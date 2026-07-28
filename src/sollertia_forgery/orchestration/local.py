"""Provides the shared batch execution engine that admits queued jobs against a core and a memory budget and
dispatches them in their pipelines' dependency order.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Lock, Thread
import contextlib
from collections import deque
from dataclasses import field, dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

import cv2
import numba
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

if TYPE_CHECKING:
    from collections.abc import Callable


RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when the core budget auto-resolves. The batch tools
forward this to ``resolve_worker_count``, which applies it only to a non-positive budget and honors an explicit
budget up to the logical core count."""


_WORKER_THREAD_CEILING: int = 1
"""The number of threads each pool worker pins its library thread pools to. Every job type either runs
single-threaded, raises its own thread count once it starts, or fans out into a sub-pool whose children each cost the
single core the allocation budgeted for them."""


_PINNED_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
)
"""The threading-layer environment variables a pool worker pins when it starts.

Notes:
    ``NUMBA_NUM_THREADS`` is deliberately absent. numba reads that variable once, when it is imported, and treats the
    value it read as the ceiling for the rest of the process. It then re-reads the variable on every compilation and
    raises if the two disagree once its thread pool has started. A worker imports numba before this pin could run, so
    writing the variable here would guarantee that disagreement and fail every job that compiles a numba function.
    The worker sets numba's thread count through its runtime API instead, which is the supported way to change it.
"""


_LIVENESS_WAIT_SECONDS: float = 10 * 60
"""The longest the manager blocks on a running job before looking at its state again. A job finishing is the only
event the loop acts on, so this bound never governs a healthy batch and exists so a future that never resolves
cannot stall the manager for good."""


_TIFF_DECODE_THREAD_CEILING: int = 4
"""The thread ceiling applied to the hidden image-decode pool some readers open. That pool otherwise sizes itself
from the host's core count entirely outside the batch's allocation."""


@dataclass(frozen=True, slots=True)
class JobAllocation:
    """Describes the cores one job type receives and how many of its jobs may run at once."""

    cores_per_job: int
    """The cores each job of this type occupies while it runs."""
    maximum_parallel: int
    """The jobs of this type that may run at once, which is what the core budget alone would allow narrowed by any
    concurrency limit the type declares. Reported for the caller's planning, since admission weighs each running job
    against both budgets and the limit directly."""
    concurrency_limit: int | None = None
    """The concurrent-job ceiling this type declares beyond the two budgets, or None when the budgets alone bound it.
    Reported alongside the resolved concurrency so a caller reading a maximum below the core budget's own can tell
    which term produced it."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single batch processing job tracked by a ``ProcessingTracker`` file.

    Notes:
        Subclasses extend this dataclass with the additional fields their worker callables need. The base fields
        carry everything the shared execution manager needs, which is the tracker a job is recorded on, the cores and
        memory it occupies, and the jobs it waits for. That leaves the manager free of any domain-specific detail.
    """

    tracker_path: Path
    """The path to the ``ProcessingTracker`` YAML file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    job_name: str = ""
    """The pipeline job type name registered in the ``ProcessingTracker``, which keys this job's core allocation."""
    core_weight: int = 1
    """The cores this job occupies while it runs, assigned from its type's allocation before dispatch."""
    memory_mb: int = 0
    """The memory this job occupies while it runs, estimated from the data it will process."""
    prerequisite_ids: tuple[str, ...] = ()
    """The identifiers of the jobs that must succeed before this job may be dispatched. Resolved from the pipeline's
    own job ordering, and empty for a job that depends on nothing."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the
        tracker path with the job ID.
        """
        return str(self.tracker_path), self.job_id

    @property
    def prerequisite_keys(self) -> tuple[tuple[str, str], ...]:
        """Returns the dispatch keys of this job's upstream jobs.

        Notes:
            A job identifier is derived from the job name and specifier alone, so the same stage of two different
            sessions shares one identifier. Pairing each identifier with this job's tracker keeps a batch spanning
            many sessions from treating one session's completed stage as every session's.
        """
        return tuple((str(self.tracker_path), prerequisite) for prerequisite in self.prerequisite_ids)


@dataclass(slots=True)
class GenericPendingJob(PendingJob):
    """Describes a single batch processing job for the system-agnostic processing tools.

    Notes:
        Extends the shared ``PendingJob`` base with the descriptor set every registered pipeline worker needs, so one
        descriptor serves every pipeline. The shared worker routes on ``pipeline`` and each session pipeline's worker
        reads ``unit_path`` and ``job_id``. Fields a pipeline does not use stay at their defaults, and a descriptor
        missing a field the engine requires is rejected before dispatch.
    """

    pipeline: str = ""
    """The pipeline this job belongs to, which the shared worker routes on so one pool serves every pipeline."""
    name: str = ""
    """The human-readable unit name used for logging and status reporting."""
    unit_path: Path = field(default_factory=Path)
    """The path to the processing unit this job operates on (the session root for session jobs)."""
    specifier: str = ""
    """The specifier that differentiates jobs of the same type within one unit, such as a camera or controller source
    identifier, a controller-module triple, or a plane index."""
    project_root: Path | None = None
    """The project root directory, carried for workers that resolve their output location above the unit path."""
    options: dict[str, Any] = field(default_factory=dict)
    """The pipeline-specific parameters the caller chose for this job, such as the mode a multi-mode pipeline runs in.
    The execution engine never reads this mapping, so a pipeline's worker interprets whichever keys it declares and
    ignores the rest. A pipeline that takes no parameters leaves it empty."""


@dataclass(slots=True)
class ActiveJob[PendingJobT: PendingJob]:
    """Tracks a single pending job currently executing as a ``Future`` on the shared process pool."""

    job: PendingJobT
    """The pending job descriptor associated with the running future."""
    future: Future[None]
    """The future returned by ``ProcessPoolExecutor.submit`` for this job."""


@dataclass(slots=True, kw_only=True)
class JobExecutionState[PendingJobT: PendingJob]:
    """Tracks runtime state for one batch execution session budgeted by both cores and memory.

    The state stores the job queues, the worker callable, the two budgets, the recorded outcomes that resolve
    ordering, the lock that serializes mutations, and the cancellation flag. The batch tools keep a single one of
    these, so one pool serves every pipeline and the status and cancel tools read it directly. The manager owns one
    ``ProcessPoolExecutor`` and admits each pending job once the running set has room for both its cores and its
    memory.

    Notes:
        The generic type parameter ``PendingJobT`` is a ``PendingJob`` subclass. Subclasses carry the fields a worker
        callable needs at dispatch time, such as the path of the unit the job processes.
    """

    worker: Callable[[PendingJobT], None]
    """The picklable module-level function invoked by ``ProcessPoolExecutor.submit`` for each pending job.
    Must accept a single argument of the pending job subclass associated with this state."""
    all_jobs: dict[tuple[str, str], PendingJobT] = field(default_factory=dict)
    """All submitted jobs keyed by ``(tracker_path, job_id)`` dispatch key."""
    pending_jobs: deque[PendingJobT] = field(default_factory=deque)
    """Jobs awaiting dispatch, held in the order the next admission pass considers them."""
    active_jobs: list[ActiveJob[PendingJobT]] = field(default_factory=list)
    """Jobs currently executing on the shared process pool."""
    core_budget: int = 1
    """The cores the batch may commit across all concurrently running jobs."""
    memory_budget_mb: int = 1024
    """The memory the batch may commit across all concurrently running jobs."""
    concurrency_limits: dict[str, int] = field(default_factory=dict)
    """The jobs of each type that may run at once, keyed by tracker job name, for the types that declare a ceiling
    beyond the two budgets. A job type absent from this mapping is bounded by the budgets alone."""
    pool_size: int = 1
    """The number of worker processes the pool spawns."""
    thread_ceiling: int = _WORKER_THREAD_CEILING
    """The thread count each worker pins its library thread pools to."""
    succeeded_job_keys: set[tuple[str, str]] = field(default_factory=set)
    """The dispatch keys of the jobs known to have succeeded, seeded from the trackers and extended as jobs finish."""
    failed_job_keys: set[tuple[str, str]] = field(default_factory=set)
    """The dispatch keys of the jobs known to have failed, whose dependents can never become runnable."""
    blocked_jobs: list[PendingJobT] = field(default_factory=list)
    """Jobs dropped without dispatch because a prerequisite failed or never ran."""
    lock: Lock = field(default_factory=Lock)
    """The lock guarding every mutation of the job queues and the recorded outcomes."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    canceled: bool = False
    """Determines whether the execution session has been canceled."""


def resolve_core_allocations(
    job_cores: dict[str, int],
    job_names: set[str],
    core_budget: int,
    job_limits: dict[str, int] | None = None,
) -> dict[str, JobAllocation]:
    """Resolves how many cores each queued job type receives and how many of its jobs run at once.

    Notes:
        A type's core count is its declared allocation, narrowed to the budget so a small host never promises a job
        more cores than it has. The concurrency that follows is the budget divided by that count, narrowed again by
        any ceiling the type declares for itself. The core term the engine treats as a guide, since admission weighs
        every running job against the same budget, while the declared ceiling admission enforces exactly.

        A job type with no registered allocation stops the batch, since dispatching it would run it at a width
        nobody chose.

    Raises:
        ValueError: If any queued job type has no registered core allocation.

    Args:
        job_cores: The cores one job of each type occupies, keyed by tracker job name.
        job_names: The job type names present in the batch.
        core_budget: The cores the batch may commit across all concurrently running jobs.
        job_limits: The concurrent-job ceilings the job types declare beyond the budgets, keyed by tracker job name.
            Only the types that declare one appear, and passing nothing bounds every type by the budgets alone.

    Returns:
        A dictionary mapping each job name to its resolved allocation.
    """
    unregistered = sorted(name for name in job_names if name not in job_cores)
    if unregistered:
        message = (
            f"Unable to resolve core allocations for the batch. No core allocation is registered for job "
            f"type(s) {unregistered}. Every dispatched job type must declare the cores one of its jobs occupies."
        )
        console.error(message=message, error=ValueError)

    limits = job_limits if job_limits is not None else {}
    allocations: dict[str, JobAllocation] = {}
    for job_name in job_names:
        cores = max(1, min(job_cores[job_name], core_budget))
        limit = limits.get(job_name)
        parallel = max(1, core_budget // cores)
        allocations[job_name] = JobAllocation(
            cores_per_job=cores,
            maximum_parallel=parallel if limit is None else min(parallel, limit),
            concurrency_limit=limit,
        )
    return allocations


def job_execution_manager[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None:
    """Dispatches queued jobs against a shared ``ProcessPoolExecutor`` under the batch's core and memory budgets.

    Notes:
        Runs as a daemon thread for the lifetime of a single execution session. The loop wakes when a running job
        finishes, so capacity is refilled the moment it is released and a long-running batch costs nothing while its
        jobs run. Each pass reaps finished futures, refreshes the recorded outcomes when one finished, and admits
        whatever the freed capacity and the pipelines' own orderings allow. Cancellation stops new admissions but
        lets running futures finish naturally.

        A pass that admits nothing while nothing is running means every remaining job waits on a prerequisite that
        neither succeeded nor is queued. The remainder is recorded as blocked and the session ends rather than
        waiting forever.

    Args:
        state: The active job execution state containing the pending queue, active jobs, worker callable, and
            budgets. Mutated under ``state.lock`` as jobs move between queues.
    """
    with ProcessPoolExecutor(
        max_workers=state.pool_size, initializer=_initialize_worker_threads, initargs=(state.thread_ceiling,)
    ) as pool:
        with state.lock:
            # Clears the recorded outcome of every job this batch holds, so the trackers report this run alone.
            _reset_queued_jobs(state=state)

            # Seeds the recorded outcomes before the first admission, so a batch that queues only a pipeline's later
            # stages still sees the earlier stages a previous run already completed. The reset above leaves those
            # earlier stages untouched, since a batch never queues them.
            _refresh_job_outcomes(state=state)

        while True:
            with state.lock:
                # Reaps finished futures and frees their share of both budgets. Each result is drained and its
                # exception discarded, because a worker records its own outcome on the tracker before returning and
                # the tracker is what the status tool and the ordering logic both read.
                still_active: list[ActiveJob[PendingJobT]] = []
                completed_any = False
                for active in state.active_jobs:
                    if active.future.done():
                        with contextlib.suppress(Exception):
                            active.future.result()
                        completed_any = True
                    else:
                        still_active.append(active)
                state.active_jobs = still_active

                if not state.pending_jobs and not state.active_jobs:
                    break

                # Re-reads the trackers only when something finished, since that is the only event that can newly
                # satisfy a prerequisite from inside this batch.
                if completed_any:
                    _refresh_job_outcomes(state=state)

                if not state.canceled:
                    _admit_pending_jobs(state=state, pool=pool)

                if not state.active_jobs and state.pending_jobs:
                    state.blocked_jobs.extend(state.pending_jobs)
                    state.pending_jobs.clear()
                    break

                pending_futures = [active.future for active in state.active_jobs]

            # Blocks until a running job finishes. A job completing is the only event that frees capacity or
            # satisfies a prerequisite, so the loop has nothing to do until one does. The loop breaks above whenever
            # the active set empties, so a future is always in flight to wait on, and the bound is a liveness
            # backstop rather than a polling interval.
            wait(pending_futures, timeout=_LIVENESS_WAIT_SECONDS, return_when=FIRST_COMPLETED)


def group_jobs_by_tracker[PendingJobT: PendingJob](
    state: JobExecutionState[PendingJobT],
) -> dict[Path, list[PendingJobT]]:
    """Groups all jobs in an execution state by their tracker file path.

    Minimizes redundant file reads by batching jobs that share the same tracker, so each tracker YAML file is
    deserialized only once when iterating over the groups.

    Args:
        state: The active job execution state containing the job registry.

    Returns:
        A dictionary mapping each tracker path to its list of pending job descriptors.
    """
    tracker_jobs: dict[Path, list[PendingJobT]] = {}
    for job in state.all_jobs.values():
        tracker_jobs.setdefault(job.tracker_path, []).append(job)
    return tracker_jobs


def _initialize_worker_threads(thread_ceiling: int = _WORKER_THREAD_CEILING) -> None:
    """Pins a pool worker's library thread pools when the worker process starts.

    Notes:
        Runs as the ``ProcessPoolExecutor`` initializer in every spawned child. Setting the environment variables
        alone is not sufficient, because the package imports numba to select its threading layer, so numba latches
        its maximum thread count from the unpinned environment before this runs. The runtime setters are therefore
        called alongside the variables. A job that needs more threads raises its own count once it starts, which
        numba permits up to the count latched at import.

        numba and OpenCV are pinned through their runtime setters alone, leaving the environment they read at import
        untouched. Both already hold the count they read when the worker imported them, so rewriting those variables
        would change nothing they consult again. For numba it would actively break the worker, since it compares the
        variable against the latched count on every compilation and rejects a disagreement once its threads have
        started, which is exactly the state a late pin creates.

    Args:
        thread_ceiling: The number of threads each library thread pool is pinned to.
    """
    ceiling = max(1, thread_ceiling)
    for variable in _PINNED_THREAD_VARIABLES:
        os.environ[variable] = str(ceiling)
    os.environ["TIFFFILE_NUM_THREADS"] = str(min(_TIFF_DECODE_THREAD_CEILING, ceiling))

    numba.set_num_threads(min(ceiling, numba.config.NUMBA_NUM_THREADS))  # type: ignore[attr-defined]
    cv2.setNumThreads(ceiling)


def _reset_queued_jobs[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None:
    """Returns every job this batch holds to the scheduled state on its tracker.

    Notes:
        Runs once, before the first admission. Without it a queued job whose tracker still records an earlier
        success reports as succeeded from the moment the batch starts, which makes a run's progress
        indistinguishable from its history and lets a status reader call a batch complete before it has dispatched
        anything. Resetting is safe because a caller queues a job precisely to have it run again.

        Jobs are grouped by tracker so each file is rewritten once, however many of its jobs the batch holds.

    Args:
        state: The active job execution state whose jobs are reset. Its trackers are rewritten in place.
    """
    for tracker_path, jobs in group_jobs_by_tracker(state=state).items():
        if not tracker_path.is_file():
            continue
        ProcessingTracker(file_path=tracker_path).reset_jobs(job_ids=[job.job_id for job in jobs])


def _refresh_job_outcomes[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None:
    """Re-reads the batch's trackers and records which jobs have succeeded or failed.

    Notes:
        The trackers are the authoritative record of every job's outcome, so prerequisite satisfaction is read from
        them rather than inferred from the futures. Reading them also picks up prerequisites that succeeded in an
        earlier batch and were never queued in this one. That is how a run asking only for a pipeline's later stages
        still resolves its ordering. Outcomes are keyed by tracker as well as identifier, so one session's completed
        stage never satisfies another session's.

    Args:
        state: The active job execution state whose tracker files are re-read. Its outcome sets are updated in place.
    """
    for tracker_path in {job.tracker_path for job in state.all_jobs.values()}:
        if not tracker_path.is_file():
            continue
        for job_id, job_state in ProcessingTracker(file_path=tracker_path).snapshot().items():
            if job_state.status is ProcessingStatus.SUCCEEDED:
                state.succeeded_job_keys.add((str(tracker_path), job_id))
            elif job_state.status is ProcessingStatus.FAILED:
                state.failed_job_keys.add((str(tracker_path), job_id))


def _admit_pending_jobs[PendingJobT: PendingJob](
    state: JobExecutionState[PendingJobT], pool: ProcessPoolExecutor
) -> None:
    """Admits every queued job whose prerequisites are met, whose type is below its concurrency limit, and whose
    cores and memory the budgets still allow.

    Notes:
        A job is weighed against both budgets, and the one that runs out first is whichever the batch's mix makes
        scarce. Committed totals are recomputed from the running set on each pass, so a job that fails or is dropped
        releases its share automatically. This running total is the batch's main resource guard, so a heavy job and
        a crowd of light ones share the host while committed cores and memory stay within both budgets.

        A job type that declares a concurrency limit is held to it by a third admission term, counted from the
        running set the same way. The budgets bound what the host can supply, which leaves a job type whose pace is
        set by storage throughput free to open far more streams than the array serves. The limit bounds that
        directly, so those types stay at the concurrency they gain from rather than the concurrency the cores allow.

        The scan considers the heaviest job first and continues past anything that does not fit, so large jobs are
        admitted as soon as the budgets allow. Small jobs backfill whatever capacity the large ones leave spare. A
        job is admitted alone when nothing is running, so a job larger than the whole budget still makes progress.
        That floor holds the prerequisite check, since dispatching a job before its input exists would fail rather
        than progress. It also holds the concurrency limit, which never binds an idle pool because every limit is at
        least one.

    Args:
        state: The active job execution state. Its pending queue is rebuilt from the jobs that were not admitted.
        pool: The process pool the admitted jobs are submitted into.
    """
    used_cores = sum(active.job.core_weight for active in state.active_jobs)
    used_memory = sum(active.job.memory_mb for active in state.active_jobs)

    running_counts: dict[str, int] = {}
    for active in state.active_jobs:
        running_counts[active.job.job_name] = running_counts.get(active.job.job_name, 0) + 1

    admitted_any = False
    deferred: deque[PendingJobT] = deque()
    candidates = sorted(state.pending_jobs, key=lambda pending: pending.memory_mb, reverse=True)
    state.pending_jobs = deque(candidates)

    while state.pending_jobs:
        job = state.pending_jobs.popleft()

        if any(prerequisite in state.failed_job_keys for prerequisite in job.prerequisite_keys):
            state.blocked_jobs.append(job)
            continue
        if not all(prerequisite in state.succeeded_job_keys for prerequisite in job.prerequisite_keys):
            deferred.append(job)
            continue

        # Bounds how many streams this job type opens against the storage array, which neither budget expresses.
        limit = state.concurrency_limits.get(job.job_name)
        if limit is not None and running_counts.get(job.job_name, 0) >= limit:
            deferred.append(job)
            continue

        forced = not state.active_jobs and not admitted_any
        fits = (
            used_cores + job.core_weight <= state.core_budget and used_memory + job.memory_mb <= state.memory_budget_mb
        )
        if not (fits or forced):
            deferred.append(job)
            continue

        future = pool.submit(state.worker, job)
        state.active_jobs.append(ActiveJob(job=job, future=future))
        used_cores += job.core_weight
        used_memory += job.memory_mb
        running_counts[job.job_name] = running_counts.get(job.job_name, 0) + 1
        admitted_any = True

    state.pending_jobs = deferred
