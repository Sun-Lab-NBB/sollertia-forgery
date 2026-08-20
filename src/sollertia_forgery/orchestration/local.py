"""Provides the local batch execution engine that admits queued jobs against a core and a memory budget, then
dispatches them onto a shared process pool in their pipelines' dependency order.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from threading import Lock, Thread
import contextlib
from collections import Counter, deque
from dataclasses import field, dataclass
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import cv2
from cindra import TIFF_DECODE_CEILING
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, initialize_worker_threads

from .graph import PendingJob, resolve_dispatch_priorities

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Iterator
    from concurrent.futures import Future


RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when the core budget auto-resolves. The batch tools
forward this to ``resolve_worker_count``, which applies it only to a non-positive budget and honors an explicit
budget up to the logical core count."""

_WORKER_THREAD_CEILING: int = 1
"""The number of threads each pool worker pins its library thread pools to. Every job type either runs
single-threaded, raises its own thread count once it starts, or fans out into a sub-pool whose children each cost the
single core the allocation budgeted for them."""

_IMPORT_LATCHED_THREAD_VARIABLES: tuple[str, ...] = ("POLARS_MAX_THREADS",)
"""The threading-layer environment variables that have to be set before a pool worker starts, because the pools they
size cannot be resized once the worker has imported the library that owns them.

Notes:
    polars builds its thread pool as it is imported, exposes no runtime setter, and is not one of the pools
    ``threadpool_limits`` manages, so the only moment its width can be chosen is before the child process imports it.
    Every job type that uses polars in the executor process itself declares a single core, so one thread is the width
    that matches what those jobs were admitted at.

    The BLAS and OpenMP variables are deliberately absent, which is why this narrower tuple stands in for the shared
    ``limit_worker_threads`` context that pins all of them. Their pools are resized at runtime for the duration of
    each job, which holds every job to its own core weight rather than to one width shared by every job a worker
    runs. A BLAS backend that reads its variable at load also treats the value it read as the widest pool it will
    ever allocate buffers for, so pinning it here would cap the compute stages at one thread for the life of the
    worker and leave the per-job resize with nothing to raise.
"""

_LIVENESS_WAIT_SECONDS: float = 10 * 60
"""The longest the manager blocks on a running job before looking at its state again. A job finishing is the only
event the loop acts on, so this bound never governs a healthy batch and exists so a future that never resolves
cannot stall the manager for good."""


@dataclass(frozen=True, slots=True)
class JobAllocation:
    """Describes the cores the host can supply for one job of a type and how many of its jobs may run at once."""

    cores_per_job: int
    """The cores the host can supply for one job of this type, resolved from the widest job the type holds. Sizing is
    per job rather than per type, so dispatch caps each job at this width instead of running every job at it, and a
    job the sizing pass placed below it keeps the narrower width its own model chose."""
    maximum_parallel: int
    """The jobs of this type that may run at once, which is what the core budget alone would allow narrowed by any
    concurrency limit the type declares. Reported for the caller's planning, since admission weighs each running job
    against both budgets and the limit directly."""
    concurrency_limit: int | None = None
    """The concurrent-job ceiling this type declares beyond the two budgets, or None when the budgets alone bound it.
    Reported alongside the resolved concurrency so a caller reading a maximum below the core budget's own can tell
    which term produced it. This ceiling holds however much capacity is idle."""
    concurrency_reservation: int | None = None
    """The concurrency this type is held to while other work can use the capacity it gives up, or None when it
    competes at its full width. A reserved type runs at this count while other jobs are runnable and widens toward
    ``maximum_parallel`` once nothing else claims the room, so both numbers describe it."""


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

    The batch tools keep a single one of these, so one pool serves every pipeline and the status and cancel tools
    read it directly. The manager owns one ``ProcessPoolExecutor`` and admits each pending job once the running set
    has room for both its cores and its memory.

    Notes:
        Subclasses of ``PendingJob`` carry the fields a worker callable needs at dispatch time, such as the path of
        the unit the job processes.
    """

    worker: Callable[[PendingJobT], None]
    """The picklable module-level function invoked by ``ProcessPoolExecutor.submit`` for each pending job."""
    all_jobs: dict[tuple[str, str], PendingJobT] = field(default_factory=dict)
    """All submitted jobs keyed by ``(unit_path, job_id)`` dispatch key."""
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
    beyond the two budgets. This ceiling holds however much capacity is idle, since a type recorded here waits on a
    resource that spare cores and spare memory do not supply. A job type absent from this mapping is bounded by the
    budgets alone."""
    concurrency_reservations: dict[str, int] = field(default_factory=dict)
    """The jobs of each type that run at once while other work can still use the capacity the type gives up, keyed
    by tracker job name. Admission offers that capacity to every other runnable job first and then releases the
    reservation over whatever remains, so a reserved type widens rather than idling the host."""
    dispatch_priorities: dict[tuple[str, str], int] = field(default_factory=dict)
    """The cores each job's transitive dependents commit, keyed by dispatch key, which is the weight admission
    considers candidates in. Resolved once from the job set when the manager starts, since the batch's dependency
    graph does not change while it runs."""
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
    """The background thread running the execution manager, or None before the session starts it."""
    canceled: bool = False
    """Determines whether the execution session has been canceled."""


def resolve_core_allocations(
    job_cores: dict[str, int],
    job_names: set[str],
    core_budget: int,
    job_limits: dict[str, int] | None = None,
    job_reservations: dict[str, int] | None = None,
) -> dict[str, JobAllocation]:
    """Resolves the cores the host can supply for one job of each queued job type and how many of its jobs run at once.

    Notes:
        A type's core count is the allocation its representative job declares, narrowed to the budget so a small host
        never promises a job more cores than it has. The caller supplies the widest job of each type as that
        representative, because sizing is per job and one type can hold jobs of several widths. Dispatch then caps
        each job at the count resolved here rather than dispatching every job of the type at it, so a narrower job
        keeps the width its own sizing pass chose. The concurrency that follows is the budget divided by that count,
        narrowed again by any ceiling the type declares for itself. The core term the engine treats as a guide, since
        admission weighs every running job against the same budget at that job's own width, while the declared ceiling
        admission enforces exactly.

        A job type with no registered allocation stops the batch, since dispatching it would run it at a width
        nobody chose.

    Args:
        job_cores: The cores the widest job of each type occupies, keyed by tracker job name, which is the
            representative width the type's allocation is resolved from.
        job_names: The job type names present in the batch.
        core_budget: The cores the batch may commit across all concurrently running jobs.
        job_limits: The concurrent-job ceilings the job types declare beyond the budgets, keyed by tracker job name.
            Only the types that declare one appear, and passing nothing bounds every type by the budgets alone.
        job_reservations: The concurrency the job types are held to while other work can use the capacity they give
            up, keyed by tracker job name. Only the types that declare one appear, and passing nothing lets every
            type compete at its full width.

    Returns:
        A dictionary mapping each job name to its resolved allocation.

    Raises:
        ValueError: If any queued job type has no registered core allocation.
    """
    unregistered = sorted(name for name in job_names if name not in job_cores)
    if unregistered:
        message = (
            f"Unable to resolve core allocations for the batch. No core allocation is registered for job "
            f"type(s) {unregistered}. Every dispatched job type must declare the cores one of its jobs occupies."
        )
        console.error(message=message, error=ValueError)

    limits = job_limits if job_limits is not None else {}
    reservations = job_reservations if job_reservations is not None else {}
    allocations: dict[str, JobAllocation] = {}
    for job_name in job_names:
        cores = max(1, min(job_cores[job_name], core_budget))
        limit = limits.get(job_name)
        parallel = max(1, core_budget // cores)
        allocations[job_name] = JobAllocation(
            cores_per_job=cores,
            maximum_parallel=parallel if limit is None else min(parallel, limit),
            concurrency_limit=limit,
            concurrency_reservation=reservations.get(job_name),
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
    with (
        _pinned_pool_imports(),
        ProcessPoolExecutor(
            max_workers=state.pool_size,
            initializer=_initialize_worker_threads,
            # A spawned worker re-imports the library and comes up with a freshly enabled console while inheriting
            # this process' stdout, so the parent's silence has to travel with it. Mirroring the parent is what keeps
            # a batch run's worker output intact and silences only the children of a parent that is already silent,
            # which is the MCP server on the stdio transport, where an echoed line would corrupt the JSON-RPC stream.
            initargs=(state.thread_ceiling, not console.enabled),
        ) as pool,
    ):
        with state.lock:
            # Weighs every job by the work waiting on it, so admission can favor the batch's critical path. The
            # dependency graph is fixed for the session, so this is resolved once rather than on every pass.
            state.dispatch_priorities = resolve_dispatch_priorities(jobs=state.all_jobs)

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


@contextlib.contextmanager
def _pinned_pool_imports() -> Iterator[None]:
    """Pins the thread pools a spawned worker sizes while importing, then restores the environment.

    Notes:
        Scopes the creation of the shared pool. A spawned worker re-imports rather than inheriting the parent's
        modules, so a library that sizes its pool at import does so before any code the worker runs, and the width it
        picks comes from the environment the parent handed it. A worker that inherits the defaults reserves a thread
        per core of the whole machine for a job that was admitted at one.

        Scoping this to the pool's lifetime rather than setting it once keeps the parent's own threading untouched,
        which matters because the parent has already imported the same libraries.

    Yields:
        None. The caps are in effect for the duration of the block.
    """
    previous = {variable: os.environ.get(variable) for variable in _IMPORT_LATCHED_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(_IMPORT_LATCHED_THREAD_VARIABLES, str(_WORKER_THREAD_CEILING)))
    try:
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value


def apply_decode_thread_ceiling(cores: int) -> None:
    """Bounds the default image-decode width of the calling process, from the cores one job holds.

    Notes:
        tifffile resolves this variable the first time a decode asks for a default width and holds the result for the
        life of the process. The value a pool worker writes as it starts is therefore the one every read in that
        worker sees, and a later write in the same process reaches nothing.

        cindra names its own decode width on each read, so the image conversion stage sizes its pool from the cores
        the batch allocated it rather than from this bound. What this bounds is any other TIFF read a worker performs.

    Args:
        cores: The cores the job about to run holds.
    """
    os.environ["TIFFFILE_NUM_THREADS"] = str(max(1, min(TIFF_DECODE_CEILING, cores)))


def _initialize_worker_threads(
    thread_ceiling: int = _WORKER_THREAD_CEILING,
    console_disabled: bool = False,  # noqa: FBT001, FBT002 - pool initargs are positional.
) -> None:
    """Pins a pool worker's library thread pools and mirrors the parent's console state when the worker process starts.

    Notes:
        Runs as the ``ProcessPoolExecutor`` initializer in every spawned child. The pinning itself belongs to
        ataraxis-data-structures, which writes the threading-layer variables the lazily-initialized backends still
        read at this point and pins numba through its own runtime setter, since numba latched its ceiling from the
        unpinned environment when the worker imported it. A job that needs more threads raises its own count once it
        starts, which numba permits up to that latched ceiling.

        The OpenCV core thread count is pinned here rather than there, because it is a runtime setter that library
        declines to reach for. Its FFmpeg decoder reads a variable of its own and is covered by the shared pin.

        The shared pin writes the image-decode width at the same count as every other backend, so the decode ceiling
        is reapplied over it and a worker starting at a wider count still opens no wider a decode pool than a decode
        gains from.

        A spawned worker re-imports the library rather than inheriting the parent's modules, so it comes up with a
        freshly enabled console however the parent left its own. Since the worker also inherits the parent's standard
        output stream, a parent that silenced its console has to have that silence carried across, and only a parent
        that silenced its own is mirrored.

    Args:
        thread_ceiling: The number of threads each library thread pool is pinned to.
        console_disabled: Determines whether the parent process silenced its console, which the worker mirrors.
    """
    ceiling = max(1, thread_ceiling)
    initialize_worker_threads(thread_count=ceiling)
    apply_decode_thread_ceiling(cores=ceiling)
    cv2.setNumThreads(ceiling)
    if console_disabled:
        console.disable()


def _reset_queued_jobs[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None:
    """Returns every job this batch holds to the scheduled state on its tracker.

    Notes:
        Runs once, before the first admission. Without it a queued job whose tracker still records an earlier
        success reports as succeeded from the moment the batch starts. That makes a run's progress indistinguishable
        from its history, and lets a status reader call a batch complete before it has dispatched anything.
        Resetting is safe because a caller queues a job precisely to have it run again.

        Jobs are grouped by tracker so each file is rewritten once, however many of its jobs the batch holds.

    Args:
        state: The active job execution state whose jobs are reset. Its trackers are rewritten in place.

    Raises:
        ValueError: If the batch names an identifier the unit's tracker does not hold.
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
        still resolves its ordering.

        Each tracker's jobs are recorded under the unit that tracker belongs to, matching how a job's dispatch key is
        formed, so one unit's completed stage never satisfies another unit's. Units are paired with their trackers
        rather than read from them, because a tracker file states which jobs it holds and not which unit holds it.

    Args:
        state: The active job execution state whose tracker files are re-read. Its outcome sets are updated in place.
    """
    for unit_path, tracker_path in {(job.unit_path, job.tracker_path) for job in state.all_jobs.values()}:
        if not tracker_path.is_file():
            continue
        for job_id, job_state in ProcessingTracker(file_path=tracker_path).snapshot().items():
            if job_state.status is ProcessingStatus.SUCCEEDED:
                state.succeeded_job_keys.add((str(unit_path), job_id))
            elif job_state.status is ProcessingStatus.FAILED:
                state.failed_job_keys.add((str(unit_path), job_id))


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
        Idle capacity never lifts that ceiling, since a type held by it waits on a resource the idle capacity does
        not supply.

        A job type that declares a reservation is held to it only while other jobs can use the capacity it gives up.
        Admission runs a second pass over what the first deferred, with the reservations released, so a reserved
        type widens into capacity nothing else claimed. That keeps a reservation from idling the host once its own
        queue is the only queue left.

        The scan considers the job with the most work waiting on it first, and settles ties by size so the heavier of
        two equally blocking jobs is placed while the budgets are still open. Ordering this way keeps the batch on
        its critical path, since a job at the root of a long chain is admitted ahead of a leaf that unblocks nothing.
        Size alone would invert that for a cheap root, which leaves the host idle later when the leaves are spent and
        the chain has yet to start.

        The scan continues past anything that does not fit, so smaller jobs backfill whatever capacity the larger
        ones leave spare. A job is admitted alone when nothing is running, so a job larger than the whole budget
        still makes progress. That floor holds the prerequisite check, since dispatching a job before its input
        exists would fail rather than progress. It also holds the concurrency limit, which never binds an idle pool
        because every limit is at least one.

    Args:
        state: The active job execution state. Its pending queue is rebuilt from the jobs that were not admitted.
        pool: The process pool the admitted jobs are submitted into.
    """
    used_cores = sum(active.job.core_weight for active in state.active_jobs)
    used_memory = sum(active.job.memory_mb for active in state.active_jobs)

    running_counts: Counter[str] = Counter(active.job.job_name for active in state.active_jobs)

    admitted_any = False
    remaining: deque[PendingJobT] = deque(
        sorted(
            state.pending_jobs,
            key=lambda pending: (state.dispatch_priorities.get(pending.dispatch_key, 0), pending.memory_mb),
            reverse=True,
        )
    )

    # The first pass holds every reservation, which offers the capacity a reserved type gives up to every other
    # runnable job. The second pass releases the reservations over whatever capacity that left, so a reserved type
    # widens instead of idling the host once nothing else can use the room. A batch holding no reserved type at all
    # settles in the first pass, since the second would only rescan jobs no term newly admits.
    passes = (True, False) if state.concurrency_reservations else (True,)
    for honor_reservations in passes:
        deferred: deque[PendingJobT] = deque()
        while remaining:
            job = remaining.popleft()

            if any(prerequisite in state.failed_job_keys for prerequisite in job.prerequisite_keys):
                state.blocked_jobs.append(job)
                continue
            if not all(prerequisite in state.succeeded_job_keys for prerequisite in job.prerequisite_keys):
                deferred.append(job)
                continue

            running = running_counts.get(job.job_name, 0)

            # Bounds how many streams this job type opens against the storage array, which neither budget expresses.
            # A type held here waits on a resource the spare capacity does not supply, so this ceiling stands in
            # both passes and idle cores never lift it.
            limit = state.concurrency_limits.get(job.job_name)
            if limit is not None and running >= limit:
                deferred.append(job)
                continue

            # Holds a type to the room it leaves others only while others can take that room.
            reservation = state.concurrency_reservations.get(job.job_name)
            if honor_reservations and reservation is not None and running >= reservation:
                deferred.append(job)
                continue

            forced = not state.active_jobs and not admitted_any
            fits = (
                used_cores + job.core_weight <= state.core_budget
                and used_memory + job.memory_mb <= state.memory_budget_mb
            )
            if not (fits or forced):
                deferred.append(job)
                continue

            future = pool.submit(state.worker, job)
            state.active_jobs.append(ActiveJob(job=job, future=future))
            used_cores += job.core_weight
            used_memory += job.memory_mb
            running_counts[job.job_name] = running + 1
            admitted_any = True

        remaining = deferred

    state.pending_jobs = remaining
