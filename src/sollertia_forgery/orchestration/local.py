"""Provides the generic batch-orchestration primitives that the interface Model Context Protocol tools drive."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from threading import Lock, Thread
import contextlib
from collections import deque
from dataclasses import field, dataclass
from concurrent.futures import Future, ProcessPoolExecutor

from ataraxis_time import PrecisionTimer, TimerPrecisions

if TYPE_CHECKING:
    from collections.abc import Callable

RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when the worker budget auto-resolves. The generic
``execute_jobs_tool`` forwards this to ``resolve_worker_count``, which applies it only to a non-positive budget and
honors an explicit budget up to the physical core count."""


@dataclass(frozen=True, slots=True)
class ConcurrencyDescriptor:
    """Describes the per-pipeline concurrency policy consulted by the generic ``execute_jobs_tool``.

    Notes:
        ``cores_per_job`` is the number of CPU cores a single worker subprocess consumes. The generic tool floors
        the user-supplied parallel-job cap by ``worker_budget // cores_per_job``. ``default_max_parallel`` is the
        fallback hard cap applied when the caller does not request an explicit parallel-job ceiling. This descriptor
        is system-agnostic, so one shared type declares the concurrency policy for every pipeline in the dispatch
        table, whether system-specific or agnostic.
    """

    cores_per_job: int
    """The number of CPU cores a single worker subprocess of this pipeline consumes."""
    default_max_parallel: int
    """The default hard cap on concurrently executing jobs when the caller does not request one. A non-positive
    value defers concurrency to the resolved worker budget alone."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single batch processing job tracked by a ``ProcessingTracker`` file.

    Notes:
        Subclasses extend this dataclass with the additional fields their worker callables need. The base fields
        are sufficient for the shared execution manager to group jobs by tracker, read their status from disk, and
        cancel or reset them without knowing any domain-specific details.
    """

    tracker_path: Path
    """The path to the ``ProcessingTracker`` YAML file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the
        tracker path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)
class GenericPendingJob(PendingJob):
    """Describes a single batch processing job for the system-agnostic processing tools.

    Notes:
        Extends the shared ``PendingJob`` base with the descriptor set every registered pipeline worker needs, so
        one descriptor serves every pipeline. The picklable workers in the worker registry map these fields onto each
        pipeline's call convention (the session pipeline workers use ``unit_path``, while the forging worker uses
        ``name`` and ``project_root``). Fields that do not apply to a given pipeline stay at their defaults, and the
        processing tools assert the required fields per pipeline before dispatch.
    """

    name: str = ""
    """The human-readable unit name (the session name for session jobs or the dataset name for forging jobs)
    used for logging and status reporting."""
    unit_path: Path = field(default_factory=Path)
    """The path to the processing unit this job operates on (the session root for session jobs)."""
    job_name: str = ""
    """The pipeline job type name registered in the ``ProcessingTracker`` (paired with ``specifier`` to derive
    the job ID)."""
    specifier: str = ""
    """The job-specific specifier that differentiates jobs of the same type within a unit (the system ID, the
    controller-type-id triple, or the session name for forging assembly jobs)."""
    project_root: Path | None = None
    """The project root directory passed to the forging worker. Unused by pipelines that resolve their output
    location from the unit path alone."""


@dataclass(slots=True)
class ActiveJob[PendingJobT: PendingJob]:
    """Tracks a single pending job currently executing as a ``Future`` on the shared process pool."""

    job: PendingJobT
    """The pending job descriptor associated with the running future."""
    future: Future[None]
    """The future returned by ``ProcessPoolExecutor.submit`` for this job."""


@dataclass(slots=True, kw_only=True)
class JobExecutionState[PendingJobT: PendingJob]:
    """Tracks runtime state for a batch execution session with budget-bounded concurrency.

    The state stores the pending and active job queues, the worker callable used to dispatch each job to a
    subprocess, the lock that serializes state mutations, and the cancellation flag consulted by the manager
    thread. The generic batch tools store one of these per pipeline in a module-level registry, so the status
    and cancel tools read it directly. The manager owns a single ``ProcessPoolExecutor`` sized to
    ``worker_budget`` and dispatches each pending job as an independent future.

    The generic type parameter ``PendingJobT`` is a ``PendingJob`` subclass. Subclasses
    carry fields such as session paths, output directories, and job specifiers that the worker callable needs
    at dispatch time.
    """

    worker: Callable[[PendingJobT], None]
    """The picklable module-level function invoked by ``ProcessPoolExecutor.submit`` for each pending job.
    Must accept a single argument of the pending job subclass associated with this state."""
    all_jobs: dict[tuple[str, str], PendingJobT] = field(default_factory=dict)
    """All submitted jobs keyed by ``(tracker_path, job_id)`` dispatch key."""
    pending_queue: deque[PendingJobT] = field(default_factory=deque)
    """Jobs awaiting dispatch."""
    active_jobs: list[ActiveJob[PendingJobT]] = field(default_factory=list)
    """Jobs currently executing on the shared process pool."""
    worker_budget: int = 1
    """Total CPU cores available for the execution session."""
    max_parallel_jobs: int = -1
    """Hard cap on concurrently executing jobs. Set to -1 to defer to ``worker_budget``. Tools where each
    job consumes multiple cores internally set this to bound memory footprint independently of CPU
    allocation."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    canceled: bool = False
    """Determines whether the execution session has been canceled."""


def job_execution_manager[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None:
    """Dispatches queued jobs against a shared ``ProcessPoolExecutor``.

    Notes:
        Runs as a daemon thread for the lifetime of a single execution session. Each poll cycle collects
        completed futures, frees their budget, and dispatches new jobs from the pending queue while the number
        of active jobs stays below the worker budget. Exits when the queue is empty and no jobs remain in
        flight. Cancellation stops new dispatches but lets already-running futures finish naturally. The manager
        calls ``state.worker`` via ``ProcessPoolExecutor.submit``, so the worker must be a picklable
        module-level function that accepts a single pending-job argument.

    Args:
        state: The active job execution state containing the pending queue, active jobs, worker callable, and
            worker budget. Mutated under ``state.lock`` as jobs move between queues.
    """
    poll_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    # Resolves the concurrency cap. A non-positive ``max_parallel_jobs`` defers to the CPU budget alone.
    concurrency_limit = (
        state.worker_budget if state.max_parallel_jobs <= 0 else min(state.worker_budget, state.max_parallel_jobs)
    )

    # Creates a single ProcessPoolExecutor sized to the concurrency limit. The executor is reused across
    # every dispatch cycle so worker subprocesses are spawned once per execution session rather than per job.
    with ProcessPoolExecutor(max_workers=concurrency_limit) as pool:
        while True:
            with state.lock:
                # Reaps completed futures and frees their budget. Draining each future's result surfaces any
                # worker exception so the daemon thread does not silently lose failures. The tracker
                # remains the authoritative source for per-job outcomes since the worker is expected to
                # transition its job to a terminal state before returning.
                still_active: list[ActiveJob[PendingJobT]] = []
                for active in state.active_jobs:
                    if active.future.done():
                        with contextlib.suppress(Exception):
                            active.future.result()
                    else:
                        still_active.append(active)
                state.active_jobs = still_active

                # Exits when there is no more work to do.
                if not state.pending_queue and not state.active_jobs:
                    break

                # Dispatches as many pending jobs as the remaining budget allows. Cancellation suppresses new
                # dispatches but lets the already-running futures continue to completion.
                if not state.canceled:
                    available = concurrency_limit - len(state.active_jobs)
                    while state.pending_queue and available > 0:
                        job = state.pending_queue.popleft()
                        future = pool.submit(state.worker, job)
                        state.active_jobs.append(ActiveJob(job=job, future=future))
                        available -= 1

            poll_timer.delay(delay=1, allow_sleep=True)


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
