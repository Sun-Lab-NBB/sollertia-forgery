from pathlib import Path
from threading import Lock, Thread
import contextlib
from collections import deque
from dataclasses import field, dataclass
from collections.abc import (
    Callable as Callable,
    Iterator,
)
from concurrent.futures import Future, ProcessPoolExecutor

from .graph import (
    PendingJob as PendingJob,
    resolve_dispatch_priorities as resolve_dispatch_priorities,
)
from ..shared_assets import posix_text as posix_text

RESERVED_CORES: int
_WORKER_THREAD_CEILING: int
_IMPORT_LATCHED_THREAD_VARIABLES: tuple[str, ...]
_LIVENESS_WAIT_SECONDS: float

@dataclass(slots=True, kw_only=True)
class JobExecutionState[PendingJobT: PendingJob]:
    worker: Callable[[PendingJobT], None]
    all_jobs: dict[tuple[str, str], PendingJobT] = field(default_factory=dict)
    pending_jobs: deque[PendingJobT] = field(default_factory=deque)
    active_jobs: list[_ActiveJob[PendingJobT]] = field(default_factory=list)
    core_budget: int = ...
    memory_budget_mb: int = ...
    concurrency_limits: dict[str, int] = field(default_factory=dict)
    concurrency_reservations: dict[str, int] = field(default_factory=dict)
    dispatch_priorities: dict[tuple[str, str], int] = field(default_factory=dict)
    pool_size: int = ...
    thread_ceiling: int = ...
    succeeded_job_keys: set[tuple[str, str]] = field(default_factory=set)
    failed_job_keys: set[tuple[str, str]] = field(default_factory=set)
    blocked_jobs: list[PendingJobT] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...
    canceled: bool = ...

@dataclass(frozen=True, slots=True)
class _JobAllocation:
    cores_per_job: int
    maximum_parallel: int
    concurrency_limit: int | None = ...
    concurrency_reservation: int | None = ...

@dataclass(slots=True)
class _ActiveJob[PendingJobT: PendingJob]:
    job: PendingJobT
    future: Future[None]

def resolve_core_allocations(
    job_cores: dict[str, int],
    job_names: set[str],
    core_budget: int,
    job_limits: dict[str, int] | None = None,
    job_reservations: dict[str, int] | None = None,
) -> dict[str, _JobAllocation]: ...
def job_execution_manager[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None: ...
def group_jobs_by_tracker[PendingJobT: PendingJob](
    state: JobExecutionState[PendingJobT],
) -> dict[Path, list[PendingJobT]]: ...
def apply_decode_thread_ceiling(cores: int) -> None: ...
@contextlib.contextmanager
def _pinned_pool_imports() -> Iterator[None]: ...
def _initialize_worker_threads(thread_ceiling: int = ..., console_disabled: bool = False) -> None: ...
def _reset_queued_jobs[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None: ...
def _refresh_job_outcomes[PendingJobT: PendingJob](state: JobExecutionState[PendingJobT]) -> None: ...
def _admit_pending_jobs[PendingJobT: PendingJob](
    state: JobExecutionState[PendingJobT], pool: ProcessPoolExecutor
) -> None: ...
