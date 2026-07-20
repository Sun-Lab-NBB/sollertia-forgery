"""Provides generic batch-orchestration primitives shared across every package's Model Context Protocol tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Lock, Thread
import contextlib
from collections import deque
from dataclasses import field, dataclass
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, PrecisionTimer, TimerPrecisions, convert_time
from sollertia_shared_assets import validate_directory
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

if TYPE_CHECKING:
    from collections.abc import Callable

RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations. Each package's ``execute_*_jobs_tool`` subtracts this
value from the available core count when resolving the worker budget."""


@dataclass(frozen=True, slots=True)
class ConcurrencyDescriptor:
    """Describes the per-pipeline concurrency policy consulted by the generic ``execute_jobs_tool``.

    Notes:
        ``cores_per_job`` is the number of CPU cores a single worker subprocess consumes. The generic tool floors
        the user-supplied parallel-job cap by ``worker_budget // cores_per_job``. ``default_max_parallel`` is the
        fallback hard cap applied when the caller does not request an explicit parallel-job ceiling. This descriptor
        is system-agnostic so both the system-specific batch adapters and the agnostic forging adapters can declare
        their concurrency policy with one shared type.
    """

    cores_per_job: int
    """The number of CPU cores a single worker subprocess of this pipeline consumes."""
    default_max_parallel: int
    """The default hard cap on concurrently executing jobs when the caller does not request one. A non-positive
    value defers concurrency to the resolved worker budget alone."""


_MINIMUM_ROWS_FOR_INTERVALS: int = 2
"""The minimum number of rows required in a feather file to compute inter-row timing intervals."""

_TIME_COLUMN_CANDIDATES: tuple[str, ...] = ("timestamp_us", "time_us", "frame_time_us")
"""Column names, in priority order, that ``analyze_feather_file`` recognizes as the canonical time axis
when computing timing summaries. The first matching column present in the dataframe is used, which lets the
helper cover every feather variant produced across the Sollertia stack: ``timestamp_us`` (raw axci module
feathers), ``time_us`` (forgery runtime and microcontroller outputs), and ``frame_time_us`` (axvs camera
timestamp feathers)."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single batch processing job tracked by a ``ProcessingTracker`` file.

    Notes:
        Packages subclass this dataclass with additional fields required by their domain-specific worker
        callables. The base fields are sufficient for the shared execution manager to group jobs by tracker,
        read their status from disk, and cancel or reset them without knowing any domain-specific details.
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
        Extends the shared ``PendingJob`` base with the small descriptor set shared by every registered
        pipeline worker, so a single descriptor replaces the former per-pipeline ``PendingJob`` subclasses. The
        picklable workers in the worker registry map these fields onto each pipeline's call convention (the
        behavior worker uses ``unit_path``, while the forging worker uses ``name`` and ``project_root``). Fields that
        do not apply to a given pipeline are left at their defaults, and the processing tools assert the
        required fields per pipeline before dispatch.
    """

    name: str = ""
    """The human-readable unit name (the session name for behavior jobs or the dataset name for forging jobs)
    used for logging and status reporting."""
    unit_path: Path = field(default_factory=Path)
    """The path to the processing unit this job operates on (the session root for behavior jobs)."""
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
    thread. Each package that exposes MCP batch tools keeps its own module-level state variable so that status
    and cancel tools can read it directly. The manager owns a single ``ProcessPoolExecutor`` sized to
    ``worker_budget`` and dispatches each pending job as an independent future.

    The generic type parameter ``PendingJobT`` is the package-specific ``PendingJob`` subclass. Subclasses
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


def read_tracker_status(tracker_path: Path) -> dict[str, Any]:
    """Reads a processing tracker file and returns structured per-job status information.

    Args:
        tracker_path: The path to the ``ProcessingTracker`` YAML file.

    Returns:
        A dictionary containing per-job status details in ``jobs`` and summary counts in ``summary``. Each job
        entry has ``job_id``, ``job_name``, ``specifier``, ``status``, and optionally ``error_message`` keys.
    """
    tracker = ProcessingTracker.from_yaml(file_path=tracker_path)

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in tracker.jobs.items():
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {
            "job_id": job_id,
            "job_name": job_state.job_name,
            "specifier": job_state.specifier,
            "status": status.name,
        }
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        job_details.append(entry)

    return {
        "jobs": job_details,
        "summary": {
            "total": len(tracker.jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def derive_tracker_status(summary: dict[str, Any]) -> str:
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if
    any are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise.

    Args:
        summary: A dictionary containing ``total``, ``succeeded``, ``failed``, ``running``, and ``scheduled``
            counts.

    Returns:
        A status string: one of ``failed``, ``completed``, ``processing``, ``not_started``, or ``in_progress``.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0) > 0:
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


def clean_output_subdirectory(output_directory: str, subdirectory_name: str) -> dict[str, Any]:
    """Deletes a named subdirectory under a single output directory.

    Removes the ``<output_directory>/<subdirectory_name>`` tree via
    ``ataraxis_data_structures.delete_directory``, which performs parallel file deletion with platform-safe
    retry logic. Returns structured outcome information suitable for inclusion in MCP tool responses.

    Args:
        output_directory: The absolute path to the parent output directory containing the subdirectory to
            delete.
        subdirectory_name: The name of the subdirectory to delete under ``output_directory``.

    Returns:
        A dictionary containing ``output_directory``, a ``cleaned`` flag, and either ``data_path`` (the path
        that was removed) or ``error`` (a human-readable failure description).
    """
    error = validate_directory(output_directory)
    if error is not None:
        return {"output_directory": output_directory, "cleaned": False, "error": error}

    data_path = Path(output_directory) / subdirectory_name

    if not data_path.exists():
        return {"output_directory": output_directory, "cleaned": True, "message": "Nothing to clean."}

    try:
        delete_directory(directory_path=data_path)
    except Exception as error:
        return {
            "output_directory": output_directory,
            "cleaned": False,
            "data_path": str(data_path),
            "error": f"Unable to delete: {error}",
        }

    return {"output_directory": output_directory, "cleaned": True, "data_path": str(data_path)}


def analyze_feather_file(feather_file: str, max_sample_rows: int) -> dict[str, Any]:
    """Reads a single feather file and computes generic summary statistics.

    Computes the total row count, the list of columns, inter-row timing statistics (when a ``timestamp_us``
    column is present), and a configurable number of sample rows. Columns whose dtype is ``polars.Binary``
    are replaced in the sample rows by a boolean ``<column>_has_data`` flag so the payload stays
    JSON-serializable.

    Args:
        feather_file: The absolute path to the feather file.
        max_sample_rows: The maximum number of sample rows to include.

    Returns:
        A dictionary containing ``file``, ``summary``, ``inter_row_timing``, and ``sample_rows`` keys, or
        ``file`` and ``error`` keys if the file cannot be read.
    """
    file_path = Path(feather_file)

    if not file_path.exists():
        return {"file": feather_file, "error": f"File does not exist: {feather_file}"}

    if not file_path.is_file():
        return {"file": feather_file, "error": f"Path is not a file: {feather_file}"}

    try:
        dataframe = pl.read_ipc(source=file_path)
    except Exception as error:
        return {"file": feather_file, "error": f"Unable to read feather file: {error}"}

    total_rows = dataframe.height

    summary: dict[str, Any] = {"total_rows": total_rows, "columns": dataframe.columns}

    inter_row_timing: dict[str, Any] = {}
    time_column = next((name for name in _TIME_COLUMN_CANDIDATES if name in dataframe.columns), None)
    if time_column is not None and total_rows >= _MINIMUM_ROWS_FOR_INTERVALS:
        timestamps = dataframe[time_column].to_numpy().astype(np.int64)
        first_timestamp_us = int(timestamps[0])
        last_timestamp_us = int(timestamps[-1])
        duration_us = last_timestamp_us - first_timestamp_us
        summary["first_timestamp_us"] = first_timestamp_us
        summary["last_timestamp_us"] = last_timestamp_us
        summary["duration_us"] = duration_us
        summary["duration_seconds"] = (
            round(
                convert_time(
                    time=duration_us, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True
                ),
                6,
            )
            if duration_us > 0
            else 0.0
        )

        intervals_us = np.diff(timestamps)
        inter_row_timing = {
            "mean_us": round(float(np.mean(intervals_us)), 2),
            "median_us": round(float(np.median(intervals_us)), 2),
            "std_us": round(float(np.std(intervals_us)), 2),
            "min_us": int(np.min(intervals_us)),
            "max_us": int(np.max(intervals_us)),
        }

    sample_rows: list[dict[str, Any]] = []
    sample_count = min(max_sample_rows, total_rows)
    if sample_count > 0:
        sample_df = dataframe.head(sample_count)
        binary_columns = {name for name, dtype in dataframe.schema.items() if dtype == pl.Binary}

        for row in sample_df.iter_rows(named=True):
            sample_entry: dict[str, Any] = {}
            for column, value in row.items():
                if column in binary_columns:
                    sample_entry[f"{column}_has_data"] = value is not None
                else:
                    sample_entry[column] = value
            sample_rows.append(sample_entry)

    return {
        "file": feather_file,
        "summary": summary,
        "inter_row_timing": inter_row_timing,
        "sample_rows": sample_rows,
    }
