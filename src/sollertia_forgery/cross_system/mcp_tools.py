"""Provides the system-agnostic Model Context Protocol (MCP) tools for the checksum, manifest generation, and
project manifest inspection pipelines.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from threading import Thread
import contextlib
from collections import deque
from dataclasses import dataclass

from ataraxis_time import (
    TimeUnits,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from ataraxis_base_utilities import resolve_worker_count, resolve_parallel_job_capacity
from sollertia_shared_assets import SessionData, ProcessingTrackers, iterate_sessions, validate_directory
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .checksum import CHECKSUM_JOB_NAME, resolve_checksum
from .manifest import generate_project_manifest
from ..interfaces import mcp
from .orchestration import (
    RESERVED_CORES,
    PendingJob,
    JobExecutionState,
    prepare_tracker,
    read_tracker_status,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
)
from .project_manifest import ProjectManifest

_STATUS_COLUMNS: frozenset[str] = frozenset({"complete", "integrity", "cindra", "behavior", "video"})
"""The manifest column names that store boolean-like UInt8 processing status flags, cast to native bools by
``get_project_manifest_tool`` for readability."""


@mcp.tool()
def get_project_manifest_tool(
    manifest_file: str,
    animal: int | None = None,
    session: str | None = None,
    *,
    include_notes: bool = False,
) -> dict[str, Any]:
    """Reads a project manifest ``.feather`` file and returns structured project metadata with per-session data.

    Loads the manifest via ``ProjectManifest``, computes aggregate statistics with its ``summarize`` method, and
    returns per-session rows as dictionaries. Supports two retrieval
    modes: when ``session`` is provided, returns full data for that single session including experimenter notes,
    which is useful for answering detailed questions about a specific session. When ``session`` is omitted,
    returns data for all sessions (optionally filtered by ``animal``), with notes excluded by default.

    Args:
        manifest_file: The absolute path to the ``.feather`` manifest file.
        animal: An optional animal identifier. When provided, only sessions belonging to that animal are included
            in the ``sessions`` list. Ignored when ``session`` is specified. The ``summary`` always reflects the
            full manifest regardless of this filter.
        session: An optional session identifier for targeted retrieval. When provided, returns full data for that
            single session including experimenter notes, ignoring ``animal`` and ``include_notes``.
        include_notes: Determines whether to include the ``notes`` column in each session entry. Defaults to
            ``False`` to keep responses concise. Ignored when ``session`` is specified, as notes are always
            included in targeted retrieval mode.

    Returns:
        A dictionary containing the ``manifest_file`` path, a ``summary`` with aggregate statistics, a
        ``sessions`` list of per-session dictionaries, and ``total_sessions`` count. Returns an ``error`` key
        on failure.
    """
    file_path = Path(manifest_file)

    if not file_path.exists():
        return {"error": f"Manifest file does not exist: {manifest_file}"}

    if not file_path.is_file():
        return {"error": f"Path is not a file: {manifest_file}"}

    try:
        manifest = ProjectManifest(manifest_file=file_path)
    except Exception as error:
        return {"error": f"Unable to read manifest file: {error}"}

    # Computes aggregate statistics from the full manifest.
    summary = manifest.summarize()

    # Targeted session retrieval mode — returns full data for a single session including notes.
    if session is not None:
        session_df = manifest.get_session_data(session=session)
        if session_df.is_empty():
            available = list(manifest.get_sessions(animal=None, exclude_incomplete=False))
            return {"error": f"Session '{session}' not found in manifest. Available sessions: {available}."}

        row: dict[str, Any] = session_df.to_dicts()[0]
        for column in _STATUS_COLUMNS:
            if column in row:
                row[column] = bool(row[column])

        return {
            "manifest_file": manifest_file,
            "summary": summary,
            "sessions": [row],
            "total_sessions": 1,
        }

    # Validates the animal filter against the manifest's known animals.
    if animal is not None and animal not in manifest.animals:
        return {"error": f"Animal ID '{animal}' not found in manifest. Available animals: {list(manifest.animals)}."}

    # Retrieves per-session data using ProjectManifest's public filtering API.
    session_names = manifest.get_sessions(animal=animal, exclude_incomplete=False)

    session_rows: list[dict[str, Any]] = []

    for session_name in session_names:
        session_df = manifest.get_session_data(session=session_name)
        if session_df.is_empty():
            continue

        row = session_df.to_dicts()[0]

        # Excludes the notes column unless explicitly requested.
        if not include_notes and "notes" in row:
            del row["notes"]

        # Casts boolean-like UInt8 status columns to native bools for readability.
        for column in _STATUS_COLUMNS:
            if column in row:
                row[column] = bool(row[column])

        session_rows.append(row)

    return {
        "manifest_file": manifest_file,
        "summary": summary,
        "sessions": session_rows,
        "total_sessions": len(session_rows),
    }


_CHECKSUM_MAX_WORKERS_PER_JOB: int = 20
"""The hard cap on CPU cores allocated to a single checksum job. Checksum computation shows no throughput benefit
beyond this core count per session."""

_CHECKSUM_PREFERRED_WORKERS_PER_JOB: int = 10
"""The preferred number of CPU cores per checksum job used by saturating allocation. The allocator attempts to run
as many concurrent jobs as possible at this worker count before trading parallelism for per-job throughput."""

_CHECKSUM_MINIMUM_WORKERS_PER_JOB: int = 5
"""The minimum acceptable workers per checksum job when running multiple jobs concurrently. If the budget cannot
sustain this floor with more than one concurrent job, parallelism is reduced until each job meets the minimum."""

_CHECKSUM_WORKER_MULTIPLE: int = 5
"""Worker counts are rounded down to the nearest multiple of this value for clean process-pool sizing."""


@dataclass(slots=True)
class _ChecksumPendingJob(PendingJob):
    """Describes a single checksum resolution job queued for background execution.

    Extends the shared ``PendingJob`` base with the session root path, the session's human-readable name, the
    flag controlling whether the checksum is verified against the stored value or regenerated, and the resolved
    per-job worker count for parallel checksum computation.
    """

    session_path: Path
    """The path to the session root directory containing the session data hierarchy."""
    session_name: str
    """The human-readable session name used for logging and status reporting."""
    regenerate_checksum: bool
    """Determines whether to overwrite the stored checksum instead of verifying it."""
    workers: int = 1
    """The number of CPU cores allocated to this checksum job by the batch-level saturating resolver. Defaults to
    1 so that jobs can be constructed before worker resolution; ``execute_checksum_jobs_tool`` overwrites this
    value after resolving the saturating allocation."""


_checksum_execution_state: JobExecutionState[_ChecksumPendingJob] | None = None
"""Stores the active execution state for batch checksum resolution jobs."""


@mcp.tool()
def prepare_checksum_batch_tool(
    session_paths: list[str],
) -> dict[str, Any]:
    """Prepares an execution manifest for batch checksum resolution without starting execution.

    Accepts confirmed session paths from the caller, loads :class:`SessionData` for each session to resolve the
    ``raw_data`` directory, and initializes a :class:`ProcessingTracker` at
    ``{raw_data_path}/checksum_processing_tracker.yaml``. Idempotent: if a tracker already exists, returns its
    current job registry and status instead of reinitializing.

    Important:
        The AI agent calling this tool MUST run :func:`discover_sessions_tool` first to obtain confirmed session
        root paths. Do not assume or guess session paths. Each session produces exactly one checksum resolution
        job identified by the ``checksum_resolution`` job name and the session name as the specifier.

    Args:
        session_paths: The list of absolute paths to session root directories. Accepts paths from the
            ``session_paths`` list returned by :func:`discover_sessions_tool`.

    Returns:
        A dictionary containing per-session manifests in ``sessions`` with tracker paths and job lists, total
        counts, and any invalid session paths.
    """
    result_sessions: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for session_path_str in session_paths:
        session_path = Path(session_path_str)

        if not session_path.exists() or not session_path.is_dir():
            invalid_paths.append(session_path_str)
            continue

        try:
            session = SessionData.load(session_path=session_path)
        except Exception as error:
            result_sessions[session_path_str] = {
                "error": f"Unable to load session: {error}",
                "jobs": [],
                "summary": {},
            }
            continue

        tracker_path = session.raw_data.checksum_tracker_path
        expected_jobs: list[tuple[str, str]] = [(CHECKSUM_JOB_NAME, session.session_name)]

        # Initializes the tracker with stale entry detection. If the tracker already exists, foreign entries
        # are detected and the tracker is reset before reinitializing with the expected job set. If the tracker
        # does not exist, it is created from scratch with the expected jobs.
        tracker = ProcessingTracker(file_path=tracker_path)
        prepare_tracker(tracker=tracker, jobs=expected_jobs)

        # Reads the (possibly just-repaired or freshly-created) tracker state to return to the caller.
        try:
            tracker_status = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_status = {"jobs": [], "summary": {}}

        # Augments each tracker entry with dispatch metadata so the caller can feed the manifest directly
        # into execute_checksum_jobs_tool without re-deriving per-job fields.
        enriched_jobs: list[dict[str, Any]] = [
            {
                **tracker_entry,
                "session_path": session_path_str,
                "session_name": session.session_name,
                "tracker_path": str(tracker_path),
            }
            for tracker_entry in tracker_status.get("jobs", [])
        ]

        result_sessions[session_path_str] = {
            "tracker_path": str(tracker_path),
            "session_name": session.session_name,
            "jobs": enriched_jobs,
            "summary": tracker_status.get("summary", {}),
        }
        total_jobs += len(enriched_jobs)

    result: dict[str, Any] = {
        "success": True,
        "sessions": result_sessions,
        "total_sessions": len(result_sessions),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def execute_checksum_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    workers_per_job: int = -1,
    max_parallel_jobs: int = -1,
    regenerate_checksum: bool = False,
) -> dict[str, Any]:
    """Dispatches checksum resolution jobs for background execution with saturating core allocation.

    Takes job descriptors from the manifest produced by :func:`prepare_checksum_batch_tool` and starts a
    background execution manager that runs each job in a separate worker subprocess. The CPU budget is
    distributed across concurrent jobs using a saturating allocation strategy: the allocator maximizes
    parallelism at a preferred per-job worker count, reduces parallelism when workers would drop below a
    minimum floor, and enforces a hard cap of ``_CHECKSUM_MAX_WORKERS_PER_JOB`` cores per job. Each job's
    resolved worker count is bound to its pending descriptor and forwarded to :func:`resolve_checksum` at
    dispatch time.

    Important:
        Only one checksum execution session can be active at a time. Use :func:`cancel_checksum_tool` to cancel
        an active session before starting a new one.

    Args:
        jobs: The list of job descriptors from :func:`prepare_checksum_batch_tool`. Each dictionary must have
            ``session_path``, ``tracker_path``, and ``job_id`` keys.
        workers_per_job: The number of CPU cores to allocate to each individual checksum job. Set to -1 for
            automatic resolution via saturating allocation. Capped at ``_CHECKSUM_MAX_WORKERS_PER_JOB``
            regardless of the requested value.
        max_parallel_jobs: The maximum number of checksum jobs to execute concurrently. Set to -1 for automatic
            resolution based on the available CPU budget and the resolved per-job worker count.
        regenerate_checksum: Determines whether to overwrite the stored checksum instead of verifying it. Applies
            to all jobs in the batch.

    Returns:
        A dictionary containing a ``started`` flag, ``total_jobs``, ``workers_per_job``, ``max_parallel_jobs``,
        and any invalid jobs.
    """
    global _checksum_execution_state

    # Enforces single-session constraint.
    if (
        _checksum_execution_state is not None
        and _checksum_execution_state.manager_thread is not None
        and _checksum_execution_state.manager_thread.is_alive()
    ):
        return {"error": "A checksum execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"session_path", "tracker_path", "job_id"}
    pending: list[_ChecksumPendingJob] = []
    all_jobs: dict[tuple[str, str], _ChecksumPendingJob] = {}
    invalid_jobs: list[dict[str, Any]] = []

    for job_dict in jobs:
        missing = required_keys - job_dict.keys()
        if missing:
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {sorted(missing)}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        pending_job = _ChecksumPendingJob(
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            session_path=Path(job_dict["session_path"]),
            session_name=str(job_dict.get("session_name", Path(job_dict["session_path"]).name)),
            regenerate_checksum=regenerate_checksum,
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves per-job worker count and maximum concurrent jobs using saturating allocation. The four resolution
    # scenarios mirror the cindra pipeline pattern: fully automatic, fixed workers with automatic parallelism,
    # automatic workers with fixed parallelism, and both explicitly specified.
    budget = resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES)
    total_jobs = max(1, len(pending))

    if workers_per_job <= 0 and max_parallel_jobs <= 0:
        actual_workers, actual_max_parallel = _resolve_checksum_saturating_allocation(
            budget=budget, total_jobs=total_jobs
        )
    elif workers_per_job > 0 >= max_parallel_jobs:
        actual_workers = min(
            resolve_worker_count(requested_workers=workers_per_job, reserved_cores=RESERVED_CORES),
            _CHECKSUM_MAX_WORKERS_PER_JOB,
        )
        actual_max_parallel = resolve_parallel_job_capacity(workers_per_job=actual_workers)
    elif workers_per_job <= 0 < max_parallel_jobs:
        raw_workers = budget // max_parallel_jobs
        actual_workers = min(
            max(1, (raw_workers // _CHECKSUM_WORKER_MULTIPLE) * _CHECKSUM_WORKER_MULTIPLE),
            _CHECKSUM_MAX_WORKERS_PER_JOB,
        )
        actual_max_parallel = max_parallel_jobs
    else:
        actual_workers = min(
            resolve_worker_count(requested_workers=workers_per_job, reserved_cores=RESERVED_CORES),
            _CHECKSUM_MAX_WORKERS_PER_JOB,
        )
        actual_max_parallel = max_parallel_jobs

    # Binds the resolved worker count to each pending job so the worker subprocess receives it at dispatch time.
    for pending_job in pending:
        pending_job.workers = actual_workers

    # Creates the execution state and starts the shared manager thread. The worker_budget is set to the maximum
    # number of concurrent jobs rather than total cores, since each job subprocess internally spawns its own
    # worker pool sized to actual_workers.
    _checksum_execution_state = JobExecutionState[_ChecksumPendingJob](
        worker=_run_checksum_job,
        all_jobs=all_jobs,
        pending_queue=deque(pending),
        worker_budget=actual_max_parallel,
    )

    manager = Thread(
        target=job_execution_manager,
        kwargs={"state": _checksum_execution_state},
        daemon=True,
    )
    manager.start()
    _checksum_execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "workers_per_job": actual_workers,
        "max_parallel_jobs": actual_max_parallel,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()
def get_checksum_status_tool() -> dict[str, Any]:
    """Returns the current status of the active checksum resolution execution session.

    Reads :class:`ProcessingTracker` files from disk for each job to report per-job progress. When no execution
    session exists, returns an inactive status.

    Returns:
        A dictionary containing an ``active`` flag, per-job status entries in ``jobs``, and a ``summary`` with
        counts for scheduled, running, succeeded, and failed jobs.
    """
    if _checksum_execution_state is None:
        return {"active": False, "message": "No checksum execution session exists."}

    state = _checksum_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    # Groups jobs by tracker so each YAML file is deserialized only once per status refresh.
    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:
            job_details.extend(
                {
                    "job_id": job.job_id,
                    "session_name": job.session_name,
                    "status": "UNKNOWN",
                }
                for job in path_jobs
            )
            continue

        # Resolves per-job status from the tracker and tallies outcomes for the aggregate summary.
        for job in path_jobs:
            if job.job_id in tracker.jobs:
                job_state = tracker.jobs[job.job_id]
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
                    "job_id": job.job_id,
                    "session_name": job.session_name,
                    "status": status.name,
                }
                if job_state.error_message is not None:
                    entry["error_message"] = job_state.error_message
                job_details.append(entry)
            else:
                job_details.append(
                    {
                        "job_id": job.job_id,
                        "session_name": job.session_name,
                        "status": "UNKNOWN",
                    }
                )

    return {
        "active": manager_alive,
        "canceled": state.canceled,
        "jobs": job_details,
        "summary": {
            "total": len(state.all_jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


@mcp.tool()
def get_checksum_timing_tool() -> dict[str, Any]:
    """Returns timing information for all jobs in the active checksum resolution execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps recorded in the :class:`ProcessingTracker`.

    Returns:
        A dictionary containing an ``active`` flag, per-job timing in ``jobs``, and a ``session`` summary with
        total elapsed seconds and throughput.
    """
    if _checksum_execution_state is None:
        return {"active": False, "message": "No checksum execution session exists."}

    state = _checksum_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    # Captures the current timestamp once for computing elapsed time on running jobs.
    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    job_timing: list[dict[str, Any]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0

    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:  # noqa: S112
            continue

        for job in path_jobs:
            if job.job_id not in tracker.jobs:
                continue

            job_info = tracker.jobs[job.job_id]
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "session_name": job.session_name,
            }

            # Tracks the earliest start timestamp across all jobs for session-level elapsed time.
            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            # Computes wall-clock elapsed seconds for jobs that are still running.
            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)

            # Computes total duration for jobs that have reached a terminal state.
            if job_info.completed_at is not None:
                entry["completed_at"] = int(job_info.completed_at)
                if job_info.started_at is not None:
                    duration_seconds = convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    )
                    entry["duration_seconds"] = round(duration_seconds, 2)

            if job_info.status == ProcessingStatus.SUCCEEDED:
                completed_count += 1
            elif job_info.status == ProcessingStatus.FAILED:
                failed_count += 1

            job_timing.append(entry)

    # Computes session-level elapsed time from the earliest job start to now.
    total_elapsed_seconds = 0.0
    if earliest_start is not None:
        total_elapsed_seconds = round(
            convert_time(
                time=current_us - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            2,
        )

    running_count = sum(1 for job_entry in job_timing if "elapsed_seconds" in job_entry)
    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": len(state.all_jobs) - completed_count - failed_count - running_count,
    }

    # Derives throughput as completed jobs per hour since the earliest start.
    if completed_count > 0 and earliest_start is not None:
        elapsed_hours = convert_time(
            time=current_us - earliest_start,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.HOUR,
            as_float=True,
        )
        if elapsed_hours > 0:
            session["throughput_jobs_per_hour"] = round(completed_count / elapsed_hours, 2)

    return {"active": manager_alive, "jobs": job_timing, "session": session}


@mcp.tool()
def cancel_checksum_tool() -> dict[str, Any]:
    """Cancels the active checksum resolution execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a ``canceled`` flag, a ``message``, and ``final_state`` with counts for
        succeeded, failed, and active jobs at the time of cancellation.
    """
    if _checksum_execution_state is None:
        return {"canceled": False, "message": "No checksum execution session is active."}

    state = _checksum_execution_state

    # Atomically halts dispatch and snapshots queue sizes before reading trackers.
    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_jobs)

    # Reads tracker files to tally terminal outcomes for the final state report.
    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        with contextlib.suppress(Exception):
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
            for job_state in tracker.jobs.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1

    return {
        "canceled": True,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} job(s) still completing.",
        "final_state": {
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "active_jobs_at_cancel": active_count,
        },
    }


@mcp.tool()
def reset_checksum_jobs_tool(
    tracker_path: str,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a checksum tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the checksum :class:`ProcessingTracker` YAML file.
        job_ids: An optional list of job IDs (the hexadecimal identifiers returned by
            :func:`prepare_checksum_batch_tool`) to reset. If not provided, every job in the tracker is reset.

    Returns:
        A dictionary containing a ``reset`` flag, the number of jobs reset, and updated job statuses.
    """
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Identifies which job IDs to reset. When the caller does not specify a filter, every job in the tracker
    # is reset. Otherwise, the filter is intersected with the tracker's job registry so that stale or unknown
    # IDs are silently ignored.
    tracker_ids = set(tracker.jobs.keys())
    target_ids = tracker_ids if job_ids is None else tracker_ids & set(job_ids)

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Collects (job_name, specifier) tuples for the jobs to reset, then rebuilds the tracker registry so that
    # every reset entry returns to SCHEDULED state with cleared timing and error metadata.
    reset_jobs: list[tuple[str, str]] = [
        (tracker.jobs[job_id].job_name, tracker.jobs[job_id].specifier) for job_id in target_ids
    ]

    for job_id in target_ids:
        del tracker.jobs[job_id]
    tracker.to_yaml(file_path=path)

    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    try:
        updated_status = read_tracker_status(tracker_path=path)
    except Exception:
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_checksum_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:
    """Discovers and summarizes checksum resolution status for all sessions under a root directory.

    Iterates every session marker under the root via :func:`iterate_sessions` and, for each session whose
    canonical ``SessionData.checksum_tracker_path`` exists, reads the tracker and aggregates its status.

    Args:
        root_directory: The absolute path to the root directory to search for sessions.

    Returns:
        A dictionary containing per-session status summaries and aggregate counts.
    """
    error = validate_directory(root_directory)
    if error is not None:
        return {"error": error}

    root_path = Path(root_directory)
    session_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for session in iterate_sessions(root_path=root_path):
        tracker_path = session.raw_data.checksum_tracker_path
        if not tracker_path.is_file():
            continue

        session_root = session.raw_data_path.parent
        try:
            # Reads the tracker and accumulates per-session counts into the project-wide aggregate.
            status = read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})

            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            dir_status = derive_tracker_status(summary=summary)

            session_statuses.append(
                {
                    "session_path": str(session_root),
                    "tracker_path": str(tracker_path),
                    "status": dir_status,
                    **status,
                }
            )
        except Exception:
            session_statuses.append(
                {
                    "session_path": str(session_root),
                    "tracker_path": str(tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    return {
        "sessions": session_statuses,
        "total_sessions": len(session_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()
def clean_checksum_tracker_tool(session_paths: list[str]) -> dict[str, Any]:
    """Deletes checksum processing tracker files and their lock files for one or more sessions.

    For each session, loads :class:`SessionData` to resolve ``raw_data_path``, then removes
    ``checksum_processing_tracker.yaml`` and its companion ``.lock`` file from the ``raw_data`` directory.
    The tool refuses to run while a checksum execution session is still active.

    Important:
        The AI agent calling this tool SHOULD call :func:`get_checksum_batch_status_overview_tool` first to
        confirm all jobs have reached a terminal state before cleaning up trackers.

    Args:
        session_paths: The list of absolute paths to session root directories whose checksum tracker files
            should be deleted.

    Returns:
        A dictionary containing a ``results`` list with per-session outcomes and a ``total_cleaned`` count.
    """
    # Refuses to clean while an execution session is still writing to tracker files.
    if (
        _checksum_execution_state is not None
        and _checksum_execution_state.manager_thread is not None
        and _checksum_execution_state.manager_thread.is_alive()
    ):
        return {"error": "A checksum execution session is still active. Wait for completion or cancel first."}

    results: list[dict[str, Any]] = []

    for session_path_str in session_paths:
        session_path = Path(session_path_str)

        if not session_path.exists() or not session_path.is_dir():
            results.append(
                {"session_path": session_path_str, "cleaned": False, "error": "Session path does not exist."}
            )
            continue

        try:
            session = SessionData.load(session_path=session_path)
        except Exception as load_error:
            results.append(
                {"session_path": session_path_str, "cleaned": False, "error": f"Unable to load session: {load_error}"}
            )
            continue

        # Resolves the tracker and lock file paths from the session's raw_data directory.
        tracker_file = session.raw_data.checksum_tracker_path
        lock_file = tracker_file.with_suffix(tracker_file.suffix + ".lock")
        deleted_files: list[str] = []

        # Removes both the tracker YAML and its companion lock file if they exist.
        try:
            for target in (tracker_file, lock_file):
                if target.exists():
                    target.unlink()
                    deleted_files.append(target.name)

            results.append(
                {
                    "session_path": session_path_str,
                    "cleaned": True,
                    "deleted_files": deleted_files,
                }
            )
        except Exception as delete_error:
            results.append(
                {"session_path": session_path_str, "cleaned": False, "error": f"Unable to delete: {delete_error}"}
            )

    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_sessions": len(results)}


@mcp.tool()
def generate_project_manifest_tool(project_directory: str) -> dict[str, Any]:
    """Generates a project manifest ``.feather`` file capturing the snapshot of a project's state.

    Calls :func:`generate_project_manifest` to scan the entire project for sessions, processing status, and
    multi-recording datasets, then writes the result to ``{project_name}_manifest.feather`` in the project root.
    A :class:`ProcessingTracker` at ``{project_root}/manifest_processing_tracker.yaml`` records the outcome.

    Important:
        The AI agent calling this tool MUST confirm the project directory path with the user before calling.
        Manifest generation scans every session in the project and may take several seconds for large projects.
        After generation completes, use :func:`get_project_manifest_tool` to read the generated manifest.

    Args:
        project_directory: The absolute path to the project's root directory.

    Returns:
        A dictionary containing a ``success`` flag, ``project_directory``, ``manifest_file`` path, and
        ``tracker_path``. Returns an ``error`` key on failure.
    """
    error = validate_directory(project_directory)
    if error is not None:
        return {"error": error}

    # Resolves expected output paths so they can be reported in the response even on failure.
    project_path = Path(project_directory)
    manifest_file = project_path / f"{project_path.stem}_manifest.feather"
    tracker_file = project_path / ProcessingTrackers.MANIFEST

    # Delegates to the pipeline function which handles tracker lifecycle, session scanning, and feather output.
    try:
        generate_project_manifest(project_directory=project_path)
    except Exception as generation_error:
        return {
            "error": f"Manifest generation failed: {generation_error}",
            "project_directory": project_directory,
            "tracker_path": str(tracker_file),
        }

    return {
        "success": True,
        "project_directory": project_directory,
        "manifest_file": str(manifest_file),
        "tracker_path": str(tracker_file),
    }


@mcp.tool()
def get_manifest_generation_status_tool(project_directory: str) -> dict[str, Any]:
    """Returns the status of the most recent manifest generation for a project.

    Reads the :class:`ProcessingTracker` at ``{project_root}/manifest_processing_tracker.yaml`` and returns
    its job status. Useful for checking whether manifest generation has been run and whether it succeeded.

    Args:
        project_directory: The absolute path to the project's root directory.

    Returns:
        A dictionary containing the ``tracker_path`` and per-job status information from the tracker, or an
        ``error`` key if the tracker does not exist.
    """
    error = validate_directory(project_directory)
    if error is not None:
        return {"error": error}

    tracker_path = Path(project_directory) / ProcessingTrackers.MANIFEST

    if not tracker_path.exists():
        return {
            "error": f"No manifest tracker found at '{tracker_path}'. Manifest generation may not have been run.",
            "project_directory": project_directory,
        }

    try:
        tracker_status = read_tracker_status(tracker_path=tracker_path)
    except Exception as read_error:
        return {"error": f"Unable to read manifest tracker: {read_error}", "tracker_path": str(tracker_path)}

    return {
        "tracker_path": str(tracker_path),
        "project_directory": project_directory,
        **tracker_status,
    }


@mcp.tool()
def clean_project_manifest_tool(project_directory: str) -> dict[str, Any]:
    """Deletes all manifest generation artifacts from a project's root directory.

    Removes the ``manifest_processing_tracker.yaml`` tracker, the ``{project_name}_manifest.feather`` data file,
    and their companion ``.lock`` files. After cleanup, :func:`generate_project_manifest_tool` can be called to
    regenerate the manifest from scratch.

    Args:
        project_directory: The absolute path to the project's root directory.

    Returns:
        A dictionary containing a ``cleaned`` flag, the ``project_directory``, and a ``deleted_files`` list
        naming each removed file, or an ``error`` key on failure.
    """
    error = validate_directory(project_directory)
    if error is not None:
        return {"error": error}

    # Resolves paths for all manifest artifacts: tracker, data file, and their companion lock files.
    project_path = Path(project_directory)
    tracker_file = project_path / ProcessingTrackers.MANIFEST
    manifest_file = project_path / f"{project_path.stem}_manifest.feather"
    manifest_lock = manifest_file.with_suffix(manifest_file.suffix + ".lock")
    tracker_lock = tracker_file.with_suffix(tracker_file.suffix + ".lock")

    deleted_files: list[str] = []

    # Removes each artifact if present, accumulating the list of deleted filenames for the response.
    try:
        for target in (tracker_file, tracker_lock, manifest_file, manifest_lock):
            if target.exists():
                target.unlink()
                deleted_files.append(target.name)
    except Exception as delete_error:
        return {
            "error": f"Unable to delete: {delete_error}",
            "project_directory": project_directory,
            "deleted_files": deleted_files,
        }

    return {"cleaned": True, "project_directory": project_directory, "deleted_files": deleted_files}


def _resolve_checksum_saturating_allocation(budget: int, total_jobs: int) -> tuple[int, int]:
    """Resolves per-job worker count and maximum parallel job count for checksum batch execution.

    Distributes the CPU budget across as many concurrent jobs as possible at the preferred worker count, then
    reduces parallelism until each job meets the minimum worker floor. Worker counts are rounded down to the
    nearest multiple of ``_CHECKSUM_WORKER_MULTIPLE`` and capped at ``_CHECKSUM_MAX_WORKERS_PER_JOB``.

    Args:
        budget: The total number of available CPU cores after reserving system cores.
        total_jobs: The total number of checksum jobs to execute.

    Returns:
        A ``(workers_per_job, max_parallel_jobs)`` tuple.
    """
    max_at_preferred = max(1, budget // _CHECKSUM_PREFERRED_WORKERS_PER_JOB)
    max_parallel = min(total_jobs, max_at_preferred)
    raw_workers = budget // max_parallel
    workers = min(
        max(1, (raw_workers // _CHECKSUM_WORKER_MULTIPLE) * _CHECKSUM_WORKER_MULTIPLE),
        _CHECKSUM_MAX_WORKERS_PER_JOB,
    )

    # Reduces parallelism until each job has at least the minimum worker count.
    while workers < _CHECKSUM_MINIMUM_WORKERS_PER_JOB and max_parallel > 1:
        max_parallel -= 1
        raw_workers = budget // max_parallel
        workers = min(
            max(1, (raw_workers // _CHECKSUM_WORKER_MULTIPLE) * _CHECKSUM_WORKER_MULTIPLE),
            _CHECKSUM_MAX_WORKERS_PER_JOB,
        )

    return workers, max_parallel


def _run_checksum_job(job: _ChecksumPendingJob) -> None:
    """Executes a single checksum resolution job in-process via ``resolve_checksum``.

    Serves as the picklable worker callable stored on ``JobExecutionState`` and dispatched to the batch
    manager's ``ProcessPoolExecutor``. Delegates to ``resolve_checksum`` with the job's session path and the
    resolved per-job worker count bound at dispatch time by ``execute_checksum_jobs_tool``. The
    ``resolve_checksum`` function internally manages its own ``ProcessingTracker`` lifecycle (idempotent
    initialization, start, complete, or fail).

    Args:
        job: The pending job descriptor produced by ``prepare_checksum_batch_tool`` and attached to the active
            ``JobExecutionState`` by ``execute_checksum_jobs_tool``.
    """
    resolve_checksum(
        session_path=job.session_path,
        regenerate_checksum=job.regenerate_checksum,
        workers=job.workers,
    )
