"""Provides Model Context Protocol (MCP) tools for the behavior processing pipeline."""

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
from ataraxis_base_utilities import resolve_worker_count
from sollertia_shared_assets import Directories, SessionData, iterate_sessions, validate_directory
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .pipeline import (
    discover_behavior_jobs,
    run_behavior_processing_pipeline,
)
from ..interfaces import mcp
from ..shared_assets import (
    RESERVED_CORES,
    PendingJob,
    JobExecutionState,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)


@dataclass(slots=True)
class _BehaviorPendingJob(PendingJob):
    """Describes a single behavior processing job queued for background execution.

    Extends the shared ``PendingJob`` base with the session-level metadata required by the behavior worker
    callable: the session root path, the session's human-readable name, and the ``(job_name, specifier)`` pair
    that identifies the atomic unit of work inside the session's ``ProcessingTracker`` registry. The per-job
    output directory is not stored on the descriptor because behavior outputs always live under
    ``{session.processed_data_path}/behavior_data/``; the worker resolves that path from ``session_path`` at
    dispatch time via ``SessionData``.
    """

    session_path: Path
    """The path to the session root directory containing the session data hierarchy."""
    session_name: str
    """The human-readable session name used for logging and status reporting."""
    job_name: str
    """The behavior job type name (one of ``runtime_processing``, ``camera_processing``, or
    ``microcontroller_processing``)."""
    specifier: str
    """The job-specific specifier that differentiates jobs of the same type (system ID, camera source ID, or
    ``controller-type-id`` triple)."""


_job_execution_state: JobExecutionState[_BehaviorPendingJob] | None = None
"""Stores the active execution state for batch behavior processing jobs."""


@mcp.tool()
def prepare_behavior_processing_batch_tool(
    session_paths: list[str],
) -> dict[str, Any]:
    """Prepares an execution manifest for batch behavior processing without starting execution.

    Accepts confirmed session paths from the caller, discovers the behavior jobs available for each session via
    :func:`discover_behavior_jobs`, and initializes a :class:`ProcessingTracker` under each session's
    ``{processed_data_path}/behavior_data/`` subdirectory. The output location is static — behavior outputs
    always live under the session's ``processed_data_path`` and the caller never chooses where they go.
    Idempotent: if a tracker already exists, returns its current job registry and status instead of
    reinitializing.

    Important:
        The AI agent calling this tool MUST run :func:`discover_behavior_sessions_tool` first to obtain
        confirmed session root paths. Do not assume or guess session paths. The output directory is resolved
        statically from the session's :class:`SessionData` marker — there is no user-selectable output location.

    Args:
        session_paths: The list of absolute paths to session root directories. Accepts paths from the
            ``session_paths`` list returned by :func:`discover_behavior_sessions_tool`.

    Returns:
        A dictionary containing per-session manifests in ``sessions`` with tracker paths and job lists, total
        counts, and any invalid session paths. Each per-session entry carries a ``data_path`` field that
        identifies where ``behavior_data/`` was created under the session's processed data hierarchy.
    """
    result_sessions: dict[str, Any] = {}
    invalid_paths: list[str] = []
    total_jobs = 0

    for session_path_str in session_paths:
        session_path = Path(session_path_str)

        if not session_path.exists() or not session_path.is_dir():
            invalid_paths.append(session_path_str)
            continue

        # Runs discovery for the session. Discovery loads the SessionData, validates the session type, and
        # enumerates the runtime, camera, and microcontroller jobs available on disk. Failures are surfaced per
        # session so that other sessions in the batch are unaffected.
        try:
            session, discovered_jobs = discover_behavior_jobs(session_path=session_path)
        except Exception as error:
            result_sessions[session_path_str] = {
                "error": f"Discovery failed: {error}",
                "jobs": [],
                "summary": {},
            }
            continue

        if not discovered_jobs:
            result_sessions[session_path_str] = {
                "error": "No processable behavior jobs discovered for this session.",
                "jobs": [],
                "summary": {},
            }
            continue

        # Resolves the static output location. The ``behavior_data/`` subdirectory always lives under the
        # session's ``processed_data_path``, co-located with the upstream ``camera_timestamps/`` and
        # ``microcontroller_data/`` produced by axvs and axci. The caller does not choose where behavior
        # outputs go.
        data_path = session.behavior_data_path
        data_path.mkdir(parents=True, exist_ok=True)
        tracker_path = session.behavior_tracker_path

        if tracker_path.exists():
            # Idempotent path: returns existing tracker state without rebuilding the job registry.
            try:
                tracker_status = read_tracker_status(tracker_path=tracker_path)
            except Exception:
                tracker_status = {"jobs": [], "summary": {}}

            tracker_jobs_by_id = {
                ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
                for job_name, specifier in discovered_jobs
            }

            # Augments each tracker entry with the enclosing dispatch metadata so the caller can feed the manifest
            # directly into execute_behavior_processing_jobs_tool without re-deriving per-job fields.
            enriched_jobs: list[dict[str, Any]] = []
            for tracker_entry in tracker_status.get("jobs", []):
                entry_job_id = tracker_entry["job_id"]
                if entry_job_id not in tracker_jobs_by_id:
                    continue
                job_name, specifier = tracker_jobs_by_id[entry_job_id]
                enriched_jobs.append(
                    {
                        **tracker_entry,
                        "job_name": job_name,
                        "specifier": specifier,
                        "session_path": session_path_str,
                        "session_name": session.session_name,
                        "tracker_path": str(tracker_path),
                    }
                )

            result_sessions[session_path_str] = {
                "tracker_path": str(tracker_path),
                "data_path": str(data_path),
                "session_name": session.session_name,
                "jobs": enriched_jobs,
                "summary": tracker_status.get("summary", {}),
            }
            total_jobs += len(enriched_jobs)
            continue

        # Initializes a new tracker with jobs for the discovered (job_name, specifier) tuples. The helper
        # autoloads on construction so the ProcessingTracker created here is immediately backed by the YAML
        # file once initialize_jobs persists the registry.
        tracker = ProcessingTracker(file_path=tracker_path)
        tracker.initialize_jobs(jobs=discovered_jobs)

        jobs: list[dict[str, Any]] = [
            {
                "job_id": ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier),
                "job_name": job_name,
                "specifier": specifier,
                "status": ProcessingStatus.SCHEDULED.name,
                "session_path": session_path_str,
                "session_name": session.session_name,
                "tracker_path": str(tracker_path),
            }
            for job_name, specifier in discovered_jobs
        ]

        result_sessions[session_path_str] = {
            "tracker_path": str(tracker_path),
            "data_path": str(data_path),
            "session_name": session.session_name,
            "jobs": jobs,
            "summary": {
                "total": len(jobs),
                "succeeded": 0,
                "failed": 0,
                "running": 0,
                "scheduled": len(jobs),
            },
        }
        total_jobs += len(jobs)

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
def execute_behavior_processing_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    worker_budget: int = -1,
) -> dict[str, Any]:
    """Dispatches behavior processing jobs for background execution with budget-bounded concurrency.

    Takes job descriptors from the manifest produced by :func:`prepare_behavior_processing_batch_tool` and starts
    a background execution manager that runs each job in a separate worker subprocess. Each job invokes
    :func:`run_behavior_processing_pipeline` in remote mode with the descriptor's ``job_id`` so that only that
    single (job_name, specifier) pair is executed against the session. The worker budget directly controls memory
    footprint since each worker spawns a separate process -- it also bounds the maximum number of concurrently
    executing behavior jobs.

    Important:
        Only one execution session can be active at a time. Use :func:`cancel_behavior_processing_tool` to cancel
        an active session before starting a new one.

    Args:
        jobs: The list of job descriptors from :func:`prepare_behavior_processing_batch_tool`. Each dictionary
            must have 'session_path', 'tracker_path', 'job_id', 'job_name', and 'specifier' keys. The output
            location is not part of the descriptor — workers resolve it from the session's
            :class:`SessionData` marker at dispatch time.
        worker_budget: The total number of CPU cores available for the execution session. Directly controls
            memory footprint. Set to -1 for automatic resolution via
            :func:`ataraxis_base_utilities.resolve_worker_count`.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved worker budget, and any invalid jobs.
    """
    global _job_execution_state

    # Enforces single-session constraint.
    if (
        _job_execution_state is not None
        and _job_execution_state.manager_thread is not None
        and _job_execution_state.manager_thread.is_alive()
    ):
        return {"error": "An execution session is already active. Cancel it first or wait for completion."}

    # Validates and builds pending jobs.
    required_keys = {"session_path", "tracker_path", "job_id", "job_name", "specifier"}
    pending: list[_BehaviorPendingJob] = []
    all_jobs: dict[tuple[str, str], _BehaviorPendingJob] = {}
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

        pending_job = _BehaviorPendingJob(
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            session_path=Path(job_dict["session_path"]),
            session_name=str(job_dict.get("session_name", Path(job_dict["session_path"]).name)),
            job_name=job_dict["job_name"],
            specifier=job_dict["specifier"],
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the total worker budget. The budget simultaneously caps the process pool size and the maximum
    # number of behavior jobs that can execute concurrently.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=RESERVED_CORES)

    # Creates the execution state and starts the shared manager thread. The manager owns the
    # ProcessPoolExecutor and dispatches pending jobs as budget becomes available, invoking
    # ``_run_behavior_job`` for each job it picks up.
    _job_execution_state = JobExecutionState[_BehaviorPendingJob](
        worker=_run_behavior_job,
        all_jobs=all_jobs,
        pending_queue=deque(pending),
        worker_budget=resolved_budget,
    )

    manager = Thread(
        target=job_execution_manager,
        kwargs={"state": _job_execution_state},
        daemon=True,
    )
    manager.start()
    _job_execution_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()
def get_behavior_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of the active behavior processing execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. When no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts
        for pending, running, succeeded, and failed jobs.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
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
                    "job_name": job.job_name,
                    "specifier": job.specifier,
                    "session_name": job.session_name,
                    "status": "UNKNOWN",
                }
                for job in path_jobs
            )
            continue

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
                    "job_name": job.job_name,
                    "specifier": job.specifier,
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
                        "job_name": job.job_name,
                        "specifier": job.specifier,
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
def get_behavior_processing_timing_tool() -> dict[str, Any]:
    """Returns timing information for all jobs in the active behavior processing execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps recorded in the ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with total
        elapsed seconds and throughput.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    state = _job_execution_state
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
                "job_name": job.job_name,
                "specifier": job.specifier,
                "session_name": job.session_name,
            }

            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)

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
def cancel_behavior_processing_tool() -> dict[str, Any]:
    """Cancels the active behavior processing execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    if _job_execution_state is None:
        return {"canceled": False, "message": "No execution session is active."}

    state = _job_execution_state

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_jobs)

    # Counts final job statuses from tracker files.
    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}

    for tracker_path in tracker_paths:
        # Skips trackers that cannot be deserialized so that one unreadable file does not suppress the final
        # tally for every other tracker in the batch.
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
def reset_behavior_processing_jobs_tool(
    tracker_path: str,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the behavior processing :class:`ProcessingTracker` YAML file.
        job_ids: An optional list of job IDs (the hexadecimal identifiers returned by
            :func:`prepare_behavior_processing_batch_tool`) to reset. If not provided, every job in the tracker
            is reset.

    Returns:
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses.
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
    # IDs in the caller's request are silently ignored.
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
def get_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:
    """Discovers and summarizes behavior processing status for all sessions under a root directory.

    Iterates every session marker under the root via :func:`iterate_sessions` and, for each session whose
    canonical ``SessionData.behavior_tracker_path`` exists, reads the tracker and aggregates its status.

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
        tracker_path = session.behavior_tracker_path
        if not tracker_path.is_file():
            continue

        data_path = session.behavior_data_path
        session_root = session.raw_data_path.parent
        try:
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
                    "data_path": str(data_path),
                    "tracker_path": str(tracker_path),
                    "status": dir_status,
                    **status,
                }
            )
        except Exception:
            session_statuses.append(
                {
                    "session_path": str(session_root),
                    "data_path": str(data_path),
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
def verify_behavior_processing_output_tool(session_path: str) -> dict[str, Any]:
    """Verifies the completeness of processed behavior data output for a single session.

    Loads the session's :class:`SessionData` marker to resolve ``processed_data_path``, then scans the
    ``behavior_data/`` subdirectory for feather files produced by the behavior processing pipeline. Each
    feather file is loaded to confirm it is readable and to report its row and column counts. The processing
    tracker is also read to report per-job statuses alongside the output file inventory.

    Args:
        session_path: The absolute path to the session root directory. The session's ``processed_data_path``
            is resolved from :class:`SessionData`, and verification inspects
            ``{processed_data_path}/behavior_data/``.

    Returns:
        A dictionary containing a 'verified' flag, per-file results in 'files' (each with path, readability,
        row count, and column names), tracker status in 'tracker', and aggregate counts.
    """
    error = validate_directory(session_path)
    if error is not None:
        return {"error": error}

    session_root = Path(session_path)

    try:
        session = SessionData.load(session_path=session_root)
    except Exception as error:
        return {"error": f"Unable to load session: {error}"}

    data_path = session.behavior_data_path

    if not data_path.exists():
        return {
            "error": (
                f"No '{Directories.BEHAVIOR_DATA}' subdirectory found under '{session.processed_data_path}'. "
                f"Processing may not have been run yet."
            ),
        }

    file_results: list[dict[str, Any]] = []
    all_valid = True

    feather_files = sorted(data_path.glob("*.feather"))

    for feather_file in feather_files:
        # Reuses the shared feather inspector with zero sample rows, since verify only needs the row count
        # and column list. ``analyze_feather_file`` returns an ``error`` key when the file cannot be read.
        analysis = analyze_feather_file(feather_file=str(feather_file), max_sample_rows=0)
        entry: dict[str, Any] = {"file": str(feather_file), "filename": feather_file.name}

        if "error" in analysis:
            entry["valid"] = False
            entry["error"] = analysis["error"]
            all_valid = False
            file_results.append(entry)
            continue

        summary = analysis.get("summary", {})
        entry["valid"] = True
        entry["columns"] = summary.get("columns", [])
        entry["row_count"] = summary.get("total_rows", 0)
        file_results.append(entry)

    tracker_path = session.behavior_tracker_path
    tracker_info: dict[str, Any] = {}
    if tracker_path.exists():
        try:
            tracker_info = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_info = {"error": "Unable to read tracker file."}

    return {
        "verified": all_valid and bool(feather_files),
        "session_path": str(session_root),
        "data_path": str(data_path),
        "files": file_results,
        "total_files": len(file_results),
        "tracker": tracker_info,
    }


@mcp.tool()
def query_behavior_data_tool(
    feather_files: list[str],
    max_sample_rows: int = 10,
) -> dict[str, Any]:
    """Reads one or more processed behavior feather files and returns row counts, column metadata, and samples.

    For each file, computes the total row count, the list of columns, inter-row timing statistics (when a
    ``timestamp_us`` column is present), and a configurable number of sample rows. Binary data payloads are
    omitted from the sample rows for readability. Accepts feather file paths from the 'files' list returned by
    :func:`verify_behavior_processing_output_tool`.

    Args:
        feather_files: The list of absolute paths to feather files produced by the behavior processing pipeline.
        max_sample_rows: The maximum number of sample rows to include per file. Defaults to 10.

    Returns:
        A dictionary containing a 'results' list with per-file summaries and a 'total_files' count. Files that
        cannot be read produce an entry with 'file' and 'error' keys.
    """
    results = [
        analyze_feather_file(feather_file=feather_file, max_sample_rows=max_sample_rows)
        for feather_file in feather_files
    ]

    return {"results": results, "total_files": len(results)}


@mcp.tool()
def clean_behavior_processing_output_tool(session_paths: list[str]) -> dict[str, Any]:
    """Deletes the behavior_data subdirectory under one or more sessions' processed_data directories.

    For each session, loads :class:`SessionData` to resolve ``processed_data_path``, then removes
    ``{processed_data_path}/behavior_data/`` and all of its contents (processed feather files plus the
    processing tracker). Uses :func:`ataraxis_data_structures.delete_directory` for parallel file deletion
    with platform-safe retry logic. After cleanup, the same session paths can be passed back to
    :func:`prepare_behavior_processing_batch_tool` to reinitialize from scratch.

    Args:
        session_paths: The list of absolute paths to session root directories whose behavior processing
            output should be deleted.

    Returns:
        A dictionary containing a 'results' list with per-session outcomes (each with 'session_path',
        'cleaned' flag, and either 'data_path' or 'error') and a 'total_cleaned' count.
    """
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
        except Exception as error:
            results.append(
                {"session_path": session_path_str, "cleaned": False, "error": f"Unable to load session: {error}"}
            )
            continue

        outcome = clean_output_subdirectory(
            output_directory=str(session.processed_data_path),
            subdirectory_name=Directories.BEHAVIOR_DATA,
        )
        # Rewrites the helper's ``output_directory`` key into ``session_path`` so that the MCP surface
        # consistently identifies each entry by its session root.
        outcome.pop("output_directory", None)
        outcome = {"session_path": session_path_str, **outcome}
        results.append(outcome)

    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_sessions": len(results)}


def _run_behavior_job(job: _BehaviorPendingJob) -> None:
    """Executes a single behavior processing job in-process via the pipeline's remote mode.

    Serves as the picklable worker callable stored on ``JobExecutionState`` and dispatched to the batch
    manager's ``ProcessPoolExecutor``. Delegates to ``run_behavior_processing_pipeline`` with the job's
    session path in remote mode so that only the single ``(job_name, specifier)`` pair identified by
    ``job.job_id`` is executed against the session. The output location is resolved inside
    ``run_behavior_processing_pipeline`` from the session's ``SessionData`` marker.

    Args:
        job: The pending job descriptor produced by ``prepare_behavior_processing_batch_tool`` and attached
            to the active ``JobExecutionState`` by ``execute_behavior_processing_jobs_tool``.
    """
    run_behavior_processing_pipeline(
        session_path=job.session_path,
        job_id=job.job_id,
    )
