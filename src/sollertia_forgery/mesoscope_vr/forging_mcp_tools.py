"""Provides Model Context Protocol (MCP) tools for the forging (dataset assembly) pipeline."""

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
from sollertia_shared_assets import (
    SurgeryData,
    RawDataFiles,
    SessionTypes,
    ProcessingTrackers,
    MesoscopeExperimentDescriptor,
    validate_directory,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

from .forging import (
    FORGING_JOB_NAME,
    run_forging_pipeline,
)
from ..interfaces import mcp
from ..cross_system import (
    RESERVED_CORES,
    PendingJob,
    DatasetData,
    DatasetFiles,
    JobExecutionState,
    prepare_tracker,
    resolve_dataset,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
)


@dataclass(slots=True)
class _ForgingPendingJob(PendingJob):
    """Describes a single forging assembly job queued for background execution.

    Extends the shared ``PendingJob`` base with the dataset-level metadata required by the forging worker
    callable: the dataset name and the project root (passed to ``run_forging_pipeline`` in remote mode), and
    the session name for human-readable logging and status reporting. Path resolution (session directories,
    output paths) happens inside the worker subprocess via the pipeline's remote-mode routing.
    """

    dataset_name: str
    """The name of the dataset this job belongs to."""
    project_root: Path
    """The project root directory passed to ``run_forging_pipeline``."""
    session_name: str
    """The human-readable session name used for logging and status reporting."""


_job_execution_state: JobExecutionState[_ForgingPendingJob] | None = None
"""Stores the active execution state for batch forging jobs."""

_CORES_PER_JOB: int = 3
"""CPU cores consumed per forging worker: one subprocess plus a two-thread pool that parallelizes behavior
and runtime assembly. Divides the worker budget to yield the core-bounded concurrent-job ceiling."""


@mcp.tool()
def prepare_forging_batch_tool(
    datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prepares an execution manifest for batch dataset forging without starting execution.

    Accepts a list of dataset specifications, resolves each dataset hierarchy (creating, loading, or recreating
    as needed via :func:`resolve_dataset`), and initializes a :class:`ProcessingTracker` under each dataset
    directory. Idempotent: if a tracker already exists and the session set matches, returns the existing tracker
    state instead of reinitializing.

    Important:
        The AI agent calling this tool MUST run session discovery and filtering first to obtain confirmed
        session names. Do not assume or guess session names. Each dataset spec is resolved independently so
        that a failure in one dataset does not block the others.

    Args:
        datasets: The list of dataset specification dictionaries. Each dictionary must have a 'name' key
            (the dataset name) and a 'project_root' key (the absolute path to the project root). Optional keys
            are 'session_names' (a list of session name strings; empty or omitted to reuse an existing dataset)
            and 'force_recreate' (a boolean; defaults to False).

    Returns:
        A dictionary containing per-dataset manifests in ``datasets`` (keyed by dataset name) with tracker
        paths and job lists, total counts, and any invalid dataset specifications in ``invalid_datasets``.
    """
    result_datasets: dict[str, Any] = {}
    invalid_datasets: list[dict[str, Any]] = []
    total_jobs = 0

    for dataset_spec in datasets:
        # Validates the required keys.
        name = dataset_spec.get("name")
        project_root_str = dataset_spec.get("project_root")
        if not name or not project_root_str:
            invalid_datasets.append({**dataset_spec, "error": "Missing required 'name' or 'project_root' key."})
            continue

        name = str(name)
        project_root_str = str(project_root_str)
        error = validate_directory(directory=project_root_str)
        if error is not None:
            invalid_datasets.append({"name": name, "error": error})
            continue

        project_root = Path(project_root_str)
        session_names = tuple(dataset_spec.get("session_names", []))
        force_recreate = bool(dataset_spec.get("force_recreate", False))

        # Resolves the dataset hierarchy without triggering execution.
        try:
            dataset = resolve_dataset(
                name=name,
                session_names=session_names,
                project_root=project_root,
                required_session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
                force_recreate=force_recreate,
            )
        except Exception as resolve_error:
            invalid_datasets.append({"name": name, "error": str(resolve_error)})
            continue

        dataset_path = dataset.dataset_data_path.parent
        tracker_path = dataset_path / ProcessingTrackers.FORGING

        # Prepares the processing tracker and aligns it with the session set.
        tracker = ProcessingTracker(file_path=tracker_path)
        jobs_tuples = [(FORGING_JOB_NAME, entry.session) for entry in dataset.sessions]
        prepare_tracker(tracker=tracker, jobs=jobs_tuples)

        # Builds enriched job descriptors directly from the in-memory tracker, which
        # prepare_tracker just aligned. This avoids a redundant YAML deserialization.
        session_by_job_id = {
            ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=entry.session): entry
            for entry in dataset.sessions
        }

        enriched_jobs: list[dict[str, Any]] = []
        succeeded_count = 0
        failed_count = 0
        running_count = 0
        scheduled_count = 0

        for job_id, job_state in tracker.jobs.items():
            if job_id not in session_by_job_id:
                continue
            session_entry = session_by_job_id[job_id]
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
                "session_name": session_entry.session,
                "dataset_name": name,
                "project_root": project_root_str,
                "tracker_path": str(tracker_path),
            }
            if job_state.error_message is not None:
                entry["error_message"] = job_state.error_message
            enriched_jobs.append(entry)

        # Assembles the per-dataset manifest entry.
        result_datasets[name] = {
            "dataset_path": str(dataset_path),
            "tracker_path": str(tracker_path),
            "dataset_name": name,
            "project_root": project_root_str,
            "jobs": enriched_jobs,
            "summary": {
                "total": len(tracker.jobs),
                "succeeded": succeeded_count,
                "failed": failed_count,
                "running": running_count,
                "scheduled": scheduled_count,
            },
        }
        total_jobs += len(enriched_jobs)

    # Packages the final response with aggregate counts and any invalid specs.
    result: dict[str, Any] = {
        "success": True,
        "datasets": result_datasets,
        "total_datasets": len(result_datasets),
        "total_jobs": total_jobs,
    }

    if invalid_datasets:
        result["invalid_datasets"] = invalid_datasets

    return result


@mcp.tool()
def execute_forging_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    worker_budget: int = -1,
    max_parallel_jobs: int = 10,
) -> dict[str, Any]:
    """Dispatches forging assembly jobs for background execution with budget-bounded concurrency.

    Takes job descriptors from the manifest produced by :func:`prepare_forging_batch_tool` and starts a
    background execution manager that runs each job in a separate worker subprocess. Each job invokes
    :func:`run_forging_pipeline` in remote mode with the descriptor's ``job_id`` so that only that single
    session assembly is executed. Concurrency is bounded by ``min(max_parallel_jobs, worker_budget //
    _CORES_PER_JOB)``; lower ``max_parallel_jobs`` to reduce per-session memory peaks.

    Important:
        Only one execution session can be active at a time. Use :func:`cancel_forging_tool` to cancel an
        active session before starting a new one.

    Args:
        jobs: The list of job descriptors from :func:`prepare_forging_batch_tool`. Each dictionary must have
            'tracker_path', 'job_id', 'dataset_name', 'project_root', and 'session_name' keys.
        worker_budget: Total CPU cores available for the execution session. Set to -1 for automatic
            resolution via :func:`ataraxis_base_utilities.resolve_worker_count`.
        max_parallel_jobs: Hard cap on concurrent forging jobs. Set to -1 to fall back to the default of 10.

    Returns:
        A dictionary containing a 'started' flag, 'total_jobs', resolved 'worker_budget', the effective
        'max_parallel_jobs' after applying the CPU floor, and any invalid jobs.
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
    required_keys = {"tracker_path", "job_id", "dataset_name", "project_root", "session_name"}
    pending: list[_ForgingPendingJob] = []
    all_jobs: dict[tuple[str, str], _ForgingPendingJob] = {}
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

        pending_job = _ForgingPendingJob(
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            dataset_name=job_dict["dataset_name"],
            project_root=Path(job_dict["project_root"]),
            session_name=job_dict["session_name"],
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the total worker budget.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=RESERVED_CORES)

    # Floors the user-supplied parallel-job cap by the CPU budget divided by the per-worker core cost.
    requested_parallel = max_parallel_jobs if max_parallel_jobs > 0 else 10
    core_bounded_parallel = max(1, resolved_budget // _CORES_PER_JOB)
    effective_parallel = min(requested_parallel, core_bounded_parallel)

    # Creates the execution state and starts the shared manager thread.
    _job_execution_state = JobExecutionState[_ForgingPendingJob](
        worker=_run_forging_job,
        all_jobs=all_jobs,
        pending_queue=deque(pending),
        worker_budget=resolved_budget,
        max_parallel_jobs=effective_parallel,
    )

    manager = Thread(
        target=job_execution_manager,
        kwargs={"state": _job_execution_state},
        daemon=True,
    )
    manager.start()
    _job_execution_state.manager_thread = manager

    # Packages the final response with job count and resolved budget.
    result: dict[str, Any] = {
        "started": True,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "max_parallel_jobs": effective_parallel,
    }

    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs

    return result


@mcp.tool()
def get_forging_status_tool() -> dict[str, Any]:
    """Returns the current status of the active forging execution session.

    Reads ProcessingTracker files from disk for each job to report per-job progress. When no execution session
    exists, returns an inactive status.

    Returns:
        A dictionary containing an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts
        for pending, running, succeeded, and failed jobs.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    # Snapshots current manager liveness and initializes per-status counters.
    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    # Reads each tracker file from disk and classifies jobs by status.
    for tracker_path, path_jobs in group_jobs_by_tracker(state=state).items():
        try:
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
        except Exception:
            # Marks all jobs under an unreadable tracker as UNKNOWN.
            job_details.extend(
                {
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "session_name": job.session_name,
                    "status": "UNKNOWN",
                }
                for job in path_jobs
            )
            continue

        # Classifies each job's status and builds the detail entry.
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
                    "dataset_name": job.dataset_name,
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
                        "dataset_name": job.dataset_name,
                        "session_name": job.session_name,
                        "status": "UNKNOWN",
                    }
                )

    # Assembles the response with per-job details and aggregate summary.
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
def get_forging_timing_tool() -> dict[str, Any]:
    """Returns timing information for all jobs in the active forging execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps recorded in the ProcessingTracker.

    Returns:
        A dictionary containing an 'active' flag, per-job timing in 'jobs', and a 'session' summary with total
        elapsed seconds and throughput.
    """
    if _job_execution_state is None:
        return {"active": False, "message": "No execution session exists."}

    # Captures current wall-clock time for elapsed-time calculations.
    state = _job_execution_state
    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    job_timing: list[dict[str, Any]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0
    running_count = 0

    # Reads each tracker and extracts per-job timing information.
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
                "dataset_name": job.dataset_name,
                "session_name": job.session_name,
            }

            # Records start timestamp and tracks the earliest start across all jobs.
            if job_info.started_at is not None:
                started_at_us = int(job_info.started_at)
                entry["started_at"] = started_at_us
                if earliest_start is None or started_at_us < earliest_start:
                    earliest_start = started_at_us

            # Computes live elapsed time for currently running jobs.
            if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
                elapsed_seconds = convert_time(
                    time=current_us - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                )
                entry["elapsed_seconds"] = round(elapsed_seconds, 2)
                running_count += 1

            # Computes final duration for completed jobs.
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

    # Calculates total wall-clock elapsed time from the earliest job start to now.
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

    # Assembles the session-level summary with per-status counts.
    session: dict[str, Any] = {
        "total_elapsed_seconds": total_elapsed_seconds,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": len(state.all_jobs) - completed_count - failed_count - running_count,
    }

    # Derives throughput as completed jobs per hour when at least one job has finished.
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
def cancel_forging_tool() -> dict[str, Any]:
    """Cancels the active forging execution session.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs
    are started.

    Returns:
        A dictionary containing a 'canceled' flag, a 'message', and 'final_state' with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    if _job_execution_state is None:
        return {"canceled": False, "message": "No execution session is active."}

    state = _job_execution_state

    # Atomically drains the pending queue so no new jobs are dispatched.
    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_jobs)

    # Scans all trackers to tally final succeeded/failed counts at the point of cancellation.
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

    # Assembles the cancellation summary with final state snapshot.
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
def reset_forging_jobs_tool(
    tracker_path: str,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a forging tracker to scheduled status for re-runs.

    Args:
        tracker_path: The absolute path to the forging :class:`ProcessingTracker` YAML file.
        job_ids: An optional list of job IDs to reset. If not provided, every job in the tracker is reset.

    Returns:
        A dictionary containing a 'reset' flag, the number of jobs reset, and updated job statuses.
    """
    # Loads and validates the tracker file.
    path = Path(tracker_path)

    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Resolves which job IDs to reset: all jobs if none specified, otherwise the intersection.
    tracker_ids = set(tracker.jobs.keys())
    target_ids = tracker_ids if job_ids is None else tracker_ids & set(job_ids)

    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Removes targeted jobs and re-initializes them as SCHEDULED in a fresh tracker.
    reset_jobs: list[tuple[str, str]] = [
        (tracker.jobs[job_id].job_name, tracker.jobs[job_id].specifier) for job_id in target_ids
    ]

    for job_id in target_ids:
        del tracker.jobs[job_id]
    tracker.to_yaml(file_path=path)

    reset_tracker = ProcessingTracker(file_path=path)
    reset_tracker.initialize_jobs(jobs=reset_jobs)

    # Reads back the updated tracker state to confirm the reset.
    try:
        updated_status = read_tracker_status(tracker_path=path)
    except Exception:
        updated_status = {"jobs": [], "summary": {}}

    return {"reset": True, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_forging_batch_status_overview_tool(root_directory: str) -> dict[str, Any]:
    """Discovers and summarizes forging status for all datasets under a root directory.

    Datasets live at the canonical ``<project_root>/<dataset_name>/`` layout and each holds its tracker at
    ``<dataset_root>/forging_tracker.yaml``. This tool iterates the top-level children of ``root_directory``
    and reads the tracker from each candidate dataset directory rather than walking the whole project.

    Args:
        root_directory: The absolute path to the project root containing dataset directories.

    Returns:
        A dictionary containing per-dataset status summaries and aggregate counts.
    """
    # Validates the root directory path.
    error = validate_directory(directory=root_directory)
    if error is not None:
        return {"error": error}

    root_path = Path(root_directory)
    dataset_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    # Iterates top-level children only; datasets are never nested under animals or sessions.
    for dataset_path in sorted(root_path.iterdir()):
        if not dataset_path.is_dir():
            continue
        tracker_path = dataset_path.joinpath(ProcessingTrackers.FORGING)
        if not tracker_path.is_file():
            continue
        dataset_name = dataset_path.name
        try:
            status = read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})

            # Accumulates cross-dataset totals for the aggregate summary.
            aggregate_succeeded += summary.get("succeeded", 0)
            aggregate_failed += summary.get("failed", 0)
            aggregate_running += summary.get("running", 0)
            aggregate_scheduled += summary.get("scheduled", 0)

            dataset_status = derive_tracker_status(summary=summary)

            dataset_statuses.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "tracker_path": str(tracker_path),
                    "status": dataset_status,
                    **status,
                }
            )
        except Exception:
            dataset_statuses.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "tracker_path": str(tracker_path),
                    "status": "error",
                    "error": "Unable to read tracker file.",
                }
            )

    # Assembles the response with per-dataset entries and cross-dataset aggregates.
    return {
        "datasets": dataset_statuses,
        "total_datasets": len(dataset_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()
def verify_forging_output_tool(dataset_path: str) -> dict[str, Any]:
    """Verifies the completeness of forged data output for a single dataset.

    Loads the dataset's :class:`DatasetData` marker, then for each session in the dataset hierarchy checks the
    presence and readability of ``data.feather`` and the presence and parseability of the per-session
    ``session_descriptor.yaml``. For each animal in the dataset, also checks the presence and parseability
    of the per-animal ``surgery_metadata.yaml``. The forging processing tracker is read to report per-job statuses.

    Args:
        dataset_path: The absolute path to the dataset root directory (containing ``dataset_data.yaml``).

    Returns:
        A dictionary containing a 'verified' flag, per-session results in 'files', per-animal surgery results
        in 'animals', tracker status in 'tracker', and aggregate counts.
    """
    # Validates the dataset directory path and loads the dataset metadata.
    error = validate_directory(directory=dataset_path)
    if error is not None:
        return {"error": error}

    dataset_root = Path(dataset_path)

    try:
        dataset = DatasetData.load(dataset_path=dataset_root)
    except Exception as load_error:
        return {"error": f"Unable to load dataset: {load_error}"}

    file_results: list[dict[str, Any]] = []
    all_valid = True

    # Checks each session's feather file for existence and readability, plus the per-session experiment
    # descriptor for existence and parseability.
    for session_entry in dataset.sessions:
        data_path = session_entry.data_path
        entry: dict[str, Any] = {
            "session_name": session_entry.session,
            "animal": session_entry.animal,
            "file": str(data_path),
        }

        feather_valid = True
        if not data_path.exists():
            entry["valid"] = False
            entry["error"] = f"{DatasetFiles.DATA} not found."
            all_valid = False
            feather_valid = False
        else:
            # Loads the feather file without sampling to confirm readability and extract metadata.
            analysis = analyze_feather_file(feather_file=str(data_path), max_sample_rows=0)
            if "error" in analysis:
                entry["valid"] = False
                entry["error"] = analysis["error"]
                all_valid = False
                feather_valid = False
            else:
                summary = analysis.get("summary", {})
                entry["valid"] = True
                entry["columns"] = summary.get("columns", [])
                entry["row_count"] = summary.get("total_rows", 0)

        # Verifies the per-session experiment descriptor exists and parses as MesoscopeExperimentDescriptor.
        descriptor_path = session_entry.descriptor_path
        descriptor_entry: dict[str, Any] = {"file": str(descriptor_path)}
        if not descriptor_path.exists():
            descriptor_entry["valid"] = False
            descriptor_entry["error"] = f"{RawDataFiles.SESSION_DESCRIPTOR} not found."
            all_valid = False
        else:
            try:
                MesoscopeExperimentDescriptor.from_yaml(file_path=descriptor_path)
                descriptor_entry["valid"] = True
            except Exception as descriptor_error:
                descriptor_entry["valid"] = False
                descriptor_entry["error"] = f"Unable to load descriptor: {descriptor_error}"
                all_valid = False
        entry["descriptor"] = descriptor_entry

        # Promotes a feather-only valid flag to overall invalid when the descriptor failed.
        if feather_valid and not descriptor_entry["valid"]:
            entry["valid"] = False

        file_results.append(entry)

    # Verifies the per-animal surgery file for each unique animal in the dataset.
    animal_results: list[dict[str, Any]] = []
    for dataset_animal in dataset.animals:
        surgery_path = dataset_animal.surgery_path
        animal_entry: dict[str, Any] = {"animal": dataset_animal.animal, "file": str(surgery_path)}
        if not surgery_path.exists():
            animal_entry["valid"] = False
            animal_entry["error"] = f"{RawDataFiles.SURGERY_METADATA} not found."
            all_valid = False
        else:
            try:
                SurgeryData.from_yaml(file_path=surgery_path)
                animal_entry["valid"] = True
            except Exception as surgery_error:
                animal_entry["valid"] = False
                animal_entry["error"] = f"Unable to load surgery data: {surgery_error}"
                all_valid = False
        animal_results.append(animal_entry)

    # Reads the forging tracker to include per-job pipeline statuses alongside file checks.
    tracker_path = dataset.dataset_data_path.parent / ProcessingTrackers.FORGING
    tracker_info: dict[str, Any] = {}
    if tracker_path.exists():
        try:
            tracker_info = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_info = {"error": "Unable to read tracker file."}

    # Assembles the verification result with per-file outcomes, per-animal surgery outcomes, and tracker state.
    return {
        "verified": all_valid and bool(file_results) and bool(animal_results),
        "dataset_path": str(dataset_root),
        "dataset_name": dataset.name,
        "files": file_results,
        "total_files": len(file_results),
        "animals": animal_results,
        "total_animals": len(animal_results),
        "tracker": tracker_info,
    }


@mcp.tool()
def query_forging_data_tool(
    feather_files: list[str],
    max_sample_rows: int = 10,
) -> dict[str, Any]:
    """Reads one or more forged dataset feather files and returns row counts, column metadata, and samples.

    For each file, computes the total row count, the list of columns, inter-row timing statistics (when a
    time column is present), and a configurable number of sample rows. Accepts feather file paths from the
    'files' list returned by :func:`verify_forging_output_tool`.

    Args:
        feather_files: The list of absolute paths to feather files produced by the forging pipeline.
        max_sample_rows: The maximum number of sample rows to include per file. Defaults to 10.

    Returns:
        A dictionary containing a 'results' list with per-file summaries and a 'total_files' count.
    """
    # Analyzes each feather file for row count, column metadata, and sample rows.
    results = [
        analyze_feather_file(feather_file=feather_file, max_sample_rows=max_sample_rows)
        for feather_file in feather_files
    ]

    return {"results": results, "total_files": len(results)}


@mcp.tool()
def clean_forging_output_tool(dataset_paths: list[str]) -> dict[str, Any]:
    """Deletes the full dataset hierarchy for one or more datasets.

    For each dataset path, removes the entire directory tree (tracker, dataset metadata, and all per-session
    ``data.feather`` files). After cleanup, the same dataset specifications can be passed back to
    :func:`prepare_forging_batch_tool` to reinitialize from scratch.

    Important:
        This tool refuses to run while a forging execution session is active. Cancel any running session
        before calling this tool.

    Args:
        dataset_paths: The list of absolute paths to dataset root directories to delete.

    Returns:
        A dictionary containing per-dataset outcomes and a 'total_cleaned' count.
    """
    # Refuses to clean while jobs are still running to prevent data loss.
    if (
        _job_execution_state is not None
        and _job_execution_state.manager_thread is not None
        and _job_execution_state.manager_thread.is_alive()
    ):
        return {"error": "Cannot clean while an execution session is active. Cancel it first."}

    results: list[dict[str, Any]] = []

    # Validates and deletes each dataset directory tree.
    for dataset_path_str in dataset_paths:
        dataset_path = Path(dataset_path_str)

        if not dataset_path.exists():
            results.append({"dataset_path": dataset_path_str, "cleaned": True, "message": "Nothing to clean."})
            continue

        if not dataset_path.is_dir():
            results.append({"dataset_path": dataset_path_str, "cleaned": False, "error": "Path is not a directory."})
            continue

        try:
            delete_directory(directory_path=dataset_path)
            results.append({"dataset_path": dataset_path_str, "cleaned": True})
        except Exception as delete_error:
            results.append(
                {"dataset_path": dataset_path_str, "cleaned": False, "error": f"Unable to delete: {delete_error}"}
            )

    # Tallies how many datasets were successfully cleaned.
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))

    return {"results": results, "total_cleaned": total_cleaned, "total_datasets": len(results)}


def _run_forging_job(job: _ForgingPendingJob) -> None:
    """Executes a single forging assembly job in-process via the pipeline's remote mode.

    Serves as the picklable worker callable stored on ``JobExecutionState`` and dispatched to the batch
    manager's ``ProcessPoolExecutor``. Delegates to ``run_forging_pipeline`` with an empty session list (the
    dataset is already materialized) and the job's ``job_id`` in remote mode so that only the single session
    identified by the job ID is assembled.

    Args:
        job: The pending job descriptor produced by ``prepare_forging_batch_tool`` and attached to the active
            ``JobExecutionState`` by ``execute_forging_jobs_tool``.
    """
    run_forging_pipeline(
        name=job.dataset_name,
        session_names=(),
        project_root=job.project_root,
        job_id=job.job_id,
    )
