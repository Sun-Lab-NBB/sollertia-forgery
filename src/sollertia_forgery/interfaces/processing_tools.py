"""Provides the system-agnostic batch Model Context Protocol (MCP) tools for the local processing and forging
pipelines.

Notes:
    Every tool takes a ``pipeline`` argument (one of the ``MCP_BATCH_PIPELINES`` members, currently ``behavior`` and
    ``forging``) and dispatches the system-specific processing code (discovery, per-job execution, verification,
    cleanup, and status iteration) through the registries. The generic batch orchestration — the worker pool,
    processing trackers, and status/timing math — is shared across every pipeline, so an agent drives any acquisition
    system's batch pipelines through one tool surface. The acquisition system is resolved inside each registered
    adapter from the session or dataset it is pointed at, so these tools never name a system package.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from threading import Thread
import contextlib
from collections import deque

from ataraxis_time import (
    TimeUnits,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from ataraxis_base_utilities import resolve_worker_count
from sollertia_shared_assets import validate_directory
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..pipelines import ProcessingPipelines
from ..registries import (
    MCP_BATCH_PIPELINES,
    resolve_clean,
    resolve_verify,
    resolve_worker,
    resolve_prepare,
    resolve_overview,
    resolve_concurrency,
)
from .mcp_instance import mcp
from ..local_orchestration import (
    RESERVED_CORES,
    GenericPendingJob,
    JobExecutionState,
    read_tracker_status,
    analyze_feather_file,
    group_jobs_by_tracker,
    job_execution_manager,
)

_EXECUTION_STATE: dict[ProcessingPipelines, JobExecutionState[GenericPendingJob]] = {}
"""Stores the active batch execution state per pipeline. Each pipeline runs at most one execution session at a time;
the status, timing, cancel, and execute tools read and write the entry for the requested pipeline."""

_REQUIRED_JOB_KEYS: frozenset[str] = frozenset({"tracker_path", "job_id", "job_name", "specifier"})
"""The job-descriptor keys every pipeline's worker needs to dispatch a single job. Per-pipeline fields (the session
path for behavior, the project root for forging) are mapped onto the generic descriptor when present."""


def _resolve_batch_pipeline(pipeline: str) -> ProcessingPipelines | None:
    """Coerces a pipeline identifier to a batch ``ProcessingPipelines`` member, or returns None when invalid.

    Args:
        pipeline: The pipeline identifier string (e.g., ``"behavior"`` or ``"forging"``).

    Returns:
        The corresponding ``ProcessingPipelines`` member when it is a batch pipeline, otherwise None.
    """
    try:
        member = ProcessingPipelines(pipeline)
    except ValueError:
        return None
    if member not in MCP_BATCH_PIPELINES:
        return None
    return member


def _invalid_pipeline_error(pipeline: str) -> dict[str, Any]:
    """Builds the error response returned when a tool receives an unsupported ``pipeline`` argument."""
    valid = ", ".join(sorted(member.value for member in MCP_BATCH_PIPELINES))
    return {"error": f"Unsupported batch pipeline '{pipeline}'. The batch tools expose only: {valid}."}


@mcp.tool()
def prepare_batch_tool(pipeline: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepares an execution manifest for a batch of processing units without starting execution.

    Discovers the jobs available for each unit via the pipeline's registered prepare adapter and initializes a
    ``ProcessingTracker`` for each. The output location is resolved statically by the adapter from the unit's
    on-disk marker; the caller never chooses where outputs go. Idempotent: if a tracker already exists, the adapter
    returns its current job registry and status instead of reinitializing.

    Args:
        pipeline: The batch pipeline to prepare (one of ``behavior`` or ``forging``).
        units: The list of unit specifications. For ``behavior``, each unit carries a ``session_path``. For
            ``forging``, each unit carries a ``name`` (dataset name), a ``project_root``, and optionally
            ``session_names`` and a ``force_recreate`` flag.

    Returns:
        A dictionary carrying the per-unit manifests in ``units`` (each with its tracker path, job descriptors, and
        summary), plus the total unit and job counts. Per-unit discovery failures are surfaced as a unit entry with
        an ``error`` key, leaving the rest of the batch unaffected.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    prepare = resolve_prepare(member)
    unit_results: list[dict[str, Any]] = []
    total_jobs = 0

    for unit in units:
        try:
            entry = prepare(unit)
        except Exception as error:
            unit_results.append({"error": f"Preparation failed: {error}", "unit": unit, "jobs": [], "summary": {}})
            continue
        unit_results.append(entry)
        total_jobs += len(entry.get("jobs", []))

    return {
        "success": True,
        "pipeline": member.value,
        "units": unit_results,
        "total_units": len(unit_results),
        "total_jobs": total_jobs,
    }


@mcp.tool()
def execute_jobs_tool(
    pipeline: str,
    jobs: list[dict[str, Any]],
    *,
    worker_budget: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, Any]:
    """Dispatches prepared jobs for background execution with budget-bounded concurrency.

    Takes job descriptors from the manifest produced by ``prepare_batch_tool`` and starts a background execution
    manager that runs each job in a separate worker subprocess via the pipeline's registered worker. Only one
    execution session can be active per pipeline at a time; use ``cancel_processing_tool`` to cancel an active
    session before starting a new one.

    Args:
        pipeline: The batch pipeline to execute (one of ``behavior`` or ``forging``).
        jobs: The list of job descriptors from ``prepare_batch_tool``. Each must carry ``tracker_path``,
            ``job_id``, ``job_name``, and ``specifier``; behavior jobs additionally carry ``session_path`` and
            forging jobs carry ``project_root`` and ``dataset_name``.
        worker_budget: The total number of CPU cores available for the execution session. Set to -1 for automatic
            resolution. Controls the process-pool size and bounds the memory footprint.
        max_parallel_jobs: An optional hard cap on concurrently executing jobs. Set to -1 to apply the pipeline's
            default policy, which floors the cap by ``worker_budget`` divided by the pipeline's per-job core cost.

    Returns:
        A dictionary carrying a ``started`` flag, the resolved worker budget, the effective parallel-job cap, the
        total job count, and any invalid job descriptors.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    # Enforces the single-session-per-pipeline constraint.
    active = _EXECUTION_STATE.get(member)
    if active is not None and active.manager_thread is not None and active.manager_thread.is_alive():
        return {
            "error": (
                f"An execution session is already active for the '{member.value}' pipeline. Cancel it first or "
                f"wait for completion."
            )
        }

    worker = resolve_worker(member)
    concurrency = resolve_concurrency(member)

    pending: list[GenericPendingJob] = []
    all_jobs: dict[tuple[str, str], GenericPendingJob] = {}
    invalid_jobs: list[dict[str, Any]] = []

    for job_dict in jobs:
        missing = _REQUIRED_JOB_KEYS - job_dict.keys()
        if missing:
            invalid_jobs.append({**job_dict, "error": f"Missing required keys: {sorted(missing)}"})
            continue

        tracker_path = Path(job_dict["tracker_path"])
        if not tracker_path.exists():
            invalid_jobs.append({**job_dict, "error": f"Tracker file not found: {job_dict['tracker_path']}"})
            continue

        # Maps the per-pipeline descriptor keys onto the generic job. The behavior worker uses ``unit_path`` (the
        # session path); the forging worker uses ``name`` (the dataset name) and ``project_root``. Absent keys
        # default to harmless values for the pipelines that do not consume them.
        unit_path_raw = job_dict.get("session_path") or job_dict.get("unit_path") or job_dict.get("project_root")
        project_root_raw = job_dict.get("project_root")
        pending_job = GenericPendingJob(
            tracker_path=tracker_path,
            job_id=job_dict["job_id"],
            name=str(job_dict.get("dataset_name") or job_dict.get("session_name") or ""),
            unit_path=Path(unit_path_raw) if unit_path_raw else Path(),
            job_name=job_dict["job_name"],
            specifier=job_dict["specifier"],
            project_root=Path(project_root_raw) if project_root_raw else None,
        )
        pending.append(pending_job)
        all_jobs[pending_job.dispatch_key] = pending_job

    if not pending:
        return {"error": "No valid jobs to execute.", "invalid_jobs": invalid_jobs}

    # Resolves the worker budget (the process-pool size) and the effective parallel-job cap. When the caller does
    # not request an explicit cap, a pipeline that declares a positive default floors the cap by the cores each job
    # consumes; a non-positive default defers concurrency entirely to the worker budget.
    resolved_budget = resolve_worker_count(requested_workers=worker_budget, reserved_cores=RESERVED_CORES)
    if max_parallel_jobs and max_parallel_jobs > 0:
        effective_max = max_parallel_jobs
    elif concurrency.default_max_parallel > 0:
        budget_cap = max(1, resolved_budget // max(1, concurrency.cores_per_job))
        effective_max = min(concurrency.default_max_parallel, budget_cap)
    else:
        effective_max = -1

    state = JobExecutionState[GenericPendingJob](
        worker=worker,
        all_jobs=all_jobs,
        pending_queue=deque(pending),
        worker_budget=resolved_budget,
        max_parallel_jobs=effective_max,
    )

    manager = Thread(target=job_execution_manager, kwargs={"state": state}, daemon=True)
    manager.start()
    state.manager_thread = manager
    _EXECUTION_STATE[member] = state

    result: dict[str, Any] = {
        "started": True,
        "pipeline": member.value,
        "total_jobs": len(pending),
        "worker_budget": resolved_budget,
        "max_parallel_jobs": effective_max,
    }
    if invalid_jobs:
        result["invalid_jobs"] = invalid_jobs
    return result


@mcp.tool()
def get_processing_status_tool(pipeline: str) -> dict[str, Any]:
    """Returns the current status of the active execution session for a pipeline.

    Reads the ``ProcessingTracker`` file for each job from disk to report per-job progress. When no execution
    session exists for the pipeline, returns an inactive status.

    Args:
        pipeline: The batch pipeline whose execution session to inspect (one of ``behavior`` or ``forging``).

    Returns:
        A dictionary carrying an ``active`` flag, per-job status entries in ``jobs``, and a ``summary`` with counts
        for scheduled, running, succeeded, and failed jobs.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    state = _EXECUTION_STATE.get(member)
    if state is None:
        return {"active": False, "pipeline": member.value, "message": "No execution session exists."}

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
                    "name": job.name,
                    "status": "UNKNOWN",
                }
                for job in path_jobs
            )
            continue

        for job in path_jobs:
            if job.job_id not in tracker.jobs:
                job_details.append(
                    {
                        "job_id": job.job_id,
                        "job_name": job.job_name,
                        "specifier": job.specifier,
                        "name": job.name,
                        "status": "UNKNOWN",
                    }
                )
                continue

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
                "name": job.name,
                "status": status.name,
            }
            if job_state.error_message is not None:
                entry["error_message"] = job_state.error_message
            job_details.append(entry)

    return {
        "active": manager_alive,
        "pipeline": member.value,
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
def get_processing_timing_tool(pipeline: str) -> dict[str, Any]:
    """Returns timing information for every job in the active execution session for a pipeline.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC
    timestamps recorded in the ``ProcessingTracker``.

    Args:
        pipeline: The batch pipeline whose execution session to inspect (one of ``behavior`` or ``forging``).

    Returns:
        A dictionary carrying an ``active`` flag, per-job timing in ``jobs``, and a ``session`` summary with total
        elapsed seconds and throughput.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    state = _EXECUTION_STATE.get(member)
    if state is None:
        return {"active": False, "pipeline": member.value, "message": "No execution session exists."}

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

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
                "name": job.name,
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

    return {"active": manager_alive, "pipeline": member.value, "jobs": job_timing, "session": session}


@mcp.tool()
def cancel_processing_tool(pipeline: str) -> dict[str, Any]:
    """Cancels the active execution session for a pipeline.

    Clears the pending job queue so no new jobs are dispatched. Active jobs complete naturally but no new jobs are
    started.

    Args:
        pipeline: The batch pipeline whose execution session to cancel (one of ``behavior`` or ``forging``).

    Returns:
        A dictionary carrying a ``canceled`` flag, a ``message``, and a ``final_state`` with counts for succeeded,
        failed, and active jobs at the time of cancellation.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    state = _EXECUTION_STATE.get(member)
    if state is None:
        return {"canceled": False, "pipeline": member.value, "message": "No execution session is active."}

    with state.lock:
        state.canceled = True
        cleared_count = len(state.pending_queue)
        state.pending_queue.clear()
        active_count = len(state.active_jobs)

    succeeded = 0
    failed = 0
    tracker_paths: set[Path] = {job.tracker_path for job in state.all_jobs.values()}
    for tracker_path in tracker_paths:
        # Skips trackers that cannot be deserialized so one unreadable file does not suppress the final tally.
        with contextlib.suppress(Exception):
            tracker = ProcessingTracker.from_yaml(file_path=tracker_path)
            for job_state in tracker.jobs.values():
                if job_state.status == ProcessingStatus.SUCCEEDED:
                    succeeded += 1
                elif job_state.status == ProcessingStatus.FAILED:
                    failed += 1

    return {
        "canceled": True,
        "pipeline": member.value,
        "message": f"Canceled. Cleared {cleared_count} pending job(s). {active_count} job(s) still completing.",
        "final_state": {
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "active_jobs_at_cancel": active_count,
        },
    }


@mcp.tool()
def reset_processing_jobs_tool(
    pipeline: str,
    tracker_path: str,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Resets specific jobs or all jobs in a tracker to scheduled status for re-runs.

    Operates directly on the supplied tracker file and is independent of any active execution session.

    Args:
        pipeline: The batch pipeline the tracker belongs to (validated for surface uniformity; one of ``behavior``
            or ``forging``).
        tracker_path: The absolute path to the pipeline's ``ProcessingTracker`` YAML file.
        job_ids: An optional list of job IDs to reset. If not provided, every job in the tracker is reset.

    Returns:
        A dictionary carrying a ``reset`` flag, the number of jobs reset, and the updated job statuses.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    path = Path(tracker_path)
    if not path.exists():
        return {"error": f"Tracker file not found: {tracker_path}"}

    try:
        tracker = ProcessingTracker.from_yaml(file_path=path)
    except Exception as error:
        return {"error": f"Unable to read tracker: {error}"}

    # Intersects the requested IDs with the tracker registry so stale or unknown IDs are silently ignored.
    tracker_ids = set(tracker.jobs.keys())
    target_ids = tracker_ids if job_ids is None else tracker_ids & set(job_ids)
    if not target_ids:
        return {"reset": False, "message": "No matching jobs found to reset."}

    # Collects the (job_name, specifier) tuples for the reset jobs, then rebuilds the registry so each returns to
    # SCHEDULED with cleared timing and error metadata.
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

    return {"reset": True, "pipeline": member.value, "jobs_reset": len(target_ids), **updated_status}


@mcp.tool()
def get_batch_status_overview_tool(pipeline: str, root_directory: str) -> dict[str, Any]:
    """Discovers and summarizes processing status for every unit under a root directory.

    Iterates the pipeline's units under the root via its registered overview iterator and aggregates each unit's
    tracker status.

    Args:
        pipeline: The batch pipeline to summarize (one of ``behavior`` or ``forging``).
        root_directory: The absolute path to the root directory to search (a project root for both pipelines).

    Returns:
        A dictionary carrying the per-unit status summaries in ``units`` and aggregate counts.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    error = validate_directory(root_directory)
    if error is not None:
        return {"error": error}

    iterate = resolve_overview(member)

    unit_statuses: list[dict[str, Any]] = []
    aggregate_succeeded = 0
    aggregate_failed = 0
    aggregate_running = 0
    aggregate_scheduled = 0

    for descriptor in iterate(root_directory):
        unit_statuses.append(descriptor)
        summary = descriptor.get("summary", {})
        aggregate_succeeded += summary.get("succeeded", 0)
        aggregate_failed += summary.get("failed", 0)
        aggregate_running += summary.get("running", 0)
        aggregate_scheduled += summary.get("scheduled", 0)

    return {
        "pipeline": member.value,
        "units": unit_statuses,
        "total_units": len(unit_statuses),
        "summary": {
            "succeeded": aggregate_succeeded,
            "failed": aggregate_failed,
            "running": aggregate_running,
            "scheduled": aggregate_scheduled,
        },
    }


@mcp.tool()
def verify_processing_output_tool(pipeline: str, unit_path: str) -> dict[str, Any]:
    """Verifies the completeness of a single processing unit's output.

    Delegates the system-specific verification (which feather files and metadata to check, and how the ``verified``
    verdict is computed) to the pipeline's registered verify adapter, which owns the full result.

    Args:
        pipeline: The batch pipeline whose output to verify (one of ``behavior`` or ``forging``).
        unit_path: The absolute path to the unit root (a session root for behavior, a dataset root for forging).

    Returns:
        The verify adapter's result merged into the response: a ``verified`` flag, per-file results, the tracker
        block, and aggregate counts. Returns an ``error`` key when the unit cannot be loaded or located.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    error = validate_directory(unit_path)
    if error is not None:
        return {"error": error}

    verify = resolve_verify(member)
    return {"pipeline": member.value, **verify(Path(unit_path))}


@mcp.tool()
def query_data_tool(
    pipeline: str,
    feather_files: list[str],
    max_sample_rows: int = 10,
) -> dict[str, Any]:
    """Reads one or more processed feather files and returns row counts, column metadata, and samples.

    For each file, computes the total row count, the column list, inter-row timing statistics (when a recognized
    timestamp column is present), and a configurable number of sample rows. Binary payloads are omitted from the
    sample rows for readability.

    Args:
        pipeline: The batch pipeline the files belong to (validated for surface uniformity; one of ``behavior`` or
            ``forging``).
        feather_files: The list of absolute paths to feather files to inspect.
        max_sample_rows: The maximum number of sample rows to include per file. Defaults to 10.

    Returns:
        A dictionary carrying a ``results`` list with per-file summaries and a ``total_files`` count. Files that
        cannot be read produce an entry with ``file`` and ``error`` keys.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    results = [
        analyze_feather_file(feather_file=feather_file, max_sample_rows=max_sample_rows)
        for feather_file in feather_files
    ]
    return {"pipeline": member.value, "results": results, "total_files": len(results)}


@mcp.tool()
def clean_output_tool(pipeline: str, unit_paths: list[str]) -> dict[str, Any]:
    """Deletes the processing output for one or more units.

    Delegates the system-specific deletion scope (a per-session subdirectory for behavior, the full dataset tree
    for forging) to the pipeline's registered clean adapter. Pipelines whose cleanup must not run while an
    execution session is still writing are guarded: the tool refuses to clean while that pipeline has an active
    execution session.

    Args:
        pipeline: The batch pipeline whose output to delete (one of ``behavior`` or ``forging``).
        unit_paths: The list of absolute paths to the unit roots whose output should be deleted.

    Returns:
        A dictionary carrying a ``results`` list with per-unit outcomes and a ``total_cleaned`` count.
    """
    member = _resolve_batch_pipeline(pipeline)
    if member is None:
        return _invalid_pipeline_error(pipeline)

    clean_fn, guard_on_active = resolve_clean(member)

    if guard_on_active:
        state = _EXECUTION_STATE.get(member)
        if state is not None and state.manager_thread is not None and state.manager_thread.is_alive():
            return {
                "error": (
                    f"An execution session is active for the '{member.value}' pipeline. Cancel it before cleaning "
                    f"its output."
                )
            }

    results = [clean_fn(Path(unit_path)) for unit_path in unit_paths]
    total_cleaned = sum(1 for result in results if result.get("cleaned", False))
    return {
        "pipeline": member.value,
        "results": results,
        "total_cleaned": total_cleaned,
        "total_units": len(results),
    }
