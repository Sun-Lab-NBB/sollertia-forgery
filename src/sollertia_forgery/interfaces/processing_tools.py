"""Provides the generic Model Context Protocol (MCP) tools for running any session-processing pipeline as a local
batch: preparing jobs, executing them non-blocking on the in-process engine, checking status, cancelling, and
resetting for replay. An agent selects the pipeline and decides which jobs to run, how many, and in what order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
import functools
from threading import Thread
from collections import deque

from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .mcp_instance import mcp
from ..orchestration import (
    RESERVED_CORES,
    JobExecutionState,
    group_jobs_by_tracker,
    job_execution_manager,
)
from ..orchestration.dispatch import (
    BATCH_PIPELINES,
    resolve_dispatch,
    build_pending_job,
    prepare_pipeline_jobs,
)

if TYPE_CHECKING:
    from ..orchestration import GenericPendingJob
    from ..orchestration.pipelines import ProcessingPipelines

_EXECUTION_STATE: dict[ProcessingPipelines, JobExecutionState[GenericPendingJob]] = {}
"""Per-pipeline batch execution state, keyed by pipeline. Each entry holds a running batch's job queues, worker
callable, and cancellation flag so the status, cancel, and reset tools read it directly."""

_STATUS_COUNT_KEYS: dict[ProcessingStatus, str] = {
    ProcessingStatus.SUCCEEDED: "succeeded",
    ProcessingStatus.FAILED: "failed",
    ProcessingStatus.RUNNING: "running",
    ProcessingStatus.SCHEDULED: "scheduled",
}
"""Maps each tracker job status to its aggregate-count key in the status response."""


def _ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def _error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}


def _unsupported_message(pipeline: str) -> str:
    """Builds the error message returned when a caller names a pipeline the batch tools do not support."""
    available = ", ".join(sorted(member.value for member in BATCH_PIPELINES))
    return f"Unsupported batch pipeline '{pipeline}'. Available: {available}."


def _collect_status(state: JobExecutionState[GenericPendingJob]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Re-reads the trackers of an execution state's jobs and returns per-job status entries and aggregate counts.

    Args:
        state: The batch execution state whose jobs to report.

    Returns:
        A tuple of the per-job status entries and a summary dict counting succeeded, failed, running, and scheduled
        jobs alongside the total.
    """
    per_job: list[dict[str, Any]] = []
    counts = {"succeeded": 0, "failed": 0, "running": 0, "scheduled": 0}
    for tracker_path, jobs in group_jobs_by_tracker(state=state).items():
        snapshot = ProcessingTracker(file_path=tracker_path).snapshot()
        for job in jobs:
            job_state = snapshot.get(job.job_id)
            status = job_state.status if job_state is not None else ProcessingStatus.SCHEDULED
            counts[_STATUS_COUNT_KEYS.get(status, "scheduled")] += 1
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "specifier": job.specifier,
                "status": status.name.lower(),
            }
            if job_state is not None and job_state.error_message is not None:
                entry["error_message"] = job_state.error_message
            per_job.append(entry)
    return per_job, {"total": len(per_job), **counts}


@mcp.tool()
def prepare_batch_tool(pipeline: str, session_paths: list[str]) -> dict[str, Any]:
    """Discovers and tracker-aligns the batch jobs for a session pipeline over one or more sessions.

    For each session, resolves the pipeline's runnable jobs, aligns the session's processing tracker so the job
    slots exist, and returns the dispatchable job descriptors. A session that cannot be prepared is reported in its
    own entry with an ``error`` key and does not abort the others.

    Args:
        pipeline: The batch pipeline to prepare, one of ``runtime``, ``microcontroller``, ``video``, ``two_photon``.
        session_paths: The session root directories to prepare jobs for.

    Returns:
        A response dict with ``pipeline`` and a ``units`` list, one entry per session carrying its ``session_path``,
        ``tracker_path``, and ``jobs``, or an ``error``. ``total_jobs`` counts the dispatchable jobs across sessions.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    units: list[dict[str, Any]] = []
    total_jobs = 0
    for session_path in session_paths:
        try:
            prepared = prepare_pipeline_jobs(dispatch=dispatch, session_path=Path(session_path))
        except Exception as exception:
            units.append({"session_path": session_path, "error": str(exception), "jobs": []})
            continue
        prepared["session_path"] = session_path
        units.append(prepared)
        total_jobs += len(prepared["jobs"])

    return _ok_response(pipeline=dispatch.pipeline.value, units=units, total_units=len(units), total_jobs=total_jobs)


@mcp.tool()
def execute_jobs_tool(
    pipeline: str,
    jobs: list[dict[str, Any]],
    *,
    worker_budget: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, Any]:
    """Dispatches a batch of prepared jobs for a session pipeline to the in-process engine, returning immediately.

    Execution runs on a background process pool. Only one batch per pipeline may run at a time, so the tool refuses
    while a batch is still live. Malformed job dicts are collected into ``invalid_jobs`` and the valid remainder
    still starts. The resolved core budget is split into concurrent jobs by the pipeline's per-job core count.

    Args:
        pipeline: The batch pipeline to run, one of ``runtime``, ``microcontroller``, ``video``, ``two_photon``.
        jobs: The job descriptors from ``prepare_batch_tool``, each carrying ``tracker_path``, ``job_id``, and
            ``session_path``.
        worker_budget: The total CPU cores the batch may use. A positive value is honored up to the physical core
            count, so a batch can claim every core. A non-positive value auto-resolves to all cores minus the reserved
            system cores.
        max_parallel_jobs: The hard cap on concurrently running jobs. A non-positive value defers to the pipeline's
            core-budget-derived default.

    Returns:
        A response dict with ``started``, the ``pipeline``, ``total_jobs`` dispatched, the resolved ``worker_budget``,
        and the effective ``max_parallel_jobs``. A partial batch also carries ``invalid_jobs``.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    active = _EXECUTION_STATE.get(dispatch.pipeline)
    if active is not None and active.manager_thread is not None and active.manager_thread.is_alive():
        return _error_response(
            message=(f"A {dispatch.pipeline.value} batch is already running. Wait for it to finish or cancel it first.")
        )

    pending: list[GenericPendingJob] = []
    invalid_jobs: list[dict[str, Any]] = []
    for job in jobs:
        try:
            pending.append(build_pending_job(job=job))
        except (KeyError, TypeError) as exception:
            invalid_jobs.append({"job": job, "error": str(exception)})

    if not pending:
        response = _error_response(message="No valid jobs to execute.")
        if invalid_jobs:
            response["invalid_jobs"] = invalid_jobs
        return response

    usable_cores = resolve_worker_count(requested_workers=worker_budget, reserved_cores=RESERVED_CORES)
    cores_per_job = dispatch.concurrency.cores_per_job
    budget_cap = max(1, usable_cores // cores_per_job)
    requested_cap = max_parallel_jobs if max_parallel_jobs > 0 else dispatch.concurrency.default_max_parallel
    effective_parallel = budget_cap if requested_cap <= 0 else min(requested_cap, budget_cap)

    state: JobExecutionState[GenericPendingJob] = JobExecutionState(
        worker=functools.partial(dispatch.worker, cores_per_job=cores_per_job),
        all_jobs={job.dispatch_key: job for job in pending},
        pending_queue=deque(pending),
        worker_budget=effective_parallel,
    )
    _EXECUTION_STATE[dispatch.pipeline] = state
    thread = Thread(target=job_execution_manager, args=(state,), daemon=True)
    state.manager_thread = thread
    thread.start()

    response = _ok_response(
        pipeline=dispatch.pipeline.value,
        started=True,
        total_jobs=len(pending),
        worker_budget=usable_cores,
        max_parallel_jobs=effective_parallel,
    )
    if invalid_jobs:
        response["invalid_jobs"] = invalid_jobs
    return response


@mcp.tool()
def get_processing_status_tool(pipeline: str) -> dict[str, Any]:
    """Reports the live status of the active batch for a session pipeline by re-reading its processing trackers.

    Args:
        pipeline: The batch pipeline to report, one of ``runtime``, ``microcontroller``, ``video``, ``two_photon``.

    Returns:
        A response dict with ``active`` (whether the manager thread is still running), ``canceled``, a per-job
        ``jobs`` list, and a ``summary`` counting succeeded, failed, running, and scheduled jobs. If no batch has
        run for the pipeline, ``active`` is False with an explanatory ``message``.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    state = _EXECUTION_STATE.get(dispatch.pipeline)
    if state is None:
        return _ok_response(
            pipeline=dispatch.pipeline.value,
            active=False,
            message="No batch has been executed for this pipeline.",
        )

    per_job, summary = _collect_status(state=state)
    running = state.manager_thread is not None and state.manager_thread.is_alive()
    return _ok_response(
        pipeline=dispatch.pipeline.value, active=running, canceled=state.canceled, jobs=per_job, summary=summary
    )


@mcp.tool()
def cancel_processing_tool(pipeline: str) -> dict[str, Any]:
    """Cooperatively cancels the active batch for a session pipeline. In-flight jobs finish, queued jobs are dropped.

    Args:
        pipeline: The batch pipeline to cancel, one of ``runtime``, ``microcontroller``, ``video``, ``two_photon``.

    Returns:
        A response dict with ``canceled`` and the number of queued jobs ``dropped``. Returns an error when no batch
        is running for the pipeline.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    state = _EXECUTION_STATE.get(dispatch.pipeline)
    if state is None:
        return _error_response(message=f"No {dispatch.pipeline.value} batch is running.")

    with state.lock:
        state.canceled = True
        dropped = len(state.pending_queue)
        state.pending_queue.clear()

    return _ok_response(
        pipeline=dispatch.pipeline.value,
        canceled=True,
        dropped_jobs=dropped,
        message="Cancellation requested. In-flight jobs will finish, queued jobs were dropped.",
    )


@mcp.tool()
def reset_processing_jobs_tool(pipeline: str, tracker_path: str, job_ids: list[str] | None = None) -> dict[str, Any]:
    """Resets tracked jobs to SCHEDULED so a subsequent execute reruns only them, preserving untargeted jobs.

    Opens the tracker file directly, so it works independently of any running batch. Requested job IDs absent from
    the tracker are ignored. When ``job_ids`` is omitted, every job in the tracker is reset.

    Args:
        pipeline: The batch pipeline the tracker belongs to, one of ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``.
        tracker_path: The absolute path to the pipeline's processing tracker file.
        job_ids: The job identifiers to reset. Omit to reset every job in the tracker.

    Returns:
        A response dict with ``tracker_path`` and the ``jobs_reset`` list. Returns an error when the tracker is
        missing, empty, or none of the requested IDs exist.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    path = Path(tracker_path)
    if not path.is_file():
        return _error_response(message=f"No tracker file found at '{tracker_path}'.")

    tracker = ProcessingTracker(file_path=path)
    snapshot = tracker.snapshot()
    if not snapshot:
        return _error_response(message=f"The tracker at '{tracker_path}' has no jobs.")

    target_ids = list(snapshot.keys()) if job_ids is None else [job_id for job_id in job_ids if job_id in snapshot]
    if not target_ids:
        return _error_response(message="None of the requested job IDs exist in the tracker.")

    tracker.reset_jobs(job_ids=target_ids)
    return _ok_response(pipeline=dispatch.pipeline.value, tracker_path=tracker_path, jobs_reset=target_ids)
