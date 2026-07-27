"""Provides the generic Model Context Protocol (MCP) tools for preparing session-processing jobs, inspecting what
they will cost, running them as one local batch, and checking, canceling, or resetting that batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Thread
from collections import deque

from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .mcp_instance import mcp
from ..orchestration import (
    RESERVED_CORES,
    BATCH_PIPELINES,
    JobExecutionState,
    run_batch_job,
    resolve_dispatch,
    build_pending_job,
    group_jobs_by_tracker,
    job_execution_manager,
    prepare_pipeline_jobs,
    resolve_host_memory_mb,
    resolve_core_allocations,
)

if TYPE_CHECKING:
    from ..orchestration import GenericPendingJob

_EXECUTION_STATE: JobExecutionState[GenericPendingJob] | None = None
"""The single batch execution state. One pool serves every pipeline, so a batch may hold any mix of jobs and the
engine packs them against one pair of budgets."""

_MEMORY_BUDGET_FRACTION: float = 0.85
"""The share of the host's memory a batch commits when the caller does not name one."""

_MINIMUM_MEMORY_BUDGET_MB: int = 1024
"""The floor the resolved memory budget never falls below, so a small host still admits one job at a time."""

_STATUS_COUNT_KEYS: dict[ProcessingStatus, str] = {
    ProcessingStatus.SUCCEEDED: "succeeded",
    ProcessingStatus.FAILED: "failed",
    ProcessingStatus.RUNNING: "running",
    ProcessingStatus.SCHEDULED: "scheduled",
}
"""Maps each tracker job status to its aggregate-count key in the status response."""


@mcp.tool()
def prepare_batch_tool(
    pipeline: str, session_paths: list[str], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Discovers and tracker-aligns the batch jobs for a session pipeline over one or more sessions.

    For each session, resolves the pipeline's runnable jobs, aligns the session's processing tracker so the job
    slots exist, and returns the dispatchable job descriptors. A session that cannot be prepared is reported in its
    own entry with an ``error`` key and does not abort the others.

    Args:
        pipeline: The batch pipeline to prepare, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``.
        session_paths: The session root directories to prepare jobs for.
        options: The pipeline-specific parameters to run the prepared jobs with, carried on every descriptor this
            call returns. The ``checksum`` pipeline reads ``regenerate_checksum``, a boolean selecting re-baselining
            of the stored value over verification against it, which defaults to verification. The other pipelines
            take no parameters.

    Returns:
        A response dict with ``pipeline``, ``total_units``, and a ``units`` list, one entry per session carrying its
        ``session_path``, ``session_name``, ``tracker_path``, and ``jobs``, or an ``error``. ``total_jobs`` counts
        the dispatchable jobs across sessions.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    units: list[dict[str, Any]] = []
    total_jobs = 0
    for session_path in session_paths:
        try:
            prepared = prepare_pipeline_jobs(dispatch=dispatch, session_path=Path(session_path), options=options)
        except Exception as exception:
            units.append({"session_path": session_path, "error": str(exception), "jobs": []})
            continue
        prepared["session_path"] = session_path
        units.append(prepared)
        total_jobs += len(prepared["jobs"])

    return _ok_response(pipeline=dispatch.pipeline.value, units=units, total_units=len(units), total_jobs=total_jobs)


@mcp.tool()
def inspect_job_resources_tool(
    pipeline: str, session_paths: list[str], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Reports the cores and memory every runnable job of a pipeline will need, without executing any of them.

    Estimates each job's memory from the data it will process, so a long recording is not charged the same as a
    short one. The figures already carry the shared tolerance, so they are the values to plan a local batch against
    or to request from a remote scheduler. Discovery runs as it does for a batch, so each session's tracker is
    created and aligned to its job universe.

    Args:
        pipeline: The pipeline to inspect, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``.
        session_paths: The session root directories to inspect.
        options: The pipeline-specific parameters the inspected jobs would run with, forwarded to preparation. See
            ``prepare_batch_tool`` for the keys each pipeline reads.

    Returns:
        A response dict with the host's ``total_memory_mb``, the batch-available ``total_cores`` left after the
        reserved system cores, and a ``units`` list carrying each session's per-job ``cores``, ``memory_mb``, and
        ``memory_modeled`` flag. The ``totals`` summary gives ``jobs``, ``jobs_without_a_modeled_estimate``,
        ``widest_job_cores``, ``largest_job_memory_mb``, and ``summed_memory_mb``, the maxima being taken
        independently over the same job list.
    """
    prepared = prepare_batch_tool(pipeline=pipeline, session_paths=session_paths, options=options)
    if not prepared["success"]:
        return prepared

    jobs = [job for unit in prepared["units"] for job in unit.get("jobs", [])]
    return _ok_response(
        pipeline=prepared["pipeline"],
        units=prepared["units"],
        total_cores=resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES),
        total_memory_mb=resolve_host_memory_mb(),
        totals={
            "jobs": len(jobs),
            "jobs_without_a_modeled_estimate": sum(1 for job in jobs if not job.get("memory_modeled", False)),
            "widest_job_cores": max((int(job["cores"]) for job in jobs), default=0),
            "largest_job_memory_mb": max((int(job["memory_mb"]) for job in jobs), default=0),
            "summed_memory_mb": sum(int(job["memory_mb"]) for job in jobs),
        },
    )


@mcp.tool()
def execute_jobs_tool(
    jobs: list[dict[str, Any]],
    *,
    core_budget_override: int = -1,
    memory_budget_mb: int = -1,
) -> dict[str, Any]:
    """Dispatches prepared jobs of any pipeline onto the shared pool, returning immediately.

    One pool serves every pipeline, so a single call may mix jobs from as many pipelines and sessions as the caller
    wants. Each job carries the cores and memory it needs, and the engine admits jobs continuously against both
    budgets, refilling the capacity a finished job frees as soon as it is released. Jobs run in their pipeline's
    own dependency order, so a batch may safely hold every stage of a pipeline at once.

    Args:
        jobs: The job descriptors from ``prepare_batch_tool``, each carrying ``tracker_path``, ``job_id``,
            ``session_path``, ``pipeline``, ``job_name``, ``specifier``, ``cores``, ``memory_mb``,
            ``prerequisite_ids``, and ``options``.
        core_budget_override: The cores the batch may use in total. A non-positive value auto-resolves to all cores
            minus the reserved system cores.
        memory_budget_mb: The memory the batch may use in total. A non-positive value auto-resolves to a share of
            the host's memory.

    Returns:
        A response dict with ``started``, ``total_jobs`` dispatched, the resolved ``core_budget`` and
        ``memory_budget_mb``, the ``pool_size``, a ``pipelines`` list naming what the batch holds, and a
        ``job_allocations`` entry per job type giving the cores it was narrowed to. A partial batch also carries
        ``invalid_jobs``.
    """
    global _EXECUTION_STATE

    if _EXECUTION_STATE is not None and (
        _EXECUTION_STATE.manager_thread is not None and _EXECUTION_STATE.manager_thread.is_alive()
    ):
        return _error_response(
            message="A batch is already running. Wait for it to finish or cancel it before starting another."
        )

    pending: list[GenericPendingJob] = []
    invalid_jobs: list[dict[str, Any]] = []
    for job in jobs:
        try:
            pending.append(build_pending_job(job=job))
        except (KeyError, TypeError) as exception:
            invalid_jobs.append({"job": job, "error": str(exception)})

    unsupported = sorted({job.pipeline for job in pending if resolve_dispatch(pipeline=job.pipeline) is None})
    if unsupported:
        return _error_response(message=_unsupported_message(pipeline=unsupported[0]))

    if not pending:
        response = _error_response(message="No valid jobs to execute.")
        if invalid_jobs:
            response["invalid_jobs"] = invalid_jobs
        return response

    core_budget = resolve_worker_count(requested_workers=core_budget_override, reserved_cores=RESERVED_CORES)
    resolved_memory = (
        memory_budget_mb
        if memory_budget_mb > 0
        else max(_MINIMUM_MEMORY_BUDGET_MB, int(resolve_host_memory_mb() * _MEMORY_BUDGET_FRACTION))
    )

    # Narrows every job to the resolved budget, since a descriptor prepared against a larger host would otherwise
    # tell its pipeline to fan out wider than this host can supply.
    allocations = resolve_core_allocations(
        job_cores={job.job_name: job.core_weight for job in pending},
        job_names={job.job_name for job in pending},
        core_budget=core_budget,
    )
    for pending_job in pending:
        pending_job.core_weight = allocations[pending_job.job_name].cores_per_job

    # Sizes the pool by how many of the narrowest jobs the core budget could admit at once, so worker processes are
    # never spawned for capacity the core budget cannot supply.
    narrowest = min((job.core_weight for job in pending), default=1)
    pool_size = max(1, min(len(pending), core_budget // max(1, narrowest)))

    state: JobExecutionState[GenericPendingJob] = JobExecutionState(
        worker=run_batch_job,
        all_jobs={job.dispatch_key: job for job in pending},
        pending_jobs=deque(pending),
        core_budget=core_budget,
        memory_budget_mb=resolved_memory,
        pool_size=pool_size,
    )
    _EXECUTION_STATE = state
    thread = Thread(target=job_execution_manager, args=(state,), daemon=True)
    state.manager_thread = thread
    thread.start()

    response = _ok_response(
        started=True,
        total_jobs=len(pending),
        core_budget=core_budget,
        memory_budget_mb=resolved_memory,
        pool_size=pool_size,
        pipelines=sorted({job.pipeline for job in pending}),
        job_allocations={
            job_name: {"cores_per_job": allocation.cores_per_job, "maximum_parallel": allocation.maximum_parallel}
            for job_name, allocation in allocations.items()
        },
    )
    if invalid_jobs:
        response["invalid_jobs"] = invalid_jobs
    return response


@mcp.tool()
def get_processing_status_tool() -> dict[str, Any]:
    """Reports the live status of the active batch by re-reading the processing trackers of every job it holds.

    Returns:
        A response dict with ``active`` (whether the manager thread is still running), ``canceled``, a per-job
        ``jobs`` list, and a ``summary`` counting succeeded, failed, running, and scheduled jobs. A batch that could
        not dispatch some jobs also carries ``blocked_jobs`` and a ``blocked_reason`` naming the upstream failure.
        If no batch has run, ``active`` is False with an explanatory ``message``.
    """
    state = _EXECUTION_STATE
    if state is None:
        return _ok_response(active=False, message="No batch has been executed yet.")

    per_job, summary = _collect_status(state=state)
    running = state.manager_thread is not None and state.manager_thread.is_alive()
    response = _ok_response(active=running, canceled=state.canceled, jobs=per_job, summary=summary)
    if state.blocked_jobs:
        response["blocked_jobs"] = [
            {"job_id": job.job_id, "pipeline": job.pipeline, "job_name": job.job_name, "specifier": job.specifier}
            for job in state.blocked_jobs
        ]
        response["blocked_reason"] = (
            "These jobs were never dispatched because a job they depend on failed or was never run. Run the upstream "
            "stage first, then execute them again."
        )
    return response


@mcp.tool()
def cancel_processing_tool() -> dict[str, Any]:
    """Cooperatively cancels the active batch.

    In-flight jobs finish and queued jobs are dropped.

    Returns:
        A response dict with ``canceled`` and the number of queued jobs in ``dropped_jobs``. Returns an error when no
        batch is currently running.
    """
    state = _EXECUTION_STATE
    if state is None or state.manager_thread is None or not state.manager_thread.is_alive():
        return _error_response(message="No batch is running.")

    with state.lock:
        state.canceled = True
        dropped = len(state.pending_jobs)
        state.pending_jobs.clear()

    return _ok_response(
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
        pipeline: The batch pipeline the tracker belongs to, one of ``checksum``, ``runtime``, ``microcontroller``,
            ``video``, ``two_photon``.
        tracker_path: The absolute path to the pipeline's processing tracker file.
        job_ids: The job identifiers to reset. Omit to reset every job in the tracker.

    Returns:
        A response dict with ``pipeline``, ``tracker_path``, and the ``jobs_reset`` list. Returns an error when the
        pipeline is not a supported batch pipeline, when the tracker is missing or empty, or when none of the
        requested identifiers exist.
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
                "pipeline": job.pipeline,
                "job_name": job.job_name,
                "specifier": job.specifier,
                "status": status.name.lower(),
            }
            if job_state is not None and job_state.error_message is not None:
                entry["error_message"] = job_state.error_message
            per_job.append(entry)
    return per_job, {"total": len(per_job), **counts}
