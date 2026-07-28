"""Provides the generic Model Context Protocol (MCP) tools for preparing session-processing jobs, inspecting what
they will cost, running them as one local batch, and checking, canceling, or resetting that batch.
"""

from __future__ import annotations

from time import time_ns
from uuid import uuid4
from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Thread
from collections import deque

from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker, delete_directory

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
    resolve_concurrency_limits,
    resolve_concurrency_reservations,
)

if TYPE_CHECKING:
    from ..orchestration import GenericPendingJob

_EXECUTION_STATE: JobExecutionState[GenericPendingJob] | None = None
"""The single batch execution state. One pool serves every pipeline, so a batch may hold any mix of jobs and the
engine packs them against one pair of budgets."""

_PREPARED_BATCHES: dict[str, list[dict[str, Any]]] = {}
"""The job descriptors every preparation produced, keyed by the identifier it returned. Execution resolves its jobs
from here when the caller names a batch, so dispatching a large batch costs one identifier rather than a copy of
every descriptor. Preparing one pipeline at a time yields one identifier each, and execution accepts them together,
which is how a single pool run comes to hold every pipeline."""

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
    pipeline: str,
    session_paths: list[str],
    options: dict[str, Any] | None = None,
    *,
    include_job_descriptors: bool = True,
) -> dict[str, Any]:
    """Discovers and tracker-aligns the batch jobs for a session pipeline over one or more sessions.

    For each session, resolves the pipeline's runnable jobs, aligns the session's processing tracker so the job
    slots exist, and registers the dispatchable job descriptors under a returned ``batch_id``. A session that cannot
    be prepared is reported in its own entry with an ``error`` key and does not abort the others.

    Pass the returned ``batch_id`` to ``execute_jobs_tool`` to dispatch the batch. Preparing several pipelines
    yields one identifier each, and execution accepts them together, which is how one pool run holds every pipeline.

    Args:
        pipeline: The batch pipeline to prepare, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories to prepare jobs for, which are session roots for every
            session pipeline and dataset roots for ``forging``.
        options: The pipeline-specific parameters to run the prepared jobs with, carried on every descriptor this
            call registers. The ``checksum`` pipeline reads ``regenerate_checksum``, a boolean selecting
            re-baselining of the stored value over verification against it, which defaults to verification. The
            ``forging`` pipeline reads ``session_names``, ``force_recreate``, and ``recreate_animals``, which its
            definition job applies to the dataset hierarchy and every other forging job ignores. The other pipelines
            take no parameters.
        include_job_descriptors: Determines whether each unit carries its full ``jobs`` list. Dispatch reads the
            descriptors from the identifier rather than from this response, so a batch spanning many sessions can
            omit them and report counts alone.

    Returns:
        A response dict with ``batch_id``, ``pipeline``, ``total_units``, ``total_jobs``, and a ``units`` list, one
        entry per session carrying its ``session_path``, ``session_name``, ``tracker_path``, and its ``job_count``,
        or an ``error``. Each unit also carries its ``jobs`` list unless the descriptors were omitted.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    units: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for session_path in session_paths:
        try:
            prepared = prepare_pipeline_jobs(dispatch=dispatch, unit_path=Path(session_path), options=options)
        except Exception as exception:
            units.append({"session_path": session_path, "error": str(exception), "job_count": 0, "jobs": []})
            continue
        prepared["session_name"] = prepared.pop("unit_name")
        prepared["session_path"] = session_path
        descriptors.extend(prepared["jobs"])
        prepared["job_count"] = len(prepared["jobs"])
        if not include_job_descriptors:
            del prepared["jobs"]
        units.append(prepared)

    batch_id = uuid4().hex[:16]
    _PREPARED_BATCHES[batch_id] = descriptors

    return _ok_response(
        batch_id=batch_id,
        pipeline=dispatch.pipeline.value,
        units=units,
        total_units=len(units),
        total_jobs=len(descriptors),
    )


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
            ``two_photon``, ``forging``.
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
    jobs: list[dict[str, Any]] | None = None,
    batch_ids: list[str] | None = None,
    *,
    core_budget_override: int = -1,
    memory_budget_mb: int = -1,
) -> dict[str, Any]:
    """Dispatches prepared jobs of any pipeline onto the shared pool, returning immediately.

    One pool serves every pipeline, so a single call may mix jobs from as many pipelines and sessions as the caller
    wants. Each job carries the cores and memory it needs, and the engine admits jobs continuously against both
    budgets, refilling the capacity a finished job frees as soon as it is released. Jobs run in their pipeline's
    own dependency order, so a batch may safely hold every stage of a pipeline at once.

    Name the batches to dispatch by their identifiers, which is what keeps the cost of starting a large run flat.
    Passing descriptors directly stays available for a caller that assembled or filtered its own job list.

    Args:
        jobs: The job descriptors to dispatch, each carrying ``tracker_path``, ``job_id``, ``unit_path``,
            ``pipeline``, ``job_name``, ``specifier``, ``cores``, ``memory_mb``, ``prerequisite_ids``, and
            ``options``. Supply this or ``batch_ids``, or both to dispatch their union.
        batch_ids: The identifiers ``prepare_batch_tool`` returned, whose registered descriptors are dispatched.
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

    unknown = sorted(batch for batch in (batch_ids or []) if batch not in _PREPARED_BATCHES)
    if unknown:
        return _error_response(
            message=(
                f"No prepared batch exists for identifier(s) {unknown}. Prepare the pipeline again to register its "
                f"jobs, since identifiers live only for the lifetime of the server that issued them."
            )
        )

    descriptors: list[dict[str, Any]] = list(jobs or [])
    for batch in batch_ids or []:
        descriptors.extend(_PREPARED_BATCHES[batch])

    pending: list[GenericPendingJob] = []
    invalid_jobs: list[dict[str, Any]] = []
    for job in descriptors:
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
    # tell its pipeline to fan out wider than this host can supply. The concurrency limits enter here too, so the
    # reported maximum for a storage-bound job type is the one admission will actually hold it to.
    concurrency_limits = resolve_concurrency_limits(job_names={job.job_name for job in pending})
    concurrency_reservations = resolve_concurrency_reservations(job_names={job.job_name for job in pending})
    allocations = resolve_core_allocations(
        job_cores={job.job_name: job.core_weight for job in pending},
        job_names={job.job_name for job in pending},
        core_budget=core_budget,
        job_limits=concurrency_limits,
        job_reservations=concurrency_reservations,
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
        concurrency_limits=concurrency_limits,
        concurrency_reservations=concurrency_reservations,
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
            job_name: {
                "cores_per_job": allocation.cores_per_job,
                "maximum_parallel": allocation.maximum_parallel,
                "concurrency_limit": allocation.concurrency_limit,
                "concurrency_reservation": allocation.concurrency_reservation,
            }
            for job_name, allocation in allocations.items()
        },
    )
    if invalid_jobs:
        response["invalid_jobs"] = invalid_jobs
    return response


@mcp.tool()
def get_processing_status_tool(
    status_filter: str | None = None,
    session_paths: list[str] | None = None,
    job_ids: list[str] | None = None,
    job_names: list[str] | None = None,
    pipelines: list[str] | None = None,
    *,
    include_jobs: bool = False,
) -> dict[str, Any]:
    """Reports the live status of the active batch by re-reading the processing trackers of every job it holds.

    Reports counts by default, because a batch spanning many sessions holds more jobs than a single response can
    carry. The ``breakdown`` resolves those counts per pipeline and job type, which is what tracking a run needs,
    and every failed job is always named in full so a failure is never hidden behind a count.

    Naming any filter narrows the reported jobs and returns them without asking for the listing separately, which is
    how a caller reads one job in full. Filters combine, so a session and a job name together name a single job. The
    counts and the breakdown always span the whole batch, so narrowing what is listed never distorts what is
    reported.

    Args:
        status_filter: Restricts the reported jobs to one status, one of ``succeeded``, ``failed``, ``running``, or
            ``scheduled``.
        session_paths: Restricts the reported jobs to these session root directories.
        job_ids: Restricts the reported jobs to these tracker job identifiers.
        job_names: Restricts the reported jobs to these job type names, such as ``motion_energy``.
        pipelines: Restricts the reported jobs to these pipelines.
        include_jobs: Determines whether the response carries an entry for every job the batch holds when no filter
            narrows them. A large batch omits them, since the counts and the failures answer what a run is doing.

    Returns:
        A response dict with ``active`` (whether the manager thread is still running), ``canceled``, a ``summary``
        counting succeeded, failed, running, and scheduled jobs, a ``breakdown`` of those counts per pipeline and
        job type, and ``failed_jobs`` naming every failure. Carries a ``jobs`` list, each entry holding the job's
        identity, its allocated cores and memory, its options and prerequisites, and the tracker's whole record of
        it, whenever a filter is named or the listing is requested. A batch that could not dispatch some jobs also
        carries ``blocked_jobs`` and a ``blocked_reason`` naming the upstream failure. If no batch has run,
        ``active`` is False with an explanatory ``message``.
    """
    state = _EXECUTION_STATE
    if state is None:
        return _ok_response(active=False, message="No batch has been executed yet.")

    if status_filter is not None and status_filter not in _STATUS_COUNT_KEYS.values():
        return _error_response(
            message=f"Unknown status '{status_filter}'. Available: {', '.join(sorted(_STATUS_COUNT_KEYS.values()))}."
        )

    per_job, summary = _collect_status(state=state)
    running = state.manager_thread is not None and state.manager_thread.is_alive()

    # Counts each pipeline's job types by status, which tracks a run at a size the response can always carry.
    tallies: dict[tuple[str, str, str], int] = {}
    for entry in per_job:
        key = (entry["pipeline"], entry["job_name"], entry["status"])
        tallies[key] = tallies.get(key, 0) + 1
    breakdown = [
        {"pipeline": pipeline, "job_name": job_name, "status": status, "count": count}
        for (pipeline, job_name, status), count in sorted(tallies.items())
    ]

    response = _ok_response(
        active=running,
        canceled=state.canceled,
        summary=summary,
        breakdown=breakdown,
        failed_jobs=[entry for entry in per_job if entry["status"] == "failed"],
    )

    selectors: dict[str, list[str] | None] = {
        "status": [status_filter] if status_filter is not None else None,
        "session_path": session_paths,
        "job_id": job_ids,
        "job_name": job_names,
        "pipeline": pipelines,
    }
    narrowed = any(values is not None for values in selectors.values())
    if narrowed or include_jobs:
        matches = [
            entry
            for entry in per_job
            if all(values is None or entry[field] in values for field, values in selectors.items())
        ]
        response["jobs"] = matches
        response["matched_jobs"] = len(matches)
    if state.blocked_jobs:
        response["blocked_jobs"] = [
            {
                "job_id": job.job_id,
                "pipeline": job.pipeline,
                "job_name": job.job_name,
                "specifier": job.specifier,
                "session_path": str(job.unit_path),
            }
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
            ``video``, ``two_photon``, ``forging``.
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


@mcp.tool()
def describe_jobs_tool(
    pipeline: str,
    session_paths: list[str],
    job_ids: list[str] | None = None,
    job_names: list[str] | None = None,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """Reports everything the processing trackers record about a pipeline's jobs for the named sessions.

    Reads the trackers on disk rather than a running batch, so it answers for work this server never dispatched and
    for runs that finished long ago. That makes it the tool for inspecting one job in full, while
    ``get_processing_status_tool`` follows a batch that is currently running.

    Every job the tracker holds is reported, whether the pipeline would resolve it as runnable today, so a
    stage that stopped being applicable stays visible. Sessions are read independently, and one that cannot be read
    is reported in its own entry without aborting the others.

    Args:
        pipeline: The pipeline whose tracker to read, one of ``checksum``, ``runtime``, ``microcontroller``,
            ``video``, ``two_photon``, ``forging``.
        session_paths: The session root directories to describe.
        job_ids: Restricts the reported jobs to these tracker job identifiers.
        job_names: Restricts the reported jobs to these job type names.
        status_filter: Restricts the reported jobs to one status, one of ``succeeded``, ``failed``, ``running``, or
            ``scheduled``.

    Returns:
        A response dict with ``pipeline``, ``total_units``, ``total_jobs`` matched across sessions, an aggregate
        ``summary`` of their statuses, and a ``units`` list. Each unit carries its ``session_path``,
        ``session_name``, ``tracker_path``, and whether the tracker ``tracker_exists``. Its ``jobs`` list holds each
        job's identity, the executor that ran it, its start and completion timestamps, its elapsed seconds, and any
        recorded error. A unit that could not be read carries an ``error`` instead.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    if status_filter is not None and status_filter not in _STATUS_COUNT_KEYS.values():
        return _error_response(
            message=f"Unknown status '{status_filter}'. Available: {', '.join(sorted(_STATUS_COUNT_KEYS.values()))}."
        )

    units: list[dict[str, Any]] = []
    counts = {"succeeded": 0, "failed": 0, "running": 0, "scheduled": 0}
    total_jobs = 0
    for session_path in session_paths:
        try:
            unit, _, _ = dispatch.discover(Path(session_path))
        except Exception as exception:
            units.append({"session_path": session_path, "error": str(exception)})
            continue

        tracker_path = dispatch.tracker_path(unit)
        snapshot = ProcessingTracker(file_path=tracker_path).snapshot() if tracker_path.is_file() else {}

        jobs: list[dict[str, Any]] = []
        for job_id, job_state in snapshot.items():
            status = job_state.status.name.lower()
            if job_ids is not None and job_id not in job_ids:
                continue
            if job_names is not None and job_state.job_name not in job_names:
                continue
            if status_filter is not None and status != status_filter:
                continue
            counts[_STATUS_COUNT_KEYS.get(job_state.status, "scheduled")] += 1
            jobs.append(
                {
                    "job_id": job_id,
                    "pipeline": dispatch.pipeline.value,
                    "job_name": job_state.job_name,
                    "specifier": job_state.specifier,
                    "session_path": session_path,
                    "tracker_path": str(tracker_path),
                    "status": status,
                    **_job_state_record(job_state=job_state),
                }
            )

        total_jobs += len(jobs)
        units.append(
            {
                "session_path": session_path,
                "session_name": dispatch.unit_name(unit),
                "tracker_path": str(tracker_path),
                "tracker_exists": tracker_path.is_file(),
                "jobs": jobs,
            }
        )

    return _ok_response(
        pipeline=dispatch.pipeline.value,
        units=units,
        total_units=len(units),
        total_jobs=total_jobs,
        summary={"total": total_jobs, **counts},
    )


@mcp.tool()
def clean_processing_output_tool(pipeline: str, session_paths: list[str]) -> dict[str, Any]:
    """Removes a pipeline's output and processing tracker for one or more sessions, returning them to an unprocessed
    state.

    Deletes the directory the pipeline owns outright alongside its tracker, so a subsequent preparation rediscovers
    every job from the acquired data rather than resuming a partial run. Sessions are cleaned independently, and one
    that cannot be cleaned is reported in its own entry without aborting the others.

    The ``checksum`` pipeline owns no directory, because it writes its stored value into the acquired data itself.
    Cleaning it removes its tracker and leaves that stored value in place, so the session keeps the baseline a later
    verification compares against. The ``forging`` pipeline owns its whole dataset hierarchy, so cleaning it removes
    every assembled feather in that dataset alongside the tracker.

    Args:
        pipeline: The batch pipeline to clean, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The session root directories to clean.

    Returns:
        A response dict with ``pipeline``, ``total_units``, ``removed_bytes`` freed across every session, and a
        ``units`` list carrying each session's ``removed_paths`` and ``removed_bytes``, or an ``error``. Returns an
        error when a batch is running, since removing the output of a job in flight would fail that job.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return _error_response(message=_unsupported_message(pipeline=pipeline))

    # A running batch holds open the very files this removes, so cleaning waits for the pool to drain.
    state = _EXECUTION_STATE
    if state is not None and state.manager_thread is not None and state.manager_thread.is_alive():
        return _error_response(
            message="A batch is currently running. Wait for it to finish or cancel it before cleaning output."
        )

    units: list[dict[str, Any]] = []
    total_removed = 0
    for session_path in session_paths:
        try:
            unit, _, _ = dispatch.discover(Path(session_path))
        except Exception as exception:
            units.append({"session_path": session_path, "error": str(exception)})
            continue

        targets = [dispatch.tracker_path(unit)]
        owned = dispatch.output_path(unit)
        if owned is not None:
            targets.append(owned)

        removed_paths: list[str] = []
        removed_bytes = 0
        for target in targets:
            if not target.exists():
                continue
            removed_bytes += _directory_size(path=target)
            if target.is_dir():
                delete_directory(directory_path=target)
            else:
                target.unlink()
                # The tracker's lock file is bookkeeping beside it rather than tracked output of its own.
                target.with_suffix(target.suffix + ".lock").unlink(missing_ok=True)
            removed_paths.append(str(target))

        total_removed += removed_bytes
        units.append(
            {
                "session_path": session_path,
                "session_name": dispatch.unit_name(unit),
                "removed_paths": removed_paths,
                "removed_bytes": removed_bytes,
            }
        )

    return _ok_response(
        pipeline=dispatch.pipeline.value,
        units=units,
        total_units=len(units),
        removed_bytes=total_removed,
    )


def _directory_size(path: Path) -> int:
    """Sums the bytes a path holds, counting a directory's whole tree and a file's own size.

    Args:
        path: The file or directory to measure.

    Returns:
        The size in bytes.
    """
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


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

    Notes:
        Every entry names the unit it belongs to, because a job identifier is derived from the job name and the
        specifier alone. A pipeline whose specifier does not vary by session therefore gives every session's copy of
        that stage one identifier, and the entries would be indistinguishable without the unit that separates them.

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
                "session_path": str(job.unit_path),
                "tracker_path": str(tracker_path),
                "status": status.name.lower(),
                "cores": job.core_weight,
                "memory_mb": job.memory_mb,
                "options": dict(job.options),
                "prerequisite_ids": list(job.prerequisite_ids),
            }
            entry.update(_job_state_record(job_state=job_state))
            per_job.append(entry)
    return per_job, {"total": len(per_job), **counts}


def _job_state_record(job_state: JobState | None) -> dict[str, Any]:
    """Renders a tracker's record of one job as a response payload.

    Notes:
        Reports the whole record rather than the status alone, because a caller asking about one job wants which
        executor ran it, when it started, and how long it took. A job the tracker does not know yet reports empty
        timing rather than an absent key, so every entry carries the same shape.

    Args:
        job_state: The tracker's record of the job, or None when the tracker holds no entry for it.

    Returns:
        A dictionary carrying the executor identifier, the start and completion timestamps, the elapsed seconds, and
        any recorded error message.
    """
    if job_state is None:
        return {"executor_id": None, "started_at": None, "completed_at": None, "elapsed_seconds": None}

    record: dict[str, Any] = {
        "executor_id": job_state.executor_id,
        "started_at": job_state.started_at,
        "completed_at": job_state.completed_at,
        "elapsed_seconds": _elapsed_seconds(job_state=job_state),
    }
    if job_state.error_message is not None:
        record["error_message"] = job_state.error_message
    return record


def _elapsed_seconds(job_state: JobState) -> float | None:
    """Resolves how long a job has run, measuring a finished job to its completion and a running one to now.

    Args:
        job_state: The tracker's record of the job.

    Returns:
        The elapsed seconds, or None when the job has not started.
    """
    if job_state.started_at is None:
        return None
    end = job_state.completed_at if job_state.completed_at is not None else time_ns() // 1000
    return round((end - job_state.started_at) / 1_000_000, 3)
