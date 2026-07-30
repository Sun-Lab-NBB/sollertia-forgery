"""Provides the generic Model Context Protocol (MCP) tools for preparing session-processing jobs, inspecting what
they will cost, running them as one local batch, and checking, canceling, or resetting that batch.
"""

from __future__ import annotations

from time import time_ns
from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Thread
from contextlib import contextmanager
from collections import deque

from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker, delete_directory

from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    error_response,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from .remote_tools import remote_batch_cancel, remote_batch_status
from ..orchestration import (
    RESERVED_CORES,
    BATCH_PIPELINES,
    LOCAL_HOST_LABEL,
    REMOTE_HOST_LABEL,
    REMOTE_JOB_WALLTIME_MINUTES,
    LocalHost,
    RemoteHost,
    JobExecutionState,
    read_ledger,
    submit_batch,
    prepare_batch,
    run_batch_job,
    resolve_dispatch,
    build_pending_job,
    connect_to_server,
    query_submissions,
    resolve_batch_host,
    reconcile_local_jobs,
    group_jobs_by_tracker,
    job_execution_manager,
    read_prepared_batches,
    reconcile_remote_jobs,
    record_prepared_batch,
    remote_batch_directory,
    resolve_host_memory_mb,
    resolve_core_allocations,
    resolve_concurrency_limits,
    resolve_concurrency_reservations,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..orchestration import ExecutionHost, GenericPendingJob

_EXECUTION_STATE: JobExecutionState[GenericPendingJob] | None = None
"""The single batch execution state. One pool serves every pipeline, so a batch may hold any mix of jobs and the
engine packs them against one pair of budgets."""

_HOST_LABELS: frozenset[str] = frozenset({LOCAL_HOST_LABEL, REMOTE_HOST_LABEL})
"""The hosts a batch may be prepared against and dispatched to."""

_BLOCKED_SEMI_FIELDS: tuple[str, ...] = (
    "job_id",
    "pipeline",
    "job_name",
    "specifier",
    "unit_name",
    "unsatisfied_prerequisite_ids",
)
"""The fields a blocked-job listing carries, naming the job and the upstream jobs this run could neither dispatch nor
find already succeeded."""

_MEMORY_BUDGET_FRACTION: float = 0.85
"""The share of the host's memory a batch commits when the caller does not name one."""

_MINIMUM_MEMORY_BUDGET_MB: int = 1024
"""The floor the resolved memory budget never falls below, so a small host still admits one job at a time."""

_STATUS_AXES: tuple[str, ...] = ("pipeline", "job_name", "status", "session_path")
"""The job attributes a caller may filter a batch by, and the axes a status breakdown counts."""

_STATUS_SEMI_FIELDS: tuple[str, ...] = ("job_id", "pipeline", "job_name", "specifier", "status", "session_path")
"""The job fields a semi-detail listing carries. ``job_id`` is included because it is the identifier a reset
targets."""

_STATUS_DETAIL_FIELDS: tuple[str, ...] = (
    "cores",
    "memory_mb",
    "elapsed_seconds",
    "executor_id",
    "error_message",
    "started_at",
    "completed_at",
    "options",
    "prerequisite_ids",
    "tracker_path",
)
"""The job fields detail adds, which are the resources the job was admitted at, its timing and provenance, the
parameters it ran with, and the jobs it waited for."""

_RESOURCE_SEMI_FIELDS: tuple[str, ...] = ("job_id", "job_name", "specifier", "cores", "memory_mb")
"""The job fields a semi-detail resource listing carries, which is the job's identity and its planned figures. The
unit path sits on the unit entry rather than on every job of that unit."""

_RESOURCE_DETAIL_FIELDS: tuple[str, ...] = ("memory_modeled", "prerequisite_ids", "unit_path", "options")
"""The job fields detail adds, stating whether the memory figure was modeled, which jobs it waits for, and the
parameters it would run with."""

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
    host: str = "local",
    *,
    replan: bool = False,
    include_job_descriptors: bool = False,
) -> dict[str, Any]:
    """Resolves a pipeline's dispatchable jobs for one or more units, on this machine or on the compute server.

    Runs the project's planning and state steps on whichever host holds the data, reads the resulting artifacts here,
    and builds the batch from them. One path serves both hosts, so a local batch and a remote one are the same
    document and differ only in where their jobs will run.

    A job the unit cannot run never reaches its processing tracker, so its absence from the project's state artifact is
    what rules it out. A job whose upstream stage this run can neither dispatch nor find already succeeded is reported
    under ``blocked_jobs`` rather than dispatched.

    Pass the returned ``batch_id`` to ``execute_jobs_tool``. A batch runs where it was prepared, so execution reads the
    host from the batch itself. Identifiers are recorded on disk and outlive the server that issued them.

    Args:
        pipeline: The batch pipeline to prepare, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories to prepare jobs for, which are session roots for every session
            pipeline and dataset roots for ``forging``. For ``remote`` these are paths ON THE SERVER. Every unit must
            belong to one project, since the artifacts a batch is resolved from are written per project.
        options: The pipeline-specific parameters to run the prepared jobs with, carried on every descriptor this call
            registers. The ``checksum`` pipeline reads ``regenerate_checksum``, a boolean selecting re-baselining of
            the stored value over verification against it, which defaults to verification. The ``forging`` pipeline
            reads ``session_names``, ``force_recreate``, and ``recreate_animals``, which its definition job applies to
            the dataset hierarchy and every other forging job ignores. The other pipelines take no parameters.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        replan: Determines whether to re-estimate the cores and memory the units' plan caches already hold. Leave False
            unless a deliberate retune should be adopted, since a submission may already have been sized against the
            recorded figures.
        include_job_descriptors: Determines whether the response carries the full ``jobs`` list. Omitted by default,
            since dispatch reads the descriptors from the identifier rather than from this response.

    Returns:
        A response dict with ``batch_id``, ``pipeline``, ``host``, ``total_units``, ``total_jobs``,
        ``total_blocked_jobs``, a ``units`` list carrying each unit's ``unit_path``, ``unit_name`` and ``job_count`` or
        an ``error``, and a ``blocked_jobs`` list naming what each blocked job waits on. Carries a ``jobs`` list when
        the descriptors are asked for.
    """
    if pipeline not in {member.value for member in BATCH_PIPELINES}:
        return error_response(message=_unsupported_message(pipeline=pipeline))
    if host not in _HOST_LABELS:
        return error_response(message=_unsupported_host_message(host=host))

    try:
        with _resolve_host(host=host) as execution_host:
            document = prepare_batch(
                host=execution_host,
                pipeline=pipeline,
                unit_paths=session_paths,
                options=options,
                replan=replan,
            )
    except Exception as exception:
        return error_response(message=f"Unable to prepare the {host} '{pipeline}' batch. {exception}")

    batch_id = record_prepared_batch(document=document)
    response = ok_response(
        batch_id=batch_id,
        pipeline=document.pipeline,
        host=document.host,
        units=document.units,
        total_units=len(document.units),
        total_jobs=len(document.jobs),
        total_blocked_jobs=len(document.blocked_jobs),
        blocked_jobs=[project_item(item=entry, fields=_BLOCKED_SEMI_FIELDS) for entry in document.blocked_jobs],
    )
    if include_job_descriptors:
        response["jobs"] = document.jobs
    return response


@mcp.tool()
def inspect_job_resources_tool(
    pipeline: str,
    session_paths: list[str],
    options: dict[str, Any] | None = None,
    host: str = "local",
    job_names: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reports the cores and memory a pipeline's possible jobs will need, in three widening stages, running none.

    A bare call reports the figures a batch is planned against alongside a ``breakdown`` naming every job type and how
    many of each the named sessions resolve. Naming a filter adds a page of jobs carrying their figures, and opting into
    detail adds whether each memory figure was modeled from the job's own input.

    Estimates each job's memory from the data it will process, so a long recording is not charged the same as a short
    one. The figures already carry the shared tolerance, so they are the values to plan a local batch against or to
    request from a remote scheduler. Discovery runs as it does for a batch, so each session's tracker is created and
    aligned to its job universe.

    Args:
        pipeline: The pipeline to inspect, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The session root directories to inspect.
        options: The pipeline-specific parameters the inspected jobs would run with, forwarded to preparation. See
            ``prepare_batch_tool`` for the keys each pipeline reads.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        job_names: Restricts the listing to these job type names.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs report whether their memory figure was modeled.

    Returns:
        A response dict with the host's ``total_memory_mb``, the batch-available ``total_cores`` left after the reserved
        system cores, a ``totals`` summary giving ``jobs``, ``jobs_without_a_modeled_estimate``, ``widest_job_cores``,
        ``largest_job_memory_mb``, and ``summed_memory_mb``, a ``breakdown`` per job type, and a ``units`` list naming
        each session and how many jobs it resolved. Carries a ``jobs`` list with ``rows``, ``matched_rows``,
        ``start_row``, and ``next_start_row`` whenever a filter is named or the listing is requested.
    """
    prepared = prepare_batch_tool(
        pipeline=pipeline, session_paths=session_paths, options=options, host=host, include_job_descriptors=True
    )
    if not prepared["success"]:
        return prepared

    jobs = prepared.get("jobs", [])
    units = prepared["units"]
    response = ok_response(
        pipeline=prepared["pipeline"],
        units=units,
        total_units=len(units),
        total_cores=resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES),
        total_memory_mb=resolve_host_memory_mb(),
        totals={
            "jobs": len(jobs),
            "jobs_without_a_modeled_estimate": sum(1 for job in jobs if not job.get("memory_modeled", False)),
            "widest_job_cores": max((int(job["cores"]) for job in jobs), default=0),
            "largest_job_memory_mb": max((int(job["memory_mb"]) for job in jobs), default=0),
            "summed_memory_mb": sum(int(job["memory_mb"]) for job in jobs),
        },
        breakdown={"job_name": count_values(values=[job["job_name"] for job in jobs])},
    )

    if job_names is None and not include_items:
        return response

    matched = jobs if job_names is None else [job for job in jobs if job["job_name"] in job_names]
    fields = (*_RESOURCE_SEMI_FIELDS, *_RESOURCE_DETAIL_FIELDS) if detailed else _RESOURCE_SEMI_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["jobs"] = [project_item(item=job, fields=fields) for job in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


@mcp.tool()
def execute_jobs_tool(
    batch_ids: list[str],
    *,
    core_budget_override: int = -1,
    memory_budget_mb: int = -1,
    walltime_minutes: int = -1,
) -> dict[str, Any]:
    """Dispatches prepared batches, onto this machine's process pool or onto the server's scheduler.

    A batch runs where it was prepared, so the host is read from the batch itself rather than named again. Batches
    prepared against different hosts are rejected rather than mixed.

    Before anything is dispatched, every job the trackers already record as running is reconciled. Locally that record
    describes a pool that died, so the job is rerun. Remotely the submission ledger and the tracker's executor
    identifier are consulted, and a job whose allocation is still live is adopted rather than submitted twice, with its
    dependents wired to wait on the allocation already running it. Every job that is dispatched has its recorded state
    cleared first, which is what keeps a status read honest across the window before it starts.

    Locally one pool serves every pipeline, so several batches may be dispatched together and the engine packs them
    against one pair of budgets. Remotely the scheduler sequences the dependency graph, so nothing has to stay running
    here for the batch to finish.

    Args:
        batch_ids: The identifiers ``prepare_batch_tool`` returned, whose recorded jobs are dispatched.
        core_budget_override: The cores a local batch may use in total. A non-positive value auto-resolves to all cores
            minus the reserved system cores. Ignored for a remote batch, where each job requests its own allocation.
        memory_budget_mb: The memory a local batch may use in total. A non-positive value auto-resolves to a share of
            the host's memory. Ignored for a remote batch.
        walltime_minutes: The wall-time every remote allocation requests. A non-positive value takes the shared
            default, which exists to stop a run that has stopped progressing. Ignored for a local batch.

    Returns:
        A response dict with ``started``, the ``host`` it dispatched to, ``total_jobs``, the ``pipelines`` the batch
        holds, and any ``adopted_jobs`` it left to an allocation already running them. A local dispatch adds the
        resolved ``core_budget``, ``memory_budget_mb``, ``pool_size``, and a ``job_allocations`` entry per job type. A
        remote dispatch adds ``walltime_minutes``, the ``batch_directory`` on the server, and a ``submissions`` list
        pairing each job with the allocation it runs as.
    """
    documents, missing = read_prepared_batches(batch_ids=batch_ids)
    if missing:
        return error_response(
            message=(
                f"No prepared batch exists for identifier(s) {missing}. Prepare the pipeline again to register its "
                f"jobs."
            )
        )
    if not documents:
        return error_response(message="No batch was named.")

    try:
        host = resolve_batch_host(documents=documents)
    except ValueError as exception:
        return error_response(message=str(exception))

    descriptors = [job for document in documents for job in document.jobs]
    if not descriptors:
        return error_response(message="No dispatchable jobs. Every prepared job is blocked or already succeeded.")

    pending: list[GenericPendingJob] = []
    invalid_jobs: list[dict[str, Any]] = []
    for job in descriptors:
        try:
            pending.append(build_pending_job(job=job))
        except (KeyError, TypeError) as exception:
            invalid_jobs.append(
                {
                    "job_id": job.get("job_id", ""),
                    "pipeline": job.get("pipeline", ""),
                    "unit_path": job.get("unit_path", ""),
                    "error": str(exception),
                }
            )
    if not pending:
        response = error_response(message="No valid jobs to execute.")
        if invalid_jobs:
            response["invalid_jobs"] = invalid_jobs
        return response

    if host == REMOTE_HOST_LABEL:
        response = _execute_remote_batch(pending=pending, batch_id=batch_ids[0], walltime_minutes=walltime_minutes)
    else:
        response = _execute_local_batch(
            host=LocalHost(),
            pending=pending,
            core_budget_override=core_budget_override,
            memory_budget_mb=memory_budget_mb,
        )
    if invalid_jobs and response["success"]:
        response["invalid_jobs"] = invalid_jobs
    return response


@mcp.tool()
def get_processing_status_tool(
    host: str = "local",
    batch_ids: list[str] | None = None,
    status_filter: str | None = None,
    session_paths: list[str] | None = None,
    job_ids: list[str] | None = None,
    job_names: list[str] | None = None,
    pipelines: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reports the live status of the active batch, in three widening stages.

    A bare call re-reads the processing trackers of every job the batch holds and reports the counts alongside a
    ``breakdown`` naming every pipeline, job type, status, and session in the batch. That is what tracks a run at a size
    a response can always carry, however many jobs it holds, and the counts are where a failure first shows.

    Naming a filter adds a page of jobs carrying identity and status. Filtering to ``failed`` is how a caller reads
    which jobs failed, and opting into detail adds each one's error text, timing, and the resources it was admitted at.

    Args:
        host: Which batch to report on, either ``local`` for this machine's pool or ``remote`` for the outstanding
            allocations on the server's scheduler.
        batch_ids: Restricts a ``remote`` report to these outstanding batches. Omit to cover all of them. Ignored for
            ``local``, where one pool holds one batch.
        status_filter: Restricts the listing to one status. Locally one of ``succeeded``, ``failed``, ``running``, or
            ``scheduled``, and remotely a scheduler state such as ``FAILED``, ``RUNNING``, or ``BLOCKED``.
        session_paths: Restricts the listing to these session root directories.
        job_ids: Restricts the listing to these tracker job identifiers.
        job_names: Restricts the listing to these job type names, such as ``motion_energy``.
        pipelines: Restricts the listing to these pipelines.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry their resources, timing, provenance, and error text.

    Returns:
        For ``remote``, a response dict with ``active``, the ``batches`` covered, a ``summary`` counting the allocations
        by scheduler state, and a ``breakdown`` per axis. For ``local``, a response dict with ``active`` (whether the
        manager thread is still running), ``canceled``, a ``summary``
        counting succeeded, failed, running, and scheduled jobs, and a ``breakdown`` per axis. Carries a ``jobs`` list
        with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a filter is named or the
        listing is requested. A batch that could not dispatch some jobs also reports ``blocked_jobs`` as a count with a
        ``blocked_reason``, and those jobs are listed by filtering to ``scheduled``. If no batch has run, ``active`` is
        False with an explanatory ``message``.
    """
    if host not in _HOST_LABELS:
        return error_response(message=_unsupported_host_message(host=host))
    if host == REMOTE_HOST_LABEL:
        return remote_batch_status(
            batch_ids=batch_ids,
            status_filter=status_filter,
            session_paths=session_paths,
            job_ids=job_ids,
            job_names=job_names,
            pipelines=pipelines,
            limit=limit,
            start_row=start_row,
            include_items=include_items,
            detailed=detailed,
        )

    state = _EXECUTION_STATE
    if state is None:
        return ok_response(active=False, message="No batch has been executed yet.")

    if status_filter is not None and status_filter not in _STATUS_COUNT_KEYS.values():
        return error_response(
            message=f"Unknown status '{status_filter}'. Available: {', '.join(sorted(_STATUS_COUNT_KEYS.values()))}."
        )

    per_job, summary = _collect_status(state=state)
    response = ok_response(
        active=state.manager_thread is not None and state.manager_thread.is_alive(),
        canceled=state.canceled,
        summary=summary,
        breakdown={axis: count_values(values=[entry[axis] for entry in per_job]) for axis in _STATUS_AXES},
    )
    if state.blocked_jobs:
        response["blocked_jobs"] = len(state.blocked_jobs)
        response["blocked_reason"] = (
            "These jobs were never dispatched because a job they depend on failed or was never run. Run the upstream "
            "stage first, then execute them again. List them by filtering to the 'scheduled' status."
        )

    selectors: dict[str, list[str] | None] = {
        "status": [status_filter] if status_filter is not None else None,
        "session_path": session_paths,
        "job_id": job_ids,
        "job_name": job_names,
        "pipeline": pipelines,
    }
    if not any(values is not None for values in selectors.values()) and not include_items:
        return response

    matched = [
        entry
        for entry in per_job
        if all(values is None or entry[field] in values for field, values in selectors.items())
    ]
    fields = (*_STATUS_SEMI_FIELDS, *_STATUS_DETAIL_FIELDS) if detailed else _STATUS_SEMI_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["jobs"] = [project_item(item=entry, fields=fields) for entry in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


@mcp.tool()
def cancel_processing_tool(host: str = "local", batch_ids: list[str] | None = None) -> dict[str, Any]:
    """Cancels the active local batch, or the outstanding allocations of the remote batches.

    Locally this is cooperative: in-flight jobs finish and queued jobs are dropped. Remotely it cancels queued and
    running allocations alike, and the scheduler cancels a dependent of a canceled allocation in turn because its
    dependency can no longer complete successfully.

    Args:
        host: Which batch to cancel, either ``local`` for this machine's pool or ``remote`` for the server's scheduler.
        batch_ids: The outstanding remote batches to cancel. Omit to cancel all of them. Ignored for ``local``, where
            one pool holds one batch.

    Returns:
        A response dict with ``canceled`` and, for ``local``, the number of queued jobs in ``dropped_jobs``. For
        ``remote`` it carries the ``canceled_jobs`` count and the ``batch_ids`` the cancellation covered. Returns an
        error when nothing is running or outstanding.
    """
    if host not in _HOST_LABELS:
        return error_response(message=_unsupported_host_message(host=host))
    if host == REMOTE_HOST_LABEL:
        return remote_batch_cancel(batch_ids=batch_ids)

    state = _EXECUTION_STATE
    if state is None or state.manager_thread is None or not state.manager_thread.is_alive():
        return error_response(message="No batch is running.")

    with state.lock:
        state.canceled = True
        dropped = len(state.pending_jobs)
        state.pending_jobs.clear()

    return ok_response(
        canceled=True,
        dropped_jobs=dropped,
        message="Cancellation requested. In-flight jobs will finish, queued jobs were dropped.",
    )


@mcp.tool()
def reset_processing_jobs_tool(pipeline: str, unit_path: str, job_ids: list[str] | None = None) -> dict[str, Any]:
    """Resets tracked jobs to SCHEDULED so a subsequent execute reruns only them, preserving untargeted jobs.

    Resolves the pipeline's tracker from the unit itself, so a caller names what it wants reset rather than where the
    tracker sits. That makes a mismatched pipeline and path impossible to express, and it is why no read tool has to
    carry tracker locations. Works independently of any running batch. Requested job IDs absent from the tracker are
    ignored, and omitting them resets every job the tracker holds.

    Args:
        pipeline: The pipeline whose jobs to reset, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        unit_path: The absolute path to the processing unit, which is a session root for every session pipeline and a
            dataset root for ``forging``.
        job_ids: The job identifiers to reset, as reported by any read tool's listing. Omit to reset every job.

    Returns:
        A response dict with ``pipeline``, ``unit_path``, the resolved ``tracker_path``, and the ``jobs_reset`` list.
        Returns an error when the pipeline is not supported, when the unit cannot be loaded, when the tracker is
        missing or empty, or when none of the requested identifiers exist.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return error_response(message=_unsupported_message(pipeline=pipeline))

    # Resolves the tracker without running discovery, so resetting never writes anything the way a preparation does.
    try:
        path = dispatch.tracker_path(dispatch.load(Path(unit_path)))
    except Exception as exception:
        return error_response(message=f"Unable to load the unit at '{unit_path}'. {exception}")

    if not path.is_file():
        return error_response(
            message=f"The '{dispatch.pipeline.value}' pipeline has no tracker for '{unit_path}', expected at '{path}'."
        )

    tracker = ProcessingTracker(file_path=path)
    snapshot = tracker.snapshot()
    if not snapshot:
        return error_response(message=f"The tracker at '{path}' has no jobs.")

    target_ids = list(snapshot.keys()) if job_ids is None else [job_id for job_id in job_ids if job_id in snapshot]
    if not target_ids:
        return error_response(message="None of the requested job IDs exist in the tracker.")

    tracker.reset_jobs(job_ids=target_ids)
    return ok_response(
        pipeline=dispatch.pipeline.value, unit_path=unit_path, tracker_path=str(path), jobs_reset=target_ids
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
        return error_response(message=_unsupported_message(pipeline=pipeline))

    # A running batch holds open the very files this removes, so cleaning waits for the pool to drain.
    state = _EXECUTION_STATE
    if state is not None and state.manager_thread is not None and state.manager_thread.is_alive():
        return error_response(
            message="A batch is currently running. Wait for it to finish or cancel it before cleaning output."
        )

    units: list[dict[str, Any]] = []
    total_removed = 0
    for session_path in session_paths:
        # Loads the unit rather than resolving its jobs, since a cleanup needs the unit's own locations alone and a
        # unit whose pipeline has never been prepared still has output to remove.
        try:
            unit = dispatch.load(Path(session_path))
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

    return ok_response(
        pipeline=dispatch.pipeline.value,
        units=units,
        total_units=len(units),
        removed_bytes=total_removed,
    )


@contextmanager
def _resolve_host(host: str) -> Iterator[ExecutionHost]:
    """Opens the named execution host, closing a server connection when the caller is done with it.

    Args:
        host: Either ``local`` for this machine or ``remote`` for the configured compute server.

    Yields:
        The execution host preparation drives.
    """
    if host == REMOTE_HOST_LABEL:
        with connect_to_server() as server:
            yield RemoteHost(server=server)
        return
    yield LocalHost()


def _reset_batch_jobs(host: ExecutionHost, jobs: list[GenericPendingJob]) -> None:
    """Clears the recorded state of every job about to be dispatched, on the host that records it.

    Notes:
        Jobs are grouped by the pipeline and unit that record them, so each tracker is rewritten once however many of
        its jobs the batch holds.

    Args:
        host: The host holding the trackers.
        jobs: The jobs whose records to clear.
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    for job in jobs:
        grouped.setdefault((job.pipeline, str(job.unit_path)), []).append(job.job_id)
    for (pipeline, unit_path), job_ids in grouped.items():
        host.reset_jobs(pipeline=pipeline, unit_path=Path(unit_path), job_ids=job_ids)


def _execute_local_batch(
    host: ExecutionHost, pending: list[GenericPendingJob], core_budget_override: int, memory_budget_mb: int
) -> dict[str, Any]:
    """Reconciles a local batch and dispatches it onto the shared process pool.

    Args:
        host: The host holding the trackers, which the reset is applied through.
        pending: The batch's jobs.
        core_budget_override: The cores the batch may use in total, or a non-positive value to auto-resolve.
        memory_budget_mb: The memory the batch may use in total, or a non-positive value to auto-resolve.

    Returns:
        The response dict the calling tool returns.
    """
    global _EXECUTION_STATE

    if _EXECUTION_STATE is not None and (
        _EXECUTION_STATE.manager_thread is not None and _EXECUTION_STATE.manager_thread.is_alive()
    ):
        return error_response(
            message="A batch is already running. Wait for it to finish or cancel it before starting another."
        )

    reconciliation = reconcile_local_jobs(jobs=pending)
    _reset_batch_jobs(host=host, jobs=reconciliation.resettable)
    dispatchable = reconciliation.dispatchable

    core_budget = resolve_worker_count(requested_workers=core_budget_override, reserved_cores=RESERVED_CORES)
    resolved_memory = (
        memory_budget_mb
        if memory_budget_mb > 0
        else max(_MINIMUM_MEMORY_BUDGET_MB, int(resolve_host_memory_mb() * _MEMORY_BUDGET_FRACTION))
    )

    # Narrows every job to the resolved budget, since a descriptor planned against a wider host would otherwise tell
    # its pipeline to fan out wider than this host can supply. The concurrency limits enter here too, so the reported
    # maximum for a storage-bound job type is the one admission actually enforces.
    concurrency_limits = resolve_concurrency_limits(job_names={job.job_name for job in dispatchable})
    concurrency_reservations = resolve_concurrency_reservations(job_names={job.job_name for job in dispatchable})
    allocations = resolve_core_allocations(
        job_cores={job.job_name: job.core_weight for job in dispatchable},
        job_names={job.job_name for job in dispatchable},
        core_budget=core_budget,
        job_limits=concurrency_limits,
        job_reservations=concurrency_reservations,
    )
    for pending_job in dispatchable:
        pending_job.core_weight = allocations[pending_job.job_name].cores_per_job

    # Sizes the pool by how many of the narrowest jobs the core budget could admit at once, so worker processes are
    # never spawned for capacity the core budget cannot supply.
    narrowest = min((job.core_weight for job in dispatchable), default=1)
    pool_size = max(1, min(len(dispatchable), core_budget // max(1, narrowest)))

    state: JobExecutionState[GenericPendingJob] = JobExecutionState(
        worker=run_batch_job,
        all_jobs={job.dispatch_key: job for job in dispatchable},
        pending_jobs=deque(dispatchable),
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

    return ok_response(
        started=True,
        host=LOCAL_HOST_LABEL,
        total_jobs=len(dispatchable),
        core_budget=core_budget,
        memory_budget_mb=resolved_memory,
        pool_size=pool_size,
        pipelines=sorted({job.pipeline for job in dispatchable}),
        adopted_jobs=[],
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


def _execute_remote_batch(pending: list[GenericPendingJob], batch_id: str, walltime_minutes: int) -> dict[str, Any]:
    """Reconciles a remote batch and submits it to the server's scheduler as a dependency graph.

    Notes:
        A submission spanning several batches writes the scripts and logs of them all into one directory, named after
        the first batch.

    Args:
        pending: The batch's jobs.
        batch_id: The identifier naming the directory the scripts and logs are written into.
        walltime_minutes: The wall-time every allocation requests, or a non-positive value to take the shared default.

    Returns:
        The response dict the calling tool returns.
    """
    walltime = walltime_minutes if walltime_minutes > 0 else REMOTE_JOB_WALLTIME_MINUTES
    try:
        with connect_to_server() as server:
            reconciliation = reconcile_remote_jobs(server=server, jobs=pending)
            _reset_batch_jobs(host=RemoteHost(server=server), jobs=reconciliation.resettable)
            descriptors = [_render_descriptor(job=job) for job in reconciliation.dispatchable]

            submissions = submit_batch(
                server=server,
                jobs=descriptors,
                batch_id=batch_id,
                adopted=reconciliation.adopted,
                walltime_minutes=walltime,
            )
            batch_directory = str(remote_batch_directory(server=server, batch_id=batch_id))

            # Retires the earlier batches that finished while this one was prepared, so the ledger sheds them without
            # waiting for a status read that may never come.
            outstanding = [submission for batch in read_ledger().batches for submission in batch.submissions]
            if outstanding:
                query_submissions(server=server, submissions=outstanding)
    except Exception as exception:
        return error_response(message=f"Unable to submit the remote batch. {exception}")

    return ok_response(
        started=True,
        host=REMOTE_HOST_LABEL,
        batch_id=batch_id,
        total_jobs=len(submissions),
        walltime_minutes=walltime,
        pipelines=sorted({submission.pipeline for submission in submissions}),
        batch_directory=batch_directory,
        adopted_jobs=[
            {"unit_path": unit_path, "job_id": job_id, "slurm_job_id": allocation}
            for (unit_path, job_id), allocation in sorted(reconciliation.adopted.items())
        ],
        submissions=[
            {"job_id": submission.job_id, "slurm_job_id": submission.slurm_job_id, "job_name": submission.job_name}
            for submission in submissions
        ],
    )


def _render_descriptor(job: GenericPendingJob) -> dict[str, Any]:
    """Renders one reconciled job as the descriptor a submission dispatches.

    Args:
        job: The job to render.

    Returns:
        The job descriptor.
    """
    return {
        "job_id": job.job_id,
        "job_name": job.job_name,
        "specifier": job.specifier,
        "unit_path": str(job.unit_path),
        "unit_name": job.name,
        "pipeline": job.pipeline,
        "tracker_path": str(job.tracker_path),
        "cores": job.core_weight,
        "memory_mb": job.memory_mb,
        "prerequisite_ids": list(job.prerequisite_ids),
        "options": dict(job.options),
    }


def _unsupported_host_message(host: str) -> str:
    """Builds the error message returned when a caller names a host the tools do not support."""
    return f"Unsupported host '{host}'. Available: {', '.join(sorted(_HOST_LABELS))}."


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
