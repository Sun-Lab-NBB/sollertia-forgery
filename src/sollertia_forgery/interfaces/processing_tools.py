"""Provides the generic Model Context Protocol (MCP) tools for preparing processing jobs, inspecting their cost,
running them as one batch on this machine or the compute server, checking, canceling, or resetting that batch, and
removing pipeline output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Thread
from collections import deque

from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

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
    close_batch,
    read_ledger,
    submit_batch,
    prepare_batch,
    run_batch_job,
    build_pending_job,
    connect_to_server,
    current_timestamp,
    query_submissions,
    read_batch_outcome,
    resolve_batch_host,
    reconcile_local_jobs,
    close_settled_batches,
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
from .host_resolution import (
    HOST_LABELS,
    resolve_execution_host,
    unsupported_host_message,
)

if TYPE_CHECKING:
    from ..orchestration import ExecutionHost, GenericPendingJob

_EXECUTION_STATE: JobExecutionState[GenericPendingJob] | None = None
"""The single batch execution state. One pool serves every pipeline, so a batch may hold any mix of jobs and the
engine packs them against one pair of budgets."""

_CLOSED_BATCH_MESSAGE: str = (
    "No batch is running in this process. The named batches have closed, so their outcomes are read from the snapshot "
    "closure recorded rather than from a live pool."
)
"""The message returned when a caller asks about batches that already reached closure. Their counts are durable, so an
answer survives the process that ran them."""

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
"""The floor an auto-resolved memory budget never falls below, so a small host still admits one job at a time. A budget
the caller names explicitly is honored as given."""

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
"""The job fields a semi-detail resource listing carries, which is the job's identity and its planned figures. The unit
path is left off a semi-detail row because the unit entry already names it, and detail adds it back for a caller
reading one job closely."""

_RESOURCE_DETAIL_FIELDS: tuple[str, ...] = ("prerequisite_ids", "unit_path", "options")
"""The job fields detail adds, naming the unit the job reads, which jobs it waits for, and the parameters it would run
with."""

_STATUS_LABELS: tuple[str, ...] = tuple(member.name.lower() for member in ProcessingStatus)
"""The status labels a tracked job reports, which are the tracker's own status names in lower case. These are the
values a caller filters the listing by, and the keys a tracker summary counts under."""


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
            the stored value over verification against it, which defaults to verification. Every other pipeline,
            including ``forging``, takes no parameters. Build or extend a dataset hierarchy with
            ``define_forging_dataset_tool``, which takes ``session_names``, ``force_recreate``, and
            ``recreate_animals`` directly.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        replan: Determines whether to re-estimate the cores and memory the units' plan caches already hold. Leave False
            unless a deliberate retune should be adopted, since a submission may already have been sized against the
            recorded figures.
        include_job_descriptors: Determines whether the response carries the full ``jobs`` list. Omitted by default,
            since dispatch reads the descriptors from the identifier rather than from this response.

    Returns:
        A response dict with ``batch_id``, ``pipeline``, ``host``, ``total_units``, ``total_jobs``,
        ``total_blocked_jobs``, a ``units`` list carrying each unit's ``unit_path``, ``unit_name``, ``job_count``, and
        ``blocked_count``, or its ``unit_path`` and an ``error``, and a ``blocked_jobs`` list naming what each blocked
        job waits on. Carries a ``jobs`` list when the descriptors are asked for.
    """
    if pipeline not in {member.value for member in BATCH_PIPELINES}:
        return error_response(message=_unsupported_message(pipeline=pipeline))
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    try:
        with resolve_execution_host(host=host) as execution_host:
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
    """Reports the cores and memory a pipeline's outstanding jobs will need, in three widening stages, running none.

    A bare call reports the figures a batch is planned against alongside a ``breakdown`` naming every job type and how
    many of each the named sessions still have to run. A job the units already recorded as succeeded, and a job this
    run could not unblock, are both absent, so this reports what a batch would dispatch rather than the whole universe.
    Naming a filter adds a page of jobs carrying their figures, and opting into detail adds the unit each job reads,
    the jobs it waits for, and the parameters it would run with.

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
        detailed: Determines whether the listed jobs report the unit they read, the jobs they wait for, and the
            parameters they would run with.

    Returns:
        A response dict with a ``totals`` summary giving ``jobs``, ``widest_job_cores``, ``largest_job_memory_mb``,
        and ``summed_memory_mb``. Carries a ``breakdown`` per job type and a ``units`` list naming each session and how
        many jobs it resolved. Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and
        ``next_start_row`` whenever a filter is named or the listing is requested. For ``local`` it also carries this
        machine's ``total_memory_mb`` and the batch-available ``total_cores`` left after the reserved system cores;
        both are absent for ``remote``, where the scheduler holds the budgets and the caller names what a job requests.
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
        host=host,
        units=units,
        total_units=len(units),
        totals={
            "jobs": len(jobs),
            "widest_job_cores": max((int(job["cores"]) for job in jobs), default=0),
            "largest_job_memory_mb": max((int(job["memory_mb"]) for job in jobs), default=0),
            "summed_memory_mb": sum(int(job["memory_mb"]) for job in jobs),
        },
        breakdown={"job_name": count_values(values=[job["job_name"] for job in jobs])},
    )

    # Both figures read the machine this process runs on, so they describe the host only when the data sits here. A
    # remote batch is submitted with the budgets its caller names and the scheduler enforces its own limits, so
    # reporting this workstation's cores and memory against a server project would describe the wrong machine.
    if host == LOCAL_HOST_LABEL:
        response["total_cores"] = resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES)
        response["total_memory_mb"] = resolve_host_memory_mb()

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
        remote dispatch adds the ``batch_id`` its scripts and logs are filed under, ``walltime_minutes``, the
        ``batch_directory`` on the server, and a ``submissions`` list pairing each job with the allocation it runs as.
        Either host adds an ``invalid_jobs`` list when a recorded descriptor could not be built into a job.
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
        response = _execute_remote_batch(pending=pending, batch_ids=batch_ids, walltime_minutes=walltime_minutes)
    else:
        response = _execute_local_batch(
            host=LocalHost(),
            pending=pending,
            batch_ids=batch_ids,
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
    a response can always carry, however many jobs it holds, and the counts are where a failure first shows. The counts
    cover the batch's own jobs alone, so their total is the number of jobs the batch dispatched and a job of the same
    tracker that this batch did not dispatch is left out of them. The ``status`` label the response carries is what
    those same counts resolve to, so the label and the counts describe one set.

    Naming a filter adds a page of jobs carrying identity and status. Filtering to ``failed`` is how a caller reads
    which jobs failed, and opting into detail adds each one's error text, timing, and the resources it was admitted at.

    A batch that has finished carries an ``outcomes`` entry, which is the durable snapshot closure took of what its
    jobs recorded. Read ``complete``, ``succeeded``, ``failed``, ``blocked``, and ``outstanding`` from it to decide
    whether the run needs anything further, and ``failed_jobs`` for the error text each failure recorded.

    Args:
        host: Which batch to report on, either ``local`` for this machine's pool or ``remote`` for the outstanding
            allocations on the server's scheduler.
        batch_ids: Restricts a ``remote`` report to these outstanding batches. Omit to cover all of them. Naming any
            batch also counts as a filter, so the response carries a page of jobs. Ignored for ``local``, where one
            pool holds one batch.
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
        by scheduler state, a ``breakdown`` per axis, and the ``outcomes`` of any batch that closed on this call. For
        ``local``, a response dict with ``active`` (whether the manager thread is still running), ``canceled``, a
        ``summary`` counting the batch's succeeded, failed, running, and scheduled jobs alongside their total, the
        ``status`` label those counts resolve to, and a ``breakdown`` per axis. Carries a ``jobs`` list with ``rows``,
        ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a filter is named or the listing is requested.
        A batch that could not dispatch some jobs also reports ``blocked_jobs`` as a count with a ``blocked_reason``,
        and those jobs are listed by filtering to ``scheduled``. If no batch has run, ``active`` is False with an
        explanatory ``message``.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
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
        recorded = [outcome for batch in batch_ids or [] if (outcome := read_batch_outcome(batch_id=batch)) is not None]
        if recorded:
            return ok_response(active=False, outcomes=recorded, message=_CLOSED_BATCH_MESSAGE)
        return ok_response(active=False, message="No batch is running in this process.")

    if status_filter is not None and status_filter not in _STATUS_LABELS:
        return error_response(
            message=f"Unknown status '{status_filter}'. Available: {', '.join(sorted(_STATUS_LABELS))}."
        )

    per_job, summary = _collect_status(state=state)
    response = ok_response(
        active=state.manager_thread is not None and state.manager_thread.is_alive(),
        canceled=state.canceled,
        status=ProcessingTracker.resolve_status(summary=summary).value,
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
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
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
def reset_processing_jobs_tool(
    pipeline: str, unit_paths: list[str], job_ids: list[str] | None = None, host: str = "local"
) -> dict[str, Any]:
    """Resets tracked jobs to SCHEDULED across one or more units, so a later execute reruns only them.

    Resolves each pipeline tracker from its unit, so a caller names what it wants reset rather than where the tracker
    sits. That makes a mismatched pipeline and path impossible to express, and it is why no read tool has to carry
    tracker locations.

    One call covers every named unit, because each unit resets only the identifiers it actually tracks. Passing a whole
    batch's identifiers alongside all of its units therefore costs a single operation, which remotely is one lightweight
    server-side invocation rather than one per unit. Omitting the identifiers resets every job each unit tracks.

    Works independently of any running batch, and note that ``execute_jobs_tool`` already resets what it dispatches, so
    this is for the case where a caller wants a unit returned to a clean slate without running anything.

    Args:
        pipeline: The pipeline whose jobs to reset, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        unit_paths: The processing unit directories whose jobs to reset, which are session roots for every session
            pipeline and dataset roots for ``forging``. These are paths ON THE SERVER for ``remote``.
        job_ids: The job identifiers to reset, as reported by any read tool's listing. Omit to reset every job each
            named unit tracks.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        A response dict with ``pipeline``, ``host``, ``total_units``, a ``jobs_reset`` count of the identifiers the
        caller named, which is null when none were named, and a ``message`` stating what was reset. The host does not
        report how many records it actually cleared, so ``jobs_reset`` is an upper bound rather than an outcome.
        Returns an error when the pipeline or the host is not supported.
    """
    if pipeline not in {member.value for member in BATCH_PIPELINES}:
        return error_response(message=_unsupported_message(pipeline=pipeline))
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    units = [Path(path) for path in unit_paths]
    try:
        with resolve_execution_host(host=host) as execution_host:
            execution_host.reset_jobs(pipeline=pipeline, job_ids_by_unit=dict.fromkeys(units, tuple(job_ids or ())))
    except Exception as exception:
        return error_response(message=f"Unable to reset the {host} '{pipeline}' jobs. {exception}")

    return ok_response(
        pipeline=pipeline,
        host=host,
        total_units=len(units),
        jobs_reset=len(job_ids) if job_ids else None,
        message=(
            "Reset every job each named unit tracks."
            if not job_ids
            else "Reset the named identifiers each unit tracks, skipping the ones it does not."
        ),
    )


@mcp.tool()
def clean_processing_output_tool(pipeline: str, session_paths: list[str], host: str = "local") -> dict[str, Any]:
    """Removes a pipeline's output and processing tracker for one or more units, on this machine or on the server.

    Returns each unit to an unprocessed state, so a later preparation rediscovers every job from the acquired data
    rather than resuming a partial run. Removal reports the bytes each path held either way, since the host runs the
    same removal whichever side of the connection it sits on.

    The ``checksum`` pipeline owns no directory, because it writes its stored value into the acquired data itself.
    Cleaning it removes its tracker and leaves that stored value in place, so the unit keeps the baseline a later
    verification compares against. The ``forging`` pipeline owns its whole dataset hierarchy, so cleaning it removes
    every assembled feather in that dataset alongside the tracker.

    Args:
        pipeline: The batch pipeline to clean, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories to clean, which are paths ON THE SERVER for ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        A response dict with ``pipeline``, ``host``, ``total_paths`` removed, ``removed_bytes`` freed across every
        unit, and a ``removed`` list carrying each path and the bytes it held. Returns an error when a batch is
        running locally, since removing the output of a job in flight would fail that job.
    """
    if pipeline not in {member.value for member in BATCH_PIPELINES}:
        return error_response(message=_unsupported_message(pipeline=pipeline))
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    # A running batch holds open the very files this removes, so cleaning waits for the pool to drain.
    state = _EXECUTION_STATE
    running = state is not None and state.manager_thread is not None and state.manager_thread.is_alive()
    if host == LOCAL_HOST_LABEL and running:
        return error_response(
            message="A batch is currently running. Wait for it to finish or cancel it before cleaning output."
        )

    try:
        with resolve_execution_host(host=host) as execution_host:
            removed = execution_host.clean(pipeline=pipeline, unit_paths=[Path(path) for path in session_paths])
    except Exception as exception:
        return error_response(message=f"Unable to clean the {host} '{pipeline}' output. {exception}")

    return ok_response(
        pipeline=pipeline,
        host=host,
        removed=removed,
        total_paths=len(removed),
        removed_bytes=sum(int(entry["removed_bytes"]) for entry in removed),
    )


def _reset_batch_jobs(host: ExecutionHost, jobs: list[GenericPendingJob]) -> None:
    """Clears the recorded state of every job about to be dispatched, on the host that records it.

    Notes:
        Each unit carries the identifiers of its own dispatched jobs alone. A job identifier is derived from the job
        name and the specifier alone, so two units of one project share the identifier of the same stage, and a flat
        set applied to both would clear a succeeded record this batch never dispatched.

        Jobs are still grouped by pipeline, so one operation carries every unit of a pipeline and a batch spanning many
        units stays at one round trip per pipeline.

    Args:
        host: The host holding the trackers.
        jobs: The jobs whose records to clear.
    """
    grouped: dict[str, dict[Path, set[str]]] = {}
    for job in jobs:
        grouped.setdefault(job.pipeline, {}).setdefault(job.unit_path, set()).add(job.job_id)

    for pipeline, identifiers_by_unit in grouped.items():
        host.reset_jobs(
            pipeline=pipeline,
            job_ids_by_unit={unit_path: sorted(identifiers) for unit_path, identifiers in identifiers_by_unit.items()},
        )


def _run_and_close_local_batch(
    state: JobExecutionState[GenericPendingJob], host: ExecutionHost, batch_ids: list[str]
) -> None:
    """Runs a local batch to completion, then closes every batch it held.

    Notes:
        Closure runs in the same thread the manager did, so it happens the moment the queue drains rather than waiting
        for a caller to ask. A failure to close is reported and swallowed, because the jobs themselves have already run
        and recorded their outcomes.

    Args:
        state: The batch execution state the manager dispatches from.
        host: The host holding the data the batch's jobs read.
        batch_ids: The identifiers of the batches this run dispatched.
    """
    job_execution_manager(state=state)
    for batch_id in batch_ids:
        try:
            close_batch(host=host, batch_id=batch_id)
        except Exception as exception:
            console.echo(
                message=f"Unable to close the finished batch '{batch_id}'. {exception}", level=LogLevel.WARNING
            )


def _execute_local_batch(
    host: ExecutionHost,
    pending: list[GenericPendingJob],
    batch_ids: list[str],
    core_budget_override: int,
    memory_budget_mb: int,
) -> dict[str, Any]:
    """Reconciles a local batch and dispatches it onto the shared process pool.

    Args:
        host: The host holding the trackers, which the reset is applied through.
        pending: The batch's jobs.
        batch_ids: The identifiers of the batches this run dispatches, which closure records its outcome onto.
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

    # Represents each type by the widest job carrying its name. Sizing is per-job rather than per-type, so one type
    # holds jobs of several widths, and the resolved figure serves as the cap those jobs are held to below and as the
    # divisor the reported concurrency follows from. Taking the widest is what leaves every library-chosen width
    # intact, since a narrower representative would cap a type's own wide jobs down to a sibling's figure, and it
    # settles on one representative whatever order the batch holds its jobs in. Admission reads neither figure and
    # weighs each job's own width against the budget, so this governs the cap and the report rather than what runs.
    type_cores: dict[str, int] = {}
    for job in dispatchable:
        type_cores[job.job_name] = max(type_cores.get(job.job_name, 0), job.core_weight)

    allocations = resolve_core_allocations(
        job_cores=type_cores,
        job_names=set(type_cores),
        core_budget=core_budget,
        job_limits=concurrency_limits,
        job_reservations=concurrency_reservations,
    )

    # Caps each job at what the host can supply for its type rather than replacing its width with that figure, which
    # preserves the width the owning library sized this particular job at. A job the library read a small archive for
    # and sized at one core stays at one core, and only a job wider than the host allows is brought down.
    for pending_job in dispatchable:
        pending_job.core_weight = max(1, min(pending_job.core_weight, allocations[pending_job.job_name].cores_per_job))

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
    thread = Thread(target=_run_and_close_local_batch, args=(state, host, batch_ids), daemon=True)
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


def _execute_remote_batch(
    pending: list[GenericPendingJob], batch_ids: list[str], walltime_minutes: int
) -> dict[str, Any]:
    """Reconciles a remote batch and submits it to the server's scheduler as a dependency graph.

    Notes:
        A submission spanning several batches writes the scripts and logs of them all into one directory, named after
        the first batch. Every batch it dispatched is recorded on the ledger entry, so closure snapshots an outcome for
        each rather than for the first alone.

    Args:
        pending: The batch's jobs.
        batch_ids: The prepared batches being dispatched. The first names the directory the scripts and logs are
            written into.
        walltime_minutes: The wall-time every allocation requests, or a non-positive value to take the shared default.

    Returns:
        The response dict the calling tool returns.
    """
    walltime = walltime_minutes if walltime_minutes > 0 else REMOTE_JOB_WALLTIME_MINUTES
    batch_id = batch_ids[0]
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
                covered_batch_ids=batch_ids,
                walltime_minutes=walltime,
            )
            batch_directory = str(remote_batch_directory(server=server, batch_id=batch_id))

            # Closes the earlier batches that finished while this one was prepared, so the ledger sheds them without
            # waiting for a status read that may never come. Observing a state does not retire it, so the closure
            # step has to follow the query.
            ledger = read_ledger()
            outstanding = [submission for batch in ledger.batches for submission in batch.submissions]
            if outstanding:
                statuses = query_submissions(server=server, submissions=outstanding)
                close_settled_batches(host=RemoteHost(server=server), batches=ledger.batches, statuses=statuses)
    except Exception as exception:
        return error_response(message=f"Unable to submit the remote batch. {exception}")

    return ok_response(
        started=True,
        host=REMOTE_HOST_LABEL,
        batch_id=batch_id,
        batch_ids=batch_ids,
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


def _unsupported_message(pipeline: str) -> str:
    """Builds the error message returned when a caller names a pipeline the batch tools do not support.

    Args:
        pipeline: The name the caller supplied.

    Returns:
        The error message.
    """
    available = ", ".join(sorted(member.value for member in BATCH_PIPELINES))
    return f"Unsupported batch pipeline '{pipeline}'. Available: {available}."


def _collect_status(state: JobExecutionState[GenericPendingJob]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Re-reads the trackers of an execution state's jobs and returns per-job status entries and aggregate counts.

    Notes:
        Every entry names the unit it belongs to, because a job identifier is derived from the job name and the
        specifier alone. A pipeline whose specifier does not vary by session therefore gives every session's copy of
        that stage one identifier, and the entries would be indistinguishable without the unit that separates them.

        Each tracker is summarized once, so the record of every job the batch holds comes from one read. The counts
        are then tallied from the batch's own jobs rather than taken from the trackers' aggregates, because a tracker
        holds the whole job universe of the unit's pipeline while the batch holds the subset it dispatched. Counting
        the aggregates would report a total the batch's job count does not match.

        A tracker's record carries the executor that ran the job, when it started, and when it finished, because a
        caller asking about one job wants all three. A job the tracker does not know yet reports empty timing rather
        than absent keys, so every entry carries the same fields. Only the error text is conditional, because a job
        that recorded none has nothing to report.

    Args:
        state: The batch execution state whose jobs to report.

    Returns:
        A tuple of the per-job status entries and a summary dict counting succeeded, failed, running, and scheduled
        jobs alongside the total.
    """
    per_job: list[dict[str, Any]] = []
    summary = dict.fromkeys(("total", *_STATUS_LABELS), 0)
    for tracker_path, jobs in group_jobs_by_tracker(state=state).items():
        payload = ProcessingTracker(file_path=tracker_path).summarize()
        records: dict[str, dict[str, Any]] = {record["job_id"]: record for record in payload["jobs"]}
        for job in jobs:
            record = records.get(job.job_id, {})
            status = record.get("status", ProcessingStatus.SCHEDULED.name).lower()
            summary["total"] += 1
            summary[status] += 1
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "pipeline": job.pipeline,
                "job_name": job.job_name,
                "specifier": job.specifier,
                "session_path": str(job.unit_path),
                "tracker_path": str(tracker_path),
                "status": status,
                "cores": job.core_weight,
                "memory_mb": job.memory_mb,
                "options": dict(job.options),
                "prerequisite_ids": list(job.prerequisite_ids),
                "executor_id": record.get("executor_id"),
                "started_at": record.get("started_at"),
                "completed_at": record.get("completed_at"),
                "elapsed_seconds": _elapsed_seconds(
                    started_at=record.get("started_at"), completed_at=record.get("completed_at")
                ),
            }
            if "error_message" in record:
                entry["error_message"] = record["error_message"]
            per_job.append(entry)
    return per_job, summary


def _elapsed_seconds(started_at: int | None, completed_at: int | None) -> float | None:
    """Resolves how long a job has run, measuring a finished job to its completion and a running one to now.

    Args:
        started_at: The microsecond-precision epoch the tracker recorded the job as starting at, or None when the job
            has not started.
        completed_at: The microsecond-precision epoch the tracker recorded the job as finishing at, or None when the
            job is still running.

    Returns:
        The elapsed seconds, or None when the job has not started.
    """
    if started_at is None:
        return None
    # A running job is measured to now on the same clock the tracker stamps its own timestamps with, so the two are
    # directly subtractable.
    end = completed_at if completed_at is not None else current_timestamp()
    seconds = convert_time(
        time=end - started_at, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True
    )
    return round(seconds, 3)
