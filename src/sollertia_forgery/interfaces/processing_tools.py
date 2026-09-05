"""Provides the generic Model Context Protocol (MCP) tools for preparing processing jobs, inspecting their cost,
running them as one batch on this machine or the compute server, checking, canceling, or resetting that batch, and
removing pipeline output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from threading import Thread
from collections import deque
from dataclasses import dataclass

from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    bounded_counts,
    error_response,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from .remote_tools import remote_batch_cancel, remote_batch_retire, remote_batch_status
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
    read_batch_outcome,
    resolve_batch_host,
    resolve_allocations,
    reconcile_local_jobs,
    close_settled_batches,
    group_jobs_by_tracker,
    job_execution_manager,
    read_prepared_batches,
    reconcile_remote_jobs,
    record_prepared_batch,
    read_scheduler_records,
    remote_batch_directory,
    resolve_host_memory_mb,
    resolve_tracker_claims,
    resolve_core_allocations,
    resolve_concurrency_limits,
    resolve_queried_allocations,
    resolve_concurrency_reservations,
)
from .host_resolution import (
    HOST_LABELS,
    resolve_execution_host,
    unsupported_host_message,
)

if TYPE_CHECKING:
    from ..server import Server
    from ..orchestration import ExecutionHost, GenericPendingJob

_CLOSED_BATCH_MESSAGE: str = (
    "No batch is running in this process. The reported batches have closed, so their outcomes are read from the "
    "snapshot closure recorded rather than from a live pool."
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
"""The floor below which an auto-resolved memory budget never falls, so a small host still admits one job at a time. A
budget the caller names explicitly is honored as given."""

_STATUS_AXES: tuple[str, ...] = ("pipeline", "job_name", "status", "session_path")
"""The job attributes by which a caller may filter a batch, and the axes a status breakdown counts."""

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
"""The job fields detail adds, which are the resources the job occupies, its timing and provenance, its runtime
parameters, and its prerequisite jobs."""

_RESOURCE_SEMI_FIELDS: tuple[str, ...] = (
    "job_id",
    "job_name",
    "specifier",
    "unit_path",
    "cores",
    "memory_mb",
    "resident_mb",
)
"""The job fields a semi-detail resource listing carries, which are the job's identity, the unit it reads, and its
planned figures. The unit path rides every row because a job identifier derives from the job name and the specifier
alone, so the units of one call share the identifier of the same stage and nothing else in the row separates them."""

_RESOURCE_DETAIL_FIELDS: tuple[str, ...] = ("prerequisite_ids", "options")
"""The job fields detail adds, naming the job's prerequisite jobs and the parameters it would use."""

_STATUS_LABELS: tuple[str, ...] = tuple(member.name.lower() for member in ProcessingStatus)
"""The status labels a tracked job reports, which are the tracker's own status names in lower case. These are the values
by which a caller filters the listing, and the keys under which a tracker summary counts."""

_UNREACHABLE_SERVER: str = (
    "Unable to reach the remote compute server, so nothing was submitted and every prepared batch stays as it was."
)
"""The cause reported when the connection itself cannot be opened. It is named apart from every step behind it because
it is repaired on the network rather than on either machine."""

_RECONCILIATION_FAILURE: str = (
    "Unable to resolve which of this batch's jobs an allocation already runs, so nothing was submitted. Submitting "
    "without that resolution could run a second allocation over a job another one is still carrying."
)
"""The cause reported when the reconciliation cannot read the ledger or the scheduler's accounting. The submission is
refused rather than attempted, since the resolution is what keeps one job from being run twice."""

_RESET_FAILURE: str = (
    "Unable to clear the recorded state of the jobs this batch dispatches on the remote compute server, so nothing "
    "was submitted."
)
"""The cause reported when the host refuses the reset that precedes a submission. A job dispatched over an uncleared
record would report the previous run's outcome for the window before it starts."""

_SUBMISSION_FAILURE: str = (
    "Unable to submit the remote batch to the compute server's scheduler. Every allocation the scheduler accepted "
    "before the rejection is recorded on the submission ledger, so re-running this batch submits what it left."
)
"""The cause reported when the scheduler rejects a submission. The allocations it already accepted stay queued and
recorded, which is why the message names the rerun rather than describing the batch as unsubmitted."""


@dataclass(frozen=True, slots=True)
class _LocalRun:
    """Holds this process's one local batch together with the prepared batches that batch covers.

    Notes:
        A reader takes the pair in one reference and a writer replaces it whole, so a state is never read beside the
        identifiers of another run.
    """

    state: JobExecutionState[GenericPendingJob] | None = None
    """The execution state the dispatch installed, or None once the run closed and recorded a durable outcome."""
    batch_ids: tuple[str, ...] = ()
    """The prepared batches this run covers. They outlive the state, so a bare status call reads the outcomes of the
    run that just finished."""


_LOCAL_RUN: _LocalRun = _LocalRun()
"""The single batch execution state and the batches it covers. One pool serves every pipeline, so a batch may hold any
mix of jobs and the engine packs them against one pair of budgets."""


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

    A job that the unit cannot run never reaches its processing tracker, so its absence from the project's state
    artifact is what rules it out. A job whose upstream stage this run can neither dispatch nor find already succeeded
    is reported under ``blocked_jobs`` rather than dispatched.

    Pass the returned ``batch_id`` to ``execute_jobs_tool``. A batch runs where it was prepared, so execution reads the
    host from the batch itself. Identifiers are recorded on disk and outlive the server that issued them.

    Args:
        pipeline: The batch pipeline to prepare, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories whose jobs to prepare, which are session roots for every session
            pipeline and dataset roots for ``forging``. For ``remote`` these are paths ON THE SERVER. Every unit must
            belong to one project, since the artifacts from which a batch is resolved are written per project.
        options: The pipeline-specific parameters for the prepared jobs, carried on every descriptor this call
            registers. The ``checksum`` pipeline reads ``regenerate_checksum``, a boolean selecting re-baselining of the
            stored value over verification against it, which defaults to verification. Every other pipeline, including
            ``forging``, takes no parameters. Build or extend a dataset hierarchy with ``define_forging_dataset_tool``,
            which takes ``session_names``, ``force_recreate``, and ``recreate_animals`` directly.
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
        job awaits. Carries a ``jobs`` list when the caller requests the descriptors.
    """
    response = _prepare_batch_response(
        pipeline=pipeline, session_paths=session_paths, options=options, host=host, replan=replan, record=True
    )
    if response["success"] and not include_job_descriptors:
        response.pop("jobs")
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

    A bare call reports the figures against which a batch is planned, alongside a ``breakdown`` naming every job type
    and how many of each the named sessions still have to run. A job that the units already recorded as succeeded, and a
    job that this run could not unblock, are both absent, so this reports what a batch would dispatch rather than the
    whole universe. Naming a filter adds a page of jobs carrying their figures, and opting into detail adds the unit
    each job reads, its prerequisite jobs, and the parameters it would use.

    Estimates each job's memory from the data it will process, so a long recording is not charged the same as a short
    one. The figures already carry the shared tolerance, so a caller plans a local batch against them or requests them
    from a remote scheduler. Discovery runs as it does for a batch, so each session's tracker is created and aligned to
    its job universe.

    Args:
        pipeline: The pipeline to inspect, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories to inspect, which are session roots for every session
            pipeline and dataset roots for ``forging``. For ``remote`` these are paths ON THE SERVER.
        options: The pipeline-specific parameters for the inspected jobs, forwarded to preparation. See
            ``prepare_batch_tool`` for the keys each pipeline reads.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        job_names: Restricts the listing to these job type names.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs report the unit they read, their prerequisite jobs, and the
            parameters they would use.

    Returns:
        A response dict with the ``pipeline`` inspected, the ``host`` that holds the data, ``total_units``, and a
        ``totals`` summary giving ``jobs``, ``widest_job_cores``, ``largest_job_memory_mb``, ``summed_memory_mb``,
        ``largest_job_resident_mb``, and ``summed_resident_mb``. A caller sizing this machine's pool budgets against
        the anonymous totals, and a caller sizing a scheduler submission budgets against the resident ones.
        Carries a ``breakdown`` per job type and a ``units`` list naming each session and how many jobs it resolved.
        Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. For ``local`` it also carries this machine's ``total_memory_mb``
        and the batch-available ``total_cores`` left after the reserved system cores. Both are absent for ``remote``,
        where the scheduler holds the budgets and the caller names what a job requests.
    """
    prepared = _prepare_batch_response(
        pipeline=pipeline, session_paths=session_paths, options=options, host=host, replan=False, record=False
    )
    if not prepared["success"]:
        return prepared

    jobs = prepared["jobs"]
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
            "largest_job_resident_mb": max((int(job["resident_mb"] or job["memory_mb"]) for job in jobs), default=0),
            "summed_resident_mb": sum(int(job["resident_mb"] or job["memory_mb"]) for job in jobs),
        },
        breakdown={"job_name": count_values(values=[job["job_name"] for job in jobs])},
    )

    # Both figures read the machine on which this process runs, so they describe the host only when the data sits here.
    # A remote batch is submitted with the budgets its caller names and the scheduler enforces its own limits, so
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

    Before anything is dispatched, every job that the trackers already record as running is reconciled. Locally that
    record describes a pool that died, so the job is rerun. Remotely each job is resolved through the same three
    records against which ``get_processing_status_tool`` resolves it, and the ``verdict`` decides what happens to it. A
    job that resolves as running is left alone, adopted onto the allocation already running it, with its dependents
    wired to wait on that allocation. Every other verdict releases the job to this run. A job that resolves as running
    while naming no allocation, which is one whose tracker claims an executor outside the scheduler, is withheld along
    with its dependents rather than run a second time, and reported under ``withheld_jobs``. Every job that is
    dispatched has its recorded state cleared first, so a status read stays honest across the window before it starts.

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
        A response dict with ``started``, the ``host`` that received the dispatch, ``total_jobs``, the ``pipelines`` the
        batch holds, and any ``adopted_jobs`` it left to an allocation already running them. A local dispatch adds the
        resolved ``core_budget``, ``memory_budget_mb``, ``pool_size``, and a ``job_allocations`` entry per job type. A
        remote dispatch adds the ``batch_id`` under which its scripts and logs are filed, the ``batch_ids`` the
        submission covered, ``walltime_minutes``, and the ``batch_directory`` on the server. It also adds a
        ``withheld_jobs`` list naming each job it neither submitted nor adopted alongside the executor its tracker
        claims, and a ``submissions`` list pairing each job with the allocation that runs it. Either host adds an
        ``invalid_jobs`` list when a recorded descriptor could not be built into a job. Returns an error when the
        prepared-batch registry cannot be read, when an identifier resolves to no prepared batch, or when no batch is
        named. Returns an error as well when the named batches mix hosts, when every prepared job is blocked or already
        succeeded, and when no recorded descriptor builds into a job. A local dispatch also returns an error when a
        batch is already running in this process, since one pool holds one batch, and when the recorded state of the
        dispatched jobs cannot be cleared. A remote dispatch reports each of its
        own steps as itself. It returns an error when the server cannot be reached, when the reconciliation cannot
        resolve what already runs, when the host refuses to clear the dispatched jobs' records, and when the scheduler
        rejects the submission.
    """
    try:
        documents, missing = read_prepared_batches(batch_ids=batch_ids)
    except Exception as exception:
        return error_response(message=f"Unable to read the prepared batches {batch_ids}. {exception}")
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
    ``breakdown`` naming every pipeline, job type, status, and session in the batch. An axis holding more distinct
    values than the shared cap reports how many it holds in place of its counts, which is what tracks a run at a size a
    response can always carry, however many jobs it holds. The counts are where a failure first shows. They cover the
    batch's own jobs alone, so their total is the number of jobs the batch dispatched and a job of the same tracker
    that this batch did not dispatch is left out of them. Those same counts resolve to the ``status`` label the
    response carries, so the label and the counts describe one set.

    Naming a filter adds a page of jobs carrying identity and status. Filtering to ``failed`` is how a caller reads
    which jobs failed, and opting into detail adds each one's error text, timing, and the resources it occupies.

    A ``remote`` call carries an ``outcomes`` entry for any batch that settled and closed on it. It resolves every
    allocation of every batch that remains outstanding against three records, which are the scheduler's accounting, the
    scheduler's queue, and the processing tracker of the job the allocation carries. Each allocation reports the
    ``scheduler_state`` those first two records place it in, the ``tracker_status`` the third holds, the ``verdict``
    the three of them carry, and the ``remediation`` that verdict prescribes. A verdict of ``running`` means the
    scheduler still holds the recorded allocation, holds the one this job's own tracker claims, or that tracker claims
    to be running under an executor for which neither scheduler record answers. Nothing is remediated for such a
    verdict, which is how work another machine submitted is left alone by a ledger that never recorded it.
    ``finished`` and ``failed`` mean the job recorded an outcome its tracker keeps, ``abandoned`` means nothing claims
    the job, and ``stranded`` means its tracker still claims to be running while no allocation is. Each batch also
    carries a ``progress`` verdict of ``progressing``, ``stalled``, or ``awaiting_closure`` alongside the remedy for
    it. A batch is stalled when none of its allocations resolves as running and at least one is gone from both
    scheduler records, which no later query changes. ``retire_remote_batches_tool`` is what remediates such a batch. A
    ``local`` call carries an ``outcomes`` entry once the run this process dispatched has closed and recorded an
    outcome for at least one of its batches. The report covers the batches named in ``batch_ids``, or the batches the
    last run covered when the argument names none. Each entry is the durable snapshot that closure took of what a
    batch's jobs recorded. Read ``complete``, ``succeeded``, ``failed``, ``blocked``, and ``outstanding`` from it to
    decide whether the run needs anything further, and ``failed_jobs`` for the error text each failure recorded.

    Args:
        host: Which batch to report, either ``local`` for this machine's pool or ``remote`` for the outstanding
            allocations on the server's scheduler.
        batch_ids: Restricts a ``remote`` report to these outstanding batches. Omit to cover all of them. Naming any
            batch also counts as a filter, so the response carries a page of jobs. For ``local`` it names the closed
            batches whose recorded outcomes to report, and omitting it reports the batches the last run covered. A
            ``local`` call made while a batch runs returns one error naming every identifier the running batch does
            not cover, since one pool holds one batch.
        status_filter: Restricts the listing to one status. Locally one of ``succeeded``, ``failed``, ``running``, or
            ``scheduled``, and remotely an accounting state such as ``FAILED``, ``RUNNING``, or ``BLOCKED``.
        session_paths: Restricts the listing to these session root directories.
        job_ids: Restricts the listing to these tracker job identifiers.
        job_names: Restricts the listing to these job type names, such as ``motion_energy``.
        pipelines: Restricts the listing to these pipelines.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry their resources, timing, provenance, and error text.

    Returns:
        For ``remote``, a response dict with ``active`` and the ``batches`` covered. Each batch entry carries its
        ``outstanding_seconds``, its ``progress`` verdict, a ``verdicts`` count per allocation verdict, its
        ``running_allocations``, ``stranded_allocations``, and ``unresolvable_allocations``, and its ``remedy``. The
        response also carries ``stalled_batch_ids``, naming the batches that can no longer settle. It adds
        ``uncovered_batch_ids``, naming any batch another process recorded while this call ran, about which this call's
        records say nothing and which the next read covers. It closes with a ``summary`` counting the allocations by
        accounting state and a ``breakdown`` per axis. The final keys are a ``scheduler_read_error`` that is empty
        unless one of the scheduler's records could not be read, and the ``outcomes`` of any batch that closed on this
        call. For ``local``, a response dict with ``active``, which reports whether the manager thread is still
        running, and ``canceled``. It also carries a ``summary`` counting the batch's succeeded, failed, running, and
        scheduled jobs alongside their total, the ``status`` label resolved from those counts, and a ``breakdown`` per
        axis. Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. A batch that could not dispatch some jobs also reports
        ``blocked_jobs`` as a count with a ``blocked_reason``, and those jobs are listed by filtering to ``scheduled``.
        A ``local`` call made once the run closed and recorded an outcome reports ``active`` as False alongside the
        ``batch_ids`` it covered and their ``outcomes``. If no batch has run, ``active`` is False with an explanatory
        ``message``.
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

    run = _LOCAL_RUN
    state = run.state
    if state is None:
        # A bare call falls back to the batches the last run covered, so the run that just finished still answers.
        named = list(batch_ids) if batch_ids else list(run.batch_ids)
        try:
            recorded = [outcome for batch in named if (outcome := read_batch_outcome(batch_id=batch)) is not None]
        except Exception as exception:
            return error_response(message=f"Unable to read the recorded outcomes of {named}. {exception}")
        if recorded:
            return ok_response(active=False, batch_ids=named, outcomes=recorded, message=_CLOSED_BATCH_MESSAGE)
        return ok_response(active=False, message="No batch is running in this process.")

    uncovered = sorted(set(batch_ids or ()) - set(run.batch_ids))
    if uncovered:
        return error_response(
            message=(
                f"Unable to report the named batch(es) {uncovered}. A local report must name the batches the running "
                f"batch covers, which are {sorted(run.batch_ids)}, since one pool holds one batch. Read the others "
                f"once this one finishes."
            )
        )

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
        breakdown={axis: bounded_counts(values=[entry[axis] for entry in per_job]) for axis in _STATUS_AXES},
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
    dependency can no longer complete successfully. A remote cancellation covers every allocation the ledger recorded
    for the named batches. Resolving those allocations the way ``get_processing_status_tool`` resolves them, it also
    covers every further allocation a job's own tracker claims and the scheduler still holds, which is how it reaches a
    run another machine submitted.

    Args:
        host: Which batch to cancel, either ``local`` for this machine's pool or ``remote`` for the server's scheduler.
        batch_ids: The outstanding remote batches to cancel. Omit to cancel all of them. Ignored for ``local``, where
            one pool holds one batch.

    Returns:
        A response dict with ``canceled``, a ``message`` stating what the cancellation did, and, for ``local``, the
        number of queued jobs in ``dropped_jobs``. For ``remote`` it carries the ``canceled_jobs`` count, the
        ``canceled_allocations`` it named, and the ``batch_ids`` the cancellation covered. Returns an error when
        nothing is running or outstanding.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
    if host == REMOTE_HOST_LABEL:
        return remote_batch_cancel(batch_ids=batch_ids)

    state = _LOCAL_RUN.state
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
def retire_remote_batches_tool(
    batch_ids: list[str], *, force: bool = False, drop_without_outcome: bool = False
) -> dict[str, Any]:
    """Applies the resolved remediation to every allocation of the named batches, then drops their ledger entries.

    This is the one tool that acts on what ``get_processing_status_tool`` resolves with ``host='remote'``. Every
    allocation is resolved there and here by the same three records, and the ``verdict`` that resolution carries
    decides what happens to it. A ``stranded`` allocation, whose job's tracker still claims to be running while no
    allocation is, has that job returned to the scheduled state, which is what releases a job no rerun could otherwise
    reach. A ``finished``, ``failed``, or ``abandoned`` allocation has its tracker left exactly as it stands, so no
    result is discarded and no failure is silently cleared. Every named batch is then snapshotted through the same
    closure applied to a settled batch, and its ledger entry is dropped, which is what stops the batch being
    outstanding and stops its jobs being claimed during reconciliation.

    Two guarantees stand in front of that, and each is waived by its own flag and by nothing else. A batch holding an
    allocation that resolves as ``running`` is refused, because remediating it would disturb work the scheduler is
    still carrying. ``force`` waives that one, and then every allocation those entries leave held is canceled before
    any tracker is written, the allocation a job's own tracker claims included. That claim may name another machine's
    allocation, and it is the one actually carrying the job. A batch whose outcome cannot be snapshotted is refused,
    because the ledger entry is the last record naming the run, and ``drop_without_outcome`` waives that one.
    An unreachable server hides every record, so every allocation resolves as ``running``, both waivers are needed to
    remediate through it, and remediating then cancels nothing and writes no tracker, because reaching either needs
    the connection that failed.

    This drops the ledger entries of the named batches, and the snapshot it takes first replaces each covered batch's
    prepared document with the outcome recorded for it. It removes neither those outcomes nor the state snapshots
    beside them: ``forget_prepared_batches_tool`` drops the prepared document and the recorded outcome, while the
    delivered state snapshots under the batch directory are left on disk for the caller to remove.

    Args:
        batch_ids: The outstanding batches to remediate, as ``get_processing_status_tool`` reports them with
            ``host='remote'``. Naming none is an error, since remediation never defaults to the whole ledger.
        force: Determines whether to remediate batches holding an allocation that resolves as running. Every
            allocation those entries leave held is canceled before any tracker is written, the one a job's tracker
            claims included. An allocation whose state could not be read at all still resolves as running and is
            canceled by nothing, so waiving the refusal for it drops its entry while leaving its tracker untouched.
        drop_without_outcome: Determines whether to drop the ledger entries when their outcome cannot be snapshotted.

    Returns:
        A response dict with ``retired``, the ``batch_ids`` the ledger held and dropped, and ``total_allocations``
        counting the allocations they held. It carries a ``batches`` list, whose entries hold each batch's
        ``batch_id``, its ``covered_batch_ids``, its ``allocations``, and its ``outstanding_seconds``. It carries an
        ``allocations`` list as well, whose entries hold each allocation's identity, its ``scheduler_state``, its
        ``tracker_status``, its ``verdict``, and the ``remediation`` applied. Each allocation entry also reports
        whether it was ``cancelled``, whether its ``tracker_reset`` ran, whether its ``snapshot_recorded``, and whether
        its ``entry_dropped``. The response also carries ``cancelled_allocations``, a ``reset_jobs`` count, the
        ``outcomes`` closure recorded, the ``outcome_directory`` on this machine holding those outcome files and the
        state snapshots they cite, a ``snapshot_error`` that is empty when the snapshot succeeded, and a ``message``.
        Returns an error when the ledger cannot be read or written, when no batch is outstanding, when no identifier is
        named, and when a named identifier is not outstanding. It returns an error as well when the jobs' own
        processing trackers cannot be read, when the scheduler's accounting cannot be read, and when an allocation
        resolves as running while ``force`` is not set. A cancellation or a tracker reset that fails is an error too,
        and so is a snapshot that fails while ``drop_without_outcome`` is not set.
    """
    return remote_batch_retire(batch_ids=batch_ids, force=force, drop_without_outcome=drop_without_outcome)


@mcp.tool()
def reset_processing_jobs_tool(
    pipeline: str, unit_paths: list[str], job_ids: list[str] | None = None, host: str = "local"
) -> dict[str, Any]:
    """Resets tracked jobs to SCHEDULED across one or more units, so a later execute reruns only them.

    Resolves each pipeline tracker from its unit, so a caller names what it wants reset rather than where the tracker
    sits. That makes a mismatched pipeline and path impossible to express, and it is why no read tool has to carry
    tracker locations.

    One call covers every named unit, because each unit resets only the identifiers it actually tracks. Passing a whole
    batch's identifiers alongside all of its units therefore costs a single operation, and remotely the per-unit
    invocations are chained into one round trip, so a batch spanning many units still costs a single connection.
    Omitting the identifiers resets every job each unit tracks.

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
        Returns an error when the pipeline or the host is not supported, and when the host cannot carry out the reset.
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
    same removal on whichever side of the connection it sits.

    The ``checksum`` pipeline owns no directory, because it writes its stored value into the acquired data itself.
    Cleaning it removes its tracker and leaves that stored value in place, so the unit keeps the baseline against which
    a later verification compares. The ``forging`` pipeline owns its whole dataset hierarchy, so cleaning it removes
    every assembled feather in that dataset alongside the tracker. It also owns one cross-recording directory inside
    every source session the dataset names, holding what its cross-recording stages wrote into that session's own
    imaging output. Cleaning the pipeline removes each of those too, leaving the session's single-recording output and
    the directories any other dataset owns beside it in place. Those directories are resolved before anything is
    removed, so a resolver that fails leaves the unit untouched, and a source session that no longer loads is reported
    and passed over.

    Args:
        pipeline: The batch pipeline to clean, one of ``checksum``, ``runtime``, ``microcontroller``, ``video``,
            ``two_photon``, ``forging``.
        session_paths: The processing unit directories to clean, which are paths ON THE SERVER for ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        A response dict with ``pipeline``, ``host``, ``total_paths`` removed, ``removed_bytes`` freed across every
        unit, and a ``removed`` list carrying each path and the bytes it held. Returns an error when a batch is running
        in this process, since removing the output of a job in flight would fail that job. It also returns an error
        when the pipeline or the host is not supported, and when the host cannot carry out the removal.
    """
    if pipeline not in {member.value for member in BATCH_PIPELINES}:
        return error_response(message=_unsupported_message(pipeline=pipeline))
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    # A running batch holds open the very files this removes, so cleaning waits for the pool to drain.
    state = _LOCAL_RUN.state
    running = state is not None and state.manager_thread is not None and state.manager_thread.is_alive()
    if host == LOCAL_HOST_LABEL and running:
        return error_response(
            message=(
                "A batch is currently running in this process. Wait for it to finish or cancel it before cleaning "
                "output."
            )
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


def _prepare_batch_response(
    pipeline: str,
    session_paths: list[str],
    options: dict[str, Any] | None,
    host: str,
    *,
    replan: bool,
    record: bool,
) -> dict[str, Any]:
    """Resolves a pipeline's dispatchable jobs on the host that holds the data and renders them as a response.

    Notes:
        Recording sits inside the guarded region alongside the resolution it completes, because it writes under the
        platform working directory and a caller is owed an error naming whichever step failed.

        A caller that only reads the resolved jobs records nothing, because dispatch resolves a batch by its
        recorded identifier, and a record that no caller can name is never retired.

    Args:
        pipeline: The batch pipeline to resolve.
        session_paths: The processing unit directories whose jobs to resolve.
        options: The pipeline-specific parameters carried on every resolved descriptor.
        host: Where the data sits, either ``local`` or ``remote``.
        replan: Determines whether to re-estimate the cores and memory the units' plan caches already hold.
        record: Determines whether the resolved batch is recorded under an identifier the response carries.

    Returns:
        The response dict carrying the resolved jobs, or the error response naming what failed.
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
        batch_id = record_prepared_batch(document=document) if record else None
    except Exception as exception:
        return error_response(message=f"Unable to prepare the {host} '{pipeline}' batch. {exception}")

    response = ok_response(
        pipeline=document.pipeline,
        host=document.host,
        units=document.units,
        total_units=len(document.units),
        total_jobs=len(document.jobs),
        total_blocked_jobs=len(document.blocked_jobs),
        blocked_jobs=[project_item(item=entry, fields=_BLOCKED_SEMI_FIELDS) for entry in document.blocked_jobs],
        jobs=document.jobs,
    )
    if batch_id is not None:
        response["batch_id"] = batch_id
    return response


def _reset_batch_jobs(host: ExecutionHost, jobs: list[GenericPendingJob]) -> None:
    """Clears the recorded state of every job about to be dispatched, on the host that records it.

    Notes:
        Each unit carries the identifiers of its own dispatched jobs alone. A job identifier is derived from the job
        name and the specifier alone, so two units of one project share the identifier of the same stage. A flat set
        applied to both would clear a succeeded record this batch never dispatched.

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
    """Runs a local batch to completion and closes every batch it held, releasing the execution state behind a
    recorded outcome.

    Notes:
        Closure runs in the same thread the manager did, so it happens the moment the queue drains rather than waiting
        for a caller to ask. A failure to close is reported and swallowed, because the jobs themselves have already run
        and recorded their outcomes.

        The state is released only when closure recorded at least one durable outcome, and only while it is still the
        state this run installed. A run whose closure recorded nothing therefore keeps its state, which leaves its
        counts readable from the trackers. The identifiers outlive the state either way, so a bare status read reaches
        the outcomes of the run that just finished.

    Args:
        state: The batch execution state from which the manager dispatches.
        host: The host holding the data the batch's jobs read.
        batch_ids: The identifiers of the batches this run dispatched.
    """
    global _LOCAL_RUN

    job_execution_manager(state=state)
    settled: list[str] = []
    for batch_id in batch_ids:
        try:
            if close_batch(host=host, batch_id=batch_id) is not None:
                settled.append(batch_id)
        except Exception as exception:
            console.echo(
                message=f"Unable to close the finished batch '{batch_id}'. {exception}", level=LogLevel.WARNING
            )

    # Every dispatched identifier survives the release, including one whose closure failed, so a later status read
    # still names the batch and reports that it recorded no outcome rather than losing it from the run entirely.
    if settled and _LOCAL_RUN.state is state:
        _LOCAL_RUN = _LocalRun(batch_ids=tuple(batch_ids))


def _execute_local_batch(
    host: ExecutionHost,
    pending: list[GenericPendingJob],
    batch_ids: list[str],
    core_budget_override: int,
    memory_budget_mb: int,
) -> dict[str, Any]:
    """Reconciles a local batch and dispatches it onto the shared process pool.

    Args:
        host: The host holding the trackers, through which the reset is applied.
        pending: The batch's jobs.
        batch_ids: The identifiers of the batches this run dispatches, onto which closure records its outcome.
        core_budget_override: The cores the batch may use in total, or a non-positive value to auto-resolve.
        memory_budget_mb: The memory the batch may use in total, or a non-positive value to auto-resolve.

    Returns:
        The response dict the calling tool returns.
    """
    global _LOCAL_RUN

    running = _LOCAL_RUN.state
    if running is not None and running.manager_thread is not None and running.manager_thread.is_alive():
        return error_response(
            message=(
                "A batch is already running in this process. Wait for it to finish or cancel it before starting "
                "another."
            )
        )

    reconciliation = reconcile_local_jobs(jobs=pending)

    # The reset takes each unit's tracker lock, which a concurrent writer can hold past the acquisition timeout, so the
    # dispatch reports that contention rather than raising out of the tool.
    try:
        _reset_batch_jobs(host=host, jobs=reconciliation.resettable)
    except Exception as exception:
        return error_response(message=f"Unable to clear the recorded state of the batch's jobs. {exception}")

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
    # holds jobs of several widths, and the resolved figure serves as the cap to which those jobs are held below and as
    # the divisor from which the reported concurrency follows. Taking the widest is what leaves every library-chosen
    # width intact, since a narrower representative would cap a type's own wide jobs down to a sibling's figure, and it
    # settles on one representative however the batch orders its jobs. Admission reads neither figure and weighs each
    # job's own width against the budget, so this governs the cap and the report rather than what runs.
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
    # preserves the width at which the owning library sized this particular job. A job the library sized at one core
    # from a small archive stays at one core, and only a job wider than the host allows is brought down.
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
    thread = Thread(target=_run_and_close_local_batch, args=(state, host, batch_ids), daemon=True)
    state.manager_thread = thread
    _LOCAL_RUN = _LocalRun(state=state, batch_ids=tuple(batch_ids))
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

        Each step names its own cause, because what a caller does next differs by cause. A connection is restored, a
        scheduler outage is waited out, a host that refuses a reset is repaired, and a rejected submission is read off
        the scheduler's own answer. Every step ahead of the submission leaves the batch exactly as it was, so retrying
        it costs nothing.

    Args:
        pending: The batch's jobs.
        batch_ids: The prepared batches being dispatched. The first names the directory into which the scripts and logs
            are written.
        walltime_minutes: The wall-time every allocation requests, or a non-positive value to take the shared default.

    Returns:
        The response dict the calling tool returns.
    """
    walltime = walltime_minutes if walltime_minutes > 0 else REMOTE_JOB_WALLTIME_MINUTES
    batch_id = batch_ids[0]
    try:
        connection = connect_to_server()
    except Exception as exception:
        return error_response(message=f"{_UNREACHABLE_SERVER} {exception}")

    with connection as server:
        host = RemoteHost(server=server)
        try:
            reconciliation = reconcile_remote_jobs(server=server, jobs=pending)
        except Exception as exception:
            return error_response(message=f"{_RECONCILIATION_FAILURE} {exception}")

        try:
            _reset_batch_jobs(host=host, jobs=reconciliation.resettable)
        except Exception as exception:
            return error_response(message=f"{_RESET_FAILURE} {exception}")

        try:
            submissions = submit_batch(
                server=server,
                jobs=[_render_descriptor(job=job) for job in reconciliation.dispatchable],
                batch_id=batch_id,
                adopted=reconciliation.adopted,
                covered_batch_ids=batch_ids,
                walltime_minutes=walltime,
            )
        except Exception as exception:
            return error_response(message=f"{_SUBMISSION_FAILURE} {exception}")

        batch_directory = str(remote_batch_directory(server=server, batch_id=batch_id))
        _close_finished_batches(server=server, host=host, batch_id=batch_id)

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
        withheld_jobs=[
            {"unit_path": str(job.unit_path), "job_id": job.job_id, "executor_id": job.executor_id}
            for job in reconciliation.withheld
        ],
        submissions=[
            {
                "job_id": submission.job_id,
                "slurm_job_id": submission.slurm_job_id,
                "job_name": submission.job_name,
                "specifier": submission.specifier,
                "unit_path": submission.unit_path,
                "unit_name": submission.unit_name,
            }
            for submission in submissions
        ],
    )


def _close_finished_batches(server: Server, host: ExecutionHost, batch_id: str) -> None:
    """Closes the outstanding batches that finished before this submission was accepted.

    Notes:
        The ledger sheds a finished batch here rather than waiting for a status read that may never come. The batches
        are resolved exactly as a status read resolves them and closed from those resolutions, so this housekeeping
        drops nothing a status read would report as still held.

        A failure is reported as a warning rather than raised, because it runs behind an accepted submission and
        raising would answer a submitted batch as a failed one. The batches it could not close stay outstanding, so
        the next query tries them again.

    Args:
        server: The connected server that runs the allocations.
        host: The host holding the data the batches' jobs read.
        batch_id: The identifier of the batch this submission dispatched, which the warning names.
    """
    try:
        ledger = read_ledger()
        if not ledger.batches:
            return
        outstanding = [submission for batch in ledger.batches for submission in batch.submissions]
        claims = resolve_tracker_claims(host=host, submissions=outstanding)
        reading = read_scheduler_records(
            server=server, allocations=resolve_queried_allocations(submissions=outstanding, claims=claims)
        )
        close_settled_batches(
            host=host,
            batches=ledger.batches,
            resolutions=resolve_allocations(batches=ledger.batches, reading=reading, claims=claims),
        )
    except Exception as exception:
        console.echo(
            message=(
                f"Unable to close the batches that finished before '{batch_id}' was submitted, which stay "
                f"outstanding so the next query can try again. {exception}"
            ),
            level=LogLevel.WARNING,
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
        Every entry names the unit to which it belongs, because a job identifier is derived from the job name and the
        specifier alone. A pipeline whose specifier does not vary by session therefore gives every session's copy of
        that stage one identifier, and the entries would be indistinguishable without the unit that separates them.

        Each tracker is summarized once, so the record of every job the batch holds comes from one read. The counts
        are then tallied from the batch's own jobs rather than taken from the trackers' aggregates, because a tracker
        holds the whole job universe of the unit's pipeline while the batch holds the subset it dispatched. Counting
        the aggregates would report a total the batch's job count does not match.

        A tracker's record carries the executor that ran the job, when it started, and when it finished, because a
        caller asking about one job wants all three. A job that the tracker does not know yet reports empty timing
        rather than absent keys, so every entry carries the same fields. Only the error text is conditional, because a
        job that recorded none has nothing to report.

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
        started_at: The microsecond-precision epoch the tracker recorded as the job's start, or None when the job has
            not started.
        completed_at: The microsecond-precision epoch the tracker recorded as the job's completion, or None when the job
            is still running.

    Returns:
        The elapsed seconds, or None when the job has not started.
    """
    if started_at is None:
        return None
    # A running job is measured to now on the same clock that stamps the tracker's own timestamps, so the two are
    # directly subtractable.
    end = completed_at if completed_at is not None else current_timestamp()
    seconds = convert_time(
        time=end - started_at, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True
    )
    return round(seconds, 3)
