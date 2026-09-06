"""Provides the remote halves of the processing tools, which resolve, cancel, and remediate what the scheduler ran."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import asdict

from natsort import natsorted
from ataraxis_time import TimeUnits, convert_time

from ..server import JobStatus
from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    error_response,
    resolve_detail_limit,
)
from ..orchestration import (
    STALLED_BATCH,
    NO_REMEDIATION,
    GONE_ALLOCATION,
    DROP_REMEDIATION,
    RESET_REMEDIATION,
    CANCEL_REMEDIATION,
    RUNNING_ALLOCATION,
    STRANDED_ALLOCATION,
    AWAITING_CLOSURE_BATCH,
    RemoteHost,
    SchedulerReading,
    SubmissionLedger,
    AllocationResolution,
    read_ledger,
    classify_batch,
    forget_batches,
    batch_directory,
    resolve_batches,
    connect_to_server,
    current_timestamp,
    render_allocation,
    cancel_allocations,
    cancel_submissions,
    reset_stranded_jobs,
    resolve_allocations,
    close_covered_batches,
    close_settled_batches,
    read_scheduler_records,
    resolve_tracker_claims,
    resolve_live_allocations,
    resolve_queried_allocations,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..server import Server
    from ..orchestration import SubmissionBatch

_REMOTE_STATUS_AXES: tuple[str, ...] = (
    "batch_id",
    "pipeline",
    "job_name",
    "status",
    "scheduler_state",
    "tracker_status",
    "verdict",
    "unit_path",
)
"""The axes a remote status breakdown counts. The three resolved axes are counted alongside the raw accounting state,
so a caller reads how many allocations carry each verdict without listing them. A caller filters on the ``batch_id``,
``pipeline``, ``job_name``, ``status`` and ``unit_path`` axes and on the job identifier, while the resolved axes are
counted alone."""

_REMOTE_STATUS_SEMI_FIELDS: tuple[str, ...] = (
    "batch_id",
    "job_id",
    "slurm_job_id",
    "pipeline",
    "job_name",
    "specifier",
    "status",
    "scheduler_state",
    "tracker_status",
    "verdict",
    "remediation",
    "unit_name",
)
"""The job fields a semi-detail listing carries, which are the job's identity, the allocation that runs it, and the
resolved state of both records that decide what may be done about it."""

_REMOTE_STATUS_DETAIL_FIELDS: tuple[str, ...] = (
    "queued",
    "tracker_executor_id",
    "claimed_allocation",
    "claim_state",
    "cores",
    "resident_mb",
    "slurm_job_name",
    "unit_path",
    "output_log",
    "error_log",
)
"""The job fields detail adds, which are the evidence behind the resolved state, the resources the allocation
requested, and where its output landed. A caller reads the log paths to see a failed allocation's own diagnostics."""

_NAMED_ALLOCATION_LIMIT: int = 50
"""The allocations one batch entry of a status read names under each verdict it reports. The count beside each list
always covers the whole batch, so a batch of any size reports its totals while naming this many examples. The
remediation report is uncapped instead, because it is the last moment at which those identifiers can be read at all."""

_FINISHED_BATCH_GUIDANCE: str = (
    "Read what a finished run produced from the outcome closure recorded on the batch, which "
    "get_processing_status_tool reports and 'slf server batches' prints, or from the project's own job artifact, "
    "which the MCP tool read_project_jobs_tool reads with host='remote'."
)
"""The guidance appended wherever a caller reaches for a batch the ledger no longer holds. The ledger names outstanding
allocations alone, so the answer for a finished batch is its recorded outcome or the project's own job artifact.

Notes:
    The outcome reader is named as both a tool and a command, because one function answers each of them. The job
    artifact reader is marked as an MCP tool alone, because it carries no command, and a caller reading this in a
    terminal would otherwise search for one that does not exist.
"""

_NOTHING_OUTSTANDING: str = f"No remote batch is outstanding. {_FINISHED_BATCH_GUIDANCE}"
"""The message reported when the ledger holds no batch, which means every submitted batch has finished, was retired, or
was never submitted."""

_ALL_BATCHES_CLOSED: str = (
    f"Every batch this read covered resolved entirely to a plain drop and closed on it, so none of them is "
    f"outstanding any longer. {_FINISHED_BATCH_GUIDANCE}"
)
"""The message reported when the read's own closure retired every batch it covered. The outstanding batches are
resolved again after that closure, so a batch that just closed is reported as closed rather than as outstanding."""

_NO_BATCH_NAMED: str = (
    "Unable to remediate a batch without an identifier. Name the batches to remediate, since remediating drops the "
    "ledger's record of the named runs rather than everything it holds. Read the outstanding identifiers, and the "
    "verdict resolved for each of their allocations, from get_processing_status_tool with host='remote'."
)
"""The message reported when a remediation names no batch. Remediation is destructive and irreversible, so it never
takes the whole ledger as its default target the way a read does."""

_LEDGER_READ_FAILURE: str = (
    "Unable to read this machine's submission ledger, which is the record of the allocations it has outstanding on "
    "the compute server's scheduler."
)
"""The cause reported when the ledger itself cannot be read. It is named separately from every server-side failure
because it is repaired on this machine rather than on the server."""

_LEDGER_WRITE_FAILURE: str = (
    "Unable to drop the named batches from this machine's submission ledger. Their allocations were remediated, so "
    "remediating them again is safe once the ledger's lock is free."
)
"""The cause reported when the ledger's lock cannot be taken for the drop. The remediation ahead of the drop has
already run at that point, so the message says the retry is safe rather than leaving the caller to guess."""

_UNREACHABLE_SERVER: str = (
    "Unable to reach the remote compute server, so neither the state of the named batches' allocations nor what their "
    "jobs recorded could be read."
)
"""The cause reported when the connection itself cannot be opened. It hides both records at once, which is why it is
the one failure that leaves every allocation resolving as running."""

_ACCOUNTING_READ_FAILURE: str = (
    "Unable to read the state of the named batches' allocations from the remote compute server's scheduler accounting."
)
"""The cause reported when the accounting query fails. Accounting that cannot answer writes nothing, which is
indistinguishable from an answer holding no row for anything, so the failure is reported rather than resolved."""

_TRACKER_READ_FAILURE: str = (
    "Unable to read what the named batches' jobs recorded on their own processing trackers, which is the record that "
    "decides whether a job may be run again."
)
"""The cause reported when the host cannot rewrite or deliver the state artifacts. Without them a verdict would rest
on the scheduler alone, which cannot tell a job that finished from one whose tracker still claims to be running."""

_CLOSURE_FAILURE: str = (
    "Unable to retire the batches every allocation of which resolves to a plain drop from the submission ledger."
)
"""The cause reported when the closure that runs inside a status read or a cancellation cannot write. Closure
snapshots each such batch before dropping it, so a failure here leaves those batches outstanding and the next read
retries them."""

_CANCEL_ISSUE_FAILURE: str = (
    "Unable to cancel the allocations the named batches hold on the remote compute server's scheduler, so nothing was "
    "canceled and every named batch stays outstanding."
)
"""The cause reported when the cancellation itself fails. It is named apart from the reads that follow it, because a
cancellation that never reached the scheduler leaves work running while the reads behind it leave the cancellation
standing."""

_CLAIMED_CANCEL_FAILURE: str = (
    "Unable to cancel the further allocations the named batches' jobs claim on their own processing trackers, which "
    "are the ones this machine's ledger never recorded. The allocations the ledger does record were canceled, so a "
    "job may still be carried by the allocation another submitter started for it."
)
"""The cause reported when the second cancellation, which names the allocations the resolution found held beyond the
ledger's own, fails. It is named apart from the first because the first one stands, so a caller reads which half of the
work was actually stopped."""

_CANCELLATION_STANDS: str = (
    "The cancellation itself was issued, so retrying it is safe, and the named batches stay outstanding for the next "
    "status read to resolve and close."
)
"""The note appended to every failure a cancellation meets after the scheduler accepted it. Each step names its own
cause, and this is what tells a caller which part of the call still stands."""

_CANCEL_FAILURE: str = (
    "Unable to cancel the allocations the scheduler still holds, so nothing was remediated. The trackers were left "
    "untouched, because resetting one while its allocation still runs would let that allocation write into it."
)
"""The cause reported when the override's cancellation fails. The cancellation runs ahead of every step that writes,
which is the tracker reset, the snapshot, and the drop, so a failure here stops the remediation with nothing yet
changed."""

_RESET_FAILURE: str = (
    "Unable to return the stranded jobs to the scheduled state, so nothing was retired. Dropping a ledger entry while "
    "its job's tracker still claims to be running would leave that job claimed by a record no rerun can clear."
)
"""The cause reported when a stranded job's tracker cannot be reset. The reset is the whole point of remediating such
a job, so the drop is refused rather than waived: retrying it once the host answers again costs nothing."""


def remote_batch_status(
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
    """Resolves the state of the outstanding remote batches exactly, in three widening stages.

    A bare call covers every outstanding batch and resolves each of its allocations against three records: the
    scheduler's accounting, the scheduler's queue, and the processing tracker of the job that allocation carries. It
    reports the counts alongside a ``breakdown`` naming every batch, pipeline, job type, processing unit, accounting
    state, scheduler state, tracker status, and verdict. Naming a filter adds a page of allocations, and opting into
    detail adds the evidence behind each verdict, the resources the allocation requested, and the log files it wrote.

    Every allocation resolves to one ``scheduler_state``. It is ``held`` when the queue carries it or accounting
    reports a state it has yet to leave. It is ``settled`` when accounting reports a state it never leaves and the
    queue no longer carries it. It is ``gone`` only when accounting returns no row for it and the queue does not carry
    it either. A record that could not be read holds every allocation, since a source that did not answer is no
    evidence of absence. A ``BLOCKED`` allocation is the exception the queue does not hold, because its dependency can
    never be satisfied and it will never run whatever the queue still carries. It resolves as ``settled`` on every read
    rather than as ``held`` on one and ``settled`` on the next.

    That state and the job's own tracker together carry the ``verdict``, which is the value on which a caller acts.
    ``running`` means the scheduler still holds the allocation, holds the one its tracker claims, or that tracker
    claims to be running under an executor for which neither scheduler record answers, and nothing is remediated for it.
    ``finished`` and ``failed`` mean the job recorded an outcome, and its tracker is left exactly as it stands.
    ``abandoned`` means nothing claims the job, so it is already runnable. ``stranded`` means the job's tracker still
    claims to be running while no allocation is, which is the one case whose remediation writes to a tracker. Each
    verdict also carries the ``remediation`` a default ``retire_remote_batches_tool`` call would apply to it.

    This tracks a run in flight. A batch stops being reported as outstanding once every one of its allocations
    resolves to the plain ``drop`` remediation. The call that observes that closes the batch, carries its outcome, and
    reports what remains outstanding afterwards rather than what was outstanding before. A batch holding an
    allocation whose remediation is anything else stays outstanding for ``retire_remote_batches_tool``, since closing
    it would drop a record while work is still held, a tracker still claims a run, or an allocation still needs
    canceling. That closure is a derivation from the verdicts reported here rather than a second reading, so what
    closes and what this reports cannot disagree. Each call rewrites the state artifacts of the projects it covers
    before reading them, because a job records its outcome on its own tracker and nothing else regenerates those
    artifacts while a batch runs. A read is therefore one server-side regeneration per project rather than a free
    lookup.

    Args:
        batch_ids: Restricts the report to these outstanding batches. Omit to cover all of them. Naming any batch also
            counts as a filter, so the response carries a page of allocations.
        status_filter: Restricts the listing to one accounting state, such as ``FAILED``, ``RUNNING``, or ``BLOCKED``.
        session_paths: Restricts the listing to these processing unit directories.
        job_ids: Restricts the listing to these tracker job identifiers.
        job_names: Restricts the listing to these job type names, such as ``motion_energy``.
        pipelines: Restricts the listing to these pipelines.
        limit: The allocations to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero
            lists every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list allocations when no filter is named.
        detailed: Determines whether the listed allocations carry the evidence behind their verdict, their requested
            resources, and their log paths.

    Returns:
        A response dict with ``active``, which reports whether any covered allocation resolves as running and is the
        reading behind each batch's ``progressing`` verdict, alongside the ``batches`` covered. It carries
        ``stalled_batch_ids``, naming the batches that can no longer settle, and ``uncovered_batch_ids``, naming any
        batch another process recorded while this call ran, about which this call's records say nothing and which the
        next read covers. It carries a ``summary`` counting the allocations by accounting state alongside the total, a
        ``breakdown`` per axis, and a ``scheduler_read_error`` that is empty unless one of the scheduler's records could
        not be read. It also carries the ``outcomes`` of any batch that closed on this call, which is empty when none
        did. Each ``batches`` entry carries the batch's ``batch_id``, ``submitted_at``, ``outstanding_seconds``,
        ``pipelines``, ``batch_directory``, and ``total_jobs``, alongside its ``progress`` verdict of ``progressing``,
        ``stalled``, or ``awaiting_closure`` and a ``verdicts`` count per allocation verdict. It also carries its
        ``live_allocations`` count with the ``running_allocations`` it names, its ``stranded_allocation_count`` with the
        ``stranded_allocations`` it names, its ``unresolvable_allocation_count`` with the ``unresolvable_allocations``
        it names, and the ``remedy`` naming what to do about that verdict. Carries a ``jobs`` list with ``rows``,
        ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a filter is named or the listing is requested.
        If no batch is outstanding, ``active`` is False with an explanatory ``message``. Returns an error when the
        ledger cannot be read, when the server cannot be reached, when accounting or the trackers cannot be read, and
        when the closure of a settled batch cannot be written.
    """
    try:
        ledger = read_ledger()
    except Exception as exception:
        return error_response(message=_failure_message(cause=_LEDGER_READ_FAILURE, exception=exception))

    if not ledger.batches:
        return ok_response(active=False, message=_NOTHING_OUTSTANDING)

    if status_filter is not None and status_filter not in {member.value for member in JobStatus}:
        return error_response(
            message=(
                f"Unknown scheduler state '{status_filter}'. "
                f"Available: {', '.join(sorted(member.value for member in JobStatus))}."
            )
        )

    if batch_ids is not None:
        unknown = sorted(batch for batch in batch_ids if ledger.resolve_batch(batch_id=batch) is None)
        if unknown:
            return error_response(message=_unknown_batch_message(unknown=unknown, ledger=ledger))

    batches = resolve_batches(ledger=ledger, batch_ids=batch_ids)
    try:
        connection = connect_to_server()
    except Exception as exception:
        return error_response(message=_failure_message(cause=_UNREACHABLE_SERVER, exception=exception))

    with connection as server:
        host = RemoteHost(server=server)
        submissions = [entry for batch in batches for entry in batch.submissions]
        try:
            claims = resolve_tracker_claims(host=host, submissions=submissions)
        except Exception as exception:
            return error_response(message=_failure_message(cause=_TRACKER_READ_FAILURE, exception=exception))

        try:
            reading = read_scheduler_records(
                server=server, allocations=resolve_queried_allocations(submissions=submissions, claims=claims)
            )
        except Exception as exception:
            return error_response(message=_failure_message(cause=_ACCOUNTING_READ_FAILURE, exception=exception))

        # Resolves every covered allocation once, ahead of the closure, so the verdicts this read publishes and the
        # batches it closes are one computation over one set of entries rather than two that could disagree.
        resolved = resolve_allocations(batches=batches, reading=reading, claims=claims)

        # Closes every batch whose entries all prescribe a plain drop before it leaves the ledger, so such a batch is
        # answerable from a durable snapshot rather than forgotten the moment it stops being outstanding.
        try:
            closed = close_settled_batches(host=host, batches=batches, resolutions=resolved)
        except Exception as exception:
            return error_response(message=_failure_message(cause=_CLOSURE_FAILURE, exception=exception))

        # The ledger is read again to learn which of the covered batches that closure retired, so a batch that just
        # closed is reported as closed rather than as outstanding. A batch another process recorded in between is
        # named rather than resolved, because the records this call gathered say nothing about it.
        try:
            held = {batch.batch_id for batch in resolve_batches(ledger=read_ledger(), batch_ids=batch_ids)}
        except Exception as exception:
            return error_response(message=_failure_message(cause=_LEDGER_READ_FAILURE, exception=exception))

    outstanding = [batch for batch in batches if batch.batch_id in held]
    uncovered = natsorted(held - {batch.batch_id for batch in batches})
    resolutions = [resolution for resolution in resolved if resolution.batch_id in held]

    per_job = [render_allocation(resolution=resolution, reading=reading) for resolution in resolutions]
    grouped: dict[str, list[AllocationResolution]] = {batch.batch_id: [] for batch in outstanding}
    for resolution in resolutions:
        grouped[resolution.batch_id].append(resolution)

    now = current_timestamp()
    diagnosed = [_diagnose_batch(batch=batch, resolutions=grouped[batch.batch_id], now=now) for batch in outstanding]
    response = ok_response(
        batches=diagnosed,
        stalled_batch_ids=[entry["batch_id"] for entry in diagnosed if entry["progress"] == STALLED_BATCH],
        uncovered_batch_ids=uncovered,
        active=any(resolution.verdict == RUNNING_ALLOCATION for resolution in resolutions),
        summary={"total": len(per_job), **count_values(values=[entry["status"] for entry in per_job])},
        breakdown={axis: count_values(values=[entry[axis] for entry in per_job]) for axis in _REMOTE_STATUS_AXES},
        outcomes=[asdict(outcome) for outcome in closed],
        scheduler_read_error=reading.unreadable_reason,
    )
    messages = [] if outstanding else [_ALL_BATCHES_CLOSED]
    if uncovered:
        messages.append(_uncovered_batch_message(uncovered=uncovered))
    if messages:
        response["message"] = " ".join(messages)

    selectors: dict[str, list[str] | None] = {
        "batch_id": batch_ids,
        "status": [status_filter] if status_filter is not None else None,
        "unit_path": session_paths,
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
    fields = (*_REMOTE_STATUS_SEMI_FIELDS, *_REMOTE_STATUS_DETAIL_FIELDS) if detailed else _REMOTE_STATUS_SEMI_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["jobs"] = [project_item(item=entry, fields=fields) for entry in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


def remote_batch_cancel(batch_ids: list[str] | None = None) -> dict[str, Any]:
    """Cancels the allocations of the outstanding remote batches.

    Cancels queued and running allocations alike in one command. A dependent of a canceled allocation is canceled by
    the scheduler in turn, because its dependency can no longer complete successfully. The batches are resolved from
    the submission ledger, so a batch submitted before this server started is cancelable too.

    Canceling then resolves every named batch exactly as ``get_processing_status_tool`` resolves it. That resolution
    names the allocations a job's own tracker claims alongside the ones this machine recorded, and every one of those
    the scheduler still holds is canceled in turn. A job carried by an allocation another machine submitted is
    therefore stopped rather than left running behind a report that its batch was canceled. The ledger's own
    allocations are canceled first, ahead of the two reads the resolution needs, so a read that cannot answer still
    leaves the cancellation issued.

    The batches whose every allocation resolves to a plain ``drop`` are then closed. The scheduler applies a
    cancellation asynchronously, so an allocation it still carries leaves its batch outstanding for a later read to
    close. So does a job whose own tracker still claims to be running the run this call just stopped, which
    ``retire_remote_batches_tool`` releases.

    Args:
        batch_ids: The outstanding batches to cancel. Omit to cancel all of them.

    Returns:
        A response dict with ``canceled``, a ``canceled_jobs`` count of every allocation the cancellation named,
        including the ones that had already finished, the ``canceled_allocations`` it named, the ``batch_ids`` it
        covered, and a ``message``. Returns an error when the ledger cannot be read, when no batch is outstanding, when
        a named identifier is not outstanding, and when the named batches hold no allocation. It returns an error as
        well when the server cannot be reached, when either cancellation fails, and when the reads or the closure
        behind an issued cancellation fail, which each report as themselves.
    """
    try:
        ledger = read_ledger()
    except Exception as exception:
        return error_response(message=_failure_message(cause=_LEDGER_READ_FAILURE, exception=exception))

    if not ledger.batches:
        return error_response(message=_NOTHING_OUTSTANDING)

    if batch_ids is not None:
        unknown = sorted(batch for batch in batch_ids if ledger.resolve_batch(batch_id=batch) is None)
        if unknown:
            return error_response(message=_unknown_batch_message(unknown=unknown, ledger=ledger))

    batches = resolve_batches(ledger=ledger, batch_ids=batch_ids)
    submissions = [submission for batch in batches for submission in batch.submissions]
    if not submissions:
        return error_response(message="The named batches hold no allocation to cancel.")

    try:
        connection = connect_to_server()
    except Exception as exception:
        return error_response(message=_failure_message(cause=_UNREACHABLE_SERVER, exception=exception))

    # Each step names its own cause, because a cancellation that never reached the scheduler leaves work running while
    # a read that failed behind an accepted one leaves that cancellation standing.
    with connection as server:
        try:
            allocations = cancel_submissions(server=server, submissions=submissions)
        except Exception as exception:
            return error_response(message=_failure_message(cause=_CANCEL_ISSUE_FAILURE, exception=exception))

        # Resolves the canceled batches the way a status read resolves them, then closes the ones every entry of
        # which prescribes a plain drop, so a canceled run leaves the same durable record as a completed one.
        host = RemoteHost(server=server)
        try:
            claims = resolve_tracker_claims(host=host, submissions=submissions)
        except Exception as exception:
            return error_response(
                message=_standing_cancellation_message(cause=_TRACKER_READ_FAILURE, exception=exception)
            )

        try:
            reading = read_scheduler_records(
                server=server, allocations=resolve_queried_allocations(submissions=submissions, claims=claims)
            )
        except Exception as exception:
            return error_response(
                message=_standing_cancellation_message(cause=_ACCOUNTING_READ_FAILURE, exception=exception)
            )

        resolutions = resolve_allocations(batches=batches, reading=reading, claims=claims)

        # Reaches the allocation a job's own tracker claims, which the ledger never recorded and the cancellation above
        # therefore never named. That claim is the allocation actually carrying the job, so leaving it running would
        # report a batch as canceled while its work continued. It is resolved rather than assumed, through the same
        # call the override's cancellation uses, and only the allocations that resolution leaves held are named.
        claimed = [
            allocation
            for allocation in resolve_live_allocations(
                resolutions=[entry for entry in resolutions if entry.verdict == RUNNING_ALLOCATION]
            )
            if allocation not in set(allocations)
        ]
        if claimed:
            try:
                allocations.extend(cancel_allocations(server=server, allocations=claimed))
            except Exception as exception:
                return error_response(
                    message=_standing_cancellation_message(cause=_CLAIMED_CANCEL_FAILURE, exception=exception)
                )

        try:
            close_settled_batches(host=host, batches=batches, resolutions=resolutions)
        except Exception as exception:
            return error_response(message=_standing_cancellation_message(cause=_CLOSURE_FAILURE, exception=exception))

    return ok_response(
        canceled=True,
        canceled_jobs=len(allocations),
        canceled_allocations=natsorted(set(allocations)),
        batch_ids=[batch.batch_id for batch in batches],
        message=(
            "Cancellation issued for every allocation of the named batches, including the ones already finished and "
            "the ones their jobs' own trackers claim."
        ),
    )


def remote_batch_retire(
    batch_ids: list[str], *, force: bool = False, drop_without_outcome: bool = False
) -> dict[str, Any]:
    """Applies the resolved remediation to every allocation of the named batches, then drops their ledger entries.

    Every allocation is resolved exactly as the status read resolves it, and the verdict that resolution carries
    decides what happens to it. A ``stranded`` allocation has its job returned to the scheduled state on that job's own
    tracker, which is what releases a job no rerun could otherwise reach. A ``finished``, ``failed``, or ``abandoned``
    allocation has its tracker left exactly as it stands, so no result is discarded and no failure is silently
    cleared. What every named batch's jobs recorded is then snapshotted through the same closure applied to a settled
    batch, and the ledger entries are dropped.

    Two guarantees stand in front of that, and each is waived by its own flag and by nothing else. A batch holding an
    allocation that resolves as ``running`` is refused, because remediating it would disturb work the scheduler is
    still carrying. ``force`` waives that one, and then every allocation those entries leave held is canceled before
    any tracker is written, so no tracker is reset underneath an allocation the scheduler was never told to stop. That
    cancellation covers the allocation a job's own tracker claims as well as the one this machine recorded, including
    a claim another machine submitted. That claim is the allocation actually carrying the job, and resetting its
    tracker while it runs is the destruction this resolution exists to refuse. An allocation for which neither of the
    scheduler's records answers is cancelable by nothing, so waiving the refusal for it drops its entry with nothing
    canceled and its tracker exactly as it stands. A batch whose outcome cannot be snapshotted is refused, because the
    ledger entry is the last record naming the run, and ``drop_without_outcome`` waives that one.

    A failure that leaves nothing changed is reported rather than waived. An unreachable server hides both scheduler
    records and the trackers, so every allocation resolves as ``running`` and the refusal names ``force``. A
    cancellation or a tracker reset that fails stops the remediation with the ledger untouched, so retrying it costs
    nothing.

    Each named batch is snapshotted on its own, so one that cannot be read leaves the outcomes of the others recorded
    and reported. The drop stays all or nothing, so a caller that waives the failure remediates exactly the batches it
    named.

    Args:
        batch_ids: The outstanding batches to remediate. Naming none is an error, since remediation never defaults to
            the whole ledger.
        force: Determines whether to remediate batches holding an allocation that resolves as running. Every
            allocation those entries leave held is canceled before any tracker is written, the one a job's tracker
            claims included. An allocation whose state could not be read at all still resolves as running and is
            canceled by nothing, so waiving the refusal for it drops its entry while leaving its tracker untouched.
        drop_without_outcome: Determines whether to drop the ledger entries when their outcome cannot be snapshotted.
            What the jobs recorded is then read from the project's own job artifact instead.

    Returns:
        A response dict with ``retired``, the ``batch_ids`` the ledger held and dropped, and ``total_allocations``
        counting the allocations they held. It carries a ``batches`` list, whose entries hold each batch's
        ``batch_id``, its ``covered_batch_ids``, its ``allocations``, and its ``outstanding_seconds``. It carries an
        ``allocations`` list as well, whose entries hold each allocation's identity, its ``scheduler_state``, its
        ``tracker_status``, its ``verdict``, and the ``remediation`` applied. Each allocation entry also reports whether
        it was ``canceled``, whether its ``tracker_reset`` ran, whether its ``snapshot_recorded``, and whether its
        ``entry_dropped``. The response also carries ``canceled_allocations``, a ``reset_jobs`` count, the ``outcomes``
        closure recorded, and the ``outcome_directory`` on this machine holding those outcome files and the state
        snapshots they cite. It closes with a ``snapshot_error`` that is empty when every snapshot succeeded and names
        the batches that failed otherwise, and a ``message``. Returns an error when the ledger cannot be read or
        written, when no batch is outstanding, when no identifier is named, and when a named identifier is not
        outstanding. It returns an error as well when the jobs' own processing trackers cannot be read, when the
        scheduler's accounting cannot be read, and when an allocation resolves as running while ``force`` is not set. A
        cancellation or a tracker reset that fails is an error too, and so is a snapshot that fails while
        ``drop_without_outcome`` is not set.
    """
    try:
        ledger = read_ledger()
    except Exception as exception:
        return error_response(message=_failure_message(cause=_LEDGER_READ_FAILURE, exception=exception))

    if not ledger.batches:
        return error_response(message=_NOTHING_OUTSTANDING)
    if not batch_ids:
        return error_response(message=_NO_BATCH_NAMED)

    unknown = sorted(batch for batch in batch_ids if ledger.resolve_batch(batch_id=batch) is None)
    if unknown:
        return error_response(message=_unknown_batch_message(unknown=unknown, ledger=ledger))

    batches = resolve_batches(ledger=ledger, batch_ids=batch_ids)
    try:
        connection = connect_to_server()
    except Exception as exception:
        return _remediate_unreadable(
            batches=batches,
            reason=_failure_message(cause=_UNREACHABLE_SERVER, exception=exception),
            force=force,
            drop_without_outcome=drop_without_outcome,
        )

    with connection as server:
        return _remediate_batches(
            server=server, batches=batches, force=force, drop_without_outcome=drop_without_outcome
        )


def _remediate_batches(
    server: Server, batches: Sequence[SubmissionBatch], *, force: bool, drop_without_outcome: bool
) -> dict[str, Any]:
    """Resolves every allocation of the named batches against the connected server and applies its remediation.

    Args:
        server: The connected server that runs the allocations.
        batches: The recorded batches to remediate.
        force: Determines whether to remediate batches holding an allocation that resolves as running.
        drop_without_outcome: Determines whether to drop the entries when their outcome cannot be snapshotted.

    Returns:
        The response, which is a refusal when a guarantee holds and the report of what was applied otherwise.
    """
    host = RemoteHost(server=server)
    submissions = [entry for batch in batches for entry in batch.submissions]
    try:
        claims = resolve_tracker_claims(host=host, submissions=submissions)
    except Exception as exception:
        return error_response(message=_failure_message(cause=_TRACKER_READ_FAILURE, exception=exception))

    try:
        reading = read_scheduler_records(
            server=server, allocations=resolve_queried_allocations(submissions=submissions, claims=claims)
        )
    except Exception as exception:
        return error_response(message=_failure_message(cause=_ACCOUNTING_READ_FAILURE, exception=exception))

    resolutions = resolve_allocations(batches=batches, reading=reading, claims=claims)
    running = [resolution for resolution in resolutions if resolution.verdict == RUNNING_ALLOCATION]
    if running and not force:
        return error_response(
            message=_running_allocation_message(resolutions=running, reason=reading.unreadable_reason)
        )

    # The cancellation runs ahead of the tracker reset, the snapshot, and the drop, so no tracker is reset underneath
    # an allocation the scheduler was never told to stop. It names every allocation these entries leave held, the
    # recorded one and the one a job's tracker claims alike, because the allocation carrying the job is not always the
    # one this ledger recorded. The verdicts are then resolved again against what this call actually canceled, because
    # the scheduler applies a cancellation asynchronously and a query issued straight afterwards reports the old state.
    canceled: list[str] = []
    if running:
        try:
            canceled = cancel_allocations(server=server, allocations=resolve_live_allocations(resolutions=running))
        except Exception as exception:
            return error_response(message=_failure_message(cause=_CANCEL_FAILURE, exception=exception))
        resolutions = resolve_allocations(
            batches=batches, reading=reading.canceling(allocations=canceled), claims=claims
        )

    try:
        reset = reset_stranded_jobs(host=host, resolutions=resolutions)
    except Exception as exception:
        return error_response(message=_failure_message(cause=_RESET_FAILURE, exception=exception))

    outcomes: list[dict[str, Any]] = []
    failures: list[str] = []
    for batch in batches:
        try:
            outcomes.extend(asdict(outcome) for outcome in close_covered_batches(host=host, batch=batch))
        except Exception as exception:
            failures.append(f"'{batch.batch_id}' ({exception})")

    return _drop_batches(
        batches=batches,
        resolutions=resolutions,
        outcomes=outcomes,
        canceled=canceled,
        reset=reset,
        snapshot_error=f"Unable to snapshot what the jobs of {', '.join(failures)} recorded." if failures else "",
        drop_without_outcome=drop_without_outcome,
    )


def _remediate_unreadable(
    batches: Sequence[SubmissionBatch], reason: str, *, force: bool, drop_without_outcome: bool
) -> dict[str, Any]:
    """Applies the remediation for batches whose state could not be read at all.

    Notes:
        A reading that answered nothing holds every allocation, so every one of them resolves as running and the
        refusal that names ``force`` is what a caller meets first. Nothing is canceled and no tracker is written,
        because reaching either would need the connection that failed.

    Args:
        batches: The recorded batches to remediate.
        reason: What stopped the state from being read, which the refusals report as the cause.
        force: Determines whether to remediate batches holding an allocation that resolves as running.
        drop_without_outcome: Determines whether to drop the entries when their outcome cannot be snapshotted.

    Returns:
        The response, which is a refusal until both waivers are given and the report of the drop afterwards.
    """
    resolutions = resolve_allocations(batches=batches, reading=SchedulerReading(unreadable_reason=reason), claims={})
    running = [resolution for resolution in resolutions if resolution.verdict == RUNNING_ALLOCATION]
    if running and not force:
        return error_response(message=_running_allocation_message(resolutions=running, reason=reason))

    return _drop_batches(
        batches=batches,
        resolutions=resolutions,
        outcomes=[],
        canceled=[],
        reset=set(),
        snapshot_error=reason,
        drop_without_outcome=drop_without_outcome,
    )


def _drop_batches(
    batches: Sequence[SubmissionBatch],
    resolutions: Sequence[AllocationResolution],
    outcomes: list[dict[str, Any]],
    canceled: Sequence[str],
    reset: set[tuple[str, str]],
    snapshot_error: str,
    *,
    drop_without_outcome: bool,
) -> dict[str, Any]:
    """Drops the remediated batches from the submission ledger and reports what each allocation had applied to it.

    Args:
        batches: The recorded batches being remediated.
        resolutions: The resolutions on which the remediation acted.
        outcomes: The outcomes the snapshot recorded, one per covered batch it could read.
        canceled: The allocations the cancellation named.
        reset: The unit path and job identifier of each job whose tracker was reset.
        snapshot_error: What stopped a snapshot, or empty when every one of them succeeded.
        drop_without_outcome: Determines whether to drop the entries despite a failed snapshot.

    Returns:
        The response, which is a refusal when the snapshot failed and the waiver was withheld.
    """
    if snapshot_error and not drop_without_outcome:
        return error_response(message=_snapshot_failure_message(reason=snapshot_error))

    try:
        dropped = forget_batches(batch_ids=[batch.batch_id for batch in batches])
    except Exception as exception:
        return error_response(message=_failure_message(cause=_LEDGER_WRITE_FAILURE, exception=exception))

    held = set(dropped)
    retired = [batch for batch in batches if batch.batch_id in held]
    snapshotted = {str(outcome["batch_id"]) for outcome in outcomes}
    recorded = {batch.batch_id for batch in batches if snapshotted.intersection(batch.covered_batch_ids)}
    now = current_timestamp()
    return ok_response(
        retired=True,
        batch_ids=dropped,
        total_allocations=sum(len(batch.submissions) for batch in retired),
        batches=[
            {
                "batch_id": batch.batch_id,
                "covered_batch_ids": batch.covered_batch_ids,
                "allocations": [entry.slurm_job_id for entry in batch.submissions],
                "outstanding_seconds": _outstanding_seconds(submitted_at=batch.submitted_at, now=now),
            }
            for batch in retired
        ],
        allocations=[
            _render_remediation(
                resolution=resolution, canceled=set(canceled), reset=reset, recorded=recorded, dropped=held
            )
            for resolution in resolutions
        ],
        canceled_allocations=sorted(set(canceled)),
        reset_jobs=len(reset),
        outcomes=outcomes,
        outcome_directory=str(batch_directory()),
        snapshot_error=snapshot_error,
        message=(
            f"Remediated {len(dropped)} remote batch(es) and dropped them from the submission ledger. Their jobs are "
            f"no longer claimed by a recorded allocation, so a new batch prepares and runs whatever they left "
            f"outstanding."
        ),
    )


def _render_remediation(
    resolution: AllocationResolution,
    canceled: set[str],
    reset: set[tuple[str, str]],
    recorded: set[str],
    dropped: set[str],
) -> dict[str, Any]:
    """Renders what one allocation had applied to it.

    Notes:
        An entry counts as canceled when either of the allocations it resolves was named, since the cancellation
        covers the allocation its job's tracker claims alongside the one the ledger recorded.

    Args:
        resolution: The resolution on which the remediation acted.
        canceled: The allocations the cancellation named.
        reset: The unit path and job identifier of each job whose tracker was reset.
        recorded: The batches whose snapshot was recorded.
        dropped: The batches the ledger held and dropped.

    Returns:
        The allocation's report entry.
    """
    submission = resolution.submission
    was_canceled = submission.slurm_job_id in canceled or resolution.tracker.allocation in canceled
    was_reset = (submission.unit_path, submission.job_id) in reset
    return {
        "batch_id": resolution.batch_id,
        "slurm_job_id": submission.slurm_job_id,
        "job_id": submission.job_id,
        "pipeline": submission.pipeline,
        "job_name": submission.job_name,
        "specifier": submission.specifier,
        "unit_name": submission.unit_name,
        "unit_path": submission.unit_path,
        "scheduler_state": resolution.scheduler_state,
        "tracker_status": resolution.tracker.status,
        "verdict": resolution.verdict,
        "remediation": _applied_remediation(
            canceled=was_canceled, reset=was_reset, dropped=resolution.batch_id in dropped
        ),
        "canceled": was_canceled,
        "tracker_reset": was_reset,
        "snapshot_recorded": resolution.batch_id in recorded,
        "entry_dropped": resolution.batch_id in dropped,
    }


def _applied_remediation(*, canceled: bool, reset: bool, dropped: bool) -> str:
    """Resolves the remediation one allocation actually had applied to it.

    Notes:
        This is composed from what ran rather than copied from the verdict. The verdict a caller was shown is the one
        resolved before the cancellation, and the tracker of a canceled allocation is written only when its
        post-cancellation verdict is stranded. A canceled allocation whose job recorded an outcome therefore reports
        the drop it received rather than a reset it was deliberately spared, and the ``canceled`` flag beside it is
        what says the scheduler was told to stop it.

    Args:
        canceled: Determines whether the cancellation named either of this allocation's identifiers.
        reset: Determines whether this job's tracker was returned to the scheduled state.
        dropped: Determines whether the ledger held this allocation's batch and dropped it.

    Returns:
        One of ``none``, ``drop``, ``reset_and_drop``, or ``cancel_reset_and_drop``.
    """
    if not dropped:
        return NO_REMEDIATION
    if reset:
        return CANCEL_REMEDIATION if canceled else RESET_REMEDIATION
    return DROP_REMEDIATION


def _diagnose_batch(batch: SubmissionBatch, resolutions: Sequence[AllocationResolution], now: int) -> dict[str, Any]:
    """Renders one outstanding batch alongside the verdict its allocations carry.

    Notes:
        The verdict rests on the states the batch's allocations hold rather than on how long the batch has been
        outstanding, so nothing has to be tuned and a slow run is never mistaken for a stopped one. How long it has
        been outstanding is reported beside the verdict, for a caller deciding whether a progressing batch is worth
        waiting on.

        The batch verdict and the per-allocation verdicts are one resolution rather than two, so a batch is
        progressing exactly when one of its allocations resolves as running.

    Args:
        batch: The recorded batch to render.
        resolutions: The resolutions of the allocations this batch holds.
        now: The moment of this observation, as a microsecond-precision epoch.

    Returns:
        The batch's response entry, carrying what the ledger records about it alongside the verdict, the allocations
        behind that verdict, and the remedy it names.
    """
    progress = classify_batch(resolutions=resolutions)
    running = _named_allocations(resolutions=resolutions, verdict=RUNNING_ALLOCATION)
    stranded = _named_allocations(resolutions=resolutions, verdict=STRANDED_ALLOCATION)
    unresolvable = [
        resolution.submission.slurm_job_id
        for resolution in resolutions
        if resolution.scheduler_state == GONE_ALLOCATION
    ]

    return {
        "batch_id": batch.batch_id,
        "submitted_at": batch.submitted_at,
        "outstanding_seconds": _outstanding_seconds(submitted_at=batch.submitted_at, now=now),
        "pipelines": batch.pipelines,
        "batch_directory": batch.batch_directory,
        "total_jobs": len(batch.submissions),
        "progress": progress,
        "verdicts": count_values(values=[resolution.verdict for resolution in resolutions]),
        "live_allocations": len(running),
        "running_allocations": running[:_NAMED_ALLOCATION_LIMIT],
        "stranded_allocation_count": len(stranded),
        "stranded_allocations": stranded[:_NAMED_ALLOCATION_LIMIT],
        "unresolvable_allocation_count": len(unresolvable),
        "unresolvable_allocations": unresolvable[:_NAMED_ALLOCATION_LIMIT],
        "remedy": _batch_remedy(batch_id=batch.batch_id, progress=progress, stranded=len(stranded)),
    }


def _named_allocations(resolutions: Sequence[AllocationResolution], verdict: str) -> list[str]:
    """Returns the allocations that resolved to one verdict.

    Args:
        resolutions: The resolutions from which to select.
        verdict: The verdict whose allocations to name.

    Returns:
        The allocation identifiers, in the order the batch holds them.
    """
    return [resolution.submission.slurm_job_id for resolution in resolutions if resolution.verdict == verdict]


def _outstanding_seconds(submitted_at: int, now: int) -> float | None:
    """Resolves how long a batch has been outstanding, measured from the moment its submission was recorded.

    Args:
        submitted_at: The microsecond-precision epoch at which the batch was submitted, or zero for a record written
            before the ledger carried that field.
        now: The moment of this observation, as a microsecond-precision epoch.

    Returns:
        The elapsed seconds, or None when the record carries no submission time.
    """
    if submitted_at <= 0:
        return None
    seconds = convert_time(
        time=now - submitted_at, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.SECOND, as_float=True
    )
    return round(seconds, 3)


def _batch_remedy(batch_id: str, progress: str, stranded: int) -> str:
    """Builds the instruction a caller acts on for one batch, naming the tool and the command that carry it out.

    Args:
        batch_id: The identifier of the batch the instruction names.
        progress: The verdict the batch carries.
        stranded: The allocations of the batch whose jobs are stranded on their own trackers.

    Returns:
        The instruction.
    """
    released = (
        f" That also returns its {stranded} stranded job(s) to the scheduled state, which is what lets them run again."
        if stranded
        else ""
    )
    if progress == STALLED_BATCH:
        return (
            f"Nothing the scheduler does will settle this batch. Remediate it with "
            f"retire_remote_batches_tool(batch_ids=['{batch_id}']), or with 'slf server retire-batch -b {batch_id}', "
            f"which snapshots what its jobs recorded before dropping the ledger entry and refuses while any of its "
            f"allocations resolves as running.{released}"
        )
    if progress == AWAITING_CLOSURE_BATCH:
        # Closure releases a job whose tracker no longer claims an allocation, and a stranded job is exactly the job
        # whose tracker still claims one. Advising a re-read while any job is stranded therefore names a step that
        # cannot close this batch however many times it runs, so the retirement leads instead.
        if stranded:
            return (
                f"Every allocation of this batch has settled, and {stranded} of its job(s) remain stranded on their "
                f"own trackers, which closure does not release. Remediate it with "
                f"retire_remote_batches_tool(batch_ids=['{batch_id}']), or with 'slf server retire-batch "
                f"-b {batch_id}'.{released}"
            )
        return (
            f"Every allocation of this batch has settled, so read this status again to close it. Remediate it with "
            f"retire_remote_batches_tool(batch_ids=['{batch_id}']) if that closure keeps failing."
        )
    return (
        "Wait. At least one allocation of this batch resolves as running, so the run may still advance. Cancel the "
        "batch with cancel_processing_tool using host='remote' if it should not, and remediate it afterwards."
    )


def _failure_message(cause: str, exception: Exception) -> str:
    """Builds the error message returned when one step of a remote read or remediation failed.

    Notes:
        Each step names its own cause, so a caller reads which record could not be answered rather than one shared
        report that the remote batches could not be read. What a caller does next differs by cause: a connection is
        restored, an accounting outage is waited out, and a project whose artifacts cannot be regenerated is repaired.

    Args:
        cause: The description of the step that failed.
        exception: The failure that step raised.

    Returns:
        The error message.
    """
    return f"{cause} {exception}"


def _standing_cancellation_message(cause: str, exception: Exception) -> str:
    """Builds the error message returned when a step behind an accepted cancellation failed.

    Args:
        cause: The description of the step that failed.
        exception: The failure that step raised.

    Returns:
        The error message, which names the failed step and reports that the cancellation itself stands.
    """
    return f"{_failure_message(cause=cause, exception=exception)} {_CANCELLATION_STANDS}"


def _running_allocation_message(resolutions: Sequence[AllocationResolution], reason: str) -> str:
    """Builds the error message returned when a remediation would disturb allocations that resolve as running.

    Args:
        resolutions: The resolutions that carry the running verdict.
        reason: What stopped a scheduler record from being read, or empty when both of them answered.

    Returns:
        The error message.
    """
    running = [resolution.submission.slurm_job_id for resolution in resolutions]
    cause = (
        f"{reason} Every allocation of the named batch(es) therefore resolves as '{RUNNING_ALLOCATION}', since a "
        f"record that could not be read is no evidence that the scheduler has finished with an allocation."
        if reason
        else (
            f"{len(running)} of their allocation(s) resolve as '{RUNNING_ALLOCATION}', because the scheduler still "
            f"holds the allocation, still holds the one the job's own tracker claims, or that tracker claims an "
            f"executor outside the scheduler that neither of its records answers for."
        )
    )
    return (
        f"Unable to remediate the named remote batch(es). {cause} Running: {running[:_NAMED_ALLOCATION_LIMIT]}. Wait "
        f"for them to finish, cancel them with cancel_processing_tool using host='remote', or pass force=True "
        f"('--force' on 'slf server retire-batch') to cancel each of them first and remediate anyway."
    )


def _snapshot_failure_message(reason: str) -> str:
    """Builds the error message returned when a remediation could not snapshot what the batches' jobs recorded.

    Args:
        reason: The description of what failed.

    Returns:
        The error message.
    """
    return (
        f"{reason} Nothing was retired, because dropping the ledger entry now would discard the only record naming "
        f"the run. Restore access to the server and remediate again, or pass drop_without_outcome=True "
        f"('--drop-without-outcome' on 'slf server retire-batch') to drop the entries regardless. "
        f"{_FINISHED_BATCH_GUIDANCE}"
    )


def _uncovered_batch_message(uncovered: list[str]) -> str:
    """Builds the note reported when the ledger gained a batch while this read ran.

    Notes:
        Such a batch is named rather than resolved, because the scheduler records and the tracker claims this call
        gathered were taken before it existed and say nothing about the allocations it holds.

    Args:
        uncovered: The identifiers the ledger holds that this read's records do not cover.

    Returns:
        The note.
    """
    return (
        f"Batch(es) {uncovered} were recorded while this read ran, so the records it gathered say nothing about them "
        f"and they are left unresolved here. Read the status again to cover them."
    )


def _unknown_batch_message(unknown: list[str], ledger: SubmissionLedger) -> str:
    """Builds the error message returned when a caller names a batch the ledger does not hold.

    Notes:
        A batch that the ledger held earlier is absent because it finished or was retired, so the message names where
        its outcome is read instead of only reporting the identifier as unknown.

    Args:
        unknown: The identifiers the ledger does not hold.
        ledger: The ledger against which the identifiers were resolved.

    Returns:
        The error message.
    """
    return (
        f"No outstanding remote batch has identifier(s) {unknown}. A batch leaves the ledger when every one of its "
        f"allocations resolves to a plain drop, which is the reading that closes it, or when a caller remediates it "
        f"explicitly, so a batch that is absent here has finished, was remediated, or was never submitted. "
        f"Outstanding: {sorted(batch.batch_id for batch in ledger.batches)}. {_FINISHED_BATCH_GUIDANCE}"
    )
