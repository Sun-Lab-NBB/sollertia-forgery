"""Provides the remote halves of the processing tools, which read and cancel what the scheduler ran."""

from __future__ import annotations

from typing import Any
from dataclasses import asdict

from ..server import TERMINAL_JOB_STATUSES, JobStatus
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
    RemoteHost,
    SubmissionLedger,
    read_ledger,
    resolve_batches,
    connect_to_server,
    query_submissions,
    render_submission,
    cancel_submissions,
    close_settled_batches,
)

_REMOTE_STATUS_AXES: tuple[str, ...] = ("batch_id", "pipeline", "job_name", "status", "unit_path")
"""The job attributes a caller may filter a remote batch by, and the axes a status breakdown counts."""

_REMOTE_STATUS_SEMI_FIELDS: tuple[str, ...] = (
    "batch_id",
    "job_id",
    "slurm_job_id",
    "pipeline",
    "job_name",
    "specifier",
    "status",
    "unit_name",
)
"""The job fields a semi-detail listing carries, which is the job's identity, the allocation it runs as, and its
scheduler state."""

_REMOTE_STATUS_DETAIL_FIELDS: tuple[str, ...] = (
    "cores",
    "memory_mb",
    "slurm_job_name",
    "unit_path",
    "output_log",
    "error_log",
)
"""The job fields detail adds, which are the resources the allocation requested and where its output landed. A caller
reads the log paths to see a failed allocation's own diagnostics."""

_FINISHED_BATCH_GUIDANCE: str = (
    "Read what a finished run produced from the outcome closure recorded on the batch, which "
    "get_processing_status_tool reports, or from read_project_jobs_tool with host='remote'."
)
"""The guidance appended wherever a caller reaches for a batch the ledger no longer holds. The ledger names outstanding
allocations alone, so the answer for a finished batch is its recorded outcome or the project's own job artifact."""

_NOTHING_OUTSTANDING: str = f"No remote batch is outstanding. {_FINISHED_BATCH_GUIDANCE}"
"""The message reported when the ledger holds no batch, which means every submitted batch has finished or none was
ever submitted."""


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
    """Reports the scheduler state of the outstanding remote batches, in three widening stages.

    A bare call covers every outstanding batch, queries the scheduler for all of them in one accounting call, and
    reports the counts alongside a ``breakdown`` naming every batch, pipeline, job type, state, and unit. Naming a
    filter adds a page of jobs, and opting into detail adds the resources each allocation requested and the log files
    it wrote.

    This tracks a run in flight. A batch stops being reported as outstanding once its allocations all finish, and the
    call that observes them finishing closes the batch and carries its outcome, which is the durable record of what its
    jobs reached.

    Args:
        batch_ids: Restricts the report to these outstanding batches. Omit to cover all of them. Naming any batch also
            counts as a filter, so the response carries a page of jobs.
        status_filter: Restricts the listing to one scheduler state, such as ``FAILED``, ``RUNNING``, or ``BLOCKED``.
        session_paths: Restricts the listing to these processing unit directories.
        job_ids: Restricts the listing to these tracker job identifiers.
        job_names: Restricts the listing to these job type names, such as ``motion_energy``.
        pipelines: Restricts the listing to these pipelines.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry their requested resources and their log paths.

    Returns:
        A response dict with ``active`` (whether any allocation has yet to reach a terminal state), the ``batches``
        covered, a ``summary`` counting the allocations by state alongside the total, and a ``breakdown`` per axis. It
        also carries the ``outcomes`` of any batch that settled and closed on this call, which is empty when none did.
        Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. If no batch is outstanding, ``active`` is False with an
        explanatory ``message``.
    """
    ledger = read_ledger()
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
    submissions = [(batch, submission) for batch in batches for submission in batch.submissions]

    try:
        with connect_to_server() as server:
            statuses = query_submissions(server=server, submissions=[entry for _, entry in submissions])

            # Closes every batch the scheduler has finished with before it leaves the ledger, so a settled batch is
            # answerable from a durable snapshot rather than forgotten the moment it stops being outstanding.
            closed = close_settled_batches(host=RemoteHost(server=server), batches=batches, statuses=statuses)
    except Exception as exception:
        return error_response(message=f"Unable to query the remote batches. {exception}")

    per_job = [
        {
            **render_submission(submission=submission),
            "batch_id": batch.batch_id,
            "status": statuses[submission.slurm_job_id].value,
        }
        for batch, submission in submissions
    ]

    response = ok_response(
        batches=[
            {
                "batch_id": batch.batch_id,
                "submitted_at": batch.submitted_at,
                "pipelines": batch.pipelines,
                "batch_directory": batch.batch_directory,
                "total_jobs": len(batch.submissions),
            }
            for batch in batches
        ],
        active=any(status not in TERMINAL_JOB_STATUSES for status in statuses.values()),
        summary={"total": len(per_job), **count_values(values=[entry["status"] for entry in per_job])},
        breakdown={axis: count_values(values=[entry[axis] for entry in per_job]) for axis in _REMOTE_STATUS_AXES},
        outcomes=[asdict(outcome) for outcome in closed],
    )

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

    Canceling settles the named batches, so they are retired from the ledger and stop being reported as outstanding.

    Args:
        batch_ids: The outstanding batches to cancel. Omit to cancel all of them.

    Returns:
        A response dict with ``canceled``, a ``canceled_jobs`` count of every allocation the cancellation named,
        including the ones that had already finished, the ``batch_ids`` it covered, and a ``message``. Returns an error
        when no batch is outstanding, when the named batches hold no allocation, or when the scheduler cannot be
        reached.
    """
    ledger = read_ledger()
    if not ledger.batches:
        return error_response(message=_NOTHING_OUTSTANDING)

    batches = resolve_batches(ledger=ledger, batch_ids=batch_ids)
    submissions = [submission for batch in batches for submission in batch.submissions]
    if not submissions:
        return error_response(message="The named batches hold no allocation to cancel.")

    try:
        with connect_to_server() as server:
            allocations = cancel_submissions(server=server, submissions=submissions)

            # Re-reads the scheduler after canceling, then closes these batches so a canceled run leaves the same
            # durable record of what its jobs reached as a completed one.
            statuses = query_submissions(server=server, submissions=submissions)
            close_settled_batches(host=RemoteHost(server=server), batches=batches, statuses=statuses)
    except Exception as exception:
        return error_response(message=f"Unable to cancel the remote batches. {exception}")

    return ok_response(
        canceled=True,
        canceled_jobs=len(allocations),
        batch_ids=[batch.batch_id for batch in batches],
        message="Cancellation issued for every allocation of the named batches, including the ones already finished.",
    )


def _unknown_batch_message(unknown: list[str], ledger: SubmissionLedger) -> str:
    """Builds the error message returned when a caller names a batch the ledger does not hold.

    Notes:
        A batch the ledger held earlier is absent precisely because it finished, so the message names where its
        outcome is read instead of only reporting the identifier as unknown.

    Args:
        unknown: The identifiers the ledger does not hold.
        ledger: The ledger the identifiers were resolved against.

    Returns:
        The error message.
    """
    return (
        f"No outstanding remote batch has identifier(s) {unknown}. A batch is retired once its allocations all "
        f"finish, so a batch that is absent here has either finished or was never submitted. Outstanding: "
        f"{sorted(batch.batch_id for batch in ledger.batches)}. {_FINISHED_BATCH_GUIDANCE}"
    )
