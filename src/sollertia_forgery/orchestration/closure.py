"""Provides the closure that snapshots what a finished batch's jobs recorded, before anything stops tracking it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import field, asdict, dataclass

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingStatus

from .graph import index_rows_by_unit
from .hosts import state_artifact_paths
from .ledger import forget_batches, current_timestamp
from .remote import DROP_REMEDIATION
from .batches import (
    batch_directory,
    read_prepared_batch,
    record_batch_outcome,
    retire_prepared_batch,
)
from .dispatch import resolve_unit_kind
from .preparation import resolve_project_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .graph import BatchDocument
    from .hosts import ExecutionHost
    from .ledger import SubmissionBatch
    from .remote import AllocationResolution

_OUTCOME_FIELD_LIMIT: int = 50
"""The failed and blocked jobs one outcome enumerates. The counts always cover the whole batch, so a larger batch
reports every total while listing this many examples of each problem."""


@dataclass(slots=True)
class _BatchOutcome:
    """Records what one batch's jobs finally recorded, read back from the project's own state artifact."""

    batch_id: str = ""
    """The identifier of the batch this outcome describes."""
    pipeline: str = ""
    """The pipeline the batch dispatched."""
    host: str = ""
    """The host that ran the batch."""
    total: int = 0
    """The jobs the batch held, counting the ones it dispatched and the ones it reported blocked."""
    succeeded: int = 0
    """The dispatched jobs the state artifact records as succeeded."""
    failed: int = 0
    """The dispatched jobs the state artifact records as failed."""
    blocked: int = 0
    """The jobs that never ran because an upstream job could not supply their input, counting the ones preparation
    reported and the ones whose prerequisite failed during the run."""
    outstanding: int = 0
    """The dispatched jobs that neither succeeded nor failed and wait on nothing that failed. A batch cut short leaves
    these behind."""
    complete: bool = False
    """Determines whether every job held by the batch succeeded."""
    failed_jobs: list[dict[str, Any]] = field(default_factory=list)
    """The failed jobs, each naming its unit and carrying the error text its worker recorded."""
    blocked_jobs: list[dict[str, Any]] = field(default_factory=list)
    """The blocked jobs, each naming its unit and the upstream jobs on which it waited."""
    snapshot_paths: list[str] = field(default_factory=list)
    """Where this machine holds the state artifacts from which the outcome was read, so a caller can inspect them
    after the run without reaching back to the host."""
    verified_at: int = 0
    """The UTC timestamp (microsecond-precision epoch) at which the outcome was read."""


def close_batch(host: ExecutionHost, batch_id: str) -> _BatchOutcome | None:
    """Snapshots what one finished batch's jobs recorded and stores the result on the batch itself.

    Notes:
        Rewrites the project's artifacts on the host, reads each of the batch's jobs out of them where the host holds
        them, and delivers a copy of each artifact to this machine. Regenerating first is what makes the snapshot
        describe the state after the run rather than the state against which the run was prepared.

        The outcome is written into the batch's own outcome file, so a finished batch stays answerable once nothing is
        running and nothing is queued. The prepared document is retired behind that write, because it describes work
        for which the outcome now answers.

    Args:
        host: The host that holds the data the batch's jobs read.
        batch_id: The identifier of the batch to close.

    Returns:
        The recorded outcome, or None when this host holds no batch under that identifier.

    Raises:
        RuntimeError: If a step fails on the host.
        Timeout: If the outcome file's lock cannot be acquired within the timeout period.
    """
    document = read_prepared_batch(batch_id=batch_id)
    if document is None:
        return None

    outcome = _verify_batch(host=host, document=document, batch_id=batch_id)
    record_batch_outcome(batch_id=batch_id, outcome=asdict(outcome))
    retire_prepared_batch(batch_id=batch_id)
    return outcome


def close_covered_batches(host: ExecutionHost, batch: SubmissionBatch) -> list[_BatchOutcome]:
    """Snapshots what every prepared batch of one submission recorded, then retires each prepared document.

    Notes:
        One submission may dispatch several prepared batches, and each carries its own document, so each is
        snapshotted separately rather than folded into the identifier that keys the ledger.

        The submission's own ledger entry is left alone. Retiring that entry is the caller's decision, whether the
        caller is the closure of a settled batch or the explicit retirement of a stalled one.

    Args:
        host: The host that holds the data the batch's jobs read.
        batch: The recorded submission whose prepared batches to snapshot.

    Returns:
        The recorded outcomes, holding one entry per prepared batch this host still held a document for.

    Raises:
        RuntimeError: If a step fails on the host.
        Timeout: If an outcome file's lock cannot be acquired within the timeout period.
    """
    return [
        outcome
        for covered in batch.covered_batch_ids
        if (outcome := close_batch(host=host, batch_id=covered)) is not None
    ]


def close_settled_batches(
    host: ExecutionHost, batches: Sequence[SubmissionBatch], resolutions: Sequence[AllocationResolution]
) -> list[_BatchOutcome]:
    """Closes every batch whose resolved allocations all prescribe a plain drop, then retires the ones that closed.

    Notes:
        A batch closes here exactly when every entry the resolution holds for it prescribes ``drop``, which is the one
        remediation that leaves every tracker as it stands and needs nothing canceled. That is a derivation from the
        caller's own resolution rather than a second reading of the scheduler. It keeps this closure from diverging from
        the verdicts a caller is shown. The caller resolves once and hands the same entries to both, and nothing here
        consults a record of its own. An entry prescribing ``none`` holds its batch open because the work may still be
        live, and one prescribing a reset holds it open because a tracker still claims a run no allocation is carrying.
        Each of those is released by the explicit remediation rather than automatically.

        A batch holding no allocation carries no entry at all, so nothing it holds prescribes anything but a drop and
        it closes. That is the exit such a record needs, since closure is what drops it.

        Retirement is issued per batch identifier, and only for a batch whose own closure completed. A batch that
        fails to close therefore stays in the submission ledger however its siblings fared. The entry keeps it
        answerable and lets the next query try again.

    Args:
        host: The host that holds the data the batches' jobs read.
        batches: The batches the resolution covered.
        resolutions: The verdict and remediation resolved for every allocation those batches hold, as
            ``resolve_allocations`` returned it for these same batches.

    Returns:
        The outcomes of the batches that were closed. A closed batch with no prepared record on this host is retired
        without producing one.
    """
    prescribed: dict[str, list[str]] = {batch.batch_id: [] for batch in batches}
    for resolution in resolutions:
        prescribed.setdefault(resolution.batch_id, []).append(resolution.remediation)

    settled = [
        batch for batch in batches if all(remediation == DROP_REMEDIATION for remediation in prescribed[batch.batch_id])
    ]

    closed: list[_BatchOutcome] = []
    retired: list[str] = []
    for batch in settled:
        try:
            outcomes = close_covered_batches(host=host, batch=batch)
        except Exception as exception:
            console.echo(
                message=(
                    f"Unable to close the finished batch '{batch.batch_id}', which stays outstanding so the next "
                    f"query can try again. {exception}"
                ),
                level=LogLevel.WARNING,
            )
            continue
        closed.extend(outcomes)
        retired.append(batch.batch_id)

    if retired:
        forget_batches(batch_ids=retired)
    return closed


def _verify_batch(host: ExecutionHost, document: BatchDocument, batch_id: str) -> _BatchOutcome:
    """Reads what a batch's jobs recorded out of freshly regenerated project artifacts.

    Notes:
        The rows are read from each artifact where the host holds it, since a host resolves every path it is given
        against its own filesystem and a delivered copy sits on this machine instead. The delivery is taken separately,
        so the outcome and the snapshot it cites come from the same regeneration.

        Each artifact is delivered under the directory in which it sits on the host, because a dataset batch reads one
        same-named table per dataset and a shared destination would leave only the last one.

        A job absent from the state artifact counts as outstanding rather than missing, since a tracker that lost an
        entry describes a job that never ran.

        A job that neither succeeded nor failed is reported as blocked when any of its prerequisites failed, because
        no rerun of it alone can succeed. That is what separates work a stopped batch leaves behind from work it can
        never reach.

    Args:
        host: The host that holds the data the batch's jobs read.
        document: The prepared batch to verify.
        batch_id: The identifier under which the outcome is recorded.

    Returns:
        The batch's outcome.
    """
    unit_kind = resolve_unit_kind(pipeline=document.pipeline)
    unit_paths = [Path(entry["unit_path"]) for entry in document.units]
    project_root = resolve_project_root(unit_paths=unit_paths, unit_kind=unit_kind)

    host.materialize(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind, replan=False)

    artifacts = state_artifact_paths(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind)
    recorded = index_rows_by_unit(
        rows=[row for artifact in artifacts for row in host.read_rows(path=artifact)], key=unit_kind
    )

    snapshots = [
        delivered
        for artifact in artifacts
        if (
            delivered := host.fetch(
                path=artifact, destination=batch_directory().joinpath(batch_id, artifact.parent.name)
            )
        )
        is not None
    ]

    return _resolve_outcome(document=document, batch_id=batch_id, recorded=recorded, snapshots=snapshots)


def _resolve_outcome(
    document: BatchDocument,
    batch_id: str,
    recorded: dict[str, dict[str, dict[str, Any]]],
    snapshots: Sequence[Path],
) -> _BatchOutcome:
    """Counts a batch's jobs against the status each one recorded.

    Args:
        document: The prepared batch being verified.
        batch_id: The identifier under which the outcome is recorded.
        recorded: The refreshed state rows, keyed by unit name and then by job identifier.
        snapshots: Where this machine holds the artifacts from which the rows were read.

    Returns:
        The batch's outcome.
    """
    outcome = _BatchOutcome(
        batch_id=batch_id,
        pipeline=document.pipeline,
        host=document.host,
        total=len(document.jobs) + len(document.blocked_jobs),
        blocked=len(document.blocked_jobs),
        blocked_jobs=[
            {
                "job_id": entry["job_id"],
                "job_name": entry["job_name"],
                "specifier": entry["specifier"],
                "unit_name": entry["unit_name"],
                "unsatisfied_prerequisite_ids": entry["unsatisfied_prerequisite_ids"],
            }
            for entry in document.blocked_jobs[:_OUTCOME_FIELD_LIMIT]
        ],
        snapshot_paths=[str(path) for path in snapshots],
        verified_at=current_timestamp(),
    )

    # A state table records each job's status by the name of its ``ProcessingStatus`` member, so the comparisons below
    # read those names rather than literals restating them.
    for job in document.jobs:
        unit_state = recorded.get(job["unit_name"], {})
        status = (unit_state.get(job["job_id"]) or {}).get("status")
        if status == ProcessingStatus.SUCCEEDED.name:
            outcome.succeeded += 1
            continue
        if status == ProcessingStatus.FAILED.name:
            outcome.failed += 1
            if len(outcome.failed_jobs) < _OUTCOME_FIELD_LIMIT:
                outcome.failed_jobs.append(_failed_entry(job=job, state_row=unit_state.get(job["job_id"]) or {}))
            continue

        unsatisfied = [
            prerequisite
            for prerequisite in job.get("prerequisite_ids", ())
            if (unit_state.get(prerequisite) or {}).get("status") == ProcessingStatus.FAILED.name
        ]
        if unsatisfied:
            outcome.blocked += 1
            if len(outcome.blocked_jobs) < _OUTCOME_FIELD_LIMIT:
                outcome.blocked_jobs.append(_blocked_entry(job=job, unsatisfied=unsatisfied))
            continue
        outcome.outstanding += 1

    outcome.complete = outcome.succeeded == outcome.total
    return outcome


def _failed_entry(job: dict[str, Any], state_row: dict[str, Any]) -> dict[str, Any]:
    """Renders one failed job, carrying the error text its worker recorded.

    Args:
        job: The job's descriptor.
        state_row: The job's row in the refreshed state artifact.

    Returns:
        The failed-job entry.
    """
    return {
        "job_id": job["job_id"],
        "job_name": job["job_name"],
        "specifier": job["specifier"],
        "unit_name": job["unit_name"],
        "error_message": state_row.get("error_message"),
    }


def _blocked_entry(job: dict[str, Any], unsatisfied: list[str]) -> dict[str, Any]:
    """Renders one job blocked by a prerequisite that failed during the run.

    Args:
        job: The job's descriptor.
        unsatisfied: The identifiers of the prerequisites that failed.

    Returns:
        The blocked-job entry.
    """
    return {
        "job_id": job["job_id"],
        "job_name": job["job_name"],
        "specifier": job["specifier"],
        "unit_name": job["unit_name"],
        "unsatisfied_prerequisite_ids": unsatisfied,
    }
