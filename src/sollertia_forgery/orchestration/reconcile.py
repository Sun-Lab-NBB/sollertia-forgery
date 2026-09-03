"""Provides the reconciliation that decides whether to adopt or rerun a job the trackers already record as running."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from .graph import resolve_submission_order
from .ledger import SubmissionBatch, RemoteSubmission, read_ledger
from .remote import (
    HELD_ALLOCATION,
    RUNNING_ALLOCATION,
    TrackerClaim,
    SchedulerReading,
    resolve_allocations,
    read_scheduler_records,
    resolve_slurm_allocation,
    resolve_queried_allocations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .graph import GenericPendingJob
    from .ledger import SubmissionLedger
    from .remote import AllocationResolution
    from ..server import Server

LOCAL_HOST_LABEL: str = "local"
"""The label a batch carries when it was prepared against this machine."""

REMOTE_HOST_LABEL: str = "remote"
"""The label a batch carries when it was prepared against the remote compute server."""


@dataclass(slots=True)
class _Reconciliation:
    """Records what reconciliation decided about a batch's jobs.

    Notes:
        ``dispatchable``, ``adopted``, and ``withheld`` together cover every job the batch held, so a caller dispatches
        the first set and reports the other two without running them again.
    """

    dispatchable: list[GenericPendingJob] = field(default_factory=list)
    """The jobs this run carries out, which are the ones the resolution does not report as running."""
    adopted: dict[tuple[str, str], str] = field(default_factory=dict)
    """The identifier of the allocation already running each adopted job, keyed by dispatch key. A dependent of an
    adopted job waits on the allocation recorded here rather than on one this run submits."""
    resettable: list[GenericPendingJob] = field(default_factory=list)
    """The dispatchable jobs whose recorded state is cleared before they run, which is every one of them. The reset
    itself drops the identifiers a unit does not track."""
    withheld: list[GenericPendingJob] = field(default_factory=list)
    """The jobs this run neither dispatches nor adopts, which are the ones the resolution reports as running while
    naming no allocation a dependent could wait on, together with the jobs that depend on them. Their records are left
    exactly as they stand."""


def reconcile_local_jobs(jobs: Sequence[GenericPendingJob]) -> _Reconciliation:
    """Decides what to do with each job of a batch prepared against this machine.

    Notes:
        Nothing else on this machine should be running a job this batch holds, so a record left in the running state
        describes a pool that died without recording an outcome. Every job is therefore dispatched and its record
        cleared.

        A recorded process identifier is not checked for liveness, because a tracker on shared storage can hold a
        process identifier belonging to another machine and identifiers are reused.

    Args:
        jobs: The batch's jobs.

    Returns:
        The reconciliation, which adopts nothing, withholds nothing, and dispatches every job.
    """
    return _Reconciliation(dispatchable=list(jobs), resettable=list(jobs))


def reconcile_remote_jobs(server: Server, jobs: Sequence[GenericPendingJob]) -> _Reconciliation:
    """Decides what to do with each job of a batch prepared against the remote compute server.

    Notes:
        Every job is resolved through the same published resolution a status read and a remediation resolve, and the
        verdict that resolution carries is what decides the job's fate. Nothing here reads a scheduler record on its
        own account, so what this dispatch adopts and what a status read reports as running cannot disagree.

        Two records can show that a job already has an allocation, and they cover different windows. The submission
        ledger names allocations this host submitted, including ones still queued, but it knows nothing about a batch
        submitted from another machine. The executor identifier that a job's tracker recorded travels with the data
        and so covers every submitter, but it only appears once the allocation starts running. Both therefore reach
        the resolution, the ledger's as the recorded allocation of each job's submission and the tracker's as that
        job's claim, which is the pairing the resolution already reads. The tracker's record is taken off the batch's
        own descriptors, which preparation filled from the host's state artifact, rather than by opening a tracker
        across the transport while the batch runs.

        A job whose verdict is ``running`` is left alone, since every allocation the resolution reports that way is
        one the scheduler still holds, one whose job a still-held allocation claims on its tracker, or one whose
        tracker claims an executor outside the scheduler. The first two name an allocation, so the job is adopted onto
        it and its dependents wait on that allocation rather than on a second one this run would submit. The third
        names none, so there is nothing to adopt and nothing for a dependent to wait on: that job is withheld with its
        dependents, because dispatching it would run a second copy over a tracker its own executor may still be
        writing to, and that is the one verdict whose remediation is to leave everything as it stands.

        Every other verdict releases the job. ``finished``, ``failed``, and ``abandoned`` mean nothing carries it any
        longer, and ``stranded`` means its tracker claims a run no allocation is carrying, which is exactly the claim
        the dispatch's own reset clears.

        The scheduler is read only when a job carries an allocation to resolve. A batch whose jobs carry none resolves
        entirely from what their trackers recorded, so a routine dispatch costs no scheduler round trip at all, and a
        queue that cannot be read is carried rather than raised, which leaves every claimed job adopted and disturbs
        nothing.

    Args:
        server: The connected server that runs the batch.
        jobs: The batch's jobs.

    Returns:
        The reconciliation, naming which jobs to submit, which allocations to adopt, which jobs to withhold, and which
        records to clear.

    Raises:
        RuntimeError: If the accounting query fails.
    """
    claims = _resolve_job_claims(jobs=jobs)
    submissions = _resolve_recorded_submissions(ledger=read_ledger(), jobs=jobs)
    resolutions = resolve_allocations(
        batches=[SubmissionBatch(submissions=submissions)],
        reading=_read_scheduler_state(server=server, submissions=submissions, claims=claims),
        claims=claims,
    )

    reconciliation = _Reconciliation()
    withheld: set[tuple[str, str]] = set()
    for job, resolution in zip(jobs, resolutions, strict=True):
        if resolution.verdict != RUNNING_ALLOCATION:
            continue
        allocation = _resolve_adoptable_allocation(resolution=resolution)
        if allocation:
            reconciliation.adopted[job.dispatch_key] = allocation
        else:
            withheld.add(job.dispatch_key)

    # Withholding propagates, because a job whose upstream stage this run neither submits nor adopts has no allocation
    # to wait on and would otherwise start against input nothing here is producing. The submission order is what makes
    # one pass enough, since it places every job behind the prerequisites it holds.
    for job in resolve_submission_order(jobs=jobs):
        if job.dispatch_key not in reconciliation.adopted and any(
            prerequisite in withheld for prerequisite in job.prerequisite_keys
        ):
            withheld.add(job.dispatch_key)

    for job in jobs:
        if job.dispatch_key in reconciliation.adopted:
            continue
        if job.dispatch_key in withheld:
            reconciliation.withheld.append(job)
            continue
        reconciliation.dispatchable.append(job)
        reconciliation.resettable.append(job)
    return reconciliation


def _resolve_job_claims(jobs: Sequence[GenericPendingJob]) -> dict[tuple[str, str], TrackerClaim]:
    """Resolves what each job's own processing tracker recorded, read off the batch's descriptors.

    Notes:
        Preparation regenerates the host's state artifact from its trackers and reads that artifact, so every recorded
        status and executor has already crossed into the batch. Opening a tracker here would read the same fact a
        second time, and for a remote batch it would have to cross the transport while the batch runs.

    Args:
        jobs: The jobs whose recorded claims to resolve.

    Returns:
        What each job's tracker recorded, keyed by the unit path and job identifier that name the job.
    """
    return {
        job.dispatch_key: TrackerClaim(
            status=job.status,
            executor_id=job.executor_id,
            allocation=resolve_slurm_allocation(executor_id=job.executor_id),
        )
        for job in jobs
    }


def _resolve_recorded_submissions(
    ledger: SubmissionLedger, jobs: Sequence[GenericPendingJob]
) -> list[RemoteSubmission]:
    """Renders each job as the submission the resolution reads, carrying the allocation the ledger recorded for it.

    Notes:
        Every job is rendered, including one the ledger holds no entry for. Such a job carries no allocation, which
        the resolution reads as a record naming nothing rather than as a record it could not resolve, and its verdict
        then rests on its tracker and on the allocation that tracker claims.

        A job the ledger records under two batches takes the allocation of the batch recorded last, which is the
        submission that superseded the earlier one.

    Args:
        ledger: The recorded ledger, which holds one entry per allocation this host submitted.
        jobs: The jobs to render.

    Returns:
        One submission per job, in the order the batch holds them.
    """
    recorded = {
        (submission.unit_path, submission.job_id): submission.slurm_job_id
        for batch in ledger.batches
        for submission in batch.submissions
    }
    return [
        RemoteSubmission(
            job_id=job.job_id,
            slurm_job_id=recorded.get(job.dispatch_key, ""),
            pipeline=job.pipeline,
            job_name=job.job_name,
            specifier=job.specifier,
            unit_path=str(job.unit_path),
            unit_name=job.name,
            cores=job.core_weight,
            memory_mb=job.memory_mb,
        )
        for job in jobs
    ]


def _read_scheduler_state(
    server: Server, submissions: Sequence[RemoteSubmission], claims: Mapping[tuple[str, str], TrackerClaim]
) -> SchedulerReading:
    """Reads the scheduler's records for the allocations this reconciliation has to resolve.

    Notes:
        A batch whose jobs name no allocation at all is resolved without reaching the scheduler, since neither record
        answers for allocations nobody named and the empty reading resolves each of those jobs from its tracker
        alone. That keeps a first dispatch, which is the common one, free of a scheduler round trip.

    Args:
        server: The connected server that runs the batch.
        submissions: The submissions whose recorded allocations to read.
        claims: What each job's tracker recorded, keyed by unit path and job identifier.

    Returns:
        The reading, which is empty when no allocation needed reading.

    Raises:
        RuntimeError: If the accounting query fails.
    """
    allocations = resolve_queried_allocations(submissions=submissions, claims=claims)
    if not allocations:
        return SchedulerReading()
    return read_scheduler_records(server=server, allocations=allocations)


def _resolve_adoptable_allocation(resolution: AllocationResolution) -> str:
    """Resolves the allocation an adopted job's dependents wait on, reading it off the resolution's own states.

    Notes:
        A resolution names two allocations, the one its ledger entry recorded and the one its job's tracker claims,
        and either being held is what carries the running verdict. The claimed one is preferred where both are held,
        because a tracker records an executor only once its allocation starts and therefore describes a later moment
        than this host's record of submitting one.

    Args:
        resolution: The resolution to read.

    Returns:
        The held allocation, or an empty string when the resolution names none, which is the job whose tracker claims
        an executor outside the scheduler.
    """
    if resolution.claim_state == HELD_ALLOCATION:
        return resolution.tracker.allocation
    if resolution.scheduler_state == HELD_ALLOCATION:
        return resolution.submission.slurm_job_id
    return ""
