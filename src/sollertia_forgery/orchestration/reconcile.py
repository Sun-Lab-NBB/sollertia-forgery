"""Provides the reconciliation that decides whether to adopt or rerun a job the trackers already record as running."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_data_structures import ProcessingStatus

from .ledger import read_ledger
from ..server import TERMINAL_JOB_STATUSES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .graph import GenericPendingJob
    from ..server import Server

LOCAL_HOST_LABEL: str = "local"
"""The label a batch carries when it was prepared against this machine."""

REMOTE_HOST_LABEL: str = "remote"
"""The label a batch carries when it was prepared against the remote compute server."""

_SLURM_EXECUTOR_SCHEME: str = "slurm"
"""The scheme an executor identifier carries when the job ran as a SLURM allocation."""


@dataclass(slots=True)
class Reconciliation:
    """Records what reconciliation decided about a batch's jobs.

    Notes:
        ``dispatchable`` and ``adopted`` together cover every job the batch held, so a caller dispatches the first set
        and reports the second without running it again.
    """

    dispatchable: list[GenericPendingJob] = field(default_factory=list)
    """The jobs this run carries out, which are the ones with no live executor behind them."""
    adopted: dict[tuple[str, str], str] = field(default_factory=dict)
    """The identifier of the allocation already running each adopted job, keyed by dispatch key. A dependent of an
    adopted job waits on the allocation recorded here rather than on one this run submits."""
    resettable: list[GenericPendingJob] = field(default_factory=list)
    """The dispatchable jobs whose recorded state is cleared before they run, which is every one of them. The reset
    itself drops the identifiers a unit does not track."""


def reconcile_local_jobs(jobs: Sequence[GenericPendingJob]) -> Reconciliation:
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
        The reconciliation, which adopts nothing and dispatches every job.
    """
    return Reconciliation(dispatchable=list(jobs), resettable=list(jobs))


def reconcile_remote_jobs(server: Server, jobs: Sequence[GenericPendingJob]) -> Reconciliation:
    """Decides what to do with each job of a batch prepared against the remote compute server.

    Notes:
        Two records can show that a job already has an allocation, and they cover different windows. The submission
        ledger names allocations this host submitted, including ones still queued, but it knows nothing about a batch
        submitted from another machine. The executor identifier a job's tracker recorded travels with the data and so
        covers every submitter, but it only appears once the allocation starts running. Both are therefore consulted,
        the second through the batch's own descriptors, which preparation filled from the host's state artifact.

        Whichever allocation the two sources name is then queried. A job whose allocation has yet to reach a terminal
        state is adopted rather than submitted again. A job is dispatched with its record cleared when its allocation
        has finished, and when its record names an executor that was never a scheduler allocation at all.

    Args:
        server: The connected server the batch runs on.
        jobs: The batch's jobs.

    Returns:
        The reconciliation, naming which jobs to submit, which allocations to adopt, and which records to clear.
    """
    claimed = _resolve_claimed_allocations(jobs=jobs)
    statuses = server.get_job_statuses(slurm_job_ids=sorted(set(claimed.values())))

    reconciliation = Reconciliation()
    for job in jobs:
        allocation = claimed.get(job.dispatch_key)
        if allocation is not None and statuses.get(allocation) not in TERMINAL_JOB_STATUSES:
            reconciliation.adopted[job.dispatch_key] = allocation
            continue
        reconciliation.dispatchable.append(job)
        reconciliation.resettable.append(job)
    return reconciliation


def _resolve_claimed_allocations(jobs: Sequence[GenericPendingJob]) -> dict[tuple[str, str], str]:
    """Resolves the scheduler allocation already claiming each job, from the ledger and from the batch's own records.

    Notes:
        The executor a job's tracker recorded is read off the job descriptor rather than out of the tracker itself.
        Preparation regenerates the host's state artifact from its trackers and reads that artifact, so every recorded
        executor has already crossed into the batch. Opening a tracker here would read the same fact a second time,
        and for a remote batch it would have to cross the transport while the batch runs.

        Where both sources name an allocation the tracker's wins, since an executor appears only once the allocation
        starts and therefore describes a later moment than the ledger's record of submitting it. That is what lets an
        allocation another machine submitted be found, since this host's ledger knows nothing about it.

    Args:
        jobs: The jobs to resolve claims for.

    Returns:
        The identifier of the allocation claiming each job, keyed by dispatch key. Only the jobs that carry a claim
        appear.
    """
    claimed: dict[tuple[str, str], str] = {
        (submission.unit_path, submission.job_id): submission.slurm_job_id
        for batch in read_ledger().batches
        for submission in batch.submissions
    }

    for job in jobs:
        if job.status != ProcessingStatus.RUNNING.name:
            continue
        scheme, _, allocation = job.executor_id.partition(":")
        if scheme == _SLURM_EXECUTOR_SCHEME and allocation:
            claimed[job.dispatch_key] = allocation

    batch_keys = {job.dispatch_key for job in jobs}
    return {key: allocation for key, allocation in claimed.items() if key in batch_keys}
