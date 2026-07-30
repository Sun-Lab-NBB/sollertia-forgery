"""Provides the reconciliation that decides whether to adopt or rerun a job the trackers already record as running."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

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
    """The dispatchable jobs whose recorded state is cleared before they run, which is every one of them the trackers
    already hold a record of."""


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
    return Reconciliation(dispatchable=list(jobs), adopted={}, resettable=list(jobs))


def reconcile_remote_jobs(server: Server, jobs: Sequence[GenericPendingJob]) -> Reconciliation:
    """Decides what to do with each job of a batch prepared against the remote compute server.

    Notes:
        Two records can show that a job already has an allocation, and they cover different windows. The submission
        ledger names allocations this host submitted, including ones still queued, but it knows nothing about a batch
        submitted from another machine. A tracker's executor identifier travels with the data and so covers every
        submitter, but it only appears once the allocation starts running. Both are therefore consulted.

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
    """Resolves the scheduler allocation already claiming each job, from the ledger and from the trackers.

    Notes:
        The ledger is consulted first, because it names an allocation from the moment it is submitted while a tracker
        names one only once it starts. A tracker entry is read for the jobs the ledger does not cover, which is how an
        allocation submitted from another machine is still found.

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

    outstanding = [job for job in jobs if job.dispatch_key not in claimed]
    trackers = {job.tracker_path for job in outstanding if job.tracker_path.is_file()}
    recorded = {tracker_path: ProcessingTracker(file_path=tracker_path).snapshot() for tracker_path in trackers}

    for job in outstanding:
        job_state = recorded.get(job.tracker_path, {}).get(job.job_id)
        if job_state is None or job_state.status is not ProcessingStatus.RUNNING:
            continue
        executor_id = job_state.executor_id or ""
        scheme, _, allocation = executor_id.partition(":")
        if scheme == _SLURM_EXECUTOR_SCHEME and allocation:
            claimed[job.dispatch_key] = allocation

    return {key: allocation for key, allocation in claimed.items() if key in {job.dispatch_key for job in jobs}}
