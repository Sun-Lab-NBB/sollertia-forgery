from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence

from .graph import (
    GenericPendingJob as GenericPendingJob,
    resolve_submission_order as resolve_submission_order,
)
from .ledger import (
    SubmissionBatch as SubmissionBatch,
    RemoteSubmission as RemoteSubmission,
    SubmissionLedger as SubmissionLedger,
    read_ledger as read_ledger,
)
from .remote import (
    HELD_ALLOCATION as HELD_ALLOCATION,
    RUNNING_ALLOCATION as RUNNING_ALLOCATION,
    TrackerClaim as TrackerClaim,
    SchedulerReading as SchedulerReading,
    AllocationResolution as AllocationResolution,
    resolve_allocations as resolve_allocations,
    read_scheduler_records as read_scheduler_records,
    resolve_slurm_allocation as resolve_slurm_allocation,
    resolve_queried_allocations as resolve_queried_allocations,
)
from ..server import Server as Server

LOCAL_HOST_LABEL: str
REMOTE_HOST_LABEL: str

@dataclass(slots=True)
class _Reconciliation:
    dispatchable: list[GenericPendingJob] = field(default_factory=list)
    adopted: dict[tuple[str, str], str] = field(default_factory=dict)
    resettable: list[GenericPendingJob] = field(default_factory=list)
    withheld: list[GenericPendingJob] = field(default_factory=list)

def reconcile_local_jobs(jobs: Sequence[GenericPendingJob]) -> _Reconciliation: ...
def reconcile_remote_jobs(server: Server, jobs: Sequence[GenericPendingJob]) -> _Reconciliation: ...
def _resolve_job_claims(jobs: Sequence[GenericPendingJob]) -> dict[tuple[str, str], TrackerClaim]: ...
def _resolve_recorded_submissions(
    ledger: SubmissionLedger, jobs: Sequence[GenericPendingJob]
) -> list[RemoteSubmission]: ...
def _read_scheduler_state(
    server: Server, submissions: Sequence[RemoteSubmission], claims: Mapping[tuple[str, str], TrackerClaim]
) -> SchedulerReading: ...
def _resolve_adoptable_allocation(resolution: AllocationResolution) -> str: ...
