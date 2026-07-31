from dataclasses import field, dataclass
from collections.abc import Sequence

from .graph import GenericPendingJob as GenericPendingJob
from .ledger import read_ledger as read_ledger
from ..server import (
    TERMINAL_JOB_STATUSES as TERMINAL_JOB_STATUSES,
    Server as Server,
)

LOCAL_HOST_LABEL: str
REMOTE_HOST_LABEL: str
_SLURM_EXECUTOR_SCHEME: str

@dataclass(slots=True)
class Reconciliation:
    dispatchable: list[GenericPendingJob] = field(default_factory=list)
    adopted: dict[tuple[str, str], str] = field(default_factory=dict)
    resettable: list[GenericPendingJob] = field(default_factory=list)

def reconcile_local_jobs(jobs: Sequence[GenericPendingJob]) -> Reconciliation: ...
def reconcile_remote_jobs(server: Server, jobs: Sequence[GenericPendingJob]) -> Reconciliation: ...
def _resolve_claimed_allocations(jobs: Sequence[GenericPendingJob]) -> dict[tuple[str, str], str]: ...
