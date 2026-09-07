import re
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence

from .graph import (
    GenericPendingJob as GenericPendingJob,
    build_pending_job as build_pending_job,
    index_rows_by_unit as index_rows_by_unit,
    resolve_submission_order as resolve_submission_order,
)
from .hosts import (
    ExecutionHost as ExecutionHost,
    environment_command as environment_command,
    state_artifact_paths as state_artifact_paths,
)
from .ledger import (
    SubmissionBatch as SubmissionBatch,
    RemoteSubmission as RemoteSubmission,
    record_batch as record_batch,
    current_timestamp as current_timestamp,
)
from ..server import (
    TERMINAL_JOB_STATUSES as TERMINAL_JOB_STATUSES,
    Job as Job,
    Server as Server,
    JobStatus as JobStatus,
    discover_project_markers as discover_project_markers,
    get_server_configuration as get_server_configuration,
)
from ..forging import DATASET_STATE_FILENAME as DATASET_STATE_FILENAME
from .dispatch import (
    resolve_unit_kind as resolve_unit_kind,
    resolve_job_command as resolve_job_command,
)
from .planning import project_plan_path as project_plan_path
from ..managing import (
    project_jobs_path as project_jobs_path,
    project_manifest_path as project_manifest_path,
)
from .preparation import resolve_project_root as resolve_project_root
from ..shared_assets import posix_text as posix_text

REMOTE_JOB_WALLTIME_MINUTES: int
_BATCH_DIRECTORY_NAME: str
_MEGABYTES_PER_GIGABYTE: int
_SLURM_NAME_SANITIZER: re.Pattern[str]
_SLURM_EXECUTOR_SCHEME: str
HELD_ALLOCATION: str
SETTLED_ALLOCATION: str
GONE_ALLOCATION: str
RUNNING_ALLOCATION: str
FINISHED_ALLOCATION: str
FAILED_ALLOCATION: str
ABANDONED_ALLOCATION: str
STRANDED_ALLOCATION: str
NO_REMEDIATION: str
DROP_REMEDIATION: str
RESET_REMEDIATION: str
CANCEL_REMEDIATION: str
PROGRESSING_BATCH: str
STALLED_BATCH: str
AWAITING_CLOSURE_BATCH: str
_VERDICT_REMEDIATIONS: dict[str, str]

@dataclass(frozen=True, slots=True)
class TrackerClaim:
    status: str = ...
    executor_id: str = ...
    allocation: str = ...
    @property
    def unqueryable_executor(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class SchedulerReading:
    statuses: dict[str, JobStatus] = field(default_factory=dict)
    queued: frozenset[str] = ...
    canceled: frozenset[str] = ...
    unreadable_reason: str = ...
    def resolve_state(self, allocation: str) -> str: ...
    def resolve_status(self, allocation: str) -> JobStatus: ...
    def canceling(self, allocations: Sequence[str]) -> SchedulerReading: ...

@dataclass(frozen=True, slots=True)
class AllocationResolution:
    batch_id: str
    submission: RemoteSubmission
    scheduler_state: str
    claim_state: str
    tracker: TrackerClaim
    verdict: str
    remediation: str

def remote_batch_directory(server: Server, batch_id: str) -> Path: ...
def submit_batch(
    server: Server,
    jobs: Sequence[dict[str, Any]],
    batch_id: str,
    adopted: dict[tuple[str, str], str] | None = None,
    covered_batch_ids: Sequence[str] = (),
    *,
    walltime_minutes: int = ...,
    verbose: bool = False,
) -> list[RemoteSubmission]: ...
def cancel_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> list[str]: ...
def cancel_allocations(server: Server, allocations: Sequence[str]) -> list[str]: ...
def resolve_slurm_allocation(executor_id: str) -> str: ...
def resolve_tracker_claims(
    host: ExecutionHost, submissions: Sequence[RemoteSubmission]
) -> dict[tuple[str, str], TrackerClaim]: ...
def resolve_queried_allocations(
    submissions: Sequence[RemoteSubmission], claims: Mapping[tuple[str, str], TrackerClaim]
) -> list[str]: ...
def read_scheduler_records(server: Server, allocations: Sequence[str]) -> SchedulerReading: ...
def resolve_allocations(
    batches: Sequence[SubmissionBatch], reading: SchedulerReading, claims: Mapping[tuple[str, str], TrackerClaim]
) -> list[AllocationResolution]: ...
def resolve_live_allocations(resolutions: Sequence[AllocationResolution]) -> list[str]: ...
def classify_batch(resolutions: Sequence[AllocationResolution]) -> str: ...
def reset_stranded_jobs(host: ExecutionHost, resolutions: Sequence[AllocationResolution]) -> set[tuple[str, str]]: ...
def render_allocation(resolution: AllocationResolution, reading: SchedulerReading) -> dict[str, Any]: ...
def sync_project_state(
    server: Server, project: str, local_directory: Path, *, regenerate: bool = True
) -> list[Path]: ...
def connect_to_server() -> Server: ...
def render_submission(submission: RemoteSubmission) -> dict[str, Any]: ...
def _regenerate_recorded_state(
    host: ExecutionHost, group: tuple[str, str], unit_paths: set[str]
) -> list[dict[str, Any]]: ...
def _resolve_verdict(scheduler_state: str, claim_state: str, tracker: TrackerClaim) -> str: ...
def _submit_ordered_jobs(
    server: Server,
    ordered: Sequence[GenericPendingJob],
    batch_directory: Path,
    walltime_minutes: int,
    submissions: list[RemoteSubmission],
    allocation_of_job: dict[tuple[str, str], str],
    *,
    verbose: bool,
) -> None: ...
def _resolve_slurm_job_name(job: GenericPendingJob, index: int) -> str: ...
def _regenerate_remote_state(server: Server, project_path: Path, datasets: Sequence[Path]) -> None: ...
