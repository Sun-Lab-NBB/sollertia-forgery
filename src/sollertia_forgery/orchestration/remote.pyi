import re
from typing import Any
from pathlib import Path
from collections.abc import Sequence

from .graph import (
    BatchDocument as BatchDocument,
    GenericPendingJob as GenericPendingJob,
    build_pending_job as build_pending_job,
    resolve_submission_order as resolve_submission_order,
)
from .hosts import (
    RemoteHost as RemoteHost,
    environment_command as environment_command,
)
from .ledger import (
    SubmissionBatch as SubmissionBatch,
    RemoteSubmission as RemoteSubmission,
    record_batch as record_batch,
    current_timestamp as current_timestamp,
)
from ..server import (
    Job as Job,
    Server as Server,
    JobStatus as JobStatus,
    get_server_configuration as get_server_configuration,
)
from ..forging import (
    DATASET_STATE_FILENAME as DATASET_STATE_FILENAME,
    DATASET_MARKER_FILENAME as DATASET_MARKER_FILENAME,
)
from .dispatch import resolve_job_command as resolve_job_command
from .planning import project_plan_path as project_plan_path
from ..managing import (
    project_jobs_path as project_jobs_path,
    project_manifest_path as project_manifest_path,
)
from .preparation import prepare_batch as prepare_batch

REMOTE_JOB_WALLTIME_MINUTES: int
BATCH_DIRECTORY_NAME: str
_MEGABYTES_PER_GIGABYTE: int
_SLURM_NAME_SANITIZER: re.Pattern[str]

def remote_batch_directory(server: Server, batch_id: str) -> Path: ...
def prepare_remote_batch(
    server: Server, pipeline: str, unit_paths: Sequence[str], options: dict[str, Any] | None = None
) -> BatchDocument: ...
def submit_batch(
    server: Server,
    jobs: Sequence[dict[str, Any]],
    batch_id: str,
    adopted: dict[tuple[str, str], str] | None = None,
    *,
    walltime_minutes: int = ...,
    verbose: bool = False,
) -> list[RemoteSubmission]: ...
def query_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> dict[str, JobStatus]: ...
def cancel_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> list[str]: ...
def sync_project_state(
    server: Server, project: str, local_directory: Path, *, regenerate: bool = True
) -> list[Path]: ...
def connect_to_server() -> Server: ...
def render_submission(submission: RemoteSubmission) -> dict[str, Any]: ...
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
def _discover_remote_datasets(server: Server, project_path: Path) -> list[Path]: ...
def _regenerate_remote_state(server: Server, project_path: Path, datasets: Sequence[Path]) -> None: ...
