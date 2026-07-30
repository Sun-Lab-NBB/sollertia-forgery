from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Sequence

from .graph import (
    SUCCEEDED_STATUS as SUCCEEDED_STATUS,
    BatchDocument as BatchDocument,
    index_rows_by_unit as index_rows_by_unit,
)
from .hosts import (
    ExecutionHost as ExecutionHost,
    state_artifact_paths as state_artifact_paths,
)
from .ledger import (
    SubmissionBatch as SubmissionBatch,
    forget_batches as forget_batches,
    batch_is_settled as batch_is_settled,
    current_timestamp as current_timestamp,
)
from ..server import JobStatus as JobStatus
from .batches import (
    batch_directory as batch_directory,
    read_prepared_batch as read_prepared_batch,
    record_batch_outcome as record_batch_outcome,
)
from .planning import (
    DATASET_UNIT as DATASET_UNIT,
    SESSION_UNIT as SESSION_UNIT,
)
from .preparation import resolve_project_root as resolve_project_root
from ..shared_assets import ProcessingPipelines as ProcessingPipelines

_FAILED_STATUS: str
_OUTCOME_FIELD_LIMIT: int

@dataclass(slots=True)
class BatchOutcome:
    batch_id: str = ...
    pipeline: str = ...
    host: str = ...
    total: int = ...
    succeeded: int = ...
    failed: int = ...
    blocked: int = ...
    outstanding: int = ...
    complete: bool = ...
    failed_jobs: list[dict[str, Any]] = field(default_factory=list)
    blocked_jobs: list[dict[str, Any]] = field(default_factory=list)
    snapshot_paths: list[str] = field(default_factory=list)
    verified_at: int = ...

def close_batch(host: ExecutionHost, batch_id: str) -> BatchOutcome | None: ...
def verify_batch(host: ExecutionHost, document: BatchDocument, batch_id: str) -> BatchOutcome: ...
def close_settled_batches(
    host: ExecutionHost, batches: Sequence[SubmissionBatch], statuses: dict[str, JobStatus]
) -> list[BatchOutcome]: ...
def _resolve_outcome(
    document: BatchDocument, batch_id: str, recorded: dict[str, dict[str, dict[str, Any]]], snapshots: Sequence[Path]
) -> BatchOutcome: ...
def _failed_entry(job: dict[str, Any], state_row: dict[str, Any]) -> dict[str, Any]: ...
def _blocked_entry(job: dict[str, Any], unsatisfied: list[str]) -> dict[str, Any]: ...
