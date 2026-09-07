from typing import Any
from collections.abc import Sequence

from ..server import (
    Server as Server,
    JobStatus as JobStatus,
)
from .responses import (
    ok_response as ok_response,
    page_fields as page_fields,
    count_values as count_values,
    project_item as project_item,
    resolve_page as resolve_page,
    error_response as error_response,
    resolve_detail_limit as resolve_detail_limit,
)
from ..orchestration import (
    STALLED_BATCH as STALLED_BATCH,
    NO_REMEDIATION as NO_REMEDIATION,
    GONE_ALLOCATION as GONE_ALLOCATION,
    DROP_REMEDIATION as DROP_REMEDIATION,
    RESET_REMEDIATION as RESET_REMEDIATION,
    CANCEL_REMEDIATION as CANCEL_REMEDIATION,
    RUNNING_ALLOCATION as RUNNING_ALLOCATION,
    STRANDED_ALLOCATION as STRANDED_ALLOCATION,
    AWAITING_CLOSURE_BATCH as AWAITING_CLOSURE_BATCH,
    RemoteHost as RemoteHost,
    SubmissionBatch as SubmissionBatch,
    SchedulerReading as SchedulerReading,
    SubmissionLedger as SubmissionLedger,
    AllocationResolution as AllocationResolution,
    read_ledger as read_ledger,
    classify_batch as classify_batch,
    forget_batches as forget_batches,
    batch_directory as batch_directory,
    resolve_batches as resolve_batches,
    connect_to_server as connect_to_server,
    current_timestamp as current_timestamp,
    render_allocation as render_allocation,
    cancel_allocations as cancel_allocations,
    cancel_submissions as cancel_submissions,
    reset_stranded_jobs as reset_stranded_jobs,
    resolve_allocations as resolve_allocations,
    close_covered_batches as close_covered_batches,
    close_settled_batches as close_settled_batches,
    read_scheduler_records as read_scheduler_records,
    resolve_tracker_claims as resolve_tracker_claims,
    resolve_live_allocations as resolve_live_allocations,
    resolve_queried_allocations as resolve_queried_allocations,
)

_REMOTE_STATUS_AXES: tuple[str, ...]
_REMOTE_STATUS_SEMI_FIELDS: tuple[str, ...]
_REMOTE_STATUS_DETAIL_FIELDS: tuple[str, ...]
_NAMED_ALLOCATION_LIMIT: int
_FINISHED_BATCH_GUIDANCE: str
_NOTHING_OUTSTANDING: str
_ALL_BATCHES_CLOSED: str
_NO_BATCH_NAMED: str
_LEDGER_READ_FAILURE: str
_LEDGER_WRITE_FAILURE: str
_UNREACHABLE_SERVER: str
_ACCOUNTING_READ_FAILURE: str
_TRACKER_READ_FAILURE: str
_CLOSURE_FAILURE: str
_CANCEL_ISSUE_FAILURE: str
_CLAIMED_CANCEL_FAILURE: str
_CANCELLATION_STANDS: str
_CANCEL_FAILURE: str
_RESET_FAILURE: str

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
) -> dict[str, Any]: ...
def remote_batch_cancel(batch_ids: list[str] | None = None) -> dict[str, Any]: ...
def remote_batch_retire(
    batch_ids: list[str], *, force: bool = False, drop_without_outcome: bool = False
) -> dict[str, Any]: ...
def _remediate_batches(
    server: Server, batches: Sequence[SubmissionBatch], *, force: bool, drop_without_outcome: bool
) -> dict[str, Any]: ...
def _remediate_unreadable(
    batches: Sequence[SubmissionBatch], reason: str, *, force: bool, drop_without_outcome: bool
) -> dict[str, Any]: ...
def _drop_batches(
    batches: Sequence[SubmissionBatch],
    resolutions: Sequence[AllocationResolution],
    outcomes: list[dict[str, Any]],
    canceled: Sequence[str],
    reset: set[tuple[str, str]],
    snapshot_error: str,
    *,
    drop_without_outcome: bool,
) -> dict[str, Any]: ...
def _render_remediation(
    resolution: AllocationResolution,
    canceled: set[str],
    reset: set[tuple[str, str]],
    recorded: set[str],
    dropped: set[str],
) -> dict[str, Any]: ...
def _applied_remediation(*, canceled: bool, reset: bool, dropped: bool) -> str: ...
def _diagnose_batch(
    batch: SubmissionBatch, resolutions: Sequence[AllocationResolution], now: int
) -> dict[str, Any]: ...
def _named_allocations(resolutions: Sequence[AllocationResolution], verdict: str) -> list[str]: ...
def _outstanding_seconds(submitted_at: int, now: int) -> float | None: ...
def _batch_remedy(batch_id: str, progress: str, stranded: int) -> str: ...
def _failure_message(cause: str, exception: Exception) -> str: ...
def _standing_cancellation_message(cause: str, exception: Exception) -> str: ...
def _running_allocation_message(resolutions: Sequence[AllocationResolution], reason: str) -> str: ...
def _snapshot_failure_message(reason: str) -> str: ...
def _uncovered_batch_message(uncovered: list[str]) -> str: ...
def _unknown_batch_message(unknown: list[str], ledger: SubmissionLedger) -> str: ...
