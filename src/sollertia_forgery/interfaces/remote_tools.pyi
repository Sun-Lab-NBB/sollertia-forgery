from typing import Any

from ..server import (
    TERMINAL_JOB_STATUSES as TERMINAL_JOB_STATUSES,
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
    RemoteHost as RemoteHost,
    SubmissionLedger as SubmissionLedger,
    read_ledger as read_ledger,
    resolve_batches as resolve_batches,
    connect_to_server as connect_to_server,
    query_submissions as query_submissions,
    render_submission as render_submission,
    cancel_submissions as cancel_submissions,
    close_settled_batches as close_settled_batches,
)

_REMOTE_STATUS_AXES: tuple[str, ...]
_REMOTE_STATUS_SEMI_FIELDS: tuple[str, ...]
_REMOTE_STATUS_DETAIL_FIELDS: tuple[str, ...]
_FINISHED_BATCH_GUIDANCE: str
_NOTHING_OUTSTANDING: str

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
def _unknown_batch_message(unknown: list[str], ledger: SubmissionLedger) -> str: ...
