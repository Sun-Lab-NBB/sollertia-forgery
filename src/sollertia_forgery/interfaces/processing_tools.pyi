from typing import Any
from dataclasses import dataclass

from ..server import Server as Server
from .responses import (
    ok_response as ok_response,
    page_fields as page_fields,
    count_values as count_values,
    project_item as project_item,
    resolve_page as resolve_page,
    bounded_counts as bounded_counts,
    error_response as error_response,
    resolve_detail_limit as resolve_detail_limit,
)
from .mcp_instance import mcp as mcp
from .remote_tools import (
    remote_batch_cancel as remote_batch_cancel,
    remote_batch_retire as remote_batch_retire,
    remote_batch_status as remote_batch_status,
)
from ..orchestration import (
    RESERVED_CORES as RESERVED_CORES,
    BATCH_PIPELINES as BATCH_PIPELINES,
    LOCAL_HOST_LABEL as LOCAL_HOST_LABEL,
    REMOTE_HOST_LABEL as REMOTE_HOST_LABEL,
    REMOTE_JOB_WALLTIME_MINUTES as REMOTE_JOB_WALLTIME_MINUTES,
    LocalHost as LocalHost,
    RemoteHost as RemoteHost,
    ExecutionHost as ExecutionHost,
    GenericPendingJob as GenericPendingJob,
    JobExecutionState as JobExecutionState,
    close_batch as close_batch,
    read_ledger as read_ledger,
    submit_batch as submit_batch,
    prepare_batch as prepare_batch,
    run_batch_job as run_batch_job,
    build_pending_job as build_pending_job,
    connect_to_server as connect_to_server,
    current_timestamp as current_timestamp,
    read_batch_outcome as read_batch_outcome,
    resolve_batch_host as resolve_batch_host,
    resolve_allocations as resolve_allocations,
    reconcile_local_jobs as reconcile_local_jobs,
    close_settled_batches as close_settled_batches,
    group_jobs_by_tracker as group_jobs_by_tracker,
    job_execution_manager as job_execution_manager,
    read_prepared_batches as read_prepared_batches,
    reconcile_remote_jobs as reconcile_remote_jobs,
    record_prepared_batch as record_prepared_batch,
    read_scheduler_records as read_scheduler_records,
    remote_batch_directory as remote_batch_directory,
    resolve_host_memory_mb as resolve_host_memory_mb,
    resolve_tracker_claims as resolve_tracker_claims,
    resolve_core_allocations as resolve_core_allocations,
    resolve_concurrency_limits as resolve_concurrency_limits,
    resolve_queried_allocations as resolve_queried_allocations,
    resolve_concurrency_reservations as resolve_concurrency_reservations,
)
from ..shared_assets import posix_text as posix_text
from .host_resolution import (
    HOST_LABELS as HOST_LABELS,
    resolve_execution_host as resolve_execution_host,
    unsupported_host_message as unsupported_host_message,
)

_CLOSED_BATCH_MESSAGE: str
_BLOCKED_SEMI_FIELDS: tuple[str, ...]
_MEMORY_BUDGET_FRACTION: float
_MINIMUM_MEMORY_BUDGET_MB: int
_STATUS_AXES: tuple[str, ...]
_STATUS_SEMI_FIELDS: tuple[str, ...]
_STATUS_DETAIL_FIELDS: tuple[str, ...]
_RESOURCE_SEMI_FIELDS: tuple[str, ...]
_RESOURCE_DETAIL_FIELDS: tuple[str, ...]
_STATUS_LABELS: tuple[str, ...]
_UNREACHABLE_SERVER: str
_RECONCILIATION_FAILURE: str
_RESET_FAILURE: str
_SUBMISSION_FAILURE: str

@dataclass(frozen=True, slots=True)
class _LocalRun:
    state: JobExecutionState[GenericPendingJob] | None = ...
    batch_ids: tuple[str, ...] = ...

_LOCAL_RUN: _LocalRun

def prepare_batch_tool(
    pipeline: str,
    session_paths: list[str],
    options: dict[str, Any] | None = None,
    host: str = "local",
    *,
    replan: bool = False,
    include_job_descriptors: bool = False,
) -> dict[str, Any]: ...
def inspect_job_resources_tool(
    pipeline: str,
    session_paths: list[str],
    options: dict[str, Any] | None = None,
    host: str = "local",
    job_names: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def execute_jobs_tool(
    batch_ids: list[str], *, core_budget_override: int = -1, memory_budget_mb: int = -1, walltime_minutes: int = -1
) -> dict[str, Any]: ...
def get_processing_status_tool(
    host: str = "local",
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
def cancel_processing_tool(host: str = "local", batch_ids: list[str] | None = None) -> dict[str, Any]: ...
def retire_remote_batches_tool(
    batch_ids: list[str], *, force: bool = False, drop_without_outcome: bool = False
) -> dict[str, Any]: ...
def reset_processing_jobs_tool(
    pipeline: str, unit_paths: list[str], job_ids: list[str] | None = None, host: str = "local"
) -> dict[str, Any]: ...
def clean_processing_output_tool(pipeline: str, session_paths: list[str], host: str = "local") -> dict[str, Any]: ...
def _prepare_batch_response(
    pipeline: str, session_paths: list[str], options: dict[str, Any] | None, host: str, *, replan: bool, record: bool
) -> dict[str, Any]: ...
def _reset_batch_jobs(host: ExecutionHost, jobs: list[GenericPendingJob]) -> None: ...
def _run_and_close_local_batch(
    state: JobExecutionState[GenericPendingJob], host: ExecutionHost, batch_ids: list[str]
) -> None: ...
def _execute_local_batch(
    host: ExecutionHost,
    pending: list[GenericPendingJob],
    batch_ids: list[str],
    core_budget_override: int,
    memory_budget_mb: int,
) -> dict[str, Any]: ...
def _execute_remote_batch(
    pending: list[GenericPendingJob], batch_ids: list[str], walltime_minutes: int
) -> dict[str, Any]: ...
def _close_finished_batches(server: Server, host: ExecutionHost, batch_id: str) -> None: ...
def _render_descriptor(job: GenericPendingJob) -> dict[str, Any]: ...
def _unsupported_message(pipeline: str) -> str: ...
def _collect_status(state: JobExecutionState[GenericPendingJob]) -> tuple[list[dict[str, Any]], dict[str, int]]: ...
def _elapsed_seconds(started_at: int | None, completed_at: int | None) -> float | None: ...
