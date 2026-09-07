from typing import Any

from ..server import (
    ServerConfiguration as ServerConfiguration,
    discover_project_markers as discover_project_markers,
    get_server_configuration as get_server_configuration,
    get_server_configuration_path as get_server_configuration_path,
)
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
from ..orchestration import (
    DATASET_UNIT as DATASET_UNIT,
    SESSION_UNIT as SESSION_UNIT,
    connect_to_server as connect_to_server,
)
from ..shared_assets import posix_text as posix_text

_MASKED_PASSWORD: str
_DISCOVERY_UNIT_KINDS: frozenset[str]
_DISCOVERY_FIELDS: tuple[str, ...]
_ACCOUNTING_VIEW: str
_QUEUE_VIEW: str
_SCHEDULER_VIEWS: frozenset[str]
_ALL_USERS: str
_ACCOUNTING_COLUMNS: dict[str, str]
_QUEUE_FORMAT: str
_QUEUE_FIELDS: tuple[str, ...]
_ACCOUNTING_AXES: tuple[str, ...]
_QUEUE_AXES: tuple[str, ...]
_ACCOUNTING_SEMI_FIELDS: tuple[str, ...]
_ACCOUNTING_DETAIL_FIELDS: tuple[str, ...]
_QUEUE_SEMI_FIELDS: tuple[str, ...]
_QUEUE_DETAIL_FIELDS: tuple[str, ...]
_FIELD_SEPARATOR: str
_STEP_SEPARATOR: str
_EXTERN_STEP_NAME: str
_MINIMUM_TABLE_LINES: int
_MEASURED_SIZE_FIELDS: frozenset[str]
_MEASURED_DURATION_FIELDS: frozenset[str]
_SIZE_SUFFIX_SCALES: dict[str, float]
_SECONDS_PER_DAY: float
_SECONDS_PER_MINUTE: float
_DAY_SEPARATOR: str
_TIME_SEPARATOR: str

def read_server_configuration_tool() -> dict[str, Any]: ...
def write_server_configuration_tool(
    configuration_payload: dict[str, Any], *, overwrite: bool = False
) -> dict[str, Any]: ...
def discover_remote_project_tool(
    project: str,
    unit_kind: str | None = None,
    animals: list[str] | None = None,
    sessions: list[str] | None = None,
    datasets: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    include_sessions: bool = True,
) -> dict[str, Any]: ...
def read_scheduler_jobs_tool(
    view: str = ...,
    user: str | None = None,
    job_ids: list[str] | None = None,
    job_names: list[str] | None = None,
    states: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def pull_remote_path_tool(remote_path: str, destination: str) -> dict[str, Any]: ...
def _render_configuration(instance: ServerConfiguration) -> dict[str, Any]: ...
def _accounting_command(user: str, job_ids: list[str] | None, start_time: str | None, end_time: str | None) -> str: ...
def _queue_command(user: str, job_ids: list[str] | None) -> str: ...
def _parse_accounting_rows(output: str) -> list[dict[str, str]]: ...
def _merge_accounting_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]: ...
def _exceeds_held(field: str, value: str, held: str) -> bool: ...
def _parse_size(value: str) -> float | None: ...
def _parse_duration(value: str) -> float | None: ...
def _parse_queue_rows(output: str) -> list[dict[str, str]]: ...
def _reject_unmatched(field: str, values: list[str], available: set[str]) -> dict[str, Any] | None: ...
