from typing import Any

import polars as pl

from .responses import (
    ok_response as ok_response,
    page_fields as page_fields,
    project_item as project_item,
    resolve_page as resolve_page,
    error_response as error_response,
    reject_unknown as reject_unknown,
    frame_breakdown as frame_breakdown,
    resolve_detail_limit as resolve_detail_limit,
    resolve_elapsed_seconds as resolve_elapsed_seconds,
)
from .mcp_instance import mcp as mcp
from ..orchestration import (
    DATASET_UNIT as DATASET_UNIT,
    SESSION_UNIT as SESSION_UNIT,
    PROJECT_PLAN_SCHEMA as PROJECT_PLAN_SCHEMA,
    project_plan_path as project_plan_path,
    resolve_project_root as resolve_project_root,
)
from .host_resolution import (
    HOST_LABELS as HOST_LABELS,
    resolve_execution_host as resolve_execution_host,
    resolve_readable_project as resolve_readable_project,
    unsupported_host_message as unsupported_host_message,
)

_PLAN_AXES: tuple[str, ...]
_PLAN_SEMI_FIELDS: tuple[str, ...]
_PLAN_DETAIL_FIELDS: tuple[str, ...]

def plan_session_jobs_tool(
    session_paths: list[str], host: str = "local", *, regenerate_plan: bool = False
) -> dict[str, Any]: ...
def plan_dataset_jobs_tool(
    dataset_paths: list[str], host: str = "local", *, regenerate_plan: bool = False
) -> dict[str, Any]: ...
def generate_project_plan_tool(project_path: str, host: str = "local") -> dict[str, Any]: ...
def read_project_plan_tool(
    project_path: str,
    host: str = "local",
    unit_kind: str | None = None,
    animal: str | None = None,
    dataset: str | None = None,
    pipelines: list[str] | None = None,
    job_names: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def _plan_units(unit_paths: list[str], unit_kind: str, host: str, *, regenerate_plan: bool) -> dict[str, Any]: ...
def _plan_totals(frame: pl.DataFrame) -> dict[str, Any]: ...
def _plan_breakdown(frame: pl.DataFrame) -> list[dict[str, Any]]: ...
