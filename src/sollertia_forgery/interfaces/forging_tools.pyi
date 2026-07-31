from typing import Any

import polars as pl
from sollertia_shared_assets import DatasetData

from ..forging import (
    DATASET_STATE_FILENAME as DATASET_STATE_FILENAME,
    dataset_state_path as dataset_state_path,
    forging_tracker_path as forging_tracker_path,
    define_forging_dataset as define_forging_dataset,
    generate_dataset_state as generate_dataset_state,
    discover_project_datasets as discover_project_datasets,
)
from .responses import (
    ok_response as ok_response,
    page_fields as page_fields,
    count_values as count_values,
    project_item as project_item,
    resolve_page as resolve_page,
    error_response as error_response,
    reject_unknown as reject_unknown,
    frame_breakdown as frame_breakdown,
    resolve_detail_limit as resolve_detail_limit,
)
from .mcp_instance import mcp as mcp
from ..orchestration import (
    DATASET_UNIT as DATASET_UNIT,
    REMOTE_HOST_LABEL as REMOTE_HOST_LABEL,
    RemoteHost as RemoteHost,
    connect_to_server as connect_to_server,
)
from .host_resolution import (
    HOST_LABELS as HOST_LABELS,
    resolve_readable_project as resolve_readable_project,
    unsupported_host_message as unsupported_host_message,
)

_DATASET_SEMI_FIELDS: tuple[str, ...]
_STATE_AXES: tuple[str, ...]
_STATE_SEMI_FIELDS: tuple[str, ...]
_STATE_DETAIL_FIELDS: tuple[str, ...]

def define_forging_dataset_tool(
    project_path: str,
    dataset_name: str,
    session_names: list[str],
    recreate_animals: list[str] | None = None,
    host: str = "local",
    *,
    force_recreate: bool = False,
) -> dict[str, Any]: ...
def generate_dataset_state_tool(dataset_paths: list[str], host: str = "local") -> dict[str, Any]: ...
def read_dataset_state_tool(
    dataset_path: str,
    host: str = "local",
    scope: str | None = None,
    animal: str | None = None,
    session: str | None = None,
    job_names: list[str] | None = None,
    status: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def list_project_datasets_tool(
    project_path: str,
    host: str = "local",
    session: str | None = None,
    animal: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    detailed: bool = False,
) -> dict[str, Any]: ...
def _dataset_state_summary(dataset: DatasetData) -> dict[str, Any]: ...
def _status_counts(frame: pl.DataFrame) -> dict[str, int]: ...
def _generate_remote_dataset_state(dataset_paths: list[str]) -> dict[str, Any]: ...
