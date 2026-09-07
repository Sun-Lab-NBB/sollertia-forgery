from typing import Any
from pathlib import Path

from ataraxis_time import PrecisionTimer

from ..managing import (
    MANIFEST_AXES as MANIFEST_AXES,
    MANIFEST_JOB_NAME as MANIFEST_JOB_NAME,
    MANIFEST_SEMI_FIELDS as MANIFEST_SEMI_FIELDS,
    ProjectManifest as ProjectManifest,
    project_jobs_path as project_jobs_path,
    project_manifest_path as project_manifest_path,
    generate_project_manifest as generate_project_manifest,
)
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
    SESSION_UNIT as SESSION_UNIT,
    REMOTE_HOST_LABEL as REMOTE_HOST_LABEL,
    RemoteHost as RemoteHost,
    connect_to_server as connect_to_server,
)
from .host_resolution import (
    HOST_LABELS as HOST_LABELS,
    resolve_readable_project as resolve_readable_project,
    unsupported_host_message as unsupported_host_message,
    resolve_reported_project_path as resolve_reported_project_path,
)

_MANIFEST_DETAIL_FIELDS: tuple[str, ...]
_JOB_AXES: tuple[str, ...]
_JOB_SEMI_FIELDS: tuple[str, ...]
_JOB_DETAIL_FIELDS: tuple[str, ...]

def generate_project_manifest_tool(project_path: str, host: str = "local") -> dict[str, Any]: ...
def read_project_manifest_tool(
    project_path: str,
    host: str = "local",
    animal: str | None = None,
    session_type: str | None = None,
    system: str | None = None,
    pipeline_done: dict[str, int] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def read_project_jobs_tool(
    project_path: str,
    host: str = "local",
    animal: str | None = None,
    session: str | None = None,
    pipelines: list[str] | None = None,
    job_names: list[str] | None = None,
    status: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]: ...
def get_manifest_status_tool(project_path: str, host: str = "local") -> dict[str, Any]: ...
def _generate_remote_manifest(project_root: Path, timer: PrecisionTimer) -> dict[str, Any]: ...
