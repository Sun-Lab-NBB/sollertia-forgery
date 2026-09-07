from typing import Any
from pathlib import Path

from ..video import (
    ENERGY_JOB_NAME as ENERGY_JOB_NAME,
    RENAME_JOB_NAME as RENAME_JOB_NAME,
    TRACKING_JOB_NAME as TRACKING_JOB_NAME,
    CAMERA_EXTRACTION_JOB_NAME as CAMERA_EXTRACTION_JOB_NAME,
)
from ..forging import (
    FORGING_JOB_NAME as FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME as MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME as MULTIDAY_EXTRACTION_JOB_NAME,
)
from ..runtime import RUNTIME_JOB_NAME as RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME as CHECKSUM_JOB_NAME
from .responses import (
    ok_response as ok_response,
    page_fields as page_fields,
    count_values as count_values,
    project_item as project_item,
    resolve_page as resolve_page,
    error_response as error_response,
    resolve_detail_limit as resolve_detail_limit,
)
from ..two_photon import SingleRecordingJobNames as SingleRecordingJobNames
from .mcp_instance import mcp as mcp
from ..orchestration import (
    RESERVED_CORES as RESERVED_CORES,
    OUTCOME_FILE_SUFFIX as OUTCOME_FILE_SUFFIX,
    batch_directory as batch_directory,
    resolve_job_cores as resolve_job_cores,
    read_batch_outcome as read_batch_outcome,
    read_prepared_batch as read_prepared_batch,
    forget_batch_records as forget_batch_records,
    resolve_host_memory_mb as resolve_host_memory_mb,
    resolve_concurrency_limits as resolve_concurrency_limits,
    resolve_concurrency_reservations as resolve_concurrency_reservations,
)
from ..shared_assets import ProcessingPipelines as ProcessingPipelines
from .host_resolution import (
    HOST_LABELS as HOST_LABELS,
    unsupported_host_message as unsupported_host_message,
)
from ..microcontrollers import (
    PARSE_JOB_NAME as PARSE_JOB_NAME,
    CONTROLLER_EXTRACTION_JOB_NAME as CONTROLLER_EXTRACTION_JOB_NAME,
)

_BATCH_FILE_PATTERN: str
_BATCH_SEMI_FIELDS: tuple[str, ...]
_BATCH_DETAIL_FIELDS: tuple[str, ...]
_MODEL_FIELDS: tuple[str, ...]
_PIPELINE_JOB_NAMES: dict[ProcessingPipelines, tuple[str, ...]]

def list_prepared_batches_tool(
    batch_ids: list[str] | None = None,
    pipelines: list[str] | None = None,
    host: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    detailed: bool = False,
) -> dict[str, Any]: ...
def forget_prepared_batches_tool(batch_ids: list[str]) -> dict[str, Any]: ...
def read_resource_model_tool(
    pipelines: list[str] | None = None, job_names: list[str] | None = None, limit: int | None = None, start_row: int = 0
) -> dict[str, Any]: ...
def _read_recorded_batches(directory: Path) -> list[dict[str, Any]]: ...
def _render_job_types() -> list[dict[str, Any]]: ...
