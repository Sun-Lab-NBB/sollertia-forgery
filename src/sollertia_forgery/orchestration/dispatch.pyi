from typing import Any
from pathlib import Path
from functools import cache
from dataclasses import dataclass
from collections.abc import Callable as Callable

from sollertia_shared_assets import DatasetData, SessionData

from .graph import GenericPendingJob as GenericPendingJob
from .local import apply_decode_thread_ceiling as apply_decode_thread_ceiling
from ..video import (
    ENERGY_JOB_NAME as ENERGY_JOB_NAME,
    RENAME_JOB_NAME as RENAME_JOB_NAME,
    TRACKING_JOB_NAME as TRACKING_JOB_NAME,
    CAMERA_EXTRACTION_JOB_NAME as CAMERA_EXTRACTION_JOB_NAME,
    discover_video_jobs as discover_video_jobs,
    video_job_prerequisites as video_job_prerequisites,
    run_video_processing_pipeline as run_video_processing_pipeline,
)
from ..forging import (
    FORGING_JOB_NAME as FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME as MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME as MULTIDAY_EXTRACTION_JOB_NAME,
    FORGING_JOB_CONCURRENCY_LIMITS as FORGING_JOB_CONCURRENCY_LIMITS,
    forging_tracker_path as forging_tracker_path,
    run_forging_pipeline as run_forging_pipeline,
    discover_forging_jobs as discover_forging_jobs,
    forging_job_prerequisites as forging_job_prerequisites,
    forging_cross_recording_paths as forging_cross_recording_paths,
)
from ..runtime import (
    RUNTIME_JOB_NAME as RUNTIME_JOB_NAME,
    discover_runtime_jobs as discover_runtime_jobs,
    runtime_job_prerequisites as runtime_job_prerequisites,
    run_runtime_processing_pipeline as run_runtime_processing_pipeline,
)
from ..managing import (
    CHECKSUM_JOB_NAME as CHECKSUM_JOB_NAME,
    discover_checksum_jobs as discover_checksum_jobs,
    checksum_job_prerequisites as checksum_job_prerequisites,
    run_checksum_processing_pipeline as run_checksum_processing_pipeline,
)
from .footprints import (
    JobFootprint as JobFootprint,
    size_dataset_jobs as size_dataset_jobs,
    size_session_jobs as size_session_jobs,
)
from ..two_photon import (
    SingleRecordingJobNames as SingleRecordingJobNames,
    discover_two_photon_jobs as discover_two_photon_jobs,
    prime_two_photon_recording as prime_two_photon_recording,
    two_photon_job_prerequisites as two_photon_job_prerequisites,
    run_two_photon_processing_pipeline as run_two_photon_processing_pipeline,
)
from ..shared_assets import (
    ProcessingPipelines as ProcessingPipelines,
    resolve_session_tracker_path as resolve_session_tracker_path,
)
from ..microcontrollers import (
    PARSE_JOB_NAME as PARSE_JOB_NAME,
    CONTROLLER_EXTRACTION_JOB_NAME as CONTROLLER_EXTRACTION_JOB_NAME,
    discover_microcontroller_jobs as discover_microcontroller_jobs,
    microcontroller_job_prerequisites as microcontroller_job_prerequisites,
    run_microcontroller_processing_pipeline as run_microcontroller_processing_pipeline,
)

BATCH_PIPELINES: frozenset[ProcessingPipelines]
SESSION_UNIT: str
DATASET_UNIT: str
_UNIT_KINDS: frozenset[str]
_JOB_CORE_ALLOCATIONS: dict[str, int]
_JOB_CONCURRENCY_LIMITS: dict[str, int]
_JOB_CONCURRENCY_RESERVATIONS: dict[str, int]

@dataclass(frozen=True, slots=True)
class PipelineDispatch[UnitT]:
    pipeline: ProcessingPipelines
    unit_kind: str
    load: Callable[[Path], UnitT]
    discover: Callable[[Path], tuple[UnitT, list[tuple[str, str]], list[tuple[str, str]]]]
    worker: Callable[..., None]
    prerequisites: Callable[[UnitT, list[tuple[str, str]]], dict[tuple[str, str], tuple[tuple[str, str], ...]]]
    tracker_path: Callable[[UnitT], Path]
    output_path: Callable[[UnitT], Path | None]
    unit_name: Callable[[UnitT], str]
    size_jobs: Callable[[UnitT, list[tuple[str, str, int]]], dict[tuple[str, str], JobFootprint]]
    command: Callable[[GenericPendingJob], tuple[str, ...]]
    prime: Callable[[Path], None] | None = ...
    external_output_paths: Callable[[UnitT], tuple[Path, ...]] | None = ...

def run_batch_job(job: GenericPendingJob) -> None: ...
def resolve_job_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def resolve_dispatch(pipeline: str | ProcessingPipelines) -> PipelineDispatch[Any] | None: ...
def resolve_unit_kind(pipeline: str | ProcessingPipelines) -> str: ...
def resolve_unit_dispatches(unit_kind: str) -> tuple[PipelineDispatch[Any], ...]: ...
def resolve_job_cores(job_name: str) -> int: ...
def resolve_concurrency_limits(job_names: set[str]) -> dict[str, int]: ...
def resolve_concurrency_reservations(job_names: set[str]) -> dict[str, int]: ...
def _run_checksum_job(job: GenericPendingJob) -> None: ...
def _run_runtime_job(job: GenericPendingJob) -> None: ...
def _run_microcontroller_job(job: GenericPendingJob) -> None: ...
def _run_video_job(job: GenericPendingJob) -> None: ...
def _run_two_photon_job(job: GenericPendingJob) -> None: ...
def _run_forging_job(job: GenericPendingJob) -> None: ...
def _checksum_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _runtime_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _microcontroller_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _video_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _two_photon_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _forging_command(job: GenericPendingJob) -> tuple[str, ...]: ...
def _session_command_preamble(job: GenericPendingJob) -> tuple[str, ...]: ...
def _load_session(session_path: Path) -> SessionData: ...
def _load_dataset(dataset_path: Path) -> DatasetData: ...
def _session_sizer(
    pipeline: ProcessingPipelines,
) -> Callable[[SessionData, list[tuple[str, str, int]]], dict[tuple[str, str], JobFootprint]]: ...
def _session_tracker(pipeline: ProcessingPipelines) -> Callable[[SessionData], Path]: ...
@cache
def _pipeline_dispatch() -> dict[ProcessingPipelines, PipelineDispatch[Any]]: ...
def _assert_dispatch_coverage() -> None: ...
