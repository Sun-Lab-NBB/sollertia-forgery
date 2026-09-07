from pathlib import Path
from functools import cache
from dataclasses import dataclass

from cindra import (
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from sollertia_shared_assets import (
    DatasetData as DatasetData,
    SessionData,
)

from ..video import (
    ENERGY_JOB_NAME as ENERGY_JOB_NAME,
    RENAME_JOB_NAME as RENAME_JOB_NAME,
    TRACKING_JOB_NAME as TRACKING_JOB_NAME,
    CAMERA_EXTRACTION_JOB_NAME as CAMERA_EXTRACTION_JOB_NAME,
    resolve_camera_video as resolve_camera_video,
)
from ..forging import (
    FORGING_JOB_NAME as FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME as MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME as MULTIDAY_EXTRACTION_JOB_NAME,
)
from ..runtime import RUNTIME_JOB_NAME as RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME as CHECKSUM_JOB_NAME
from ..registries import (
    resolve_pose_prediction_locator as resolve_pose_prediction_locator,
    resolve_two_photon_data_locator as resolve_two_photon_data_locator,
    resolve_assembly_source_resolver as resolve_assembly_source_resolver,
    resolve_assembly_geometry_resolver as resolve_assembly_geometry_resolver,
    resolve_forging_admission_pipelines as resolve_forging_admission_pipelines,
    resolve_multi_recording_configuration_resolver as resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver as resolve_single_recording_configuration_resolver,
)
from ..shared_assets import ProcessingPipelines as ProcessingPipelines
from ..microcontrollers import (
    PARSE_JOB_NAME as PARSE_JOB_NAME,
    CONTROLLER_EXTRACTION_JOB_NAME as CONTROLLER_EXTRACTION_JOB_NAME,
)

_MODEL_VERSION_DIGITS: int
_MEGABYTES_PER_GIGABYTE: int
_BYTES_PER_MEGABYTE: int
_SINGLE_PRECISION_BYTES: int
_ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE: int
_CONTROLLER_ARCHIVE_TABLE_RATIO: float
_DOUBLE_PRECISION_BYTES: int
_POSE_TABLE_COPIES: float
_DECODER_BUFFER_MEMORY_MB: int
_SPAWNED_PACKAGE_CHILD_MEMORY_MB: int
_DEPENDENCY_CHILD_SURPLUS_MB: int
_RESIDENT_ESTIMATE_TOLERANCE: float
_PLANE_MAPPING_STAGES: frozenset[SingleRecordingJobNames]
_TWO_CHANNEL_COUNT: int
_PLANE_BINARY_ELEMENT_BYTES: int
_PLANE_BINARY_PATTERN: str
_SHARED_LIBRARY_IMAGE_MB: int
_RETAINED_FRAME_BUFFERS: int
_UNRESOLVED_FRAME_PIXELS: int
_UNRESOLVED_FRAME_COUNT: int
_CHECKSUM_CHUNK_MEMORY_MB: int
_CHECKSUM_READER_MEMORY_MB: int
_TRACE_ARRAY_DIMENSIONS: int
_UNREADABLE_OUTPUT_ERRORS: tuple[type[Exception], ...]
_ASSEMBLY_SINGLE_DAY_COLUMNS: int
_ASSEMBLY_MULTI_DAY_COLUMNS: int
_ASSEMBLY_WRITE_COPIES: int
_ASSEMBLY_LOAD_TRANSIENT_COLUMNS: int
_SUB_DATASET_BYTES_PER_SAMPLE: int
_SOURCE_INPUT_BYTES_PER_SAMPLE: int
_PERCENT_PER_FRACTION: float
_TRACKED_REGION_HEADROOM: float
_MODULE_SPECIFIER_SEPARATOR: str
_ARCHIVE_JOB_NAMES: frozenset[str]

@dataclass(frozen=True, slots=True)
class JobFootprint:
    cores: int
    memory_mb: int
    mapped_mb: int = ...
    @property
    def resident_mb(self) -> int: ...

@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    regions: int
    samples: int

@dataclass(frozen=True, slots=True)
class _EnergyRecording:
    frame_pixels: int
    frame_count: int

def resolve_model_version() -> str: ...
def resolve_host_memory_mb() -> int: ...
def size_session_jobs(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], JobFootprint]: ...
def size_dataset_jobs(
    dataset: DatasetData, jobs: list[tuple[str, str, int]], *, planned_roi_count: int | None = None
) -> dict[tuple[str, str], JobFootprint]: ...
def _resolve_job_archives(behavior_directory: Path, jobs: list[tuple[str, str, int]]) -> dict[str, Path]: ...
def _bytes_to_megabytes(byte_count: float) -> int: ...
def _round_to_gigabyte(memory_mb: int) -> int: ...
def _apply_tolerance(memory_mb: int) -> int: ...
def _dependency_child_surplus(cores: int) -> int: ...
def _size_camera_extraction_job(archive_path: Path) -> JobFootprint: ...
def _size_controller_extraction_job(archive_path: Path) -> JobFootprint: ...
def _size_runtime_job(archive_path: Path, cores: int) -> JobFootprint: ...
def _size_checksum_job(cores: int) -> JobFootprint: ...
def _size_rename_job(cores: int) -> JobFootprint: ...
def _resolve_energy_recordings(
    behavior_directory: Path, camera_directory: Path, session_name: str, jobs: list[tuple[str, str, int]]
) -> dict[str, _EnergyRecording]: ...
def _read_recording_metadata(recording: Path) -> _EnergyRecording: ...
def _size_motion_energy_job(recording: _EnergyRecording, cores: int) -> JobFootprint: ...
def _size_two_photon_job(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> JobFootprint: ...
def _mapped_plane_megabytes(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> int: ...
def _mapped_recording_megabytes(
    job_name: MultiRecordingJobNames, specifier: str, directories: tuple[Path, ...]
) -> int: ...
def _size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: tuple[Path, ...],
    configuration: MultiRecordingConfiguration | None,
    *,
    planned_roi_count: int | None = None,
) -> JobFootprint: ...
def _size_pose_tracking_job(session: SessionData, cores: int) -> JobFootprint: ...
def _read_pose_table_shape(prediction: Path) -> tuple[int, int]: ...
def _size_module_parse_job(archive_path: Path, cores: int) -> JobFootprint: ...
@cache
def _two_photon_output_root(project_root: Path, animal: str, session: str) -> Path: ...
@cache
def _session_marker(project_root: Path, animal: str, session: str) -> SessionData: ...
def _animal_recording_directories(dataset: DatasetData, animal: str, project_root: Path) -> tuple[Path, ...]: ...
def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None: ...
def _read_array_shape(array_path: Path) -> tuple[int, int] | None: ...
def _resolve_tracking_configuration(dataset: DatasetData, project_root: Path) -> MultiRecordingConfiguration | None: ...
def _resolve_tracked_regions(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
    *,
    planned_roi_count: int | None = None,
) -> int: ...
def _size_forging_job(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
    cores: int,
    *,
    planned_roi_count: int | None = None,
) -> JobFootprint: ...
def _requires_imaging(dataset: DatasetData) -> bool: ...
def _size_behavior_assembly_job(
    system: str, project_root: Path, animal: str, session: str, cores: int
) -> JobFootprint: ...
