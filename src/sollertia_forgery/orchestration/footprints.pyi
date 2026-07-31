from pathlib import Path
from functools import cache
from dataclasses import dataclass

from cindra import (
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from sollertia_shared_assets import (
    DatasetData as DatasetData,
    SessionData,
)

from ..video import (
    ENERGY_JOB_NAME as ENERGY_JOB_NAME,
    TRACKING_JOB_NAME as TRACKING_JOB_NAME,
    TIMESTAMP_JOB_NAME as TIMESTAMP_JOB_NAME,
)
from ..forging import (
    MULTIDAY_DISCOVERY_JOB_NAME as MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME as MULTIDAY_EXTRACTION_JOB_NAME,
)
from ..runtime import RUNTIME_JOB_NAME as RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME as CHECKSUM_JOB_NAME
from ..registries import (
    resolve_two_photon_data_locator as resolve_two_photon_data_locator,
    resolve_multi_recording_configuration_resolver as resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver as resolve_single_recording_configuration_resolver,
)
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    ProcessingPipelines as ProcessingPipelines,
    multi_recording_dataset_directory as multi_recording_dataset_directory,
)
from ..microcontrollers import (
    PARSE_JOB_NAME as PARSE_JOB_NAME,
    EXTRACTION_JOB_NAME as EXTRACTION_JOB_NAME,
)

_MEMORY_ESTIMATE_TOLERANCE: float
_WORKER_MEMORY_MB: int
_SUBPROCESS_MEMORY_MB: int
_MEGABYTES_PER_GIGABYTE: int
_BYTES_PER_MEGABYTE: int
_SINGLE_PRECISION_BYTES: int
_RAW_SAMPLE_BYTES: int
_ARCHIVE_DIRECTORY_RATIO: float
_MODULE_TABLE_RATIO: float
_POSE_PREDICTION_RATIO: float
_DECODER_BUFFER_MEMORY_MB: int
_RETAINED_FRAME_BUFFERS: int
_DETECTION_ARRAY_MULTIPLIER: int
_BINARIZATION_BATCH_COPIES: int
_CHECKSUM_READER_MEMORY_MB: int
_FLUORESCENCE_FILENAME: str
_COMBINED_METADATA_FILENAME: str
_MULTI_RECORDING_DIRECTORY: str
_TRACE_ARRAY_DIMENSIONS: int
_DISCOVERY_PLANES_PER_RECORDING: int
_DISCOVERY_CLUSTERING_MEMORY_MB: int
_EXTRACTION_TRACE_COPIES: int
_EXTRACTION_BATCH_BYTES_PER_PIXEL: int
_EXTRACTION_BATCH_RETENTION: int
_ASSEMBLY_FLUORESCENCE_COLUMNS: int
_ASSEMBLY_WRITE_COPIES: int
_SUB_DATASET_BYTES_PER_SAMPLE: int
_PERCENT_PER_FRACTION: float
_COMBINATION_MEMORY_MB: int
_REGISTRATION_MEMORY_MB: int

@dataclass(frozen=True, slots=True)
class _RawImagingGeometry:
    sample_count: int
    sampling_rate: float
    raw_frame_pixels: int
    plane_extents: tuple[tuple[int, int], ...]

@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    regions: int
    samples: int
    pixels: int

def resolve_host_memory_mb() -> int: ...
def estimate_session_job_memory(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]: ...
def estimate_dataset_job_memory(
    dataset: DatasetData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]: ...
def _resolve_raw_imaging_geometry(
    session: SessionData, configuration: SingleRecordingConfiguration
) -> _RawImagingGeometry | None: ...
def _bytes_to_megabytes(byte_count: float) -> int: ...
def _apply_tolerance(memory_mb: int) -> int: ...
def _estimate_archive_reader_memory(archive_path: Path, cores: int) -> int: ...
def _estimate_checksum_memory(cores: int) -> int: ...
def _resolve_widest_camera_frame_pixels(camera_directory: Path) -> int: ...
def _estimate_motion_energy_memory(frame_pixels: int, cores: int) -> int: ...
def _estimate_binarization_memory(
    geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration
) -> int: ...
def _estimate_plane_registration_memory() -> int: ...
def _estimate_plane_processing_memory(
    extent: tuple[int, int], geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration
) -> int: ...
def _estimate_two_photon_memory(
    job_name: str, specifier: str, geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration
) -> int: ...
def _resolve_plane_index(specifier: str) -> int | None: ...
def _estimate_widest_file_memory(directory: Path, pattern: str, expansion_ratio: float) -> int: ...
@cache
def _two_photon_output_directory(project_root: Path, animal: str, session: str) -> Path: ...
def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None: ...
def _read_array_shape(array_path: Path) -> tuple[int, int] | None: ...
def _resolve_tracking_configuration(dataset: DatasetData, project_root: Path) -> MultiRecordingConfiguration | None: ...
def _resolve_tracked_regions(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
) -> int: ...
def _estimate_discovery_memory(dataset: DatasetData, animal: str, project_root: Path) -> tuple[int, bool]: ...
def _estimate_extraction_memory(
    geometry: _RecordingGeometry | None, regions: int, configuration: MultiRecordingConfiguration | None
) -> tuple[int, bool]: ...
def _estimate_assembly_memory(geometry: _RecordingGeometry | None, regions: int) -> tuple[int, bool]: ...
