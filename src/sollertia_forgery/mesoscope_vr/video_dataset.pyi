from pathlib import Path
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .metadata import VideoDataFiles as VideoDataFiles
from ..shared_assets import count_feather_rows as count_feather_rows
from .video_tracking import (
    PUPIL_CAMERA_NAME as PUPIL_CAMERA_NAME,
    PupilColumn as PupilColumn,
)

_BODY_CAMERA_NAME: str
_MICROSECONDS_PER_SECOND: float
_MINIMUM_CLOCK_FRAMES: int
_MOTION_ENERGY_COLUMN: str
_FRAME_LUMINANCE_COLUMN: str
_PUPIL_FLAG_COLUMNS: frozenset[str]
type _AlignedArray = NDArray[np.float32] | NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class _CameraSource:
    name: str
    timestamps_file: str
    energy_file: str
    pupil_file: str | None

@dataclass(frozen=True, slots=True)
class _ClockSelection:
    camera: str
    timestamps_path: Path
    samples: int
    mean_rate: float

_CAMERA_SOURCES: tuple[_CameraSource, ...]

def assemble_video_dataset(video_data_path: Path, reference_time: NDArray[np.uint64]) -> pl.DataFrame: ...
def resolve_slowest_camera_clock(video_data_path: Path) -> NDArray[np.uint64]: ...
def resolve_reference_clock_samples(video_data_path: Path) -> int | None: ...
def count_camera_source_samples(video_data_path: Path) -> tuple[int, ...]: ...
def _select_reference_camera(video_data_path: Path) -> _ClockSelection | None: ...
def _interpolate_linear(
    frame_time: NDArray[np.uint64], values: NDArray[np.float32], reference_time: NDArray[np.uint64]
) -> NDArray[np.float32]: ...
def _read_frame_aligned(feather_path: Path, expected_rows: int, timestamps_filename: str) -> pl.DataFrame: ...
