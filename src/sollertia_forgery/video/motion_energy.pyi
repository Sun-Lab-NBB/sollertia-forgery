from enum import StrEnum
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray as NDArray

from ..shared_assets import pinned_worker_threads as pinned_worker_threads

MOTION_ENERGY_SUFFIX: str
_VIDEO_SUFFIX: str
_SPATIAL_BIN_SIZE: int
_MINIMUM_CHUNK_FRAMES: int
_SINGLE_PLANE_DIMENSIONS: int
_MONOCHROME_PLANE_INDEX: int

class MotionEnergyColumn(StrEnum):
    MOTION_ENERGY = "motion_energy"
    FRAME_LUMINANCE = "frame_luminance"

def resolve_camera_video(camera_data_directory: Path, session_name: str, camera_name: str) -> Path | None: ...
def compute_camera_motion_energy(
    video_path: Path,
    output_path: Path,
    *,
    workers: int = -1,
    executor: ProcessPoolExecutor | None = None,
    display_progress: bool = False,
) -> None: ...
def _read_frame_count(video_path: Path) -> int: ...
def _plan_chunks(frame_count: int, workers: int) -> list[tuple[int, int]]: ...
def _submit_chunks(
    executor: ProcessPoolExecutor, video_path: Path, chunks: list[tuple[int, int]], *, display_progress: bool
) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]: ...
def _join_chunks(
    results: list[tuple[NDArray[np.float32], NDArray[np.float32]]], chunks: list[tuple[int, int]], video_path: Path
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def _energy_chunk(
    video_path: str, start_frame: int, frame_count: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def _bin_frame(frame: NDArray[np.uint8]) -> NDArray[np.float32]: ...
