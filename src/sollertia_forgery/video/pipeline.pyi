from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker

from ..registries import resolve_video_tracking as resolve_video_tracking
from .motion_energy import (
    MOTION_ENERGY_SUFFIX as MOTION_ENERGY_SUFFIX,
    resolve_camera_video as resolve_camera_video,
    compute_camera_motion_energy as compute_camera_motion_energy,
)
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    tracked_job as tracked_job,
    pinned_worker_threads as pinned_worker_threads,
)

RENAME_JOB_NAME: str
TRACKING_JOB_NAME: str
ENERGY_JOB_NAME: str
_RAW_CAMERA_LOG_PART_COUNT: int
_PARSED_CAMERA_PREFIX: str
_CAMERA_TIMESTAMP_SUFFIX: str

def run_video_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    timestamp: bool = False,
    track: bool = False,
    energy: bool = False,
    target_camera: int = -1,
    workers: int = -1,
    display_progress: bool = False,
) -> None: ...
def discover_video_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def video_job_prerequisites(
    session: SessionData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _resolve_camera_names(data_directory: Path) -> dict[int, str]: ...
def _find_camera_logs(data_directory: Path) -> list[Path]: ...
def _extract_camera_source_id(log_path: Path) -> int: ...
def _dispatch_job(
    job_name: str,
    specifier: str,
    session: SessionData,
    log_paths: dict[int, Path],
    camera_names: dict[int, str],
    video_data_directory: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> None: ...
def _link_parsed_timestamps(
    video_data_directory: Path, camera_names: dict[int, str], job_id: str, tracker: ProcessingTracker
) -> None: ...
def _run_pose_tracking(
    session: SessionData, video_data_directory: Path, job_id: str, tracker: ProcessingTracker
) -> None: ...
def _run_motion_energy(
    session: SessionData,
    camera_name: str,
    video_data_directory: Path,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> None: ...
