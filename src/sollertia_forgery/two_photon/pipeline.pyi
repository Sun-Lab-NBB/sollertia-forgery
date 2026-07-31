from pathlib import Path

from cindra import SingleRecordingConfiguration as SingleRecordingConfiguration
from sollertia_shared_assets import SessionData

from ..registries import (
    resolve_two_photon_data_locator as resolve_two_photon_data_locator,
    resolve_single_recording_configuration_resolver as resolve_single_recording_configuration_resolver,
)

_STAGE_DEFAULT_WORKERS: int
_ALL_PLANES: int
_CINDRA_CONFIGURATION_FILENAME: str

def run_two_photon_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    register: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = ...,
    workers: int = ...,
    display_progress: bool = False,
) -> None: ...
def prime_two_photon_recording(session_path: Path) -> None: ...
def discover_two_photon_jobs(
    session_path: Path,
) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def two_photon_job_prerequisites(
    session: SessionData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _configuration_path(session: SessionData) -> Path: ...
def _resolve_primed_plane_count(session: SessionData) -> int | None: ...
def _resolve_stage_request(workers: int) -> int | None: ...
def _resolve_configuration(
    session: SessionData, *, display_progress: bool, persist: bool
) -> tuple[SingleRecordingConfiguration, Path]: ...
