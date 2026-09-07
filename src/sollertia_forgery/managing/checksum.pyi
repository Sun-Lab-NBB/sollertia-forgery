from pathlib import Path

from sollertia_shared_assets import SessionData

from ..shared_assets import verify_openmp_runtime as verify_openmp_runtime

CHECKSUM_JOB_NAME: str

def run_checksum_processing_pipeline(
    session_path: Path, *, regenerate_checksum: bool = False, workers: int = -1, display_progress: bool = False
) -> None: ...
def discover_checksum_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def checksum_job_prerequisites(
    session: SessionData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _has_checksummable_data(raw_data_path: Path) -> bool: ...
