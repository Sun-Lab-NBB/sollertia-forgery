from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import SessionData

from ..registries import resolve_runtime_binding as resolve_runtime_binding
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    tracked_job as tracked_job,
    pinned_worker_threads as pinned_worker_threads,
)

RUNTIME_JOB_NAME: str

def run_runtime_processing_pipeline(
    session_path: Path, *, workers: int = -1, display_progress: bool = False
) -> None: ...
def discover_runtime_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def runtime_job_prerequisites(
    session: SessionData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _decode_archive(archive_path: Path, *, workers: int, display_progress: bool) -> pl.DataFrame: ...
def _decode_batches(
    archive_path: Path, onset_us: np.uint64, batches: list[list[str]], *, workers: int, display_progress: bool
) -> tuple[NDArray[np.uint64], list[bytes]]: ...
def _decode_batch(
    archive_path: Path, onset_us: np.uint64, keys: list[str]
) -> tuple[NDArray[np.uint64], list[bytes]]: ...
