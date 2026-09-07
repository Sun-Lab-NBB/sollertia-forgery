from pathlib import Path

import polars as pl
from sollertia_shared_assets import DatasetData as DatasetData

from .pipeline import (
    FORGING_JOB_NAME as FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME as MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME as MULTIDAY_EXTRACTION_JOB_NAME,
    forging_tracker_path as forging_tracker_path,
)
from ..shared_assets import natural_sort as natural_sort

DATASET_STATE_FILENAME: str
_LOCK_TIMEOUT_SECONDS: float
_ANIMAL_SCOPE: str
_SESSION_SCOPE: str
_DATASET_JOB_SCOPES: dict[str, str]
_DATASET_STATE_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType]

def dataset_state_path(dataset: DatasetData) -> Path: ...
def generate_dataset_state(dataset: DatasetData, *, display_progress: bool = False) -> Path: ...
def _build_job_rows(dataset: DatasetData) -> list[dict[str, str | int | None]]: ...
