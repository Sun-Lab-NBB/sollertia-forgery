"""Provides the system-agnostic dataset forging pipeline."""

from sollertia_shared_assets import (
    DatasetData,
    DatasetSession,
)

from .state import (
    DATASET_STATE_FILENAME,
    dataset_state_path,
    generate_dataset_state,
)
from .dataset import discover_project_datasets
from .pipeline import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    FORGING_JOB_CONCURRENCY_LIMITS,
    forging_tracker_path,
    run_forging_pipeline,
    discover_forging_jobs,
    define_forging_dataset,
    forging_job_prerequisites,
    forging_cross_recording_paths,
)

__all__ = [
    "DATASET_STATE_FILENAME",
    "FORGING_JOB_CONCURRENCY_LIMITS",
    "FORGING_JOB_NAME",
    "MULTIDAY_DISCOVERY_JOB_NAME",
    "MULTIDAY_EXTRACTION_JOB_NAME",
    "DatasetData",
    "DatasetSession",
    "dataset_state_path",
    "define_forging_dataset",
    "discover_forging_jobs",
    "discover_project_datasets",
    "forging_cross_recording_paths",
    "forging_job_prerequisites",
    "forging_tracker_path",
    "generate_dataset_state",
    "run_forging_pipeline",
]
