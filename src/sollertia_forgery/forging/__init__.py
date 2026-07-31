"""Provides the system-agnostic dataset forging pipeline."""

from sollertia_shared_assets import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)

from .state import (
    ANIMAL_SCOPE,
    SESSION_SCOPE,
    DATASET_JOB_SCOPES,
    DATASET_STATE_SCHEMA,
    DATASET_STATE_FILENAME,
    dataset_state_path,
    generate_dataset_state,
)
from .dataset import (
    DATASET_MARKER_FILENAME,
    resolve_dataset,
    discover_project_datasets,
)
from .pipeline import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    FORGING_JOB_CONCURRENCY_LIMITS,
    load_multiday_plan,
    forging_tracker_path,
    run_forging_pipeline,
    discover_forging_jobs,
    resolve_multiday_plan,
    build_forging_universe,
    define_forging_dataset,
    forging_job_prerequisites,
    materialize_multiday_plan,
)
from .admission import verify_session_admissibility

__all__ = [
    "ANIMAL_SCOPE",
    "DATASET_JOB_SCOPES",
    "DATASET_MARKER_FILENAME",
    "DATASET_STATE_FILENAME",
    "DATASET_STATE_SCHEMA",
    "FORGING_JOB_CONCURRENCY_LIMITS",
    "FORGING_JOB_NAME",
    "MULTIDAY_DISCOVERY_JOB_NAME",
    "MULTIDAY_EXTRACTION_JOB_NAME",
    "SESSION_SCOPE",
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "build_forging_universe",
    "dataset_state_path",
    "define_forging_dataset",
    "discover_forging_jobs",
    "discover_project_datasets",
    "forging_job_prerequisites",
    "forging_tracker_path",
    "generate_dataset_state",
    "load_multiday_plan",
    "materialize_multiday_plan",
    "resolve_dataset",
    "resolve_multiday_plan",
    "run_forging_pipeline",
    "verify_session_admissibility",
]
