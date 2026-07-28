"""Provides the system-agnostic dataset forging pipeline: dataset resolution and per-session assembly dispatch."""

from sollertia_shared_assets import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)

from .dataset import resolve_dataset
from .pipeline import (
    DEFINE_JOB_NAME,
    VERIFY_JOB_NAME,
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    load_multiday_plan,
    forging_tracker_path,
    run_forging_pipeline,
    discover_forging_jobs,
    resolve_multiday_plan,
    build_forging_universe,
    forging_job_prerequisites,
    materialize_multiday_plan,
)

__all__ = [
    "DEFINE_JOB_NAME",
    "FORGING_JOB_NAME",
    "MULTIDAY_DISCOVERY_JOB_NAME",
    "MULTIDAY_EXTRACTION_JOB_NAME",
    "VERIFY_JOB_NAME",
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "build_forging_universe",
    "discover_forging_jobs",
    "forging_job_prerequisites",
    "forging_tracker_path",
    "load_multiday_plan",
    "materialize_multiday_plan",
    "resolve_dataset",
    "resolve_multiday_plan",
    "run_forging_pipeline",
]
