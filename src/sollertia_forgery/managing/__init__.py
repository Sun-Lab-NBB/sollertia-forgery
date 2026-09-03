"""Provides system-agnostic management pipelines: raw-data checksum verification and project manifest handling."""

from .jobs import project_jobs_path
from .checksum import (
    CHECKSUM_JOB_NAME,
    discover_checksum_jobs,
    checksum_job_prerequisites,
    run_checksum_processing_pipeline,
)
from .manifest import (
    MANIFEST_AXES,
    MANIFEST_JOB_NAME,
    MANIFEST_SEMI_FIELDS,
    ProjectManifest,
    project_manifest_path,
    generate_project_manifest,
)

__all__ = [
    "CHECKSUM_JOB_NAME",
    "MANIFEST_AXES",
    "MANIFEST_JOB_NAME",
    "MANIFEST_SEMI_FIELDS",
    "ProjectManifest",
    "checksum_job_prerequisites",
    "discover_checksum_jobs",
    "generate_project_manifest",
    "project_jobs_path",
    "project_manifest_path",
    "run_checksum_processing_pipeline",
]
