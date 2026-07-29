"""Provides system-agnostic management pipelines: raw-data checksum verification and project manifest handling."""

from .jobs import PROJECT_JOBS_SCHEMA, project_jobs_path, write_project_jobs
from .checksum import (
    CHECKSUM_JOB_NAME,
    discover_checksum_jobs,
    checksum_job_prerequisites,
    run_checksum_processing_pipeline,
)
from .manifest import (
    MANIFEST_JOB_NAME,
    ProjectManifest,
    project_manifest_path,
    generate_project_manifest,
)

__all__ = [
    "CHECKSUM_JOB_NAME",
    "MANIFEST_JOB_NAME",
    "PROJECT_JOBS_SCHEMA",
    "ProjectManifest",
    "checksum_job_prerequisites",
    "discover_checksum_jobs",
    "generate_project_manifest",
    "project_jobs_path",
    "project_manifest_path",
    "run_checksum_processing_pipeline",
    "write_project_jobs",
]
