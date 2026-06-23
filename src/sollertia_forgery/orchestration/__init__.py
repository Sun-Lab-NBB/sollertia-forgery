"""Provides the unified orchestration layer: the in-process batch job-execution engine, the remote SLURM pipeline
engine, and the system-agnostic remote management orchestrators for project manifest and checksum resolution.
"""

from .local import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    GenericPendingJob,
    JobExecutionState,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)
from .managing import manage_project_data, resolve_project_manifest
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility

__all__ = [
    "RESERVED_CORES",
    "ActiveJob",
    "GenericPendingJob",
    "JobExecutionState",
    "PendingJob",
    "ProcessingPipeline",
    "analyze_feather_file",
    "check_session_eligibility",
    "clean_output_subdirectory",
    "derive_tracker_status",
    "execute_pipelines",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "manage_project_data",
    "read_tracker_status",
    "resolve_project_manifest",
]
