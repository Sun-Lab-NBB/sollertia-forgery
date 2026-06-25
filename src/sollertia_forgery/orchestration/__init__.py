"""Provides the unified orchestration layer: the in-process batch job-execution engine, the remote SLURM pipeline
engine, and the processing-tracker job-registry alignment helper.
"""

from .local import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    GenericPendingJob,
    JobExecutionState,
    ConcurrencyDescriptor,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility
from .tracking import prepare_tracker

__all__ = [
    "RESERVED_CORES",
    "ActiveJob",
    "ConcurrencyDescriptor",
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
    "prepare_tracker",
    "read_tracker_status",
]
