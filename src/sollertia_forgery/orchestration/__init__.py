"""Provides the unified orchestration layer: the in-process batch job-execution engine and the remote SLURM pipeline
engine.
"""

from .local import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    GenericPendingJob,
    JobExecutionState,
    ConcurrencyDescriptor,
    analyze_feather_file,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility

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
    "execute_pipelines",
    "group_jobs_by_tracker",
    "job_execution_manager",
]
