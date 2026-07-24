"""Provides the local orchestration layer: the in-process batch job-execution engine and the pipeline-identity
enumeration.
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
from .pipelines import ProcessingPipelines

__all__ = [
    "RESERVED_CORES",
    "ActiveJob",
    "ConcurrencyDescriptor",
    "GenericPendingJob",
    "JobExecutionState",
    "PendingJob",
    "ProcessingPipelines",
    "analyze_feather_file",
    "clean_output_subdirectory",
    "group_jobs_by_tracker",
    "job_execution_manager",
]
