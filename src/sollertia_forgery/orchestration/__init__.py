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
    group_jobs_by_tracker,
    job_execution_manager,
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
    "group_jobs_by_tracker",
    "job_execution_manager",
]
