"""Provides the local orchestration layer: the shared batch job-execution engine, its resource estimators, the
pipeline dispatch table, and the pipeline-identity enumeration.
"""

from .local import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    JobAllocation,
    GenericPendingJob,
    JobExecutionState,
    group_jobs_by_tracker,
    job_execution_manager,
    resolve_core_allocations,
)
from .dispatch import (
    BATCH_PIPELINES,
    PipelineDispatch,
    run_batch_job,
    resolve_dispatch,
    build_pending_job,
    prepare_pipeline_jobs,
    resolve_concurrency_limits,
)
from .pipelines import ProcessingPipelines
from .footprints import resolve_host_memory_mb

__all__ = [
    "BATCH_PIPELINES",
    "RESERVED_CORES",
    "ActiveJob",
    "GenericPendingJob",
    "JobAllocation",
    "JobExecutionState",
    "PendingJob",
    "PipelineDispatch",
    "ProcessingPipelines",
    "build_pending_job",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "prepare_pipeline_jobs",
    "resolve_concurrency_limits",
    "resolve_core_allocations",
    "resolve_dispatch",
    "resolve_host_memory_mb",
    "run_batch_job",
]
