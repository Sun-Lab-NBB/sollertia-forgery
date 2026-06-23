"""Provides the local batch-orchestration layer: the process-pool job-execution engine and the shared tracker and
feather helpers used to deploy batches of processing jobs on the local machine.
"""

from .orchestration import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    GenericPendingJob,
    JobExecutionState,
    prepare_tracker,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)

__all__ = [
    "RESERVED_CORES",
    "ActiveJob",
    "GenericPendingJob",
    "JobExecutionState",
    "PendingJob",
    "analyze_feather_file",
    "clean_output_subdirectory",
    "derive_tracker_status",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "prepare_tracker",
    "read_tracker_status",
]
