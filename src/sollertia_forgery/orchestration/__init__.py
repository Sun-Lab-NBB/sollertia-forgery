"""Provides the orchestration layer: the shared preparation path, the execution hosts, the batch job-execution engine
and its resource estimators, the pipeline dispatch table, the job descriptors and the graph algorithms that order them,
the job plan caches, the remote scheduler backend and its submission ledger, the running-job reconciliation, the
prepared-batch registry with its closure, and the unit maintenance operations.
"""

from .graph import (
    GenericPendingJob,
    build_pending_job,
)
from .hosts import (
    LocalHost,
    RemoteHost,
    ExecutionHost,
)
from .local import (
    RESERVED_CORES,
    JobExecutionState,
    group_jobs_by_tracker,
    job_execution_manager,
    resolve_core_allocations,
)
from .ledger import (
    SubmissionLedger,
    read_ledger,
    resolve_batches,
    current_timestamp,
)
from .remote import (
    REMOTE_JOB_WALLTIME_MINUTES,
    submit_batch,
    connect_to_server,
    query_submissions,
    render_submission,
    cancel_submissions,
    sync_project_state,
    remote_batch_directory,
)
from .batches import (
    batch_directory,
    read_batch_outcome,
    resolve_batch_host,
    read_prepared_batches,
    record_prepared_batch,
)
from .closure import (
    close_batch,
    close_settled_batches,
)
from .dispatch import (
    BATCH_PIPELINES,
    run_batch_job,
    resolve_concurrency_limits,
    resolve_concurrency_reservations,
)
from .planning import (
    DATASET_UNIT,
    SESSION_UNIT,
    PROJECT_PLAN_SCHEMA,
    project_plan_path,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
)
from .reconcile import (
    LOCAL_HOST_LABEL,
    REMOTE_HOST_LABEL,
    reconcile_local_jobs,
    reconcile_remote_jobs,
)
from .footprints import resolve_host_memory_mb
from .maintenance import (
    reset_tracked_jobs,
    clean_pipeline_output,
)
from .preparation import prepare_batch, resolve_project_root

__all__ = [
    "BATCH_PIPELINES",
    "DATASET_UNIT",
    "LOCAL_HOST_LABEL",
    "PROJECT_PLAN_SCHEMA",
    "REMOTE_HOST_LABEL",
    "REMOTE_JOB_WALLTIME_MINUTES",
    "RESERVED_CORES",
    "SESSION_UNIT",
    "ExecutionHost",
    "GenericPendingJob",
    "JobExecutionState",
    "LocalHost",
    "RemoteHost",
    "SubmissionLedger",
    "batch_directory",
    "build_pending_job",
    "cancel_submissions",
    "clean_pipeline_output",
    "close_batch",
    "close_settled_batches",
    "connect_to_server",
    "current_timestamp",
    "generate_project_plan",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "prepare_batch",
    "project_plan_path",
    "query_submissions",
    "read_batch_outcome",
    "read_ledger",
    "read_prepared_batches",
    "reconcile_local_jobs",
    "reconcile_remote_jobs",
    "record_prepared_batch",
    "remote_batch_directory",
    "render_submission",
    "reset_tracked_jobs",
    "resolve_batch_host",
    "resolve_batches",
    "resolve_concurrency_limits",
    "resolve_concurrency_reservations",
    "resolve_core_allocations",
    "resolve_dataset_plan",
    "resolve_host_memory_mb",
    "resolve_project_root",
    "resolve_session_plan",
    "run_batch_job",
    "submit_batch",
    "sync_project_state",
]
