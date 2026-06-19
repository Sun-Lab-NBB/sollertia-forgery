"""Provides system-agnostic assets shared across every acquisition system supported by the library: the analysis
dataset data hierarchy, project management (manifest generation and checksum verification), the project manifest
viewer, the camera-timestamp extraction pipeline, batch-orchestration primitives, and the generic remote
compute-server machinery (SSH/SLURM execution and remote project management).

Notes:
    This package must never import from a system-specific package (such as ``mesoscope_vr``). The dependency
    direction is strictly system-specific -> cross_system.
"""

from .job import Job
from .video import VIDEO_JOB_NAME, run_video_processing_pipeline
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .checksum import CHECKSUM_JOB_NAME, resolve_checksum
from .manifest import MANIFEST_JOB_NAME, generate_project_manifest
from .pipelines import ProcessingPipelines
from .utilities import delay_timer, delay_terminal
from .dataset_data import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)
from .orchestration import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    JobExecutionState,
    prepare_tracker,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
)
from .remote_pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility
from .project_manifest import ProjectManifest
from .remote_management import (
    manage_project_data,
    discover_project_data,
    resolve_project_manifest,
    discover_project_sessions,
)
from .server_configuration import (
    ServerConfiguration,
    get_server_configuration,
    get_server_configuration_path,
    create_server_configuration_file,
)

__all__ = [
    "CHECKSUM_JOB_NAME",
    "MANIFEST_JOB_NAME",
    "RESERVED_CORES",
    "VIDEO_JOB_NAME",
    "ActiveJob",
    "CommandResult",
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "Job",
    "JobExecutionState",
    "JobStatus",
    "PendingJob",
    "ProcessingPipeline",
    "ProcessingPipelines",
    "ProjectManifest",
    "Server",
    "ServerConfiguration",
    "analyze_feather_file",
    "check_session_eligibility",
    "clean_output_subdirectory",
    "create_server_configuration_file",
    "delay_terminal",
    "delay_timer",
    "derive_tracker_status",
    "discover_project_data",
    "discover_project_sessions",
    "execute_pipelines",
    "generate_project_manifest",
    "get_remote_job_work_directory",
    "get_server_configuration",
    "get_server_configuration_path",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "manage_project_data",
    "prepare_tracker",
    "read_tracker_status",
    "resolve_checksum",
    "resolve_project_manifest",
    "run_video_processing_pipeline",
]
