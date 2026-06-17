"""Provides assets for interfacing with remote compute servers to manage data and run processing pipelines."""

from sollertia_shared_assets import ProcessingTrackers

from .job import Job, JupyterJob
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility
from .forging_interface import forge_dataset
from .managing_interface import (
    transfer_data,
    manage_project_data,
    discover_project_data,
    resolve_project_manifest,
    discover_project_sessions,
)
from .processing_interface import process_project_data
from .server_configuration import (
    ServerConfiguration,
    get_server_configuration,
    get_server_configuration_path,
    create_server_configuration_file,
)

__all__ = [
    "CommandResult",
    "Job",
    "JobStatus",
    "JupyterJob",
    "ProcessingPipeline",
    "ProcessingTrackers",
    "Server",
    "ServerConfiguration",
    "check_session_eligibility",
    "create_server_configuration_file",
    "discover_project_data",
    "discover_project_sessions",
    "execute_pipelines",
    "forge_dataset",
    "get_remote_job_work_directory",
    "get_server_configuration",
    "get_server_configuration_path",
    "manage_project_data",
    "process_project_data",
    "resolve_project_manifest",
    "transfer_data",
]
