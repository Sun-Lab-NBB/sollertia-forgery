"""Provides assets for interfacing with remote compute servers to manage data and run processing pipelines."""

from sollertia_shared_assets import DatasetTrackers, ManagingTrackers, ProcessingTrackers

from .job import Job, JupyterJob
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .pipeline import ProcessingPipeline, check_session_eligibility, execute_pipelines
from .managing_interface import adopt_project, manage_project_data, resolve_project_manifest
from .processing_interface import process_project_data
from .forging_interface import forge_dataset
from .server_configuration import (
    ServerConfiguration,
    create_server_configuration_file,
    get_server_configuration,
    get_server_configuration_path,
)

__all__ = [
    "CommandResult",
    "DatasetTrackers",
    "Job",
    "JobStatus",
    "JupyterJob",
    "ManagingTrackers",
    "ProcessingPipeline",
    "ProcessingTrackers",
    "Server",
    "ServerConfiguration",
    "check_session_eligibility",
    "create_server_configuration_file",
    "execute_pipelines",
    "adopt_project",
    "forge_dataset",
    "get_remote_job_work_directory",
    "get_server_configuration",
    "get_server_configuration_path",
    "manage_project_data",
    "process_project_data",
    "resolve_project_manifest",
]
