"""Provides the generic remote compute-server machinery: SSH/SLURM execution, non-interactive job assembly, the remote
pipeline state machine, and server configuration.
"""

from .job import Job
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility
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
    "ProcessingPipeline",
    "Server",
    "ServerConfiguration",
    "check_session_eligibility",
    "create_server_configuration_file",
    "execute_pipelines",
    "get_remote_job_work_directory",
    "get_server_configuration",
    "get_server_configuration_path",
]
