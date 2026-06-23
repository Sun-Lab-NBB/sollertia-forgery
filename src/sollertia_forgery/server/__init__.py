"""Provides the remote compute-server transport: SSH/SLURM execution, non-interactive job assembly, project session
discovery, and server configuration.
"""

from .job import Job
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .discovery import discover_project_data, discover_project_sessions
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
    "Server",
    "ServerConfiguration",
    "create_server_configuration_file",
    "discover_project_data",
    "discover_project_sessions",
    "get_remote_job_work_directory",
    "get_server_configuration",
    "get_server_configuration_path",
]
