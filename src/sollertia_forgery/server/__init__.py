"""Provides the remote compute-server transport: SSH/SLURM execution, non-interactive job assembly, project session
discovery, and server configuration.
"""

from .job import Job
from .server import (
    TERMINAL_JOB_STATUSES,
    Server,
    JobStatus,
)
from .discovery import (
    discover_project_data,
    discover_project_markers,
)
from .server_configuration import (
    ServerConfiguration,
    remote_state_path,
    remote_state_directory,
    get_server_configuration,
    get_server_configuration_path,
    create_server_configuration_file,
)

__all__ = [
    "TERMINAL_JOB_STATUSES",
    "Job",
    "JobStatus",
    "Server",
    "ServerConfiguration",
    "create_server_configuration_file",
    "discover_project_data",
    "discover_project_markers",
    "get_server_configuration",
    "get_server_configuration_path",
    "remote_state_directory",
    "remote_state_path",
]
