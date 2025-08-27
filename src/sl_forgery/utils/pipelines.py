"""This module stores assets used by other library modules to streamline constructing, submitting, and monitoring data
processing pipelines that run on remote compute servers."""

from pathlib import Path

from sl_shared_assets import Server
from ataraxis_time.time_helpers import get_timestamp


def get_remote_job_work_directory(server: Server, job_name: str) -> Path:
    """Generates the working directory for the input job intended to be executed on the compute server managed by the
    input Server class.

    This worker function generates the current UTC timestamp, clips it down to minutes, and concatenates it to the
    job_name to construct the working directory name. It then resolves the path to that directory relative to the user
    working root on the remote server, creates the directory on the server, and returns the resolved path.
    """

    # Resolves working directory name using timestamp (accurate to minutes) and the job_name.
    timestamp = "-".join(get_timestamp().split("-")[:5])  # type: ignore
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Creates the working directory on the remote server.
    server.create_directory(remote_path=working_directory, parents=True)

    return working_directory
