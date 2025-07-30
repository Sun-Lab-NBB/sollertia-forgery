"""This module provides tools for managing the Sun lab projects stored on the remote compute server. While most of these
tools are intended to be used by data processing pipelines and require 'service' server access, a small subset of tools
is also intended to be used by lab users (and requires 'user' server access)."""

from pathlib import Path

from ataraxis_time import PrecisionTimer
from sl_shared_assets import Job, Server
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp

from ..utils import get_working_directory, get_credentials_file_path


def generate_remote_project_manifest(project: str, keep_job_logs: bool = False) -> None:
    """Generates the manifest .feather file for the specified project stored on the remote compute server.

    This function allows generating the manifest.feather files on the remote compute server outside the standard
    workflow (manually). Since this process requires 'service' access privileges, this function is not intended to be
    called directly by most lab users.

    Notes:
        All Sun lab 'service' pipelines automatically update the manifest file as part of their runtime, so it is
        typically unnecessary to use this function. The function is mostly used internally to test various lab pipelines
        and data management strategies.

        As part of its runtime, this function also fetches (pulls) the generated manifest file to the local Sun lab
        working directory. Therefore, this function also includes the functionality of the
        fetch_remote_project_manifest() function.

        The manifest file is created and stored inside the root raw data directory for the target project on the remote
        server.

    Args:
        project: The name of the project for which to generate and fetch the manifest file.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.

    Raises:
        FileNotFoundError: If the remote (server-side) project manifest generation job fails with an error and does not
            generate the manifest file.
    """

    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Resolves the path to the server access credentials file. Since manifest generation requires access to .YAML
    # processing trackers, this function requires service access privileges.
    credentials_path = get_credentials_file_path(require_service=True)

    # Uses the server access credentials file to initialize the SHH connection to the remote server.
    server = Server(credentials_path=credentials_path)

    # Resolves the working directory for the remote job, using a static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    job_name = f"{project}_manifest_generation"
    server_working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Ensures that the working directory exists on the remote server
    server.create_directory(remote_path=server_working_directory)

    # Parses the paths to the shared Sun lab directories used to store raw and processed project data on the remote
    # server.
    project_storage_root = server.raw_data_root.joinpath(project)

    # Generates the remote job header
    job = Job(
        job_name=job_name,
        output_log=server_working_directory.joinpath(f"output.txt"),
        error_log=server_working_directory.joinpath(f"errors.txt"),
        working_directory=server_working_directory,
        conda_environment="manage",
        cpus_to_use=1,
        ram_gb=10,
        time_limit=10,
    )

    # Configures the job to use the sl-shared-assets package installed on the server to generate the manifest file
    # inside the project's root raw data directory
    job.add_command(
        f"sl-project-manifest -pp {str(project_storage_root)} -pdr {str(server.processed_data_root)} "
        f"-od {str(project_storage_root)}"
    )

    # If the function is configured to remove job logs after runtime, adds a command to delete job working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {str(server_working_directory)}")

    # Submits the remote job to the server
    job = server.submit_job(job)

    # Waits for the server to complete the job
    delay_timer = PrecisionTimer("s")
    while not server.job_complete(job=job):
        message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
        console.echo(message=message, level=LogLevel.INFO)
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    # Resolves the path to the remote and local manifest files
    remote_manifest_path = project_storage_root.joinpath(f"{project}_manifest.feather")
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # Ensures that the project-specific folder exists under the local working directory
    ensure_directory_exists(local_manifest_path)

    # Verifies that the job ran as expected. For this, ensures that the remote manifest file exists (was created).
    if not server.exists(remote_path=remote_manifest_path):
        # Closes the SSH connection
        server.close()

        message = (
            f"Unable to locate the manifest file for '{project}' project one the remote server. This indicates that "
            f"the remote manifest creation job ran into an error and did not generate the file. Check the error logs "
            f"for the {job_name} job stored in the {server_working_directory} server directory for more details about "
            f"the error."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory. This ensures that the user has continued access to the most recent manifest file
    # for that project.
    console.echo(message=f"Fetching the generated manifest file from the remote compute server...")
    server.pull_file(
        local_file_path=local_manifest_path,
        remote_file_path=remote_manifest_path,
    )

    # Closes the SSH connection
    server.close()


def fetch_remote_project_manifest(project: str) -> None:
    """Fetches (pulls) the existing project manifest .feather file for the specified project stored on the remote
    compute server to the local Sun lab working directory.

    This function serves as the entry-point for all data processing and dataset formation pipelines exposed by this
    library. It is used to pull the current snapshot of all available data for the specified project on the remote
    server to the local machine, so that it can be used by other functions from this library. The pulled manifest file
    is stored under the directory named after the input project inside the local Sun lab working directory.

    Args:
        project: The name of the project for which to fetch the manifest file.

    Raises:
        FileNotFoundError: If the manifest file does not exist on the server, indicating that the file has not been
            generated.
    """

    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Unlike generating the manifest file, pulling an existing manifest file does not require service access privileges.
    credentials_path = get_credentials_file_path(require_service=False)

    # Uses the server access credentials file to initialize the SHH connection to the remote server.
    server = Server(credentials_path=credentials_path)

    # Resolves the path to the remote and local manifest files
    remote_manifest_path = server.raw_data_root.joinpath(project, f"{project}_manifest.feather")
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # Ensures that the project-specific folder exists under the local working directory
    ensure_directory_exists(local_manifest_path)

    # Verifies that the job ran as expected. For this, ensures that the remote manifest file exists (was created).
    # Otherwise, aborts with an error.
    if not server.exists(remote_path=remote_manifest_path):
        # Closes the SSH connection
        server.close()

        message = (
            f"Unable to fetch the manifest file for '{project}' project from the remote server, as the target project "
            f"does not have a manifest file. Either wait for one of the service pipelines to generate the manifest "
            f"file or use the 'generate_project_manifest' CLI command to create it manually (requires 'service' access "
            f"privileges)."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory.
    console.echo(message=f"Fetching the manifest file from the remote compute server...")
    server.pull_file(
        local_file_path=local_manifest_path,
        remote_file_path=remote_manifest_path,
    )

    # Closes the SSH connection
    server.close()
