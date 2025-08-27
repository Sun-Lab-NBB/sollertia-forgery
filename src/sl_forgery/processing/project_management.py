"""This module provides tools for managing the Sun lab projects stored on the remote compute server. While most of these
tools are intended to be used by data processing pipelines and require 'service' server access, a small subset of tools
is also intended to be used by lab users (and requires 'user' server access)."""

from ataraxis_time import PrecisionTimer
from sl_shared_assets import Job, Server, TrackerFileNames, ProcessingTracker, get_working_directory
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..utils import get_remote_job_work_directory


def generate_remote_project_manifest(project: str, server: Server, keep_job_logs: bool = False) -> None:
    """Generates the manifest .feather file for the specified project stored on the remote compute server.

    This function allows generating manifest.feather files on the remote compute server outside the standard
    workflow (manually). Since this process requires 'service' access privileges, this function is not intended to be
    called directly by most lab users. As part of its runtime, this function also fetches (pulls) the generated
    manifest file to the local Sun lab working directory.

    Notes:
        All Sun lab 'service' pipelines automatically update the manifest file as part of their runtime, so it is
        typically unnecessary to use this function. The function is mostly used internally to test processing
        pipelines and data management strategies.

        The manifest file is created and stored inside the root raw data directory for the target project on the remote
        server.

    Args:
        project: The name of the project for which to generate and fetch the manifest file.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.
        server: An initialized Server instance used to communicate with the remote server. Note, the Server must be
            configured to use the service account server access credentials.

    Raises:
        FileNotFoundError: If the remote (server-side) project manifest generation job fails with an error and does not
            generate the manifest file.
    """

    console.echo(message=f"Constructing remote project manifest regeneration job...")

    local_working_directory = get_working_directory()

    # Resolves the job name and its remote working directory.
    job_name = f"{project}_manifest_generation"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Resolves the paths to the remote and local manifest generation tracker files
    remote_manifest_tracker_path = server.raw_data_root.joinpath(project, TrackerFileNames.MANIFEST)
    local_manifest_tracker_path = local_working_directory.joinpath(project, job_name, TrackerFileNames.MANIFEST)
    ensure_directory_exists(local_manifest_tracker_path)

    # Generates the remote job header
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpus_to_use=1,
        ram_gb=1,
        time_limit=20,
    )

    # Parses the path to the shared Sun lab directory used to store raw project data on the remote server.
    project_storage_root = server.raw_data_root.joinpath(project)

    # Configures the job to use the sl-shared-assets library installed on the server to generate the manifest file
    # inside the project's root raw data directory.
    job.add_command(f"sl-manage project -pp {project_storage_root} -pdr {server.processed_data_root} manifest")

    # If the function is configured to remove job logs after runtime, adds a command to delete job working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {working_directory}")

    # Submits the remote job to the server
    job = server.submit_job(job, verbose=False)

    # Waits for the server to complete the job
    delay_timer = PrecisionTimer("s")
    message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
    console.echo(message=message, level=LogLevel.INFO)
    while not server.job_complete(job=job):
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    # Verifies the outcome of the manifest generation job by pulling the remote tracker file to the local machine and
    # Checking the final status of the job.
    console.echo(message=f"Verifying the outcome of the manifest generation job...")
    server.pull_file(
        local_file_path=local_manifest_tracker_path,
        remote_file_path=remote_manifest_tracker_path,
    )
    tracker = ProcessingTracker(file_path=local_manifest_tracker_path)

    # If the job did not complete successfully, raises an error
    if not tracker.is_complete:
        message = (
            f"Manifest generation job: Failed. Check the processing logs stored on the remote compute server for "
            f"details about the error that caused the failure."
        )
        console.error(message=message, error=RuntimeError)

    # Otherwise, fetches the created manifest file to the local machine via the fetch function.
    console.echo(message=f"Project manifest file: Generated.", level=LogLevel.SUCCESS)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory. This ensures that the user has continued access to the most recent manifest file
    # for that project.
    fetch_remote_project_manifest(project=project, server=server)


def fetch_remote_project_manifest(project: str, server: Server) -> None:
    """Fetches (pulls) the existing project manifest .feather file for the specified project stored on the remote
    compute server to the local Sun lab working directory.

    This function serves as the entry-point for all data processing and dataset formation pipelines exposed by this
    library. It is used to pull the current snapshot of all available data for the specified project on the remote
    server to the local machine so that it can be used by other functions from this library. The pulled manifest file
    is stored under the directory named after the input project inside the local Sun lab working directory.

    Args:
        project: The name of the project for which to fetch the manifest file.
        server: An initialized Server instance used to communicate with the remote server.

    Raises:
        FileNotFoundError: If the manifest file does not exist on the server, indicating that the file has not been
            generated.
    """

    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Resolves the paths to the remote and local manifest files
    remote_manifest_path = server.raw_data_root.joinpath(project, TrackerFileNames.MANIFEST)
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
            f"file or use the 'sl-project update' CLI command with the '--regenerate-manifest (-rm)' flag to create "
            f"it manually (requires 'service' access privileges)."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory.
    console.echo(
        message=(
            f"Fetching the '{project}' project's manifest file from the remote compute server to the local machine..."
        )
    )
    server.pull_file(
        local_file_path=local_manifest_path,
        remote_file_path=remote_manifest_path,
    )
    console.echo(message=f"Most recent manifest file for the '{project}' project: Fetched.", level=LogLevel.SUCCESS)
