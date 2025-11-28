"""This module provides the interface functions for resolving project manifest files from the remote compute server."""

from typing import TYPE_CHECKING

from ataraxis_time import PrecisionTimer, TimerPrecisions
from sl_shared_assets import ProcessingTracker, delete_directory, get_working_directory
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..server import Job, Server, JobStatus, ManagingTrackers, get_remote_job_work_directory

if TYPE_CHECKING:
    from pathlib import Path


def resolve_project_manifest(
    project: str,
    server: Server,
    *,
    generate: bool = False,
    keep_job_logs: bool = False,
) -> Path:
    """Resolves and fetches the project manifest .feather file for the specified project stored on the remote compute
    server.

    This function provides the entry-point for all interactions with the project's data stored on the remote compute
    server by generating and fetching the snapshot of the project's data state.

    Notes:
        If the manifest file does not exist on the remote server and the 'generate' argument is False, the function
        automatically generates the manifest before fetching it.

    Args:
        project: The name of the project for which to resolve the manifest file.
        server: The Server instance used to communicate with the remote compute server.
        generate: Determines whether to regenerate the manifest file on the remote server. If True, the manifest is
            regenerated regardless of whether it already exists. If False, the existing manifest is fetched (and
            auto-generated if missing).
        keep_job_logs: Determines whether to keep completed manifest generation job logs on the server. If the job
            fails, logs are always kept regardless of this parameter.

    Returns:
        The path to the fetched project's manifest .feather file.

    Raises:
        RuntimeError: If the remote manifest generation job fails.
    """
    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Resolves the local path where the manifest file will be stored.
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")
    ensure_directory_exists(local_manifest_path)

    # Resolves the path to the remote manifest file.
    remote_manifest_path = server.shared_storage_root.joinpath(project, f"{project}_manifest.feather")

    # Determines whether to generate the manifest.
    should_generate = generate or not server.exists(remote_path=remote_manifest_path)

    if should_generate:
        _generate_remote_manifest(
            project=project,
            server=server,
            keep_job_logs=keep_job_logs,
            local_working_directory=local_working_directory,
        )

    # Fetches the manifest file to the local machine.
    console.echo(
        message=f"Fetching the '{project}' project's manifest file from the remote server to the local machine..."
    )
    server.pull(
        local_path=local_manifest_path,
        remote_path=remote_manifest_path,
    )
    console.echo(message=f"Manifest file for the '{project}' project: Fetched.", level=LogLevel.SUCCESS)

    return local_manifest_path


def _generate_remote_manifest(
    project: str,
    server: Server,
    local_working_directory: Path,
    *,
    keep_job_logs: bool,
) -> None:
    """Generates the manifest file on the remote compute server and fetches it to the host-machine.

    Args:
        project: The name of the project for which to generate the manifest file.
        server: The Server instance used to communicate with the remote server.
        keep_job_logs: Determines whether to keep completed manifest generation job logs on the server.
        local_working_directory: The path to the local Sun lab working directory.

    Raises:
        RuntimeError: If the manifest generation job fails.
    """
    console.echo(message=f"Generating the manifest file for the '{project}' project on the remote server...")

    # Resolves the job name and its remote working directory.
    job_name = f"{project}_manifest_generation"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Resolves the paths to the remote and local manifest generation tracker files.
    remote_manifest_tracker_path = server.shared_storage_root.joinpath(project, ManagingTrackers.MANIFEST)
    local_manifest_tracker_path = local_working_directory.joinpath(project, job_name, ManagingTrackers.MANIFEST)
    ensure_directory_exists(local_manifest_tracker_path)

    # Generates the remote job header.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpu_threads=1,
        ram=1,
        time=20,
    )

    # Resolves the path to the project's directory on the remote compute server.
    project_storage_root = server.shared_storage_root.joinpath(project)

    # Configures the job to call the appropriate CLI command.
    job.add_command(f"sl-process manifest -pp {project_storage_root}")

    # If configured to remove job logs after runtime, adds a command to delete the job's working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {working_directory}")

    # Submits the job to the server.
    job = server.submit_job(job=job, verbose=False)

    # Waits for the server to complete the job.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
    console.echo(message=message, level=LogLevel.INFO)
    while server.get_job_status(slurm_job_id=int(job.job_id)) in (JobStatus.PENDING, JobStatus.RUNNING):
        delay_timer.delay(delay=5, allow_sleep=True, block=False)

    # Verifies the outcome of the manifest generation job.
    console.echo(message="Verifying the outcome of the manifest generation job...")
    server.pull(
        local_path=local_manifest_tracker_path,
        remote_path=remote_manifest_tracker_path,
    )
    tracker = ProcessingTracker(file_path=local_manifest_tracker_path)

    # If the job did not complete successfully, raises an error.
    if not tracker.complete:
        message = (
            "Manifest generation job: Failed. Check the processing logs stored on the remote compute server for "
            "details about the error that caused the failure."
        )
        console.error(message=message, error=RuntimeError)
    else:
        # If the job ran successfully, removes the local working directory.
        delete_directory(local_manifest_tracker_path.parent)

    console.echo(message=f"Manifest file for the '{project}' project: Generated.", level=LogLevel.SUCCESS)
