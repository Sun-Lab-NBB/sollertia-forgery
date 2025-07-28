"""This module contains tools and bindings for all Sun lab data processing pipelines. The tools from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

from pathlib import Path

from sl_shared_assets import Job, Server, SessionTypes
from ataraxis_base_utilities import LogLevel, console
from ataraxis_time.time_helpers import get_timestamp
from sl_shared_assets.tools.project_management_tools import ProjectManifest

from ..utils import get_working_directory
from .management import fetch_remote_project_manifest


def submit_behavior_processing_job(
    project: str, session: str, server: Server, reprocess: bool, legacy: bool
) -> Job | None:
    """Generates and submits the behavior processing job for the specified session to the remote processing server.

    This function composes the behavior processing job and instructs the specified remote server to execute the job. It
    does not wait for the server to complete the job and instead returns the submitted Job object, which behaves similar
    to an asynchronous 'future' object.

    Notes:
        Depending on current server load and other jobs in the processing queue, the job may take a significant amount
        of time to execute. Use the job_complete() method of the Server class to periodically check on the state of
        the job.

    Args:
        project: The name of the project for which to submit the behavior processing job.
        session: The name of the session for which to submit the behavior processing job.
        server: An instance of the Server class that manages access to the remote server that stores the session data
            to process.
        reprocess: A boolean flag indicating whether to reprocess sessions that have already been processed.
        legacy: A boolean flag indicating whether to use the legacy behavior processing pipeline. This pipeline is
            designed exclusively for processing 'Tyche' project data and should not be used for any other project.

    Returns:
        The Job instance representing the behavior processing job running on the server if the job is submitted. None,
        if the job is not submitted to the server for any reason.
    """
    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Resolves the path to the locally stored project manifest file
    manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # If the local manifest file does not exist, fetches it from the remote server
    if not manifest_path.exists():
        fetch_remote_project_manifest(project)

    # Parses the target session data from the manifest file
    manifest = ProjectManifest(manifest_file=manifest_path)
    session_data = manifest.get_session_info(session=session)
    session_type = session_data["type"]
    animal = session_data["animal"]
    processed = bool(session_data["behavior"])

    # If the session type is not one of the supported types, skips processing the session
    if session_type not in {SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING, SessionTypes.MESOSCOPE_EXPERIMENT}:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is of type '{session_type},' which does not support this form of processing. "
            f"Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # If the session has already been processed and the function is not running in reprocessing mode, skips
    # processing the session.
    if processed and not reprocess:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session has already been processed with at least one behavior processing pipeline. To "
            f"enable reprocessing already processed sessions, call this command with the 'reprocess' flag set to True."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Otherwise, constructs the session processing job and submits it to the remote compute server

    # Resolves the working directory for the job, using static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    job_name = f"{session}_behavior_processing"
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Ensures that the working directory exists on the remote server
    server.create_directory(remote_path=working_directory)

    # Parses the paths to the shared Sun lab directories used to store raw and processed project data on the remote
    # server.
    project_storage_root = Path(server.raw_data_root).joinpath(project)
    project_working_path = Path(server.processed_data_root).joinpath(project)

    # Generates the remote job header. Currently, all behavior processing jobs use at most 7 CPU cores and do not
    # require more than 10GB of RAM due to using memory mapping.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="behavior",
        cpus_to_use=7,
        ram_gb=10,
        time_limit=60,
    )

    # Configures the job to use the sl-behavior package installed on the server to process session's behavior data.
    # Note, depending on the legacy flag, either submits the job in the legacy or contemporary processing mode. Legacy
    # processing mode is intended exclusively for processing Tyche data using modern Sun lab tools and should not be
    # used in most cases.
    if not legacy:
        job.add_command(f"sl-sl-process-behavior -sp {str(project_storage_root)} -pdr {str(project_working_path)}")
    else:
        job.add_command(f"sl-sl-process-behavior -sp {str(project_storage_root)} -pdr {str(project_working_path)} -l")

    # Submits the remote job to the server and returns the job object updated with job tracking details to the caller
    # for monitoring and handling the results once the job completes.
    job = server.submit_job(job)
    return job
