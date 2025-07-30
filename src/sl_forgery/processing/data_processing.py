"""This module contains tools and bindings for all Sun lab data processing pipelines. The tools from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

import shutil as sh
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

from sl_shared_assets import Job, Server, SessionTypes, ProjectManifest, ProcessingTracker
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp
from ataraxis_time import PrecisionTimer

from ..utils import get_working_directory, get_credentials_file_path
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest


@dataclass()
class ProcessingJob:
    """Stores the information about a processing job running on the remote server.

    This class instance is used to aggregate information about running data processing pipelines to support the
    asynchronous job result collection. It is processing-pipeline agnostic and works for all currently
    supported Sun lab processing pipelines.
    """

    job: Job
    """The Job object that stores the metadata for the job tracked by this class instance."""
    server: Server
    """The Server object that maintains bidirectional communication with the remote server running the job."""
    remote_tracker_path: Path
    """The path to the job tracker .yaml file stored on the remote server running the job."""
    job_working_directory: Path
    """The path to the job's working directory on the remote server."""
    local_tracker_path: Path
    """The path to the local job tracker .yaml file. The remote file is pulled to this location as part of the job 
    outcome verification process."""
    session: str
    """The ID of the session whose data is being processed by the tracked job."""
    animal: str
    """The ID of the animal whose data is being processed by the tracked job."""
    project: str
    """The name of the project whose data is being processed by the tracked job."""
    keep_job_logs: bool = False
    """Determines whether to keep the logs for successfully completed jobs on the server or (default) to remove them 
    after runtime."""


def submit_behavior_processing_job(
    project: str,
    session: str,
    server: Server,
    reprocess: bool = False,
    legacy: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingJob | None:
    """Generates and submits the behavior processing job for the specified session to the remote processing server.

    This function composes the behavior processing job and instructs the specified remote server to execute the job. It
    does not wait for the server to complete the job and instead returns the submitted Job object, which behaves similar
    to an asynchronous 'future' object.

    Notes:
        Depending on the current server load and other jobs in the processing queue, the job may take a significant
        amount of time to execute. Use the job_complete() method of the Server class to periodically check on the state
        of the job.

    Args:
        project: The name of the project for which to submit the behavior processing job.
        session: The name of the session for which to submit the behavior processing job.
        server: An instance of the Server class that manages access to the remote server that stores the session data
            to process.
        reprocess: A boolean flag indicating whether to reprocess sessions that have already been processed.
        legacy: A boolean flag indicating whether to use the legacy behavior processing pipeline. This pipeline is
            designed exclusively for processing 'Tyche' project data and should not be used for any other project.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.

    Returns:
        The ProcessingJob instance representing the processing job running on the server if the job is submitted. None,
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
    session_type = session_data["type"][0]
    animal = str(session_data["animal"][0])
    processed = bool(session_data["behavior"][0])
    complete = bool(session_data["complete"][0])
    dataset = bool(session_data["dataset"][0])

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

    # Prevents processing incomplete sessions
    if not complete:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is marked as 'incomplete', which excludes it from automated data processing. "
            f"To enable processing, manually mark it as 'complete' by creating the 'telomere.bin' marker file in "
            f"the session's raw_data directory on the remote server."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    if dataset:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is currently in the 'dataset integration' mode and cannot be processed. To convert "
            f"the session back to the 'data processing' mode, call the 'sl-resolve-session-mode' CLI command from this "
            f"library with the appropriate runtime flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Otherwise, constructs the session processing job and submits it to the remote compute server

    # Resolves the working directory for the job, using a static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    job_name = f"{session}_behavior_processing"
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Ensures that the working directory exists on the remote server
    server.create_directory(remote_path=working_directory)

    # Parses the paths to the shared Sun lab directories used to store raw and processed session data on the remote
    # server.
    remote_session_path = Path(server.raw_data_root).joinpath(project, animal, session)
    processed_data_root = Path(server.processed_data_root)

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
        job.add_command(f"sl-process-behavior -sp {str(remote_session_path)} -pdr {str(processed_data_root)} -um")
    else:
        job.add_command(f"sl-process-behavior -sp {str(remote_session_path)} -pdr {str(processed_data_root)} -l -um")

    # Submits the remote job to the server and returns the job object updated with job tracking details to the caller
    # for monitoring and handling the results once the job completes.
    job = server.submit_job(job)

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.processed_data_root).joinpath(
        project, animal, session, "processed_data", "behavior_processing_tracker.yaml"
    )
    local_tracker_path = local_working_directory.joinpath(project, "temp", "behavior_tracker.yaml")

    # Packages Job data into a ProcessingJob object and returns it to the caller.
    job_data = ProcessingJob(
        job=job,
        server=server,
        remote_tracker_path=remote_tracker_path,
        job_working_directory=working_directory,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )

    return job_data


def verify_processing_job_outcome(job_data: ProcessingJob) -> bool | None:
    """Checks if the input processing job running on the remote compute server has completed successfully.

    If the job is still running, returns None. If the job has completed successfully, returns True. If the job has
    failed, returns False. This function also prints success and failure messages to the console to notify the user
    about the job evaluation status.
    """

    # Unpacks the Job and Server objects from the ProcessingJob object.
    job = job_data.job
    server = job_data.server

    # If the server has not yet completed the job, returns None to indicate that the job is still running.
    if not server.job_complete(job=job):
        return None

    # Otherwise, checks the outcome of the job by evaluating the processing status stored inside the processing
    # tracker file. To do so, first pulls the tracker file from the remote server to the local machine.
    ensure_directory_exists(job_data.local_tracker_path)  # Ensures that the local temporary directory exists
    server.pull_file(remote_file_path=job_data.remote_tracker_path, local_file_path=job_data.local_tracker_path)
    tracker = ProcessingTracker(job_data.local_tracker_path)

    # The tracker should indicate that the job is 'complete' if runtime finishes successfully.
    if not tracker.is_complete:
        # Removes the temporary directory where the local copy of the tracker file is stored.
        sh.rmtree(job_data.local_tracker_path.parent)

        return False  # Job has failed

    # If the job was configured to remove logs after completing successfully, removes the job logs from the remote
    # server.
    if not job_data.keep_job_logs:
        server.remove(remote_path=job_data.job_working_directory, recursive=True, is_dir=True)

    # Removes the temporary directory where the local copy of the tracker file is stored.
    sh.rmtree(job_data.local_tracker_path.parent)

    return True  # Job completed successfully


def process_behavior_data(
    project: str,
    update_manifest: bool = True,
    sessions: list[str] | tuple[str] | None = None,
    reprocess: bool = False,
    legacy: bool = False,
    keep_job_logs: bool = False,
) -> None:
    # Establishes SSH connection to the processing server.
    credentials = get_credentials_file_path(require_service=True)
    server = Server(credentials_path=credentials)

    # Depending on configuration, updates the project manifest file stored on the remote server and fetches it to the
    # local machine.
    if update_manifest:
        generate_remote_project_manifest(project=project)
    else:
        fetch_remote_project_manifest(project=project)

    # Loads the fetched manifest file into memory as a ProjectManifest instance.
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")
    manifest = ProjectManifest(manifest_file=manifest_path)

    # If the user did not specify a list of sessions to process, processes all available sessions for that project.
    # Ensures sessions are stored as a tuple of strings for efficiency.
    if sessions is None:
        sessions = manifest.sessions
    else:
        sessions = tuple(sessions)

    # Attempts to generate and submit a remote processing job for each session
    jobs = []
    for session in sessions:
        job = submit_behavior_processing_job(
            server=server,
            project=project,
            session=session,
            reprocess=reprocess,
            legacy=legacy,
            keep_job_logs=keep_job_logs,
        )

        # Since the job submission function also verifies whether the job should be submitted, not all jobs are
        # expected to actually be sent to the server. If the returned object is None, the job was not submitted and,
        # hence, does not require tracking.
        if job is not None:
            jobs.append(job)

    # If no jobs were submitted, aborts the runtime early
    if len(jobs) == 0:
        message = (
            f"All available sessions for project '{project}' have been excluded from behavior processing. See the "
            f"messages above for details on exclusion criteria applied to each session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Creates a progress bar to track the progress of the behavior processing jobs.
    with tqdm(total=len(jobs), desc="Processing jobs", unit="job") as pbar:

        # Initializes a timer to delay repeated job status checks
        delay_timer = PrecisionTimer("s")

        remaining_jobs = jobs.copy()  # Creates a copy to track remaining jobs
        completed_count = 0

        # Runs until all jobs are completed (successfully or not)
        while remaining_jobs:

            # Checks the status of each remaining job
            jobs_to_remove = []
            for i, job in enumerate(remaining_jobs):

                result = verify_processing_job_outcome(job_data=job)

                # If the job verification function returned True or False (completed), marks the job for removal
                if result is not None:

                    jobs_to_remove.append(i)
                    completed_count += 1
                    pbar.update(1)  # Updates progress bar

            # Removes completed jobs from tracking (in reverse order to maintain indices)
            for i in reversed(jobs_to_remove):
                remaining_jobs.pop(i)

            # If jobs are still running, waits before checking again
            if remaining_jobs:
                delay_timer.delay_noblock(delay=5, allow_sleep=True)

        # Sets the final progress bar status.
        pbar.set_postfix_str(f"All {len(jobs)} jobs completed")


    # message = (
    #     f"The remote processing job with id {job.job_id} and name '{job.job_name}' for the session "
    #     f"'{job_data.session}' performed by animal '{job_data.animal}' for '{job_data.project}' project did not "
    #     f"run successfully. Check the job error logs for the specific details about the cause of the failure."
    # )
    # console.echo(message=message, level=LogLevel.ERROR)
