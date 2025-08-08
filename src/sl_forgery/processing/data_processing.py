"""This module contains tools and bindings for all Sun lab data processing pipelines. The tools from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

from enum import IntEnum, StrEnum
import shutil as sh
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    Job,
    Server,
    SessionTypes,
    ProjectManifest,
    TrackerFileNames,
    generate_manager_id,
    get_processing_tracker,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp

from ..utils import get_working_directory, get_credentials_file_path, get_remote_filesystem_paths
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest


class _ProcessingStatus(IntEnum):
    """Maps integer-based remote processing pipeline status codes to human-readable names.

    This enumeration is used internally to standardize processing progress tracking across all processing pipelines
    supported by this module.

    Notes:
        Each processing pipeline may be associated with multiple sequential or concurrent jobs running on the server.
        Therefore, these status codes track the state of the pipeline as a whole, instead of tracking the state of each
        job.
    """

    RUNNING = 0
    """The pipeline is currently running on the remote server. It may be executed (in progress) or waiting for resources 
    to become available (queued)."""
    SUCCEEDED = 1
    """The server has successfully executed the processing pipeline."""
    FAILED = 2
    """The server has failed to complete the pipeline as one of its constituent jobs has encountered a runtime error."""


class _Suite2PStages(IntEnum):
    """Maps integer-based codes for single-day suite2p processing pipeline stages to human-readable names.

    This enumeration is used internally to run remote suite2p single-day processing pipelines by issuing multiple
    sequential stage-based jobs to the server.
    """

    BINARIZE = 1
    """Stage 1: Converts source data files into multiple suite2p plane-specific binary files."""
    PROCESS = 2
    """Stage 2: Processes each plane by registering all planes to eliminate motion, discovering cells, and extracting 
    cell fluorescence."""
    COMBINE = 3
    """Stage 3: Combines all plane-specific data into a unified 'combined' dataset. This is a prerequisite for running 
    multi-day suite2p pipeline."""


class _ProcessingPipelines(StrEnum):
    """Defines the set of supported remote processing pipeline names.

    This enumeration is primarily used to standardize the names of the remote processing pipelines used in this library.

    Notes:
        The fields in this enumeration match the fields in the ProcessingTracker enumeration, since each valid
        processing pipeline is associated with a ProcessingTracker file. However, not all ProcessingTracker files are
        associated with a data processing pipeline: some are used for preprocessing or dataset formation pipelines.
    """

    BEHAVIOR = "behavior"
    SUITE2P = "single-day suite2p"


@dataclass()
class _ProcessingPipeline:
    """Manages a single remote processing pipeline running on the remote server.

    This class instance functions as a general-purpose interface for executing processing pipelines on the Sun lab
    compute servers. It is processing-pipeline agnostic and works for all currently supported Sun lab processing
    pipelines.

    Notes:
        The processing graph for each pipeline is fully resolved at the instantiation of this class instance. This
        means that the instance is preconfigured to store the Job objects for each processing stage of the pipeline at
        instantiation.
    """

    server: Server
    """The reference to the Server object that maintains bidirectional communication with the remote server running 
    the pipeline."""
    manager_id: int
    """The unique identifier for the manager process that constructs and manages the runtime of the tracked pipeline. 
    This is used to ensure that only a single pipeline instance can work with each session's data at the same time."""
    jobs: dict[int, list[Job]]
    """Stores a dictionary that maps the processing stage integer-codes to lists of Job objects, one for each 
    independent job instance to be submitted as part of that managed processing pipeline stage."""
    job_working_directories: list[Path]
    """Stores a list of paths to each managed job's working directory on the remote server. This path is 
    used to remove the log folder for each successful job if the pipeline is not configured to preserve job logs."""
    remote_tracker_path: Path
    """The path to the processing tracker .yaml file for the pipeline stored on the remote server running the 
    pipeline."""
    local_tracker_path: Path
    """The path to the local pipeline processing tracker .yaml file. The remote file is pulled to this location as 
    part of each processing stage outcome verification process."""
    session: str
    """The ID of the session whose data is being processed by the tracked pipeline."""
    animal: str
    """The ID of the animal whose data is being processed by the tracked pipeline."""
    project: str
    """The name of the project whose data is being processed by the tracked pipeline."""
    pipeline_type: _ProcessingPipelines
    """Stores the name of the processing pipeline managed by this class instance. Primarily, this is used to identify 
    the pipeline to the user in terminal messages and logs."""
    keep_job_logs: bool = False
    """Determines whether to keep the logs for successfully completed jobs on the server or (default) to remove them 
    after pipeline successfully ends its runtime. If the pipeline fails to complete its runtime, the logs are kept 
    regardless of this setting."""
    pipeline_status: _ProcessingStatus | int = _ProcessingStatus.RUNNING
    """Stores the current status of the tracked remote pipeline. This field is the primary means with which the class 
    instance communicates with external pipeline management functions from this library."""
    _pipeline_stage: int = 0
    """Stores the current stage of the tracked pipeline. Note, each stage can be associated with one or more 
    individual jobs running on the server in-parallel. This field is initialized to a non-valid value '0' and is then 
    modified as the instance tracks the completion of each stage of the managed pipeline."""

    def job_cycle(self) -> None:
        """This is the main entry point for all interactions with the processing pipeline running on the remote server.

        During processing, this method should be called repeatedly to track the progress of the pipeline managed by this
        instance and continuously advance the pipeline across all of its inter-dependent stages. This method updates
        the 'pipeline_status' instance field to communicate whether the managed pipeline is still running, succeeded,
        or failed.
        """

        # This clause is executed the first time the method is called for the newly initialized pipeline tracker
        # instance. For one-stage pipelines, this is the only time when pipeline jobs are submitted to the server.
        if self._pipeline_stage == 0:
            self._pipeline_stage += 1
            self._submit_jobs()

        # If the server has not completed all jobs in the current processing stage, returns to caller without doing
        # any additional processing.
        for job in self.jobs[self._pipeline_stage]:
            if not self.server.job_complete(job=job):
                return

        # If all jobs for the current processing stage have completed successfully, checks the shared processing
        # tracker file to determine if all jobs completed successfully.
        ensure_directory_exists(self.local_tracker_path)  # Ensures that the local temporary directory exists
        self.server.pull_file(remote_file_path=self.remote_tracker_path, local_file_path=self.local_tracker_path)
        tracker = get_processing_tracker(root=self.local_tracker_path.parent, file_name=TrackerFileNames.BEHAVIOR)

        # Checks whether the stage has completed without errors. If the stage failed due to encountering an error,
        # removes the local tracker copy and marks the pipeline as 'failed. It is expected that the pipeline state is
        # then handed by the caller to notify the user at the appropriate time.
        if tracker.encountered_error:
            # Removes the temporary directory where the local copy of the tracker file is stored.
            sh.rmtree(self.local_tracker_path.parent)
            self.pipeline_status = _ProcessingStatus.FAILED  # Updates the processing status to 'failed'
            return

        # If this was the last processing stage, the tracker would indicate that the processing has been completed.
        # In this case, initialized the shutdown sequence:
        if tracker.is_complete:
            # If the pipeline was configured to remove logs after completing successfully, removes the job logs from
            # the remote server. Note, removes the logs from the jobs submitted across all stages.
            if not self.keep_job_logs:
                for directory in self.job_working_directories:
                    self.server.remove(remote_path=directory, recursive=True, is_dir=True)

            # Removes the temporary directory where the local copy of the tracker file is stored.
            sh.rmtree(self.local_tracker_path.parent)

            self.pipeline_status = _ProcessingStatus.SUCCEEDED  # Updates the job status to 'succeeded'
            return

        # If the processing is not complete (according to the tracker), this indicates that the pipeline has more
        # stages to execute. In this case, increments the processing stage tracker and submits the next batch of jobs
        # to the server.
        self._pipeline_stage += 1
        self._submit_jobs()

    def _submit_jobs(self) -> None:
        """This worker method submits the processing jobs for the currently active pipeline stage to the remote
        server.

        It is used internally by the job_cycle() method to iteratively execute all stages of the processing pipeline on
        the remote server.
        """
        for job in self.jobs[self._pipeline_stage]:
            self.server.submit_job(job=job)


def _construct_behavior_processing_pipeline(
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> _ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance that can be used to run the behavior processing pipeline
    on the target session's data.

    This function composes the behavior processing pipeline (graph), packages it into the ProcessingPipeline, and
    returns it to the caller. This function does not itself submit the pipeline to the server, this is done the first
    time the job_cycle() method of the returned instance is called.

    Notes:
        Depending on the current server load and other jobs / pipeline in the processing queue, the pipeline may take a
        significant amount of time to execute.

        The returned ProcessingPipeline instance represents a complete solution for executing and tracking the state of
        the processing pipeline running on the remote server. All interactions with the pipeline should be done
        exclusively through that instance.

    Args:
        project: The name of the project for which to execute the behavior processing pipeline.
        session: The name of the session for which to execute the behavior processing pipeline.
        server: The Server class instance that manages access to the remote server that stores the session data to
            process.
        manager_id: The ID of the process that is managing the constructed behavior processing pipeline.
        reprocess: A boolean flag indicating whether to reprocess sessions that have already been processed.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this parameter.

    Returns:
        The _ProcessingPipeline instance configured to execute and manage the requested processing pipeline on the
        server if the session can be processed with the requested pipeline. None, if the session is excluded from
        processing for any reason.
    """
    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Resolves the path to the locally stored project manifest file
    manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # If the local manifest file does not exist, fetches it from the remote server
    if not manifest_path.exists():
        fetch_remote_project_manifest(project=project, server=server)

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

    # Prevents processing sessions that are marked as dataset integration candidates
    if dataset:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is currently in the 'dataset integration' mode and cannot be processed. To convert "
            f"the session back to the 'data processing' mode, call the 'sl-resolve-session-mode' CLI command from this "
            f"library with the appropriate runtime flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Otherwise, constructs the session processing pipeline and returns it to caller. Behavior processing pipeline is
    # executed as a single job, so it does not require an extensive setup process similar to the suite2p setup process.

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
    job.add_command(f"sl-process-behavior -sp {str(remote_session_path)} -pdr {str(processed_data_root)} -um")

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.processed_data_root).joinpath(
        project, animal, session, "processed_data", TrackerFileNames.BEHAVIOR
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_behavior_processing", TrackerFileNames.BEHAVIOR
    )

    # Packages job data into a _ProcessingPipeline object and returns it to the caller. The end-result is a 'one-stage'
    # and 'one-job' pipeline.
    pipeline = _ProcessingPipeline(
        jobs={1: [job]},
        server=server,
        manager_id=manager_id,
        pipeline_type=_ProcessingPipelines.BEHAVIOR,
        remote_tracker_path=remote_tracker_path,
        job_working_directories=[working_directory],
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=_ProcessingStatus.RUNNING,
    )

    return pipeline


def _construct_suite2p_processing_pipeline(
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    configuration_file: str = "GCaMP6f_CA1_SD.yaml",
    plane_count: int = 3,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> _ProcessingPipeline | None:
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
        fetch_remote_project_manifest(project=project, server=server)

    # Parses the target session data from the manifest file
    manifest = ProjectManifest(manifest_file=manifest_path)
    session_data = manifest.get_session_info(session=session)
    session_type = session_data["type"][0]
    animal = str(session_data["animal"][0])
    processed = bool(session_data["suite2p"][0])
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

    # Otherwise, resolves the single-day suite2p processing graph. Note; the suite2p processing relies on multiple jobs
    # submitted in 3 distinct processing stages. All Job objects are resolved before running the pipeline on the
    # remote server (below), so that the pipeline functions as a monolithic processing graph.

    # Precreates the lists to store stage jobs
    stage_1 = []
    stage_2 = []
    stage_3 = []
    working_directories = []

    # Resolves the path to the target suite2p configuration file stored on the remote server.
    configuration_path = get_remote_filesystem_paths(server=server).suite2p_configurations_path.joinpath(
        configuration_file
    )

    # Resolves the working directory for the job, using a static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    working_root = Path(server.user_working_root).joinpath("job_logs")

    # Parses the paths to the shared Sun lab directories used to store raw and processed session data on the remote
    # server.
    remote_session_path = Path(server.raw_data_root).joinpath(project, animal, session)
    processed_data_root = Path(server.processed_data_root)

    # Stage 1, Job 1: Binarization
    job_name = f"{session}_s2p_sd_binarization"
    working_directory = working_root.joinpath(f"{job_name}_{timestamp}")
    server.create_directory(remote_path=working_directory)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=2,
        ram_gb=10,
        time_limit=60,
    )
    stage_1.append(job)
    working_directories.append(working_directory)

    # Stage 2, Jobs 2+: Plane processing
    for plane in range(plane_count):
        job_name = f"{session}_s2p_sd_plane_{plane + 1}"
        working_directory = working_root.joinpath(f"{job_name}_{timestamp}")
        server.create_directory(remote_path=working_directory)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="suite2p",
            cpus_to_use=20,
            ram_gb=50,
            time_limit=60,
        )
        stage_1.append(job)
        working_directories.append(working_directory)

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.processed_data_root).joinpath(
        project, animal, session, "processed_data", TrackerFileNames.SUITE2P
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_suite2p_processing", TrackerFileNames.SUITE2P
    )

    # Packages job data into a _ProcessingPipeline object and returns it to the caller. The end-result is a 'one-stage'
    # and 'one-job' pipeline.
    pipeline = _ProcessingPipeline(
        jobs={1: [job]},
        server=server,
        manager_id=manager_id,
        pipeline_type=_ProcessingPipelines.BEHAVIOR,
        remote_tracker_path=remote_tracker_path,
        job_working_directories=[working_directory],
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=_ProcessingStatus.RUNNING,
    )

    return pipeline


def process_project_data(
    project: str,
    update_manifest: bool = True,
    sessions: list[str] | tuple[str, ...] | None = None,
    reprocess_behavior: bool = False,
    keep_job_logs: bool = False,
) -> None:
    # Entry message
    console.echo(message=f"Initializing project '{project}' data processing...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    credentials = get_credentials_file_path(require_service=True)
    server = Server(credentials_path=credentials)

    # Depending on the configuration, updates the project manifest file stored on the remote server and fetches it to
    # the local machine.
    if update_manifest:
        generate_remote_project_manifest(project=project, server=server)
    else:
        fetch_remote_project_manifest(project=project, server=server)

    # Loads the fetched manifest file into memory as a ProjectManifest instance.
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")
    manifest = ProjectManifest(manifest_file=manifest_path)

    # If the user did not specify a list of sessions to process, processes all available sessions for that project.
    # Ensures sessions are stored as a tuple of strings for efficiency.
    if sessions is None:
        sessions = manifest.get_sessions(exclude_incomplete=True, not_dataset_ready_only=True)
    else:
        sessions = tuple(sessions)

    # Generates the unique identifier for this runtime
    manager_id = generate_manager_id()

    # Generates the list of processing pipelines to run on the target project's data.
    pipelines: list[_ProcessingPipeline] = []
    for session in sessions:
        # Behavior processing pipeline.
        behavior_pipeline = _construct_behavior_processing_pipeline(
            server=server,
            manager_id=manager_id,
            project=project,
            session=session,
            reprocess=reprocess_behavior,
            keep_job_logs=keep_job_logs,
        )
        if behavior_pipeline is not None:
            # If the session is not excluded from processing, adds the pipeline for processing the session to the
            # storage list.
            pipelines.append(behavior_pipeline)

    # If the project requires no additional processing, aborts the runtime early
    if len(pipelines) == 0:
        message = (
            f"All available sessions for project '{project}' have been excluded from all supported processing "
            f"pipelines. See the messages above for details on exclusion criteria applied to each session and pipeline "
            f"combination."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Creates a progress bar to track the runtime progress of each processing pipeline.
    console.echo(message=f"Executing {len(pipelines)} resolved processing pipelines...", level=LogLevel.INFO)

    # Initializes a timer to delay repeated pipeline status checks
    delay_timer = PrecisionTimer("s")

    # Initializes tracker variables to track the processing progress
    uncompleted_count = len(pipelines)
    successful_count = 0
    failed_count = 0
    aborted_count = 0

    # Runs until all pipelines are completed (successfully or not)
    while uncompleted_count > 0:
        # At every loop cycle, checks the status of each running job
        for pipeline in pipelines:
            # If the pipeline has been completed, skips to the next pipeline
            if pipeline.pipeline_status != _ProcessingStatus.RUNNING:
                continue

            # Resolves the state of the pipeline. If necessary, this can advance the processing stage of the
            # pipeline and submit additional jobs to the server.
            pipeline.job_cycle()

            # If the pipeline status changed to one of the completed status codes, decrements the uncompleted
            # pipeline count and notifies the user about the outcome of the processing runtime.
            if pipeline.pipeline_status == _ProcessingStatus.FAILED:
                # The pipeline has encountered a runtime error and ended early
                message = (
                    f"The {pipeline.pipeline_type} processing pipeline for the session '{pipeline.session}' "
                    f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Failed. Check the "
                    f"remote job error logs stored on the server for the specific details about the cause of the "
                    f"failure."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                failed_count += 1
                uncompleted_count -= 1
            elif pipeline.pipeline_status == _ProcessingStatus.SUCCEEDED:
                # The pipeline has successfully completed the runtime
                message = (
                    f"The {pipeline.pipeline_type} processing pipeline for the session '{pipeline.session}' "
                    f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Complete."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
                successful_count += 1
                uncompleted_count -= 1
            else:
                # A very rare case: the pipeline was aborted by another user. It is highly unrealistic to encounter
                # this case.
                message = (
                    f"The {pipeline.pipeline_type} processing pipeline for the session '{pipeline.session}' "
                    f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Aborted."
                )
                console.echo(message=message, level=LogLevel.WARNING)
                aborted_count += 1
                uncompleted_count -= 1

        # Reruns the pipeline resolution cycle every 30 seconds to avoid overwhelming the communication line.
        delay_timer.delay_noblock(delay=30, allow_sleep=True)

        # Exit message
        message = (
            f"Project '{project}' data: Processed. Successfully completed {successful_count} pipelines, failed or "
            f"aborted {failed_count + aborted_count} pipelines."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
