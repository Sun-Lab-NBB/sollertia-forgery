"""This module contains tools and bindings for all Sun lab data processing pipelines. The tools from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

from enum import IntEnum, StrEnum
import shutil as sh
from pathlib import Path
from dataclasses import dataclass

from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    Job,
    Server,
    SessionTypes,
    ProjectManifest,
    TrackerFileNames,
    ProcessingTracker,
    generate_manager_id,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp

from ..utils import get_working_directory, get_credentials_file_path, get_remote_filesystem_paths
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest


class _ProcessingStatus(IntEnum):
    """Maps integer-based processing pipeline status (state) codes to human-readable names.

    This enumeration is used internally to track and communicate the progress of data processing pipelines as they are
    executed by the remote server.

    Notes:
        The status codes from this enumeration track the state of the pipeline as a whole, instead of tracking the
        state of each job that comprises the pipeline.
    """

    RUNNING = 0
    """The pipeline is currently running on the remote server. It may be executed (in progress) or waiting for 
    the required resources to become available (queued)."""
    SUCCEEDED = 1
    """The server has completed the processing pipeline. This status indicates that the processing was complete 
    in-full."""
    FAILED = 2
    """The server has failed to complete the pipeline. This status indicates that the processing might have completed 
    partially, but was not finished."""
    ABORTED = 3
    """The pipeline has been aborted by the managing user or another user with service privilege access. This state is 
    generally not expected to be encountered often, but it is possible."""


class _ProcessingPipelines(StrEnum):
    """Defines the set of data processing pipelines supported by this library.

    All data processing pipelines currently supported by the lab codebase are defined in this enumeration. Primarily,
    the elements from this enumeration are used in terminal messages and logging to identify the pipelines to the user.

    Notes:
        The elements in this enumeration partially match the elements in the ProcessingTracker enumeration, since each
        valid data processing pipeline has an associated ProcessingTracker file. However, not all ProcessingTrackers are
        associated with a data processing pipeline; some are used in preprocessing or dataset formation pipelines.
    """

    BEHAVIOR = "behavior"
    """Behavior processing pipeline."""
    SUITE2P = "suite2p"
    """Single-day suite2p pipeline."""
    VIDEO = "video"
    """DeepLabCut (Video) processing pipeline."""


@dataclass()
class _ProcessingPipeline:
    """Encapsulates access to a processing pipeline running on the remote compute server.

    This class functions as an interface for all data processing pipelines running on Sun lab compute servers. It is
    pipeline-type-agnostic and works for all data processing pipelines supported by this library. After instantiation,
    the class automatically handles all interactions with the server necessary to run the remote processing pipeline and
    verify the runtime outcome via the runtime_cycle() method that has to be called cyclically until the pipeline is
    complete.

    Notes:
        Each pipeline may be executed in one or more stages, each stage using one or more parallel jobs. As such, each
        pipeline can be seen as an execution graph that sequentially submits batches of jobs to the remote server. The
        processing graph for each pipeline is fully resolved at the instantiation of this class instance, so each
        instance contains the necessary data to run the entire processing pipeline.

        The minimum self-contained unit of the processing pipeline is a single job. Since jobs can depend on the output
        of other jobs, they are organized into stages based on the dependency graph between jobs. Combined with cluster
        management software, such as SLURM, this class can efficiently execute processing pipelines on scalable compute
        clusters.
    """

    pipeline_type: _ProcessingPipelines
    """Stores the name of the processing pipeline managed by this instance. Primarily, this is used to identify the 
    pipeline to the user in terminal messages and logs."""
    server: Server
    """The reference to the Server object that maintains bidirectional communication with the remote server running 
    the pipeline."""
    manager_id: int
    """The unique identifier for the manager process that constructs and manages the runtime of the tracked pipeline. 
    This is used to ensure that only a single pipeline instance can work with each session's data at the same time on 
    the remote server."""
    jobs: dict[int, tuple[tuple[Job, Path], ...]]
    """Stores the dictionary that maps the pipeline processing stage integer-codes to two-element tuples. Each tuple
    stores the Job objects and the paths to their remote working directories to be submitted to the server at each 
    stage."""
    remote_tracker_path: Path
    """The path to the pipeline's processing tracker .yaml file stored on the remote compute server."""
    local_tracker_path: Path
    """The path to the pipeline's processing tracker .yaml file on the local machine. The remote file is pulled to 
    this location when the instance verifies the outcome of each tracked pipeline's processing stage."""
    session: str
    """The ID of the session whose data is being processed by the tracked pipeline."""
    animal: str
    """The ID of the animal whose data is being processed by the tracked pipeline."""
    project: str
    """The name of the project whose data is being processed by the tracked pipeline."""
    keep_job_logs: bool = False
    """Determines whether to keep the logs for the jobs making up the pipeline execution graph or (default) to remove 
    them after pipeline successfully ends its runtime. If the pipeline fails to complete its runtime, the logs are kept 
    regardless of this setting."""
    pipeline_status: _ProcessingStatus | int = _ProcessingStatus.RUNNING
    """Stores the current status of the tracked remote pipeline. This field is updated each time runtime_cycle() 
    instance method is called."""
    _pipeline_stage: int = 0
    """Stores the current stage of the tracked pipeline. This field is monotonically incremented by the runtime_cycle()
    method to sequentially submit batches of jobs to the server in a processing-stage-driven fashion."""

    def __post_init__(self) -> None:
        """Carries out the necessary filesystem setup tasks to support pipeline execution."""
        ensure_directory_exists(self.local_tracker_path)  # Ensures that the local temporary directory exists

    def runtime_cycle(self) -> None:
        """Checks the current status of the tracked pipeline and, if necessary, submits additional batches of jobs to
        the remote server to progress the pipeline.

        This method is the main entry point for all interactions with the processing pipeline managed by this instance.
        It checks the current state of the pipeline, advances the pipeline's processing stage, and submits the necessary
        jobs to the remote server. The process managing the data processing runtime should call this method repeatedly
        (cyclically) to run the pipeline until the 'is_running' property of the instance returns True.

        Notes:
            While the 'is_running' property can be used to determine whether the pipeline is still running, to resolve
            the final status of the pipeline (success or failure), the manager process should access the
            'pipeline_status' instance attribute.
        """

        # This clause is executed the first time the method is called for the newly initialized pipeline tracker
        # instance. It submits the first batch of processing jobs (first stage) to the remote server. For one-stage
        # pipelines, this is the only time when pipeline jobs are submitted to the server.
        if self._pipeline_stage == 0:
            self._pipeline_stage += 1
            self._submit_jobs()

        # Waits until all jobs submitted to the server as part of the current processing stage are completed before
        # advancing further.
        for job, _ in self.jobs[self._pipeline_stage]:  # Ignores working directories as part of this iteration.
            if not self.server.job_complete(job=job):
                return

        # If all jobs for the current processing stage have completed, checks the pipeline's processing tracker file to
        # determine if all jobs completed successfully.
        self.server.pull_file(remote_file_path=self.remote_tracker_path, local_file_path=self.local_tracker_path)
        tracker = ProcessingTracker(self.local_tracker_path)

        # If the stage failed due to encountering an error, removes the local tracker copy and marks the pipeline
        # as 'failed'. It is expected that the pipeline state is then handed by the manager process to notify the
        # user about the runtime failure.
        if tracker.encountered_error:
            sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
            self.pipeline_status = _ProcessingStatus.FAILED  # Updates the processing status to 'failed'

        # If this was the last processing stage, the tracker indicates that the processing has been completed. In this
        # case, initialized the shutdown sequence:
        elif tracker.is_complete:
            sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
            self.pipeline_status = _ProcessingStatus.SUCCEEDED  # Updates the job status to 'succeeded'

            # If the pipeline was configured to remove logs after completing successfully, removes the runtime log for
            # each job submitted as part of this pipeline from the remote server.
            if not self.keep_job_logs:
                for stage_jobs in self.jobs.values():
                    for _, directory in stage_jobs:  # Ignores job objects as part of this iteration.
                        self.server.remove(remote_path=directory, recursive=True, is_dir=True)

        # If the processing is not complete (according to the tracker), this indicates that the pipeline has more
        # stages to execute. In this case, increments the processing stage tracker and submits the next batch of jobs
        # to the server.
        elif tracker.is_running:
            self._pipeline_stage += 1
            self._submit_jobs()

        # The final and the rarest state: the pipeline was aborted before it finished the runtime. Generally, this state
        # should not be encountered during most runtimes.
        else:
            self.pipeline_status = _ProcessingStatus.ABORTED

    def _submit_jobs(self) -> None:
        """This worker method submits the processing jobs for the currently active processing stage to the remote
        server.

        It is used internally by the runtime_cycle() method to iteratively execute all stages of the managed processing
        pipeline on the remote server.
        """
        for job, _ in self.jobs[self._pipeline_stage]:
            self.server.submit_job(job=job)

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is currently running, False otherwise."""
        if self.pipeline_status == _ProcessingStatus.RUNNING:
            return True
        return False


def _construct_behavior_processing_pipeline(
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> _ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the behavior processing pipeline for the
    target session.

    This function composes the processing pipeline and packages it into the ProcessingPipeline. This pipeline extracts
    the non-video and non-brain-activity data stored inside the .npz log files acquired by Sun lab data acquisition
    systems. The extracted data is stored as a series of Polars dataframes using the .feather (IPC) format compressed
    with 'lz4' scheme.

    Notes:
        This function does not start executing the pipeline. Instead, the pipeline starts executing the first time
        the manager process calls its runtime_cycle() method.

        If the function determines that the target session cannot be processed, it instead returns None and notifies
        the user why the session was excluded from processing via the terminal.

    Args:
        project: The name of the project for which to execute the behavior processing pipeline.
        session: The name of the session to process with the behavior processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        reprocess: Determines whether to reprocess sessions that have already been processed.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The _ProcessingPipeline instance configured to execute and manage the behavior processing pipeline on the
        server if the session can be processed with this pipeline. None, if the session is excluded from processing
        for any reason.
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
    complete = bool(session_data["complete"][0]) and bool(session_data["integrity"][0])
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
            f"project. The session has already been processed. To enable reprocessing already processed sessions, call "
            f"this command with the 'reprocess' flag set to True."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Prevents processing incomplete sessions
    if not complete:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is either marked as 'incomplete,' or did not pass integrity verification when it was"
            f"moved to the remote server, which excludes it from automated data processing. To enable processing, "
            f"manually mark it as 'complete' by creating the 'telomere.bin' marker file in the session's raw_data "
            f"directory on the remote server and setting the integrity_verification_tracker.yaml file to indicate that"
            f"verification was passed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Prevents processing sessions that are marked as dataset integration candidates
    if dataset:
        message = (
            f"Unable to process behavior data for session '{session}' performed by animal '{animal}' for '{project}' "
            f"project. The session is currently in the 'dataset integration' mode and cannot be processed. To convert "
            f"the session back to the 'data processing' mode, call the 'sl-resolve-session-mode' CLI command with the "
            f"'--remove' (-r) flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Otherwise, constructs the session processing pipeline and returns it to caller. Behavior processing pipeline is
    # executed as a single job, so it does not require an extensive setup process (unlike the suite2p pipeline).

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

    # Generates the remote job header and configures it to run behavior processing
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="behavior",
        cpus_to_use=7,
        ram_gb=5,
        time_limit=180,
    )
    job.add_command(f"sl-process-behavior -sp {remote_session_path!s} -pdr {processed_data_root!s} -um")

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
        jobs={1: ((job, working_directory),)},
        server=server,
        manager_id=manager_id,
        pipeline_type=_ProcessingPipelines.BEHAVIOR,
        remote_tracker_path=remote_tracker_path,
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
    """Generates and returns the ProcessingPipeline instance used to execute the single-day suite2p processing pipeline
    for the target session.

    This function composes the processing pipeline and packages it into the ProcessingPipeline. This pipeline extracts
    the brain activity data from the mesoscope-acquired .tiff stacks. The extracted data is stored as a collection of
    NumPy .npy files and is later used during the multi-day suite2p pipeline.

    Notes:
        This function does not start executing the pipeline. Instead, the pipeline starts executing the first time
        the manager process calls its runtime_cycle() method.

        If the function determines that the target session cannot be processed, it instead returns None and notifies
        the user why the session was excluded from processing via the terminal.

    Args:
        project: The name of the project for which to execute the single-day suite2p processing pipeline.
        session: The name of the session to process with the single-day suite2p processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            processing parameters to use for this runtime. The file with this name (and a .yaml) extensions must be
            present in the shared suite2p configuration folder on the remote compute server for the pipeline to be able
            to run the processing.
        plane_count: The number of planes in the input dataset. For mesoscope images, this is the number of ROIs
            (stripes) x the number of z-planes. This determines the number of plane processing jobs to execute during
            runtime.
        reprocess: Determines whether to reprocess sessions that have already been processed.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The _ProcessingPipeline instance configured to execute and manage the single-day suite2p processing pipeline
        on the server if the session can be processed with this pipeline. None, if the session is excluded from
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
    processed = bool(session_data["suite2p"][0])
    complete = bool(session_data["complete"][0]) and bool(session_data["integrity"][0])
    dataset = bool(session_data["dataset"][0])

    # If the session type is not one of the supported types, skips processing the session
    if session_type not in {SessionTypes.MESOSCOPE_EXPERIMENT}:
        message = (
            f"Unable to process 2-photon brain activity data for session '{session}' performed by animal '{animal}' "
            f"for '{project}' project. The session is of type '{session_type},' which does not support this form of "
            f"processing. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # If the session has already been processed and the function is not running in reprocessing mode, skips
    # processing the session.
    if processed and not reprocess:
        message = (
            f"Unable to process 2-photon brain activity data for session '{session}' performed by animal '{animal}' "
            f"for '{project}' project. The session has already been processed. To enable reprocessing already "
            f"processed sessions, call this command with the 'reprocess' flag set to True."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Prevents processing incomplete sessions
    if not complete:
        message = (
            f"Unable to process 2-photon brain activity data for session '{session}' performed by animal '{animal}' "
            f"for '{project}' project. The session is either marked as 'incomplete,' or did not pass integrity "
            f"verification when it was moved to the remote server, which excludes it from automated data processing. "
            f"To enable processing, manually mark it as 'complete' by creating the 'telomere.bin' marker file in the "
            f"session's raw_data directory on the remote server and setting the integrity_verification_tracker.yaml "
            f"file to indicate that verification was passed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Prevents processing sessions that are marked as dataset integration candidates
    if dataset:
        message = (
            f"Unable to process 2-photon brain activity data for session '{session}' performed by animal '{animal}' "
            f"for '{project}' project. The session is currently in the 'dataset integration' mode and cannot be "
            f"processed. To convert the session back to the 'data processing' mode, call the 'sl-resolve-session-mode' "
            f"CLI command with the '--remove' (-r) flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Ensures that the target suite2p configuration file exists on the remote server
    configuration_path = get_remote_filesystem_paths(server=server).suite2p_configurations_path.joinpath(
        configuration_file
    )
    if not server.exists(configuration_path):
        message = (
            f"Unable to process 2-photon brain activity data for session '{session}' performed by animal '{animal}' "
            f"for '{project}' project. The suite2p configuration file '{configuration_file}' does not exist on the "
            f"remote server."
        )
        console.error(message=message, error=ValueError)

    # Otherwise, resolves the single-day suite2p processing graph. Note; the suite2p processing relies on multiple jobs
    # submitted in 3 distinct processing stages. All Job objects are resolved before running the pipeline on the
    # remote server (below), so that the pipeline functions as a monolithic processing graph.

    # Precreates the iterables to store stage jobs
    stage_1 = []
    stage_2 = []
    stage_3 = []

    # Resolves the directory where to store the data for all jobs executed as part of the pipeline and the current
    # timestamp (to use in job working directory names).
    timestamp = get_timestamp()
    working_root = Path(server.user_working_root).joinpath("job_logs")

    # Parses the paths to the shared Sun lab directories used to store raw and processed session data on the remote
    # server.
    remote_session_path = Path(server.raw_data_root).joinpath(project, animal, session)
    processed_data_root = Path(server.processed_data_root)

    # Stage 1: Binarization
    job_name = f"{session}_s2p_sd_binarization"
    working_directory = working_root.joinpath(f"{job_name}_{timestamp}")
    server.create_directory(remote_path=working_directory)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=1,
        ram_gb=5,
        time_limit=240,
    )
    job.add_command(
        f"sl-process-suite2p -i {configuration_path!s} -sp {remote_session_path!s} "
        f"-pdr {processed_data_root!s} -b -w -1 -um"
    )
    stage_1.append((job, working_directory))

    # Stage 2: Plane processing
    for plane in range(plane_count):
        job_name = f"{session}_s2p_sd_plane_{plane}"
        working_directory = working_root.joinpath(f"{job_name}_{timestamp}")
        server.create_directory(remote_path=working_directory)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="suite2p",
            cpus_to_use=42,
            ram_gb=80,
            time_limit=300,
        )
        job.add_command(
            f"sl-process-suite2p -i {configuration_path!s} -sp {remote_session_path!s} "
            f"-pdr {processed_data_root!s} -p -t {plane} -w -1 -um"
        )
        stage_2.append((job, working_directory))

    # Stage 3: Combination
    job_name = f"{session}_s2p_sd_combination"
    working_directory = working_root.joinpath(f"{job_name}_{timestamp}")
    server.create_directory(remote_path=working_directory)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=1,
        ram_gb=4,
        time_limit=90,
    )
    job.add_command(
        f"sl-process-suite2p -i {configuration_path!s} -sp {remote_session_path!s} "
        f"-pdr {processed_data_root!s} -c -w -1 -um"
    )
    stage_3.append((job, working_directory))

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
        jobs={1: tuple(stage_1), 2: tuple(stage_2), 3: tuple(stage_3)},
        server=server,
        manager_id=manager_id,
        pipeline_type=_ProcessingPipelines.SUITE2P,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=_ProcessingStatus.RUNNING,
    )

    return pipeline


def _construct_dataset_marker_job(
    project: str,
    session: str,
    server: Server,
    create: bool = False,
) -> Job | None:
    """Generates and returns the ProcessingPipeline instance used to execute the single-day suite2p processing pipeline
    for the target session.

    This function composes the processing pipeline and packages it into the ProcessingPipeline. This pipeline extracts
    the brain activity data from the mesoscope-acquired .tiff stacks. The extracted data is stored as a collection of
    NumPy .npy files and is later used during the multi-day suite2p pipeline.

    Notes:
        This function does not start executing the pipeline. Instead, the pipeline starts executing the first time
        the manager process calls its runtime_cycle() method.

        If the function determines that the target session cannot be processed, it instead returns None and notifies
        the user why the session was excluded from processing via the terminal.

    Args:
        project: The name of the project for which to execute the single-day suite2p processing pipeline.
        session: The name of the session to process with the single-day suite2p processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The _ProcessingPipeline instance configured to execute and manage the single-day suite2p processing pipeline
        on the server if the session can be processed with this pipeline. None, if the session is excluded from
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
    complete = bool(session_data["complete"][0]) and bool(session_data["integrity"][0])
    behavior = bool(session_data["behavior"][0])
    suite2p = bool(session_data["suite2p"][0])
    dataset = bool(session_data["dataset"][0])

    # Prevents resolving dataset marker for incomplete sessions.
    if not complete:
        message = (
            f"Unable to resolve the dataset marker for the session '{session}' performed by animal '{animal}' for "
            f"'{project}' project. The session is either marked as 'incomplete,' or did not pass integrity "
            f"verification when it was moved to the remote server. If necessary, resolve the marker manually by "
            f"creating or removing the 'p53.bin' marker file from the session's processed_data directory on the remote "
            f"server."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # Prevents recreating the dataset marker if it already exists.
    if create and dataset:
        message = (
            f"Session '{session}' performed by animal '{animal}' for '{project}' project is already marked as ready "
            f"for dataset integration. Skipping (re)creating the dataset marker for the session."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
        return None

    # Prevents removing the dataset marker if it does not exist.
    elif not create and not dataset:
        message = (
            f"Session '{session}' performed by animal '{animal}' for '{project}' project does not contain the dataset "
            f"integration marker. Skipping removing the nonexistent dataset marker for the session."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
        return None

    # Ensures that the session (type) supports dataset integration.
    if session_type not in {SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING, SessionTypes.MESOSCOPE_EXPERIMENT}:
        message = (
            f"Unable to resolve the dataset marker for the session '{session}' performed by animal '{animal}' for "
            f"'{project}' project. The session is of type '{session_type},' which does not support dataset "
            f"integration. Skipping resolving the dataset marker for the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    elif (
        session_type in {SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING, SessionTypes.MESOSCOPE_EXPERIMENT}
        and not behavior
    ):
        message = (
            f"Unable to resolve the dataset marker for the session '{session}' performed by animal '{animal}' for "
            f"'{project}' project. The session requires to be processed with the behavior processing pipeline before "
            f"it can be integrated into a dataset, but the manifest file for the project indicates that the session "
            f"has not been processed with this pipeline. Call the 'sl-process' command to conduct the required "
            f"processing and retry resolving the dataset marker."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    elif session_type in {SessionTypes.MESOSCOPE_EXPERIMENT} and not suite2p:
        message = (
            f"Unable to resolve the dataset marker for the session '{session}' performed by animal '{animal}' for "
            f"'{project}' project. The session requires to be processed with the single-day suite2p processing "
            f"pipeline before it can be integrated into a dataset, but the manifest file for the project indicates "
            f"that the session has not been processed with this pipeline. Call the 'sl-process' command to conduct the "
            f"required processing and retry resolving the dataset marker."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    # This section works similar to other pipeline sections in this module. However, instead of constructing a
    # pipeline object, it constructs and submits a processing job to the server. Primarily, this is because the dataset
    # marker pipeline does not rely on processing tracker files like other pipeline

    # Resolves the working directory for the job, using a static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    job_name = f"{session}_dataset_marker"
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Ensures that the working directory exists on the remote server
    server.create_directory(remote_path=working_directory)

    # Parses the paths to the shared Sun lab directories used to store raw and processed session data on the remote
    # server.
    remote_session_path = Path(server.raw_data_root).joinpath(project, animal, session)
    processed_data_root = Path(server.processed_data_root)

    # Generates the remote job header and configures the job to resolve the dataset marker for the target session.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="manage",
        cpus_to_use=4,
        ram_gb=20,
        time_limit=90,
    )
    if create:
        job.add_command(f"sl-dataset-marker -sp {remote_session_path!s} -pdr {processed_data_root!s} -um")
    else:
        job.add_command(f"sl-dataset-marker -sp {remote_session_path!s} -pdr {processed_data_root!s} -um -r")

    return job


def process_project_data(
    project: str,
    sessions: list[str] | tuple[str, ...] | None = None,
    process_behavior: bool = True,
    process_suite2p: bool = True,
    create_dataset_markers: bool = True,
    remove_dataset_markers: bool = True,
    update_manifest: bool = True,
    reprocess: bool = False,
    keep_job_logs: bool = False,
    suite2p_configuration_file: str = "GCaMP6f_CA1_SD.yaml",
    plane_count: int = 3,
) -> None:
    """Resolves and executes the necessary data processing pipelines for the specified project.

    This function acts as the main entry point for all data processing in the Sun lab. As part of its runtime, it first
    determines which processing pipelines need to be executed for each session of the project. Then it efficiently
    executes these pipelines on the remote compute server by iteratively submitting batches of remote compute jobs
    to the server.

    Args:
        project: The name of the project to process.
        sessions: An iterable of session names to process as part of this runtime. If this optional argument is not
            provided, the function automatically processes all sessions that have not been processed with one or more
            supported pipelines.
        process_behavior: Determines whether to execute behavior data processing as part of this runtime.
        process_suite2p: Determines whether to execute single-day suite2p processing as part of this runtime.
        create_dataset_markers: Determines whether to create dataset markers for sessions that have been processed with
            all supported pipelines as part of this runtime. Note, once a session is marked with a dataset marker, it
            cannot be (re)processed until the marker is removed.
        remove_dataset_markers: Determines whether to remove dataset markers from sessions that have them before
            executing data processing. This allows (re)processing sessions that have been marked for dataset
            integration, but prevents them from being included in datasets until the marker are recreated again.
        update_manifest: Determines whether to update the project manifest file stored on the remote server after each
            processing step.
        reprocess: Determines whether to reprocess the sessions that have already been processed.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each processing pipeline completes successfully. If the pipeline fails, the job logs are kept regardless
            of the value of this argument.
        suite2p_configuration_file: Specifies the name of the configuration file for the single-day suite2p processing
            pipeline. This argument is only used if the 'process_suite2p' argument is set to True. The configuration
            file with the specified name must be present in the shared suite2p configuration directory on the remote
            compute server.
        plane_count: Specifies the number of planes in the session's data to be processed with the single-day suite2p
            pipeline. Note; for mesoscope recordings this number is equal to the number of ROI(s) (stripes) * the number
            of z-planes. This argument is only used if the 'process_suite2p' argument is set to True.
    """
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
    console.echo(message=f"Resolving the data processing pipelines to run on the project data...", level=LogLevel.INFO)
    pipelines: list[_ProcessingPipeline] = []
    for session in sessions:
        # Behavior processing pipeline.
        if process_behavior:
            behavior_pipeline = _construct_behavior_processing_pipeline(
                server=server,
                manager_id=manager_id,
                project=project,
                session=session,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
            )
            if behavior_pipeline is not None:
                pipelines.append(behavior_pipeline)

        # Suite2p processing pipeline.
        if process_suite2p:
            suite2p_pipeline = _construct_suite2p_processing_pipeline(
                server=server,
                manager_id=manager_id,
                project=project,
                session=session,
                configuration_file=suite2p_configuration_file,
                plane_count=plane_count,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
            )
            if suite2p_pipeline is not None:
                pipelines.append(suite2p_pipeline)

    # If the project requires no additional processing, aborts the runtime early
    if len(pipelines) == 0:
        message = (
            f"All target sessions for project '{project}' have been excluded from all supported processing pipelines. "
            f"See the messages above for details on exclusion criteria applied to each session and pipeline "
            f"combination. Processing: Aborted."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Creates a progress bar to track the runtime progress of each processing pipeline.
    console.echo(message=f"Executing {len(pipelines)} processing pipelines...", level=LogLevel.INFO)

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
            if not pipeline.is_running:
                continue

            # Resolves the state of the pipeline. If necessary, this can advance the processing stage of the
            # pipeline and submit additional jobs to the server.
            pipeline.runtime_cycle()

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
            elif pipeline.pipeline_status == _ProcessingStatus.ABORTED:
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
