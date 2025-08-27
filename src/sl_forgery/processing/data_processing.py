"""This module contains the bindings for all Sun lab data processing pipelines. The assets from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

from pathlib import Path

from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    Job,
    Server,
    SessionLock,
    SessionTypes,
    ProcessingStatus,
    TrackerFileNames,
    AcquisitionSystems,
    ProcessingPipeline,
    ProcessingPipelines,
    generate_manager_id,
    get_working_directory,
    get_credentials_file_path,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp

from ..utils import ProjectManifest, get_remote_job_work_directory
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest


def _check_session_eligibility(
    manifest: ProjectManifest,
    project: str,
    session: str,
    pipeline: ProcessingPipelines | str,
    supported_systems: set[str | AcquisitionSystems],
    supported_sessions: set[str | SessionTypes],
    allow_reprocessing: bool = False,
) -> bool:
    """Checks whether the target session meets the eligibility criteria for being processed with the specified pipeline.

    This worker function aggregates common eligibility checks to streamline the process for all supported processing
    pipelines.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the session's project.
        session: The name (ID) of the session to be processed.
        pipeline: The processing pipeline to be used to process the session's data.
        supported_systems: A set of data acquisition systems that support this type of processing.
        supported_sessions: A set of session types that support this type of processing.
        allow_reprocessing: Determines whether to allow reprocessing already processed sessions.

    Returns:
        True if the session meets the eligibility criteria, False otherwise.
    """
    # Parses the target session data from the manifest file
    session_data = manifest.get_session_info(session=session)
    session_type = session_data["type"][0]
    session_system = session_data["system"][0]
    animal = str(session_data["animal"][0])
    complete = session_data["complete"][0]

    # Determines whether the session has already been processed using the specified pipeline
    if pipeline == ProcessingPipelines.CHECKSUM:
        processed = bool(session_data["integrity"][0])
    elif pipeline == ProcessingPipelines.PREPARATION:
        processed = bool(session_data["prepared"][0])
    elif pipeline == ProcessingPipelines.ARCHIVING:
        processed = bool(session_data["archived"][0])
    elif pipeline == ProcessingPipelines.BEHAVIOR:
        processed = bool(session_data["behavior"][0])
    elif pipeline == ProcessingPipelines.SUITE2P:
        processed = bool(session_data["suite2p"][0])
    else:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The pipeline '{pipeline}' is not supported. "
            f"Use one of the supported pipelines: {list(ProcessingPipelines)}. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # Ensures that the pipeline's name is stored as a ProcessingPipelines instance
    pipeline = ProcessingPipelines(pipeline)

    # If the session was acquired using a data acquisition system that does not support this type of processing,
    # skips processing the session
    if session_system not in supported_systems:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session was acquired using the acquisition system "
            f"'{session_system},' which does not support this form of processing. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the session type is not one of the supported types, skips processing the session
    if session_type not in supported_sessions:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session is of type '{session_type},' which does not support "
            f"this form of processing. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # Prevents processing incomplete sessions
    if not complete:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed "
            f"by the animal '{animal}' for the '{project}' project. The session is marked as 'incomplete,' which "
            f"excludes it from all further data processing. To enable processing, manually mark it as 'complete' by "
            f"creating the 'telomere.bin' marker file in the session's 'raw_data' directory on the remote server and "
            f"setting the integrity_verification_tracker.yaml file to indicate that the verification was passed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the session has already been processed and reprocessing is not allowed, skips processing the session.
    if processed and not allow_reprocessing:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session has already been processed with this pipeline "
            f"and reprocessing is disabled. To enable reprocessing, call this command with the '--reprocess (-r)' "
            f"flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # The session is eligible for processing with this pipeline.
    return True


def _acquire_session_lock(
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
    force: bool = False,
    keep_job_logs: bool = False,
) -> None:
    """Acquires exclusive access to the target session's data for the specified manager process.

    This function is used to verify that the data of each session stored on the remove compute server is accessible to
    a single manager process at a time to ensure safe access while using multiple parallel processes. Acquiring
    exclusive data access lock is a prerequisite for all other session data processing functions.

    Notes:
        Each runtime that calls this function must also call the _release_session_lock() function.

    Args:
        project: The name of the project under which the target session was acquired.
        animal: The ID of the animal that participated in the target session.
        session: The name of the session for which to acquire the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target session's
            data.
        manager_id: The unique identifier of the process that calls this function.
        force: Determines whether to forcibly reset the access lock, if it is held by a different manager process. This
            option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.

    """
    console.echo(message=f"Constructing session lock acquisition job...")

    # Resolves the job name and its remote working directory.
    job_name = f"{session}_lock_acquisition"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

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

    # Parses the path to the shared Sun lab directory used to store raw session data
    session_folder = server.raw_data_root.joinpath(project, animal, session)

    # Parses additional processing flags for the lock acquisition command
    tracker_command = ""
    if force:
        tracker_command = "-r"

    # Configures the job to use the sl-shared-assets library installed on the server to acquire exclusive access to the
    # session's data for the specified manager process
    job.add_command(
        f"sl-manage session -sp {session_folder} -pdr {server.processed_data_root} -id {manager_id} {tracker_command} "
        f"lock"
    )

    # If the function is configured to remove job logs after runtime, adds a command to delete job working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {working_directory}")

    # Submits the remote job to the server
    job = server.submit_job(job)

    # Waits for the server to complete the job
    delay_timer = PrecisionTimer("s")
    message = f"Waiting for the session data lock acquisition job with ID {job.job_id} to complete..."
    console.echo(message=message, level=LogLevel.INFO)
    while not server.job_complete(job=job):
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    # Verifies the job completion status by checking the session_lock file for the owner's ID
    console.echo(message=f"Verifying that the manager process {manager_id} has acquired the session's data lock...")

    # Resolves the paths to the local and remote session lock files.
    local_working_directory = get_working_directory()
    remote_lock_path = server.raw_data_root.joinpath(project, animal, session, "tracking_data", "session_lock.yaml")
    local_lock_path = local_working_directory.joinpath(project, job_name, "manifest.feather", "session_lock.yaml")
    ensure_directory_exists(local_lock_path)

    # Pulls the remote session lock file to the local machine
    server.pull_file(
        local_file_path=local_lock_path,
        remote_file_path=remote_lock_path,
    )

    # Ensures that the caller process has exclusive access to session's data. This raises an error if the expectation
    # is violated.
    lock = SessionLock(file_path=local_lock_path)
    lock.check_owner(manager_id=manager_id)

    # If the lock has been acquired successfully, removes the local working directory
    delete_directory(local_lock_path.parent)


def _release_session_lock(session: str, server: Server) -> None:
    pass


def _construct_checksum_resolution_pipeline(
    manifest: ProjectManifest,
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
    recreate_checksum: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the raw data integrity checksum resolution
    pipeline for the target session.

    Notes:
        This pipeline always works with data stored on the 'raw data' volume of the remote compute server.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        animal: The ID of the animal for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        reprocess: Determines whether to reprocess the session if it has already been processed.
        reset_tracker: Determines whether to reset the processing tracker for the pipeline before executing the
            processing. This option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.
        recreate_checksum: Determines whether to recalculate and overwrite the data integrity checksums stored in the
            'raw data' folder instead of verifying its' integrity. This flag is used to update the checksum following
            expected changes to the session's raw data.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
    """

    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Parses the path to the session directory on the remote server
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        pipeline=ProcessingPipelines.CHECKSUM,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={
            SessionTypes.WINDOW_CHECKING,
            SessionTypes.LICK_TRAINING,
            SessionTypes.RUN_TRAINING,
            SessionTypes.MESOSCOPE_EXPERIMENT,
        },
        allow_reprocessing=True if recreate_checksum or reprocess else False,
    ):
        # If the session is not eligible, skips processing the session.
        return None

    # Resolves the name and working directory for the job
    job_name = f"{session}_behavior_processing"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Generates the remote job header and configures it to run behavior processing
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpus_to_use=30,
        ram_gb=5,
        time_limit=180,
    )

    # Resolves additional flags for the processing CLI
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"
    recalculate_command = ""
    if recreate_checksum:
        recalculate_command = "-rc"

    # Instructs the server to execute the target processing pipeline
    job.add_command(
        f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
        f"{tracker_command} checksum {recalculate_command}"
    )

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.CHECKSUM
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_checksum_resolution", TrackerFileNames.CHECKSUM
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
        jobs={1: ((job, working_directory),)},
        server=server,
        manager_id=manager_id,
        pipeline_type=ProcessingPipelines.CHECKSUM,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
    )

    return pipeline


def _construct_preparation_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the processing preparation pipeline for
    the target session.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        reprocess: Determines whether to reprocess the session if it has already been processed.
        reset_tracker: Determines whether to reset the processing tracker for the pipeline before executing the
            processing. This option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
    """

    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Parses the target session data from the manifest file
    session_data = manifest.get_session_info(session=session)
    animal = str(session_data["animal"][0])

    # Parses the path to the session directory on the remote server
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        pipeline=ProcessingPipelines.PREPARATION,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={
            SessionTypes.LICK_TRAINING,
            SessionTypes.RUN_TRAINING,
            SessionTypes.MESOSCOPE_EXPERIMENT,
        },
        allow_reprocessing=reprocess,
    ):
        # If the session is not eligible, skips processing the session.
        return None

    # Resolves the name and working directory for the job
    job_name = f"{session}_preparation"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Generates the remote job header and configures it to run behavior processing
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpus_to_use=30,
        ram_gb=5,
        time_limit=180,
    )

    # Resolves additional flags for the processing CLI
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"

    # Instructs the server to execute the target processing pipeline
    job.add_command(
        f"sl-process-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
        f"{tracker_command} prepare"
    )

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.PREPARATION
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_checksum_resolution", TrackerFileNames.PREPARATION
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
        jobs={1: ((job, working_directory),)},
        server=server,
        manager_id=manager_id,
        pipeline_type=ProcessingPipelines.PREPARATION,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
    )

    return pipeline


def _construct_behavior_processing_pipeline(
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
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
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
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
    local_tracker_path = local_working_directory.joinpath(project, job_name, TrackerFileNames.BEHAVIOR)

    # Packages job data into a ProcessingPipeline object and returns it to the caller. The end-result is a 'one-stage'
    # and 'one-job' pipeline.
    pipeline = ProcessingPipeline(
        jobs={1: ((job, working_directory),)},
        server=server,
        manager_id=manager_id,
        pipeline_type=ProcessingPipelines.BEHAVIOR,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
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
) -> ProcessingPipeline | None:
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
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
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
    configuration_path = server.suite2p_configurations_directory.joinpath(configuration_file)
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

    # Packages job data into a ProcessingPipeline object and returns it to the caller. The end-result is a 'one-stage'
    # and 'one-job' pipeline.
    pipeline = ProcessingPipeline(
        jobs={1: tuple(stage_1), 2: tuple(stage_2), 3: tuple(stage_3)},
        server=server,
        manager_id=manager_id,
        pipeline_type=ProcessingPipelines.SUITE2P,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
    )

    return pipeline


def process_project_data(
    project: str,
    sessions: list[str] | tuple[str, ...] | None = None,
    process_behavior: bool = True,
    process_suite2p: bool = True,
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
    credentials = get_credentials_file_path(service=True)
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
        sessions = manifest.get_sessions(exclude_incomplete=True)
    else:
        sessions = tuple(sessions)

    # Generates the unique identifier for this runtime
    manager_id = generate_manager_id()

    # Generates the list of processing pipelines to run on the target project's data.
    console.echo(message=f"Resolving the data processing pipelines to run on the project data...", level=LogLevel.INFO)
    pipelines: list[ProcessingPipeline] = []
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
            if pipeline.pipeline_status == ProcessingStatus.FAILED:
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
            elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
                # The pipeline has successfully completed the runtime
                message = (
                    f"The {pipeline.pipeline_type} processing pipeline for the session '{pipeline.session}' "
                    f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Complete."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
                successful_count += 1
                uncompleted_count -= 1
            elif pipeline.pipeline_status == ProcessingStatus.ABORTED:
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
