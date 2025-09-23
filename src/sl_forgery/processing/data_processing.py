"""This module contains the bindings for all Sun lab data processing pipelines. The assets from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks."""

from pathlib import Path

from tqdm import tqdm
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
    delete_directory,
    generate_manager_id,
    get_working_directory,
    get_credentials_file_path,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists, chunk_iterable

from ..utils import ProjectManifest, get_remote_job_work_directory
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest


def _check_session_eligibility(
    manifest: ProjectManifest,
    project: str,
    session: str,
    pipeline: ProcessingPipelines | str,
    server: Server,
    supported_systems: set[str | AcquisitionSystems],
    supported_sessions: set[str | SessionTypes],
    allow_reprocessing: bool = False,
    configuration_file: str | None = None,
) -> bool:
    """Checks whether the target session meets the eligibility criteria for being processed with the specified pipeline.

    This worker function aggregates common eligibility checks to streamline the process for all supported processing
    pipelines.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the session's project.
        session: The name (ID) of the session to be processed.
        pipeline: The processing pipeline to be used to process the session's data.
        server: The initialized Server instance that manages the access to the remote compute server that stores the
            session's data and executes the processing pipelines.
        supported_systems: A set of data acquisition systems that support this type of processing.
        supported_sessions: A set of session types that support this type of processing.
        allow_reprocessing: Determines whether to allow reprocessing already processed sessions.
        configuration_file: The name of the configuration file to use during processing, if the pipeline requires it.

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
    prepared = True
    configuration_path: Path | None = None
    if pipeline == ProcessingPipelines.CHECKSUM:
        processed = bool(session_data["integrity"][0])
    elif pipeline == ProcessingPipelines.PREPARATION:
        processed = bool(session_data["prepared"][0])
    elif pipeline == ProcessingPipelines.ARCHIVING:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["archived"][0])
    elif pipeline == ProcessingPipelines.BEHAVIOR:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["behavior"][0])
    elif pipeline == ProcessingPipelines.SUITE2P:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["suite2p"][0])
        configuration_path = server.suite2p_configurations_directory.joinpath(configuration_file)
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

    # If the target processing pipeline requires the session data to be prepared, excludes any unprepared sessions from
    # processing.
    if not prepared:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The pipeline requires the session data to be prepared for "
            f"processing before it can be executed. Call the project data processing CLI with the '--prepare (-p)' "
            f"flag to prepare the target session for processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the target processing pipeline requires a specific server-side configuration file, ensures that the file is
    # present at the expected remote server location.
    if configuration_path is not None and not server.exists(remote_path=configuration_path):
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The target configuration file '{configuration_file}' does not "
            f"exist on the remote server at the expected path: {configuration_path}."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # The session is eligible for processing with this pipeline.
    return True


def _construct_lock_acquisition_job(
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
    force: bool = False,
    keep_job_logs: bool = False,
) -> Job:
    """Constructs the remote job used to acquire exclusive access to the data of the target session for the specified
    manager ID.

    This worker function is used as part of the overall lock acquisition step to efficiently construct and submit
    session data lock acquisition jobs to the remote compute server.

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

    Returns:
        The initialized Job instance for the constructed and submitted session data lock acquisition job.
    """
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

    # Submits the remote job to the server and returns the updated Job object to caller
    return server.submit_job(job, verbose=False)


def _verify_lock_acquisition_job(
    job: Job,
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
) -> None:
    """Verifies the outcome of a session data lock acquisition job that ran on a remote compute server.

    This worker function is used as part of the overall session data lock acquisition step to efficiently verify the
    outcome of completed session data lock acquisition jobs submitted to the remote compute server.

    Args:
        job: The initialized Job instance for the lock acquisition job to be verified.
        project: The name of the project under which the target session was acquired.
        animal: The ID of the animal that participated in the target session.
        session: The name of the session for which to acquire the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target session's
            data.
        manager_id: The unique identifier of the process that calls this function.
    """

    # Resolves the paths to the local and remote session lock files.
    local_working_directory = get_working_directory()
    remote_lock_path = server.raw_data_root.joinpath(project, animal, session, "tracking_data", "session_lock.yaml")
    local_lock_path = local_working_directory.joinpath(project, job.job_name, "manifest.feather", "session_lock.yaml")
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


def _acquire_session_lock(
    manifest: ProjectManifest,
    project: str,
    sessions: tuple[str, ...],
    server: Server,
    manager_id: int,
    force: bool = False,
    keep_job_logs: bool = False,
) -> None:
    """Acquires exclusive access to the target sessions' data for the specified manager process.

    This function is used to verify that the data of each processed session stored on the remote compute server is
    accessible to a single manager process at a time to ensure safe access while using multiple parallel processes.
    Acquiring exclusive data access lock is a prerequisite for all other session data processing functions.

    Notes:
        Each runtime that calls this function must also call the _release_session_lock() function.

    Args:
        project: The name of the project under which the target session was acquired.
        sessions: The sessions for which to acquire the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target session's
            data.
        manager_id: The unique identifier of the process that calls this function.
        force: Determines whether to forcibly reset the access lock, if it is held by a different manager process. This
            option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.

    """
    # Pre-creates a list of animal IDs for each session to be processed
    animals = [manifest.get_animal_for_session(session=session) for session in sessions]

    # Constructs and submits remote processing jobs to the server
    jobs = []
    for session, animal in tqdm(
        zip(sessions, animals), total=len(sessions), desc="Submitting session lock acquisition jobs", unit="job"
    ):
        jobs.append(
            _construct_lock_acquisition_job(
                project=project,
                animal=animal,
                session=session,
                server=server,
                manager_id=manager_id,
                force=force,
                keep_job_logs=keep_job_logs,
            )
        )

    completed_jobs = []
    with tqdm(total=len(jobs), desc="Waiting for the session lock acquisition jobs to complete", unit="job") as pbar:
        for index, job in enumerate(jobs):
            # Waits for each job to complete. Ensures that each job is verified exactly once.
            if not server.job_complete(job=job) or index in completed_jobs:
                continue

            _verify_lock_acquisition_job(
                job=job,
                project=project,
                animal=animals[index],
                session=sessions[index],
                server=server,
                manager_id=manager_id,
            )

            # Ensures that this job is not processed again as part of this function's cycle
            completed_jobs.append(index)

            # Increments the progress bar
            pbar.update()


def _construct_lock_release_job(
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
    keep_job_logs: bool = False,
) -> Job:
    """Constructs the remote job used to release exclusive access to the data of the target session for the specified
    manager ID.

    This worker function is used as part of the overall lock release step to efficiently construct and submit
    session data lock release jobs to the remote compute server.

    Args:
        project: The name of the project under which the target session was acquired.
        animal: The ID of the animal that participated in the target session.
        session: The name of the session for which to release the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target session's
            data.
        manager_id: The unique identifier of the process that calls this function.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.

    Returns:
        The initialized Job instance for the constructed and submitted session data lock release job.
    """
    # Resolves the job name and its remote working directory.
    job_name = f"{session}_lock_release"
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

    # Configures the job to use the sl-shared-assets library installed on the server to release the exclusive access to
    # the session's data from the specified manager process
    job.add_command(f"sl-manage session -sp {session_folder} -pdr {server.processed_data_root} -id {manager_id} unlock")

    # If the function is configured to remove job logs after runtime, adds a command to delete job working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {working_directory}")

    # Submits the remote job to the server and returns the updated Job object to caller
    return server.submit_job(job, verbose=False)


def _verify_lock_release_job(
    job: Job,
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
) -> None:
    """Verifies the outcome of a session data lock release job that ran on a remote compute server.

    This worker function is used as part of the overall session data lock release step to efficiently verify the
    outcome of completed session data lock release jobs submitted to the remote compute server.

    Args:
        job: The initialized Job instance for the lock release job to be verified.
        project: The name of the project under which the target session was acquired.
        animal: The ID of the animal that participated in the target session.
        session: The name of the session for which to release the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target session's
            data.
        manager_id: The unique identifier of the process that calls this function.
    """

    # Resolves the paths to the local and remote session lock files.
    local_working_directory = get_working_directory()
    remote_lock_path = server.raw_data_root.joinpath(project, animal, session, "tracking_data", "session_lock.yaml")
    local_lock_path = local_working_directory.joinpath(project, job.job_name, "manifest.feather", "session_lock.yaml")
    ensure_directory_exists(local_lock_path)

    # Pulls the remote session lock file to the local machine
    server.pull_file(
        local_file_path=local_lock_path,
        remote_file_path=remote_lock_path,
    )

    # Ensures that the caller process has exclusive access to session's data. This raises an error if the expectation
    # is violated.
    lock = SessionLock(file_path=local_lock_path)
    try:
        lock.check_owner(manager_id=manager_id)
    except Exception:
        # Since the lock is expected to be released, the 'success' outcome of this check is failing with an exception.
        # If the lock has been released successfully, removes the local working directory.
        delete_directory(local_lock_path.parent)
        return
    else:
        message = (
            f"Failed to release the session data lock from the manager process {manager_id}. Check the job logs "
            f"stored on the remote server for the details on the error that prevented releasing the lock."
        )
        console.error(message=message, error=RuntimeError)


def _release_session_lock(
    manifest: ProjectManifest,
    project: str,
    sessions: tuple[str, ...],
    server: Server,
    manager_id: int,
    keep_job_logs: bool = False,
) -> None:
    """Releases exclusive access to the target sessions' data if it is currently held by the specified manager.

    This function is used to release the lock after it has been acquired via the _acquire_session_lock() function
    runtime. Releasing the lock allows other manager processes to acquire the lock and work with the sessions' data.

    Args:
        manifest: The ProjectManifest instance containing session metadata.
        project: The name of the project under which the target sessions were acquired.
        sessions: The sessions for which to release the exclusive data access rights.
        server: The Server class instance that manages access to the remote server that stores the target sessions'
            data.
        manager_id: The unique identifier of the process that calls this function.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.
    """
    # Pre-creates a list of animal IDs for each session to be processed
    animals = [manifest.get_animal_for_session(session=session) for session in sessions]

    # Constructs and submits remote processing jobs to the server
    jobs = []
    for session, animal in tqdm(
        zip(sessions, animals), total=len(sessions), desc="Submitting session lock release jobs", unit="job"
    ):
        jobs.append(
            _construct_lock_release_job(
                project=project,
                animal=animal,
                session=session,
                server=server,
                manager_id=manager_id,
                keep_job_logs=keep_job_logs,
            )
        )

    completed_jobs = []
    with tqdm(total=len(jobs), desc="Waiting for the session lock release jobs to complete", unit="job") as pbar:
        for index, job in enumerate(jobs):
            # Waits for each job to complete. Ensures that each job is verified exactly once.
            if not server.job_complete(job=job) or index in completed_jobs:
                continue

            _verify_lock_release_job(
                job=job,
                project=project,
                animal=animals[index],
                session=sessions[index],
                server=server,
                manager_id=manager_id,
            )

            # Ensures that this job is not processed again as part of this function's cycle
            completed_jobs.append(index)

            # Increments the progress bar
            pbar.update()


def _construct_checksum_resolution_pipeline(
    manifest: ProjectManifest,
    project: str,
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
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
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

    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Parses the path to the session directory on the remote server.
    animal = manifest.get_animal_for_session(session=session)
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        server=server,
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

    # Resolves the name and working directory for the job.
    job_name = f"{session}_checksum"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Generates the remote job header and configures it to run behavior processing.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpus_to_use=1,
        ram_gb=17,
        time_limit=20,
    )

    # Resolves additional flags for the processing CLI.
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"
    recalculate_command = ""
    if recreate_checksum:
        recalculate_command = "-rc"

    # Instructs the server to execute the target processing pipeline.
    job.add_command(
        f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
        f"{tracker_command} checksum {recalculate_command}"
    )

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.CHECKSUM
    )
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_checksum", TrackerFileNames.CHECKSUM)

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
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
        reset_tracker: Determines whether to reset the processing tracker for the pipeline before executing the
            processing. This option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
    """

    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Extracts the ID of the animal that performed the session.
    animal = manifest.get_animal_for_session(session=session)

    # Parses the path to the session directory on the remote server.
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        server=server,
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

    # Resolves the name and working directory for the job.
    job_name = f"{session}_preparation"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)

    # Generates the remote job header and configures it to run behavior processing.
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

    # Resolves additional flags for the processing CLI.
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"

    # Instructs the server to execute the target processing pipeline.
    job.add_command(
        f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
        f"{tracker_command} prepare"
    )

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.PREPARATION
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_preparation", TrackerFileNames.PREPARATION
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
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    reprocess: bool = False,
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the behavior processing pipeline for the
    target session.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
        reset_tracker: Determines whether to reset the processing tracker for the pipeline before executing the
            processing. This option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
    """

    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Extracts additional metadata about the processed session.
    animal = manifest.get_animal_for_session(session=session)
    system = manifest.get_system_for_session(session=session)

    # Parses the path to the session directory on the remote server.
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        server=server,
        pipeline=ProcessingPipelines.BEHAVIOR,
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

    # Resolves additional shared flags for the processing CLI.
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"

    # Different acquisition systems require slightly different stack of Job objects, so the processing graph is
    # purpose-built for each acquisition system.
    stage_1 = []
    if system == AcquisitionSystems.MESOSCOPE_VR:
        # All processing jobs are intended to run in parallel with no cross-hierarchical dependencies.

        # Runtime data processing job
        job_name = f"{session}_runtime_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=2,
            ram_gb=4,
            time_limit=90,
        )
        # Note, the tracker reset command is only included with the job that is issued first. Otherwise, multiple jobs
        # resetting the tracker in rapid succession may overwrite legitimate job completion data written to the tracker
        # file by the jobs that complete quickly.
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 1 "
            f"{tracker_command} runtime"
        )
        stage_1.append((job, working_directory))

        # Face camera processing job
        job_name = f"{session}_face_camera_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=90,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 51  "
            f"camera"
        )
        stage_1.append((job, working_directory))

        # Left camera processing job
        job_name = f"{session}_left_camera_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 62 "
            f"camera"
        )
        stage_1.append((job, working_directory))

        # Right camera processing job
        job_name = f"{session}_right_camera_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 73 "
            f"camera"
        )
        stage_1.append((job, working_directory))

        # Actor microcontroller data processing job
        job_name = f"{session}_actor_microcontroller_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=5,
            ram_gb=10,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 101  "
            f"microcontroller"
        )
        stage_1.append((job, working_directory))

        # Sensor microcontroller data processing job
        job_name = f"{session}_sensor_microcontroller_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=15,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 152  "
            f"microcontroller"
        )
        stage_1.append((job, working_directory))

        # Encoder microcontroller data processing job
        job_name = f"{session}_encoder_microcontroller_processing"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=180,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.processed_data_root} -j 7 -id {manager_id} -l 203  "
            f"microcontroller"
        )
        stage_1.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.BEHAVIOR
    )
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_behavior", TrackerFileNames.BEHAVIOR)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
        jobs={1: tuple(stage_1)},
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
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    manager_id: int,
    configuration_file: str = "GCaMP6f_CA1_SD.yaml",
    plane_count: int = 3,
    reprocess: bool = False,
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the single-day suite2p processing pipeline
    for the target session.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the sl-suite2p single-day pipeline.
        plane_count: The number of imaging planes in the processed cell activity movie.
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
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

    # Extracts additional metadata about the processed session.
    animal = manifest.get_animal_for_session(session=session)

    # Parses the path to the session directory on the remote server.
    remote_session_path = server.raw_data_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not _check_session_eligibility(
            manifest=manifest,
            project=project,
            session=session,
            server=server,
            pipeline=ProcessingPipelines.SUITE2P,
            supported_systems={AcquisitionSystems.MESOSCOPE_VR},
            supported_sessions={SessionTypes.MESOSCOPE_EXPERIMENT},
            allow_reprocessing=reprocess,
            configuration_file=configuration_file,
    ):
        # If the session is not eligible, skips processing the session.
        return None

    # Resolves additional shared flags for the processing CLI.
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"
    configuration_command = f"-i {server.suite2p_configurations_directory.joinpath(configuration_file)}"
    job_command = f"-j {plane_count + 2}"  # Static +2 is for binarization and combination steps.

    # Precreates the iterables to store stage jobs
    stage_1 = []
    stage_2 = []
    stage_3 = []

    # Stage 1: Binarization
    job_name = f"{session}_ss2p_binarization"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=1,
        ram_gb=10,
        time_limit=180,
    )
    # Note, reset tracker command is only issued as part of the binarization processing stage.
    job.add_command(
        f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
        f"-pdr {server.processed_data_root} -id {manager_id} {job_command} {tracker_command} -b"
    )
    stage_1.append((job, working_directory))

    # Stage 2: Plane processing
    for plane in range(plane_count):
        job_name = f"{session}_ss2p_plane_{plane}"
        working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
        server.create_directory(remote_path=working_directory)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="suite2p",
            cpus_to_use=30,
            ram_gb=80,
            time_limit=180,
        )
        job.add_command(
            f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
            f"-pdr {server.processed_data_root} -id {manager_id} {job_command} -p -t {plane}"
        )
        stage_2.append((job, working_directory))

    # Stage 3: Combination
    job_name = f"{session}_ss2p_combination"
    working_directory = get_remote_job_work_directory(server=server, job_name=job_name)
    server.create_directory(remote_path=working_directory)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=1,
        ram_gb=30,
        time_limit=180,
    )
    job.add_command(
        f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
        f"-pdr {server.processed_data_root} -id {manager_id} {job_command} -c"
    )
    stage_3.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(
        project, animal, session, "tracking_data", TrackerFileNames.SUITE2P
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_ss2p_sd_processing", TrackerFileNames.SUITE2P
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
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


def _execute_pipelines(
        pipelines: tuple[ProcessingPipeline],
        batch_size: int,
        stage_name: str,
        poll_delay: int = 30,
) -> tuple[int, int, int]:
    """Executes the input pipelines as sequential batches.

    This worker function is used by the main process_project_data() function to efficiently execute batches of
    processing pipelines.

    Args:
        pipelines: The ProcessingPipelines to be executed for the current processing stage.
        batch_size: The maximum number of pipelines to be executed concurrently.
        stage_name: The name of the current processing stage.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of three integer values. The first value specifies the number of input pipelines that has been
        completed successfully. The second value specifies the number of failed pipelines. The third value specifies
        the number of aborted pipelines.

    """

    # If the list of pipelines is empty, returns 0 for all count updates.
    if not pipelines:
        return 0, 0, 0

    # Initializes counters
    uncompleted_count = len(pipelines)
    successful_count = 0
    failed_count = 0
    aborted_count = 0

    # Tracks which pipelines have been counted using their index
    counted_indices = set()

    # Splits the overall sequence of pipelines into batches
    batches = tuple(chunk_iterable([out for out in enumerate(pipelines)], batch_size))

    # Initializes a timer to delay repeated pipeline status checks
    delay_timer = PrecisionTimer("s")

    # Executes the current processing stage with a progress bar
    with tqdm(total=len(pipelines), desc=f"Executing {stage_name} pipelines", unit="pipeline") as pbar:

        # Processes each batch sequentially (one at a time)
        for batch in batches:
            batch_complete = False

            # Processes the current batch until all batch pipelines are completed
            while not batch_complete:
                batch_complete = True

                for idx, pipeline in batch:
                    # Check if the pipeline is still running
                    if pipeline.is_running:
                        pipeline.runtime_cycle()
                        batch_complete = False

                    # If the pipeline status changes to one of the completed status codes and the pipeline is not yet
                    # counted, updates the counters
                    if idx not in counted_indices:
                        if pipeline.pipeline_status == ProcessingStatus.FAILED:
                            failed_count += 1
                            uncompleted_count -= 1
                            counted_indices.add(idx)
                            pbar.update()
                        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
                            successful_count += 1
                            uncompleted_count -= 1
                            counted_indices.add(idx)
                            pbar.update()
                        elif pipeline.pipeline_status == ProcessingStatus.ABORTED:
                            aborted_count += 1
                            uncompleted_count -= 1
                            counted_indices.add(idx)
                            pbar.update()

                # Reruns the pipeline resolution cycle every poll_delay seconds to avoid overwhelming the
                # communication line.
                if not batch_complete:
                    delay_timer.delay_noblock(delay=poll_delay, allow_sleep=True)

    return successful_count, failed_count, aborted_count


def process_project_data(
        project: str,
        sessions: list[str] | tuple[str, ...] | None = None,
        animals: list[str | int] | tuple[str | int, ...] | set[str] | None = None,
        management_batch_size: int = 1,
        processing_batch_size: int = 4,
        process_checksum: bool = False,
        prepare_sessions: bool = False,
        process_behavior: bool = False,
        process_suite2p: bool = False,
        update_manifest: bool = False,
        reprocess: bool = False,
        keep_job_logs: bool = False,
        force_lock: bool = False,
        recalculate_checksum: bool = False,
        reset_trackers: bool = False,
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
        animals: An iterable of animal IDs to process as part of this runtime. If this optional argument is not
            provided, the function automatically processes the data for all animals participating in the project. Note,
            the animal ID filtering is applied after initially selecting the sessions according to the 'sessions'
            argument value.
        management_batch_size: The number of processing pipelines that can be submitted to the remote compute server at
            a time when running session data management pipelines. These pipelines are primarily limited by the I/O
            speed of the slow 'storage' server volume.
        processing_batch_size: Same as 'management_batch_size', but works with data processing pipelines that work with
            the data stored on the fast 'working' volume of the server. These pipelines are primarily limited by the
            available RAM / CPU resources rather than the fast drive I/O speed.
        process_checksum: Determines whether to recreate or verify the raw data integrity checksum for the target
            sessions as part of this runtime.
        prepare_sessions: Determines whether to prepare the target sessions for data processing as part of this runtime.
            Executing this pipeline is a prerequisite for running all data processing pipelines other than the checksum
            processing pipeline.
        process_behavior: Determines whether to execute the behavior data processing pipeline.
        process_suite2p: Determines whether to execute the single-day suite2p data processing pipeline.
        update_manifest: Determines whether to regenerate the project manifest file before resolving the processing
            pipeline graph. Generally, this is not required for most use cases, as all processing pipelines
            automatically update the project manifest as part of their runtime.
        reprocess: Determines whether to reprocess the sessions that have already been processed. This setting applies
            to all requested processing pipelines.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each processing pipeline completes successfully. If the pipeline fails, the job logs are kept regardless
            of the value of this argument.
        force_lock: Determines whether to forcibly acquire the exclusive session data access lock for all processed
            sessions, even if it is currently held by a different manager process. This argument should be disabled for
            most runtimes, as session locking is a critical safety mechanism used to prevent data corruption. This
            argument should only be used when recovering from improper runtime termination errors.
        reset_trackers: Determines whether to reset the processing tracker file for each processing pipeline to be
            executed. This argument should only be used when recovering from improper runtime termination errors and
            should be disabled for most runtimes.
        recalculate_checksum: Determines whether to regenerate and overwrite the raw data integrity checksum, instead
            of verifying its integrity, for each target session. This argument is only used when the 'process_checksum'
            argument is set to True.
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

    # Initializes a delay timer to support better visual separation of various terminal printouts and progress bars.
    delay_timer = PrecisionTimer("s")

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

    # If optional animal filtering is enabled, filters the resolved list of sessions to only include the sessions
    # performed by the requested animals.
    if animals is not None:
        animals = set([str(animal) for animal in animals])  # Converts to a string set for efficient lookup
        filtered_sessions = []
        for session in sessions:
            if manifest.get_animal_for_session(session) in animals:
                filtered_sessions.append(session)
        sessions = tuple(filtered_sessions)

    # Generates the unique identifier for this runtime
    manager_id = generate_manager_id()

    # Tracks all sessions that have been locked across all processing phases
    locked_sessions: set[str] = set()

    # Tracks all pipelines executed across all processing phases for final outcome reporting
    all_pipelines: list[ProcessingPipeline] = []

    # Tracks processing overall statistics
    total_successful = 0
    total_failed = 0
    total_aborted = 0

    # PHASE 1: CHECKSUM
    if process_checksum:
        console.echo(message="Phase 1: Checksum Resolution", level=LogLevel.INFO)

        # Ensures the visual separation between terminal printouts
        delay_timer.delay_noblock(delay=1, allow_sleep=True)

        # Resolves the checksum processing graph
        checksum_pipelines = []
        checksum_sessions = set()
        for session in tqdm(sessions, desc="Resolving the checksum processing graph", unit="session"):
            pipeline = _construct_checksum_resolution_pipeline(
                manifest=manifest,
                server=server,
                manager_id=manager_id,
                project=project,
                session=session,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
                recreate_checksum=recalculate_checksum,
                reset_tracker=reset_trackers,
            )
            if pipeline is not None:
                checksum_pipelines.append(pipeline)
                checksum_sessions.add(session)
                all_pipelines.append(pipeline)

        if checksum_pipelines:
            # Ensures that the manager process holds the session data locks for all sessions to be processed
            sessions_to_lock = checksum_sessions - locked_sessions
            if sessions_to_lock:
                _acquire_session_lock(
                    manifest=manifest,
                    project=project,
                    sessions=tuple(sorted(sessions_to_lock)),
                    server=server,
                    manager_id=manager_id,
                    force=force_lock,
                    keep_job_logs=keep_job_logs,
                )
                locked_sessions.update(sessions_to_lock)

            # Executes checksum pipelines and saves the runtime statistics data
            success, failed, aborted = _execute_pipelines(
                pipelines=tuple(checksum_pipelines),
                batch_size=management_batch_size,
                stage_name="checksum",
                poll_delay=5,
            )
            total_successful += success
            total_failed += failed
            total_aborted += aborted

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

            # Refreshes the local manifest file to include the processing outcome data
            fetch_remote_project_manifest(project=project, server=server)
            manifest = ProjectManifest(manifest_file=manifest_path)

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

    # PHASE 2: PREPARATION
    if prepare_sessions:
        console.echo(message="Phase 2: Processing Preparation", level=LogLevel.INFO)

        # Ensures the visual separation between terminal printouts
        delay_timer.delay_noblock(delay=1, allow_sleep=True)

        # Resolves the session data preparation graph
        prep_pipelines = []
        prep_sessions = set()
        for session in tqdm(sessions, desc="Resolving the processing preparation graph", unit="session"):
            pipeline = _construct_preparation_pipeline(
                manifest=manifest,
                server=server,
                manager_id=manager_id,
                project=project,
                session=session,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
                reset_tracker=reset_trackers,
            )
            if pipeline is not None:
                prep_pipelines.append(pipeline)
                prep_sessions.add(session)
                all_pipelines.append(pipeline)

        if prep_pipelines:
            # Ensures that the manager process holds the session data locks for all sessions to be processed
            sessions_to_lock = prep_sessions - locked_sessions
            if sessions_to_lock:
                _acquire_session_lock(
                    manifest=manifest,
                    project=project,
                    sessions=tuple(sorted(sessions_to_lock)),
                    server=server,
                    manager_id=manager_id,
                    force=force_lock,
                    keep_job_logs=keep_job_logs,
                )
                locked_sessions.update(sessions_to_lock)

            # Executes preparation pipelines and saves the runtime data
            success, failed, aborted = _execute_pipelines(
                pipelines=tuple(prep_pipelines),
                batch_size=management_batch_size,
                stage_name="preparation",
                poll_delay=5,
            )
            total_successful += success
            total_failed += failed
            total_aborted += aborted

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

            # Refreshes the local manifest file to include the processing outcome data
            fetch_remote_project_manifest(project=project, server=server)
            manifest = ProjectManifest(manifest_file=manifest_path)

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

    # PHASE 3: DATA PROCESSING
    if process_behavior or process_suite2p:
        console.echo(message="Phase 3: Data Processing", level=LogLevel.INFO)

        # Ensures the visual separation between terminal printouts
        delay_timer.delay_noblock(delay=1, allow_sleep=True)

        # Build processing pipelines
        processing_pipelines = []
        processing_sessions = set()

        for session in tqdm(sessions, desc="Resolving the data processing graph", unit="session"):
            # Behavior pipeline
            if process_behavior:
                pipeline = _construct_behavior_processing_pipeline(
                    manifest=manifest,
                    server=server,
                    manager_id=manager_id,
                    project=project,
                    session=session,
                    reprocess=reprocess,
                    keep_job_logs=keep_job_logs,
                    reset_tracker=reset_trackers,
                )
                if pipeline is not None:
                    processing_pipelines.append(pipeline)
                    processing_sessions.add(session)
                    all_pipelines.append(pipeline)

            # Suite2p pipeline
            if process_suite2p:
                pipeline = _construct_suite2p_processing_pipeline(
                    manifest=manifest,
                    server=server,
                    manager_id=manager_id,
                    project=project,
                    session=session,
                    configuration_file=suite2p_configuration_file,
                    plane_count=plane_count,
                    reprocess=reprocess,
                    keep_job_logs=keep_job_logs,
                    reset_tracker=reset_trackers,
                )
                if pipeline is not None:
                    processing_pipelines.append(pipeline)
                    processing_sessions.add(session)
                    all_pipelines.append(pipeline)

        if processing_pipelines:
            # Ensures that the manager process holds the session data locks for all sessions to be processed
            sessions_to_lock = processing_sessions - locked_sessions
            if sessions_to_lock:
                _acquire_session_lock(
                    manifest=manifest,
                    project=project,
                    sessions=tuple(sorted(sessions_to_lock)),
                    server=server,
                    manager_id=manager_id,
                    force=force_lock,
                    keep_job_logs=keep_job_logs,
                )
                locked_sessions.update(sessions_to_lock)

            # Executes processing pipelines and saves the runtime statistics
            success, failed, aborted = _execute_pipelines(
                pipelines=tuple(processing_pipelines),
                batch_size=processing_batch_size,
                stage_name="data processing",
                poll_delay=10,
            )
            total_successful += success
            total_failed += failed
            total_aborted += aborted

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

            # Refreshes the local manifest file to include the processing outcome data
            fetch_remote_project_manifest(project=project, server=server)
            manifest = ProjectManifest(manifest_file=manifest_path)

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

    # Checks if any processing was done
    if not all_pipelines:
        message = (
            f"All target sessions for project '{project}' have been excluded from all supported processing pipelines. "
            f"Processing: Aborted."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Creates a visual separation between the final processing outcome message and any progress bars used during
    # processing
    delay_timer.delay_noblock(delay=1, allow_sleep=True)

    # Displays the overall processing summary message
    message = (
        f"Project '{project}' data: Processed. Successfully completed {total_successful} pipelines, "
        f"failed {total_failed} pipelines, and aborted {total_aborted} pipelines. "
        f"The details about the processing outcome for each processed session are available below:"
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for all pipelines
    for pipeline in all_pipelines:
        if pipeline.pipeline_status == ProcessingStatus.FAILED:
            message = (
                f"The {pipeline.pipeline_type} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Failed."
            )
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = (
                f"The {pipeline.pipeline_type} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Complete."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
        elif pipeline.pipeline_status == ProcessingStatus.ABORTED:
            message = (
                f"The {pipeline.pipeline_type} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Aborted."
            )
            console.echo(message=message, level=LogLevel.WARNING)

    # Release all session locks acquired during processing
    if locked_sessions:
        _release_session_lock(
            manifest=manifest,
            project=project,
            sessions=tuple(sorted(locked_sessions)),
            server=server,
            manager_id=manager_id,
            keep_job_logs=keep_job_logs,
        )

    console.echo(message=f"Processing: Complete.", level=LogLevel.SUCCESS)
