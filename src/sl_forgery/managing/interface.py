"""This module provides the interface functions for using the assets from this package while working with the
Sun lab's remote compute servers.
"""

from typing import TYPE_CHECKING

from tqdm import tqdm
from ataraxis_time import PrecisionTimer, TimerPrecisions
from sl_shared_assets import (
    SessionTypes,
    ProcessingStatus,
    ProcessingTracker,
    AcquisitionSystems,
    delete_directory,
    get_working_directory,
    get_server_configuration,
)
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..server import Job, Server, JobStatus, ProcessingPipeline, get_remote_job_work_directory
from ..shared_assets import (
    ProjectManifest,
    SessionMetadata,
    ManagingTrackers,
    ProcessingPipelines,
    execute_pipelines,
    check_session_eligibility,
)

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
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MANIFEST
    )

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


def _execute_adoption_jobs(
    sessions: list[SessionMetadata],
    project: str,
    server: Server,
    *,
    keep_job_logs: bool = False,
    poll_delay: int = 10,
) -> tuple[int, int]:
    """Executes the adoption jobs for the specified sessions sequentially (batch size of 1).

    Unlike other pipelines, adoption does not use ProcessingPipeline tracking since the manifest is not available
    until after adoption completes. Instead, this worker function submits and monitors SLURM jobs directly.

    Args:
        sessions: A list of SessionMetadata instances specifying the sessions to adopt.
        project: The name of the project containing the sessions.
        server: The Server instance used to communicate with the remote compute server.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job fails, its logs are kept regardless of this argument's value.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of two integers: (successful_count, failed_count).
    """
    if not sessions:
        return 0, 0

    successful_count = 0
    failed_count = 0

    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    with tqdm(total=len(sessions), desc="Executing adoption jobs", unit="session") as pbar:
        for session_metadata in sessions:
            # Resolves source and destination paths
            source_path = server.shared_storage_root.joinpath(
                project, session_metadata.animal, session_metadata.session
            )
            destination_path = server.user_working_root.joinpath(
                project, session_metadata.animal, session_metadata.session
            )

            # Resolves the job name and working directory
            job_name = f"{session_metadata.session}_adoption"
            working_directory = get_remote_job_work_directory(
                server=server, job_name=job_name, pipeline_name=ProcessingPipelines.ADOPTION
            )

            # Creates and configures the adoption job
            job = Job(
                job_name=job_name,
                output_log=working_directory.joinpath("output.txt"),
                error_log=working_directory.joinpath("errors.txt"),
                working_directory=working_directory,
                conda_environment="forge",
                cpu_threads=1,
                ram=20,
                time=60,
            )

            # Adds the transfer command
            job.add_command(f"sl-process transfer -sp {source_path} -dp {destination_path}")

            # Submits the job to the server
            job = server.submit_job(job=job, verbose=False)

            # Waits for the job to complete
            while True:
                job_status = server.get_job_status(slurm_job_id=int(job.job_id))
                if job_status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    break
                delay_timer.delay(delay=poll_delay, allow_sleep=True, block=False)

            # Checks the outcome and updates counters
            if job_status == JobStatus.COMPLETED:
                # If the job completed successfully, increments the successful count.
                successful_count += 1
                # Removes job logs if configured to do so
                if not keep_job_logs:
                    server.remove(remote_path=working_directory, recursive=True, is_dir=True)
            else:
                # Otherwise, increments the failed counter and notifies the user about the failed job.
                failed_count += 1
                console.echo(
                    message=(
                        f"Adoption job for session '{session_metadata.session}' performed by animal "
                        f"'{session_metadata.animal}': Failed (status: {job_status}). "
                        f"Check job logs at: {working_directory}"
                    ),
                    level=LogLevel.ERROR,
                )

            pbar.update()

    return successful_count, failed_count


def _discover_sessions_from_project_folder(project: str, server: Server) -> dict[str, list[str]]:
    """Discovers the sessions potentially available for adoption by scanning the project's directory on the remote
    server.

    This function recursively searches the project directory for session_data.yaml files to identify available
    sessions that can be adopted.

    Args:
        project: The name of the project for which to discover sessions.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        A dictionary mapping the unique animal identifiers to lists of session names.
    """
    project_path = server.shared_storage_root.joinpath(project)

    # Uses the server to find all session_data.yaml files in the project directory
    console.echo(message=f"Discovering '{project}' project's sessions on the remote server...", level=LogLevel.INFO)

    # Builds a dictionary that uses animal IDs as keys and lists all available sessions for each animal.
    animal_sessions: dict[str, list[str]] = {}
    for animal_dir in server.list_directory(remote_path=project_path):
        animal_path = project_path.joinpath(animal_dir)

        # Skips non-directory entries (like manifest files)
        if not server.is_directory(remote_path=animal_path):
            continue

        # Finds valid sessions (those containing session_data.yaml)
        valid_sessions = [
            session_dir
            for session_dir in server.list_directory(remote_path=animal_path)
            if server.exists(remote_path=animal_path.joinpath(session_dir, "raw_data", "session_data.yaml"))
        ]

        if valid_sessions:
            animal_sessions[animal_dir] = valid_sessions

    total_sessions = sum(len(sessions) for sessions in animal_sessions.values())
    console.echo(
        message=f"Discovered {total_sessions} sessions across {len(animal_sessions)} animals.",
        level=LogLevel.SUCCESS,
    )

    return animal_sessions


def _check_session_already_adopted(
    project: str,
    animal: str,
    session: str,
    server: Server,
) -> bool:
    """Returns True if the user has already adopted the target session.

    The session is considered adopted if its destination directory exists and contains the ax_checksum.txt file,
    which indicates that the data transfer has completed successfully.

    Args:
        project: The name of the project containing the session.
        animal: The unique identifier of the animal that performed the session.
        session: The name of the session to check.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        True if the session has already been adopted, False otherwise.
    """
    destination_path = server.user_working_root.joinpath(project, animal, session)
    checksum_file_path = destination_path.joinpath("raw_data", "ax_checksum.txt")

    return server.exists(remote_path=checksum_file_path)


def _construct_checksum_resolution_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    *,
    reprocess: bool = False,
    keep_job_logs: bool = False,
    recreate_checksum: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the raw data integrity checksum resolution
    pipeline for the target session.

    Notes:
        This pipeline only works with sessions stored under the user's server working directory.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.
        recreate_checksum: Determines whether to recalculate and overwrite the data integrity checksum stored in the
            session's 'raw data' directory instead of verifying its' integrity. This flag allows updating the checksum
            following expected changes to the session's raw data.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline. None,
        if the session is excluded from processing for any reason.
    """
    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Parses the path to the session's directory on the remote server.
    animal = manifest.get_animal_for_session(session=session)
    remote_session_path = server.user_working_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        pipeline=ProcessingPipelines.CHECKSUM,
        server=server,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={
            SessionTypes.LICK_TRAINING,
            SessionTypes.RUN_TRAINING,
            SessionTypes.MESOSCOPE_EXPERIMENT,
        },
        allow_reprocessing=recreate_checksum or reprocess,
    ):
        # If the session is not eligible, skips processing the session.
        return None

    # Resolves the name and working directory for the job.
    job_name = f"{session}_checksum"
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.CHECKSUM
    )

    # Generates the remote job header and configures it to run checksum verification.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="forge",
        cpu_threads=1,
        ram=20,
        time=40,
    )

    # Instructs the server to execute the target processing pipeline via the sl-process CLI.
    job.add_command(f"sl-process checksum -sp {remote_session_path} {'-rc' if recreate_checksum else ''}")

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = remote_session_path.joinpath("tracking_data", ManagingTrackers.CHECKSUM)
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_checksum", ManagingTrackers.CHECKSUM)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.CHECKSUM,
        server=server,
        data_path=remote_session_path,
        jobs={1: ((job, working_directory),)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def manage_project_data(
    project: str,
    sessions: tuple[SessionMetadata, ...],
    *,
    repeat_adoption: bool = False,
    repeat_checksum_verification: bool = False,
    keep_job_logs: bool = False,
    recalculate_checksum: bool = False,
) -> None:
    """Resolves and executes the necessary data adoption and management pipelines for the specified project.

    This function acts as the entry point for all data management operations in the Sun lab. It allows users to 'adopt'
    the project's data for further processing and analysis by copying it from the shared read-only repositories, and
    then verifies the integrity of the adopted data using checksum verification.

    Notes:
        The input sessions are expected to be pre-filtered before calling this function. The function resolves session
        paths relative to the shared storage root on the remote server, verifies the sessions exist, and proceeds with
        adoption and checksum verification.

    Args:
        project: The name of the project to work with.
        sessions: A tuple of SessionMetadata instances representing the sessions to process. These sessions are
            expected to be pre-filtered and valid.
        repeat_adoption: Determines whether to re-adopt sessions that have already been adopted. If False (default),
            already-adopted sessions are skipped during the adoption stage.
        repeat_checksum_verification: Determines whether to re-verify checksums for sessions that have already been
            verified. If False (default), already-verified sessions are skipped during the checksum stage.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each pipeline completes successfully. If the pipeline fails, the job logs are kept regardless of this
            argument's value.
        recalculate_checksum: Determines whether to regenerate and overwrite the raw data integrity checksum instead
            of verifying its integrity. Setting this to True implies repeat_checksum_verification=True.
    """
    if not sessions:
        console.echo(
            message=f"No sessions provided for project '{project}'. Management: Aborted.",
            level=LogLevel.WARNING,
        )
        return

    console.echo(message=f"Initializing project '{project}' data management...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Initializes a delay timer to support better visual separation of terminal printouts
    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    # Tracks checksum pipelines for final outcome reporting
    checksum_pipelines: list[ProcessingPipeline] = []

    # Tracks processing overall statistics
    total_adoption_successful = 0
    total_adoption_failed = 0
    total_checksum_successful = 0
    total_checksum_failed = 0

    # STAGE 1: ADOPTION
    console.echo(message="Stage 1: Session Adoption", level=LogLevel.INFO)
    delay_timer.delay(delay=1, allow_sleep=True, block=False)

    # Resolves and validates sessions from the shared storage, filtering out already adopted sessions if needed
    sessions_to_adopt: list[SessionMetadata] = []
    for session_metadata in tqdm(sessions, desc="Resolving session paths", unit="session"):
        # Checks if already adopted and skips if not repeating
        if not repeat_adoption and _check_session_already_adopted(
            project=project, animal=session_metadata.animal, session=session_metadata.session, server=server
        ):
            console.echo(
                message=(
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}' has already "
                    f"been adopted. Skipping. Use '--repeat-adoption' to force re-adoption."
                ),
                level=LogLevel.WARNING,
            )
            continue

        sessions_to_adopt.append(session_metadata)

    if sessions_to_adopt:
        # Executes adoption jobs sequentially (batch size of 1)
        total_adoption_successful, total_adoption_failed = _execute_adoption_jobs(
            sessions=sessions_to_adopt,
            project=project,
            server=server,
            keep_job_logs=keep_job_logs,
            poll_delay=5,
        )
        delay_timer.delay(delay=1, allow_sleep=True, block=False)

    # STAGE 2: CHECKSUM VERIFICATION
    console.echo(message="Stage 2: Checksum Verification", level=LogLevel.INFO)
    delay_timer.delay(delay=1, allow_sleep=True, block=False)

    # Refreshes the user-specific project manifest file and pulls it to the local machine
    resolve_project_manifest(project=project, server=server, generate=True)

    # Loads the manifest data
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Determines which sessions to verify: all adopted sessions from the input list
    sessions_to_verify: list[SessionMetadata] = [
        session_metadata
        for session_metadata in sessions
        if _check_session_already_adopted(
            project=project, animal=session_metadata.animal, session=session_metadata.session, server=server
        )
    ]

    # Recalculating checksum implies repeating the verification
    allow_reprocessing = repeat_checksum_verification or recalculate_checksum

    for session_metadata in tqdm(sessions_to_verify, desc="Resolving the checksum processing graph", unit="session"):
        pipeline = _construct_checksum_resolution_pipeline(
            manifest=manifest,
            project=project,
            session=session_metadata.session,
            server=server,
            reprocess=allow_reprocessing,
            keep_job_logs=keep_job_logs,
            recreate_checksum=recalculate_checksum,
        )
        if pipeline is not None:
            checksum_pipelines.append(pipeline)

    if checksum_pipelines:
        # Executes checksum pipelines sequentially (batch size of 1)
        total_checksum_successful, total_checksum_failed = execute_pipelines(
            pipelines=tuple(checksum_pipelines),
            stage_name="checksum",
            poll_delay=5,
        )
        delay_timer.delay(delay=1, allow_sleep=True, block=False)

        # Refreshes the manifest to include verification results
        resolve_project_manifest(project=project, server=server, generate=True)

    # Checks if any processing was done
    total_processed = len(sessions_to_adopt) + len(checksum_pipelines)
    if total_processed == 0:
        message = (
            f"All target sessions for project '{project}' have been excluded from all management pipelines. "
            f"Management: Aborted."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Creates a visual separation before the final summary
    delay_timer.delay(delay=1, allow_sleep=True, block=False)

    # Displays the overall processing summary message
    message = (
        f"Project '{project}' data: Managed. "
        f"Adoption: {total_adoption_successful} succeeded, {total_adoption_failed} failed. "
        f"Checksum: {total_checksum_successful} succeeded, {total_checksum_failed} failed."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for checksum pipelines
    for pipeline in checksum_pipelines:
        if pipeline.pipeline_status == ProcessingStatus.FAILED:
            message = (
                f"The {pipeline.pipeline} pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Failed."
            )
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = (
                f"The {pipeline.pipeline} pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Complete."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

    console.echo(message="Management: Complete.", level=LogLevel.SUCCESS)
