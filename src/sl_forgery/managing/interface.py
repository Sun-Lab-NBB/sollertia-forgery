"""This module provides the interface functions for using the assets from this package while working with the
Sun lab's remote compute servers.
"""

from typing import TYPE_CHECKING

from tqdm import tqdm
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
    delay_timer,
    delay_terminal,
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


def _discover_adoption_candidates(project: str, server: Server) -> tuple[SessionMetadata, ...]:
    """Discovers the sessions potentially available for adoption by scanning the project's directory on the remote
    server for session_data.yaml files.

    Args:
        project: The name of the project for which to discover sessions.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        A tuple of SessionMetadata instances representing all discovered sessions.
    """
    project_path = server.shared_storage_root.joinpath(project)

    # Uses the server to find all session_data.yaml files in the project directory
    console.echo(message=f"Discovering '{project}' project's sessions available for adoption...", level=LogLevel.INFO)
    delay_terminal()

    # Builds a list of SessionMetadata instances for all discovered sessions
    discovered_sessions: list[SessionMetadata] = []
    for animal_dir in tqdm(
        server.list_directory(remote_path=project_path), desc="Evaluating animal directories", unit="directory"
    ):
        animal_path = project_path.joinpath(animal_dir)

        # Skips non-directory entries (like manifest files)
        if not server.is_directory(remote_path=animal_path):
            continue

        # Finds valid sessions (those containing session_data.yaml files)
        discovered_sessions.extend(
            SessionMetadata(session=session_dir, animal=animal_dir)
            for session_dir in server.list_directory(remote_path=animal_path)
            if server.exists(remote_path=animal_path.joinpath(session_dir, "raw_data", "session_data.yaml"))
        )

    return tuple(discovered_sessions)


def _execute_adoption_jobs(
    sessions: list[SessionMetadata],
    project: str,
    server: Server,
    *,
    keep_job_logs: bool = False,
    poll_delay: int = 10,
) -> tuple[tuple[SessionMetadata, JobStatus], ...]:
    """Executes the session data adoption jobs for the specified sessions.

    Notes:
        Unlike other pipelines, adoption does not use the ProcessingPipeline instance, since most necessary tracking
        assets are not available until the sessions are owned (adopted) by the calling user.

        The adoption jobs are executed sequentially due to being primarily I/O bound.

    Args:
        sessions: The list of SessionMetadata instances that define the sessions to adopt.
        project: The name of the project containing the sessions.
        server: The Server instance used to communicate with the remote compute server.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job fails, its logs are kept regardless of this argument's value.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of (SessionMetadata, JobStatus) pairs representing the outcome of each adoption job.
    """
    results: list[tuple[SessionMetadata, JobStatus]] = []

    with tqdm(total=len(sessions), desc="Executing session adoption jobs", unit="session") as pbar:
        for session_metadata in sessions:
            # Resolves the source and destination paths. Limits the adoption process to the raw_data directory.
            source_path = server.shared_storage_root.joinpath(
                project, session_metadata.animal, session_metadata.session, "raw_data"
            )
            destination_path = server.user_working_root.joinpath(
                project, session_metadata.animal, session_metadata.session, "raw_data"
            )

            # Resolves the job's name and working directory.
            job_name = f"{session_metadata.session}_adoption"
            working_directory = get_remote_job_work_directory(
                server=server, job_name=job_name, pipeline_name=ProcessingPipelines.ADOPTION
            )

            # Creates and configures the adoption job.
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
            job.add_command(f"sl-process transfer -sp {source_path} -dp {destination_path}")

            # Submits the job to the server.
            job = server.submit_job(job=job, verbose=False)

            # Waits for the job to complete.
            while True:
                job_status = server.get_job_status(slurm_job_id=int(job.job_id))
                if job_status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    break
                delay_timer.delay(delay=poll_delay, allow_sleep=True, block=False)

            # Records the outcome for this session.
            results.append((session_metadata, job_status))

            # Removes job logs if configured to do so and the job completed successfully.
            if job_status == JobStatus.COMPLETED and not keep_job_logs:
                server.remove(remote_path=working_directory, recursive=True, is_dir=True)

            pbar.update()

    return tuple(results)


def _delete_remote_session_data(
    manifest: ProjectManifest,
    sessions: list[SessionMetadata],
    project: str,
    server: Server,
    *,
    keep_job_logs: bool = False,
    poll_delay: int = 10,
) -> tuple[tuple[SessionMetadata, JobStatus], ...]:
    """Deletes the specified sessions from the user's working directory.

    This function generates and submits the session data deletion jobs using SLURM and verifies that they successfully
    delete the target sessions.

    Args:
        manifest: The ProjectManifest instance that stores the processed project's metadata.
        sessions: The list of SessionMetadata instances that define the sessions to delete.
        project: The name of the project containing the sessions.
        server: The Server instance used to communicate with the remote compute server.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job fails, its logs are kept regardless of this argument's value.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of (SessionMetadata, JobStatus) pairs representing the outcome of each deletion job.
    """
    results: list[tuple[SessionMetadata, JobStatus]] = []

    with tqdm(total=len(sessions), desc="Executing session deletion jobs", unit="session") as pbar:
        for session_metadata in sessions:
            # Resolves the path to the session directory using the manifest.
            animal = manifest.get_animal_for_session(session=session_metadata.session)
            session_path = server.user_working_root.joinpath(project, animal, session_metadata.session)

            # Resolves the job's name and working directory.
            job_name = f"{session_metadata.session}_deletion"
            working_directory = get_remote_job_work_directory(
                server=server, job_name=job_name, pipeline_name=ProcessingPipelines.ADOPTION
            )

            # Creates and configures the deletion job.
            job = Job(
                job_name=job_name,
                output_log=working_directory.joinpath("output.txt"),
                error_log=working_directory.joinpath("errors.txt"),
                working_directory=working_directory,
                conda_environment="forge",
                cpu_threads=1,
                ram=4,
                time=30,
            )
            job.add_command(f"sl-process transfer -sp {session_path} -rm")

            # Submits the job to the server.
            job = server.submit_job(job=job, verbose=False)

            # Waits for the job to complete.
            while True:
                job_status = server.get_job_status(slurm_job_id=int(job.job_id))
                if job_status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    break
                delay_timer.delay(delay=poll_delay, allow_sleep=True, block=False)

            # Records the outcome for this session.
            results.append((session_metadata, job_status))

            # Removes job logs if configured to do so and the job completed successfully.
            if job_status == JobStatus.COMPLETED and not keep_job_logs:
                server.remove(remote_path=working_directory, recursive=True, is_dir=True)

            pbar.update()

    return tuple(results)


def _delete_sessions_for_readoption(
    sessions: list[SessionMetadata],
    project: str,
    server: Server,
    *,
    keep_job_logs: bool = False,
    poll_delay: int = 10,
) -> tuple[tuple[SessionMetadata, JobStatus], ...]:
    """Deletes existing session data to prepare for clean re-adoption.

    This function is similar to _delete_remote_session_data but does not require a manifest, as the animal IDs are
    already available from the discovery phase. It is used during the adoption process when re-adopting sessions that
    were previously adopted.

    Args:
        sessions: The list of SessionMetadata instances that define the sessions to delete before re-adoption.
        project: The name of the project containing the sessions.
        server: The Server instance used to communicate with the remote compute server.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job fails, its logs are kept regardless of this argument's value.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of (SessionMetadata, JobStatus) pairs representing the outcome of each deletion job.
    """
    results: list[tuple[SessionMetadata, JobStatus]] = []

    with tqdm(total=len(sessions), desc="Executing pre-adoption deletion jobs", unit="session") as pbar:
        for session_metadata in sessions:
            # Resolves the path to the session directory using the session metadata directly.
            session_path = server.user_working_root.joinpath(project, session_metadata.animal, session_metadata.session)

            # Resolves the job's name and working directory.
            job_name = f"{session_metadata.session}_preadopt_deletion"
            working_directory = get_remote_job_work_directory(
                server=server, job_name=job_name, pipeline_name=ProcessingPipelines.ADOPTION
            )

            # Creates and configures the deletion job.
            job = Job(
                job_name=job_name,
                output_log=working_directory.joinpath("output.txt"),
                error_log=working_directory.joinpath("errors.txt"),
                working_directory=working_directory,
                conda_environment="forge",
                cpu_threads=1,
                ram=4,
                time=30,
            )
            job.add_command(f"sl-process transfer -sp {session_path} -rm")

            # Submits the job to the server.
            job = server.submit_job(job=job, verbose=False)

            # Waits for the job to complete.
            while True:
                job_status = server.get_job_status(slurm_job_id=int(job.job_id))
                if job_status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    break
                delay_timer.delay(delay=poll_delay, allow_sleep=True, block=False)

            # Records the outcome for this session.
            results.append((session_metadata, job_status))

            # Removes job logs if configured to do so and the job completed successfully.
            if job_status == JobStatus.COMPLETED and not keep_job_logs:
                server.remove(remote_path=working_directory, recursive=True, is_dir=True)

            pbar.update()

    return tuple(results)


def _construct_checksum_resolution_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    *,
    reprocess: bool = False,
    keep_job_logs: bool = False,
    recreate_checksum: bool = False,
) -> ProcessingPipeline | str:
    """Generates and returns the ProcessingPipeline instance used to execute the raw data integrity checksum resolution
    pipeline for the target session.

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
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline.
        Otherwise, returns a string describing why the session was excluded from processing.
    """
    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Parses the path to the session's directory on the remote server.
    animal = manifest.get_animal_for_session(session=session)
    remote_session_path = server.user_working_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    exclusion_reason = check_session_eligibility(
        manifest=manifest,
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
    )
    if exclusion_reason is not None:
        return exclusion_reason

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


def adopt_project(
    project: str,
    *,
    repeat_adoption: bool = False,
    keep_job_logs: bool = False,
) -> None:
    """Discovers and adopts all unadopted project's sessions from the remote compute server's shared storage directory.

    This function serves as the entry point for adopting project data for further processing and analysis. It scans the
    project's directory on the shared server's volume, identifies sessions that have not yet been adopted (copied to the
    user's working directory), and executes the adoption pipeline followed by the data integrity verification
    pipeline for each session.

    Notes:
        Unlike manage_project_data, this function does not require a project manifest file. It discovers sessions
        directly from the project's directory structure on the remote server.

    Args:
        project: The name of the project to adopt.
        repeat_adoption: Determines whether to re-adopt sessions that have already been adopted. If False (default),
            already-adopted sessions are skipped during the adoption stage.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each pipeline completes successfully. If the pipeline fails, the job logs are kept regardless of this
            argument's value.
    """
    console.echo(message=f"Initializing '{project}' project adoption...", level=LogLevel.INFO)

    # Establishes communication with the compute server
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # STAGE 0: DISCOVERY
    # Discovers all sessions hypothetically available for adoption
    discovered_sessions = _discover_adoption_candidates(project=project, server=server)

    # Builds the list of sessions that need to be adopted by filtering out already-adopted sessions. If session
    # discovery did not yield any sessions, this loop is skipped, producing an empty list. Sessions that are being
    # re-adopted are tracked separately to allow for pre-adoption deletion.
    sessions_to_adopt: list[SessionMetadata] = []
    sessions_to_readopt: list[SessionMetadata] = []
    skipped_sessions: list[SessionMetadata] = []
    for session_metadata in tqdm(discovered_sessions, desc="Resolving adoption tasks", unit="task"):
        # Checks if the session has already been adopted by verifying the presence of the session's data and the
        # integrity checksum file in the user's working directory. A session is considered adopted if the
        # ax_checksum.txt file exists in the user's directory.
        checksum_file_path = server.user_working_root.joinpath(
            project, session_metadata.animal, session_metadata.session, "raw_data", "ax_checksum.txt"
        )
        is_already_adopted = server.exists(remote_path=checksum_file_path)

        # Unless the function is instructed to repeat the adoption process, skips reprocessing already adopted sessions.
        if is_already_adopted and not repeat_adoption:
            skipped_sessions.append(session_metadata)
            continue

        # Tracks sessions requiring re-adoption separately - they will need pre-adoption deletion.
        if is_already_adopted and repeat_adoption:
            sessions_to_readopt.append(session_metadata)
        else:
            sessions_to_adopt.append(session_metadata)

    # If all sessions are already adopted and re-adoption is not requested, ends the runtime early.
    delay_terminal()
    if not sessions_to_adopt and not sessions_to_readopt:
        console.echo(
            message=(
                f"All {len(discovered_sessions)} '{project}' project's sessions are already adopted. Processing: "
                f"Complete."
            ),
            level=LogLevel.SUCCESS,
        )
        return

    # STAGE 1: PRE-ADOPTION CLEANUP (only for sessions being re-adopted)
    # Deletes existing session data before re-adoption to ensure a clean slate.
    deletion_failed_sessions: list[SessionMetadata] = []
    if sessions_to_readopt:
        console.echo(message="Stage 1: Pre-Adoption Cleanup...", level=LogLevel.INFO)
        delay_terminal()

        deletion_results = _delete_sessions_for_readoption(
            sessions=sessions_to_readopt,
            project=project,
            server=server,
            keep_job_logs=keep_job_logs,
            poll_delay=10,
        )

        # Only proceed with adoption for sessions where deletion succeeded.
        for session_metadata, job_status in deletion_results:
            if job_status == JobStatus.COMPLETED:
                sessions_to_adopt.append(session_metadata)
            else:
                deletion_failed_sessions.append(session_metadata)

        delay_terminal()

    # STAGE 2: ADOPTION
    console.echo(message="Stage 2: Adoption...", level=LogLevel.INFO)
    delay_terminal()

    # Executes adoption jobs sequentially (batch size of 1)
    adoption_results = _execute_adoption_jobs(
        sessions=sessions_to_adopt,
        project=project,
        server=server,
        keep_job_logs=keep_job_logs,
        poll_delay=10,
    )
    delay_terminal()

    # STAGE 3: INTEGRITY VERIFICATION
    console.echo(message="Stage 3: Integrity Verification...", level=LogLevel.INFO)
    delay_terminal()

    # Refreshes the user-specific project manifest file and pulls it to the local machine. Following the adoption
    # procedure, the manifest should reflect the state of the newly adopted sessions.
    resolve_project_manifest(project=project, server=server, generate=True)

    # Loads the manifest data
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Constructs checksum verification pipelines for sessions that were successfully adopted.
    # Tracks both successfully constructed pipelines and exclusion reasons for sessions that couldn't be processed.
    checksum_pipelines: dict[str, ProcessingPipeline] = {}  # Maps session names to pipelines
    checksum_exclusions: dict[str, str] = {}  # Maps session names to exclusion reasons

    for session_metadata, job_status in adoption_results:
        # Only verifies the checksum for successfully adopted sessions.
        if job_status != JobStatus.COMPLETED:
            continue

        result = _construct_checksum_resolution_pipeline(
            manifest=manifest,
            project=project,
            session=session_metadata.session,
            server=server,
            reprocess=True,  # Technically it is impossible for the newly adopted sessions to be already verified.
            keep_job_logs=keep_job_logs,
            recreate_checksum=False,
        )
        if isinstance(result, str):
            checksum_exclusions[session_metadata.session] = result
        else:
            checksum_pipelines[session_metadata.session] = result

    # Tracks checksum execution results
    total_checksum_successful = 0
    total_checksum_failed = 0

    if checksum_pipelines:
        # Executes checksum pipelines sequentially
        # noinspection PyTypeChecker
        total_checksum_successful, total_checksum_failed = execute_pipelines(
            pipelines=tuple(checksum_pipelines.values()),
            stage_name="checksum",
            poll_delay=10,
        )
        delay_terminal()

        # Refreshes the manifest to include verification results
        resolve_project_manifest(project=project, server=server, generate=True)

    # Determines the runtime's outcome
    total_adoption_successful = sum(1 for _, status in adoption_results if status == JobStatus.COMPLETED)
    total_adoption_failed = len(adoption_results) - total_adoption_successful
    total_deletion_failed = len(deletion_failed_sessions)
    total_skipped = len(skipped_sessions)

    # Displays the overall processing summary message
    delay_terminal()
    message = (
        f"Project '{project}': Adopted. Successfully completed {total_checksum_successful} pipelines, "
        f"failed {total_adoption_failed + total_checksum_failed + total_deletion_failed} pipelines, "
        f"skipped {total_skipped} sessions. "
        f"The details about the processing outcome for each session are available below:"
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for each session (adoption + checksum)
    for session_metadata, job_status in adoption_results:
        if job_status == JobStatus.COMPLETED:
            # Checks if the session was excluded from checksum processing
            if session_metadata.session in checksum_exclusions:
                message = (
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
                    f"Adopted, but excluded from verification. Reason: {checksum_exclusions[session_metadata.session]}"
                )
                console.echo(message=message, level=LogLevel.WARNING)
                continue

            # For successful adoptions, finds the corresponding checksum pipeline result
            checksum_result = checksum_pipelines.get(session_metadata.session)
            if checksum_result is not None and checksum_result.pipeline_status == ProcessingStatus.SUCCEEDED:
                message = (
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
                    f"Adopted and verified."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
            elif checksum_result is not None and checksum_result.pipeline_status == ProcessingStatus.FAILED:
                message = (
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
                    f"Adopted, but failed verification."
                )
                console.echo(message=message, level=LogLevel.ERROR)
        else:
            message = (
                f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
                f"Failed to be adopted (SLURM status: {job_status})."
            )
            console.echo(message=message, level=LogLevel.ERROR)

    # Prints skipped sessions (already adopted)
    for session_metadata in skipped_sessions:
        message = (
            f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
            f"Already adopted, skipped."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Prints sessions where pre-adoption deletion failed (blocked from re-adoption)
    for session_metadata in deletion_failed_sessions:
        message = (
            f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
            f"Pre-adoption deletion failed, re-adoption aborted."
        )
        console.echo(message=message, level=LogLevel.ERROR)

    console.echo(message="Adoption: Complete.", level=LogLevel.SUCCESS)


def manage_project_data(
    manifest_path: Path,
    project: str,
    sessions: tuple[SessionMetadata, ...],
    *,
    verify_checksum: bool = False,
    recompute_checksum: bool = False,
    delete_sessions: bool = False,
    keep_job_logs: bool = False,
) -> None:
    """Resolves and executes the necessary data management pipelines for the target project.

    This function allows managing the sessions adopted by the user for further processing. Specifically, it can be used
    to either verify or recompute the session's data integrity checksum or to delete the adopted session's data from the
    user's working directory.

    Notes:
        The verify_checksum/recompute_checksum operations and delete_sessions operation are mutually exclusive.
        If delete_sessions is True, checksum operations are skipped.

    Args:
        manifest_path: The path to the project's manifest .feather file.
        project: The name of the project whose data to manage.
        sessions: A tuple of SessionMetadata instances defining the project's sessions to manage.
        verify_checksum: Determines whether to verify the data integrity checksum for the target sessions.
        recompute_checksum: Determines whether to recompute (regenerate) the data integrity checksum for the target
            sessions. This overwrites the existing checksum stored in the ax_checksum.txt file for each session.
        delete_sessions: Determines whether to delete the target sessions from the user's working directory.
            If True, checksum operations are skipped.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each pipeline completes successfully. If the pipeline fails, the job logs are kept regardless of this
            argument's value.
    """
    # Ensures that the caller has specified the processing pipeline to execute.
    if not verify_checksum and not recompute_checksum and not delete_sessions:
        console.error(
            message=(
                f"Unable to manage the '{project}' project's data, as no management pipeline was selected. "
                f"Call the data management CLI command with --verify-checksum (-vc), --recompute-checksum (-rc), or "
                f"--delete (-d) flag to execute the desired management pipeline."
            ),
            error=RuntimeError,
        )

    console.echo(message=f"Initializing '{project}' project data management...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Loads the project's manifest data.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # SESSION DELETION PIPELINE
    if delete_sessions:
        console.echo(message="Pipeline: Deletion...", level=LogLevel.INFO)
        delay_terminal()

        # Executes the deletion jobs.
        deletion_results = _delete_remote_session_data(
            manifest=manifest,
            sessions=list(sessions),
            project=project,
            server=server,
            keep_job_logs=keep_job_logs,
            poll_delay=10,
        )
        delay_terminal()

        # Refreshes the locally stored manifest file to reflect the processing outcome.
        resolve_project_manifest(project=project, server=server, generate=True)

        # Calculates the deletion outcome statistics.
        total_deleted = sum(1 for _, status in deletion_results if status == JobStatus.COMPLETED)
        total_failed = len(deletion_results) - total_deleted

        # Displays the overall deletion summary message.
        delay_terminal()
        message = (
            f"Project '{project}' session deletion: Complete. Deleted: {total_deleted}, Failed: {total_failed}. "
            f"The details about the processing outcome for each session are available below:"
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Prints detailed results for each deletion job.
        for session_metadata, job_status in deletion_results:
            if job_status == JobStatus.COMPLETED:
                message = (
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': Deleted."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
            else:
                message = (
                    f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
                    f"Failed to be deleted (SLURM status: {job_status})."
                )
                console.echo(message=message, level=LogLevel.ERROR)

        console.echo(message="Management: Complete.", level=LogLevel.SUCCESS)
        return

    # CHECKSUM VERIFICATION/RECOMPUTATION PIPELINE
    console.echo(
        message=f"Pipeline: Integrity {'Recomputation' if recompute_checksum else 'Verification'}...",
        level=LogLevel.INFO,
    )
    delay_terminal()

    # Constructs checksum processing pipelines for all processed sessions. Tracks both successfully constructed
    # pipelines and exclusion reasons for sessions that couldn't be processed.
    checksum_pipelines: list[ProcessingPipeline] = []
    checksum_exclusions: dict[str, tuple[SessionMetadata, str]] = {}  # Maps session names to (metadata, reason)

    for session_metadata in tqdm(sessions, desc="Resolving the checksum processing graph", unit="session"):
        result = _construct_checksum_resolution_pipeline(
            manifest=manifest,
            project=project,
            session=session_metadata.session,
            server=server,
            reprocess=verify_checksum or recompute_checksum,
            keep_job_logs=keep_job_logs,
            recreate_checksum=recompute_checksum,
        )
        if isinstance(result, str):
            checksum_exclusions[session_metadata.session] = (session_metadata, result)
        else:
            checksum_pipelines.append(result)

    # Executes checksum pipelines sequentially.
    total_successful = 0
    total_failed = 0
    if checksum_pipelines:
        total_successful, total_failed = execute_pipelines(
            pipelines=tuple(checksum_pipelines),
            stage_name="checksum",
            poll_delay=5,
        )
        delay_terminal()

        # Refreshes the manifest to include the processing results.
        resolve_project_manifest(project=project, server=server, generate=True)

    # Creates a visual separation before the final summary.
    delay_terminal()

    # Displays the overall processing summary message.
    operation_name = "recomputation" if recompute_checksum else "verification"
    total_excluded = len(checksum_exclusions)
    message = (
        f"Project '{project}' checksum {operation_name}: Complete. "
        f"Processed: {total_successful}, Failed: {total_failed}, Excluded: {total_excluded}. "
        f"The details about the processing outcome for each session are available below:"
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for checksum pipelines.
    for pipeline in checksum_pipelines:
        if pipeline.pipeline_status == ProcessingStatus.FAILED:
            message = (
                f"Session '{pipeline.session}' performed by animal '{pipeline.animal}': "
                f"Checksum {operation_name} failed."
            )
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = (
                f"Session '{pipeline.session}' performed by animal '{pipeline.animal}': "
                f"Checksum {operation_name} complete."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

    # Prints exclusion reasons for sessions that couldn't be processed
    for session_metadata, reason in checksum_exclusions.values():
        message = (
            f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
            f"Excluded. Reason: {reason}"
        )
        console.echo(message=message, level=LogLevel.WARNING)

    console.echo(message="Management: Complete.", level=LogLevel.SUCCESS)
