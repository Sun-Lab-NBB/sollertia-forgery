"""Provides interface functions for data management pipelines and tasks.

Notes:
    The assets from this module manage data stored on the remote compute server and assume the server is properly
    configured to execute all data management tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import (
    SessionTypes,
    AcquisitionSystems,
    ProcessingTrackers,
    get_working_directory,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

from ..server import (
    Job,
    Server,
    JobStatus,
    ProcessingPipeline,
    execute_pipelines,
    get_server_configuration,
    check_session_eligibility,
    get_remote_job_work_directory,
)
from ..pipelines import ProcessingPipelines
from ..shared_assets import DatasetSession, delay_timer, delay_terminal
from .project_manifest import ProjectManifest

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
    # Resolves the path to the local directory used to work with Sollertia data.
    local_working_directory = get_working_directory()

    # Resolves the local path where the manifest file will be stored.
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")
    ensure_directory_exists(local_manifest_path)

    # Resolves the path to the remote manifest file.
    remote_manifest_path = server.root.joinpath(project, f"{project}_manifest.feather")

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
        local_working_directory: The path to the local Sollertia working directory.

    Raises:
        RuntimeError: If the manifest generation job fails.
    """
    console.echo(message=f"Generating the manifest file for the '{project}' project on the remote server...")

    # Resolves the path to the project's directory on the remote compute server.
    project_storage_root = server.root.joinpath(project)

    # Resolves the job name and its remote log directory.
    job_name = f"{project}_manifest_generation"
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MANIFEST, base_path=project_storage_root
    )

    # Resolves the paths to the remote and local manifest generation tracker files.
    remote_manifest_tracker_path = project_storage_root.joinpath(ProcessingTrackers.MANIFEST)
    local_manifest_tracker_path = local_working_directory.joinpath(project, job_name, ProcessingTrackers.MANIFEST)
    ensure_directory_exists(local_manifest_tracker_path)

    # Generates the remote job header.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment=server.environment,
        cpu_threads=1,
        ram=1,
        time=20,
    )

    # Configures the job to call the appropriate CLI command.
    job.add_command(f"slf manifest generate -pp {project_storage_root}")

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
    tracker = ProcessingTracker.from_yaml(file_path=local_manifest_tracker_path)

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


def discover_project_sessions(project: str, server: Server) -> tuple[DatasetSession, ...]:
    """Discovers all sessions stored under the project's directory on the remote compute server's data root.

    Notes:
        This function explicitly skips dataset directories (those containing dataset_data.yaml) to avoid confusing
        dataset session hierarchies with actual animal/session directories.

    Args:
        project: The name of the project for which to discover sessions.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        A tuple of DatasetSession instances representing all discovered sessions.
    """
    project_path = server.root.joinpath(project)

    # Builds a list of DatasetSession instances for all discovered sessions.
    discovered_sessions: list[DatasetSession] = []
    for animal_dir in console.track(
        server.list_directory(remote_path=project_path), description="Evaluating animal directories", unit="directory"
    ):
        animal_path = project_path.joinpath(animal_dir)

        # Skips non-directory entries (like manifest files).
        if not server.is_directory(remote_path=animal_path):
            continue

        # Skips dataset directories (those containing dataset_data.yaml) to avoid confusing dataset session
        # hierarchies with actual animal/session directories.
        if server.exists(remote_path=animal_path.joinpath("dataset_data.yaml")):
            continue

        # Finds valid sessions (those containing session_data.yaml files).
        discovered_sessions.extend(
            DatasetSession(session=session_dir, animal=animal_dir)
            for session_dir in server.list_directory(remote_path=animal_path)
            if server.exists(remote_path=animal_path.joinpath(session_dir, "raw_data", "session_data.yaml"))
        )

    return tuple(discovered_sessions)


def discover_project_data(project: str) -> tuple[DatasetSession, ...]:
    """Discovers and reports all sessions stored under the project's directory on the remote compute server.

    This function serves as the entry point for discovering project data available on the remote server. It connects
    to the server, scans the project's directory under the data root, and prints the discovered sessions to the
    terminal.

    Args:
        project: The name of the project whose data to discover.

    Returns:
        A tuple of DatasetSession instances representing all discovered sessions.
    """
    console.echo(message=f"Discovering '{project}' project's sessions on the remote server...", level=LogLevel.INFO)

    # Establishes communication with the compute server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    try:
        discovered_sessions = discover_project_sessions(project=project, server=server)
    finally:
        server.close()

    delay_terminal()
    console.echo(
        message=f"Discovered {len(discovered_sessions)} session(s) for the '{project}' project:", level=LogLevel.INFO
    )
    for session_metadata in discovered_sessions:
        console.echo(message=f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}'.")

    return discovered_sessions


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
            session's 'raw data' directory instead of verifying its integrity. This flag allows updating the checksum
            following expected changes to the session's raw data.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline.
        Otherwise, returns a string describing why the session was excluded from processing.
    """
    # Resolves the path to the local Sollertia working directory.
    local_working_directory = get_working_directory()

    # Parses the path to the session's directory on the remote server.
    animal = manifest.get_animal_for_session(session=session)
    remote_session_path = server.root.joinpath(project, animal, session)

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

    # Resolves the name and log directory for the job.
    job_name = f"{session}_checksum"
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=str(remote_session_path))
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.CHECKSUM, base_path=remote_session_path
    )

    # Generates the remote job header and configures it to run checksum verification.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment=server.environment,
        cpu_threads=1,
        ram=20,
        time=40,
    )

    # Instructs the server to execute the target processing pipeline via the slf CLI.
    job.add_command(f"slf checksum -sp {remote_session_path} {'-rc' if recreate_checksum else ''}")

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = remote_session_path.joinpath("tracking_data", ProcessingTrackers.CHECKSUM)
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_checksum", ProcessingTrackers.CHECKSUM)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.CHECKSUM,
        server=server,
        data_path=remote_session_path,
        jobs={1: ((job, working_directory, job_id),)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def manage_project_data(
    manifest_path: Path,
    project: str,
    sessions: tuple[DatasetSession, ...],
    *,
    verify_checksum: bool = False,
    recompute_checksum: bool = False,
    keep_job_logs: bool = False,
) -> None:
    """Resolves and executes the necessary data management pipelines for the target project.

    This function allows managing the sessions stored on the remote compute server. Specifically, it can be used to
    verify or recompute the session's data integrity checksum.

    Args:
        manifest_path: The path to the project's manifest .feather file.
        project: The name of the project whose data to manage.
        sessions: A tuple of DatasetSession instances defining the project's sessions to manage.
        verify_checksum: Determines whether to verify the data integrity checksum for the target sessions.
        recompute_checksum: Determines whether to recompute (regenerate) the data integrity checksum for the target
            sessions. This overwrites the existing checksum stored in the ax_checksum.txt file for each session.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each pipeline completes successfully. If the pipeline fails, the job logs are kept regardless of this
            argument's value.
    """
    # Ensures that the caller has specified the processing pipeline to execute.
    if not verify_checksum and not recompute_checksum:
        console.error(
            message=(
                f"Unable to manage the '{project}' project's data, as no management pipeline was selected. "
                f"Call the data management CLI command with the --verify-checksum (-vc) or "
                f"--recompute-checksum (-rc) flag to execute the desired management pipeline."
            ),
            error=RuntimeError,
        )

    console.echo(message=f"Initializing '{project}' project data management...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Loads the project's manifest data.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # CHECKSUM VERIFICATION/RECOMPUTATION PIPELINE
    console.echo(
        message=f"Pipeline: Integrity {'Recomputation' if recompute_checksum else 'Verification'}...",
        level=LogLevel.INFO,
    )
    delay_terminal()

    # Constructs checksum processing pipelines for all processed sessions. Tracks both successfully constructed
    # pipelines and exclusion reasons for sessions that couldn't be processed.
    checksum_pipelines: list[ProcessingPipeline] = []
    checksum_exclusions: dict[str, tuple[DatasetSession, str]] = {}  # Maps session names to (metadata, reason)

    for session_metadata in console.track(
        sessions, description="Resolving the checksum processing graph", unit="session"
    ):
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

    # Prints exclusion reasons for sessions that couldn't be processed.
    for session_metadata, reason in checksum_exclusions.values():
        message = (
            f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}': "
            f"Excluded. Reason: {reason}"
        )
        console.echo(message=message, level=LogLevel.WARNING)

    console.echo(message="Management: Complete.", level=LogLevel.SUCCESS)
