"""This module provides the interface functions for all Sun lab data processing pipelines. The assets from this module
are designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks.
"""

from typing import TYPE_CHECKING

from tqdm import tqdm
from sl_shared_assets import (
    SessionTypes,
    ProcessingStatus,
    ProcessingTracker,
    AcquisitionSystems,
    get_working_directory,
    get_server_configuration,
)
from ataraxis_base_utilities import LogLevel, console

from ..server import Job, Server, ProcessingPipeline, get_remote_job_work_directory
from ..managing import ProjectManifest, resolve_project_manifest
from ..shared_assets import (
    SessionMetadata,
    ProcessingTrackers,
    ProcessingPipelines,
    delay_terminal,
    execute_pipelines,
    check_session_eligibility,
)

if TYPE_CHECKING:
    from pathlib import Path


def _construct_behavior_processing_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    *,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | str:
    """Generates and returns the ProcessingPipeline instance used to execute the behavior processing pipeline for the
    target session.

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

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline.
        Otherwise, returns a string describing why the session was excluded from processing.
    """
    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Extracts additional metadata about the processed session.
    animal = manifest.get_animal_for_session(session=session)
    system = manifest.get_system_for_session(session=session)

    # Parses the path to the session directory on the remote server.
    remote_session_path = server.shared_storage_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    exclusion_reason = check_session_eligibility(
        manifest=manifest,
        session=session,
        pipeline=ProcessingPipelines.BEHAVIOR,
        server=server,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={
            SessionTypes.LICK_TRAINING,
            SessionTypes.RUN_TRAINING,
            SessionTypes.MESOSCOPE_EXPERIMENT,
        },
        allow_reprocessing=reprocess,
    )
    if exclusion_reason is not None:
        return exclusion_reason

    # Different acquisition systems require slightly different stack of Job objects, so the processing graph is
    # purpose-built for each acquisition system.
    stage_1 = []
    if system == AcquisitionSystems.MESOSCOPE_VR:
        # All processing jobs are intended to run in parallel with no cross-hierarchical dependencies.

        # Runtime data processing job
        job_name = f"{session}_runtime_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=2,
            ram=4,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 1 runtime")
        stage_1.append((job, working_directory))

        # Face camera processing job
        job_name = f"{session}_face_camera_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=30,
            ram=90,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 51 camera")
        stage_1.append((job, working_directory))

        # Left camera processing job
        job_name = f"{session}_left_camera_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=30,
            ram=60,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 62 camera")
        stage_1.append((job, working_directory))

        # Right camera processing job
        job_name = f"{session}_right_camera_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=30,
            ram=60,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 73 camera")
        stage_1.append((job, working_directory))

        # Actor microcontroller data processing job
        job_name = f"{session}_actor_microcontroller_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=5,
            ram=10,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 101 microcontroller")
        stage_1.append((job, working_directory))

        # Sensor microcontroller data processing job
        job_name = f"{session}_sensor_microcontroller_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=15,
            ram=60,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 152 microcontroller")
        stage_1.append((job, working_directory))

        # Encoder microcontroller data processing job
        job_name = f"{session}_encoder_microcontroller_processing"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpu_threads=30,
            ram=200,
            time=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -id {job_id} -l 203 microcontroller")
        stage_1.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = server.shared_storage_root.joinpath(
        project, animal, session, "tracking_data", ProcessingTrackers.BEHAVIOR
    )
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_behavior", ProcessingTrackers.BEHAVIOR)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.BEHAVIOR,
        server=server,
        data_path=remote_session_path,
        jobs={1: tuple(stage_1)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def _construct_suite2p_processing_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
    *,
    configuration_file: str = "GCaMP6f_CA1_SD.yaml",
    plane_count: int = 3,
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | str:
    """Generates and returns the ProcessingPipeline instance used to execute the single-day suite2p processing pipeline
    for the target session.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        session: The name of the session to process with the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the sl-suite2p single-day pipeline.
        plane_count: The number of imaging planes in the processed cell activity movie.
        reprocess: Determines whether to reprocess the session if it has already been processed with the target
            processing pipeline.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if the target session can be processed with this pipeline.
        Otherwise, returns a string describing why the session was excluded from processing.
    """
    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Extracts additional metadata about the processed session.
    animal = manifest.get_animal_for_session(session=session)

    # Parses the path to the session directory on the remote server.
    remote_session_path = server.shared_storage_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    configuration_path = server.suite2p_configurations_directory.joinpath(configuration_file)
    exclusion_reason = check_session_eligibility(
        manifest=manifest,
        session=session,
        pipeline=ProcessingPipelines.SUITE2P,
        server=server,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={SessionTypes.MESOSCOPE_EXPERIMENT},
        allow_reprocessing=reprocess,
        configuration_path=configuration_path,
    )
    if exclusion_reason is not None:
        return exclusion_reason

    # Resolves additional shared flags for the processing CLI.
    configuration_command = f"-i {server.suite2p_configurations_directory.joinpath(configuration_file)}"

    # Precreates the iterables to store stage jobs
    stage_1 = []
    stage_2 = []
    stage_3 = []

    # Stage 1: Binarization
    job_name = f"{session}_ss2p_binarization"
    job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.SUITE2P
    )
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpu_threads=1,
        ram=10,
        time=180,
    )
    job.add_command(f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} -id {job_id} -b")
    stage_1.append((job, working_directory))

    # Stage 2: Plane processing
    for plane in range(plane_count):
        job_name = f"{session}_ss2p_plane_{plane}"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.SUITE2P
        )
        server.create(remote_path=working_directory, is_dir=True)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="suite2p",
            cpu_threads=30,
            ram=80,
            time=180,
        )
        job.add_command(
            f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} -id {job_id} -p -t {plane}"
        )
        stage_2.append((job, working_directory))

    # Stage 3: Combination
    job_name = f"{session}_ss2p_combination"
    job_id = ProcessingTracker.generate_job_id(session_path=remote_session_path, job_name=job_name)
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.SUITE2P
    )
    server.create(remote_path=working_directory, is_dir=True)
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpu_threads=1,
        ram=30,
        time=180,
    )
    job.add_command(f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} -id {job_id} -c")
    stage_3.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = server.shared_storage_root.joinpath(
        project, animal, session, "tracking_data", ProcessingTrackers.SUITE2P
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_ss2p_sd_processing", ProcessingTrackers.SUITE2P
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.SUITE2P,
        server=server,
        data_path=remote_session_path,
        jobs={1: tuple(stage_1), 2: tuple(stage_2), 3: tuple(stage_3)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def process_project_data(
    manifest_path: Path,
    project: str,
    sessions: tuple[SessionMetadata, ...],
    *,
    process_behavior: bool = False,
    process_suite2p: bool = False,
    reprocess: bool = False,
    keep_job_logs: bool = False,
    suite2p_configuration_file: str = "GCaMP6f_CA1_SD.yaml",
    plane_count: int = 3,
    processing_batch_size: int = 4,
) -> None:
    """Resolves and executes the necessary data processing pipelines for the target project.

    This function acts as the main entry point for all data processing in the Sun lab. As part of its runtime, it first
    determines which processing pipelines need to be executed for each session of the project. Then it efficiently
    executes these pipelines on the remote compute server by iteratively submitting batches of remote compute jobs
    to the server.

    Args:
        manifest_path: The path to the project's manifest .feather file.
        project: The name of the project whose data to process.
        sessions: A tuple of SessionMetadata instances defining the project's sessions to process.
        process_behavior: Determines whether to execute the behavior data processing pipeline.
        process_suite2p: Determines whether to execute the single-day suite2p data processing pipeline.
        reprocess: Determines whether to reprocess the sessions that have already been processed. This setting applies
            to all requested processing pipelines.
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
        processing_batch_size: The number of processing pipelines that can be submitted to the remote compute server at
            a time. These pipelines are primarily limited by the available RAM / CPU resources.
    """
    # Ensures that the caller has specified the processing pipeline to execute.
    if not process_behavior and not process_suite2p:
        console.error(
            message=(
                f"Unable to process the '{project}' project's data, as no processing pipeline was selected. "
                f"Call the data processing CLI command with --behavior (-b) or --suite2p (-s) flag to execute "
                f"the desired processing pipeline."
            ),
            error=RuntimeError,
        )

    console.echo(message=f"Initializing '{project}' project data processing...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Loads the project's manifest data.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Tracks all pipelines executed across all processing phases for final outcome reporting
    all_pipelines: list[ProcessingPipeline] = []

    # DATA PROCESSING PIPELINE
    console.echo(message="Pipeline: Data Processing...", level=LogLevel.INFO)
    delay_terminal()

    # Build processing pipelines
    processing_pipelines: list[ProcessingPipeline] = []
    processing_exclusions: dict[str, tuple[SessionMetadata, str]] = {}  # Maps session names to (metadata, reason)

    for session_metadata in tqdm(sessions, desc="Resolving the data processing graph", unit="session"):
        # Behavior pipeline
        if process_behavior:
            result = _construct_behavior_processing_pipeline(
                manifest=manifest,
                project=project,
                session=session_metadata.session,
                server=server,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
            )
            if isinstance(result, str):
                processing_exclusions[session_metadata.session] = (session_metadata, result)
            else:
                processing_pipelines.append(result)
                all_pipelines.append(result)

        # Suite2p pipeline
        if process_suite2p:
            result = _construct_suite2p_processing_pipeline(
                manifest=manifest,
                project=project,
                session=session_metadata.session,
                server=server,
                configuration_file=suite2p_configuration_file,
                plane_count=plane_count,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
            )
            if isinstance(result, str):
                processing_exclusions[session_metadata.session] = (session_metadata, result)
            else:
                processing_pipelines.append(result)
                all_pipelines.append(result)

    # Executes processing pipelines
    total_successful = 0
    total_failed = 0
    if processing_pipelines:
        total_successful, total_failed = execute_pipelines(
            pipelines=tuple(processing_pipelines),
            batch_size=processing_batch_size,
            stage_name="data processing",
            poll_delay=10,
        )
        delay_terminal()

        # Refreshes the manifest to include the processing results.
        resolve_project_manifest(project=project, server=server, generate=True)

    # Creates a visual separation before the final summary.
    delay_terminal()

    # Displays the overall processing summary message.
    message = (
        f"Project '{project}' data processing: Complete. Successfully completed {total_successful} pipelines, "
        f"failed {total_failed} pipelines. "
        f"The details about the processing outcome for each session are available below:"
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for all pipelines
    for pipeline in all_pipelines:
        if pipeline.pipeline_status == ProcessingStatus.FAILED:
            message = (
                f"The {pipeline.pipeline} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}': Failed."
            )
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = (
                f"The {pipeline.pipeline} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}': Complete."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

    # Prints exclusion reasons for sessions that couldn't be processed
    for session_name, (session_metadata, reason) in processing_exclusions.items():
        message = f"Session '{session_name}' performed by animal '{session_metadata.animal}': Excluded ({reason})."
        console.echo(message=message, level=LogLevel.WARNING)

    console.echo(message="Processing: Complete.", level=LogLevel.SUCCESS)
