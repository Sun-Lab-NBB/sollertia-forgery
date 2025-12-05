"""This module contains the bindings for all Sun lab data processing pipelines. The assets from this module are
designed to process the data stored on the remote Sun lab compute server and assume that the server is properly
configured to execute all data processing tasks.
"""

from tqdm import tqdm
from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    SessionTypes,
    ProcessingStatus,
    AcquisitionSystems,
    get_working_directory,
    get_server_configuration,
)
from ataraxis_base_utilities import LogLevel, console, chunk_iterable

from ..server import Job, Server, ProcessingPipeline, get_remote_job_work_directory
from ..managing import ProjectManifest, resolve_project_manifest
from ..shared_assets import ProcessingTrackers, ProcessingPipelines, check_session_eligibility
from ..managing.interface import _construct_checksum_resolution_pipeline


def _construct_behavior_processing_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
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
    remote_session_path = server.shared_storage_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    if not check_session_eligibility(
        manifest=manifest,
        project=project,
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
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
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
            f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 1 "
            f"{tracker_command} runtime"
        )
        stage_1.append((job, working_directory))

        # Face camera processing job
        job_name = f"{session}_face_camera_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=90,
            time_limit=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 51 camera")
        stage_1.append((job, working_directory))

        # Left camera processing job
        job_name = f"{session}_left_camera_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 62 camera")
        stage_1.append((job, working_directory))

        # Right camera processing job
        job_name = f"{session}_right_camera_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 73 camera")
        stage_1.append((job, working_directory))

        # Actor microcontroller data processing job
        job_name = f"{session}_actor_microcontroller_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=5,
            ram_gb=10,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 101 microcontroller"
        )
        stage_1.append((job, working_directory))

        # Sensor microcontroller data processing job
        job_name = f"{session}_sensor_microcontroller_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=15,
            ram_gb=60,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 152 microcontroller"
        )
        stage_1.append((job, working_directory))

        # Encoder microcontroller data processing job
        job_name = f"{session}_encoder_microcontroller_processing"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.BEHAVIOR
        )
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="forge",
            cpus_to_use=30,
            ram_gb=200,
            time_limit=90,
        )
        job.add_command(
            f"sl-behavior -sp {remote_session_path} -pdr {server.shared_working_root} -j 7 -l 203 microcontroller"
        )
        stage_1.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = server.shared_storage_root.joinpath(
        project, animal, session, "tracking_data", ProcessingTrackers.BEHAVIOR
    )
    local_tracker_path = local_working_directory.joinpath(project, f"{session}_behavior", ProcessingTrackers.BEHAVIOR)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
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

    return pipeline


def _construct_suite2p_processing_pipeline(
    manifest: ProjectManifest,
    project: str,
    session: str,
    server: Server,
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
    remote_session_path = server.shared_storage_root.joinpath(project, animal, session)

    # Determines whether the session is eligible for processing.
    configuration_path = server.suite2p_configurations_directory.joinpath(configuration_file)
    if not check_session_eligibility(
        manifest=manifest,
        project=project,
        session=session,
        pipeline=ProcessingPipelines.SUITE2P,
        server=server,
        supported_systems={AcquisitionSystems.MESOSCOPE_VR},
        supported_sessions={SessionTypes.MESOSCOPE_EXPERIMENT},
        allow_reprocessing=reprocess,
        configuration_path=configuration_path,
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
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.SUITE2P
    )
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpus_to_use=1,
        ram_gb=10,
        time_limit=180,
    )
    # Note, reset tracker command is only issued as part of the binarization processing stage.
    job.add_command(
        f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
        f"-pdr {server.shared_working_root} {job_command} {tracker_command} -b"
    )
    stage_1.append((job, working_directory))

    # Stage 2: Plane processing
    for plane in range(plane_count):
        job_name = f"{session}_ss2p_plane_{plane}"
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
            cpus_to_use=30,
            ram_gb=80,
            time_limit=180,
        )
        job.add_command(
            f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
            f"-pdr {server.shared_working_root} {job_command} -p -t {plane}"
        )
        stage_2.append((job, working_directory))

    # Stage 3: Combination
    job_name = f"{session}_ss2p_combination"
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
        cpus_to_use=1,
        ram_gb=30,
        time_limit=180,
    )
    job.add_command(
        f"ss2p run {configuration_command} -w -1 sl-single-day -sp {remote_session_path} "
        f"-pdr {server.shared_working_root} {job_command} -c"
    )
    stage_3.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = server.shared_storage_root.joinpath(
        project, animal, session, "tracking_data", ProcessingTrackers.SUITE2P
    )
    local_tracker_path = local_working_directory.joinpath(
        project, f"{session}_ss2p_sd_processing", ProcessingTrackers.SUITE2P
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
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
    process_behavior: bool = False,
    process_suite2p: bool = False,
    update_manifest: bool = False,
    reprocess: bool = False,
    keep_job_logs: bool = False,
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
    configuration = get_server_configuration(service=True)
    server = Server(configuration=configuration)

    # Initializes a delay timer to support better visual separation of various terminal printouts and progress bars.
    delay_timer = PrecisionTimer("s")

    # Resolves the project manifest file, optionally regenerating it on the remote server.
    resolve_project_manifest(project=project, server=server, generate=update_manifest)

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

            # Refreshes the local manifest file to include the processing outcome data.
            resolve_project_manifest(project=project, server=server, generate=False)
            manifest = ProjectManifest(manifest_file=manifest_path)

            # Ensures the visual separation between terminal printouts
            delay_timer.delay_noblock(delay=1, allow_sleep=True)

    # PHASE 2: DATA PROCESSING
    if process_behavior or process_suite2p:
        console.echo(message="Phase 2: Data Processing", level=LogLevel.INFO)

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

            # Refreshes the local manifest file to include the processing outcome data.
            resolve_project_manifest(project=project, server=server, generate=False)
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
                f"The {pipeline.pipeline} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Failed."
            )
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = (
                f"The {pipeline.pipeline} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Complete."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)
        elif pipeline.pipeline_status == ProcessingStatus.ABORTED:
            message = (
                f"The {pipeline.pipeline} processing pipeline for session '{pipeline.session}' "
                f"performed by animal '{pipeline.animal}' for '{pipeline.project}' project: Aborted."
            )
            console.echo(message=message, level=LogLevel.WARNING)

    console.echo(message="Processing: Complete.", level=LogLevel.SUCCESS)
