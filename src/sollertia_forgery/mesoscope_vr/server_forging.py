"""Provides interface functions for dataset forging pipelines.

Notes:
    The assets from this module forge (assemble) datasets from processed data stored on the remote compute server
    and assume the server is properly configured to execute all forging tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    Directories,
    SessionTypes,
    AcquisitionSystems,
    ProcessingTrackers,
    filter_sessions,
    get_working_directory,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..server import Job, Server, JobStatus, get_server_configuration, get_remote_job_work_directory
from .forging import FORGING_JOB_NAME
from ..managing import ProjectManifest
from ..pipelines import ProcessingPipelines
from ..orchestration import (
    ProcessingPipeline,
    execute_pipelines,
    resolve_project_manifest,
    check_session_eligibility,
)
from ..shared_assets import DatasetData, DatasetSession, delay_timer, delay_terminal

if TYPE_CHECKING:
    from pathlib import Path


def _define_dataset_remote(
    project: str,
    dataset_name: str,
    filtered_sessions: set[DatasetSession],
    server: Server,
    *,
    keep_job_logs: bool = False,
) -> DatasetData:
    """Creates a new forged dataset on the remote compute server.

    This function submits a job to the remote server to create the dataset directory structure and metadata file,
    waits for the job to complete, and then loads the created DatasetData instance. The session type and acquisition
    system are derived from the first session's metadata on the remote server.

    Args:
        project: The name of the project from which the dataset's sessions originate.
        dataset_name: The unique name for the dataset to create.
        filtered_sessions: The set of DatasetSession instances representing the sessions to include in the dataset.
        server: The Server class instance that manages access to the remote server.
        keep_job_logs: Determines whether to keep completed job logs on the server.

    Returns:
        The initialized DatasetData instance loaded from the remote server.

    Raises:
        RuntimeError: If the dataset definition job fails.
    """
    # Resolves the path to the local Sollertia working directory.
    local_working_directory = get_working_directory()

    # Resolves paths. Note, the project root serves as both the session data root and the datasets root.
    project_root = server.root.joinpath(project)
    remote_dataset_path = project_root.joinpath(dataset_name)

    # Constructs the session specifications for the CLI command.
    session_specs = " ".join(f"-s {s.session}:{s.animal}" for s in filtered_sessions)

    # Resolves the job name and working directory.
    job_name = f"{dataset_name}_definition"
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.FORGING, base_path=project_root
    )

    # Creates the job.
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment=server.environment,
        cpu_threads=1,
        ram=4,
        time=10,
    )
    job.add_command(f"slf forge -dn {dataset_name} -pp {project_root} {session_specs}")

    # If configured to remove job logs after runtime, adds a command to delete the job's working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {working_directory}")

    # Submits the job to the server.
    console.echo(message=f"Submitting dataset definition job for '{dataset_name}'...")
    job = server.submit_job(job=job, verbose=False)

    # Waits for the server to complete the job.
    message = f"Waiting for dataset definition job with ID {job.job_id} to complete..."
    console.echo(message=message, level=LogLevel.INFO)
    while server.get_job_status(slurm_job_id=int(job.job_id)) in (JobStatus.PENDING, JobStatus.RUNNING):
        delay_timer.delay(delay=5, allow_sleep=True, block=False)

    # Verifies the dataset was created successfully by checking if the dataset_data.yaml file exists.
    remote_dataset_data_path = remote_dataset_path.joinpath("dataset_data.yaml")
    if not server.exists(remote_path=remote_dataset_data_path):
        message = (
            f"Dataset definition job failed. The dataset_data.yaml file was not created at "
            f"'{remote_dataset_data_path}'. Check the job logs for details."
        )
        console.error(message=message, error=RuntimeError)

    # Fetches the dataset_data.yaml file to the local machine.
    local_dataset_path = local_working_directory.joinpath(project, dataset_name)
    local_dataset_data_path = local_dataset_path.joinpath("dataset_data.yaml")
    local_dataset_path.mkdir(parents=True, exist_ok=True)
    server.pull(local_path=local_dataset_data_path, remote_path=remote_dataset_data_path)

    # Loads and returns the DatasetData instance.
    dataset = DatasetData.load(dataset_path=local_dataset_data_path)
    console.echo(
        message=f"Created dataset '{dataset_name}' with {len(filtered_sessions)} sessions.",
        level=LogLevel.SUCCESS,
    )
    return dataset


def _construct_cindra_multiday_pipeline(
    manifest: ProjectManifest,
    project: str,
    animal: str,
    sessions: list[str],
    dataset_name: str,
    server: Server,
    *,
    configuration_file: str = "GCaMP6f_CA1_MD.yaml",
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | str:
    """Generates and returns the ProcessingPipeline instance used to execute the multi-day cindra processing pipeline
    for the target set of sessions.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        animal: The unique identifier of the animal that performed the sessions processed by this pipeline.
        sessions: The list of session names to process using the target processing pipeline.
        dataset_name: The name of the dataset to process.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the cindra multi-day pipeline.
        reprocess: Determines whether to reprocess the dataset if it has already been processed with this pipeline.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if all sessions can be processed with this pipeline.
        Otherwise, returns a string describing why the dataset was excluded from processing.
    """
    # Resolves the path to the local Sollertia working directory.
    local_working_directory = get_working_directory()

    # Resolves the configuration path on the remote server.
    configuration_path = server.cindra_configurations_directory.joinpath(configuration_file)

    # Resolves full session paths using the project structure.
    project_root = server.root.joinpath(project)
    session_paths = [project_root.joinpath(animal, session) for session in sessions]

    # Builds CLI session path arguments for the new cindra interface.
    session_path_args = " ".join(f"-sp {path}" for path in session_paths)

    # Determines the main session (first after natural sort) for tracker storage.
    sorted_sessions = natsorted(sessions)
    main_session = sorted_sessions[0]
    main_session_path = project_root.joinpath(animal, main_session)
    # Cindra writes multi-recording output to ``processed_data/cindra/multi_recording/<dataset_name>/``. The
    # dataset directory here intentionally omits the animal_id prefix used on local disk because the server
    # execution context operates on a single animal at a time.
    main_session_multiday = main_session_path.joinpath(
        "processed_data", Directories.CINDRA, Directories.MULTI_RECORDING, dataset_name
    )

    # Validates each session for eligibility with the multiday pipeline.
    for session in sessions:
        exclusion_reason = check_session_eligibility(
            manifest=manifest,
            session=session,
            pipeline=ProcessingPipelines.CINDRA_MULTI_RECORDING,
            server=server,
            supported_systems={AcquisitionSystems.MESOSCOPE_VR},
            supported_sessions={SessionTypes.MESOSCOPE_EXPERIMENT},
            allow_reprocessing=reprocess,
            configuration_path=configuration_path,
        )
        if exclusion_reason is not None:
            return f"Session '{session}': {exclusion_reason}"

    # Precreates the iterables to store stage jobs.
    stage_1 = []
    stage_2 = []

    # Stage 1: Multi-day cell tracking (discovery).
    job_name = f"{dataset_name}_cindra_discovery"
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=str(main_session_multiday))
    working_directory = get_remote_job_work_directory(
        server=server,
        job_name=job_name,
        pipeline_name=ProcessingPipelines.CINDRA_MULTI_RECORDING,
        base_path=main_session_multiday,
    )
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment=server.environment,
        cpu_threads=30,
        ram=80,
        time=180,
    )
    job.add_command(f"slf forge -i {configuration_path} {session_path_args} -id {job_id} -d")
    stage_1.append((job, working_directory, job_id))

    # Stage 2: Across-day-tracked cell fluorescence extraction.
    for session in sessions:
        job_name = f"{dataset_name}_cindra_extraction_session_{session}"
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=str(main_session_multiday))
        working_directory = get_remote_job_work_directory(
            server=server,
            job_name=job_name,
            pipeline_name=ProcessingPipelines.CINDRA_MULTI_RECORDING,
            base_path=main_session_multiday,
        )
        server.create(remote_path=working_directory, is_dir=True)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment=server.environment,
            cpu_threads=30,
            ram=80,
            time=180,
        )
        job.add_command(f"slf forge -i {configuration_path} {session_path_args} -id {job_id} -e -t {session}")
        stage_2.append((job, working_directory, job_id))

    # Resolves the paths to the local and remote job tracker files (now in main session's multiday folder).
    remote_tracker_path = main_session_multiday.joinpath("multiday_tracker.json")
    local_tracker_path = local_working_directory.joinpath(
        project,
        animal,
        main_session,
        "processed_data",
        "mesoscope_data",
        "multiday",
        dataset_name,
        "multiday_tracker.json",
    )

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.CINDRA_MULTI_RECORDING,
        server=server,
        data_path=main_session_multiday,
        jobs={1: tuple(stage_1), 2: tuple(stage_2)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=dataset_name,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def _construct_data_assembly_pipeline(
    dataset: DatasetData,
    project: str,
    server: Server,
    *,
    keep_job_logs: bool = False,
) -> ProcessingPipeline:
    """Generates and returns the ProcessingPipeline instance used to execute the data assembly pipeline for the target
    dataset's sessions.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        project: The name of the project for which to execute the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance for the data assembly pipeline.
    """
    # Resolves the path to the local Sollertia working directory.
    local_working_directory = get_working_directory()

    # Resolves dataset and project paths.
    remote_dataset_path = server.root.joinpath(project, dataset.name)
    project_root = server.root.joinpath(project)

    # Collects the session names from the dataset.
    session_names = [s.session for s in dataset.sessions]

    # Extracts the first animal from the dataset for pipeline metadata.
    first_animal = dataset.animals[0].animal if dataset.animals else "unknown"

    # Precreates the iterable to store the assembly jobs (single stage pipeline).
    stage_1 = []

    # Creates an assembly job for each session.
    for session in session_names:
        job_name = f"{dataset.name}_{ProcessingPipelines.FORGING}_session_{session}"
        job_id = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.FORGING, base_path=remote_dataset_path
        )
        server.create(remote_path=working_directory, is_dir=True)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment=server.environment,
            cpu_threads=8,
            ram=32,
            time=60,
        )
        job.add_command(f"slf forge -dn {remote_dataset_path.name} -pp {project_root} -id {job_id}")
        stage_1.append((job, working_directory, job_id))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = remote_dataset_path.joinpath(ProcessingTrackers.FORGING)
    local_tracker_path = local_working_directory.joinpath(project, dataset.name, ProcessingTrackers.FORGING)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.FORGING,
        server=server,
        data_path=remote_dataset_path,
        jobs={1: tuple(stage_1)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=dataset.name,
        animal=first_animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )


def forge_dataset(
    manifest_path: Path,
    project: str,
    sessions: tuple[DatasetSession, ...],
    dataset_name: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sessions: set[str] | None = None,
    exclude_sessions: set[str] | None = None,
    include_animals: set[str] | None = None,
    exclude_animals: set[str] | None = None,
    process_multiday: bool = False,
    assemble_data: bool = False,
    reprocess: bool = False,
    keep_job_logs: bool = False,
    cindra_configuration_file: str = "GCaMP6f_CA1_MD.yaml",
    processing_batch_size: int = 4,
) -> None:
    """Resolves and executes the necessary dataset forging pipelines for the target project.

    This function acts as the main entry point for all dataset forging. As part of its runtime, it first
    defines the dataset by filtering the available sessions and creating the dataset hierarchy structure. Then, it
    executes the requested processing pipelines on the remote compute server by iteratively submitting batches of
    remote compute jobs to the server. The session type and acquisition system are derived automatically from the
    first session's metadata.

    Args:
        manifest_path: The path to the project's manifest .feather file.
        project: The name of the project whose data to forge into a dataset.
        sessions: A tuple of DatasetSession instances defining the project's sessions available for dataset creation.
        dataset_name: The unique name to assign to the created dataset.
        start_date: The start date for the date range filter. Sessions recorded on or after this date are included.
        end_date: The end date for the date range filter. Sessions recorded on or before this date are included.
        include_sessions: A set of session names to include regardless of the date range.
        exclude_sessions: A set of session names to exclude from the dataset.
        include_animals: A set of animal names to include. If specified, only sessions from these animals are
            included in the dataset.
        exclude_animals: A set of animal names to exclude. Sessions from these animals are removed from the dataset.
        process_multiday: Determines whether to execute the multi-day cindra processing pipeline.
        assemble_data: Determines whether to execute the data assembly pipeline.
        reprocess: Determines whether to reprocess sessions that have already been processed with either of the
            supported dataset forging pipelines.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            each processing pipeline completes successfully. If the pipeline fails, the job logs are kept regardless
            of the value of this argument.
        cindra_configuration_file: Specifies the name of the configuration file for the multi-day cindra processing
            pipeline. This argument is only used if the 'process_multiday' argument is set to True. The configuration
            file with the specified name must be present in the shared cindra configuration directory on the remote
            compute server.
        processing_batch_size: The number of processing pipelines that can be submitted to the remote compute server at
            a time.
    """
    # Ensures that the caller has specified at least one processing pipeline to execute.
    if not process_multiday and not assemble_data:
        console.error(
            message=(
                f"Unable to forge the '{dataset_name}' dataset, as no processing pipeline was selected. "
                f"Call the forge command with the --multiday (-m), --assemble (-a), or both flags to execute "
                f"the desired dataset forging pipelines."
            ),
            error=RuntimeError,
        )

    console.echo(message=f"Initializing the '{dataset_name}' dataset forging...", level=LogLevel.INFO)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Loads the project's manifest data.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Tracks all pipelines executed across all processing phases for final outcome reporting.
    all_pipelines: list[ProcessingPipeline] = []

    # Applies the filtering rules to the provided sessions. sollertia-shared-assets' filter_sessions
    # operates on plain (session_name, animal) tuples; the map rehydrates the filtered keys back to
    # DatasetSession.
    session_map = {(session.session, session.animal): session for session in sessions}
    filtered_sessions = {
        session_map[key]
        for key in filter_sessions(
            sessions=set(session_map.keys()),
            start_date=start_date,
            end_date=end_date,
            include_sessions=include_sessions,
            exclude_sessions=exclude_sessions,
            include_animals=include_animals,
            exclude_animals=exclude_animals,
            utc_timezone=True,
        )
    }

    # Ensures at least one session passed the filtering.
    if not filtered_sessions:
        console.error(
            message=f"Unable to create the '{dataset_name}' dataset. No sessions passed the filtering criteria.",
            error=ValueError,
        )

    # PHASE 1: MULTI-DAY PROCESSING
    console.echo(message="Phase 1: Multi-Day Processing...", level=LogLevel.INFO)
    delay_terminal()

    multiday_pipelines: list[ProcessingPipeline] = []
    multiday_exclusions: dict[str, tuple[str, str]] = {}  # Maps animal to (dataset_name, reason)

    if process_multiday:
        # Groups sessions by animal for multi-day processing.
        animal_sessions: dict[str, list[str]] = {}
        for session_meta in filtered_sessions:
            animal_sessions.setdefault(session_meta.animal, []).append(session_meta.session)

        # For each animal, constructs the multi-day pipeline.
        for animal, animal_session_list in console.track(
            animal_sessions.items(), description="Resolving the multi-day processing graph", unit="animal"
        ):
            result = _construct_cindra_multiday_pipeline(
                manifest=manifest,
                project=project,
                animal=animal,
                sessions=animal_session_list,
                dataset_name=dataset_name,
                server=server,
                configuration_file=cindra_configuration_file,
                reprocess=reprocess,
                keep_job_logs=keep_job_logs,
            )
            if isinstance(result, str):
                multiday_exclusions[animal] = (dataset_name, result)
            else:
                multiday_pipelines.append(result)
                all_pipelines.append(result)

        # Executes multi-day processing pipelines.
        if multiday_pipelines:
            execute_pipelines(
                pipelines=tuple(multiday_pipelines),
                batch_size=processing_batch_size,
                stage_name="multi-day processing",
                poll_delay=10,
            )
            delay_terminal()

            # Refreshes the manifest to include the processing results.
            resolve_project_manifest(project=project, server=server, generate=True)

    # PHASE 2: DATASET DEFINITION AND DATA ASSEMBLY
    console.echo(message="Phase 2: Dataset Definition and Data Assembly...", level=LogLevel.INFO)
    delay_terminal()

    assembly_pipelines: list[ProcessingPipeline] = []
    assembly_exclusions: dict[str, str] = {}  # Maps session name to exclusion reason

    if assemble_data:
        # Creates the dataset on the remote server (only needed for assembly).
        dataset = _define_dataset_remote(
            project=project,
            dataset_name=dataset_name,
            filtered_sessions=filtered_sessions,
            server=server,
            keep_job_logs=keep_job_logs,
        )
        delay_terminal()

        # Constructs the data assembly pipeline for the dataset.
        assembly_pipeline = _construct_data_assembly_pipeline(
            dataset=dataset,
            project=project,
            server=server,
            keep_job_logs=keep_job_logs,
        )
        assembly_pipelines.append(assembly_pipeline)
        all_pipelines.append(assembly_pipeline)

        # Executes assembly pipelines.
        if assembly_pipelines:
            execute_pipelines(
                pipelines=tuple(assembly_pipelines),
                batch_size=processing_batch_size,
                stage_name="data assembly",
                poll_delay=10,
            )
            delay_terminal()

            # Refreshes the manifest to include the processing results.
            resolve_project_manifest(project=project, server=server, generate=True)

    # Creates a visual separation before the final summary.
    delay_terminal()

    # Calculates overall statistics.
    total_successful = sum(1 for p in all_pipelines if p.pipeline_status == ProcessingStatus.SUCCEEDED)
    total_failed = sum(1 for p in all_pipelines if p.pipeline_status == ProcessingStatus.FAILED)

    # Displays the overall processing summary message.
    message = (
        f"Dataset '{dataset_name}' forging: Complete. Successfully completed {total_successful} pipelines, "
        f"failed {total_failed} pipelines. "
        f"The details about the processing outcome are available below:"
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Prints detailed results for all pipelines.
    for pipeline in all_pipelines:
        if pipeline.pipeline_status == ProcessingStatus.FAILED:
            message = f"The {pipeline.pipeline} pipeline for '{pipeline.session}': Failed."
            console.echo(message=message, level=LogLevel.ERROR)
        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
            message = f"The {pipeline.pipeline} pipeline for '{pipeline.session}': Complete."
            console.echo(message=message, level=LogLevel.SUCCESS)

    # Prints exclusion reasons for multi-day processing.
    for animal, (ds_name, reason) in multiday_exclusions.items():
        message = f"Multi-day processing for animal '{animal}' in dataset '{ds_name}': Excluded ({reason})."
        console.echo(message=message, level=LogLevel.WARNING)

    # Prints exclusion reasons for assembly.
    for session_name, reason in assembly_exclusions.items():
        message = f"Assembly for session '{session_name}': Excluded ({reason})."
        console.echo(message=message, level=LogLevel.WARNING)

    console.echo(message="Forging: Complete.", level=LogLevel.SUCCESS)
