
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
from ..processing import fetch_remote_project_manifest


def _construct_forging_pipeline(
    manifest: ProjectManifest,
    project: str,
    sessions: list[str],
    server: Server,
    manager_id: int,
    configuration_file: str = "GCaMP6f_CA1_MD.yaml",
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the single-day suite2p processing pipeline
    for the target session.

    Args:
        manifest: The initialized ProjectManifest instance that stores the processed project's metadata.
        project: The name of the project for which to execute the target processing pipeline.
        sessions: The names of the sessions to process.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the sl-suite2p multi-day pipeline.
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
    system = manifest.get_system_for_session(session=session)

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

    if system == AcquisitionSystems.MESOSCOPE_VR:

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
