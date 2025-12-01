from pathlib import Path

from sl_shared_assets import (
    Job,
    Server,
    ProcessingStatus,
    TrackerFileNames,
    ProcessingPipeline,
    get_working_directory,
)

from ..server import get_remote_job_work_directory
from ..shared_assets import ProcessingPipelines


def _construct_suite2p_multiday_pipeline(
    project: str,
    animal: int,
    sessions: list[str],
    server: Server,
    manager_id: int,
    dataset_name: str,
    configuration_file: str = "GCaMP6f_CA1_MD.yaml",
    reset_tracker: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | None:
    """Generates and returns the ProcessingPipeline instance used to execute the multi-day suite2p processing pipeline
    for the target set of sessions.

    Notes:
        Unlike processing pipeline constructors used in the 'processing' package, this constructor expects the input
        sessions to be pre-validated and always assumes all input sessions can be processed.

    Args:
        project: The name of the project for which to execute the target processing pipeline.
        animal: The unique identifier of the animal that performed the sessions processed by this pipeline.
        sessions: The list of session names to process using the target processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        manager_id: The unique identifier of the process that calls this function to construct the pipeline.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the sl-suite2p multi-day pipeline.
        reset_tracker: Determines whether to reset the processing tracker for the pipeline before executing the
            processing. This option should only be enabled when recovering from improper runtime terminations.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any pipeline job fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance.
    """
    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Constructs the list of session paths to use in the multiday processing command.
    session_command = ""
    for session in sessions:
        session_path = server.shared_storage_root.joinpath(project, str(animal), session)
        session_command += f"-sp {session_path} "

    # Resolves additional shared flags for the processing CLI.
    tracker_command = ""
    if reset_tracker:
        tracker_command = "-r"
    configuration_command = f"-i {server.suite2p_configurations_directory.joinpath(configuration_file)}"
    job_command = f"-j {len(sessions) + 1}"  # Static +1 is for the multi-day cell discovery step.

    # Precreates the iterables to store stage jobs
    stage_1 = []
    stage_2 = []

    # Stage 1: Discovery
    job_name = f"{dataset_name}_ss2p_discovery"
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MULTIDAY
    )
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
        f"ss2p run {configuration_command} -w -1 sl-multi-day {session_command} "
        f"-pdr {server.shared_working_root} -id {manager_id} {job_command} {tracker_command} -o "
        f"{server.user_working_root.joinpath(dataset_name)} -d"
    )
    stage_1.append((job, working_directory))

    # Stage 2: Session data extraction.
    for session in sessions:
        job_name = f"{dataset_name}_ss2p_session_{session}"
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MULTIDAY
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
            f"ss2p run {configuration_command} -w -1 sl-multi-day {session_command} "
            f"-pdr {server.shared_working_root} -id {manager_id} {job_command} {tracker_command} -o "
            f"{server.user_working_root.joinpath(project, dataset_name)} -e -t {session}"
        )
        stage_2.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = Path(server.user_working_root).joinpath(
        project, dataset_name, str(animal), TrackerFileNames.MULTIDAY
    )
    local_tracker_path = local_working_directory.joinpath(project, dataset_name, str(animal), TrackerFileNames.MULTIDAY)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    pipeline = ProcessingPipeline(
        jobs={1: tuple(stage_1), 2: tuple(stage_2)},
        server=server,
        manager_id=manager_id,
        pipeline_type=ProcessingPipelines.MULTIDAY,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=dataset_name,
        animal=str(animal),
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
    )

    return pipeline
