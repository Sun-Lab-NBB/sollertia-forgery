from pathlib import Path

from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp

from ..server import (
    Job,
    Server,
    ProcessingStatus,
    TrackerFileNames,
    ProcessingPipeline,
    ProcessingPipelines,
)
from ..data_classes import get_working_directory


def _get_remote_job_work_directory(server: Server, job_name: str) -> Path:
    """Generates the working directory for the input job intended to be executed on the compute server managed by the
    input Server class.

    This worker function generates the current UTC timestamp, clips it down to minutes, and concatenates it to the
    job_name to construct the working directory name. It then resolves the path to that directory relative to the user
    working root on the remote server, creates the directory on the server, and returns the resolved path.
    """

    # Resolves working directory name using timestamp (accurate to minutes) and the job_name.
    timestamp = "-".join(get_timestamp().split("-")[:5])  # type: ignore
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Creates the working directory on the remote server.
    server.create_directory(remote_path=working_directory, parents=True)

    return working_directory


def compose_remote_processing_pipeline(
    pipeline: ProcessingPipelines,
    project: str,
    animal: str,
    session: str,
    server: Server,
    manager_id: int,
    keep_job_logs: bool = False,
    reset_tracker: bool = False,
    recalculate_checksum: bool = False,
) -> ProcessingPipeline:
    """Generates and returns the ProcessingPipeline instance used to execute the requested pipeline for the target
    session on the specified remote compute server.

    This function composes the processing pipeline execution graph and packages it into the ProcessingPipeline instance.
    As part of this process, it resolves the necessary local and remote filesystem paths and generates the Job
    instances for all jobs making up the pipeline.

    Notes:
        This function does not start executing the pipeline. Instead, the pipeline starts executing the first time
        the manager process calls its runtime_cycle() method.

    Args:
        pipeline: The ProcessingPipelines enumeration value specifying the target pipeline to compose.
        project: The name of the project for which to compose the processing pipeline.
        animal: The ID of the animal for which to compose the processing pipeline.
        session: The ID of the session to process with the processing pipeline.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session data.
        manager_id: The unique identifier of the process that calls this function to compose the pipeline.
        keep_job_logs: Determines whether to keep the logs for completed pipeline jobs on the server or (default)
            remove them after runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of
            this argument.
        reset_tracker: Determines whether the constructed pipeline should forcibly reset its tracker before executing
            the runtime. This option should only be used in extreme circumstances to recover from processing deadlocks
            due to improper runtime termination. Most runtimes should have this argument disabled to ensure that the
            session data is accessed in a process-safe manner.
        recalculate_checksum: Only for integrity verification pipelines. Determines whether to verify the checksum
            integrity (if false) or whether to recalculate and overwrite the checksum stored in the ax_checksum.txt file
            (if true).

    Returns:
        The ProcessingPipeline instance configured to execute and manage the target processing pipeline on the remote
        server.
    """
    # Parses the paths to the shared Sun lab directories used to store raw session data on the remote server.
    remote_session_path = Path(server.raw_data_root).joinpath(project, animal, session)

    # Resolves the path to the local Sun lab working directory
    local_working_directory = get_working_directory()

    # Precreates the dictionary for storing job instances and their working directories
    jobs: dict[int, tuple[tuple[Job, Path], ...]] = {}
    stage = 1  # This is used to iteratively fill processing stage data for multi-stage pipelines.
    # This is used to store the Tracker file name for the constructed pipeline once it is resolved below
    tracker: TrackerFileNames

    # Integrity verification pipeline
    if pipeline == ProcessingPipelines.CHECKSUM:
        job_name = f"{session}_integrity_verification"
        working_directory = _get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="manage",
            cpus_to_use=10,
            ram_gb=10,
            time_limit=30,
        )

        # Resolves processing flags for the job
        reset_flag = ""
        if reset_tracker:
            reset_flag = "-r"
        recalculate_flag = ""
        if recalculate_checksum:
            recalculate_flag = "-rc"

        # Adds the main runtime command
        job.add_command(
            f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
            f"{reset_flag} checksum {recalculate_flag}"
        )
        jobs[stage] = ((job, working_directory),)
        tracker = TrackerFileNames.CHECKSUM

    # Processing preparation pipeline
    elif pipeline == ProcessingPipelines.PREPARATION:
        job_name = f"{session}_processing_preparation"
        working_directory = _get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="manage",
            cpus_to_use=20,
            ram_gb=10,
            time_limit=30,
        )
        reset_flag = ""
        if reset_tracker:
            reset_flag = "-r"
        job.add_command(
            f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
            f"{reset_flag} prepare"
        )
        jobs[stage] = ((job, working_directory),)
        tracker = TrackerFileNames.PREPARATION

    # Behavior processing pipeline
    elif pipeline == ProcessingPipelines.BEHAVIOR:
        job_name = f"{session}_behavior_processing"
        working_directory = _get_remote_job_work_directory(server=server, job_name=job_name)
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
        job.add_command(f"sl-process-behavior -sp {remote_session_path} -um")
        jobs[stage] = ((job, working_directory),)
        tracker = TrackerFileNames.BEHAVIOR

    # Data Archiving
    elif pipeline == ProcessingPipelines.ARCHIVING:
        job_name = f"{session}_data_archiving"
        working_directory = _get_remote_job_work_directory(server=server, job_name=job_name)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath(f"output.txt"),
            error_log=working_directory.joinpath(f"errors.txt"),
            working_directory=working_directory,
            conda_environment="manage",
            cpus_to_use=20,
            ram_gb=10,
            time_limit=30,
        )
        reset_flag = ""
        if reset_tracker:
            reset_flag = "-r"
        job.add_command(
            f"sl-manage session -sp {remote_session_path} -pdr {server.processed_data_root} -id {manager_id} "
            f"{reset_flag} archive"
        )
        jobs[stage] = ((job, working_directory),)
        tracker = TrackerFileNames.ARCHIVING

    else:
        message = (
            f"Unsupported pipeline {pipeline} encountered when attempting to construct a ProcessingPipeline instance. "
            f"Currently, only the following pipelines listed in the ProcessingPipelines enumeration are supported: "
            f"{', '.join(tuple(ProcessingPipelines))}."
        )
        console.error(message=message, error=ValueError)
        raise ValueError(message)  # Fallback to appease mypy, should not be reachable

    # Resolves paths to pipeline tracker files.
    remote_tracker_path = Path(server.raw_data_root).joinpath(project, animal, session, "raw_data", tracker)
    local_tracker_path = local_working_directory.joinpath(tracker)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    processing_pipeline = ProcessingPipeline(
        jobs=jobs,
        server=server,
        manager_id=manager_id,
        pipeline_type=pipeline,
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=session,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
        pipeline_status=ProcessingStatus.RUNNING,
    )
    return processing_pipeline


def generate_remote_project_manifest(project: str, server: Server, keep_job_logs: bool = False) -> None:
    """Generates the manifest .feather file for the specified project stored on the remote compute server.

    This function allows generating the manifest.feather files on the remote compute server outside the standard
    workflow (manually). Since this process requires 'service' access privileges, this function is not intended to be
    called directly by most lab users. As part of its runtime, this function also fetches (pulls) the generated manifest
    file to the local Sun lab working directory. Therefore, this function also includes the functionality of the
    fetch_remote_project_manifest() function.

    Notes:
        All Sun lab 'service' pipelines automatically update the manifest file as part of their runtime, so it is
        typically unnecessary to use this function. The function is mostly used internally to test various lab pipelines
        and data management strategies.

        The manifest file is created and stored inside the root raw data directory for the target project on the remote
        server.

    Args:
        project: The name of the project for which to generate and fetch the manifest file.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If the job fails, the logs are always kept regardless of this parameter.
        server: An initialized Server instance used to communicate with the remote server. Note, the Server must be
            configured to use the service account server access credentials.

    Raises:
        FileNotFoundError: If the remote (server-side) project manifest generation job fails with an error and does not
            generate the manifest file.
    """

    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Resolves the name and the working directory for the remote job
    job_name = f"{project}_manifest_generation"
    server_working_directory = _get_remote_job_work_directory(server=server, job_name=job_name)

    # Parses the paths to the shared Sun lab directories used to store raw and processed project data on the remote
    # server.
    project_storage_root = server.raw_data_root.joinpath(project)

    # Generates the remote job header
    job = Job(
        job_name=job_name,
        output_log=server_working_directory.joinpath(f"output.txt"),
        error_log=server_working_directory.joinpath(f"errors.txt"),
        working_directory=server_working_directory,
        conda_environment="manage",
        cpus_to_use=1,
        ram_gb=10,
        time_limit=20,
    )

    # Configures the job to use the sl-shared-assets package installed on the server to generate the manifest file
    # inside the project's root raw data directory
    job.add_command(f"sl-project-manifest -pp {project_storage_root!s} -pdr {server.processed_data_root!s}")

    # If the function is configured to remove job logs after runtime, adds a command to delete job working directory.
    if not keep_job_logs:
        job.add_command(f"rm -rf {server_working_directory!s}")

    # Submits the remote job to the server
    job = server.submit_job(job)

    # Waits for the server to complete the job
    delay_timer = PrecisionTimer("s")
    message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
    console.echo(message=message, level=LogLevel.INFO)
    while not server.job_complete(job=job):
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    # Resolves the path to the remote and local manifest files
    remote_manifest_path = project_storage_root.joinpath(f"{project}_manifest.feather")
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # Ensures that the project-specific folder exists under the local working directory
    ensure_directory_exists(local_manifest_path)

    # Verifies that the job ran as expected. For this, ensures that the remote manifest file exists (was created).
    if not server.exists(remote_path=remote_manifest_path):
        # Closes the SSH connection
        server.close()

        message = (
            f"Unable to locate the manifest file for '{project}' project one the remote server. This indicates that "
            f"the remote manifest creation job ran into an error and did not generate the file. Check the error logs "
            f"for the {job_name} job stored in the {server_working_directory} server directory for more details about "
            f"the error."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory. This ensures that the user has continued access to the most recent manifest file
    # for that project.
    console.echo(message=f"Fetching the generated manifest file from the remote compute server...")
    server.pull_file(
        local_file_path=local_manifest_path,
        remote_file_path=remote_manifest_path,
    )


def fetch_remote_project_manifest(project: str, server: Server) -> None:
    """Fetches (pulls) the existing project manifest .feather file for the specified project stored on the remote
    compute server to the local Sun lab working directory.

    This function serves as the entry-point for most other Sun lab data processing pipelines. It is used to pull the
    current snapshot of all available data for the specified project on the remote server to the local machine, so that
    it can be used by other functions from this library. The pulled manifest file is stored under the directory named
    after the input project inside the local Sun lab working directory.

    Args:
        project: The name of the project for which to fetch the manifest file.
        server: An initialized Server instance used to communicate with the remote server.

    Raises:
        FileNotFoundError: If the manifest file does not exist on the server, indicating that the file has not been
            generated.
    """

    # Resolves the path to the local directory used to work with Sun lab data.
    local_working_directory = get_working_directory()

    # Resolves the path to the remote and local manifest files
    remote_manifest_path = server.raw_data_root.joinpath(project, f"{project}_manifest.feather")
    local_manifest_path = local_working_directory.joinpath(project, "manifest.feather")

    # Ensures that the project-specific folder exists under the local working directory
    ensure_directory_exists(local_manifest_path)

    # Verifies that the job ran as expected. For this, ensures that the remote manifest file exists (was created).
    # Otherwise, aborts with an error.
    if not server.exists(remote_path=remote_manifest_path):
        # Closes the SSH connection
        server.close()

        message = (
            f"Unable to fetch the manifest file for '{project}' project from the remote server, as the target project "
            f"does not have a manifest file. Either wait for one of the service pipelines to generate the manifest "
            f"file or use the 'generate_project_manifest' CLI command to create it manually (requires 'service' access "
            f"privileges)."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the job completes as expected, pulls the generated manifest file to the project-specific subdirectory under
    # the local working directory.
    console.echo(message=f"Fetching the manifest file from the remote compute server...")
    server.pull_file(
        local_file_path=local_manifest_path,
        remote_file_path=remote_manifest_path,
    )
