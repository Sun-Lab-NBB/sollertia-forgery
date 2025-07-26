from pathlib import Path
from tempfile import TemporaryDirectory

from ataraxis_time import PrecisionTimer
from sl_shared_assets import Job, Server, ServerCredentials
from ataraxis_base_utilities import LogLevel, console
from ataraxis_time.time_helpers import get_timestamp
from sl_shared_assets.tools.project_management_tools import ProjectManifest


def fetch_project_manifest(credentials_path: Path, project: str) -> ProjectManifest:
    """Generates and fetches the manifest .feather file for the specified project stored on the remote compute server.

    This function serves as the entry-point for all data processing and dataset formation tasks performed by this
    library. It is used to generate the current snapshot of all available data for the specified project on the remote
    server and pull it to the local machine, so that it can be used by other functions from this library.

    Args:
        credentials_path: The path to the .yaml file that stores the access credentials and filesystem information
            for the remote compute server that stores the project data.
        project: The name of the project for which to generate and fetch the manifest file.

    Returns:
        A ProjectManifest instance containing the manifest data for the specified project.
    """

    # Loads the server access credential data, which includes remote server's filesystem data. Also uses the file to
    # initialize the SHH connection to the remote server.
    credentials: ServerCredentials = ServerCredentials.from_yaml(credentials_path)  # type: ignore
    server = Server(credentials_path=credentials_path)

    # Resolves the working directory for the job, using static job name and the current timestamp in UTC.
    timestamp = get_timestamp()
    job_name = f"{project}_manifest_generation"
    working_directory = Path(credentials.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Ensures that the working directory exists on the remote server
    server.create_directory(remote_path=working_directory)

    # Parses the paths to the shared Sun lab directories used to store raw and processed project data on the remote
    # server.
    project_storage_root = Path(credentials.storage_root).joinpath(credentials.shared_directory_name, project)
    project_working_path = Path(credentials.working_root).joinpath(credentials.shared_directory_name, project)

    # Generates the remote job header
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath(f"output.txt"),
        error_log=working_directory.joinpath(f"errors.txt"),
        working_directory=working_directory,
        conda_environment="manage",
        cpus_to_use=1,
        ram_gb=10,
        time_limit=10,
    )

    # Configures the job to use the sl-shared-assets package installed on the server to generate the manifest file
    # inside the working directory
    job.add_command(
        f"sl-project-manifest -pp {str(project_storage_root)} -ppp {str(project_working_path)} "
        f"-od {str(working_directory)}"
    )

    # Submits the remote job to the server
    job = server.submit_job(job)

    # Waits for the server to complete the job
    delay_timer = PrecisionTimer("s")
    while not server.job_complete(job=job):
        message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
        console.echo(message=message, level=LogLevel.INFO)
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    # After the server completes the job, pulls the generated manifest file to a temporary directory on the local PC,
    # loads the manifest data into memory as a ProjectManifest instance and returns it to caller
    with TemporaryDirectory() as temp_directory:
        server.pull_file(
            local_file_path=Path(temp_directory).joinpath(f"manifest.feather"),
            remote_file_path=working_directory.joinpath(f"{project}_manifest.feather"),
        )
        return ProjectManifest(manifest_file=Path(temp_directory).joinpath(f"manifest.feather"))


def view_sessions(credentials_path: Path, project: str, animal: int) -> None:
    manifest = fetch_project_manifest(credentials_path=credentials_path, project=project)
    sessions = manifest.get_sessions_for_animal(animal=animal, exclude_incomplete=False)
    console.echo(
        message=f"The animal '{animal}' has performed the following sessions for '{project}' project :", level=LogLevel.INFO
    )
    for num, session in enumerate(sessions, start=1):
        console.echo(message=f"{num}: {session}", level=LogLevel.INFO)

