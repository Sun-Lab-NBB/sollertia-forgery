from pathlib import Path

from ataraxis_time import PrecisionTimer
from sl_shared_assets import Job, Server, ServerCredentials
from ataraxis_base_utilities import LogLevel, console
from ataraxis_time.time_helpers import get_timestamp
from sl_shared_assets.tools.project_management_tools import ProjectManifest


def fetch_project_manifest(credentials_path: Path, project: str, target_directory: Path) -> None:
    """Generates and fetches the manifest .feather file for the specified project stored on the remote compute server.

    This function serves as the entry-point for all data processing and dataset formation tasks performed by this
    library. It is used to generate the current snapshot of all available data for the specified Sun lab project. In
    turn, this information is used to direct processing pipelines to work with specific sessions.
    """
    credentials: ServerCredentials = ServerCredentials.from_yaml(credentials_path)  # type: ignore
    timestamp = get_timestamp()
    job_name = f"{project}_manifest_generation"
    working_directory = Path(credentials.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")
    server = Server(credentials_path=credentials_path)

    project_storage_root = Path(credentials.storage_root).joinpath(credentials.shared_directory_name, project)
    project_working_path = Path(credentials.working_root).joinpath(credentials.shared_directory_name, project)

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

    job.add_command(
        f"sl-project-manifest -pp {str(project_storage_root)} -ppp {str(project_working_path)} "
        f"-od {str(working_directory)}"
    )

    server.create_directory(remote_path=working_directory)

    job = server.submit_job(job)

    delay_timer = PrecisionTimer("s")
    while not server.job_complete(job=job):
        message = f"Waiting for the manifest generation job with ID {job.job_id} to complete..."
        console.echo(message=message, level=LogLevel.INFO)
        delay_timer.delay_noblock(delay=5, allow_sleep=True)

    server.pull_file(
        local_file_path=target_directory.joinpath(f"{project}_manifest.feather"),
        remote_file_path=working_directory.joinpath(f"{project}_manifest.feather"),
    )

def process_behavior_data(manifest_file: Path, target_directory: Path) -> None:


# fetch_project_manifest(
#     credentials_path=Path("/home/cyberaxolotl/Data/server_credentials.yaml"),
#     project="MaalstroomicFlow",
#     target_directory=Path("/home/cyberaxolotl/Data"),
# )

target = Path("/home/cyberaxolotl/Data/MaalstroomicFlow_manifest.feather")
manifest = ProjectManifest(manifest_file=target)
manifest.print_summary(15)