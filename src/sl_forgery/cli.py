"""This module provides the Command Line Interfaces (CLIs) exposed by the library upon installation into a python
environment.
"""

import click
from sl_shared_assets import Server, ProjectManifest, JupyterJob
from ataraxis_base_utilities import console

from .utils import get_working_directory, get_credentials_file_path
from .processing import fetch_remote_project_manifest, generate_remote_project_manifest


@click.command()
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to print the manifest data.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=False,
    help=(
        "The name of the animal for which to print the manifest data. If not provided, this CLI prints the data for "
        "all animals that participate in the specified project."
    ),
)
@click.option(
    "-n",
    "--notes",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to print the experimenter note view of the available manifest data. This data view is "
        "optimized for checking the outcome of each session conducted as part of the target project and, optionally, "
        "by the specified animal."
    ),
)
@click.option(
    "-s",
    "--summary",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to print the data processing view of the available manifest data. This view is optimized "
        "for tracking the data processing state of each session conducted as part of the project."
    ),
)
@click.option(
    "-u",
    "--update_manifest",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to fetch the most recent project manifest version stored on the remote server before "
        "displaying the data. Since the manifest file is cached locally, this option is only required if the project "
        "data stored on the server has updated since the last call to this CLI."
    ),
)
@click.option(
    "-r",
    "--regenerate_manifest",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to regenerate the manifest file on the remote server before fetching it it to the local "
        "working directory. This flag requires service access privileges and is not recommend for most use cases, as "
        "all lab pipelines automatically update the manifest file as part of their runtime."
    ),
)
def print_project_manifest_data(
    project: str,
    animal: str | None,
    notes: bool,
    summary: bool,
    update_manifest: bool,
    regenerate_manifest: bool,
) -> None:
    if not summary and not notes:
        message = (
            f"No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            f"'summary' (-s), or both flags when calling the command to display the data using the target format."
        )
        console.error(message=message, error=ValueError)

    # Resolves the path to the manifest file
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remove server
    if not manifest_path.exists() and not regenerate_manifest and not update_manifest:
        update_manifest = True

    # If requested, fetches the most recent manifest file instance from the remote server to the working directory
    # before printing the project data. Note, the default is expected to be 'update' as it does not requre service
    # account credentials.
    if update_manifest:
        # Establishes SSH connection to the processing server using the user account credentials.
        credentials = get_credentials_file_path(require_service=False)
        server = Server(credentials_path=credentials)
        fetch_remote_project_manifest(project=project, server=server)

    # Manifest regeneration requires service account credentials and re-creates the manifest before fetching it to the
    # local machine.
    elif regenerate_manifest:
        # Establishes SSH connection to the processing server using the service account credentials.
        credentials = get_credentials_file_path(require_service=True)
        server = Server(credentials_path=credentials)
        generate_remote_project_manifest(project=project, server=server)

    # Loads the manifest file data into memory
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Ensures that the specified animal exists in the manifest data. Since the manifest is optimized for the Sun lab
    # data format, it stores animal IDs as integers. To improve the flexibility of this CLI, converts animal IDs to
    # strings before running the check.
    if animal is not None and animal not in [str(animal) for animal in manifest.animals]:
        message = (
            f"Unable to display the data for the target animal ({animal}), as the animal does not belong to the "
            f"target project ({project})."
        )
        console.error(message=message, error=ValueError)

    # If requested, prints the experimenter note view of the manifest data
    if notes:
        manifest.print_notes(animal=animal)

    # If requested, prints the data processing view of the manifest data
    if summary:
        manifest.print_summary(animal=animal)


@click.option(
    "-e",
    "--environment",
    type=str,
    required=True,
    help=(
        "The name of the conda environment to use for running the Jupyter server. At a minimum, the target environment "
        "must contain the 'jupyterlab' and the 'notebook' Python packages. Note, the user whose credentials are used "
        "to connect to the server must have a configured conda / mamba shell that exposes the target environment for "
        "the job to run as expected."
    ),
)
@click.option(
    "-c",
    "--cores",
    type=int,
    required=True,
    show_default=True,
    default=2,
    help="The number of CPU cores to allocate to the Jupyter server.",
)
@click.option(
    "-m",
    "--memory",
    type=int,
    required=True,
    show_default=True,
    default=32,
    help="The memory (RAM), in Gigabytes, to allocate to the Jupyter server.",
)
@click.option(
    "-t",
    "--time",
    type=int,
    required=True,
    show_default=True,
    default=240,
    help=(
        "The maximum runtime duration for this Jupyter server instance, in minutes. If the server job is still running "
        "at the end of this time limit, the job will be forcibly terminated by SLURM. To prevent hogging the server, "
        "make sure this parameter is always set to the smallest feasible period of time."
    ),
)
@click.option(
    "-p",
    "--port",
    type=int,
    required=True,
    show_default=True,
    default=0,
    help=(
        "The port to use for the Jupyter server communication on the remote server. Valid port values are from 8888 "
        "to 9999. Most runtimes should leave this set to the default value (0), which randomly selects one of the "
        "valid ports. Using random selection minimizes the chances of colliding with other interactive jupyter "
        "sessions."
    ),
)
def start_jupyter_server(environment: str, cores: int, memory: int, time: int, port: int) -> None:
    """Starts an interactive Jupyter session on the remote Sun lab server.

    This command allows running Jupyter lab and notebook sessions on the remote Sun lab server. Since all lab data is
    stored on the server, this allows running interactive analysis sessions on the same node as the data,
    while leveraging considerable compute resources of the server.

    Calling this command initializes a SLURM session that runs the interactive Jupyter server. Since this server
    directly competes for resources with all other headless jobs running on the server, it is imperative that each
    jupyter runtime uses the minimum amount of resources as necessary. Do not use this command to run
    heavy data processing pipelines! Instead, consult the API documentation for this library and use the headless
    Job or Pipeline class.
    """
    # Initializes server connection
    credentials_path = get_credentials_file_path(require_service=False)
    server = Server(credentials_path)

    job: JupyterJob | None = None
    job_name = f"interactive_jupyter_server"
    try:
        # Launches the Jupyter server
        job = server.launch_jupyter_server(
            job_name=job_name,
            conda_environment=environment,
            notebook_directory=server.user_working_root,
            cpus_to_use=cores,
            ram_gb=memory,
            port=port,
            time_limit=time,
        )

        # Displays the server connection details to the user via terminal
        job.print_connection_info()

        # Blocks in-place until the user shuts down the server. This allows terminating the jupyter job early if the
        # user is done working with the server
        input("Enter anything to shut down the server: ")

    # Ensures that the server created as part of this CLI is always terminated when the CLI terminates
    finally:
        # Terminates the server job
        if isinstance(job, JupyterJob) and not server.job_complete(job):
            server.abort_job(job)

        # Closes the server connection if it is still open
        server.close()
