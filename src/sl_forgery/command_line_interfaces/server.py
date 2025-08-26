import click
from sl_shared_assets import Server, JupyterJob, get_credentials_file_path


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
    credentials_path = get_credentials_file_path(service=False)
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
