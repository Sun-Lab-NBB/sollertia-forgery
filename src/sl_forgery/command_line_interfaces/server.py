"""This module provides the command-line interface for interacting with the remote Sun lab compute server."""

import click
from sl_shared_assets import get_server_configuration

from ..server import Server


@click.command(name="jupyter")
@click.option(
    "-e",
    "--environment",
    type=str,
    required=True,
    help=(
        "The name of the conda environment to use for running the Jupyter notebook session. The environment "
        "must contain the 'jupyterlab' and the 'notebook' Python packages. Note, the user whose credentials are used "
        "to connect to the server must have a configured conda / mamba shell that exposes the target environment for "
        "the job to run as expected."
    ),
)
@click.option(
    "-c",
    "--cores",
    type=int,
    default=2,
    show_default=True,
    help="The number of CPU cores to allocate to the Jupyter session.",
)
@click.option(
    "-m",
    "--memory",
    type=int,
    default=32,
    show_default=True,
    help="The memory (RAM), in Gigabytes, to allocate to the Jupyter session.",
)
@click.option(
    "-t",
    "--time",
    type=int,
    default=120,
    show_default=True,
    help=("The maximum uptime duration for the Jupyter session, in minutes."),
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=0,
    show_default=True,
    help=(
        "The port to use for communicating with the Jupyter session. Valid port values are from 8888 to 9999. Most "
        "use contexts should leave this set to the default value (0), which randomly selects one of the valid ports. "
        "Using random selection minimizes the chance of colliding with other interactive jupyter sessions."
    ),
)
def start_jupyter_server(environment: str, cores: int, memory: int, time: int, port: int) -> None:
    """Starts the interactive Jupyter notebook session on the remote compute server.

    Calling this command initializes a SLURM job that runs the interactive Jupyter notebook session. Since this session
    directly competes for resources with all other headless jobs running on the server, it is imperative that each
    jupyter runtime uses the minimum amount of resources necessary to support its runtime. Jupyter sessions are intended
    for lightweight data exploration and visualization tasks and should not be used for resource-intensive data
    processing tasks. Those tasks should be executed using the headless processing pipeline classes from this library.
    """
    # Initializes server connection
    configuration = get_server_configuration(service=False)
    server = Server(configuration=configuration)

    try:
        # Launches the Jupyter server. This method establishes an SSH tunnel, prints connection info, blocks until
        # the user terminates the session, and handles job cleanup automatically.
        server.launch_jupyter_server(
            job_name="interactive_jupyter_server",
            conda_environment=environment,
            notebook_directory=server.user_working_root,
            cpu_threads=cores,
            ram=memory,
            port=port,
            time=time,
        )

    finally:
        # Closes the server connection
        server.close()
