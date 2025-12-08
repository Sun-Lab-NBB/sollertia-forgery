"""This module provides the Command Line Interfaces (CLIs) used to directly interact with the remote Sun lab compute
server.
"""

import click
from tabulate import tabulate
from sl_shared_assets import get_server_configuration
from ataraxis_base_utilities import LogLevel, console

from ..server import Server

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}

# Hardcoded SLURM output formats
SACCT_FORMAT = "JobID,JobName%50,ReqMem,MaxRSS,AveRSS,MaxVMSize,NCPUS,AveCPU,Elapsed,State"
"""The format for the slurm accounting 'sacct' command used to display and evaluate completed job's efficiency."""
SQUEUE_FORMAT = "%.10i %.9P %.50j %.8u %.8T %.6D %.6C %.10m %.10M %.12l %.12L"
"""The format for the slurm queue 'squeue' command used to display running and pending jobs."""


def _format_slurm_output(raw_output: str) -> str:
    """Formats raw SLURM command output strings into nicely formatted tables.

    Notes:
        This worker function is used to format the output of the SLURM's 'squeue' and 'sacct' commands.

    Args:
        raw_output: The raw output string from the SLURM 'squeue' or 'sacct' commands.

    Returns:
        A formatted string representation of the SLURM's output.
    """
    lines = raw_output.strip().split("\n")
    if not lines:
        return "No data available."

    # Parses header and data rows
    rows = [line.split() for line in lines if line.strip()]
    if not rows:
        return "No data available."

    # Uses 'tabulate' to format the output, with the first row as headers
    headers = rows[0]
    data = rows[1:]

    return tabulate(data, headers=headers, tablefmt="simple")


@click.group("server", context_settings=CONTEXT_SETTINGS)
def server_cli() -> None:
    """This Command-Line Interface (CLI) group allows interacting with the remote Sun lab compute server.

    This CLI group provides commands for managing non-standardized server interactions, including starting interactive
    Jupyter sessions and viewing SLURM job information. Note; all data workflow interactions available through
    sl-project and sl-execute command groups must be carried out through these groups, rather than the commands
    exposed by this CLI.
    """


@server_cli.command("jupyter")
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
    help="The maximum uptime duration for the Jupyter session, in minutes.",
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
    configuration = get_server_configuration()
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


@server_cli.command("print")
@click.option(
    "-j",
    "--job-data",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to display the remote server's job accounting history (runtime statistics) using the "
        "SLURM's 'sacct' command."
    ),
)
@click.option(
    "-q",
    "--queue",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display the remote server's job queue status using the SLURM's 'squeue' command.",
)
@click.option(
    "-u",
    "--user",
    type=str,
    default=None,
    help=(
        "Allows filtering the displayed queue and job data to only include the jobs submitted by the specified user. "
        "Defaults to the username used for the server authentication."
    ),
)
@click.option(
    "-st--start-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that started on or after this date "
        "(format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
    ),
)
@click.option(
    "-et--end-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that ended on or before this date "
        "(format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
    ),
)
def print_slurm_info(
    *, job_data: bool, queue: bool, user: str | None, start_time: str | None, end_time: str | None
) -> None:
    """Displays remote server's SLURM queue status or job data as a formatted table."""
    if not job_data and not queue:
        message = (
            "No data display options were selected when calling the command. Pass either the '--job-data' (-j), "
            "'--queue' (-q), or both flags to display the requested remote server's SLURM information."
        )
        console.error(message=message, error=ValueError)

    # Initializes communication with the server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Resolves the username from the server configuration file if an explicit override is not provided.
    if user is None:
        user = configuration.username

    try:
        # Displays sacct output if requested
        if job_data:
            cmd = f'sacct -u {user} -o "{SACCT_FORMAT}" --units=G'
            if start_time:
                cmd += f" --starttime={start_time}"
            if end_time:
                cmd += f" --endtime={end_time}"

            console.echo(message=f"Fetching job accounting data for the user '{user}'...", level=LogLevel.INFO)
            result = server.execute_command(command=cmd)

            if result.return_code != 0:
                console.error(
                    message=f"Failed to execute the sacct command on the remote server: {result.stderr}",
                    error=RuntimeError,
                )

            if result.stdout.strip():
                formatted_output = _format_slurm_output(result.stdout)
                console.echo(message=f"Job accounting (sacct) data for the user '{user}':")
                click.echo(formatted_output)
            else:
                console.echo(
                    message="No job accounting data found for the specified filtering criteria.", level=LogLevel.WARNING
                )

        # Displays squeue output if requested
        if queue:
            cmd = f'squeue -o "{SQUEUE_FORMAT}" -u {user}'

            console.echo(message=f"Fetching queue status for user '{user}'...", level=LogLevel.INFO)
            result = server.execute_command(command=cmd)

            if result.return_code != 0:
                console.error(
                    message=f"Failed to execute the squeue command on the remote server: {result.stderr}",
                    error=RuntimeError,
                )

            if result.stdout.strip():
                formatted_output = _format_slurm_output(result.stdout)
                console.echo(message=f"Queue status (squeue) for the user '{user}':")
                click.echo(formatted_output)
            else:
                console.echo(message="No jobs found in the queue for the specified user.", level=LogLevel.WARNING)

    finally:
        server.close()
