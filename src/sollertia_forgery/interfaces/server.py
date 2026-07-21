"""Provides CLIs for directly interacting with the remote compute server."""

import click
from tabulate import tabulate
from ataraxis_base_utilities import LogLevel, console

from ..server import Server, discover_project_data, get_server_configuration, create_server_configuration_file

# Ensures that displayed CLICK help messages are formatted according to the sollertia platform standard.
CONTEXT_SETTINGS = {"max_content_width": 120}

# Hardcoded SLURM output formats
SACCT_FORMAT = "JobID,JobName%50,ReqMem,MaxRSS,AveRSS,MaxVMSize,NCPUS,AveCPU,Elapsed,State"
"""The format for the slurm accounting 'sacct' command used to display and evaluate completed job's efficiency."""
SACCT_HEADERS = ["JobID", "JobName", "ReqMem", "MaxRSS", "AveRSS", "MaxVMSize", "NCPUS", "AveCPU", "Elapsed", "State"]
"""The headers corresponding to SACCT_FORMAT, used for display after merging rows."""
SQUEUE_FORMAT = "%.10i %.9P %.50j %.8u %.8T %.6D %.6C %.10m %.10M %.12l %.12L"
"""The format for the slurm queue 'squeue' command used to display running and pending jobs."""

# Minimum number of rows required for valid sacct output (header + at least one data row).
_MINIMUM_SACCT_ROWS: int = 2

# Number of columns expected in sacct output based on SACCT_FORMAT.
_SACCT_COLUMN_COUNT: int = 10


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

    # Parses header and data rows, skipping separator lines (lines with only dashes and spaces)
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped and not all(c in "- " for c in stripped):
            rows.append(stripped.split())
    if not rows:
        return "No data available."

    # Uses 'tabulate' to format the output, with the first row as headers
    headers = rows[0]
    data = rows[1:]

    return tabulate(data, headers=headers, tablefmt="simple", colalign=["center"] * len(headers))


def _format_sacct_output(raw_output: str) -> str:
    """Formats raw 'sacct' output (parsable format) into a nicely formatted table with merged rows.

    This function parses pipe-delimited sacct output and merges job rows that share the same base JobID. This
    handles both standard jobs (where parent rows are followed by .batch step rows) and bash jobs (where multiple
    rows share the same JobID).

    Args:
        raw_output: The raw output string from the 'sacct' command with the --parsable2 flag.

    Returns:
        A formatted string representation of the merged job data.
    """
    lines = raw_output.strip().split("\n")
    if not lines:
        return "No data available."

    # Parses pipe-delimited rows.
    rows = [line.split("|") for line in lines if line.strip()]
    if len(rows) < _MINIMUM_SACCT_ROWS:
        return "No data available."

    # Skips the header row from sacct, uses predefined headers
    data = rows[1:]

    # Merges rows by base JobID
    merged_data: list[list[str]] = []
    parent_jobs: dict[str, list[str]] = {}

    for row in data:
        if len(row) < _SACCT_COLUMN_COUNT:
            continue

        job_id = row[0]

        # Skips extern step rows
        if ".extern" in job_id:
            continue

        # Extracts the base job ID (without .batch suffix if present)
        base_job_id = job_id.split(".")[0]

        if base_job_id in parent_jobs:
            # Merges this row's non-empty fields into the existing parent row
            parent_row = parent_jobs[base_job_id]
            for i in range(len(row[:_SACCT_COLUMN_COUNT])):
                if row[i] and not parent_row[i]:
                    parent_row[i] = row[i]
        else:
            # First occurrence of this job ID - creates a new entry
            merged_row = list(row[:_SACCT_COLUMN_COUNT])
            parent_jobs[base_job_id] = merged_row
            merged_data.append(merged_row)

    if not merged_data:
        return "No data available."

    return tabulate(merged_data, headers=SACCT_HEADERS, tablefmt="simple", colalign=["center"] * len(SACCT_HEADERS))


@click.group("server", context_settings=CONTEXT_SETTINGS)
def server_cli() -> None:
    """Provides commands for interacting with the remote Sollertia compute server.

    This CLI group provides commands for managing non-standardized server interactions, including authoring the
    server access configuration, discovering project sessions, and viewing SLURM job information.
    """


@server_cli.command("configure", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-u",
    "--username",
    type=str,
    required=True,
    help="The username to use for server authentication.",
)
@click.option(
    "-p",
    "--password",
    type=str,
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
    help="The password to use for server authentication. Prompted interactively (with hidden input) if not provided.",
)
@click.option(
    "-h",
    "--host",
    type=str,
    required=True,
    help="The host name or IP address of the server.",
)
@click.option(
    "-r",
    "--root",
    type=str,
    required=True,
    help="The absolute path, on the remote server, to the root directory that stores all Sollertia data.",
)
@click.option(
    "-e",
    "--environment",
    type=str,
    required=True,
    help=(
        "The name of the shared conda environment, on the remote server, in which sollertia-forgery and all of its "
        "processing dependencies are installed."
    ),
)
def configure_server(
    username: str,
    password: str,
    host: str,
    root: str,
    environment: str,
) -> None:  # pragma: no cover
    """Creates the remote compute server configuration file in the Sollertia platform working directory."""
    create_server_configuration_file(
        username=username,
        password=password,
        host=host,
        root=root,
        environment=environment,
    )


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
        "Set to 'all' to display data for all users. Defaults to the username used for the server authentication."
    ),
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help="Determines the job for which to display the accounting data. Bypasses user and date filtering options.",
)
@click.option(
    "-st",
    "--start-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that started on or after this date "
        "(format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."
    ),
)
@click.option(
    "-et",
    "--end-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that ended on or before this date "
        "(format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."
    ),
)
def print_slurm_info(
    *,
    job_data: bool,
    queue: bool,
    user: str | None,
    job_id: str | None,
    start_time: str | None,
    end_time: str | None,
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

    # Determines whether to display data for all users
    all_users = user.lower() == "all"

    try:
        # Displays sacct output if requested
        if job_data:
            # If a specific job ID is requested, bypasses user and date filtering
            if job_id is not None:
                cmd = f'sacct -j {job_id} -o "{SACCT_FORMAT}" --parsable2 --units=G'
                console.echo(message=f"Fetching job accounting data for job ID '{job_id}'...", level=LogLevel.INFO)
            else:
                # Builds the command with optional user filtering
                if all_users:
                    cmd = f'sacct -a -o "{SACCT_FORMAT}" --parsable2 --units=G'
                else:
                    cmd = f'sacct -u {user} -o "{SACCT_FORMAT}" --parsable2 --units=G'
                if start_time:
                    cmd += f" --starttime={start_time}"
                if end_time:
                    cmd += f" --endtime={end_time}"

                if all_users:
                    console.echo(message="Fetching job accounting data for all users...", level=LogLevel.INFO)
                else:
                    console.echo(message=f"Fetching job accounting data for the user '{user}'...", level=LogLevel.INFO)

            result = server.execute_command(command=cmd)

            if result.return_code != 0:
                console.error(
                    message=f"Failed to execute the sacct command on the remote server: {result.stderr}",
                    error=RuntimeError,
                )

            if result.stdout.strip():
                formatted_output = _format_sacct_output(result.stdout)
                if job_id is not None:
                    console.echo(message=f"Job accounting (sacct) data for job ID '{job_id}':")
                elif all_users:
                    console.echo(message="Job accounting (sacct) data for all users:")
                else:
                    console.echo(message=f"Job accounting (sacct) data for the user '{user}':")
                click.echo(formatted_output)
            else:
                console.echo(
                    message="No job accounting data found for the specified filtering criteria.", level=LogLevel.WARNING
                )

        # Displays squeue output if requested
        if queue:
            if all_users:
                cmd = f'squeue -o "{SQUEUE_FORMAT}"'
                console.echo(message="Fetching queue status for all users...", level=LogLevel.INFO)
            else:
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
                if all_users:
                    console.echo(message="Queue status (squeue) for all users:")
                else:
                    console.echo(message=f"Queue status (squeue) for the user '{user}':")
                click.echo(formatted_output)
            elif all_users:
                console.echo(message="No jobs found in the queue.", level=LogLevel.WARNING)
            else:
                console.echo(message="No jobs found in the queue for the specified user.", level=LogLevel.WARNING)

    finally:
        server.close()


@server_cli.command("discover")
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project whose sessions to discover under the server's data root.",
)
def discover_project_sessions_command(project: str) -> None:
    """Discovers and prints the sessions stored under the project's directory on the remote compute server."""
    discover_project_data(project=project)
