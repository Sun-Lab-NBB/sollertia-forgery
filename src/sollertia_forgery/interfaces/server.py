"""Provides CLIs for directly interacting with the remote compute server."""

from __future__ import annotations

import click
from tabulate import tabulate
from ataraxis_base_utilities import LogLevel, console

from ..server import Server, discover_project_data, get_server_configuration, create_server_configuration_file

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""

_SACCT_FORMAT: str = "JobID,JobName%50,ReqMem,MaxRSS,AveRSS,MaxVMSize,NCPUS,AveCPU,Elapsed,State"
"""The format for the SLURM accounting 'sacct' command that reports a completed job's resource efficiency."""

_SACCT_HEADERS: list[str] = [
    "JobID",
    "JobName",
    "ReqMem",
    "MaxRSS",
    "AveRSS",
    "MaxVMSize",
    "NCPUS",
    "AveCPU",
    "Elapsed",
    "State",
]
"""The headers corresponding to _SACCT_FORMAT, used for display after merging rows."""

_SQUEUE_FORMAT: str = "%.10i %.9P %.50j %.8u %.8T %.6D %.6C %.10m %.10M %.12l %.12L"
"""The format for the SLURM queue 'squeue' command used to display running and pending jobs."""

_MINIMUM_SACCT_ROWS: int = 2
"""The smallest row count a usable 'sacct' response carries, being the header row plus one data row."""

_SACCT_COLUMN_COUNT: int = 10
"""The number of columns each parsed 'sacct' row carries."""


@click.group("server", context_settings=_CONTEXT_SETTINGS)
def server_cli() -> None:
    """Interacts with the remote Sollertia compute server.

    Authors the server access configuration, discovers a project's sessions, and reports SLURM queue and job
    accounting data.
    """


@server_cli.command("configure", context_settings=_CONTEXT_SETTINGS)
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
) -> None:
    """Creates the remote compute server configuration file in the Sollertia platform working directory."""
    create_server_configuration_file(
        username=username,
        password=password,
        host=host,
        root=root,
        environment=environment,
    )


@server_cli.command("print", context_settings=_CONTEXT_SETTINGS)
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

    # Loads and validates the local server access configuration.
    configuration = get_server_configuration()

    # Resolves the username from the server configuration file if an explicit override is not provided.
    if user is None:
        user = configuration.username

    all_users = user.lower() == "all"

    with Server(configuration=configuration) as server:
        if job_data:
            _report_job_accounting(
                server=server,
                user=user,
                job_id=job_id,
                start_time=start_time,
                end_time=end_time,
                all_users=all_users,
            )

        if queue:
            _report_queue_status(server=server, user=user, all_users=all_users)


@server_cli.command("discover", context_settings=_CONTEXT_SETTINGS)
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


def _report_job_accounting(
    server: Server,
    user: str,
    job_id: str | None,
    start_time: str | None,
    end_time: str | None,
    *,
    all_users: bool,
) -> None:
    """Reads the server's SLURM job accounting history and prints it as a formatted table.

    Args:
        server: The connected compute server to query.
        user: The username whose jobs to report.
        job_id: The single job to report, or None to report the jobs matching the remaining filters.
        start_time: The earliest start date a reported job may carry.
        end_time: The latest end date a reported job may carry.
        all_users: Determines whether to report the jobs of every user.

    Raises:
        RuntimeError: If the 'sacct' command fails on the server.
    """
    # A named job is reported on its own, which is what lets a caller bypass the user and date filtering.
    if job_id is not None:
        command = f'sacct -j {job_id} -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        console.echo(message=f"Fetching job accounting data for job ID '{job_id}'...", level=LogLevel.INFO)
    else:
        # Builds the command with optional user filtering.
        if all_users:
            command = f'sacct -a -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        else:
            command = f'sacct -u {user} -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        if start_time:
            command += f" --starttime={start_time}"
        if end_time:
            command += f" --endtime={end_time}"

        if all_users:
            console.echo(message="Fetching job accounting data for all users...", level=LogLevel.INFO)
        else:
            console.echo(message=f"Fetching job accounting data for the user '{user}'...", level=LogLevel.INFO)

    result = server.execute_command(command=command)

    if result.return_code != 0:
        message = (
            f"Unable to read the job accounting data from the remote compute server. The 'sacct' command exited "
            f"with the status {result.return_code} and reported: {result.stderr}."
        )
        console.error(message=message, error=RuntimeError)

    if not result.stdout.strip():
        console.echo(
            message="No job accounting data found for the specified filtering criteria.", level=LogLevel.WARNING
        )
        return

    formatted_output = _format_sacct_output(raw_output=result.stdout)
    if job_id is not None:
        console.echo(message=f"Job accounting (sacct) data for job ID '{job_id}':")
    elif all_users:
        console.echo(message="Job accounting (sacct) data for all users:")
    else:
        console.echo(message=f"Job accounting (sacct) data for the user '{user}':")
    click.echo(formatted_output)


def _report_queue_status(server: Server, user: str, *, all_users: bool) -> None:
    """Reads the server's SLURM job queue and prints it as a formatted table.

    Args:
        server: The connected compute server to query.
        user: The username whose queued jobs to report.
        all_users: Determines whether to report the queued jobs of every user.

    Raises:
        RuntimeError: If the 'squeue' command fails on the server.
    """
    if all_users:
        command = f'squeue -o "{_SQUEUE_FORMAT}"'
        console.echo(message="Fetching queue status for all users...", level=LogLevel.INFO)
    else:
        command = f'squeue -o "{_SQUEUE_FORMAT}" -u {user}'
        console.echo(message=f"Fetching queue status for user '{user}'...", level=LogLevel.INFO)

    result = server.execute_command(command=command)

    if result.return_code != 0:
        message = (
            f"Unable to read the job queue status from the remote compute server. The 'squeue' command exited with "
            f"the status {result.return_code} and reported: {result.stderr}."
        )
        console.error(message=message, error=RuntimeError)

    if not result.stdout.strip():
        if all_users:
            console.echo(message="No jobs found in the queue.", level=LogLevel.WARNING)
        else:
            console.echo(message="No jobs found in the queue for the specified user.", level=LogLevel.WARNING)
        return

    formatted_output = _format_slurm_output(raw_output=result.stdout)
    if all_users:
        console.echo(message="Queue status (squeue) for all users:")
    else:
        console.echo(message=f"Queue status (squeue) for the user '{user}':")
    click.echo(formatted_output)


def _format_slurm_output(raw_output: str) -> str:
    """Formats raw SLURM command output strings into nicely formatted tables.

    Args:
        raw_output: The raw output string from the SLURM 'squeue' command.

    Returns:
        The output as a formatted table, or a notice when it holds no rows.
    """
    lines = raw_output.strip().split("\n")
    if not lines:
        return "No data available."

    # A separator line carries only dashes and spaces, so dropping it leaves the header and the data rows.
    rows: list[list[str]] = [
        line.strip().split()
        for line in lines
        if line.strip() and not all(character in "- " for character in line.strip())
    ]
    if not rows:
        return "No data available."

    # Uses 'tabulate' to format the output, with the first row as headers.
    headers = rows[0]
    data = rows[1:]

    return tabulate(tabular_data=data, headers=headers, tablefmt="simple", colalign=["center"] * len(headers))


def _format_sacct_output(raw_output: str) -> str:
    """Formats raw 'sacct' output (parsable format) into a nicely formatted table with merged rows.

    Merges the rows sharing a base JobID, which covers a standard job whose parent row is followed by a '.batch'
    step row and a bash job whose steps repeat the same JobID.

    Args:
        raw_output: The raw output string from the 'sacct' command with the --parsable2 flag.

    Returns:
        The merged job data as a formatted table, or a notice when the output holds no usable row.
    """
    lines = raw_output.strip().split("\n")
    if not lines:
        return "No data available."

    # Parses pipe-delimited rows.
    rows = [line.split("|") for line in lines if line.strip()]
    if len(rows) < _MINIMUM_SACCT_ROWS:
        return "No data available."

    # The response repeats the requested format in its first row, so the predefined headers carry the display names.
    data = rows[1:]

    merged_data: list[list[str]] = []
    parent_jobs: dict[str, list[str]] = {}

    for row in data:
        if len(row) < _SACCT_COLUMN_COUNT:
            continue

        job_id = row[0]

        # A '.extern' step row carries no accounting figures, so it contributes nothing to the merged job row.
        if ".extern" in job_id:
            continue

        # Extracts the base job ID (without the '.batch' suffix if present).
        base_job_id = job_id.split(".")[0]

        if base_job_id in parent_jobs:
            parent_row = parent_jobs[base_job_id]
            for index in range(len(row[:_SACCT_COLUMN_COUNT])):
                if row[index] and not parent_row[index]:
                    parent_row[index] = row[index]
        else:
            # The first row carrying this job ID seeds the merged entry that later rows fill in.
            merged_row = list(row[:_SACCT_COLUMN_COUNT])
            parent_jobs[base_job_id] = merged_row
            merged_data.append(merged_row)

    if not merged_data:
        return "No data available."

    return tabulate(
        tabular_data=merged_data,
        headers=_SACCT_HEADERS,
        tablefmt="simple",
        colalign=["center"] * len(_SACCT_HEADERS),
    )
