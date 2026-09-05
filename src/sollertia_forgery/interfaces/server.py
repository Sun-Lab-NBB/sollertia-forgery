"""Provides CLIs for directly interacting with the remote compute server."""

from __future__ import annotations

import shlex
from typing import Any
from pathlib import Path

import click
from tabulate import tabulate
from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import LogLevel, console

from ..server import Server, discover_project_data, get_server_configuration, create_server_configuration_file
from .remote_tools import remote_batch_retire, remote_batch_status
from ..orchestration import STALLED_BATCH

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

_SQUEUE_MISSING_JOB_ERROR: str = "Invalid job id specified"
"""The error text 'squeue' reports when it is asked for a job the SLURM controller no longer holds. The command exits
with a nonzero status in that case, which is the normal outcome of looking up any job that has already finished. The
queue report therefore treats this text as a successful command rather than as a failure. The report calls the queue
empty only when the command also returned no rows, since a caller may name several jobs at once and a list mixing
purged and live jobs still returns the rows of the live ones."""

_BATCH_HEADERS: list[str] = [
    "batch_id",
    "progress",
    "outstanding_h",
    "jobs",
    "running",
    "stranded",
    "gone",
    "pipelines",
]
"""The headers of the outstanding-batch report. It pairs each batch's size with the verdict on whether the scheduler is
still advancing it. It also carries the counts of the two verdicts on which a caller acts and the count of the
allocations for which neither scheduler record answers."""

_ALLOCATION_HEADERS: list[str] = [
    "batch_id",
    "allocation",
    "job_name",
    "specifier",
    "scheduler",
    "tracker",
    "verdict",
    "remediation",
]
"""The headers of the per-allocation report, which pairs the state in which each record places an allocation with the
verdict they carry together and the remediation that verdict prescribes."""

_REMEDIATION_HEADERS: list[str] = [
    "allocation",
    "job_name",
    "specifier",
    "verdict",
    "cancelled",
    "reset",
    "snapshot",
    "dropped",
]
"""The headers of the remediation report, which pairs each allocation's verdict with each of the four steps that
verdict either did or did not apply to it."""

_MINIMUM_SACCT_ROWS: int = 2
"""The smallest row count a usable 'sacct' response carries, being the header row plus one data row."""

_SACCT_COLUMN_COUNT: int = 10
"""The number of columns each parsed 'sacct' row carries."""


@click.group("server", context_settings=_CONTEXT_SETTINGS)
def server_cli() -> None:
    """Interacts with the remote Sollertia compute server.

    Authors the server access configuration, discovers a project's sessions, reports SLURM queue and job accounting
    data, and resolves and remediates the batches this machine has outstanding on the scheduler.
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
        "Determines whether to display the remote server's job accounting history (runtime statistics) using "
        "SLURM's 'sacct' command."
    ),
)
@click.option(
    "-q",
    "--queue",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display the remote server's job queue status using SLURM's 'squeue' command.",
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
    help=(
        "Determines the job for which to display the queue and accounting data. Bypasses user and date filtering "
        "options."
    ),
)
@click.option(
    "-st",
    "--start-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that started on or after this date "
        "(format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Only applies to the job accounting ('--job-data') data, as the "
        "queue reports the currently active jobs and supports no date window."
    ),
)
@click.option(
    "-et",
    "--end-time",
    type=str,
    required=False,
    help=(
        "Allows filtering displayed job data to only include the jobs that ended on or before this date "
        "(format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Only applies to the job accounting ('--job-data') data, as the "
        "queue reports the currently active jobs and supports no date window."
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
    """Displays the remote server's SLURM queue status or job data as a formatted table."""
    if not job_data and not queue:
        message = (
            "No data display options were selected when calling the command. Pass either the '--job-data' (-j), "
            "'--queue' (-q), or both flags to display the requested remote server's SLURM information."
        )
        console.error(message=message, error=ValueError)

    configuration = get_server_configuration()

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
            _report_queue_status(server=server, user=user, job_id=job_id, all_users=all_users)


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


@server_cli.command("batches", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-b",
    "--batch-id",
    type=str,
    default=(),
    multiple=True,
    help=(
        "The identifier of an outstanding batch to report. Can be specified multiple times. Omit to report every "
        "outstanding batch."
    ),
)
@click.option(
    "-a",
    "--allocations",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to additionally report every allocation on its own row, carrying the state each "
        "scheduler record places it in, what its job's tracker holds, and the verdict they carry together."
    ),
)
def report_remote_batches_command(batch_id: tuple[str, ...], *, allocations: bool) -> None:
    """Reports the batches this machine still has outstanding on the remote compute server's scheduler.

    Resolves every allocation the submission ledger holds against three records, which are the scheduler's accounting,
    the scheduler's queue, and the processing tracker of the job the allocation carries. Reports per batch how long it
    has been outstanding and the state to which those records resolve. A batch is 'stalled' when none of its
    allocations resolves as running and at least one is gone from both scheduler records, which no later query changes,
    and the reported remedy names the command that remediates it. Closes and retires any batch whose allocations all
    resolve to a plain drop, exactly as the agentic status read does.
    """
    response = remote_batch_status(batch_ids=list(batch_id) or None, limit=0, include_items=allocations)
    _reject_failed_response(response=response)

    # A read that found nothing outstanding never reached the scheduler, so its response carries no reading to report.
    if response.get("scheduler_read_error"):
        console.echo(message=response["scheduler_read_error"], level=LogLevel.WARNING)

    batches = response.get("batches")
    if not batches:
        console.echo(message=response["message"], level=LogLevel.WARNING)
        return

    rows = [
        [
            batch["batch_id"],
            batch["progress"],
            _format_outstanding(seconds=batch["outstanding_seconds"]),
            batch["total_jobs"],
            batch["live_allocations"],
            batch["stranded_allocation_count"],
            batch["unresolvable_allocation_count"],
            ", ".join(batch["pipelines"]),
        ]
        for batch in batches
    ]
    console.echo(message="Outstanding remote batches:")
    console.echo(
        message=tabulate(
            tabular_data=rows, headers=_BATCH_HEADERS, tablefmt="simple", colalign=["center"] * len(_BATCH_HEADERS)
        ),
        raw=True,
    )

    if allocations:
        _report_allocations(jobs=response["jobs"])

    for batch in batches:
        if batch["progress"] == STALLED_BATCH:
            console.echo(
                message=(
                    f"Batch '{batch['batch_id']}' holds allocation(s) {batch['unresolvable_allocations']} that both "
                    f"scheduler records disclaim. {batch['remedy']}"
                ),
                level=LogLevel.WARNING,
            )


@server_cli.command("retire-batch", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-b",
    "--batch-id",
    type=str,
    required=True,
    multiple=True,
    help="The identifier of an outstanding batch to retire. Can be specified multiple times.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to remediate batches holding an allocation that resolves as running. Cancels each such "
        "allocation before any tracker is written, and abandons one whose state could not be read at all."
    ),
)
@click.option(
    "-do",
    "--drop-without-outcome",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to drop the ledger entries when what their jobs recorded cannot be snapshotted, which an "
        "unreachable server is one cause of."
    ),
)
def retire_remote_batch_command(batch_id: tuple[str, ...], *, force: bool, drop_without_outcome: bool) -> None:
    """Remediates the named batches and drops them from this machine's submission ledger.

    This is the remedy for a batch the 'batches' report calls stalled, whose ledger entry would otherwise keep it
    outstanding forever and keep its jobs claimed against a rerun. Every allocation is resolved exactly as that report
    resolves it, and a job its own tracker still claims to be running while no allocation is, which the report calls
    stranded, is returned to the scheduled state. A job that recorded success or failure keeps that record, so nothing
    a run produced is discarded here. What the batch's jobs recorded is then snapshotted, through the same closure
    applied to a finished batch, so the run stays answerable once the entry is gone. A batch holding an allocation
    that resolves as running is refused without '--force', and a snapshot that fails refuses the drop without
    '--drop-without-outcome'. This drops the ledger entries and replaces each covered batch's prepared document with
    its recorded outcome, leaving the outcomes themselves in the batch registry.
    """
    response = remote_batch_retire(batch_ids=list(batch_id), force=force, drop_without_outcome=drop_without_outcome)
    _reject_failed_response(response=response)

    if response["snapshot_error"]:
        console.echo(message=response["snapshot_error"], level=LogLevel.WARNING)
    for batch in response["batches"]:
        console.echo(
            message=f"Retired '{batch['batch_id']}', which held allocation(s) {batch['allocations']}.",
            level=LogLevel.SUCCESS,
        )

    rows = [
        [
            allocation["slurm_job_id"],
            allocation["job_name"],
            allocation["specifier"],
            allocation["verdict"],
            allocation["cancelled"],
            allocation["tracker_reset"],
            allocation["snapshot_recorded"],
            allocation["entry_dropped"],
        ]
        for allocation in response["allocations"]
    ]
    if rows:
        console.echo(
            message=tabulate(
                tabular_data=rows,
                headers=_REMEDIATION_HEADERS,
                tablefmt="simple",
                colalign=["center"] * len(_REMEDIATION_HEADERS),
            ),
            raw=True,
        )
    console.echo(message=f"{response['message']} Outcomes: {response['outcome_directory']}.")


def _report_allocations(jobs: list[dict[str, Any]]) -> None:
    """Prints one row per resolved allocation, carrying the verdict and the remediation it prescribes.

    Notes:
        A listed row leaves out the fields that hold nothing, so the specifier and the tracker status are read with
        a default rather than indexed. A job that never started carries no tracker status, and a job of a single-stage
        pipeline carries no specifier.

    Args:
        jobs: The allocation rows the status read listed.
    """
    rows = [
        [
            job["batch_id"],
            job["slurm_job_id"],
            job["job_name"],
            job.get("specifier", ""),
            job["scheduler_state"],
            job.get("tracker_status", ""),
            job["verdict"],
            job["remediation"],
        ]
        for job in jobs
    ]
    if not rows:
        return

    console.echo(message="Resolved allocations:")
    console.echo(
        message=tabulate(
            tabular_data=rows,
            headers=_ALLOCATION_HEADERS,
            tablefmt="simple",
            colalign=["center"] * len(_ALLOCATION_HEADERS),
        ),
        raw=True,
    )


def _reject_failed_response(response: dict[str, Any]) -> None:
    """Raises the error a batch tool reported, so a failed command exits rather than printing an empty report.

    Args:
        response: The response the tool returned.

    Raises:
        RuntimeError: If the response reports a failure.
    """
    if not response["success"]:
        message: str = response["error"]
        console.error(message=message, error=RuntimeError)


def _format_outstanding(seconds: float | None) -> str:
    """Renders how long a batch has been outstanding as the hours the report displays.

    Args:
        seconds: The elapsed seconds, or None when the ledger record carries no submission time.

    Returns:
        The elapsed hours, or a notice that the record dates the submission nowhere.
    """
    if seconds is None:
        return "unknown"
    hours = convert_time(time=seconds, from_units=TimeUnits.SECOND, to_units=TimeUnits.HOUR, as_float=True)
    return f"{hours:.2f}"


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
    # A named job is reported on its own, so a caller bypasses the user and date filtering.
    if job_id is not None:
        command = f'sacct -j {shlex.quote(job_id)} -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        console.echo(message=f"Fetching job accounting data for job ID '{job_id}'...", level=LogLevel.INFO)
    else:
        if all_users:
            command = f'sacct -a -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        else:
            command = f'sacct -u {user} -o "{_SACCT_FORMAT}" --parsable2 --units=G'
        if start_time is not None:
            command += f" --starttime={start_time}"
        if end_time is not None:
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
    console.echo(message=formatted_output, raw=True)


def _report_queue_status(server: Server, user: str, job_id: str | None, *, all_users: bool) -> None:
    """Reads the server's SLURM job queue and prints it as a formatted table.

    Notes:
        The 'squeue' command exits with an error when it is asked for a job the SLURM controller no longer holds,
        which is the normal state of every job that has already finished. That outcome is not treated as a command
        failure, so only a genuine failure is raised. It is reported as an empty queue result only when the command
        resolved no job at all. A caller may name several jobs at once, and a list mixing purged and live jobs still
        returns the rows of the live ones.

    Args:
        server: The connected compute server to query.
        user: The username whose queued jobs to report.
        job_id: The single job to report, or None to report the jobs matching the remaining filters.
        all_users: Determines whether to report the queued jobs of every user.

    Raises:
        RuntimeError: If the 'squeue' command fails on the server.
    """
    # A named job is reported on its own, so a caller bypasses the user filtering.
    if job_id is not None:
        command = f'squeue -o "{_SQUEUE_FORMAT}" -j {shlex.quote(job_id)}'
        console.echo(message=f"Fetching queue status for job ID '{job_id}'...", level=LogLevel.INFO)
    elif all_users:
        command = f'squeue -o "{_SQUEUE_FORMAT}"'
        console.echo(message="Fetching queue status for all users...", level=LogLevel.INFO)
    else:
        command = f'squeue -o "{_SQUEUE_FORMAT}" -u {user}'
        console.echo(message=f"Fetching queue status for user '{user}'...", level=LogLevel.INFO)

    result = server.execute_command(command=command)

    # A job the controller no longer holds makes 'squeue' exit with an error, so that lookup is separated from a
    # genuine command failure before the shared guard below.
    missing_job = job_id is not None and result.return_code != 0 and _SQUEUE_MISSING_JOB_ERROR in result.stderr

    # A list naming both purged and live jobs still writes the live rows to the output, so the missing-job notice is
    # given only when the command resolved nothing at all. Otherwise, the resolved rows are formatted as usual.
    if missing_job and not result.stdout.strip():
        console.echo(
            message=f"The queue no longer holds the job ID '{job_id}', which is expected for a job that has already "
            f"finished.",
            level=LogLevel.WARNING,
        )
        return

    if result.return_code != 0 and not missing_job:
        message = (
            f"Unable to read the job queue status from the remote compute server. The 'squeue' command exited with "
            f"the status {result.return_code} and reported: {result.stderr}."
        )
        console.error(message=message, error=RuntimeError)

    if not result.stdout.strip():
        if job_id is not None:
            console.echo(message="No jobs found in the queue for the specified job ID.", level=LogLevel.WARNING)
        elif all_users:
            console.echo(message="No jobs found in the queue.", level=LogLevel.WARNING)
        else:
            console.echo(message="No jobs found in the queue for the specified user.", level=LogLevel.WARNING)
        return

    formatted_output = _format_slurm_output(raw_output=result.stdout)
    if job_id is not None:
        console.echo(message=f"Queue status (squeue) for job ID '{job_id}':")
    elif all_users:
        console.echo(message="Queue status (squeue) for all users:")
    else:
        console.echo(message=f"Queue status (squeue) for the user '{user}':")
    console.echo(message=formatted_output, raw=True)


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


@server_cli.command("pull", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-r",
    "--remote-path",
    type=str,
    required=True,
    help="The absolute path, on the server, to the file or directory to copy.",
)
@click.option(
    "-d",
    "--destination",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the local directory that receives the copy.",
)
def pull_command(remote_path: str, destination: Path) -> None:
    """Copies a file or directory off the remote compute server onto this machine.

    A directory is copied whole. The copy lands inside the destination directory under the remote path's own final
    component, which is how a session's processed data, one feather, or a batch's logs are brought back.
    """
    source = Path(remote_path)
    with Server(configuration=get_server_configuration()) as server:
        if not server.exists(remote_path=source):
            message = (
                f"Unable to copy '{remote_path}' off the compute server. The server holds no file or directory at "
                f"that path."
            )
            console.error(message=message, error=FileNotFoundError)

        destination.mkdir(parents=True, exist_ok=True)
        local_path = destination.joinpath(source.name)
        server.pull(local_path=local_path, remote_path=source)

    copied = sorted(path for path in local_path.rglob("*") if path.is_file()) if local_path.is_dir() else [local_path]
    total = sum(path.stat().st_size for path in copied)
    console.echo(
        message=f"Copied {len(copied)} file(s), {total / 1024**2:.1f} MB, to '{local_path}'.",
        level=LogLevel.SUCCESS,
    )
