"""This module provides the Command Line Interfaces (CLIs) used to work with the data of any Sun lab project stored
on the remote compute server.
"""

from typing import Any

import click
from sl_shared_assets import get_working_directory, get_server_configuration
from ataraxis_base_utilities import console

from ..server import Server
from ..managing import adopt_project, manage_project_data, resolve_project_manifest
from ..shared_assets import ProjectManifest, SessionMetadata, filter_sessions

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("project", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project to work with.",
)
def project_cli(ctx: Any, project: str) -> None:
    """This Command-Line Interface (CLI) group allows working with Sun lab projects stored on the remote compute server.

    This CLI group is intended to be called on user machines as part of the shared Sun lab data workflow interface.
    Primarily, commands from this CLI group are intended to be used as entry-points for all further interactions with
    the target project's data.
    """
    ctx.ensure_object(dict)
    ctx.obj["project"] = project


@project_cli.command("update")
@click.option(
    "-rm",
    "--regenerate-manifest",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to regenerate the manifest file on the remote server before fetching it it to the local "
        "working directory."
    ),
)
@click.pass_context
def update_manifest(ctx: Any, regenerate_manifest: bool) -> None:
    """Actualizes the target project's manifest file stored on the local machine.

    The project manifest file communicates the current state of the project's data stored on the remote server, which
    informs all other pipelines accessible from this library on how to interact with the project's data. This command
    ensures that the local copy of the manifest file reflects the current state of the project's data stored on the
    remote compute server.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Resolves the project manifest file.
    resolve_project_manifest(project=project, server=server, generate=regenerate_manifest)


@project_cli.command("print")
@click.option(
    "-a",
    "--animal",
    type=str,
    required=False,
    help=(
        "The name of the animal for which to print the manifest data. If not provided, this command prints the data "
        "for all animals participating in the target project."
    ),
)
@click.option(
    "-n",
    "--notes",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to print the 'experimenter notes' view of the available manifest data. This data view is "
        "optimized for checking the outcome of each data acquisition session conducted for the target project."
    ),
)
@click.option(
    "-s",
    "--summary",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to print the 'data processing' view of the available manifest data. This view is optimized "
        "for tracking the data processing state of each data acquisition session conducted for the target project."
    ),
)
@click.pass_context
def print_project_manifest_data(
    ctx: Any,
    animal: str | None,
    notes: bool,
    summary: bool,
) -> None:
    """Prints the requested data from the target project's manifest file to the terminal as a formatted table.

    This command is designed to inform the user about the current state of the project's data stored on the remote
    server. It is recommended to always call the 'sl-project update' command before calling this command to ensure that
    the local manifest file contains up-to-date information.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]

    if not summary and not notes:
        message = (
            "No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            "'summary' (-s), or both flags when calling the command to display the data using the target format."
        )
        console.error(message=message, error=ValueError)

    # Resolves the path to the manifest file
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remote server before
    # continuing with this command.
    if not manifest_path.exists():
        # Establishes SSH connection to the processing server using the user account credentials.
        configuration = get_server_configuration()
        server = Server(configuration=configuration)
        resolve_project_manifest(project=project, server=server, generate=False)

    # Loads the manifest file data into memory
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Ensures that the specified animal exists in the manifest data.
    if animal is not None and animal not in manifest.animals:
        message = (
            f"Unable to display the data for the target animal '{animal}', as it did not participate in the "
            f"target project '{project}'."
        )
        console.error(message=message, error=ValueError)

    # If requested, prints the experimenter note view of the manifest data
    if notes:
        manifest.print_notes(animal=animal)

    # If requested, prints the data processing view of the manifest data
    if summary:
        manifest.print_summary(animal=animal)


@project_cli.command("adopt")
@click.option(
    "-r",
    "--repeat-adoption",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to re-adopt sessions that have already been adopted. If False (default), already-adopted "
        "sessions are skipped during the adoption stage."
    ),
)
@click.option(
    "-k",
    "--keep-job-logs",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to keep completed job logs on the server or (default) remove them after each pipeline "
        "completes successfully. If the pipeline fails, the job logs are kept regardless of this argument's value."
    ),
)
@click.pass_context
def adopt_project_data(ctx: Any, repeat_adoption: bool, keep_job_logs: bool) -> None:
    """Discovers and adopts all unadopted project sessions from the remote compute server.

    This command scans the project's directory on the shared server's volume, identifies sessions that have not yet
    been adopted (copied to the user's working directory), and executes the adoption pipeline followed by the data
    integrity verification pipeline for each session.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]

    # Executes the adoption process.
    adopt_project(
        project=project,
        repeat_adoption=repeat_adoption,
        keep_job_logs=keep_job_logs,
    )


@project_cli.command("manage")
@click.option(
    "-a",
    "--animal",
    type=str,
    multiple=True,
    help=(
        "The animal(s) whose sessions to manage. Can be specified multiple times to include multiple animals. "
        "If not specified, sessions from all animals are considered."
    ),
)
@click.option(
    "-s",
    "--session",
    type=str,
    multiple=True,
    help=(
        "The specific session(s) to manage. Can be specified multiple times to include multiple sessions. "
        "If not specified, all sessions matching other criteria are considered."
    ),
)
@click.option(
    "--start-date",
    type=str,
    required=False,
    help=(
        "The start date for filtering sessions (format: YYYY-MM-DD). Sessions recorded on or after this date are "
        "included."
    ),
)
@click.option(
    "--end-date",
    type=str,
    required=False,
    help=(
        "The end date for filtering sessions (format: YYYY-MM-DD). Sessions recorded on or before this date are "
        "included."
    ),
)
@click.option(
    "-vc",
    "--verify-checksum",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to verify the data integrity checksum for the target sessions.",
)
@click.option(
    "-rc",
    "--recompute-checksum",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to recompute (regenerate) the data integrity checksum for the target sessions. "
        "This overwrites the existing checksum stored in the ax_checksum.txt file for each session."
    ),
)
@click.option(
    "-d",
    "--delete",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to delete the target sessions from the user's working directory. If True, checksum "
        "operations are skipped."
    ),
)
@click.option(
    "-k",
    "--keep-job-logs",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to keep completed job logs on the server or (default) remove them after each pipeline "
        "completes successfully. If the pipeline fails, the job logs are kept regardless of this argument's value."
    ),
)
@click.pass_context
def manage_sessions(
    ctx: Any,
    animal: tuple[str, ...],
    session: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    verify_checksum: bool,
    recompute_checksum: bool,
    delete: bool,
    keep_job_logs: bool,
) -> None:
    """Manages the adopted project sessions on the remote compute server.

    This command allows verifying or recomputing the session's data integrity checksum or deleting the adopted
    session's data from the user's working directory. Use the filtering options to select which sessions to manage.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]

    # Resolves the path to the manifest file.
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remote server.
    if not manifest_path.exists():
        configuration = get_server_configuration()
        server = Server(configuration=configuration)
        resolve_project_manifest(project=project, server=server, generate=False)

    # Loads the manifest file data into memory.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Builds the set of all available sessions from the manifest.
    all_sessions: set[SessionMetadata] = set()
    for animal_id in manifest.animals:
        for session_name in manifest.get_sessions(animal=animal_id, exclude_incomplete=False):
            all_sessions.add(SessionMetadata(session=session_name, animal=animal_id))

    # Applies filtering based on the provided options.
    filtered_sessions = filter_sessions(
        sessions=all_sessions,
        start_date=start_date,
        end_date=end_date,
        include_sessions=set(session) if session else None,
        include_animals=set(animal) if animal else None,
    )

    # If no sessions match the filter criteria, raises an error.
    if not filtered_sessions:
        message = (
            "No sessions match the specified filtering criteria. Please adjust the filtering options and try again."
        )
        console.error(message=message, error=ValueError)

    # Executes the management operation.
    manage_project_data(
        manifest_path=manifest_path,
        project=project,
        sessions=tuple(filtered_sessions),
        verify_checksum=verify_checksum,
        recompute_checksum=recompute_checksum,
        delete_sessions=delete,
        keep_job_logs=keep_job_logs,
    )
