"""This module provides the Command Line Interfaces (CLIs) used to interact with the project's stored on the Sun lab's
remote compute server. These interfaces allow fetching and displaying the project's data and processing state snapshots
and 'adopting' the shared project's data for further processing by the calling user.
"""

import click
from sl_shared_assets import get_server_configuration
from ataraxis_base_utilities import console

from ..server import Server
from ..managing import adopt_project, resolve_project_manifest
from ..shared_assets import ProjectManifest

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
def project_cli(ctx: click.Context, project: str) -> None:
    """This Command-Line Interface (CLI) group allows working with Sun lab projects stored on the remote compute server.

    This CLI group is intended to be called on user machines as part of the shared Sun lab data workflow interface.
    Primarily, commands from this CLI group are intended to be used as entry-points for all further interactions with
    the target project's data.
    """
    ctx.ensure_object(dict)
    ctx.obj["project"] = project


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
@click.option(
    "-r",
    "--regenerate",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to regenerate the manifest file on the remote server before fetching it. Use this option "
        "to ensure the manifest reflects the latest state of the project's data on the server."
    ),
)
@click.pass_context
def print_project_manifest_data(
    ctx: click.Context,
    *,
    animal: str | None,
    notes: bool,
    summary: bool,
    regenerate: bool,
) -> None:
    """Prints the requested data from the target project's manifest file to the terminal as a formatted table."""
    # Retrieves shared context data.
    project = ctx.obj["project"]

    if not summary and not notes:
        message = (
            "No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            "'summary' (-s), or both flags when calling the command to display the data using the target format."
        )
        console.error(message=message, error=ValueError)

    # Establishes SSH connection to the processing server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    # Always fetches the manifest from the server. Regenerates if requested or if manifest doesn't exist.
    manifest_path = resolve_project_manifest(project=project, server=server, generate=regenerate)

    # Loads the manifest file data into memory.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Ensures that the specified animal exists in the manifest data.
    if animal is not None and animal not in manifest.animals:
        message = (
            f"Unable to display the data for the target animal '{animal}', as it did not participate in the "
            f"target project '{project}'."
        )
        console.error(message=message, error=ValueError)

    # If requested, prints the experimenter note view of the manifest data.
    if notes:
        manifest.print_notes(animal=animal)

    # If requested, prints the data processing view of the manifest data.
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
def adopt_project_data(ctx: click.Context, *, repeat_adoption: bool, keep_job_logs: bool) -> None:
    """Discovers and adopts all unadopted project sessions from the remote compute server.

    This command scans the project's directory on the shared server's volume, identifies sessions that have not yet
    been adopted (copied to the user's working directory), and executes the adoption pipeline followed by the data
    integrity verification pipeline. Adopting the project's session data in this way is the prerequisite for running
    all further processing and analysis workflows.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]

    # Executes the adoption process.
    adopt_project(
        project=project,
        repeat_adoption=repeat_adoption,
        keep_job_logs=keep_job_logs,
    )
