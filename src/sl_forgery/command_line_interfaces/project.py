"""This module provides the Command Line Interfaces (CLIs) used to work with the data of any Sun lab project stored
on the remote compute server.
"""

from typing import Any

import click
from sl_shared_assets import get_working_directory, get_server_configuration
from ataraxis_base_utilities import console

from ..server import Server
from ..managing import resolve_project_manifest
from ..managing.processing import ProjectManifest

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
