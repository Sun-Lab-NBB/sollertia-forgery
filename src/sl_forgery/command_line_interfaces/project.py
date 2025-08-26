"""This module provides the Command Line Interfaces (CLIs) used to work with the data of any Sun lab project stored
on the remote compute server.
"""

from typing import Any

import click
from sl_shared_assets import Server, ProjectManifest, get_working_directory, get_credentials_file_path
from ataraxis_base_utilities import console

from ..processing import fetch_remote_project_manifest, generate_remote_project_manifest


@click.group("project")
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
        "working directory. This flag requires service access privileges and is not recommend for most use cases, as "
        "all lab pipelines automatically update the manifest file as part of their runtime."
    ),
)
@click.pass_context
def updated_manifest(ctx: Any, regenerate_manifest: bool) -> None:
    # Retrieves shared context data.
    project = ctx.obj["project"]

    # If requested, rebuilds the manifest file on the server before pulling it to the local machine. Manifest
    # regeneration requires service account credentials.
    if regenerate_manifest:
        # Establishes SSH connection to the processing server using the service account credentials.
        credentials = get_credentials_file_path(service=True)
        server = Server(credentials_path=credentials)
        generate_remote_project_manifest(project=project, server=server)

    # Otherwise, fetches the most recent manifest file instance from the remote server to the working directory
    else:
        # Establishes SSH connection to the processing server using the user account credentials.
        credentials = get_credentials_file_path(service=False)
        server = Server(credentials_path=credentials)
        fetch_remote_project_manifest(project=project, server=server)


@project_cli.command("print")
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
def print_project_manifest_data(
    project: str,
    animal: str | None,
    notes: bool,
    summary: bool,
) -> None:
    if not summary and not notes:
        message = (
            f"No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            f"'summary' (-s), or both flags when calling the command to display the data using the target format."
        )
        console.error(message=message, error=ValueError)

    # Resolves the path to the manifest file
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remove server before
    # continuing with this command.
    if not manifest_path.exists():
        # Establishes SSH connection to the processing server using the user account credentials.
        credentials = get_credentials_file_path(service=False)
        server = Server(credentials_path=credentials)
        fetch_remote_project_manifest(project=project, server=server)

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
