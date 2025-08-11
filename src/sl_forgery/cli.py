"""This module provides the Command Line Interfaces (CLIs) exposed by the library upon installation into a python
environment.
"""

from pathlib import Path

import click
from sl_shared_assets import Server, ProjectManifest
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from .utils import get_working_directory, set_working_directory, get_credentials_file_path
from .processing import fetch_remote_project_manifest, generate_remote_project_manifest


@click.command()
@click.option(
    "-d",
    "--directory",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the directory to use for working with Sun lab data.",
)
def designate_working_directory(directory: Path) -> None:
    """Sets the input directory as the Sun lab working directory, creating any missing directory path components.

    All future calls to this library will use this directory to store the intermediate data required to perform the
    requested task. This system allows the library to behave consistently across different user machines and runtime
    contexts.
    """
    # Creates the directory if it does not exist
    ensure_directory_exists(directory)

    # Sets the directory as the local working directory
    set_working_directory(path=directory)

    console.echo(message=f"Sun lab working directory set to: {directory}.", level=LogLevel.SUCCESS)


@click.command()
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to print the manifest data.",
)
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
@click.option(
    "-u",
    "--update_manifest",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to fetch the most recent project manifest version stored on the remote server before "
        "displaying the data. Since the manifest file is cached locally, this option is only required if the project "
        "data stored on the server has updated since the last call to this CLI."
    ),
)
@click.option(
    "-r",
    "--regenerate_manifest",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to regenerate the manifest file on the remote server before fetching it it to the local "
        "working directory. This flag requires service access privileges and is not recommend for most use cases, as "
        "all lab pipelines automatically update the manifest file as part of their runtime."
    ),
)
def print_project_manifest_data(
    project: str,
    animal: str | None,
    notes: bool,
    summary: bool,
    update_manifest: bool,
    regenerate_manifest: bool,
) -> None:
    if not summary and not notes:
        message = (
            f"No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            f"'summary' (-s), or both flags when calling the command to display the data using the target format."
        )
        console.error(message=message, error=ValueError)

    # Resolves the path to the manifest file
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remove server
    if not manifest_path.exists() and not regenerate_manifest and not update_manifest:
        update_manifest = True

    # If requested, fetches the most recent manifest file instance from the remote server to the working directory
    # before printing the project data. Note, the default is expected to be 'update' as it does not requre service
    # account credentials.
    if update_manifest:
        # Establishes SSH connection to the processing server using the user account credentials.
        credentials = get_credentials_file_path(require_service=False)
        server = Server(credentials_path=credentials)
        fetch_remote_project_manifest(project=project, server=server)

    # Manifest regeneration requires service account credentials and re-creates the manifest before fetching it to the
    # local machine.
    elif regenerate_manifest:
        # Establishes SSH connection to the processing server using the service account credentials.
        credentials = get_credentials_file_path(require_service=True)
        server = Server(credentials_path=credentials)
        generate_remote_project_manifest(project=project, server=server)

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
