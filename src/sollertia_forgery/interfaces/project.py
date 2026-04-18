"""Provides CLIs for interacting with Sollertia project manifest files."""

from pathlib import Path

import click
from ataraxis_base_utilities import console

from ..managing import generate_project_manifest
from ..shared_assets import ProjectManifest

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("project", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The path to the project's root data directory.",
)
def project_cli(ctx: click.Context, project_path: Path) -> None:
    """Provides commands for working with Sollertia projects stored on local machine."""
    ctx.ensure_object(dict)
    ctx.obj["project_path"] = project_path


@project_cli.command("print")
@click.option(
    "-a",
    "--animal",
    type=int,
    required=False,
    help=(
        "The identifier of the animal for which to print the manifest data. If not provided, this command prints "
        "the data for all animals participating in the target project."
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
        "Determines whether to regenerate the manifest file before loading it. Use this option to ensure the manifest "
        "reflects the latest state of the project's data."
    ),
)
@click.pass_context
def print_project_manifest_data(
    ctx: click.Context,
    *,
    animal: int | None,
    notes: bool,
    summary: bool,
    regenerate: bool,
) -> None:
    """Prints the requested data from the target project's manifest file to the terminal as a formatted table."""
    project_path: Path = ctx.obj["project_path"]

    if not summary and not notes:
        message = (
            "No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            "'summary' (-s), or both flags when calling the command."
        )
        console.error(message=message, error=ValueError)

    # Resolves the manifest path and regenerates if requested or absent.
    manifest_path = project_path.joinpath(f"{project_path.stem}_manifest.feather")
    if regenerate or not manifest_path.exists():
        generate_project_manifest(project_directory=project_path)

    manifest = ProjectManifest(manifest_file=manifest_path)

    # Ensures that the specified animal exists in the manifest data.
    if animal is not None and animal not in manifest.animals:
        message = (
            f"Unable to display the data for the target animal '{animal}', as it did not participate in the "
            f"target project '{project_path.stem}'."
        )
        console.error(message=message, error=ValueError)

    # If requested, prints the experimenter note view of the manifest data.
    if notes:
        manifest.print_notes(animal=animal)

    # If requested, prints the data processing view of the manifest data.
    if summary:
        manifest.print_summary(animal=animal)
