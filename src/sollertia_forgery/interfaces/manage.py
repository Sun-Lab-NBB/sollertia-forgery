"""Provides the system-agnostic management CLI commands exposed by the ``slf`` root group: project manifest
generation and inspection, and session raw-data integrity checksum verification.
"""

from pathlib import Path
from dataclasses import dataclass

import click
from ataraxis_base_utilities import console

from ..managing import ProjectManifest, resolve_checksum, generate_project_manifest

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@dataclass(frozen=True, slots=True)
class _SharedManifestParameters:
    """Bundles the option parsed on the ``manifest`` group and shared across its ``generate`` and ``print``
    subcommands.

    The group callback builds one of these from its option and stores it on the Click context, and each subcommand
    reads it back through the ``_pass_shared_parameters`` decorator.
    """

    project_path: Path | None
    """The path to the project root data directory both subcommands operate on."""

    def require_project_path(self) -> Path:
        """Returns the project root path, raising a Click usage error when ``--project-path`` was not supplied."""
        if self.project_path is None:
            message = "Missing option '-pp' / '--project-path'."
            raise click.UsageError(message=message)
        return self.project_path


_pass_shared_parameters = click.make_pass_decorator(_SharedManifestParameters)
"""Injects the ``manifest`` group's ``_SharedManifestParameters`` as each subcommand's first argument."""


@click.group("manifest", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="The absolute path to the project's root data directory.",
)
@click.pass_context
def manifest_cli(context: click.Context, project_path: Path | None) -> None:
    """Generates and inspects the project manifest .feather file that snapshots a project's state.

    The project path is parsed on this group and shared by every subcommand, so it must be given before the
    subcommand name.
    """
    context.obj = _SharedManifestParameters(project_path=project_path)


@manifest_cli.command("create", context_settings=CONTEXT_SETTINGS)
@_pass_shared_parameters
def create_manifest(shared: _SharedManifestParameters) -> None:
    """Creates the manifest .feather file that captures the snapshot of the target project's state.

    An existing manifest for the project is recreated (overwritten) with a fresh snapshot.
    """
    generate_project_manifest(project_directory=shared.require_project_path())


@manifest_cli.command("print", context_settings=CONTEXT_SETTINGS)
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
@_pass_shared_parameters
def print_project_manifest_data(
    shared: _SharedManifestParameters,
    *,
    animal: int | None,
    notes: bool,
    summary: bool,
    regenerate: bool,
) -> None:
    """Prints the requested data from the target project's manifest file to the terminal as a formatted table."""
    if not summary and not notes:
        message = (
            "No data display options were selected when calling the command. Pass either the 'notes' (-n), "
            "'summary' (-s), or both flags when calling the command."
        )
        console.error(message=message, error=ValueError)

    # Resolves the manifest path and regenerates if requested or absent.
    project_path = shared.require_project_path()
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


@click.command("checksum", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the processed session's root data directory.",
)
@click.option(
    "-rc",
    "--regenerate-checksum",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to recalculate and overwrite the cached session's checksum value. When "
        "the command is called with this flag, it re-checksums the data instead of verifying its integrity."
    ),
)
def checksum_command(session_path: Path, *, regenerate_checksum: bool) -> None:
    """Resolves the data integrity checksum for the target session's 'raw_data' directory.

    This command can be used to either verify the integrity of the session's data or to update the session's data
    integrity checksum to include expected changes.
    """
    resolve_checksum(session_path=session_path, regenerate_checksum=regenerate_checksum)
