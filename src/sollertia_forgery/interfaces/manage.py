"""Provides the system-agnostic management CLI commands exposed by the ``slf`` root group: project manifest
generation and inspection, dataset forging-state snapshotting, session raw-data integrity checksum verification,
tracked-job resetting, and pipeline output cleaning.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import click
from ataraxis_base_utilities import console
from sollertia_shared_assets import DatasetData

from ..forging import generate_dataset_state
from ..managing import (
    ProjectManifest,
    project_manifest_path,
    generate_project_manifest,
    run_checksum_processing_pipeline,
)
from ..orchestration import BATCH_PIPELINES, reset_tracked_jobs, clean_pipeline_output

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@dataclass(frozen=True, slots=True)
class _SharedManifestParameters:
    """Bundles the option parsed on the ``manifest`` group and shared across its ``create`` and ``print``
    subcommands.
    """

    project_path: Path | None
    """The path to the project root data directory on which both subcommands operate."""

    def require_project_path(self) -> Path:
        """Returns the project root path, raising a Click usage error when ``--project-path`` was not supplied."""
        if self.project_path is None:
            message = (
                "Unable to resolve the project root directory for the 'manifest' command. The '-pp' / "
                "'--project-path' option must be supplied before the subcommand name, but it was omitted."
            )
            console.error(message=message, error=click.UsageError)
        return self.project_path


_pass_shared_parameters = click.make_pass_decorator(_SharedManifestParameters)
"""Injects the ``manifest`` group's ``_SharedManifestParameters`` as each subcommand's first argument."""


@click.group("manifest", context_settings=_CONTEXT_SETTINGS)
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


@manifest_cli.command("create", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to suppress the preamble and completion messages during manifest generation. These "
        "messages are displayed by default."
    ),
)
@_pass_shared_parameters
def create_manifest(shared: _SharedManifestParameters, *, no_progress: bool) -> None:
    """Creates the manifest .feather file that captures the snapshot of the target project's state.

    An existing manifest for the project is recreated (overwritten) with a fresh snapshot.
    """
    generate_project_manifest(project_directory=shared.require_project_path(), display_progress=not no_progress)


@manifest_cli.command("print", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-a",
    "--animal",
    type=str,
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
@_pass_shared_parameters
def print_project_manifest_data(
    shared: _SharedManifestParameters,
    *,
    animal: str | None,
    notes: bool,
    summary: bool,
) -> None:
    """Prints the requested data from the target project's manifest file to the terminal as a formatted table."""
    if not summary and not notes:
        message = (
            "Unable to print the manifest data. At least one of the '--notes' (-n) and '--summary' (-s) "
            "display flags must be supplied, but neither was given."
        )
        console.error(message=message, error=ValueError)

    # Printing reads an existing manifest snapshot. Generation is a separate step ('manifest create'), so a missing
    # manifest is a loud error rather than an implicit regeneration.
    project_path = shared.require_project_path()
    manifest_path = project_manifest_path(project_directory=project_path)
    if not manifest_path.exists():
        message = (
            f"Unable to print the manifest data for the '{project_path.stem}' project. No manifest file exists at "
            f"'{manifest_path}'. Generate it first with 'slf manifest -pp {project_path} create'."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = ProjectManifest(manifest_file=manifest_path)

    if animal is not None and animal not in manifest.animals:
        message = (
            f"Unable to display the data for the target animal '{animal}', as it did not participate in the "
            f"target project '{project_path.stem}'."
        )
        console.error(message=message, error=ValueError)

    if notes:
        manifest.print_notes(animal=animal)

    if summary:
        manifest.print_summary(animal=animal)


@click.command("checksum", context_settings=_CONTEXT_SETTINGS)
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
@click.option(
    "-w",
    "--workers",
    type=int,
    show_default=True,
    default=-1,
    help=(
        "The number of parallel worker processes to use for hashing the session's files. Values below 1 request all "
        "available cores minus the reserved system cores, and a value of 1 disables parallelism."
    ),
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to suppress the preamble message and progress bar during checksum resolution. These "
        "are displayed by default."
    ),
)
def checksum_command(session_path: Path, workers: int, *, regenerate_checksum: bool, no_progress: bool) -> None:
    """Resolves the data integrity checksum for the target session's 'raw_data' directory.

    Verifies the stored checksum by default, and recalculates it to absorb expected changes when --regenerate-checksum
    is given.
    """
    run_checksum_processing_pipeline(
        session_path=session_path,
        regenerate_checksum=regenerate_checksum,
        workers=workers,
        display_progress=not no_progress,
    )


@click.command("dataset-state", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-dp",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help="The absolute path to a dataset root directory to snapshot. Can be specified multiple times.",
)
def dataset_state_command(dataset_path: tuple[Path, ...]) -> None:
    """Snapshots each named dataset's forging job state into a shippable table at the dataset root.

    The snapshot is what carries a dataset's forging progress to another host, since the project manifest reports one
    row per session while a dataset's jobs sit at differing scopes.
    """
    for path in dataset_path:
        dataset = DatasetData.load(dataset_path=path)
        console.echo(message=str(generate_dataset_state(dataset=dataset, display_progress=True)), raw=True)


@click.command("reset", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(sorted(str(pipeline) for pipeline in BATCH_PIPELINES), case_sensitive=False),
    required=True,
    help="The pipeline whose tracked jobs to reset.",
)
@click.option(
    "-up",
    "--unit-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help=(
        "The absolute path to a processing unit whose jobs to reset, which is a session root for a session pipeline "
        "and a dataset root for 'forging'. Can be specified multiple times."
    ),
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=(),
    multiple=True,
    help=(
        "The hexadecimal identifier of a tracked job to reset. Can be specified multiple times. Omit to reset every "
        "job each named unit tracks."
    ),
)
def reset_command(pipeline: str, unit_path: tuple[Path, ...], job_id: tuple[str, ...]) -> None:
    """Returns tracked jobs of the named units to the scheduled state, leaving every untargeted job's record intact.

    One invocation covers every named unit, and each unit resets only the identifiers it actually tracks, so a batch
    spanning many units costs a single call.
    """
    reset = reset_tracked_jobs(pipeline=pipeline, unit_paths=unit_path, job_ids=job_id)
    console.echo(message=f"Reset {len(reset)} job(s) across {len(unit_path)} unit(s).")


@click.command("clean", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(sorted(str(pipeline) for pipeline in BATCH_PIPELINES), case_sensitive=False),
    required=True,
    help="The pipeline whose output to remove.",
)
@click.option(
    "-up",
    "--unit-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help="The absolute path to a processing unit to clean. Can be specified multiple times.",
)
def clean_command(pipeline: str, unit_path: tuple[Path, ...]) -> None:
    """Removes a pipeline's output and processing tracker for the named units.

    Returns each unit to an unprocessed state, so a later preparation rediscovers every job from the acquired data
    rather than resuming a partial run. Each removed path is reported with the bytes it held, one per line.
    """
    for removed in clean_pipeline_output(pipeline=pipeline, unit_paths=unit_path):
        console.echo(message=f"{removed['removed_bytes']} {removed['path']}", raw=True)
