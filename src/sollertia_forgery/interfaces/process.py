"""Provides CLIs for executing data management, processing, and analysis pipelines exposed by the library."""

from pathlib import Path

import click

from ..forging import run_forging_pipeline
from ..managing import resolve_checksum, transfer_session, generate_project_manifest
from ..processing import run_behavior_processing_pipeline

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("process", context_settings=CONTEXT_SETTINGS)
def process_cli() -> None:
    """Executes data management, processing, or forging pipelines on the local machine."""


@process_cli.command("manifest")
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the project's root data directory.",
)
def generate_manifest(project_path: Path) -> None:
    """Generates the manifest .feather file that captures the snapshot of the target project's state."""
    generate_project_manifest(project_directory=project_path)


@process_cli.command("checksum")
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
def resolve_session_checksum(session_path: Path, *, regenerate_checksum: bool) -> None:
    """Resolves the data integrity checksum for the target session's 'raw_data' directory.

    This command can be used to either verify the integrity of the session's data or to update the session's data
    integrity checksum to include expected changes.
    """
    resolve_checksum(
        session_path=session_path,
        regenerate_checksum=regenerate_checksum,
    )


@process_cli.command("transfer")
@click.option(
    "-sp",
    "--source-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the transferred session's source data directory.",
)
@click.option(
    "-dp",
    "--destination-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help="The absolute path to the destination directory where to transfer the session's data.",
)
@click.option(
    "-rm",
    "--remove-source",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to delete the source session directory after completing the transfer. If the destination "
        "path is not provided, this command deletes the source session directory without transferring."
    ),
)
def transfer_session_data(source_path: Path, destination_path: Path | None, *, remove_source: bool) -> None:
    """Transfers the session's data from source to destination or deletes the source session.

    This command can be used to move session's data between storage locations or to delete the session data that is no
    longer needed.
    """
    transfer_session(
        source_path=source_path,
        destination_path=destination_path,
        remove_source=remove_source,
    )


@process_cli.command("behavior")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the session root directory to process.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching "
        "job (remote mode). If not provided, discovers and runs every available job for the session "
        "(local mode)."
    ),
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help=(
        "The number of worker processes to use for parallel processing. Set to -1 for automatic "
        "resolution via 'resolve_worker_count'. Set to 1 for sequential execution without spawning "
        "worker processes. Ignored when '--job-id' is provided (remote mode runs the single job "
        "in-process)."
    ),
)
@click.option(
    "-pr",
    "--progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display a progress bar during processing. Only meaningful in local mode.",
)
def run_behavior_pipeline_command(
    session_path: Path,
    job_id: str | None,
    workers: int,
    *,
    progress: bool,
) -> None:
    """Runs the behavior processing pipeline on the target session."""
    run_behavior_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@process_cli.command("forge")
@click.option(
    "-dn",
    "--dataset-name",
    type=str,
    required=True,
    help="The unique name for the dataset to create or use.",
)
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The path to the project's root data directory that stores the animal and session data directories.",
)
@click.option(
    "-s",
    "--session",
    type=str,
    multiple=True,
    help=(
        "The session name to include in the dataset. Can be specified multiple times. When the dataset already "
        "exists and sessions are provided, the set is verified against the existing definition. Omit to work "
        "with an already-defined dataset without triggering verification."
    ),
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching "
        "job (remote mode). If not provided, discovers and runs every available job for the dataset "
        "(local mode)."
    ),
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help=(
        "The number of worker processes to use for parallel processing. Set to -1 for automatic "
        "resolution via 'resolve_worker_count'. Set to 1 for sequential execution without spawning "
        "worker processes. Ignored when '--job-id' is provided (remote mode runs the single job "
        "in-process)."
    ),
)
@click.option(
    "-pr",
    "--progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display a progress bar during processing. Only meaningful in local mode.",
)
@click.option(
    "-f",
    "--force-recreate",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to delete any existing dataset hierarchy before creating it fresh. Use this flag when "
        "extending, shrinking, or modifying the session set of an existing dataset. Any prior assembled data and "
        "the existing processing tracker are discarded."
    ),
)
def run_forging_pipeline_command(
    dataset_name: str,
    project_path: Path,
    session: tuple[str, ...],
    job_id: str | None,
    workers: int,
    *,
    progress: bool,
    force_recreate: bool,
) -> None:
    """Runs the forging (dataset assembly) pipeline on the target dataset.

    Defines the dataset hierarchy if it does not exist, then executes per-session data assembly jobs. When
    called with an existing dataset and no session list, reuses the on-disk definition.
    """
    run_forging_pipeline(
        name=dataset_name,
        session_names=session,
        project_root=project_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
        force_recreate=force_recreate,
    )
