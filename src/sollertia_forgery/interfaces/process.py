"""Provides CLIs for executing data management, processing, and analysis pipelines exposed by the library."""

from pathlib import Path

import click
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import DatasetData, SessionData, SessionTypes, DatasetSession
from ataraxis_data_structures import delete_directory

from ..managing import resolve_checksum, transfer_session, generate_project_manifest
from ..processing import run_behavior_processing_pipeline
from ..forging.pipeline import run_forging_pipeline

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


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
    help="The absolute path to the transferred session's source data directory",
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


@process_cli.command("define")
@click.option(
    "-dn",
    "--dataset-name",
    type=str,
    required=True,
    help="The unique name for the dataset to create.",
)
@click.option(
    "-pr",
    "--project-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The path to the root directory of the project for which to create the dataset.",
)
@click.option(
    "-s",
    "--session",
    type=str,
    multiple=True,
    required=True,
    help=(
        "The session to include in the dataset specified using the 'session_name:animal_name' format. This argument "
        "can be specified multiple times to include multiple sessions."
    ),
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
def define_dataset_command(
    dataset_name: str,
    project_root: Path,
    session: tuple[str, ...],
    *,
    force_recreate: bool,
) -> None:
    """Defines a new analysis dataset by creating its data hierarchy and metadata files.

    Dataset creation is currently limited to mesoscope experiment sessions.
    """
    # Parses the session specifications into DatasetSession instances.
    sessions: list[DatasetSession] = []
    expected_parts = 2
    for session_spec in session:
        parts = session_spec.split(":")
        if len(parts) != expected_parts:
            message = (
                f"Invalid session specification '{session_spec}' encountered when defining the '{dataset_name}' "
                f"analysis dataset's data hierarchy. All session entries must follow the 'session_name:animal_name' "
                f"format."
            )
            console.error(message=message, error=ValueError)
        sessions.append(DatasetSession(session=parts[0], animal=parts[1]))

    # Removes the existing dataset hierarchy if the caller explicitly requested recreation.
    dataset_directory = project_root.joinpath(dataset_name)
    if force_recreate and dataset_directory.exists():
        delete_directory(directory_path=dataset_directory)
        console.echo(
            message=f"Dataset '{dataset_name}': Removed existing hierarchy for recreation.",
            level=LogLevel.INFO,
        )

    # Derives session type and acquisition system from the first session's metadata. Validates that the session type
    # is MESOSCOPE_EXPERIMENT (the only type currently supported by the forging pipeline).
    first_session_path = project_root.joinpath(sessions[0].animal, sessions[0].session)
    first_session_data = SessionData.load(session_path=first_session_path)
    if first_session_data.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
        message = (
            f"Unable to define dataset '{dataset_name}'. Dataset creation is currently supported only for mesoscope "
            f"experiment sessions, but the first session's type resolved to '{first_session_data.session_type}'."
        )
        console.error(message=message, error=ValueError)

    # Creates the dataset hierarchy and metadata files.
    dataset = DatasetData.create(
        name=dataset_name,
        project=project_root.name,
        session_type=first_session_data.session_type,
        acquisition_system=first_session_data.acquisition_system,
        sessions=tuple(sessions),
        datasets_root=project_root,
    )

    console.echo(
        message=(
            f"Dataset '{dataset_name}' data hierarchy: Defined with {len(sessions)} sessions from "
            f"{len(dataset.animals)} animals."
        ),
        level=LogLevel.SUCCESS,
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


@process_cli.command("assemble")
@click.option(
    "-dp",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The path to the dataset's root directory (containing dataset_data.yaml).",
)
@click.option(
    "-pr",
    "--project-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The path to the project's root directory that stores the animal and session data directories.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode)."
    ),
)
@click.option(
    "-t",
    "--target-session",
    type=str,
    default=None,
    help="If provided, limits the assembly to the specified session only.",
)
def assemble_dataset_command(
    dataset_path: Path,
    project_root: Path,
    job_id: str | None,
    target_session: str | None,
) -> None:
    """Assembles forged data for the target dataset's sessions.

    This command reads the dataset metadata and assembles each session's data into a unified data.feather file
    within the dataset hierarchy.
    """
    # Loads the existing dataset's metadata to extract its name and session set. The forging pipeline's internal
    # create-or-load path then reuses the on-disk definition instead of re-creating it.
    dataset = DatasetData.load(dataset_path=dataset_path)

    # Runs the unified forging pipeline.
    run_forging_pipeline(
        name=dataset.name,
        sessions=dataset.sessions,
        project_root=project_root,
        job_id=job_id,
        target_session=target_session,
        display_progress=True,
    )
