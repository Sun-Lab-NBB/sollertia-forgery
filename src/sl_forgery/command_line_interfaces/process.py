"""This module provides the Command-Line Interfaces (CLIs) for executing all data management, processing, and analysis
pipelines intended to run on the remote compute server. These CLIs are intended to be used exclusively by other
library components and should not be called directly by the end-users.
"""

from pathlib import Path

import click

from ..managing.processing import (
    resolve_checksum,
    transfer_session,
    generate_project_manifest,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("process", context_settings=CONTEXT_SETTINGS)
def process_cli() -> None:
    """This Command-Line Interface (CLI) allows executing local data management, processing, and analysis pipelines.

    This CLI is intended to run on the Sun lab remote compute server(s) and should not be called by the end-user
    directly. Instead, these commands are called by other library components to execute the requested data processing
    tasks.
    """


@process_cli.command("manifest")
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the project's root data directory.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    required=True,
    help="The unique identifier of this processing job.",
)
def generate_manifest(project_path: Path, job_id: str) -> None:
    """Generates the manifest .feather file that captures the snapshot of the target project's state."""
    generate_project_manifest(
        project_directory=project_path,
        job_id=job_id,
    )


@process_cli.command("checksum")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the processed session's root data directory.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    required=True,
    help="The unique identifier of this processing job.",
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
def resolve_session_checksum(session_path: Path, job_id: str, *, regenerate_checksum: bool) -> None:
    """Resolves the data integrity checksum for the target session's 'raw_data' directory.

    This command can be used to either verify the integrity of the session's data or to update the session's data
    integrity checksum to include expected changes.
    """
    resolve_checksum(
        session_path=session_path,
        job_id=job_id,
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
