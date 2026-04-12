"""Provides assets for transferring or deleting session data directories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import delete_directory, transfer_directory

if TYPE_CHECKING:
    from pathlib import Path


def transfer_session(
    source_path: Path,
    destination_path: Path | None = None,
    *,
    remove_source: bool = False,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Transfers the session's data from source to destination or deletes the session's data directory.

    Notes:
        Deletes the source directory instead of transferring it when the destination path is None and remove_source
        is True.

    Args:
        source_path: The path to the source session's data directory to be transferred.
        destination_path: The path to the destination directory where to transfer the session's data.
        remove_source: Determines whether to delete the source session directory after completing the transfer. If the
            destination path is not provided, this command deletes the source session directory without transferring.
        workers: The number of threads to use for the file transfer. Setting this to a value less than 1 uses all
            available CPU cores (minus reserved cores). Setting this to 1 conducts the transfer sequentially without
            spawning additional threads.
        display_progress: Determines whether to display console messages and progress bars during the transfer.

    Raises:
        ValueError: If no destination path is provided and source removal is not requested.
        FileNotFoundError: If the source path does not contain a valid session data hierarchy.
    """
    # Validates the source path by loading the session data and reconstructs the session root relative to the
    # resolved session_data.yaml location. This ensures only well-formed sessions are transferred or deleted.
    session_data = SessionData.load(session_path=source_path)
    session_root = session_data.raw_data_path.parent

    # If the destination path is None and remove_source is True, deletes the source directory.
    if destination_path is None and remove_source:
        if display_progress:
            console.echo(
                message=f"Deleting the session '{session_data.session_name}' at '{session_root}'...",
                level=LogLevel.INFO,
            )
        delete_directory(directory_path=session_root)
        if display_progress:
            console.echo(
                message=f"Session '{session_data.session_name}': Deleted.",
                level=LogLevel.SUCCESS,
            )
        return

    elif destination_path is None:
        message = (
            f"Unable to transfer the session '{session_data.session_name}' at '{session_root}'. No destination "
            f"path was provided and source removal was not requested. Provide a destination path to transfer the "
            f"session or set remove_source to True to delete it."
        )
        console.error(message=message, error=ValueError)

    if display_progress:
        console.echo(
            message=f"Transferring the session '{session_data.session_name}' to '{destination_path}'...",
            level=LogLevel.INFO,
        )

    # Resolves the thread count and transfers the session data to the destination.
    resolved_workers = resolve_worker_count(requested_workers=workers)
    transfer_directory(
        source=session_root,
        destination=destination_path,
        num_threads=resolved_workers,
        verify_integrity=False,
        remove_source=remove_source,
        progress=display_progress,
    )

    if display_progress:
        console.echo(
            message=f"Session '{session_data.session_name}' transferred to '{destination_path}'.",
            level=LogLevel.SUCCESS,
        )
