"""Provides functions for discovering the project sessions stored on the remote compute server's data root."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    RAW_DATA_DIRECTORY,
    DATASET_MARKER_FILENAME,
    RawDataFiles,
    DatasetSession,
)

from .server import Server
from ..shared_assets import delay_terminal
from .server_configuration import get_server_configuration

if TYPE_CHECKING:
    from pathlib import Path

_DATASET_MARKER_DEPTH: int = 2
"""The number of path components that separate a dataset's marker file from the project root."""

_SESSION_MARKER_DEPTH: int = 4
"""The number of path components that separate an acquired session's marker file from the project root."""


@dataclass(frozen=True, slots=True)
class ProjectMarkers:
    """Stores the dataset directories and acquired sessions one remote project holds, as resolved from its marker
    files.
    """

    datasets: tuple[Path, ...]
    """The absolute paths to the project's forged dataset directories, in natural sort order."""
    sessions: tuple[DatasetSession, ...]
    """The project's acquired sessions, in natural sort order, excluding the sessions a dataset directory holds."""


def discover_project_markers(project_path: Path, server: Server, *, include_sessions: bool = True) -> ProjectMarkers:
    """Discovers the dataset and session marker files a remote project holds.

    Notes:
        Both marker kinds are read in one server-side search, since both answer the same question about the same tree.
        A directory carrying a dataset marker is a forged dataset rather than an animal, so the sessions it holds are
        never reported as an animal's sessions.

        The search reads every directory above the depth it covers, so a caller that needs the datasets alone narrows
        it to the depth their markers sit at. That keeps the answer from depending on the session and output
        directories every animal holds, which a project shared between accounts need not leave readable.

    Args:
        project_path: The absolute path to the project's root directory on the remote compute server.
        server: The Server instance used to communicate with the remote compute server.
        include_sessions: Determines whether the search covers the project's acquired sessions alongside its datasets.

    Returns:
        A ProjectMarkers instance holding the project's dataset directories and acquired sessions. The sessions are
        empty when the search did not cover them.

    Raises:
        FileNotFoundError: If the server holds no directory at the project path.
        RuntimeError: If the server-side search reached only part of the project's tree.
    """
    records = server.find_paths(
        remote_path=project_path,
        names=(DATASET_MARKER_FILENAME, RawDataFiles.SESSION_DATA),
        minimum_depth=_DATASET_MARKER_DEPTH,
        maximum_depth=_SESSION_MARKER_DEPTH if include_sessions else _DATASET_MARKER_DEPTH,
    )

    datasets: list[Path] = []
    sessions: list[DatasetSession] = []
    for record in records:
        parts = record.relative_to(project_path).parts

        # The search matches on the file name alone, so the depth and the parent directory of each match are what
        # separate a dataset marker from a session marker and both from a same-named file at another position.
        if len(parts) == _DATASET_MARKER_DEPTH and parts[1] == DATASET_MARKER_FILENAME:
            datasets.append(project_path.joinpath(parts[0]))
        elif (
            len(parts) == _SESSION_MARKER_DEPTH
            and parts[2] == RAW_DATA_DIRECTORY
            and parts[3] == RawDataFiles.SESSION_DATA
        ):
            sessions.append(DatasetSession(session=parts[1], animal=parts[0]))

    # The search orders the marker paths, where the separator that follows a directory's name orders a name against a
    # sibling that extends it. Both collections are therefore ordered again on the names they are reported by.
    dataset_names = {dataset.name for dataset in datasets}
    return ProjectMarkers(
        datasets=tuple(natsorted(datasets)),
        sessions=tuple(
            natsorted(
                (session for session in sessions if session.animal not in dataset_names),
                key=lambda entry: (entry.animal, entry.session),
            )
        ),
    )


def discover_project_sessions(project: str, server: Server) -> tuple[DatasetSession, ...]:
    """Discovers all sessions stored under the project's directory on the remote compute server's data root.

    Notes:
        Dataset directories carry a dataset.yaml marker and are skipped, so a dataset's session hierarchy is never
        returned as an animal's sessions.

    Args:
        project: The name of the project for which to discover sessions.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        A tuple of DatasetSession instances representing all discovered sessions, in natural sort order.

    Raises:
        FileNotFoundError: If the server holds no directory for the named project.
        RuntimeError: If the server-side search reached only part of the project's tree.
    """
    markers = discover_project_markers(project_path=server.root.joinpath(project), server=server)
    return markers.sessions


def discover_project_data(project: str) -> tuple[DatasetSession, ...]:
    """Discovers and reports all sessions stored under the project's directory on the remote compute server.

    Serves as the entry point for discovering project data, connecting to the server and reporting the discovered
    sessions to the terminal.

    Args:
        project: The name of the project whose data to discover.

    Returns:
        A tuple of DatasetSession instances representing all discovered sessions.
    """
    console.echo(message=f"Discovering '{project}' project's sessions on the remote server...", level=LogLevel.INFO)

    # Establishes communication with the compute server.
    configuration = get_server_configuration()
    server = Server(configuration=configuration)

    try:
        discovered_sessions = discover_project_sessions(project=project, server=server)
    finally:
        server.close()

    delay_terminal()
    console.echo(
        message=f"Discovered {len(discovered_sessions)} session(s) for the '{project}' project:", level=LogLevel.INFO
    )
    for session_metadata in discovered_sessions:
        console.echo(message=f"Session '{session_metadata.session}' performed by animal '{session_metadata.animal}'.")

    return discovered_sessions
