"""Provides functions for discovering the project sessions stored on the remote compute server's data root."""

from __future__ import annotations

from ataraxis_base_utilities import LogLevel, console

from .server import Server
from ..shared_assets import DatasetSession, delay_terminal
from .server_configuration import get_server_configuration


def discover_project_sessions(project: str, server: Server) -> tuple[DatasetSession, ...]:
    """Discovers all sessions stored under the project's directory on the remote compute server's data root.

    Notes:
        This function explicitly skips dataset directories (those containing dataset_data.yaml) to avoid confusing
        dataset session hierarchies with actual animal/session directories.

    Args:
        project: The name of the project for which to discover sessions.
        server: The Server instance used to communicate with the remote compute server.

    Returns:
        A tuple of DatasetSession instances representing all discovered sessions.
    """
    project_path = server.root.joinpath(project)

    # Builds a list of DatasetSession instances for all discovered sessions.
    discovered_sessions: list[DatasetSession] = []
    for animal_dir in console.track(
        server.list_directory(remote_path=project_path), description="Evaluating animal directories", unit="directory"
    ):
        animal_path = project_path.joinpath(animal_dir)

        # Skips non-directory entries (like manifest files).
        if not server.is_directory(remote_path=animal_path):
            continue

        # Skips dataset directories (those containing dataset_data.yaml) to avoid confusing dataset session
        # hierarchies with actual animal/session directories.
        if server.exists(remote_path=animal_path.joinpath("dataset_data.yaml")):
            continue

        # Finds valid sessions (those containing session_data.yaml files).
        discovered_sessions.extend(
            DatasetSession(session=session_dir, animal=animal_dir)
            for session_dir in server.list_directory(remote_path=animal_path)
            if server.exists(remote_path=animal_path.joinpath(session_dir, "raw_data", "session_data.yaml"))
        )

    return tuple(discovered_sessions)


def discover_project_data(project: str) -> tuple[DatasetSession, ...]:
    """Discovers and reports all sessions stored under the project's directory on the remote compute server.

    This function serves as the entry point for discovering project data available on the remote server. It connects
    to the server, scans the project's directory under the data root, and prints the discovered sessions to the
    terminal.

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
