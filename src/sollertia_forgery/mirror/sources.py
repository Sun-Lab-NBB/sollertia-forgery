"""Defines the data-source abstraction that lets the mirror synchronize identically from a local data root or a
remote compute server.

The ``MirrorSource`` protocol is the single seam that isolates the local-versus-remote distinction. Every consumer of
the mirror operates on the synchronized local copy and never branches on where the data originated, so orchestration
and monitoring behave the same regardless of whether the project lives on this machine or on the compute server.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Protocol

from sollertia_shared_assets import RAW_DATA_DIRECTORY, RawDataFiles, DatasetSession

from ..server import Server, discover_project_sessions

if TYPE_CHECKING:
    from pathlib import Path

_DATASET_MARKER_FILENAME: str = "dataset_data.yaml"
"""The filename whose presence marks an animal-level directory as a forged dataset rather than an acquisition animal.
Matches the remote discovery convention so local and remote session enumeration exclude the same directories."""


class MirrorSource(Protocol):
    """Declares the operations the mirror requires from a project data source.

    A source resolves session identifiers under a hierarchy root and copies individual state files out of that root
    into the local mirror. Both a local data root and a remote compute server implement this protocol, which is what
    makes synchronization behave identically for local and remote projects.
    """

    @property
    def root(self) -> Path:
        """Returns the hierarchy root under which the source stores all project directories."""
        ...

    def discover_sessions(self, project: str) -> tuple[DatasetSession, ...]:
        """Enumerates the acquisition sessions stored under the target project on the source.

        Args:
            project: The name of the project whose sessions to enumerate.

        Returns:
            A tuple of the discovered sessions, each identifying an animal and a session directory name.
        """
        ...

    def fetch(self, source_path: Path, destination_path: Path) -> bool:
        """Copies a single file from the source hierarchy into the local mirror.

        Missing source files are tolerated because unprocessed sessions legitimately lack processing tracker files.

        Args:
            source_path: The absolute path to the file under the source root.
            destination_path: The absolute path under the mirror root where the file is written.

        Returns:
            True if the file existed on the source and was copied, False if the source file was absent.
        """
        ...


class LocalDataSource:
    """Synchronizes the mirror from a project hierarchy stored on the local machine's filesystem.

    Notes:
        The data root is supplied explicitly by the caller rather than resolved from the platform data-root setting,
        matching the rest of sollertia-forgery, which always operates on caller-supplied paths.
    """

    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root

    def __repr__(self) -> str:
        """Returns a string representation of the LocalDataSource instance."""
        return f"LocalDataSource(data_root={self._data_root})"

    @property
    def root(self) -> Path:
        """Returns the local data root under which all project directories are stored."""
        return self._data_root

    def discover_sessions(self, project: str) -> tuple[DatasetSession, ...]:
        """Enumerates the acquisition sessions stored under the target project on the local data root.

        Skips dataset directories and only reports directories that contain a session_data.yaml marker, applying the
        same predicate the remote discovery uses so both sources return identical session sets.

        Args:
            project: The name of the project whose sessions to enumerate.

        Returns:
            A tuple of the discovered sessions, each identifying an animal and a session directory name.
        """
        project_path = self._data_root.joinpath(project)
        if not project_path.is_dir():
            return ()

        return tuple(
            DatasetSession(session=session_path.name, animal=animal_path.name)
            for animal_path in sorted(project_path.iterdir())
            if animal_path.is_dir() and not animal_path.joinpath(_DATASET_MARKER_FILENAME).exists()
            for session_path in sorted(animal_path.iterdir())
            if session_path.joinpath(RAW_DATA_DIRECTORY, RawDataFiles.SESSION_DATA).is_file()
        )

    def fetch(self, source_path: Path, destination_path: Path) -> bool:
        """Copies a single file from the local data root into the mirror, preserving file metadata.

        Args:
            source_path: The absolute path to the file under the local data root.
            destination_path: The absolute path under the mirror root where the file is written.

        Returns:
            True if the source file existed and was copied, False if it was absent.
        """
        if not source_path.is_file():
            return False
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=source_path, dst=destination_path)
        return True


class RemoteDataSource:
    """Synchronizes the mirror from a project hierarchy stored on the remote compute server.

    Notes:
        The instance borrows an already-connected ``Server`` and does not own its lifecycle. The caller is responsible
        for opening the server connection and closing it once synchronization completes.
    """

    def __init__(self, server: Server) -> None:
        self._server = server

    def __repr__(self) -> str:
        """Returns a string representation of the RemoteDataSource instance."""
        return f"RemoteDataSource(host={self._server.host}, root={self._server.root})"

    @property
    def root(self) -> Path:
        """Returns the remote data root under which all project directories are stored."""
        return self._server.root

    def discover_sessions(self, project: str) -> tuple[DatasetSession, ...]:
        """Enumerates the acquisition sessions stored under the target project on the remote data root.

        Args:
            project: The name of the project whose sessions to enumerate.

        Returns:
            A tuple of the discovered sessions, each identifying an animal and a session directory name.
        """
        return discover_project_sessions(project=project, server=self._server)

    def fetch(self, source_path: Path, destination_path: Path) -> bool:
        """Downloads a single file from the remote data root into the mirror over SFTP.

        Args:
            source_path: The absolute path to the file on the remote server.
            destination_path: The absolute path under the mirror root where the file is written.

        Returns:
            True if the remote file existed and was downloaded, False if it was absent.
        """
        if not self._server.exists(remote_path=source_path):
            return False
        self._server.pull(local_path=destination_path, remote_path=source_path)
        return True
