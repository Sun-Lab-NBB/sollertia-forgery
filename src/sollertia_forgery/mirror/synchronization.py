"""Provides the project mirror: a persistent, shallow local copy of project state that presents a uniform surface for
orchestrating and monitoring processing jobs regardless of whether the project's data lives on the local machine or on
the remote compute server.

The mirror holds only lightweight coordination state, namely the session markers, the per-pipeline processing
trackers, and the project manifest. It never holds the heavy raw or processed payloads, which remain on their data
root. The slsa session grammar re-derives every asset path from the session marker's location, so copying only the
marker and the trackers into the mirror is enough to resolve and inspect a session's entire asset layout locally.

Synchronization is the one operation that differs between a local and a remote project, and it is confined to the
``MirrorSource`` seam. ``synchronize`` pulls state from either a ``LocalDataSource`` or a ``RemoteDataSource`` into an
identical mirror, so every downstream consumer reads the same local files and never branches on data origin.

Notes:
    Callers interface with the mirror under a small contract. First, treat the mirror as a snapshot and call
    ``synchronize`` to refresh it before reading state, because the mirror does not track the source live. Second,
    read all state from the mirror rather than from the source, so behavior stays independent of where the data
    resides. Third, resolve payload locations on the real data root, never in the mirror, which by design contains no
    payloads. Fourth, build asset paths under the mirror and translate them onto the source root with ``rebase_path``
    before touching or transferring payloads. Fifth, treat trackers owned by external tools, such as the cindra
    two-photon tracker, as read-only. Sixth, never execute jobs or mutate trackers through the mirror, because
    execution happens on the data root and its results reach the mirror only through a subsequent synchronization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    RAW_DATA_DIRECTORY,
    PROCESSED_DATA_DIRECTORY,
    RawData,
    ProjectData,
    SessionData,
    ProcessedData,
    ProcessingTrackers,
)

from ..server import Server, get_server_configuration
from .sources import LocalDataSource, RemoteDataSource
from .location import rebase_path, get_mirror_directory
from ..managing import project_manifest_path
from ..shared_assets import delay_terminal

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import DatasetSession

    from .sources import MirrorSource


@dataclass(frozen=True, slots=True)
class SynchronizationReport:
    """Summarizes the outcome of a single project synchronization run."""

    project: str
    """The name of the synchronized project."""
    source_root: Path
    """The hierarchy root the state was pulled from, either a local data root or a remote server root."""
    mirror_directory: Path
    """The local mirror root the state was pulled into."""
    session_count: int
    """The number of acquisition sessions discovered under the project on the source."""
    pulled_count: int
    """The number of state files that existed on the source and were copied into the mirror."""
    absent_count: int
    """The number of state files that were absent on the source, which is expected for unprocessed sessions."""


def synchronize(source: MirrorSource, project: str) -> SynchronizationReport:
    """Pulls a project's coordination state from a data source into the local mirror.

    Discovers the project's sessions on the source, then copies the project manifest, the manifest tracker, each
    session marker, and each per-pipeline processing tracker into the mirror at the same hierarchy-relative location.
    Absent source files are skipped rather than treated as errors, because unprocessed sessions have no tracker files.

    Args:
        source: The data source to pull state from, either a local data root or a remote compute server.
        project: The name of the project to synchronize.

    Returns:
        A report describing how many sessions were discovered and how many state files were pulled or absent.

    Raises:
        FileNotFoundError: If the Sollertia platform working directory has not been configured for the host machine.
    """
    mirror_directory = get_mirror_directory()
    source_root = source.root

    console.echo(
        message=f"Synchronizing '{project}' project state from {source_root} into the local mirror...",
        level=LogLevel.INFO,
    )

    sessions = source.discover_sessions(project=project)

    # Pulls the project-level state (manifest and manifest tracker) once, before descending into sessions.
    project_pulled, project_absent = _pull_state_files(
        source=source,
        state_files=_project_state_files(project_directory=source_root.joinpath(project)),
        source_root=source_root,
        mirror_directory=mirror_directory,
    )

    pulled_count = project_pulled
    absent_count = project_absent
    for session in console.track(sessions, description="Synchronizing session state", unit="session"):
        session_pulled, session_absent = _pull_state_files(
            source=source,
            state_files=_session_state_files(root=source_root, project=project, session=session),
            source_root=source_root,
            mirror_directory=mirror_directory,
        )
        pulled_count += session_pulled
        absent_count += session_absent

    delay_terminal()
    console.echo(
        message=(
            f"Synchronized {len(sessions)} session(s) for the '{project}' project: {pulled_count} file(s) pulled, "
            f"{absent_count} absent."
        ),
        level=LogLevel.SUCCESS,
    )

    return SynchronizationReport(
        project=project,
        source_root=source_root,
        mirror_directory=mirror_directory,
        session_count=len(sessions),
        pulled_count=pulled_count,
        absent_count=absent_count,
    )


def synchronize_local(data_root: Path, project: str) -> SynchronizationReport:
    """Synchronizes a project's state from a local data root into the mirror.

    This is the turnkey local entry point that constructs the local data source for the caller, so orchestration and
    monitoring tooling can synchronize without assembling a source themselves.

    Args:
        data_root: The local data root under which the project hierarchy is stored.
        project: The name of the project to synchronize.

    Returns:
        A report describing how many sessions were discovered and how many state files were pulled or absent.
    """
    return synchronize(source=LocalDataSource(data_root=data_root), project=project)


def synchronize_remote(project: str) -> SynchronizationReport:
    """Synchronizes a project's state from the configured remote compute server into the mirror.

    Bootstraps a server connection from the stored server configuration, synchronizes, and closes the connection
    afterward, so the caller does not have to manage the server lifecycle. Remote and local synchronization produce
    an identical mirror.

    Args:
        project: The name of the project to synchronize.

    Returns:
        A report describing how many sessions were discovered and how many state files were pulled or absent.

    Raises:
        FileNotFoundError: If the working directory or the server configuration has not been configured for the host.
    """
    configuration = get_server_configuration()
    server = Server(configuration=configuration)
    try:
        return synchronize(source=RemoteDataSource(server=server), project=project)
    finally:
        server.close()


def load_mirrored_session(project: str, animal: str, session: str) -> SessionData:
    """Loads a synchronized session from the mirror with every asset path resolved under the mirror root.

    The returned session must already have been synchronized. Its resolved paths point into the mirror, so callers
    that need the real payloads must translate the relevant paths onto the data root with ``rebase_path``.

    Args:
        project: The name of the project the session belongs to.
        animal: The identifier of the animal that participated in the session.
        session: The name of the session directory to load.

    Returns:
        The loaded session data with raw and processed asset paths resolved under the mirror root.

    Raises:
        FileNotFoundError: If the session marker has not been synchronized into the mirror.
    """
    mirror_directory = get_mirror_directory()
    session_directory = (
        ProjectData(root=mirror_directory, project_name=project)
        .animal(animal_id=animal)
        .session_path(session_name=session)
    )
    return SessionData.load(session_path=session_directory)


def _pull_state_files(
    source: MirrorSource,
    state_files: tuple[Path, ...],
    source_root: Path,
    mirror_directory: Path,
) -> tuple[int, int]:
    """Copies a set of source state files into the mirror and counts the present and absent files.

    Args:
        source: The data source to pull the files from.
        state_files: The absolute paths of the state files under the source root.
        source_root: The source hierarchy root the state files are resolved against.
        mirror_directory: The mirror root the state files are copied into.

    Returns:
        A tuple of the number of files pulled and the number of files that were absent on the source.
    """
    pulled = 0
    absent = 0
    for source_path in state_files:
        destination_path = rebase_path(path=source_path, source_root=source_root, destination_root=mirror_directory)
        if source.fetch(source_path=source_path, destination_path=destination_path):
            pulled += 1
        else:
            absent += 1
    return pulled, absent


def _session_state_files(root: Path, project: str, session: DatasetSession) -> tuple[Path, ...]:
    """Resolves the coordination state files the mirror tracks for a single session under a hierarchy root.

    Args:
        root: The hierarchy root to resolve the session's paths against.
        project: The name of the project the session belongs to.
        session: The session to resolve state file paths for.

    Returns:
        The absolute paths of the session marker and the per-pipeline processing trackers, with the marker first.
    """
    animal = ProjectData(root=root, project_name=project).animal(animal_id=session.animal)
    session_directory = animal.session_path(session_name=session.session)
    raw_data = RawData.build(root=session_directory.joinpath(RAW_DATA_DIRECTORY))
    processed_data = ProcessedData.build(root=session_directory.joinpath(PROCESSED_DATA_DIRECTORY))
    return (
        raw_data.session_data_path,
        raw_data.checksum_tracker_path,
        processed_data.runtime_tracker_path,
        processed_data.video_tracker_path,
        processed_data.microcontroller_tracker_path,
        processed_data.two_photon_tracker_path,
    )


def _project_state_files(project_directory: Path) -> tuple[Path, ...]:
    """Resolves the project-level coordination state files the mirror tracks under a project directory.

    Args:
        project_directory: The project's root directory under a hierarchy root.

    Returns:
        The absolute paths of the project manifest and the manifest processing tracker.
    """
    return (
        project_manifest_path(project_directory=project_directory),
        project_directory.joinpath(ProcessingTrackers.MANIFEST),
    )
