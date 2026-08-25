"""Provides the shared host resolution every tool that takes a ``host`` parameter routes through."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import contextmanager

from ..server import remote_state_directory
from ..orchestration import (
    LOCAL_HOST_LABEL,
    REMOTE_HOST_LABEL,
    LocalHost,
    RemoteHost,
    connect_to_server,
    sync_project_state,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..orchestration import ExecutionHost

HOST_LABELS: frozenset[str] = frozenset({LOCAL_HOST_LABEL, REMOTE_HOST_LABEL})
"""The hosts a tool may be pointed at, which is this machine or the configured compute server."""


def unsupported_host_message(host: str) -> str:
    """Builds the error message returned when a caller names a host the tools do not support.

    Args:
        host: The unrecognized host name to report back.
    """
    return f"Unsupported host '{host}'. Available: {', '.join(sorted(HOST_LABELS))}."


@contextmanager
def resolve_execution_host(host: str) -> Iterator[ExecutionHost]:
    """Opens the named execution host, closing a server connection when the caller is done with it.

    Args:
        host: Either ``local`` for this machine or ``remote`` for the configured compute server.

    Yields:
        The execution host the operation runs against.
    """
    if host == REMOTE_HOST_LABEL:
        with connect_to_server() as server:
            yield RemoteHost(server=server)
        return
    yield LocalHost()


def resolve_readable_project(project_path: str, host: str) -> Path:
    """Resolves the directory a read tool opens a project's artifacts from.

    Notes:
        A remote project is mirrored onto this machine and read from the mirror, which is what lets one reader serve
        both hosts. The mirror keeps the project directory's name, so every artifact keeps the filename its writer
        derived from the project. Mirroring leaves the server's artifacts as they stand, so a read never regenerates
        them.

    Args:
        project_path: The path to the project's root directory. A local read opens this path as given. A remote read
            takes the project name from the path's final component and resolves the project under the server's
            configured data root, so the directories above that component are ignored.
        host: Either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        The local directory holding the project's artifacts.

    Raises:
        FileNotFoundError: If the server holds no directory for the named project.
        RuntimeError: If the server-side search for the project's datasets reached only part of its tree.
    """
    if host != REMOTE_HOST_LABEL:
        return Path(project_path)

    project = Path(project_path).name
    local_directory = remote_state_directory(project=project)
    with connect_to_server() as server:
        sync_project_state(server=server, project=project, local_directory=local_directory, regenerate=False)
    return local_directory
