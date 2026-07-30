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
        host: The name the caller supplied.

    Returns:
        The error message.
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
        A local project is read where it sits. A remote project is mirrored onto this machine first and read from the
        mirror, which is what lets one reader serve both hosts without knowing which it was given. The mirror reproduces
        the project directory by name, so every artifact keeps the filename its writer derived from the project.

        Mirroring rewrites nothing on the server, so a read reports what the project currently records rather than
        regenerating it. Regeneration is a deliberate act, which ``generate_project_manifest_tool`` and
        ``generate_dataset_state_tool`` perform and which a batch's closure performs on its own.

    Args:
        project_path: The path to the project's root directory, on this machine or on the server.
        host: Either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        The local directory holding the project's artifacts.

    Raises:
        FileNotFoundError: If the server holds no directory for the named project.
    """
    if host != REMOTE_HOST_LABEL:
        return Path(project_path)

    project = Path(project_path).name
    local_directory = remote_state_directory(project=project)
    with connect_to_server() as server:
        sync_project_state(server=server, project=project, local_directory=local_directory, regenerate=False)
    return local_directory
