"""Tests the discovery of a project's sessions stored under the remote compute server's data root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sollertia_forgery.server import discover_project_data, discover_project_sessions

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import StubSSHTransport

    from sollertia_forgery.server import Server, ServerConfiguration

_PROJECT: str = "TestProject"
"""The name of the project whose remote hierarchy every test in this module builds."""

_FIRST_SESSION: str = "2024_01_01-12_00_00-000000"
"""The name of the acquired session the first animal holds."""

_SECOND_SESSION: str = "2024_02_02-09_30_00-000000"
"""The name of the acquired session the second animal holds."""


def _build_remote_project(transport: StubSSHTransport) -> Path:
    """Builds one server-side project holding two acquired sessions beside the entries discovery skips.

    The hierarchy carries an animal with one acquired session and one directory that never finished acquisition, a
    second animal with one acquired session, a forged dataset carrying its own marker, and a project-level artifact
    file.

    Args:
        transport: The transport whose temporary server-side filesystem the hierarchy is created inside.

    Returns:
        The path to the created project directory, as it is laid out on the stubbed server.
    """
    project_path = transport.local_path(f"/data/sollertia/{_PROJECT}")

    for animal, session in (("305", _FIRST_SESSION), ("321", _SECOND_SESSION)):
        marker = project_path.joinpath(animal, session, "raw_data", "session_data.yaml")
        marker.parent.mkdir(parents=True)
        marker.write_text("session: marker\n")

    # An acquisition that never wrote its marker is not an acquired session.
    project_path.joinpath("305", "aborted_acquisition").mkdir()

    # A forged dataset carries its own marker and holds a session hierarchy that is not an animal's.
    dataset_path = project_path.joinpath("dataset_alpha")
    dataset_path.joinpath("305", _FIRST_SESSION, "raw_data").mkdir(parents=True)
    dataset_path.joinpath("305", _FIRST_SESSION, "raw_data", "session_data.yaml").write_text("session: marker\n")
    dataset_path.joinpath("dataset.yaml").write_text("dataset: marker\n")

    # A project-level artifact is not a directory and is skipped before it is descended into.
    project_path.joinpath("project_manifest.feather").write_text("manifest")

    return project_path


def test_discover_project_sessions_returns_every_acquired_session(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that discovery returns one entry per acquired session, skipping datasets and non-directory entries."""
    _build_remote_project(transport=stub_ssh_transport)

    discovered = discover_project_sessions(project=_PROJECT, server=connected_server)

    assert [(session.animal, session.session) for session in discovered] == [
        ("305", _FIRST_SESSION),
        ("321", _SECOND_SESSION),
    ]


def test_discover_project_sessions_returns_nothing_for_an_empty_project(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a project directory holding no animal directories yields no sessions."""
    stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}").mkdir(parents=True)

    assert discover_project_sessions(project=_PROJECT, server=connected_server) == ()


def test_discover_project_data_connects_reports_and_closes(
    stub_ssh_transport: StubSSHTransport,
    server_configuration: ServerConfiguration,  # noqa: ARG001 - requested so the credentials resolve from disk.
) -> None:
    """Verifies that the entry point opens its own connection, discovers the sessions, and closes the connection."""
    _build_remote_project(transport=stub_ssh_transport)

    discovered = discover_project_data(project=_PROJECT)

    assert [(session.animal, session.session) for session in discovered] == [
        ("305", _FIRST_SESSION),
        ("321", _SECOND_SESSION),
    ]
    assert stub_ssh_transport.connections == [("test.server.com", "tester")]
    assert stub_ssh_transport.closed is True
