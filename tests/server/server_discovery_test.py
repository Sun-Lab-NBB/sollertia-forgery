"""Tests the discovery of a project's sessions stored under the remote compute server's data root."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sollertia_forgery.server import discover_project_data, discover_project_markers, discover_project_sessions

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
    server_configuration: ServerConfiguration,  # Requested so the credentials resolve from disk.
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


def test_discover_project_sessions_skips_a_session_a_dataset_directory_holds(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A marked directory is a dataset rather than an animal, so a session it holds is not the project's session."""
    project_path = stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}")
    marker = project_path.joinpath("hybrid", "S1", "raw_data", "session_data.yaml")
    marker.parent.mkdir(parents=True)
    marker.write_text("session: marker\n")
    project_path.joinpath("hybrid", "dataset.yaml").write_text("dataset: marker\n")

    assert discover_project_sessions(project=_PROJECT, server=connected_server) == ()


def test_discover_project_sessions_ignores_a_marker_the_hierarchy_does_not_place_at_a_session(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """The search matches on file name alone, so a same-named file at another depth or parent is not a session."""
    project_path = stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}")
    for relative in (
        ("305", _FIRST_SESSION, "processed_data", "session_data.yaml"),
        ("raw_data", "session_data.yaml"),
        ("901", "S1", "raw_data", "dataset.yaml"),
    ):
        decoy = project_path.joinpath(*relative)
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_text("decoy\n")

    assert discover_project_sessions(project=_PROJECT, server=connected_server) == ()


def test_discover_project_sessions_orders_the_sessions_naturally(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """The server reports its matches in directory order, so the discovered sessions are ordered on this host."""
    project_path = stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}")
    for animal in ("100", "10", "9"):
        marker = project_path.joinpath(animal, _FIRST_SESSION, "raw_data", "session_data.yaml")
        marker.parent.mkdir(parents=True)
        marker.write_text("session: marker\n")

    discovered = discover_project_sessions(project=_PROJECT, server=connected_server)

    assert [session.animal for session in discovered] == ["9", "10", "100"]


def test_discover_project_sessions_rejects_a_project_the_server_does_not_hold(connected_server: Server) -> None:
    """A project the server holds no directory for is an absent project rather than a project holding no sessions."""
    with pytest.raises(FileNotFoundError, match=r"holds no directory at that path"):
        discover_project_sessions(project="Absent", server=connected_server)


def test_discover_project_sessions_rejects_a_search_that_covered_part_of_the_tree(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A search the server could not complete would answer for the part of the tree it read, which is not an answer."""
    _build_remote_project(transport=stub_ssh_transport)
    stub_ssh_transport.respond(
        prefix="find -L ", stdout="", stderr="find: '/data/sollertia/TestProject/305': Permission denied", return_code=1
    )

    with pytest.raises(RuntimeError, match=r"reached only part of the tree"):
        discover_project_sessions(project=_PROJECT, server=connected_server)


def test_discover_project_markers_reports_the_datasets_beside_the_sessions(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Both marker kinds answer the same question about the same tree, so one search reports both."""
    project_path = _build_remote_project(transport=stub_ssh_transport)

    markers = discover_project_markers(project_path=connected_server.root.joinpath(_PROJECT), server=connected_server)

    assert [path.name for path in markers.datasets] == ["dataset_alpha"]
    assert [(session.animal, session.session) for session in markers.sessions] == [
        ("305", _FIRST_SESSION),
        ("321", _SECOND_SESSION),
    ]
    assert project_path.joinpath("dataset_alpha").is_dir()


def test_discover_project_markers_orders_the_datasets_naturally(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a dataset name carrying fewer digits is listed ahead of a longer one, as a reader reads them."""
    project_path = stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}")
    for name in ("dataset_100", "dataset_10", "dataset_9"):
        marker = project_path.joinpath(name, "dataset.yaml")
        marker.parent.mkdir(parents=True)
        marker.write_text("dataset: marker\n")

    markers = discover_project_markers(project_path=connected_server.root.joinpath(_PROJECT), server=connected_server)

    assert [path.name for path in markers.datasets] == ["dataset_9", "dataset_10", "dataset_100"]


def test_discover_project_markers_reads_both_marker_names_in_one_search(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Both marker kinds answer one search, so it carries them as alternatives rather than as conditions to satisfy."""
    _build_remote_project(transport=stub_ssh_transport)

    discover_project_markers(project_path=connected_server.root.joinpath(_PROJECT), server=connected_server)

    assert stub_ssh_transport.commands == [
        (
            f"find -L {connected_server.root.joinpath(_PROJECT)} -mindepth 2 -maxdepth 4 "
            f"'(' -name dataset.yaml -o -name session_data.yaml ')' '!' -type l -print0"
        )
    ]


def test_discover_project_markers_reads_only_the_dataset_depth_when_sessions_are_excluded(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A caller needing the datasets alone leaves the session and output directories every animal holds unread."""
    _build_remote_project(transport=stub_ssh_transport)

    markers = discover_project_markers(
        project_path=connected_server.root.joinpath(_PROJECT), server=connected_server, include_sessions=False
    )

    assert [path.name for path in markers.datasets] == ["dataset_alpha"]
    assert markers.sessions == ()
    assert "-mindepth 2 -maxdepth 2" in stub_ssh_transport.commands[0]


def test_discover_project_sessions_orders_a_name_ahead_of_the_sibling_that_extends_it(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """The order is taken over the names each session is reported by rather than over the paths they were found at."""
    project_path = stub_ssh_transport.local_path(f"/data/sollertia/{_PROJECT}")
    for animal, session in (("305", "S2"), ("305", "S10"), ("305-repeat", "S1"), ("305", "S1")):
        marker = project_path.joinpath(animal, session, "raw_data", "session_data.yaml")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("session: marker\n")

    discovered = discover_project_sessions(project=_PROJECT, server=connected_server)

    assert [(session.animal, session.session) for session in discovered] == [
        ("305", "S1"),
        ("305", "S2"),
        ("305", "S10"),
        ("305-repeat", "S1"),
    ]
