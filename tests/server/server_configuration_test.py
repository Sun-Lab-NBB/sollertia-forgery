"""Contains tests for the ServerConfiguration dataclass and the helpers that create and resolve its file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sollertia_forgery.server.server_configuration import (
    ServerConfiguration,
    remote_state_path,
    remote_state_directory,
    get_server_configuration,
    get_server_configuration_path,
    create_server_configuration_file,
)

if TYPE_CHECKING:
    from pathlib import Path

_CONFIGURATION_RELATIVE_PATH: tuple[str, str] = ("configuration", "server_configuration.yaml")
"""The working-directory-relative location the server configuration file is written to."""


# Tests for ServerConfiguration dataclass


def test_server_configuration_default_initialization() -> None:
    """Verifies default initialization of ServerConfiguration."""
    config = ServerConfiguration()

    assert config.username == ""
    assert config.password == ""
    assert config.host == ""
    assert config.root == ""
    assert config.environment == ""


def test_server_configuration_custom_initialization() -> None:
    """Verifies custom initialization of ServerConfiguration."""
    config = ServerConfiguration(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
    )

    assert config.username == "test_user"
    assert config.password == "test_pass"  # noqa: S105 - literal test credential.
    assert config.host == "test.server.com"


def test_server_configuration_yaml_roundtrip(tmp_path: Path) -> None:
    """Verifies that ServerConfiguration survives YAML serialization."""
    original = ServerConfiguration(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
        root="/remote/sollertia/root",
        environment="forge",
    )
    yaml_path = tmp_path / "server_configuration.yaml"
    original.to_yaml(file_path=yaml_path)

    loaded = ServerConfiguration.from_yaml(file_path=yaml_path)

    assert loaded.username == original.username
    assert loaded.password == original.password
    assert loaded.host == original.host
    assert loaded.root == original.root
    assert loaded.environment == original.environment


# Tests for create_server_configuration_file


def test_create_server_configuration_file(isolated_working_directory: Path) -> None:
    """Verifies that create_server_configuration_file creates a user-visible YAML under the working directory."""
    create_server_configuration_file(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
        root="/remote/sollertia/root",
        environment="forge",
    )

    config_file = isolated_working_directory.joinpath(*_CONFIGURATION_RELATIVE_PATH)
    assert config_file.exists()

    loaded = ServerConfiguration.from_yaml(file_path=config_file)
    assert loaded.username == "test_user"
    assert loaded.password == "test_pass"  # noqa: S105 - literal test credential.
    assert loaded.host == "test.server.com"
    assert loaded.root == "/remote/sollertia/root"
    assert loaded.environment == "forge"


# Tests for get_server_configuration


def test_get_server_configuration_returns_the_written_credentials(
    isolated_working_directory: Path,  # Requested for working directory isolation.
) -> None:
    """Verifies that get_server_configuration loads the YAML created by create_server_configuration_file."""
    create_server_configuration_file(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
        root="/remote/sollertia/root",
        environment="forge",
    )

    config = get_server_configuration()

    assert config.username == "test_user"
    assert config.host == "test.server.com"


def test_get_server_configuration_raises_error_if_missing(
    isolated_working_directory: Path,  # Requested for working directory isolation.
) -> None:
    """Verifies that get_server_configuration raises FileNotFoundError when no configuration is present."""
    with pytest.raises(FileNotFoundError, match=r"Unable to locate the 'server_configuration\.yaml' file"):
        get_server_configuration()


@pytest.mark.parametrize("blank_field", ["username", "password", "host", "root", "environment"])
def test_get_server_configuration_rejects_a_configuration_missing_one_field(
    isolated_working_directory: Path, blank_field: str
) -> None:
    """Verifies that a configuration filling in every field but one is refused rather than partially used.

    A blank root builds every server-side path relative to the login account's home directory, and a blank
    environment runs every allocation under whatever the login shell defaults to, so a configuration missing one
    field is as unusable as one missing all of them.
    """
    fields = {
        "username": "test_user",
        "password": "test_pass",
        "host": "test.server.com",
        "root": "/remote/sollertia/root",
        "environment": "forge",
    }
    fields[blank_field] = ""
    ServerConfiguration(**fields).to_yaml(file_path=isolated_working_directory.joinpath(*_CONFIGURATION_RELATIVE_PATH))

    with pytest.raises(ValueError, match=r"(?i)unconfigured"):
        get_server_configuration()


def test_get_server_configuration_raises_error_if_unconfigured(isolated_working_directory: Path) -> None:
    """Verifies that get_server_configuration raises ValueError for a YAML with placeholder credentials."""
    config_file = isolated_working_directory.joinpath(*_CONFIGURATION_RELATIVE_PATH)
    ServerConfiguration().to_yaml(file_path=config_file)

    with pytest.raises(ValueError, match=r"(?i)unconfigured"):
        get_server_configuration()


# Tests for the path helpers


def test_get_server_configuration_path_resolves_the_written_file(isolated_working_directory: Path) -> None:
    """Verifies that the resolved configuration path names the file the creator writes."""
    create_server_configuration_file(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
        root="/remote/sollertia/root",
        environment="forge",
    )

    resolved = get_server_configuration_path()

    assert resolved == isolated_working_directory.joinpath(*_CONFIGURATION_RELATIVE_PATH)
    assert resolved.exists()


def test_get_server_configuration_path_resolves_without_reading_the_file(isolated_working_directory: Path) -> None:
    """Verifies that the configuration path resolves even when no configuration file has been written."""
    resolved = get_server_configuration_path()

    assert resolved == isolated_working_directory.joinpath(*_CONFIGURATION_RELATIVE_PATH)
    assert not resolved.exists()


def test_remote_state_path_resolves_under_the_working_directory(isolated_working_directory: Path) -> None:
    """Verifies that the remote state directory sits directly under the platform working directory."""
    assert remote_state_path() == isolated_working_directory.joinpath("remote_state")


def test_remote_state_directory_mirrors_the_project_by_name(isolated_working_directory: Path) -> None:
    """Verifies that a project's mirror directory reproduces the project name under the remote state directory."""
    mirror = remote_state_directory(project="TestProject")

    assert mirror == isolated_working_directory.joinpath("remote_state", "TestProject")
    assert mirror.parent == remote_state_path()
