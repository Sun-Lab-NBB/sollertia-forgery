"""Contains tests for the ServerConfiguration dataclass and associated helpers relocated from
sollertia-shared-assets.
"""

from pathlib import Path

import pytest
import platformdirs
from sollertia_shared_assets import set_working_directory

from sollertia_forgery.server.server_configuration import (
    ServerConfiguration,
    create_server_configuration_file,
    get_server_configuration,
)


@pytest.fixture
def clean_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configures platformdirs to resolve to a pristine tmp-backed working directory."""
    data_dir = tmp_path / "user_data"
    data_dir.mkdir()
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda appname, appauthor: str(data_dir))  # noqa: ARG005
    working_dir = tmp_path / "working"
    set_working_directory(path=working_dir)
    return working_dir


# Tests for ServerConfiguration dataclass


def test_server_configuration_default_initialization() -> None:
    """Verifies default initialization of ServerConfiguration."""
    config = ServerConfiguration()

    assert config.username == ""
    assert config.password == ""
    assert config.host == ""


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
    )
    yaml_path = tmp_path / "server_configuration.yaml"
    original.to_yaml(file_path=yaml_path)

    loaded = ServerConfiguration.from_yaml(file_path=yaml_path)

    assert loaded.username == original.username
    assert loaded.password == original.password
    assert loaded.host == original.host


# Tests for create_server_configuration_file


def test_create_server_configuration_file(clean_working_directory: Path) -> None:
    """Verifies that create_server_configuration_file creates a user-visible YAML under the working directory."""
    create_server_configuration_file(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
    )

    config_file = clean_working_directory / "configuration" / "server_configuration.yaml"
    assert config_file.exists()

    loaded = ServerConfiguration.from_yaml(file_path=config_file)
    assert loaded.username == "test_user"
    assert loaded.password == "test_pass"  # noqa: S105 - literal test credential.
    assert loaded.host == "test.server.com"


# Tests for get_server_configuration


def test_get_server_configuration_user(clean_working_directory: Path) -> None:
    """Verifies that get_server_configuration loads the YAML created by create_server_configuration_file."""
    create_server_configuration_file(
        username="test_user",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
    )

    config = get_server_configuration()

    assert config.username == "test_user"
    assert config.host == "test.server.com"


def test_get_server_configuration_raises_error_if_missing(clean_working_directory: Path) -> None:  # noqa: ARG001
    """Verifies that get_server_configuration raises FileNotFoundError when no configuration is present."""
    with pytest.raises(FileNotFoundError):
        get_server_configuration()


def test_get_server_configuration_raises_error_if_unconfigured(clean_working_directory: Path) -> None:
    """Verifies that get_server_configuration raises ValueError for a YAML with placeholder credentials."""
    config_file = clean_working_directory / "configuration" / "server_configuration.yaml"
    ServerConfiguration().to_yaml(file_path=config_file)

    with pytest.raises(ValueError, match=r"(?i)unconfigured"):
        get_server_configuration()
