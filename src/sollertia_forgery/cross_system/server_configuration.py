"""Provides the server configuration dataclass and helpers used to access the Sollertia platform remote compute
server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import get_working_directory
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path

_SERVER_CONFIG_FILENAME: str = "server_configuration.yaml"
"""Canonical filename for the ServerConfiguration YAML stored under the working directory's configuration
subdirectory."""

_CONFIGURATION_DIR: str = "configuration"
"""Subdirectory under the working directory that stores the server configuration YAML alongside other Sollertia
configuration assets."""


@dataclass
class ServerConfiguration(YamlConfig):
    """Defines the access credentials and data root for the Sollertia platform remote compute server."""

    username: str = ""
    """The username to use for server authentication."""
    password: str = ""
    """The password to use for server authentication."""
    host: str = ""
    """The hostname or IP address of the server to connect to."""
    root: str = ""
    """The absolute path, on the remote compute server, to the single root directory that stores all Sollertia data
    (raw and processed). All server-side data operations resolve their paths relative to this root."""


def create_server_configuration_file(
    username: str,
    password: str,
    host: str,
    root: str,
) -> None:
    """Creates the .YAML configuration file for the Sollertia platform compute server and configures the local machine
    (PC) to use this file for all future server-related calls.

    Args:
        username: The username to use for server authentication.
        password: The password to use for server authentication.
        host: The hostname or IP address of the server to connect to.
        root: The absolute path, on the remote compute server, to the root directory that stores all Sollertia data.
    """
    output_directory = get_working_directory().joinpath(_CONFIGURATION_DIR)
    ServerConfiguration(
        username=username,
        password=password,
        host=host,
        root=root,
    ).to_yaml(file_path=output_directory.joinpath(_SERVER_CONFIG_FILENAME))
    console.echo(message="Server configuration file: Created.", level=LogLevel.SUCCESS)


def get_server_configuration() -> ServerConfiguration:
    """Resolves and returns the Sollertia platform compute server's configuration data as a ServerConfiguration
    instance.

    Returns:
        The loaded and validated server configuration data, stored in a ServerConfiguration instance.

    Raises:
        FileNotFoundError: If the 'server_configuration.yaml' file does not exist in the local Sollertia platform
            working directory.
        ValueError: If the loaded server configuration is unconfigured or contains placeholder access credentials.
    """
    configuration_directory = get_working_directory().joinpath(_CONFIGURATION_DIR)

    config_path = configuration_directory.joinpath(_SERVER_CONFIG_FILENAME)

    if not config_path.exists():
        message = (
            f"Unable to locate the 'server_configuration.yaml' file in the Sollertia platform working directory "
            f"{config_path}. Call the 'sl-server configure' CLI command to create the server configuration file."
        )
        console.error(message=message, error=FileNotFoundError)

    configuration = ServerConfiguration.from_yaml(file_path=config_path)

    if not all((configuration.username, configuration.password, configuration.host, configuration.root)):
        message = (
            "Unable to load the server configuration. The 'server_configuration.yaml' file appears to be unconfigured "
            "or contains placeholder values for one or more required fields (username, password, host, root). Call the "
            "'sl-server configure' CLI command to reconfigure the server access credentials."
        )
        console.error(message=message, error=ValueError)

    message = f"Server configuration: Resolved. Using the {configuration.username} account."
    console.echo(message=message, level=LogLevel.SUCCESS)
    return configuration


def get_server_configuration_path() -> Path:
    """Returns the path under which the ``server_configuration.yaml`` file is stored.

    Used by tools that write to the configuration file directly and need to verify its destination path without
    loading the configuration contents.
    """
    return get_working_directory().joinpath(_CONFIGURATION_DIR, _SERVER_CONFIG_FILENAME)
