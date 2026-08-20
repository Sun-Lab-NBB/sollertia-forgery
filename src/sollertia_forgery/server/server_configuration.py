"""Provides the server configuration dataclass and helpers used to access the Sollertia platform remote compute
server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import CONFIGURATION_DIRECTORY, get_working_directory
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path

_SERVER_CONFIGURATION_FILENAME: str = "server_configuration.yaml"
"""Canonical filename for the ServerConfiguration YAML stored under the working directory's configuration
subdirectory."""

_REMOTE_STATE_DIRECTORY: str = "remote_state"
"""Subdirectory under the working directory that holds this host's remote-run state, which is the artifacts mirrored
from the compute server alongside this host's own record of what it submitted."""


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
    environment: str = ""
    """The name of the shared conda environment, on the remote compute server, in which sollertia-forgery and all of
    its processing dependencies are installed. Every remote compute job activates this single environment before
    invoking the ``slf`` CLI, so all processing and forging pipelines share one environment."""


def create_server_configuration_file(
    username: str,
    password: str,
    host: str,
    root: str,
    environment: str,
) -> None:
    """Creates the ``server_configuration.yaml`` file for the Sollertia platform compute server at the canonical path
    every server-related call resolves.

    Args:
        username: The username to use for server authentication.
        password: The password to use for server authentication.
        host: The hostname or IP address of the server to connect to.
        root: The absolute path, on the remote compute server, to the root directory that stores all Sollertia data.
        environment: The name of the shared conda environment, on the remote compute server, in which
            sollertia-forgery and all of its processing dependencies are installed.
    """
    output_directory = get_working_directory().joinpath(CONFIGURATION_DIRECTORY)
    ServerConfiguration(
        username=username,
        password=password,
        host=host,
        root=root,
        environment=environment,
    ).to_yaml(file_path=output_directory.joinpath(_SERVER_CONFIGURATION_FILENAME))
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
    configuration_directory = get_working_directory().joinpath(CONFIGURATION_DIRECTORY)

    configuration_path = configuration_directory.joinpath(_SERVER_CONFIGURATION_FILENAME)

    if not configuration_path.exists():
        message = (
            f"Unable to locate the 'server_configuration.yaml' file in the Sollertia platform working directory "
            f"{configuration_path}. Call the 'slf server configure' CLI command to create the server configuration "
            f"file."
        )
        console.error(message=message, error=FileNotFoundError)

    configuration = ServerConfiguration.from_yaml(file_path=configuration_path)

    if not all(
        (
            configuration.username,
            configuration.password,
            configuration.host,
            configuration.root,
            configuration.environment,
        )
    ):
        message = (
            "Unable to load the server configuration. The 'server_configuration.yaml' file appears to be unconfigured "
            "or contains placeholder values for one or more required fields (username, password, host, root, "
            "environment). Call the 'slf server configure' CLI command to reconfigure the server access credentials."
        )
        console.error(message=message, error=ValueError)

    message = f"Server configuration: Resolved. Using the {configuration.username} account."
    console.echo(message=message, level=LogLevel.SUCCESS)
    return configuration


def get_server_configuration_path() -> Path:
    """Returns the path under which the ``server_configuration.yaml`` file is stored.

    Returns:
        The path to the configuration file under the Sollertia platform working directory, resolved without reading
        the file.
    """
    return get_working_directory().joinpath(CONFIGURATION_DIRECTORY, _SERVER_CONFIGURATION_FILENAME)


def remote_state_path() -> Path:
    """Returns the local directory holding everything this host records about remote runs.

    Notes:
        One directory holds both halves of what a remote run leaves behind, namely the state artifacts pulled from
        the server and this host's own record of what it submitted. Keeping them together means a run's whole local
        footprint is one directory to find, inspect, or remove.

    Returns:
        The path to the remote state directory under the Sollertia platform working directory.
    """
    return get_working_directory().joinpath(_REMOTE_STATE_DIRECTORY)


def remote_state_directory(project: str) -> Path:
    """Returns the local directory mirroring one remote project's state artifacts.

    Notes:
        The mirror reproduces the project directory by name, so an artifact pulled into it keeps the filename its
        writer derived from the project. Every read tool resolves an artifact from the project directory it is given,
        so a mirrored project is read exactly as a local one is.

    Args:
        project: The name of the project whose remote state is mirrored.

    Returns:
        The path to the project's mirror directory under the Sollertia platform working directory.
    """
    return remote_state_path().joinpath(project)
