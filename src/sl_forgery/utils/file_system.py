"""This module provides various tools and assets for working with the filesystem of the local machine (PC) and the
remote compute server. The assets from this module support running all other pipelines and runtimes exposed by this
library."""

from pathlib import Path
from dataclasses import dataclass

import appdirs
from sl_shared_assets import Server, ServerCredentials
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists


def set_working_directory(path: Path) -> None:
    """Sets the directory specified by the input path as the default working directory for the local machine (PC).

    This function is used to initially configure any machine to work with Sun lab data stored on remote compute
    server(s). Specifically, the working directory acts as a stable access point for running all supported tasks on the
    compute server, including data processing and dataset formation. The path to the working directory is stored inside
    the user's data directory so that all pipelines exposed by this library automatically access that information during
    every runtime. Since the storage directory is typically hidden and varies between OSes and machines, this function
    provides a convenient way for setting that path without manually editing the storage cache.

    Notes:
        If the input path does not point to an existing directory, the function will automatically generate the
        requested directory.

        If the directory does not have the 'user_credentials.yaml' or 'service_credentials.yaml' files, the precursors
        for these files will be created as part of runtime.

    Args:
        path: The path to the new working directory to use for working with Sun lab data from the local machine (PC).
    """

    # If the directory specified by the 'path' does not exist, generates the specified directory tree. As part of this
    # process, also generate the precursor server_credentials.yaml file to use for accessing the remote server used to
    # store project data.
    if not path.exists():
        message = (
            f"The specified working directory ({path}) does not exist. Generating the directory at the "
            f"specified path..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Resolves the path to the static .txt file used to store the path to the system configuration file
    app_dir = Path(appdirs.user_data_dir(appname="sun_lab_data", appauthor="sun_lab"))
    path_file = app_dir.joinpath("working_directory_path.txt")

    # In case this function is called before the app directory is created, ensures the app directory exists
    ensure_directory_exists(path_file)

    # Ensures that the input path's directory exists
    ensure_directory_exists(path)

    # Replaces the contents of the working_directory_path.txt file with the provided path
    with open(path_file, "w") as f:
        f.write(str(path))

    if not path.joinpath("user_credentials.yaml").exists():
        message = (
            f"Unable to locate the 'user_credentials.yaml' file in the Sun lab working directory {path}. Creating a "
            f"precursor credentials file in the directory. Edit the file to store your BioHPC access credentials. This "
            f"is a prerequisite for generating datasets and analysing the processed data stored on the remote server."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        ServerCredentials().to_yaml(file_path=path.joinpath("user_credentials.yaml"))

    if not path.joinpath("service_credentials.yaml").exists():
        message = (
            f"Unable to locate the 'service_credentials.yaml' file in the Sun lab working directory {path}. Creating a "
            f"precursor credentials file in the directory. Edit the file to store the Sun lab service BioHPC access "
            f"credentials. Editing this file is only required if you intend to run remote data processing pipelines. "
            f"Most lab users should not need to use this file."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        ServerCredentials().to_yaml(file_path=path.joinpath("service_credentials.yaml"))


def get_working_directory() -> Path:
    """Resolves the path to the local Sun lab working directory.

    This service function is used by most runtimes from this library to ensure that all local processing is confined to
    the same directory. This directory also stores important configuration metadata necessary to access the remote
    compute server that stores all Sun lab data.

    Returns:
        The path to the local working directory.

    Raises:
        FileNotFoundError: If the local machine does not have the Sun lab data directory, or the local working directory
            does not exist (has not been configured).
    """
    # Uses appdirs to locate the user data directory and resolve the path to the configuration file
    app_dir = Path(appdirs.user_data_dir(appname="sun_lab_data", appauthor="sun_lab"))
    path_file = app_dir.joinpath("working_directory_path.txt")

    # If the cache file or the Sun lab data directory does not exist, aborts with an error
    if not path_file.exists():
        message = (
            "Unable to resolve the path to the local working directory, as local machine does not have the "
            "Sun lab data directory. This indicates that the local directory has not been designated. Designate the "
            "local working directory by calling the 'sl-set-working-directory' CLI command and rerun the command that "
            "produced this error."
        )
        console.error(message=message, error=FileNotFoundError)

    # Once the location of the path storage file is resolved, reads the file path from the file
    with open(path_file, "r") as f:
        working_directory = Path(f.read().strip())

    # If the configuration file does not exist, also aborts with an error
    if not working_directory.exists():
        message = (
            "Unable to resolve the path to the local working directory, as the directory pointed by the path stored "
            "in Sun lab data directory does not exist. Designate a new working directory by calling the "
            "'sl-set-working-directory' CLI command and rerun the command that produced this error."
        )
        console.error(message=message, error=FileNotFoundError)

    # Returns the path to the working directory
    return working_directory


def get_credentials_file_path(require_service: bool = False) -> Path:
    """Resolves and returns the path to the requested .yaml file that stores access credentials for the Sun lab
    remote compute server.

    Depending on the configuration, either returns the path to the 'user_credentials.yaml' file (default) or the
    'service_credentials.yaml' file if the 'require_service' flag is set to True.

    Notes:
        Assumes that the local working directory has been configured before calling this function.

    Args:
        require_service: Determines whether this function must evaluate and return the path to the
            'service_credentials.yaml' file (if true) or the 'user_credentials.yaml' file (if false).

    Raises:
        FileNotFoundError: If either the 'service_credentials.yaml' or the 'user_credentials.yaml' files do not exist
            in the local Sun lab working directory.
        ValueError: If both credential files exist, but the requested credentials file is not configured.
    """

    # Gets the path to the local working directory.
    working_directory = get_working_directory()

    # Resolves the paths to the credential files.
    service_path = working_directory.joinpath("service_credentials.yaml")
    user_path = working_directory.joinpath("user_credentials.yaml")

    # Aborts with an error if one of the files does not exist
    if not service_path.exists() or not user_path.exists():
        message = (
            f"Unable to resolve the path to the preferred Sun lab server access credentials file, as at least one of "
            f"the expected files ('service_credentials.yaml' or 'user_credentials.yaml') does not exist in the local "
            f"Sun lab working directory {working_directory}. Rerun the 'sl-set-working-directory' CLI command to "
            f"generate the missing files and rerun the command that produced this error."
        )
        console.error(message=message, error=FileNotFoundError)

    # If the caller requires the service account, evaluates the service credentials file.
    if require_service:
        credentials: ServerCredentials = ServerCredentials.from_yaml(file_path=service_path)  # type: ignore

        # If the service account is not configured, aborts with an error.
        if credentials.username == "YourNetID" or credentials.password == "YourPassword":
            message = (
                f"The 'service_credentials.yaml' file appears to be unconfigured or contains placeholder credentials. "
                f"Manually edit the file to include proper access credentials for the Sun lab remote compute server "
                f"and rerun the command that produced this error."
            )
            console.error(message=message, error=ValueError)
            raise ValueError(message)  # Fallback to appease mypy, should not be reachable

        # If the service account is configured, returns the path to the service credentials file to caller
        else:
            message = f"Server access credentials: Resolved. Using the service {credentials.username} account."
            console.echo(message=message, level=LogLevel.SUCCESS)
            return service_path

    # Otherwise, evaluates the user credentials file.
    credentials: ServerCredentials = ServerCredentials.from_yaml(file_path=user_path)  # type: ignore

    # If the user account is not configured, aborts with an error.
    if credentials.username == "YourNetID" or credentials.password == "YourPassword":
        message = (
            f"The 'user_credentials.yaml' file appears to be unconfigured or contains placeholder credentials. "
            f"Manually edit the file to include proper access credentials for the Sun lab remote compute server and "
            f"rerun the command that produced this error."
        )
        console.error(message=message, error=ValueError)
        raise ValueError(message)  # Fallback to appease mypy, should not be reachable

    # Otherwise, returns the path to the user credentials file to caller
    message = f"Server access credentials: Resolved. Using the {credentials.username} account."
    console.echo(message=message, level=LogLevel.SUCCESS)
    return user_path


@dataclass()
class RemotePaths:
    """Stores the paths to configuration directories for some data processing pipelines stored on the remote server.

    These configuration directories are stored in the shared Sun lab data directory on the remote compute server and
    are used by various processing pipelines to maintain consistent configuration across all sessions, projects, and
    users.

    Notes:
        This class should be instantiated via the get_remote_filesystem_paths() function exposed by the 'utils' package
        of this library.

        All paths in this class are resolved relative to the remote compute server's shared working and storage
        directories.
    """

    suite2p_configurations_path: Path = Path()
    """The path to the shared Sun lab directory that contains single-day and multi-day suite2p configuration files."""
    dlc_projects_path: Path = Path()
    """The path to the shared Sun lab directory that contains DeepLabCut project directories."""


def get_remote_filesystem_paths(server: Server) -> RemotePaths:
    """Resolves and returns a RemotePaths instance that provides the paths to certain server-side directories used by
    processing pipelines.

    Primarily, this function is used to resolve the paths to shared server-side configuration directories used by
    pipelines such as DeepLabCut and sl-suite2p single-day and multi-day.

    Args:
        server: The Server class instance that manages the bidirectional communication with the remote compute server
            that executes processing pipelines.

    Returns:
        The initialized RemotePaths instance that stores the resolved paths data.
    """

    return RemotePaths(
        suite2p_configurations_path=server.raw_data_root.joinpath("suite2p_configurations"),
        dlc_projects_path=server.raw_data_root.joinpath("deeplabcut_projects"),
    )
