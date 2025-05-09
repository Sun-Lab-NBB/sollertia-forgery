"""Provides utilities to select a .yml settings file interactively and parse
settings or data info from YAML configurations.
"""

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from ipyfilechooser import FileChooser
from IPython.display import display


def select_settings_file(start_path: Optional[Path] = None) -> FileChooser:
    """Open a GUI dialog for selecting a .yml settings file.

    This function is designed for use in interactive environments (e.g., IPython notebooks).
    It should not be used in non-interactive scripts or environments.

    Args:
        start_path (Path | None): Initial directory to open in the file chooser. If None,
            defaults to the current working directory.

    Returns:
        FileChooser: The file chooser widget object pointing to the selected file.
    """
    # Use starting path, otherwise defaults to the current working directory
    file_chooser = FileChooser(str(start_path)) if start_path else FileChooser()

    # Enhances the search to specifically look for the .yml files.
    file_chooser.use_dir_icons = True
    file_chooser.filter_pattern = "*.yml"

    display(file_chooser)
    return file_chooser


def parse_settings(file: Union[Path, str]) -> dict[str, Any]:
    """Parse settings from a YAML configuration file.

    Args:
        file (Path | str): Path to the YAML settings file.

    Returns:
        dict[str, Any]: Dictionary containing the parsed configuration settings.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        NameError: If any required configuration parameters are missing.
    """
    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"Settings file not found at path '{file_path}'.")

    required_keys = {"server", "cell_detection", "registration", "clustering", "demix"}

    with open(file_path, "r") as data_file:
        settings = yaml.load(data_file, Loader=yaml.FullLoader)

    missing_keys = required_keys - settings.keys()
    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise NameError(f"Missing required keys in settings file: {missing_list}")

    return settings


def parse_data_info(file: Union[Path, str]) -> dict[str, Any]:
    """Parse data identification information from a YAML file.

    Args:
        file (Path | str): Path to the data information YAML file.

    Returns:
        dict[str, Any]: Dictionary containing metadata about the data and animal IDs.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        NameError: If required data fields are missing in the file.
    """
    file_path = Path(file)
    if not file_path.is_file():
        msg = f"Information file not found at path '{file_path}'."
        raise FileNotFoundError(msg)

    required_keys = {"data", "animal"}

    # Using FullLoader to support custom Python objects and YAML constructors
    with open(file_path) as data_file:
        information = yaml.load(data_file, Loader=yaml.FullLoader)

        # Checks for missing required keys
        missing_keys = required_keys - information.keys()
        if missing_keys:
            msg = f"Missing required keys in information file: {', '.join(missing_keys)}"
            raise NameError(msg)

        return information
