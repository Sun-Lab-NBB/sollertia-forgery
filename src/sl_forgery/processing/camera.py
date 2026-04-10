"""Provides assets for discovering and processing pre-extracted camera timestamp feather files produced by the
ataraxis-video-system library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path

_CAMERA_FEATHER_PATTERN: str = "camera_*_timestamps.feather"
"""The glob pattern used to discover camera timestamp feather files produced by ataraxis-video-system."""


def find_camera_feather(data_directory: Path, source_id: int) -> Path:
    """Searches for a single camera timestamp feather file matching the target source ID under the data directory.

    Recursively searches the data_directory and all subdirectories for a feather file matching the
    ``camera_{source_id}_timestamps.feather`` naming convention used by ataraxis-video-system. Expects exactly one
    match per source ID within the directory tree.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively, so feather
            files may be nested at any depth below this path.
        source_id: The numeric source ID of the camera to match. Corresponds to the camera's log source identifier
            used during data acquisition.

    Returns:
        The path to the discovered camera timestamp feather file.

    Raises:
        FileNotFoundError: If the data_directory does not exist, is not a directory, or no feather file matching the
            source ID is found.
        ValueError: If multiple feather files matching the source ID are found under the data directory.
    """
    if not data_directory.exists() or not data_directory.is_dir():
        message = (
            f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. The path does "
            f"not exist or is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    pattern = f"camera_{source_id}_timestamps.feather"
    matches = sorted(data_directory.rglob(pattern))

    if not matches:
        message = (
            f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. No file "
            f"matching '{pattern}' was found."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. Multiple "
            f"files matching '{pattern}' were found: {[str(match) for match in matches]}. Expected exactly one match."
        )
        console.error(message=message, error=ValueError)

    return matches[0]


def find_camera_feathers(data_directory: Path) -> list[Path]:
    """Discovers all camera timestamp feather files under the data directory.

    Recursively searches the data_directory for feather files matching the ``camera_*_timestamps.feather`` naming
    convention used by ataraxis-video-system.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively.

    Returns:
        A sorted list of paths to all discovered camera timestamp feather files. Returns an empty list if no files
        are found.
    """
    if not data_directory.exists() or not data_directory.is_dir():
        return []

    return sorted(data_directory.rglob(_CAMERA_FEATHER_PATTERN))


def extract_camera_source_id(feather_path: Path) -> int:
    """Extracts the numeric camera source ID from a camera timestamp feather filename.

    Args:
        feather_path: The path to the camera timestamp feather file. The filename must follow the
            ``camera_{source_id}_timestamps.feather`` naming convention.

    Returns:
        The numeric source ID extracted from the filename.

    Raises:
        ValueError: If the filename does not follow the expected naming convention.
    """
    stem = feather_path.stem  # e.g., "camera_51_timestamps"
    parts = stem.split("_")

    _expected_part_count = 3
    if len(parts) < _expected_part_count or parts[0] != "camera" or parts[-1] != "timestamps":
        message = (
            f"Unable to extract camera source ID from '{feather_path.name}'. The filename does not follow the "
            f"expected 'camera_{{source_id}}_timestamps.feather' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1])


def process_camera_timestamps(feather_path: Path, output_directory: Path) -> None:
    """Reads a pre-extracted camera timestamp feather file and writes it to the behavior output directory.

    Notes:
        Camera timestamp feather files produced by ataraxis-video-system already contain the final ``frame_time_us``
        column in the correct format. This function relocates the file to the behavior processing output directory
        with uncompressed IPC format to support memory-mapping during downstream analysis.

    Args:
        feather_path: The path to the input camera timestamp feather file produced by ataraxis-video-system.
        output_directory: The path to the output directory where the processed feather file will be written.
    """
    console.echo(message=f"Processing camera timestamps from '{feather_path.name}'...")

    # Reads the pre-extracted camera timestamp data.
    dataframe = pl.read_ipc(source=feather_path)

    # Ensures the output directory exists.
    output_directory.mkdir(parents=True, exist_ok=True)

    # Writes the data to the output directory using uncompressed feather format for memory-mapping support.
    dataframe.write_ipc(file=output_directory / feather_path.name, compression="uncompressed")

    console.echo(message=f"Camera timestamp processing for '{feather_path.name}': Complete.", level=LogLevel.SUCCESS)
