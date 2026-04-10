"""Provides assets for discovering and processing pre-extracted camera timestamp feather files produced by the
ataraxis-video-system library.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

if TYPE_CHECKING:
    from pathlib import Path

_CAMERA_FEATHER_PATTERN: str = "camera_*_timestamps.feather"
"""The glob pattern used to discover camera timestamp feather files produced by ataraxis-video-system."""

_CAMERA_OUTPUT_NAMES: dict[int, str] = {
    51: "face_camera_timestamps.feather",
    62: "body_camera_timestamps.feather",
}
"""Maps camera source IDs to their output feather filenames, matching the naming convention used by sl-behavior."""


def find_camera_feathers(data_directory: Path, source_id: int | None = None) -> list[Path]:
    """Discovers camera timestamp feather files under the data directory.

    Recursively searches the data_directory for feather files matching the ``camera_*_timestamps.feather`` naming
    convention used by ataraxis-video-system. When a source_id is provided, narrows the search to the specific
    ``camera_{source_id}_timestamps.feather`` file and validates that exactly one match exists.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively, so feather
            files may be nested at any depth below this path.console
        source_id: The numeric source ID of the camera to match. When provided, restricts discovery to the specific
            source ID and enforces that exactly one matching file exists. When omitted, discovers all camera timestamp
            feather files.

    Returns:
        A sorted list of paths to the discovered camera timestamp feather files. Returns an empty list when no
        source_id filter is applied and no files are found.

    Raises:
        FileNotFoundError: If a source_id is specified and the data_directory does not exist, is not a directory, or
            no feather file matching the source ID is found.
        ValueError: If a source_id is specified and multiple feather files matching the source ID are found.
    """
    if not data_directory.exists() or not data_directory.is_dir():
        if source_id is not None:
            message = (
                f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. The path "
                f"does not exist or is not a directory."
            )
            console.error(message=message, error=FileNotFoundError)
        return []

    pattern = f"camera_{source_id}_timestamps.feather" if source_id is not None else _CAMERA_FEATHER_PATTERN
    matches = sorted(data_directory.rglob(pattern))

    if source_id is not None:
        if not matches:
            message = (
                f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. No file "
                f"matching '{pattern}' was found."
            )
            console.error(message=message, error=FileNotFoundError)

        if len(matches) > 1:
            message = (
                f"Unable to find camera timestamp feather for source '{source_id}' in '{data_directory}'. Multiple "
                f"files matching '{pattern}' were found: {[str(match) for match in matches]}. Expected exactly one "
                f"match."
            )
            console.error(message=message, error=ValueError)

    return matches


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

    expected_part_count = 3
    if len(parts) < expected_part_count or parts[0] != "camera" or parts[-1] != "timestamps":
        message = (
            f"Unable to extract camera source ID from '{feather_path.name}'. The filename does not follow the "
            f"expected 'camera_{{source_id}}_timestamps.feather' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1])


def process_camera_timestamps(feather_path: Path, output_directory: Path, source_id: int) -> None:
    """Copies a pre-extracted camera timestamp feather file to the behavior output directory under the legacy name.

    Notes:
        Camera timestamp feather files produced by ataraxis-video-system already contain the final ``frame_time_us``
        column in the correct format. This function copies the file to the behavior processing output directory using
        the legacy naming convention (e.g., ``face_camera_timestamps.feather``) without modifying its contents.

    Args:
        feather_path: The path to the input camera timestamp feather file produced by ataraxis-video-system.
        output_directory: The path to the output directory where the processed feather file will be written.
        source_id: The numeric camera source ID used to resolve the output filename.

    Raises:
        ValueError: If the source ID does not have a registered output name.
    """
    if source_id not in _CAMERA_OUTPUT_NAMES:
        message = (
            f"Unable to process camera timestamps for source '{source_id}'. No output filename is registered for "
            f"this source ID. Registered source IDs: {sorted(_CAMERA_OUTPUT_NAMES.keys())}."
        )
        console.error(message=message, error=ValueError)

    output_filename = _CAMERA_OUTPUT_NAMES[source_id]
    console.echo(message=f"Processing camera timestamps from '{feather_path.name}' -> '{output_filename}'...")

    ensure_directory_exists(path=output_directory)
    shutil.copy2(src=feather_path, dst=output_directory / output_filename)

    console.echo(message=f"Camera timestamp processing for '{output_filename}': Complete.", level=LogLevel.SUCCESS)
