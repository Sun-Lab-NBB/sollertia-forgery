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
"""Maps camera source IDs to their output feather filenames, matching the naming convention used by
sollertia-forgery processing."""


def find_camera_feathers(data_directory: Path) -> list[Path]:
    """Discovers camera timestamp feather files under the data directory.

    Recursively searches the data_directory for feather files matching the ``camera_*_timestamps.feather`` naming
    convention used by ataraxis-video-system.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively, so feather
            files may be nested at any depth below this path.

    Returns:
        A sorted list of paths to the discovered camera timestamp feather files. Returns an empty list if the
        directory does not exist or if no matching files are found.
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

    expected_part_count = 3
    if len(parts) < expected_part_count or parts[0] != "camera" or parts[-1] != "timestamps":
        message = (
            f"Unable to extract camera source ID from '{feather_path.name}'. The filename does not follow the "
            f"expected 'camera_{{source_id}}_timestamps.feather' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1])


def process_camera_timestamps(feather_path: Path, output_directory: Path) -> None:
    """Copies a pre-extracted camera timestamp feather file to the behavior output directory under the legacy name.

    Notes:
        Camera timestamp feather files produced by ataraxis-video-system already contain the final ``frame_time_us``
        column in the correct format. This function copies the file to the behavior processing output directory using
        the legacy naming convention (e.g., ``face_camera_timestamps.feather``) without modifying its contents. The
        camera source ID is recovered from the input filename, which already encodes it per the
        ``camera_{source_id}_timestamps.feather`` convention.

    Args:
        feather_path: The path to the input camera timestamp feather file produced by ataraxis-video-system.
        output_directory: The path to the output directory where the processed feather file will be written.

    Raises:
        ValueError: If the source ID encoded in the feather filename does not have a registered output name.
    """
    source_id = extract_camera_source_id(feather_path=feather_path)

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
