"""Provides assets for discovering and processing pre-extracted camera timestamp feather files produced by the
ataraxis-video-system library.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ataraxis_base_utilities import console, ensure_directory_exists

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
    """Discovers camera timestamp feather files inside the canonical camera timestamps directory.

    Searches ``data_directory`` non-recursively for feather files matching the ``camera_*_timestamps.feather``
    naming convention used by ataraxis-video-system. The directory is expected to be the session's canonical
    ``processed_data/camera_timestamps`` location exposed by ``SessionData.camera_timestamps_path``.

    Args:
        data_directory: The path to the session's camera timestamps directory.

    Returns:
        A sorted list of paths to the discovered camera timestamp feather files. Returns an empty list if the
        directory does not exist or if no matching files are found.
    """
    if not data_directory.is_dir():
        return []
    return sorted(data_directory.glob(_CAMERA_FEATHER_PATTERN))


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
    """Hardlinks a pre-extracted camera timestamp feather file into the behavior output directory under the legacy name.

    Notes:
        Camera timestamp feather files produced by ataraxis-video-system already contain the final ``frame_time_us``
        column in the correct format. This function creates a hardlink to the file inside the behavior processing
        output directory using the legacy naming convention (e.g., ``face_camera_timestamps.feather``) without
        modifying its contents or duplicating bytes on disk. The original ``camera_{source_id}_timestamps.feather``
        remains in place so that ataraxis-video-system discovery tools continue to locate it. The camera source ID
        is recovered from the input filename, which already encodes it per the
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

    ensure_directory_exists(path=output_directory)
    output_path = output_directory / output_filename
    output_path.unlink(missing_ok=True)
    os.link(src=feather_path, dst=output_path)
