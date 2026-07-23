"""Provides the Mesoscope-VR training-session data-assembly worker donated to the system-agnostic forging pipeline.
The worker combines a run or lick training session's behavior and video sub-datasets on the slowest camera's clock into
the session's unified ``data.feather``, since a training session carries no mesoscope fluorescence clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import SessionData, ProcessingTrackers

from .video_dataset import assemble_video_dataset, resolve_slowest_camera_clock
from .behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path


def assemble_training_dataset(source_session_path: Path, output_path: Path) -> None:
    """Assembles a single Mesoscope-VR training session's unified data feather.

    Resolves the reference clock from the slowest camera, then combines the session's behavior and video sub-datasets
    onto that clock into a single Polars DataFrame, written as an uncompressed ``data.feather`` at ``output_path``. The
    behavior sub-dataset supplies the ``time_us`` and ``elapsed_minutes`` columns, and the video sub-dataset contributes
    columns only when the session carries processed camera feathers. The meaning of each emitted column is documented by
    ``DatasetColumn`` and donated to the dataset's ``data_descriptions.feather`` via ``MESOSCOPE_COLUMN_DESCRIPTIONS``.

    Notes:
        Training sessions carry no mesoscope imaging, so the assembler requires neither the experiment configuration nor
        any cindra output, and it takes no dataset name because a training session has no cindra multi-recording output
        to resolve.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy.

    Raises:
        FileNotFoundError: If the session's processed microcontroller-data or runtime-data directory is missing, or if
            no camera clock is available to serve as the reference clock.
        ValueError: If a sub-dataset cannot be assembled (for example, a required hardware-state field is missing).
    """
    session = SessionData.load(session_path=source_session_path)

    microcontroller_data_path = session.processed_data.microcontroller_data_path
    runtime_data_path = session.processed_data.runtime_data_path
    video_data_path = session.processed_data.video_data_path
    raw_data_path = session.raw_data_path

    # Validates that the processed microcontroller and runtime outputs exist before any expensive work. A training
    # session has no cindra output, so only the behavior sources are required. The parsed behavior feathers are split
    # across the per-worker ``microcontroller_data`` (module parsing) and ``runtime_data`` (runtime decode) directories.
    if not microcontroller_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the processed "
            f"microcontroller data directory '{microcontroller_data_path}' to exist and contain "
            f"'{ProcessingTrackers.MICROCONTROLLER}'."
        )
        console.error(message=message, error=FileNotFoundError)
    if not runtime_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the processed runtime "
            f"data directory '{runtime_data_path}' to exist and contain '{ProcessingTrackers.RUNTIME}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # Resolves the reference clock from the slowest camera, since a training session has no fluorescence clock. Every
    # other data source is interpolated onto this clock. Resolving before creating the output directory avoids leaving
    # an empty directory behind when no camera clock is available.
    reference_time = resolve_slowest_camera_clock(video_data_path=video_data_path)

    ensure_directory_exists(path=output_path)

    # Assembles the behavior sub-dataset with its own time columns, since it supplies the unified feather's time axis.
    # The video sub-dataset aligns to the same reference clock and is empty when the session carries no camera feathers.
    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=reference_time,
        drop_time_columns=False,
    )
    video_data = assemble_video_dataset(video_data_path=video_data_path, reference_time=reference_time)

    # Concatenates the behavior and video sub-datasets into the unified feather and writes it uncompressed so downstream
    # consumers can memory-map it. The video sub-dataset joins only when it produced columns.
    sub_datasets = [behavior_data]
    if video_data.width > 0:
        sub_datasets.append(video_data)
    result = pl.concat(items=sub_datasets, how="horizontal")
    result.write_ipc(file=output_path)
