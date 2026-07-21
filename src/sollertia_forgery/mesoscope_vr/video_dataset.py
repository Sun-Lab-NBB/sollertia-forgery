"""Provides the Mesoscope-VR video sub-dataset assembler donated to the system-agnostic forging pipeline.

The assembler reads the fixed camera set's per-frame timestamp, motion-energy, and pupil-tracking feathers and aligns
their values onto the mesoscope fluorescence reference clock, mirroring how the behavior and runtime sub-datasets are
built. The camera set is hardcoded for the Mesoscope-VR system, matching the fixed microcontroller module set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from ataraxis_base_utilities import console
from ataraxis_data_structures import interpolate_data

from .metadata import VideoDataFiles
from .video_tracking import PUPIL_CAMERA_NAME, PupilColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_BODY_CAMERA_NAME: str = "body_camera"
"""The colloquial name of the Mesoscope-VR body camera, used to prefix its motion-energy dataset columns."""

_FRAME_TIME_COLUMN: str = "frame_time_us"
"""The single column of each camera timestamp feather, holding one acquisition timestamp per recorded frame in
microseconds since the UTC epoch. It is the source clock each camera's values are interpolated from."""

_MOTION_ENERGY_COLUMN: str = "motion_energy"
"""The motion-energy column read from each camera's energy feather."""

_FRAME_LUMINANCE_COLUMN: str = "frame_luminance"
"""The frame-luminance column read from each camera's energy feather."""

_PUPIL_FLAG_COLUMNS: frozenset[str] = frozenset({PupilColumn.BLINKING_STATE, PupilColumn.DILATION_STATE})
"""The pupil feather columns holding boolean state flags. They are interpolated by nearest-prior sample and stored as
unsigned 8-bit integers, since a boolean state cannot be linearly interpolated."""

_CAMERA_SOURCES: tuple[tuple[str, str, str, str | None], ...] = (
    (
        PUPIL_CAMERA_NAME,
        VideoDataFiles.FACE_CAMERA_TIMESTAMPS,
        VideoDataFiles.FACE_CAMERA_ENERGY,
        VideoDataFiles.FACE_CAMERA_PUPIL,
    ),
    (
        _BODY_CAMERA_NAME,
        VideoDataFiles.BODY_CAMERA_TIMESTAMPS,
        VideoDataFiles.BODY_CAMERA_ENERGY,
        None,
    ),
)
"""The fixed Mesoscope-VR camera set, each entry pairing a camera's colloquial name with its timestamp, motion-energy,
and optional pupil feather filenames. The face camera carries the eye, so only it contributes a pupil feather."""


def assemble_video_dataset(video_data_path: Path, reference_time: NDArray[np.uint64]) -> pl.DataFrame:
    """Assembles the target session's video dataset and aligns it to the reference time vector.

    Reads the fixed camera set's per-frame video feathers from the processed video-data directory and interpolates
    each present feather's values onto the reference time vector. Motion-energy, frame-luminance, and pupil-geometry
    values are interpolated linearly, while the two pupil state flags are interpolated by nearest-prior sample. A
    camera whose timestamp feather is absent is skipped, and the dataset is empty when no camera feather is present, so
    a session processed without video still forges.

    Args:
        video_data_path: The path to the processed video-data directory holding the per-camera timestamp,
            motion-energy, and pupil-tracking feathers.
        reference_time: The reference time vector, in microseconds since the UTC epoch, to which to align the assembled
            video values. It is the mesoscope fluorescence frame clock.

    Returns:
        A Polars DataFrame aligned to the reference time vector with the per-camera motion-energy and frame-luminance
        columns plus the face-camera pupil columns. The DataFrame has no columns when no camera feather is present.

    Raises:
        ValueError: If a camera's motion-energy or pupil feather has a different row count than its timestamp feather.
    """
    aligned_data: dict[str, NDArray[Any]] = {}
    if not video_data_path.is_dir():
        return pl.DataFrame()

    for camera_name, timestamps_file, energy_file, pupil_file in _CAMERA_SOURCES:
        timestamps_path = video_data_path.joinpath(timestamps_file)
        if not timestamps_path.is_file():
            continue

        # The timestamp feather is the camera's source clock. Every other feather for this camera holds one row per
        # recorded frame in the same acquisition order, so its values align to this clock by row position.
        frame_time = pl.read_ipc(source=timestamps_path, memory_map=True)[_FRAME_TIME_COLUMN].to_numpy()

        energy_path = video_data_path.joinpath(energy_file)
        if energy_path.is_file():
            energy_frame = _read_frame_aligned(
                feather_path=energy_path, expected_rows=frame_time.size, axis_name=timestamps_path.name
            )
            for source_column in (_MOTION_ENERGY_COLUMN, _FRAME_LUMINANCE_COLUMN):
                aligned_data[f"{camera_name}_{source_column}"] = _interpolate_linear(
                    frame_time=frame_time,
                    values=energy_frame[source_column].to_numpy(),
                    reference_time=reference_time,
                )

        if pupil_file is None:
            continue
        pupil_path = video_data_path.joinpath(pupil_file)
        if pupil_path.is_file():
            pupil_frame = _read_frame_aligned(
                feather_path=pupil_path, expected_rows=frame_time.size, axis_name=timestamps_path.name
            )
            for column in pupil_frame.columns:
                if column in _PUPIL_FLAG_COLUMNS:
                    # A boolean state has no meaningful linear blend, so the flag takes the nearest prior frame's value.
                    aligned_data[column] = interpolate_data(
                        source_coordinates=frame_time,
                        source_values=pupil_frame[column].to_numpy().astype(np.uint8),
                        target_coordinates=reference_time,
                        is_discrete=True,
                    )
                else:
                    aligned_data[column] = _interpolate_linear(
                        frame_time=frame_time,
                        values=pupil_frame[column].to_numpy(),
                        reference_time=reference_time,
                    )

    if not aligned_data:
        return pl.DataFrame()
    return pl.DataFrame(aligned_data)


def _interpolate_linear(
    frame_time: NDArray[np.uint64], values: NDArray[np.float32], reference_time: NDArray[np.uint64]
) -> NDArray[np.float32]:
    """Linearly interpolates a continuous per-frame camera value onto the reference time vector.

    Args:
        frame_time: The camera's per-frame acquisition timestamps, the source clock the values are sampled at.
        values: The per-frame values to interpolate.
        reference_time: The reference time vector to interpolate the values onto.

    Returns:
        The interpolated values as single-precision floats, matching the source feather precision. A not-a-number
        source value propagates to the samples that bracket it, so an unmeasured frame stays unmeasured.
    """
    return interpolate_data(
        source_coordinates=frame_time,
        source_values=values,
        target_coordinates=reference_time,
        is_discrete=False,
    ).astype(np.float32)


def _read_frame_aligned(feather_path: Path, expected_rows: int, axis_name: str) -> pl.DataFrame:
    """Reads a per-camera value feather and verifies it carries exactly one row per recorded frame.

    Args:
        feather_path: The path to the motion-energy or pupil feather to read.
        expected_rows: The camera timestamp feather's row count, which every per-camera feather must match.
        axis_name: The timestamp feather filename, used only for the error message.

    Returns:
        The loaded feather.

    Raises:
        ValueError: If the feather's row count differs from the timestamp feather's row count, which means the recording
            and its timestamps disagree on the acquired frame count.
    """
    frame = pl.read_ipc(source=feather_path, memory_map=True)
    if frame.height != expected_rows:
        message = (
            f"Unable to assemble the video dataset. The feather '{feather_path.name}' has {frame.height} rows, but the "
            f"camera timestamp feather '{axis_name}' has {expected_rows}. Every per-camera video feather must hold "
            f"exactly one row per recorded frame."
        )
        console.error(message=message, error=ValueError)
    return frame
