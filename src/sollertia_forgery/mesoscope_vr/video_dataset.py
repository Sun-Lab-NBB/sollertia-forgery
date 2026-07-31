"""Provides the Mesoscope-VR video sub-dataset assembler and camera-clock resolver donated to the system-agnostic
forging pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

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
"""The colloquial name of the Mesoscope-VR body camera, used to prefix its motion-energy and frame-luminance dataset
columns."""

_MICROSECONDS_PER_SECOND: float = 1_000_000.0
"""The number of microseconds in one second, used to convert a camera's timestamp span into a mean frame rate."""

_MINIMUM_CLOCK_FRAMES: int = 2
"""The fewest frames a camera timestamp feather must hold to define a reference clock, since a mean frame rate needs at
least two timestamps spanning a positive duration."""

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

type _AlignedArray = NDArray[np.float32] | NDArray[np.uint8]
"""The per-camera array types the assembled columns take: linearly interpolated geometry and the nearest-prior state
flags."""


@dataclass(frozen=True, slots=True)
class _CameraSource:
    """Pairs a Mesoscope-VR camera's colloquial name with the per-camera feathers the assembler reads for it."""

    name: str
    """The camera's colloquial name, used to prefix its motion-energy and frame-luminance dataset columns."""
    timestamps_file: str
    """The camera's timestamp feather filename, holding its source clock."""
    energy_file: str
    """The camera's motion-energy feather filename."""
    pupil_file: str | None
    """The camera's pupil-tracking feather filename, present only for the camera that carries the eye."""


_CAMERA_SOURCES: tuple[_CameraSource, ...] = (
    _CameraSource(
        name=PUPIL_CAMERA_NAME,
        timestamps_file=VideoDataFiles.FACE_CAMERA_TIMESTAMPS,
        energy_file=VideoDataFiles.FACE_CAMERA_ENERGY,
        pupil_file=VideoDataFiles.FACE_CAMERA_PUPIL,
    ),
    _CameraSource(
        name=_BODY_CAMERA_NAME,
        timestamps_file=VideoDataFiles.BODY_CAMERA_TIMESTAMPS,
        energy_file=VideoDataFiles.BODY_CAMERA_ENERGY,
        pupil_file=None,
    ),
)
"""The fixed Mesoscope-VR camera set, each entry naming a camera and the feathers read for it. The face camera carries
the eye, so only it contributes a pupil feather."""


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
            video values. It is the mesoscope fluorescence clock for experiment sessions and the slowest camera's clock
            for training sessions.

    Returns:
        A Polars DataFrame aligned to the reference time vector with the per-camera motion-energy and frame-luminance
        columns plus the face-camera pupil columns. The DataFrame has no columns when no camera feather is present.

    Raises:
        ValueError: If a camera's motion-energy or pupil feather has a different row count than its timestamp feather.
    """
    if not video_data_path.is_dir():
        return pl.DataFrame()

    aligned_data: dict[str, _AlignedArray] = {}
    for camera in _CAMERA_SOURCES:
        timestamps_path = video_data_path.joinpath(camera.timestamps_file)
        if not timestamps_path.is_file():
            continue

        # The timestamp feather is the camera's source clock. Every other feather for this camera holds one row per
        # recorded frame in the same acquisition order, so its values align to this clock by row position.
        frame_time = pl.read_ipc(source=timestamps_path, memory_map=True)[_FRAME_TIME_COLUMN].to_numpy()

        energy_path = video_data_path.joinpath(camera.energy_file)
        if energy_path.is_file():
            energy_frame = _read_frame_aligned(
                feather_path=energy_path, expected_rows=frame_time.size, timestamps_filename=timestamps_path.name
            )
            for source_column in (_MOTION_ENERGY_COLUMN, _FRAME_LUMINANCE_COLUMN):
                aligned_data[f"{camera.name}_{source_column}"] = _interpolate_linear(
                    frame_time=frame_time,
                    values=energy_frame[source_column].to_numpy(),
                    reference_time=reference_time,
                )

        if camera.pupil_file is None:
            continue
        pupil_path = video_data_path.joinpath(camera.pupil_file)
        if pupil_path.is_file():
            pupil_frame = _read_frame_aligned(
                feather_path=pupil_path, expected_rows=frame_time.size, timestamps_filename=timestamps_path.name
            )
            for column in pupil_frame.columns:
                if column in _PUPIL_FLAG_COLUMNS:
                    # A boolean state has no meaningful linear blend, so the flag takes the nearest prior frame's value.
                    aligned_data[column] = interpolate_data(
                        source_coordinates=frame_time,
                        source_values=pupil_frame[column].to_numpy().astype(np.uint8),
                        target_coordinates=reference_time,
                        is_discrete=True,
                    ).astype(np.uint8)
                else:
                    aligned_data[column] = _interpolate_linear(
                        frame_time=frame_time,
                        values=pupil_frame[column].to_numpy(),
                        reference_time=reference_time,
                    )

    if not aligned_data:
        return pl.DataFrame()
    return pl.DataFrame(aligned_data)


def resolve_slowest_camera_clock(video_data_path: Path) -> NDArray[np.uint64]:
    """Resolves the slowest camera's acquisition clock, used as the assembly reference clock for training sessions.

    Reads each present camera's timestamp feather from the processed video-data directory, computes its mean frame rate
    as the recorded frame count divided by the timestamp span, and returns the timestamps of the camera with the lowest
    mean rate verbatim. The slowest camera is chosen because every other data source can be interpolated onto its
    coarser grid without inventing samples between its frames. Training sessions carry no fluorescence clock, so this
    camera clock stands in as the reference the behavior and video sub-datasets align to.

    Args:
        video_data_path: The path to the processed video-data directory holding the per-camera timestamp feathers.

    Returns:
        The slowest camera's per-frame acquisition timestamps, in microseconds since the UTC epoch.

    Raises:
        FileNotFoundError: If no camera timestamp feather with at least two frames spanning a positive duration is
            present, so no camera clock can serve as the reference.
    """
    slowest_clock: NDArray[np.uint64] | None = None
    slowest_rate = float("inf")
    slowest_camera = ""

    if video_data_path.is_dir():
        for camera in _CAMERA_SOURCES:
            timestamps_path = video_data_path.joinpath(camera.timestamps_file)
            if not timestamps_path.is_file():
                continue

            frame_time = pl.read_ipc(source=timestamps_path, memory_map=True)[_FRAME_TIME_COLUMN].to_numpy()

            # A mean frame rate needs at least two frames spanning a positive duration. Casts the endpoints to float
            # first, since the timestamps are unsigned and their difference would wrap on an out-of-order feather.
            if frame_time.size < _MINIMUM_CLOCK_FRAMES:
                continue
            duration_seconds = (float(frame_time[-1]) - float(frame_time[0])) / _MICROSECONDS_PER_SECOND
            if duration_seconds <= 0:
                continue

            mean_rate = frame_time.size / duration_seconds
            if mean_rate < slowest_rate:
                slowest_rate = mean_rate
                slowest_clock = frame_time
                slowest_camera = camera.name

    if slowest_clock is None:
        message = (
            f"Unable to resolve the reference clock for the training session. No camera timestamp feather with at "
            f"least two frames spanning a positive duration was found under '{video_data_path}', so no camera clock "
            f"can serve as the assembly reference clock."
        )
        console.error(message=message, error=FileNotFoundError)

    console.echo(message=f"Resolved the '{slowest_camera}' clock ({slowest_rate:.2f} fps) as the reference clock.")
    return slowest_clock


def _interpolate_linear(
    frame_time: NDArray[np.uint64], values: NDArray[np.float32], reference_time: NDArray[np.uint64]
) -> NDArray[np.float32]:
    """Linearly interpolates a continuous per-frame camera value onto the reference time vector.

    Args:
        frame_time: The camera's per-frame acquisition timestamps, the source clock the values are sampled at.
        values: The per-frame values to interpolate.
        reference_time: The reference time vector to interpolate the values onto.

    Returns:
        The interpolated values, carried at the source feather precision. A not-a-number source value propagates to
        the samples that bracket it, so an unmeasured frame stays unmeasured.
    """
    return interpolate_data(
        source_coordinates=frame_time,
        source_values=values,
        target_coordinates=reference_time,
        is_discrete=False,
    ).astype(np.float32)


def _read_frame_aligned(feather_path: Path, expected_rows: int, timestamps_filename: str) -> pl.DataFrame:
    """Reads a per-camera value feather and verifies it carries exactly one row per recorded frame.

    Args:
        feather_path: The path to the motion-energy or pupil feather to read.
        expected_rows: The camera timestamp feather's row count, which every per-camera feather must match.
        timestamps_filename: The timestamp feather filename, used only for the error message.

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
            f"camera timestamp feather '{timestamps_filename}' has {expected_rows}. Every per-camera video feather "
            f"must hold exactly one row per recorded frame."
        )
        console.error(message=message, error=ValueError)
    return frame
