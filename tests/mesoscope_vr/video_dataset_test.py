"""Tests for the Mesoscope-VR video sub-dataset assembler and its training-session camera-clock resolver."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from tests.mesoscope_vr.video_tracking_test import _build_points, _frame_specification

from sollertia_forgery.mesoscope_vr.metadata import VideoDataFiles
from sollertia_forgery.mesoscope_vr.video_dataset import (
    _BODY_CAMERA_NAME,
    assemble_video_dataset,
    resolve_slowest_camera_clock,
)
from sollertia_forgery.mesoscope_vr.video_tracking import (
    PUPIL_CAMERA_NAME,
    PupilColumn,
    process_mesoscope_video_tracking,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_CLOCK_ORIGIN: int = 1_700_000_000_000_000
"""The microsecond epoch every synthetic camera clock starts at."""


def _clock(*offsets_us: int) -> NDArray[np.uint64]:
    """Builds one camera clock from microsecond offsets measured off the shared clock origin.

    Args:
        *offsets_us: The per-frame offsets, in microseconds after the origin.

    Returns:
        The absolute per-frame acquisition timestamps.
    """
    return np.asarray([_CLOCK_ORIGIN + offset for offset in offsets_us], dtype=np.uint64)


def _write_energy(path: Path, motion_energy: Sequence[float], frame_luminance: Sequence[float]) -> Path:
    """Writes one camera's motion-energy feather in the layout the assembler reads.

    Args:
        path: The output feather path.
        motion_energy: The per-frame motion energy values.
        frame_luminance: The per-frame frame luminance values.

    Returns:
        The written path.
    """
    pl.DataFrame(
        {
            "motion_energy": np.asarray(motion_energy, dtype=np.float32),
            "frame_luminance": np.asarray(frame_luminance, dtype=np.float32),
        }
    ).write_ipc(file=path, compression="uncompressed")
    return path


def _write_pupil(path: Path, diameter: Sequence[float], blinking: Sequence[bool]) -> Path:
    """Writes a two-column stand-in for the pupil feather, carrying one geometry column and one state flag.

    Args:
        path: The output feather path.
        diameter: The per-frame pupil diameter values.
        blinking: The per-frame blink flags.

    Returns:
        The written path.
    """
    pl.DataFrame(
        {
            PupilColumn.PUPIL_DIAMETER_PX.value: np.asarray(diameter, dtype=np.float32),
            PupilColumn.BLINKING_STATE.value: np.asarray(blinking, dtype=np.bool_),
        }
    ).write_ipc(file=path, compression="uncompressed")
    return path


@pytest.fixture
def video_data_path(tmp_path: Path) -> Path:
    """Creates the processed video-data directory the assembler and the clock resolver read.

    Args:
        tmp_path: The temporary directory the video-data directory is created under.

    Returns:
        The created directory path.
    """
    path = tmp_path.joinpath("processed_data", "video_data")
    path.mkdir(parents=True)
    return path


def test_assemble_video_dataset_returns_an_empty_frame_when_the_directory_is_absent(tmp_path: Path) -> None:
    assembled = assemble_video_dataset(video_data_path=tmp_path.joinpath("missing"), reference_time=_clock(0, 1000))

    assert assembled.width == 0
    assert assembled.height == 0


def test_assemble_video_dataset_returns_an_empty_frame_when_no_camera_feather_is_present(
    video_data_path: Path,
) -> None:
    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(0, 1000))

    assert assembled.width == 0


def test_assemble_video_dataset_returns_an_empty_frame_for_a_camera_carrying_only_its_clock(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000))

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(500, 1500))

    assert assembled.width == 0


def test_assemble_video_dataset_interpolates_the_body_camera_onto_the_reference_clock(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000))
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_ENERGY),
        motion_energy=[0.0, 10.0, 20.0],
        frame_luminance=[100.0, 110.0, 120.0],
    )

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(500, 1500))

    # The body camera carries no eye, so it contributes its two energy columns and no pupil column at all.
    assert assembled.columns == [f"{_BODY_CAMERA_NAME}_motion_energy", f"{_BODY_CAMERA_NAME}_frame_luminance"]
    assert assembled.schema[f"{_BODY_CAMERA_NAME}_motion_energy"] == pl.Float32
    assert assembled[f"{_BODY_CAMERA_NAME}_motion_energy"].to_list() == pytest.approx([5.0, 15.0])
    assert assembled[f"{_BODY_CAMERA_NAME}_frame_luminance"].to_list() == pytest.approx([105.0, 115.0])


def test_assemble_video_dataset_interpolates_the_face_camera_energy_and_pupil_columns(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000))
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_ENERGY),
        motion_energy=[1.0, 3.0, 5.0],
        frame_luminance=[10.0, 20.0, 30.0],
    )
    _write_pupil(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_PUPIL),
        diameter=[20.0, 30.0, 40.0],
        blinking=[True, False, True],
    )

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(500, 1500, 2000))

    assert assembled.columns == [
        f"{PUPIL_CAMERA_NAME}_motion_energy",
        f"{PUPIL_CAMERA_NAME}_frame_luminance",
        PupilColumn.PUPIL_DIAMETER_PX.value,
        PupilColumn.BLINKING_STATE.value,
    ]
    assert assembled[f"{PUPIL_CAMERA_NAME}_motion_energy"].to_list() == pytest.approx([2.0, 4.0, 5.0])
    assert assembled[PupilColumn.PUPIL_DIAMETER_PX].to_list() == pytest.approx([25.0, 35.0, 40.0])
    # A boolean state cannot be blended, so each flag takes the value of the last frame acquired at or before the
    # reference sample.
    assert assembled.schema[PupilColumn.BLINKING_STATE] == pl.UInt8
    assert assembled[PupilColumn.BLINKING_STATE].to_list() == [1, 0, 1]


def test_assemble_video_dataset_skips_a_camera_whose_clock_is_absent(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    # Only the body camera recorded, yet the face camera's energy feather is present from an earlier partial run.
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), _clock(0, 2000))
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_ENERGY),
        motion_energy=[0.0, 8.0],
        frame_luminance=[1.0, 9.0],
    )
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_ENERGY),
        motion_energy=[0.0, 1.0],
        frame_luminance=[2.0, 3.0],
    )

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(1000))

    assert assembled.columns == [f"{_BODY_CAMERA_NAME}_motion_energy", f"{_BODY_CAMERA_NAME}_frame_luminance"]


def test_assemble_video_dataset_skips_a_face_camera_carrying_no_pupil_feather(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 2000))
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_ENERGY),
        motion_energy=[0.0, 4.0],
        frame_luminance=[1.0, 5.0],
    )

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(1000))

    assert assembled.columns == [f"{PUPIL_CAMERA_NAME}_motion_energy", f"{PUPIL_CAMERA_NAME}_frame_luminance"]


def test_assemble_video_dataset_rejects_an_energy_feather_disagreeing_with_the_clock(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000))
    _write_energy(
        video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_ENERGY),
        motion_energy=[0.0, 4.0],
        frame_luminance=[1.0, 5.0],
    )

    with pytest.raises(ValueError, match=re.escape("has 2 rows, but the")):
        assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(1000))


def test_assemble_video_dataset_rejects_a_pupil_feather_disagreeing_with_the_clock(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 1000))
    _write_pupil(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_PUPIL), diameter=[20.0, 30.0, 40.0], blinking=[True] * 3
    )

    with pytest.raises(ValueError, match=re.escape("has 3 rows, but the")):
        assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(500))


def test_assemble_video_dataset_reads_back_the_feather_the_tracking_worker_wrote(
    video_data_path: Path,
    experiment_session: SessionData,
    write_dlc_predictions: Callable[..., Path],
    write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path],
) -> None:
    experiment_session.raw_data.camera_data_path.mkdir(parents=True, exist_ok=True)
    write_dlc_predictions(
        experiment_session.raw_data.camera_data_path.joinpath("face_eye_tracking.h5"),
        _build_points([_frame_specification(), _frame_specification(), _frame_specification()]),
    )
    process_mesoscope_video_tracking(session=experiment_session, output_directory=video_data_path)
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000))

    assembled = assemble_video_dataset(video_data_path=video_data_path, reference_time=_clock(500, 1500))

    assert assembled.columns == [column.value for column in PupilColumn]
    assert assembled[PupilColumn.PUPIL_DIAMETER_PX].to_list() == pytest.approx([20.0, 20.0], abs=1e-3)
    assert assembled[PupilColumn.BLINKING_STATE].to_list() == [0, 0]
    assert assembled[PupilColumn.DILATION_STATE].to_list() == [0, 0]


def test_resolve_slowest_camera_clock_rejects_an_absent_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("No camera timestamp feather")):
        resolve_slowest_camera_clock(video_data_path=tmp_path.joinpath("missing"))


def test_resolve_slowest_camera_clock_rejects_a_directory_holding_no_clock(video_data_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("No camera timestamp feather")):
        resolve_slowest_camera_clock(video_data_path=video_data_path)


def test_resolve_slowest_camera_clock_rejects_a_single_frame_clock(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0))

    with pytest.raises(FileNotFoundError, match=re.escape("at least two frames")):
        resolve_slowest_camera_clock(video_data_path=video_data_path)


def test_resolve_slowest_camera_clock_rejects_a_clock_spanning_no_duration(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), _clock(0, 0, 0))

    with pytest.raises(FileNotFoundError, match=re.escape("spanning a positive duration")):
        resolve_slowest_camera_clock(video_data_path=video_data_path)


def test_resolve_slowest_camera_clock_rejects_a_clock_whose_frames_run_backwards(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    # The timestamps are unsigned, so an out-of-order feather's endpoint difference wraps to a span of roughly six
    # hundred thousand years. That reads as the slowest camera in the session and would be handed back as the
    # reference clock every other data source is interpolated onto, so the span is measured in floating point.
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(5000, 1000))

    with pytest.raises(FileNotFoundError, match=re.escape("spanning a positive duration")):
        resolve_slowest_camera_clock(video_data_path=video_data_path)


def test_resolve_slowest_camera_clock_returns_the_body_camera_when_it_records_the_fewest_frames(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    write_camera_timestamps(
        video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000, 3000, 4000)
    )
    body_clock = _clock(0, 2000, 4000)
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), body_clock)

    resolved = resolve_slowest_camera_clock(video_data_path=video_data_path)

    assert resolved.tolist() == body_clock.tolist()
    assert resolved.dtype == np.uint64


def test_resolve_slowest_camera_clock_keeps_the_face_camera_when_the_body_camera_runs_faster(
    video_data_path: Path, write_camera_timestamps: Callable[[Path, NDArray[np.uint64]], Path]
) -> None:
    face_clock = _clock(0, 2000, 4000)
    write_camera_timestamps(video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS), face_clock)
    write_camera_timestamps(
        video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS), _clock(0, 1000, 2000, 3000, 4000)
    )

    resolved = resolve_slowest_camera_clock(video_data_path=video_data_path)

    assert resolved.tolist() == face_clock.tolist()
