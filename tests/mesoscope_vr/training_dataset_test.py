"""Contains tests for the Mesoscope-VR training-session data assembler, and for the geometry resolver reporting the
heights that assembler works at, against on-disk sessions built from real feathers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import SessionTypes

from sollertia_forgery.mesoscope_vr.forging import assemble_mesoscope_session
from sollertia_forgery.mesoscope_vr.metadata import VideoDataFiles, BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.assembly_sources import resolve_mesoscope_assembly_sources
from sollertia_forgery.mesoscope_vr.training_dataset import (
    assemble_training_dataset,
    resolve_mesoscope_assembly_geometry,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_BODY_FRAME_PERIOD_US: int = 200_000
"""The interval between two consecutive body-camera frames, which makes the body camera the slower of the two."""

_BODY_FRAME_COUNT: int = 30
"""The number of frames the body camera records, whose timestamps become the assembly reference clock."""

_FACE_FRAME_PERIOD_US: int = 100_000
"""The interval between two consecutive face-camera frames, which makes the face camera the faster of the two."""

_FACE_FRAME_COUNT: int = 60
"""The number of frames the face camera records."""

_VALVE_SAMPLE_COUNT: int = 4
"""The samples the synthetic valve feather holds, which is the height at which the assembly reads it."""

_LICK_SAMPLE_COUNT: int = 3
"""The samples the synthetic lick feather holds."""

_ENCODER_SAMPLE_COUNT: int = 60
"""The samples the synthetic encoder feather holds, one every hundred milliseconds over the recorded span."""

_STATE_SAMPLE_COUNT: int = 3
"""The samples each of the two synthetic runtime state feathers holds."""

_SESSION_START_US: int = 1_000_000
"""The timestamp at which the acquisition system first leaves idle, which anchors the head of the clipped dataset."""

_REST_ONSET_US: int = 4_000_000
"""The timestamp at which the acquisition system enters the rest state for the remainder of the session."""

_RUNTIME_END_US: int = 5_000_000
"""The timestamp of the final runtime-state entry, which anchors the tail of the clipped dataset."""

_BEHAVIOR_COLUMNS: frozenset[str] = frozenset(
    {"time_us", "elapsed_minutes", "distance_cm", "speed_cm_s", "lick", "water_uL", "reward", "system_state"}
)
"""The columns a run-training session's behavior sub-dataset contributes to the assembled feather."""

_VIDEO_COLUMNS: frozenset[str] = frozenset(
    {
        "face_camera_motion_energy",
        "face_camera_frame_luminance",
        "body_camera_motion_energy",
        "body_camera_frame_luminance",
    }
)
"""The columns the video sub-dataset contributes when both cameras carry a motion-energy feather."""


def _write_feather(path: Path, columns: dict[str, NDArray[np.number]]) -> None:
    """Writes one uncompressed feather holding the given columns.

    Args:
        path: The path of the feather file to write.
        columns: The column name to column value mapping the feather stores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_ipc(file=path, compression="uncompressed")


def _body_clock() -> NDArray[np.uint64]:
    """Returns the body camera's per-frame acquisition timestamps, which serve as the reference clock.

    Returns:
        The body-camera timestamps in microseconds since the UTC epoch.
    """
    return np.arange(_BODY_FRAME_COUNT, dtype=np.uint64) * np.uint64(_BODY_FRAME_PERIOD_US)


def _write_behavior_sources(session: SessionData) -> None:
    """Writes the module-parsed and runtime-parsed feathers read by a run-training session's behavior assembly and
    session-bounds clip.

    The valve stream plays one reward tone that dispenses water, the encoder advances at a constant rate, and the
    system leaves idle at the session start before settling into rest.

    Args:
        session: The loaded training session whose processed directories receive the feathers.
    """
    microcontroller_data_path = session.processed_data.microcontroller_data_path
    runtime_data_path = session.processed_data.runtime_data_path

    _write_feather(
        path=microcontroller_data_path.joinpath(BehaviorDataFiles.VALVE),
        columns={
            "time_us": np.array([0, 2_000_000, 2_400_000, 2_800_000], dtype=np.uint64),
            "dispensed_water_volume_uL": np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64),
            "tone_state": np.array([0, 1, 1, 0], dtype=np.uint8),
        },
    )
    _write_feather(
        path=microcontroller_data_path.joinpath(BehaviorDataFiles.LICK),
        columns={
            "time_us": np.array([0, 2_400_000, 2_600_000], dtype=np.uint64),
            "lick_state": np.array([0, 1, 0], dtype=np.uint8),
        },
    )
    encoder_time = np.arange(_ENCODER_SAMPLE_COUNT, dtype=np.uint64) * np.uint64(100_000)
    _write_feather(
        path=microcontroller_data_path.joinpath(BehaviorDataFiles.ENCODER),
        columns={
            "time_us": encoder_time,
            "traveled_distance_cm": np.arange(encoder_time.size, dtype=np.float64) * 2.0,
        },
    )
    _write_feather(
        path=runtime_data_path.joinpath(BehaviorDataFiles.SYSTEM_STATE),
        columns={
            "time_us": np.array([0, _SESSION_START_US, _REST_ONSET_US], dtype=np.uint64),
            "system_state": np.array([0, 2, 1], dtype=np.uint8),
        },
    )
    _write_feather(
        path=runtime_data_path.joinpath(BehaviorDataFiles.RUNTIME_STATE),
        columns={
            "time_us": np.array([0, _SESSION_START_US, _RUNTIME_END_US], dtype=np.uint64),
            "runtime_state": np.array([0, 1, 1], dtype=np.uint8),
        },
    )


def _write_camera_clocks(session: SessionData) -> None:
    """Writes both camera timestamp feathers, which the slowest-camera clock resolver reads.

    Args:
        session: The loaded training session whose processed video directory receives the feathers.
    """
    video_data_path = session.processed_data.video_data_path
    _write_feather(
        path=video_data_path.joinpath(VideoDataFiles.BODY_CAMERA_TIMESTAMPS),
        columns={"frame_time_us": _body_clock()},
    )
    _write_feather(
        path=video_data_path.joinpath(VideoDataFiles.FACE_CAMERA_TIMESTAMPS),
        columns={"frame_time_us": np.arange(_FACE_FRAME_COUNT, dtype=np.uint64) * np.uint64(_FACE_FRAME_PERIOD_US)},
    )


def _write_camera_energies(session: SessionData) -> None:
    """Writes both cameras' motion-energy feathers, holding one row per recorded frame.

    Args:
        session: The loaded training session whose processed video directory receives the feathers.
    """
    video_data_path = session.processed_data.video_data_path
    for filename, frame_count in (
        (VideoDataFiles.BODY_CAMERA_ENERGY, _BODY_FRAME_COUNT),
        (VideoDataFiles.FACE_CAMERA_ENERGY, _FACE_FRAME_COUNT),
    ):
        _write_feather(
            path=video_data_path.joinpath(filename),
            columns={
                "motion_energy": np.arange(frame_count, dtype=np.float32),
                "frame_luminance": np.full(frame_count, 128.0, dtype=np.float32),
            },
        )


@pytest.fixture
def prepared_training_session(training_session: SessionData) -> SessionData:
    """Builds a fully processed run-training session carrying its behavior, camera clock, and camera energy feathers.

    Args:
        training_session: The acquired run-training session that receives the processed feathers.

    Returns:
        The same session, now holding every input the training assembler reads.
    """
    _write_behavior_sources(session=training_session)
    _write_camera_clocks(session=training_session)
    _write_camera_energies(session=training_session)
    return training_session


def test_assemble_training_dataset_writes_the_clipped_feather(
    prepared_training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies the assembled feather sits on the slowest camera's clock and is clipped to the session bounds."""
    output_path = tmp_path.joinpath("forged", "data.feather")

    assemble_mesoscope_session(
        source_session_path=prepared_training_session.raw_data_path.parent,
        output_path=output_path,
        dataset_name="training_dataset",
    )

    assembled = pl.read_ipc(source=output_path)

    # The reference clock is the body camera's, and the clip keeps only the samples inside the session bounds.
    retained = [timestamp for timestamp in _body_clock().tolist() if _SESSION_START_US <= timestamp <= _RUNTIME_END_US]
    assert assembled["time_us"].to_list() == retained
    assert set(assembled.columns) == _BEHAVIOR_COLUMNS | _VIDEO_COLUMNS


def test_assemble_training_dataset_reports_the_recorded_behavior(
    prepared_training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies the assembled columns carry the recorded states, rewards, and camera values."""
    output_path = tmp_path.joinpath("forged", "data.feather")

    assemble_mesoscope_session(
        source_session_path=prepared_training_session.raw_data_path.parent,
        output_path=output_path,
        dataset_name="training_dataset",
    )

    assembled = pl.read_ipc(source=output_path)
    states = dict(zip(assembled["time_us"].to_list(), assembled["system_state"].to_list(), strict=True))
    rewards = dict(zip(assembled["time_us"].to_list(), assembled["reward"].to_list(), strict=True))

    assert states[_SESSION_START_US] == "run"
    assert states[_REST_ONSET_US] == "rest"
    # The tone opens at 2_000_000 and the delivery lands at 2_400_000, so every sample of that span reads as rewarded.
    assert rewards[2_200_000] == "yes"
    assert rewards[3_000_000] == "no"
    # The reference clock starts at zero, so the first retained sample sits one second into the recording.
    assert assembled["elapsed_minutes"].to_list()[0] == pytest.approx(0.02)
    # The body camera's luminance is constant, and its motion energy is its own frame index.
    assert assembled["body_camera_frame_luminance"].to_list() == [128.0] * assembled.height
    assert assembled["body_camera_motion_energy"].to_list()[0] == pytest.approx(
        _SESSION_START_US / _BODY_FRAME_PERIOD_US
    )


def test_assemble_training_dataset_forges_without_camera_energy(training_session: SessionData, tmp_path: Path) -> None:
    """Verifies a session whose cameras produced only timestamps forges with the behavior columns alone."""
    _write_behavior_sources(session=training_session)
    _write_camera_clocks(session=training_session)
    output_path = tmp_path.joinpath("forged", "data.feather")

    assemble_training_dataset(source_session_path=training_session.raw_data_path.parent, output_path=output_path)

    assert set(pl.read_ipc(source=output_path).columns) == _BEHAVIOR_COLUMNS


def test_assemble_training_dataset_rejects_a_missing_microcontroller_directory(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies an unprocessed microcontroller stage is reported before any assembly work starts."""
    _write_camera_clocks(session=training_session)
    session_name = training_session.session_name

    with pytest.raises(FileNotFoundError, match=rf"(?s)session '{session_name}'.*microcontroller data\s+directory"):
        assemble_training_dataset(
            source_session_path=training_session.raw_data_path.parent,
            output_path=tmp_path.joinpath("forged", "data.feather"),
        )


def test_assemble_training_dataset_rejects_a_missing_runtime_directory(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies an unprocessed runtime stage is reported even when the microcontroller stage finished."""
    _write_behavior_sources(session=training_session)
    for feather in training_session.processed_data.runtime_data_path.iterdir():
        feather.unlink()
    training_session.processed_data.runtime_data_path.rmdir()

    with pytest.raises(FileNotFoundError, match=r"(?s)processed runtime\s+data directory"):
        assemble_training_dataset(
            source_session_path=training_session.raw_data_path.parent,
            output_path=tmp_path.joinpath("forged", "data.feather"),
        )


def test_assemble_training_dataset_rejects_a_session_without_a_camera_clock(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies a session with no camera timestamps fails before the output directory is created."""
    _write_behavior_sources(session=training_session)
    output_path = tmp_path.joinpath("forged", "data.feather")

    with pytest.raises(FileNotFoundError, match="no camera clock"):
        assemble_training_dataset(source_session_path=training_session.raw_data_path.parent, output_path=output_path)

    assert not output_path.parent.exists()


def test_resolve_mesoscope_assembly_geometry_reports_the_reference_clock_and_every_source(
    prepared_training_session: SessionData,
) -> None:
    """Verifies the geometry reports both heights the assembly works at: the reference clock its columns are placed
    on, and the height each source it reads stands at.

    The face camera runs at twice the body camera's rate, so the assembly settles on the body camera's clock while
    holding the face camera's feathers at twice that height. The two heights are reported separately because neither
    states the other.
    """
    geometry = resolve_mesoscope_assembly_geometry(session=prepared_training_session)

    assert geometry.reference_samples == _BODY_FRAME_COUNT

    # One entry per clock the assembly reads, in the order it reads them: the face and body cameras, whose three
    # feathers each share one clock, then the valve, lick and encoder feathers, then the two runtime state feathers.
    # The optional screen, brake and torque feathers a run-training session never writes are absent from both.
    assert geometry.source_samples == (
        _FACE_FRAME_COUNT,
        _BODY_FRAME_COUNT,
        _VALVE_SAMPLE_COUNT,
        _LICK_SAMPLE_COUNT,
        _ENCODER_SAMPLE_COUNT,
        _STATE_SAMPLE_COUNT,
        _STATE_SAMPLE_COUNT,
    )


def test_resolve_mesoscope_assembly_geometry_counts_the_sources_a_session_actually_wrote(
    training_session: SessionData,
) -> None:
    """Verifies a session that wrote none of its behavior feathers reports its camera clocks alone, since a source
    the assembler never opens holds nothing for the job to be charged.
    """
    _write_camera_clocks(session=training_session)

    geometry = resolve_mesoscope_assembly_geometry(session=training_session)

    assert geometry.reference_samples == _BODY_FRAME_COUNT
    assert geometry.source_samples == (_FACE_FRAME_COUNT, _BODY_FRAME_COUNT)


def test_resolve_mesoscope_assembly_geometry_refuses_a_session_without_a_reference_clock(
    training_session: SessionData,
) -> None:
    """Verifies a session whose cameras left no usable clock is refused rather than measured at a floor, which is the
    answer its assembler gives for it as well.
    """
    _write_behavior_sources(session=training_session)

    with pytest.raises(FileNotFoundError, match="no camera clock"):
        resolve_mesoscope_assembly_geometry(session=training_session)


def test_resolve_mesoscope_assembly_sources_refuses_an_uncovered_session_type(
    session_factory: Callable[..., SessionData],
) -> None:
    """Verifies a session type no Mesoscope-VR assembler covers is refused, and that the refusal names the types that
    are covered, since no assembly of an uncovered type exists whose sources could be reported.
    """
    window_checking_session = session_factory(session_type=SessionTypes.WINDOW_CHECKING)

    with pytest.raises(ValueError, match="not a supported forging session type"):
        resolve_mesoscope_assembly_sources(session=window_checking_session)


def test_resolve_mesoscope_assembly_geometry_refuses_an_uncovered_session_type(
    session_factory: Callable[..., SessionData],
) -> None:
    """Verifies the refusal reaches the geometry resolver, which reports it once the reference clock resolves, so a
    job sizing an uncovered session fails rather than being charged a training session's sources.
    """
    window_checking_session = session_factory(session_type=SessionTypes.WINDOW_CHECKING)
    _write_camera_clocks(session=window_checking_session)

    with pytest.raises(ValueError, match="not a supported forging session type"):
        resolve_mesoscope_assembly_geometry(session=window_checking_session)
