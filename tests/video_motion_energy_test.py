"""Tests for the system-agnostic per-camera motion-energy analysis.

Every test builds its own recording with ``cv2.VideoWriter`` into ``tmp_path``, so no session data is required. The
fixtures deliberately use frame dimensions that are not multiples of the spatial bin size, so the block-mean crop path
is exercised by default rather than only on real recordings.

Notes:
    The fixture writer is lossy, exactly like the acquisition-time encoder, so tests that assert on values compare
    against energy recomputed from the DECODED frames rather than from the frames that were written. That is the
    honest comparison: the decoded frames are what the analysis actually consumes.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData
from sollertia_shared_assets import ProcessingTrackers
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.video import (
    ENERGY_JOB_NAME,
    SPATIAL_BIN_SIZE,
    MINIMUM_CHUNK_FRAMES,
    MOTION_ENERGY_SUFFIX,
    MotionEnergyColumn,
    resolve_camera_video,
    compute_camera_motion_energy,
    run_video_processing_pipeline,
)
from sollertia_forgery.video import pipeline as pipeline_module
from sollertia_forgery.video.motion_energy import _bin_frame, _plan_chunks, _energy_chunk

_FRAME_HEIGHT: int = 100
"""The fixture frame height. Not a multiple of the spatial bin size, so the block-mean crop path always runs."""

_FRAME_WIDTH: int = 64
"""The fixture frame width. Not a multiple of the spatial bin size, so the block-mean crop path always runs."""


def _write_video(path: Path, frames: np.ndarray, fps: int = 30) -> Path:
    """Writes a stack of grayscale frames into a video file the analysis can decode.

    Args:
        path: The path of the video file to write.
        frames: The ``(frame_count, height, width)`` uint8 frame stack.
        fps: The frame rate to record in the container.

    Returns:
        The path the video was written to.
    """
    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=True)
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def _decoded_frames(path: Path) -> list[np.ndarray]:
    """Decodes every frame of a video the same way the analysis does, for use as a reference.

    Args:
        path: The path to the video file.

    Returns:
        The decoded frames, each reduced to its block means.
    """
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    frames = []
    while True:
        decoded, frame = capture.read()
        if not decoded:
            break
        frames.append(_bin_frame(frame=frame))
    capture.release()
    return frames


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    """Builds a recording whose every frame is identical."""
    rng = np.random.default_rng(seed=17)
    single = rng.integers(0, 256, size=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    return _write_video(tmp_path.joinpath("static.mp4"), np.repeat(single[None], 60, axis=0))


@pytest.fixture
def moving_video(tmp_path: Path) -> Path:
    """Builds a recording containing a bright block that moves between frames."""
    frames = np.zeros((60, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for index in range(frames.shape[0]):
        offset = (index * 3) % 40
        frames[index, 20 + offset : 40 + offset, 10:40] = 255
    return _write_video(tmp_path.joinpath("moving.mp4"), frames)


def test_binning_matches_exact_block_mean() -> None:
    """Verifies the block-mean reduction is exact, which resizing with pixel-area interpolation is not.

    This is the regression guard against replacing the box filter with ``cv2.resize(..., INTER_AREA)``: that is only
    an exact block mean when both dimensions divide evenly by the bin size, and silently blends across block
    boundaries when they do not, which both real cameras' dimensions trigger.
    """
    rng = np.random.default_rng(seed=3)
    frame = rng.integers(0, 256, size=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)

    binned = _bin_frame(frame=frame)

    bin_height = _FRAME_HEIGHT // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE
    bin_width = _FRAME_WIDTH // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE
    expected = (
        frame[:bin_height, :bin_width]
        .reshape(bin_height // SPATIAL_BIN_SIZE, SPATIAL_BIN_SIZE, bin_width // SPATIAL_BIN_SIZE, SPATIAL_BIN_SIZE)
        .mean(axis=(1, 3), dtype=np.float64)
    )

    assert binned.shape == expected.shape
    assert binned.dtype == np.float32
    assert np.abs(binned - expected).max() < 1e-4


def test_binning_crops_partial_blocks(static_video: Path) -> None:
    """Verifies frame dimensions that are not multiples of the bin size crop cleanly to whole blocks."""
    binned = _decoded_frames(static_video)[0]
    assert binned.shape == (_FRAME_HEIGHT // SPATIAL_BIN_SIZE, _FRAME_WIDTH // SPATIAL_BIN_SIZE)


def test_chunked_result_is_bit_identical_to_sequential(moving_video: Path) -> None:
    """Verifies splitting a recording into decode chunks changes nothing about the result.

    The seam invariant the whole parallel design rests on: each chunk beyond the first decodes a priming frame so the
    difference spanning its leading boundary is computed rather than lost or duplicated.
    """
    sequential_energy, sequential_luminance = _energy_chunk(str(moving_video), 0, 60)

    chunk_energies, chunk_luminances = [], []
    for start in range(0, 60, 10):
        energy, luminance = _energy_chunk(str(moving_video), start, 10)
        chunk_energies.append(energy)
        chunk_luminances.append(luminance)
    chunked_energy = np.concatenate(chunk_energies)
    chunked_luminance = np.concatenate(chunk_luminances)

    assert np.array_equal(np.isnan(sequential_energy), np.isnan(chunked_energy))
    finite = ~np.isnan(sequential_energy)
    assert np.array_equal(sequential_energy[finite], chunked_energy[finite])
    assert np.array_equal(sequential_luminance, chunked_luminance)


def test_first_frame_is_the_only_missing_energy(tmp_path: Path, moving_video: Path) -> None:
    """Verifies energy is missing exactly at the first frame, and luminance is never missing."""
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=1)

    frame = pl.read_ipc(output_path)
    energy = frame[MotionEnergyColumn.MOTION_ENERGY].to_numpy()
    luminance = frame[MotionEnergyColumn.FRAME_LUMINANCE].to_numpy()

    assert np.isnan(energy[0])
    assert not np.isnan(energy[1:]).any()
    assert not np.isnan(luminance).any()


def test_static_recording_yields_near_zero_energy(tmp_path: Path, static_video: Path) -> None:
    """Verifies a recording with no movement floors the energy, leaving only codec noise."""
    output_path = tmp_path.joinpath("static_energy.feather")
    compute_camera_motion_energy(video_path=static_video, output_path=output_path, workers=1)

    energy = pl.read_ipc(output_path)[MotionEnergyColumn.MOTION_ENERGY].to_numpy()
    assert np.nanmax(energy) < 1.0


def test_energy_matches_the_decoded_frame_difference(tmp_path: Path, moving_video: Path) -> None:
    """Verifies the written energy is the mean absolute difference of the frames the decoder actually produced."""
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=1)
    energy = pl.read_ipc(output_path)[MotionEnergyColumn.MOTION_ENERGY].to_numpy()

    frames = _decoded_frames(moving_video)
    expected = [float(np.mean(np.abs(current - previous))) for previous, current in zip(frames, frames[1:])]

    assert np.allclose(energy[1 : len(expected) + 1], expected, atol=1e-4)


def test_luminance_tracks_a_global_brightness_step(tmp_path: Path) -> None:
    """Verifies the luminance column reproduces a whole-field brightness step and flags it as the energy peak.

    This pins the artifact-diagnostic contract: a display or illuminator changing lands on every pixel at once, and
    the luminance column is what lets a consumer separate that from real movement.
    """
    frames = np.full((40, _FRAME_HEIGHT, _FRAME_WIDTH), 60, dtype=np.uint8)
    frames[20:] = 180
    video_path = _write_video(tmp_path.joinpath("step.mp4"), frames)

    output_path = tmp_path.joinpath("step_energy.feather")
    compute_camera_motion_energy(video_path=video_path, output_path=output_path, workers=1)

    frame = pl.read_ipc(output_path)
    energy = frame[MotionEnergyColumn.MOTION_ENERGY].to_numpy()
    luminance = frame[MotionEnergyColumn.FRAME_LUMINANCE].to_numpy()

    assert luminance[:20].mean() < luminance[20:].mean()
    # The step frame is where the whole field changed at once, so it carries the largest difference in the recording.
    assert int(np.nanargmax(energy)) == 20


def test_output_schema_and_frame_index(tmp_path: Path, moving_video: Path) -> None:
    """Verifies the feather's columns, dtypes, and one-based frame index."""
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=1)

    frame = pl.read_ipc(output_path)
    assert dict(frame.schema) == {
        MotionEnergyColumn.FRAME.value: pl.UInt32,
        MotionEnergyColumn.MOTION_ENERGY.value: pl.Float32,
        MotionEnergyColumn.FRAME_LUMINANCE.value: pl.Float32,
    }
    assert frame[MotionEnergyColumn.FRAME][0] == 1
    assert frame[MotionEnergyColumn.FRAME][-1] == len(frame)


def test_unresolved_worker_count_is_resolved(tmp_path: Path, moving_video: Path) -> None:
    """Verifies a caller may pass an unresolved worker count, as remote mode does.

    Remote mode forwards the raw ``workers`` value, which may be -1. Passing that straight to a process pool raises,
    so the analysis must resolve it itself.
    """
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=-1)
    assert output_path.is_file()


def test_plan_chunks_tiles_the_recording_exactly() -> None:
    """Verifies the planned chunks cover every frame exactly once, with no gap and no overlap."""
    for frame_count, workers in ((3000, 8), (100_000, 64), (252_477, 64), (1, 8)):
        chunks = _plan_chunks(frame_count=frame_count, workers=workers)

        assert chunks[0][0] == 0
        assert sum(size for _, size in chunks) == frame_count
        assert len(chunks) <= max(1, workers)
        for (start, size), (next_start, _) in zip(chunks, chunks[1:]):
            assert start + size == next_start


def test_plan_chunks_respects_the_minimum_chunk_size() -> None:
    """Verifies a recording too short to split is decoded as a single chunk."""
    assert _plan_chunks(frame_count=MINIMUM_CHUNK_FRAMES - 1, workers=64) == [(0, MINIMUM_CHUNK_FRAMES - 1)]


def test_missing_recording_resolves_to_none(tmp_path: Path) -> None:
    """Verifies a camera with no recording resolves to None so its stage can no-op instead of failing."""
    tmp_path.joinpath("session_face_camera.mp4").touch()

    assert (
        resolve_camera_video(camera_data_directory=tmp_path, session_name="session", camera_name="body_camera") is None
    )
    assert (
        resolve_camera_video(
            camera_data_directory=tmp_path.joinpath("absent"), session_name="session", camera_name="face_camera"
        )
        is None
    )


def test_camera_recording_is_resolved_on_the_whole_name(tmp_path: Path) -> None:
    """Verifies a camera name is not matched as a suffix of a different camera's name.

    ``body_camera`` itself ends in ``_camera``, so a suffix match would resolve a camera named ``camera`` to the body
    camera's recording and silently measure the wrong camera.
    """
    tmp_path.joinpath("session_body_camera.mp4").touch()
    tmp_path.joinpath("session_face_camera.mp4").touch()

    assert resolve_camera_video(camera_data_directory=tmp_path, session_name="session", camera_name="camera") is None
    resolved = resolve_camera_video(camera_data_directory=tmp_path, session_name="session", camera_name="face_camera")
    assert resolved is not None
    assert resolved.name == "session_face_camera.mp4"


def _make_session(tmp_path: Path, cameras: dict[int, str]) -> SimpleNamespace:
    """Builds a stand-in for SessionData exposing only what the video pipeline reads, plus its camera manifest.

    Args:
        tmp_path: The temporary directory to build the session hierarchy under.
        cameras: The mapping of camera source ID to colloquial name to register in the manifest.

    Returns:
        The session stand-in.
    """
    behavior_path = tmp_path.joinpath("raw_data", "behavior_data")
    camera_path = tmp_path.joinpath("raw_data", "camera_data")
    behavior_path.mkdir(parents=True)
    camera_path.mkdir(parents=True)

    manifest = CameraManifest(
        sources=[CameraSourceData(id=source_id, name=name) for source_id, name in cameras.items()]
    )
    manifest.to_yaml(file_path=behavior_path.joinpath(CAMERA_MANIFEST_FILENAME))

    return SimpleNamespace(
        session_name="test_session",
        acquisition_system="mesoscope",
        raw_data=SimpleNamespace(behavior_data_path=behavior_path, camera_data_path=camera_path),
        processed_data=SimpleNamespace(video_data_path=tmp_path.joinpath("processed_data", "video_data")),
    )


def test_pipeline_writes_one_energy_feather_per_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the energy stage runs per camera and names each output after its manifest name."""
    session = _make_session(tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    frames = np.zeros((30, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for index in range(frames.shape[0]):
        frames[index, index : index + 10, 5:25] = 255
    for name in ("face_camera", "body_camera"):
        _write_video(session.raw_data.camera_data_path.joinpath(f"test_session_{name}.mp4"), frames)

    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    video_data = session.processed_data.video_data_path
    for name in ("face_camera", "body_camera"):
        output_path = video_data.joinpath(f"{name}{MOTION_ENERGY_SUFFIX}")
        assert output_path.is_file()
        assert len(pl.read_ipc(output_path)) == frames.shape[0]


def test_pipeline_energy_stage_no_ops_without_a_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a camera with no recording completes its job rather than failing the shared tracker.

    A rig that ran only one of its registered cameras must not wedge the video tracker on the camera it did not run.
    """
    session = _make_session(tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _write_video(session.raw_data.camera_data_path.joinpath("test_session_face_camera.mp4"), frames)

    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    video_data = session.processed_data.video_data_path
    assert video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").exists()

    # Both energy jobs must be recorded complete: the absent recording is a no-op, not a failure.
    tracker = ProcessingTracker(file_path=video_data.joinpath(ProcessingTrackers.VIDEO))
    for source_id in (51, 62):
        job_id = ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier=str(source_id))
        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


def test_pipeline_energy_stage_honors_the_target_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies selecting a single camera measures only that camera."""
    session = _make_session(tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for name in ("face_camera", "body_camera"):
        _write_video(session.raw_data.camera_data_path.joinpath(f"test_session_{name}.mp4"), frames)

    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    run_video_processing_pipeline(session_path=tmp_path, energy=True, target_camera=51, workers=1)

    video_data = session.processed_data.video_data_path
    assert video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").exists()


def test_pipeline_dispatches_a_single_energy_job_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies remote mode runs exactly the requested energy job, leaving its sibling untouched."""
    session = _make_session(tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for name in ("face_camera", "body_camera"):
        _write_video(session.raw_data.camera_data_path.joinpath(f"test_session_{name}.mp4"), frames)

    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    job_id = ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier="62")
    run_video_processing_pipeline(session_path=tmp_path, job_id=job_id, workers=1)

    video_data = session.processed_data.video_data_path
    assert video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").exists()


def test_pipeline_universe_carries_an_energy_job_per_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies every registered camera contributes an energy job to the tracker-alignment universe.

    The universe must cover every registered camera rather than only those that ran, so that a partial invocation
    never wipes a sibling job from the shared video tracker.
    """
    session = _make_session(tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _write_video(session.raw_data.camera_data_path.joinpath("test_session_face_camera.mp4"), frames)

    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    tracker = ProcessingTracker(file_path=session.processed_data.video_data_path.joinpath(ProcessingTrackers.VIDEO))
    for source_id in (51, 62):
        job_id = ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier=str(source_id))
        assert tracker.get_job_status(job_id=job_id) is not None


def test_unreadable_recording_errors(tmp_path: Path) -> None:
    """Verifies a file that cannot be decoded raises rather than writing an empty feather."""
    broken_path = tmp_path.joinpath("broken.mp4")
    broken_path.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="could not be opened"):
        compute_camera_motion_energy(video_path=broken_path, output_path=tmp_path.joinpath("out.feather"), workers=1)
