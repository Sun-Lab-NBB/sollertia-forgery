"""Contains tests for the system-agnostic per-camera motion-energy analysis."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from pathlib import Path
from itertools import pairwise
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import polars as pl
import pytest
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData
from sollertia_shared_assets import ProcessingTrackers
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, limit_worker_threads

from sollertia_forgery.video import ENERGY_JOB_NAME, MotionEnergyColumn, run_video_processing_pipeline
import sollertia_forgery.video.pipeline as pipeline_module
from sollertia_forgery.video.motion_energy import (
    _SPATIAL_BIN_SIZE,
    MOTION_ENERGY_SUFFIX,
    _MINIMUM_CHUNK_FRAMES,
    _bin_frame,
    _join_chunks,
    _plan_chunks,
    _energy_chunk,
    resolve_camera_video,
    compute_camera_motion_energy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_FRAME_HEIGHT: int = 100
"""The fixture frame height, chosen to leave a partial block at the frame edge so the block-mean crop path runs."""

_FRAME_WIDTH: int = 64
"""The fixture frame width, chosen to leave a partial block at the frame edge so the block-mean crop path runs."""

_FIXTURE_FRAME_COUNT: int = 60
"""The frame count of the fixture recordings, which the chunk-seam test tiles exactly."""

_CHUNK_FRAMES: int = 10
"""The frame count of each chunk the seam test decodes, dividing the fixture recording evenly."""

_ENERGY_TOLERANCE: float = 1e-4
"""The absolute tolerance the value assertions allow, covering float32 accumulation in the block-mean reduction."""

_THREAD_LIMIT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
    "TIFFFILE_NUM_THREADS",
)
"""The threading-layer environment variables the motion-energy decode pool relies on being capped for it. Naming them
here rather than reading the library's own tuple keeps a rename loud, since a variable that silently stops being
capped leaves each decode worker opening a pool sized to the whole machine."""


def _write_video(path: Path, frames: NDArray[np.uint8], fps: int = 30) -> Path:
    """Writes a stack of grayscale frames into a video file the analysis can decode, and returns its path."""
    height, width = frames.shape[1:]
    writer = cv2.VideoWriter(
        filename=str(path), fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=(width, height), isColor=True
    )
    for frame in frames:
        writer.write(cv2.cvtColor(src=frame, code=cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def _decoded_frames(path: Path) -> list[NDArray[np.float32]]:
    """Decodes every frame of a video into the block means the analysis computes, for use as a reference.

    The fixture writer is lossy in the same way the acquisition encoder is, so a value assertion compares against the
    frames the decoder produced rather than against the frames that were written.
    """
    capture = cv2.VideoCapture(str(path))
    capture.set(propId=cv2.CAP_PROP_CONVERT_RGB, value=0)
    frames = []
    while True:
        decoded, frame = capture.read()
        if not decoded:
            break
        frames.append(_bin_frame(frame=frame))
    capture.release()
    return frames


def _make_session(tmp_path: Path, cameras: dict[int, str]) -> SimpleNamespace:
    """Builds a stand-in for SessionData exposing only what the video pipeline reads, plus its camera manifest keyed
    by camera source ID.
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


def _record_cameras(session: SimpleNamespace, names: tuple[str, ...], frames: NDArray[np.uint8]) -> None:
    """Writes the given frame stack as each named camera's recording under the session's raw camera directory."""
    for name in names:
        _write_video(path=session.raw_data.camera_data_path.joinpath(f"test_session_{name}.mp4"), frames=frames)


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    """Builds a recording whose every frame is identical."""
    rng = np.random.default_rng(seed=17)
    single = rng.integers(low=0, high=256, size=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    return _write_video(
        path=tmp_path.joinpath("static.mp4"), frames=np.repeat(a=single[None], repeats=_FIXTURE_FRAME_COUNT, axis=0)
    )


@pytest.fixture
def moving_video(tmp_path: Path) -> Path:
    """Builds a recording containing a bright block that moves between frames."""
    frames = np.zeros((_FIXTURE_FRAME_COUNT, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for index in range(frames.shape[0]):
        offset = (index * 3) % 40
        frames[index, 20 + offset : 40 + offset, 10:40] = 255
    return _write_video(path=tmp_path.joinpath("moving.mp4"), frames=frames)


@pytest.fixture
def patched_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Builds a two-camera stand-in session and installs it as the loader the video pipeline resolves."""
    session = _make_session(tmp_path=tmp_path, cameras={51: "face_camera", 62: "body_camera"})
    monkeypatch.setattr(
        pipeline_module,
        "SessionData",
        SimpleNamespace(load=lambda session_path: session),  # noqa: ARG005
    )
    return session


def test_binning_matches_exact_block_mean() -> None:
    """Verifies the block-mean reduction is exact, which resizing with pixel-area interpolation is not.

    This is the regression guard against replacing the box filter with ``cv2.resize(..., INTER_AREA)``. Pixel-area
    interpolation is an exact block mean only when both dimensions divide evenly by the bin size. Both real cameras'
    dimensions fail that, so the interpolation blends across block boundaries.
    """
    rng = np.random.default_rng(seed=3)
    frame = rng.integers(low=0, high=256, size=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)

    binned = _bin_frame(frame=frame)

    bin_height = _FRAME_HEIGHT // _SPATIAL_BIN_SIZE * _SPATIAL_BIN_SIZE
    bin_width = _FRAME_WIDTH // _SPATIAL_BIN_SIZE * _SPATIAL_BIN_SIZE
    expected = (
        frame[:bin_height, :bin_width]
        .reshape(bin_height // _SPATIAL_BIN_SIZE, _SPATIAL_BIN_SIZE, bin_width // _SPATIAL_BIN_SIZE, _SPATIAL_BIN_SIZE)
        .mean(axis=(1, 3), dtype=np.float64)
    )

    assert binned.shape == expected.shape
    assert binned.dtype == np.float32
    assert np.abs(binned - expected).max() < _ENERGY_TOLERANCE


def test_binning_crops_partial_blocks(static_video: Path) -> None:
    """Verifies frame dimensions that are not multiples of the bin size crop cleanly to whole blocks."""
    binned = _decoded_frames(static_video)[0]
    assert binned.shape == (_FRAME_HEIGHT // _SPATIAL_BIN_SIZE, _FRAME_WIDTH // _SPATIAL_BIN_SIZE)


def test_chunked_result_is_bit_identical_to_sequential(moving_video: Path) -> None:
    """Verifies splitting a recording into decode chunks changes nothing about the result.

    The seam invariant the whole parallel design rests on: each chunk beyond the first decodes a priming frame so the
    difference spanning its leading boundary is computed rather than lost or duplicated.
    """
    sequential_energy, sequential_luminance = _energy_chunk(
        video_path=str(moving_video), start_frame=0, frame_count=_FIXTURE_FRAME_COUNT
    )

    chunk_energies, chunk_luminances = [], []
    for start in range(0, _FIXTURE_FRAME_COUNT, _CHUNK_FRAMES):
        energy, luminance = _energy_chunk(video_path=str(moving_video), start_frame=start, frame_count=_CHUNK_FRAMES)
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
    expected = [float(np.mean(np.abs(current - previous))) for previous, current in pairwise(frames)]

    assert np.allclose(energy[1 : len(expected) + 1], expected, atol=_ENERGY_TOLERANCE)


def test_luminance_tracks_a_global_brightness_step(tmp_path: Path) -> None:
    """Verifies the luminance column reproduces a whole-field brightness step and flags it as the energy peak."""
    step_index = 20
    frames = np.full((40, _FRAME_HEIGHT, _FRAME_WIDTH), fill_value=60, dtype=np.uint8)
    frames[step_index:] = 180
    video_path = _write_video(path=tmp_path.joinpath("step.mp4"), frames=frames)

    output_path = tmp_path.joinpath("step_energy.feather")
    compute_camera_motion_energy(video_path=video_path, output_path=output_path, workers=1)

    frame = pl.read_ipc(output_path)
    energy = frame[MotionEnergyColumn.MOTION_ENERGY].to_numpy()
    luminance = frame[MotionEnergyColumn.FRAME_LUMINANCE].to_numpy()

    assert luminance[:step_index].mean() < luminance[step_index:].mean()
    # The step frame is where the whole field changed at once, so it carries the largest difference in the recording.
    assert int(np.nanargmax(energy)) == step_index


def test_output_schema_is_positional(tmp_path: Path, moving_video: Path) -> None:
    """Verifies the feather's columns, dtypes, and one row per decoded frame."""
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=1)

    frame = pl.read_ipc(output_path)
    assert dict(frame.schema) == {
        MotionEnergyColumn.MOTION_ENERGY: pl.Float32,
        MotionEnergyColumn.FRAME_LUMINANCE: pl.Float32,
    }
    # The feather is a positional table, so it holds exactly one row per decoded frame.
    assert len(frame) == len(_decoded_frames(moving_video))


def test_unresolved_worker_count_is_resolved(tmp_path: Path, moving_video: Path) -> None:
    """Verifies a caller may pass an unresolved worker count, as remote mode does.

    Remote mode forwards the raw ``workers`` value, which may be -1. Passing that straight to a process pool raises,
    so the analysis must resolve it itself.
    """
    output_path = tmp_path.joinpath("moving_energy.feather")
    compute_camera_motion_energy(video_path=moving_video, output_path=output_path, workers=-1)
    assert len(pl.read_ipc(output_path)) == _FIXTURE_FRAME_COUNT


@pytest.mark.parametrize(("frame_count", "workers"), [(3000, 8), (100_000, 64), (252_477, 64), (1, 8)])
def test_plan_chunks_tiles_the_recording_exactly(frame_count: int, workers: int) -> None:
    """Verifies the planned chunks cover every frame exactly once, with no gap and no overlap."""
    chunks = _plan_chunks(frame_count=frame_count, workers=workers)

    assert chunks[0][0] == 0
    assert sum(size for _, size in chunks) == frame_count
    assert len(chunks) <= max(1, workers)
    for (start, size), (next_start, _) in pairwise(chunks):
        assert start + size == next_start


def test_plan_chunks_respects_the_minimum_chunk_size() -> None:
    """Verifies a recording too short to split is decoded as a single chunk."""
    assert _plan_chunks(frame_count=_MINIMUM_CHUNK_FRAMES - 1, workers=64) == [(0, _MINIMUM_CHUNK_FRAMES - 1)]


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


def test_pipeline_writes_one_energy_feather_per_camera(tmp_path: Path, patched_session: SimpleNamespace) -> None:
    """Verifies the energy stage runs per camera and names each output after its manifest name."""
    frames = np.zeros((30, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for index in range(frames.shape[0]):
        frames[index, index : index + 10, 5:25] = 255
    _record_cameras(session=patched_session, names=("face_camera", "body_camera"), frames=frames)

    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    video_data = patched_session.processed_data.video_data_path
    for name in ("face_camera", "body_camera"):
        output_path = video_data.joinpath(f"{name}{MOTION_ENERGY_SUFFIX}")
        assert output_path.is_file()
        assert len(pl.read_ipc(output_path)) == frames.shape[0]


def test_pipeline_energy_stage_no_ops_without_a_recording(tmp_path: Path, patched_session: SimpleNamespace) -> None:
    """Verifies a camera with no recording completes its job rather than failing the shared tracker.

    A rig that ran only one of its registered cameras must not wedge the video tracker on the camera it did not run.
    """
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _record_cameras(session=patched_session, names=("face_camera",), frames=frames)

    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    video_data = patched_session.processed_data.video_data_path
    assert video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").exists()

    # Both energy jobs must be recorded complete: the absent recording is a no-op, not a failure.
    tracker = ProcessingTracker(file_path=video_data.joinpath(ProcessingTrackers.VIDEO))
    for source_id in (51, 62):
        job_id = ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier=str(source_id))
        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED


def test_pipeline_energy_stage_honors_the_target_camera(tmp_path: Path, patched_session: SimpleNamespace) -> None:
    """Verifies selecting a single camera measures only that camera."""
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _record_cameras(session=patched_session, names=("face_camera", "body_camera"), frames=frames)

    run_video_processing_pipeline(session_path=tmp_path, energy=True, target_camera=51, workers=1)

    video_data = patched_session.processed_data.video_data_path
    assert video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").exists()


def test_pipeline_dispatches_a_single_energy_job_by_id(tmp_path: Path, patched_session: SimpleNamespace) -> None:
    """Verifies remote mode runs exactly the requested energy job, leaving its sibling untouched."""
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _record_cameras(session=patched_session, names=("face_camera", "body_camera"), frames=frames)

    job_id = ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier="62")
    run_video_processing_pipeline(session_path=tmp_path, job_id=job_id, workers=1)

    video_data = patched_session.processed_data.video_data_path
    assert video_data.joinpath(f"body_camera{MOTION_ENERGY_SUFFIX}").is_file()
    assert not video_data.joinpath(f"face_camera{MOTION_ENERGY_SUFFIX}").exists()


def test_pipeline_universe_carries_an_energy_job_per_camera(tmp_path: Path, patched_session: SimpleNamespace) -> None:
    """Verifies every registered camera contributes an energy job to the tracker-alignment universe.

    The universe must cover every registered camera rather than only those the invocation runs, so that a partial
    invocation aligns the tracker without wiping the sibling job an earlier run already completed.
    """
    frames = np.zeros((20, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    _record_cameras(session=patched_session, names=("face_camera", "body_camera"), frames=frames)
    run_video_processing_pipeline(session_path=tmp_path, energy=True, workers=1)

    # Re-runs the stage for one camera alone, which dispatches a single job while aligning the tracker against the
    # full universe the manifest defines.
    run_video_processing_pipeline(session_path=tmp_path, energy=True, target_camera=51, workers=1)

    tracker = ProcessingTracker(
        file_path=patched_session.processed_data.video_data_path.joinpath(ProcessingTrackers.VIDEO)
    )
    energy_jobs = [
        ProcessingTracker.generate_job_id(job_name=ENERGY_JOB_NAME, specifier=str(source_id)) for source_id in (51, 62)
    ]
    # The camera the second invocation passed over keeps its completed record, rather than losing it or falling back
    # to a scheduled one.
    for job_id in energy_jobs:
        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
    # Neither invocation registers a job it did not dispatch, which would leave the session reporting scheduled work
    # against the stage flags it was never given.
    assert set(tracker.snapshot()) == set(energy_jobs)


def test_limited_worker_threads_cap_and_restore_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the thread caps are set inside the block and the prior environment is restored on exit.

    The caps must not leak past the pool they were set for. Every caller outside a decode pool relies on the numeric
    backends opening their full thread pool, so a leaked cap would silently narrow them for the rest of the process.
    """
    sentinel = "OMP_NUM_THREADS"
    monkeypatch.setenv(name=sentinel, value="13")

    with limit_worker_threads():
        assert all(os.environ[variable] == "1" for variable in _THREAD_LIMIT_VARIABLES)
    assert os.environ[sentinel] == "13"


def test_limited_worker_threads_remove_variables_they_introduced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a variable absent before the block is absent again after it, rather than left set to one."""
    sentinel = _THREAD_LIMIT_VARIABLES[0]
    monkeypatch.delenv(name=sentinel, raising=False)

    with limit_worker_threads():
        assert os.environ[sentinel] == "1"
    assert sentinel not in os.environ


def test_unreadable_recording_errors(tmp_path: Path) -> None:
    """Verifies a file that cannot be decoded raises rather than writing an empty feather."""
    broken_path = tmp_path.joinpath("broken.mp4")
    broken_path.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="could not be opened"):
        compute_camera_motion_energy(video_path=broken_path, output_path=tmp_path.joinpath("out.feather"), workers=1)


@pytest.fixture(scope="module")
def chunked_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Builds a recording long enough for the analysis to split it into more than one decode chunk.

    The frames are kept small so the encode stays cheap, since only the frame count decides the chunk plan.
    """
    frame_count = _MINIMUM_CHUNK_FRAMES * 2
    frames = np.zeros((frame_count, 16, 16), dtype=np.uint8)
    for index in range(frame_count):
        frames[index, index % 10 : index % 10 + 4, 2:6] = 255
    return _write_video(path=tmp_path_factory.mktemp("chunked").joinpath("chunked.mp4"), frames=frames)


def test_own_pool_multi_chunk_result_matches_the_sequential_pass(tmp_path: Path, chunked_video: Path) -> None:
    """Verifies a recording split across a pool the analysis owns yields exactly the sequential result."""
    output_path = tmp_path.joinpath("chunked_energy.feather")
    compute_camera_motion_energy(video_path=chunked_video, output_path=output_path, workers=2)

    written = pl.read_ipc(output_path)
    sequential_energy, sequential_luminance = _energy_chunk(
        video_path=str(chunked_video), start_frame=0, frame_count=_MINIMUM_CHUNK_FRAMES * 2
    )

    energy = written[MotionEnergyColumn.MOTION_ENERGY].to_numpy()
    assert np.array_equal(np.isnan(energy), np.isnan(sequential_energy))
    finite = ~np.isnan(sequential_energy)
    assert np.array_equal(energy[finite], sequential_energy[finite])
    assert np.array_equal(written[MotionEnergyColumn.FRAME_LUMINANCE].to_numpy(), sequential_luminance)


def test_shared_pool_decodes_the_chunks_and_reports_progress(tmp_path: Path, chunked_video: Path) -> None:
    """Verifies a caller-owned pool is used as-is, and the per-chunk progress report leaves the result unchanged."""
    own_path = tmp_path.joinpath("own_energy.feather")
    shared_path = tmp_path.joinpath("shared_energy.feather")
    compute_camera_motion_energy(video_path=chunked_video, output_path=own_path, workers=2)

    with ProcessPoolExecutor(max_workers=2) as executor:
        compute_camera_motion_energy(
            video_path=chunked_video,
            output_path=shared_path,
            workers=2,
            executor=executor,
            display_progress=True,
        )

    assert pl.read_ipc(shared_path).equals(pl.read_ipc(own_path))


def test_non_positive_reported_frame_count_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a container that opens but reports no frames errors instead of writing an empty feather."""

    class _EmptyCapture:
        """Stands in for a decoder whose container reports no frames at all."""

        def __init__(self, filename: str) -> None:
            self.filename = filename

        def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV capture interface.
            """Reports the recording as openable."""
            return True

        def get(self, propId: int) -> float:  # noqa: N803, ARG002 - mirrors the OpenCV capture interface.
            """Reports every queried container property as zero."""
            return 0.0

        def release(self) -> None:
            """Closes the stand-in capture."""

    monkeypatch.setattr(cv2, "VideoCapture", _EmptyCapture)

    with pytest.raises(ValueError, match="must be a positive"):
        compute_camera_motion_energy(
            video_path=tmp_path.joinpath("empty.mp4"), output_path=tmp_path.joinpath("out.feather"), workers=1
        )


def test_early_end_before_the_last_chunk_errors() -> None:
    """Verifies a chunk other than the last running out of frames is reported as a truncated recording."""
    results = [
        (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
        (np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)),
    ]

    with pytest.raises(ValueError, match="ended after 3 of its 5 frames"):
        _join_chunks(results=results, chunks=[(0, 5), (5, 5)], video_path=Path("truncated.mp4"))


def test_short_final_chunk_is_joined_and_announced() -> None:
    """Verifies the last chunk running out early yields the frames that decoded rather than an error."""
    results = [
        (np.array([np.nan, 1.0], dtype=np.float32), np.array([4.0, 5.0], dtype=np.float32)),
        (np.array([2.0], dtype=np.float32), np.array([6.0], dtype=np.float32)),
    ]

    energy, luminance = _join_chunks(results=results, chunks=[(0, 2), (2, 2)], video_path=Path("short.mp4"))

    assert energy.size == 3
    assert np.array_equal(energy[1:], np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(luminance, np.array([4.0, 5.0, 6.0], dtype=np.float32))


def test_chunk_decode_rejects_an_unopenable_recording(tmp_path: Path) -> None:
    """Verifies a chunk worker handed a file it cannot open names the recording it failed on."""
    broken_path = tmp_path.joinpath("broken.mp4")
    broken_path.write_bytes(b"not a video")

    with pytest.raises(ValueError, match="Unable to open"):
        _energy_chunk(video_path=str(broken_path), start_frame=0, frame_count=5)


def test_chunk_decode_rejects_an_undecodable_priming_frame(moving_video: Path) -> None:
    """Verifies a chunk whose priming frame lies past the end of the recording errors rather than skipping it."""
    with pytest.raises(ValueError, match="Unable to decode the frame preceding"):
        _energy_chunk(video_path=str(moving_video), start_frame=5000, frame_count=_CHUNK_FRAMES)


def test_chunk_decode_stops_at_the_end_of_the_recording(moving_video: Path) -> None:
    """Verifies a chunk planned longer than the recording returns only the frames that decoded."""
    energy, luminance = _energy_chunk(video_path=str(moving_video), start_frame=0, frame_count=200)

    assert energy.size == len(_decoded_frames(moving_video))
    assert energy.size == luminance.size
    assert not np.isnan(luminance).any()


def test_binning_takes_one_plane_of_a_multi_plane_frame() -> None:
    """Verifies a decoder falling back to a three-channel expansion is reduced on a single plane."""
    rng = np.random.default_rng(seed=11)
    plane = rng.integers(low=0, high=256, size=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    expanded = np.stack([np.zeros_like(plane), plane, np.full_like(plane, fill_value=255)], axis=2)

    assert np.array_equal(_bin_frame(frame=expanded), _bin_frame(frame=plane))
