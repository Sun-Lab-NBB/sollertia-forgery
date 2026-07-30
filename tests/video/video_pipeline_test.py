"""Tests for the system-agnostic camera video-processing pipeline and its job discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest, CameraSourceData
from sollertia_shared_assets import ProcessingTrackers
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker
from ataraxis_video_system.video import TIMESTAMP_JOB_NAME

from sollertia_forgery.video.pipeline import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    _dispatch_job,
    _find_camera_logs,
    discover_video_jobs,
    video_job_prerequisites,
    _extract_camera_source_id,
    run_video_processing_pipeline,
)
from sollertia_forgery.video.motion_energy import MOTION_ENERGY_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Callable

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_FACE_SOURCE_ID: int = 51
"""The manifest source identifier of the camera every fixture gives both an archive and a recording."""

_BODY_SOURCE_ID: int = 62
"""The manifest source identifier of the second registered camera, used to prove per-camera job independence."""

_FACE_CAMERA: str = "face_camera"
"""The colloquial manifest name of the first registered camera, which also names the pupil-tracking output."""

_BODY_CAMERA: str = "body_camera"
"""The colloquial manifest name of the second registered camera."""

_FRAME_HEIGHT: int = 16
"""The height of every synthetic recording frame, kept small so encoding and decoding stay cheap."""

_FRAME_WIDTH: int = 16
"""The width of every synthetic recording frame, kept small so encoding and decoding stay cheap."""

_RECORDING_FRAMES: int = 12
"""The number of frames every synthetic recording carries."""

_PUPIL_POINTS: tuple[str, ...] = (
    "pupil_right",
    "pupil_bottom_right",
    "pupil_bottom",
    "pupil_bottom_left",
    "pupil_left",
    "pupil_top_left",
    "pupil_top",
    "pupil_top_right",
)
"""The eight pupil-perimeter bodyparts the Mesoscope-VR tracking function requires, in its own ring order."""

_EYE_POINTS: tuple[str, ...] = ("eye_right", "eye_bottom", "eye_left", "eye_top")
"""The four eye-perimeter bodyparts the Mesoscope-VR tracking function requires, in its own ring order."""


def _write_manifest(directory: Path, cameras: Mapping[int, str]) -> Path:
    """Writes the acquisition-time camera manifest the pipeline reads its job universe from.

    Args:
        directory: The raw behavior-data directory the manifest is written into.
        cameras: The mapping of camera source identifier to colloquial camera name to register.

    Returns:
        The path to the written manifest file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory.joinpath(CAMERA_MANIFEST_FILENAME)
    CameraManifest(sources=[CameraSourceData(id=source_id, name=name) for source_id, name in cameras.items()]).to_yaml(
        file_path=manifest_path
    )
    return manifest_path


def _moving_frames(frame_count: int = _RECORDING_FRAMES) -> NDArray[np.uint8]:
    """Builds a small grayscale frame stack holding one bright block that moves between consecutive frames.

    Args:
        frame_count: The number of frames the stack carries.

    Returns:
        The frame stack shaped as frames by height by width.
    """
    frames = np.zeros((frame_count, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.uint8)
    for index in range(frame_count):
        frames[index, index % 8 : index % 8 + 5, 3:9] = 255
    return frames


def _ring(center: tuple[float, float], radius: float, count: int, frame_count: int) -> list[NDArray[np.float64]]:
    """Builds one per-bodypart prediction array for every point of an evenly spaced ring.

    Args:
        center: The horizontal and vertical center of the ring, in pixels.
        radius: The ring radius, in pixels.
        count: The number of points spaced evenly around the ring.
        frame_count: The number of frames each point is predicted for.

    Returns:
        A list holding one ``(frame_count, 3)`` array of horizontal position, vertical position, and likelihood per
        ring point, in the ring order the tracking function expects.
    """
    arrays: list[NDArray[np.float64]] = []
    for index in range(count):
        angle = 2.0 * np.pi * index / count
        point = np.empty((frame_count, 3), dtype=np.float64)
        point[:, 0] = center[0] + radius * np.cos(angle)
        point[:, 1] = center[1] + radius * np.sin(angle)
        point[:, 2] = 1.0
        arrays.append(point)
    return arrays


@pytest.fixture
def camera_session(experiment_session: SessionData) -> SessionData:
    """Registers two cameras in the session's acquisition-time manifest.

    Args:
        experiment_session: The created Mesoscope-VR experiment session the manifest is written under.

    Returns:
        The same session, whose raw behavior-data directory now carries the camera manifest.
    """
    _write_manifest(
        experiment_session.raw_data.behavior_data_path,
        {_FACE_SOURCE_ID: _FACE_CAMERA, _BODY_SOURCE_ID: _BODY_CAMERA},
    )
    return experiment_session


@pytest.fixture
def write_frame_archive(write_log_archive: Callable[..., Path]) -> Callable[..., Path]:
    """Returns a writer that builds one camera's raw log archive holding the requested number of frame messages.

    Every camera frame message carries an empty payload, which is what the extraction binding counts as one acquired
    frame.

    Args:
        write_log_archive: The shared writer that serializes the DataLogger archive.

    Returns:
        A callable taking the raw behavior-data directory, the camera source identifier, and the frame count, and
        returning the written archive path.
    """

    def _write(directory: Path, source_id: int, frames: int = _RECORDING_FRAMES) -> Path:
        messages = [(1000 * (index + 1), b"") for index in range(frames)]
        return write_log_archive(directory.joinpath(f"{source_id}_log.npz"), source_id, messages)

    return _write


@pytest.fixture
def write_recording(write_grayscale_video: Callable[..., Path]) -> Callable[[SessionData, str], Path]:
    """Returns a writer that encodes one camera's recording under the name the analysis resolves it by.

    Args:
        write_grayscale_video: The shared writer that encodes the frame stack.

    Returns:
        A callable taking the loaded session and the colloquial camera name, and returning the written path.
    """

    def _write(session: SessionData, camera_name: str) -> Path:
        directory = session.raw_data.camera_data_path
        directory.mkdir(parents=True, exist_ok=True)
        return write_grayscale_video(directory.joinpath(f"{session.session_name}_{camera_name}.mp4"), _moving_frames())

    return _write


def _session_path(session: SessionData) -> Path:
    """Resolves the root session directory every pipeline entry point takes as its path argument.

    Args:
        session: The loaded session.

    Returns:
        The path to the root session directory.
    """
    return session.raw_data_path.parent


def _video_directory(session: SessionData) -> Path:
    """Resolves the processed video-data directory every pipeline job writes into.

    Args:
        session: The loaded session.

    Returns:
        The processed video-data directory path.
    """
    return session.processed_data.video_data_path


def _job_status(session: SessionData, job_name: str, specifier: str) -> ProcessingStatus | None:
    """Reads one job's recorded status out of the session's shared video tracker.

    Args:
        session: The loaded session whose video tracker is read.
        job_name: The name of the job to read.
        specifier: The job's specifier, which is a camera source identifier or an empty string.

    Returns:
        The recorded status, or None when the tracker does not hold the job.
    """
    tracker = ProcessingTracker(file_path=_video_directory(session).joinpath(ProcessingTrackers.VIDEO))
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
    job_state = tracker.snapshot().get(job_id)
    return job_state.status if job_state is not None else None


# Local mode, full runs


def test_unflagged_run_executes_every_stage(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
    write_recording: Callable[[SessionData, str], Path],
) -> None:
    """Verifies a run with no stage flag parses, publishes, tracks, and measures every registered camera."""
    behavior_directory = camera_session.raw_data.behavior_data_path
    write_frame_archive(behavior_directory, _FACE_SOURCE_ID)
    write_frame_archive(behavior_directory, _BODY_SOURCE_ID)
    # An archive belonging to a source the camera manifest does not register must be ignored rather than parsed.
    write_frame_archive(behavior_directory, 101)
    for camera_name in (_FACE_CAMERA, _BODY_CAMERA):
        write_recording(camera_session, camera_name)

    run_video_processing_pipeline(session_path=_session_path(camera_session), workers=1)

    video_directory = _video_directory(camera_session)
    for source_id, camera_name in ((_FACE_SOURCE_ID, _FACE_CAMERA), (_BODY_SOURCE_ID, _BODY_CAMERA)):
        parsed = video_directory.joinpath(f"camera_{source_id}_timestamps.feather")
        canonical = video_directory.joinpath(f"{camera_name}_timestamps.feather")
        assert pl.read_ipc(parsed).height == _RECORDING_FRAMES
        assert canonical.stat().st_ino == parsed.stat().st_ino
        assert pl.read_ipc(video_directory.joinpath(f"{camera_name}{MOTION_ENERGY_SUFFIX}")).height == _RECORDING_FRAMES
        assert _job_status(camera_session, TIMESTAMP_JOB_NAME, str(source_id)) == ProcessingStatus.SUCCEEDED
        assert _job_status(camera_session, ENERGY_JOB_NAME, str(source_id)) == ProcessingStatus.SUCCEEDED
    assert not video_directory.joinpath("camera_101_timestamps.feather").exists()
    assert _job_status(camera_session, RENAME_JOB_NAME, "") == ProcessingStatus.SUCCEEDED
    assert _job_status(camera_session, TRACKING_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


def test_multiple_workers_share_one_pool_across_the_run(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
    write_recording: Callable[[SessionData, str], Path],
) -> None:
    """Verifies a run given more than one worker still writes every output, sharing one pool across its jobs."""
    write_frame_archive(camera_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)
    write_recording(camera_session, _FACE_CAMERA)

    run_video_processing_pipeline(session_path=_session_path(camera_session), workers=2)

    video_directory = _video_directory(camera_session)
    assert pl.read_ipc(video_directory.joinpath(f"{_FACE_CAMERA}_timestamps.feather")).height == _RECORDING_FRAMES
    assert pl.read_ipc(video_directory.joinpath(f"{_FACE_CAMERA}{MOTION_ENERGY_SUFFIX}")).height == _RECORDING_FRAMES


def test_pose_tracking_writes_the_pupil_feather(
    camera_session: SessionData,
    write_dlc_predictions: Callable[..., Path],
) -> None:
    """Verifies the tracking job runs the acquisition system's donated function into the video-data directory."""
    reflection = np.empty((_RECORDING_FRAMES, 3), dtype=np.float64)
    reflection[:, 0], reflection[:, 1], reflection[:, 2] = 40.0, 30.0, 1.0
    pupil_ring = _ring(center=(50.0, 50.0), radius=8.0, count=len(_PUPIL_POINTS), frame_count=_RECORDING_FRAMES)
    eye_ring = _ring(center=(50.0, 50.0), radius=20.0, count=len(_EYE_POINTS), frame_count=_RECORDING_FRAMES)
    points: dict[str, NDArray[np.float64]] = {"reflection": reflection}
    points.update(dict(zip(_PUPIL_POINTS, pupil_ring, strict=True)))
    points.update(dict(zip(_EYE_POINTS, eye_ring, strict=True)))
    write_dlc_predictions(
        camera_session.raw_data.camera_data_path.joinpath("face_camera_eye_tracking_predictions.h5"), points
    )

    run_video_processing_pipeline(session_path=_session_path(camera_session), track=True, workers=1)

    pupil_frame = pl.read_ipc(_video_directory(camera_session).joinpath(f"{_FACE_CAMERA}_pupil.feather"))
    assert pupil_frame.height == _RECORDING_FRAMES
    assert _job_status(camera_session, TRACKING_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


# Local mode, stage selection


def test_timestamp_stage_honors_the_target_camera(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies naming one camera parses that camera alone and leaves the tracking and energy jobs unregistered."""
    behavior_directory = camera_session.raw_data.behavior_data_path
    write_frame_archive(behavior_directory, _FACE_SOURCE_ID)
    write_frame_archive(behavior_directory, _BODY_SOURCE_ID)

    run_video_processing_pipeline(
        session_path=_session_path(camera_session), timestamp=True, target_camera=_FACE_SOURCE_ID, workers=1
    )

    video_directory = _video_directory(camera_session)
    assert video_directory.joinpath(f"camera_{_FACE_SOURCE_ID}_timestamps.feather").is_file()
    assert not video_directory.joinpath(f"camera_{_BODY_SOURCE_ID}_timestamps.feather").exists()
    assert _job_status(camera_session, TIMESTAMP_JOB_NAME, str(_FACE_SOURCE_ID)) == ProcessingStatus.SUCCEEDED
    # A stage-scoped invocation registers only the jobs it runs, leaving its siblings out of the shared tracker.
    assert _job_status(camera_session, TIMESTAMP_JOB_NAME, str(_BODY_SOURCE_ID)) is None
    assert _job_status(camera_session, TRACKING_JOB_NAME, "") is None
    assert _job_status(camera_session, ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)) is None
    assert _job_status(camera_session, RENAME_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


def test_timestamp_stage_without_archives_errors(camera_session: SessionData) -> None:
    """Verifies the timestamp stage refuses to run when no registered camera archive is on disk."""
    with pytest.raises(ValueError, match="No registered camera log archives were"):
        run_video_processing_pipeline(session_path=_session_path(camera_session), timestamp=True, workers=1)


def test_timestamp_stage_rejects_a_target_camera_without_an_archive(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies naming a camera whose archive is absent errors rather than silently parsing nothing."""
    write_frame_archive(camera_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)

    with pytest.raises(ValueError, match=f"requested camera source ID {_BODY_SOURCE_ID}"):
        run_video_processing_pipeline(
            session_path=_session_path(camera_session), timestamp=True, target_camera=_BODY_SOURCE_ID, workers=1
        )


def test_energy_stage_rejects_an_unregistered_target_camera(camera_session: SessionData) -> None:
    """Verifies the energy stage refuses a camera the acquisition-time manifest does not register."""
    with pytest.raises(ValueError, match="It is not registered in the camera manifest"):
        run_video_processing_pipeline(
            session_path=_session_path(camera_session), energy=True, target_camera=999, workers=1
        )


def test_manifest_without_cameras_errors(experiment_session: SessionData) -> None:
    """Verifies an empty camera manifest stops the run instead of aligning an empty job universe."""
    _write_manifest(experiment_session.raw_data.behavior_data_path, {})

    with pytest.raises(ValueError, match="does not register any cameras"):
        run_video_processing_pipeline(session_path=_session_path(experiment_session), workers=1)


def test_missing_manifest_errors(experiment_session: SessionData) -> None:
    """Verifies a session with no camera manifest reports the missing file rather than an empty universe."""
    with pytest.raises(FileNotFoundError, match=f"No camera manifest \\('{CAMERA_MANIFEST_FILENAME}'\\)"):
        run_video_processing_pipeline(session_path=_session_path(experiment_session), workers=1)


# Remote mode


def test_remote_mode_runs_only_the_requested_timestamp_job(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies a job identifier selects exactly one job, ignoring the stage flags and the target camera."""
    behavior_directory = camera_session.raw_data.behavior_data_path
    write_frame_archive(behavior_directory, _FACE_SOURCE_ID)
    write_frame_archive(behavior_directory, _BODY_SOURCE_ID)
    job_id = ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=str(_BODY_SOURCE_ID))

    run_video_processing_pipeline(
        session_path=_session_path(camera_session), job_id=job_id, energy=True, target_camera=_FACE_SOURCE_ID, workers=1
    )

    video_directory = _video_directory(camera_session)
    assert video_directory.joinpath(f"camera_{_BODY_SOURCE_ID}_timestamps.feather").is_file()
    assert not video_directory.joinpath(f"camera_{_FACE_SOURCE_ID}_timestamps.feather").exists()
    assert _job_status(camera_session, TIMESTAMP_JOB_NAME, str(_BODY_SOURCE_ID)) == ProcessingStatus.SUCCEEDED
    # The stage flag and the target camera are ignored, so the energy job the flags name never reaches the tracker.
    assert _job_status(camera_session, ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)) is None


def test_remote_mode_runs_the_rename_job(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies the rename job is dispatchable on its own, publishing whichever parsed feathers are already present."""
    write_frame_archive(camera_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)
    parse_id = ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=str(_FACE_SOURCE_ID))
    run_video_processing_pipeline(session_path=_session_path(camera_session), job_id=parse_id, workers=1)

    rename_id = ProcessingTracker.generate_job_id(job_name=RENAME_JOB_NAME, specifier="")
    run_video_processing_pipeline(session_path=_session_path(camera_session), job_id=rename_id, workers=1)

    video_directory = _video_directory(camera_session)
    assert video_directory.joinpath(f"{_FACE_CAMERA}_timestamps.feather").is_file()
    assert not video_directory.joinpath(f"{_BODY_CAMERA}_timestamps.feather").exists()
    assert _job_status(camera_session, RENAME_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


def test_remote_mode_rejects_an_unknown_job_id(camera_session: SessionData) -> None:
    """Verifies an identifier outside the session's job universe errors and names the valid identifiers."""
    with pytest.raises(ValueError, match="does not match any camera processing job"):
        run_video_processing_pipeline(session_path=_session_path(camera_session), job_id="deadbeef", workers=1)


def test_remote_mode_rejects_a_timestamp_job_without_an_archive(camera_session: SessionData) -> None:
    """Verifies dispatching a parse job whose camera archive is absent reports the missing archive."""
    job_id = ProcessingTracker.generate_job_id(job_name=TIMESTAMP_JOB_NAME, specifier=str(_FACE_SOURCE_ID))

    with pytest.raises(FileNotFoundError, match="No raw log archive was discovered for"):
        run_video_processing_pipeline(session_path=_session_path(camera_session), job_id=job_id, workers=1)


# Canonical publication


def test_rename_job_skips_a_camera_without_a_parsed_feather(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies a camera whose parse job has not run is passed over rather than failing the rename job."""
    write_frame_archive(camera_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)

    run_video_processing_pipeline(session_path=_session_path(camera_session), timestamp=True, workers=1)

    video_directory = _video_directory(camera_session)
    assert video_directory.joinpath(f"{_FACE_CAMERA}_timestamps.feather").is_file()
    assert not video_directory.joinpath(f"{_BODY_CAMERA}_timestamps.feather").exists()
    assert _job_status(camera_session, RENAME_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


def test_rename_job_replaces_a_stale_canonical_feather(
    experiment_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies a canonical feather left by an earlier run is re-pointed at the freshly parsed feather."""
    _write_manifest(experiment_session.raw_data.behavior_data_path, {_FACE_SOURCE_ID: _FACE_CAMERA})
    write_frame_archive(experiment_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)
    video_directory = _video_directory(experiment_session)
    video_directory.mkdir(parents=True, exist_ok=True)
    stale_path = video_directory.joinpath(f"{_FACE_CAMERA}_timestamps.feather")
    pl.DataFrame({"frame_time_us": np.arange(3, dtype=np.uint64)}).write_ipc(file=stale_path)

    run_video_processing_pipeline(session_path=_session_path(experiment_session), timestamp=True, workers=1)

    parsed_path = video_directory.joinpath(f"camera_{_FACE_SOURCE_ID}_timestamps.feather")
    assert stale_path.stat().st_ino == parsed_path.stat().st_ino
    assert pl.read_ipc(stale_path).height == _RECORDING_FRAMES


def test_rename_job_preserves_a_feather_already_named_canonically(
    experiment_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies a manifest name that already equals the parsed filename leaves that feather untouched.

    Unlinking the canonical name in that case would destroy the parsed feather the job is meant to publish.
    """
    canonical_name = f"camera_{_FACE_SOURCE_ID}"
    _write_manifest(experiment_session.raw_data.behavior_data_path, {_FACE_SOURCE_ID: canonical_name})
    write_frame_archive(experiment_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)

    run_video_processing_pipeline(session_path=_session_path(experiment_session), timestamp=True, workers=1)

    parsed_path = _video_directory(experiment_session).joinpath(f"{canonical_name}_timestamps.feather")
    assert pl.read_ipc(parsed_path).height == _RECORDING_FRAMES
    assert _job_status(experiment_session, RENAME_JOB_NAME, "") == ProcessingStatus.SUCCEEDED


def test_rename_job_copies_when_hardlinking_is_unavailable(
    experiment_session: SessionData,
    write_frame_archive: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a filesystem that refuses hardlinks still publishes the canonical name, as an independent copy."""
    _write_manifest(experiment_session.raw_data.behavior_data_path, {_FACE_SOURCE_ID: _FACE_CAMERA})
    write_frame_archive(experiment_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)

    def _refuse(self: Path, target: Path) -> None:  # noqa: ARG001
        message = "Hardlinks are not supported on this filesystem."
        raise OSError(message)

    monkeypatch.setattr(Path, "hardlink_to", _refuse)
    run_video_processing_pipeline(session_path=_session_path(experiment_session), timestamp=True, workers=1)

    video_directory = _video_directory(experiment_session)
    parsed_path = video_directory.joinpath(f"camera_{_FACE_SOURCE_ID}_timestamps.feather")
    canonical_path = video_directory.joinpath(f"{_FACE_CAMERA}_timestamps.feather")
    assert canonical_path.stat().st_ino != parsed_path.stat().st_ino
    assert canonical_path.read_bytes() == parsed_path.read_bytes()


# Discovery and ordering


def test_discovery_reports_the_universe_and_the_possible_subset(
    camera_session: SessionData,
    write_frame_archive: Callable[..., Path],
) -> None:
    """Verifies discovery keeps every registered camera in the universe while gating parse jobs on their archives."""
    write_frame_archive(camera_session.raw_data.behavior_data_path, _FACE_SOURCE_ID)

    session, universe, possible = discover_video_jobs(session_path=_session_path(camera_session))

    assert session.session_name == camera_session.session_name
    assert universe == [
        (TIMESTAMP_JOB_NAME, str(_FACE_SOURCE_ID)),
        (TIMESTAMP_JOB_NAME, str(_BODY_SOURCE_ID)),
        (RENAME_JOB_NAME, ""),
        (TRACKING_JOB_NAME, ""),
        (ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)),
        (ENERGY_JOB_NAME, str(_BODY_SOURCE_ID)),
    ]
    assert possible == [
        (TIMESTAMP_JOB_NAME, str(_FACE_SOURCE_ID)),
        (RENAME_JOB_NAME, ""),
        (TRACKING_JOB_NAME, ""),
        (ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)),
        (ENERGY_JOB_NAME, str(_BODY_SOURCE_ID)),
    ]


def test_discovery_omits_the_rename_job_without_any_archive(camera_session: SessionData) -> None:
    """Verifies the rename job leaves the possible subset when no camera can produce a parsed feather."""
    _session, _universe, possible = discover_video_jobs(session_path=_session_path(camera_session))

    assert possible == [
        (TRACKING_JOB_NAME, ""),
        (ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)),
        (ENERGY_JOB_NAME, str(_BODY_SOURCE_ID)),
    ]


def test_discovery_rejects_a_manifest_without_cameras(experiment_session: SessionData) -> None:
    """Verifies discovery on an empty manifest errors rather than returning an empty universe."""
    _write_manifest(experiment_session.raw_data.behavior_data_path, {})

    with pytest.raises(ValueError, match="does not register any cameras"):
        discover_video_jobs(session_path=_session_path(experiment_session))


def test_prerequisites_order_the_rename_job_after_every_parse_job(camera_session: SessionData) -> None:
    """Verifies only the rename job carries prerequisites, and they are exactly the parse jobs in the given set."""
    session, universe, _possible = discover_video_jobs(session_path=_session_path(camera_session))

    prerequisites = video_job_prerequisites(session=session, universe=universe)

    assert prerequisites[RENAME_JOB_NAME, ""] == (
        (TIMESTAMP_JOB_NAME, str(_FACE_SOURCE_ID)),
        (TIMESTAMP_JOB_NAME, str(_BODY_SOURCE_ID)),
    )
    assert prerequisites[TRACKING_JOB_NAME, ""] == ()
    assert prerequisites[ENERGY_JOB_NAME, str(_FACE_SOURCE_ID)] == ()
    assert prerequisites[TIMESTAMP_JOB_NAME, str(_FACE_SOURCE_ID)] == ()
    assert set(prerequisites) == set(universe)


# Archive discovery and dispatch guards


def test_archive_discovery_tolerates_a_missing_directory(tmp_path: Path) -> None:
    """Verifies discovery over a behavior directory that was never created reports no archives."""
    assert _find_camera_logs(data_directory=tmp_path.joinpath("absent")) == []


def test_archive_discovery_sorts_the_archives_naturally(tmp_path: Path, write_log_archive: Callable[..., Path]) -> None:
    """Verifies the discovered archives are ordered by their numeric source identifier rather than lexically."""
    for source_id in (2, 10, 1):
        write_log_archive(tmp_path.joinpath(f"{source_id}_log.npz"), source_id, [(1, b"")])

    assert [path.name for path in _find_camera_logs(data_directory=tmp_path)] == [
        "1_log.npz",
        "2_log.npz",
        "10_log.npz",
    ]


@pytest.mark.parametrize("filename", ["sync_log.npz", "51_frames.npz", "51_log_extra.npz"])
def test_source_id_extraction_rejects_a_foreign_archive_name(tmp_path: Path, filename: str) -> None:
    """Verifies an archive whose name breaks the convention errors instead of yielding a wrong source identifier."""
    with pytest.raises(ValueError, match="does not follow the expected"):
        _extract_camera_source_id(log_path=tmp_path.joinpath(filename))


def test_dispatch_rejects_an_unknown_job_name(camera_session: SessionData) -> None:
    """Verifies a job name outside the pipeline's four kinds fails rather than silently completing."""
    video_directory = _video_directory(camera_session)
    video_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=video_directory.joinpath(ProcessingTrackers.VIDEO))

    with pytest.raises(ValueError, match="does not identify any pipeline job"):
        _dispatch_job(
            job_name="unregistered_job",
            specifier="",
            session=camera_session,
            log_paths={},
            camera_names={_FACE_SOURCE_ID: _FACE_CAMERA},
            video_data_directory=video_directory,
            tracker=tracker,
            workers=1,
            display_progress=False,
            executor=None,
        )
