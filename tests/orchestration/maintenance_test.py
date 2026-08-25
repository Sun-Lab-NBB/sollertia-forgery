"""Contains tests for the maintenance operations that return a processing unit to an earlier state, by record or by
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import pytest
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.shared_assets import ProcessingPipelines, resolve_session_tracker_path
from sollertia_forgery.orchestration.maintenance import (
    _resolve_path_size,
    reset_tracked_jobs,
    clean_pipeline_output,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_VIDEO_JOBS: list[tuple[str, str]] = [("motion_energy", "face_camera"), ("camera_timestamps", "face_camera")]
"""The video jobs against which every video tracker written by these tests is aligned."""


def job_identifier(job: tuple[str, str]) -> str:
    """Resolves the tracker identifier of one named job.

    Args:
        job: The job name and specifier pair under which the tracker records the job.

    Returns:
        The hexadecimal identifier by which the tracker keys the job.
    """
    return ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])


def session_root(session: SessionData) -> Path:
    """Resolves the session's root directory, which every maintenance call takes as its unit path.

    Args:
        session: The loaded session whose root to resolve.

    Returns:
        The absolute path to the session directory.
    """
    return session.raw_data_path.parent


def tracker_statuses(path: Path) -> dict[str, ProcessingStatus]:
    """Reads the status each job of one tracker records.

    Args:
        path: The path to the processing tracker to read.

    Returns:
        The recorded status of every job, keyed by job identifier.
    """
    return {job_id: job_state.status for job_id, job_state in ProcessingTracker(file_path=path).snapshot().items()}


@pytest.fixture
def video_tracker(
    experiment_session: SessionData, write_tracker: Callable[..., ProcessingTracker]
) -> ProcessingTracker:
    """Writes the session's video tracker holding both video jobs in the succeeded state.

    Args:
        experiment_session: The session for which the tracker is written.
        write_tracker: The builder that writes the tracker.

    Returns:
        The written tracker.
    """
    return write_tracker(
        resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO),
        _VIDEO_JOBS,
        succeeded=_VIDEO_JOBS,
    )


def test_a_pipeline_outside_the_dispatch_table_resets_nothing(experiment_session: SessionData) -> None:
    """Verifies that a caller naming an unsupported pipeline gets an empty result rather than a partial reset of
    something else.
    """
    assert reset_tracked_jobs(pipeline="analysis", unit_paths=[session_root(experiment_session)]) == []


def test_naming_no_identifier_returns_the_whole_unit_to_a_clean_slate(
    experiment_session: SessionData, video_tracker: ProcessingTracker
) -> None:
    """Verifies that resetting without naming a job is how a caller returns one unit's entire pipeline to the scheduled
    state.
    """
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session_root(experiment_session)])

    assert sorted(reset) == sorted(job_identifier(job) for job in _VIDEO_JOBS)
    assert set(tracker_statuses(tracker_path).values()) == {ProcessingStatus.SCHEDULED}
    assert video_tracker.file_path == tracker_path


def test_only_the_named_identifiers_are_reset(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies that naming one job leaves every other record the unit holds exactly as the run left it."""
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)
    target = job_identifier(job=_VIDEO_JOBS[0])

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session_root(experiment_session)], job_ids=[target])

    assert reset == [target]
    statuses = tracker_statuses(tracker_path)
    assert statuses[target] is ProcessingStatus.SCHEDULED
    assert statuses[job_identifier(job=_VIDEO_JOBS[1])] is ProcessingStatus.SUCCEEDED


def test_an_identifier_the_unit_does_not_track_is_dropped(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies one call carries a whole batch's identifiers, so a unit resets its own share and ignores the rest."""
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    reset = reset_tracked_jobs(
        pipeline="video", unit_paths=[session_root(experiment_session)], job_ids=["0123456789abcdef"]
    )

    assert reset == []
    assert set(tracker_statuses(tracker_path).values()) == {ProcessingStatus.SUCCEEDED}


def test_a_unit_holding_no_tracker_is_skipped(experiment_session: SessionData) -> None:
    """Verifies a pipeline that never ran for the unit has no record to clear, so the call reports nothing for it."""
    assert reset_tracked_jobs(pipeline="video", unit_paths=[session_root(experiment_session)]) == []


def test_a_unit_that_cannot_be_loaded_leaves_its_siblings_reset(
    tmp_path: Path,
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the healthy unit carries records to clear.
) -> None:
    """Verifies that one unresolvable unit is skipped rather than abandoning the reset of the others."""
    unresolvable = tmp_path.joinpath("not_a_session")
    unresolvable.mkdir()

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[unresolvable, session_root(experiment_session)])

    assert sorted(reset) == sorted(job_identifier(job) for job in _VIDEO_JOBS)


def test_a_pipeline_outside_the_dispatch_table_removes_nothing(experiment_session: SessionData) -> None:
    """Verifies that an unsupported pipeline identifier never reaches a unit's files at all."""
    assert clean_pipeline_output(pipeline="analysis", unit_paths=[session_root(experiment_session)]) == []


def test_a_pipeline_that_owns_a_directory_removes_it_alongside_its_tracker(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies a later preparation must rediscover every job from the acquired data, so the whole owned tree goes."""
    output_directory = experiment_session.processed_data.video_data_path
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)
    output_directory.joinpath("motion_energy.feather").write_bytes(b"0123456789")

    removed = clean_pipeline_output(pipeline="video", unit_paths=[session_root(experiment_session)])

    assert [entry["path"] for entry in removed] == [str(tracker_path), str(output_directory)]
    assert removed[1]["removed_bytes"] == 10
    assert not output_directory.exists()
    assert not tracker_path.with_suffix(tracker_path.suffix + ".lock").exists()


def test_a_pipeline_that_owns_no_directory_removes_its_tracker_alone(
    experiment_session: SessionData, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that the checksum pipeline verifies the acquired data in place, so cleaning it must leave that data
    untouched.
    """
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.CHECKSUM)
    tracker = write_tracker(tracker_path, [("checksum", "")])
    lock_path = Path(tracker.lock_path)
    descriptor_path = experiment_session.raw_data.session_descriptor_path
    # Another pipeline's output directory sits beside the checksum tracker, so a cleanup scoped to checksum has to
    # leave it standing rather than take the directory some other pipeline owns.
    runtime_directory = experiment_session.processed_data.runtime_data_path
    runtime_directory.mkdir(parents=True, exist_ok=True)
    runtime_output = runtime_directory.joinpath("runtime_data.feather")
    runtime_output.write_bytes(b"runtime")

    assert lock_path.is_file(), "the tracker did not leave the lock file the cleanup is expected to remove"

    removed = clean_pipeline_output(pipeline="checksum", unit_paths=[session_root(experiment_session)])

    assert [entry["path"] for entry in removed] == [str(tracker_path)]
    assert not tracker_path.exists()
    # The lock is bookkeeping beside the tracker, so it goes with it rather than outliving the record it guarded.
    assert not lock_path.exists()
    assert descriptor_path.is_file(), "cleaning the checksum pipeline removed acquired data"
    assert runtime_output.is_file(), "cleaning the checksum pipeline removed another pipeline's output"


def test_a_unit_with_nothing_recorded_removes_nothing(experiment_session: SessionData) -> None:
    """Verifies that a pipeline that never ran leaves neither a tracker nor an output directory behind to remove."""
    assert clean_pipeline_output(pipeline="video", unit_paths=[session_root(experiment_session)]) == []


def test_a_unit_that_cannot_be_loaded_leaves_its_siblings_cleaned(
    tmp_path: Path,
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the healthy unit carries output to remove.
) -> None:
    """Verifies that one unresolvable unit is skipped rather than abandoning the cleanup of the others."""
    unresolvable = tmp_path.joinpath("not_a_session")
    unresolvable.mkdir()
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    removed = clean_pipeline_output(pipeline="video", unit_paths=[unresolvable, session_root(experiment_session)])

    assert [entry["path"] for entry in removed] == [
        str(tracker_path),
        str(experiment_session.processed_data.video_data_path),
    ]


def test_a_directory_is_measured_across_its_whole_tree(tmp_path: Path) -> None:
    """Verifies that the reported figure is what the removal frees, so it has to count nested files rather than the top
    level.
    """
    root = tmp_path.joinpath("tree")
    root.joinpath("nested").mkdir(parents=True)
    root.joinpath("top.bin").write_bytes(b"abcd")
    root.joinpath("nested", "deep.bin").write_bytes(b"efghij")

    assert _resolve_path_size(path=root) == 10


def test_a_file_is_measured_by_its_own_size(tmp_path: Path) -> None:
    """Verifies that a tracker is a single file, so its measurement never walks a tree that does not exist."""
    target = tmp_path.joinpath("tracker.yaml")
    target.write_bytes(b"abcde")

    assert _resolve_path_size(path=target) == 5
