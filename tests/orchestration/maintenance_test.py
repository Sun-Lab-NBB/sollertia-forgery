"""Contains tests for the maintenance operations that return a processing unit to an earlier state, by record or by
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import pytest
from sollertia_shared_assets import (
    DatasetData,
    SessionTypes,
    DatasetSession,
    AcquisitionSystems,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import forging_tracker_path
from sollertia_forgery.shared_assets import ProcessingPipelines, resolve_session_tracker_path
from sollertia_forgery.orchestration.maintenance import (
    _resolve_path_size,
    reset_tracked_jobs,
    clean_pipeline_output,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sollertia_shared_assets import ProjectData, SessionData

_VIDEO_JOBS: list[tuple[str, str]] = [("motion_energy", "face_camera"), ("camera_timestamps", "face_camera")]
"""The video jobs against which every video tracker written by these tests is aligned."""

_FORGING_JOBS: list[tuple[str, str]] = [("multiday_discovery", "305"), ("session_data_assembly", "a_session")]
"""The forging jobs against which every forging tracker written by these tests is aligned."""

_DATASET_NAME: str = "TestDataset"
"""The name of the dataset these tests forge, carrying uppercase so the lowercasing of the cindra output directory
stays observable."""

_SIBLING_DATASET_DIRECTORY: str = "305_otherdataset"
"""The cross-recording directory of a second dataset the same session belongs to, which the cleanup of this dataset
has to leave standing."""

_MULTI_RECORDING_DIRECTORY: str = "multi_recording"
"""The directory under a session's cindra output holding one subdirectory per dataset that tracks the session across
recordings."""


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


@pytest.fixture
def forged_dataset(project: ProjectData, experiment_session: SessionData) -> DatasetData:
    """Creates a dataset naming the session, which gives the forging cleanup a hierarchy and a source session to reach.

    Args:
        project: The project under which the dataset hierarchy is created.
        experiment_session: The session the dataset names as its only source.

    Returns:
        The created dataset.
    """
    return DatasetData.create(
        name=_DATASET_NAME,
        project=project.project_name,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=(DatasetSession(session=experiment_session.session_name, animal=str(experiment_session.animal_id)),),
        datasets_root=project.path,
        column_descriptions={},
    )


def cross_recording_root(session: SessionData) -> Path:
    """Resolves the directory holding one subdirectory per dataset that tracks the session across recordings.

    Args:
        session: The session whose cross-recording root to resolve.

    Returns:
        The path to the session's multi-recording root inside its cindra output.
    """
    return session.processed_data.cindra_data_path.joinpath(_MULTI_RECORDING_DIRECTORY)


def test_a_pipeline_outside_the_dispatch_table_resets_nothing(experiment_session: SessionData) -> None:
    """Verifies that a caller naming an unsupported pipeline gets an empty result rather than a partial reset of
    something else.
    """
    assert (
        reset_tracked_jobs(pipeline="unregistered_pipeline", unit_paths=[session_root(session=experiment_session)])
        == []
    )


def test_naming_no_identifier_returns_the_whole_unit_to_a_clean_slate(
    experiment_session: SessionData, video_tracker: ProcessingTracker
) -> None:
    """Verifies that resetting without naming a job is how a caller returns one unit's entire pipeline to the scheduled
    state.
    """
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session_root(session=experiment_session)])

    assert sorted(reset) == sorted(job_identifier(job=job) for job in _VIDEO_JOBS)
    assert set(tracker_statuses(path=tracker_path).values()) == {ProcessingStatus.SCHEDULED}
    assert video_tracker.file_path == tracker_path


def test_only_the_named_identifiers_are_reset(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies that naming one job leaves every other record the unit holds exactly as the run left it."""
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)
    target = job_identifier(job=_VIDEO_JOBS[0])

    reset = reset_tracked_jobs(
        pipeline="video", unit_paths=[session_root(session=experiment_session)], job_ids=[target]
    )

    assert reset == [target]
    statuses = tracker_statuses(path=tracker_path)
    assert statuses[target] is ProcessingStatus.SCHEDULED
    assert statuses[job_identifier(job=_VIDEO_JOBS[1])] is ProcessingStatus.SUCCEEDED


def test_an_identifier_the_unit_does_not_track_is_dropped(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies one call carries a whole batch's identifiers, so a unit resets its own share and ignores the rest."""
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    reset = reset_tracked_jobs(
        pipeline="video", unit_paths=[session_root(session=experiment_session)], job_ids=["0123456789abcdef"]
    )

    assert reset == []
    assert set(tracker_statuses(path=tracker_path).values()) == {ProcessingStatus.SUCCEEDED}


def test_a_unit_holding_no_tracker_is_skipped(experiment_session: SessionData) -> None:
    """Verifies a pipeline that never ran for the unit has no record to clear, so the call reports nothing for it."""
    assert reset_tracked_jobs(pipeline="video", unit_paths=[session_root(session=experiment_session)]) == []


def test_a_unit_that_cannot_be_loaded_leaves_its_siblings_reset(
    tmp_path: Path,
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the healthy unit carries records to clear.
) -> None:
    """Verifies that one unresolvable unit is skipped rather than abandoning the reset of the others."""
    unresolvable = tmp_path.joinpath("not_a_session")
    unresolvable.mkdir()

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[unresolvable, session_root(session=experiment_session)])

    assert sorted(reset) == sorted(job_identifier(job=job) for job in _VIDEO_JOBS)


def test_a_pipeline_outside_the_dispatch_table_removes_nothing(experiment_session: SessionData) -> None:
    """Verifies that an unsupported pipeline identifier never reaches a unit's files at all."""
    assert (
        clean_pipeline_output(pipeline="unregistered_pipeline", unit_paths=[session_root(session=experiment_session)])
        == []
    )


def test_a_pipeline_that_owns_a_directory_removes_it_alongside_its_tracker(
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the tracker exists on disk.
) -> None:
    """Verifies a later preparation must rediscover every job from the acquired data, so the whole owned tree goes."""
    output_directory = experiment_session.processed_data.video_data_path
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)
    output_directory.joinpath("motion_energy.feather").write_bytes(b"0123456789")

    removed = clean_pipeline_output(pipeline="video", unit_paths=[session_root(session=experiment_session)])

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

    # A tracker leaves its lock file behind on POSIX and removes it as it releases the lock on Windows, so the
    # cleanup is handed one to remove on either platform.
    lock_path.touch()

    removed = clean_pipeline_output(pipeline="checksum", unit_paths=[session_root(session=experiment_session)])

    assert [entry["path"] for entry in removed] == [str(tracker_path)]
    assert not tracker_path.exists()
    # The lock is bookkeeping beside the tracker, so it goes with it rather than outliving the record it guarded.
    assert not lock_path.exists()
    assert descriptor_path.is_file(), "cleaning the checksum pipeline removed acquired data"
    assert runtime_output.is_file(), "cleaning the checksum pipeline removed another pipeline's output"


def test_a_unit_with_nothing_recorded_removes_nothing(experiment_session: SessionData) -> None:
    """Verifies that a pipeline that never ran leaves neither a tracker nor an output directory behind to remove."""
    assert clean_pipeline_output(pipeline="video", unit_paths=[session_root(session=experiment_session)]) == []


def test_a_unit_that_cannot_be_loaded_leaves_its_siblings_cleaned(
    tmp_path: Path,
    experiment_session: SessionData,
    video_tracker: ProcessingTracker,  # Requested so the healthy unit carries output to remove.
) -> None:
    """Verifies that one unresolvable unit is skipped rather than abandoning the cleanup of the others."""
    unresolvable = tmp_path.joinpath("not_a_session")
    unresolvable.mkdir()
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO)

    removed = clean_pipeline_output(
        pipeline="video", unit_paths=[unresolvable, session_root(session=experiment_session)]
    )

    assert [entry["path"] for entry in removed] == [
        str(tracker_path),
        str(experiment_session.processed_data.video_data_path),
    ]


def test_cleaning_a_dataset_removes_the_cross_recording_output_it_owns_in_each_source_session(
    forged_dataset: DatasetData,
    experiment_session: SessionData,
    write_tracker: Callable[..., ProcessingTracker],
) -> None:
    """Verifies that the output the cross-recording stages left in a source session goes with the dataset that
    registered it, while everything that session holds for another dataset stays.
    """
    dataset_root = forged_dataset.dataset_data_path.parent
    tracker_path = forging_tracker_path(dataset=forged_dataset)
    write_tracker(tracker_path, _FORGING_JOBS, succeeded=_FORGING_JOBS)
    owned = cross_recording_root(session=experiment_session).joinpath(
        f"{experiment_session.animal_id}_{_DATASET_NAME}".lower()
    )
    owned.mkdir(parents=True)
    owned.joinpath("cell_registration.npz").write_bytes(b"0123456789")
    sibling = cross_recording_root(session=experiment_session).joinpath(_SIBLING_DATASET_DIRECTORY)
    sibling.mkdir(parents=True)
    single_recording = experiment_session.processed_data.cindra_data_path.joinpath("plane_0")
    single_recording.mkdir(parents=True)

    removed = clean_pipeline_output(pipeline="forging", unit_paths=[dataset_root])

    assert [entry["path"] for entry in removed] == [str(tracker_path), str(dataset_root), str(owned)]
    assert removed[2]["removed_bytes"] == 10
    assert not owned.exists()
    assert sibling.is_dir(), "cleaning the dataset removed the cross-recording output of another dataset"
    assert single_recording.is_dir(), "cleaning the dataset removed the session's single-recording output"


def test_a_source_session_holding_no_cross_recording_output_reports_nothing_for_it(
    forged_dataset: DatasetData, experiment_session: SessionData
) -> None:
    """Verifies that a dataset whose cross-recording stages never ran reports the dataset tree alone, leaving the
    source session's own output untouched.
    """
    dataset_root = forged_dataset.dataset_data_path.parent
    single_recording = experiment_session.processed_data.cindra_data_path
    single_recording.mkdir(parents=True, exist_ok=True)

    removed = clean_pipeline_output(pipeline="forging", unit_paths=[dataset_root])

    assert [entry["path"] for entry in removed] == [str(dataset_root)]
    assert single_recording.is_dir()


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
