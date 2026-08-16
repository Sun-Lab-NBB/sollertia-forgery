"""Tests the project manifest artifact: the walk that generates it and the query surface that reads it back."""

from __future__ import annotations

import time
import shutil
from typing import TYPE_CHECKING, Any
from datetime import UTC, datetime
import threading

import polars as pl
import pytest
from filelock import FileLock
from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionTypes, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker
from sollertia_shared_assets.registries import DESCRIPTOR_REGISTRY

from sollertia_forgery.managing import (
    MANIFEST_JOB_NAME,
    ProjectManifest,
    project_jobs_path,
    project_manifest_path,
    generate_project_manifest,
)
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.managing.manifest import (
    PIPELINE_STATUS_COLUMNS,
    PROJECT_MANIFEST_SCHEMA,
    _assert_status_column_coverage,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

LOCK_POLL_TIMEOUT_S: float = 30.0
"""The longest a test waits for the generation thread to reach the manifest lock before giving up on it."""


@pytest.fixture
def reported_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects every message the shared console is asked to echo, so a reported outcome is assertable.

    Args:
        monkeypatch: The fixture used to replace the console's echo method for the duration of one test.

    Returns:
        The list the recorder appends each echoed message to, in the order they were emitted.
    """
    messages: list[str] = []
    monkeypatch.setattr(console, "echo", lambda message, **_keywords: messages.append(message))
    return messages


@pytest.fixture
def manifest(project_manifest: Path) -> ProjectManifest:
    """Opens the generated project manifest through the reader every consumer uses.

    Args:
        project_manifest: The path to the generated manifest artifact.

    Returns:
        The manifest reader bound to the generated artifact.
    """
    return ProjectManifest(manifest_file=project_manifest)


def read_manifest(project_root: Path) -> pl.DataFrame:
    """Reads the manifest artifact of one project into a frame.

    Args:
        project_root: The project whose manifest to read.

    Returns:
        The manifest contents, one row per recorded session.
    """
    return pl.read_ipc(source=project_manifest_path(project_directory=project_root), memory_map=True)


# Tests for the generation walk


def test_generation_records_one_row_per_session_with_its_pipeline_state(
    project_root: Path,
    experiment_session: SessionData,
    training_session: SessionData,
    mark_session_processed: Callable[[SessionData], None],
) -> None:
    """A processed session reports every pipeline as done and an untouched session reports every one as not done."""
    mark_session_processed(experiment_session)

    generate_project_manifest(project_directory=project_root)

    frame = read_manifest(project_root=project_root)
    assert dict(frame.schema) == PROJECT_MANIFEST_SCHEMA
    # The rows are ordered by animal, so the experiment animal precedes the training animal.
    assert frame.get_column("animal").to_list() == [305, 321]

    processed = frame.filter(pl.col("session") == experiment_session.session_name).to_dicts()[0]
    untouched = frame.filter(pl.col("session") == training_session.session_name).to_dicts()[0]

    assert processed["session_path"] == f"305/{experiment_session.session_name}"
    assert processed["type"] == str(SessionTypes.MESOSCOPE_EXPERIMENT)
    assert processed["system"] == experiment_session.acquisition_system
    assert processed["notes"] == "A synthetic session."
    assert processed["complete"] == 1
    assert [processed[column] for column in PIPELINE_STATUS_COLUMNS.values()] == [1, 1, 1, 1, 1]
    assert [untouched[column] for column in PIPELINE_STATUS_COLUMNS.values()] == [0, 0, 0, 0, 0]


def test_the_recorded_date_is_the_session_name_read_as_utc(project_root: Path, training_session: SessionData) -> None:
    """The session name is a UTC acquisition timestamp, so the date column reproduces it to the microsecond."""
    generate_project_manifest(project_directory=project_root)

    components = [int(part) for part in training_session.session_name.split("-")]
    recorded = read_manifest(project_root=project_root).get_column("date").to_list()[0]

    assert recorded == datetime(*components, tzinfo=UTC)


def test_generation_marks_its_own_job_as_succeeded(project_root: Path, training_session: SessionData) -> None:
    """The manifest run is itself a tracked job, recorded against the project rather than a session."""
    generate_project_manifest(project_directory=project_root)

    jobs = ProcessingTracker(file_path=project_root.joinpath(ProcessingTrackers.MANIFEST)).snapshot()
    state = next(iter(jobs.values()))

    assert state.job_name == MANIFEST_JOB_NAME
    assert state.specifier == project_root.stem
    assert state.status.name == "SUCCEEDED"


def test_generation_writes_the_job_artifact_beside_the_manifest(
    project_root: Path,
    experiment_session: SessionData,
    mark_session_processed: Callable[[SessionData], None],
) -> None:
    """Both artifacts are written under one lock, so a reader never sees one refreshed without the other."""
    mark_session_processed(experiment_session)

    generate_project_manifest(project_directory=project_root)

    jobs = pl.read_ipc(source=project_jobs_path(project_directory=project_root), memory_map=True)
    assert jobs.height == len(PIPELINE_STATUS_COLUMNS)
    assert set(jobs.get_column("pipeline").to_list()) == {pipeline.value for pipeline in PIPELINE_STATUS_COLUMNS}
    assert jobs.get_column("session").unique().to_list() == [experiment_session.session_name]
    assert jobs.get_column("status").unique().to_list() == ["SUCCEEDED"]


def test_generation_announces_its_start_and_its_completion(
    project_root: Path,
    training_session: SessionData,
    reported_messages: list[str],
) -> None:
    """Requesting progress brackets the run with a preamble and a completion message naming the project."""
    generate_project_manifest(project_directory=project_root, display_progress=True)

    assert reported_messages == [
        f"Generating the project manifest for the '{project_root.stem}' project...",
        f"Project '{project_root.stem}' manifest: Generated.",
    ]


def test_generating_for_an_absent_project_reports_the_missing_directory(tmp_path: Path) -> None:
    """A project path that names nothing fails before any discovery work is attempted."""
    with pytest.raises(FileNotFoundError, match="The specified project directory does not"):
        generate_project_manifest(project_directory=tmp_path.joinpath("NoSuchProject"))


def test_generating_for_a_project_without_sessions_reports_the_requirement(project_root: Path) -> None:
    """A created but unused project holds nothing to snapshot, which is a hard error rather than an empty manifest."""
    with pytest.raises(FileNotFoundError, match="The project directory does not contain any"):
        generate_project_manifest(project_directory=project_root)


def test_a_session_missing_its_descriptor_fails_the_manifest_job(
    project_root: Path, training_session: SessionData
) -> None:
    """The walk records the fault on the manifest tracker before it propagates, so the aborted run is visible."""
    training_session.raw_data.session_descriptor_path.unlink()

    with pytest.raises(FileNotFoundError):
        generate_project_manifest(project_directory=project_root)

    jobs = ProcessingTracker(file_path=project_root.joinpath(ProcessingTrackers.MANIFEST)).snapshot()
    state = next(iter(jobs.values()))

    assert state.status.name == "FAILED"
    assert state.error_message is not None


def test_a_session_type_without_a_descriptor_names_the_supported_types(
    project_root: Path,
    training_session: SessionData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session type the descriptor registry does not cover cannot be snapshotted, so the run names what is."""
    monkeypatch.delitem(DESCRIPTOR_REGISTRY, SessionTypes.RUN_TRAINING)

    with pytest.raises(ValueError, match="An unsupported session type 'run training' was"):
        generate_project_manifest(project_directory=project_root)


def test_a_session_emptied_after_discovery_is_left_out_of_the_manifest(
    project_root: Path,
    experiment_session: SessionData,
    training_session: SessionData,
) -> None:
    """The walk re-reads each session, so one whose acquired data went away contributes no row.

    Discovery materializes the session list before the manifest lock is taken. Holding that lock from the test parks
    the run between the two, which is the only window in which a session can lose its raw data mid-generation.
    """
    manifest_path = project_manifest_path(project_directory=project_root)
    outer_lock = FileLock(str(manifest_path) + ".lock")
    tracker_path = project_root.joinpath(ProcessingTrackers.MANIFEST)
    failures: list[BaseException] = []

    def _generate() -> None:
        """Runs one manifest generation, keeping whatever it raised for the test to re-report."""
        try:
            generate_project_manifest(project_directory=project_root)
        except BaseException as exception:
            failures.append(exception)

    outer_lock.acquire()
    worker = threading.Thread(target=_generate)
    worker.start()
    try:
        # The manifest tracker is registered after the session list is built and before the lock is taken, so its
        # appearance marks the point past which emptying a session is invisible to discovery.
        deadline = time.monotonic() + LOCK_POLL_TIMEOUT_S
        while not tracker_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tracker_path.exists()
        shutil.rmtree(training_session.raw_data_path)
    finally:
        outer_lock.release()
    worker.join(timeout=LOCK_POLL_TIMEOUT_S)

    assert not failures
    assert read_manifest(project_root=project_root).get_column("session").to_list() == [experiment_session.session_name]


def test_every_session_pipeline_declares_a_status_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """The import-time check names the pipelines that disagree, so a new pipeline fails loudly rather than silently."""
    monkeypatch.delitem(PIPELINE_STATUS_COLUMNS, ProcessingPipelines.VIDEO)

    with pytest.raises(RuntimeError, match=r"the sets differ by \['video'\]"):
        _assert_status_column_coverage()


# Tests for the manifest query surface


def test_the_reader_reports_how_many_sessions_it_holds(manifest: ProjectManifest) -> None:
    """The representation is the row count, which is what a caller checks a loaded manifest against."""
    assert repr(manifest) == "ProjectManifest(sessions=2)"


def test_the_reader_exposes_its_frame_and_its_animals(manifest: ProjectManifest) -> None:
    """The frame is the stored artifact and the animals are its sorted unique subjects."""
    assert manifest.data.height == 2
    assert manifest.animals == (305, 321)


def test_every_session_is_listed_for_the_whole_project(
    manifest: ProjectManifest, experiment_session: SessionData, training_session: SessionData
) -> None:
    """An unfiltered listing spans every animal, ordered by session name."""
    assert manifest.get_sessions() == tuple(sorted((experiment_session.session_name, training_session.session_name)))


def test_an_animal_filter_narrows_the_listing_to_its_sessions(
    manifest: ProjectManifest, experiment_session: SessionData
) -> None:
    """Naming an animal returns only the sessions that animal participated in."""
    assert manifest.get_sessions(animal=305) == (experiment_session.session_name,)


def test_incomplete_sessions_are_excluded_unless_they_are_asked_for(
    project_root: Path,
    training_session: SessionData,
    session_factory: Callable[..., SessionData],
) -> None:
    """The completeness filter is what separates the sessions worth processing from the aborted ones."""
    aborted = session_factory(animal_id="321", session_type=SessionTypes.LICK_TRAINING, incomplete=True)
    generate_project_manifest(project_directory=project_root)
    reader = ProjectManifest(manifest_file=project_manifest_path(project_directory=project_root))

    assert reader.get_sessions() == (training_session.session_name,)
    assert reader.get_sessions(exclude_incomplete=False) == tuple(
        sorted((training_session.session_name, aborted.session_name))
    )


def test_listing_an_unknown_animal_names_the_available_ones(manifest: ProjectManifest) -> None:
    """A mistyped animal reports the project's roster rather than returning an empty tuple."""
    with pytest.raises(ValueError, match="Unable to filter sessions using animal ID '999'"):
        manifest.get_sessions(animal=999)


def test_a_session_row_carries_every_manifest_column(
    manifest: ProjectManifest, experiment_session: SessionData
) -> None:
    """The per-session view is the stored row, so a caller reads acquisition and processing state from one place."""
    row = manifest.get_session_data(session=experiment_session.session_name)

    assert row.height == 1
    assert row.columns == list(PROJECT_MANIFEST_SCHEMA)
    assert row.select("animal").item() == 305


def test_a_session_resolves_to_the_animal_that_recorded_it(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """The manifest is the lookup that maps a session name back to its subject."""
    assert manifest.get_animal_for_session(session=training_session.session_name) == 321


def test_an_unknown_session_lookup_names_the_available_sessions(manifest: ProjectManifest) -> None:
    """A session the manifest does not hold reports what it does hold, including the incomplete ones."""
    with pytest.raises(ValueError, match="Unable to look up the participating animal using session ID 'absent'"):
        manifest.get_animal_for_session(session="absent")


def test_a_session_resolves_to_the_system_that_acquired_it(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """The acquisition system decides which pipelines a session supports, so it is queryable on its own."""
    assert manifest.get_system_for_session(session=training_session.session_name) == training_session.acquisition_system


def test_an_unknown_system_lookup_names_the_available_sessions(manifest: ProjectManifest) -> None:
    """The system lookup fails the same way the animal lookup does, so both report the same roster."""
    with pytest.raises(ValueError, match="Unable to look up the acquisition system using session ID 'absent'"):
        manifest.get_system_for_session(session="absent")


def test_the_summary_counts_what_each_pipeline_finished(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """One processed session beside one untouched session gives every pipeline the same one-and-one split."""
    summary = manifest.summarize()

    assert summary["total_sessions"] == 2
    assert summary["total_rows"] == 2
    assert summary["total_animals"] == 2
    assert summary["animals"] == [305, 321]
    assert summary["complete_count"] == 2
    assert summary["session_types"] == {str(SessionTypes.MESOSCOPE_EXPERIMENT): 1, str(SessionTypes.RUN_TRAINING): 1}
    assert summary["acquisition_systems"] == {training_session.acquisition_system: 2}
    assert summary["columns"] == list(PROJECT_MANIFEST_SCHEMA)
    assert summary["pipeline_status_counts"] == {
        pipeline.value: {"done": 1, "not_done": 1} for pipeline in PIPELINE_STATUS_COLUMNS
    }


def test_the_summary_skips_a_pipeline_the_artifact_holds_no_column_for(tmp_path: Path) -> None:
    """A manifest written before a pipeline existed reports the pipelines it does carry rather than failing."""
    columns: dict[str, Any] = {name: [] for name in PROJECT_MANIFEST_SCHEMA if name != "video"}
    schema = {name: dtype for name, dtype in PROJECT_MANIFEST_SCHEMA.items() if name != "video"}
    path = tmp_path.joinpath("Legacy_manifest.feather")
    pl.DataFrame(data=columns, schema=schema).write_ipc(file=path, compression="uncompressed")

    summary = ProjectManifest(manifest_file=path).summarize()

    assert ProcessingPipelines.VIDEO.value not in summary["pipeline_status_counts"]
    assert summary["pipeline_status_counts"][ProcessingPipelines.CHECKSUM.value] == {"done": 0, "not_done": 0}
    assert summary["total_sessions"] == 0


# Tests for the printed views


def test_printing_the_data_emits_every_stored_column(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """The full view is the whole artifact, notes included, printed as one table."""
    manifest.print_data()

    printed = reported_messages[0]
    assert "notes" in printed
    assert "A synthetic session." in printed
    assert "session_path" in printed


def test_the_summary_view_indexes_sessions_per_animal(
    manifest: ProjectManifest, experiment_session: SessionData, reported_messages: list[str]
) -> None:
    """The printed session column counts from one within each animal, so a long session name never widens the table."""
    manifest.print_summary()

    printed = reported_messages[0]
    assert experiment_session.session_name not in printed
    assert "notes" not in printed
    assert "two_photon" in printed
    assert "305" in printed
    assert "321" in printed


def test_the_summary_view_honors_an_animal_filter(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """Naming an animal prints that animal's rows alone."""
    manifest.print_summary(animal=305)

    printed = reported_messages[0]
    assert "305" in printed
    assert "321" not in printed


def test_the_notes_view_reports_the_experimenter_text(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """The notes view is identity plus notes, which is what an operator reviews a run's outcome from."""
    manifest.print_notes()

    printed = reported_messages[0]
    assert "A synthetic session." in printed
    assert "two_photon" not in printed


def test_the_notes_view_honors_an_animal_filter(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """The notes view filters by animal the same way the summary view does."""
    manifest.print_notes(animal=321)

    printed = reported_messages[0]
    assert "321" in printed
    assert "305" not in printed
