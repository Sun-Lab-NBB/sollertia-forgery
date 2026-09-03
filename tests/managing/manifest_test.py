"""Contains tests for the project manifest artifact: the walk that generates it and the query surface that reads it
back.
"""

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
from sollertia_shared_assets import DESCRIPTOR_REGISTRY, SessionTypes, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.managing import (
    MANIFEST_AXES,
    MANIFEST_JOB_NAME,
    MANIFEST_SEMI_FIELDS,
    ProjectManifest,
    project_jobs_path,
    project_manifest_path,
    generate_project_manifest,
)
from sollertia_forgery.shared_assets import ProcessingPipelines, resolve_session_tracker_path
from sollertia_forgery.managing.manifest import (
    _MANIFEST_ROW_COLUMNS,
    _PIPELINE_STATUS_COLUMNS,
    _PROJECT_MANIFEST_SCHEMA,
    _MANIFEST_SUMMARY_COLUMNS,
    _assert_status_column_coverage,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_LOCK_POLL_TIMEOUT_S: float = 30.0
"""The longest a test waits for the generation thread to reach the manifest lock before giving up on it."""


@pytest.fixture
def reported_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects every message the shared console is asked to echo, so a reported outcome is assertable.

    Args:
        monkeypatch: The fixture used to replace the console's echo method for the duration of one test.

    Returns:
        The list to which the recorder appends each echoed message, in the order they were emitted.
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


@pytest.fixture
def numbered_animal_manifest(project_root: Path, session_factory: Callable[..., SessionData]) -> ProjectManifest:
    """Generates the manifest of a project whose animal identifiers order differently as text than as numbers.

    Args:
        project_root: The project for which the manifest is generated.
        session_factory: The builder that creates one acquired session for each of the two animals.

    Returns:
        The manifest reader bound to the generated artifact.
    """
    for animal_id in ("10", "2"):
        session_factory(animal_id=animal_id, session_type=SessionTypes.RUN_TRAINING)

    generate_project_manifest(project_directory=project_root)
    return ProjectManifest(manifest_file=project_manifest_path(project_directory=project_root))


def read_manifest(project_root: Path) -> pl.DataFrame:
    """Reads the manifest artifact of one project into a frame.

    Args:
        project_root: The project whose manifest to read.

    Returns:
        The manifest contents, one row per recorded session.
    """
    return pl.read_ipc(source=project_manifest_path(project_directory=project_root), memory_map=True)


def printed_animals(printed: str) -> list[str]:
    """Reads the animal identifier of each data row out of a printed manifest view, in the order it was printed.

    Args:
        printed: The table one of the printed views echoed.

    Returns:
        The animal identifier of every printed data row, in printed order.
    """
    rows = [line for line in printed.splitlines() if line.startswith("│")]
    # The first bordered row is the header, and every printed view leads with the animal column.
    return [row.split("┆")[0].strip("│ ") for row in rows[1:]]


def write_partial_then_fail(_frame: pl.DataFrame, file: Any, **_keywords: Any) -> None:
    """Stands in for the frame writer, writing a partial artifact into the handle it is given before it fails.

    Being handed an open handle rather than a destination path is what publishing through a temporary file offers, so
    this stand-in leaves its partial bytes in the temporary the publication discards rather than in the destination.

    Args: _frame: The frame handed to the writer, which this stand-in never serializes. file: The open file object
    receiving the artifact. **_keywords: The serialization options the caller passed, which this stand-in ignores.

    Raises: RuntimeError: Always, standing in for a writer that dies partway through.
    """
    file.write(b"partial")
    message = "the artifact writer died mid-write"
    raise RuntimeError(message)


def write_second_partial_then_fail(calls: list[int]) -> Any:
    """Builds a frame-writer stand-in that serializes normally until the manifest write, which it fails partway.

    Generation publishes the job artifact before the manifest, so failing the first call would abort before the
    manifest is ever opened. Letting that call through puts the failure on the manifest's own publication.

    Args:
        calls: The list to which the stand-in appends once per invocation, which is what sequences the two writes.

    Returns:
        The stand-in, which serializes the first frame it is handed and raises on every later one.
    """
    original = pl.DataFrame.write_ipc

    def _writer(frame: pl.DataFrame, file: Any, **keywords: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            original(frame, file, **keywords)
            return
        file.write(b"partial")
        message = "the artifact writer died mid-write"
        raise RuntimeError(message)

    return _writer


# Tests for the generation walk


def test_generation_records_one_row_per_session_with_its_pipeline_state(
    project_root: Path,
    experiment_session: SessionData,
    training_session: SessionData,
    mark_session_processed: Callable[[SessionData], None],
) -> None:
    """Verifies that a processed session reports every pipeline as done and an untouched session reports every one as
    not done.
    """
    mark_session_processed(experiment_session)

    generate_project_manifest(project_directory=project_root)

    frame = read_manifest(project_root=project_root)
    assert dict(frame.schema) == _PROJECT_MANIFEST_SCHEMA
    # The rows are ordered by animal, so the experiment animal precedes the training animal.
    assert frame.get_column("animal").to_list() == ["305", "321"]

    processed = frame.filter(pl.col("session") == experiment_session.session_name).to_dicts()[0]
    untouched = frame.filter(pl.col("session") == training_session.session_name).to_dicts()[0]

    assert processed["session_path"] == f"305/{experiment_session.session_name}"
    assert processed["type"] == str(SessionTypes.MESOSCOPE_EXPERIMENT)
    assert processed["system"] == experiment_session.acquisition_system
    assert processed["notes"] == "A synthetic session."
    assert processed["complete"] == 1
    assert [processed[column] for column in _PIPELINE_STATUS_COLUMNS.values()] == [1, 1, 1, 1, 1]
    assert [untouched[column] for column in _PIPELINE_STATUS_COLUMNS.values()] == [0, 0, 0, 0, 0]


def test_a_pipeline_holding_a_failed_or_a_running_job_is_not_recorded_as_finished(
    project_root: Path,
    experiment_session: SessionData,
    mark_session_processed: Callable[[SessionData], None],
    write_tracker: Callable[..., ProcessingTracker],
) -> None:
    """Verifies that a pipeline column reports 1 only when every one of that pipeline's jobs succeeded.

    A crashed pipeline and a pipeline still running both leave work to be done, so recording either as finished would
    read as a session that needs no re-running.
    """
    mark_session_processed(experiment_session)
    failed_job = (f"{ProcessingPipelines.VIDEO.value}_stage", "")
    running_job = (f"{ProcessingPipelines.RUNTIME.value}_stage", "")
    write_tracker(
        path=resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.VIDEO),
        jobs=[failed_job],
        failed={failed_job: "motion energy failed"},
    )
    write_tracker(
        path=resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.RUNTIME),
        jobs=[running_job],
        running=[running_job],
    )

    generate_project_manifest(project_directory=project_root)

    frame = read_manifest(project_root=project_root)
    row = frame.filter(pl.col("session") == experiment_session.session_name).to_dicts()[0]

    assert row[_PIPELINE_STATUS_COLUMNS[ProcessingPipelines.VIDEO]] == 0
    assert row[_PIPELINE_STATUS_COLUMNS[ProcessingPipelines.RUNTIME]] == 0
    untouched_pipelines = (
        ProcessingPipelines.CHECKSUM,
        ProcessingPipelines.MICROCONTROLLER,
        ProcessingPipelines.TWO_PHOTON,
    )
    assert [row[_PIPELINE_STATUS_COLUMNS[pipeline]] for pipeline in untouched_pipelines] == [1, 1, 1]


def test_the_recorded_date_is_the_session_name_read_as_utc(project_root: Path, training_session: SessionData) -> None:
    """Verifies that the session name is a UTC acquisition timestamp. The date column reproduces every microsecond."""
    generate_project_manifest(project_directory=project_root)

    components = [int(part) for part in training_session.session_name.split("-")]
    recorded = read_manifest(project_root=project_root).get_column("date").to_list()[0]

    assert recorded == datetime(*components, tzinfo=UTC)


def test_generation_marks_its_own_job_as_succeeded(project_root: Path, training_session: SessionData) -> None:
    """Verifies that the manifest run is itself a tracked job, recorded against the project rather than a session."""
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
    """Verifies that both artifacts are written under one lock, and the job artifact lands first, so a reader at worst
    holds job rows for a session the manifest does not list yet.
    """
    mark_session_processed(experiment_session)

    generate_project_manifest(project_directory=project_root)

    jobs = pl.read_ipc(source=project_jobs_path(project_directory=project_root), memory_map=True)
    assert jobs.height == len(_PIPELINE_STATUS_COLUMNS)
    assert set(jobs.get_column("pipeline").to_list()) == {pipeline.value for pipeline in _PIPELINE_STATUS_COLUMNS}
    assert jobs.get_column("session").unique().to_list() == [experiment_session.session_name]
    assert jobs.get_column("status").unique().to_list() == ["SUCCEEDED"]


def test_generation_announces_its_start_and_its_completion(
    project_root: Path,
    training_session: SessionData,
    reported_messages: list[str],
) -> None:
    """Verifies that requesting progress wraps the run in a preamble and a completion message naming the project."""
    generate_project_manifest(project_directory=project_root, display_progress=True)

    assert reported_messages == [
        f"Generating the project manifest for the '{project_root.stem}' project...",
        f"Project '{project_root.stem}' manifest: Generated.",
    ]


def test_generating_for_an_absent_project_reports_the_missing_directory(tmp_path: Path) -> None:
    """Verifies that a project path that names nothing fails before any discovery work is attempted."""
    with pytest.raises(FileNotFoundError, match="The specified project directory does not"):
        generate_project_manifest(project_directory=tmp_path.joinpath("NoSuchProject"))


def test_generating_for_a_project_without_sessions_reports_the_requirement(project_root: Path) -> None:
    """Verifies that a created but unused project holds nothing to snapshot, which is a hard error rather than an empty
    manifest.
    """
    with pytest.raises(FileNotFoundError, match="The project directory does not contain any"):
        generate_project_manifest(project_directory=project_root)


def test_a_session_missing_its_descriptor_fails_the_manifest_job(
    project_root: Path, training_session: SessionData
) -> None:
    """Verifies that the walk records the fault on the manifest tracker before it propagates, so the aborted run is
    visible.
    """
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
    """Verifies that a session type outside the descriptor registry cannot be snapshotted, so the run names what is."""
    monkeypatch.delitem(DESCRIPTOR_REGISTRY, SessionTypes.RUN_TRAINING)

    with pytest.raises(ValueError, match="An unsupported session type 'run training' was"):
        generate_project_manifest(project_directory=project_root)


def test_a_non_numeric_animal_identifier_is_recorded_as_the_marker_carries_it(
    project_root: Path,
    training_session: SessionData,
) -> None:
    """Verifies that the manifest records the animal identifier as text, matching every other artifact that carries one.

    The marker holds the identifier as free text, so a project whose animal directories are not plain numbers is
    snapshotted rather than refused at a conversion the schema no longer needs.
    """
    marker_path = training_session.raw_data_path.joinpath("session_data.yaml")
    recorded = marker_path.read_text()
    rewritten = recorded.replace(f"animal_id: '{training_session.animal_id}'", "animal_id: '305-repeat'")
    assert rewritten != recorded
    marker_path.write_text(rewritten)

    generate_project_manifest(project_directory=project_root)

    frame = pl.read_ipc(source=project_manifest_path(project_directory=project_root))
    assert "305-repeat" in frame.get_column("animal").to_list()


def test_a_session_emptied_after_discovery_is_left_out_of_the_manifest(
    project_root: Path,
    experiment_session: SessionData,
    training_session: SessionData,
) -> None:
    """Verifies that the walk re-reads each session, so one whose acquired data went away contributes no row.

    Discovery materializes the session list before the manifest lock is taken. Holding that lock from the test parks the
    run between the two, which is the only window in which a session can lose its raw data mid-generation.
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
        deadline = time.monotonic() + _LOCK_POLL_TIMEOUT_S
        while not tracker_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tracker_path.exists()
        shutil.rmtree(training_session.raw_data_path)
    finally:
        outer_lock.release()
    worker.join(timeout=_LOCK_POLL_TIMEOUT_S)

    assert not failures
    assert read_manifest(project_root=project_root).get_column("session").to_list() == [experiment_session.session_name]


def test_a_failed_write_leaves_the_previously_published_manifest_readable(
    project_root: Path,
    project_manifest: Path,  # Requested so a complete manifest is already published when the failing run starts.
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that the manifest is published by rename, so a writer that dies mid-write leaves the mapped file
    untouched.

    Rewriting the destination in place truncates it first, which would leave every reader that memory-maps the manifest
    without taking its lock facing an unreadable file. The job artifact is published first, so the stand-in lets that
    write through and dies on the manifest's own publication.
    """
    published = read_manifest(project_root=project_root).get_column("session").to_list()

    calls: list[int] = []
    monkeypatch.setattr(pl.DataFrame, "write_ipc", write_second_partial_then_fail(calls=calls))

    with pytest.raises(RuntimeError, match="died mid-write"):
        generate_project_manifest(project_directory=project_root)

    # Two writes were attempted, so the failure landed on the manifest rather than on the job artifact before it.
    assert len(calls) == 2
    assert read_manifest(project_root=project_root).get_column("session").to_list() == published
    assert [entry.name for entry in project_root.iterdir() if entry.name.endswith(".tmp")] == []


def test_every_session_pipeline_declares_a_status_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the import-time check names the pipelines that disagree, so a new pipeline fails loudly rather than
    silently.
    """
    monkeypatch.delitem(_PIPELINE_STATUS_COLUMNS, ProcessingPipelines.VIDEO)

    with pytest.raises(RuntimeError, match=r"the sets differ by \['video'\]"):
        _assert_status_column_coverage()


@pytest.mark.parametrize(
    ("roster_name", "narrowed_roster"),
    [
        (
            "_PROJECT_MANIFEST_SCHEMA",
            {column: dtype for column, dtype in _PROJECT_MANIFEST_SCHEMA.items() if column != "video"},
        ),
        ("_MANIFEST_ROW_COLUMNS", tuple(column for column in _MANIFEST_ROW_COLUMNS if column != "video")),
        ("_MANIFEST_SUMMARY_COLUMNS", tuple(column for column in _MANIFEST_SUMMARY_COLUMNS if column != "video")),
        ("MANIFEST_AXES", tuple(column for column in MANIFEST_AXES if column != "video")),
        ("MANIFEST_SEMI_FIELDS", tuple(column for column in MANIFEST_SEMI_FIELDS if column != "video")),
    ],
)
def test_a_roster_that_drops_a_status_column_names_the_roster_and_the_column(
    roster_name: str, narrowed_roster: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the import-time check names the roster missing a status column, so a column added to the manifest
    without reaching every roster that lists it fails loudly rather than going unreported.
    """
    monkeypatch.setattr(f"sollertia_forgery.managing.manifest.{roster_name}", narrowed_roster)

    with pytest.raises(RuntimeError, match=rf"{roster_name}[\s\S]*\['video'\]"):
        _assert_status_column_coverage()


@pytest.mark.parametrize(
    ("roster_name", "widened_roster"),
    [
        ("_MANIFEST_ROW_COLUMNS", (*_MANIFEST_ROW_COLUMNS, "retired_column")),
        ("_MANIFEST_SUMMARY_COLUMNS", (*_MANIFEST_SUMMARY_COLUMNS, "retired_column")),
        ("MANIFEST_AXES", (*MANIFEST_AXES, "retired_column")),
        ("MANIFEST_SEMI_FIELDS", (*MANIFEST_SEMI_FIELDS, "retired_column")),
    ],
)
def test_a_roster_naming_a_column_the_manifest_lacks_names_the_roster_and_the_entry(
    roster_name: str, widened_roster: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the import-time check names the roster holding an entry the manifest schema does not declare, so a
    renamed or retired column fails loudly rather than dropping out of every breakdown and listing that skips what the
    frame does not carry. The schema itself is left out of the parametrization, since it is the authority the other
    rosters are checked against.
    """
    monkeypatch.setattr(f"sollertia_forgery.managing.manifest.{roster_name}", widened_roster)

    with pytest.raises(RuntimeError, match=rf"{roster_name}[\s\S]*\['retired_column'\]"):
        _assert_status_column_coverage()


# Tests for the manifest query surface


def test_the_reader_reports_how_many_sessions_it_holds(manifest: ProjectManifest) -> None:
    """Verifies that the representation is the row count, which is what a caller checks against a loaded manifest."""
    assert repr(manifest) == "ProjectManifest(sessions=2)"


def test_the_reader_exposes_its_frame_and_its_animals(manifest: ProjectManifest) -> None:
    """Verifies that the frame is the stored artifact and the animals are its sorted unique subjects."""
    assert manifest.data.height == 2
    assert manifest.animals == ("305", "321")


def test_the_animal_roster_is_ordered_the_way_the_identifiers_are_written(
    numbered_animal_manifest: ProjectManifest,
) -> None:
    """Verifies that animal identifiers are numbers held as text, so the roster reports animal 2 before animal 10.

    Ordering them as plain text puts 10 first, which disagrees with the order in which the same identifiers are read
    and written everywhere else.
    """
    assert numbered_animal_manifest.animals == ("2", "10")


def test_every_session_is_listed_for_the_whole_project(
    manifest: ProjectManifest, experiment_session: SessionData, training_session: SessionData
) -> None:
    """Verifies that an unfiltered listing spans every animal, ordered by session name."""
    assert manifest.get_sessions() == tuple(sorted((experiment_session.session_name, training_session.session_name)))


def test_an_animal_filter_narrows_the_listing_to_its_sessions(
    manifest: ProjectManifest, experiment_session: SessionData
) -> None:
    """Verifies that naming an animal returns only the sessions in which that animal participated."""
    assert manifest.get_sessions(animal="305") == (experiment_session.session_name,)


def test_incomplete_sessions_are_excluded_unless_they_are_asked_for(
    project_root: Path,
    training_session: SessionData,
    session_factory: Callable[..., SessionData],
) -> None:
    """Verifies that the completeness filter is what separates the sessions worth processing from the aborted ones."""
    aborted = session_factory(animal_id="321", session_type=SessionTypes.LICK_TRAINING, incomplete=True)
    generate_project_manifest(project_directory=project_root)
    reader = ProjectManifest(manifest_file=project_manifest_path(project_directory=project_root))

    assert reader.get_sessions() == (training_session.session_name,)
    assert reader.get_sessions(exclude_incomplete=False) == tuple(
        sorted((training_session.session_name, aborted.session_name))
    )


def test_listing_an_unknown_animal_names_the_available_ones(manifest: ProjectManifest) -> None:
    """Verifies that a mistyped animal reports the project's roster rather than returning an empty tuple."""
    with pytest.raises(ValueError, match="Unable to filter sessions using animal ID '999'"):
        manifest.get_sessions(animal="999")


def test_a_session_row_carries_every_manifest_column(
    manifest: ProjectManifest, experiment_session: SessionData
) -> None:
    """Verifies that the per-session view is the stored row, so a caller reads acquisition and processing state from one
    place.
    """
    row = manifest.get_session_data(session=experiment_session.session_name)

    assert row.height == 1
    assert row.columns == list(_PROJECT_MANIFEST_SCHEMA)
    assert row.select("animal").item() == "305"


def test_a_session_resolves_to_the_animal_that_recorded_it(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """Verifies that the manifest is the lookup that maps a session name back to its subject."""
    assert manifest.get_animal_for_session(session=training_session.session_name) == "321"


def test_an_unknown_session_lookup_names_the_available_sessions(manifest: ProjectManifest) -> None:
    """Verifies that a session that the manifest does not hold reports what it does hold, including incomplete ones."""
    with pytest.raises(ValueError, match="Unable to look up the participating animal using session ID 'absent'"):
        manifest.get_animal_for_session(session="absent")


def test_a_session_resolves_to_the_system_that_acquired_it(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """Verifies that the acquisition system decides the pipelines a session supports, so it is queryable on its own."""
    assert manifest.get_system_for_session(session=training_session.session_name) == training_session.acquisition_system


def test_an_unknown_system_lookup_names_the_available_sessions(manifest: ProjectManifest) -> None:
    """Verifies that the system lookup fails the same way the animal lookup does, so both report the same roster."""
    with pytest.raises(ValueError, match="Unable to look up the acquisition system using session ID 'absent'"):
        manifest.get_system_for_session(session="absent")


def test_the_summary_counts_what_each_pipeline_finished(
    manifest: ProjectManifest, training_session: SessionData
) -> None:
    """Verifies that one processed session beside one untouched one gives every pipeline the same one-and-one split."""
    summary = manifest.summarize()

    assert summary["total_sessions"] == 2
    assert summary["total_rows"] == 2
    assert summary["total_animals"] == 2
    assert summary["animals"] == ["305", "321"]
    assert summary["complete_count"] == 2
    assert summary["session_types"] == {str(SessionTypes.MESOSCOPE_EXPERIMENT): 1, str(SessionTypes.RUN_TRAINING): 1}
    assert summary["acquisition_systems"] == {training_session.acquisition_system: 2}
    assert summary["columns"] == list(_PROJECT_MANIFEST_SCHEMA)
    assert summary["pipeline_status_counts"] == {
        pipeline.value: {"done": 1, "not_done": 1} for pipeline in _PIPELINE_STATUS_COLUMNS
    }


def test_the_summary_skips_a_pipeline_the_artifact_holds_no_column_for(tmp_path: Path) -> None:
    """Verifies that a manifest predating a pipeline reports the pipelines it carries rather than failing."""
    columns: dict[str, Any] = {name: [] for name in _PROJECT_MANIFEST_SCHEMA if name != "video"}
    schema = {name: dtype for name, dtype in _PROJECT_MANIFEST_SCHEMA.items() if name != "video"}
    path = tmp_path.joinpath("Legacy_manifest.feather")
    pl.DataFrame(data=columns, schema=schema).write_ipc(file=path, compression="uncompressed")

    summary = ProjectManifest(manifest_file=path).summarize()

    assert ProcessingPipelines.VIDEO.value not in summary["pipeline_status_counts"]
    assert summary["pipeline_status_counts"][ProcessingPipelines.CHECKSUM.value] == {"done": 0, "not_done": 0}
    assert summary["total_sessions"] == 0


# Tests for the printed views


def test_printing_the_data_emits_every_stored_column(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """Verifies that the full view is the whole artifact, notes included, printed as one table."""
    manifest.print_data()

    printed = reported_messages[0]
    assert "notes" in printed
    assert "A synthetic session." in printed
    assert "session_path" in printed


def test_the_summary_view_indexes_sessions_per_animal(
    manifest: ProjectManifest, experiment_session: SessionData, reported_messages: list[str]
) -> None:
    """Verifies that the printed session column counts from one within each animal, so a long session name never widens
    the table.
    """
    manifest.print_summary()

    printed = reported_messages[0]
    assert experiment_session.session_name not in printed
    assert "notes" not in printed
    assert "two_photon" in printed
    assert "305" in printed
    assert "321" in printed


def test_the_summary_view_honors_an_animal_filter(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """Verifies that naming an animal prints that animal's rows alone."""
    manifest.print_summary(animal="305")

    printed = reported_messages[0]
    assert "305" in printed
    assert "321" not in printed


def test_the_summary_view_prints_the_animals_in_natural_order(
    numbered_animal_manifest: ProjectManifest, reported_messages: list[str]
) -> None:
    """Verifies that the printed rows are grouped by animal in the order an operator reads the identifiers, so 2
    precedes 10.
    """
    numbered_animal_manifest.print_summary()

    assert printed_animals(printed=reported_messages[0]) == ["2", "10"]


def test_the_notes_view_reports_the_experimenter_text(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """Verifies that the notes view is identity plus notes, which is how an operator reviews a run's outcome."""
    manifest.print_notes()

    printed = reported_messages[0]
    assert "A synthetic session." in printed
    assert "two_photon" not in printed


def test_the_notes_view_honors_an_animal_filter(manifest: ProjectManifest, reported_messages: list[str]) -> None:
    """Verifies that the notes view filters by animal the same way the summary view does."""
    manifest.print_notes(animal="321")

    printed = reported_messages[0]
    assert "321" in printed
    assert "305" not in printed
