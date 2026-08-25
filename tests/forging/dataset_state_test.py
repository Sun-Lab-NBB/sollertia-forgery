"""Contains tests for the dataset state artifact: job scope resolution, subject resolution, and the written table's
layout.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    dataset_state_path,
    forging_tracker_path,
    generate_dataset_state,
)
from sollertia_forgery.forging.state import (
    _ANIMAL_SCOPE,
    _SESSION_SCOPE,
    _DATASET_JOB_SCOPES,
    _DATASET_STATE_SCHEMA,
    _build_job_rows,
)

if TYPE_CHECKING:
    from pathlib import Path

_FIRST_SESSION: str = "2026-01-02-03-04-05-000006"
"""The session the tracked animal holds, used for both of its session-scoped jobs."""

_SECOND_SESSION: str = "2026-01-03-03-04-05-000006"
"""The session of the second animal, which carries an assembly job alone."""


def write_partial_then_fail(_frame: pl.DataFrame, file: Any, **_keywords: Any) -> None:
    """Stands in for the frame writer, writing a partial artifact into the handle it is given before it fails.

    Publishing through a temporary file hands the writer an open handle rather than a destination path, so this
    stand-in leaves its partial bytes in the temporary that the publication discards rather than in the destination.

    Args: _frame: The frame passed to the writer, which this stand-in never serializes. file: The open file object that
    receives the artifact. **_keywords: The serialization options passed by the caller, which this stand-in ignores.

    Raises: RuntimeError: Always, standing in for a writer that dies partway through.
    """
    file.write(b"partial")
    message = "the artifact writer died mid-write"
    raise RuntimeError(message)


def build_dataset(dataset_root: Path, first_animal: str, second_animal: str) -> SimpleNamespace:
    """Builds a stand-in dataset whose root and session list drive the state artifact.

    Args:
        dataset_root: The directory that receives the dataset's tracker and state artifact.
        first_animal: The identifier of the animal that owns the first session.
        second_animal: The identifier of the animal that owns the second session.

    Returns:
        The stand-in dataset, carrying the attributes the state artifact reads.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        name="test_dataset",
        dataset_data_path=dataset_root.joinpath("dataset.yaml"),
        sessions=(
            SimpleNamespace(session=_FIRST_SESSION, animal=first_animal),
            SimpleNamespace(session=_SECOND_SESSION, animal=second_animal),
        ),
    )


@pytest.fixture
def dataset(tmp_path: Path) -> SimpleNamespace:
    """Builds a stand-in dataset whose root and session list drive the state artifact."""
    return build_dataset(dataset_root=tmp_path.joinpath("dataset"), first_animal="305", second_animal="321")


def _align_tracker(dataset: SimpleNamespace, jobs: list[tuple[str, str]]) -> ProcessingTracker:
    """Creates the dataset's forging tracker holding the provided job universe.

    Aligning against an empty request is rejected by the tracker, since it would classify every recorded job as
    foreign, so an empty universe is written through the reset that clears the registry instead.
    """
    tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))
    if jobs:
        tracker.align_jobs(jobs=jobs, universe=jobs)
    else:
        tracker.reset()
    return tracker


def test_every_forging_job_name_declares_a_scope() -> None:
    """Verifies that each of the three job names in the forging universe declares the unit its specifier names."""
    assert set(_DATASET_JOB_SCOPES) == {
        MULTIDAY_DISCOVERY_JOB_NAME,
        MULTIDAY_EXTRACTION_JOB_NAME,
        FORGING_JOB_NAME,
    }
    assert _DATASET_JOB_SCOPES[MULTIDAY_DISCOVERY_JOB_NAME] == _ANIMAL_SCOPE
    assert _DATASET_JOB_SCOPES[MULTIDAY_EXTRACTION_JOB_NAME] == _SESSION_SCOPE
    assert _DATASET_JOB_SCOPES[FORGING_JOB_NAME] == _SESSION_SCOPE


def test_a_dataset_without_a_tracker_reports_no_jobs(dataset: SimpleNamespace) -> None:
    """Verifies that a dataset whose jobs have never been aligned reports an empty table."""
    assert _build_job_rows(dataset=dataset) == []


def test_a_tracker_holding_no_jobs_reports_no_rows(dataset: SimpleNamespace) -> None:
    """Verifies that a tracker with an empty universe reports an empty table, so an emptied dataset stays readable."""
    _align_tracker(dataset=dataset, jobs=[])

    assert forging_tracker_path(dataset=dataset).is_file()
    assert _build_job_rows(dataset=dataset) == []


def test_an_empty_dataset_writes_an_artifact_carrying_the_declared_schema(dataset: SimpleNamespace) -> None:
    """Verifies that a dataset with no tracked job still writes the artifact, so a reader always finds a file."""
    written = generate_dataset_state(dataset=dataset)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert frame.height == 0
    assert dict(frame.schema) == _DATASET_STATE_SCHEMA


def test_a_failed_write_leaves_the_previously_published_state_readable(
    dataset: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a writer dying mid-write leaves the published artifact whole rather than truncated.

    The remote backend reads this artifact from a server path without taking the writer's lock, so only publishing
    by rename keeps it off a file that is being rewritten.
    """
    _align_tracker(dataset=dataset, jobs=[(FORGING_JOB_NAME, _FIRST_SESSION)])
    published = generate_dataset_state(dataset=dataset)

    monkeypatch.setattr(pl.DataFrame, "write_ipc", write_partial_then_fail)

    with pytest.raises(RuntimeError, match="died mid-write"):
        generate_dataset_state(dataset=dataset)

    assert pl.read_ipc(source=published, memory_map=True).get_column("job_name").to_list() == [FORGING_JOB_NAME]
    assert [entry.name for entry in published.parent.iterdir() if entry.name.endswith(".tmp")] == []


@pytest.fixture
def reported_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects every message the shared console is asked to echo, so a reported summary is assertable.

    Args:
        monkeypatch: The fixture used to replace the console's echo method for the duration of one test.

    Returns:
        The list into which the recorder appends each echoed message, in the order they were emitted.
    """
    messages: list[str] = []
    monkeypatch.setattr(console, "echo", lambda message, **_keywords: messages.append(message))
    return messages


def test_progress_reporting_announces_the_serialized_job_count(
    dataset: SimpleNamespace, reported_messages: list[str]
) -> None:
    """Verifies that requesting progress reports how many jobs the written artifact holds."""
    _align_tracker(
        dataset=dataset,
        jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305"), (FORGING_JOB_NAME, _FIRST_SESSION)],
    )

    generate_dataset_state(dataset=dataset, display_progress=True)

    assert reported_messages == ["Dataset 'test_dataset' state: Generated with 2 job(s)."]


def test_generation_stays_silent_without_progress_reporting(
    dataset: SimpleNamespace, reported_messages: list[str]
) -> None:
    """Verifies that a generation asked for no progress emits nothing, so a batch of them stays quiet."""
    _align_tracker(dataset=dataset, jobs=[(FORGING_JOB_NAME, _FIRST_SESSION)])

    generate_dataset_state(dataset=dataset)

    assert reported_messages == []


def test_each_scope_resolves_its_own_subject(dataset: SimpleNamespace) -> None:
    """Verifies an animal-scoped job takes its specifier as the animal, and a session-scoped job resolves its animal."""
    _align_tracker(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305"),
            (MULTIDAY_EXTRACTION_JOB_NAME, _FIRST_SESSION),
            (FORGING_JOB_NAME, _SECOND_SESSION),
        ],
    )

    rows = {row["job_name"]: row for row in _build_job_rows(dataset=dataset)}

    assert rows[MULTIDAY_DISCOVERY_JOB_NAME]["animal"] == "305"
    assert rows[MULTIDAY_DISCOVERY_JOB_NAME]["session"] is None
    assert rows[MULTIDAY_EXTRACTION_JOB_NAME]["animal"] == "305"
    assert rows[MULTIDAY_EXTRACTION_JOB_NAME]["session"] == _FIRST_SESSION
    assert rows[FORGING_JOB_NAME]["animal"] == "321"
    assert rows[FORGING_JOB_NAME]["session"] == _SECOND_SESSION


def test_a_session_the_dataset_dropped_reports_without_an_animal(dataset: SimpleNamespace) -> None:
    """Verifies that a tracker still holding a dropped session stays readable, so state survives a rebuild window."""
    _align_tracker(dataset=dataset, jobs=[(FORGING_JOB_NAME, "2020-01-01-00-00-00-000000")])

    rows = _build_job_rows(dataset=dataset)

    assert len(rows) == 1
    assert rows[0]["animal"] is None
    assert rows[0]["session"] == "2020-01-01-00-00-00-000000"


def test_a_job_name_without_a_scope_stops_the_serialization(dataset: SimpleNamespace) -> None:
    """Verifies that a tracker recording an unscoped job name fails loudly."""
    _align_tracker(dataset=dataset, jobs=[("unregistered_stage", _FIRST_SESSION)])

    with pytest.raises(ValueError, match="declare no scope"):
        _build_job_rows(dataset=dataset)


def test_the_written_artifact_matches_the_declared_schema(dataset: SimpleNamespace) -> None:
    """Verifies that the stored table carries exactly the declared columns and dtypes, and that a failed row carries
    every field the tracker recorded for it, while a running row and a scheduled row carry the identifier, the scope,
    and the timestamps their state implies.

    A consumer reads this artifact by schema and addresses a job it finds here by the identifier the row publishes, so a
    column that carries the wrong field ships a snapshot that names jobs no tracker holds.
    """
    tracker = _align_tracker(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305"),
            (MULTIDAY_EXTRACTION_JOB_NAME, _FIRST_SESSION),
            (FORGING_JOB_NAME, _FIRST_SESSION),
        ],
    )
    running_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_EXTRACTION_JOB_NAME, specifier=_FIRST_SESSION)
    tracker.start_job(job_id=running_id, executor_id="slurm:12344")
    failed_id = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=_FIRST_SESSION)
    tracker.start_job(job_id=failed_id, executor_id="slurm:12345")
    tracker.fail_job(job_id=failed_id, error_message="assembly failed")

    written = generate_dataset_state(dataset=dataset)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == dataset_state_path(dataset=dataset)
    assert dict(frame.schema) == _DATASET_STATE_SCHEMA
    assert frame["job_name"].to_list() == [
        MULTIDAY_EXTRACTION_JOB_NAME,
        FORGING_JOB_NAME,
        MULTIDAY_DISCOVERY_JOB_NAME,
    ]

    failure = frame.filter(pl.col("job_name") == FORGING_JOB_NAME).to_dicts()[0]
    assert failure["dataset"] == "test_dataset"
    assert failure["scope"] == _SESSION_SCOPE
    assert failure["job_id"] == failed_id
    assert failure["specifier"] == _FIRST_SESSION
    assert failure["status"] == "FAILED"
    assert failure["executor_id"] == "slurm:12345"
    assert failure["error_message"] == "assembly failed"
    assert failure["completed_at"] >= failure["started_at"]

    running = frame.filter(pl.col("job_name") == MULTIDAY_EXTRACTION_JOB_NAME).to_dicts()[0]
    assert running["job_id"] == running_id
    assert running["status"] == "RUNNING"
    assert running["started_at"] is not None
    assert running["completed_at"] is None

    scheduled = frame.filter(pl.col("job_name") == MULTIDAY_DISCOVERY_JOB_NAME).to_dicts()[0]
    assert scheduled["job_id"] == ProcessingTracker.generate_job_id(
        job_name=MULTIDAY_DISCOVERY_JOB_NAME, specifier="305"
    )
    assert scheduled["scope"] == _ANIMAL_SCOPE
    assert scheduled["specifier"] == "305"
    assert scheduled["error_message"] is None
    assert scheduled["started_at"] is None


def test_the_written_artifact_orders_its_animals_the_way_a_reader_reads_them(tmp_path: Path) -> None:
    """Verifies that the published rows are ordered naturally, so animal 9 precedes animal 10 rather than following it.

    Animal identifiers are numbers held as text, and every other listing produced by this stack reads them in numeric
    order, so ordering the snapshot as plain text would disagree with all of them.
    """
    numbered = build_dataset(dataset_root=tmp_path.joinpath("dataset"), first_animal="10", second_animal="9")
    _align_tracker(dataset=numbered, jobs=[(FORGING_JOB_NAME, _FIRST_SESSION), (FORGING_JOB_NAME, _SECOND_SESSION)])

    frame = pl.read_ipc(source=generate_dataset_state(dataset=numbered), memory_map=True)

    assert frame["animal"].to_list() == ["9", "10"]
    assert frame["session"].to_list() == [_SECOND_SESSION, _FIRST_SESSION]
