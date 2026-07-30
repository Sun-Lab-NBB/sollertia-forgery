"""Tests the dataset state artifact: job scope resolution, subject resolution, and the written table's layout."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import polars as pl
import pytest
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.forging import (
    ANIMAL_SCOPE,
    SESSION_SCOPE,
    FORGING_JOB_NAME,
    DATASET_JOB_SCOPES,
    DATASET_STATE_SCHEMA,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    dataset_state_path,
    forging_tracker_path,
    generate_dataset_state,
)
from sollertia_forgery.forging.state import _build_job_rows

if TYPE_CHECKING:
    from pathlib import Path

_FIRST_SESSION: str = "2026-01-02-03-04-05-000006"
"""The session the tracked animal holds, used for both of its session-scoped jobs."""

_SECOND_SESSION: str = "2026-01-03-03-04-05-000006"
"""The session of the second animal, which carries an assembly job alone."""


@pytest.fixture
def dataset(tmp_path: Path) -> SimpleNamespace:
    """Builds a stand-in dataset whose root and session list drive the state artifact."""
    dataset_root = tmp_path.joinpath("dataset")
    dataset_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        name="test_dataset",
        dataset_data_path=dataset_root.joinpath("dataset.yaml"),
        sessions=(
            SimpleNamespace(session=_FIRST_SESSION, animal="305"),
            SimpleNamespace(session=_SECOND_SESSION, animal="321"),
        ),
    )


def _align_tracker(dataset: SimpleNamespace, jobs: list[tuple[str, str]]) -> ProcessingTracker:
    """Creates the dataset's forging tracker holding the provided job universe."""
    tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))
    tracker.align_jobs(jobs=jobs, universe=jobs)
    return tracker


def test_every_forging_job_name_declares_a_scope() -> None:
    """Verifies that each of the three job names the forging universe emits declares the unit its specifier names."""
    assert set(DATASET_JOB_SCOPES) == {
        MULTIDAY_DISCOVERY_JOB_NAME,
        MULTIDAY_EXTRACTION_JOB_NAME,
        FORGING_JOB_NAME,
    }
    assert DATASET_JOB_SCOPES[MULTIDAY_DISCOVERY_JOB_NAME] == ANIMAL_SCOPE
    assert DATASET_JOB_SCOPES[MULTIDAY_EXTRACTION_JOB_NAME] == SESSION_SCOPE
    assert DATASET_JOB_SCOPES[FORGING_JOB_NAME] == SESSION_SCOPE


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
    assert dict(frame.schema) == DATASET_STATE_SCHEMA


@pytest.fixture
def reported_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects every message the shared console is asked to echo, so a reported summary is assertable.

    Args:
        monkeypatch: The fixture used to replace the console's echo method for the duration of one test.

    Returns:
        The list the recorder appends each echoed message to, in the order they were emitted.
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
    """Verifies that an animal-scoped job takes its specifier as the animal, and a session-scoped job resolves its
    animal.
    """
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
    """Verifies that the stored table carries exactly the declared columns and dtypes, since consumers read it by
    schema.
    """
    tracker = _align_tracker(
        dataset=dataset,
        jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305"), (FORGING_JOB_NAME, _FIRST_SESSION)],
    )
    failed_id = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=_FIRST_SESSION)
    tracker.start_job(job_id=failed_id, executor_id="slurm:12345")
    tracker.fail_job(job_id=failed_id, error_message="assembly failed")

    written = generate_dataset_state(dataset=dataset)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == dataset_state_path(dataset=dataset)
    assert dict(frame.schema) == DATASET_STATE_SCHEMA
    assert frame["job_name"].to_list() == [FORGING_JOB_NAME, MULTIDAY_DISCOVERY_JOB_NAME]
    failure = frame.filter(pl.col("job_name") == FORGING_JOB_NAME).to_dicts()[0]
    assert failure["status"] == "FAILED"
    assert failure["executor_id"] == "slurm:12345"
    assert failure["error_message"] == "assembly failed"
