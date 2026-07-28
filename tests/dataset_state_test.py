"""Tests the dataset state artifact: job scope resolution, subject resolution, and the written table's layout.

A dataset's forging jobs sit at differing scopes, so the artifact is reported per job rather than per session. These
tests pin the scope mapping and the resolved subject columns that make the table joinable against the project
manifest.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import polars as pl
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

FIRST_SESSION: str = "2026-01-02-03-04-05-000006"
"""The session the tracked animal holds, used for both of its session-scoped jobs."""

SECOND_SESSION: str = "2026-01-03-03-04-05-000006"
"""The session of the second animal, which carries an assembly job alone."""


def make_dataset(dataset_root: Path) -> SimpleNamespace:
    """Builds a stand-in dataset whose root and session list drive the state artifact."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        name="test_dataset",
        dataset_data_path=dataset_root.joinpath("dataset.yaml"),
        sessions=(
            SimpleNamespace(session=FIRST_SESSION, animal="305"),
            SimpleNamespace(session=SECOND_SESSION, animal="321"),
        ),
    )


def align_tracker(dataset: SimpleNamespace, jobs: list[tuple[str, str]]) -> ProcessingTracker:
    """Creates the dataset's forging tracker holding the provided job universe."""
    tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))
    tracker.align_jobs(jobs=jobs, universe=jobs)
    return tracker


def test_every_forging_job_name_declares_a_scope() -> None:
    """The three job names the forging universe emits each declare the unit their specifier names."""
    assert set(DATASET_JOB_SCOPES) == {
        MULTIDAY_DISCOVERY_JOB_NAME,
        MULTIDAY_EXTRACTION_JOB_NAME,
        FORGING_JOB_NAME,
    }
    assert DATASET_JOB_SCOPES[MULTIDAY_DISCOVERY_JOB_NAME] == ANIMAL_SCOPE
    assert DATASET_JOB_SCOPES[MULTIDAY_EXTRACTION_JOB_NAME] == SESSION_SCOPE
    assert DATASET_JOB_SCOPES[FORGING_JOB_NAME] == SESSION_SCOPE


def test_a_dataset_without_a_tracker_reports_no_jobs(tmp_path: Path) -> None:
    """A dataset whose jobs have never been aligned reports an empty table rather than failing."""
    assert _build_job_rows(dataset=make_dataset(tmp_path.joinpath("dataset"))) == []


def test_each_scope_resolves_its_own_subject(tmp_path: Path) -> None:
    """An animal-scoped job takes its specifier as the animal, and a session-scoped job resolves its animal."""
    dataset = make_dataset(tmp_path.joinpath("dataset"))
    align_tracker(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305"),
            (MULTIDAY_EXTRACTION_JOB_NAME, FIRST_SESSION),
            (FORGING_JOB_NAME, SECOND_SESSION),
        ],
    )

    rows = {row["job_name"]: row for row in _build_job_rows(dataset=dataset)}

    assert rows[MULTIDAY_DISCOVERY_JOB_NAME]["animal"] == "305"
    assert rows[MULTIDAY_DISCOVERY_JOB_NAME]["session"] is None
    assert rows[MULTIDAY_EXTRACTION_JOB_NAME]["animal"] == "305"
    assert rows[MULTIDAY_EXTRACTION_JOB_NAME]["session"] == FIRST_SESSION
    assert rows[FORGING_JOB_NAME]["animal"] == "321"
    assert rows[FORGING_JOB_NAME]["session"] == SECOND_SESSION


def test_a_session_the_dataset_dropped_reports_without_an_animal(tmp_path: Path) -> None:
    """A tracker still holding a dropped session stays readable, so state survives a rebuild window."""
    dataset = make_dataset(tmp_path.joinpath("dataset"))
    align_tracker(dataset=dataset, jobs=[(FORGING_JOB_NAME, "2020-01-01-00-00-00-000000")])

    rows = _build_job_rows(dataset=dataset)

    assert len(rows) == 1
    assert rows[0]["animal"] is None
    assert rows[0]["session"] == "2020-01-01-00-00-00-000000"


def test_a_job_name_without_a_scope_stops_the_serialization(tmp_path: Path) -> None:
    """A tracker recording an unscoped job name fails loudly rather than emitting a row with no scope."""
    dataset = make_dataset(tmp_path.joinpath("dataset"))
    align_tracker(dataset=dataset, jobs=[("unregistered_stage", FIRST_SESSION)])

    with pytest.raises(ValueError, match="declare no scope"):
        _build_job_rows(dataset=dataset)


def test_the_written_artifact_matches_the_declared_schema(tmp_path: Path) -> None:
    """The stored table carries exactly the declared columns and dtypes, since consumers read it by schema."""
    dataset = make_dataset(tmp_path.joinpath("dataset"))
    tracker = align_tracker(
        dataset=dataset,
        jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305"), (FORGING_JOB_NAME, FIRST_SESSION)],
    )
    failed_id = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=FIRST_SESSION)
    tracker.start_job(job_id=failed_id, executor_id="slurm:12345")
    tracker.fail_job(job_id=failed_id, error_message="assembly failed")

    written = generate_dataset_state(dataset=dataset)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == dataset_state_path(dataset=dataset)
    assert dict(frame.schema) == DATASET_STATE_SCHEMA
    assert frame.height == 2
    failure = frame.filter(pl.col("job_name") == FORGING_JOB_NAME).to_dicts()[0]
    assert failure["status"] == "FAILED"
    assert failure["executor_id"] == "slurm:12345"
    assert failure["error_message"] == "assembly failed"
