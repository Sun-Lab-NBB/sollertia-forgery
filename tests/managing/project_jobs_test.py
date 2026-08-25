"""Contains tests for the project's job artifact schema and its join contract with the session manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from ataraxis_data_structures import TrackerStatus

from sollertia_forgery.managing import (
    project_jobs_path,
    project_manifest_path,
)
from sollertia_forgery.managing.jobs import _PROJECT_JOBS_SCHEMA, write_project_jobs
from sollertia_forgery.managing.manifest import _PIPELINE_STATUS_COLUMNS, _read_pipeline_state

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from ataraxis_data_structures import ProcessingTracker

    from sollertia_forgery.shared_assets import ProcessingPipelines

_SUBJECT_COLUMNS: tuple[str, ...] = ("animal", "session")
"""The columns that identify which session recorded a job row, and the keys the manifest joins against."""


def write_partial_then_fail(_frame: pl.DataFrame, file: Any, **_keywords: Any) -> None:
    """Stands in for the frame writer, writing a partial artifact into the handle it is given before it fails.

    Being handed an open handle rather than a destination path is what publishing through a temporary file offers, so
    this stand-in leaves its partial bytes in the temporary the publication discards rather than in the destination.

    Args: _frame: The frame the writer was called on, which this stand-in never serializes. file: The open file object
    the artifact is written to. **_keywords: The serialization options the caller passed, which this stand-in ignores.

    Raises: RuntimeError: Always, standing in for a writer that dies partway through.
    """
    file.write(b"partial")
    message = "the artifact writer died mid-write"
    raise RuntimeError(message)


def test_the_job_artifact_sits_beside_the_manifest(tmp_path: Path) -> None:
    """Verifies that the job artifact and the manifest share a parent directory and the project-name prefix."""
    project_directory = tmp_path.joinpath("Proj")
    manifest = project_manifest_path(project_directory=project_directory)
    jobs = project_jobs_path(project_directory=project_directory)

    assert manifest.parent == jobs.parent
    assert manifest.name == "Proj_manifest.feather"
    assert jobs.name == "Proj_jobs.feather"


def test_the_job_schema_carries_the_subject_and_the_pipeline_discriminator() -> None:
    """Verifies that the job schema carries the subject columns and the pipeline discriminator."""
    for column in (*_SUBJECT_COLUMNS, "pipeline"):
        assert column in _PROJECT_JOBS_SCHEMA

    assert _PROJECT_JOBS_SCHEMA["animal"] == pl.String
    assert _PROJECT_JOBS_SCHEMA["session"] == pl.String


def test_the_job_schema_covers_every_field_a_tracker_reports(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that the job schema covers every field the tracker reader emits."""
    jobs = [("stage", "specifier")]
    tracker = write_tracker(
        path=tmp_path.joinpath("tracker.yaml"), jobs=jobs, failed={jobs[0]: "stage failed"}, executor_id="slurm:1"
    )

    emitted = set(tracker.summarize()["jobs"][0])
    declared = set(_PROJECT_JOBS_SCHEMA) - set(_SUBJECT_COLUMNS) - {"pipeline"}

    assert emitted == declared


@pytest.mark.parametrize("pipeline", list(_PIPELINE_STATUS_COLUMNS))
def test_reading_a_pipeline_state_returns_a_label_and_its_job_entries(
    pipeline: ProcessingPipelines, tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that reading a pipeline state returns its rolled-up label and its job entries."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    jobs = [("stage", "one"), ("stage", "two")]
    write_tracker(path=tracker_path, jobs=jobs, succeeded=jobs)

    status, entries = _read_pipeline_state(pipeline=pipeline, tracker_path=tracker_path)

    assert status is TrackerStatus.COMPLETED
    assert len(entries) == len(jobs)
    assert {entry["pipeline"] for entry in entries} == {pipeline.value}


def test_the_written_artifact_groups_rows_by_subject_then_pipeline(tmp_path: Path) -> None:
    """Verifies that the written rows are ordered by animal, then session, then pipeline."""
    rows: list[dict[str, str | None]] = [
        {"animal": "321", "session": "s1", "pipeline": "video", "job_name": "motion_energy", "specifier": "face"},
        {"animal": "305", "session": "s2", "pipeline": "checksum", "job_name": "checksum_resolution", "specifier": ""},
        {"animal": "305", "session": "s1", "pipeline": "video", "job_name": "motion_energy", "specifier": "body"},
        {"animal": "305", "session": "s1", "pipeline": "runtime", "job_name": "extraction", "specifier": None},
    ]

    written = write_project_jobs(project_directory=tmp_path, job_rows=rows)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == project_jobs_path(project_directory=tmp_path)
    assert dict(frame.schema) == _PROJECT_JOBS_SCHEMA
    assert list(
        zip(frame.get_column("animal"), frame.get_column("session"), frame.get_column("pipeline"), strict=True)
    ) == [
        ("305", "s1", "runtime"),
        ("305", "s1", "video"),
        ("305", "s2", "checksum"),
        ("321", "s1", "video"),
    ]
    # A row that names no specifier still records every other field it carried.
    assert frame.get_column("specifier").to_list() == [None, "body", "", "face"]


def test_the_written_rows_order_every_identifier_the_way_it_is_written(tmp_path: Path) -> None:
    """Verifies that the rows place animal 2 ahead of animal 10 and specifier 2 ahead of specifier 10.

    The animal and the specifier are numbers held as text, so ordering the rows as plain text would put 10 ahead of 2
    and leave this artifact disagreeing with the manifest a reader joins it against.
    """
    rows: list[dict[str, str | None]] = [
        {"animal": "10", "session": "s1", "pipeline": "two_photon", "job_name": "registration", "specifier": "10"},
        {"animal": "2", "session": "s1", "pipeline": "two_photon", "job_name": "registration", "specifier": "10"},
        {"animal": "2", "session": "s1", "pipeline": "two_photon", "job_name": "registration", "specifier": "2"},
    ]

    frame = pl.read_ipc(source=write_project_jobs(project_directory=tmp_path, job_rows=rows), memory_map=True)

    assert list(zip(frame.get_column("animal"), frame.get_column("specifier"), strict=True)) == [
        ("2", "2"),
        ("2", "10"),
        ("10", "10"),
    ]


def test_a_project_that_recorded_no_job_still_gets_its_artifact(tmp_path: Path) -> None:
    """Verifies that a project holding no job still writes an empty artifact carrying the full schema."""
    frame = pl.read_ipc(source=write_project_jobs(project_directory=tmp_path, job_rows=[]), memory_map=True)

    assert frame.height == 0
    assert dict(frame.schema) == _PROJECT_JOBS_SCHEMA


def test_a_failed_write_leaves_the_previously_published_artifact_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a writer dying mid-write leaves the published artifact whole rather than truncated.

    The readers memory-map the artifact without taking the manifest's lock, so only publishing by rename keeps them
    off a file that is being rewritten.
    """
    published = write_project_jobs(
        project_directory=tmp_path,
        job_rows=[{"animal": "305", "session": "s1", "pipeline": "video", "job_name": "motion_energy"}],
    )

    monkeypatch.setattr(pl.DataFrame, "write_ipc", write_partial_then_fail)

    with pytest.raises(RuntimeError, match="died mid-write"):
        write_project_jobs(project_directory=tmp_path, job_rows=[])

    assert pl.read_ipc(source=published, memory_map=True).get_column("job_name").to_list() == ["motion_energy"]
    assert [entry.name for entry in tmp_path.iterdir() if entry.name.endswith(".tmp")] == []


@pytest.mark.parametrize("pipeline", list(_PIPELINE_STATUS_COLUMNS))
def test_an_absent_tracker_contributes_no_job_rows(pipeline: ProcessingPipelines, tmp_path: Path) -> None:
    """Verifies that an absent tracker reports not_started and contributes no job rows."""
    status, entries = _read_pipeline_state(pipeline=pipeline, tracker_path=tmp_path.joinpath("absent.yaml"))

    assert status is TrackerStatus.NOT_STARTED
    assert entries == []
