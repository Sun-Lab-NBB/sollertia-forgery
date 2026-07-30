"""Contains tests for the project's job artifact schema and its join contract with the session manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from sollertia_forgery.managing import (
    PROJECT_JOBS_SCHEMA,
    project_jobs_path,
    write_project_jobs,
    project_manifest_path,
)
from sollertia_forgery.shared_assets import summarize_tracker
from sollertia_forgery.managing.manifest import PIPELINE_STATUS_COLUMNS, _read_pipeline_state

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from ataraxis_data_structures import ProcessingTracker

    from sollertia_forgery.shared_assets import ProcessingPipelines

_SUBJECT_COLUMNS: tuple[str, ...] = ("animal", "session")
"""The columns that identify which session recorded a job row, and the keys the manifest joins against."""


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
        assert column in PROJECT_JOBS_SCHEMA

    assert PROJECT_JOBS_SCHEMA["animal"] == pl.String
    assert PROJECT_JOBS_SCHEMA["session"] == pl.String


def test_the_job_schema_covers_every_field_a_tracker_reports(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that the job schema covers every field the tracker reader emits."""
    jobs = [("stage", "specifier")]
    tracker = write_tracker(
        path=tmp_path.joinpath("tracker.yaml"), jobs=jobs, failed={jobs[0]: "stage failed"}, executor_id="slurm:1"
    )

    emitted = set(summarize_tracker(jobs=tracker.snapshot())["jobs"][0])
    declared = set(PROJECT_JOBS_SCHEMA) - set(_SUBJECT_COLUMNS) - {"pipeline"}

    assert emitted == declared


@pytest.mark.parametrize("pipeline", list(PIPELINE_STATUS_COLUMNS))
def test_reading_a_pipeline_state_returns_a_label_and_its_job_entries(
    pipeline: ProcessingPipelines, tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that reading a pipeline state returns its rolled-up label and its job entries."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    jobs = [("stage", "one"), ("stage", "two")]
    write_tracker(path=tracker_path, jobs=jobs, succeeded=jobs)

    status, entries = _read_pipeline_state(pipeline=pipeline, tracker_path=tracker_path)

    assert status == "completed"
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
    assert dict(frame.schema) == PROJECT_JOBS_SCHEMA
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


def test_a_project_that_recorded_no_job_still_gets_its_artifact(tmp_path: Path) -> None:
    """Verifies that a project holding no job still writes an empty artifact carrying the full schema."""
    frame = pl.read_ipc(source=write_project_jobs(project_directory=tmp_path, job_rows=[]), memory_map=True)

    assert frame.height == 0
    assert dict(frame.schema) == PROJECT_JOBS_SCHEMA


@pytest.mark.parametrize("pipeline", list(PIPELINE_STATUS_COLUMNS))
def test_an_absent_tracker_contributes_no_job_rows(pipeline: ProcessingPipelines, tmp_path: Path) -> None:
    """Verifies that an absent tracker reports not_started and contributes no job rows."""
    status, entries = _read_pipeline_state(pipeline=pipeline, tracker_path=tmp_path.joinpath("absent.yaml"))

    assert status == "not_started"
    assert entries == []
