"""Tests the project's job artifact schema and the split that moved the job registry out of the manifest.

The manifest is session-rowed and the job artifact is job-rowed, so a reader pages jobs one job at a time rather than
one session at a time. These tests pin the column contract the two artifacts join on and the tracker fields the job
rows carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.managing import PROJECT_JOBS_SCHEMA, project_jobs_path, project_manifest_path
from sollertia_forgery.shared_assets import summarize_tracker
from sollertia_forgery.managing.manifest import PIPELINE_STATUS_COLUMNS, _read_pipeline_state

if TYPE_CHECKING:
    from pathlib import Path

SUBJECT_COLUMNS: tuple[str, ...] = ("animal", "session")
"""The columns that identify which session recorded a job row, and the keys the manifest joins against."""


def test_the_job_artifact_sits_beside_the_manifest(tmp_path: Path) -> None:
    """Both artifacts live at the project root under the project's own name, so one pull location serves both."""
    manifest = project_manifest_path(project_directory=tmp_path.joinpath("Proj"))
    jobs = project_jobs_path(project_directory=tmp_path.joinpath("Proj"))

    assert manifest.parent == jobs.parent
    assert manifest.name == "Proj_manifest.feather"
    assert jobs.name == "Proj_jobs.feather"


def test_the_job_schema_carries_the_subject_and_the_pipeline_discriminator() -> None:
    """A job row identifies its session and its pipeline, which is what makes the artifact joinable and filterable."""
    for column in (*SUBJECT_COLUMNS, "pipeline"):
        assert column in PROJECT_JOBS_SCHEMA

    assert PROJECT_JOBS_SCHEMA["animal"] == pl.String
    assert PROJECT_JOBS_SCHEMA["session"] == pl.String


def test_the_job_schema_covers_every_field_a_tracker_reports(tmp_path: Path) -> None:
    """The schema matches what the tracker reader emits, so a job row never silently drops a recorded field.

    Pins the contract against the tracker's own job-state fields, which is what would break if the tracking library
    started reporting a field the artifact has no column for.
    """
    tracker_path = tmp_path.joinpath("tracker.yaml")
    tracker = ProcessingTracker(file_path=tracker_path)
    jobs = [("stage", "specifier")]
    tracker.align_jobs(jobs=jobs, universe=jobs)
    job_id = ProcessingTracker.generate_job_id(job_name="stage", specifier="specifier")
    tracker.start_job(job_id=job_id, executor_id="slurm:1")
    tracker.fail_job(job_id=job_id, error_message="stage failed")

    emitted = set(summarize_tracker(jobs=tracker.snapshot())["jobs"][0])
    declared = set(PROJECT_JOBS_SCHEMA) - set(SUBJECT_COLUMNS) - {"pipeline"}

    assert emitted == declared


def test_reading_a_pipeline_state_returns_a_label_and_its_job_entries(tmp_path: Path) -> None:
    """Each pipeline contributes its rolled-up label to the manifest and its jobs to the job artifact."""
    pipeline = next(iter(PIPELINE_STATUS_COLUMNS))
    tracker_path = tmp_path.joinpath("tracker.yaml")
    tracker = ProcessingTracker(file_path=tracker_path)
    jobs = [("stage", "one"), ("stage", "two")]
    tracker.align_jobs(jobs=jobs, universe=jobs)
    for specifier in ("one", "two"):
        job_id = ProcessingTracker.generate_job_id(job_name="stage", specifier=specifier)
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    status, entries = _read_pipeline_state(pipeline=pipeline, tracker_path=tracker_path)

    assert status == "completed"
    assert len(entries) == 2
    assert {entry["pipeline"] for entry in entries} == {pipeline.value}


def test_an_absent_tracker_contributes_no_job_rows(tmp_path: Path) -> None:
    """A pipeline that never ran reports not_started and adds nothing to the job artifact."""
    status, entries = _read_pipeline_state(
        pipeline=next(iter(PIPELINE_STATUS_COLUMNS)), tracker_path=tmp_path.joinpath("absent.yaml")
    )

    assert status == "not_started"
    assert entries == []
