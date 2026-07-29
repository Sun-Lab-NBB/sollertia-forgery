"""Tests the aggregation and filtering the planning tools layer performs over a stored plan projection.

The tools are thin over the planning functions themselves, so these tests target what the layer adds: the totals a
submission is sized against, the per-pipeline breakdown, and the filters that narrow what is listed without distorting
what is reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from sollertia_forgery.orchestration import DATASET_UNIT, SESSION_UNIT, PROJECT_PLAN_SCHEMA, project_plan_path
from sollertia_forgery.interfaces.forging_tools import _status_counts
from sollertia_forgery.interfaces.planning_tools import read_project_plan_tool

if TYPE_CHECKING:
    from pathlib import Path

PLANNED_JOBS: list[dict[str, Any]] = [
    {
        "unit_kind": SESSION_UNIT,
        "animal": "305",
        "session": "2026-01-02-03-04-05-000006",
        "dataset": None,
        "pipeline": "video",
        "job_name": "motion_energy",
        "specifier": "51",
        "cores": 16,
        "memory_mb": 4000,
        "memory_modeled": True,
    },
    {
        "unit_kind": SESSION_UNIT,
        "animal": "305",
        "session": "2026-01-02-03-04-05-000006",
        "dataset": None,
        "pipeline": "video",
        "job_name": "camera_timestamp_rename",
        "specifier": "",
        "cores": 1,
        "memory_mb": 400,
        "memory_modeled": False,
    },
    {
        "unit_kind": SESSION_UNIT,
        "animal": "321",
        "session": "2026-01-03-03-04-05-000006",
        "dataset": None,
        "pipeline": "checksum",
        "job_name": "checksum_resolution",
        "specifier": "",
        "cores": 8,
        "memory_mb": 900,
        "memory_modeled": True,
    },
    {
        "unit_kind": DATASET_UNIT,
        "animal": None,
        "session": None,
        "dataset": "ds_a",
        "pipeline": "forging",
        "job_name": "session_data_assembly",
        "specifier": "2026-01-02-03-04-05-000006",
        "cores": 1,
        "memory_mb": 6000,
        "memory_modeled": True,
    },
]
"""A projection holding both unit kinds, two pipelines for one session, and one job carrying no modeled estimate."""


def install_projection(project_root: Path) -> Path:
    """Writes the stand-in projection at the location the tools read it from."""
    project_root.mkdir(parents=True, exist_ok=True)
    plan_path = project_plan_path(project_directory=project_root)
    pl.DataFrame(data=PLANNED_JOBS, schema=PROJECT_PLAN_SCHEMA, strict=False).write_ipc(
        file=plan_path, compression="uncompressed"
    )
    return plan_path


def test_reading_a_projection_reports_the_figures_a_submission_is_sized_against(tmp_path: Path) -> None:
    """The totals sum memory across every row and take the maxima independently, over the whole projection."""
    install_projection(project_root=tmp_path.joinpath("Proj"))

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert response["success"]
    assert response["total_jobs"] == len(PLANNED_JOBS)
    assert response["summed_memory_mb"] == 11300
    assert response["largest_job_memory_mb"] == 6000
    assert response["widest_job_cores"] == 16
    assert response["jobs_without_a_modeled_estimate"] == 1


def test_the_breakdown_groups_by_unit_kind_and_pipeline(tmp_path: Path) -> None:
    """Each pipeline reports its own job count, summed memory, and widest allocation."""
    install_projection(project_root=tmp_path.joinpath("Proj"))

    breakdown = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")))["breakdown"]

    entries = {(entry["unit_kind"], entry["pipeline"]): entry for entry in breakdown}
    assert entries[(SESSION_UNIT, "video")]["jobs"] == 2
    assert entries[(SESSION_UNIT, "video")]["summed_memory_mb"] == 4400
    assert entries[(SESSION_UNIT, "video")]["widest_job_cores"] == 16
    assert entries[(DATASET_UNIT, "forging")]["jobs"] == 1


def test_a_filter_narrows_the_listing_without_narrowing_the_totals(tmp_path: Path) -> None:
    """Filters restrict the listed jobs while the totals keep spanning every row, so a narrowed read stays honest."""
    install_projection(project_root=tmp_path.joinpath("Proj"))

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")), unit_kind=DATASET_UNIT)

    assert response["matched_rows"] == 1
    assert response["total_jobs"] == len(PLANNED_JOBS)
    assert [job["pipeline"] for job in response["jobs"]] == ["forging"]


def test_a_limit_caps_the_listing_and_reports_the_truncation(tmp_path: Path) -> None:
    """A capped read says so, so a caller never mistakes a page for the whole projection."""
    install_projection(project_root=tmp_path.joinpath("Proj"))

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")), limit=2)

    assert response["rows"] == 2
    assert response["matched_rows"] == len(PLANNED_JOBS)
    assert response["truncated"] is True


def test_an_unknown_pipeline_filter_names_what_is_available(tmp_path: Path) -> None:
    """A filter the projection cannot satisfy reports the available values rather than returning nothing."""
    install_projection(project_root=tmp_path.joinpath("Proj"))

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")), pipelines=["bogus"])

    assert not response["success"]
    assert "checksum" in response["error"]


def test_reading_an_absent_projection_points_at_the_command_that_writes_it(tmp_path: Path) -> None:
    """A project that has never been projected reports how to produce one."""
    tmp_path.joinpath("Proj").mkdir()

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert not response["success"]
    assert "generate_project_plan_tool" in response["error"]


def test_status_counts_total_every_job_and_group_by_status() -> None:
    """The dataset state summary counts each status the snapshot holds alongside the total."""
    frame = pl.DataFrame({"status": ["SUCCEEDED", "SUCCEEDED", "FAILED", "SCHEDULED"]})

    assert _status_counts(frame=frame) == {"total": 4, "FAILED": 1, "SCHEDULED": 1, "SUCCEEDED": 2}
