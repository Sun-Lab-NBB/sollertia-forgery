"""Tests the aggregation and filtering the planning tools layer performs over a stored plan projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from sollertia_forgery.orchestration import DATASET_UNIT, SESSION_UNIT, PROJECT_PLAN_SCHEMA, project_plan_path
from sollertia_forgery.interfaces.forging_tools import _status_counts
from sollertia_forgery.interfaces.planning_tools import read_project_plan_tool

if TYPE_CHECKING:
    from pathlib import Path

_PLANNED_JOBS: list[dict[str, Any]] = [
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


def install_projection(project_root: Path) -> None:
    """Writes the stand-in projection at the location the tools read it from."""
    project_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data=_PLANNED_JOBS, schema=PROJECT_PLAN_SCHEMA, strict=False).write_ipc(
        file=project_plan_path(project_directory=project_root), compression="uncompressed"
    )


@pytest.fixture
def projected_project(tmp_path: Path) -> Path:
    """Provides a project root carrying the stand-in plan projection.

    Args:
        tmp_path: The temporary directory the project root is created under.

    Returns:
        The path to the project root the projection was written into.
    """
    project_root = tmp_path.joinpath("Proj")
    install_projection(project_root=project_root)
    return project_root


def test_reading_a_projection_reports_the_figures_a_submission_is_sized_against(projected_project: Path) -> None:
    """Verifies that the totals sum memory across every row and take the maxima over the whole projection."""
    response = read_project_plan_tool(project_path=str(projected_project))

    assert response["success"]
    assert response["total_jobs"] == len(_PLANNED_JOBS)
    assert response["summed_memory_mb"] == 11300
    assert response["largest_job_memory_mb"] == 6000
    assert response["widest_job_cores"] == 16
    assert response["jobs_without_a_modeled_estimate"] == 1


def test_a_bare_call_reports_the_axes_a_caller_can_filter_on(projected_project: Path) -> None:
    """Verifies that the first stage names every filterable value and how much each matches, while listing nothing."""
    response = read_project_plan_tool(project_path=str(projected_project))

    assert "jobs" not in response
    assert response["breakdown"]["pipeline"] == {"checksum": 1, "forging": 1, "video": 2}
    assert response["breakdown"]["unit_kind"] == {DATASET_UNIT: 1, SESSION_UNIT: 3}
    assert response["breakdown"]["animal"] == {"305": 2, "321": 1, "none": 1}


def test_a_filter_narrows_the_listing_without_narrowing_the_totals(projected_project: Path) -> None:
    """Verifies that filters restrict the listed jobs while the totals span every row, so the read stays honest."""
    response = read_project_plan_tool(project_path=str(projected_project), unit_kind=DATASET_UNIT)

    assert response["matched_rows"] == 1
    assert response["total_jobs"] == len(_PLANNED_JOBS)
    assert [job["pipeline"] for job in response["jobs"]] == ["forging"]
    # A filter implies the listing, so the tool pages the matched rows on its own.
    assert response["start_row"] == 0


def test_a_page_reports_where_the_next_one_begins(projected_project: Path) -> None:
    """Verifies that a caller walks a long result by following next_start_row until it is null."""
    first = read_project_plan_tool(project_path=str(projected_project), include_items=True, limit=2)

    assert first["rows"] == 2
    assert first["matched_rows"] == len(_PLANNED_JOBS)
    assert first["start_row"] == 0
    assert first["next_start_row"] == 2

    last = read_project_plan_tool(
        project_path=str(projected_project), include_items=True, limit=2, start_row=first["next_start_row"]
    )

    assert last["rows"] == 2
    assert last["next_start_row"] is None


def test_opting_into_detail_adds_the_expensive_fields(projected_project: Path) -> None:
    """Verifies that semi-detail carries the job's figures and that detail adds whether the figure was modeled."""
    arguments = {"project_path": str(projected_project), "include_items": True}

    semi = read_project_plan_tool(**arguments)["jobs"][0]
    full = read_project_plan_tool(**arguments, detailed=True)["jobs"][0]

    assert "memory_modeled" not in semi
    assert "cores" in semi
    assert "memory_modeled" in full


def test_an_unknown_pipeline_filter_names_what_is_available(projected_project: Path) -> None:
    """Verifies that a filter the projection cannot satisfy reports the available values rather than nothing."""
    response = read_project_plan_tool(project_path=str(projected_project), pipelines=["bogus"])

    assert not response["success"]
    assert "checksum" in response["error"]


def test_reading_an_absent_projection_points_at_the_command_that_writes_it(tmp_path: Path) -> None:
    """Verifies that a project that has never been projected reports how to produce one."""
    tmp_path.joinpath("Proj").mkdir()

    response = read_project_plan_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert not response["success"]
    assert "generate_project_plan_tool" in response["error"]


def test_status_counts_totals_every_job_and_groups_by_status() -> None:
    """Verifies that the dataset state summary counts each status the snapshot holds alongside the total."""
    frame = pl.DataFrame({"status": ["SUCCEEDED", "SUCCEEDED", "FAILED", "SCHEDULED"]})

    assert _status_counts(frame=frame) == {"total": 4, "FAILED": 1, "SCHEDULED": 1, "SUCCEEDED": 2}
