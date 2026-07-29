"""Tests the paging primitives every read tool shares and the three stages the project read tools report in.

The stages are continuous: a bare call summarizes, a filter adds a page, and detail enriches that page. These tests
pin that progression, the page arithmetic that walks a long result, and the field projection that keeps a semi-detail
row small.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from sollertia_forgery.forging import DATASET_MARKER_FILENAME
from sollertia_forgery.managing import PROJECT_JOBS_SCHEMA, project_jobs_path
from sollertia_forgery.interfaces.responses import (
    DEFAULT_ITEM_LIMIT,
    DEFAULT_DETAILED_LIMIT,
    count_values,
    project_item,
    resolve_page,
    resolve_detail_limit,
)
from sollertia_forgery.interfaces.forging_tools import list_project_datasets_tool
from sollertia_forgery.interfaces.management_tools import read_project_jobs_tool

if TYPE_CHECKING:
    from pathlib import Path


def make_job(animal: str, session: str, pipeline: str, job_name: str, status: str) -> dict[str, Any]:
    """Builds one job row of the project's job artifact."""
    return {
        "animal": animal,
        "session": session,
        "pipeline": pipeline,
        "job_id": f"{pipeline}{job_name}{session}"[:16],
        "job_name": job_name,
        "specifier": "",
        "status": status,
        "executor_id": "slurm:1" if status != "SCHEDULED" else None,
        "error_message": "it failed" if status == "FAILED" else None,
        "started_at": 1 if status != "SCHEDULED" else None,
        "completed_at": 2 if status != "SCHEDULED" else None,
    }


def install_jobs(project_root: Path, count: int = 7) -> Path:
    """Writes a job artifact holding a mix of pipelines and statuses."""
    project_root.mkdir(parents=True, exist_ok=True)
    rows = [make_job("305", f"s{index}", "video", "motion_energy", "SUCCEEDED") for index in range(count - 2)]
    rows.append(make_job("305", "s9", "video", "motion_energy", "FAILED"))
    rows.append(make_job("321", "s9", "checksum", "checksum_resolution", "SCHEDULED"))
    path = project_jobs_path(project_directory=project_root)
    pl.DataFrame(data=rows, schema=PROJECT_JOBS_SCHEMA, strict=False).write_ipc(file=path, compression="uncompressed")
    return path


# Tests for the shared paging primitives


def test_a_page_reports_where_the_next_one_begins() -> None:
    """Walking a matched set means following next_start_row until it is null."""
    first = resolve_page(total=10, limit=4, start_row=0)
    assert (first.start, first.length, first.next_start_row) == (0, 4, 4)

    middle = resolve_page(total=10, limit=4, start_row=4)
    assert (middle.start, middle.length, middle.next_start_row) == (4, 4, 8)

    last = resolve_page(total=10, limit=4, start_row=8)
    assert (last.start, last.length, last.next_start_row) == (8, 2, None)


def test_a_page_past_the_end_is_empty_rather_than_an_error() -> None:
    """A start row beyond the matches yields nothing and ends the walk."""
    window = resolve_page(total=10, limit=4, start_row=99)
    assert (window.length, window.next_start_row) == (0, None)


def test_a_negative_start_row_begins_at_the_first_match() -> None:
    """A nonsensical start row is clamped rather than rejected, since it names no meaningful position."""
    assert resolve_page(total=10, limit=4, start_row=-5).start == 0


def test_a_limit_at_or_below_zero_lifts_the_cap() -> None:
    """The unlimited escape is deliberate, so a caller reading under a tight filter takes everything at once."""
    for limit in (0, -1):
        window = resolve_page(total=10, limit=limit, start_row=0)
        assert window.length is None
        assert window.next_start_row is None


def test_the_default_page_shrinks_when_detail_is_requested() -> None:
    """Detail carries several times what semi-detail does, so its default page is correspondingly shorter."""
    assert resolve_detail_limit(limit=None, detailed=False) == DEFAULT_ITEM_LIMIT
    assert resolve_detail_limit(limit=None, detailed=True) == DEFAULT_DETAILED_LIMIT
    assert DEFAULT_DETAILED_LIMIT < DEFAULT_ITEM_LIMIT
    # A named limit always wins over the default.
    assert resolve_detail_limit(limit=3, detailed=True) == 3


def test_counting_values_reports_absent_subjects_as_a_category() -> None:
    """A null is itself a value a caller filters on, so it is counted rather than dropped."""
    assert count_values(values=["a", "a", None, "b"]) == {"a": 2, "b": 1, "none": 1}


def test_projecting_an_item_leaves_out_what_carries_nothing() -> None:
    """An absent key reads as empty, so omitting an empty field costs a reader no information."""
    item = {"kept": "value", "empty_text": "", "empty_list": [], "absent": None, "zero": 0}

    assert project_item(item=item, fields=("kept", "empty_text", "empty_list", "absent", "zero", "missing")) == {
        "kept": "value",
        "zero": 0,
    }


def test_projecting_can_keep_empty_fields_when_asked() -> None:
    """A caller that needs a uniform shape across rows can keep the empty fields."""
    projected = project_item(item={"a": None}, fields=("a",), drop_empty=False)
    assert projected == {"a": None}


# Tests for the three stages of a project read tool


def test_a_bare_call_summarizes_and_lists_nothing(tmp_path: Path) -> None:
    """The first stage orients a caller on the axes it can filter, without paying for any listing."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))

    response = read_project_jobs_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert "jobs" not in response
    assert response["total_jobs"] == 7
    assert response["breakdown"]["status"] == {"FAILED": 1, "SCHEDULED": 1, "SUCCEEDED": 5}
    assert response["breakdown"]["pipeline"] == {"checksum": 1, "video": 6}


def test_a_filter_adds_a_page_and_leaves_the_totals_whole(tmp_path: Path) -> None:
    """Narrowing what is listed never narrows what is reported, so a filtered read stays honest about the asset."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))

    response = read_project_jobs_tool(project_path=str(tmp_path.joinpath("Proj")), status="FAILED")

    assert response["total_jobs"] == 7
    assert response["matched_rows"] == 1
    assert len(response["jobs"]) == 1
    assert response["jobs"][0]["status"] == "FAILED"


def test_semi_detail_omits_the_provenance_that_detail_adds(tmp_path: Path) -> None:
    """Semi-detail is identity and state, and detail adds the timing, executor, and error text."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))
    arguments = {"project_path": str(tmp_path.joinpath("Proj")), "status": "FAILED"}

    semi = read_project_jobs_tool(**arguments)["jobs"][0]
    full = read_project_jobs_tool(**arguments, detailed=True)["jobs"][0]

    assert "executor_id" not in semi
    assert "error_message" not in semi
    assert "job_id" in semi
    assert full["error_message"] == "it failed"
    assert full["executor_id"] == "slurm:1"


def test_a_scheduled_job_omits_the_fields_it_has_no_value_for(tmp_path: Path) -> None:
    """A job that never ran carries no executor or timing, so those keys are absent rather than null."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))

    scheduled = read_project_jobs_tool(project_path=str(tmp_path.joinpath("Proj")), status="SCHEDULED", detailed=True)[
        "jobs"
    ][0]

    assert "executor_id" not in scheduled
    assert "started_at" not in scheduled


def test_walking_the_pages_covers_every_match(tmp_path: Path) -> None:
    """Following next_start_row reaches every matching job exactly once."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))

    seen: list[str] = []
    start: int | None = 0
    while start is not None:
        page = read_project_jobs_tool(
            project_path=str(tmp_path.joinpath("Proj")), include_items=True, limit=2, start_row=start
        )
        seen.extend(job["job_id"] for job in page["jobs"])
        start = page["next_start_row"]

    assert len(seen) == 7


def test_an_unknown_filter_value_names_what_is_available(tmp_path: Path) -> None:
    """A mistyped filter reports the available values rather than returning an empty page."""
    install_jobs(project_root=tmp_path.joinpath("Proj"))

    response = read_project_jobs_tool(project_path=str(tmp_path.joinpath("Proj")), status="BOGUS")

    assert not response["success"]
    assert "SUCCEEDED" in response["error"]


def test_reading_an_absent_artifact_points_at_the_tool_that_writes_it(tmp_path: Path) -> None:
    """A project whose artifacts were never generated reports how to produce them."""
    tmp_path.joinpath("Proj").mkdir()

    response = read_project_jobs_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert not response["success"]
    assert "generate_project_manifest_tool" in response["error"]


# Tests for the project dataset listing


def install_dataset(project_root: Path, name: str, members: list[tuple[str, str]]) -> Path:
    """Writes a dataset marker holding the given animal and session pairs."""
    root = project_root.joinpath(name)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath(DATASET_MARKER_FILENAME).write_text(
        f"name: {name}\nproject: {project_root.name}\nsession_type: mesoscope experiment\n"
        "acquisition_system: mesoscope\nsessions:\n"
        + "".join(f"- session: {session}\n  animal: '{animal}'\n  session_path: ''\n" for animal, session in members)
    )
    return root


def test_listing_reports_every_dataset_a_project_holds(tmp_path: Path) -> None:
    """A project holds few datasets, so the listing is the summary and appears without an opt-in."""
    project = tmp_path.joinpath("Proj")
    install_dataset(project_root=project, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    install_dataset(project_root=project, name="ds_b", members=[("305", "s1")])

    response = list_project_datasets_tool(project_path=str(project))

    assert response["total_datasets"] == 2
    assert response["total_memberships"] == 3
    assert [entry["name"] for entry in response["datasets"]] == ["ds_a", "ds_b"]
    assert response["datasets"][0]["animal_count"] == 2


def test_a_session_resolves_to_every_dataset_holding_it(tmp_path: Path) -> None:
    """This is the membership question the manifest's dataset column used to answer."""
    project = tmp_path.joinpath("Proj")
    install_dataset(project_root=project, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    install_dataset(project_root=project, name="ds_b", members=[("305", "s1")])

    both = list_project_datasets_tool(project_path=str(project), session="s1")
    one = list_project_datasets_tool(project_path=str(project), session="s2")
    none = list_project_datasets_tool(project_path=str(project), session="absent")

    assert [entry["name"] for entry in both["datasets"]] == ["ds_a", "ds_b"]
    assert [entry["name"] for entry in one["datasets"]] == ["ds_a"]
    assert none["matched_rows"] == 0
    # The totals still span the project, so a miss is distinguishable from an empty project.
    assert none["total_datasets"] == 2


def test_an_animal_resolves_to_every_dataset_holding_it(tmp_path: Path) -> None:
    """An animal filter answers which datasets a subject participates in."""
    project = tmp_path.joinpath("Proj")
    install_dataset(project_root=project, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    install_dataset(project_root=project, name="ds_b", members=[("305", "s1")])

    response = list_project_datasets_tool(project_path=str(project), animal="321")

    assert [entry["name"] for entry in response["datasets"]] == ["ds_a"]


def test_detail_reports_an_absent_state_snapshot_rather_than_guessing(tmp_path: Path) -> None:
    """A dataset whose state was never snapshotted says so, rather than falling back to its tracker."""
    project = tmp_path.joinpath("Proj")
    install_dataset(project_root=project, name="ds_a", members=[("305", "s1")])

    semi = list_project_datasets_tool(project_path=str(project))["datasets"][0]
    full = list_project_datasets_tool(project_path=str(project), detailed=True)["datasets"][0]

    assert "state_exists" not in semi
    assert full["state_exists"] is False
    assert full["animals"] == ["305"]


def test_a_project_holding_no_datasets_reports_none(tmp_path: Path) -> None:
    """A project with no forged datasets lists nothing rather than failing."""
    tmp_path.joinpath("Proj").mkdir()

    response = list_project_datasets_tool(project_path=str(tmp_path.joinpath("Proj")))

    assert response["success"]
    assert response["total_datasets"] == 0
    assert response["datasets"] == []
