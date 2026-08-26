"""Contains tests for the paging primitives every read tool shares, the three stages in which read_project_jobs_tool
reports, and the project dataset listing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from ataraxis_time import PrecisionTimer, TimerPrecisions
from sollertia_shared_assets import DATASET_MARKER_FILENAME

from sollertia_forgery.video import ENERGY_JOB_NAME
from sollertia_forgery.managing import (
    CHECKSUM_JOB_NAME,
    project_jobs_path,
)
from sollertia_forgery.managing.jobs import _PROJECT_JOBS_SCHEMA
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.interfaces.responses import (
    _DEFAULT_ITEM_LIMIT,
    _BREAKDOWN_AXIS_LIMIT,
    _DEFAULT_DETAILED_LIMIT,
    count_values,
    project_item,
    resolve_page,
    bounded_counts,
    reject_unknown,
    resolve_detail_limit,
    resolve_elapsed_seconds,
)
from sollertia_forgery.interfaces.forging_tools import list_project_datasets_tool
from sollertia_forgery.interfaces.management_tools import read_project_jobs_tool

if TYPE_CHECKING:
    from pathlib import Path

_JOB_COUNT: int = 7
"""The number of job rows the installed job artifact holds."""


@pytest.fixture
def project_directory(tmp_path: Path) -> Path:
    """Returns the project directory at which every read tool in this module is pointed."""
    return tmp_path.joinpath("Proj")


def _make_job(
    animal: str, session: str, pipeline: ProcessingPipelines, job_name: str, status: str
) -> dict[str, str | int | None]:
    """Builds one job row of the project's job artifact."""
    return {
        "animal": animal,
        "session": session,
        "pipeline": pipeline.value,
        "job_id": f"{pipeline.value}{job_name}{session}"[:16],
        "job_name": job_name,
        "specifier": "",
        "status": status,
        "executor_id": "slurm:1" if status != "SCHEDULED" else None,
        "error_message": "it failed" if status == "FAILED" else None,
        "started_at": 1 if status != "SCHEDULED" else None,
        "completed_at": 2 if status != "SCHEDULED" else None,
    }


def _install_jobs(project_directory: Path, count: int = _JOB_COUNT) -> Path:
    """Writes a job artifact holding a mix of pipelines and statuses."""
    project_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        _make_job(
            animal="305",
            session=f"s{index}",
            pipeline=ProcessingPipelines.VIDEO,
            job_name=ENERGY_JOB_NAME,
            status="SUCCEEDED",
        )
        for index in range(count - 2)
    ]
    rows.append(
        _make_job(
            animal="305",
            session="s9",
            pipeline=ProcessingPipelines.VIDEO,
            job_name=ENERGY_JOB_NAME,
            status="FAILED",
        )
    )
    rows.append(
        _make_job(
            animal="321",
            session="s9",
            pipeline=ProcessingPipelines.CHECKSUM,
            job_name=CHECKSUM_JOB_NAME,
            status="SCHEDULED",
        )
    )
    path = project_jobs_path(project_directory=project_directory)
    pl.DataFrame(data=rows, schema=_PROJECT_JOBS_SCHEMA, strict=False).write_ipc(file=path, compression="uncompressed")
    return path


def _install_dataset(project_directory: Path, name: str, members: list[tuple[str, str]]) -> Path:
    """Writes a dataset marker holding the given animal and session pairs."""
    root = project_directory.joinpath(name)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath(DATASET_MARKER_FILENAME).write_text(
        f"name: {name}\nproject: {project_directory.name}\nsession_type: mesoscope experiment\n"
        f"acquisition_system: mesoscope\nsessions:\n"
        + "".join(f"- session: {session}\n  animal: '{animal}'\n  session_path: ''\n" for animal, session in members)
    )
    return root


# Tests for the shared paging primitives


def test_a_page_reports_where_the_next_one_begins() -> None:
    """Verifies that walking a matched set means following next_start_row until it is null."""
    first = resolve_page(total=10, limit=4, start_row=0)
    assert (first.start, first.length, first.next_start_row) == (0, 4, 4)

    middle = resolve_page(total=10, limit=4, start_row=4)
    assert (middle.start, middle.length, middle.next_start_row) == (4, 4, 8)

    last = resolve_page(total=10, limit=4, start_row=8)
    assert (last.start, last.length, last.next_start_row) == (8, 2, None)


def test_a_page_past_the_end_is_empty_rather_than_an_error() -> None:
    """Verifies that a start row beyond the matches yields nothing and ends the walk."""
    window = resolve_page(total=10, limit=4, start_row=99)
    assert (window.length, window.next_start_row) == (0, None)


def test_a_negative_start_row_begins_at_the_first_match() -> None:
    """Verifies that a nonsensical start row is clamped rather than rejected, since it names no meaningful row."""
    assert resolve_page(total=10, limit=4, start_row=-5).start == 0


def test_a_limit_at_or_below_zero_lifts_the_cap() -> None:
    """Verifies that the unlimited escape lets a caller reading under a tight filter take everything at once."""
    for limit in (0, -1):
        window = resolve_page(total=10, limit=limit, start_row=0)
        assert window.length is None
        assert window.next_start_row is None


def test_the_default_page_shrinks_when_detail_is_requested() -> None:
    """Verifies that detail carries several times what semi-detail does, so its default page is shorter."""
    assert resolve_detail_limit(limit=None, detailed=False) == _DEFAULT_ITEM_LIMIT
    assert resolve_detail_limit(limit=None, detailed=True) == _DEFAULT_DETAILED_LIMIT
    assert _DEFAULT_DETAILED_LIMIT < _DEFAULT_ITEM_LIMIT
    # A named limit always wins over the default.
    assert resolve_detail_limit(limit=3, detailed=True) == 3


def test_counting_values_reports_absent_subjects_as_a_category() -> None:
    """Verifies that a null is itself a value on which a caller filters, so it is counted rather than dropped.

    A breakdown is read top to bottom as the list of values on which an axis can be filtered, so the counts are
    reported in value order rather than in the order the column happens to hold them.
    """
    counts = count_values(values=["a", "a", None, "b"])

    assert counts == {"a": 2, "b": 1, "none": 1}
    assert list(counts) == ["a", "b", "none"]


def test_filtering_on_a_column_the_table_does_not_hold_names_the_columns_it_does() -> None:
    """Verifies that a filter naming a column missing from the artifact is answered rather than failing on the read."""
    frame = pl.DataFrame({"status": ["FAILED"], "pipeline": ["video"]})

    response = reject_unknown(frame=frame, column="state", values=["FAILED"], subject="job")

    assert response is not None
    assert not response["success"]
    assert "state" in response["error"]
    assert "['pipeline', 'status']" in response["error"]


def test_the_elapsed_runtime_a_response_reports_is_measured_in_seconds() -> None:
    """Verifies that an operation timed in milliseconds is reported in the seconds every response uses."""
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    timer.reset()
    timer.delay(delay=100, allow_sleep=True, block=False)

    elapsed = resolve_elapsed_seconds(timer=timer)

    assert 0.09 <= elapsed < 1


def test_projecting_an_item_leaves_out_what_carries_nothing() -> None:
    """Verifies that an absent key reads as empty, so omitting an empty field costs a reader no information."""
    item = {"kept": "value", "empty_text": "", "empty_list": [], "absent": None, "zero": 0}

    assert project_item(item=item, fields=("kept", "empty_text", "empty_list", "absent", "zero", "missing")) == {
        "kept": "value",
        "zero": 0,
    }


def test_projecting_can_keep_empty_fields_when_asked() -> None:
    """Verifies that a caller needing a uniform shape across rows keeps the empty fields."""
    projected = project_item(item={"a": None}, fields=("a",), drop_empty=False)
    assert projected == {"a": None}


# Tests for the three stages of a project read tool


def test_a_bare_call_summarizes_and_lists_nothing(project_directory: Path) -> None:
    """Verifies that the first stage orients a caller on the axes it can filter, without paying for a listing."""
    _install_jobs(project_directory=project_directory)

    response = read_project_jobs_tool(project_path=str(project_directory))

    assert "jobs" not in response
    assert response["total_jobs"] == _JOB_COUNT
    assert response["breakdown"]["status"] == {"FAILED": 1, "SCHEDULED": 1, "SUCCEEDED": 5}
    assert response["breakdown"]["pipeline"] == {"checksum": 1, "video": 6}


def test_a_filter_adds_a_page_and_leaves_the_totals_whole(project_directory: Path) -> None:
    """Verifies that narrowing what is listed never narrows what is reported about the whole asset."""
    _install_jobs(project_directory=project_directory)

    response = read_project_jobs_tool(project_path=str(project_directory), status="FAILED")

    assert response["total_jobs"] == _JOB_COUNT
    assert response["matched_rows"] == 1
    assert len(response["jobs"]) == 1
    assert response["jobs"][0]["status"] == "FAILED"


def test_semi_detail_omits_the_provenance_that_detail_adds(project_directory: Path) -> None:
    """Verifies that semi-detail is identity and state, and detail adds the timing, executor, and error text."""
    _install_jobs(project_directory=project_directory)
    arguments = {"project_path": str(project_directory), "status": "FAILED"}

    semi = read_project_jobs_tool(**arguments)["jobs"][0]
    full = read_project_jobs_tool(**arguments, detailed=True)["jobs"][0]

    assert "executor_id" not in semi
    assert "error_message" not in semi
    assert "job_id" in semi
    assert full["error_message"] == "it failed"
    assert full["executor_id"] == "slurm:1"
    assert full["started_at"] == 1


def test_a_scheduled_job_omits_the_fields_it_has_no_value_for(project_directory: Path) -> None:
    """Verifies that a job that never ran carries no executor or timing, so those keys are absent."""
    _install_jobs(project_directory=project_directory)

    scheduled = read_project_jobs_tool(project_path=str(project_directory), status="SCHEDULED", detailed=True)["jobs"][
        0
    ]

    assert "executor_id" not in scheduled
    assert "started_at" not in scheduled


def test_walking_the_pages_covers_every_match(project_directory: Path) -> None:
    """Verifies that following next_start_row reaches every matching job exactly once."""
    _install_jobs(project_directory=project_directory)

    seen: list[tuple[str, str]] = []
    start: int | None = 0
    while start is not None:
        page = read_project_jobs_tool(project_path=str(project_directory), include_items=True, limit=2, start_row=start)
        seen.extend((job["pipeline"], job["session"]) for job in page["jobs"])
        start = page["next_start_row"]

    assert sorted(seen) == sorted(
        [("checksum", "s9"), ("video", "s9"), *(("video", f"s{index}") for index in range(_JOB_COUNT - 2))]
    )


def test_an_unknown_filter_value_names_what_is_available(project_directory: Path) -> None:
    """Verifies that a mistyped filter reports the values the axis actually holds."""
    _install_jobs(project_directory=project_directory)

    response = read_project_jobs_tool(project_path=str(project_directory), status="BOGUS")

    assert not response["success"]
    assert "SUCCEEDED" in response["error"]


def test_reading_an_absent_artifact_points_at_the_tool_that_writes_it(project_directory: Path) -> None:
    """Verifies that a project whose artifacts were never generated reports how to produce them."""
    project_directory.mkdir()

    response = read_project_jobs_tool(project_path=str(project_directory))

    assert not response["success"]
    assert "generate_project_manifest_tool" in response["error"]


# Tests for the project dataset listing


def test_listing_reports_every_dataset_a_project_holds(project_directory: Path) -> None:
    """Verifies that a project holds few datasets, so the listing is the summary and appears without an opt-in."""
    _install_dataset(project_directory=project_directory, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    _install_dataset(project_directory=project_directory, name="ds_b", members=[("305", "s1")])

    response = list_project_datasets_tool(project_path=str(project_directory))

    assert response["total_datasets"] == 2
    assert response["total_memberships"] == 3
    assert [entry["name"] for entry in response["datasets"]] == ["ds_a", "ds_b"]
    assert response["datasets"][0]["animal_count"] == 2


def test_a_session_resolves_to_every_dataset_holding_it(project_directory: Path) -> None:
    """Verifies that a session filter names every dataset that holds the session."""
    _install_dataset(project_directory=project_directory, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    _install_dataset(project_directory=project_directory, name="ds_b", members=[("305", "s1")])

    both = list_project_datasets_tool(project_path=str(project_directory), session="s1")
    one = list_project_datasets_tool(project_path=str(project_directory), session="s2")
    none = list_project_datasets_tool(project_path=str(project_directory), session="absent")

    assert [entry["name"] for entry in both["datasets"]] == ["ds_a", "ds_b"]
    assert [entry["name"] for entry in one["datasets"]] == ["ds_a"]
    assert none["matched_rows"] == 0
    # The totals still span the project, so a miss is distinguishable from an empty project.
    assert none["total_datasets"] == 2


def test_an_animal_resolves_to_every_dataset_holding_it(project_directory: Path) -> None:
    """Verifies that an animal filter answers the datasets in which a subject participates."""
    _install_dataset(project_directory=project_directory, name="ds_a", members=[("305", "s1"), ("321", "s2")])
    _install_dataset(project_directory=project_directory, name="ds_b", members=[("305", "s1")])

    response = list_project_datasets_tool(project_path=str(project_directory), animal="321")

    assert [entry["name"] for entry in response["datasets"]] == ["ds_a"]


def test_detail_reports_an_absent_state_snapshot_rather_than_guessing(project_directory: Path) -> None:
    """Verifies that a dataset whose state was never snapshotted reports the snapshot as absent."""
    _install_dataset(project_directory=project_directory, name="ds_a", members=[("305", "s1")])

    semi = list_project_datasets_tool(project_path=str(project_directory))["datasets"][0]
    full = list_project_datasets_tool(project_path=str(project_directory), detailed=True)["datasets"][0]

    assert "state_exists" not in semi
    assert not full["state_exists"]
    assert full["animals"] == ["305"]


def test_a_project_holding_no_datasets_reports_none(project_directory: Path) -> None:
    """Verifies that a project with no forged datasets reports an empty listing and a zero total."""
    project_directory.mkdir()

    response = list_project_datasets_tool(project_path=str(project_directory))

    assert response["success"]
    assert response["total_datasets"] == 0
    assert response["datasets"] == []


# Bounded breakdown axes


def test_an_axis_within_the_limit_reports_its_counts() -> None:
    """Verifies an axis holding few enough distinct values is counted exactly as an unbounded one is."""
    values = [f"unit-{index}" for index in range(_BREAKDOWN_AXIS_LIMIT)]

    assert bounded_counts(values=values) == count_values(values=values)


def test_an_axis_past_the_limit_reports_its_size_instead_of_its_counts() -> None:
    """Verifies an axis that grows with the project reports how many values it holds rather than listing them."""
    values = [f"unit-{index}" for index in range(_BREAKDOWN_AXIS_LIMIT + 1)]

    bounded = bounded_counts(values=values)

    assert bounded["distinct_values"] == _BREAKDOWN_AXIS_LIMIT + 1
    assert "elided" in bounded
    assert "unit-0" not in bounded
