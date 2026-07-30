"""Tests the job plan caches and the project plan projection.

A plan entry is frozen on first write, since the cores and memory a job is submitted with must not change between
planning and running. These tests pin that freeze, the append behaviour when a unit's job universe widens, and the
layout of the projection that ships the caches.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path
from dataclasses import replace

import polars as pl
import pytest

from sollertia_forgery.runtime import RUNTIME_JOB_NAME
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.orchestration import (
    DATASET_UNIT,
    SESSION_UNIT,
    PROJECT_PLAN_SCHEMA,
    JobPlan,
    planning as planning_module,
    dataset_plan_path,
    project_plan_path,
    session_plan_path,
    generate_project_plan,
)
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.orchestration.dispatch import PipelineDispatch

CHECKSUM_JOBS: list[tuple[str, str]] = [(CHECKSUM_JOB_NAME, "")]
"""A single-job universe standing in for the checksum pipeline."""

RUNTIME_JOBS: list[tuple[str, str]] = [(RUNTIME_JOB_NAME, "51")]
"""A single-job universe standing in for the runtime pipeline."""


def make_session(root: Path) -> SimpleNamespace:
    """Builds a stand-in session exposing the attributes the planner and the projection read."""
    processed = root.joinpath("processed_data")
    processed.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        session_name=root.name, animal_id="305", processed_data_path=processed, unit_kind=SESSION_UNIT
    )


def make_dataset(root: Path) -> SimpleNamespace:
    """Builds a stand-in dataset exposing the attributes the planner and the projection read."""
    root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(name=root.name, dataset_data_path=root.joinpath("dataset.yaml"))


def make_dispatch(
    pipeline: ProcessingPipelines,
    unit: SimpleNamespace,
    universe: list[tuple[str, str]],
    memory_mb: int = 1000,
    *,
    fails: bool = False,
) -> PipelineDispatch[Any]:
    """Builds a dispatch entry whose resolver returns a fixed universe and whose estimator returns a fixed figure."""

    def discover(_path: Path) -> tuple[Any, list[tuple[str, str]], list[tuple[str, str]]]:
        if fails:
            message = f"{pipeline.value} resolves nothing here"
            raise FileNotFoundError(message)
        return unit, universe, universe

    def estimate(_unit: Any, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], tuple[int, bool]]:  # noqa: ANN401
        return {(job_name, specifier): (memory_mb, True) for job_name, specifier, _cores in jobs}

    return PipelineDispatch[Any](
        pipeline=pipeline,
        load=lambda _path: unit,
        discover=discover,
        worker=lambda _job: None,
        prerequisites=lambda _unit, _universe: {},
        # Planning registers each pipeline's possible jobs on its tracker, so the stand-in resolves a real writable
        # path beside the unit rather than a placeholder.
        tracker_path=lambda resolved: (
            getattr(resolved, "processed_data_path", None) or resolved.dataset_data_path.parent
        ).joinpath(f"{pipeline.value}_tracker.yaml"),
        output_path=lambda _unit: None,
        unit_name=lambda resolved: getattr(resolved, "session_name", None) or resolved.name,
        estimate_memory=estimate,
        command=lambda job: ("slf", pipeline.value, job.job_id),
    )


def plan_session(unit: SimpleNamespace, dispatches: list[PipelineDispatch[Any]], **kwargs: Any) -> JobPlan:  # noqa: ANN401
    """Plans a stand-in session through the private core, bypassing the real dispatch table."""
    return planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=dispatches,
        unit_path=unit.processed_data_path.parent,
        unit_kind=SESSION_UNIT,
        regenerate_plan=kwargs.get("regenerate_plan", False),
        display_progress=kwargs.get("display_progress", False),
    )


def test_a_plan_records_every_resolved_job_and_persists_it(tmp_path: Path) -> None:
    """Planning writes one entry per resolved job and the cache reloads to the same entries."""
    session = make_session(tmp_path.joinpath("session"))
    plan = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, memory_mb=3200),
            make_dispatch(ProcessingPipelines.RUNTIME, session, RUNTIME_JOBS, memory_mb=900),
        ],
    )

    assert plan.unit_kind == SESSION_UNIT
    assert {entry.key for entry in plan.entries} == {
        ("checksum", CHECKSUM_JOB_NAME, ""),
        ("runtime", RUNTIME_JOB_NAME, "51"),
    }
    # Cores come from the real allocation table rather than from the stand-in dispatch.
    assert plan.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].cores > 1
    assert JobPlan.from_yaml(file_path=session_plan_path(session=session)).entry_map() == plan.entry_map()


def test_recorded_figures_are_frozen_across_replanning(tmp_path: Path) -> None:
    """A second plan leaves recorded figures untouched, even when the estimator would now report a different one."""
    session = make_session(tmp_path.joinpath("session"))
    plan_session(unit=session, dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 3200)])

    replanned = plan_session(
        unit=session, dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 99999)]
    )

    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 3200


def test_forcing_re_estimates_recorded_figures(tmp_path: Path) -> None:
    """Forcing is the only way a recorded figure changes, so a deliberate retune can be adopted."""
    session = make_session(tmp_path.joinpath("session"))
    plan_session(unit=session, dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 3200)])

    replanned = plan_session(
        unit=session,
        dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 99999)],
        regenerate_plan=True,
    )

    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 99999


def test_a_widened_universe_appends_without_disturbing_recorded_entries(tmp_path: Path) -> None:
    """A job appearing later is estimated and appended while every earlier entry keeps its figure."""
    session = make_session(tmp_path.joinpath("session"))
    plan_session(unit=session, dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 3200)])

    widened = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(ProcessingPipelines.CHECKSUM, session, [*CHECKSUM_JOBS, (CHECKSUM_JOB_NAME, "extra")], 7777)
        ],
    )

    entries = widened.entry_map()
    assert entries[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 3200
    assert entries[("checksum", CHECKSUM_JOB_NAME, "extra")].memory_mb == 7777


def test_a_pipeline_resolving_nothing_is_skipped_rather_than_failing_the_plan(tmp_path: Path) -> None:
    """A resolver that rejects the unit leaves the other pipelines' jobs planned."""
    session = make_session(tmp_path.joinpath("session"))
    plan = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(ProcessingPipelines.TWO_PHOTON, session, CHECKSUM_JOBS, fails=True),
            make_dispatch(ProcessingPipelines.RUNTIME, session, RUNTIME_JOBS, memory_mb=900),
        ],
    )

    assert {entry.pipeline for entry in plan.entries} == {"runtime"}


def test_a_unit_no_pipeline_resolves_stops_the_plan(tmp_path: Path) -> None:
    """A unit every resolver rejects names no plan file, so it fails with the reasons the resolvers gave."""
    session = make_session(tmp_path.joinpath("session"))

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to plan the jobs"):
        plan_session(
            unit=session,
            dispatches=[make_dispatch(ProcessingPipelines.TWO_PHOTON, session, CHECKSUM_JOBS, fails=True)],
        )


def test_the_projection_carries_both_unit_kinds_in_the_declared_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The projection reads the caches of both unit kinds into one table matching the declared layout."""
    project = tmp_path.joinpath("Project")
    session = make_session(project.joinpath("305", "2026-01-02-03-04-05-000006"))
    dataset = make_dataset(project.joinpath("ds_a"))

    plan_session(unit=session, dispatches=[make_dispatch(ProcessingPipelines.CHECKSUM, session, CHECKSUM_JOBS, 3200)])
    planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=[make_dispatch(ProcessingPipelines.FORGING, dataset, RUNTIME_JOBS, 6400)],
        unit_path=project.joinpath("ds_a"),
        unit_kind=DATASET_UNIT,
        regenerate_plan=False,
        display_progress=False,
    )

    monkeypatch.setattr(planning_module, "iterate_sessions", lambda root_path: [session])  # noqa: ARG005
    monkeypatch.setattr(planning_module, "discover_project_datasets", lambda project_root: [dataset])  # noqa: ARG005

    written = generate_project_plan(project_directory=project)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == project_plan_path(project_directory=project)
    assert dict(frame.schema) == PROJECT_PLAN_SCHEMA
    assert set(frame["unit_kind"].to_list()) == {SESSION_UNIT, DATASET_UNIT}
    session_row = frame.filter(pl.col("unit_kind") == SESSION_UNIT).to_dicts()[0]
    assert session_row["animal"] == "305"
    assert session_row["dataset"] is None
    assert session_row["memory_mb"] == 3200
    dataset_row = frame.filter(pl.col("unit_kind") == DATASET_UNIT).to_dicts()[0]
    assert dataset_row["dataset"] == "ds_a"
    assert dataset_row["session"] is None


def test_an_unplanned_unit_contributes_no_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit carrying no cache is absent from the projection, so a reader treats its jobs as unplanned."""
    project = tmp_path.joinpath("Project")
    session = make_session(project.joinpath("305", "2026-01-02-03-04-05-000006"))

    monkeypatch.setattr(planning_module, "iterate_sessions", lambda root_path: [session])  # noqa: ARG005
    monkeypatch.setattr(planning_module, "discover_project_datasets", lambda project_root: [])  # noqa: ARG005

    frame = pl.read_ipc(source=generate_project_plan(project_directory=project), memory_map=True)

    assert frame.height == 0
    assert dict(frame.schema) == PROJECT_PLAN_SCHEMA


def test_the_dataset_cache_lands_at_the_dataset_root(tmp_path: Path) -> None:
    """A dataset's plan sits at its root beside its marker, so the dataset stays self-contained."""
    dataset = make_dataset(tmp_path.joinpath("ds_a"))
    assert dataset_plan_path(dataset=dataset).parent == tmp_path.joinpath("ds_a")


def test_planning_registers_the_possible_jobs_on_the_pipeline_tracker(tmp_path: Path) -> None:
    """The job artifact a remote batch is resolved from is built from trackers, so planning is what creates them."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=CHECKSUM_JOBS)

    planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=[dispatch],
        unit_path=tmp_path.joinpath("2024_11_04"),
        unit_kind=SESSION_UNIT,
        regenerate_plan=False,
        display_progress=False,
    )

    tracker_path = dispatch.tracker_path(session)
    assert tracker_path.is_file()
    recorded = ProcessingTracker(file_path=tracker_path).snapshot()
    assert [state.job_name for state in recorded.values()] == [CHECKSUM_JOB_NAME]


def test_a_job_the_unit_cannot_run_never_reaches_the_tracker(tmp_path: Path) -> None:
    """A job absent from the tracker is the statement that the unit cannot run it, which is what a scheduler reads."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, ""), (CHECKSUM_JOB_NAME, "unreachable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    # Narrows the possible subset to the first job, as a resolver does for a job whose input is absent.
    dispatch = replace(dispatch, discover=lambda _path: (session, universe, [universe[0]]))

    plan = planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=[dispatch],
        unit_path=tmp_path.joinpath("2024_11_04"),
        unit_kind=SESSION_UNIT,
        regenerate_plan=False,
        display_progress=False,
    )

    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    assert [state.specifier for state in recorded.values()] == [""]
    # The plan still sizes every job the pipeline could produce, since a plan describes cost rather than eligibility.
    assert len(plan.entries) == len(universe)


def test_the_plan_records_the_ordering_a_scheduler_builds_its_graph_from(tmp_path: Path) -> None:
    """Prerequisites live in the plan so a scheduler resolves a job's upstream stages without loading the unit."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, "upstream"), (CHECKSUM_JOB_NAME, "downstream")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    dispatch = replace(
        dispatch,
        discover=lambda _path: (session, universe, universe),
        prerequisites=lambda _unit, _jobs: {universe[0]: (), universe[1]: (universe[0],)},
    )

    plan = planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=[dispatch],
        unit_path=tmp_path.joinpath("2024_11_04"),
        unit_kind=SESSION_UNIT,
        regenerate_plan=False,
        display_progress=False,
    )

    entries = plan.entry_map()
    upstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "upstream")]
    downstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "downstream")]
    assert upstream.prerequisite_ids == []
    assert downstream.prerequisite_ids == [upstream.job_id]


def test_the_ordering_covers_the_universe_rather_than_the_possible_subset(tmp_path: Path) -> None:
    """A recorded edge is the pipeline's own, so it survives a unit that cannot currently produce its upstream job."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, "upstream"), (CHECKSUM_JOB_NAME, "downstream")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    dispatch = replace(
        dispatch,
        # The unit supports the downstream job alone, so its upstream stage stays in the universe by itself.
        discover=lambda _path: (session, universe, [universe[1]]),
        # Mirrors a real resolver by building ordering from the job set it is handed, so an edge appears only when
        # planning resolves ordering over the whole universe.
        prerequisites=lambda _unit, jobs: {
            job: ((universe[0],) if job == universe[1] and universe[0] in jobs else ()) for job in jobs
        },
    )

    plan = planning_module._resolve_unit_plan(  # noqa: SLF001
        dispatches=[dispatch],
        unit_path=tmp_path.joinpath("2024_11_04"),
        unit_kind=SESSION_UNIT,
        regenerate_plan=False,
        display_progress=False,
    )

    entries = plan.entry_map()
    upstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "upstream")]
    downstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "downstream")]
    assert downstream.prerequisite_ids == [upstream.job_id]

    # The tracker holds the possible subset alone, which is what tells a consumer to drop the recorded edge.
    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    assert [state.specifier for state in recorded.values()] == ["downstream"]
