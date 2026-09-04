"""Contains tests for the job plan caches, the project plan projection, and the registry that records a
prepared batch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from ataraxis_video_system import OutputLayout, ExtractedDataColumns
from sollertia_shared_assets import DatasetData, SessionData, SessionTypes, DatasetSession
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.runtime import RUNTIME_JOB_NAME
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.orchestration import (
    DATASET_UNIT,
    SESSION_UNIT,
    PROJECT_PLAN_SCHEMA,
    planning as planning_module,
    batch_directory,
    project_plan_path,
    read_batch_outcome,
    resolve_batch_host,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
    read_prepared_batches,
    record_prepared_batch,
)
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.orchestration.graph import BatchDocument
from sollertia_forgery.orchestration.batches import (
    _batch_path,
    _outcome_path,
    read_prepared_batch,
    forget_batch_records,
    record_batch_outcome,
    retire_prepared_batch,
)
from sollertia_forgery.orchestration.dispatch import _JOB_CORE_ALLOCATIONS, PipelineDispatch
from sollertia_forgery.orchestration.planning import (
    _JobPlan,
    _dataset_plan_path,
    _session_plan_path,
)
from sollertia_forgery.orchestration.footprints import _POSE_TABLE_COPIES, JobFootprint

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

_CHECKSUM_JOBS: list[tuple[str, str]] = [(CHECKSUM_JOB_NAME, "")]
"""A single-job universe standing in for the checksum pipeline."""

_RUNTIME_JOBS: list[tuple[str, str]] = [(RUNTIME_JOB_NAME, "51")]
"""A single-job universe standing in for the runtime pipeline."""


def write_partial_then_fail(_frame: pl.DataFrame, file: Any, **_keywords: Any) -> None:
    """Stands in for the frame writer, writing a partial artifact into the handle it is given before it fails.

    Being handed an open handle rather than a destination path is what publishing through a temporary file offers, so
    this stand-in leaves its partial bytes in the temporary the publication discards rather than in the destination.

    Args:
        _frame: The frame passed to the writer, which this stand-in never serializes.
        file: The open file object receiving the artifact.
        **_keywords: The serialization options the caller passed, which this stand-in ignores.

    Raises:
        RuntimeError: Always, standing in for a writer that dies partway through.
    """
    file.write(b"partial")
    message = "the artifact writer died mid-write"
    raise RuntimeError(message)


def write_camera_clock(session: SessionData, *, frames: int, period_us: int = 33_000) -> Path:
    """Writes the camera timestamp feather from which a training dataset's assembly job is sized.

    A training session records no imaging, so its assembly places every column on a camera clock and the estimate
    counts that feather's rows. Writing one is therefore what makes the session's own assembly job sizable.

    Args:
        session: The session whose processed video data receives the feather.
        frames: The frames the camera acquired, which is the samples its clock holds.
        period_us: The microseconds separating consecutive frames.

    Returns:
        The path to the written feather.
    """
    directory = session.processed_data.video_data_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(f"face_camera{OutputLayout.TIMESTAMPS_INFIX}{OutputLayout.FILE_SUFFIX}")
    pl.DataFrame(
        {ExtractedDataColumns.FRAME_TIME: np.arange(frames, dtype=np.uint64) * np.uint64(period_us)}
    ).write_ipc(file=path, compression="uncompressed")
    return path


def define_planned_dataset(project_root: Path, session: SessionData) -> DatasetData:
    """Creates the single-session forged dataset hierarchy on which the dataset planning tests operate."""
    return DatasetData.create(
        name="ds_planned",
        project=project_root.stem,
        session_type=SessionTypes.RUN_TRAINING,
        acquisition_system=session.acquisition_system,
        sessions=(DatasetSession(session=session.session_name, animal=str(session.animal_id)),),
        datasets_root=project_root,
        column_descriptions={"time_us": "The sample timestamp."},
    )


def refuse_to_size(_unit: Any, _jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
    """Stands in for a sizing pass whose job input cannot be read, naming the input the way the real pass does.

    Args:
        _unit: The unit on which the jobs operate, which this stand-in never reads.
        _jobs: The jobs to size, which this stand-in never sizes.

    Raises:
        FileNotFoundError: Always, standing in for a job whose input is absent.
    """
    message = "Unable to size the job. The archive 'camera_77.npz' does not exist."
    raise FileNotFoundError(message)


def refuse_one_job(refused: tuple[str, str]) -> Callable[..., dict[tuple[str, str], JobFootprint]]:
    """Builds a sizing pass that refuses one job of the universe and answers for every other job it is handed."""

    def estimate(_unit: Any, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
        """Sizes every requested job, raising where the request holds the job whose input cannot be read."""
        if any((job_name, specifier) == refused for job_name, specifier, _cores in jobs):
            message = f"Unable to size the job. The archive '{refused[1]}.npz' does not exist."
            raise FileNotFoundError(message)
        return {(job_name, specifier): JobFootprint(cores=cores, memory_mb=1000) for job_name, specifier, cores in jobs}

    return estimate


def omit_one_job(omitted: tuple[str, str]) -> Callable[..., dict[tuple[str, str], JobFootprint]]:
    """Builds a sizing pass that leaves one job out of the mapping it returns while raising nothing."""

    def estimate(_unit: Any, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
        """Sizes every requested job apart from the one this stand-in silently drops."""
        return {
            (job_name, specifier): JobFootprint(cores=cores, memory_mb=1000)
            for job_name, specifier, cores in jobs
            if (job_name, specifier) != omitted
        }

    return estimate


def refuse_the_whole_pass(_unit: Any, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
    """Stands in for a sizing pass that raises for a whole-pipeline request and answers for a single job.

    Raises:
        RuntimeError: If more than one job is requested, standing in for a shared reader that died mid-pass.
    """
    if len(jobs) > 1:
        message = "the shared reader died before it answered"
        raise RuntimeError(message)
    return {(job_name, specifier): JobFootprint(cores=cores, memory_mb=1000) for job_name, specifier, cores in jobs}


def make_session(root: Path, animal_id: str = "305") -> SimpleNamespace:
    """Builds a stand-in session exposing the attributes the planner and the projection read."""
    processed = root.joinpath("processed_data")
    processed.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        session_name=root.name, animal_id=animal_id, processed_data_path=processed, unit_kind=SESSION_UNIT
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
    unit_kind: str = SESSION_UNIT,
    *,
    fails: bool = False,
) -> PipelineDispatch[Any]:
    """Builds a dispatch entry whose resolver returns a fixed universe and whose estimator returns a fixed figure."""

    def discover(_path: Path) -> tuple[Any, list[tuple[str, str]], list[tuple[str, str]]]:
        """Returns the fixed unit and job universe, or raises where the stand-in is set to reject the unit."""
        if fails:
            message = f"{pipeline.value} resolves nothing here"
            raise FileNotFoundError(message)
        return unit, universe, universe

    def estimate(_unit: Any, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
        """Returns the fixed memory figure for every job it is handed, at the width the caller declared for it."""
        return {
            (job_name, specifier): JobFootprint(cores=cores, memory_mb=memory_mb) for job_name, specifier, cores in jobs
        }

    return PipelineDispatch[Any](
        pipeline=pipeline,
        unit_kind=unit_kind,
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
        size_jobs=estimate,
        command=lambda job: ("slf", pipeline.value, job.job_id),
    )


def plan_unit(
    unit_path: Path,
    unit_kind: str,
    dispatches: list[PipelineDispatch[Any]],
    *,
    regenerate_plan: bool = False,
    display_progress: bool = False,
) -> _JobPlan:
    """Plans a stand-in unit of either kind through the private core against the supplied dispatch entries."""
    return planning_module._resolve_unit_plan(
        dispatches=dispatches,
        unit_path=unit_path,
        unit_kind=unit_kind,
        regenerate_plan=regenerate_plan,
        display_progress=display_progress,
    )


def plan_session(
    unit: SimpleNamespace,
    dispatches: list[PipelineDispatch[Any]],
    *,
    regenerate_plan: bool = False,
    display_progress: bool = False,
) -> _JobPlan:
    """Plans a stand-in session through the private core against the supplied dispatch entries."""
    return plan_unit(
        unit_path=unit.processed_data_path.parent,
        unit_kind=SESSION_UNIT,
        dispatches=dispatches,
        regenerate_plan=regenerate_plan,
        display_progress=display_progress,
    )


def make_document(host: str = "workstation", pipeline: str = "checksum") -> BatchDocument:
    """Builds a prepared batch document holding one dispatchable job and one blocked job."""
    return BatchDocument(
        pipeline=pipeline,
        host=host,
        options={"regenerate_checksum": True},
        units=[{"unit_path": "/nonexistent/session", "unit_name": "session", "job_count": 1, "blocked_count": 1}],
        jobs=[{"job_id": "a_job", "unit_path": "/nonexistent/session", "cores": 8, "memory_mb": 1024}],
        blocked_jobs=[{"job_id": "a_blocked_job", "unsatisfied_prerequisite_ids": ["a_job"]}],
    )


def test_a_plan_records_every_resolved_job_and_persists_it(tmp_path: Path) -> None:
    """Verifies that planning writes one entry per resolved job and that the cache reloads to the same entries."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200),
            make_dispatch(pipeline=ProcessingPipelines.RUNTIME, unit=session, universe=_RUNTIME_JOBS, memory_mb=900),
        ],
    )

    assert plan.unit_kind == SESSION_UNIT
    assert {entry.key for entry in plan.entries} == {
        ("checksum", CHECKSUM_JOB_NAME, ""),
        ("runtime", RUNTIME_JOB_NAME, "51"),
    }
    # The declared allocation reaches the sizing pass and comes back unchanged for a stage that holds one width, so
    # the recorded figure is the real allocation table's rather than the stand-in's.
    assert plan.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].cores == _JOB_CORE_ALLOCATIONS[CHECKSUM_JOB_NAME]
    assert _JobPlan.from_yaml(file_path=_session_plan_path(session=session)).entry_map() == plan.entry_map()


def test_a_plan_records_the_width_the_sizing_pass_resolved(tmp_path: Path) -> None:
    """Verifies that a stage whose library picks a width per job records that width rather than the declared one."""
    session = make_session(root=tmp_path.joinpath("session"))
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS)
    # Stands in for a library that read its job's input and picked a narrower width than the job type declares.
    dispatch = replace(
        dispatch,
        size_jobs=lambda _unit, jobs: {
            (job_name, specifier): JobFootprint(cores=1, memory_mb=2048) for job_name, specifier, _cores in jobs
        },
    )

    plan = plan_session(unit=session, dispatches=[dispatch])

    # The declared allocation reaches the sizing pass as the fallback width, so the plan entry, and therefore the
    # scheduler, carries the width the pass itself picked.
    assert _JOB_CORE_ALLOCATIONS[CHECKSUM_JOB_NAME] != 1
    assert plan.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].cores == 1


def test_recorded_figures_are_frozen_across_replanning(tmp_path: Path) -> None:
    """Verifies that a second plan leaves every recorded figure untouched, whatever the estimator now reports."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )

    replanned = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=99999)
        ],
    )

    # The cores and memory with which a job is submitted must match the figures against which it was planned, so the
    # first write freezes the entry.
    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 3200


def test_replanning_never_re_reads_the_input_of_a_job_the_plan_already_holds(tmp_path: Path) -> None:
    """Verifies that a replan sizes the outstanding jobs alone, leaving the recorded ones' inputs unread."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS)
    # Stands in for the job's raw input having been archived since the figure was recorded.
    dispatch = replace(dispatch, size_jobs=refuse_to_size)

    replanned = plan_session(unit=session, dispatches=[dispatch])

    # A refused sizing pass drops the whole pipeline out of the plan, and a unit left with no pipeline is rejected.
    # Re-reading an archived input would therefore break the replan outright rather than merely cost a pass over
    # every container the pipeline opens.
    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 3200


def test_forcing_re_estimates_recorded_figures(tmp_path: Path) -> None:
    """Verifies that forcing re-estimates a recorded figure, which is how a deliberate retune is adopted."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )

    replanned = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=99999)
        ],
        regenerate_plan=True,
    )

    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 99999


def test_retuning_a_sizing_constant_re_estimates_a_recorded_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a retuned sizing constant re-estimates a recorded figure without a caller asking for it."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )

    monkeypatch.setattr("sollertia_forgery.orchestration.footprints._POSE_TABLE_COPIES", _POSE_TABLE_COPIES + 1.0)
    replanned = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=99999)
        ],
    )

    # The stamp a plan carries is a digest of the sizing constants that produced its figures, so a retune answers
    # with another stamp and every cache holding the old one is read as absent.
    assert replanned.entry_map()[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 99999


def test_a_widened_universe_appends_without_disturbing_recorded_entries(tmp_path: Path) -> None:
    """Verifies that a job appearing later is estimated and appended while every earlier entry keeps its figure."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )

    widened = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(
                pipeline=ProcessingPipelines.CHECKSUM,
                unit=session,
                universe=[*_CHECKSUM_JOBS, (CHECKSUM_JOB_NAME, "extra")],
                memory_mb=7777,
            )
        ],
    )

    entries = widened.entry_map()
    assert entries[("checksum", CHECKSUM_JOB_NAME, "")].memory_mb == 3200
    assert entries[("checksum", CHECKSUM_JOB_NAME, "extra")].memory_mb == 7777


def test_a_pipeline_resolving_nothing_is_skipped_rather_than_failing_the_plan(tmp_path: Path) -> None:
    """Verifies that a resolver rejecting the unit leaves the other pipelines' jobs planned."""
    session = make_session(root=tmp_path.joinpath("session"))
    plan = plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.TWO_PHOTON, unit=session, universe=_CHECKSUM_JOBS, fails=True),
            make_dispatch(pipeline=ProcessingPipelines.RUNTIME, unit=session, universe=_RUNTIME_JOBS, memory_mb=900),
        ],
    )

    assert {entry.pipeline for entry in plan.entries} == {"runtime"}


def test_a_unit_no_pipeline_resolves_stops_the_plan(tmp_path: Path) -> None:
    """Verifies that a unit rejected by every resolver names no plan file and fails with the reasons they gave."""
    session = make_session(root=tmp_path.joinpath("session"))

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to plan the jobs"):
        plan_session(
            unit=session,
            dispatches=[
                make_dispatch(
                    pipeline=ProcessingPipelines.TWO_PHOTON, unit=session, universe=_CHECKSUM_JOBS, fails=True
                )
            ],
        )


def test_a_pipeline_whose_input_cannot_be_read_is_dropped_rather_than_planned(tmp_path: Path) -> None:
    """Verifies that a pipeline refused by the sizing pass leaves the other pipelines' jobs planned."""
    session = make_session(root=tmp_path.joinpath("session"))
    refused = make_dispatch(pipeline=ProcessingPipelines.VIDEO, unit=session, universe=_CHECKSUM_JOBS)
    refused = replace(refused, size_jobs=refuse_to_size)
    plan = plan_session(
        unit=session,
        dispatches=[
            refused,
            make_dispatch(pipeline=ProcessingPipelines.RUNTIME, unit=session, universe=_RUNTIME_JOBS, memory_mb=900),
        ],
    )

    # Every job is modeled from the data it will read, so a pipeline whose sizing pass cannot read one of its inputs
    # says nothing about what this unit costs. That pipeline drops out of the plan entirely rather than contributing
    # a figure nothing measured, and it takes the same reporting path a rejected resolver takes.
    assert {entry.pipeline for entry in plan.entries} == {"runtime"}
    # Sizing precedes the tracker write, so the dropped pipeline registers no job a scheduler would then read.
    assert not refused.tracker_path(session).exists()


def test_a_unit_no_pipeline_can_size_stops_the_plan(tmp_path: Path) -> None:
    """Verifies that a unit whose every pipeline is refused writes no plan and names the input that could not be
    read.
    """
    session = make_session(root=tmp_path.joinpath("session"))
    dispatch = make_dispatch(pipeline=ProcessingPipelines.VIDEO, unit=session, universe=_CHECKSUM_JOBS)
    dispatch = replace(dispatch, size_jobs=refuse_to_size)

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to plan the jobs") as failure:
        plan_session(unit=session, dispatches=[dispatch])

    # The refusal names what it could not read, so a caller learns which input to restore, and the dropped unit is
    # absent from the plan rather than present at a floor.
    assert "camera_77.npz" in " ".join(str(failure.value).split())
    assert not _session_plan_path(session=session).exists()


def test_a_job_the_sizing_pass_omits_without_raising_is_recorded(tmp_path: Path) -> None:
    """Verifies that a job for which the sizing pass returns neither a footprint nor a refusal is still recorded."""
    session = make_session(root=tmp_path.joinpath("session"))
    universe = [(CHECKSUM_JOB_NAME, "sized"), (CHECKSUM_JOB_NAME, "omitted")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    dispatch = replace(dispatch, size_jobs=omit_one_job(omitted=universe[1]))

    plan = plan_session(unit=session, dispatches=[dispatch])

    assert {entry.specifier for entry in plan.entries} == {"sized"}
    # A job that vanishes from the pass with no reason recorded would leave the plan silently short, so the omission
    # itself is recorded against the job that carries it.
    assert "no footprint" in plan.unsized_jobs[f"checksum/{CHECKSUM_JOB_NAME} (omitted)"]


def test_a_one_pass_sizing_failure_is_recorded_even_when_every_job_then_sizes(tmp_path: Path) -> None:
    """Verifies that the refusal ending a pipeline's one-pass sizing survives a fallback that sizes every job."""
    session = make_session(root=tmp_path.joinpath("session"))
    universe = [(CHECKSUM_JOB_NAME, "first"), (CHECKSUM_JOB_NAME, "second")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    dispatch = replace(dispatch, size_jobs=refuse_the_whole_pass)

    plan = plan_session(unit=session, dispatches=[dispatch])

    # Sizing each job on its own recovers the figures the one pass withheld, so the account of why that pass ended
    # is the only record of it that remains.
    assert {entry.specifier for entry in plan.entries} == {"first", "second"}
    assert "the shared reader died" in plan.unsized_jobs["checksum/all jobs"]


def test_the_projection_carries_both_unit_kinds_in_the_declared_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the projection reads the caches of both unit kinds into one table matching the layout."""
    project = tmp_path.joinpath("Project")
    session = make_session(root=project.joinpath("305", "2026-01-02-03-04-05-000006"))
    dataset = make_dataset(root=project.joinpath("ds_a"))
    universe = [(CHECKSUM_JOB_NAME, "upstream"), (CHECKSUM_JOB_NAME, "downstream")]

    plan_session(
        unit=session,
        dispatches=[
            replace(
                make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe, memory_mb=3200),
                prerequisites=lambda _unit, jobs: {jobs[1]: (jobs[0],)},
            )
        ],
    )
    plan_unit(
        unit_path=project.joinpath("ds_a"),
        unit_kind=DATASET_UNIT,
        dispatches=[
            make_dispatch(
                pipeline=ProcessingPipelines.FORGING,
                unit=dataset,
                universe=_RUNTIME_JOBS,
                memory_mb=6400,
                unit_kind=DATASET_UNIT,
            )
        ],
    )

    monkeypatch.setattr(
        target=planning_module,
        name="iterate_sessions",
        value=lambda root_path: [session],  # noqa: ARG005
    )
    monkeypatch.setattr(
        target=planning_module,
        name="discover_project_datasets",
        value=lambda project_root: [dataset],  # noqa: ARG005
    )

    written = generate_project_plan(project_directory=project)
    frame = pl.read_ipc(source=written, memory_map=True)

    assert written == project_plan_path(project_directory=project)
    assert dict(frame.schema) == PROJECT_PLAN_SCHEMA
    assert set(frame["unit_kind"].to_list()) == {SESSION_UNIT, DATASET_UNIT}
    # Every job the session's cache holds carries its own row, so the join covers the whole unit rather than one job.
    session_rows = frame.filter(pl.col("unit_kind") == SESSION_UNIT).sort(by="specifier").to_dicts()
    upstream_id = ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier="upstream")
    # Every column is checked whole, since a scheduler joins this table against the recorded state on the hashed job
    # identifier and resolves each job's upstream stages from the ordering carried beside it.
    assert session_rows == [
        {
            "unit_kind": SESSION_UNIT,
            "animal": "305",
            "session": session.session_name,
            "dataset": None,
            "pipeline": ProcessingPipelines.CHECKSUM.value,
            "job_id": ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier="downstream"),
            "job_name": CHECKSUM_JOB_NAME,
            "specifier": "downstream",
            "cores": _JOB_CORE_ALLOCATIONS[CHECKSUM_JOB_NAME],
            "memory_mb": 3200,
            "memory_modeled": True,
            "prerequisite_ids": [upstream_id],
        },
        {
            "unit_kind": SESSION_UNIT,
            "animal": "305",
            "session": session.session_name,
            "dataset": None,
            "pipeline": ProcessingPipelines.CHECKSUM.value,
            "job_id": upstream_id,
            "job_name": CHECKSUM_JOB_NAME,
            "specifier": "upstream",
            "cores": _JOB_CORE_ALLOCATIONS[CHECKSUM_JOB_NAME],
            "memory_mb": 3200,
            "memory_modeled": True,
            "prerequisite_ids": [],
        },
    ]
    dataset_row = frame.filter(pl.col("unit_kind") == DATASET_UNIT).to_dicts()[0]
    assert dataset_row["dataset"] == "ds_a"
    assert dataset_row["session"] is None


def test_the_projection_orders_its_animals_the_way_their_identifiers_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the projection orders animal 2 ahead of animal 10 rather than behind it."""
    project = tmp_path.joinpath("Project")
    early = make_session(root=project.joinpath("2", "2026-01-02-03-04-05-000006"), animal_id="2")
    late = make_session(root=project.joinpath("10", "2026-01-02-03-04-05-000007"), animal_id="10")
    for session in (early, late):
        plan_session(
            unit=session,
            dispatches=[
                make_dispatch(
                    pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200
                )
            ],
        )

    monkeypatch.setattr(
        target=planning_module,
        name="iterate_sessions",
        value=lambda root_path: [late, early],  # noqa: ARG005
    )
    monkeypatch.setattr(
        target=planning_module,
        name="discover_project_datasets",
        value=lambda project_root: [],  # noqa: ARG005
    )

    frame = pl.read_ipc(source=generate_project_plan(project_directory=project), memory_map=True)

    # Every identifier on which this table sorts embeds a number in text, so ordering the rows as plain text
    # disagrees with the order in which the same animals are read and written everywhere else in the project.
    assert frame["animal"].to_list() == ["2", "10"]


def test_an_unplanned_unit_contributes_no_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a unit carrying no cache is absent from the projection, so a reader reads it as unplanned."""
    project = tmp_path.joinpath("Project")
    session = make_session(root=project.joinpath("305", "2026-01-02-03-04-05-000006"))

    monkeypatch.setattr(
        target=planning_module,
        name="iterate_sessions",
        value=lambda root_path: [session],  # noqa: ARG005
    )
    monkeypatch.setattr(
        target=planning_module,
        name="discover_project_datasets",
        value=lambda project_root: [],  # noqa: ARG005
    )

    frame = pl.read_ipc(source=generate_project_plan(project_directory=project), memory_map=True)

    assert frame.height == 0
    assert dict(frame.schema) == PROJECT_PLAN_SCHEMA


def test_a_failed_projection_leaves_the_previously_published_one_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a writer dying mid-write leaves the published projection whole rather than truncated."""
    project = tmp_path.joinpath("Project")
    session = make_session(root=project.joinpath("305", "2026-01-02-03-04-05-000006"))

    plan_session(
        unit=session,
        dispatches=[
            make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS, memory_mb=3200)
        ],
    )
    monkeypatch.setattr(
        target=planning_module,
        name="iterate_sessions",
        value=lambda root_path: [session],  # noqa: ARG005
    )
    monkeypatch.setattr(
        target=planning_module,
        name="discover_project_datasets",
        value=lambda project_root: [],  # noqa: ARG005
    )
    published = generate_project_plan(project_directory=project)

    monkeypatch.setattr(pl.DataFrame, "write_ipc", write_partial_then_fail)

    with pytest.raises(RuntimeError, match="died mid-write"):
        generate_project_plan(project_directory=project)

    # A scheduler memory-maps the projection without taking the writer's lock, so only publishing by rename keeps a
    # reader clear of a file that is being rewritten.
    assert pl.read_ipc(source=published, memory_map=True).get_column("memory_mb").to_list() == [3200]
    assert [entry.name for entry in project.iterdir() if entry.name.endswith(".tmp")] == []


def test_projecting_a_project_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """Verifies that a missing project holds neither a unit to read nor a writable location, so it is named here
    rather than surfacing as a walk failure partway through the projection.
    """
    with pytest.raises(FileNotFoundError, match=r"does\s+not\s+name\s+an\s+existing\s+directory"):
        generate_project_plan(project_directory=tmp_path.joinpath("NeverCreated"))


def test_the_dataset_cache_lands_at_the_dataset_root(tmp_path: Path) -> None:
    """Verifies that a dataset's plan sits at its root beside its marker, so the dataset stays self-contained."""
    dataset = make_dataset(root=tmp_path.joinpath("ds_a"))
    assert _dataset_plan_path(dataset=dataset).parent == tmp_path.joinpath("ds_a")


def test_planning_registers_the_possible_jobs_on_the_pipeline_tracker(tmp_path: Path) -> None:
    """Verifies that planning creates the pipeline trackers from which a remote batch's job artifact is resolved."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS)

    plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    tracker_path = dispatch.tracker_path(session)
    assert tracker_path.is_file()
    recorded = ProcessingTracker(file_path=tracker_path).snapshot()
    assert [state.job_name for state in recorded.values()] == [CHECKSUM_JOB_NAME]


def test_a_job_the_unit_cannot_run_never_reaches_the_tracker(tmp_path: Path) -> None:
    """Verifies that a job the unit cannot run stays off the tracker, which is what a scheduler reads."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, ""), (CHECKSUM_JOB_NAME, "unreachable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    # Narrows the possible subset to the first job, as a resolver does for a job whose input is absent.
    dispatch = replace(dispatch, discover=lambda _path: (session, universe, [universe[0]]))

    plan = plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    assert [state.specifier for state in recorded.values()] == [""]
    # The plan still sizes every job the pipeline could produce, since a plan describes cost rather than eligibility.
    assert len(plan.entries) == len(universe)


def test_replanning_keeps_the_recorded_state_of_a_job_the_unit_cannot_run_this_time(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that a job outside this run's possible subset keeps the state its tracker already recorded."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, ""), (CHECKSUM_JOB_NAME, "unreachable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    write_tracker(path=dispatch.tracker_path(session), jobs=universe, succeeded=[universe[1]])
    # Narrows the possible subset to the first job, as a resolver does while the second job's input is out of reach.
    dispatch = replace(dispatch, discover=lambda _path: (session, universe, [universe[0]]))

    plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    finished = ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier="unreachable")
    # Alignment discards every registry entry outside the universe it is handed, treating it as a job the pipeline no
    # longer defines. Declaring the possible subset as that universe would therefore delete the record of a job that
    # already succeeded, the moment its input is temporarily out of reach, and the deleted work would be run again.
    assert recorded[finished].status == ProcessingStatus.SUCCEEDED


def test_a_pipeline_supporting_no_job_records_its_figures_without_writing_a_tracker(tmp_path: Path) -> None:
    """Verifies that a pipeline resolving a universe but no runnable job is still planned, and writes no tracker."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, ""), (CHECKSUM_JOB_NAME, "unreachable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    # Empties the possible subset, as a resolver does for a unit carrying none of the inputs its jobs read.
    dispatch = replace(dispatch, discover=lambda _path: (session, universe, []))

    plan = plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    # A tracker states which jobs a unit supports, so an empty registry states nothing and is not written at all.
    assert not dispatch.tracker_path(session).exists()
    # A plan describes what its jobs would cost wherever the unit is eventually processed, so the figures the
    # pipeline resolved still belong in the cache.
    assert {entry.specifier for entry in plan.entries} == {"", "unreachable"}


def test_a_job_the_sizing_pass_refuses_is_retired_from_the_tracker(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that a refused job an earlier plan registered leaves the tracker, so preparation resolves the unit."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, "readable"), (CHECKSUM_JOB_NAME, "unreadable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    write_tracker(path=dispatch.tracker_path(session), jobs=universe)
    dispatch = replace(dispatch, size_jobs=refuse_one_job(refused=universe[1]))

    plan = plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    # Preparation reads a tracked job carrying no planned figures as an unplanned job and refuses the whole unit over
    # it, so retiring the entry alongside the figures that sized it keeps the remaining jobs dispatchable.
    assert [state.specifier for state in recorded.values()] == ["readable"]
    assert "unreadable.npz" in plan.unsized_jobs[f"checksum/{CHECKSUM_JOB_NAME} (unreadable)"]


def test_a_refused_job_the_tracker_records_as_succeeded_keeps_its_entry(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that a refused job that already succeeded keeps its recorded outcome, so the work is not run again."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, "readable"), (CHECKSUM_JOB_NAME, "unreadable")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    write_tracker(path=dispatch.tracker_path(session), jobs=universe, succeeded=[universe[1]])
    dispatch = replace(dispatch, size_jobs=refuse_one_job(refused=universe[1]))

    plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    finished = ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier="unreadable")
    # A completed job's record outlives the input that produced it, and preparation already leaves a succeeded job
    # out of the work it dispatches, so retiring the entry gains nothing.
    assert recorded[finished].status == ProcessingStatus.SUCCEEDED


def test_priming_is_handed_the_unit_root_rather_than_the_directory_holding_it(tmp_path: Path) -> None:
    """Verifies a pipeline whose job model lives in state written by a dependency is primed against the unit itself."""
    root = tmp_path.joinpath("2024_11_04")
    session = make_session(root=root)
    primed: list[Path] = []
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=_CHECKSUM_JOBS)
    dispatch = replace(dispatch, prime=primed.append)

    plan = plan_unit(unit_path=root, unit_kind=SESSION_UNIT, dispatches=[dispatch])

    # Priming loads the unit from the path it is handed, so any other path raises. Both priming and resolution run
    # inside the same guard, which swallows that refusal into the skip report and takes the whole pipeline out of the
    # plan, so the second assertion is what makes a wrong path loud.
    assert primed == [root]
    assert {entry.pipeline for entry in plan.entries} == {ProcessingPipelines.CHECKSUM.value}


def test_the_plan_records_the_ordering_a_scheduler_builds_its_graph_from(tmp_path: Path) -> None:
    """Verifies that prerequisites live in the plan, so a scheduler resolves a job's upstream stages on its own."""
    session = make_session(root=tmp_path.joinpath("2024_11_04"))
    universe = [(CHECKSUM_JOB_NAME, "upstream"), (CHECKSUM_JOB_NAME, "downstream")]
    dispatch = make_dispatch(pipeline=ProcessingPipelines.CHECKSUM, unit=session, universe=universe)
    dispatch = replace(
        dispatch,
        discover=lambda _path: (session, universe, universe),
        prerequisites=lambda _unit, _jobs: {universe[0]: (), universe[1]: (universe[0],)},
    )

    plan = plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    entries = plan.entry_map()
    upstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "upstream")]
    downstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "downstream")]
    assert upstream.prerequisite_ids == []
    assert downstream.prerequisite_ids == [upstream.job_id]


def test_the_ordering_covers_the_universe_rather_than_the_possible_subset(tmp_path: Path) -> None:
    """Verifies that a recorded edge is the pipeline's own, so it survives a unit lacking its upstream job."""
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

    plan = plan_unit(unit_path=tmp_path.joinpath("2024_11_04"), unit_kind=SESSION_UNIT, dispatches=[dispatch])

    entries = plan.entry_map()
    upstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "upstream")]
    downstream = entries[(ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, "downstream")]
    assert downstream.prerequisite_ids == [upstream.job_id]

    # The tracker holds the possible subset alone, which is what tells a consumer to drop the recorded edge.
    recorded = ProcessingTracker(file_path=dispatch.tracker_path(session)).snapshot()
    assert [state.specifier for state in recorded.values()] == ["downstream"]


# The real dispatch table


def test_planning_an_acquired_session_records_the_pipelines_that_resolve_jobs(
    experiment_session: SessionData,
) -> None:
    """Verifies that planning drives every registered session pipeline, keeping the ones that resolve a job."""
    session_path = experiment_session.raw_data_path.parent

    plan = resolve_session_plan(session_path=session_path, display_progress=True)

    assert plan.unit_name == experiment_session.session_name
    assert plan.unit_kind == SESSION_UNIT
    # The session carries acquisition markers but no imaging, so the checksum stage plans and two-photon is skipped.
    assert (ProcessingPipelines.CHECKSUM.value, CHECKSUM_JOB_NAME, experiment_session.session_name) in plan.entry_map()
    assert ProcessingPipelines.TWO_PHOTON.value not in {entry.pipeline for entry in plan.entries}
    assert _JobPlan.from_yaml(file_path=_session_plan_path(session=experiment_session)).entry_map() == plan.entry_map()


def test_planning_a_defined_dataset_records_its_forging_jobs(project_root: Path, training_session: SessionData) -> None:
    """Verifies that a dataset is plannable as soon as its sessions carry the outputs its jobs consume, since every
    figure it records follows from those outputs.
    """
    write_camera_clock(session=training_session, frames=1200)
    dataset = define_planned_dataset(project_root=project_root, session=training_session)

    plan = resolve_dataset_plan(dataset_path=dataset.dataset_data_path.parent, display_progress=True)

    assert plan.unit_kind == DATASET_UNIT
    assert plan.unit_name == "ds_planned"
    # A training session records no imaging at all, so its assembly job consumes the camera clock its columns are
    # placed on rather than any fluorescence.
    assert {entry.pipeline for entry in plan.entries} == {ProcessingPipelines.FORGING.value}
    assert [entry.specifier for entry in plan.entries] == [training_session.session_name]
    assert _JobPlan.from_yaml(file_path=_dataset_plan_path(dataset=dataset)).entry_map() == plan.entry_map()


def test_planning_a_dataset_covers_every_pipeline_scoped_to_a_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the pass drives every pipeline declaring the dataset unit, so a second dataset pipeline is planned
    without being named here.
    """
    dataset_path = tmp_path.joinpath("Project", "ds_a")
    dataset = make_dataset(root=dataset_path)
    dispatches = (
        make_dispatch(
            pipeline=ProcessingPipelines.FORGING, unit=dataset, universe=_RUNTIME_JOBS, unit_kind=DATASET_UNIT
        ),
        make_dispatch(
            pipeline=ProcessingPipelines.CHECKSUM, unit=dataset, universe=_CHECKSUM_JOBS, unit_kind=DATASET_UNIT
        ),
    )
    monkeypatch.setattr(planning_module, "resolve_unit_dispatches", lambda unit_kind: dispatches)  # noqa: ARG005

    plan = resolve_dataset_plan(dataset_path=dataset_path)

    assert plan.unit_kind == DATASET_UNIT
    assert {entry.pipeline for entry in plan.entries} == {
        ProcessingPipelines.FORGING.value,
        ProcessingPipelines.CHECKSUM.value,
    }


def test_a_dataset_whose_session_carries_no_processed_output_is_refused(
    project_root: Path, training_session: SessionData
) -> None:
    """Verifies that a dataset whose assembly job has nothing to read is dropped rather than planned at a floor."""
    dataset = define_planned_dataset(project_root=project_root, session=training_session)

    # The assembly stage is charged the frame it builds, which a training session places on the clock of the camera
    # it recorded. A session whose video processing wrote no clock therefore states nothing from which the stage
    # could be sized.
    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to plan the jobs") as failure:
        resolve_dataset_plan(dataset_path=dataset.dataset_data_path.parent)

    # An unmodeled figure would hand a scheduler a reservation nothing measured, so the whole dataset drops out of
    # the plan and the refusal names the session it could not read.
    assert training_session.session_name in " ".join(str(failure.value).split())
    assert not _dataset_plan_path(dataset=dataset).exists()


def test_the_projection_reports_the_units_that_carry_no_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dataset carrying no cache contributes no row, so the projection holds the planned one alone."""
    project = tmp_path.joinpath("Project")
    session = make_session(root=project.joinpath("305", "2026-01-02-03-04-05-000006"))
    planned = make_dataset(root=project.joinpath("ds_planned"))
    unplanned = make_dataset(root=project.joinpath("ds_unplanned"))

    plan_unit(
        unit_path=project.joinpath("ds_planned"),
        unit_kind=DATASET_UNIT,
        dispatches=[
            make_dispatch(
                pipeline=ProcessingPipelines.FORGING,
                unit=planned,
                universe=_RUNTIME_JOBS,
                memory_mb=6400,
                unit_kind=DATASET_UNIT,
            )
        ],
    )

    monkeypatch.setattr(
        target=planning_module,
        name="iterate_sessions",
        value=lambda root_path: [session],  # noqa: ARG005
    )
    monkeypatch.setattr(
        target=planning_module,
        name="discover_project_datasets",
        value=lambda project_root: [planned, unplanned],  # noqa: ARG005
    )

    frame = pl.read_ipc(source=generate_project_plan(project_directory=project, display_progress=True), memory_map=True)

    assert frame["dataset"].to_list() == ["ds_planned"]
    assert frame["unit_kind"].to_list() == [DATASET_UNIT]


# The prepared-batch registry


def test_a_recorded_batch_is_read_back_whole(isolated_working_directory: Path, deterministic_batch_ids: Any) -> None:
    """Verifies that the document is stored whole, so executing a recorded batch re-resolves no membership."""
    document = make_document()

    batch_id = record_prepared_batch(document=document)

    assert batch_id == "batch00".ljust(16, "0")
    assert _batch_path(batch_id=batch_id) == batch_directory().joinpath(f"{batch_id}.yaml")
    assert batch_directory().is_relative_to(isolated_working_directory)
    assert read_prepared_batch(batch_id=batch_id) == document
    assert deterministic_batch_ids.issued == ["batch00"]


def test_an_unheld_batch_identifier_resolves_to_nothing(isolated_working_directory: Path) -> None:
    """Verifies that a host that never recorded a batch answers for it, so a caller reports it as missing."""
    assert read_prepared_batch(batch_id="never_recorded") is None
    assert read_batch_outcome(batch_id="never_recorded") is None
    assert not record_batch_outcome(batch_id="never_recorded", outcome={"succeeded": 1})
    assert forget_batch_records(batch_ids=["never_recorded"]) == []


def test_reading_several_batches_reports_the_identifiers_this_host_lacks(
    isolated_working_directory: Path,
    deterministic_batch_ids: Any,
) -> None:
    """Verifies that a batch prepared elsewhere is named back to the caller, sorted, rather than dropped."""
    document = make_document()
    batch_id = record_prepared_batch(document=document)

    found, missing = read_prepared_batches(batch_ids=[batch_id, "zulu_batch", "alpha_batch"])

    assert found == [document]
    assert missing == ["alpha_batch", "zulu_batch"]


def test_a_finished_batch_answers_with_what_its_jobs_recorded(
    isolated_working_directory: Path,
    deterministic_batch_ids: Any,
) -> None:
    """Verifies that closure writes the outcome onto the batch's own file, which makes a finished batch answerable."""
    batch_id = record_prepared_batch(document=make_document())

    assert read_batch_outcome(batch_id=batch_id) is None
    assert record_batch_outcome(batch_id=batch_id, outcome={"succeeded": 3, "failed": 1})
    assert read_batch_outcome(batch_id=batch_id) == {"succeeded": 3, "failed": 1}
    # Recording an outcome leaves the document itself untouched, so the batch stays executable and readable.
    assert read_prepared_batch(batch_id=batch_id) == make_document()


def test_forgetting_a_batch_removes_its_record_and_its_lock(
    isolated_working_directory: Path,
    deterministic_batch_ids: Any,
) -> None:
    """Verifies that retiring a batch takes its whole footprint, leaving the outstanding records alone."""
    first = record_prepared_batch(document=make_document())
    second = record_prepared_batch(document=make_document())
    outstanding = record_prepared_batch(document=make_document())
    record_batch_outcome(batch_id=first, outcome={"succeeded": 1})

    removed = forget_batch_records(batch_ids=[first, "never_recorded", second])

    assert removed == [first, second]
    # The batch this call did not name keeps both its document and its lock.
    assert read_prepared_batch(batch_id=outstanding) is not None
    assert _batch_path(batch_id=outstanding).is_file()
    assert not _batch_path(batch_id=first).is_file()
    assert not _batch_path(batch_id=first).with_suffix(".yaml.lock").is_file()
    # A forget takes both halves of what a batch leaves behind, so the outcome goes with the document.
    assert not _outcome_path(batch_id=first).is_file()
    assert read_prepared_batch(batch_id=second) is None


def test_retiring_a_closed_batch_keeps_the_outcome_that_answers_for_it(
    isolated_working_directory: Path,
    deterministic_batch_ids: Any,
) -> None:
    """Verifies that closure drops the prepared document alone, since the outcome beside it is what a later caller
    reads.
    """
    batch_id = record_prepared_batch(document=make_document())
    record_batch_outcome(batch_id=batch_id, outcome={"succeeded": 2})

    retire_prepared_batch(batch_id=batch_id)

    assert read_prepared_batch(batch_id=batch_id) is None
    assert not _batch_path(batch_id=batch_id).with_suffix(".yaml.lock").is_file()
    assert read_batch_outcome(batch_id=batch_id) == {"succeeded": 2}


def test_forgetting_a_settled_batch_takes_the_outcome_it_is_held_by(
    isolated_working_directory: Path,
    deterministic_batch_ids: Any,
) -> None:
    """Verifies that a batch stays forgettable after closure retires its document, which is what keeps the outcome from
    outliving every record of the run.
    """
    batch_id = record_prepared_batch(document=make_document())
    record_batch_outcome(batch_id=batch_id, outcome={"succeeded": 2})
    retire_prepared_batch(batch_id=batch_id)

    assert forget_batch_records(batch_ids=[batch_id]) == [batch_id]
    assert read_batch_outcome(batch_id=batch_id) is None
    assert not _outcome_path(batch_id=batch_id).is_file()
    assert not _outcome_path(batch_id=batch_id).with_suffix(".yaml.lock").is_file()


def test_batches_prepared_against_one_host_resolve_to_that_host(isolated_working_directory: Path) -> None:
    """Verifies that a batch runs where it was prepared, since its jobs read the data that host holds."""
    documents = [make_document(host="workstation"), make_document(host="workstation", pipeline="runtime")]

    assert resolve_batch_host(documents=documents) == "workstation"


def test_batches_prepared_against_different_hosts_are_rejected(isolated_working_directory: Path) -> None:
    """Verifies that dispatching two hosts' batches together is refused rather than resolved to one of them."""
    documents = [make_document(host="workstation"), make_document(host="server")]

    with pytest.raises(ValueError, match="prepared against the hosts"):
        resolve_batch_host(documents=documents)


def test_an_empty_set_of_batches_names_no_host(isolated_working_directory: Path) -> None:
    """Verifies that executing nothing names no host, which stops the run rather than guessing one."""
    with pytest.raises(ValueError, match="empty set of prepared batches"):
        resolve_batch_host(documents=[])
