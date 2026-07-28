"""Provides the per-unit job plan caches that record each job's resource figures, and the project-level projection
that ships them. Replanning keeps the figures a cache already holds, so the cores and memory a job is submitted with
stay the ones it was planned with. Re-estimating them is a deliberate act, requested through a forced replan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import DatasetData, iterate_sessions
from ataraxis_data_structures import YamlConfig

from .dispatch import resolve_dispatch, resolve_job_cores
from ..shared_assets import SESSION_PIPELINES, ProcessingPipelines

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

    from .dispatch import PipelineDispatch

_PLAN_FILENAME: str = "job_plan.yaml"
"""The filename of a unit's job plan cache, written beside the outputs the unit's jobs produce."""

_LOCK_TIMEOUT_SECONDS: float = 20.0
"""The period a writer waits for a plan file's lock before giving up, matching the project manifest's writer."""

SESSION_UNIT: str = "session"
"""The unit label of a plan describing the jobs of one acquisition session."""

DATASET_UNIT: str = "dataset"
"""The unit label of a plan describing the jobs of one forged dataset."""

PROJECT_PLAN_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
    "unit_kind": pl.String,
    "animal": pl.String,
    "session": pl.String,
    "dataset": pl.String,
    "pipeline": pl.String,
    "job_name": pl.String,
    "specifier": pl.String,
    "cores": pl.UInt16,
    "memory_mb": pl.UInt32,
    "memory_modeled": pl.Boolean,
}
"""The column layout of the project plan projection, one row per planned job.

Notes:
    The subject columns mirror the dataset state artifact, so a reader joins a job's figures against its recorded
    state on the same keys. A session row carries no dataset and a dataset row carries neither animal nor session.
"""


@dataclass
class JobPlanEntry:
    """Records the resource figures one job occupies while it runs."""

    pipeline: str = ""
    """The pipeline that dispatches this job."""
    job_name: str = ""
    """The tracker job name identifying this job's stage."""
    specifier: str = ""
    """The specifier that differentiates this job from others of its stage within the same unit."""
    cores: int = 1
    """The cores this job occupies, from its type's declared allocation."""
    memory_mb: int = 0
    """The memory this job occupies, estimated from the data it will process."""
    memory_modeled: bool = False
    """Determines whether the memory figure follows from this job's own input rather than from a flat allowance."""

    @property
    def key(self) -> tuple[str, str, str]:
        """Returns the triple that identifies this entry within its unit's plan."""
        return self.pipeline, self.job_name, self.specifier


@dataclass
class JobPlan(YamlConfig):
    """Records the resource figures every job of one processing unit occupies.

    Notes:
        Replanning estimates only the jobs the plan does not already hold, so widening a unit's job universe appends
        the new jobs alone. A caller that wants recorded figures re-estimated requests a forced replan, which is the
        one path through this module that changes them.

        Nothing guards the file itself, so a run reading a plan assumes it is the plan its submissions were sized
        against.
    """

    unit_name: str = ""
    """The name of the unit this plan describes."""
    unit_kind: str = ""
    """Whether this plan describes a session or a dataset."""
    entries: list[JobPlanEntry] = field(default_factory=list)
    """The planned jobs, one entry per job the unit's pipelines resolve."""

    def entry_map(self) -> dict[tuple[str, str, str], JobPlanEntry]:
        """Returns the plan's entries keyed by their identifying triple."""
        return {entry.key: entry for entry in self.entries}


def session_plan_path(session: SessionData) -> Path:
    """Resolves the path to a session's job plan cache.

    Args:
        session: The loaded session whose plan cache to locate.

    Returns:
        The path to the session's job plan file, beside the outputs its jobs produce.
    """
    return session.processed_data_path.joinpath(_PLAN_FILENAME)


def dataset_plan_path(dataset: DatasetData) -> Path:
    """Resolves the path to a dataset's job plan cache.

    Args:
        dataset: The resolved dataset whose plan cache to locate.

    Returns:
        The path to the dataset's job plan file, at the dataset root beside its forging tracker.
    """
    return dataset.dataset_data_path.parent.joinpath(_PLAN_FILENAME)


def project_plan_path(project_directory: Path) -> Path:
    """Resolves the path to a project's job plan projection.

    Args:
        project_directory: The path to the project's root directory.

    Returns:
        The path to the project's plan .feather file.
    """
    return project_directory.joinpath(f"{project_directory.stem}_plan.feather")


def resolve_session_plan(session_path: Path, *, force: bool = False, display_progress: bool = False) -> JobPlan:
    """Plans every processing job of one session, estimating only the jobs the cache does not already hold.

    Notes:
        Estimation reads the session's raw acquisition data, so it costs far more than reading the cache it writes.
        A pipeline that resolves no jobs for this session is skipped, which is the ordinary case for a session
        carrying no imaging or no camera data.

    Args:
        session_path: The path to the session root directory to plan.
        force: Determines whether to re-estimate the jobs the cache already holds. Leave False to keep every recorded
            figure, since a submission may already have been sized against it.
        display_progress: Determines whether to report the pipelines that resolved no jobs for this session and why.

    Returns:
        The session's plan, holding an entry for every job its pipelines resolve.
    """
    dispatches = [
        dispatch
        for dispatch in (resolve_dispatch(pipeline=pipeline) for pipeline in SESSION_PIPELINES)
        if dispatch is not None
    ]
    return _resolve_unit_plan(
        dispatches=dispatches,
        unit_path=session_path,
        unit_kind=SESSION_UNIT,
        force=force,
        display_progress=display_progress,
    )


def resolve_dataset_plan(dataset_path: Path, *, force: bool = False, display_progress: bool = False) -> JobPlan:
    """Plans every forging job of one dataset, estimating only the jobs the cache does not already hold.

    Notes:
        A dataset's figures follow from the single-day outputs its jobs consume, and admission requires a session to
        carry those outputs already, so a dataset is plannable as soon as its hierarchy is defined.

    Args:
        dataset_path: The path to the dataset's root directory to plan.
        force: Determines whether to re-estimate the jobs the cache already holds.
        display_progress: Determines whether to report the reason when the forging pipeline resolves no jobs.

    Returns:
        The dataset's plan, holding an entry for every forging job it resolves.
    """
    dispatch = resolve_dispatch(pipeline=ProcessingPipelines.FORGING)
    dispatches = [] if dispatch is None else [dispatch]
    return _resolve_unit_plan(
        dispatches=dispatches,
        unit_path=dataset_path,
        unit_kind=DATASET_UNIT,
        force=force,
        display_progress=display_progress,
    )


def generate_project_plan(project_directory: Path, *, display_progress: bool = False) -> Path:
    """Projects every plan cache under a project into one table and saves it at the project root.

    Notes:
        Reads the caches alone and estimates nothing, so this is the cheap half of planning and the half that ships.
        A unit carrying no cache contributes no rows, so a reader sizing a submission against this table treats an
        absent job as unplanned rather than as free.

    Args:
        project_directory: The path to the project whose plan caches to project.
        display_progress: Determines whether to report what the projection covered once it is written.

    Returns:
        The path the projection was written to.
    """
    rows: list[dict[str, Any]] = []
    planned_units = 0
    unplanned_units = 0

    for session in iterate_sessions(root_path=project_directory):
        plan = _load_plan(plan_path=session_plan_path(session=session))
        if plan is None:
            unplanned_units += 1
            continue
        planned_units += 1
        rows.extend(
            _projection_row(entry=entry, animal=str(session.animal_id), session=session.session_name, dataset=None)
            for entry in plan.entries
        )

    for dataset in _iterate_datasets(project_directory=project_directory):
        plan = _load_plan(plan_path=dataset_plan_path(dataset=dataset))
        if plan is None:
            unplanned_units += 1
            continue
        planned_units += 1
        rows.extend(
            _projection_row(entry=entry, animal=None, session=None, dataset=dataset.name) for entry in plan.entries
        )

    plan_path = project_plan_path(project_directory=project_directory)
    lock = FileLock(str(plan_path.with_suffix(plan_path.suffix + ".lock")))
    with lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        pl.DataFrame(data=rows, schema=PROJECT_PLAN_SCHEMA, strict=False).sort(
            by=["unit_kind", "animal", "session", "dataset", "pipeline", "job_name", "specifier"], nulls_last=True
        ).write_ipc(file=plan_path, compression="uncompressed")

    if display_progress:
        console.echo(
            message=(
                f"Project '{project_directory.stem}' plan: Projected {len(rows)} job(s) from {planned_units} planned "
                f"unit(s), with {unplanned_units} unit(s) carrying no plan."
            ),
            level=LogLevel.SUCCESS,
        )
    return plan_path


def _resolve_unit_plan(
    dispatches: list[PipelineDispatch[Any]], unit_path: Path, unit_kind: str, *, force: bool, display_progress: bool
) -> JobPlan:
    """Plans one unit across the pipelines that operate on it, preserving every figure already recorded.

    Notes:
        A pipeline whose resolver rejects this unit is skipped rather than failing the whole plan, since a resolver
        raises precisely when the unit carries none of the data that pipeline consumes. Each skip is reported with the
        reason its resolver gave, so a pipeline absent because its input is malformed is distinguishable from one
        absent because the unit never carried that data.

    Args:
        dispatches: The dispatch entries of the pipelines that operate on this kind of unit.
        unit_path: The path to the unit to plan.
        unit_kind: Whether the unit is a session or a dataset.
        force: Determines whether to re-estimate the jobs the cache already holds.
        display_progress: Determines whether to report the pipelines that resolved no jobs and why.

    Returns:
        The unit's plan as it now stands on disk.

    Raises:
        ValueError: If no pipeline resolves any job for this unit, since a unit with no plannable job names no path
            to a plan file.
    """
    # Resolves every pipeline's job set first, so the recorded plan seeds the entry set before any pipeline's
    # outstanding jobs are worked out against it.
    resolved: list[tuple[PipelineDispatch[Any], Any, list[tuple[str, str]]]] = []
    located: tuple[Path, str] | None = None
    skipped: dict[str, str] = {}
    for dispatch in dispatches:
        discovered = _discover_unit(dispatch=dispatch, unit_path=unit_path, skipped=skipped)
        if discovered is None:
            continue
        unit, universe = discovered
        if located is None:
            unit_plan_path = (
                session_plan_path(session=unit) if unit_kind == SESSION_UNIT else dataset_plan_path(dataset=unit)
            )
            located = (unit_plan_path, dispatch.unit_name(unit))
        resolved.append((dispatch, unit, universe))

    if located is None:
        message = (
            f"Unable to plan the jobs of '{unit_path}'. No pipeline resolved any job for it, so the unit carries "
            f"none of the data the pipelines that operate on a {unit_kind} consume. Each pipeline reported: "
            f"{skipped}."
        )
        console.error(message=message, error=ValueError)

    if display_progress and skipped:
        for pipeline, reason in skipped.items():
            console.echo(message=f"Pipeline '{pipeline}': Resolved no job for '{unit_path}'. {reason}")

    plan_path, unit_name = located
    recorded = _load_plan(plan_path=plan_path)
    entries: dict[tuple[str, str, str], JobPlanEntry] = {} if recorded is None or force else dict(recorded.entry_map())

    for dispatch, unit, universe in resolved:
        cores = {job_name: resolve_job_cores(job_name=job_name) for job_name, _ in universe}
        outstanding = [
            (job_name, specifier)
            for job_name, specifier in universe
            if (dispatch.pipeline.value, job_name, specifier) not in entries
        ]
        if not outstanding:
            continue

        estimates = dispatch.estimate_memory(
            unit, [(job_name, specifier, cores[job_name]) for job_name, specifier in outstanding]
        )
        for job_name, specifier in outstanding:
            memory_mb, memory_modeled = estimates.get((job_name, specifier), (0, False))
            entry = JobPlanEntry(
                pipeline=dispatch.pipeline.value,
                job_name=job_name,
                specifier=specifier,
                cores=cores[job_name],
                memory_mb=memory_mb,
                memory_modeled=memory_modeled,
            )
            entries[entry.key] = entry

    plan = JobPlan(unit_name=unit_name, unit_kind=unit_kind, entries=[entries[key] for key in natsorted(entries)])
    _save_plan(plan=plan, plan_path=plan_path)
    return plan


def _discover_unit(
    dispatch: PipelineDispatch[Any], unit_path: Path, skipped: dict[str, str]
) -> tuple[Any, list[tuple[str, str]]] | None:
    """Resolves one pipeline's job universe for a unit, recording the reason when the pipeline resolves nothing.

    Args:
        dispatch: The pipeline's dispatch entry.
        unit_path: The path to the unit to resolve jobs for.
        skipped: The mapping this call records its pipeline's reason into when resolution does not succeed.

    Returns:
        The loaded unit and its job universe, or None when this pipeline resolves no job for the unit.
    """
    try:
        unit, universe, _ = dispatch.discover(unit_path)
    except Exception as exception:
        skipped[dispatch.pipeline.value] = str(exception)
        return None
    return unit, universe


def _load_plan(plan_path: Path) -> JobPlan | None:
    """Reads a unit's plan cache.

    Args:
        plan_path: The path to the unit's plan file.

    Returns:
        The recorded plan, or None when the unit carries no plan file.
    """
    if not plan_path.is_file():
        return None
    return JobPlan.from_yaml(file_path=plan_path)


def _save_plan(plan: JobPlan, plan_path: Path) -> None:
    """Writes a unit's plan cache under its own lock.

    Args:
        plan: The plan to record.
        plan_path: The path to write it to.

    Raises:
        Timeout: If the plan file's lock cannot be acquired within the timeout period.
    """
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(plan_path.with_suffix(plan_path.suffix + ".lock")))
    with lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        plan.to_yaml(file_path=plan_path)


def _iterate_datasets(project_directory: Path) -> list[DatasetData]:
    """Loads every forged dataset under a project root.

    Notes:
        A forged dataset is a top-level directory carrying a ``dataset.yaml`` marker, which is how the project
        manifest discovers them too.

    Args:
        project_directory: The path to the project's root directory.

    Returns:
        The loaded datasets, ordered by directory name.
    """
    return [
        DatasetData.load(dataset_path=directory)
        for directory in natsorted(project_directory.iterdir(), key=lambda path: path.name)
        if directory.is_dir() and directory.joinpath("dataset.yaml").is_file()
    ]


def _projection_row(
    entry: JobPlanEntry, animal: str | None, session: str | None, dataset: str | None
) -> dict[str, Any]:
    """Renders one plan entry as a row of the project projection.

    Args:
        entry: The planned job to render.
        animal: The animal the job's session belongs to, or None for a dataset job.
        session: The session the job processes, or None for a dataset job.
        dataset: The dataset the job belongs to, or None for a session job.

    Returns:
        The projection row for this entry.
    """
    return {
        "unit_kind": DATASET_UNIT if dataset is not None else SESSION_UNIT,
        "animal": animal,
        "session": session,
        "dataset": dataset,
        "pipeline": entry.pipeline,
        "job_name": entry.job_name,
        "specifier": entry.specifier,
        "cores": entry.cores,
        "memory_mb": entry.memory_mb,
        "memory_modeled": entry.memory_modeled,
    }
