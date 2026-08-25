"""Provides the per-unit job plan caches that record each job's resource figures, and the project-level projection that
ships them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn
from dataclasses import field, dataclass

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import iterate_sessions
from ataraxis_data_structures import YamlConfig, ProcessingTracker, atomic_write

from ..forging import discover_project_datasets
from .dispatch import resolve_dispatch, resolve_job_cores
from ..shared_assets import SESSION_PIPELINES, ProcessingPipelines, natural_sort

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import DatasetData, SessionData

    from .dispatch import PipelineDispatch
    from .footprints import JobFootprint

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
    "job_id": pl.String,
    "job_name": pl.String,
    "specifier": pl.String,
    "cores": pl.UInt16,
    "memory_mb": pl.UInt32,
    "prerequisite_ids": pl.List(pl.String),
}
"""The column layout of the project plan projection, one row per planned job.

Notes:
    The subject columns mirror the dataset state artifact, so a reader joins a job's figures against its recorded
    state on the same keys. A session row carries no dataset and a dataset row carries neither animal nor session.

    The project job artifact keys its own rows by ``job_id``, so the two tables join on it directly. Carrying the
    ordering here as well lets a scheduler build a job's dependency graph from this table alone, without resolving
    the unit that owns the jobs.
"""


@dataclass(slots=True)
class _JobPlanEntry:
    """Records the resource figures one job occupies while it runs."""

    pipeline: str = ""
    """The pipeline that dispatches this job."""
    job_name: str = ""
    """The tracker job name identifying this job's stage."""
    specifier: str = ""
    """The specifier that differentiates this job from others of its stage within the same unit."""
    cores: int = 1
    """The cores this job occupies, as its own sizing pass resolved them. A stage that a dependency owns answers with
    the width that dependency picked for this job's input, and every other stage takes its type's declared
    allocation."""
    memory_mb: int = 0
    """The memory this job occupies, as its own sizing pass modeled it from the data the job will read."""
    prerequisite_ids: list[str] = field(default_factory=list)
    """The identifiers of the jobs that must succeed before this job may run, from its pipeline's own ordering."""

    @property
    def key(self) -> tuple[str, str, str]:
        """Returns the triple that identifies this entry within its unit's plan."""
        return self.pipeline, self.job_name, self.specifier

    @property
    def job_id(self) -> str:
        """Returns the identifier under which the processing tracker records this job."""
        return ProcessingTracker.generate_job_id(job_name=self.job_name, specifier=self.specifier)


@dataclass
class _JobPlan(YamlConfig):
    """Records the resource figures every job of one processing unit occupies.

    Notes:
        Replanning estimates only the jobs the plan does not already hold, so widening a unit's job universe appends
        the new jobs alone. A caller that wants recorded figures re-estimated regenerates the plan, which is the one
        path through this module that changes them.

        Each write to a plan file takes that file's own lock, and the lock spans a single write, so a run reading a
        plan assumes it is the plan against which its submissions were sized.
    """

    unit_name: str = ""
    """The name of the unit this plan describes."""
    unit_kind: str = ""
    """The kind of unit this plan describes, either a session or a dataset."""
    entries: list[_JobPlanEntry] = field(default_factory=list)
    """The planned jobs, one entry per job the unit's pipelines resolve."""

    def entry_map(self) -> dict[tuple[str, str, str], _JobPlanEntry]:
        """Returns the plan's entries keyed by their identifying triple."""
        return {entry.key: entry for entry in self.entries}


def _session_plan_path(session: SessionData) -> Path:
    """Resolves the path to a session's job plan cache.

    Args:
        session: The loaded session whose plan cache to locate.

    Returns:
        The path to the session's job plan file, beside the outputs its jobs produce.
    """
    return session.processed_data_path.joinpath(_PLAN_FILENAME)


def _dataset_plan_path(dataset: DatasetData) -> Path:
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


def resolve_session_plan(
    session_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> _JobPlan:
    """Plans every processing job of one session, estimating only the jobs the cache does not already hold.

    Notes:
        Estimation reads the session's raw acquisition data, so it costs far more than reading the cache it writes.
        A pipeline that resolves no jobs for this session is skipped, which is the ordinary case for a session
        carrying no imaging or no camera data.

    Args:
        session_path: The path to the session root directory to plan.
        regenerate_plan: Determines whether to re-estimate the jobs the cache already holds. Leave False to keep
            every recorded figure, since a submission may already have been sized against it.
        display_progress: Determines whether to report the pipelines that resolved no jobs for this session and why.

    Returns:
        The session's plan, holding an entry for every job its pipelines resolve.

    Raises:
        ValueError: If no pipeline resolves any job for this session, since a session with no plannable job names no
            path to a plan file.
        TimeoutError: If a pipeline's processing tracker lock or the plan file's own lock cannot be acquired within
            the timeout period.
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
        regenerate_plan=regenerate_plan,
        display_progress=display_progress,
    )


def resolve_dataset_plan(
    dataset_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> _JobPlan:
    """Plans every forging job of one dataset, estimating only the jobs the cache does not already hold.

    Notes:
        A dataset's figures follow from the single-day outputs its jobs consume, and admission requires a session to
        carry those outputs already, so a dataset is plannable as soon as its hierarchy is defined.

    Args:
        dataset_path: The path to the dataset's root directory to plan.
        regenerate_plan: Determines whether to re-estimate the jobs the cache already holds.
        display_progress: Determines whether to report the reason when the forging pipeline resolves no jobs.

    Returns:
        The dataset's plan, holding an entry for every forging job it resolves.

    Raises:
        ValueError: If the forging pipeline resolves no job for this dataset, since a dataset with no plannable job
            names no path to a plan file.
        TimeoutError: If a pipeline's processing tracker lock or the plan file's own lock cannot be acquired within
            the timeout period.
    """
    dispatch = resolve_dispatch(pipeline=ProcessingPipelines.FORGING)
    dispatches = [] if dispatch is None else [dispatch]
    return _resolve_unit_plan(
        dispatches=dispatches,
        unit_path=dataset_path,
        unit_kind=DATASET_UNIT,
        regenerate_plan=regenerate_plan,
        display_progress=display_progress,
    )


def generate_project_plan(project_directory: Path, *, display_progress: bool = False) -> Path:
    """Projects every plan cache under a project into one table and saves it at the project root.

    Notes:
        Reads the caches alone, so this is the cheap half of planning and the half that ships. A unit carrying no
        cache contributes no rows, so a reader sizing a submission against this table treats an absent job as
        unplanned rather than as free.

    Args:
        project_directory: The path to the project whose plan caches to project.
        display_progress: Determines whether to report what the projection covered once it is written.

    Returns:
        Where the projection was written.

    Raises:
        FileNotFoundError: If the project directory does not exist, since a projection has nowhere to be written and
            no unit to read.
        Timeout: If the projection file's lock cannot be acquired within the timeout period.
    """
    # Session discovery walks the tree and reports a root it cannot read, so a missing project is named here rather
    # than surfacing as a walk failure partway through the projection.
    if not project_directory.is_dir():
        message = (
            f"Unable to project the job plans of '{project_directory}'. The path does not name an existing "
            f"directory, so the project holds neither a readable unit nor a destination for the projection."
        )
        console.error(message=message, error=FileNotFoundError)

    rows: list[dict[str, Any]] = []
    planned_units = 0
    unplanned_units = 0

    for session in iterate_sessions(root_path=project_directory):
        plan = _load_plan(plan_path=_session_plan_path(session=session))
        if plan is None:
            unplanned_units += 1
            continue
        planned_units += 1
        rows.extend(
            _projection_row(entry=entry, animal=str(session.animal_id), session=session.session_name, dataset=None)
            for entry in plan.entries
        )

    for dataset in discover_project_datasets(project_root=project_directory):
        plan = _load_plan(plan_path=_dataset_plan_path(dataset=dataset))
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
        frame = natural_sort(
            frame=pl.DataFrame(data=rows, schema=PROJECT_PLAN_SCHEMA, strict=False),
            by=["unit_kind", "animal", "session", "dataset", "pipeline", "job_name", "specifier"],
            nulls_last=True,
        )
        # Published through a temporary file renamed over the destination. The lock serializes the writers, while
        # the readers memory-map the projection without taking it, so only the rename keeps them off a torn file.
        with atomic_write(file_path=plan_path, binary=True) as file:
            frame.write_ipc(file=file, compression="uncompressed")

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
    dispatches: list[PipelineDispatch[Any]],
    unit_path: Path,
    unit_kind: str,
    *,
    regenerate_plan: bool,
    display_progress: bool,
) -> _JobPlan:
    """Plans one unit across the pipelines that operate on it, preserving every figure already recorded.

    Notes:
        A pipeline whose resolver rejects this unit is skipped rather than failing the whole plan, since a resolver
        raises precisely when the unit carries none of the data that pipeline consumes. Each skip is reported with the
        reason its resolver gave, so a pipeline absent because its input is malformed is distinguishable from one
        absent because the unit never carried that data.

        Sizing takes the same path. Every job is modeled from the data it will read, so a job whose input cannot be
        read is refused rather than planned at a figure nothing measured. That refusal names the input, and it drops
        the job's whole pipeline out of this unit's plan, because a pipeline that cannot size one of its stages
        cannot state what the unit costs to run. The reason lands in the same skip report a rejected resolver fills,
        so a caller reads one account of everything this unit did not plan and why.

        Sizing therefore runs before a pipeline's tracker is aligned, so a dropped pipeline registers no job the plan
        does not cover. Each surviving pipeline's processing tracker is aligned with the jobs the unit can actually
        run, so a unit that has never been processed still carries a job registry once it is planned. The project job
        artifact is built from that registry, which is how a scheduler on another host learns which jobs exist. A job
        the unit cannot run never reaches the tracker, so its absence there is the statement that it is not possible.
        A pipeline that resolves a universe but no runnable job therefore writes no tracker rather than failing the
        plan, since its figures still belong in the cache the plan records.

        The recorded figures cover the whole universe while the tracker holds the possible subset, so a plan describes
        every job the pipeline defines and the job artifact states which of them this unit supports.

    Args:
        dispatches: The dispatch entries of the pipelines that operate on this kind of unit.
        unit_path: The path to the unit to plan.
        unit_kind: Whether the unit is a session or a dataset.
        regenerate_plan: Determines whether to re-estimate the jobs the cache already holds.
        display_progress: Determines whether to report the pipelines that planned no jobs and why.

    Returns:
        The unit's plan as it now stands on disk.

    Raises:
        ValueError: If no pipeline plans any job for this unit, since a unit with no plannable job names no path to a
            plan file, or if a pipeline requests a job outside its own declared universe.
        TimeoutError: If a pipeline's processing tracker lock cannot be acquired within the timeout period.
    """
    # Resolves every pipeline's job set first, so the recorded plan seeds the entry set before any pipeline's
    # outstanding jobs are worked out against it.
    resolved: list[tuple[PipelineDispatch[Any], Any, list[tuple[str, str]], list[tuple[str, str]]]] = []
    located: tuple[Path, str] | None = None
    skipped: dict[str, str] = {}
    for dispatch in dispatches:
        discovered = _discover_unit(dispatch=dispatch, unit_path=unit_path, skipped=skipped)
        if discovered is None:
            continue
        unit, universe, possible = discovered
        if located is None:
            unit_plan_path = (
                _session_plan_path(session=unit) if unit_kind == SESSION_UNIT else _dataset_plan_path(dataset=unit)
            )
            located = (unit_plan_path, dispatch.unit_name(unit))
        resolved.append((dispatch, unit, universe, possible))

    if located is None:
        _reject_unit(unit_path=unit_path, unit_kind=unit_kind, skipped=skipped)

    plan_path, unit_name = located
    recorded = _load_plan(plan_path=plan_path)
    entries: dict[tuple[str, str, str], _JobPlanEntry] = (
        {} if recorded is None or regenerate_plan else dict(recorded.entry_map())
    )

    planned_pipelines = 0
    for dispatch, unit, universe, possible in resolved:
        # The declared allocation reaches the sizing pass as the answer for a stage that holds one width whatever
        # data it reads, and a stage that a dependency sizes overrides it with the width that dependency picked.
        declared = {job_name: resolve_job_cores(job_name=job_name) for job_name, _ in universe}
        outstanding = [
            (job_name, specifier)
            for job_name, specifier in universe
            if (dispatch.pipeline.value, job_name, specifier) not in entries
        ]

        # Sizing precedes every write this pipeline makes, so a pipeline whose input cannot be read leaves neither a
        # tracker nor a plan entry behind and is reported alongside the pipelines whose resolvers rejected the unit.
        footprints = _size_unit(dispatch=dispatch, unit=unit, jobs=outstanding, declared=declared, skipped=skipped)
        if footprints is None:
            continue
        planned_pipelines += 1

        # Registers the jobs this unit can run, so the job artifact built from this tracker enumerates them. A
        # pipeline that resolves no possible job contributes no registry at all, since a tracker states which jobs a
        # unit supports and an empty registry states nothing.
        if possible:
            tracker_path = dispatch.tracker_path(unit)
            tracker_path.parent.mkdir(parents=True, exist_ok=True)
            ProcessingTracker(file_path=tracker_path).align_jobs(jobs=possible, universe=universe)

        # Ordering resolves over the whole universe, so every recorded edge is the pipeline's own, independent of what
        # this unit happened to carry when it was planned. A consumer drops the edges whose upstream job carries no
        # recorded state, which is how a stage stops waiting on a job the unit can never produce.
        ordering = dispatch.prerequisites(unit, universe)

        for job_name, specifier in outstanding:
            footprint = footprints[job_name, specifier]
            entry = _JobPlanEntry(
                pipeline=dispatch.pipeline.value,
                job_name=job_name,
                specifier=specifier,
                cores=footprint.cores,
                memory_mb=footprint.memory_mb,
                prerequisite_ids=[
                    ProcessingTracker.generate_job_id(job_name=upstream_name, specifier=upstream_specifier)
                    for upstream_name, upstream_specifier in ordering.get((job_name, specifier), ())
                ],
            )
            entries[entry.key] = entry

    if not planned_pipelines:
        _reject_unit(unit_path=unit_path, unit_kind=unit_kind, skipped=skipped)

    if display_progress and skipped:
        for pipeline, reason in skipped.items():
            console.echo(message=f"Pipeline '{pipeline}': Planned no job for '{unit_path}'. {reason}")

    plan = _JobPlan(unit_name=unit_name, unit_kind=unit_kind, entries=[entries[key] for key in natsorted(entries)])
    _save_plan(plan=plan, plan_path=plan_path)
    return plan


def _discover_unit(
    dispatch: PipelineDispatch[Any], unit_path: Path, skipped: dict[str, str]
) -> tuple[Any, list[tuple[str, str]], list[tuple[str, str]]] | None:
    """Primes a unit if its pipeline needs it, then resolves the pipeline's job sets, recording the reason when the
    pipeline resolves nothing.

    Notes:
        Priming belongs to planning, because planning is the preparation pass that runs before anything reads a unit's
        job set. A pipeline whose job model lives in state that a dependency writes therefore has that state
        materialized here. Priming is idempotent, so a unit already carrying what it needs is read rather than
        rewritten, and resolution itself stays read-only.

    Args:
        dispatch: The pipeline's dispatch entry.
        unit_path: The path to the unit whose jobs to resolve.
        skipped: The mapping into which this call records its pipeline's reason when resolution does not succeed.

    Returns:
        The loaded unit, its job universe, and the subset it can run, or None when this pipeline resolves no job for
        the unit.
    """
    try:
        if dispatch.prime is not None:
            dispatch.prime(unit_path)
        unit, universe, possible = dispatch.discover(unit_path)
    except Exception as exception:
        skipped[dispatch.pipeline.value] = str(exception)
        return None
    return unit, universe, possible


def _size_unit(
    dispatch: PipelineDispatch[Any],
    unit: Any,
    jobs: list[tuple[str, str]],
    declared: dict[str, int],
    skipped: dict[str, str],
) -> dict[tuple[str, str], JobFootprint] | None:
    """Sizes the jobs of one pipeline, recording the reason when an input the sizing pass reads cannot be read.

    Notes:
        Every job is modeled from the data it will process, so the sizing pass refuses a job whose input is absent,
        ambiguous, or unparsable rather than answering with a flat allowance. That refusal is the statement that the
        pipeline cannot say what this unit costs, so the whole pipeline drops out of the plan and its reason joins the
        reasons the resolvers gave. Reporting through the same map keeps a dropped pipeline visible to a caller
        instead of silently absent.

    Args:
        dispatch: The pipeline's dispatch entry.
        unit: The loaded unit on which the jobs operate.
        jobs: The jobs to size, as ``(job_name, specifier)`` pairs.
        declared: The cores each job type declares, which answer for a stage that holds one width whatever data it
            reads.
        skipped: The mapping into which this call records its pipeline's reason when sizing does not succeed.

    Returns:
        The footprint of every job this call was handed, or None when the pipeline cannot size one of them.
    """
    # A pipeline whose jobs are already recorded reads nothing, so it keeps its tracker and its recorded figures
    # without paying for a pass that would answer about jobs the plan does not need.
    if not jobs:
        return {}
    try:
        return dispatch.size_jobs(unit, [(job_name, specifier, declared[job_name]) for job_name, specifier in jobs])
    except Exception as exception:
        skipped[dispatch.pipeline.value] = str(exception)
        return None


def _reject_unit(unit_path: Path, unit_kind: str, skipped: dict[str, str]) -> NoReturn:
    """Reports a unit no pipeline plans, naming what each pipeline reported.

    Args:
        unit_path: The path to the unit that was planned.
        unit_kind: Whether the unit is a session or a dataset.
        skipped: Each pipeline's reason for contributing nothing.

    Raises:
        ValueError: Always, since a unit with no plannable job names no path to a plan file.
    """
    message = (
        f"Unable to plan the jobs of '{unit_path}'. No pipeline planned any job for it, so the unit either carries "
        f"none of the data consumed by the pipelines that operate on a {unit_kind}, or carries it in a state that "
        f"none of them can read. Each pipeline reported: {skipped}."
    )
    console.error(message=message, error=ValueError)


def _load_plan(plan_path: Path) -> _JobPlan | None:
    """Reads a unit's plan cache.

    Args:
        plan_path: The path to the unit's plan file.

    Returns:
        The recorded plan, or None when the unit carries no plan file.
    """
    if not plan_path.is_file():
        return None
    return _JobPlan.from_yaml(file_path=plan_path)


def _save_plan(plan: _JobPlan, plan_path: Path) -> None:
    """Writes a unit's plan cache under its own lock.

    Args:
        plan: The plan to record.
        plan_path: Where to write it.

    Raises:
        Timeout: If the plan file's lock cannot be acquired within the timeout period.
    """
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(plan_path.with_suffix(plan_path.suffix + ".lock")))
    with lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        plan.to_yaml(file_path=plan_path)


def _projection_row(
    entry: _JobPlanEntry, animal: str | None, session: str | None, dataset: str | None
) -> dict[str, Any]:
    """Renders one plan entry as a row of the project projection.

    Args:
        entry: The planned job to render.
        animal: The animal that owns the job's session, or None for a dataset job.
        session: The session the job processes, or None for a dataset job.
        dataset: The dataset that owns the job, or None for a session job.

    Returns:
        The projection row for this entry.
    """
    return {
        "unit_kind": DATASET_UNIT if dataset is not None else SESSION_UNIT,
        "animal": animal,
        "session": session,
        "dataset": dataset,
        "pipeline": entry.pipeline,
        "job_id": entry.job_id,
        "job_name": entry.job_name,
        "specifier": entry.specifier,
        "cores": entry.cores,
        "memory_mb": entry.memory_mb,
        "prerequisite_ids": list(entry.prerequisite_ids),
    }
