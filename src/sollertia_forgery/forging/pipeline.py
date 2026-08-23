"""Provides the system-agnostic, end-to-end dataset forging pipeline that runs the cross-recording cell-tracking stages
and assembles the data.feather file for each session.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from contextlib import nullcontext
from collections import deque
from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

from cindra import (
    MULTI_RECORDING_CONFIGURATION_FILENAME,
    MultiRecordingJobNames,
    prime_dataset,
    execute_multi_recording_job,
    resolve_multi_recording_jobs,
    resolve_multi_recording_prerequisites,
)
import polars as pl
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import (
    DATASET_MARKER_FILENAME,
    DatasetData,
    SessionData,
    DatasetFiles,
    RawDataFiles,
    ProcessingTrackers,
)
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    limit_worker_threads,
    initialize_worker_threads,
)

from .dataset import resolve_dataset
from ..registries import resolve_forging_assembly_worker, resolve_multi_recording_configuration_resolver
from ..shared_assets import verify_openmp_runtime, multi_recording_dataset_name

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Collection

    from sollertia_shared_assets import DatasetSession

    from ..registries import ForgingAssembler

MULTIDAY_DISCOVERY_JOB_NAME: str = "multiday_discovery"
"""The job name identifying a per-animal cross-recording cell-discovery job in the forging tracker. The job's
specifier is the animal identifier, and the job exists only for animals whose acquisition system resolves a
multi-recording configuration."""

MULTIDAY_EXTRACTION_JOB_NAME: str = "multiday_extraction"
"""The job name identifying a per-session aligned-fluorescence extraction job in the forging tracker. The job's
specifier is the session name, and the job runs after its animal's discovery job."""

FORGING_JOB_NAME: str = "session_data_assembly"
"""The job name identifying per-session assembly jobs in the forging tracker."""

FORGING_JOB_CONCURRENCY_LIMITS: dict[str, int] = {
    # Reads every fluorescence array and sub-dataset feather its session holds, then writes the merged result, and
    # computes almost nothing between the two.
    FORGING_JOB_NAME: 4,
}
"""The jobs of each forging type that may run at once regardless of the cores a budget could still supply, keyed by
the tracker job name.

Notes:
    Only assembly declares a ceiling. It holds one core per job, so this is what sets the width of the pool it
    opens. The cross-recording jobs exist for two-photon sessions alone, and each of them takes a wide core
    allocation of its own, so the core budget already bounds how many run at once and no separate ceiling applies.

    This mapping is the single source for both the local assembly pool and the shared batch layer, which merges it
    into its own concurrency table.
"""

_MULTIDAY_JOB_NAMES: dict[MultiRecordingJobNames, str] = {
    MultiRecordingJobNames.DISCOVER: MULTIDAY_DISCOVERY_JOB_NAME,
    MultiRecordingJobNames.EXTRACT: MULTIDAY_EXTRACTION_JOB_NAME,
}
"""The forging tracker job name each cindra cross-recording stage is recorded under, keyed by the cindra job name.

Notes:
    cindra owns the cross-recording pipeline, while the forging tracker interleaves those stages with the per-session
    assembly stage this library owns and records all of them under its own names. This table is the only place the two
    vocabularies meet, so composing another cindra stage into the forging graph is a matter of naming it here.
"""


@dataclass(frozen=True, slots=True)
class _MultidayStage:
    """Bundles one cindra cross-recording stage with everything the forging pipeline needs to dispatch it."""

    configuration_path: Path
    """The path to the materialized multi-recording configuration of the animal the stage belongs to."""
    job_name: MultiRecordingJobNames
    """The cindra job the stage executes."""
    specifier: str
    """The cindra specifier the stage executes under, which is the recording identifier for a per-recording stage and
    an empty string for a stage spanning the whole animal.
    """
    prime: bool
    """Determines whether the stage writes the shared multi-recording bootstrap before it runs."""


def define_forging_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    display_progress: bool = False,
    force_recreate: bool = False,
    recreate_animals: tuple[str, ...] = (),
) -> DatasetData:
    """Builds the dataset hierarchy and materializes each tracked animal's multi-recording configuration.

    Notes:
        Every forging job runs against a hierarchy this call established, so it precedes them rather than joining
        them on the processing tracker. Rebuilding an animal resets that animal's tracked jobs, since its new
        session set makes every stage outstanding again. The tracked jobs of a session the animal no longer holds
        fall outside the resulting universe and are discarded when the next pipeline run aligns the tracker.

        The animals this call adds or rebuilds have their multi-recording configuration materialized, alongside any the
        dataset holds without one on disk. An animal that already carries its configuration is left alone, so extending
        a dataset reads no source data for it and an animal whose sessions have moved off this machine does not block
        the growth of a dataset it was already forged into.

    Args:
        name: The unique name of the dataset.
        session_names: The session names the dataset must contain. A session the dataset does not hold is appended,
            subject to the resolution policy.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        display_progress: The progress-bar flag recorded in each materialized configuration.
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions the provided
            list holds for them.

    Returns:
        The resolved dataset.

    Raises:
        ValueError: If the arguments contradict each other, or if the dataset's acquisition system is unknown. The
            dataset resolution policy raises for a request it cannot satisfy.
        FileNotFoundError: If a provided session name resolves to no directory under the project root, or if the
            acquisition system's resolver reports a missing input it needs for an animal.
        RuntimeError: If a provided session name resolves to more than one directory under the project root.
    """
    console.echo(message=f"Defining the '{name}' dataset...", level=LogLevel.INFO)

    # Captured before resolution, so the animals this call adds are the ones absent from this set afterwards.
    existing_animals: frozenset[str] = frozenset()
    dataset_directory = project_root.joinpath(name)
    if not force_recreate and dataset_directory.joinpath(DATASET_MARKER_FILENAME).is_file():
        existing = DatasetData.load(dataset_path=dataset_directory)
        existing_animals = frozenset(dataset_animal.animal for dataset_animal in existing.animals)

    dataset = resolve_dataset(
        name=name,
        session_names=session_names,
        project_root=project_root,
        force_recreate=force_recreate,
        recreate_animals=recreate_animals,
    )

    resolved_animals = frozenset(dataset_animal.animal for dataset_animal in dataset.animals)

    # Membership in the dataset marker is not evidence that a configuration was written, because the marker is
    # committed before the configurations are. An animal the marker holds but the disk does not is therefore
    # materialized again, which is what makes a definition that failed partway self-healing on the next identical call.
    unconfigured_animals = frozenset(
        dataset_animal.animal
        for dataset_animal in dataset.animals
        if not dataset_animal.animal_path.joinpath(MULTI_RECORDING_CONFIGURATION_FILENAME).is_file()
    )
    materialize_multiday_plan(
        dataset=dataset,
        project_root=project_root,
        display_progress=display_progress,
        animals=(resolved_animals - existing_animals) | frozenset(recreate_animals) | unconfigured_animals,
    )

    if recreate_animals:
        tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))
        _reset_animal_jobs(tracker=tracker, dataset=dataset, animals=recreate_animals)

    console.echo(
        message=(
            f"Dataset '{name}': Defined with {len(dataset.sessions)} session(s) across {len(dataset.animals)} "
            f"animal(s)."
        ),
        level=LogLevel.SUCCESS,
    )
    return dataset


def run_forging_pipeline(
    name: str,
    project_root: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Runs the outstanding cross-recording and assembly jobs of an already-defined dataset.

    Notes:
        The forging tracker records every stage as a job. There is one cross-recording discovery job per tracked
        animal, one extraction job per that animal's session, and one assembly job per session in the dataset. The
        cross-recording jobs exist only for animals whose acquisition system resolves a multi-recording
        configuration, so a dataset of sessions without one carries assembly jobs alone.

        ``define_forging_dataset`` roots the ordering. It resolves the hierarchy from the requested session list and
        writes each animal's multi-recording configuration, so no job runs before it completes.

        Every stage the tracker already records as succeeded is skipped, so an invocation runs only the jobs still
        outstanding. Rebuilding an animal resets that animal's jobs first.

        In local mode (``job_id`` is None) every outstanding stage runs in sequence: each animal's discovery and its
        per-session extractions, then the assembly jobs across a parallel pool. The assembly stage runs sequentially
        when the resolved worker count or the outstanding session count is one. In remote mode (``job_id`` is
        provided) only the single job matching the identifier runs, so an external scheduler drives cross-job
        ordering by dispatching each identifier in prerequisite order.

        ``define_forging_dataset`` owns the hierarchy in both modes, so the session set, ``force_recreate``, and
        ``recreate_animals`` take effect there alone.

    Args:
        name: The unique name of the dataset, which ``define_forging_dataset`` has already built.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy also lives under this directory.
        job_id: The hexadecimal identifier of the single job to execute (remote mode). If not provided, every
            outstanding job runs (local mode).
        workers: The number of workers to use. A value less than 1 uses all available CPU cores (minus reserved
            cores), and 1 forces sequential assembly.
        display_progress: Determines whether to display progress bars during the multi-day and assembly stages.

    Raises:
        ValueError: If the dataset is not defined, if its acquisition system is unknown, if the provided job_id
            does not match any job, or if a discovery job's multi-recording configuration names fewer than two
            recording directories or no dataset name.
        FileNotFoundError: If the dataset carries no ``data_descriptions.feather`` companion file, or if a discovery
            job's multi-recording configuration is missing, is not a .yaml file, is not a valid multi-recording
            configuration, or names a recording that holds no combined metadata archive.
        RuntimeError: If a discovery job's multi-recording configuration names a recording directory holding several
            combined metadata archives, or names recording paths that carry no unique identifying component. It is
            also raised when the host is macOS and carries no loadable OpenMP runtime for the Numba threading layer.
    """
    # Every worker count below one means the same thing throughout this library, which is every available core. The
    # cross-recording stages are dispatched into cindra, which spells that request as -1 and rejects every other
    # non-positive value, so the argument is normalized once here rather than read two ways by the two stages.
    workers = workers if workers > 0 else -1

    # The cross-recording stages reach a parallelized kernel, so a host whose threading layer has no runtime to load
    # fails here rather than partway through a dataset.
    verify_openmp_runtime()
    console.echo(message=f"Initializing the forging pipeline for dataset '{name}'...", level=LogLevel.INFO)

    dataset = resolve_dataset(name=name, session_names=(), project_root=project_root)
    worker = resolve_forging_assembly_worker(system=dataset.acquisition_system)
    described_columns = frozenset(dataset.column_descriptions())

    dataset_path = dataset.dataset_data_path.parent
    session_lookup: dict[str, DatasetSession] = {entry.session: entry for entry in dataset.sessions}

    multiday_plan = load_multiday_plan(dataset=dataset)
    multiday_stages = _resolve_multiday_stages(multiday_plan=multiday_plan)
    universe = build_forging_universe(dataset=dataset, multiday_plan=multiday_plan)

    dataset_path.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))

    # Requesting only the outstanding jobs while declaring the full universe preserves the recorded state of every
    # job this invocation skips. An invocation with nothing outstanding has every job of the universe already
    # registered as succeeded, so its registry needs no alignment and the tracker rejects an empty request anyway.
    runnable = _resolve_runnable_jobs(tracker=tracker, universe=universe)
    if runnable:
        tracker.align_jobs(jobs=runnable, universe=universe)
    runnable_jobs = set(runnable)

    console.echo(message=f"Prepared {len(runnable)} outstanding forging job(s) out of {len(universe)} total.")

    if job_id is not None:
        _execute_remote_forging_job(
            job_id=job_id,
            universe=universe,
            dataset=dataset,
            session_lookup=session_lookup,
            multiday_stages=multiday_stages,
            project_root=project_root,
            tracker=tracker,
            worker=worker,
            described_columns=described_columns,
            workers=workers,
        )
        console.echo(message="Forging job completed successfully.", level=LogLevel.SUCCESS)
        return

    # The stages are resolved in the order cindra executes them, animal by animal, and the priming step each animal's
    # first stage performs writes the shared bootstrap on disk. An outstanding later stage therefore runs correctly
    # even when the stage that primed the bootstrap is skipped by this invocation.
    for job, stage in multiday_stages.items():
        if job not in runnable_jobs:
            continue
        _run_multiday_job(
            stage=stage,
            job=job,
            tracker=tracker,
            job_id=ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1]),
            workers=workers,
        )

    dataset_session_names = [
        entry.session for entry in dataset.sessions if (FORGING_JOB_NAME, entry.session) in runnable_jobs
    ]
    assembly_job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        for session in dataset_session_names
    }
    # Narrowed to the assembly type's concurrency ceiling, since each job holds one core and the stage's pace is set
    # by the merge it performs rather than by the cores a budget would supply.
    resolved_workers = min(
        resolve_worker_count(requested_workers=workers), FORGING_JOB_CONCURRENCY_LIMITS[FORGING_JOB_NAME]
    )
    if not dataset_session_names:
        console.echo(message="Every session in the dataset is already assembled.", level=LogLevel.INFO)
    elif resolved_workers > 1 and len(dataset_session_names) > 1:
        _execute_jobs_parallel(
            sessions=dataset_session_names,
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_ids=assembly_job_ids,
            worker=worker,
            described_columns=described_columns,
            workers=resolved_workers,
            display_progress=display_progress,
        )
    else:
        _execute_jobs_sequential(
            sessions=dataset_session_names,
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_ids=assembly_job_ids,
            worker=worker,
            described_columns=described_columns,
            display_progress=display_progress,
        )

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def forging_tracker_path(dataset: DatasetData) -> Path:
    """Resolves the forging processing tracker path from a loaded dataset.

    Args:
        dataset: The loaded dataset whose tracker to locate.

    Returns:
        The path to the dataset's forging tracker, which sits beside the dataset's own marker.
    """
    return dataset.dataset_data_path.parent.joinpath(ProcessingTrackers.FORGING)


def discover_forging_jobs(dataset_path: Path) -> tuple[DatasetData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the forging pipeline's job universe and possible subset for an already-defined dataset.

    Notes:
        Reads the dataset marker and each animal's materialized configuration, loading no session marker. Discovery
        therefore reads nothing outside the dataset hierarchy, so a dataset whose source sessions have moved off this
        machine still resolves its jobs. Every job the universe names is possible, because ``define_forging_dataset``
        admits a session only once it carries the single-day outputs the forging stages consume, so the possible
        subset equals the universe.

        Every stage is specified by the animals and sessions the dataset hierarchy holds, so a batch is prepared
        against a dataset ``define_forging_dataset`` has already built. Preparing it afterwards names everything the
        hierarchy now holds, and the recorded job state decides which of those a run dispatches.

    Args:
        dataset_path: The path to the dataset's root directory inside the project hierarchy.

    Returns:
        A tuple of the loaded dataset, the job universe as a list of ``(job_name, specifier)`` pairs, and the possible
        subset.

    Raises:
        FileNotFoundError: If the dataset's own marker is not present under the provided path.
    """
    dataset = DatasetData.load(dataset_path=dataset_path)
    multiday_plan = load_multiday_plan(dataset=dataset)
    universe = build_forging_universe(dataset=dataset, multiday_plan=multiday_plan)
    return dataset, universe, list(universe)


def materialize_multiday_plan(
    dataset: DatasetData, project_root: Path, *, display_progress: bool, animals: Collection[str] | None = None
) -> dict[str, tuple[Path, list[str]]]:
    """Resolves the per-animal cross-recording plan and materializes each tracked animal's configuration.

    Notes:
        The multi-recording stage registers an animal's recordings against each other, so the plan is resolved once
        per animal. The acquisition system's resolver decides whether the stage applies, and an animal whose
        resolver returns None is omitted.

        Writing a configuration truncates the file in place under no lock, so only ``define_forging_dataset`` calls
        this. Every other invocation reads the plan back through ``load_multiday_plan``.

        Each materialized animal has every one of its sessions loaded from the project root, so restricting the call
        to the animals that need one keeps a dataset's growth independent of the source data of the animals it already
        holds.

    Args:
        dataset: The resolved dataset whose animals are planned.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        display_progress: The progress-bar flag recorded in each materialized configuration.
        animals: The identifiers of the animals to materialize a configuration for. Pass None to materialize every
            animal the dataset holds, which is what a freshly created dataset needs.

    Returns:
        A mapping of each tracked animal to a tuple of its materialized configuration path and its session names, in
        the animal's dataset order. Empty when no animal needs multi-day processing.

    Raises:
        FileNotFoundError: If the acquisition system's resolver reports a missing input it needs for an animal.
        ValueError: If the dataset's acquisition system is unknown, or if its resolver cannot resolve a configuration
            for an animal.
    """
    resolve_multi_recording_configuration = resolve_multi_recording_configuration_resolver(
        system=dataset.acquisition_system
    )

    plan: dict[str, tuple[Path, list[str]]] = {}
    for dataset_animal in dataset.animals:
        animal = dataset_animal.animal
        if animals is not None and animal not in animals:
            continue

        animal_entries = dataset.get_sessions_for_animal(animal=animal)
        animal_sessions = [
            SessionData.load(session_path=project_root.joinpath(animal, entry.session)) for entry in animal_entries
        ]

        configuration = resolve_multi_recording_configuration(animal_sessions[0])
        if configuration is None:
            continue

        # Each cindra output directory holds the combined_metadata.npz the multi-day stage consumes.
        configuration.recording_io.recording_directories = tuple(
            session.processed_data.cindra_data_path for session in animal_sessions
        )
        # cindra folds the configured name when it builds the output directory, and every reader resolves that
        # directory through cindra's own resolver, so the qualified name is written as-is.
        configuration.recording_io.dataset_name = multi_recording_dataset_name(
            animal_id=animal, dataset_name=dataset.name
        )
        configuration.runtime.display_progress_bars = display_progress

        configuration_path = dataset_animal.animal_path.joinpath(MULTI_RECORDING_CONFIGURATION_FILENAME)
        configuration.save(file_path=configuration_path)

        plan[animal] = (configuration_path, [entry.session for entry in animal_entries])

    return plan


def load_multiday_plan(dataset: DatasetData) -> dict[str, tuple[Path, list[str]]]:
    """Reads back the per-animal cross-recording plan a defining invocation materialized.

    Notes:
        An animal needs multi-day processing exactly when its configuration is on disk, since
        ``define_forging_dataset`` writes that file only for the animals whose resolver returns one. Reading it back
        loads no session marker, so a single-job invocation costs one path check per animal.

    Args:
        dataset: The resolved dataset whose animals are read.

    Returns:
        A mapping of each tracked animal to a tuple of its configuration path and its session names, in the animal's
        dataset order. Empty when no animal carries a materialized configuration.
    """
    plan: dict[str, tuple[Path, list[str]]] = {}
    for dataset_animal in dataset.animals:
        configuration_path = dataset_animal.animal_path.joinpath(MULTI_RECORDING_CONFIGURATION_FILENAME)
        if not configuration_path.is_file():
            continue

        # An animal holding no session has nothing to register against, so it carries no cross-recording job.
        animal_entries = dataset.get_sessions_for_animal(animal=dataset_animal.animal)
        if not animal_entries:
            continue

        plan[dataset_animal.animal] = (configuration_path, [entry.session for entry in animal_entries])

    return plan


def build_forging_universe(
    dataset: DatasetData, multiday_plan: dict[str, tuple[Path, list[str]]]
) -> list[tuple[str, str]]:
    """Builds the full forging job universe for the dataset's tracker.

    Notes:
        cindra declares which cross-recording stages an animal runs and the order they run in, so the universe takes
        that stretch of the graph from it and appends the per-session assembly stage this library owns. The result
        holds one discovery job per tracked animal, one extraction job per that animal's session, and one assembly job
        per session in the dataset. Datasets whose animals need no multi-day processing carry assembly jobs alone.

    Args:
        dataset: The resolved dataset whose sessions are assembled.
        multiday_plan: The per-animal multi-day plan from ``materialize_multiday_plan`` or ``load_multiday_plan``.

    Returns:
        The list of ``(job_name, specifier)`` pairs the forging tracker aligns against.
    """
    universe: list[tuple[str, str]] = list(_resolve_multiday_stages(multiday_plan=multiday_plan))
    universe.extend((FORGING_JOB_NAME, entry.session) for entry in dataset.sessions)
    return universe


def forging_job_prerequisites(
    dataset: DatasetData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the forging pipeline.

    Notes:
        cindra owns the ordering of the cross-recording stretch, so the prerequisites of every discovery and
        extraction job come from its resolver rather than from a chain spelled out here. cindra orders one animal's
        recordings at a time while the universe interleaves every animal, so the cross-recording jobs are regrouped
        under the animal that owns them before the resolver runs over each group.

        This library owns the assembly stage, which reads the aligned fluorescence its session's extraction wrote, so
        that edge is added on top of the ordering cindra supplies. A dataset needing no multi-day processing carries
        assembly jobs that depend on nothing.

        The dataset supplies the animal each session belongs to, which the universe pairs do not carry, since an
        extraction is specified by its session while its discovery is specified by its animal. A job whose animal the
        dataset no longer holds keeps an empty prerequisite tuple, so every job in the universe is answered for.

    Args:
        dataset: The resolved dataset the universe was built from.
        universe: The job set to build ordering over, as returned by ``build_forging_universe``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs, following the discovery to extraction to assembly
        chain.
    """
    animal_of_session = {entry.session: entry.animal for entry in dataset.sessions}
    tracked = set(universe)

    sessions_by_animal: dict[str, list[str]] = {}
    for job_name, specifier in universe:
        if job_name == MULTIDAY_DISCOVERY_JOB_NAME:
            sessions_by_animal.setdefault(specifier, [])
        elif job_name == MULTIDAY_EXTRACTION_JOB_NAME and specifier in animal_of_session:
            sessions_by_animal.setdefault(animal_of_session[specifier], []).append(specifier)

    ordering: dict[tuple[str, str], tuple[tuple[str, str], ...]] = dict.fromkeys(universe, ())
    for animal, sessions in sessions_by_animal.items():
        # Narrowed to the stages the universe actually tracks, so a job the universe omits is not resolved into a
        # prerequisite of the jobs that follow it.
        jobs = [
            job
            for job in resolve_multi_recording_jobs(recording_ids=sessions)
            if _forging_job(animal=animal, job=job) in tracked
        ]
        for job, prerequisites in resolve_multi_recording_prerequisites(jobs=jobs).items():
            ordering[_forging_job(animal=animal, job=job)] = tuple(
                _forging_job(animal=animal, job=prerequisite) for prerequisite in prerequisites
            )

    extractions = {specifier for job_name, specifier in universe if job_name == MULTIDAY_EXTRACTION_JOB_NAME}
    for job_name, specifier in universe:
        if job_name == FORGING_JOB_NAME and specifier in extractions:
            ordering[job_name, specifier] = ((MULTIDAY_EXTRACTION_JOB_NAME, specifier),)
    return ordering


def _forging_job(animal: str, job: tuple[str, str]) -> tuple[str, str]:
    """Renames one cindra cross-recording job into the forging job the tracker records it under.

    Notes:
        cindra specifies a stage spanning the whole animal with an empty string, since it knows the dataset it
        registers rather than the animal that dataset belongs to. The forging tracker interleaves several animals, so
        it records such a stage under the animal instead. Every other stage is specified by its recording, which the
        forging layout names by its session.

    Args:
        animal: The identifier of the animal whose cross-recording pipeline the job belongs to.
        job: The cindra ``(job_name, specifier)`` pair to rename.

    Returns:
        The ``(job_name, specifier)`` pair the forging tracker records the job under.
    """
    job_name, specifier = job
    return _MULTIDAY_JOB_NAMES[MultiRecordingJobNames(job_name)], specifier or animal


def _resolve_multiday_stages(multiday_plan: dict[str, tuple[Path, list[str]]]) -> dict[tuple[str, str], _MultidayStage]:
    """Resolves every cross-recording stage the multi-day plan implies, keyed by the forging job that tracks it.

    Notes:
        Which stages an animal runs, the order they run in, and the stage that has nothing before it are all cindra's
        to declare, so they come from its resolvers and this call only renames the result into the forging tracker's
        vocabulary. The returned mapping preserves that order, animal by animal, so iterating it dispatches an
        animal's stages in the order cindra executes them.

        Every cindra stage reads the shared multi-recording bootstrap rather than writing it, so the animal's first
        stage is the one marked to prime it. Priming is single-threaded by contract, and that stage precedes every
        other stage of its animal, so it is the only point at which no peer stage of the same animal can be running.

    Args:
        multiday_plan: The per-animal multi-day plan from ``materialize_multiday_plan`` or ``load_multiday_plan``.

    Returns:
        A mapping of each cross-recording forging job to the cindra stage it dispatches. Empty when no animal needs
        multi-day processing.
    """
    stages: dict[tuple[str, str], _MultidayStage] = {}
    for animal, (configuration_path, sessions) in multiday_plan.items():
        jobs = resolve_multi_recording_jobs(recording_ids=sessions)
        prerequisites = resolve_multi_recording_prerequisites(jobs=jobs)
        for job_name, specifier in jobs:
            stages[_forging_job(animal=animal, job=(job_name, specifier))] = _MultidayStage(
                configuration_path=configuration_path,
                job_name=MultiRecordingJobNames(job_name),
                specifier=specifier,
                prime=not prerequisites[job_name, specifier],
            )
    return stages


def _resolve_runnable_jobs(tracker: ProcessingTracker, universe: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Resolves the subset of the job universe the tracker does not already record as succeeded.

    Notes:
        Reading the tracker leaves a file that does not yet exist uncreated, so a dataset forged for the first time
        reports its whole universe as outstanding.

    Args:
        tracker: The forging processing tracker to read the recorded job states from.
        universe: Every ``(job_name, specifier)`` pair the dataset could produce.

    Returns:
        The outstanding pairs, in the order the universe lists them.
    """
    snapshot = tracker.snapshot()
    return [
        (job_name, specifier)
        for job_name, specifier in universe
        if (state := snapshot.get(ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier))) is None
        or state.status != ProcessingStatus.SUCCEEDED
    ]


def _reset_animal_jobs(tracker: ProcessingTracker, dataset: DatasetData, animals: tuple[str, ...]) -> None:
    """Resets every tracked forging job belonging to the specified animals back to the scheduled state.

    Notes:
        A rebuilt animal has a new session set, so its discovery stage is outstanding again, along with the multi-day
        extraction and assembly stages of every session it now holds, including the sessions it kept across the
        rebuild. Only the identifiers the tracker already holds are reset, since a tracker rejects a request naming a
        job it does not track.

    Args:
        tracker: The forging processing tracker whose job states to reset.
        dataset: The dataset as it stands after the rebuild, used to resolve each animal's current session set.
        animals: The identifiers of the animals whose jobs to reset.
    """
    snapshot = tracker.snapshot()
    if not snapshot:
        return

    targets: list[str] = []
    for animal in animals:
        targets.append(ProcessingTracker.generate_job_id(job_name=MULTIDAY_DISCOVERY_JOB_NAME, specifier=animal))
        targets.extend(
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=entry.session)
            for entry in dataset.get_sessions_for_animal(animal=animal)
            for job_name in (MULTIDAY_EXTRACTION_JOB_NAME, FORGING_JOB_NAME)
        )

    tracked_targets = [target for target in targets if target in snapshot]
    if tracked_targets:
        tracker.reset_jobs(job_ids=tracked_targets)
        console.echo(
            message=f"Reset {len(tracked_targets)} tracked job(s) for the rebuilt animal(s) {natsorted(animals)}.",
            level=LogLevel.INFO,
        )


def _run_multiday_job(
    stage: _MultidayStage, job: tuple[str, str], tracker: ProcessingTracker, job_id: str, workers: int
) -> None:
    """Runs one cindra cross-recording stage as a tracked forging job.

    Notes:
        cindra records this job's state directly on the forging tracker under job_id, and identifies each recording by
        the unique component of its recording directory path, which for the forging layout is the session name.

        A stage that primes writes the shared multi-recording bootstrap every cindra stage reads. The priming precedes
        the tracked job, so a bootstrap that cannot be written leaves the job unstarted rather than recorded as failed.

    Args:
        stage: The cindra stage to execute, resolved by ``_resolve_multiday_stages``.
        job: The forging ``(job_name, specifier)`` pair the stage is tracked under, used for logging.
        tracker: The forging processing tracker this job is recorded on.
        job_id: The unique hexadecimal identifier for this job.
        workers: The workers this stage runs under.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, is not a valid multi-recording
            configuration, or names a recording that holds no combined metadata archive.
        ValueError: If the configuration names fewer than two recording directories or no dataset name.
        RuntimeError: If the configuration names a recording directory holding several combined metadata archives, or
            names recording paths that carry no unique identifying component.
    """
    job_name, specifier = job
    console.echo(
        message=f"Running '{job_name}' job with specifier '{specifier}' (ID: {job_id})...",
        level=LogLevel.INFO,
    )
    if stage.prime:
        prime_dataset(configuration_path=stage.configuration_path)
    execute_multi_recording_job(
        configuration_path=stage.configuration_path,
        job_name=stage.job_name,
        specifier=stage.specifier,
        job_id=job_id,
        tracker=tracker,
        workers=workers,
    )


def _execute_remote_forging_job(
    job_id: str,
    universe: list[tuple[str, str]],
    dataset: DatasetData,
    session_lookup: dict[str, DatasetSession],
    multiday_stages: dict[tuple[str, str], _MultidayStage],
    project_root: Path,
    tracker: ProcessingTracker,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    workers: int,
) -> None:
    """Executes the single forging job matching the provided identifier (remote mode).

    Notes:
        The resolved cross-recording stages carry everything cindra needs to run any one of them on its own, so a job
        they name is dispatched through them and every other job is an assembly job this library runs itself.

    Args:
        job_id: The hexadecimal identifier of the job to execute.
        universe: Every ``(job_name, specifier)`` pair the dataset could produce, used to resolve the job.
        dataset: The resolved dataset being forged.
        session_lookup: The mapping from session name to its DatasetSession metadata, used by assembly jobs.
        multiday_stages: The cross-recording stages from ``_resolve_multiday_stages``, keyed by the forging job each
            one is tracked under.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        worker: The registered per-session assembly worker.
        described_columns: The column names the dataset describes, which every assembled session is held to.
        workers: The workers a multi-day stage runs under. The assembly stage takes its own fan-out instead.

    Raises:
        ValueError: If the job_id does not match any job available for this dataset.
    """
    id_to_job = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
        for job_name, specifier in universe
    }
    if job_id not in id_to_job:
        message = (
            f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any forging job "
            f"available for this dataset. Valid job IDs: {natsorted(id_to_job.keys())}."
        )
        console.error(message=message, error=ValueError)

    job = id_to_job[job_id]
    stage = multiday_stages.get(job)
    if stage is not None:
        _run_multiday_job(stage=stage, job=job, tracker=tracker, job_id=job_id, workers=workers)
    else:
        _execute_job(
            session_name=job[1],
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_id=job_id,
            worker=worker,
            described_columns=described_columns,
        )


def _execute_jobs_sequential(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    *,
    display_progress: bool,
) -> None:
    """Runs the provided assembly jobs sequentially in the parent process with an optional progress bar.

    Notes:
        Each job is fully owned by the parent process, so the first exception aborts the remaining jobs.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_ids: The mapping from session name to job ID.
        worker: The registered per-session assembly worker.
        described_columns: The column names the dataset describes, which every assembled session is held to.
        display_progress: Determines whether to display a per-session progress bar.
    """
    progress_context = (
        console.progress(total=len(sessions), description="Assembling dataset sessions", unit="session")
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for session_name in sessions:
            _execute_job(
                session_name=session_name,
                session_lookup=session_lookup,
                dataset_name=dataset_name,
                project_root=project_root,
                tracker=tracker,
                job_id=job_ids[session_name],
                worker=worker,
                described_columns=described_columns,
            )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_jobs_parallel(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    workers: int,
    *,
    display_progress: bool,
) -> None:
    """Runs the provided assembly jobs concurrently across a shared ProcessPoolExecutor.

    Notes:
        Every dispatched job is tracked individually, and in-flight futures are allowed to finish on failure so the
        tracker stays accurate for all of them. The first captured exception is re-raised after all futures resolve.

        A job is marked running as its pool slot opens rather than as the queue is built, so the tracker never reports
        more jobs running than the pool can execute and a recorded start time is the time the work began.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_ids: The mapping from session name to job ID.
        worker: The registered per-session assembly worker. Must be picklable for the worker processes.
        described_columns: The column names the dataset describes, which every assembled session is held to.
        workers: The resolved worker-process count for the shared pool.
        display_progress: Determines whether to display a per-session progress bar.
    """
    first_exception: Exception | None = None

    # Each assembly child re-imports and sizes its library thread pools before any of this code runs inside it, so the
    # caps are placed around the pool's construction rather than inside its workers. numba is the exception: it
    # latches its ceiling while it is imported and refuses a later disagreement, so the environment never reaches it
    # and each child pins it through its own runtime setter in the pool initializer instead.
    with (
        limit_worker_threads(),
        ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker_threads) as executor,
    ):
        queued = deque(sessions)
        future_to_job_id: dict[Future[None], str] = {}

        def _dispatch_next() -> None:
            """Marks the next queued session's job as running and submits it, doing nothing when none is queued."""
            if not queued:
                return
            session_name = queued.popleft()
            job_id = job_ids[session_name]
            session_metadata = session_lookup[session_name]

            console.echo(message=f"Running '{FORGING_JOB_NAME}' job with specifier '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _forge_session,
                source_session_path=project_root.joinpath(session_metadata.animal, session_name),
                output_path=session_metadata.data_path,
                dataset_name=dataset_name,
                worker=worker,
                described_columns=described_columns,
            )
            future_to_job_id[future] = job_id

        for _ in range(min(workers, len(sessions))):
            _dispatch_next()

        progress_context = (
            console.progress(total=len(sessions), description="Assembling dataset sessions", unit="session")
            if display_progress
            else nullcontext()
        )

        with progress_context as progress_bar:
            while future_to_job_id:
                completed, _ = wait(future_to_job_id, return_when=FIRST_COMPLETED)
                for completed_future in completed:
                    completed_job_id = future_to_job_id.pop(completed_future)
                    try:
                        completed_future.result()
                        tracker.complete_job(job_id=completed_job_id)
                    except Exception as exception:
                        tracker.fail_job(job_id=completed_job_id, error_message=str(exception))
                        if first_exception is None:
                            first_exception = exception
                    if progress_bar is not None:
                        progress_bar.update(1)
                    _dispatch_next()

    if first_exception is not None:
        raise first_exception


def _execute_job(
    session_name: str,
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_id: str,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
) -> None:
    """Executes a single session assembly job in-process with full tracker state management.

    Args:
        session_name: The name of the session whose data to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_id: The unique hexadecimal identifier for this assembly job.
        worker: The registered per-session assembly worker.
        described_columns: The column names the dataset describes, which every assembled session is held to.
    """
    console.echo(message=f"Running '{FORGING_JOB_NAME}' job with specifier '{session_name}' (ID: {job_id})...")
    with tracker.run_job(job_id=job_id):
        session_metadata = session_lookup[session_name]
        source_session_path = project_root.joinpath(session_metadata.animal, session_name)
        _forge_session(
            source_session_path=source_session_path,
            output_path=session_metadata.data_path,
            dataset_name=dataset_name,
            worker=worker,
            described_columns=described_columns,
        )
    console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)


def _forge_session(
    source_session_path: Path,
    output_path: Path,
    dataset_name: str,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
) -> None:
    """Assembles a single session by running the system worker, holding the written feather to the dataset's described
    columns, then re-exporting the shared assets.

    Notes:
        This is the atomic unit the parallel path dispatches to worker processes, so it stays importable at module
        level and accepts only picklable arguments. The session descriptor is written by every acquisition runtime,
        while the VR and experiment configurations are present only for the session types that carry them. A session
        missing a required asset fails before any expensive work, and whichever assets the session holds are
        re-exported alongside the assembled feather.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the session's ``data.feather`` in the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, forwarded to the assembly worker.
        worker: The registered per-session assembly worker for the dataset's acquisition system.
        described_columns: The column names the dataset describes, which every assembled session is held to.

    Raises:
        FileNotFoundError: If a shared asset the session is required to carry is missing from the source session.
        ValueError: If the assembled feather carries a column the dataset's description companion file does not
            describe.
    """
    session = SessionData.load(session_path=source_session_path)
    reexported_assets = {
        RawDataFiles.SESSION_DESCRIPTOR: session.raw_data.session_descriptor_path,
        RawDataFiles.VR_CONFIGURATION: session.raw_data.vr_configuration_path,
        RawDataFiles.EXPERIMENT_CONFIGURATION: session.raw_data.experiment_configuration_path,
    }

    # The session's required-asset policy is the single source of truth for which re-exported assets are mandatory.
    required_filenames = {filename for filename, _ in session.required_raw_assets()}
    for filename, source_path in reexported_assets.items():
        if filename in required_filenames and not source_path.is_file():
            message = (
                f"Unable to assemble session '{source_session_path.name}'. The session's raw data directory does not "
                f"contain the required shared asset '{filename}' at '{source_path}'."
            )
            console.error(message=message, error=FileNotFoundError)

    worker(source_session_path=source_session_path, output_path=output_path, dataset_name=dataset_name)

    # The dataset describes the columns its sessions may emit, so the session that emitted an undescribed one is the
    # session whose assembly fails.
    undescribed = natsorted(set(pl.read_ipc_schema(source=output_path)) - described_columns)
    if undescribed:
        message = (
            f"Unable to assemble session '{source_session_path.name}'. Every column written into a session's "
            f"'{DatasetFiles.DATA}' must have a matching description in the '{dataset_name}' dataset's "
            f"'{DatasetFiles.DESCRIPTIONS}' companion file, but the following columns are undescribed: "
            f"{undescribed}."
        )
        console.error(message=message, error=ValueError)

    output_directory = output_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, source_path in reexported_assets.items():
        if source_path.is_file():
            shutil.copy2(src=source_path, dst=output_directory.joinpath(filename))
