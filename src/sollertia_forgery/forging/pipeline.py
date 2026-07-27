"""Provides the system-agnostic, end-to-end dataset forging pipeline that defines the dataset hierarchy, runs the
cindra multi-day cell-tracking stages, and assembles the data.feather files for each session.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from cindra import MultiRecordingJobNames, execute_multi_recording_job
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, RawDataFiles, ProcessingTrackers
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .dataset import resolve_dataset
from ..registries import resolve_forging_assembly_worker, resolve_multi_recording_configuration_resolver
from ..shared_assets import tracked_job, pinned_worker_threads, multi_recording_dataset_directory

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import DatasetData, DatasetSession

    from ..registries import ForgingAssembler

DEFINE_JOB_NAME: str = "dataset_definition"
"""The job name identifying the single dataset-definition job in the forging processing tracker. The job records that
the dataset hierarchy was resolved, which happens before the tracker exists, so it is recorded complete without
running any work of its own."""

MULTIDAY_DISCOVERY_JOB_NAME: str = "multiday_discovery"
"""The job name identifying a per-animal cindra multi-day cross-recording cell-discovery job in the forging tracker.
The job's specifier is the animal identifier."""

MULTIDAY_EXTRACTION_JOB_NAME: str = "multiday_extraction"
"""The job name identifying a per-session cindra multi-day aligned-fluorescence extraction job in the forging tracker.
The job's specifier is the session name, and the job runs after its animal's discovery job."""

FORGING_JOB_NAME: str = "session_data_assembly"
"""The job name identifying per-session assembly jobs in the forging processing tracker."""

_MULTI_RECORDING_CONFIGURATION_FILENAME: str = "multi_recording_configuration.yaml"
"""The filename under which the per-animal cindra multi-recording configuration is materialized in the animal's forged
dataset directory before the multi-day cell-tracking stage runs."""


def run_forging_pipeline(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
    force_recreate: bool = False,
    recreate_animals: tuple[str, ...] = (),
) -> None:
    """Defines the dataset hierarchy, runs the cindra multi-day cell-tracking stages, and assembles the target sessions.

    Notes:
        The forging tracker records every stage as a job: one dataset-definition job, one cindra multi-day discovery
        job per tracked animal, one multi-day extraction job per that animal's session, and one assembly job per
        session. The multi-day jobs exist only for animals whose acquisition system returns a multi-recording
        configuration, so training-session datasets carry only the definition and assembly jobs.

        Every stage the tracker already records as succeeded is skipped, so an invocation runs only the jobs still
        outstanding. Rebuilding an animal resets that animal's jobs first.

        In local mode (``job_id`` is None), every outstanding stage runs in sequence: definition, then each animal's
        discovery and its per-session extractions, then the assembly jobs across a parallel pool. In remote mode
        (``job_id`` is provided) only the single job matching the identifier runs, so an external scheduler drives
        cross-job ordering by dispatching each identifier in prerequisite order.

        The definition job owns the hierarchy in both modes, so ``session_names``, ``force_recreate``, and
        ``recreate_animals`` take effect on that job alone. A remote run may therefore pass one set of arguments with
        every dispatched job and still define the hierarchy exactly once.

    Args:
        name: The unique name of the dataset.
        session_names: The session names the dataset must contain. A session the dataset does not hold is appended,
            subject to the resolution policy. Pass an empty tuple to work with an already-defined dataset without
            changing its session set. Applied by the definition job.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        job_id: The hexadecimal identifier of the single job to execute (remote mode). If not provided, the whole
            pipeline runs (local mode).
        workers: The number of workers to use. A value less than 1 uses all available CPU cores (minus reserved
            cores), and 1 forces sequential assembly.
        display_progress: Determines whether to display progress bars during the multi-day and assembly stages.
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list. Applied by the definition job.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions the
            provided list holds for them. Applied by the definition job.

    Raises:
        ValueError: If the dataset's acquisition system is unknown, or if the provided job_id does not match any job.
            The dataset resolution policy raises for a request it cannot satisfy.
    """
    console.echo(message=f"Initializing the forging pipeline for dataset '{name}'...", level=LogLevel.INFO)

    # The definition job's identifier follows from the job name alone, so it resolves before the dataset is loaded.
    define_id = ProcessingTracker.generate_job_id(job_name=DEFINE_JOB_NAME, specifier="")
    defines_hierarchy = job_id is None or job_id == define_id

    dataset = resolve_dataset(
        name=name,
        session_names=session_names if defines_hierarchy else (),
        project_root=project_root,
        force_recreate=force_recreate and defines_hierarchy,
        recreate_animals=recreate_animals if defines_hierarchy else (),
    )

    worker = resolve_forging_assembly_worker(dataset.acquisition_system)

    dataset_path = dataset.dataset_data_path.parent
    session_lookup: dict[str, DatasetSession] = {entry.session: entry for entry in dataset.sessions}

    # Only the invocation that owns the hierarchy writes the per-animal configurations. Every other invocation reads
    # the plan those writes left behind, so sibling jobs dispatched against one dataset never rewrite a file another
    # one is reading.
    multiday_plan = (
        _materialize_multiday_plan(
            dataset=dataset, project_root=project_root, workers=workers, display_progress=display_progress
        )
        if defines_hierarchy
        else _load_multiday_plan(dataset=dataset)
    )
    universe = _build_forging_universe(dataset=dataset, multiday_plan=multiday_plan)
    session_to_configuration = {
        session: configuration_path for configuration_path, sessions in multiday_plan.values() for session in sessions
    }

    dataset_path.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(ProcessingTrackers.FORGING))

    # A job belonging to a session the rebuild dropped falls outside the universe, so align_jobs discards it below.
    if recreate_animals and defines_hierarchy:
        _reset_animal_jobs(tracker=tracker, dataset=dataset, animals=recreate_animals)

    # Requesting only the outstanding jobs while declaring the full universe preserves the recorded state of every
    # job this invocation skips.
    runnable = _resolve_runnable_jobs(tracker=tracker, universe=universe)
    tracker.align_jobs(jobs=runnable, universe=universe)
    runnable_jobs = set(runnable)

    console.echo(message=f"Prepared {len(runnable)} outstanding forging job(s) out of {len(universe)} total.")

    if job_id is not None:
        _execute_remote_forging_job(
            job_id=job_id,
            universe=universe,
            dataset=dataset,
            session_lookup=session_lookup,
            session_to_configuration=session_to_configuration,
            multiday_plan=multiday_plan,
            project_root=project_root,
            tracker=tracker,
            worker=worker,
        )
        console.echo(message="Forging job completed successfully.", level=LogLevel.SUCCESS)
        return

    # Local mode. The dataset was resolved above, so the definition job is recorded complete before the tracked stages.
    if (DEFINE_JOB_NAME, "") in runnable_jobs:
        tracker.start_job(job_id=define_id)
        tracker.complete_job(job_id=define_id)

    # cindra persists the shared bootstrap to disk, so an outstanding extraction runs correctly even when its
    # animal's discovery job is skipped.
    for animal, (configuration_path, sessions) in multiday_plan.items():
        if (MULTIDAY_DISCOVERY_JOB_NAME, animal) in runnable_jobs:
            discovery_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_DISCOVERY_JOB_NAME, specifier=animal)
            _run_discovery_job(
                configuration_path=configuration_path, animal=animal, tracker=tracker, job_id=discovery_id
            )
        for session in sessions:
            if (MULTIDAY_EXTRACTION_JOB_NAME, session) not in runnable_jobs:
                continue
            extraction_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_EXTRACTION_JOB_NAME, specifier=session)
            _run_extraction_job(
                configuration_path=configuration_path, session=session, tracker=tracker, job_id=extraction_id
            )

    dataset_session_names = [
        entry.session for entry in dataset.sessions if (FORGING_JOB_NAME, entry.session) in runnable_jobs
    ]
    assembly_job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        for session in dataset_session_names
    }
    resolved_workers = resolve_worker_count(requested_workers=workers)
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
            display_progress=display_progress,
        )

    # Every session's data.feather now exists, so the dataset's data-description contract is enforced against the fully
    # composed dataset. A violation means the acquisition system emitted an undescribed column.
    console.echo(message="Verifying assembled-data column descriptions...", level=LogLevel.INFO)
    dataset.verify_data_descriptions()

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def _materialize_multiday_plan(
    dataset: DatasetData, project_root: Path, *, workers: int, display_progress: bool
) -> dict[str, tuple[Path, list[str]]]:
    """Resolves the per-animal cindra multi-day plan and materializes each tracked animal's configuration.

    Notes:
        The multi-recording stage registers an animal's recordings against each other, so the plan is resolved once
        per animal. The acquisition system's resolver decides whether the stage applies, and an animal whose
        resolver returns None is omitted.

        Writing a configuration truncates the file in place under no lock, so only the invocation that owns the
        dataset hierarchy calls this. Every other invocation reads the same plan back through
        ``_load_multiday_plan``, which keeps concurrently dispatched jobs off the files their siblings read.

    Args:
        dataset: The resolved dataset whose animals are planned.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        workers: The numba worker budget recorded in each materialized configuration.
        display_progress: The progress-bar flag recorded in each materialized configuration.

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
        animal_entries = dataset.get_sessions_for_animal(animal)
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
        # The helper applies the same lowercasing cindra does, so the written output directory and the path the
        # assembler reads back agree.
        configuration.recording_io.dataset_name = multi_recording_dataset_directory(
            animal_id=animal, dataset_name=dataset.name
        )
        configuration.runtime.parallel_workers = workers
        configuration.runtime.display_progress_bars = display_progress

        configuration_path = dataset_animal.animal_path.joinpath(_MULTI_RECORDING_CONFIGURATION_FILENAME)
        configuration.save(file_path=configuration_path)

        plan[animal] = (configuration_path, [entry.session for entry in animal_entries])

    return plan


def _load_multiday_plan(dataset: DatasetData) -> dict[str, tuple[Path, list[str]]]:
    """Reads back the per-animal cindra multi-day plan a defining invocation materialized.

    Notes:
        An animal needs multi-day processing exactly when its configuration is on disk, since that file is written
        only for the animals whose acquisition system resolves one. Reading the plan this way touches no session
        marker and resolves no configuration, so an invocation that runs a single job costs a directory listing per
        animal rather than a load per session.

    Args:
        dataset: The resolved dataset whose animals are read.

    Returns:
        A mapping of each tracked animal to a tuple of its configuration path and its session names, in the animal's
        dataset order. Empty when no animal carries a materialized configuration.
    """
    plan: dict[str, tuple[Path, list[str]]] = {}
    for dataset_animal in dataset.animals:
        configuration_path = dataset_animal.animal_path.joinpath(_MULTI_RECORDING_CONFIGURATION_FILENAME)
        if not configuration_path.is_file():
            continue
        animal_entries = dataset.get_sessions_for_animal(dataset_animal.animal)
        plan[dataset_animal.animal] = (configuration_path, [entry.session for entry in animal_entries])

    return plan


def _build_forging_universe(
    dataset: DatasetData, multiday_plan: dict[str, tuple[Path, list[str]]]
) -> list[tuple[str, str]]:
    """Builds the full forging job universe for the dataset's tracker.

    Notes:
        The universe holds the single definition job, one discovery job per tracked animal, one extraction job per
        that animal's session, and one assembly job per session in the dataset. Datasets whose animals need no
        multi-day processing carry only the definition and assembly jobs.

    Args:
        dataset: The resolved dataset whose sessions are assembled.
        multiday_plan: The per-animal multi-day plan from ``_resolve_multiday_plan``.

    Returns:
        The list of ``(job_name, specifier)`` pairs the forging tracker aligns against.
    """
    universe: list[tuple[str, str]] = [(DEFINE_JOB_NAME, "")]
    for animal, (_, sessions) in multiday_plan.items():
        universe.append((MULTIDAY_DISCOVERY_JOB_NAME, animal))
        universe.extend((MULTIDAY_EXTRACTION_JOB_NAME, session) for session in sessions)
    universe.extend((FORGING_JOB_NAME, entry.session) for entry in dataset.sessions)
    return universe


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
            for entry in dataset.get_sessions_for_animal(animal)
            for job_name in (MULTIDAY_EXTRACTION_JOB_NAME, FORGING_JOB_NAME)
        )

    tracked_targets = [target for target in targets if target in snapshot]
    if tracked_targets:
        tracker.reset_jobs(job_ids=tracked_targets)
        console.echo(
            message=f"Reset {len(tracked_targets)} tracked job(s) for the rebuilt animal(s) {natsorted(animals)}.",
            level=LogLevel.INFO,
        )


def _run_discovery_job(configuration_path: Path, animal: str, tracker: ProcessingTracker, job_id: str) -> None:
    """Runs the cindra cross-recording cell-discovery stage for one animal as a tracked forging job.

    Notes:
        cindra records this job's state directly on the forging tracker under job_id. The job runs single-threaded
        ahead of its animal's extraction jobs, so it is the one that persists the shared multi-recording bootstrap
        those jobs read.

    Args:
        configuration_path: The path to the animal's materialized cindra multi-recording configuration.
        animal: The animal identifier, used for logging.
        tracker: The forging processing tracker cindra records this job on.
        job_id: The unique hexadecimal identifier for this discovery job.
    """
    console.echo(message=f"Running multi-day discovery for animal '{animal}' (ID: {job_id})...", level=LogLevel.INFO)
    execute_multi_recording_job(
        configuration_path=configuration_path,
        job_name=MultiRecordingJobNames.DISCOVER,
        specifier="",
        job_id=job_id,
        tracker=tracker,
        persist_bootstrap=True,
    )


def _run_extraction_job(configuration_path: Path, session: str, tracker: ProcessingTracker, job_id: str) -> None:
    """Runs the cindra aligned-fluorescence extraction stage for one recording as a tracked forging job.

    Notes:
        cindra identifies each recording by the unique component of its recording directory path, which for the
        forging layout is the session name. cindra records this job's state directly on the forging tracker under
        job_id, and reads the shared bootstrap the animal's discovery job wrote.

    Args:
        configuration_path: The path to the owning animal's materialized cindra multi-recording configuration.
        session: The session name, which is also the cindra recording identifier for the extraction.
        tracker: The forging processing tracker cindra records this job on.
        job_id: The unique hexadecimal identifier for this extraction job.
    """
    console.echo(message=f"Running multi-day extraction for session '{session}' (ID: {job_id})...", level=LogLevel.INFO)
    execute_multi_recording_job(
        configuration_path=configuration_path,
        job_name=MultiRecordingJobNames.EXTRACT,
        specifier=session,
        job_id=job_id,
        tracker=tracker,
    )


def _execute_remote_forging_job(
    job_id: str,
    universe: list[tuple[str, str]],
    dataset: DatasetData,
    session_lookup: dict[str, DatasetSession],
    session_to_configuration: dict[str, Path],
    multiday_plan: dict[str, tuple[Path, list[str]]],
    project_root: Path,
    tracker: ProcessingTracker,
    worker: ForgingAssembler,
) -> None:
    """Executes the single forging job matching the provided identifier (remote mode).

    Args:
        job_id: The hexadecimal identifier of the job to execute.
        universe: Every ``(job_name, specifier)`` pair the dataset could produce, used to resolve the job.
        dataset: The resolved dataset being forged.
        session_lookup: The mapping from session name to its DatasetSession metadata, used by assembly jobs.
        session_to_configuration: The mapping from each multi-day session to its animal's configuration path.
        multiday_plan: The per-animal multi-day plan, used to resolve a discovery job's configuration.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        worker: The registered per-session assembly worker.

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

    job_name, specifier = id_to_job[job_id]
    if job_name == DEFINE_JOB_NAME:
        # The dataset was resolved before the tracker was created, so the definition job is recorded complete.
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)
    elif job_name == MULTIDAY_DISCOVERY_JOB_NAME:
        _run_discovery_job(
            configuration_path=multiday_plan[specifier][0], animal=specifier, tracker=tracker, job_id=job_id
        )
    elif job_name == MULTIDAY_EXTRACTION_JOB_NAME:
        _run_extraction_job(
            configuration_path=session_to_configuration[specifier], session=specifier, tracker=tracker, job_id=job_id
        )
    else:
        _execute_job(
            session_name=specifier,
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_id=job_id,
            worker=worker,
        )


def _execute_jobs_sequential(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssembler,
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
    workers: int,
    *,
    display_progress: bool,
) -> None:
    """Runs the provided assembly jobs concurrently across a shared ProcessPoolExecutor.

    Notes:
        Every dispatched job is tracked individually, and in-flight futures are allowed to finish on failure so the
        tracker stays accurate for all of them. The first captured exception is re-raised after all futures resolve.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_ids: The mapping from session name to job ID.
        worker: The registered per-session assembly worker. Must be picklable for the worker processes.
        workers: The resolved worker-process count for the shared pool.
        display_progress: Determines whether to display a per-session progress bar.
    """
    first_exception: Exception | None = None

    # Each assembly child re-imports and sizes its library thread pools before any of this code runs inside it, so the
    # caps are placed around the pool's construction rather than inside its workers.
    with pinned_worker_threads(), ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for session_name in sessions:
            job_id = job_ids[session_name]
            session_metadata = session_lookup[session_name]
            source_session_path = project_root.joinpath(session_metadata.animal, session_name)

            console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _forge_session,
                source_session_path=source_session_path,
                output_path=session_metadata.data_path,
                dataset_name=dataset_name,
                worker=worker,
            )
            future_to_job_id[future] = job_id

        progress_context = (
            console.progress(total=len(sessions), description="Assembling dataset sessions", unit="session")
            if display_progress
            else nullcontext()
        )

        with progress_context as progress_bar:
            for completed_future in as_completed(future_to_job_id):
                completed_job_id = future_to_job_id[completed_future]
                try:
                    completed_future.result()
                    tracker.complete_job(job_id=completed_job_id)
                except Exception as exception:
                    tracker.fail_job(job_id=completed_job_id, error_message=str(exception))
                    if first_exception is None:
                        first_exception = exception
                if progress_bar is not None:
                    progress_bar.update(1)

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
    """
    console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
    with tracked_job(tracker=tracker, job_id=job_id):
        session_metadata = session_lookup[session_name]
        source_session_path = project_root.joinpath(session_metadata.animal, session_name)
        _forge_session(
            source_session_path=source_session_path,
            output_path=session_metadata.data_path,
            dataset_name=dataset_name,
            worker=worker,
        )
    console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)


def _forge_session(
    source_session_path: Path,
    output_path: Path,
    dataset_name: str,
    worker: ForgingAssembler,
) -> None:
    """Forges and assembles a single session: runs the system worker, then re-exports the shared assets.

    Notes:
        The atomic unit dispatched to worker processes by the parallel path, so it must stay importable at module
        level and accept only picklable arguments. The session descriptor is written by every acquisition runtime,
        while the VR and experiment configurations are present only for the session types that carry them. The
        session's own required-asset policy decides which of them are mandatory, and a session missing a required
        asset fails before any expensive work. Whichever assets the session holds are re-exported alongside the
        assembled feather.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the session's ``data.feather`` in the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, forwarded to the assembly worker.
        worker: The registered per-session assembly worker for the dataset's acquisition system.

    Raises:
        FileNotFoundError: If a shared asset the session is required to carry is missing from the source session.
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

    worker(source_session_path, output_path, dataset_name)

    output_directory = output_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, source_path in reexported_assets.items():
        if source_path.is_file():
            shutil.copy2(src=source_path, dst=output_directory.joinpath(filename))
