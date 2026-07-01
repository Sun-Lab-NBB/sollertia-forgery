"""Provides the two-stage microcontroller log processing pipeline that extracts raw per-module data from
controller log archives and parses each module into a domain-specific feather using the parser function registered
for the session's acquisition system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker
from ataraxis_communication_interface.microcontroller import (
    EXTRACTION_JOB_NAME,
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    MicroControllerManifest,
    execute_job,
)

from ..registries import resolve_microcontroller_parsers
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX,
    tracked_job,
    prepare_tracker,
    partition_events,
    find_module_feathers,
    parse_module_feather_name,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping, Callable
    from concurrent.futures import Future

    from ataraxis_communication_interface.microcontroller import ControllerExtractionConfig

# The registered parser for a single module, resolved from the central MICROCONTROLLER_PARSER_REGISTRY: a plain
# module-level function ``parse(event_partition, output_directory, session) -> None``. The PEP 695 alias is evaluated
# lazily, so its annotation-only operands (Callable, Path, SessionData) need not exist at runtime.
type ModuleParser = Callable[[dict[int, pl.DataFrame], Path, SessionData], None]

PARSE_JOB_NAME: str = "module_parsing"
"""The job name identifying per-module parsing (Stage 2) jobs in the microcontroller processing tracker. Stage 1
extraction jobs use the acquisition library's own ``EXTRACTION_JOB_NAME`` instead."""


def run_microcontroller_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes microcontroller log processing jobs for the target session.

    Notes:
        This is a two-stage pipeline. Stage 1 (extraction) reads each ``{controller_id}_log.npz`` archive via the
        ataraxis-communication-interface binding and writes raw per-module feathers into the session's
        ``microcontroller_data`` directory. Stage 2 (parsing) partitions each raw feather by event code and runs the
        parser registered for the session's acquisition system (in ``MICROCONTROLLER_PARSER_REGISTRY``), writing the
        domain-specific feather into ``microcontroller_data``; the pipeline stays system-agnostic.

        In local mode (job_id is None) every present controller is extracted, then every eligible module is parsed
        (across a worker pool when more than one worker is available and more than one module is runnable). In
        remote mode (job_id is provided) only the single matching job runs in-process. The processing tracker is
        co-located with the extracted and parsed output in ``microcontroller_data``.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The hexadecimal identifier of the single job to execute (remote mode). If not provided, the whole
            pipeline runs (local mode).
        workers: The number of worker processes to use. A value less than 1 uses all available CPU cores (minus
            reserved cores); 1 forces sequential processing.
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        FileNotFoundError: If the session's extraction configuration or microcontroller manifest is missing, or,
            in remote mode, if a requested extraction job's log archive is not present.
        ValueError: If the session's acquisition system is unknown, if a configured controller ID is not registered
            in the microcontroller manifest, if no processable controllers are discovered, or if the provided
            job_id does not match any available job.
    """
    session = SessionData.load(session_path=session_path)
    console.echo(
        message=f"Initializing microcontroller processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Looks up the parser function for every module this session's acquisition system can parse from the central
    # registry, inferring the system from the session.
    parsers = resolve_microcontroller_parsers(system=session.acquisition_system)

    # Loads the per-controller extraction configurations (validated against the microcontroller manifest).
    controllers = _resolve_controllers(session=session)

    log_directory = session.raw_data.behavior_data_path
    extraction_output = session.processed_data.microcontroller_data_path
    parse_output = session.processed_data.microcontroller_data_path

    universe, requested, extraction_archives, parse_specifiers = _discover_jobs(
        controllers=controllers, parsers=parsers, log_directory=log_directory, extraction_job_name=EXTRACTION_JOB_NAME
    )

    if not requested:
        message = (
            f"Unable to process microcontroller data for session '{session.session_name}'. No configured controller "
            f"with both a present log archive and at least one eligible module was discovered."
        )
        console.error(message=message, error=ValueError)

    console.echo(
        message=(
            f"Discovered {len(extraction_archives)} controller archive(s) and {len(parse_specifiers)} module parse "
            f"job(s)."
        )
    )

    # The tracker lives alongside the extracted and parsed data in ``microcontroller_data``. The same job universe
    # drives foreign-entry detection in both local and remote modes, so a single concurrent remote job aligns the
    # tracker without resetting its sibling jobs.
    tracker_directory = session.processed_data.microcontroller_data_path
    tracker_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_directory / ProcessingTrackers.MICROCONTROLLER)
    prepare_tracker(tracker=tracker, jobs=requested, universe=universe)

    if job_id is not None:
        _execute_remote_job(
            job_id=job_id,
            universe=universe,
            extraction_job_name=EXTRACTION_JOB_NAME,
            controllers=controllers,
            parsers=parsers,
            session=session,
            log_directory=log_directory,
            extraction_output=extraction_output,
            parse_output=parse_output,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
        )
    else:
        # Resolves the worker budget once and creates a single process pool that spans BOTH stages. The stages run
        # strictly in sequence, so one pool serves the extraction stage (intra-archive batch decoding) and then the
        # parse stage (one future per module), avoiding a worker re-spawn between them.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        shared_executor = ProcessPoolExecutor(max_workers=resolved_workers) if resolved_workers > 1 else None
        try:
            _run_extraction_stage(
                extraction_archives=extraction_archives,
                controllers=controllers,
                extraction_output=extraction_output,
                tracker=tracker,
                extraction_job_name=EXTRACTION_JOB_NAME,
                workers=resolved_workers,
                executor=shared_executor,
                display_progress=display_progress,
            )
            _run_parse_stage(
                parse_specifiers=parse_specifiers,
                parsers=parsers,
                session=session,
                extraction_output=extraction_output,
                parse_output=parse_output,
                tracker=tracker,
                executor=shared_executor,
                display_progress=display_progress,
            )
        finally:
            if shared_executor is not None:
                shared_executor.shutdown(wait=True)

    console.echo(message="All microcontroller processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _resolve_controllers(session: SessionData) -> dict[str, ControllerExtractionConfig]:
    """Resolves the per-controller extraction configurations for the target session.

    Notes:
        Loads the acquisition-time extraction configuration (the source of truth for which controllers, modules,
        and event codes to extract) from the session's raw behavior data directory, and validates every configured
        controller ID against the microcontroller manifest written alongside the log archives. The manifest check
        confirms the archives were produced by ataraxis-communication-interface, which also distinguishes the
        microcontroller controllers from the runtime DataLogger archive that shares the same directory.

    Args:
        session: The loaded session whose microcontroller logs are being processed.

    Returns:
        An ordered mapping from each configured controller ID (as a string) to its ControllerExtractionConfig.

    Raises:
        FileNotFoundError: If the extraction configuration or the microcontroller manifest is not present at the
            session's canonical raw behavior data location.
        ValueError: If a configured controller ID is not registered in the microcontroller manifest.
    """
    log_directory = session.raw_data.behavior_data_path

    config_path = log_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    if not config_path.is_file():
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. No extraction "
            f"configuration was found at '{config_path}'. The extraction configuration is authored during "
            f"acquisition and defines the per-controller event codes the extraction stage processes."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest_path = log_directory.joinpath(MICROCONTROLLER_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. No "
            f"microcontroller manifest was found at '{manifest_path}'. The manifest is required to confirm the log "
            f"archives were produced by ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    config = ExtractionConfig.load(file_path=config_path)
    manifest = MicroControllerManifest.load(file_path=manifest_path)
    manifest_ids = {str(controller.id) for controller in manifest.controllers}

    controllers = {str(controller.controller_id): controller for controller in config.controllers}

    unregistered = natsorted(controller_id for controller_id in controllers if controller_id not in manifest_ids)
    if unregistered:
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. The following "
            f"configured controller IDs are not registered in the microcontroller manifest: "
            f"{', '.join(unregistered)}. Registered IDs: {natsorted(manifest_ids)}."
        )
        console.error(message=message, error=ValueError)

    return controllers


def _find_controller_archive(log_directory: Path, controller_id: str) -> Path | None:
    """Locates the raw log archive for a controller, if it is present under the log directory.

    Notes:
        Searches recursively for the ``{controller_id}_log.npz`` archive (the same recursive glob the
        ataraxis-communication-interface log reader uses). Unlike that reader, it returns None when no archive is
        present so an unstaged controller is skipped rather than failing the session, and it takes the first match
        when several exist.

    Args:
        log_directory: The session's raw behavior data directory holding the controller log archives.
        controller_id: The controller ID whose archive to locate.

    Returns:
        The path to the controller's log archive, or None if no matching archive exists.
    """
    if not log_directory.is_dir():
        return None
    matches = natsorted(log_directory.rglob(f"{controller_id}{LOG_ARCHIVE_SUFFIX}"))
    return matches[0] if matches else None


def _extract_controller(
    archive_path: Path,
    output_directory: Path,
    controller_id: str,
    controller_config: ControllerExtractionConfig,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None = None,
) -> None:
    """Extracts the target controller's log archive into raw per-module feather files via
    ataraxis-communication-interface.

    Notes:
        Delegates to the acquisition library's ``execute_job`` binding, which reads the archive once and filters
        messages by the configured per-module event codes. The binding writes a
        ``controller_{id}_module_{type}_{id}.feather`` file per module that produced data and manages this job's
        state on the passed-in tracker (start, complete, or fail). When kernel extraction is configured it may also
        write a ``controller_{id}_kernel.feather``, which this pipeline does not consume. The output directory is
        created if it does not exist.

    Args:
        archive_path: The path to the controller's ``{controller_id}_log.npz`` archive.
        output_directory: The directory where the raw per-module feather files are written (the session's
            microcontroller data directory).
        controller_id: The controller ID whose archive is being extracted.
        controller_config: The controller's extraction configuration (its modules and per-module event codes).
        job_id: The hexadecimal identifier of this extraction job in the shared processing tracker.
        tracker: The shared processing tracker the extraction job records its state against.
        workers: The number of worker processes the extraction may use to parallelize message decoding within the
            archive. Set to a value less than 1 to use all available CPU cores (minus reserved cores).
        display_progress: Determines whether to display a progress bar during extraction.
        executor: An optional shared process pool to reuse for parallel message decoding, so a sequence of
            controller extractions does not create and tear down a pool per controller.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=controller_id,
        job_id=job_id,
        workers=workers,
        tracker=tracker,
        controller_config=controller_config,
        display_progress=display_progress,
        executor=executor,
    )


def _discover_jobs(
    controllers: dict[str, ControllerExtractionConfig],
    parsers: Mapping[tuple[int, int], ModuleParser],
    log_directory: Path,
    extraction_job_name: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Path], dict[str, tuple[str, int, int]]]:
    """Builds the job universe and the requested-job set for the session.

    Notes:
        A controller contributes jobs only if at least one of its configured modules is eligible (present in the
        resolved parser mapping); extracting a controller with no parseable modules would produce intermediate
        feathers that nothing consumes. The universe enumerates every job the configuration could produce (one
        extraction job per such controller plus one parse job per eligible module), which stays stable across
        invocations for foreign-entry detection and remote-job validation. The requested set narrows the universe
        to controllers whose log archive is actually present on disk.

    Args:
        controllers: The per-controller extraction configurations, keyed by controller ID.
        parsers: The eligible module parsers for the session, keyed by ``(module_type, module_id)``.
        log_directory: The raw behavior data directory holding the controller log archives.
        extraction_job_name: The acquisition library's extraction job name used for Stage 1 tracker entries.

    Returns:
        A tuple of (universe, requested, extraction_archives, parse_specifiers). ``universe`` and ``requested`` are
        lists of ``(job_name, specifier)`` tuples. ``extraction_archives`` maps each present controller ID to its
        archive path. ``parse_specifiers`` maps each requested parse specifier (``"{controller}-{type}-{id}"``) to
        its ``(controller_id, module_type, module_id)`` triple.
    """
    universe: list[tuple[str, str]] = []
    requested: list[tuple[str, str]] = []
    extraction_archives: dict[str, Path] = {}
    parse_specifiers: dict[str, tuple[str, int, int]] = {}

    for controller_id, controller_config in controllers.items():
        eligible = [
            (module.module_type, module.module_id)
            for module in controller_config.modules
            if (module.module_type, module.module_id) in parsers
        ]
        if not eligible:
            continue

        universe.append((extraction_job_name, controller_id))
        for module_type, module_id in eligible:
            universe.append((PARSE_JOB_NAME, f"{controller_id}-{module_type}-{module_id}"))

        archive_path = _find_controller_archive(log_directory=log_directory, controller_id=controller_id)
        if archive_path is None:
            continue

        extraction_archives[controller_id] = archive_path
        requested.append((extraction_job_name, controller_id))
        for module_type, module_id in eligible:
            specifier = f"{controller_id}-{module_type}-{module_id}"
            requested.append((PARSE_JOB_NAME, specifier))
            parse_specifiers[specifier] = (controller_id, module_type, module_id)

    return universe, requested, extraction_archives, parse_specifiers


def _run_extraction_stage(
    extraction_archives: dict[str, Path],
    controllers: dict[str, ControllerExtractionConfig],
    extraction_output: Path,
    tracker: ProcessingTracker,
    extraction_job_name: str,
    *,
    workers: int,
    executor: ProcessPoolExecutor | None,
    display_progress: bool,
) -> None:
    """Runs Stage 1: extracts each present controller's log archive into raw per-module feathers.

    Notes:
        Controllers are extracted one at a time within a session. Each archive already fans its message decoding across
        the shared process pool, so a single controller saturates the session's worker budget, matching how the
        acquisition library orchestrates a multi-controller directory. Parallelism across sessions is handled by the
        orchestration layer, which runs independent sessions concurrently under a per-session worker cap; extracting a
        session's controllers sequentially therefore avoids oversubscribing cores across those concurrent sessions. The
        shared tracker is file-lock guarded and safe under concurrent access, so this ordering is a throughput choice
        rather than a correctness constraint. The pool is owned by the caller and shared with the parse stage, so this
        helper neither creates nor shuts it down.

    Args:
        extraction_archives: The present controllers' archive paths, keyed by controller ID.
        controllers: The per-controller extraction configurations, keyed by controller ID.
        extraction_output: The directory where raw per-module feathers are written.
        tracker: The shared processing tracker.
        extraction_job_name: The acquisition library's extraction job name used to derive each job identifier.
        workers: The resolved worker-process count, passed through to size each archive's decode batches.
        executor: The shared process pool spanning both pipeline stages, or None for sequential processing.
        display_progress: Determines whether to display a per-controller progress bar.
    """
    if not extraction_archives:
        return

    progress_context = (
        console.progress(
            total=len(extraction_archives), description="Extracting microcontroller logs", unit="controller"
        )
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for controller_id, archive_path in extraction_archives.items():
            extraction_job_id = ProcessingTracker.generate_job_id(job_name=extraction_job_name, specifier=controller_id)
            console.echo(
                message=(
                    f"Running '{extraction_job_name}' job for controller '{controller_id}' (ID: {extraction_job_id})..."
                )
            )
            _extract_controller(
                archive_path=archive_path,
                output_directory=extraction_output,
                controller_id=controller_id,
                controller_config=controllers[controller_id],
                job_id=extraction_job_id,
                tracker=tracker,
                workers=workers,
                display_progress=False,
                executor=executor,
            )
            if progress_bar is not None:
                progress_bar.update(1)


def _run_parse_stage(
    parse_specifiers: dict[str, tuple[str, int, int]],
    parsers: Mapping[tuple[int, int], ModuleParser],
    session: SessionData,
    extraction_output: Path,
    parse_output: Path,
    tracker: ProcessingTracker,
    *,
    executor: ProcessPoolExecutor | None,
    display_progress: bool,
) -> None:
    """Runs Stage 2: parses each eligible module's raw feather into its domain-specific feather.

    Notes:
        The extraction outputs are indexed once up front, so each parse job resolves its input feather with an O(1)
        lookup rather than re-globbing and re-scanning the output directory per module. The acquisition binding
        writes a raw feather only for modules that produced at least one message, so a configured, eligible module
        can legitimately have no feather; such a parse job is completed with no output rather than left unresolved.
        Modules with a feather are dispatched to the shared process pool when one is available and more than one
        module is runnable, with the parent owning all tracker state transitions.

    Args:
        parse_specifiers: The requested parse specifiers mapped to their ``(controller_id, type, id)`` triples.
        parsers: The registered module parsers for the session's acquisition system, keyed by
            ``(module_type, module_id)``.
        session: The loaded session, passed through to each parser so it can resolve its own system configuration.
        extraction_output: The directory holding the raw per-module feathers.
        parse_output: The directory the parsers write their domain-specific feathers into (the session's
            ``microcontroller_data`` directory).
        tracker: The shared processing tracker.
        executor: The shared process pool spanning both pipeline stages, or None for sequential processing.
        display_progress: Determines whether to display a per-module progress bar.
    """
    if not parse_specifiers:
        return

    # Indexes every extracted module feather once, keyed by (controller_id, module_type, module_id), so each parse
    # job resolves its input with a single dict lookup instead of re-globbing and re-scanning the output directory.
    feather_index = _index_module_feathers(extraction_output=extraction_output)

    runnable: dict[str, tuple[Path, ModuleParser]] = {}
    for specifier, (controller_id, module_type, module_id) in parse_specifiers.items():
        feather_path = feather_index.get((controller_id, module_type, module_id))
        if feather_path is None:
            job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier=specifier)
            console.echo(
                message=(
                    f"No extracted data was found for module '{specifier}'; completing its parse job with no output."
                ),
                level=LogLevel.WARNING,
            )
            tracker.start_job(job_id=job_id)
            tracker.complete_job(job_id=job_id)
            continue
        runnable[specifier] = (feather_path, parsers[(module_type, module_id)])

    if not runnable:
        return

    if executor is not None and len(runnable) > 1:
        _execute_parse_jobs_parallel(
            runnable=runnable,
            tracker=tracker,
            session=session,
            parse_output=parse_output,
            executor=executor,
            display_progress=display_progress,
        )
    else:
        _execute_parse_jobs_sequential(
            runnable=runnable,
            tracker=tracker,
            session=session,
            parse_output=parse_output,
            display_progress=display_progress,
        )


def _execute_parse_jobs_sequential(
    runnable: dict[str, tuple[Path, ModuleParser]],
    tracker: ProcessingTracker,
    session: SessionData,
    parse_output: Path,
    *,
    display_progress: bool,
) -> None:
    """Runs the parse jobs sequentially in the parent process with full tracker state management.

    Args:
        runnable: The parse jobs mapping each specifier to its ``(feather_path, module_parser)`` pair.
        tracker: The shared processing tracker.
        session: The loaded session, passed through to each parser.
        parse_output: The directory the parsers write their domain-specific feathers into.
        display_progress: Determines whether to display a per-module progress bar.
    """
    progress_context = (
        console.progress(total=len(runnable), description="Parsing microcontroller modules", unit="module")
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for specifier, (feather_path, module_parser) in runnable.items():
            job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier=specifier)
            console.echo(message=f"Running '{PARSE_JOB_NAME}' job with specifier '{specifier}' (ID: {job_id})...")
            with tracked_job(tracker=tracker, job_id=job_id):
                _run_parse(
                    feather_path=feather_path,
                    module_parser=module_parser,
                    output_directory=parse_output,
                    session=session,
                )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_parse_jobs_parallel(
    runnable: dict[str, tuple[Path, ModuleParser]],
    tracker: ProcessingTracker,
    session: SessionData,
    parse_output: Path,
    *,
    executor: ProcessPoolExecutor,
    display_progress: bool,
) -> None:
    """Runs the parse jobs concurrently across the shared process pool, with the parent owning tracker state.

    Notes:
        The pool is the one shared with the extraction stage and is owned by the caller, so this helper submits to
        it without shutting it down. Each job's tracker state is advanced to running immediately before its future
        is submitted, then resolved as the future completes. In-flight futures are allowed to finish on failure so
        the tracker stays accurate for every dispatched job; the first captured exception is re-raised after all
        futures resolve.

    Args:
        runnable: The parse jobs mapping each specifier to its ``(feather_path, module_parser)`` pair.
        tracker: The shared processing tracker.
        session: The loaded session, passed through to each parser; must be picklable for the worker processes.
        parse_output: The directory the parsers write their domain-specific feathers into.
        executor: The shared process pool to submit the parse jobs to. Owned by the caller; not shut down here.
        display_progress: Determines whether to display a per-module progress bar.
    """
    first_exception: Exception | None = None

    future_to_job_id: dict[Future[None], str] = {}
    for specifier, (feather_path, module_parser) in runnable.items():
        job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier=specifier)
        console.echo(message=f"Running '{PARSE_JOB_NAME}' job with specifier '{specifier}' (ID: {job_id})...")
        tracker.start_job(job_id=job_id)
        future = executor.submit(
            _run_parse,
            feather_path=feather_path,
            module_parser=module_parser,
            output_directory=parse_output,
            session=session,
        )
        future_to_job_id[future] = job_id

    progress_context = (
        console.progress(total=len(runnable), description="Parsing microcontroller modules", unit="module")
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


def _execute_remote_job(
    job_id: str,
    universe: list[tuple[str, str]],
    extraction_job_name: str,
    controllers: dict[str, ControllerExtractionConfig],
    parsers: Mapping[tuple[int, int], ModuleParser],
    session: SessionData,
    log_directory: Path,
    extraction_output: Path,
    parse_output: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None:
    """Executes the single job matching the provided identifier (remote mode).

    Args:
        job_id: The hexadecimal identifier of the job to execute.
        universe: Every ``(job_name, specifier)`` tuple the configuration could produce, used to resolve the job.
        extraction_job_name: The acquisition library's extraction job name distinguishing Stage 1 from Stage 2 jobs.
        controllers: The per-controller extraction configurations, keyed by controller ID.
        parsers: The registered module parsers for the session's acquisition system, keyed by
            ``(module_type, module_id)``.
        session: The loaded session, passed through to the parser for a remote parse job.
        log_directory: The raw behavior data directory holding the controller log archives.
        extraction_output: The directory holding (or receiving) the raw per-module feathers.
        parse_output: The directory a parser writes its domain-specific feather into.
        tracker: The shared processing tracker.
        workers: The requested worker-process count.
        display_progress: Determines whether to display a progress bar.

    Raises:
        ValueError: If the job_id does not match any job available for this session.
        FileNotFoundError: If a requested extraction job's log archive is not present.
    """
    id_to_job = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
        for job_name, specifier in universe
    }
    if job_id not in id_to_job:
        message = (
            f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any job "
            f"available for this session. Valid job IDs: {natsorted(id_to_job.keys())}."
        )
        console.error(message=message, error=ValueError)

    job_name, specifier = id_to_job[job_id]

    if job_name == extraction_job_name:
        controller_id = specifier
        archive_path = _find_controller_archive(log_directory=log_directory, controller_id=controller_id)
        if archive_path is None:
            message = (
                f"Unable to run the extraction job for controller '{controller_id}'. No log archive "
                f"'{controller_id}_log.npz' was found under '{log_directory}'."
            )
            console.error(message=message, error=FileNotFoundError)
        resolved_workers = resolve_worker_count(requested_workers=workers)
        console.echo(message=f"Running '{extraction_job_name}' job for controller '{controller_id}' (ID: {job_id})...")
        _extract_controller(
            archive_path=archive_path,
            output_directory=extraction_output,
            controller_id=controller_id,
            controller_config=controllers[controller_id],
            job_id=job_id,
            tracker=tracker,
            workers=resolved_workers,
            display_progress=display_progress,
        )
        return

    controller_id, module_type, module_id = _split_parse_specifier(specifier=specifier)
    module_parser = parsers[(module_type, module_id)]
    feather_path = _index_module_feathers(extraction_output=extraction_output).get(
        (controller_id, module_type, module_id)
    )

    console.echo(message=f"Running '{PARSE_JOB_NAME}' job with specifier '{specifier}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)
    if feather_path is None:
        console.echo(
            message=(
                f"No extracted data was found for module '{specifier}'; completing its parse job with no output. "
                f"Ensure the controller's extraction job has run first."
            ),
            level=LogLevel.WARNING,
        )
        tracker.complete_job(job_id=job_id)
        return
    try:
        _run_parse(
            feather_path=feather_path, module_parser=module_parser, output_directory=parse_output, session=session
        )
        tracker.complete_job(job_id=job_id)
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _run_parse(feather_path: Path, module_parser: ModuleParser, output_directory: Path, session: SessionData) -> None:
    """Parses one raw module feather into its domain-specific feather.

    Notes:
        This is the atomic unit of work dispatched to worker processes by the parallel parse path, so it must
        remain importable at module level and accept only picklable arguments. It reads the raw module feather via
        memory mapping, partitions it by event code in a single pass, and delegates to the registered parser. The
        parser then resolves any system configuration from the session and writes its feather into the output
        directory.

    Args:
        feather_path: The path to the raw per-module feather produced by the extraction stage.
        module_parser: The registered parser function for this module.
        output_directory: The directory the parser writes its domain-specific feather into.
        session: The loaded session, from which the parser resolves its own system configuration.
    """
    module_dataframe = pl.read_ipc(source=feather_path, memory_map=True)
    event_partition = partition_events(module_dataframe=module_dataframe)
    output_directory.mkdir(parents=True, exist_ok=True)
    module_parser(event_partition, output_directory, session)


def _index_module_feathers(extraction_output: Path) -> dict[tuple[str, int, int], Path]:
    """Indexes the raw module feathers in the extraction output directory by their module identity.

    Notes:
        Globs the directory once and parses each feather name a single time, building a lookup keyed by
        ``(controller_id, module_type, module_id)``. Callers resolve a module's feather with an O(1) dict lookup
        instead of re-globbing and re-scanning the directory once per module.

    Args:
        extraction_output: The directory holding the raw per-module feathers.

    Returns:
        A mapping from each ``(controller_id, module_type, module_id)`` triple to its raw feather path. The
        controller ID is stored as a string to match the specifier form used throughout the pipeline.
    """
    index: dict[tuple[str, int, int], Path] = {}
    for feather_path in find_module_feathers(data_directory=extraction_output):
        feather_controller, feather_type, feather_id = parse_module_feather_name(feather_path=feather_path)
        index[(str(feather_controller), feather_type, feather_id)] = feather_path
    return index


def _split_parse_specifier(specifier: str) -> tuple[str, int, int]:
    """Splits a parse-job specifier into its controller ID, module type, and module ID components.

    Args:
        specifier: The parse specifier in ``"{controller_id}-{module_type}-{module_id}"`` form.

    Returns:
        A tuple of (controller_id, module_type, module_id).
    """
    controller_id, module_type, module_id = specifier.split("-")
    return controller_id, int(module_type), int(module_id)
