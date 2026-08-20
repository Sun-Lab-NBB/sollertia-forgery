"""Provides the two-stage microcontroller log processing pipeline that extracts raw per-module data from
controller log archives and parses each module into a domain-specific feather using the parser function registered
for the session's acquisition system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import ExitStack, nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker, limit_worker_threads, initialize_worker_threads
from ataraxis_communication_interface import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    EXTRACTION_CONFIGURATION_FILENAME,
    JobUniverse,
    ExtractionConfig,
    ModuleExtractionConfig,
    ControllerExtractionConfig,
    execute_job,
    resolve_jobs,
    partition_events,
    resolve_module_path,
)

from ..registries import (
    resolve_microcontroller_parsers,
    resolve_microcontroller_event_codes,
    resolve_eligible_microcontroller_modules,
)
from ..shared_assets import verify_openmp_runtime

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping
    from concurrent.futures import Future

    from ..registries import MicrocontrollerParser

PARSE_JOB_NAME: str = "module_parsing"
"""The job name identifying per-module parsing (Stage 2) jobs in the microcontroller processing tracker. Stage 1
extraction jobs use the acquisition library's own ``CONTROLLER_EXTRACTION_JOB_NAME`` instead."""


def run_microcontroller_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes microcontroller log processing jobs for the target session.

    Notes:
        This is a two-stage pipeline. Stage 1 (extraction) reads each controller's log archive via the
        ataraxis-communication-interface binding and writes raw per-module feathers into the session's
        ``microcontroller_data`` directory. Stage 2 (parsing) partitions each raw feather by event code and runs the
        parser registered for the session's acquisition system (resolved via ``resolve_microcontroller_parsers``),
        writing the domain-specific feather into ``microcontroller_data``. The pipeline is system-agnostic.

        In local mode (job_id is None) every present controller is extracted, then every eligible module is parsed
        (across a worker pool when more than one worker is available and more than one module is runnable). In
        remote mode (job_id is provided) only the single matching job runs. That job still honors the worker budget,
        so a remote extraction fans intra-archive decoding across the pool while a remote parse runs single-core. The
        processing tracker is co-located with the extracted and parsed output in ``microcontroller_data``.

        The extraction configuration is materialized into ``microcontroller_data`` on every invocation, before any
        job is dispatched. The acquisition binding reads each controller's extraction targets from that file rather
        than from an in-memory object, and a scheduler may dispatch a single extraction job into a fresh process, so
        writing the file unconditionally is what lets a remote job read the same configuration the local run used.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The hexadecimal identifier of the single job to execute (remote mode). If not provided, the whole
            pipeline runs (local mode).
        workers: The number of worker processes to use. A value less than 1 uses all available CPU cores (minus
            reserved cores), and 1 forces sequential processing.
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        FileNotFoundError: If the session's microcontroller manifest is missing, or, in remote mode, if a requested
            extraction job's log archive is not present.
        RuntimeError: If the host is macOS and carries no loadable OpenMP runtime for the Numba threading layer.
        ValueError: If the session's acquisition system is unknown, if the microcontroller manifest is malformed, if
            the raw behavior data tree holds more than one microcontroller manifest, if no manifest controller
            declares a module the session's acquisition system extracts, if no processable controllers are
            discovered, or if the provided job_id does not match any available job.
    """
    # A stage this pipeline dispatches may reach a parallelized kernel, so a host whose threading layer has no
    # runtime to load fails here rather than partway through a session.
    verify_openmp_runtime()
    session = SessionData.load(session_path=session_path)
    console.echo(
        message=f"Initializing microcontroller processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Looks up the parser function and the extracted event codes for every module this session's acquisition system
    # can parse from the central registries, inferring the system from the session.
    parsers = resolve_microcontroller_parsers(system=session.acquisition_system)
    event_codes = _resolve_eligible_event_codes(session=session)

    log_directory = session.raw_data.behavior_data_path
    extraction_output = session.processed_data.microcontroller_data_path
    parse_output = session.processed_data.microcontroller_data_path

    # Reads the manifest and indexes every registered controller's archive once, so the configuration derivation and
    # the job discovery that both need that topology share one read.
    job_universe = resolve_jobs(log_directory=log_directory)

    # Derives the per-controller extraction configurations from the manifest topology and the event codes.
    controllers = _resolve_controllers(session=session, event_codes=event_codes, job_universe=job_universe)

    universe, requested, extraction_archives, parse_specifiers = _discover_jobs(
        controllers=controllers, parsers=parsers, job_universe=job_universe
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

    # Co-locates the tracker with the extracted and parsed data in ``microcontroller_data``. The same job universe
    # drives foreign-entry detection in both local and remote modes, so a single concurrent remote job aligns the
    # tracker without resetting its sibling jobs.
    tracker_directory = session.processed_data.microcontroller_data_path
    tracker_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_directory.joinpath(ProcessingTrackers.MICROCONTROLLER))
    tracker.align_jobs(jobs=requested, universe=universe)

    # Writes the configuration before any job is dispatched, in both modes, since the extraction binding reads each
    # controller's targets from the file rather than from memory and a remotely dispatched job may be the only work
    # this process does.
    config_path = _materialize_extraction_config(controllers=controllers, output_directory=extraction_output)

    if job_id is not None:
        _execute_remote_job(
            job_id=job_id,
            universe=universe,
            parsers=parsers,
            session=session,
            log_directory=log_directory,
            extraction_archives=extraction_archives,
            extraction_output=extraction_output,
            parse_output=parse_output,
            config_path=config_path,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
        )
    else:
        # Resolves the worker budget once and creates a single process pool that spans BOTH stages. The stages run
        # strictly in sequence, so one pool serves the extraction stage (intra-archive batch decoding) and then the
        # parse stage (one future per module) with a single worker spawn. The caps cover the pool's whole life,
        # since it starts its children on demand and each child sizes its library thread pools while importing,
        # before any code of this pipeline runs inside it. numba latches its own ceiling while it is imported and
        # rejects an environment variable that disagrees afterwards, so it is pinned instead by the initializer every
        # child runs through its runtime setter.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        with limit_worker_threads(), ExitStack() as pool_scope:
            shared_executor = (
                pool_scope.enter_context(
                    ProcessPoolExecutor(max_workers=resolved_workers, initializer=initialize_worker_threads)
                )
                if resolved_workers > 1
                else None
            )
            _run_extraction_stage(
                extraction_archives=extraction_archives,
                extraction_output=extraction_output,
                config_path=config_path,
                tracker=tracker,
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

    console.echo(message="All microcontroller processing jobs completed successfully.", level=LogLevel.SUCCESS)


def discover_microcontroller_jobs(
    session_path: Path,
) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the microcontroller pipeline's job universe and possible subset for the target session.

    Notes:
        The universe enumerates every job the session's microcontroller manifest could produce: one extraction job per
        controller that declares at least one module the acquisition system parses and the session configured for use,
        plus one parse job per such module. The possible subset narrows the universe to controllers whose log archive
        is present on disk, since a controller with no archive can be neither extracted nor parsed. Locating those
        archives is delegated to the acquisition library's own resolver, so discovery reads the manifest and indexes
        the archive names, leaving the archives' contents and every output file untouched.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the possible
        subset of that universe. Extraction specifiers are controller IDs and parse specifiers are
        ``"{controller_id}-{module_type}-{module_id}"``.

    Raises:
        FileNotFoundError: If the session's microcontroller manifest is not present.
        ValueError: If the session's acquisition system is unknown, if the microcontroller manifest is malformed, if
            the raw behavior data tree holds more than one microcontroller manifest, or if no manifest controller
            declares a module the acquisition system extracts.
    """
    session = SessionData.load(session_path=session_path)
    parsers = resolve_microcontroller_parsers(system=session.acquisition_system)
    event_codes = _resolve_eligible_event_codes(session=session)
    job_universe = resolve_jobs(log_directory=session.raw_data.behavior_data_path)
    controllers = _resolve_controllers(session=session, event_codes=event_codes, job_universe=job_universe)
    universe, requested, _, _ = _discover_jobs(controllers=controllers, parsers=parsers, job_universe=job_universe)
    return session, universe, requested


def microcontroller_job_prerequisites(
    session: SessionData,  # noqa: ARG001
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the microcontroller pipeline.

    Notes:
        Each parse job reads the raw per-module feather its controller's extraction job writes, so every parse job
        requires that extraction job to have succeeded. Extraction jobs read the raw archive directly and have no
        upstream dependency. Extraction jobs use the acquisition library's ``CONTROLLER_EXTRACTION_JOB_NAME`` and each
        parse specifier encodes its controller as the leading ``"{controller_id}-..."`` segment.

    Args:
        session: The loaded session, accepted for the shared dispatch contract and not read by this ordering.
        universe: The job universe as returned by ``discover_microcontroller_jobs``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs. Parse jobs map to their controller's extraction job,
        and extraction jobs map to an empty tuple.
    """
    return {
        (job_name, specifier): ((CONTROLLER_EXTRACTION_JOB_NAME, specifier.split("-")[0]),)
        if job_name == PARSE_JOB_NAME
        else ()
        for job_name, specifier in universe
    }


def _resolve_eligible_event_codes(session: SessionData) -> dict[tuple[int, int], tuple[int, ...]]:
    """Resolves the event codes of the hardware modules the target session configured for use.

    Notes:
        A session records which hardware modules it used, and its acquisition system's parsers skip the modules it
        did not. Narrowing the event codes to the eligible modules keeps the extraction stage and the parse job
        universe aligned with those parsers, so an unused module contributes neither an intermediate feather nor a
        job that completes without writing an output.

    Args:
        session: The loaded session whose microcontroller logs are being processed.

    Returns:
        A mapping from each eligible ``(module_type, module_id)`` pair to the tuple of event codes its parser reads.
    """
    event_codes = resolve_microcontroller_event_codes(system=session.acquisition_system)
    eligible = resolve_eligible_microcontroller_modules(system=session.acquisition_system, session=session)
    return {module_key: codes for module_key, codes in event_codes.items() if module_key in eligible}


def _resolve_controllers(
    session: SessionData,
    event_codes: Mapping[tuple[int, int], tuple[int, ...]],
    job_universe: JobUniverse,
) -> dict[str, ControllerExtractionConfig]:
    """Derives the per-controller extraction configurations for the target session.

    Notes:
        The configurations are built in memory. The resolved job universe supplies the controller and module topology
        the manifest declares, and the session's acquisition system supplies the event codes each module's parser
        reads. Taking the topology from the universe is what lets one manifest read serve both this derivation and
        the job discovery that shares it. A manifest module the system does not parse, or that the session did not
        configure for use, is excluded, since extracting it would produce an intermediate feather nothing consumes,
        and a controller left with no such module contributes no configuration at all. Requiring the manifest also
        confirms the archives were produced by ataraxis-communication-interface, which distinguishes the
        microcontroller controllers from the runtime DataLogger archive that shares the same directory. Kernel
        extraction is never configured, because this pipeline does not consume the kernel feather.

    Args:
        session: The loaded session whose microcontroller logs are being processed.
        event_codes: The event codes of the modules that the session's acquisition system parses and the session
            configured for use, keyed by ``(module_type, module_id)``.
        job_universe: The resolved job universe, whose sources carry the modules each registered controller declares.

    Returns:
        An ordered mapping from each manifest controller ID (as a string) to its derived ControllerExtractionConfig.

    Raises:
        FileNotFoundError: If the microcontroller manifest is not present at the session's canonical raw behavior
            data location.
        ValueError: If no manifest controller declares a module the session's acquisition system extracts.
    """
    if job_universe.manifest_path is None:
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. No "
            f"microcontroller manifest was found in '{job_universe.log_directory}'. The manifest enumerates the "
            f"controllers and modules to extract and confirms the log archives were produced by "
            f"ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    controllers: dict[str, ControllerExtractionConfig] = {}
    for source in job_universe.sources:
        modules = tuple(
            ModuleExtractionConfig(
                module_type=module.module_type,
                module_id=module.module_id,
                event_codes=event_codes[(module.module_type, module.module_id)],
            )
            for module in source.modules
            if (module.module_type, module.module_id) in event_codes
        )
        if not modules:
            continue
        controllers[source.source_id] = ControllerExtractionConfig(
            controller_id=int(source.source_id), modules=modules, kernel=None
        )

    if not controllers:
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. None of the "
            f"controllers registered in the microcontroller manifest at '{job_universe.manifest_path}' declares a "
            f"module the "
            f"'{session.acquisition_system}' acquisition system extracts."
        )
        console.error(message=message, error=ValueError)

    return controllers


def _materialize_extraction_config(
    controllers: Mapping[str, ControllerExtractionConfig], output_directory: Path
) -> Path:
    """Writes the session's derived extraction configuration into the microcontroller data directory.

    Notes:
        The acquisition binding reads each controller's extraction targets from a configuration file rather than from
        an in-memory object, so the configuration slf derives from the manifest and its own event code registry has
        to reach disk before any extraction job runs. The file is written under the acquisition library's own
        configuration filename, next to the extracted output and the processing tracker.

        The write is unconditional. A scheduler may dispatch a single extraction job into a fresh process, so the
        invocation that runs one job has to produce the same configuration a whole-pipeline run would have written.

    Args:
        controllers: The per-controller extraction configurations, keyed by controller ID.
        output_directory: The session's microcontroller data directory, which receives the configuration alongside
            the extracted feathers.

    Returns:
        The path to the written extraction configuration file.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = output_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    ExtractionConfig(controllers=list(controllers.values())).to_yaml(file_path=config_path)
    return config_path


def _extract_controller(
    archive_path: Path,
    output_directory: Path,
    controller_id: str,
    config_path: Path,
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
        Delegates to the acquisition library's ``execute_job`` binding, which reads this controller's entry from the
        materialized extraction configuration, then reads the archive once and filters messages by the configured
        per-module event codes. The binding writes a ``controller_{id}_module_{type}_{id}.feather`` file per module
        that produced data and manages this job's state on the passed-in tracker (start, complete, or fail). When
        kernel extraction is configured it may also write a ``controller_{id}_kernel.feather``, which this pipeline
        does not consume. The output directory is created if it does not exist.

    Args:
        archive_path: The path to the controller's log archive, as the communication library resolved it.
        output_directory: The directory where the raw per-module feather files are written (the session's
            microcontroller data directory).
        controller_id: The controller ID whose archive is being extracted.
        config_path: The path to the materialized extraction configuration declaring every controller's modules and
            per-module event codes.
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
        config_path=config_path,
        display_progress=display_progress,
        executor=executor,
    )


def _discover_jobs(
    controllers: dict[str, ControllerExtractionConfig],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    job_universe: JobUniverse,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Path], dict[str, tuple[str, int, int]]]:
    """Builds the job universe and the requested-job set for the session.

    Notes:
        Locating the controllers is the acquisition library's own job, so the resolved universe already carries each
        registered controller's archive, in ascending identifier order and resolved only when exactly one file under
        the directory carries that controller's name. This function composes that locating with the eligibility rule
        the library knows nothing about: a
        controller contributes jobs only if at least one of its configured modules is eligible (present in the
        resolved parser mapping). Extracting a controller with no parseable modules would produce intermediate
        feathers that nothing consumes.

        The universe enumerates every job the configuration could produce (one extraction job per such controller
        plus one parse job per eligible module), which stays stable across invocations for foreign-entry detection
        and remote-job validation. The requested set narrows the universe to controllers whose archive resolved.

    Args:
        controllers: The per-controller extraction configurations, keyed by controller ID.
        parsers: The eligible module parsers for the session, keyed by ``(module_type, module_id)``.
        job_universe: The resolved job universe, whose sources carry each registered controller's archive.

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

    for source in job_universe.sources:
        controller_config = controllers.get(source.source_id)
        if controller_config is None:
            continue

        eligible = [
            (module.module_type, module.module_id)
            for module in controller_config.modules
            if (module.module_type, module.module_id) in parsers
        ]
        if not eligible:
            continue

        universe.append((CONTROLLER_EXTRACTION_JOB_NAME, source.source_id))
        for module_type, module_id in eligible:
            universe.append((PARSE_JOB_NAME, f"{source.source_id}-{module_type}-{module_id}"))

        if source.archive_path is None:
            continue

        extraction_archives[source.source_id] = source.archive_path
        requested.append((CONTROLLER_EXTRACTION_JOB_NAME, source.source_id))
        for module_type, module_id in eligible:
            specifier = f"{source.source_id}-{module_type}-{module_id}"
            requested.append((PARSE_JOB_NAME, specifier))
            parse_specifiers[specifier] = (source.source_id, module_type, module_id)

    return universe, requested, extraction_archives, parse_specifiers


def _run_extraction_stage(
    extraction_archives: dict[str, Path],
    extraction_output: Path,
    config_path: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    executor: ProcessPoolExecutor | None,
    display_progress: bool,
) -> None:
    """Runs Stage 1: extracts each present controller's log archive into raw per-module feathers.

    Notes:
        Controllers are extracted one at a time within a session. Each archive already fans its message decoding
        across the shared process pool, so a single controller saturates the session's worker budget. Sequential
        extraction therefore keeps cores available to the sessions the orchestration layer runs concurrently. The
        pool is owned by the caller and shared with the parse stage, so this helper neither creates nor shuts it
        down.

    Args:
        extraction_archives: The present controllers' archive paths, keyed by controller ID.
        extraction_output: The directory where raw per-module feathers are written.
        config_path: The path to the materialized extraction configuration each job reads its targets from.
        tracker: The shared processing tracker.
        workers: The resolved worker-process count, passed through to size each archive's decode batches.
        executor: The shared process pool spanning both pipeline stages, or None for sequential processing.
        display_progress: Determines whether to display a per-controller progress bar.
    """
    if not extraction_archives:
        return

    # The identifier follows from the job name and the controller it runs over, which is how the tracker derives the
    # identifier it records, so both sides name the same job without either passing the identifier to the other.
    extraction_job_ids = {
        controller_id: ProcessingTracker.generate_job_id(
            job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier=controller_id
        )
        for controller_id in extraction_archives
    }
    for controller_id, extraction_job_id in extraction_job_ids.items():
        console.echo(
            message=(
                f"Running '{CONTROLLER_EXTRACTION_JOB_NAME}' job for controller '{controller_id}' "
                f"(ID: {extraction_job_id})..."
            )
        )

    progress_context = (
        console.progress(
            total=len(extraction_archives), description="Extracting microcontroller logs", unit="controller"
        )
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for controller_id, archive_path in extraction_archives.items():
            # Silences the binding's per-controller announcement so it does not bisect the bar, restoring the
            # console's prior state once the extraction returns. console.error still raises while the console is
            # disabled, so a failing extraction still surfaces.
            console_enabled = console.enabled
            console.disable()
            try:
                _extract_controller(
                    archive_path=archive_path,
                    output_directory=extraction_output,
                    controller_id=controller_id,
                    config_path=config_path,
                    job_id=extraction_job_ids[controller_id],
                    tracker=tracker,
                    workers=workers,
                    display_progress=False,
                    executor=executor,
                )
            finally:
                if console_enabled:
                    console.enable()
            if progress_bar is not None:
                progress_bar.update(1)


def _run_parse_stage(
    parse_specifiers: dict[str, tuple[str, int, int]],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
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
        Each parse job derives the path of its input feather from the module's own identity through the acquisition
        library's naming convention, so no directory listing is involved. The acquisition binding writes a raw
        feather only for modules that produced at least one message, so a configured, eligible module can
        legitimately have no feather. Such a parse job is completed with no output. Modules with a feather are
        dispatched to the shared process pool when one is available and more than one module is runnable, with the
        parent owning all tracker state transitions.

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

    runnable: dict[str, tuple[Path, MicrocontrollerParser]] = {}
    for specifier, (controller_id, module_type, module_id) in parse_specifiers.items():
        feather_path = resolve_module_path(
            output_directory=extraction_output, source_id=controller_id, module_type=module_type, module_id=module_id
        )
        if not feather_path.is_file():
            job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier=specifier)
            console.echo(
                message=(
                    f"No extracted data was found for module '{specifier}'. Completing its parse job with no output."
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
    runnable: dict[str, tuple[Path, MicrocontrollerParser]],
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
            with tracker.run_job(job_id=job_id):
                _run_parse(
                    feather_path=feather_path,
                    module_parser=module_parser,
                    output_directory=parse_output,
                    session=session,
                )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_parse_jobs_parallel(
    runnable: dict[str, tuple[Path, MicrocontrollerParser]],
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
        the tracker stays accurate for every dispatched job. The first captured exception is re-raised after all
        futures resolve.

    Args:
        runnable: The parse jobs mapping each specifier to its ``(feather_path, module_parser)`` pair.
        tracker: The shared processing tracker.
        session: The loaded session, passed through to each parser. Must be picklable for the worker processes.
        parse_output: The directory the parsers write their domain-specific feathers into.
        executor: The shared process pool to submit the parse jobs to, owned by the caller.
        display_progress: Determines whether to display a per-module progress bar.

    Raises:
        Exception: The first exception raised by any parse job, re-raised after every dispatched future resolves.
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
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    session: SessionData,
    log_directory: Path,
    extraction_archives: Mapping[str, Path],
    extraction_output: Path,
    parse_output: Path,
    config_path: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None:
    """Executes the single job matching the provided identifier (remote mode).

    Notes:
        The archives are the ones job discovery already resolved through the acquisition library, so dispatching a
        single remote job costs no additional walk of the session's raw behavior data tree.

    Args:
        job_id: The hexadecimal identifier of the job to execute.
        universe: Every ``(job_name, specifier)`` tuple the configuration could produce, used to resolve the job.
        parsers: The registered module parsers for the session's acquisition system, keyed by
            ``(module_type, module_id)``.
        session: The loaded session, passed through to the parser for a remote parse job.
        log_directory: The raw behavior data directory the archives were resolved from, reported when the requested
            controller has none.
        extraction_archives: The resolved archive path of every processable controller, keyed by controller ID.
        extraction_output: The directory holding (or receiving) the raw per-module feathers.
        parse_output: The directory a parser writes its domain-specific feather into.
        config_path: The path to the materialized extraction configuration a remote extraction job reads.
        tracker: The shared processing tracker.
        workers: The requested worker-process count.
        display_progress: Determines whether to display a progress bar.

    Raises:
        ValueError: If the job_id does not match any job available for this session.
        FileNotFoundError: If a requested extraction job's log archive is not present.
    """
    job_name, specifier = tracker.resolve_job(job_id=job_id, universe=universe)

    if job_name == CONTROLLER_EXTRACTION_JOB_NAME:
        controller_id = specifier
        archive_path = extraction_archives.get(controller_id)
        if archive_path is None:
            message = (
                f"Unable to run the extraction job for controller '{controller_id}'. The communication library "
                f"resolved no log archive for that controller in '{log_directory}'. A controller whose archive is "
                f"absent, or whose name resolves to several archives under that directory, cannot be extracted."
            )
            console.error(message=message, error=FileNotFoundError)
        resolved_workers = resolve_worker_count(requested_workers=workers)
        console.echo(
            message=f"Running '{CONTROLLER_EXTRACTION_JOB_NAME}' job for controller '{controller_id}' (ID: {job_id})..."
        )
        _extract_controller(
            archive_path=archive_path,
            output_directory=extraction_output,
            controller_id=controller_id,
            config_path=config_path,
            job_id=job_id,
            tracker=tracker,
            workers=resolved_workers,
            display_progress=display_progress,
        )
        return

    # Splits the parse specifier, which is built in '{controller_id}-{module_type}-{module_id}' form.
    controller_id, module_type_text, module_id_text = specifier.split("-")
    module_type, module_id = int(module_type_text), int(module_id_text)
    module_parser = parsers[(module_type, module_id)]
    feather_path = resolve_module_path(
        output_directory=extraction_output, source_id=controller_id, module_type=module_type, module_id=module_id
    )

    console.echo(message=f"Running '{PARSE_JOB_NAME}' job with specifier '{specifier}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)
    if not feather_path.is_file():
        console.echo(
            message=(
                f"No extracted data was found for module '{specifier}'. Completing its parse job with no output. "
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


def _run_parse(
    feather_path: Path, module_parser: MicrocontrollerParser, output_directory: Path, session: SessionData
) -> None:
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
    module_parser(event_partition=event_partition, output_directory=output_directory, session=session)
