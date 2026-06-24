"""Provides the end-to-end, two-stage microcontroller log processing pipeline: extracts raw per-module data from
controller log archives via the acquisition library, then parses each module into a domain-specific feather using
the parser provider registered for the session's acquisition system.
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
from ataraxis_communication_interface.microcontroller import EXTRACTION_JOB_NAME

from .parsers import resolve_parsers
from .extraction import extract_controller, resolve_controllers, find_controller_archive
from ..orchestration import prepare_tracker
from ..shared_assets import partition_events, find_module_feathers, parse_module_feather_name

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping
    from concurrent.futures import Future

    from ataraxis_communication_interface.microcontroller import ControllerExtractionConfig

    from .parsers import ModuleParser

PARSE_JOB_NAME: str = "module_parsing"
"""The job name identifying per-module parsing jobs in the microcontroller processing tracker. Stage 1 extraction
jobs reuse the acquisition library's own extraction job name so their tracker identifiers match the binding that
records their state."""


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
        ataraxis-communication-interface binding and writes raw per-module feather files into the session's
        ``processed_data/microcontroller_data`` directory. Stage 2 (parsing) reads each raw module feather,
        partitions it by event code, and runs the per-system parser to write the domain-specific feather. The
        per-system parsers are looked up from the unified parser registry by the session's acquisition system, so
        the pipeline itself stays system-agnostic.

        In local mode (job_id is None), Stage 1 runs for every configured controller whose archive is present
        (sequentially, with message decoding parallelized within each archive), then Stage 2 parses all eligible
        modules (distributed across a worker pool when more than one worker is available). In remote mode (job_id
        is provided), only the single matching job runs in-process: an extraction job for one controller, or a
        parse job for one module (whose raw feather must already exist from a prior extraction run).

        The processing tracker is co-located with the parsed output in the session's ``behavior_data`` directory,
        while the raw per-module feathers remain in ``microcontroller_data``.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The hexadecimal identifier of the single job to execute (remote mode). If not provided, the whole
            pipeline runs (local mode).
        workers: The number of worker processes to use. A value less than 1 uses all available CPU cores (minus
            reserved cores); 1 forces sequential processing.
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        ValueError: If no parser provider is registered for the session's acquisition system, if no processable
            controllers are discovered, or if the provided job_id does not match any available job.
    """
    session = SessionData.load(session_path=session_path)
    console.echo(
        message=f"Initializing microcontroller processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Looks up the per-system parsers from the unified registry by the session's acquisition system, then resolves
    # the eligible module parsers for this session (binding any system configuration and output paths).
    provider = resolve_parsers(session.acquisition_system)
    parsers = provider.resolve(session=session)

    # Loads the per-controller extraction configurations (validated against the microcontroller manifest).
    controllers = resolve_controllers(session=session)

    log_directory = session.raw_data.behavior_data_path
    extraction_output = session.processed_data.microcontroller_data_path

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

    # The tracker lives with the final parsed data in ``behavior_data``, not with the raw intermediates in
    # ``microcontroller_data``. The same job universe drives foreign-entry detection in both local and remote modes,
    # so a single concurrent remote job aligns the tracker without resetting its sibling jobs.
    tracker_directory = session.processed_data.behavior_data_path
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
            log_directory=log_directory,
            extraction_output=extraction_output,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
        )
    else:
        _run_extraction_stage(
            extraction_archives=extraction_archives,
            controllers=controllers,
            extraction_output=extraction_output,
            tracker=tracker,
            extraction_job_name=EXTRACTION_JOB_NAME,
            workers=workers,
            display_progress=display_progress,
        )
        _run_parse_stage(
            parse_specifiers=parse_specifiers,
            parsers=parsers,
            extraction_output=extraction_output,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
        )

    console.echo(message="All microcontroller processing jobs completed successfully.", level=LogLevel.SUCCESS)


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

        archive_path = find_controller_archive(log_directory=log_directory, controller_id=controller_id)
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
    display_progress: bool,
) -> None:
    """Runs Stage 1: extracts each present controller's archive into raw per-module feathers.

    Notes:
        Controllers are extracted sequentially in the parent process because the acquisition binding manages each
        extraction job's tracker state internally; running the bindings concurrently would race on the shared
        tracker file. Parallelism instead stays inside each archive via a shared process pool that decodes message
        batches, matching how the acquisition library orchestrates a multi-controller directory.

    Args:
        extraction_archives: The present controllers' archive paths, keyed by controller ID.
        controllers: The per-controller extraction configurations, keyed by controller ID.
        extraction_output: The directory where raw per-module feathers are written.
        tracker: The shared processing tracker.
        extraction_job_name: The acquisition library's extraction job name used to derive each job identifier.
        workers: The requested worker-process count.
        display_progress: Determines whether to display a per-controller progress bar.
    """
    if not extraction_archives:
        return

    resolved_workers = resolve_worker_count(requested_workers=workers)
    shared_executor = ProcessPoolExecutor(max_workers=resolved_workers) if resolved_workers > 1 else None

    progress_context = (
        console.progress(
            total=len(extraction_archives), description="Extracting microcontroller logs", unit="controller"
        )
        if display_progress
        else nullcontext()
    )

    try:
        with progress_context as progress_bar:
            for controller_id, archive_path in extraction_archives.items():
                extraction_job_id = ProcessingTracker.generate_job_id(
                    job_name=extraction_job_name, specifier=controller_id
                )
                console.echo(
                    message=(
                        f"Running '{extraction_job_name}' job for controller '{controller_id}' "
                        f"(ID: {extraction_job_id})..."
                    )
                )
                extract_controller(
                    archive_path=archive_path,
                    output_directory=extraction_output,
                    controller_id=controller_id,
                    controller_config=controllers[controller_id],
                    job_id=extraction_job_id,
                    tracker=tracker,
                    workers=resolved_workers,
                    display_progress=False,
                    executor=shared_executor,
                )
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)


def _run_parse_stage(
    parse_specifiers: dict[str, tuple[str, int, int]],
    parsers: Mapping[tuple[int, int], ModuleParser],
    extraction_output: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None:
    """Runs Stage 2: parses each eligible module's raw feather into its domain-specific feather.

    Notes:
        The acquisition binding writes a raw feather only for modules that produced at least one message, so a
        configured, eligible module can legitimately have no feather. Such a parse job is completed with no output
        rather than left unresolved. Modules with a feather are dispatched to a worker pool when more than one
        worker is available, with the parent owning all tracker state transitions.

    Args:
        parse_specifiers: The requested parse specifiers mapped to their ``(controller_id, type, id)`` triples.
        parsers: The eligible module parsers, keyed by ``(module_type, module_id)``.
        extraction_output: The directory holding the raw per-module feathers.
        tracker: The shared processing tracker.
        workers: The requested worker-process count.
        display_progress: Determines whether to display a per-module progress bar.
    """
    if not parse_specifiers:
        return

    runnable: dict[str, tuple[Path, ModuleParser]] = {}
    for specifier, (controller_id, module_type, module_id) in parse_specifiers.items():
        feather_path = _locate_module_feather(
            extraction_output=extraction_output,
            controller_id=controller_id,
            module_type=module_type,
            module_id=module_id,
        )
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

    resolved_workers = resolve_worker_count(requested_workers=workers)
    if resolved_workers > 1 and len(runnable) > 1:
        _execute_parse_jobs_parallel(
            runnable=runnable, tracker=tracker, workers=resolved_workers, display_progress=display_progress
        )
    else:
        _execute_parse_jobs_sequential(runnable=runnable, tracker=tracker, display_progress=display_progress)


def _execute_parse_jobs_sequential(
    runnable: dict[str, tuple[Path, ModuleParser]],
    tracker: ProcessingTracker,
    *,
    display_progress: bool,
) -> None:
    """Runs the parse jobs sequentially in the parent process with full tracker state management.

    Args:
        runnable: The parse jobs mapping each specifier to its ``(feather_path, module_parser)`` pair.
        tracker: The shared processing tracker.
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
            tracker.start_job(job_id=job_id)
            try:
                _run_parse(feather_path=feather_path, module_parser=module_parser)
                tracker.complete_job(job_id=job_id)
            except Exception as exception:
                tracker.fail_job(job_id=job_id, error_message=str(exception))
                raise
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_parse_jobs_parallel(
    runnable: dict[str, tuple[Path, ModuleParser]],
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None:
    """Runs the parse jobs concurrently across a process pool, with the parent owning tracker state.

    Notes:
        Each job's tracker state is advanced to running immediately before its future is submitted, then resolved
        as the future completes. In-flight futures are allowed to finish on failure so the tracker stays accurate
        for every dispatched job; the first captured exception is re-raised after all futures resolve.

    Args:
        runnable: The parse jobs mapping each specifier to its ``(feather_path, module_parser)`` pair.
        tracker: The shared processing tracker.
        workers: The resolved worker-process count for the pool.
        display_progress: Determines whether to display a per-module progress bar.
    """
    first_exception: Exception | None = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for specifier, (feather_path, module_parser) in runnable.items():
            job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier=specifier)
            console.echo(message=f"Running '{PARSE_JOB_NAME}' job with specifier '{specifier}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(_run_parse, feather_path=feather_path, module_parser=module_parser)
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
    log_directory: Path,
    extraction_output: Path,
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
        parsers: The eligible module parsers, keyed by ``(module_type, module_id)``.
        log_directory: The raw behavior data directory holding the controller log archives.
        extraction_output: The directory holding (or receiving) the raw per-module feathers.
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
        archive_path = find_controller_archive(log_directory=log_directory, controller_id=controller_id)
        if archive_path is None:
            message = (
                f"Unable to run the extraction job for controller '{controller_id}'. No log archive "
                f"'{controller_id}_log.npz' was found under '{log_directory}'."
            )
            console.error(message=message, error=FileNotFoundError)
        resolved_workers = resolve_worker_count(requested_workers=workers)
        console.echo(message=f"Running '{extraction_job_name}' job for controller '{controller_id}' (ID: {job_id})...")
        extract_controller(
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
    feather_path = _locate_module_feather(
        extraction_output=extraction_output, controller_id=controller_id, module_type=module_type, module_id=module_id
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
        _run_parse(feather_path=feather_path, module_parser=module_parser)
        tracker.complete_job(job_id=job_id)
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _run_parse(feather_path: Path, module_parser: ModuleParser) -> None:
    """Parses one raw module feather into its domain-specific feather.

    Notes:
        This is the atomic unit of work dispatched to worker processes by the parallel parse path, so it must
        remain importable at module level and accept only picklable arguments. It reads the raw module feather via
        memory mapping, partitions it by event code in a single pass, and delegates to the system-bound parser.

    Args:
        feather_path: The path to the raw per-module feather produced by the extraction stage.
        module_parser: The system-bound parser and output destination for this module.
    """
    module_dataframe = pl.read_ipc(source=feather_path, memory_map=True)
    event_partition = partition_events(module_dataframe=module_dataframe)
    module_parser.output_path.parent.mkdir(parents=True, exist_ok=True)
    module_parser.parse(event_partition, module_parser.output_path)


def _locate_module_feather(
    extraction_output: Path, controller_id: str, module_type: int, module_id: int
) -> Path | None:
    """Locates the raw feather for a specific module among the extraction outputs.

    Args:
        extraction_output: The directory holding the raw per-module feathers.
        controller_id: The controller ID the module belongs to.
        module_type: The module type code.
        module_id: The module instance ID.

    Returns:
        The path to the module's raw feather, or None if the extraction stage produced none for it.
    """
    for feather_path in find_module_feathers(data_directory=extraction_output):
        feather_controller, feather_type, feather_id = parse_module_feather_name(feather_path=feather_path)
        if str(feather_controller) == controller_id and feather_type == module_type and feather_id == module_id:
            return feather_path
    return None


def _split_parse_specifier(specifier: str) -> tuple[str, int, int]:
    """Splits a parse-job specifier into its controller ID, module type, and module ID components.

    Args:
        specifier: The parse specifier in ``"{controller_id}-{module_type}-{module_id}"`` form.

    Returns:
        A tuple of (controller_id, module_type, module_id).
    """
    controller_id, module_type, module_id = specifier.split("-")
    return controller_id, int(module_type), int(module_id)
