"""Provides the single-recording two-photon (calcium-imaging) processing pipeline that resolves the
target session's raw imaging input and processed-output locations, obtains a runnable cindra configuration from the
acquisition system's donated resolver, and drives the cindra binarization, per-plane processing, and combination
stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cindra import SingleRecordingJobNames, execute_single_recording_job
from cindra.io import PARAMETERS_FILENAME, resolve_single_recording_contexts
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker

from ..registries import resolve_two_photon_data_locator, resolve_single_recording_configuration_resolver

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import SingleRecordingConfiguration

DEFAULT_CINDRA_WORKERS: int = 10
"""The numba thread count recorded when a caller names none. It matches the lowest count cindra documents as
supported for per-plane processing, so a configuration written by job discovery alone still names a workable value."""

CINDRA_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The filename a locally driven run materializes its single-recording configuration under, inside the session's
cindra directory (``session.processed_data.cindra_data_path``). A run driven by an external scheduler names its
copy after the job it executes, so concurrently dispatched jobs each read their own file."""


def run_two_photon_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Materializes a session-bound cindra configuration and runs the single-recording two-photon processing pipeline.

    Resolves the session's raw imaging directory (cindra input) through the two-photon data registry and its
    processed-data root (cindra output) from the session hierarchy. Obtains the cindra single-recording configuration
    from the acquisition system's donated resolver, overrides its data path, output path, worker count, and progress
    flag, and materializes the result in the session's cindra directory. Then owns the two-photon processing tracker
    and dispatches each stage to cindra as a single tracked job.

    Notes:
        The pipeline runs the single binarization job, one processing job per virtual imaging plane, and the single
        combination job, all sharing one tracker. The virtual-plane count is a property of the recording's acquisition
        parameters on disk, recovered by resolving the session's cindra runtime contexts, which also persists the
        shared bootstrap in local mode so the per-job stages can read it. The full binarization, per-plane processing,
        and combination universe defines tracker alignment, so a partial invocation never wipes sibling jobs from the
        shared tracker.

        cindra records each dispatched job's start, completion, and failure directly on this tracker, which lives in
        the session's cindra output directory (``session.processed_data.two_photon_tracker_path``). Two runtimes
        select which jobs execute. In local mode (``job_id`` is None) the requested stages run in binarization to
        processing to combination order, and every stage runs when no flag is set. The processing stage honors
        ``target_plane`` to narrow the pass to one plane. In remote mode (a ``job_id`` is provided) only the single
        job matching that identifier runs, so the stage flags and ``target_plane`` are ignored. This lets an external
        scheduler drive cross-job parallelism by dispatching each identifier concurrently.

        An acquisition system that is not a supported AcquisitionSystems member raises before any cindra work begins.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the cindra job to execute. If provided, only the matching job
            runs (remote mode). Otherwise, every requested stage runs (local mode).
        binarize: Determines whether to run the binarization stage. Ignored in remote mode.
        process: Determines whether to run the per-plane motion-correction, ROI-detection, and trace-extraction stage.
            Ignored in remote mode.
        combine: Determines whether to run the multi-plane combination stage. Ignored in remote mode.
        target_plane: The imaging plane to process when running the processing stage. Set to -1 to process all planes.
            Ignored in remote mode, where the job to run is selected entirely by job_id.
        workers: The number of numba worker threads cindra may use, recorded into the session's configuration. Set
            to -1 to use all available CPU cores minus the reserved cores. Applies in local mode only, since a job
            selected by identifier reads the configuration its preparation step already wrote.
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory does not exist, if no cindra acquisition
            parameters file is available for the recording, or if the acquisition system's resolver reports missing
            inputs it needs to resolve the configuration.
        ValueError: If the session's acquisition system is not a supported AcquisitionSystems member, if the
            acquisition system's resolver cannot resolve a configuration for the session, or if job_id does not match
            any available job.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=(
            f"Initializing single-recording two-photon processing pipeline for session '{session.session_name}'..."
        ),
        level=LogLevel.INFO,
    )

    # A remotely dispatched run reads the configuration its preparation step wrote, so the worker count recorded in
    # that copy governs and this run's own worker argument does not apply.
    configuration, materialized_configuration_path = _resolve_configuration(
        session=session, workers=workers, display_progress=display_progress, persist=job_id is None
    )

    # A local run primes the recording single-threaded, while a remote run reads the copy its prepare step primed, so
    # the scheduler's concurrent per-plane jobs share one primed bootstrap.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=job_id is None)
    plane_count = len(contexts)

    universe: list[tuple[str, str]] = [
        (str(SingleRecordingJobNames.BINARIZE), ""),
        *((str(SingleRecordingJobNames.PROCESS), f"plane_{plane_index}") for plane_index in range(plane_count)),
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]

    tracker = ProcessingTracker(file_path=session.processed_data.two_photon_tracker_path)

    if job_id is not None:
        # Aligning against the full universe keeps the sibling jobs this single-job invocation does not run.
        id_to_job = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in universe
        }
        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any two-photon "
                f"processing job available for this session. Valid job IDs: {sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)

        tracker.align_jobs(jobs=universe, universe=universe)

        job_name, specifier = id_to_job[job_id]
        execute_single_recording_job(
            configuration_path=materialized_configuration_path,
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            job_id=job_id,
            tracker=tracker,
        )
        console.echo(
            message=(
                f"Single-recording two-photon processing job for session '{session.session_name}' completed "
                f"successfully."
            ),
            level=LogLevel.SUCCESS,
        )
        return

    # Local mode. An invocation naming no stage runs every stage, matching cindra's single-recording resolution.
    if not (binarize or process or combine):
        binarize = process = combine = True

    jobs: list[tuple[str, str]] = []
    if binarize:
        jobs.append((str(SingleRecordingJobNames.BINARIZE), ""))
    if process:
        if target_plane == -1:
            jobs.extend(
                (str(SingleRecordingJobNames.PROCESS), f"plane_{plane_index}") for plane_index in range(plane_count)
            )
        else:
            jobs.append((str(SingleRecordingJobNames.PROCESS), f"plane_{target_plane}"))
    if combine:
        jobs.append((str(SingleRecordingJobNames.COMBINE), ""))

    tracker.align_jobs(jobs=jobs, universe=universe)

    console.echo(message=f"Running {len(jobs)} two-photon processing job(s).")

    for job_name, specifier in jobs:
        job_identifier = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        execute_single_recording_job(
            configuration_path=materialized_configuration_path,
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            job_id=job_identifier,
            tracker=tracker,
        )

    console.echo(
        message=f"Single-recording two-photon processing for session '{session.session_name}' completed successfully.",
        level=LogLevel.SUCCESS,
    )


def discover_two_photon_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the two-photon pipeline's job universe and runnable subset for the target session.

    Notes:
        cindra owns this pipeline's job model, so the universe is the single binarization job, one processing job per
        virtual imaging plane, and the single combination job. The virtual-plane count is a property of the recording's
        acquisition parameters on disk (ROI x physical plane for MROI data), recovered by resolving the session's
        cindra runtime contexts. Every stage is runnable once the raw imaging directory and acquisition parameters
        exist, which the resolution enforces, so the runnable subset equals the universe. This resolver additionally
        writes the session's cindra configuration.yaml and persists the per-plane bootstrap, priming the recording
        single-threaded before its jobs dispatch.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the runnable
        subset, which equals the universe. Processing specifiers are ``"plane_{index}"`` and the binarization and
        combination specifiers are empty.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory or its cindra acquisition parameters file
            is not present.
        ValueError: If the session's acquisition system is not a supported AcquisitionSystems member, or if the
            acquisition system's resolver cannot resolve a configuration for the session.
    """
    session = SessionData.load(session_path=session_path)
    configuration, _ = _resolve_configuration(
        session=session, workers=DEFAULT_CINDRA_WORKERS, display_progress=False, persist=True
    )

    # Persisting the bootstrap here primes the recording so its jobs can dispatch.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=True)
    plane_count = len(contexts)

    universe: list[tuple[str, str]] = [
        (str(SingleRecordingJobNames.BINARIZE), ""),
        *((str(SingleRecordingJobNames.PROCESS), f"plane_{plane_index}") for plane_index in range(plane_count)),
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]
    return session, universe, list(universe)


def two_photon_job_prerequisites(
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the two-photon pipeline.

    Notes:
        cindra's stages run in a strict chain: binarization writes the inputs each per-plane processing job reads, and
        combination merges every processing job's output. So each processing job requires the binarization job, the
        combination job requires every processing job, and the binarization job has no upstream dependency.

    Args:
        universe: The job universe as returned by ``discover_two_photon_jobs``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs, following the binarization to processing to
        combination chain.
    """
    binarize_job = (str(SingleRecordingJobNames.BINARIZE), "")
    process_name = str(SingleRecordingJobNames.PROCESS)
    combine_name = str(SingleRecordingJobNames.COMBINE)
    process_jobs = tuple(job for job in universe if job[0] == process_name)

    return {
        job: (binarize_job,) if job[0] == process_name else process_jobs if job[0] == combine_name else ()
        for job in universe
    }


def materialize_cindra_configuration(session: SessionData, workers: int) -> Path:
    """Writes the session's cindra configuration with the worker count its processing jobs will run under.

    Notes:
        cindra reads the worker count from this file rather than from a call argument, and only its per-plane
        processing stage consumes the value. An external scheduler therefore prepares this file once, before any job
        of the session dispatches, and every job of that session reads the same copy.

    Args:
        session: The loaded session whose configuration is written.
        workers: The numba thread count each per-plane processing job runs under.

    Returns:
        The path the configuration was written to.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory or its acquisition parameters file is
            not present.
        ValueError: If the session's acquisition system cannot resolve a two-photon configuration.
    """
    _, configuration_path = _resolve_configuration(
        session=session, workers=workers, display_progress=False, persist=True
    )
    return configuration_path


def _resolve_configuration(
    session: SessionData, *, workers: int, display_progress: bool, persist: bool
) -> tuple[SingleRecordingConfiguration, Path]:
    """Resolves the session's cindra input and output locations and materializes its single-recording configuration.

    Notes:
        The output root is the session's processed-data root, under which cindra creates its ``cindra`` subdirectory.
        The configuration comes from the acquisition system's resolver with only the session-bound locations and the
        supplied runtime settings overridden, so every system-resolved processing parameter stands as returned.

    Args:
        session: The loaded session whose two-photon data is being resolved.
        workers: The number of numba worker threads to record in the configuration's runtime section.
        display_progress: Determines whether cindra displays progress bars during processing. Recorded in the
            configuration's runtime section.
        persist: Determines whether the resolved configuration is written to disk. Only the invocation that owns the
            session's configuration writes it, so a job dispatched alongside its siblings reads a stable file.

    Returns:
        A tuple of the resolved single-recording configuration and the path it was materialized to, which is the
        session's shared ``configuration.yaml`` in local mode and a per-job file in remote mode.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory or its cindra acquisition parameters file
            is not present.
        ValueError: If the session's acquisition system is not a supported AcquisitionSystems member, or if the
            acquisition system's resolver cannot resolve a configuration for the session.
    """
    locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
    data_path = locate_two_photon_data(session)
    output_path = session.processed_data_path
    cindra_directory = session.processed_data.cindra_data_path

    # A session that acquired no two-photon data has no raw imaging directory, so this also screens those out.
    if not data_path.is_dir():
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. The raw two-photon imaging "
            f"directory '{data_path}' does not exist, so the session has no calcium-imaging data to process."
        )
        console.error(message=message, error=FileNotFoundError)

    # The raw parameters file is the canonical source of the recording's acquisition metadata.
    if not any(data_path.rglob(PARAMETERS_FILENAME)):
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. No cindra acquisition "
            f"parameters file ('{PARAMETERS_FILENAME}') was found under the raw two-photon imaging directory "
            f"'{data_path}'. Every system that produces two-photon data must write this file at acquisition time so "
            f"the cindra pipeline can recover the recording's acquisition metadata."
        )
        console.error(message=message, error=FileNotFoundError)

    resolve_single_recording_configuration = resolve_single_recording_configuration_resolver(
        system=session.acquisition_system
    )
    configuration = resolve_single_recording_configuration(session)

    configuration.file_io.data_path = data_path
    configuration.file_io.output_path = output_path
    configuration.runtime.parallel_workers = workers
    configuration.runtime.display_progress_bars = display_progress

    cindra_directory.mkdir(parents=True, exist_ok=True)

    # cindra reads this file once per job and never writes it, so one copy per session serves every job the session
    # dispatches. Confining the write to the owning invocation keeps it outside the window in which jobs run
    # concurrently.
    materialized_configuration_path = cindra_directory.joinpath(CINDRA_CONFIGURATION_FILENAME)
    if persist:
        configuration.save(file_path=materialized_configuration_path)
    elif not materialized_configuration_path.is_file():
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. No cindra configuration was "
            f"found at '{materialized_configuration_path}'. A job dispatched by an external scheduler reads the "
            f"configuration its preparation step wrote, so that step must run before the job."
        )
        console.error(message=message, error=FileNotFoundError)
    return configuration, materialized_configuration_path
