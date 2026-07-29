"""Provides the single-recording two-photon (calcium-imaging) processing pipeline that drives cindra's stages for one
session.
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

STAGE_DEFAULT_WORKERS: int = -1
"""The worker count that leaves the allocation to cindra's measured default for the stage being run. The orchestration
layer names a positive count instead, which overrides the default with the width the job was admitted at."""

CINDRA_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The filename a locally driven run materializes its single-recording configuration under, inside the session's
cindra directory (``session.processed_data.cindra_data_path``). A run driven by an external scheduler names its
copy after the job it executes, so concurrently dispatched jobs each read their own file."""


def run_two_photon_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    register: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    workers: int = STAGE_DEFAULT_WORKERS,
    display_progress: bool = False,
) -> None:
    """Materializes a session-bound cindra configuration and runs the single-recording two-photon processing pipeline.

    Resolves the session's raw imaging directory (cindra input) through the two-photon data registry and its
    processed-data root (cindra output) from the session hierarchy. Obtains the cindra single-recording configuration
    from the acquisition system's donated resolver, overrides its data path, output path, and progress flag, and
    materializes the result in the session's cindra directory. Then owns the two-photon processing tracker and
    dispatches each stage to cindra as a single tracked job.

    Notes:
        The pipeline runs the single binarization job, one registration job and one processing job per virtual imaging
        plane, and the single combination job, all sharing one tracker. The virtual-plane count is a property of the
        recording's acquisition parameters on disk, recovered by resolving the session's cindra runtime contexts, which
        also persists the shared bootstrap in local mode so the per-job stages can read it. The full four-stage
        universe defines tracker alignment, so a partial invocation never wipes sibling jobs from the shared tracker.

        cindra records each dispatched job's start, completion, and failure directly on this tracker, which lives in
        the session's cindra output directory (``session.processed_data.two_photon_tracker_path``). Two runtimes
        select which jobs execute. In local mode (``job_id`` is None) the requested stages run in binarization to
        registration to processing to combination order, and every stage runs when no flag is set. The registration and
        processing stages honor ``target_plane`` to narrow the pass to one plane. In remote mode (a ``job_id`` is
        provided) only the single job matching that identifier runs, so the stage flags and ``target_plane`` are
        ignored. This lets an external scheduler drive cross-job parallelism by dispatching each identifier
        concurrently.

        The worker count reaches cindra as a call argument, so it applies to the stage this invocation runs rather than
        to every stage a configuration file would have covered.

        An acquisition system that is not a supported AcquisitionSystems member raises before any cindra work begins.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the cindra job to execute. If provided, only the matching job
            runs (remote mode). Otherwise, every requested stage runs (local mode).
        binarize: Determines whether to run the binarization stage. Ignored in remote mode.
        register: Determines whether to run the per-plane motion-correction stage. Ignored in remote mode.
        process: Determines whether to run the per-plane ROI-detection and trace-extraction stage. Ignored in remote
            mode.
        combine: Determines whether to run the multi-plane combination stage. Ignored in remote mode.
        target_plane: The imaging plane to run the per-plane stages for. Set to -1 to cover all planes. Ignored in
            remote mode, where the job to run is selected entirely by job_id.
        workers: The number of workers cindra allocates to each dispatched stage. Set to -1 to accept cindra's measured
            default for the stage it runs.
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

    configuration, materialized_configuration_path = _resolve_configuration(
        session=session, display_progress=display_progress, persist=job_id is None
    )

    # A local run primes the recording single-threaded, while a remote run reads the copy its prepare step primed, so
    # the scheduler's concurrent per-plane jobs share one primed bootstrap.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=job_id is None)
    plane_count = len(contexts)

    universe = _resolve_job_universe(plane_count=plane_count)

    tracker = ProcessingTracker(file_path=session.processed_data.two_photon_tracker_path)

    if job_id is not None:
        # Registers the requested job alone while detecting foreign entries against the full universe, which keeps the
        # sibling jobs this single-job invocation does not run.
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

        job_name, specifier = id_to_job[job_id]
        tracker.align_jobs(jobs=[(job_name, specifier)], universe=universe)

        execute_single_recording_job(
            configuration_path=materialized_configuration_path,
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            job_id=job_id,
            tracker=tracker,
            workers=_resolve_stage_request(workers=workers),
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
    if not (binarize or register or process or combine):
        binarize = register = process = combine = True

    planes = range(plane_count) if target_plane == -1 else (target_plane,)

    jobs: list[tuple[str, str]] = []
    if binarize:
        jobs.append((str(SingleRecordingJobNames.BINARIZE), ""))
    if register:
        jobs.extend((str(SingleRecordingJobNames.REGISTER), f"plane_{plane}") for plane in planes)
    if process:
        jobs.extend((str(SingleRecordingJobNames.PROCESS), f"plane_{plane}") for plane in planes)
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
            workers=_resolve_stage_request(workers=workers),
        )

    console.echo(
        message=f"Single-recording two-photon processing for session '{session.session_name}' completed successfully.",
        level=LogLevel.SUCCESS,
    )


def discover_two_photon_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the two-photon pipeline's job universe and runnable subset for the target session.

    Notes:
        cindra owns this pipeline's job model, so the universe is the single binarization job, one registration job and
        one processing job per virtual imaging plane, and the single combination job. The virtual-plane count is a
        property of the recording's acquisition parameters on disk (ROI x physical plane for MROI data), recovered by
        resolving the session's cindra runtime contexts. Every stage is runnable once the raw imaging directory and
        acquisition parameters exist, which the resolution enforces, so the runnable subset equals the universe. This
        resolver additionally writes the session's cindra configuration.yaml and persists the per-plane bootstrap,
        priming the recording single-threaded before its jobs dispatch.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the runnable
        subset, which equals the universe. Registration and processing specifiers are ``"plane_{index}"`` and the
        binarization and combination specifiers are empty.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory or its cindra acquisition parameters file
            is not present.
        ValueError: If the session's acquisition system is not a supported AcquisitionSystems member, or if the
            acquisition system's resolver cannot resolve a configuration for the session.
    """
    session = SessionData.load(session_path=session_path)
    configuration, _ = _resolve_configuration(session=session, display_progress=False, persist=True)

    # Persisting the bootstrap here primes the recording so its jobs can dispatch.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=True)

    universe = _resolve_job_universe(plane_count=len(contexts))
    return session, universe, list(universe)


def _resolve_job_universe(plane_count: int) -> list[tuple[str, str]]:
    """Builds the two-photon job universe for a recording of the given plane count.

    Notes:
        This is the single statement of the pipeline's job model, so the runtime and the resolver enumerate the same
        jobs in the same order. Registration and processing are the per-plane stages, which cindra declares.

    Args:
        plane_count: The virtual imaging planes the recording holds.

    Returns:
        The job universe as a list of ``(job_name, specifier)`` pairs, in execution order.
    """
    return [
        (str(SingleRecordingJobNames.BINARIZE), ""),
        *((str(SingleRecordingJobNames.REGISTER), f"plane_{plane}") for plane in range(plane_count)),
        *((str(SingleRecordingJobNames.PROCESS), f"plane_{plane}") for plane in range(plane_count)),
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]


def _resolve_stage_request(workers: int) -> int | None:
    """Renders a caller's worker count as the request cindra takes.

    Args:
        workers: The worker count the caller named, or -1 to accept cindra's measured default for the stage.

    Returns:
        The worker count to pass to cindra, or None to accept its measured default.
    """
    return None if workers == STAGE_DEFAULT_WORKERS else workers


def two_photon_job_prerequisites(
    session: SessionData,  # noqa: ARG001
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the two-photon pipeline.

    Notes:
        cindra's stages run in a strict chain: binarization writes the binary every registration job reads,
        registration writes the motion-corrected plane its own processing job reads, and combination merges every
        processing job's output.

        Registration and processing are both per-plane, so a processing job waits on the registration job for its own
        plane alone rather than on all of them. That is what lets one plane reach detection while another is still
        being registered.

    Args:
        session: The loaded session, accepted for the shared dispatch contract and not read by this ordering.
        universe: The job universe as returned by ``discover_two_photon_jobs``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs, following the binarization to registration to
        processing to combination chain.
    """
    binarize_job = (str(SingleRecordingJobNames.BINARIZE), "")
    register_name = str(SingleRecordingJobNames.REGISTER)
    process_name = str(SingleRecordingJobNames.PROCESS)
    combine_name = str(SingleRecordingJobNames.COMBINE)
    process_jobs = tuple(job for job in universe if job[0] == process_name)

    ordering: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for job in universe:
        job_name, specifier = job
        if job_name == register_name:
            ordering[job] = (binarize_job,)
        elif job_name == process_name:
            ordering[job] = ((register_name, specifier),)
        elif job_name == combine_name:
            ordering[job] = process_jobs
        else:
            ordering[job] = ()
    return ordering


def _resolve_configuration(
    session: SessionData, *, display_progress: bool, persist: bool
) -> tuple[SingleRecordingConfiguration, Path]:
    """Resolves the session's cindra input and output locations and materializes its single-recording configuration.

    Notes:
        The output root is the session's processed-data root, under which cindra creates its ``cindra`` subdirectory.
        The configuration comes from the acquisition system's resolver with only the session-bound locations and the
        supplied runtime settings overridden, so every system-resolved processing parameter stands as returned.

        The configuration carries no worker count. cindra takes that as a call argument, so the allocation belongs to
        the invocation that runs a stage rather than to a file every stage of the session shares.

    Args:
        session: The loaded session whose two-photon data is being resolved.
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
