"""Provides the single-recording two-photon (calcium-imaging) processing pipeline that resolves the
target session's raw imaging input and processed-output locations, obtains a runnable cindra configuration from the
acquisition system's donated resolver, and drives the cindra binarization, per-plane processing, and combination
stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cindra import SingleRecordingJobNames, run_single_recording_pipeline
from cindra.io import PARAMETERS_FILENAME, resolve_single_recording_contexts
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import SessionData

from ..registries import resolve_two_photon_data_locator, resolve_single_recording_configuration_resolver

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import SingleRecordingConfiguration

_MATERIALIZED_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The filename cindra expects for the shared single-recording configuration. The pipeline materializes the
caller's template under this name in the session's cindra directory (``session.processed_data.cindra_data_path``)."""


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
    processed-data root (cindra output) from the session hierarchy. It obtains the cindra single-recording
    configuration from the acquisition system's donated resolver, overrides its data path and output path with these
    session-resolved locations, and overrides its worker count and progress flag with the supplied ``workers`` and
    ``display_progress`` arguments. It then writes the result as the session's cindra ``configuration.yaml`` and
    delegates the binarization, per-plane processing, and combination stages to cindra. When none of ``binarize``,
    ``process``, or ``combine`` is requested, all three stages run in sequence (local mode). A supplied ``job_id``
    instead runs only the matching job.

    Notes:
        The raw-imaging input directory and the configuration are resolved through the system-agnostic two-photon
        registries, which dispatch to the acquisition system's donated assets. Each system decides for itself how its
        configuration is derived, keeping the pipeline agnostic to every system's configuration source. If the
        session's acquisition system is not a supported AcquisitionSystems member, the lookup raises and the pipeline
        fails before any cindra work begins. cindra owns the heavy work and records the run on the two-photon processing
        tracker (``single_recording_tracker.yaml``, ``ProcessingTrackers.TWO_PHOTON``) inside its output subdirectory
        (``session.processed_data.cindra_data_path``). The stage flags map directly onto its stages. Additional
        ``FileNotFoundError``/``ValueError`` conditions may propagate from the resolver or the underlying cindra
        pipeline.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the cindra job to execute. If provided, only the matching job
            runs (remote mode). Otherwise, every requested stage runs (local mode).
        binarize: Determines whether to run the binarization stage.
        process: Determines whether to run the per-plane motion-correction, ROI-detection, and trace-extraction stage.
        combine: Determines whether to run the multi-plane combination stage.
        target_plane: The imaging plane to process when running the processing stage. Set to -1 to process all planes.
        workers: The number of numba worker threads cindra may use. Set to -1 to use all available CPU cores (minus
            reserved cores).
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        FileNotFoundError: If the session's raw two-photon imaging directory does not exist, if no cindra acquisition
            parameters file is available for the recording, or if the acquisition system's resolver reports missing
            inputs it needs to resolve the configuration.
        ValueError: If the session's acquisition system is not a supported AcquisitionSystems member, or if the
            acquisition system's resolver cannot resolve a configuration for the session.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=(
            f"Initializing single-recording two-photon processing pipeline for session '{session.session_name}'..."
        ),
        level=LogLevel.INFO,
    )

    # Resolves the session-bound cindra input and output locations and materializes the run's configuration.yaml,
    # overriding the resolver's worker count and progress flag with the supplied runtime settings.
    _, materialized_configuration_path = _resolve_configuration(
        session=session, workers=workers, display_progress=display_progress
    )

    # Requests all stages when the caller selected none (a local "run everything" invocation), mirroring the cindra
    # single-recording binding's own default.
    if not (binarize or process or combine):
        binarize = process = combine = True

    run_single_recording_pipeline(
        configuration_path=materialized_configuration_path,
        job_id=job_id,
        binarize=binarize,
        process=process,
        combine=combine,
        target_plane=target_plane,
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
        writes the session's cindra configuration.yaml and persists the per-plane bootstrap, matching how a cindra
        prepare step primes a recording single-threaded before its jobs dispatch.

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
    configuration, _ = _resolve_configuration(session=session, workers=-1, display_progress=False)

    # Recovers the virtual-plane count by resolving cindra's per-plane runtime contexts. Persisting the bootstrap here
    # primes the recording so its jobs can dispatch, matching cindra's single-threaded prepare step.
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
        combination job requires every processing job, and the binarization job has no upstream dependency. This is the
        forward reading of the same dependency chain cindra expands in reverse when it resets a phase.

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


def _resolve_configuration(
    session: SessionData, *, workers: int, display_progress: bool
) -> tuple[SingleRecordingConfiguration, Path]:
    """Resolves the session's cindra input and output locations and materializes its single-recording configuration.

    Notes:
        The raw imaging input directory is resolved through the two-photon data registry, which dispatches to the
        acquisition system's donated locator, and the output root is the session's processed-data root. cindra creates
        its ``cindra`` output subdirectory there, which is the session's canonical processed cindra directory. The
        configuration comes from the acquisition system's donated resolver, with only the session-bound locations and
        the supplied runtime settings overridden, leaving every system-resolved processing parameter as returned. The
        materialized copy is written into the cindra output directory so the run is self-describing.

    Args:
        session: The loaded session whose two-photon data is being resolved.
        workers: The number of numba worker threads to record in the configuration's runtime section.
        display_progress: Determines whether cindra displays progress bars during processing. Recorded in the
            configuration's runtime section.

    Returns:
        A tuple of the resolved single-recording configuration and the path to the materialized cindra
        ``configuration.yaml``.

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

    # Confirms the recording exposes the data cindra needs. The raw imaging directory must exist for the binarization
    # stage, which also excludes sessions that did not acquire two-photon data (their imaging directory is absent).
    if not data_path.is_dir():
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. The raw two-photon imaging "
            f"directory '{data_path}' does not exist, so the session has no calcium-imaging data to process."
        )
        console.error(message=message, error=FileNotFoundError)

    # Confirms the cindra acquisition parameters file is available. Every system producing two-photon data writes
    # 'cindra_parameters.json' alongside the raw imaging data at acquisition time, and that raw file is the canonical
    # source of the recording's acquisition metadata.
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
    materialized_configuration_path = cindra_directory.joinpath(_MATERIALIZED_CONFIGURATION_FILENAME)
    configuration.save(file_path=materialized_configuration_path)
    return configuration, materialized_configuration_path
