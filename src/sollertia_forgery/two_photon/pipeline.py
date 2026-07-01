"""Provides the single-recording two-photon (calcium-imaging) processing pipeline that resolves the
target session's raw imaging input and processed-output locations, materializes a runnable cindra configuration from
the supplied processing parameters, and drives the cindra binarization, per-plane processing, and combination stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cindra import SingleRecordingConfiguration, run_single_recording_pipeline
from cindra.io import PARAMETERS_FILENAME
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import SessionData

from ..registries import resolve_two_photon_data_locator

if TYPE_CHECKING:
    from pathlib import Path

_MATERIALIZED_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The filename cindra expects for the shared single-recording configuration. The pipeline materializes the
caller's template under this name in the session's cindra directory (``session.processed_data.cindra_data_path``)."""

_SAVED_ACQUISITION_PARAMETERS_FILENAME: str = "acquisition_parameters.yaml"
"""The filename under which cindra persists acquisition parameters in its output directory after the first run. The
pipeline accepts its presence as an alternative to the raw-side ``cindra_parameters.json`` when validating that the
recording is processable."""


def run_two_photon_processing_pipeline(
    session_path: Path,
    configuration_path: Path,
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
    processed-data root (cindra output) from the session hierarchy. It overrides the configuration template's data
    path and output path with these session-resolved locations, and its worker count and progress flag with the
    supplied ``workers`` and ``display_progress`` arguments. It then writes the result as the session's cindra
    ``configuration.yaml`` and delegates the binarization, per-plane processing, and combination stages to cindra. When
    none of ``binarize``, ``process``, or ``combine`` is requested, all three stages run in sequence (local mode); a
    supplied ``job_id`` instead runs only the matching job.

    Notes:
        The raw-imaging input directory is resolved through the system-agnostic two-photon data registry, which
        dispatches to the acquisition system's donated locator. If the registry holds no locator for the session's
        acquisition system, the lookup raises and the pipeline fails before any cindra work begins. cindra owns the
        heavy work and records the run on the two-photon processing tracker (``single_recording_tracker.yaml``,
        ``ProcessingTrackers.TWO_PHOTON``) inside its output subdirectory (``session.processed_data.cindra_data_path``);
        the stage flags map directly onto its stages. Additional ``FileNotFoundError``/``ValueError`` conditions may
        propagate from the underlying cindra pipeline.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        configuration_path: The path to the cindra single-recording configuration template. Its data path, output
            path, worker count, and progress flag are overridden; every other parameter is used as supplied.
        job_id: The unique hexadecimal identifier for the cindra job to execute. If provided, only the matching job
            runs (remote mode); otherwise every requested stage runs (local mode).
        binarize: Determines whether to run the binarization stage.
        process: Determines whether to run the per-plane motion-correction, ROI-detection, and trace-extraction stage.
        combine: Determines whether to run the multi-plane combination stage.
        target_plane: The imaging plane to process when running the processing stage. Set to -1 to process all planes.
        workers: The number of numba worker threads cindra may use. Set to -1 to use all available CPU cores (minus
            reserved cores).
        display_progress: Determines whether to display progress bars during processing.

    Raises:
        KeyError: If the two-photon data registry holds no locator for the session's acquisition system.
        FileNotFoundError: If the configuration file does not exist or is not a YAML file, if the session's raw
            two-photon imaging directory does not exist, or if no cindra acquisition parameters file is available for
            the recording.
        ValueError: If the configuration file cannot be loaded as a cindra single-recording configuration.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=(
            f"Initializing single-recording two-photon processing pipeline for session '{session.session_name}'..."
        ),
        level=LogLevel.INFO,
    )

    # Resolves the recording's raw two-photon imaging directory (cindra input) through the two-photon data registry,
    # which dispatches to the acquisition system's donated locator, and the session's processed-data root (cindra
    # output) from the shared session hierarchy. If the registry holds no locator for the session's acquisition system,
    # this lookup raises, failing the pipeline before any cindra work. cindra creates its 'cindra' output subdirectory
    # under the processed-data root, which is exactly the session's canonical processed cindra directory, so downstream
    # tools find the outputs where they expect them.
    locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
    data_path = locate_two_photon_data(session)
    output_path = session.processed_data_path
    cindra_directory = session.processed_data.cindra_data_path

    # Validates the caller-supplied processing configuration before loading it, so a missing or non-YAML file fails
    # with an actionable message rather than deep inside cindra.
    if not configuration_path.is_file() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. The supplied cindra "
            f"single-recording configuration '{configuration_path}' does not exist or is not a '.yaml' file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Confirms the recording exposes the data cindra needs. The raw imaging directory must exist for the binarization
    # stage, which also excludes sessions that did not acquire two-photon data (their imaging directory is absent).
    if not data_path.is_dir():
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. The raw two-photon imaging "
            f"directory '{data_path}' does not exist, so the session has no calcium-imaging data to process."
        )
        console.error(message=message, error=FileNotFoundError)

    # Confirms the cindra acquisition parameters file is available. Every system producing two-photon data is expected
    # to write 'cindra_parameters.json' alongside the raw imaging data at acquisition time. Once a recording has been
    # processed, cindra also persists the same metadata as 'acquisition_parameters.yaml' next to its outputs, which is
    # accepted here so a re-run can proceed even if the raw data has since been relocated.
    acquisition_parameters_available = (
        any(data_path.rglob(PARAMETERS_FILENAME))
        or cindra_directory.joinpath(_SAVED_ACQUISITION_PARAMETERS_FILENAME).is_file()
    )
    if not acquisition_parameters_available:
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. No cindra acquisition "
            f"parameters file ('{PARAMETERS_FILENAME}') was found under the raw two-photon imaging directory "
            f"'{data_path}'. Every system that produces two-photon data must write this file at acquisition time so "
            f"the cindra pipeline can recover the recording's acquisition metadata."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads the supplied configuration template and overrides only the session-bound locations and runtime settings,
    # leaving every data-specific processing parameter as authored. The materialized copy is written into the cindra
    # output directory, decoupling the reusable template from this session's run.
    try:
        configuration: SingleRecordingConfiguration = SingleRecordingConfiguration.from_yaml(
            file_path=configuration_path
        )
    except Exception:
        message = (
            f"Unable to process two-photon data for session '{session.session_name}'. The file "
            f"'{configuration_path}' could not be loaded as a cindra single-recording configuration. Ensure it is a "
            f"valid single-recording configuration '.yaml' file."
        )
        console.error(message=message, error=ValueError)

    configuration.file_io.data_path = data_path
    configuration.file_io.output_path = output_path
    configuration.runtime.parallel_workers = workers
    configuration.runtime.display_progress_bars = display_progress

    cindra_directory.mkdir(parents=True, exist_ok=True)
    materialized_configuration_path = cindra_directory.joinpath(_MATERIALIZED_CONFIGURATION_FILENAME)
    configuration.save(file_path=materialized_configuration_path)

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
