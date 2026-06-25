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

if TYPE_CHECKING:
    from pathlib import Path

_MATERIALIZED_CONFIGURATION_FILENAME: str = "configuration.yaml"
"""The filename of the session-resident cindra processing configuration that the pipeline materializes from the
caller-supplied configuration template with the session's data and output paths injected. cindra reads this file
(and re-persists it) during processing; it is written into the session's canonical processed cindra directory
alongside the cindra outputs, matching the location cindra itself uses for the shared configuration."""

_SAVED_ACQUISITION_PARAMETERS_FILENAME: str = "acquisition_parameters.yaml"
"""The filename of the acquisition parameters cindra persists into its output directory after the first run. Its
presence lets re-runs recover the recording's acquisition metadata without the raw imaging data, so the pipeline
treats it as an alternative to the raw-side ``cindra_parameters.json`` when confirming the recording is processable."""


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

    Notes:
        This is the end-to-end, single-recording calcium-imaging pipeline. It resolves the recording's raw two-photon
        imaging directory (the cindra input) and the session's processed-data root (the cindra output) from the
        shared session hierarchy, so neither location is passed as an argument. It then loads the supplied processing
        configuration template, overrides its input and output paths with the session-resolved locations and its
        runtime worker count, and writes the result as the session's cindra ``configuration.yaml``. The cindra
        single-recording binding consumes that materialized configuration and owns the heavy work: it reads the
        recording's acquisition parameters, decomposes the run into the binarization, per-plane processing, and
        combination stages, and records every stage on its own processing tracker at the cindra output root.
        All outputs land in the session's canonical processed cindra directory
        (``session.processed_data.cindra_data_path``).

        The configuration is always supplied by the caller and is never defaulted: it carries the data-specific
        processing parameters (registration, ROI detection, signal extraction, and so on), while the session supplies
        only the data and output locations. The recording-specific acquisition metadata is read by cindra from the
        cindra acquisition parameters file (``cindra_parameters.json``) that every system producing two-photon data
        writes alongside the raw imaging data at acquisition time.

        When none of ``binarize``, ``process``, or ``combine`` is requested, all three stages run in sequence, so a
        single local invocation performs the full pipeline. Passing individual stage flags runs only the requested
        stages, which (together with ``target_plane`` and ``job_id``) lets an external scheduler run each stage, and
        each imaging plane, as an independent job.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        configuration_path: The path to the cindra single-recording configuration file describing the per-recording
            processing parameters. The pipeline overrides the file's data path, output path, worker count, and
            progress flag before processing; every other parameter is used as supplied.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the matching
            cindra job is executed (remote mode); otherwise every requested stage is executed (local mode).
        binarize: Determines whether to run the binarization stage.
        process: Determines whether to run the per-plane motion-correction, ROI-detection, and trace-extraction stage.
        combine: Determines whether to run the multi-plane combination stage.
        target_plane: The imaging plane to process when running the processing stage. Set to -1 to process all planes.
        workers: The number of worker processes the cindra pipeline may use. Set to -1 to use all available CPU cores
            (minus reserved cores).
        display_progress: Determines whether to display progress bars during processing.

    Raises:
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

    # Resolves the recording's raw two-photon imaging directory (cindra input) and the session's processed-data root
    # (cindra output) from the shared session hierarchy. cindra creates its 'cindra' output subdirectory under the
    # processed-data root, which is exactly the session's canonical processed cindra directory, so downstream tools
    # find the outputs where they expect them.
    data_path = session.system_raw_data.mesoscope_data_path
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
    # to write 'cindra_parameters.json' alongside the raw imaging data at acquisition time; once a recording has been
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

    # A local "run everything" invocation requests all stages when the caller did not select any specific stage,
    # mirroring the cindra single-recording binding's own default.
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
