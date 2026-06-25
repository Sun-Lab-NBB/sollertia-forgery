"""Provides the end-to-end multi-recording (across-session cell tracking) cell-activity processing pipeline,
threading the cindra binding in-process as the first stage of ``slf forge``.

Notes:
    cindra owns the heavy per-recording / per-stage job decomposition and writes its own processing trackers at the
    recording root resolved from the configuration file. This wrapper exposes the same decomposition flags so a
    single local invocation and a per-stage remote SLURM graph can both route through the ``slf`` interface instead
    of calling the ``cindra`` CLI directly. The cindra import is deferred to call time so importing this module does
    not require the calcium-imaging library to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path


def run_multidataset_processing_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_recording: str | None = None,
) -> None:
    """Runs the multi-recording cindra processing pipeline (across-session cell tracking) for a dataset.

    Notes:
        When neither ``discover`` nor ``extract`` is requested, both stages run in sequence so a single local
        invocation performs the full multi-recording pipeline. Passing individual stage flags (as the remote SLURM
        graph does) runs only the requested stages, allowing per-recording decomposition via ``target_recording``.
        This is the first stage of the dataset workflow; the assembled fluorescence it produces is consumed by the
        forging pipeline.

    Args:
        configuration_path: The path to the cindra multi-recording configuration file.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the matching
            job is executed (remote mode); otherwise cindra runs every stage requested below.
        discover: Determines whether to run the cross-recording cell-discovery stage.
        extract: Determines whether to run the per-recording aligned-fluorescence extraction stage.
        target_recording: The recording to process when running the ``extract`` stage. Set to None to process all
            recordings.
    """
    # Deferred import: the calcium-imaging binding is only required when a multi-recording job actually runs.
    from cindra import run_multi_recording_pipeline  # noqa: PLC0415

    # A local "run everything" invocation requests all stages when the caller did not select any specific stage.
    if not (discover or extract):
        discover = extract = True

    console.echo(
        message=f"Initializing multi-recording cell-activity processing for '{configuration_path}'...",
        level=LogLevel.INFO,
    )
    run_multi_recording_pipeline(
        configuration_path=configuration_path,
        job_id=job_id,
        discover=discover,
        extract=extract,
        target_recording=target_recording,
    )
    console.echo(message="Multi-recording cell-activity processing completed successfully.", level=LogLevel.SUCCESS)
