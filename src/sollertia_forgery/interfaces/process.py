"""Provides the generic ``slf process`` CLI group that runs a system-agnostic processing pipeline on a single
session. Each command invokes one system-agnostic worker package's local processing pipeline directly. Each
pipeline infers the acquisition system from the target session internally, resolving its donated parsers and
workers from the registries, so each command needs only the target session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import click

from ..video import run_video_processing_pipeline
from ..runtime import run_runtime_processing_pipeline
from ..two_photon import run_two_photon_processing_pipeline
from ..microcontrollers import run_microcontroller_processing_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    from click.decorators import FC

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""

_SESSION_PATH_OPTION: Callable[[FC], FC] = click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the session root directory to process.",
)
"""Defines the shared session-path option that supplies the session root directory to process."""
_JOB_ID_OPTION: Callable[[FC], FC] = click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode). If not provided, discovers and runs every available job for the session (local mode)."
    ),
)
"""Defines the shared job-id option that selects a single processing job or, when omitted, every available job."""
_WORKERS_OPTION: Callable[[FC], FC] = click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help=(
        "The number of parallel workers to use for processing. Set to -1 for automatic resolution. Set to "
        "1 for sequential execution. Jobs that parallelize internally consume this budget as their per-job core "
        "allotment, and single-core jobs treat it as a no-op."
    ),
)
"""Defines the shared workers option that sets the parallel worker budget for a processing command."""
_PROGRESS_OPTION: Callable[[FC], FC] = click.option(
    "-pr",
    "--progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display a progress bar during processing.",
)
"""Defines the shared progress option that toggles the processing progress bar."""


@click.group("process", context_settings=CONTEXT_SETTINGS)
def process_cli() -> None:
    """Runs the requested data extraction pipelines on a single session."""


@process_cli.command("video")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@click.option(
    "-tr/-nt",
    "--track/--no-track",
    default=True,
    show_default=True,
    help=(
        "Determines whether to run the video-tracking stage, which post-processes externally-produced pose "
        "predictions (e.g. DeepLabCut '.h5' files) into tracking feathers. It is a no-op when no predictions are "
        "present. Ignored when '--job-id' is provided (remote mode selects the job by ID)."
    ),
)
@click.option(
    "-en/-ne",
    "--energy/--no-energy",
    default=True,
    show_default=True,
    help=(
        "Determines whether to run the motion-energy stage, which measures each camera's recording into a per-frame "
        "movement signal. It is a no-op for a camera whose recording is absent. Ignored when '--job-id' is provided "
        "(remote mode selects the job by ID)."
    ),
)
@_WORKERS_OPTION
@_PROGRESS_OPTION
def video_command(
    session_path: Path, job_id: str | None, workers: int, *, track: bool, energy: bool, progress: bool
) -> None:
    """Extracts camera frame timestamps, post-processes pose predictions, and measures per-camera motion energy."""
    # Runs the full pipeline (parse + rename), with the tracking and motion-energy stages toggled by their own flags.
    # The stages are passed explicitly so disabling either does not suppress the timestamp stages. In remote mode
    # (job_id set) the stage flags are ignored and the job is selected by ID.
    run_video_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        parse=True,
        rename=True,
        track=track,
        energy=energy,
        workers=workers,
        display_progress=progress,
    )


@process_cli.command("microcontroller")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def microcontroller_command(session_path: Path, job_id: str | None, workers: int, *, progress: bool) -> None:
    """Extracts the microcontroller module log archives and parses them into domain-specific behavior feathers."""
    run_microcontroller_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@process_cli.command("runtime")
@_SESSION_PATH_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def runtime_command(session_path: Path, workers: int, *, progress: bool) -> None:
    """Decodes the acquisition runtime log archive and parses it into the session's runtime behavior feathers."""
    run_runtime_processing_pipeline(
        session_path=session_path,
        workers=workers,
        display_progress=progress,
    )


@process_cli.command("two-photon")
@_SESSION_PATH_OPTION
@click.option(
    "-c",
    "--configuration-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The path to the cindra single-recording configuration file supplying the processing parameters.",
)
@_JOB_ID_OPTION
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    default=False,
    help="Determines whether to run the binarization stage.",
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    default=False,
    help="Determines whether to run the per-plane processing stage.",
)
@click.option(
    "-cb",
    "--combine",
    is_flag=True,
    default=False,
    help="Determines whether to run the multi-plane combination stage.",
)
@click.option(
    "-tp",
    "--target-plane",
    type=int,
    default=-1,
    show_default=True,
    help="The imaging plane to process when running the processing stage. Set to -1 to process all planes.",
)
@_WORKERS_OPTION
@_PROGRESS_OPTION
def two_photon_command(
    session_path: Path,
    configuration_path: Path,
    job_id: str | None,
    target_plane: int,
    workers: int,
    *,
    binarize: bool,
    process: bool,
    combine: bool,
    progress: bool,
) -> None:
    """Runs the single-recording two-photon (calcium-imaging) processing pipeline for a session.

    When none of ``--binarize``, ``--process``, or ``--combine`` is requested, all three stages run in sequence
    (local mode). Supplying ``--job-id`` instead runs only the matching job.
    """
    run_two_photon_processing_pipeline(
        session_path=session_path,
        configuration_path=configuration_path,
        job_id=job_id,
        binarize=binarize,
        process=process,
        combine=combine,
        target_plane=target_plane,
        workers=workers,
        display_progress=progress,
    )
