"""Provides the generic ``slf process`` CLI group that runs a system-agnostic processing pipeline on a single
session.

Notes:
    Each command invokes one system-agnostic worker package's local processing pipeline directly. The acquisition
    system is inferred from the target session by each pipeline internally (resolving its donated parsers/workers from
    the registries), so these commands carry no system selector. The heavy acquisition-library bindings are imported
    lazily inside each command callback so resolving ``slf process --help`` stays inexpensive.
"""

from pathlib import Path

import click

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""

_SESSION_PATH_OPTION = click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the session root directory to process.",
)
_JOB_ID_OPTION = click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode). If not provided, discovers and runs every available job for the session (local mode)."
    ),
)
_WORKERS_OPTION = click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help=(
        "The number of worker processes to use for parallel processing. Set to -1 for automatic resolution. Set to "
        "1 for sequential execution. Ignored when '--job-id' is provided (remote mode runs the single job in-process)."
    ),
)
_PROGRESS_OPTION = click.option(
    "-pr",
    "--progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display a progress bar during processing. Only meaningful in local mode.",
)


@click.group("process", context_settings=CONTEXT_SETTINGS)
def process_cli() -> None:
    """Runs the system-agnostic, in-process data extraction pipelines on a single session."""


@process_cli.command("video")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@click.option(
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
    from ..video import run_video_processing_pipeline  # noqa: PLC0415

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
    from ..microcontrollers import run_microcontroller_processing_pipeline  # noqa: PLC0415

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
    from ..runtime import run_runtime_processing_pipeline  # noqa: PLC0415

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
@click.option("-b", "--binarize", is_flag=True, default=False, help="Run the binarization stage.")
@click.option("-p", "--process", is_flag=True, default=False, help="Run the per-plane processing stage.")
@click.option("-cb", "--combine", is_flag=True, default=False, help="Run the multi-plane combination stage.")
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

    When none of ``--binarize``, ``--process``, or ``--combine`` is requested, all three stages run in sequence.
    """
    from ..two_photon import run_two_photon_processing_pipeline  # noqa: PLC0415

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
