"""Provides the generic ``slf process`` CLI group that runs a system-specific processing pipeline on a single
session, dispatching to the entry point registered for the session's acquisition system.

Notes:
    The acquisition system is always inferred from the target session's metadata, so these commands carry no system
    selector. The system-agnostic ``video`` pipeline runs directly without registry dispatch.
"""

from pathlib import Path

import click

from .dispatch import infer_system_from_session
from ..pipelines import ProcessingPipelines
from ..registries import resolve_local_pipeline

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
    """Runs the end-to-end, in-process data extraction pipelines on a single session."""


@process_cli.command("behavior")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def behavior_command(session_path: Path, job_id: str | None, workers: int, *, progress: bool) -> None:
    """Extracts the microcontroller log archives and parses runtime and module data into behavior feathers."""
    run_pipeline = resolve_local_pipeline(infer_system_from_session(session_path), ProcessingPipelines.BEHAVIOR)
    run_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@process_cli.command("video")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def video_command(session_path: Path, job_id: str | None, workers: int, *, progress: bool) -> None:
    """Extracts the camera frame acquisition timestamps from the raw VideoSystem log archives."""
    from ..video import run_video_processing_pipeline  # noqa: PLC0415

    run_video_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )
