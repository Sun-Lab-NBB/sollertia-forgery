"""Provides the generic ``slf process`` CLI group that runs a system-agnostic processing pipeline on one session."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import click
from ataraxis_base_utilities import console

from ..video import run_video_processing_pipeline
from ..runtime import run_runtime_processing_pipeline
from ..two_photon import run_two_photon_processing_pipeline
from ..microcontrollers import run_microcontroller_processing_pipeline

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@dataclass(frozen=True, slots=True)
class _SharedProcessingParameters:
    """Bundles the options parsed on the ``process`` group and shared across its ``video``, ``microcontroller``,
    ``runtime``, and ``two-photon`` subcommands.

    The group callback builds one of these from its options and stores it on the Click context, and each subcommand
    reads it back through the ``_pass_shared_parameters`` decorator. The ``runtime`` subcommand uses only
    ``session_path``, ``workers``, and ``display_progress``, since its single-job pipeline has no remote-dispatch job.
    """

    session_path: Path | None
    """The path to the session root directory every subcommand processes."""

    job_id: str | None
    """The unique hexadecimal identifier selecting a single processing job (remote mode), or None to run every
    available job for the session (local mode)."""

    workers: int
    """The parallel worker budget for the pipeline. A value of -1 resolves the host's available cores for the video,
    microcontroller, and runtime pipelines, and accepts cindra's measured per-stage default for the two-photon
    pipeline. A value of 1 forces sequential execution."""

    display_progress: bool
    """Determines whether the pipeline displays a progress bar during processing."""

    def require_session_path(self) -> Path:
        """Returns the session root path, raising a Click usage error when ``--session-path`` was not supplied."""
        if self.session_path is None:
            message = (
                "Unable to resolve the session root directory for the 'process' command. The '-sp' / "
                "'--session-path' option must be supplied before the subcommand name, but it was omitted."
            )
            console.error(message=message, error=click.UsageError)
        return self.session_path


_pass_shared_parameters = click.make_pass_decorator(_SharedProcessingParameters)
"""Injects the ``process`` group's ``_SharedProcessingParameters`` as each subcommand's first argument."""


@click.group("process", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="The absolute path to the session root directory to process.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode). If not provided, discovers and runs every available job for the session (local mode). The "
        "'runtime' subcommand ignores it, since its single-job pipeline has no remote-dispatch job."
    ),
)
@click.option(
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
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to suppress the progress bar during processing. The progress bar is displayed by default.",
)
@click.pass_context
def process_cli(
    context: click.Context, session_path: Path | None, job_id: str | None, workers: int, *, no_progress: bool
) -> None:
    """Runs the requested data extraction pipelines on a single session.

    The session path, job id, worker budget, and no-progress flag are parsed on this group and shared by every
    subcommand, so they must be given before the subcommand name.
    """
    context.obj = _SharedProcessingParameters(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=not no_progress,
    )


@process_cli.command("video", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-ts",
    "--timestamp",
    is_flag=True,
    default=False,
    help=(
        "Determines whether to run the timestamp stage, which parses each camera's log archive into a frame-timestamp "
        "feather and republishes it under its canonical manifest name. Ignored when '--job-id' is provided (remote "
        "mode selects the job by ID)."
    ),
)
@click.option(
    "-tr",
    "--track",
    is_flag=True,
    default=False,
    help=(
        "Determines whether to run the video-tracking stage, which post-processes externally-produced pose "
        "predictions (e.g. DeepLabCut '.h5' files) into tracking feathers. It is a no-op when no predictions are "
        "present. Ignored when '--job-id' is provided (remote mode selects the job by ID)."
    ),
)
@click.option(
    "-en",
    "--energy",
    is_flag=True,
    default=False,
    help=(
        "Determines whether to run the motion-energy stage, which measures each camera's recording into a per-frame "
        "movement signal. It is a no-op for a camera whose recording is absent. Ignored when '--job-id' is provided "
        "(remote mode selects the job by ID)."
    ),
)
@click.option(
    "-tc",
    "--target-camera",
    type=int,
    default=-1,
    show_default=True,
    help=(
        "The numeric source ID of the single camera to process for the timestamp and motion-energy stages. Set to -1 "
        "to process every camera. Ignored when '--job-id' is provided (remote mode selects the job by ID)."
    ),
)
@_pass_shared_parameters
def video_command(
    shared: _SharedProcessingParameters,
    target_camera: int,
    *,
    timestamp: bool,
    track: bool,
    energy: bool,
) -> None:
    """Extracts camera frame timestamps, processes pose predictions, and measures per-camera motion energy.

    When none of '--timestamp', '--track', or '--energy' is requested, all three stages run (local mode). Supplying
    '--job-id' instead runs only the matching job.
    """
    run_video_processing_pipeline(
        session_path=shared.require_session_path(),
        job_id=shared.job_id,
        timestamp=timestamp,
        track=track,
        energy=energy,
        target_camera=target_camera,
        workers=shared.workers,
        display_progress=shared.display_progress,
    )


@process_cli.command("microcontroller", context_settings=_CONTEXT_SETTINGS)
@_pass_shared_parameters
def microcontroller_command(shared: _SharedProcessingParameters) -> None:
    """Extracts the microcontroller module log archives and parses them into domain-specific behavior feathers."""
    run_microcontroller_processing_pipeline(
        session_path=shared.require_session_path(),
        job_id=shared.job_id,
        workers=shared.workers,
        display_progress=shared.display_progress,
    )


@process_cli.command("runtime", context_settings=_CONTEXT_SETTINGS)
@_pass_shared_parameters
def runtime_command(shared: _SharedProcessingParameters) -> None:
    """Decodes the acquisition runtime log archive and parses it into the session's runtime behavior feathers."""
    run_runtime_processing_pipeline(
        session_path=shared.require_session_path(),
        workers=shared.workers,
        display_progress=shared.display_progress,
    )


@process_cli.command("two-photon", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    default=False,
    help="Determines whether to run the binarization stage.",
)
@click.option(
    "-r",
    "--register",
    is_flag=True,
    default=False,
    help="Determines whether to run the per-plane registration stage.",
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
    help="The imaging plane that the per-plane stages process. Set to -1 to cover all planes.",
)
@_pass_shared_parameters
def two_photon_command(
    shared: _SharedProcessingParameters,
    target_plane: int,
    *,
    binarize: bool,
    register: bool,
    process: bool,
    combine: bool,
) -> None:
    """Runs the single-recording two-photon (calcium-imaging) processing pipeline for a session.

    The acquisition system resolves the cindra processing configuration. When none of '--binarize', '--register',
    '--process', or '--combine' is requested, all four stages run in sequence (local mode). Supplying '--job-id'
    instead runs only the matching job.
    """
    run_two_photon_processing_pipeline(
        session_path=shared.require_session_path(),
        job_id=shared.job_id,
        binarize=binarize,
        register=register,
        process=process,
        combine=combine,
        target_plane=target_plane,
        workers=shared.workers,
        display_progress=shared.display_progress,
    )
