"""Provides the Mesoscope-VR system-specific CLI command group exposed by the ``slf`` root group. Bundles the
end-to-end, in-process data extraction pipelines (behavior, video, and cell activity) and the dataset forging
pipeline (multi-recording cell tracking followed by per-session assembly).

Notes:
    The pipeline entry points are imported lazily inside each command callback so that ``slf --help`` and the
    system-agnostic commands do not import the heavy acquisition-library bindings.
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


@click.group("mesoscope", context_settings=CONTEXT_SETTINGS)
def mesoscope() -> None:
    """Processes and forges data acquired with the Mesoscope-VR data acquisition system."""


@mesoscope.group("process", context_settings=CONTEXT_SETTINGS)
def process() -> None:
    """Runs the end-to-end, in-process data extraction pipelines on a single Mesoscope-VR session."""


@process.command("behavior")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def behavior_command(session_path: Path, job_id: str | None, workers: int, *, progress: bool) -> None:
    """Extracts the microcontroller log archives and parses runtime and module data into behavior feathers."""
    from ..mesoscope_vr import run_behavior_processing_pipeline  # noqa: PLC0415

    run_behavior_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@process.command("video")
@_SESSION_PATH_OPTION
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
def video_command(session_path: Path, job_id: str | None, workers: int, *, progress: bool) -> None:
    """Extracts the camera frame acquisition timestamps from the raw VideoSystem log archives."""
    from ..cross_system import run_video_processing_pipeline  # noqa: PLC0415

    run_video_processing_pipeline(
        session_path=session_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
    )


@process.command("activity")
@click.option(
    "-i",
    "--configuration",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The path to the cindra single-recording configuration file describing the recording to process.",
)
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="The optional session root directory. Reserved for session-scoped configuration resolution.",
)
@_JOB_ID_OPTION
@click.option("-b", "--binarize", is_flag=True, default=False, help="Run the binarization stage.")
@click.option(
    "-p", "--process", "process_stage", is_flag=True, default=False, help="Run the per-plane processing stage."
)
@click.option("-c", "--combine", is_flag=True, default=False, help="Run the multi-plane combination stage.")
@click.option(
    "-t",
    "--target-plane",
    type=int,
    default=-1,
    show_default=True,
    help="The imaging plane to process when running the processing stage. Set to -1 to process all planes.",
)
def activity_command(
    configuration: Path,
    session_path: Path | None,  # noqa: ARG001 - reserved for session-scoped configuration resolution.
    job_id: str | None,
    target_plane: int,
    *,
    binarize: bool,
    process_stage: bool,
    combine: bool,
) -> None:
    """Runs the single-recording cindra cell-activity processing pipeline for the configured recording.

    When no stage flag is passed, all stages run in sequence (a full local single-recording pipeline). Passing
    individual stage flags runs only the requested stages, mirroring the remote per-plane job decomposition.
    """
    from ..mesoscope_vr import run_activity_processing_pipeline  # noqa: PLC0415

    run_activity_processing_pipeline(
        configuration_path=configuration,
        job_id=job_id,
        binarize=binarize,
        process=process_stage,
        combine=combine,
        target_plane=target_plane,
    )


@mesoscope.command("dataset")
@click.option("-dn", "--dataset-name", type=str, default=None, help="The unique name for the dataset to create or use.")
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="The path to the project's root data directory storing the animal and session data directories.",
)
@click.option(
    "-s",
    "--session",
    type=str,
    multiple=True,
    help="The session name to include in the dataset. Can be specified multiple times.",
)
@click.option(
    "-i",
    "--activity-configuration",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help=(
        "The path to the cindra multi-recording configuration file. When provided, the multi-recording "
        "(across-session cell tracking) stage runs before assembly."
    ),
)
@_JOB_ID_OPTION
@_WORKERS_OPTION
@_PROGRESS_OPTION
@click.option("-d", "--discover", is_flag=True, default=False, help="Run the multi-recording cell-discovery stage.")
@click.option("-e", "--extract", is_flag=True, default=False, help="Run the multi-recording extraction stage.")
@click.option(
    "-t",
    "--target-recording",
    type=str,
    default=None,
    help="The recording to process when running the multi-recording extraction stage.",
)
@click.option(
    "-f",
    "--force-recreate",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to delete any existing dataset hierarchy before creating it fresh.",
)
def dataset_command(
    dataset_name: str | None,
    project_path: Path | None,
    session: tuple[str, ...],
    activity_configuration: Path | None,
    job_id: str | None,
    workers: int,
    target_recording: str | None,
    *,
    progress: bool,
    discover: bool,
    extract: bool,
    force_recreate: bool,
) -> None:
    """Forges an analysis dataset: optionally tracks cells across recordings, then assembles per-session data.

    When an ``--activity-configuration`` is provided, the multi-recording (across-session cell tracking) cindra
    stage runs first. The forging assembly then runs when a ``--dataset-name`` and ``--project-path`` are provided.
    """
    from ..mesoscope_vr import run_forging_pipeline, run_multidataset_processing_pipeline  # noqa: PLC0415

    if activity_configuration is not None:
        run_multidataset_processing_pipeline(
            configuration_path=activity_configuration,
            job_id=job_id,
            discover=discover,
            extract=extract,
            target_recording=target_recording,
        )

    if dataset_name is not None and project_path is not None:
        run_forging_pipeline(
            name=dataset_name,
            session_names=session,
            project_root=project_path,
            job_id=job_id,
            workers=workers,
            display_progress=progress,
            force_recreate=force_recreate,
        )
