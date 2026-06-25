"""Provides the generic ``slf forge`` command that assembles a dataset from a project's processed sessions.

Notes:
    The command runs up to two stages. When an activity configuration is supplied, the multi-recording cell-tracking
    stage runs first, dispatched through the local-pipeline registry after inferring the acquisition system from the
    supplied session paths. The assembly stage then invokes the system-agnostic forging pipeline directly; that
    pipeline infers the acquisition system from the resolved dataset and resolves the system-specific assembly worker
    internally through the forging-assembly registry. The command carries no system selector.
"""

from pathlib import Path

import click
from ataraxis_base_utilities import console

from ..forging import run_forging_pipeline
from .dispatch import infer_system_from_session
from ..pipelines import ProcessingPipelines
from ..registries import resolve_local_pipeline

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.command("forge", context_settings=CONTEXT_SETTINGS)
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
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    multiple=True,
    help=(
        "A session root directory feeding the multi-recording cell-tracking stage. Can be specified multiple times. "
        "Used to locate the recordings and to infer the acquisition system."
    ),
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
@click.option(
    "-id",
    "--job-id",
    type=str,
    default=None,
    help="The unique hexadecimal identifier for this job. If provided, runs only the matching job (remote mode).",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    show_default=True,
    help="The number of worker processes to use for parallel assembly. Set to -1 for automatic resolution.",
)
@click.option(
    "-pr",
    "--progress",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to display a progress bar during assembly.",
)
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
def forge_command(
    dataset_name: str | None,
    project_path: Path | None,
    session: tuple[str, ...],
    session_path: tuple[Path, ...],
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
    """Forges a dataset: optionally tracks cells across recordings, then assembles per-session data.

    When an ``--activity-configuration`` is provided, the multi-recording (across-session cell tracking) stage runs
    first. The forging assembly then runs when a ``--dataset-name`` and ``--project-path`` are provided. The
    acquisition system is inferred from the supplied session paths (cell tracking) or the project root (assembly).
    """
    if activity_configuration is not None:
        if not session_path:
            message = (
                "Unable to run the multi-recording cell-tracking stage. Provide at least one '--session-path' (-sp) "
                "so the acquisition system can be inferred."
            )
            console.error(message=message, error=ValueError)
        run_pipeline = resolve_local_pipeline(
            infer_system_from_session(session_path[0]), ProcessingPipelines.CINDRA_MULTI_RECORDING
        )
        run_pipeline(
            configuration_path=activity_configuration,
            job_id=job_id,
            discover=discover,
            extract=extract,
            target_recording=target_recording,
        )

    if dataset_name is not None and project_path is not None:
        # Forging is system-agnostic: the pipeline resolves the dataset's system-specific assembly worker internally
        # from the central registry, so the interface invokes it directly rather than through the local-pipeline
        # registry.
        run_forging_pipeline(
            name=dataset_name,
            session_names=session,
            project_root=project_path,
            job_id=job_id,
            workers=workers,
            display_progress=progress,
            force_recreate=force_recreate,
        )
