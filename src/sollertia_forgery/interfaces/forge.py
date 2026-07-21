"""Provides the generic ``slf forge`` command that assembles a dataset from a project's processed sessions.

Notes:
    The assembly stage invokes the system-agnostic forging pipeline directly. That pipeline infers the acquisition
    system from the resolved dataset and resolves the system-specific assembly worker internally through the
    forging-assembly registry. The command carries no system selector.
"""

from pathlib import Path

import click

from ..forging import run_forging_pipeline

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.command("forge", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-dn", "--dataset-name", type=str, required=True, help="The unique name for the dataset to create or use."
)
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
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
@click.option(
    "-f",
    "--force-recreate",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to delete any existing dataset hierarchy before creating it fresh.",
)
def forge_command(
    dataset_name: str,
    project_path: Path,
    session: tuple[str, ...],
    job_id: str | None,
    workers: int,
    *,
    progress: bool,
    force_recreate: bool,
) -> None:
    """Forges a dataset by assembling per-session data from a project's processed sessions.

    The forging pipeline is system-agnostic: it resolves the dataset's system-specific assembly worker internally
    from the central registry and infers the acquisition system from the resolved dataset, so the command carries no
    system selector.
    """
    run_forging_pipeline(
        name=dataset_name,
        session_names=session,
        project_root=project_path,
        job_id=job_id,
        workers=workers,
        display_progress=progress,
        force_recreate=force_recreate,
    )
