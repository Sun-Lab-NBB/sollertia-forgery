"""Provides the generic ``slf forge`` command that assembles a dataset from a project's processed sessions."""

from pathlib import Path

import click

from ..forging import run_forging_pipeline, define_forging_dataset

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.command("forge", context_settings=_CONTEXT_SETTINGS)
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
    help="The session name the dataset must contain. A session the dataset does not hold is appended to it. Can be "
    "specified multiple times.",
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
    "-f",
    "--force-recreate",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to delete the whole existing dataset hierarchy and rebuild it from the provided sessions.",
)
@click.option(
    "-ra",
    "--recreate-animal",
    type=str,
    multiple=True,
    help="The identifier of an animal already in the dataset to rebuild from the sessions provided for it, leaving "
    "every other animal untouched. Can be specified multiple times.",
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to suppress the progress bars during the multi-day and assembly stages. These are "
        "displayed by default."
    ),
)
def forge_command(
    dataset_name: str,
    project_path: Path,
    session: tuple[str, ...],
    job_id: str | None,
    workers: int,
    recreate_animal: tuple[str, ...],
    *,
    force_recreate: bool,
    no_progress: bool,
) -> None:
    """Forges a dataset by assembling per-session data from a project's processed sessions.

    Provided sessions the dataset does not hold are appended to it, so a dataset grows by naming the sessions to add.
    An animal already in the dataset is frozen, because widening its session set invalidates the outputs already
    forged for it and requires rebuilding the animal as a whole. Name that animal with --recreate-animal to rebuild
    it from the provided sessions while every other animal keeps its data.

    The forging pipeline is system-agnostic: it resolves the dataset's system-specific assembly worker internally
    from the central registry and infers the acquisition system from the resolved dataset, so the command carries no
    system selector.
    """
    # Defining the hierarchy precedes the tracked jobs, so a command naming sessions or a rebuild builds it first.
    if session or force_recreate or recreate_animal:
        define_forging_dataset(
            name=dataset_name,
            session_names=session,
            project_root=project_path,
            display_progress=not no_progress,
            force_recreate=force_recreate,
            recreate_animals=recreate_animal,
        )

    run_forging_pipeline(
        name=dataset_name,
        project_root=project_path,
        job_id=job_id,
        workers=workers,
        display_progress=not no_progress,
    )
