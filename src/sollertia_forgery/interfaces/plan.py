"""Provides the ``slf plan`` CLI group that records what a unit's jobs will cost and projects those records into one
table per project.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..orchestration import (
    project_plan_path,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
)

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.group("plan", context_settings=_CONTEXT_SETTINGS)
def plan_cli() -> None:
    """Estimates and records the cores and memory every processing or forging job will occupy.

    Each unit's figures are cached beside its outputs and are kept across later runs, so a submission is never sized
    against figures that changed after it was planned. Use 'project' to project every cache under a project into the
    single table that ships.
    """


@plan_cli.command("session", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help="The absolute path to a session root directory to plan. Can be specified multiple times.",
)
@click.option(
    "-rp",
    "--regenerate-plan",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to recalculate and overwrite the figures the session's plan already holds. When the "
        "command is called with this flag, it re-estimates every job instead of only the ones the plan lacks."
    ),
)
def plan_session_command(session_path: tuple[Path, ...], *, regenerate_plan: bool) -> None:
    """Records what every processing job of each named session will cost."""
    for path in session_path:
        plan = resolve_session_plan(session_path=path, regenerate_plan=regenerate_plan, display_progress=True)
        click.echo(f"{plan.unit_name}: {len(plan.entries)} job(s) planned.")


@plan_cli.command("dataset", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-dp",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help="The absolute path to a dataset root directory to plan. Can be specified multiple times.",
)
@click.option(
    "-rp",
    "--regenerate-plan",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to recalculate and overwrite the figures the dataset's plan already holds. When the "
        "command is called with this flag, it re-estimates every job instead of only the ones the plan lacks."
    ),
)
def plan_dataset_command(dataset_path: tuple[Path, ...], *, regenerate_plan: bool) -> None:
    """Records what every forging job of each named dataset will cost."""
    for path in dataset_path:
        plan = resolve_dataset_plan(dataset_path=path, regenerate_plan=regenerate_plan, display_progress=True)
        click.echo(f"{plan.unit_name}: {len(plan.entries)} job(s) planned.")


@plan_cli.command("project", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the project root directory whose plan caches to project.",
)
def plan_project_command(project_path: Path) -> None:
    """Projects every plan cache under the project into one table at the project root.

    Reads the caches alone and estimates nothing, so a unit that has not been planned contributes no rows.
    """
    generate_project_plan(project_directory=project_path, display_progress=True)
    click.echo(str(project_plan_path(project_directory=project_path)))
