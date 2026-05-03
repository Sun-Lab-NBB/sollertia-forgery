"""Provides CLIs for executing the per-dataset analysis pipelines (bleaching, tuning, SCE, drift).

Each subcommand calls the matching ``run_*_analysis`` orchestrator, which is responsible for persisting
its report's artifacts. Run order is fixed at bleaching -> tuning -> sce -> drift because drift consumes
both the persisted bleaching report (per-session mask + per-cell baseline slope) and the per-session
tuning reports; ``analyze all`` enforces that order automatically.
"""

from pathlib import Path

import click

# The ``__main__`` guard around ``main`` is required for ``ProcessPoolExecutor`` under the
# ``forkserver`` / ``spawn`` start methods that Python 3.14 uses by default on Linux: subprocess
# workers re-import this module, so any code that spawns workers must sit behind the guard or it
# will recurse on import.
from ..shared_assets import DatasetData
from ..analysis import (
    run_bleaching_analysis,
    run_drift_analysis,
    run_sce_analysis,
    run_tuning_analysis,
)


CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""

_ANALYSIS_NAMES: tuple[str, ...] = ("bleaching", "tuning", "sce", "drift")
"""Canonical run order; ``analyze all`` dispatches in this order."""

_RUNNERS = {
    "bleaching": run_bleaching_analysis,
    "tuning": run_tuning_analysis,
    "sce": run_sce_analysis,
    "drift": run_drift_analysis,
}
"""Maps each analysis name to its ``run_*_analysis`` orchestrator. Each orchestrator owns its own
console banner, so this CLI does not echo a prefatory message before dispatching."""


def _dispatch(ctx: click.Context, names: tuple[str, ...]) -> None:
    """Loads the dataset and runs each named analysis pipeline in the supplied order."""
    dataset_path: Path = ctx.obj["dataset_path"]
    animal: str | None = ctx.obj["animal"]
    dataset = DatasetData.load(dataset_path=dataset_path)
    for name in names:
        _RUNNERS[name](dataset=dataset, animal=animal)


@click.group("analyze", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-dp",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the dataset's root data directory.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    default=None,
    show_default=True,
    help=(
        "Restrict every selected analysis to this animal id. When omitted, the analyses run for every "
        "animal in the dataset."
    ),
)
def analyze_cli(ctx: click.Context, dataset_path: Path, animal: str | None) -> None:
    """Runs the per-dataset analysis pipelines (bleaching, tuning, SCE, drift) on the local machine.

    Each subcommand executes its analysis orchestrator, which persists the report on disk. ``analyze
    all`` dispatches every pipeline in the canonical run order so drift can consume the upstream
    reports.
    """
    ctx.ensure_object(dict)
    ctx.obj["dataset_path"] = dataset_path
    ctx.obj["animal"] = animal


@analyze_cli.command("bleaching")
@click.pass_context
def analyze_bleaching(ctx: click.Context) -> None:
    """Runs the bleaching analysis pipeline for the configured dataset/animal selection."""
    _dispatch(ctx=ctx, names=("bleaching",))


@analyze_cli.command("tuning")
@click.pass_context
def analyze_tuning(ctx: click.Context) -> None:
    """Runs the tuning analysis pipeline for the configured dataset/animal selection."""
    _dispatch(ctx=ctx, names=("tuning",))


@analyze_cli.command("sce")
@click.pass_context
def analyze_sce(ctx: click.Context) -> None:
    """Runs the SCE analysis pipeline for the configured dataset/animal selection."""
    _dispatch(ctx=ctx, names=("sce",))


@analyze_cli.command("drift")
@click.pass_context
def analyze_drift(ctx: click.Context) -> None:
    """Runs the drift analysis pipeline for the configured dataset/animal selection."""
    _dispatch(ctx=ctx, names=("drift",))


@analyze_cli.command("all")
@click.pass_context
def analyze_all(ctx: click.Context) -> None:
    """Runs every analysis pipeline in the canonical run order (bleaching -> tuning -> sce -> drift)."""
    _dispatch(ctx=ctx, names=_ANALYSIS_NAMES)
