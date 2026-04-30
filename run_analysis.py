"""Top-level analysis driver for the MaalstroomicFlow project.

Runs any subset of the per-dataset analyses (bleaching, tuning, SCE, drift) and persists each report's
artifacts plus the canonical plots under the dataset root. Selection is by flag; passing ``--all`` or
no analysis flag both run the full set in dependency order.

Run order is fixed at bleaching -> tuning -> sce -> drift because drift consumes both the persisted
bleaching report (per-session mask + per-cell baseline slope) and the per-session tuning reports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The ``__main__`` guard around ``main`` is required for ``ProcessPoolExecutor`` under the
# ``forkserver`` / ``spawn`` start methods that Python 3.14 uses by default on Linux: subprocess
# workers re-import this module, so any code that spawns workers must sit behind the guard or it
# will recurse on import.
from sollertia_forgery.shared_assets import DatasetData
from sollertia_forgery.analysis import (
    plot_baseline_trend,
    plot_classification_raster,
    plot_dataset_baseline_trend,
    plot_drift_vs_bleaching,
    plot_peak_shift_distribution,
    plot_population_vector_correlation_vs_lag,
    plot_recurrence_heatmap,
    plot_snr_distributions,
    plot_within_session,
    plot_within_session_average,
    run_bleaching_analysis,
    run_drift_analysis,
    run_sce_analysis,
    run_tuning_analysis,
)


DEFAULT_DATASET_PATH = Path("/home/data/Data/MaalstroomicFlow/void")
ANALYSIS_NAMES: tuple[str, ...] = ("bleaching", "tuning", "sce", "drift")


def _run_bleaching(dataset: DatasetData, animal: str | None) -> None:
    """Runs the bleaching pipeline and writes per-animal + dataset-level plots next to each report."""
    reports = run_bleaching_analysis(dataset=dataset, animal=animal)
    target_animals = (dataset.get_animal(animal=animal),) if animal is not None else dataset.animals
    dataset_root = dataset.dataset_data_path.parent
    for dataset_animal, report in zip(target_animals, reports, strict=True):
        report.print_summary()
        prefix = f"bleaching_animal_{dataset_animal.animal}"
        plot_baseline_trend(report).savefig(dataset_root / f"{prefix}_baseline_trend.png", bbox_inches="tight")
        plot_within_session(report).savefig(dataset_root / f"{prefix}_within_session.png", bbox_inches="tight")
        plot_within_session_average(report).savefig(
            dataset_root / f"{prefix}_within_session_average.png", bbox_inches="tight"
        )
        plot_snr_distributions(report).savefig(
            dataset_root / f"{prefix}_snr_distributions.png", bbox_inches="tight"
        )
    plot_dataset_baseline_trend(dataset=dataset).savefig(
        dataset_root / "bleaching_dataset_baseline_trend.png", bbox_inches="tight"
    )


def _run_tuning(dataset: DatasetData, animal: str | None) -> None:
    """Runs the tuning pipeline; per-session reports persist via the orchestrator's own save path."""
    run_tuning_analysis(dataset=dataset, animal=animal)


def _run_sce(dataset: DatasetData, animal: str | None) -> None:
    """Runs the SCE pipeline; per-session reports persist via the orchestrator's own save path."""
    run_sce_analysis(dataset=dataset, animal=animal)


def _run_drift(dataset: DatasetData, animal: str | None) -> None:
    """Runs the drift pipeline and writes the canonical plots next to each persisted report."""
    reports = run_drift_analysis(dataset=dataset, animal=animal)
    target_animals = (dataset.get_animal(animal=animal),) if animal is not None else dataset.animals
    dataset_root = dataset.dataset_data_path.parent
    for dataset_animal, report in zip(target_animals, reports, strict=True):
        report.print_summary()
        animal_id = dataset_animal.animal
        prefix = f"drift_animal_{animal_id}"
        plot_classification_raster(report, animal_id=animal_id).savefig(
            dataset_root / f"{prefix}_classification_raster.png", bbox_inches="tight"
        )
        plot_population_vector_correlation_vs_lag(report, animal_id=animal_id).savefig(
            dataset_root / f"{prefix}_pv_correlation_vs_lag.png", bbox_inches="tight"
        )
        plot_peak_shift_distribution(report, animal_id=animal_id).savefig(
            dataset_root / f"{prefix}_peak_shift_distribution.png", bbox_inches="tight"
        )
        plot_drift_vs_bleaching(report, animal_id=animal_id).savefig(
            dataset_root / f"{prefix}_drift_vs_bleaching.png", bbox_inches="tight"
        )
        plot_recurrence_heatmap(report, animal_id=animal_id).savefig(
            dataset_root / f"{prefix}_recurrence_heatmap.png", bbox_inches="tight"
        )


_RUNNERS = {
    "bleaching": _run_bleaching,
    "tuning": _run_tuning,
    "sce": _run_sce,
    "drift": _run_drift,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH,
        help="Dataset root directory (default: %(default)s).",
    )
    parser.add_argument(
        "--animal", default=None,
        help="Restrict every selected analysis to this animal id. Default: every animal in the dataset.",
    )
    for name in ANALYSIS_NAMES:
        parser.add_argument(f"--{name}", action="store_true", help=f"Run the {name} analysis.")
    parser.add_argument(
        "--all", action="store_true",
        help="Run every analysis. Equivalent to passing --bleaching --tuning --sce --drift.",
    )
    args = parser.parse_args()

    if args.all or not any(getattr(args, name) for name in ANALYSIS_NAMES):
        selected = ANALYSIS_NAMES
    else:
        selected = tuple(name for name in ANALYSIS_NAMES if getattr(args, name))

    dataset = DatasetData.load(dataset_path=args.dataset)
    print(f"Loaded dataset {dataset.name!r} from {args.dataset}")
    print(f"Running analyses in order: {', '.join(selected)}")
    if args.animal is not None:
        print(f"Restricted to animal: {args.animal}")

    for name in selected:
        print(f"\n=== {name} ===")
        _RUNNERS[name](dataset=dataset, animal=args.animal)
    plt.close("all")


if __name__ == "__main__":
    main()
