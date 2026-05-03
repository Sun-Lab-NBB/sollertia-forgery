"""Per-animal reward-shift figures for the void cohort.

Loads each requested animal's saved drift report and writes the per-animal Ziv-2013-style cross-session
panels under ``/home/data/Data/MF_Drift/``:

* drift recurrence heatmap (place / reward) — ``plot_recurrence_heatmap``
* sessions-active distribution + per-session activity-rate inset — ``plot_sessions_active_distribution``
* active-cell vs place-field recurrence probability vs lag — ``plot_recurrence_probability_vs_lag``
* reference-day-sorted rate-map heatmaps across every session — ``plot_reference_day_sorted_rate_maps``
* per-cell drift profile category bar chart — ``plot_drift_profile_categories``

Currently configured for the void dataset and animals 11 / 15 / 16. Adjust ``DATASET_PATH`` or
``ANIMALS`` at the top of the file when extending to other animals or projects.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sollertia_forgery.shared_assets import DatasetData
from sollertia_forgery.analysis import (
    DriftReport,
    DriftDetectionConfiguration,
    plot_drift_profile_categories,
    plot_recurrence_heatmap,
    plot_recurrence_probability_vs_lag,
    plot_reference_day_sorted_rate_maps,
    plot_sessions_active_distribution,
)
from sollertia_forgery.analysis.drift import compute_drift_report


DATASET_PATH = Path("/home/data/Data/void")
OUTPUT_DIR = Path("/home/data/Data/MF_Drift")
ANIMALS: tuple[str, ...] = ("11", "15", "16")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = DatasetData.load(dataset_path=DATASET_PATH)
    config = DriftDetectionConfiguration(peak_shift_shuffle_count=200)

    for animal_id in ANIMALS:
        print(f"\n=== animal {animal_id} ===")
        dataset_animal = dataset.get_animal(animal=animal_id)
        sessions = tuple(sorted(
            dataset.get_sessions_for_animal(animal=animal_id), key=lambda s: s.session,
        ))
        # Prefer the saved per-animal drift report. Fall back to an on-the-fly compute when no report
        # has been persisted yet (animal 16 had no shift and may not have been included in a prior
        # ``run_analysis.py --drift`` pass).
        if dataset_animal.drift_summary_path.exists():
            report = DriftReport.load(animal=dataset_animal)
        elif len(sessions) >= 2:
            print(f"  no saved drift report; computing on the fly from {len(sessions)} sessions...")
            report = compute_drift_report(
                animal=dataset_animal, sessions=sessions,
                use_bleaching=True, configuration=config,
            )
        else:
            print(f"  fewer than 2 sessions ({len(sessions)}); skipping")
            continue
        prefix = f"reward_shift_animal_{animal_id}"

        plot_recurrence_heatmap(report, animal_id=animal_id).savefig(
            OUTPUT_DIR / f"{prefix}_recurrence_heatmap.png", bbox_inches="tight",
        )
        plot_sessions_active_distribution(report, animal_id=animal_id).savefig(
            OUTPUT_DIR / f"{prefix}_sessions_active_distribution.png", bbox_inches="tight",
        )
        plot_recurrence_probability_vs_lag(report, animal_id=animal_id).savefig(
            OUTPUT_DIR / f"{prefix}_recurrence_vs_lag.png", bbox_inches="tight",
        )
        plot_reference_day_sorted_rate_maps(
            report=report, sessions=sessions, display_sessions=(1, 5, 10, 15, 20),
            animal_id=animal_id,
        ).savefig(
            OUTPUT_DIR / f"{prefix}_reference_day_rate_maps.png", bbox_inches="tight",
        )
        plot_drift_profile_categories(
            report=report, sessions=sessions, animal_id=animal_id,
        ).savefig(
            OUTPUT_DIR / f"{prefix}_drift_profile_categories.png", bbox_inches="tight",
        )
        print(f"  wrote 5 figures for animal {animal_id} under {OUTPUT_DIR}")
    plt.close("all")


if __name__ == "__main__":
    main()
