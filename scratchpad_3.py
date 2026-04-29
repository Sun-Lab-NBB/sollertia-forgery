from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import (
    run_drift_analysis,
    plot_drift_vs_bleaching,
    plot_recurrence_heatmap,
    plot_classification_raster,
    plot_peak_shift_distribution,
    plot_population_vector_correlation_vs_lag,
)
from sollertia_forgery.shared_assets import DatasetData


if __name__ == "__main__":
    dataset_path = Path("/home/data/Data/MaalstroomicFlow/void")
    animal: str | None = None

    dataset = DatasetData.load(dataset_path=dataset_path)
    reports = run_drift_analysis(dataset=dataset, animal=animal)
    target_animals = (dataset.get_animal(animal=animal),) if animal is not None else dataset.animals

    dataset_root = dataset.dataset_data_path.parent
    for dataset_animal, report in zip(target_animals, reports, strict=True):
        report.print_summary()
        file_prefix = f"drift_animal_{dataset_animal.animal}"
        plot_classification_raster(report).savefig(
            dataset_root / f"{file_prefix}_classification_raster.png", bbox_inches="tight"
        )
        plot_population_vector_correlation_vs_lag(report).savefig(
            dataset_root / f"{file_prefix}_pv_correlation_vs_lag.png", bbox_inches="tight"
        )
        plot_peak_shift_distribution(report).savefig(
            dataset_root / f"{file_prefix}_peak_shift_distribution.png", bbox_inches="tight"
        )
        plot_drift_vs_bleaching(report).savefig(
            dataset_root / f"{file_prefix}_drift_vs_bleaching.png", bbox_inches="tight"
        )
        plot_recurrence_heatmap(report).savefig(
            dataset_root / f"{file_prefix}_recurrence_heatmap.png", bbox_inches="tight"
        )

    plt.close("all")
