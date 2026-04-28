from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import (
    plot_baseline_trend,
    plot_within_session,
    plot_snr_distributions,
    run_bleaching_analysis,
    plot_within_session_average,
    plot_dataset_baseline_trend,
)
from sollertia_forgery.shared_assets import DatasetData


# The ``__main__`` guard is required for ``ProcessPoolExecutor`` under the ``forkserver`` / ``spawn`` start
# methods that Python 3.14 uses by default on Linux: subprocess workers re-import this module, so any code that
# spawns workers must sit behind the guard or it will recurse on import.
if __name__ == "__main__":
    dataset_path = Path("/home/data/Data/MaalstroomicFlow/void")
    animal: str | None = None

    dataset = DatasetData.load(dataset_path=dataset_path)
    reports = run_bleaching_analysis(dataset=dataset, animal=animal)
    target_animals = (dataset.get_animal(animal=animal),) if animal is not None else dataset.animals

    dataset_root = dataset.dataset_data_path.parent
    for dataset_animal, report in zip(target_animals, reports, strict=True):
        report.print_summary()
        file_prefix = f"bleaching_animal_{dataset_animal.animal}"
        plot_baseline_trend(report).savefig(dataset_root / f"{file_prefix}_baseline_trend.png", bbox_inches="tight")
        plot_within_session(report).savefig(dataset_root / f"{file_prefix}_within_session.png", bbox_inches="tight")
        plot_within_session_average(report).savefig(
            dataset_root / f"{file_prefix}_within_session_average.png", bbox_inches="tight"
        )
        plot_snr_distributions(report).savefig(
            dataset_root / f"{file_prefix}_snr_distributions.png", bbox_inches="tight"
        )

    plot_dataset_baseline_trend(dataset=dataset).savefig(
        dataset_root / "bleaching_dataset_baseline_trend.png", bbox_inches="tight"
    )
    plt.close("all")
