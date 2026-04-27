from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import evaluate_and_save_bleaching
from sollertia_forgery.shared_assets import DatasetData

dataset_path = Path("/home/data/Data/MaalstroomicFlow/void")
animal = "11"

dataset = DatasetData.load(dataset_path=dataset_path)
print(f"Loaded dataset {dataset.name!r} from {dataset_path}")
print(f"Animals in dataset: {[entry.animal for entry in dataset.animals]}")

animal_sessions = sorted(dataset.get_sessions_for_animal(animal=animal), key=lambda s: s.session)
print(f"Sessions for animal {animal!r} ({len(animal_sessions)}):")
for session in animal_sessions:
    print(f"  {session.session}")
print()

report = evaluate_and_save_bleaching(dataset=dataset, animal=animal)
report.print_summary()

dataset_root = dataset.dataset_data_path.parent
file_prefix = f"bleaching_animal_{animal}"
report.plot_baseline_trend().savefig(dataset_root / f"{file_prefix}_baseline_trend.png", bbox_inches="tight")
report.plot_within_session().savefig(dataset_root / f"{file_prefix}_within_session.png", bbox_inches="tight")
report.plot_snr_distributions().savefig(dataset_root / f"{file_prefix}_snr_distributions.png", bbox_inches="tight")
plt.close("all")

print()
print(f"Bleaching artifacts saved to: {dataset.get_animal(animal=animal).animal_path}")
print(f"Diagnostic figures saved to: {dataset_root}")
