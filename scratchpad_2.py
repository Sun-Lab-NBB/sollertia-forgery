from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import (
    CellAnalysisReport,
    evaluate_and_save_cell_analysis,
)
from sollertia_forgery.shared_assets import DatasetData, TrialGeometry

dataset_path = Path("/home/data/Data/MaalstroomicFlow/void")
animal = "11"

dataset = DatasetData.load(dataset_path=dataset_path)
print(f"Loaded dataset {dataset.name!r} from {dataset_path}")

animal_sessions = sorted(dataset.get_sessions_for_animal(animal=animal), key=lambda s: s.session)
if not animal_sessions:
    raise SystemExit(f"No sessions found for animal {animal!r}.")
session = animal_sessions[-1]
print(f"Target session for animal {animal!r}: {session.session}")

# Resolves the trial type from the target session's geometry. Picks the first declared trial type.
geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
trial_type = next(iter(geometry.entries))
print(f"Using trial type: {trial_type!r} (available: {list(geometry.entries)})")
print()

dataset_root = dataset.dataset_data_path.parent
file_prefix = f"cell_analysis_animal_{animal}_{session.session}"

print(f"--- Running cell-analysis evaluation on {session.session} ---")
report: CellAnalysisReport = evaluate_and_save_cell_analysis(session=session, trial_type=trial_type)
print(report.summarize())
print()

# Plots that work entirely off the persisted feathers + YAML.
report.plot_place_cell_heatmap().savefig(dataset_root / f"{file_prefix}_place_cell_heatmap.png", bbox_inches="tight")
report.plot_reward_com_histogram().savefig(
    dataset_root / f"{file_prefix}_reward_com_histogram.png", bbox_inches="tight"
)
report.plot_rate_map_heatmap().savefig(dataset_root / f"{file_prefix}_rate_map_heatmap.png", bbox_inches="tight")
report.plot_population_activity_by_position().savefig(
    dataset_root / f"{file_prefix}_population_activity_by_position.png", bbox_inches="tight"
)
report.plot_sce_assemblies().savefig(dataset_root / f"{file_prefix}_sce_assemblies.png", bbox_inches="tight")

# Plots that also reload data.feather to recover speed and per-trial fluorescence.
report.plot_speed_and_activity_by_position(session=session).savefig(
    dataset_root / f"{file_prefix}_speed_and_activity_by_position.png", bbox_inches="tight"
)
report.plot_per_trial_activity(session=session, trial_type=trial_type).savefig(
    dataset_root / f"{file_prefix}_per_trial_activity.png", bbox_inches="tight"
)
report.plot_sce_rest_run_rest_sequence(session=session).savefig(
    dataset_root / f"{file_prefix}_sce_rest_run_rest_sequence.png", bbox_inches="tight"
)
plt.close("all")

print()
print(f"Cell-analysis artifacts saved under: {session.session_path}")
print(f"Per-session figures saved under:    {dataset_root}")
