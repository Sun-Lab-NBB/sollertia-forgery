from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import (
    SCEReport,
    TuningReport,
    plot_sce_assemblies,
    plot_per_trial_activity,
    plot_place_cell_heatmap,
    plot_rate_map_heatmap,
    plot_reward_com_histogram,
    evaluate_and_save_sce_report,
    plot_sce_cells_across_periods,
    evaluate_and_save_tuning_report,
    plot_speed_and_activity_by_position,
    plot_population_activity_by_position,
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

print(f"--- Running tuning evaluation on {session.session} ---")
tuning_report: TuningReport = evaluate_and_save_tuning_report(session=session, trial_type=trial_type)
print(tuning_report.summarize())
print()

print(f"--- Running SCE evaluation on {session.session} ---")
sce_report: SCEReport = evaluate_and_save_sce_report(session=session)
print(sce_report.summarize())
print()

# Plots that work entirely off the persisted feathers + YAML.
plot_place_cell_heatmap(tuning_report).savefig(
    dataset_root / f"{file_prefix}_place_cell_heatmap.png", bbox_inches="tight"
)
plot_reward_com_histogram(tuning_report).savefig(
    dataset_root / f"{file_prefix}_reward_com_histogram.png", bbox_inches="tight"
)
plot_rate_map_heatmap(tuning_report).savefig(
    dataset_root / f"{file_prefix}_rate_map_heatmap.png", bbox_inches="tight"
)
plot_population_activity_by_position(tuning_report).savefig(
    dataset_root / f"{file_prefix}_population_activity_by_position.png", bbox_inches="tight"
)
plot_sce_assemblies(sce_report).savefig(dataset_root / f"{file_prefix}_sce_assemblies.png", bbox_inches="tight")

# Plots that also reload data.feather to recover speed and per-trial fluorescence.
plot_speed_and_activity_by_position(tuning_report, session=session).savefig(
    dataset_root / f"{file_prefix}_speed_and_activity_by_position.png", bbox_inches="tight"
)
plot_per_trial_activity(tuning_report, session=session, trial_type=trial_type).savefig(
    dataset_root / f"{file_prefix}_per_trial_activity.png", bbox_inches="tight"
)
plot_sce_cells_across_periods(sce_report, session=session).savefig(
    dataset_root / f"{file_prefix}_sce_cells_across_periods.png", bbox_inches="tight"
)
plt.close("all")

print()
print(f"Per-session artifacts saved under: {session.session_path}")
print(f"Per-session figures saved under:    {dataset_root}")
