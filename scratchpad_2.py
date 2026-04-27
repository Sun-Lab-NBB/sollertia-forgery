from pathlib import Path

import matplotlib.pyplot as plt

from sollertia_forgery.analysis import (
    CellAnalysisReport,
    evaluate_and_save_cell_analysis,
    plot_dataset_cell_count,
    plot_dataset_place_cell_fraction,
    plot_dataset_reward_cell_fraction,
    plot_dataset_sce_rate,
)
from sollertia_forgery.shared_assets import DatasetData, TrialGeometry

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

if len(animal_sessions) < 2:
    raise SystemExit(f"Need at least two sessions for animal {animal!r}; found {len(animal_sessions)}.")

target_sessions = {"first": animal_sessions[0], "last": animal_sessions[-1]}
dataset_root = dataset.dataset_data_path.parent

# Resolves the trial type to evaluate from the first session's geometry. Picks the first declared trial type.
geometry = TrialGeometry.from_yaml(file_path=target_sessions["first"].geometry_path)
trial_type = next(iter(geometry.entries))
print(f"Using trial type: {trial_type!r} (available: {list(geometry.entries)})")
print()

# Per-session evaluation, persistence, and figure rendering.
for label, session in target_sessions.items():
    print(f"--- {label} session: {session.session} ---")
    report: CellAnalysisReport = evaluate_and_save_cell_analysis(session=session, trial_type=trial_type)
    print(report.summarize())
    print()

    file_prefix = f"cell_analysis_animal_{animal}_{label}_{session.session}"

    # Plots that work entirely off the persisted feathers + YAML.
    report.plot_place_cell_heatmap().savefig(
        dataset_root / f"{file_prefix}_place_cell_heatmap.png", bbox_inches="tight"
    )
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

# Cross-session aggregates over every animal whose CellAnalysisReport has been persisted.
plot_dataset_place_cell_fraction(dataset=dataset).savefig(
    dataset_root / "cell_analysis_dataset_place_cell_fraction.png", bbox_inches="tight"
)
plot_dataset_reward_cell_fraction(dataset=dataset).savefig(
    dataset_root / "cell_analysis_dataset_reward_cell_fraction.png", bbox_inches="tight"
)
plot_dataset_sce_rate(dataset=dataset).savefig(dataset_root / "cell_analysis_dataset_sce_rate.png", bbox_inches="tight")
plot_dataset_cell_count(dataset=dataset).savefig(
    dataset_root / "cell_analysis_dataset_cell_count.png", bbox_inches="tight"
)
plt.close("all")

print()
print(f"Cell analysis artifacts saved under each session directory inside: {dataset_root}")
print(f"Diagnostic figures saved to: {dataset_root}")
