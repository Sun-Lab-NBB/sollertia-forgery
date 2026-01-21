from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from sl_forgery.analysis.io import extract_data, behavior_to_numpy

files = [
    "2025-06-23-13-32-06-980761",
    "2025-06-24-13-17-47-781337",
    "2025-06-25-16-35-01-581331",
    "2025-06-26-12-57-38-495382",
    "2025-06-27-12-44-58-770644",
]


def get_multiday_data(file):
    filepath = "/Users/cs963/Desktop/TM_06_pilot/6/" + file
    session_root = Path(filepath)

    fluorescence, neuropil, spikes, iscell = extract_data(filepath, "single_day")

    frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
        source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
    )

    fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])

    iscell_df = pl.DataFrame(iscell)

    iscell_df = iscell_df.with_row_index("cell_idx")  # add cell id index to cell df, 0-indexed to match F_df

    behavior_df = pl.DataFrame(
        {
            "frame": frame_index,
            "timestamp": timestamps,
            "distance": traveled_distance,
            "trial": trial,
            "lick": lick,
            "reward": reward,
            "stage": experiment_stage,
            "state": system_state,
        }
    )

    cell_mask = iscell_df[:, 1].to_numpy()

    cell_fluorescence_df = fluorescence_df.select([fluorescence_df.columns[i] for i in np.where(cell_mask)[0]])

    return cell_fluorescence_df, behavior_df


d1F, d1b = get_multiday_data(files[0])
d2F, d2b = get_multiday_data(files[1])
d3F, d3b = get_multiday_data(files[2])
d4F, d4b = get_multiday_data(files[3])
d5F, d5b = get_multiday_data(files[4])

# split sessions in half and plot avgs for both
fsignals = [d1F, d2F, d3F, d4F, d5F]
labels = ["day 1", "day 2", "day 3", "day 4", "day 5"]
total_mean = []
trial_mean = []

transitions_df = d1b.with_row_index().filter(pl.col("state") != pl.col("state").shift(1))
transition_indices = transitions_df["index"].to_list()
print(transition_indices), len(transition_indices)

for i in fsignals:
    numeric_types = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    )
    overall_mean = i.select(pl.col(numeric_types)).to_numpy().mean()
    total_mean.append(overall_mean)
    by_frame_mean = i.with_columns(pl.mean_horizontal(pl.all()).alias("row_avg"))
    trial_mean.append(by_frame_mean.select(pl.col("row_avg")))

# Simple plot showing just the overall mean
plt.scatter(range(5), total_mean)
plt.title("Overall Mean F")
plt.ylabel("Mean Value")
plt.xticks(range(5))
plt.show()


plt.figure(figsize=(10, 6))
for i, x in enumerate(trial_mean):
    plt.plot(x, label=labels[i])
    plt.axvline(transition_indices[i], color="black", linestyle="--", alpha=0.8, linewidth=1)
plt.xlabel("Row Index")
plt.ylabel("Row Average")
plt.title("Row Averages by Index")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
plt.plot(d1F.shape[1], d1F.mean())
# for f in fsignals:
#     cellmean = f.mean()
#     entire_mean = cellmean.mean_horizontal()
#     plt.plot(f, label=str(f))
#
plt.show()
