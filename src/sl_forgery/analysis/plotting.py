from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.analysis.io import behavior_to_numpy, extract_data
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt


def plotting(mouse, kind):

    session_root = Path("/Users/cs963/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/")

    date = 1
    fluorescence, neuropil, spikes, iscell = extract_data(mouse, date, "single_day")

    frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
        source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
    )

    track_length = 240      #soft-code later, in cm
    cue_length = 30     #cm
    bin_size = 5        #cm
    n_bins = track_length/bin_size   #here, 48 bins of 5 cm each

#####Use trial array to bin the track

    track_bins = np.arange(0, traveled_distance.max(), track_length)        #divide the array into 240 cm trials
    trial_bins = np.arange(0, traveled_distance.max(), bin_size)         #divide the array into 5 cm bins, 48 per trial
    cue_bins = np.arange(0, cue_length, track_length)


    # Assign each distance to a bin (0-indexed using -1)
    track_bin_indices = trial           #indexing 1 lap on track (trial)
    trial_bin_indices = np.digitize(traveled_distance, trial_bins) - 1      #indexing trials in 5 cm bins
    cue_bin_indices = np.digitize(traveled_distance, cue_bins) - 1          #indexing trials into 30 cm bins i.e. cues

    track_masks = np.array(0)
    trial_bin_masks = np.array(0)


    #find locations of rewards; rewards coded as consecutive sequence of 3 timestamps
    diff = np.diff(reward)
    diff_indices = np.where(diff > 1)[0]  # +1 because diff is one element shorter

    # Add index 0 (first element is always a start)
    start_rew_indices = np.concatenate([[0], diff_indices + 1])

    print(start_rew_indices)


    reward_mask = np.where(reward == 1)[0]
    print(reward_mask, len(traveled_distance[reward_mask]))

    #print("indiices", track_bin_indices, trial_bin_indices)
    #print("bins", track_bins, trial_bins, cue_bins)
    #print(np.where(reward==1)[0], reward_mask)
    #print(len(traveled_distance[reward_mask]), len(reward_mask))
    print(reward[0:50])



    fig, ax = plt.subplots()

    if kind == "place":

        #ax.plot(traveled_distance, fluorescence[0])
        #ax.plot(traveled_distance, spikes[0])
        #ax.plot(traveled_distance, bin_indices)
        #ax.vline(traveled_distance[reward_mask], reward[reward_mask], 'o')


        plt.plot(reward_mask, reward[reward_mask])
        plt.show()

plotting(mouse="6", kind="place")

print("#################################")

def find_sequence_starts(arr, min_gap=2):
    """
    Find indices where consecutive sequences start

    Args:
        arr: sorted array with groups of consecutive integers
        min_gap: minimum gap to consider a new sequence (default=2)
    """
    if len(arr) == 0:
        return np.array([])

    # Find where differences are >= min_gap
    diff = np.diff(arr)
    break_points = np.where(diff >= min_gap)[0] + 1

    # Include the first index
    start_indices = np.concatenate([[0], break_points])

    return start_indices


# Usage
arr = np.array([0, 1, 9, 10, 11, 6638, 6639, 6640])
start_indices = find_sequence_starts(arr)

print(f"Start indices: {start_indices}")  # [0, 2, 5]
print(f"First values: {arr[start_indices]}")  # [0, 9, 6638]


