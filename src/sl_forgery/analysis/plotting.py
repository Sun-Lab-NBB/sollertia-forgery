from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.utils.dataclass import Data
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import polars as pl


    # TODO - fix date variable and folder search
    #   add arguments for cue length, bin size, day type for meso (single, multi)
    #   what are the other "kind" options

bin_size = 5 #cm

def plot_session(mouse, session, target_group, cell, data : Data):

    # %%%%%%%%%%%%%%%%%%
    # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
    #  plotting;  integrate with plotly when jacob is done;  plot cue regions under the graph; basically thick little
    #  vlines of different colors
    # TODO make this pull the column names for
    # cell ID

    session_avg_df, sess_sem, result, trial_avg_df = data.process_data(mouse, session, target_group)


    cell_val = session_avg_df['cell_{}_signal_binned'.format(cell)][-1]  # selects the last row of the col,
    # which has the avg session data

    mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
    # pulling ndarrays from polars df
    sem = sess_sem[cell] # Before Chelsea indexed sess_sem[0] every time, I think she meant to get the sem for the cell being plotted
    
    
    fig, ax = plt.subplots()
    xaxis = np.arange(bin_size / 2, 240, bin_size)  # plot the avg signal in center of bin
    ax.plot(xaxis, mean)
    plt.fill_between(xaxis, mean - sem, mean + sem,
                        color='blue', alpha=0.2, label='Mean +/- SEM')

    plt.title("cell {} session avg day5".format(cell))
    plt.xlabel("distance in cm")
    plt.ylabel("Fluorescent signal")

def plot_session_trials(mouse, session, target_group, cell, data : Data):

    # %%%%%%%%%%%%%%%%%%
    # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
    #  plotting;  integrate with plotly when jacob is done;  plot cue regions under the graph; basically thick little
    #  vlines of different colors
    # TODO make this pull the column names for
    # cell ID

    session_avg_df, sess_sem, result, trial_avg_df = data.process_data(mouse, session, target_group)

    cell_val = session_avg_df['cell_{}_signal_binned'.format(cell)][-1]  # selects the last row of the col,
    # which has the avg session data

    mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
    # pulling ndarrays from polars df
    sem = sess_sem[0]
    
    xaxis = np.arange(bin_size / 2, 240, bin_size)  # plot the avg signal in center of bin

    # plot trial avgs
    fig, ax = plt.subplots()

    for i in range(result.shape[0]):
        ax.plot(xaxis, trial_avg_df[i, cell], label=f"{i}")

    # Hide the x-tick labels
    plt.title("cell {} trial avgs day5".format(cell))
    plt.xlabel("distance in cm")
    plt.ylabel("Fluorescent signal")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    data = Data(project_root / "data")

    cells = range(5)
    for cell in cells:
        plot_session_trials(6, "2025-06-23-13-32-06-980761", "single_day", cell, data)
        plot_session(6, "2025-06-23-13-32-06-980761", "single_day", cell, data)
    plt.show()

