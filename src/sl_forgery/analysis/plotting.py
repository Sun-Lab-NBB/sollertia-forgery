from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.utils.dataclass import Data
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import polars as pl


    # TODO - fix date variable and folder search
    #   add arguments for cue length, bin size, day type for meso (single, multi)
    #   what are the other "kind" options

def plotting(mouse, session, target_group,  data : Data):

    # %%%%%%%%%%%%%%%%%%
    # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
    #  plotting;  integrate with plotly when jacob is done;  plot cue regions under the graph; basically thick little
    #  vlines of different colors

    session_avg_df, sess_sem, result, trial_avg_df = data.process_data(mouse, session, target_group)

    cells = range(5)

    for cell in cells:
        xaxis = np.arange(2.5, 240, 5)  # # 5 cm bins, plot the avg signal in center of bin
        fig, ax = plt.subplots()

        cell_val = session_avg_df['cell_{}_signal_binned'.format(cell)][-1]  # selects the last row of the col,
        # which has the avg session data

        mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
        # pulling ndarrays from polars df
        sem = sess_sem[0]
        ax.plot(xaxis, mean)
        plt.fill_between(xaxis, mean - sem, mean + sem,
                         color='blue', alpha=0.2, label='Mean +/- SEM')

        plt.title("cell {} session avg day5".format(cell))
        plt.xlabel("distance in cm")
        plt.ylabel("Fluorescent signal")

        # plot trial avgs
        fig, ax = plt.subplots()

        for i in range(result.shape[0]):
            ax.plot(xaxis, trial_avg_df[i, cell], label=f"{i}")
        # if kind == "place":
        #     for i in range(result.shape[0]):
        #         ax.plot(normalized_arrays[i], result["cell_1_signal"][i])

        # ax.vlines(trial_start["distance"][:10], ymin=-150, ymax=0, color='r', lw=2)

        # TODO make this cycle through a random subset of cells when calling the function and pull the column names for
        # cell ID

        # Hide the x-tick labels
        plt.title("cell {} trial avgs day5".format(cell))
        plt.xlabel("distance in cm")
        plt.ylabel("Fluorescent signal")
        plt.show()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    data = Data(project_root / "data")
    plotting(6, "2025-06-23-13-32-06-980761", "single_day", data)
