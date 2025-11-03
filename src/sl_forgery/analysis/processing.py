from src.sl_forgery.analysis.io import behavior_to_numpy, extract_data, behavior_from_feather, extract_data_new
import io
from pathlib import Path
from scipy import stats
from matplotlib import pyplot as plt

import numpy as np
import polars as pl

'''rewriting this to separate the data processing step from the plotting/analysis, and to match the new data structure
 this will save the processed data as memory-efficient parquet forms that can be quickly accessed for data analysis 
 and plotting, using polars
  run as a script, output is a parquet file that we can access for plotting/further analysis
  saves: 
  - trial-average signals for each cell w trial types and system states  (need to think about Ivan)
  - session-averaged signals for each cell
        '''

#right now do this manually, session filepath
session_root = Path(path)

#get all of the data (beh and meso) from the .feather file in a polars dataframe
behavior_df = pl.read_ipc(session_root / '.feather')


def process_trial_based_activity(df, bin_size_cm=5, include_averages=True):
    """
 Transform frame-based data to trial-based with distance-binned cell activity.

 Parameters:
 -----------
 df : pl.DataFrame
     Input dataframe with columns:
     - frames: frame number
     - cell_activity: array of flourescent value per cell per frame
     - active_trial: "run"; other options are "idle" and "rest"
     - trial_number: trial identifier
     - trial_type: right now ABC or ABCD: need to make more flexible. select(pl.col('trial_type').unique())
     - cumulative_distance: cumulative distance in cm
     - cue_identity: cue type
 bin_size_cm : int
     Size of distance bins in cm (default 5)
 include_averages : bool
     Whether to compute trial-level and session-level averages (default True)

 Returns:
 --------
 Dictionary with:
     - 'binned_data': Main dataframe with binned arrays per trial
     - 'trial_averages': Average signal per cell per trial
     - 'session_averages': Average signal per cell across entire session
 """

    # Step 1: Filter for active trials only <-- this could be an arg in the function, for Ivan
    active_df = df.filter(pl.col('active_trial'))

    # Step 2: Normalize distance within each trial (0 to trial_length)
    # This is crucial for consistent binning across trials
    normalized_df = active_df.with_columns([
        # Group by trial and normalize distance to start at 0
        (pl.col('cumulative_distance') -
         pl.col('cumulative_distance').min().over('trial_number')).alias('trial_distance')
    ])

    # Step 3: Assign bins based on normalized distance
    # For 240cm trials: 48 bins (0-5, 5-10, ..., 235-240)
    # For 280cm trials: 56 bins (0-5, 5-10, ..., 275-280)
    normalized_df = normalized_df.with_columns([
        (pl.col('trial_distance') // bin_size_cm).cast(pl.Int32).alias('distance_bin')
    ])

    # Step 4: Expand cell_activity array into separate columns for each cell
    # struct is like a dict; unnest expands struct into the columns
    expanded_df = normalized_df.with_columns([
        pl.col('cell_activity').list.to_struct(fields=lambda idx: f'cell_{idx}')
    ]).unnest('cell_activity')

