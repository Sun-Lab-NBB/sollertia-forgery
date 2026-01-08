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
session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')


#get all of the data (beh and meso) from the .feather file in a polars dataframe
behavior_df = pl.read_ipc(session_root / '2025-09-03-17-02-46-836208.feather')


def process_trial_based_activity(df, bin_size_cm=5, include_averages=True, save=True):
    """
 Transform frame-based data to trial-based with distance-binned cell activity for each recorded cell.

 Parameters:
 -----------
 df : pl.DataFrame
     Input dataframe with columns:
        - frame (u32): mesoscope frame, all data aligned
        - time_us (u64): UTC time
        - elapsed_minutes (f32): minutes since start of session
        - single_day_f (array(f32)): fluorescence from single session (from S2P)
        - single_day_f_neuropil (array(f32)): neuropil signal from single session (from S2P)
        - single_day_dff (array(f32)): normalized fluorescence from single session (from S2P)
        - single_day_spikes (array(f32)): deconvolved signal into spikes (from S2P)
        - multi_day_f (array(f32)): multiday signal, from all sessions up to current
        - multi_day_f_neuropil (array(f32)): neuropil signal, from all sessions up to current
        - multi_day_dff (array(f32)): normalized fluorescence from all sessions up to current
        - multi_day_spikes (array(f32)): deconvolved signal into spikes, from all sessions up to current
        - break (u1): this is the brake information; 1 is on, 0 is off
        - screens (u8): screen state; on (1) or off (0)
        - torque_N_cm (f32): torque recorded from torque sensor
        - distance_cm (f64): cumulative distance during session (0 until start of 1st active period)
        - speed_cm_s (f32): in cm/s, derived from time and distance
        - lick (u8): lick sensed (1) or no lick (0)
        - water_uL (f32): cumulative water delivered in uL
        - reward (enum): was reward delivered; string 'yes' or 'no'
        - system_state (enum): system state; string 'idle', 'rest', or 'run'
        - trial (u16): trial number
        - trial_type (enum): for chelsea, base or extended tracks; 'ABC', 'ABCD', etc.  specific to each task
        - cue (u8): cue region identifier: 0 is gray, and the rest are scalars that correspond with cue position on track
        - in_reward_zone (u8): 0 for no, 1 for yes
        - experiment_state (enum): stages of imaging -- 'baseline', 'run', 'cooldown', 'idle'
        - guided (enum): was it a guided trial? 0 for no, 1 for yes

    bin_size_cm : int
        Bin size desired for binning track length (in cm, default 5)
    include_averages : bool
        Whether to compute session average (default True)
    save : bool
        Whether to save processed data (default True)  # could add option later for parquet or csv?

 Returns:
 --------
 Polars dataframe saved as a parquet file with:
     - 'binned_data': Main dataframe with binned arrays per trial
     - 'trial_averages': Average signal per cell per trial
     - 'session_averages': Average signal per cell across entire session

    where the data is organized by trial and each cell has it's own column for activity during that trial
 """
# TODO: i think we might need to get the track data from the yaml file, including cue regions and offset



    num_cells = df['single_day_f'].arr.len()[0]

    # filter for active trials only <-- this could be an arg in the function, for Ivan's rest periods
    active_df = df.filter(pl.col('system_state')=="run")

    trial_distances = (
        active_df.group_by(['trial', 'trial_type'])
        .agg([
            (pl.col('distance_cm').max() - pl.col('distance_cm').min()).alias('trial_distance_cm')
        ])
    )

    # check what distances each trial type typically covers (could also get from yaml file, at least to confirm; rn
    # this works)
    print("Trial type distance summary:")
    print(
        trial_distances.group_by('trial_type')
        .agg([
            pl.col('trial_distance_cm').mean().alias('avg_distance'),
            pl.col('trial_distance_cm').min().alias('min_distance'),
            pl.col('trial_distance_cm').max().alias('max_distance'),
            pl.col('trial_distance_cm').count().alias('n_trials')
        ])
    )


    # normalize distance within each trial (0:trial_length)
    # needed for consistent binning across trials
    normalized_df = active_df.with_columns([
        # group by trial and normalize distance to start at 0
        (pl.col('distance_cm') -
         pl.col('distance_cm').min().over('trial')).alias('trial_distance')
    ])

  #inspect the max and min trial distance.  how can we take care of the discrepancies due to speed/frame rate?
    result = normalized_df.group_by('trial').agg([
        pl.col('trial_distance').first().alias('start_distance'),
        pl.col('trial_distance').last().alias('final_distance'),
        pl.len().alias('num_rows')
    ]).sort('trial')

    print(result)
    stats = result.select([
        pl.col('final_distance').filter(pl.col('final_distance') > 14).min().alias('min_distance'),
        pl.col('final_distance').max().alias('max_distance')
    ])

    print(stats)

    #print(normalized_df.head(300))

    # assign bins based on normalized distance; this uses the
    # for 240cm trials: 48 bins (0-5, 5-10, ..., 235-240)
    # etc

    normalized_df = normalized_df.with_columns([
        (pl.col('trial_distance') // bin_size_cm).cast(pl.Int32).alias('distance_bin')
    ])

    #print(normalized_df.head())

    # expand cell_activity array into separate columns for each cell
    # struct is like a dict; unnest expands struct into the columns

    cell_field_names = [f'cell_{i}' for i in range(num_cells)]
    #
    # expanded_df = normalized_df.with_columns([
    #     pl.col('single_day_f').list.to_struct(fields=cell_field_names)
    # ]).unnest('single_day_f')

#TODO save this as a parquet file
# create simple plotting code that is also able to plot the average using the df (thought might be easier to add that
# in the original file?

    return normalized_df




process_trial_based_activity(behavior_df)