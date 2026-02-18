"""
DataFrame Processing Module

Converts frame-based calcium imaging data into a cue-corrected, binned frame-level DataFrame for spatial analysis of
place cells and neural manifolds.

Core pipeline:
    1. fix_cue_offset() - Reassign frames to correct trials to fix Unity mismatch, drop incomplete
    2. add_position_and_bin() - Add within-trial position + spatial bin index

Output is a flat frame-level df (one row per frame) that can write_parquet / read_parquet.

Analysis helpers:
    - compute_binned_average() - Single-cell spatial tuning via Polars (fast)
    - compute_session_averages() - All-cells averages via numpy (vectorized)
"""

from pathlib import Path

import numpy as np
import polars as pl
import yaml


# CONFIGURATION

def load_experiment_config(yaml_path: Path) -> dict:
    """Load experiment configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_track_length(config: dict, trial_type: str) -> float | None:
    """Get track length for a trial type from config."""
    trial_structures = config.get('trial_structures', {})
    if trial_type in trial_structures:
        return trial_structures[trial_type].get('trial_length_cm')
    return None

def get_cue_regions(
    config: dict,
    trial_type: str,
) -> dict:
    """
    Extract cue region boundaries from the experiment config file.
    Boundaries are nominal (from config), not measured from encoder data.

    Returns
    -------
    dict
        {cue_id: [(start_cm, end_cm), ...]}
    """
    trial_structure = config.get('trial_structures', {}).get(trial_type)
    if not trial_structure:
        return {}

    cue_sequence = trial_structure['cue_sequence']
    cue_widths = config.get('cue_map', {})

    regions = {}
    position = 0.0

    for cue_id in cue_sequence:
        if cue_id not in cue_widths:
            raise KeyError(f"Cue ID {cue_id} not found in cue_map config")
        width = cue_widths.get(cue_id, 30.0)

        if cue_id not in regions:
            regions[cue_id] = [(position, position + width)]
        else:
            regions[cue_id].append((position, position + width))

        position += width

    return regions


# PREPROCESSING

def fix_cue_offset(
    df: pl.DataFrame,
    config: dict,
    system_state: str = 'run',
) -> pl.DataFrame:
    """
    Reassign frames across trial boundaries to correct for cue offset. Realigns to cue values instead, location is
    preserved
    
    The VR starts 10cm into the track, so trial boundaries in the data
    don't align with visual cue positions. This function:
    1. Identifies frames in the 0-10cm physical zone (end of recorded trial) using the cue col information
    2. Reassigns them to the next trial
    3. Drops first trial (incomplete) and last trial (incomplete)
    
    Parameters
    ----------
    df : pl.DataFrame
        Frame-based dataframe with cumulative distance
    config : dict
        Experiment configuration with trial_structures and cue offset information
    system_state : str
        Filter to this system state (default 'run')
    
    Returns
    -------
    pl.DataFrame
        Frame-based dataframe with corrected trial #, trial type, and guided values
    """
    # Get offset
    cue_offset_cm = config.get('cue_offset_cm', 0.0)

    # Filter to active running
    active_df = df.filter(pl.col('system_state') == system_state).sort('frame')

    if cue_offset_cm == 0:
        return active_df

    # Get the first cue ID from config (should be the same for all trial types)
    first_cues = set()
    for structure in config.get('trial_structures', {}).values():
        seq = structure.get('cue_sequence', [])
        if seq:
            first_cues.add(seq[0])

    if len(first_cues) != 1:
        raise ValueError(f"Expected all trial types to start with same cue, got: {first_cues}")
    first_cue = first_cues.pop()

    # Detect trial starts: cue == first_cue AND previous frame was != first_cue
    active_df = active_df.with_columns([
        (pl.col('cue') == first_cue).alias('_is_first_cue'),
        (pl.col('cue').shift(1) != first_cue).fill_null(True).alias('_prev_not_first_cue'),
    ])

    active_df = active_df.with_columns(
        (pl.col('_is_first_cue') & pl.col('_prev_not_first_cue'))
        .cum_sum()
        .alias('_new_trial')
    )

    # Sanity check: compare distance per new trial to expected track lengths
    trial_distances = (
        active_df.group_by('_new_trial')
        .agg((pl.col('distance_cm').max() - pl.col('distance_cm').min()).alias('distance'))
    )
    median_dist = trial_distances['distance'].median()
    print(f"  Median trial distance: {median_dist:.1f}cm")

    # Get trial_type and guided from original labels (mode per new trial)
    active_df = active_df.with_columns([
        pl.col('trial_type').mode().first().over('_new_trial').alias('_new_trial_type'),
        pl.col('guided').mode().first().over('_new_trial').alias('_new_guided'),
    ])

    # Replace old columns
    active_df = active_df.with_columns([
        pl.col('_new_trial').alias('trial'),
        pl.col('_new_trial_type').alias('trial_type'),
        pl.col('_new_guided').alias('guided'),
    ])

    # Drop first and last trials (incomplete)
    trials = active_df['trial'].unique().sort().to_list()
    active_df = active_df.filter(
        (pl.col('trial') != trials[0]) & (pl.col('trial') != trials[-1])
    )

    # Clean up temp columns
    temp_cols = [c for c in active_df.columns if c.startswith('_')]
    active_df = active_df.drop(temp_cols)

    n_final = active_df['trial'].n_unique()
    print(f"Cue offset correction: {len(trials)} recorded → {n_final} complete trials")

    return active_df


def add_position_and_bins(
    df: pl.DataFrame,
    config: dict = None,
    bin_size_cm: int = 5
) -> pl.DataFrame:
    """
    Add within-trial position and spatial bin index.

    Adds columns:
        - position: distance_cm normalized to 0 at each trial start
        - distance_bin: integer bin index (0 to n_bins-1), clipped per trial type
        - nominal_track_length: from config, per trial type
    
    Parameters
    ----------
    trial_df : pl.DataFrame
        Trial-indexed dataframe with 'signals' and 'position' columns
    bin_size_cm : int
        Spatial bin size in cm
    config : dict, optional
        Experiment config with track lengths (for consistent bin counts)
    
    Returns
    -------
    pl.DataFrame
        Input df with position, distance_bin, and nominal_track_length columns added
    """
    # Within-trial position: distance relative to first frame in each trial. Takes first distance_cm value for each
    # trial and subtracts it from all the other frames to normalize to nominal track length
    df = df.with_columns(
        (pl.col('distance_cm') - pl.col('distance_cm').first().over('trial'))
        .alias('position')
    )

    # Create a dict with the trial types and their lengths from the config file
    length_map = {
        tt: get_track_length(config, tt)
        for tt in df['trial_type'].unique().to_list()
    }
    missing = [tt for tt, v in length_map.items() if v is None]
    if missing:
        raise ValueError(f"No track length in config for: {missing}")

    # Map to column
    df = df.with_columns(
        pl.col('trial_type')
        .replace_strict(length_map)
        .cast(pl.Float64)
        .alias('nominal_track_length')
    )

    # Bin index: floor(position / bin_size)
    df = df.with_columns(
        (pl.col('position') / bin_size_cm)
        .floor()
        .cast(pl.Int32)
        .alias('distance_bin')
    )

    return df

# MAIN PIPELINE

def process_session(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'single_day_f',
    bin_size_cm: int | None = 5,
    system_state: str = 'run',
) -> TrialData:
    """
    Complete pipeline: frame data → trial-indexed structure.
    
    Parameters
    ----------
    df : pl.DataFrame
        Raw frame-based dataframe
    config : dict
        Experiment configuration
    signal_col : str
        Column containing neural signals
    bin_size_cm : int
        Spatial bin size if binning is used, default is 5 cm; if None, no binning will be applied
    system_state : str
        Filter to this system state
    
    Returns
    -------
    TrialData
        Container with processed trial dataframe
    """
    print("Step 1: Fixing cue offset...")       #this could be an optional argument if we don't want to do this
    corrected_df = fix_cue_offset(df, config, system_state=system_state)
    
    print("\nStep 2: Grouping into trials...")
    trial_df = group_into_trials(corrected_df, signal_col=signal_col)

    binning = bin_size_cm is not None
    if binning:
        if not isinstance(bin_size_cm, int):
            raise TypeError("bin_size_cm must be an int")
        print("\nStep 3: Adding binned signals...")
        trial_df = add_binned_signals(trial_df, bin_size_cm=bin_size_cm, config=config)
    
    print("\nPipeline complete!")
    
    return TrialData(
        trial_df=trial_df,
        config=config,
        metadata={
            'signal_col': signal_col,
            'bin_size_cm': bin_size_cm,
            'system_state': system_state,
            'binning': binning,
        }
    )


# SAVE / LOAD

def save_trial_data(data: TrialData, output_path: Path):
    """Save trial data: parquet for dataframe, npz for numpy arrays."""
    output_path = Path(output_path)

    df = data.trial_df
    object_cols = [c for c in df.columns if df[c].dtype == pl.Object]

    # Save numpy arrays separately with trial index
    arrays = {'trial_index': df['trial'].to_numpy()}
    for col in object_cols:
        arrays[col] = np.array(df[col].to_list(), dtype=object)

    np.savez(output_path.with_suffix('.npz'), **arrays)

    # Save dataframe without Object columns
    df_clean = df.drop(object_cols)
    df_clean.write_parquet(output_path.with_suffix('.parquet'))

    # Save metadata
    meta = {
        'n_trials': len(df),
        'trial_types': data.trial_types,
        'processing': data.metadata,
        'object_columns': object_cols,
    }
    with open(output_path.with_suffix('.meta.yaml'), 'w') as f:
        yaml.dump(meta, f)

    print(f"Saved: {output_path.with_suffix('.parquet')}")
    print(f"Saved: {output_path.with_suffix('.npz')}")
    print(f"Saved: {output_path.with_suffix('.meta.yaml')}")


def load_trial_data(path: Path, config_path: Path = None) -> TrialData:
    """Load trial data from parquet + npz, and metadata from yaml
    This was the easiest way to avoid object errors from nested arrays in the df """
    path = Path(path).with_suffix('')  # Strip any extension

    trial_df = pl.read_parquet(path.with_suffix('.parquet'))

    # Load numpy arrays and verify alignment
    with np.load(path.with_suffix('.npz'), allow_pickle=True) as npz:
        saved_trials = npz['trial_index']
        assert np.array_equal(saved_trials, trial_df['trial'].to_numpy()), "Trial index mismatch!"

        for col in npz.files:
            if col == 'trial_index':
                continue
            trial_df = trial_df.with_columns(
                pl.Series(col, list(npz[col]), dtype=pl.Object)
            )

    config = None
    if config_path:
        config = load_experiment_config(config_path)

    metadata = {}
    meta_path = path.with_suffix('.meta.yaml')
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f).get('processing', {})

    return TrialData(trial_df=trial_df, config=config, metadata=metadata)



# UTILITIES


def compute_session_averages(
    trial_df: pl.DataFrame,
    by_trial_type: bool = True,
) -> dict:
    """
    Compute session-level spatial averages.
    
    Returns
    -------
    dict
        Per trial_type: {'session_avg': (n_bins, n_cells), 
                         'session_sem': (n_bins, n_cells),
                         'n_trials': int}
    """
    from scipy import stats
    
    if not by_trial_type:
        all_binned = np.stack(trial_df['binned_signals'].to_list())
        return {
            'all': {
                'session_avg': np.nanmean(all_binned, axis=0),
                'session_sem': stats.sem(all_binned, axis=0, nan_policy='omit'),
                'n_trials': len(trial_df),
            }
        }
    
    results = {}
    for trial_type in trial_df['trial_type'].unique().sort().to_list():
        type_df = trial_df.filter(pl.col('trial_type') == trial_type)
        all_binned = np.stack(type_df['binned_signals'].to_list())
        
        results[trial_type] = {
            'session_avg': np.nanmean(all_binned, axis=0),
            'session_sem': stats.sem(all_binned, axis=0, nan_policy='omit'),
            'n_trials': len(type_df),
        }
    
    return results





if __name__ == "__main__":
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    behavior_df = pl.read_ipc(session_root / '2025-09-16-18-44-32-476061.feather')

    experiment_config = load_experiment_config(
        session_root / 'experiment_configuration.yaml') #for working at home

    frame_df = fix_cue_offset(behavior_df, experiment_config, system_state='run')

    print(frame_df.columns)
    with pl.Config(tbl_cols=100, tbl_rows=100, set_tbl_hide_dataframe_shape=False):
        print(frame_df.head(100))



    #
    # # Run pipeline - returns TrialData container
    # data = process_session(
    #     behavior_df,
    #     config=experiment_config,
    #     signal_col='single_day_f',
    #     bin_size_cm=5,
    #     system_state='run'
    # )
    #
    #
    # save_trial_data(data, session_root)