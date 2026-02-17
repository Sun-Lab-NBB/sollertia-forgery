"""
Trial-Based DataFrame Processing Module

Converts frame-based calcium imaging data into trial-indexed structure
for spatial analysis of place cells and neural manifolds.

Core pipeline:
    1. fix_cue_offset() - Reassign frames to correct trials, drop incomplete
    2. group_into_trials() - Frame df → trial-indexed df with arrays
    3. add_binned_signals() - Add spatially binned neural activity
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import polars as pl
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_experiment_config(yaml_path: Path) -> dict:
    """Load experiment configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_track_length(config: dict, trial_type: str) -> Optional[float]:
    """Get track length for a trial type from config."""
    trial_structures = config.get('trial_structures', {})
    if trial_type in trial_structures:
        return trial_structures[trial_type].get('trial_length_cm')
    return None


# =============================================================================
# CORE PIPELINE
# =============================================================================

def fix_cue_offset(
    df: pl.DataFrame,
    config: dict,
    system_state: str = 'run',
    cue_offset_cm: float = None,
) -> pl.DataFrame:
    """
    Reassign frames across trial boundaries to correct for cue offset.
    
    The VR starts 10cm into the track, so trial boundaries in the data
    don't align with visual cue positions. This function:
    1. Identifies frames in the 0-10cm physical zone (end of recorded trial)
    2. Reassigns them to the next trial
    3. Drops first trial (incomplete) and last trial (incomplete)
    
    Parameters
    ----------
    df : pl.DataFrame
        Frame-based dataframe with cumulative distance
    config : dict
        Experiment configuration with trial_structures
    system_state : str
        Filter to this system state (default 'run')
    cue_offset_cm : float, optional
        Override offset from config (default: config['cue_offset_cm'])
    
    Returns
    -------
    pl.DataFrame
        Frame-based dataframe with corrected trial assignments
    """
    # Get offset
    if cue_offset_cm is None:
        cue_offset_cm = config.get('cue_offset_cm', 10.0)
    
    if cue_offset_cm == 0:
        return df.filter(pl.col('system_state') == system_state)
    
    # Filter to active running
    active_df = df.filter(pl.col('system_state') == system_state).sort('frame')
    
    # Build track length lookup from config
    track_lengths = {}
    for trial_type in active_df['trial_type'].unique().to_list():
        length = get_track_length(config, trial_type)
        if length is None:
            # Infer from data if not in config
            type_data = active_df.filter(pl.col('trial_type') == trial_type)
            per_trial = type_data.group_by('trial').agg(
                (pl.col('distance_cm').max() - pl.col('distance_cm').min()).alias('length')
            )
            length = round(per_trial['length'].median() / 20) * 20
        track_lengths[trial_type] = float(length)
    
    # Add track length column via join
    track_length_df = pl.DataFrame({
        'trial_type': list(track_lengths.keys()),
        '_track_length': list(track_lengths.values()),
    }).cast({'trial_type': active_df['trial_type'].dtype})
    active_df = active_df.join(track_length_df, on='trial_type', how='left')
    
    # Compute distance within each recorded trial
    active_df = active_df.with_columns(
        (pl.col('distance_cm') - pl.col('distance_cm').first().over('trial'))
        .alias('_dist_in_trial')
    )
    
    # Compute physical position and identify early portion (0 to offset)
    active_df = active_df.with_columns(
        ((pl.col('_dist_in_trial') + cue_offset_cm) % pl.col('_track_length'))
        .alias('_physical_position')
    )
    active_df = active_df.with_columns(
        (pl.col('_physical_position') < cue_offset_cm).alias('_is_early_portion')
    )
    
    # Get trial list for reassignment mapping
    trials = active_df['trial'].unique().sort().to_list()
    
    # Map: early portion frames get assigned to next trial
    next_trial_map = {trials[i]: trials[i + 1] for i in range(len(trials) - 1)}
    next_trial_map[trials[-1]] = None  # Last trial's early portion has nowhere to go
    
    # Reassign trials
    active_df = active_df.with_columns(
        pl.when(pl.col('_is_early_portion'))
        .then(pl.col('trial').replace(next_trial_map))
        .otherwise(pl.col('trial'))
        .alias('trial')
    )
    
    # Drop incomplete trials:
    # - First trial (missing 0-10cm at start)
    # - Orphaned frames from last trial (assigned to None)
    active_df = active_df.filter(
        (pl.col('trial') != trials[0]) & (pl.col('trial').is_not_null())
    )
    
    # Also drop last trial if incomplete (didn't reach near track length)
    remaining_trials = active_df['trial'].unique().sort().to_list()
    if remaining_trials:
        last_trial = remaining_trials[-1]
        last_trial_df = active_df.filter(pl.col('trial') == last_trial)
        last_dist = last_trial_df['distance_cm']
        last_dist_range = last_dist.max() - last_dist.min()
        last_type = last_trial_df['trial_type'][0]
        expected_length = track_lengths.get(last_type, 180)
        
        # If less than 90% complete, drop it
        if last_dist_range < expected_length * 0.9:
            active_df = active_df.filter(pl.col('trial') != last_trial)
    
    # Clean up temp columns
    active_df = active_df.drop([
        '_track_length', '_dist_in_trial', '_physical_position', '_is_early_portion'
    ])
    
    n_original = len(trials)
    n_final = active_df['trial'].n_unique()
    print(f"Cue offset correction: {n_original} recorded → {n_final} complete trials")
    
    return active_df


def group_into_trials(
    df: pl.DataFrame,
    signal_col: str = 'single_day_f',
) -> pl.DataFrame:
    """
    Convert frame-based dataframe to trial-indexed structure.
    
    Parameters
    ----------
    df : pl.DataFrame
        Frame-based dataframe (after fix_cue_offset)
    signal_col : str
        Column containing neural signals (2D arrays)
    
    Returns
    -------
    pl.DataFrame
        Trial-indexed dataframe with:
        - Scalars: trial, trial_type, system_state, experiment_state, guided,
                   rewarded, n_frames, track_length_cm
        - Arrays: position, speed, cue, lick, reward, signals, etc.
    """
    # Define scalar columns (use mode for trial_type, first for others)
    scalar_cols = ['system_state', 'experiment_state', 'guided']
    
    # Define columns to skip (handled separately or excluded)
    skip_cols = {'trial', 'trial_type', 'frame', signal_col, 'distance_cm'} | set(scalar_cols)

    
    # Identify array columns (everything else)
    array_cols = [c for c in df.columns if c not in skip_cols]
    
    # Build aggregation expressions
    agg_exprs = [
        # Trial type: mode (most common after reassignment)
        pl.col('trial_type').mode().first().alias('trial_type'),
        
        # Other scalars: first value
        *[pl.col(c).first().alias(c) for c in scalar_cols if c in df.columns],
        
        # Frame indices (for reference)
        pl.col('frame').alias('frames'),
        pl.len().alias('n_frames'),
        
        # Array columns: keep as lists
        *[pl.col(c) for c in array_cols if c in df.columns],
    ]
    
    # Group and aggregate
    trial_df = (
        df.sort('frame')
        .group_by('trial', maintain_order=True)
        .agg(agg_exprs)
        .sort('trial')
    )
    
    # Add derived scalar: rewarded (True if any 'yes' in trial)
    if 'reward' in trial_df.columns:
        trial_df = trial_df.with_columns(
            pl.col('reward').list.eval(pl.element() == 'yes').list.any().alias('rewarded')
        )
    
    # Compute normalized position (0 to track_length per trial)
    positions = []
    for trial_num in trial_df['trial']:
        trial_frames = df.filter(pl.col('trial') == trial_num).sort('frame')
        cum_dist = trial_frames['distance_cm'].to_numpy()
        normalized = cum_dist - cum_dist[0]
        positions.append(normalized.tolist())
    
    trial_df = trial_df.with_columns(
        pl.Series('position', positions)
    )

    # Add measured track length from encoder (max position per trial) - byproduct of cue offset issue
    trial_df = trial_df.with_columns(
        pl.col('position').list.max().alias('measured_track_length')
    )

    # Process signals separately (handle 2D arrays)
    if signal_col in df.columns:
        print(f"Processing {signal_col}...")
        signals_per_trial = []
        
        for trial_num in trial_df['trial']:
            trial_frames = df.filter(pl.col('trial') == trial_num).sort('frame')
            signals_list = trial_frames[signal_col].to_list()
            
            if signals_list:
                signals_array = np.vstack([np.array(s) for s in signals_list])
            else:
                signals_array = np.array([])
            
            signals_per_trial.append(signals_array)
        
        trial_df = trial_df.with_columns(
            pl.Series('signals', signals_per_trial, dtype=pl.Object)
        )
    
    print(f"Created {len(trial_df)} trials")
    print(f"Trial types: {trial_df['trial_type'].unique().to_list()}")
    
    return trial_df


def add_binned_signals(
    trial_df: pl.DataFrame,
    bin_size_cm: int = 5,
    config: dict = None,
) -> pl.DataFrame:
    """
    Add spatially binned neural activity to trial dataframe.
    
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
        Trial dataframe with added columns:
        - binned_signals: (n_bins, n_cells) mean activity per bin
        - bin_counts: (n_bins,) frames per bin
        - distance_bins: (n_frames,) bin index per frame
    """
    # Get nominal track lengths for consistent binning
    nominal_lengths = {}
    if config:
        for tt in trial_df['trial_type'].unique().to_list():
            length = get_track_length(config, tt)
            if length:
                nominal_lengths[tt] = int(length)
    
    binned_signals_list = []
    bin_counts_list = []
    distance_bins_list = []
    nominal_values_list = []
    
    for row in trial_df.iter_rows(named=True):
        position = np.array(row['position'])
        signals = row['signals']
        trial_type = row['trial_type']
        
        # Use nominal track length if available, else measured
        track_length = nominal_lengths.get(trial_type, row['measured_track_length'])
        nominal_values_list.append(track_length)
        n_bins = int(track_length / bin_size_cm)
        
        # Compute bin indices for each frame
        bin_indices = np.clip(
            np.floor(position / bin_size_cm).astype(np.int32),
            0, n_bins - 1
        )
        distance_bins_list.append(bin_indices)
        
        # Handle empty or missing signals
        if signals is None or len(signals) == 0:
            n_cells = 0
            binned_signals = np.full((n_bins, n_cells), np.nan)
            bin_counts = np.zeros(n_bins, dtype=np.int32)
        else:
            if len(signals.shape) == 1:
                signals = signals.reshape(-1, 1)
            
            n_cells = signals.shape[1]
            binned_signals = np.full((n_bins, n_cells), np.nan)
            bin_counts = np.zeros(n_bins, dtype=np.int32)
            
            # Compute mean per bin
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if mask.any():
                    binned_signals[bin_idx] = signals[mask].mean(axis=0)
                    bin_counts[bin_idx] = mask.sum()
        
        binned_signals_list.append(binned_signals)
        bin_counts_list.append(bin_counts)


    return trial_df.with_columns([
        pl.Series('binned_signals', binned_signals_list, dtype=pl.Object),
        pl.Series('bin_counts', bin_counts_list, dtype=pl.Object),
        pl.Series('distance_bins', distance_bins_list, dtype=pl.Object),
        pl.Series('nominal_track_length', nominal_values_list, dtype=pl.Int32)
    ])


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class TrialData:
    """
    Container for trial-indexed data.
    
    Attributes
    ----------
    trial_df : pl.DataFrame
        Trial-indexed dataframe
    config : dict
        Experiment configuration
    metadata : dict
        Processing parameters
    """
    trial_df: pl.DataFrame
    config: dict = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def n_cells(self) -> int:
        """Number of cells in dataset."""
        if 'signals' not in self.trial_df.columns:
            return 0
        first_signals = self.trial_df['signals'][0]
        if first_signals is None or len(first_signals) == 0:
            return 0
        return first_signals.shape[1] if len(first_signals.shape) > 1 else 1
    
    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return len(self.trial_df)
    
    @property
    def trial_types(self) -> List[str]:
        """Available trial types."""
        return sorted(self.trial_df['trial_type'].unique().to_list())
    
    def get_trials(self, trial_type: str = None) -> pl.DataFrame:
        """Get trials, optionally filtered by type."""
        if trial_type is None:
            return self.trial_df
        return self.trial_df.filter(pl.col('trial_type') == trial_type)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_session(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'single_day_f',
    bin_size_cm: int = 5,
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
        Spatial bin size
    system_state : str
        Filter to this system state
    
    Returns
    -------
    TrialData
        Container with processed trial dataframe
    """
    print("Step 1: Fixing cue offset...")
    corrected_df = fix_cue_offset(df, config, system_state=system_state)
    
    print("\nStep 2: Grouping into trials...")
    trial_df = group_into_trials(corrected_df, signal_col=signal_col)
    
    print("\nStep 3: Adding binned signals...")
    trial_df = add_binned_signals(trial_df, bin_size_cm=bin_size_cm, config=config)
    
    print("\n✓ Pipeline complete!")
    
    return TrialData(
        trial_df=trial_df,
        config=config,
        metadata={
            'signal_col': signal_col,
            'bin_size_cm': bin_size_cm,
            'system_state': system_state,
        }
    )


# =============================================================================
# SAVE / LOAD
# =============================================================================
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
    """Load trial data from parquet + npz."""
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


# =============================================================================
# UTILITIES
# =============================================================================

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


def get_cue_regions(
    trial_df: pl.DataFrame,
    trial_type: str,
) -> dict:
    """
    Extract cue region boundaries from trial data.
    
    Returns
    -------
    dict
        {cue_id: (start_cm, end_cm)}
    """
    trials = trial_df.filter(pl.col('trial_type') == trial_type)
    if len(trials) == 0:
        return {}
    
    first = trials.row(0, named=True)
    cues = np.array(first['cue'])
    positions = np.array(first['position'])
    
    # Find cue transitions
    changes = np.concatenate([[0], np.where(np.diff(cues) != 0)[0] + 1, [len(cues)]])
    
    regions = {}
    for i in range(len(changes) - 1):
        start_idx = changes[i]
        end_idx = changes[i + 1] - 1
        cue_id = int(cues[start_idx])
        
        if cue_id == 0:  # Skip gray zones
            continue
        
        start_pos = positions[start_idx]
        end_pos = positions[end_idx]
        
        if cue_id not in regions:
            regions[cue_id] = (start_pos, end_pos)
    
    return regions


if __name__ == "__main__":
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    behavior_df = pl.read_ipc('/Users/cs963/Desktop/sun_lab_projects/26_explore/2025-09-16-18-44-32-476061.feather')


    experiment_config = load_experiment_config(
        '/Users/cs963/Desktop/sun_lab_projects/26_explore/experiment_configuration.yaml') #for working at home

    # Run pipeline - returns TrialData container
    data = process_session(
        behavior_df,
        config=experiment_config,
        signal_col='single_day_f',
        bin_size_cm=5,
        system_state='run'
    )


    save_trial_data(data, session_root)