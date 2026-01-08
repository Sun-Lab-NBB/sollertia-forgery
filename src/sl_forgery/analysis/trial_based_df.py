'''The purpose of this module is to organize the combined behavior and imaging data into a single usable format for
place field plotting, umap plotting, and other analyses.  The original data uses cumulative distance over the session.
This handles distance normalization, spatial binning, and session statistics.
Supports saving processed dataframe for faster subsequent loads.'''

"""
Trial-Based DataFrame Processing Module

Converts frame-based calcium imaging data into trial-indexed structure
for spatial analysis of place cells and neural manifolds.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import re

import numpy as np
import polars as pl
import yaml
from scipy import stats
from matplotlib import pyplot as plt

# CONFIGURATION

DEFAULT_CUE_COLORS = {
    0: '#CCCCCC',  # Gray
    1: '#FFD93D',  # A - Yellow
    2: '#6BCB77',  # B - Green
    3: '#4D96FF',  # C - Blue
    4: '#FF6B9D',  # D - Pink
    5: '#A23BFF',  # E - Purple
    255: '#333333'  # Dark
}

DEFAULT_CUE_LABELS = {
    0: 'Gray', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 255: 'Dark'
}

# Columns constant within a trial - aggregate to single value
DEFAULT_TRIAL_COLUMNS = [
    ('system_state', 'first'),
    ('experiment_state', 'first'),
    ('guided', 'first'),
]


@dataclass
class SessionConfig:
    """
    Session configuration from YAML and metadata from filename.

    Tracks stored as dicts:
        tracks['ABC'] = {
            'length_cm': 180,
            'cue_sequence': [1, 0, 2, 0, 3, 0],
            'reward_zone': (128, 142),
        }
    """
    # From YAML
    task_type: str = 'unknown'
    tracks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cue_offset_cm: float = 10.0
    experiment_states: Dict[str, Dict] = field(default_factory=dict)

    # From filename
    animal_id: str = 'unknown'
    date: Optional[datetime] = None
    session_timestamp: str = ''
    source_file: Optional[Path] = None

    # Processing defaults
    bin_size_cm: int = 5
    frame_rate: float = 30.0

    # Visual defaults
    cue_colors: Dict[int, str] = field(default_factory=lambda: DEFAULT_CUE_COLORS.copy())
    cue_labels: Dict[int, str] = field(default_factory=lambda: DEFAULT_CUE_LABELS.copy())

    @classmethod
    def load(
            cls,
            yaml_path: Optional[Path] = None,
            data_path: Optional[Path] = None
    ) -> 'SessionConfig':
        """
        Load config from YAML and/or extract metadata from data filename.

        Parameters
        ----------
        yaml_path : Path, optional
            Path to experiment YAML config file
        data_path : Path, optional
            Path to data file (for extracting animal_id, date from filename)
        """
        config = cls()

        # Extract metadata from filename
        if data_path is not None:
            config.source_file = Path(data_path)
            filename = config.source_file.stem

            # Try to parse animal ID from parent directory
            parent_name = config.source_file.parent.name
            if parent_name and not parent_name.startswith('.'):
                config.animal_id = parent_name

            # Try to parse timestamp from filename
            timestamp_pattern = r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'
            match = re.search(timestamp_pattern, filename)
            if match:
                config.session_timestamp = match.group(1)
                try:
                    config.date = datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')
                except ValueError:
                    pass

        # Load YAML config
        if yaml_path is not None and Path(yaml_path).exists():
            config._load_yaml(yaml_path)

        return config

    def _load_yaml(self, yaml_path: Path):
        """Parse YAML config file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            return

        self.cue_offset_cm = data.get('cue_offset_cm', 10.0)

        # Parse experiment states
        if 'experiment_states' in data:
            self.experiment_states = data['experiment_states']

        # Parse trial structures into tracks
        if 'trial_structures' in data:
            for track_name, track_data in data['trial_structures'].items():
                self.tracks[track_name] = {
                    'length_cm': track_data.get('trial_length_cm', 180),
                    'cue_sequence': track_data.get('cue_sequence', []),
                    'reward_zone': (
                        track_data.get('reward_zone_start_cm'),
                        track_data.get('reward_zone_end_cm')
                    ) if 'reward_zone_start_cm' in track_data else None,
                    'guidance_trigger_cm': track_data.get('guidance_trigger_location_cm'),
                    'reward_size_ul': track_data.get('trial_reward_size_ul', 5.0),
                }

    def get_track_length(self, trial_type: str) -> Optional[float]:
        """Get nominal track length for a trial type."""
        if trial_type in self.tracks:
            return self.tracks[trial_type].get('length_cm')
        return None


# =============================================================================
# DATA CONTAINER
# =============================================================================

@dataclass
class TrialData:
    """
    Container for trial-indexed data with references to original data.

    Attributes
    ----------
    trial_df : pl.DataFrame
        Trial-indexed dataframe with per-trial arrays
    track_lengths : dict
        Nominal track length per trial type {trial_type: length_cm}
    session_stats : dict
        Session averages and SEM per trial type
    original_df : pl.DataFrame, optional
        Filtered frame-based dataframe (by system_state)
    raw_df : pl.DataFrame, optional
        Completely unfiltered original dataframe
    config : SessionConfig, optional
        Session configuration and metadata
    metadata : dict
        Processing parameters used
    """
    trial_df: pl.DataFrame
    track_lengths: dict
    session_stats: dict = None
    original_df: pl.DataFrame = None
    raw_df: pl.DataFrame = None
    config: SessionConfig = None
    metadata: dict = field(default_factory=dict)


# CORE PROCESSING FUNCTIONS

def create_trial_indexed_dataframe(
        df: pl.DataFrame,
        system_state: Optional[str] = 'run',
        signal_col: str = 'single_day_f',
        bin_size_cm: int = 5,
        config: Optional[SessionConfig] = None,
        additional_trial_columns: Optional[List[Tuple[str, str]]] = None,
        keep_original: bool = True,
        keep_raw: bool = False,
) -> Tuple[pl.DataFrame, dict, Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Transform frame-based data into trial-based structure.

    Parameters
    ----------
    df : pl.DataFrame
        Raw frame-based dataframe
    system_state : str or None
        Filter to this system state (e.g., 'run'). None = include all.
    signal_col : str
        Column containing neural signals
    bin_size_cm : int
        Bin size for spatial binning
    config : SessionConfig, optional
        If provided, use config track lengths instead of inferring
    additional_trial_columns : list of (str, str), optional
        Extra columns to aggregate per-trial as (column, agg_method)
    keep_original : bool
        Return the filtered dataframe
    keep_raw : bool
        Return the unfiltered dataframe

    Returns
    -------
    trial_df : pl.DataFrame
        Trial-indexed dataframe
    track_lengths : dict
        Nominal track length per trial type
    original_df : pl.DataFrame or None
    raw_df : pl.DataFrame or None
    """
    # Store references
    raw_df = df if keep_raw else None

    # Filter by system state
    if system_state is None:
        active_df = df.filter(pl.col('system_state') != 'idle')  #all states besides the idle period
    else:
        active_df = df.filter(pl.col('system_state') == system_state)

    original_df = active_df if keep_original else None

    print(f"Signal column dtype: {active_df[signal_col].dtype}")
    print(f"First signal shape: {len(active_df[signal_col][0])}")
    print(f"Total frames after filtering: {len(active_df)}")

    # Sort by frame for proper ordering
    active_df = active_df.sort('frame')

    # Build aggregation list
    agg_list = [
        pl.col('frame').alias('frames'),
        pl.len().alias('n_frames'),
    ]

    # Aggregation methods for trial-level columns
    agg_methods = {
        'first': lambda c: pl.col(c).first(),       #main usage
        'last': lambda c: pl.col(c).last(),
        'mean': lambda c: pl.col(c).mean(),
        'sum': lambda c: pl.col(c).sum(),
        'max': lambda c: pl.col(c).max(),       #good for reward
        'min': lambda c: pl.col(c).min(),
    }

    # Build set of trial-level columns
    trial_level_cols = {c[0] for c in DEFAULT_TRIAL_COLUMNS}
    if additional_trial_columns:
        trial_level_cols.update(c[0] for c in additional_trial_columns)

    # Add trial-level column aggregations
    for col_name, agg_method in DEFAULT_TRIAL_COLUMNS:
        if col_name in active_df.columns:
            agg_list.append(agg_methods[agg_method](col_name).alias(col_name))

    if additional_trial_columns:
        for col_name, agg_method in additional_trial_columns:
            if col_name in active_df.columns:
                agg_list.append(agg_methods[agg_method](col_name).alias(col_name))

    # Add rewarded flag (1 if any reward delivered, 0 otherwise)
    if 'reward' in active_df.columns:
        agg_list.append(pl.col('reward').max().alias('rewarded'))

    # Everything else becomes an array (except grouping, trial-level, signal, frame)
    grouping_cols = {'trial', 'trial_type'}
    skip_cols = grouping_cols | trial_level_cols | {signal_col, 'frame'}

    for col in active_df.columns:
        if col not in skip_cols:
            agg_list.append(pl.col(col))

    # Group and aggregate
    trial_df = active_df.group_by(
        ['trial', 'trial_type'], maintain_order=True
    ).agg(agg_list).sort('trial')

    # Calculate trial duration
    trial_df = trial_df.with_columns([
        ((pl.col('elapsed_minutes').list.last() -
          pl.col('elapsed_minutes').list.first()) * 60).alias('duration_s')
    ])

    # Process signals separately (handle 2D arrays)
    print("Processing signals per trial...")
    signals_per_trial = []

    for trial_num in trial_df['trial']:
        trial_data = active_df.filter(pl.col('trial') == trial_num).sort('frame')
        signals_list = trial_data[signal_col].to_list()

        if len(signals_list) > 0:
            signals_array = np.vstack([np.array(s) for s in signals_list])
        else:
            signals_array = np.array([])

        signals_per_trial.append(signals_array)

    trial_df = trial_df.with_columns([
        pl.Series('signals', signals_per_trial, dtype=pl.Object)
    ])

    # Reset distances within each trial (create new column)
    print("Computing per-trial distances...")
    distance_in_trial_list = []

    for row in trial_df.iter_rows(named=True):
        cumulative_dists = np.array(row['distance_cm'])

        if len(cumulative_dists) > 0:
            trial_distances = cumulative_dists - cumulative_dists[0]

            # Check for distance decreases (wrong trial boundary)
            distance_diffs = np.diff(trial_distances)
            if np.any(distance_diffs < -1):
                problem_idx = np.where(distance_diffs < -1)[0]
                print(f"  Warning: Trial {row['trial']} has distance decrease at {problem_idx}")
                first_problem = problem_idx[0] + 1
                trial_distances = trial_distances[:first_problem]
                print(f"    Truncating from {len(cumulative_dists)} to {first_problem} frames")
        else:
            trial_distances = cumulative_dists

        distance_in_trial_list.append(trial_distances)

    trial_df = trial_df.with_columns([
        pl.Series('distance_in_trial', distance_in_trial_list, dtype=pl.Object)
    ])

    # Get all array columns that need truncation
    array_cols = [col for col in trial_df.columns
                  if trial_df[col].dtype == pl.Object and col != 'distance_in_trial']

    # Synchronize all array columns to match truncated distances
    print("Synchronizing array lengths...")

    for col in array_cols:
        truncated_arrays = []
        for i, row in enumerate(trial_df.iter_rows(named=True)):
            target_length = len(distance_in_trial_list[i])
            original_array = np.array(row[col]) if row[col] is not None else np.array([])

            if len(original_array) > target_length:
                truncated_arrays.append(original_array[:target_length])
            else:
                truncated_arrays.append(original_array)

        trial_df = trial_df.with_columns([
            pl.Series(col, truncated_arrays, dtype=pl.Object)
        ])

    # Update n_frames
    trial_df = trial_df.with_columns([
        pl.Series('n_frames', [len(d) for d in distance_in_trial_list])
    ])

    # Get measured track lengths
    print("Extracting track lengths from data...")
    measured_track_lengths = [
        np.array(d).max() if len(d) > 0 else 0
        for d in distance_in_trial_list
    ]

    trial_df = trial_df.with_columns([
        pl.Series('measured_track_length', measured_track_lengths)
    ])

    # Determine nominal track lengths per trial type
    print("\nDetermining nominal track lengths by trial type...")
    nominal_track_lengths = {}

    for trial_type in trial_df['trial_type'].unique().sort():
        # First check if config provides ground truth
        if config is not None:
            config_length = config.get_track_length(trial_type)
            if config_length is not None:
                nominal_track_lengths[trial_type] = float(config_length)
                print(f"  {trial_type}: {config_length} cm (from config)")
                continue

        # Fall back to inference from data
        type_trials = trial_df.filter(pl.col('trial_type') == trial_type)
        median_length = type_trials['measured_track_length'].median()
        nominal_length = round(median_length / 20) * 20
        nominal_track_lengths[trial_type] = float(nominal_length)
        print(f"  {trial_type}: measured {median_length:.1f} cm → nominal {nominal_length} cm")

    # Normalize distances
    print("Normalizing distances to nominal track lengths...")
    normalized_distance_list = []

    for i, row in enumerate(trial_df.iter_rows(named=True)):
        distances = distance_in_trial_list[i]
        measured_max = measured_track_lengths[i]
        nominal_length = nominal_track_lengths[row['trial_type']]

        if measured_max > 0:
            normalized = distances * (nominal_length / measured_max)
        else:
            normalized = distances

        normalized_distance_list.append(normalized)

    trial_df = trial_df.with_columns([
        pl.Series('distance_normalized', normalized_distance_list, dtype=pl.Object)
    ])

    # Add nominal track length column
    trial_df = trial_df.with_columns([
        pl.Series('track_length_cm',
                  [nominal_track_lengths[tt] for tt in trial_df['trial_type']])
    ])

    # Compute distance bins
    print("Computing distance bins...")
    distance_bins_list = []

    for i, row in enumerate(trial_df.iter_rows(named=True)):
        normalized = normalized_distance_list[i]
        n_bins = int(row['track_length_cm'] / bin_size_cm)
        bins = np.clip(np.floor(normalized / bin_size_cm).astype(np.int32), 0, n_bins - 1)
        distance_bins_list.append(bins)

    trial_df = trial_df.with_columns([
        pl.Series('distance_bins', distance_bins_list, dtype=pl.Object)
    ])

    print(f"\nOutput columns: {trial_df.columns}")

    return trial_df, nominal_track_lengths, original_df, raw_df


def compute_binned_activity(
        trial_df: pl.DataFrame,
        bin_size_cm: int = 5
) -> pl.DataFrame:
    """
    Add binned activity columns to trial dataframe.

    Returns trial_df with:
        - binned_signals: (n_bins, n_cells) averaged activity
        - bin_counts: frames per bin
    """
    binned_signals_list = []
    bin_counts_list = []

    for row in trial_df.iter_rows(named=True):
        signals = np.array(row['signals'])
        bin_indices = np.array(row['distance_bins'])
        n_bins = int(row['track_length_cm'] / bin_size_cm)

        if signals.shape[0] == 0 or len(bin_indices) == 0:
            num_cells = signals.shape[1] if len(signals.shape) > 1 else 0
            binned_signals = np.full((n_bins, num_cells), np.nan, dtype=np.float32)
            bin_counts = np.zeros(n_bins, dtype=np.int32)
        else:
            if len(signals.shape) == 1:
                signals = signals.reshape(-1, 1)

            num_cells = signals.shape[1]
            binned_signals = np.full((n_bins, num_cells), np.nan, dtype=np.float32)
            bin_counts = np.zeros(n_bins, dtype=np.int32)

            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if mask.any():
                    binned_signals[bin_idx] = signals[mask].mean(axis=0)
                    bin_counts[bin_idx] = mask.sum()

        binned_signals_list.append(binned_signals)
        bin_counts_list.append(bin_counts)

    return trial_df.with_columns([
        pl.Series('binned_signals', binned_signals_list, dtype=pl.Object),
        pl.Series('bin_counts', bin_counts_list, dtype=pl.Object)
    ])


def compute_session_averages(
        trial_df: pl.DataFrame,
        by_trial_type: bool = True
) -> dict:
    """
    Compute session-level averages from trial dataframe.

    Returns
    -------
    dict with per trial_type:
        - 'session_avg': (n_bins, n_cells)
        - 'session_sem': (n_bins, n_cells)
        - 'n_trials': int
    """
    if not by_trial_type:
        all_binned = np.stack([row['binned_signals'] for row in trial_df.iter_rows(named=True)])
        return {
            'session_avg': np.nanmean(all_binned, axis=0),
            'session_sem': stats.sem(all_binned, axis=0, nan_policy='omit'),
            'n_trials': len(trial_df)
        }

    results = {}
    for trial_type in trial_df['trial_type'].unique().sort():
        type_trials = trial_df.filter(pl.col('trial_type') == trial_type)
        all_binned = np.stack([row['binned_signals'] for row in type_trials.iter_rows(named=True)])

        results[trial_type] = {
            'session_avg': np.nanmean(all_binned, axis=0),
            'session_sem': stats.sem(all_binned, axis=0, nan_policy='omit'),
            'n_trials': len(type_trials)
        }

    return results


# MAIN PIPELINE

def full_pipeline(
        df: pl.DataFrame,
        system_state: Optional[str] = 'run',
        signal_col: str = 'single_day_f',
        bin_size_cm: int = 5,
        config: Optional[SessionConfig] = None,
        config_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        additional_trial_columns: Optional[List[Tuple[str, str]]] = None,
        keep_original: bool = True,
        keep_raw: bool = False,
) -> TrialData:
    """
    Complete pipeline from frame-based to trial-indexed data.

    Parameters
    ----------
    df : pl.DataFrame
        Raw frame-based dataframe
    system_state : str or None
        Filter to this system state. None = include all except 'idle'.
    signal_col : str
        Column containing neural signals
    bin_size_cm : int
        Spatial bin size
    config : SessionConfig, optional
        Pre-loaded configuration object
    config_path : Path, optional
        Path to YAML config file (loads if config not provided)
    data_path : Path, optional
        Path to data file (for metadata extraction)
    additional_trial_columns : list of (str, str), optional
        Extra per-trial aggregated columns
    keep_original : bool
        Keep reference to filtered original dataframe
    keep_raw : bool
        Keep reference to unfiltered raw dataframe

    Returns
    -------
    TrialData
        Container with trial_df, track_lengths, session_stats, and references
    """
    # Load config if path provided but config not given
    if config is None and config_path is not None:
        config = SessionConfig.load(yaml_path=config_path, data_path=data_path)
    elif config is None and data_path is not None:
        config = SessionConfig.load(data_path=data_path)

    print("Step 1: Creating trial-indexed dataframe...")
    trial_df, track_lengths, original_df, raw_df = create_trial_indexed_dataframe(
        df,
        system_state=system_state,
        signal_col=signal_col,
        bin_size_cm=bin_size_cm,
        config=config,
        additional_trial_columns=additional_trial_columns,
        keep_original=keep_original,
        keep_raw=keep_raw,
    )
    print(f"  ✓ Created {len(trial_df)} trials")
    print(f"  ✓ Trial types: {trial_df['trial_type'].unique().to_list()}")

    print("\nStep 2: Computing binned activity...")
    trial_df = compute_binned_activity(trial_df, bin_size_cm)
    print("  ✓ Binned activity computed")

    print("\nStep 3: Computing session averages...")
    session_stats = compute_session_averages(trial_df, by_trial_type=True)

    for trial_type, stats_dict in session_stats.items():
        n_bins = stats_dict['session_avg'].shape[0]
        print(f"  ✓ {trial_type}: {stats_dict['n_trials']} trials, "
              f"{n_bins} bins × {bin_size_cm} cm = {track_lengths[trial_type]:.0f} cm track")

    print("\n✓ Pipeline complete!")

    return TrialData(
        trial_df=trial_df,
        track_lengths=track_lengths,
        session_stats=session_stats,
        original_df=original_df,
        raw_df=raw_df,
        config=config,
        metadata={
            'system_state_filter': system_state,
            'signal_col': signal_col,
            'bin_size_cm': bin_size_cm,
        }
    )


# SAVE / LOAD

def save_trial_data(data: TrialData, output_path: Path):
    """
    Save processed trial data to feather file.

    Note: Config should be reloaded from original YAML when loading.
    """
    output_path = Path(output_path).with_suffix('.feather')
    data.trial_df.write_ipc(output_path)
    print(f"Saved: {output_path}")


def load_trial_data(
        feather_path: Path,
        config_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
) -> TrialData:
    """
    Load processed trial data.

    Parameters
    ----------
    feather_path : Path
        Path to saved .feather file
    config_path : Path, optional
        Path to original YAML config
    data_path : Path, optional
        Original data path for metadata extraction
    """
    trial_df = pl.read_ipc(feather_path)

    # Reload config if paths provided
    config = None
    if config_path is not None or data_path is not None:
        config = SessionConfig.load(yaml_path=config_path, data_path=data_path)

    # Infer track lengths from saved trial_df
    track_lengths = {}
    for tt in trial_df['trial_type'].unique():
        lengths = trial_df.filter(pl.col('trial_type') == tt)['track_length_cm']
        track_lengths[tt] = float(lengths[0])

    # Recompute stats if binned signals exist
    session_stats = None
    if 'binned_signals' in trial_df.columns:
        session_stats = compute_session_averages(trial_df, by_trial_type=True)

    return TrialData(
        trial_df=trial_df,
        track_lengths=track_lengths,
        session_stats=session_stats,
        config=config,
    )


# CUE REGION EXTRACTION

def get_cue_regions(
        trial_df: pl.DataFrame,
        trial_type: str = 'ABC',
        verbose: bool = False
) -> dict:
    """
    Extract cue region boundaries from trial data.

    Returns
    -------
    dict : {cue_id: (start_cm, end_cm)} or {cue_id: [(start, end), ...]}
    """
    trials = trial_df.filter(pl.col('trial_type') == trial_type)

    if len(trials) == 0:
        print(f"Warning: No trials found for trial type '{trial_type}'")
        return {}

    first_trial = trials.row(0, named=True)
    cues = np.array(first_trial['cue'])
    distances = np.array(first_trial['distance_normalized'])

    if verbose:
        print(f"\nExtracting cue regions for {trial_type} from data")
        print(f"Distance range: {distances.min():.1f} to {distances.max():.1f} cm")
        print(f"Unique cues: {np.unique(cues)}")

    # Detect cue transitions
    cue_changes = np.concatenate([[0], np.where(np.diff(cues) != 0)[0] + 1, [len(cues)]])

    cue_regions = {}
    for i in range(len(cue_changes) - 1):
        start_idx = cue_changes[i]
        end_idx = cue_changes[i + 1] - 1
        cue_id = cues[start_idx]

        if cue_id == 0:  # Skip gray regions
            continue

        start_dist = distances[start_idx]
        end_dist = distances[end_idx]

        if cue_id not in cue_regions:
            cue_regions[int(cue_id)] = []
        cue_regions[int(cue_id)].append((start_dist, end_dist))

    # Simplify single regions to tuples
    final_regions = {}
    for cue_id, regions in cue_regions.items():
        if len(regions) == 1:
            final_regions[cue_id] = regions[0]
            if verbose:
                s, e = regions[0]
                print(f"  Cue {cue_id}: {s:.1f} - {e:.1f} cm (width: {e - s:.1f} cm)")
        else:
            final_regions[cue_id] = regions
            if verbose:
                print(f"  Cue {cue_id}: {len(regions)} regions")
                for j, (s, e) in enumerate(regions):
                    print(f"    Region {j + 1}: {s:.1f} - {e:.1f} cm")

    return final_regions



# DIAGNOSTIC FUNCTIONS


def diagnose_data(data: TrialData):
    """Print diagnostic information about trial data."""
    print("\n" + "=" * 60)
    print("DATA DIAGNOSTIC")
    print("=" * 60)

    print(f"\nTotal trials: {len(data.trial_df)}")

    if data.config is not None:
        print(f"Animal: {data.config.animal_id}")
        print(f"Date: {data.config.date}")

    print("\nTrack lengths by trial type:")
    for trial_type in sorted(data.track_lengths.keys()):
        print(f"  {trial_type}: {data.track_lengths[trial_type]:.1f} cm")

    print("\nTrials by type:")
    for trial_type in data.trial_df['trial_type'].unique().sort():
        n_trials = len(data.trial_df.filter(pl.col('trial_type') == trial_type))
        n_rewarded = len(data.trial_df.filter(
            (pl.col('trial_type') == trial_type) & (pl.col('rewarded') == 1)
        ))
        print(f"  {trial_type}: {n_trials} trials ({n_rewarded} rewarded)")

    print("\nSession stats:")
    for trial_type, stats_obj in data.session_stats.items():
        n_bins = stats_obj['session_avg'].shape[0]
        n_cells = stats_obj['session_avg'].shape[1]
        print(f"  {trial_type}: {stats_obj['n_trials']} trials, {n_bins} bins, {n_cells} cells")

    print("\nFirst few trials:")
    display_cols = ['trial', 'trial_type', 'n_frames', 'duration_s',
                    'measured_track_length', 'track_length_cm', 'rewarded']
    display_cols = [c for c in display_cols if c in data.trial_df.columns]
    print(data.trial_df.select(display_cols).head(10))
    print("=" * 60)


def diagnose_cues(data: TrialData, trial_type: str = 'ABC'):
    """Detailed diagnostic of cue information for a trial type."""
    print(f"\n{'=' * 60}")
    print(f"CUE DIAGNOSTIC FOR {trial_type}")
    print(f"{'=' * 60}")

    trials = data.trial_df.filter(pl.col('trial_type') == trial_type)
    if len(trials) == 0:
        print(f"No trials found for {trial_type}")
        return

    first_trial = trials.row(0, named=True)
    cues = np.array(first_trial['cue'])
    distances = np.array(first_trial['distance_normalized'])

    print(f"\nTrial {first_trial['trial']}:")
    print(f"  Frames: {len(cues)}")
    print(f"  Distance: {distances.min():.1f} to {distances.max():.1f} cm")
    print(f"  Track length: {first_trial['track_length_cm']:.0f} cm")
    print(f"  Unique cues: {np.unique(cues)}")

    # Frames per cue
    print("\nFrames per cue:")
    for cue_id in sorted(np.unique(cues)):
        count = np.sum(cues == cue_id)
        pct = count / len(cues) * 100
        cue_mask = cues == cue_id
        if cue_mask.any():
            cue_dist = distances[cue_mask]
            print(f"  Cue {cue_id}: {count} frames ({pct:.1f}%), "
                  f"covers {cue_dist.max() - cue_dist.min():.1f} cm")

    # Extracted regions
    print("\nExtracted cue regions:")
    cue_regions = get_cue_regions(data.trial_df, trial_type, verbose=False)

    total_cue_distance = 0
    for cue_id in sorted(cue_regions.keys()):
        regions = cue_regions[cue_id]
        if isinstance(regions, tuple):
            start, end = regions
            width = end - start
            total_cue_distance += width
            print(f"  Cue {cue_id}: {start:.1f} - {end:.1f} cm (width: {width:.1f} cm)")
        else:
            for j, (start, end) in enumerate(regions):
                width = end - start
                total_cue_distance += width
                print(f"  Cue {cue_id} region {j + 1}: {start:.1f} - {end:.1f} cm")

    track_length = first_trial['track_length_cm']
    print(f"\nCue coverage: {total_cue_distance:.1f} / {track_length:.0f} cm "
          f"({total_cue_distance / track_length * 100:.1f}%)")
    print("=" * 60)


# PLOTTING

def plot_cue_regions_on_axis(
        ax: plt.Axes,
        cue_regions: dict,
        cue_colors: dict = None,
        cue_labels: dict = None,
        show_labels: bool = True,
        alpha: float = 0.15
):
    """
    Plot cue region backgrounds on a matplotlib axis.

    Handles both single regions (tuple) and multiple regions (list of tuples).
    """
    if cue_colors is None:
        cue_colors = DEFAULT_CUE_COLORS
    if cue_labels is None:
        cue_labels = DEFAULT_CUE_LABELS

    for cue_id, regions in cue_regions.items():
        color = cue_colors.get(cue_id, '#CCCCCC')

        # Normalize to list of tuples
        if isinstance(regions, tuple):
            regions_list = [regions]
        elif isinstance(regions, list):
            regions_list = regions
        else:
            continue

        for region_idx, region in enumerate(regions_list):
            if isinstance(region, tuple) and len(region) == 2:
                start, end = region
                ax.axvspan(start, end, alpha=alpha, color=color, zorder=1)

                # Label only first region
                if show_labels and region_idx == 0:
                    mid = (start + end) / 2
                    label = cue_labels.get(cue_id, f'Cue {cue_id}')
                    if len(regions_list) > 1:
                        label = f"{label} (×{len(regions_list)})"

                    ax.text(mid, 0.98, label,
                            ha='center', va='top', fontsize=9, fontweight='bold',
                            transform=ax.get_xaxis_transform(),
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor=color, alpha=0.6, edgecolor='none'))


def plot_single_cell_all_trials(
        data: TrialData,
        cell_idx: int,
        trial_type: str = 'ABC',
        bin_size_cm: int = 5,
        figsize: tuple = (12, 6),
        alpha_trials: float = 0.3,
        show_cues: bool = True,
) -> plt.Figure:
    """Plot all individual trials for a single cell plus session average."""
    cue_colors = data.config.cue_colors if data.config else DEFAULT_CUE_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    trials = data.trial_df.filter(pl.col('trial_type') == trial_type)

    if show_cues:
        cue_regions = get_cue_regions(data.trial_df, trial_type)
        plot_cue_regions_on_axis(ax, cue_regions, cue_colors, show_labels=True)

    # Plot individual trials
    for row in trials.iter_rows(named=True):
        cell_activity = row['binned_signals'][:, cell_idx]
        x = np.arange(len(cell_activity)) * bin_size_cm
        ax.plot(x, cell_activity, color='gray', linewidth=1, alpha=alpha_trials, zorder=2)

    # Plot session average
    avg = data.session_stats[trial_type]['session_avg'][:, cell_idx]
    sem = data.session_stats[trial_type]['session_sem'][:, cell_idx]
    x_avg = np.arange(len(avg)) * bin_size_cm

    ax.plot(x_avg, avg, color='#E63946', linewidth=3,
            label=f'Session average (n={len(trials)})', zorder=4)
    ax.fill_between(x_avg, avg - sem, avg + sem, color='#E63946', alpha=0.3, zorder=3)

    ax.set_xlabel('Distance (cm)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'Cell {cell_idx} - {trial_type} trials', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, axis='y', zorder=0)

    plt.tight_layout()
    return fig


def plot_single_cell_comparison(
        data: TrialData,
        cell_idx: int,
        bin_size_cm: int = 5,
        figsize: tuple = (14, 10),
        alpha_trials: float = 0.3,
        show_cues: bool = True
) -> plt.Figure:
    """Compare trial types for a single cell with all trials + averages."""
    cue_colors = data.config.cue_colors if data.config else DEFAULT_CUE_COLORS

    available_types = sorted(list(data.session_stats.keys()))
    if len(available_types) == 0:
        print("Error: No trial types in session_stats")
        return None

    if len(available_types) == 1:
        figsize = (14, 6)

    fig, axes = plt.subplots(len(available_types), 1, figsize=figsize,
                             sharex=False, squeeze=False)
    axes = axes.flatten()

    trial_colors = {'ABC': '#2E86AB', 'ABDC': '#A23B72'}
    avg_colors = {'ABC': '#0A4D68', 'ABDC': '#6B0848'}

    for ax, trial_type in zip(axes, available_types):
        trials = data.trial_df.filter(pl.col('trial_type') == trial_type)

        if len(trials) == 0:
            continue

        if show_cues:
            cue_regions = get_cue_regions(data.trial_df, trial_type)
            for cue_id, region in cue_regions.items():
                if isinstance(region, tuple):
                    start, end = region
                    ax.axvspan(start, end, alpha=0.15, color=cue_colors.get(cue_id, '#CCC'))

        # Individual trials
        trial_color = trial_colors.get(trial_type, '#2E86AB')
        for row in trials.iter_rows(named=True):
            cell_activity = row['binned_signals'][:, cell_idx]
            x = np.arange(len(cell_activity)) * bin_size_cm
            ax.plot(x, cell_activity, color=trial_color, linewidth=1,
                    alpha=alpha_trials, zorder=2)

        # Session average
        avg = data.session_stats[trial_type]['session_avg'][:, cell_idx]
        sem = data.session_stats[trial_type]['session_sem'][:, cell_idx]
        x_avg = np.arange(len(avg)) * bin_size_cm

        avg_color = avg_colors.get(trial_type, '#0A4D68')
        ax.plot(x_avg, avg, color=avg_color, linewidth=3.5,
                label=f'Average (n={len(trials)})', zorder=4)
        ax.fill_between(x_avg, avg - sem, avg + sem, color=avg_color, alpha=0.3, zorder=3)

        ax.set_ylabel('ΔF/F', fontsize=12)
        ax.set_title(f'{trial_type} trials', fontsize=12, fontweight='bold', loc='left')
        ax.legend(frameon=False, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, axis='y', zorder=0)

    axes[-1].set_xlabel('Distance (cm)', fontsize=12)

    title = f'Cell {cell_idx} - Individual trials + averages'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_split_view(
        data: TrialData,
        cell_idx: int,
        abc_length_cm: float = None,
        bin_size_cm: int = 5,
        figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Split view comparing ABC vs ABDC:
    Left: Compare ABC portions
    Right: D section

    Requires both ABC and ABDC trial types.
    """
    cue_colors = data.config.cue_colors if data.config else DEFAULT_CUE_COLORS
    cue_labels = data.config.cue_labels if data.config else DEFAULT_CUE_LABELS

    available_types = list(data.session_stats.keys())
    if 'ABC' not in available_types or 'ABDC' not in available_types:
        print(f"Split view requires both ABC and ABDC. Available: {available_types}")
        return plot_single_cell_comparison(data, cell_idx)

    if abc_length_cm is None:
        abc_length_cm = data.track_lengths.get('ABC', 180)

    abc_bins = int(abc_length_cm / bin_size_cm)

    fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[2.5, 1])

    abc_cues = get_cue_regions(data.trial_df, 'ABC')
    abdc_cues = get_cue_regions(data.trial_df, 'ABDC')

    # LEFT: ABC comparison
    ax1 = axes[0]

    for cue_id, region in abc_cues.items():
        if isinstance(region, tuple):
            start, end = region
            ax1.axvspan(start, end, alpha=0.2, color=cue_colors.get(cue_id, '#CCC'))

    # ABC trials
    abc_avg = data.session_stats['ABC']['session_avg'][:, cell_idx]
    abc_sem = data.session_stats['ABC']['session_sem'][:, cell_idx]
    abc_x = np.arange(len(abc_avg)) * bin_size_cm

    ax1.plot(abc_x, abc_avg, color='#2E86AB', linewidth=2.5,
             label='ABC trials', marker='o', markersize=3, markevery=5, zorder=3)
    ax1.fill_between(abc_x, abc_avg - abc_sem, abc_avg + abc_sem,
                     color='#2E86AB', alpha=0.2, zorder=2)

    # ABDC (ABC portion)
    abdc_full = data.session_stats['ABDC']['session_avg'][:, cell_idx]
    abdc_sem_full = data.session_stats['ABDC']['session_sem'][:, cell_idx]
    abdc_abc_portion = abdc_full[:abc_bins]
    abdc_abc_sem = abdc_sem_full[:abc_bins]
    abdc_abc_x = np.arange(len(abdc_abc_portion)) * bin_size_cm

    ax1.plot(abdc_abc_x, abdc_abc_portion, color='#A23B72', linewidth=2.5,
             label='ABDC (ABC portion)', marker='s', markersize=3, markevery=5, zorder=3)
    ax1.fill_between(abdc_abc_x, abdc_abc_portion - abdc_abc_sem,
                     abdc_abc_portion + abdc_abc_sem, color='#A23B72', alpha=0.2, zorder=2)

    ax1.set_xlabel('Distance (cm)', fontsize=12)
    ax1.set_ylabel('ΔF/F', fontsize=12)
    ax1.set_title('Shared ABC sections', fontsize=12, fontweight='bold')
    ax1.legend(frameon=False, fontsize=10, loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3, axis='y', zorder=0)

    # RIGHT: D section
    ax2 = axes[1]

    d_cues = {cue_id: (start - abc_length_cm, end - abc_length_cm)
              for cue_id, region in abdc_cues.items()
              if isinstance(region, tuple) and region[0] >= abc_length_cm
              for start, end in [region]}

    for cue_id, (start, end) in d_cues.items():
        ax2.axvspan(max(0, start), end, alpha=0.2,
                    color=cue_colors.get(cue_id, '#CCC'))

    abdc_d_portion = abdc_full[abc_bins:]
    abdc_d_sem = abdc_sem_full[abc_bins:]
    d_x = np.arange(len(abdc_d_portion)) * bin_size_cm

    ax2.plot(d_x, abdc_d_portion, color='#A23B72', linewidth=2.5,
             marker='s', markersize=4, zorder=3)
    ax2.fill_between(d_x, abdc_d_portion - abdc_d_sem,
                     abdc_d_portion + abdc_d_sem, color='#A23B72', alpha=0.2, zorder=2)

    ax2.set_xlabel('Distance in D (cm)', fontsize=12)
    ax2.set_ylabel('ΔF/F', fontsize=12)
    ax2.set_title('Extension D section', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3, axis='y', zorder=0)

    plt.suptitle(f'Cell {cell_idx} - Place field with cue regions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def quick_plot_cell(
        data: TrialData,
        cell_idx: int,
        plot_type: str = 'auto'
) -> Optional[plt.Figure]:
    """
    Quick plotting function for exploring cells.

    Parameters
    ----------
    plot_type : str
        'auto': Choose best plot based on available data
        'comparison': Show all trial types stacked
        'abc' or 'abdc': Just that trial type
        'split': Split view (requires both ABC and ABDC)
    """
    available_types = sorted(data.trial_df['trial_type'].unique().to_list())
    print(f"Available trial types: {available_types}")

    if plot_type == 'auto':
        plot_type = 'comparison' if len(available_types) >= 2 else available_types[0].lower()

    if plot_type == 'comparison':
        fig = plot_single_cell_comparison(data, cell_idx)
    elif plot_type == 'abc':
        if 'ABC' not in available_types:
            print(f"ABC not found. Available: {available_types}")
            return None
        fig = plot_single_cell_all_trials(data, cell_idx, 'ABC')
    elif plot_type == 'abdc':
        if 'ABDC' not in available_types:
            print(f"ABDC not found. Available: {available_types}")
            return None
        fig = plot_single_cell_all_trials(data, cell_idx, 'ABDC')
    elif plot_type == 'split':
        fig = plot_split_view(data, cell_idx)
    else:
        print(f"Unknown plot_type: {plot_type}")
        print("Options: 'auto', 'comparison', 'abc', 'abdc', 'split'")
        return None

    plt.show()
    return fig





if __name__ == "__main__":
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    behavior_df = pl.read_ipc(session_root / '2025-09-03-17-02-46-836208.feather')

    # Run pipeline - returns TrialData container
    data = full_pipeline(
        behavior_df,
        signal_col='single_day_f',
        bin_size_cm=5
    )

    # Access components
    # data.trial_df        - the processed dataframe
    # data.session_stats   - averages per trial type
    # data.track_lengths   - nominal lengths
    # data.original_df     - filtered frame-level data
    # data.config          - session metadata

    # Diagnostics
    diagnose_data(data)

    for trial_type in data.track_lengths.keys():
        diagnose_cues(data, trial_type)

    # Example plots
    print("\nGenerating example plots...")
    for i in range(0, 5):
        quick_plot_cell(data, cell_idx=i, plot_type='comparison')