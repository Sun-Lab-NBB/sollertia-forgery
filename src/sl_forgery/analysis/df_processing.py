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


# LOAD CONFIG FILE
def load_experiment_config(yaml_path: Path) -> dict:
    """Load experiment configuration from YAML file.

    Args:
        yaml_path: Path to the experiment configuration YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# NAMING/VALIDATION HELPERS
def get_session_prefix(session_data: dict) -> str:
    """Build filename prefix: {animal_id}_{session_date}

    """
    animal_id = session_data['animal_id']
    session_date = session_data['session_name'][:10]  # '2025-09-15'
    return f"{animal_id}_{session_date}"


def _validate_session_date(session_data: dict, behavior_filename: str):
    """Check that behavior.feather filename date matches session_data.yaml date.

    """
    expected_date = session_data['session_name'][:10]  # '2025-09-15'
    file_date = Path(behavior_filename).stem[:10]      # '2025-09-15' from feather name
    if expected_date != file_date:
        raise ValueError(
            f"Date mismatch: session_data says {expected_date}, "
            f"behavior file says {file_date}"
        )

#
def find_session_dir(mouse_dir: Path, date: str) -> Path:
    """Find the session directory matching a date.

    Expected layout:
        mouse_dir/
          {session_name}/     <-- session_name starts with date string

    Args:
        mouse_dir: Mouse-level directory (e.g., datasets/26/)
        date: Session date (e.g., '2025-09-15')

    Returns:
        Path to the session directory.
    """
    mouse_dir = Path(mouse_dir)
    matches = sorted([
        d for d in mouse_dir.iterdir()
        if d.is_dir() and d.name.startswith(date)
    ])
    if len(matches) == 0:
        raise FileNotFoundError(f"No session folder starting with '{date}' in {mouse_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple sessions for '{date}': {[d.name for d in matches]}"
        )
    return matches[0]


def load_session_context(session_dir: Path) -> tuple[dict, dict]:
    """Load session_data.yaml and experiment_configuration.yaml from a session directory.

    Expected layout:
        session_dir/
          source_data/
            session_data.yaml
            experiment_configuration.yaml

    Args:
        session_dir: Path to the session directory.

    Returns:
        (session_data, experiment_config) dicts.
    """
    source_dir = Path(session_dir) / 'source_data'
    with open(source_dir / 'session_data.yaml', 'r') as f:
        session_data = yaml.safe_load(f)     # this is from the exp session, has experiment info (project, scene, etc)
    experiment_config = load_experiment_config(source_dir / 'experiment_configuration.yaml')  # exp config file, unity

    return session_data, experiment_config


def get_session_paths(session_dir: Path, session_data: dict) -> dict:
    """Get all relevant file paths for a session. Single source of truth for naming conventions.

    Finds the raw .feather file (expects exactly one) and builds the processed output paths
    using the {animal_id}_{date}_processed naming convention.

    Args:
        session_dir: Path to the session directory.
        session_data: Parsed session_data.yaml dict (needed for animal_id and date prefix).

    Returns:
        dict with keys: 'session_dir', 'feather', 'parquet', 'metadata'
        where each value is a Path to that file (or directory).
    """
    session_dir = Path(session_dir)
    prefix = get_session_prefix(session_data)

    # find the raw feather
    feather_files = sorted(session_dir.glob('*.feather'))
    if len(feather_files) != 1:
        raise FileNotFoundError(
            f"Expected 1 feather file in {session_dir}, found {len(feather_files)}"
        )
    _validate_session_date(session_data, feather_files[0].name)

    return {
        'session_dir': session_dir,
        'feather': feather_files[0],
        'parquet': session_dir / f'{prefix}_processed.parquet',
        'metadata': session_dir / f'{prefix}_processed.yaml',
    }

# CONFIGURATIONS
def get_track_length(config: dict,
                     trial_type: str) -> float | None:
    """Get track length for a trial type from experiment config."""

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

    Returns:
        regions: {cue_id: [(start_cm, end_cm), ...]}
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
        width = cue_widths[cue_id]

        if cue_id not in regions:
            regions[cue_id] = [(position, position + width)]
        else:
            regions[cue_id].append((position, position + width))

        position += width

    return regions


def get_bin_size(
    df: pl.DataFrame,
    metadata: dict | None = None,
) -> int | None:
    """Get spatial bin size in cm from metadata or derive from DataFrame.

    Args:
        df: Frame-level DataFrame with distance_bin and position columns.
        metadata: Optional metadata dict with 'bin_size_cm' key.

    Returns:
        Bin size in cm, or None if distance_bin column doesn't exist. Which I dont think is ever the case? It's just
        populated with 'null's
    """
    if metadata and 'bin_size_cm' in metadata:
        return metadata['bin_size_cm']

    if 'distance_bin' not in df.columns:
        return None

    # Get median position at bin 0 vs bin 1 within a single trial
    first_trial = df['trial'].first()
    sub = df.filter(
        (pl.col('trial') == first_trial)
        & (pl.col('distance_bin').is_in([0, 1]))
    ).group_by('distance_bin').agg(
        pl.col('position').median()
    ).sort('distance_bin')

    if len(sub) < 2:
        return None

    return round(sub['position'][1] - sub['position'][0])


def _infer_bin_size(mouse_dir: Path, default: int = 5) -> int:
    """
    Infer bin_size_cm from existing processed session metadata in this mouse directory.
    Searches for .yaml files and reads bin_size_cm from the most recent one. This is used during automatic
    session processing, to ensure that bin size stays consistent across days.

    Args:
        mouse_dir: Path, mouse-level directory
        default: int, fallback if no metadata found

    Returns:
        bin_size_cm: int
    """
    meta_files = sorted(mouse_dir.rglob('*_processed.yaml'), reverse=True)
    for mf in meta_files:
        with open(mf, 'r') as f:
            meta = yaml.safe_load(f)
        if meta and 'bin_size_cm' in meta:
            return meta['bin_size_cm']
    return default


def ensure_processed(
    mouse_dir: Path,
    date: str,
    auto_process: bool = False,
) -> tuple[Path, dict, dict]:
    '''
    Ensure a processed parquet exists for this session. If not, prompt user to process or skip.
    Bin size is inferred from existing session metadata in the mouse directory, defaulting to 5cm.

    Args:
        mouse_dir: Path, mouse-level directory
        date: str, session date (e.g., '2025-09-15')
        auto_process: bool, False shows a prompt to autoprocess, if True skips the prompt and process automatically

    Returns:
        parquet_path: Path, path to processed parquet file
        session_data: dict, from session_data.yaml
        config: dict, from experiment_configuration.yaml

    Raises:
        FileNotFoundError: if user declines to process
    '''
    session_dir = find_session_dir(mouse_dir, date)
    session_data, config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)

    if not paths['parquet'].exists():
        bin_size_cm = _infer_bin_size(mouse_dir)
        print(f"\n  No processed file found for {date}.")
        print(f"  Raw feather: {paths['feather'].name}")
        print(f"  Bin size (inferred): {bin_size_cm}cm")

        if not auto_process:
            response = input(f"  Process now? [y/n]: ").strip().lower()
            if response != 'y':
                raise FileNotFoundError(f"User skipped processing for {date}")

        print(f"  Processing {date}...")
        behavior_df = pl.read_ipc(paths['feather'])
        data, metadata = process_session(behavior_df, config, bin_size_cm=bin_size_cm)
        save_processed_session(data, session_dir, session_data, metadata)

    return paths['parquet'], session_data, config


# PROCESSING

def fix_cue_offset(
    df: pl.DataFrame,
    config: dict,
    system_state: str = 'run',
) -> pl.DataFrame:
    """
    Reassign frames across trial boundaries to correct for cue offset. Realigns to cue values instead, location is
    preserved. Required.
    
    The VR starts 10cm into the track, so trial boundaries in the data
    don't align with visual cue positions. This function:
    1. Identifies frames in the 0-10cm physical zone (end of recorded trial) using the cue col information
    2. Reassigns them to the next trial
    3. Drops first trial (incomplete) and last trial (incomplete)
    
    Args:
        df: Frame-based dataframe with cumulative distance
        config: Experiment configuration.yaml with trial_structures and cue offset information
        system_state: Filter to this system state (default 'run')
    
    Returns:
        active_df: Frame-based dataframe with corrected trial #, trial type, and guided values
    """
    # Get offset
    cue_offset_cm = config.get('cue_offset_cm', 0.0)

    # Keep all states
    if system_state == None:
        system_state = 'run'  ## THIS IS WHAT NEEDS TO EB CHANGED

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

    Args:
        df: Frame-based dataframe after fix_cue_offset (must have 'trial', 'trial_type', 'distance_cm')
        config: Experiment config with trial structures and track lengths (for consistent bin counts)
        bin_size_cm: Spatial bin size in cm
    
    Returns:
        df: Input df with position, distance_bin, and nominal_track_length columns added
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


def add_cue_zone_id(
        df: pl.DataFrame,
        cue_names: list[int, str] | None = None,
) -> pl.DataFrame:
    """
    Add string-based cue zone identifiers.

    Maps numeric cues to letters (1→A, 2→B, etc.) and labels gray zones (cue 0)
    by the preceding cue's letter lowercase (0 after A → "0a"). Can be modified to use different cue names for a
    different task (like 'star', 'triangle', etc; gray zones become 0star, 0triangle).
    Can change this in the future if the whole thing isn't working (or we want the names to be gray1, gray2, etc)

    Args:
        df: Frame-based dataframe with 'cue' column

    Returns:
        df with new 'cue_id' column

    Raises:
        ValueError: If a cue value in the data has no corresponding name
    """
    if cue_names is None:
        cue_names = [chr(64 + i) for i in range(1, 27)]  # A-Z

    # Build map: cue 1 → cue_names[0], cue 2 → cue_names[1], etc.
    cue_to_name = {i + 1: name for i, name in enumerate(cue_names)}

    # Check that all non-zero cues in data have a mapping
    cues_in_data = set(df.filter(pl.col("cue") != 0)["cue"].unique().to_list())
    missing = cues_in_data - set(cue_to_name.keys())
    if missing:
        raise ValueError(f"No names provided for cue(s): {sorted(missing)}")

    cues = df["cue"].to_list()
    cue_ids = []
    prev_name = None

    for c in cues:
        if c != 0:
            prev_name = cue_to_name[c]
            cue_ids.append(prev_name)
        else:
            cue_ids.append(f"0{prev_name.lower()}")

    return df.with_columns(pl.Series("cue_id", cue_ids))


# AVERAGING

def compute_binned_average(
    df: pl.DataFrame,
    signal_col: str,
    cell_idx: int,
    group_cols: list[str] | None,
) -> pl.DataFrame:
    """
    Single-cell binned average using Polars. Fast — uses list.get().

    Args:
        df: Frame-level df with signal_col and distance_bin columns
        signal_col: Column containing neural signals (list of floats per frame)
        cell_idx: Which cell to extract
        group_cols: Columns to group by (default: ['trial', 'distance_bin'])

    Returns:
        Grouped df with group_cols + 'mean_signal' + 'trial_type' columns
    """
    if group_cols is None:
        group_cols = ['trial', 'distance_bin']

    return (
        df.group_by(group_cols, maintain_order=True)
        .agg(
            pl.col(signal_col).arr.get(cell_idx).mean().alias('mean_signal'),
            pl.col('trial_type').first(),
        )
    )


def compute_session_averages(
        df: pl.DataFrame,
        signal_col: str,
        config: dict,
        bin_size_cm: int = 5,
        by_trial_type: bool = True,
) -> dict:
    """
    Compute session-level spatial averages from frame-level data. For each cell at each spatial bin, averages across trials.
    Result: one smoothed tuning curve per cell per trial type.

    **Extracts all signals into a numpy matrix, accumulates into a
    (n_trials, n_bins, n_cells) array using scatter-add, then averages.

    Args:
        df: pl.DataFrame, frame-level df with position and distance_bin columns
        signal_col: str, column containing neural signals (list per frame)
        config: dict, experiment config (for track lengths / bin counts)
        bin_size_cm: int, spatial bin size in cm
        by_trial_type: bool, if True split by trial type, else average all

    Returns:
        results: dict, per trial_type (or 'all'):
            {'session_avg': (n_bins, n_cells),
             'session_sem': (n_bins, n_cells),
             'n_trials': int}
    """
    from scipy import stats

    all_signals = np.vstack(df[signal_col].to_list())  # (n_frames, n_cells)
    n_cells = all_signals.shape[1]
    trials = df['trial'].to_numpy()
    bins = df['distance_bin'].to_numpy()
    trial_types = df['trial_type'].to_numpy()


    def _avg_for_subset(mask, n_bins):
        '''
        Build (n_trials, n_bins, n_cells) from frames using scatter-add (np.add.at()), then average across trials

        Args:
            mask: np.ndarray (bool), which frames to include
            n_bins: int, spatial bins for this track type

        Returns:
            dict with session_avg, session_sem, n_trials
        '''
        sub_signals = all_signals[mask]
        sub_trials = trials[mask]
        sub_bins = bins[mask]

        unique_trials = np.unique(sub_trials)
        n_trials = len(unique_trials)
        trial_idx = np.searchsorted(unique_trials, sub_trials)
        bin_idx = sub_bins.clip(0, n_bins - 1)

        # Accumulate sums and counts per (trial, bin)
        sums = np.zeros((n_trials, n_bins, n_cells))
        counts = np.zeros((n_trials, n_bins, 1))
        np.add.at(sums, (trial_idx, bin_idx), sub_signals)
        np.add.at(counts, (trial_idx, bin_idx, 0), 1)

        # Per-trial bin averages: (n_trials, n_bins, n_cells)
        with np.errstate(invalid='ignore'):
            per_trial = sums / counts

        # Peak amplitude per trial for this track type: (n_trials, n_cells)
        trial_peaks = np.nanmax(per_trial, axis=1)

        # Average across trials: (n_bins, n_cells)
        return {
            'session_avg': np.nanmean(per_trial, axis=0),
            'session_sem': np.nanstd(per_trial, axis=0, ddof=1) / np.sqrt(n_trials),
            'n_trials': n_trials,
            'per_trial_max': np.nanmax(per_trial, axis=0),  # (n_bins, n_cells), used for setting ylims in plotting
            'per_trial_q95': np.nanpercentile(per_trial, 95, axis=0),  # (n_bins, n_cells),
            'trial_peaks': trial_peaks,  # (n_trials, n_cells)
        }

    if not by_trial_type:
        max_length = max(
            get_track_length(config, tt)
            for tt in df['trial_type'].unique().to_list()
        )
        return {'all': _avg_for_subset(np.ones(len(df), dtype=bool), int(max_length / bin_size_cm))}

    results = {}
    for tt in sorted(np.unique(trial_types)):
        mask = trial_types == tt
        n_bins = int(get_track_length(config, tt) / bin_size_cm)
        results[tt] = _avg_for_subset(mask, n_bins)

    return results


# MAIN PIPELINE

def process_session(
    df: pl.DataFrame,
    exp_config: dict,
    bin_size_cm: int | None = 5,
    system_state: str = 'run',
) -> tuple[pl.DataFrame, dict]:
    """
    Complete pipeline: raw frame df -> corrected, binned frame df. Use in analysis by filtering signals and binning
    Signal columns are untouched — specify which to use at analysis time. Examples:
        - Single cell:  compute_binned_average(df, signal_col='single_day_f', cell_idx=0)
        - All cells:    compute_session_averages(df, signal_col='single_day_spikes', config=config)
        - UMAP:         np.vstack(df['single_day_dff'].to_list())

    
    Args:
        df: Raw frame-based dataframe
        exp_config: Experiment configuration (.yaml file)
        bin_size_cm: Spatial bin size if binning is used, default is 5 cm; if None, no binning will be applied
        system_state: Filter to this system state
    
    Returns:
        results: Corrected frame-level df with position and distance_bin columns
        metadata: Information about what was modified in the new df
    """
    print("Step 1: Fixing cue offset...")       #this could be an optional argument if we don't want to do this
    corrected_df = fix_cue_offset(df, exp_config, system_state=system_state)
    
    print("\nStep 2: Normalizing position...")
    result = add_position_and_bins(corrected_df, exp_config, bin_size_cm=bin_size_cm)

    print("\nStep 3: Adding additional columns...")  # may want more than cue ids in the future, add here
    result = add_cue_zone_id(result)

    n_trials = result['trial'].n_unique()
    trial_types = result['trial_type'].unique().to_list()
    print(f"Done. {n_trials} trials ({trial_types}), {len(result)} frames.")

    #metadata for the new df, including bin size
    metadata = {
        'n_frames': len(result),
        'n_trials': result['trial'].n_unique(),
        'trial_types': result['trial_type'].unique().sort().to_list(),
        'bin_size_cm': bin_size_cm,
        'cue_offset_cm': exp_config.get('cue_offset_cm', 0.0),
        'system_state': system_state,
        'columns': result.columns,
    }

    return result, metadata


# SAVE/LOAD

def save_processed_session(
        data: pl.DataFrame,
        output_path: Path,
        session_data: dict,
        metadata: dict):
    """Save processed df as {animal_id}_{date}_processed.parquet + .meta.yaml."""

    output_path = Path(output_path)
    prefix = get_session_prefix(session_data)
    parquet_path = output_path / f'{prefix}_processed.parquet'

    data.write_parquet(parquet_path)

    # Merge session info into metadata (so now metadata from the processing adn the actual session are in one file"
    metadata['animal_id'] = session_data['animal_id']
    metadata['session_name'] = session_data['session_name']
    metadata['project_name'] = session_data.get('project_name')

    metadata_path = output_path / f'{prefix}_processed.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"Saved: {parquet_path}")
    print(f"Saved:  {metadata_path}")


def load_processed_session(path: Path) -> tuple[pl.DataFrame, dict | None]:
    """Load processed data from parquet, and metadata from yaml"""

    path = Path(path)
    if path.suffix != '.parquet':
        path = path.with_suffix('.parquet')

    df = pl.read_parquet(path)

    meta_path = path.with_suffix('.yaml')
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)

    return df, metadata


def load_multiday_sessions(
    mouse_dir: Path,
    dates: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    auto_process: bool = False,
) -> dict[str, dict]:
    '''
    Load multiple processed sessions for cross-day comparison.

    Args:
        mouse_dir: Path, mouse-level directory (e.g., 26_explore/)
        dates: list[str] | None, explicit session dates (e.g., ['2025-09-10', '2025-09-11'])
        date_range: tuple[str, str] | None, inclusive range (e.g., ('2025-09-15', '2025-09-24')).
            Auto-discovers all session folders whose date falls within the range.
            Provide either dates or date_range, not both.
        auto_process: bool, choose if you want to automatically process (cue-offset) the data is it's not found

    Returns:
        sessions: dict[str, dict], keyed by date string (sorted chronologically), each containing:
            {'data': pl.DataFrame, 'config': dict, 'session_data': dict, 'metadata': dict | None}
            Skips dates that fail to load (prints warning).
    '''
    mouse_dir = Path(mouse_dir)

    if dates is not None and date_range is not None:
        raise ValueError("Provide either dates or date_range, not both.")

    if date_range is not None:
        start, end = date_range
        # Discover session folders whose name starts with a date in range
        dates = sorted(
            d.name[:10]
            for d in mouse_dir.iterdir()
            if d.is_dir() and len(d.name) >= 10 and start <= d.name[:10] <= end
        )
        # Deduplicate (multiple folders same date would be caught by load_session_dir)
        dates = sorted(set(dates))
        print(f"Found {len(dates)} sessions in range {start} to {end}: {dates}")

    if dates is None or len(dates) == 0:
        print("No dates provided or found.")
        return {}

    sessions = {}

    for date in sorted(dates):
        try:
            parquet_path, session_data, config = ensure_processed(mouse_dir, date, auto_process=auto_process)
            data, metadata = load_processed_session(parquet_path)

            sessions[date] = {
                'data': data,
                'config': config,
                'session_data': session_data,
                'metadata': metadata,
            }
            print(f"  Loaded {date}: {len(data)} frames, "
                  f"{data['trial'].n_unique()} trials, "
                  f"types={sorted(data['trial_type'].unique().to_list())}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: Could not load {date}: {e}")

    print(f"Loaded {len(sessions)}/{len(dates)} sessions.")
    return sessions


if __name__ == "__main__":
    #load all the data
    mouse_id = '26'
    date = '2025-08-21'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, experiment_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)

    behavior_df = pl.read_ipc(paths['feather'])

    #process and save the offset-corrected df
    processed_df, meta = process_session(behavior_df, experiment_config)
    save_processed_session(processed_df, session_dir, session_data, meta)  #.parent gets session folder

    #load it back to check
    frame_df, meta = load_processed_session(paths['parquet'])

    # #check
    # print(frame_df.columns)
    # with pl.Config(tbl_cols=100, tbl_rows=100, set_tbl_hide_dataframe_shape=False):
    #     print(frame_df.head(100))

