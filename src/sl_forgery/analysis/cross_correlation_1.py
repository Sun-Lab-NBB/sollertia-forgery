"""
Cross-correlation analysis for place cell spatial tuning curves

This module implements standard population-level and single-cell analyses for
comparing spatial representations across trial types (within-session) and across
recording days (multiday). It is designed for hippocampal / cortical calcium-imaging
data recorded on a virtual-reality linear track with interleaved trial types
(e.g. ABC vs ABDC cue sequences), but could be expanded to handle different tasks

Within-session analyses:
    - Split-half reliability: odd vs even trial tuning curves per cell (sanity check
      for stable spatial tuning within a session).
    - Cross-track population vector (PV) correlation: at each spatial bin, correlate
      the population activity vector between two trial types to ask where along the
      track the representations diverge.
    - Per-cell spatial correlation: Pearson r of each cell's tuning curve between
      two conditions — tests how individual neurons remap.
    - PV correlation matrix: full bin × bin heatmap (off-diagonal structure reveals
      whether spatial representations stretch, compress, or globally remap).
    - Within-session learning curve: trial-by-trial leave-one-out PV correlation to
      own-type vs other-type templates — tracks gradual emergence of discrimination.

Across-session (multiday) analyses:
    - Per-cell tuning curve correlation between any two days, or same-day split-half.
    - PV correlation across days at each spatial bin.
    - Full pairwise day × day summary matrix (diagonal = autocorrelation via
      split-half; off-diagonal = cross-day stability).

Standard analyses following Leutgeb et al. 2005, Colgin et al. 2008, Sun et al. 2025.
"""

from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from df_processing import (compute_session_averages, get_track_length, get_cue_regions, get_bin_size)
import plot_utils as pfmt

# TODO
#  multiday pv corr uses split half but single day doesnt.
#  multiple dates make a million plots; not sure if auto is so informative

# TUNING CURVE EXTRACTION

def get_mean_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
) -> dict[str, np.ndarray]:
    """Compute session-averaged tuning curves per cell, per trial type.

    Wrapper around compute_session_averages (from df_processing) that returns just the
    mean tuning curves in (n_bins, n_cells) format needed for correlation functions.

    The averaging pipeline bins each frame by `distance_bin`, computes per-trial
    mean activity in each bin, then averages across trials for each trial type.
    This two-stage mean (within-trial → across-trial) avoids biasing toward trials
    with more frames per bin (i.e. slower running speed).

    Args:
        df: Frame-level DataFrame with position, bin columns, trial type, and signal
        config: Experiment configuration dict (track structure, cue map, etc)
        metadata: Pipeline metadata from df_processing(), has bin size in cm
        signal_col: Column containing neural signals

    Returns:
        Dict mapping trial_type -> (n_bins, n_cells) array w session-averaged tuning curve
    """
    bin_size_cm = get_bin_size(metadata)

    # compute_session_averages returns a nested dict: {trial_type: {'session_avg': ..., ...}}
    stats = compute_session_averages(
        df, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
    )
    # Extract only the session-averaged tuning curves, discarding per-trial stats
    return {tt: s['session_avg'] for tt, s in stats.items()}


def get_split_half_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> tuple[np.ndarray, np.ndarray]:
    """Split trials into odd/even halves and compute independent mean tuning curves.

    Used for split-half reliability (within-session autocorrelation) and as the
    same-day diagonal entry in multiday matrices.  Odd/even splitting is preferred
    over first-half/second-half because it is robust to slow drifts in neural
    activity or behavioral engagement across a session.

    Implementation uses scatter-add binning (np.add.at) to accumulate activity
    per (trial, bin) pair, then divides by frame counts and averages across trials.
    This is equivalent to compute_session_averages() but applied to a trial subset.

    Args:
        df: Frame-level DataFrame
        config: Experiment configuration dict — needed to look up track length.
        metadata: From df_processings, has bin size in cm
        trial_type: Which trial type to split.
        signal_col: Column containing neural signals.

    Returns:
        Tuple of (even_avg, odd_avg), each (n_bins, n_cells).
    """
    tt_df = df.filter(pl.col('trial_type') == trial_type)
    trials = tt_df['trial'].to_numpy()
    unique_trials = np.unique(trials)

    # Interleaved split: indices 0, 2, 4, ... → even; 1, 3, 5, ... → odd
    even_trials = unique_trials[::2]
    odd_trials = unique_trials[1::2]

    bin_size_cm = get_bin_size(metadata)

    # Determine array dimensions from config and data
    n_bins = int(get_track_length(config, trial_type) / bin_size_cm)
    n_cells = len(df[signal_col][0])


    def _avg_for_trials(trial_set):
        """Bin and average signals for a subset of trials.

        Uses scatter-add (np.add.at) for efficient accumulation without
        explicit loops over trials or bins. Steps:
            1. Filter frames to the requested trial subset.
            2. Map each frame to (trial_index, bin_index) coordinates.
            3. Accumulate signal sums and frame counts into a 3-D array.
            4. Divide to get per-trial bin means, then average across trials.

        Args:
            trial_set: 1-D array of trial numbers to include.

        Returns:
            ndarray of shape (n_bins, n_cells) — mean tuning curve over the subset."""
        mask_pl = pl.col('trial').is_in(trial_set.tolist())
        sub = tt_df.filter(mask_pl)

        # Stack per-frame signal vectors into (n_frames, n_cells)
        signals = np.vstack(sub[signal_col].to_list())
        sub_trials = sub['trial'].to_numpy()
        sub_bins = sub['distance_bin'].to_numpy()

        u_trials = np.unique(sub_trials)
        n_t = len(u_trials)
        trial_idx = np.searchsorted(u_trials, sub_trials)
    # TODO do we actuall want this clipping?
        bin_idx = sub_bins.clip(0, n_bins - 1)

        sums = np.zeros((n_t, n_bins, n_cells))
        counts = np.zeros((n_t, n_bins, 1))
        # Scatter-add: each frame's signal is added to the correct (trial, bin) slot
        np.add.at(sums, (trial_idx, bin_idx), signals)
        np.add.at(counts, (trial_idx, bin_idx, 0), 1)

# TODO check if this is ever the case; potentially if the mouse is running faster than the framerate but unlikely
        # Per-trial bin means; 0/0 → NaN (bins the mouse never visited on a given trial)
        with np.errstate(invalid='ignore'):
            per_trial = sums / counts
        #average across trials
        return np.nanmean(per_trial, axis=0)  # (n_bins, n_cells)

    return _avg_for_trials(even_trials), _avg_for_trials(odd_trials)


# CORE ANALYSES

def per_cell_spatial_correlation(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
    min_bins: int | None = None,
) -> np.ndarray:
    """Pearson correlation of each cell's tuning curve between two conditions (i.e. trial types).

    For each neuron independently, this correlates its mean firing-rate-by-position
    profile in condition A against condition B. This is the standard "rate remapping"
    metric: a cell with r ≈ 1 fires in the same place with the same relative rates
    in both conditions; r ≈ 0 means it remapped.

    Cells whose tuning curve has zero variance in either condition (e.g. silent cells
    or cells active in only one bin) receive NaN — they carry no spatial information
    for this comparison.

    Args:
        tuning_a: Mean tuning curves, shape (n_bins, n_cells).
        tuning_b: Mean tuning curves, shape (n_bins, n_cells).
        min_bins: Restrict to first min_bins bins (for shared-segment comparison).

    Returns:
        Per-cell Pearson r, shape (n_cells,). NaN for cells with zero variance.
    """
    # Optional truncation to shared spatial segment
    if min_bins is not None:
        tuning_a = tuning_a[:min_bins]
        tuning_b = tuning_b[:min_bins]

    # Align to the shorter array
    n_bins = min(tuning_a.shape[0], tuning_b.shape[0])
    a = tuning_a[:n_bins]
    b = tuning_b[:n_bins]
    n_cells = a.shape[1]

    # Boolean mask: True where both conditions have finite data at that bin
    valid = ~(np.isnan(a) | np.isnan(b))  # (n_bins, n_cells)
    valid_count = valid.sum(axis=0)

    corrs = np.full(n_cells, np.nan)
    # Only compute for cells with enough valid bins to estimate a correlation (3 is minimum for pearson, could do 5)
    for c in np.where(valid_count >= 3)[0]:
        m = valid[:, c]
        ac, bc = a[m, c], b[m, c]
        # Skip zero-variance cells (no spatial modulation in one condition)
        if np.std(ac) == 0 or np.std(bc) == 0:
            continue
        corrs[c] = np.corrcoef(ac, bc)[0, 1]
    return corrs


def population_vector_correlation(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
    min_bins: int | None = None,
) -> np.ndarray:
    """Population vector (PV) correlation at each spatial bin between two conditions.

    At each position along the track, this takes the vector of all cells' mean
    activity and computes Pearson r between the two conditions. High PV correlation
    at a bin means the population code at that location is similar across conditions;
    a drop in PV correlation indicates local remapping.

    This is the complement to ``per_cell_spatial_correlation``: PV correlation asks
    "does the population look the same at this position?", while per-cell correlation
    asks "does this cell fire in the same places?"

    Args:
        tuning_a: Shape (n_bins, n_cells), session-averaged tuning curves for condition A
        tuning_b: Shape (n_bins, n_cells), session-averaged tuning curves for condition B
        min_bins: Restrict to first min_bins bins (shared-segment comparison)

    Returns:
        Pearson r at each spatial bin, shape (n_shared_bins,). NaN where undefined or <3 valid cells
    """
    if min_bins is not None:
        tuning_a = tuning_a[:min_bins]
        tuning_b = tuning_b[:min_bins]

    n_bins = min(tuning_a.shape[0], tuning_b.shape[0])
    a = tuning_a[:n_bins]
    b = tuning_b[:n_bins]

    pv_corrs = np.full(n_bins, np.nan)
    for i in range(n_bins):
        # Mask out cells with NaN at this bin in either condition
        valid = ~(np.isnan(a[i]) | np.isnan(b[i]))
        if valid.sum() < 3:
            continue
        ai, bi = a[i, valid], b[i, valid]
        if np.std(ai) == 0 or np.std(bi) == 0:
            continue
        pv_corrs[i] = np.corrcoef(ai, bi)[0, 1]
    return pv_corrs


def pv_correlation_matrix(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
) -> np.ndarray:
    """Full bin-by-bin PV correlation matrix.

    Entry (i, j) = correlation of PV (Pearson r) at bin i in condition A
    with bin j in condition B. The diagonal of this matrix is the
    standard PV-by-position curve (same as ``population_vector_correlation``).
    Off-diagonal structure reveals spatial distortions: if the strongest
    correlation at row i is shifted from the diagonal, the representation at
    that location has shifted between conditions.

    This is a more information-rich version of PV correlation — the diagonal
    gives the same signal as ``population_vector_correlation``, but the full
    matrix also reveals stretching, compression, and non-monotonic remapping.

    Args:
        tuning_a: Shape (n_bins_a, n_cells).
        tuning_b: Shape (n_bins_b, n_cells).

    Returns:
        Correlation matrix, shape (n_bins_a, n_bins_b).
    """
    n_a = tuning_a.shape[0]
    n_b = tuning_b.shape[0]
    matrix = np.full((n_a, n_b), np.nan)

    for i in range(n_a):
        for j in range(n_b):
            valid = ~(np.isnan(tuning_a[i]) | np.isnan(tuning_b[j]))
            if valid.sum() < 3:
                continue
            ai, bj = tuning_a[i, valid], tuning_b[j, valid]
            if np.std(ai) == 0 or np.std(bj) == 0:
                continue
            matrix[i, j] = np.corrcoef(ai, bj)[0, 1]
    return matrix


# BIN ALIGNMENT

def get_shared_bins(
    config: dict,
    type_a: str,
    type_b: str,
    metadata: dict
) -> tuple[float,int]:
    """Identify the spatial point where two track types' cue sequences first differ.

    Walks through the ordered cue sequences of both trial types from position 0,
    accumulating cue widths as long as the cues match. The first mismatch defines
    the divergence point — the boundary between the shared segment (where sensory
    input is identical across trial types) and the divergent segment (where cues
    differ and any representational similarity reflects learned generalization
    rather than shared input).

    Returns both the raw position in cm (for plotting divergence lines) and the
    corresponding bin count (for slicing tuning curve arrays into shared vs.
    divergent segments).

    Args:
        config: Experiment configuration dict containing 'trial_structures' (cue sequences per
            trial type) and 'cue_map' (cue name → width in cm)
        type_a: First trial type
        type_b: Second trial type
        metadata: metadata from df_processing, use for bin size
    Returns:
         Tuple of:
            - divergence_cm: track position (cm) where cue sequences first differ
              Equal to the total length of the shared segment
            - n_shared_bins: number of spatial bins fully contained in the shared
              segment
    """
    ts = config.get('trial_structures', {})
    seq_a = ts.get(type_a, {}).get('cue_sequence', [])
    seq_b = ts.get(type_b, {}).get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    bin_size_cm = get_bin_size( metadata)

    # Walk through both sequences in parallel; stop at first mismatch
    shared_cm = 0.0
    for ca, cb in zip(seq_a, seq_b):
        if ca != cb:
            break
        shared_cm += cue_widths[ca]

    return shared_cm, int(shared_cm / bin_size_cm)


# WITHIN-SESSION PLOTTING

def plot_split_half(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
    figsize: tuple = (8, 5),
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
) -> Figure:
    """Split-half reliability histogram for one track type. Sanity check.
    Splits trials into odd/even for each cell, single trial type.
    High median r>.5 == stable spatial tuning within session. If the distribution is
    centered near zero, spatial coding is noisy or non-stationary.

    Per-cell Pearson r between odd and even trial tuning curves.

    Args:
        df: Frame-level DataFrame
        config: Experiment configuration dict
        trial_type: Which trial type to test
        metadata: metadata from df_processing, has bin size
        signal_col: Column containing neural signals
        figsize: Figure size
        animal_id: Which animal to plot
        date: Expeirment date
        show: Call plt.show()

    Returns:
        Matplotlib Figure
    """
    # Compute independent tuning curves from odd and even trials
    even_avg, odd_avg = get_split_half_tuning_curves(
        df, config, metadata, trial_type, signal_col=signal_col,
    )
    # Per-cell Pearson r between the two halves
    corrs = per_cell_spatial_correlation(even_avg, odd_avg)
    valid = corrs[~np.isnan(corrs)]

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(valid, bins=np.linspace(-1, 1, 41), color=color, alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(valid), color='black', linestyle='--', linewidth=1.5,
               label=f'median = {np.nanmedian(valid):.3f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Pearson r (odd vs even trials)', fontsize=11)
    ax.set_ylabel('Number of cells', fontsize=11)
    ax.set_title(pfmt.build_title('Split-Half Reliability', trial_type=trial_type,
                              animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)

    # Summary statistics annotation
    n_total = len(corrs)
    n_valid = len(valid)
    n_sig = np.sum(valid > 0.3)
    ax.text(0.02, 0.95,
            f'n = {n_valid}/{n_total} cells\n'
            f'r > 0.3: {n_sig} ({100 * n_sig / max(n_valid, 1):.0f}%)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_pv_correlation_across_position(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    figsize: tuple = (10, 5),
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
) -> Figure:
    """PV correlation at each shared spatial bin between two trial types.

    Shows where along the track the population representation is similar vs.
    different between conditions. For same-type comparisons (type_a == type_b),
    uses split-half tuning curves; otherwise uses full session averages.

    A vertical red dashed line marks the point where the two track types' cue
    sequences first diverge — PV correlation is expected to drop after this point
    if the population discriminates the two conditions.

    Cue-region shading and boundary ticks are overlaid from ``plot_utils`` for
    anatomical reference.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        metadata: metadata from df_processing, has bin size
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        figsize: Figure size.
        animal_id: Animal ID
        date: Expeirment date
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    bin_size_cm = get_bin_size(metadata)

    if type_a == type_b:
        avg_a, avg_b = get_split_half_tuning_curves(
            df, config, metadata, type_a, signal_col=signal_col,
        )
    else:
        avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
        avg_a, avg_b = avgs[type_a], avgs[type_b]

    diverge_cm, shared_bins = get_shared_bins(config, type_a, type_b, metadata)

    pv_shared = population_vector_correlation(avg_a, avg_b, min_bins=shared_bins)
    x_shared = np.arange(shared_bins) * bin_size_cm + bin_size_cm / 2

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_shared, pv_shared, color='black', linewidth=2, zorder=4)
    ax.fill_between(x_shared, pv_shared, alpha=0.15, color='black', zorder=3)

    if type_a != type_b:
        ax.axvline(diverge_cm, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Tracks diverge ({diverge_cm:.0f} cm)', zorder=5)

    if config:
        pfmt.add_cue_shading_with_labels(ax, config, type_a, alpha=0.1,
                                         max_cm=shared_bins * bin_size_cm)
        pfmt.set_cue_boundary_ticks(ax, config, type_a)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    title_desc = (f'PV Correlation — {type_a} split-half' if type_a == type_b
                  else f'PV Correlation — {type_a} vs {type_b}')
    ax.set_title(pfmt.build_title(title_desc,
                                  animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    ax.set_xlim(0, shared_bins * bin_size_cm)
    y_min = np.nanmin(pv_shared)
    y_max = np.nanmax(pv_shared)
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    if y_min > 0:
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    color = pfmt.TRIAL_TYPE_COLORS.get(type_a, 'black')
    mean_r = np.nanmean(pv_shared)
    ax.axhline(mean_r, color=color, linestyle='--', linewidth=1, alpha=0.5,
               label=f'mean r = {mean_r:.3f}')
    ax.legend(frameon=False, fontsize=10, loc='lower left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_per_cell_cross_correlation(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    segment: str = 'shared',
    figsize: tuple = (8, 5),
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
) -> Figure:
    """Histogram of per-cell spatial correlations between two trial types.

    Each cell's session-averaged tuning curve on type_a is correlated with its
    curve on type_b. The segment parameter controls whether the correlation
    uses only the shared (pre-divergence) spatial bins or the full shorter track.

    This complements the PV-by-position plot: PV correlation tells you where
    the representations differ; this histogram tells you how many cells remap.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        metadata: metadata from df_processing, has bin size
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        segment: 'shared' for shared bins only, 'full' for shorter track length.
        figsize: Figure size.
        animal_id: Animal ID for plot titles
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    bin_size_cm = get_bin_size(metadata)

    avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    min_bins = None
    if segment == 'shared':
        _, min_bins = get_shared_bins(config, type_a, type_b, metadata)

    corrs = per_cell_spatial_correlation(avg_a, avg_b, min_bins=min_bins)
    valid = corrs[~np.isnan(corrs)]

    seg_label = f'shared segment (0–{min_bins * bin_size_cm} cm)' if min_bins else 'full track'

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(valid, bins=np.linspace(-1, 1, 41), color='#555555', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(valid), color='black', linestyle='--', linewidth=1.5,
               label=f'median = {np.nanmedian(valid):.3f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f'Pearson r ({seg_label})', fontsize=11)
    ax.set_ylabel('Number of cells', fontsize=11)
    ax.set_title(pfmt.build_title(f'Per-Cell Correlation — {type_a} vs {type_b}',
                              animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)

    n_total = len(corrs)
    n_valid = len(valid)
    ax.text(0.02, 0.95,
            f'n = {n_valid}/{n_total} cells',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_pv_correlation_matrix(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    figsize: tuple = (8, 7),
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
) -> Figure:
    """Full bin×bin PV correlation heatmap between two track types.

    Delegates to pv_correlation_matrix for computation and pfmt.plot_pv_heatmap
    for rendering. The divergence point is passed so the heatmap can optionally mark
    where the tracks separate.

    On the diagonal, you see the same-position PV correlation (equivalent to
    population_vector_correlation). Off-diagonal structure shows whether
    one condition's representation at position X resembles the other condition's
    representation at a different position Y.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type
        metadata: Metadata dict from processed df, has bin size
        signal_col: Column containing neural signals.
        figsize: Figure size.
        animal_id: Animal ID for plot titles
        date: Experiment date.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    bin_size_cm = get_bin_size(metadata, df)

    avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
    matrix = pv_correlation_matrix(avgs[type_a], avgs[type_b])
    diverge, _ = get_shared_bins(config, type_a, type_b, metadata)

    fig, ax = plt.subplots(figsize=figsize)
    pfmt.plot_pv_heatmap(
        ax, matrix, config,
        x_type=type_a, y_type=type_b,
        x_label=f'{type_a} position (cm)',
        y_label=f'{type_b} position (cm)',
        bin_size_cm=bin_size_cm,
        diverge_cm=diverge,
    )
    ax.set_title(pfmt.build_title(f'PV Matrix — {type_a} vs {type_b}',
                                  animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    plt.tight_layout()
    if show:
        plt.show()

    return fig


# WITHIN-SESSION WRAPPER

def run_within_session_analysis(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Run full within-session cross-correlation analysis (single day)

    Generates:
        1. Split-half reliability histogram for each trial type.
        2. For each pair of trial types (including self-pairs):
            a. PV correlation across position.
            b. Per-cell cross-correlation histogram (cross-type pairs only).
            c. Full bin × bin PV correlation matrix.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        metadata: Metadata dict from processed df, has bin size
        signal_col: Column containing neural signals.
        animal_id: Animal ID.
        date: Experiment date.
        show: Call plt.show() for each figure.
        save_dir: Directory to save figures (optional).

    Returns:
        Dict of {name: Figure}.
    """
    trial_types = sorted(df['trial_type'].unique().to_list())

    figs = {}

    trial_counts = {
        tt: df.filter(pl.col('trial_type') == tt)['trial'].n_unique()
        for tt in trial_types
    }
    print(f"Trial types: {trial_types}")
    print(f"Trials per type: {trial_counts}\n")

    # Split-half reliability
    for tt in trial_types:
        n = trial_counts[tt]
        if n < 4:
            print(f"Skipping split-half for {tt}: only {n} trials (need ≥4)")
            continue
        print(f"Split-half reliability: {tt}...")
        fig = plot_split_half(df, config, tt, metadata, signal_col=signal_col,
                              animal_id=animal_id, date=date, show=show)
        figs[f'split_half_{tt}'] = fig

    # Cross-track comparisons
    if len(trial_types) >= 2:
        for type_a, type_b in combinations_with_replacement(trial_types, 2):
            print(f"\nCross-correlation: {type_a} vs {type_b}...")

            fig = plot_pv_correlation_across_position(
                df, config, metadata, type_a, type_b, signal_col=signal_col,
                animal_id=animal_id, date=date, show=show)
            figs[f'pv_position_{type_a}_vs_{type_b}'] = fig

            if type_a != type_b:
                fig = plot_per_cell_cross_correlation(
                    df, config, metadata, type_a, type_b, signal_col=signal_col,
                    segment='shared', animal_id=animal_id, date=date, show=show)
                figs[f'cell_corr_shared_{type_a}_vs_{type_b}'] = fig

            fig = plot_pv_correlation_matrix(
                df, config, metadata, type_a, type_b, signal_col=signal_col,
                animal_id=animal_id, date=date, show=show)
            figs[f'pv_matrix_{type_a}_vs_{type_b}'] = fig
    else:
        print("Only one trial type — skipping cross-track comparisons.")

    _save_figures(figs, save_dir)
    print(f"\nGenerated {len(figs)} figures.")
    return figs


#  MULTIDAY CORRELATIONS

def multiday_tuning_curves(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> dict[str, np.ndarray]:
    """Compute session-averaged tuning curves per cell for one trial type across days.

    Iterates over sessions (keyed by date string), calls get_mean_tuning_curves()
    for each, and collects results. Sessions where the requested trial type was
    not run are silently skipped with a console warning.

    Cell identity across days is assumed to be aligned via cross-day registration
    (e.g. Suite2P multi-session), so column indices in the returned arrays correspond
    to the same physical neurons.

    Args:
        sessions: From load_multiday_sessions(). Keys are date strings.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.

    Returns:
        Dict mapping date -> (n_bins, n_cells) tuning curve array.
        Dates missing the trial type are skipped.
    """
    result = {}
    for date, s in sorted(sessions.items()):
        df = s['data']
        config = s['config']

        tt_present = df['trial_type'].unique().to_list()
        if trial_type not in tt_present:
            print(f"  {date}: {trial_type} not present, skipping")
            continue

        avgs = get_mean_tuning_curves(df, config, s['metadata'], signal_col=signal_col)
        if trial_type in avgs:
            result[date] = avgs[trial_type]

    return result


def multiday_per_cell_correlation(
    sessions: dict[str, dict],
    day_x: str,
    day_y: str,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    cell_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Per-cell tuning curve correlation between two days.

    When day_x == day_y, computes split-half correlation (autocorrelation).
    When the days differ, it correlates full session-averaged tuning curves across days.

    Used to assess long-term stability of individual place fields

    Args:
        sessions: From load_multiday_sessions().
        day_x: First date string.
        day_y: Second date string.
        trial_type: Which trial type to compare.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells to compute. None = all cells.

    Returns:
        Per-cell Pearson r, shape (n_cells,) or (len(cell_indices),).
    """
    if day_x == day_y:
        # Autocorrelation: split-half within the same session
        s = sessions[day_x]
        even_avg, odd_avg = get_split_half_tuning_curves(
            s['data'], s['config'], s['metadata'], trial_type,
            signal_col=signal_col,
        )
        corrs = per_cell_spatial_correlation(even_avg, odd_avg)
    else:
        # Cross-day: full session averages
        tc = multiday_tuning_curves(
            {day_x: sessions[day_x], day_y: sessions[day_y]},
            trial_type, signal_col=signal_col,
        )
        if day_x not in tc or day_y not in tc:
            n_cells = len(sessions[day_x]['data'][signal_col][0])
            return np.full(n_cells, np.nan)
        corrs = per_cell_spatial_correlation(tc[day_x], tc[day_y])

    if cell_indices is not None:
        corrs = corrs[cell_indices]
    return corrs


def multiday_pv_correlation(
    sessions: dict[str, dict],
    day_x: str,
    day_y: str,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> np.ndarray:
    """PV correlation across positions between two days for one trial type.

    When day_x == day_y, uses split-half.

    Args:
        sessions: From load_multiday_sessions().
        day_x: First date string.
        day_y: Second date string.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.

    Returns:
        PV correlation per bin, shape (n_bins,).
    """


    if day_x == day_y:
        s = sessions[day_x]
        even_avg, odd_avg = get_split_half_tuning_curves(
            s['data'], s['config'], s['metadata'], trial_type,
            signal_col=signal_col,
        )
        return population_vector_correlation(even_avg, odd_avg)
    else:
        tc = multiday_tuning_curves(
            {day_x: sessions[day_x], day_y: sessions[day_y]},
            trial_type, signal_col=signal_col,
        )
        if day_x not in tc or day_y not in tc:
            return np.array([])
        return population_vector_correlation(tc[day_x], tc[day_y])


def multiday_correlation_matrix(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    cell_indices: np.ndarray | None = None,
    metric: str = 'per_cell_median',
) -> tuple[np.ndarray, list[str]]:
    """Day × day correlation matrix for a trial type.

    Computes pairwise correlations between all days including the diagonal
    (autocorrelation via split-half).

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells. None = all.
        metric: How to summarize per-pair. Options:
            'per_cell_median' — median of per-cell correlations.
            'per_cell_mean' — mean of per-cell correlations.
            'pv_mean' — mean PV correlation across bins.

    Returns:
        Tuple of (matrix, dates) where matrix is (n_days, n_days) and
        dates is the list of date strings in order.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    matrix = np.full((n_days, n_days), np.nan)

    for i, day_x in enumerate(dates):
        for j, day_y in enumerate(dates):
            if j < i:
                matrix[i, j] = matrix[j, i]  # symmetric
                continue

            if metric.startswith('per_cell'):
                corrs = multiday_per_cell_correlation(
                    sessions, day_x, day_y, trial_type,
                    signal_col=signal_col,
                    cell_indices=cell_indices,
                )
                valid = corrs[~np.isnan(corrs)]
                if len(valid) == 0:
                    continue
                if metric == 'per_cell_median':
                    matrix[i, j] = np.median(valid)
                else:
                    matrix[i, j] = np.mean(valid)

            elif metric == 'pv_mean':
                pv = multiday_pv_correlation(
                    sessions, day_x, day_y, trial_type,
                    signal_col=signal_col,
                )
                valid = pv[~np.isnan(pv)]
                if len(valid) > 0:
                    matrix[i, j] = np.mean(valid)

    return matrix, dates


# MULTIDAY PLOTTING

def plot_multiday_per_cell_histogram(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    day_x: str | None = None,
    day_y: str | None = None,
    cell_indices: np.ndarray | None = None,
    animal_id: str | None = None,
    figsize: tuple = (8, 5),
    show: bool = True,
) -> Figure:
    """Histogram of per-cell correlations between two days.

    Defaults to earliest vs latest day if day_x/day_y are not specified.
    Same-day comparisons (day_x == day_y) show split-half autocorrelation.

    Args:
        sessions: From load_multiday_sessions().
        day_x: First date string.
        day_y: Second date string.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells. None = all.
        animal_id: Which animal's data to plot.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(sessions.keys())
    if day_x is None:
        day_x = dates[0]
    if day_y is None:
        day_y = dates[-1] if len(dates) > 1 else dates[0]

    corrs = multiday_per_cell_correlation(
        sessions, day_x, day_y, trial_type,
        signal_col=signal_col,
        cell_indices=cell_indices,
    )
    valid = corrs[~np.isnan(corrs)]

    label_a = day_x[5:]  # drop year prefix for cleaner labels
    label_b = day_y[5:]
    is_auto = day_x == day_y
    pair_label = f'{label_a} split-half' if is_auto else f'{label_a} vs {label_b}'

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(valid, bins=np.linspace(-1, 1, 41), color=color, alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(valid), color='black', linestyle='--', linewidth=1.5,
               label=f'median = {np.nanmedian(valid):.3f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_ylabel('Number of cells', fontsize=11)
    ax.set_title(pfmt.build_title('Per-Cell Correlation', trial_type=trial_type,
                              animal_id=animal_id, day_x=day_x, day_y=day_y), fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)

    n_total = len(corrs)
    n_valid = len(valid)
    ax.text(0.02, 0.95,
            f'n = {n_valid}/{n_total} cells',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_multiday_pv_across_position(
    sessions: dict[str, dict],
    trial_type: str,
    day_x: str | None = None,
    day_y: str | None = None,
    config: dict | None = None,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Plot PV correlation across position between two days.

    Same visual format as ``plot_pv_correlation_across_position`` but for cross-day
    comparisons. Includes cue-region shading and a horizontal line at the mean
    correlation.

    If no data is available for the requested trial type on either day, a blank
    figure with "No data" text is returned.

    Args:
        sessions: From load_multiday_sessions().
        day_x: First date string.
        day_y: Second date string.
        trial_type: Which trial type.
        config: Experiment config for cue shading. Uses day_x's config if None.
        signal_col: Column containing neural signals.
        animal_id: Which animal's data to plot.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(sessions.keys())
    if day_x is None:
        day_x = dates[0]
    if day_y is None:
        day_y = dates[-1] if len(dates) > 1 else dates[0]

    pv = multiday_pv_correlation(
        sessions, day_x, day_y, trial_type,
        signal_col=signal_col,
    )
    if len(pv) == 0:
        print(f"No data for {trial_type} on {day_x} or {day_y}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return fig

    if config is None:
        config = sessions[day_x]['config']

    bin_size_cm = get_bin_size(sessions[day_x]['metadata'])
    x = np.arange(len(pv)) * bin_size_cm + bin_size_cm / 2
    track_len = len(pv) * bin_size_cm

    label_a = day_x[5:]
    label_b = day_y[5:]
    is_auto = day_x == day_y
    pair_label = f'{label_a} split-half' if is_auto else f'{label_a} vs {label_b}'

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, pv, color=color, linewidth=2, zorder=4)
    ax.fill_between(x, pv, alpha=0.25, color=color, zorder=3)

    pfmt.add_cue_shading_with_labels(ax, config, trial_type, alpha=0.1)
    pfmt.set_cue_boundary_ticks(ax, config, trial_type)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    ax.set_title(pfmt.build_title('PV Correlation', trial_type=trial_type, animal_id=animal_id, day_x=day_x,
                              day_y=day_y), fontsize=13, fontweight='bold')
    ax.set_xlim(0, track_len)
    y_min = np.nanmin(pv)
    y_max = np.nanmax(pv)
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    if y_min > 0:
        ax.axhline(0, color='gray', linewidth=1, alpha=0.5)

    mean_r = np.nanmean(pv)
    ax.axhline(mean_r, color=color, linestyle='--', linewidth=1, alpha=0.5,
               label=f'mean r = {mean_r:.3f}')
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_multiday_pv_correlation_matrix(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    day_x: str | None = None,
    day_y: str | None = None,
    animal_id: str | None = None,
    figsize: tuple = (8, 7),
    show: bool = True,
) -> Figure:
    """Plot a bin×bin PV correlation heatmap for one trial type across two days.

    This is the multiday analog of ``plot_pv_correlation_matrix``. It shows whether
    the spatial code at each position on one day matches the code at each position
    on another day. Strong diagonal structure means the map is stable across days;
    off-diagonal peaks indicate spatial drift.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Trial type to compare.
        signal_col: Column containing neural signals.
        day_x: First date string. Defaults to earliest.
        day_y: Second date string. Defaults to latest.
        animal_id: Animal name for title. Extracted from sessions if None.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(sessions.keys())
    if day_x is None:
        day_x = dates[0]
    if day_y is None:
        day_y = dates[-1] if len(dates) > 1 else dates[0]

    bin_size_cm = get_bin_size(sessions[day_x]['metadata'])

    tc = multiday_tuning_curves(
        {day_x: sessions[day_x], day_y: sessions[day_y]},
        trial_type, signal_col=signal_col,
    )
    if day_x not in tc or day_y not in tc:
        print(f"No data for {trial_type} on {day_x} or {day_y}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return fig

    matrix = pv_correlation_matrix(tc[day_x], tc[day_y])
    config = sessions[day_x]['config']

    if animal_id is None:
        animal_id = sessions[day_x]['session_data'].get('animal_id', '')

    label_x = day_x[5:]
    label_y = day_y[5:]

    fig, ax = plt.subplots(figsize=figsize)
    pfmt.plot_pv_heatmap(
        ax, matrix, config,
        x_type=trial_type, y_type=trial_type,
        x_label=f'{trial_type} position (cm) — {label_x}',
        y_label=f'{trial_type} position (cm) — {label_y}',
        bin_size_cm=bin_size_cm,
    )
    ax.set_title(pfmt.build_title('PV Matrix', trial_type=trial_type,
                                  animal_id=animal_id, day_x=day_x, day_y=day_y),
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_multiday_correlation_matrix_summary(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    cell_indices: np.ndarray | None = None,
    metric: str = 'per_cell_median',
    animal_id: str | None = None,
    figsize: tuple = (7, 6),
    show: bool = True,
) -> Figure:
    """Plot a day × day SUMMARY (e.g. median per-cell r) correlation heatmap.

    Diagonal = autocorrelation (split-half). Off-diagonal = cross-day.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells. None = all.
        metric: Summary metric ('per_cell_median', 'per_cell_mean', 'pv_mean').
        animal_id: Animal name for title. Extracted from sessions if None.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    matrix, dates = multiday_correlation_matrix(
        sessions, trial_type, signal_col=signal_col,
        cell_indices=cell_indices, metric=metric,
    )

    date_labels = [d[5:] for d in dates]  # drop year

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.2, vmax=1.0, aspect='equal')
    plt.colorbar(im, ax=ax, label=f'{metric} (Pearson r)', shrink=0.85)

    # Annotate values
    for i in range(len(dates)):
        for j in range(len(dates)):
            if not np.isnan(matrix[i, j]):
                color = 'white' if matrix[i, j] < 0.3 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color=color)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels(date_labels, fontsize=9)
    ax.set_xlabel('Session date', fontsize=11)
    ax.set_ylabel('Session date', fontsize=11)
    ax.set_title(pfmt.build_title(f'Day × Day Correlation ({metric})',
                              trial_type=trial_type, animal_id=animal_id), fontsize=13, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# MULTIDAY GRID PLOTS

def _clear_no_data_axes(ax: Axes) -> None:
    """Strips all ticks and spines from a subplot that has no plottable data."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_grid_date_headers(
    fig: Figure,
    axes: np.ndarray,
    date_labels: list[str],
    n_days: int,
) -> None:
    """Adds date labels as column headers (top row titles) and row headers (left-side ylabels).

    Column dates are placed as subplot titles on the top row. Row dates are placed as
    ylabels on the leftmost visible subplot in each row. Diagonal subplots get a small
    'split-half' annotation.

    Args:
        fig: Parent figure (unused but kept for future flexibility).
        axes: 2-D array of subplot Axes from plt.subplots.
        date_labels: List of date strings (e.g. ['08-30', '09-03', ...]).
        n_days: Number of days (grid dimension).
    """
    # Column headers on the top row
    for j in range(n_days):
        axes[0][j].set_title(date_labels[j], fontsize=10, fontweight='bold')

    # Row headers on the leftmost visible subplot per row
    for i in range(n_days):
        for j in range(n_days):
            if axes[i][j].get_visible():
                axes[i][j].set_ylabel(date_labels[i], fontsize=10, fontweight='bold')
                break

    # 'split-half' annotation inside each diagonal subplot
    for i in range(n_days):
        ax = axes[i][i]
        if ax.get_visible():
            ax.text(
                0.5, 0.97, 'split-half', transform=ax.transAxes,
                fontsize=7, ha='center', va='top', style='italic',
                color='#555555',
            )


def plot_multiday_pv_heatmap_grid(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple = (4, 3.5),
    show: bool = True,
) -> Figure:
    """N×N grid of bin×bin PV correlation heatmaps for one trial type across days.

    Each subplot shows the bin×bin PV correlation matrix for one day pair. Dates label
    rows and columns. Diagonal shows split-half autocorrelation. The grid is symmetric:
    subplot (i, j) shows the same comparison as (j, i).

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Trial type to compare across days.
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n_days, n_days,
        figsize=(figsize_per_cell[0] * n_days, figsize_per_cell[1] * n_days),
        squeeze=False,
    )

    tuning_curves = multiday_tuning_curves(sessions, trial_type, signal_col=signal_col)
    last_im = None

    for i in range(n_days):
        for j in range(n_days):
            ax = axes[i][j]
            day_row = dates[i]
            day_col = dates[j]

            if i == j:
                session = sessions[day_row]
                tc_x, tc_y = get_split_half_tuning_curves(
                    session['data'], session['config'], session['metadata'],
                    trial_type, signal_col=signal_col,
                )
            else:
                day_earlier = min(day_row, day_col)
                day_later = max(day_row, day_col)
                if day_earlier not in tuning_curves or day_later not in tuning_curves:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                    )
                    _clear_no_data_axes(ax)
                    continue
                tc_x = tuning_curves[day_later]
                tc_y = tuning_curves[day_earlier]

            matrix = pv_correlation_matrix(tc_x, tc_y)
            config = sessions[day_row]['config']
            bin_size_cm = get_bin_size(sessions[day_row]['metadata'])

            len_x = matrix.shape[0] * bin_size_cm
            len_y = matrix.shape[1] * bin_size_cm

            last_im = ax.imshow(
                matrix.T, origin='lower', aspect='auto',
                cmap='RdBu_r', vmin=-0.3, vmax=1.0,
                extent=[0, len_x, 0, len_y],
            )

            max_len = min(len_x, len_y)
            ax.plot(
                [0, max_len], [0, max_len], color='white', linewidth=0.8,
                linestyle='--', alpha=0.5,
            )
            pfmt.add_cue_boundary_lines(ax, config, trial_type, axis='both')

            pfmt.add_cue_bar(ax, config, trial_type, axis='x', bar_width=0.02)
            pfmt.add_cue_bar(ax, config, trial_type, axis='y', bar_width=0.02)

            x_boundaries = pfmt.get_cue_boundaries(config, trial_type, max_cm=len_x)
            y_boundaries = pfmt.get_cue_boundaries(config, trial_type, max_cm=len_y)
            ax.set_xticks(x_boundaries)
            ax.set_yticks(y_boundaries)

            if i == n_days - 1:
                ax.set_xticklabels([f'{t:.0f}' for t in x_boundaries], fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels([f'{t:.0f}' for t in y_boundaries], fontsize=6)
            else:
                ax.set_yticklabels([])

            ax.tick_params(length=0, pad=8, labelsize=6)

    _add_grid_date_headers(fig, axes, date_labels, n_days)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label='PV Correlation (r)')

    suptitle = f'PV Heatmap Grid — {trial_type}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_pv_position_grid(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple = (4, 3),
    show: bool = True,
) -> Figure:
    """N×N grid of PV correlation vs position for one trial type across all day pairs.

    Each subplot shows the population vector correlation at each spatial bin for one
    day pair with a mini cue bar along the x-axis. Dates label the rows and columns.
    Diagonal shows split-half autocorrelation. The grid is symmetric.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Trial type to compare across days.
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n_days, n_days,
        figsize=(figsize_per_cell[0] * n_days, figsize_per_cell[1] * n_days),
        squeeze=False,
    )

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    # Precompute PV correlations for all unique pairs and find shared y-range
    all_pv_values = []
    pv_data = {}
    for i in range(n_days):
        for j in range(i, n_days):
            pv = multiday_pv_correlation(
                sessions, dates[j], dates[i], trial_type, signal_col=signal_col,
            )
            pv_data[(i, j)] = pv
            if len(pv) > 0:
                valid = pv[~np.isnan(pv)]
                if len(valid) > 0:
                    all_pv_values.extend(valid.tolist())

    if all_pv_values:
        y_min = min(all_pv_values) - 0.05
        y_max = max(all_pv_values) + 0.05
    else:
        y_min, y_max = -0.1, 1.1

    def _plot_pv_line(ax: Axes, pv: np.ndarray, day_key: str, color: str, label: str | None = None) -> None:
        """Draws a PV correlation line with fill and mean annotation on the given axes."""
        bin_size_cm = get_bin_size(sessions[day_key]['metadata'])
        x_positions = np.arange(len(pv)) * bin_size_cm + bin_size_cm / 2
        track_length = len(pv) * bin_size_cm

        ax.plot(x_positions, pv, color=color, linewidth=1.5, zorder=4, label=label)
        ax.fill_between(x_positions, pv, alpha=0.12, color=color, zorder=3)

        mean_r = np.nanmean(pv)
        ax.axhline(mean_r, color=color, linestyle='--', linewidth=0.8, alpha=0.4)
        ax.set_xlim(0, track_length)

    for i in range(n_days):
        for j in range(n_days):
            ax = axes[i][j]
            pv_key = (min(i, j), max(i, j))

            if i == j:
                pv = pv_data.get((i, j), np.array([]))
                day_key = dates[i]
                config = sessions[day_key]['config']

                if len(pv) > 0:
                    pfmt.add_cue_shading(ax, config, trial_type, alpha=0.06)
                    _plot_pv_line(ax, pv, day_key, color)

                ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
                ax.set_ylim(y_min, y_max)

                pfmt.add_cue_bar(ax, config, trial_type, axis='x', bar_width=0.02)
                pfmt.set_cue_boundary_ticks(ax, config, trial_type)
            else:
                pv = pv_data.get(pv_key, np.array([]))

                if len(pv) == 0:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                    )
                    _clear_no_data_axes(ax)
                    continue

                day_key = dates[max(i, j)]
                config = sessions[day_key]['config']
                pfmt.add_cue_shading(ax, config, trial_type, alpha=0.06)
                _plot_pv_line(ax, pv, day_key, color)

                mean_r = np.nanmean(pv)
                ax.text(
                    0.97, 0.05, f'r\u0304={mean_r:.2f}', transform=ax.transAxes,
                    fontsize=7, ha='right', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                )

                ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
                ax.set_ylim(y_min, y_max)

                pfmt.add_cue_bar(ax, config, trial_type, axis='x', bar_width=0.02)
                pfmt.set_cue_boundary_ticks(ax, config, trial_type)

            if i != n_days - 1:
                ax.set_xticklabels([])

            ax.tick_params(length=0, pad=8, labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    _add_grid_date_headers(fig, axes, date_labels, n_days)

    suptitle = f'PV Correlation Grid — {trial_type}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_histogram_grid(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    cell_indices: np.ndarray | None = None,
    animal_id: str | None = None,
    figsize_per_cell: tuple = (3.5, 3),
    show: bool = True,
) -> Figure:
    """N×N grid of per-cell spatial correlation histograms for one trial type across days.

    Each subplot shows the distribution of per-cell Pearson r between tuning curves
    for one day pair. Dates label the rows and columns. Diagonal shows split-half
    autocorrelation. The grid is symmetric.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Trial type to compare across days.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells to include. None uses all cells.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n_days, n_days,
        figsize=(figsize_per_cell[0] * n_days, figsize_per_cell[1] * n_days),
        squeeze=False,
    )

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')
    histogram_bins = np.linspace(-1, 1, 41)

    # Precompute correlations for all unique pairs
    max_count = 0
    corr_data = {}
    for i in range(n_days):
        for j in range(i, n_days):
            corrs = multiday_per_cell_correlation(
                sessions, dates[j], dates[i], trial_type,
                signal_col=signal_col, cell_indices=cell_indices,
            )
            valid = corrs[~np.isnan(corrs)]
            corr_data[(i, j)] = valid
            if len(valid) > 0:
                counts, _ = np.histogram(valid, bins=histogram_bins)
                max_count = max(max_count, counts.max())

    for i in range(n_days):
        for j in range(n_days):
            ax = axes[i][j]
            corr_key = (min(i, j), max(i, j))
            valid = corr_data.get(corr_key, np.array([]))

            if i == j:
                if len(valid) > 0:
                    ax.hist(
                        valid, bins=histogram_bins, color=color, alpha=0.7,
                        edgecolor='white', linewidth=0.3,
                    )
                    median_r = np.nanmedian(valid)
                    ax.axvline(median_r, color='black', linestyle='--', linewidth=1.2)

                ax.axvline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, max_count * 1.1)
            else:
                if len(valid) == 0:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                    )
                    _clear_no_data_axes(ax)
                    continue

                ax.hist(
                    valid, bins=histogram_bins, color=color, alpha=0.7,
                    edgecolor='white', linewidth=0.3,
                )

                median_r = np.nanmedian(valid)
                ax.axvline(median_r, color='black', linestyle='--', linewidth=1.2)
                ax.axvline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)

                ax.text(
                    0.03, 0.95, f'med={median_r:.2f}\nn={len(valid)}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                )

                ax.set_xlim(-1, 1)
                ax.set_ylim(0, max_count * 1.1)

            if i != n_days - 1:
                ax.set_xticklabels([])

            ax.tick_params(length=0, labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    _add_grid_date_headers(fig, axes, date_labels, n_days)

    suptitle = f'Per-Cell Correlation Grid — {trial_type}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_combined_grid(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple[float, float] = (4, 3.5),
    show: bool = True,
) -> Figure:
    """N×N PairGrid-style plot: heatmaps on lower triangle + diagonal, mean PV r dots above.

    Layout:
        - Lower triangle: bin×bin PV correlation heatmap for each cross-day pair.
        - Diagonal: split-half PV correlation heatmap (autocorrelation).
        - Upper triangle: single correlation dot whose size and color encode the
          mean PV correlation across spatial bins (RdBu_r, -0.3 to 1.0).

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Trial type to compare across days.
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n_days, n_days,
        figsize=(figsize_per_cell[0] * n_days, figsize_per_cell[1] * n_days),
        squeeze=False,
    )

    tuning_curves = multiday_tuning_curves(sessions, trial_type, signal_col=signal_col)
    last_im = None

    for i in range(n_days):
        for j in range(n_days):
            ax = axes[i][j]
            day_row = dates[i]
            day_col = dates[j]

            if i >= j:
                # LOWER TRIANGLE + DIAGONAL — PV heatmap
                if i == j:
                    session = sessions[day_row]
                    tc_x, tc_y = get_split_half_tuning_curves(
                        session['data'], session['config'], session['metadata'],
                        trial_type, signal_col=signal_col,
                    )
                else:
                    day_earlier = min(day_row, day_col)
                    day_later = max(day_row, day_col)
                    if day_earlier not in tuning_curves or day_later not in tuning_curves:
                        ax.text(
                            0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax.transAxes,
                        )
                        _clear_no_data_axes(ax)
                        continue
                    tc_x = tuning_curves[day_later]
                    tc_y = tuning_curves[day_earlier]

                matrix = pv_correlation_matrix(tc_x, tc_y)
                config = sessions[day_row]['config']
                bin_size_cm = get_bin_size(sessions[day_row]['metadata'])

                len_x = matrix.shape[0] * bin_size_cm
                len_y = matrix.shape[1] * bin_size_cm

                last_im = ax.imshow(
                    matrix.T, origin='lower', aspect='auto',
                    cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=1.0),
                    extent=[0, len_x, 0, len_y],
                )

                max_len = min(len_x, len_y)
                ax.plot(
                    [0, max_len], [0, max_len], color='white', linewidth=0.8,
                    linestyle='--', alpha=0.5,
                )
                pfmt.add_cue_boundary_lines(ax, config, trial_type, axis='both')
                pfmt.add_cue_bar(ax, config, trial_type, axis='x', bar_width=0.02)
                pfmt.add_cue_bar(ax, config, trial_type, axis='y', bar_width=0.02)

                x_boundaries = pfmt.get_cue_boundaries(config, trial_type, max_cm=len_x)
                y_boundaries = pfmt.get_cue_boundaries(config, trial_type, max_cm=len_y)
                ax.set_xticks(x_boundaries)
                ax.set_yticks(y_boundaries)

                if i == n_days - 1:
                    ax.set_xticklabels([f'{t:.0f}' for t in x_boundaries], fontsize=6)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_yticklabels([f'{t:.0f}' for t in y_boundaries], fontsize=6)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(length=0, pad=8, labelsize=6)

            else:
                # UPPER TRIANGLE — mean PV correlation dot
                ax.set_axis_off()

                pv = multiday_pv_correlation(
                    sessions, day_col, day_row, trial_type, signal_col=signal_col,
                )
                if len(pv) == 0:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8,
                    )
                    continue

                mean_r = float(np.nanmean(pv))
                corr_text = f"{mean_r:2.2f}".replace("0.", ".").replace("-0.", "-.")
                marker_size = abs(mean_r) * 10000
                font_size = abs(mean_r) * 40 + 5

                ax.scatter(
                    [0.5], [0.5], s=marker_size, c=[mean_r], alpha=0.6,
                    cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=1.0),
                    transform=ax.transAxes,zorder=3,
                )
                ax.annotate(
                    corr_text, xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center', fontsize=font_size, fontweight='bold',
                )

    # Column headers on top row
    for j in range(n_days):
        axes[0][j].set_title(date_labels[j], fontsize=10, fontweight='bold')

    # Row headers on leftmost column (always a heatmap since lower triangle includes col 0)
    for i in range(n_days):
        axes[i][0].set_ylabel(date_labels[i], fontsize=10, fontweight='bold')

    # Split-half annotation on diagonal
    for i in range(n_days):
        ax = axes[i][i]
        if ax.get_visible():
            ax.text(
                0.5, 0.97, 'split-half', transform=ax.transAxes,
                fontsize=7, ha='center', va='top', style='italic',
                color='#555555',
            )

    cbar_ax = fig.add_axes([0.89, 0.15, 0.015, 0.7])
    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label='PV Correlation (r)')

    suptitle = f'Combined PV Grid — {trial_type}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.075, bottom=0.06, right=0.87, hspace=0.3, wspace=0.3)

    if show:
        plt.show()
    return fig


# MULTIDAY GRID PLOTS — CROSS-CONDITION (2N×2N)

def _add_cross_grid_headers(
    fig: Figure,
    axes: np.ndarray,
    date_labels: list[str],
    n_days: int,
    type_a: str,
    type_b: str,
) -> None:
    """Adds date + trial-type block headers for a 2N×2N cross-condition grid.

    Column headers show dates; the first column of each type block is prefixed with
    the trial type name. Row headers follow the same pattern on the leftmost column.

    Args:
        fig: Parent figure (unused, kept for API consistency).
        axes: 2-D array of subplot Axes (2N × 2N).
        date_labels: List of date strings.
        n_days: Number of days (half the grid dimension).
        type_a: Trial type for the first N rows/columns.
        type_b: Trial type for the last N rows/columns.
    """
    n = 2 * n_days
    # Column headers on top row
    for j in range(n):
        tt = type_a if j < n_days else type_b
        day_label = date_labels[j % n_days]
        if j == 0 or j == n_days:
            axes[0][j].set_title(f'{tt}\n{day_label}', fontsize=9, fontweight='bold')
        else:
            axes[0][j].set_title(day_label, fontsize=9, fontweight='bold')

    # Row headers on leftmost column
    for i in range(n):
        tt = type_a if i < n_days else type_b
        day_label = date_labels[i % n_days]
        if i == 0 or i == n_days:
            axes[i][0].set_ylabel(f'{tt}\n{day_label}', fontsize=9, fontweight='bold')
        else:
            axes[i][0].set_ylabel(day_label, fontsize=9, fontweight='bold')

    # Diagonal annotations for same-type blocks (split-half)
    for i in range(n_days):
        for ax in [axes[i][i], axes[n_days + i][n_days + i]]:
            if ax.get_visible():
                ax.text(
                    0.5, 0.97, 'split-half', transform=ax.transAxes,
                    fontsize=7, ha='center', va='top', style='italic',
                    color='#555555',
                )


def plot_multiday_pv_heatmap_grid_cross(
    sessions: dict[str, dict],
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple = (3.5, 3),
    show: bool = True,
) -> Figure:
    """2N×2N grid of bin×bin PV correlation heatmaps comparing two trial types across days.

    The grid is organized in four N×N blocks:
        - Top-left: type_a vs type_a across days
        - Top-right: type_a (row) vs type_b (col) across days
        - Bottom-left: type_b (row) vs type_a (col) across days
        - Bottom-right: type_b vs type_b across days

    Same-type same-day diagonal entries use split-half. Note that the top-right and
    bottom-left blocks are transposes of each other.

    Args:
        sessions: From load_multiday_sessions().
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full 2N×2N grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    n = 2 * n_days
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n, n,
        figsize=(figsize_per_cell[0] * n, figsize_per_cell[1] * n),
        squeeze=False,
    )

    tc_a = multiday_tuning_curves(sessions, type_a, signal_col=signal_col)
    tc_b = multiday_tuning_curves(sessions, type_b, signal_col=signal_col)
    tc_by_type = {type_a: tc_a, type_b: tc_b}
    last_im = None

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]

            row_type = type_a if i < n_days else type_b
            row_day = dates[i % n_days]
            col_type = type_a if j < n_days else type_b
            col_day = dates[j % n_days]

            if row_type == col_type and row_day == col_day:
                if row_day not in tc_by_type[row_type]:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                    )
                    _clear_no_data_axes(ax)
                    continue
                session = sessions[row_day]
                tc_x, tc_y = get_split_half_tuning_curves(
                    session['data'], session['config'], session['metadata'],
                    row_type, signal_col=signal_col,
                )
            else:
                tc_row = tc_by_type[row_type].get(row_day)
                tc_col = tc_by_type[col_type].get(col_day)
                if tc_row is None or tc_col is None:
                    ax.text(
                        0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes,
                    )
                    _clear_no_data_axes(ax)
                    continue
                tc_x = tc_col
                tc_y = tc_row

            matrix = pv_correlation_matrix(tc_x, tc_y)
            config = sessions[row_day]['config']
            bin_size_cm = get_bin_size(sessions[row_day]['metadata'])

            len_x = matrix.shape[0] * bin_size_cm
            len_y = matrix.shape[1] * bin_size_cm

            last_im = ax.imshow(
                matrix.T, origin='lower', aspect='auto',
                cmap='RdBu_r', vmin=-0.3, vmax=1.0,
                extent=[0, len_x, 0, len_y],
            )

            max_len = min(len_x, len_y)
            ax.plot(
                [0, max_len], [0, max_len], color='white', linewidth=0.8,
                linestyle='--', alpha=0.5,
            )

            pfmt.add_cue_boundary_lines(ax, config, col_type, axis='x')
            pfmt.add_cue_boundary_lines(ax, config, row_type, axis='y')
            pfmt.add_cue_bar(ax, config, col_type, axis='x', bar_width=0.02)
            pfmt.add_cue_bar(ax, config, row_type, axis='y', bar_width=0.02)

            x_boundaries = pfmt.get_cue_boundaries(config, col_type, max_cm=len_x)
            y_boundaries = pfmt.get_cue_boundaries(config, row_type, max_cm=len_y)
            ax.set_xticks(x_boundaries)
            ax.set_yticks(y_boundaries)

            if i == n - 1:
                ax.set_xticklabels([f'{t:.0f}' for t in x_boundaries], fontsize=5)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels([f'{t:.0f}' for t in y_boundaries], fontsize=5)
            else:
                ax.set_yticklabels([])

            ax.tick_params(length=0, pad=8, labelsize=5)

    _add_cross_grid_headers(fig, axes, date_labels, n_days, type_a, type_b)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label='PV Correlation (r)')

    suptitle = f'PV Heatmap Grid — {type_a} \u00d7 {type_b}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_pv_heatmap_grid_a_vs_b(
    sessions: dict[str, dict],
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple = (4, 3.5),
    show: bool = True,
) -> Figure:
    """Rectangular grid of bin×bin PV correlation heatmaps comparing type_a (x) vs type_b (y).

    Columns span every day that has type_a data; rows span every day that has type_b data.
    When type_b is not run on every day (e.g. ABDC introduced mid-experiment), the grid is
    rectangular rather than square — empty rows are dropped instead of left blank. Each cell
    (i, j) is the bin×bin PV correlation between type_a at the column date and type_b at the
    row date.

    No split-half is shown — same-day cells compare the two trial types on that day (the
    cross-condition counterpart of the single-day PV matrix figure).

    Args:
        sessions: From load_multiday_sessions().
        type_a: Trial type plotted on the x-axis (columns).
        type_b: Trial type plotted on the y-axis (rows).
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full N×N grid.
    """
    all_dates = sorted(sessions.keys())
    tc_a = multiday_tuning_curves(sessions, type_a, signal_col=signal_col)
    tc_b = multiday_tuning_curves(sessions, type_b, signal_col=signal_col)

    dates_x = [d for d in all_dates if d in tc_a]
    dates_y = [d for d in all_dates if d in tc_b]
    n_x = len(dates_x)
    n_y = len(dates_y)
    if n_x == 0 or n_y == 0:
        raise ValueError(f'No sessions with both {type_a} and {type_b} tuning curves.')

    x_labels = [d[5:] for d in dates_x]
    y_labels = [d[5:] for d in dates_y]

    fig, axes = plt.subplots(
        n_y, n_x,
        figsize=(figsize_per_cell[0] * n_x, figsize_per_cell[1] * n_y),
        squeeze=False,
    )
    last_im = None

    for i in range(n_y):
        for j in range(n_x):
            ax = axes[i][j]
            day_col = dates_x[j]  # type_a (x-axis)
            day_row = dates_y[i]  # type_b (y-axis)

            tc_x = tc_a.get(day_col)
            tc_y = tc_b.get(day_row)
            if tc_x is None or tc_y is None:
                ax.text(
                    0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes,
                )
                _clear_no_data_axes(ax)
                continue

            matrix = pv_correlation_matrix(tc_x, tc_y)
            config_x = sessions[day_col]['config']
            config_y = sessions[day_row]['config']
            bin_size_cm = get_bin_size(sessions[day_col]['metadata'])

            len_x = matrix.shape[0] * bin_size_cm
            len_y = matrix.shape[1] * bin_size_cm

            last_im = ax.imshow(
                matrix.T, origin='lower', aspect='auto',
                cmap='RdBu_r', vmin=-0.3, vmax=1.0,
                extent=[0, len_x, 0, len_y],
            )

            max_len = min(len_x, len_y)
            ax.plot(
                [0, max_len], [0, max_len], color='white', linewidth=0.8,
                linestyle='--', alpha=0.5,
            )

            diverge, _ = get_shared_bins(config_y, type_a, type_b, sessions[day_row]['metadata'])
            if diverge is not None:
                ax.axvline(diverge, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.axhline(diverge, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

            pfmt.add_cue_boundary_lines(ax, config_x, type_a, axis='x')
            pfmt.add_cue_boundary_lines(ax, config_y, type_b, axis='y')
            pfmt.add_cue_bar(ax, config_x, type_a, axis='x', bar_width=0.02)
            pfmt.add_cue_bar(ax, config_y, type_b, axis='y', bar_width=0.02)

            x_boundaries = pfmt.get_cue_boundaries(config_x, type_a, max_cm=len_x)
            y_boundaries = pfmt.get_cue_boundaries(config_y, type_b, max_cm=len_y)
            ax.set_xticks(x_boundaries)
            ax.set_yticks(y_boundaries)

            if i == n_y - 1:
                ax.set_xticklabels([f'{t:.0f}' for t in x_boundaries], fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_yticklabels([f'{t:.0f}' for t in y_boundaries], fontsize=6)
            else:
                ax.set_yticklabels([])

            ax.tick_params(length=0, pad=8, labelsize=6)

    # Column headers (type_a dates) and row headers (type_b dates)
    for j in range(n_x):
        label = f'{type_a}\n{x_labels[j]}' if j == 0 else x_labels[j]
        axes[0][j].set_title(label, fontsize=10, fontweight='bold')
    for i in range(n_y):
        label = f'{type_b}\n{y_labels[i]}' if i == 0 else y_labels[i]
        axes[i][0].set_ylabel(label, fontsize=10, fontweight='bold')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    if last_im is not None:
        fig.colorbar(last_im, cax=cbar_ax, label='PV Correlation (r)')

    suptitle = f'PV Heatmap Grid — {type_a} (x) vs {type_b} (y)'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    fig.subplots_adjust(right=0.9, hspace=0.3, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_pv_position_grid_cross(
    sessions: dict[str, dict],
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    figsize_per_cell: tuple = (4, 3),
    show: bool = True,
) -> Figure:
    """2N×2N grid of PV correlation vs position comparing two trial types across days.

    Four N×N blocks: type_a×type_a, type_a×type_b, type_b×type_a, type_b×type_b.
    Same-type same-day diagonal entries use split-half. Cross-type comparisons use
    the shorter track length.

    Args:
        sessions: From load_multiday_sessions().
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full 2N×2N grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    n = 2 * n_days
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n, n,
        figsize=(figsize_per_cell[0] * n, figsize_per_cell[1] * n),
        squeeze=False,
    )

    color_a = pfmt.TRIAL_TYPE_COLORS.get(type_a, '#2E86AB')
    color_b = pfmt.TRIAL_TYPE_COLORS.get(type_b, '#A23B72')

    tc_a = multiday_tuning_curves(sessions, type_a, signal_col=signal_col)
    tc_b = multiday_tuning_curves(sessions, type_b, signal_col=signal_col)
    tc_by_type = {type_a: tc_a, type_b: tc_b}
    color_by_type = {type_a: color_a, type_b: color_b}

    # Precompute all PV correlations and find shared y-range
    all_pv_values = []
    pv_data = {}
    for i in range(n):
        for j in range(n):
            row_type = type_a if i < n_days else type_b
            row_day = dates[i % n_days]
            col_type = type_a if j < n_days else type_b
            col_day = dates[j % n_days]

            if row_type == col_type and row_day == col_day:
                if row_day not in tc_by_type[row_type]:
                    pv = np.array([])
                else:
                    session = sessions[row_day]
                    even, odd = get_split_half_tuning_curves(
                        session['data'], session['config'], session['metadata'],
                        row_type, signal_col=signal_col,
                    )
                    pv = population_vector_correlation(even, odd)
            else:
                tc_row = tc_by_type[row_type].get(row_day)
                tc_col = tc_by_type[col_type].get(col_day)
                if tc_row is None or tc_col is None:
                    pv = np.array([])
                else:
                    pv = population_vector_correlation(tc_col, tc_row)

            pv_data[(i, j)] = pv
            if len(pv) > 0:
                valid = pv[~np.isnan(pv)]
                if len(valid) > 0:
                    all_pv_values.extend(valid.tolist())

    if all_pv_values:
        y_min = min(all_pv_values) - 0.05
        y_max = max(all_pv_values) + 0.05
    else:
        y_min, y_max = -0.1, 1.1

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            col_type = type_a if j < n_days else type_b
            active_color = color_by_type[col_type]

            pv = pv_data[(i, j)]
            day_key = dates[max(i % n_days, j % n_days)]
            config = sessions[day_key]['config']

            if len(pv) == 0:
                ax.text(
                    0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes,
                )
                _clear_no_data_axes(ax)
                continue

            bin_size_cm = get_bin_size(sessions[day_key]['metadata'])
            x_positions = np.arange(len(pv)) * bin_size_cm + bin_size_cm / 2
            track_length = len(pv) * bin_size_cm

            pfmt.add_cue_shading(ax, config, col_type, alpha=0.06)
            ax.plot(x_positions, pv, color=active_color, linewidth=1.5, zorder=4)
            ax.fill_between(x_positions, pv, alpha=0.12, color=active_color, zorder=3)

            mean_r = np.nanmean(pv)
            ax.axhline(mean_r, color=active_color, linestyle='--', linewidth=0.8, alpha=0.4)
            ax.text(
                0.97, 0.05, f'r\u0304={mean_r:.2f}', transform=ax.transAxes,
                fontsize=7, ha='right', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            )

            ax.set_xlim(0, track_length)
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
            ax.set_ylim(y_min, y_max)

            pfmt.add_cue_bar(ax, config, col_type, axis='x', bar_width=0.02)
            pfmt.set_cue_boundary_ticks(ax, config, col_type)

            if i != n - 1:
                ax.set_xticklabels([])
            ax.tick_params(length=0, pad=8, labelsize=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    _add_cross_grid_headers(fig, axes, date_labels, n_days, type_a, type_b)

    suptitle = f'PV Correlation Grid — {type_a} \u00d7 {type_b}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    if show:
        plt.show()
    return fig


def plot_multiday_histogram_grid_cross(
    sessions: dict[str, dict],
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_spikes',
    cell_indices: np.ndarray | None = None,
    animal_id: str | None = None,
    figsize_per_cell: tuple = (3.5, 3),
    show: bool = True,
) -> Figure:
    """2N×2N grid of per-cell correlation histograms comparing two trial types across days.

    Four N×N blocks: type_a×type_a, type_a×type_b, type_b×type_a, type_b×type_b.
    Same-type same-day entries use split-half. Cross-type and cross-day entries
    compare full session-averaged tuning curves.

    Args:
        sessions: From load_multiday_sessions().
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        cell_indices: Subset of cells. None uses all.
        animal_id: Animal identifier for the super-title.
        figsize_per_cell: Width and height per subplot cell.
        show: Determines whether to call plt.show().

    Returns:
        Matplotlib Figure containing the full 2N×2N grid.
    """
    dates = sorted(sessions.keys())
    n_days = len(dates)
    n = 2 * n_days
    date_labels = [d[5:] for d in dates]

    fig, axes = plt.subplots(
        n, n,
        figsize=(figsize_per_cell[0] * n, figsize_per_cell[1] * n),
        squeeze=False,
    )

    color_a = pfmt.TRIAL_TYPE_COLORS.get(type_a, '#2E86AB')
    color_b = pfmt.TRIAL_TYPE_COLORS.get(type_b, '#A23B72')
    color_by_type = {type_a: color_a, type_b: color_b}
    histogram_bins = np.linspace(-1, 1, 41)

    tc_a = multiday_tuning_curves(sessions, type_a, signal_col=signal_col)
    tc_b = multiday_tuning_curves(sessions, type_b, signal_col=signal_col)
    tc_by_type = {type_a: tc_a, type_b: tc_b}

    # Precompute per-cell correlations and find max histogram count
    max_count = 0
    corr_data = {}
    for i in range(n):
        for j in range(n):
            row_type = type_a if i < n_days else type_b
            row_day = dates[i % n_days]
            col_type = type_a if j < n_days else type_b
            col_day = dates[j % n_days]

            if row_type == col_type and row_day == col_day:
                if row_day not in tc_by_type[row_type]:
                    corr_data[(i, j)] = np.array([])
                    continue
                session = sessions[row_day]
                even, odd = get_split_half_tuning_curves(
                    session['data'], session['config'], session['metadata'],
                    row_type, signal_col=signal_col,
                )
                corrs = per_cell_spatial_correlation(even, odd)
            else:
                tc_row = tc_by_type[row_type].get(row_day)
                tc_col = tc_by_type[col_type].get(col_day)
                if tc_row is None or tc_col is None:
                    corr_data[(i, j)] = np.array([])
                    continue
                corrs = per_cell_spatial_correlation(tc_col, tc_row)

            if cell_indices is not None:
                corrs = corrs[cell_indices]
            valid = corrs[~np.isnan(corrs)]
            corr_data[(i, j)] = valid
            if len(valid) > 0:
                counts, _ = np.histogram(valid, bins=histogram_bins)
                max_count = max(max_count, counts.max())

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            col_type = type_a if j < n_days else type_b
            active_color = color_by_type[col_type]

            valid = corr_data.get((i, j), np.array([]))

            if len(valid) == 0:
                ax.text(
                    0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes,
                )
                _clear_no_data_axes(ax)
                continue

            ax.hist(
                valid, bins=histogram_bins, color=active_color, alpha=0.7,
                edgecolor='white', linewidth=0.3,
            )

            median_r = np.nanmedian(valid)
            ax.axvline(median_r, color='black', linestyle='--', linewidth=1.2)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.4)

            ax.text(
                0.03, 0.95, f'med={median_r:.2f}\nn={len(valid)}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            )

            ax.set_xlim(-1, 1)
            ax.set_ylim(0, max_count * 1.1)

            if i != n - 1:
                ax.set_xticklabels([])
            ax.tick_params(length=0, labelsize=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    _add_cross_grid_headers(fig, axes, date_labels, n_days, type_a, type_b)

    suptitle = f'Per-Cell Correlation Grid — {type_a} \u00d7 {type_b}'
    if animal_id:
        suptitle = f'{animal_id} — {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    if show:
        plt.show()
    return fig


# MULTIDAY WRAPPER

def run_multiday_analysis(
    sessions: dict[str, dict],
    trial_type: str | None = None,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    cell_indices: np.ndarray | None = None,
    day_pairs: list[tuple[str, str]] | None = None,
    grid: bool = False,
    grid_trial_type_b: str | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Runs all multiday cross-correlation analysis.

    Full multiday analysis pipeline:
        1. For each day-pair (including autocorrelation on the diagonal):
            a. Per-cell correlation histogram.
            b. PV correlation across position.
            c. Bin × bin PV correlation heatmap.
        2. Day × day summary matrix (if ≥ 2 days).
        3. N×N grid figures for all three plot types (if grid=True).

    If day_pairs is None, generates all unique pairs plus autocorrelation
    entries for each day. If trial_type is None, picks the first trial type
    found across all sessions (alphabetically).

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type to analyze. If None, uses first type found.
        signal_col: Column containing neural signals.
        animal_id: Animal name for title. Extracted from sessions if None.
        cell_indices: Subset of cells. None = all.
        day_pairs: Specific (day_x, day_y) pairs to analyze. None = all pairs + diagonal.
        grid: Determines whether to generate N×N grid overview figures for PV heatmaps,
            PV-across-position, and per-cell histograms in addition to individual plots.
        grid_trial_type_b: Second trial type for grid figures. When provided, the upper
            triangle shows trial_type and the lower triangle shows this type. None uses
            single-type mode (lower triangle hidden).
        show: Call plt.show().
        save_dir: Directory to save figures (optional).

    Returns:
        Dict of {name: Figure}.
    """
    dates = sorted(sessions.keys())
    if animal_id is None:
        animal_id = next(iter(sessions.values()))['session_data'].get('animal_id', '')
    figs = {}

    # Determine trial type
    if trial_type is None:
        all_types = set()
        for s in sessions.values():
            all_types.update(s['data']['trial_type'].unique().to_list())
        trial_type = sorted(all_types)[0]
        print(f"Using trial type: {trial_type}")

    # Default: all unique pairs including autocorrelation
    if day_pairs is None:
        print("day pairs none")
        day_pairs = []
        for i, d in enumerate(dates):
            day_pairs.append((d, d))  # autocorrelation
        for da, db in combinations(dates, 2):
            day_pairs.append((da, db))
        print("day pairs: ", day_pairs)

    print(f"Multiday analysis: {trial_type}, {len(dates)} sessions, {len(day_pairs)} pairs")
    n_cells_label = f'{len(cell_indices)} cells' if cell_indices is not None else 'all cells'
    print(f"  Cells: {n_cells_label}\n")

    # Per-pair histograms and PV plots
    for day_x, day_y in day_pairs:
        label = f'{day_x[5:]}_vs_{day_y[5:]}'
        if day_x == day_y:
            label = f'{day_x[5:]}_auto'

        print(f"  {label}...")

        # Per-cell histogram
        fig = plot_multiday_per_cell_histogram(
            sessions, trial_type,
            signal_col=signal_col,
            day_x=day_x,
            day_y=day_y,
            cell_indices=cell_indices,
            animal_id=animal_id,
            show=show,
        )
        figs[f'{animal_id} - multiday_cell_{trial_type}_{label}'] = fig

        # PV correlation across position
        fig = plot_multiday_pv_across_position(
            sessions, trial_type, day_x, day_y,
            signal_col=signal_col,
            animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - multiday_pv_{trial_type}_{label}'] = fig

        # Day x day PV correlation across trials
        fig = plot_multiday_pv_correlation_matrix(
            sessions, trial_type, signal_col=signal_col,
            day_x=day_x, day_y=day_y,
            animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - multiday_pv_matrix_{trial_type}_{label}'] = fig

    # Day × day matrix SUMMARY of session (all cells)
    if len(dates) >= 2:
        print(f"\n  Day × day matrix...")
        fig = plot_multiday_correlation_matrix_summary(
            sessions, trial_type, signal_col=signal_col,
            cell_indices=cell_indices, animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - multiday_matrix_{trial_type}'] = fig

    # N×N grid overview figures (single trial type)
    if grid and len(dates) >= 2:
        print(f"\n  Generating N×N grid overviews ({trial_type})...")

        fig = plot_multiday_pv_heatmap_grid(
            sessions, trial_type,
            signal_col=signal_col, animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - grid_pv_heatmap_{trial_type}'] = fig

        fig = plot_multiday_pv_position_grid(
            sessions, trial_type,
            signal_col=signal_col, animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - grid_pv_position_{trial_type}'] = fig

        fig = plot_multiday_histogram_grid(
            sessions, trial_type,
            signal_col=signal_col, cell_indices=cell_indices,
            animal_id=animal_id, show=show,
        )
        figs[f'{animal_id} - grid_histogram_{trial_type}'] = fig

        # 2N×2N cross-condition grid (both trial types)
        if grid_trial_type_b:
            cross_label = f'{trial_type}_{grid_trial_type_b}'
            print(f"\n  Generating 2N×2N cross grid overviews ({cross_label})...")

            fig = plot_multiday_pv_heatmap_grid_cross(
                sessions, trial_type, grid_trial_type_b,
                signal_col=signal_col, animal_id=animal_id, show=show,
            )
            figs[f'{animal_id} - grid_cross_pv_heatmap_{cross_label}'] = fig

            fig = plot_multiday_pv_position_grid_cross(
                sessions, trial_type, grid_trial_type_b,
                signal_col=signal_col, animal_id=animal_id, show=show,
            )
            figs[f'{animal_id} - grid_cross_pv_position_{cross_label}'] = fig

            fig = plot_multiday_histogram_grid_cross(
                sessions, trial_type, grid_trial_type_b,
                signal_col=signal_col, cell_indices=cell_indices,
                animal_id=animal_id, show=show,
            )
            figs[f'{animal_id} - grid_cross_histogram_{cross_label}'] = fig

    _save_figures(figs, save_dir)
    print(f"\nGenerated {len(figs)} figures.")
    return figs


# WITHIN-SESSION LEARNING CURVE

def _build_per_trial_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> tuple[np.ndarray, np.ndarray]:
    """Build tuning curve for each individual trial using scatter-add binning.

    Basically average the deconvolved spike values for all frames where the mouse was
    in that bin during that one trial. Result is a vector of shape (n_bins, n_cells) —
    each cell's spatial activity profile on that one lap.

    This is used by the within-session learning curve analysis to get trial-level
    resolution rather than session averages.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        metadata: Experiment metadata with bin size
        trial_type: Filter to this trial type.
        signal_col: Column containing neural signals.

    Returns:
        Tuple of (per_trial_curves, unique_trials) where per_trial_curves
        has shape (n_trials, n_bins, n_cells) and unique_trials is (n_trials,).
    """
    tt_df = df.filter(pl.col('trial_type') == trial_type)
    if len(tt_df) == 0:
        return np.empty((0, 0, 0)), np.array([])

    signals = np.vstack(tt_df[signal_col].to_list())
    trials = tt_df['trial'].to_numpy()
    bins = tt_df['distance_bin'].to_numpy()

    bin_size_cm = get_bin_size(metadata, df)

    n_bins = int(get_track_length(config, trial_type) / bin_size_cm)
    n_cells = signals.shape[1]
    unique_trials = np.unique(trials)
    n_trials = len(unique_trials)

    trial_idx = np.searchsorted(unique_trials, trials)
    bin_idx = bins.clip(0, n_bins - 1)

    #Scatter-add signals and counts into (trial, bin) slots
    sums = np.zeros((n_trials, n_bins, n_cells))
    counts = np.zeros((n_trials, n_bins, 1))
    np.add.at(sums, (trial_idx, bin_idx), signals)
    np.add.at(counts, (trial_idx, bin_idx, 0), 1)

    # Divide to get per-trial, per-bin mean activity (0/0 → NaN)
    with np.errstate(invalid='ignore'):
        per_trial = sums / counts  # (n_trials, n_bins, n_cells)

    return per_trial, unique_trials

#TODO: test: check whether the decline correlates with trial number or with elapsed time.
# If it's time-driven, plotting against elapsed_minutes instead of trial number should linearize it.
# Also check running speed — if the mouse slows down late in the session, spatial sampling gets worse and tuning curves
# get noisier, which drops correlation even without real remapping.
def within_session_learning_curve(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
    segment: str = 'all',
) -> dict:
    """Track trial-by-trial PV correlation to leave-one-out templates.

    For each trial of each type, this:
        1. Builds that trial's tuning curve via _build_per_trial_tuning_curves
        2. Computes a leave-one-out (LOO) mean of all *other* trials of the same type.
        3. Correlates the single trial against the LOO own-type template.
        4. Correlates the single trial against the *other* type's global mean.

    Plotting correlation-to-own-type over trial number reveals whether the
    representation stabilizes within the session (increasing trend) or degrades
    (decreasing, e.g. due to fatigue or reduced running speed).

    Plotting correlation-to-other-type reveals whether the two trial types start
    similar and gradually diverge (learning to discriminate) or are immediately
    distinct.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        metadata: Experiment metadata, has bin size in cm
        signal_col: Column containing neural signals.
        segment: Which spatial segment to analyze.
            'all' = full track (up to shared length),
            'shared' = only pre-divergence bins,
            'divergent' = only post-divergence bins.

    Returns:
        Dict with keys:
            'trial_numbers': dict[str, np.ndarray] — trial indices per type.
            'corr_to_own': dict[str, np.ndarray] — LOO correlation to own-type mean.
            'corr_to_other': dict[str, np.ndarray] — LOO correlation to other-type mean.
            'trial_types': tuple[str, str].
            'segment': str — which segment was used.
            'n_cells': int.
    """
    trial_types = sorted(df['trial_type'].unique().to_list())
    if len(trial_types) < 2:
        raise ValueError("Need ≥2 trial types for learning curve analysis")
    type_a, type_b = trial_types[0], trial_types[1]

    # Build per-trial tuning curves for each type
    curves_a, trials_a = _build_per_trial_tuning_curves(
        df, config, metadata, type_a, signal_col,
    )
    curves_b, trials_b = _build_per_trial_tuning_curves(
        df, config, metadata, type_b, signal_col,
    )

    # Determine bin slice based on segment
    _, n_shared = get_shared_bins(config, type_a, type_b, metadata)
    n_bins_common = min(curves_a.shape[1], curves_b.shape[1])

    if segment == 'shared':
        bin_slice = slice(0, n_shared)
    elif segment == 'divergent':
        bin_slice = slice(n_shared, n_bins_common)
    else:  # 'all'
        bin_slice = slice(0, n_bins_common)

    ca = curves_a[:, bin_slice, :]  # (n_trials_a, n_bins_seg, n_cells)
    cb = curves_b[:, bin_slice, :]

    # Global mean of each type (used for cross-type correlation)
    mean_a_global = np.nanmean(ca, axis=0)  # (n_bins_seg, n_cells)
    mean_b_global = np.nanmean(cb, axis=0)

    def _pv_corr(tuning_single: np.ndarray, tuning_template: np.ndarray) -> float:
        """Mean PV correlation across bins between a single trial and a template."""
        n_bins_seg = tuning_single.shape[0]
        corrs = np.full(n_bins_seg, np.nan)
        for b in range(n_bins_seg):
            v1 = tuning_single[b]
            v2 = tuning_template[b]
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.sum() < 3:
                continue
            s1, s2 = v1[valid], v2[valid]
            if np.std(s1) == 0 or np.std(s2) == 0:
                continue
            corrs[b] = np.corrcoef(s1, s2)[0, 1]
        return np.nanmean(corrs)

    results = {
        'trial_numbers': {},
        'corr_to_own': {},
        'corr_to_other': {},
        'trial_types': (type_a, type_b),
        'segment': segment,
        'n_cells': ca.shape[2],
    }

    # For each type: LOO correlation to own mean + correlation to other mean
    for label, curves, trials, other_mean in [
        (type_a, ca, trials_a, mean_b_global),
        (type_b, cb, trials_b, mean_a_global),
    ]:
        n_t = curves.shape[0]
        corr_own = np.full(n_t, np.nan)
        corr_other = np.full(n_t, np.nan)

        for i in range(n_t):
            trial_curve = curves[i]  # (n_bins_seg, n_cells)

            # Leave-one-out mean of own type
            if n_t > 1:
                loo_mean = (np.nansum(curves, axis=0) - trial_curve) / (n_t - 1)
            else:
                loo_mean = trial_curve  # degenerate, corr will be 1.0

            corr_own[i] = _pv_corr(trial_curve, loo_mean)
            corr_other[i] = _pv_corr(trial_curve, other_mean)

        results['trial_numbers'][label] = trials
        results['corr_to_own'][label] = corr_own
        results['corr_to_other'][label] = corr_other

    return results


def plot_within_session_learning_curve(
    result: dict,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple = (12, 5),
    show: bool = True,
) -> Figure:
    """Plot within-session learning curve: PV correlation over trials.

    Two-panel figure:
        Left: correlation to own-type template (LOO). An increasing trend means the
        representation is stabilizing over trials; decreasing suggests fatigue or drift.
        Right: correlation to other-type template. A decreasing trend means the
        representations are gradually diverging (the animal is learning to discriminate).

    Points are colored by trial type; linear trend lines (OLS) are overlaid when
    enough valid data points exist.

    Args:
        result: Output from within_session_learning_curve().
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    type_a, type_b = result['trial_types']
    segment = result['segment']
    color_a = pfmt.TRIAL_TYPE_COLORS.get(type_a, '#2E86AB')
    color_b = pfmt.TRIAL_TYPE_COLORS.get(type_b, '#A23B72')

    fig, (ax_own, ax_cross) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Assign sequential x-positions by interleaving trial order
    # (trials are interleaved ABC/ABDC, so plot by actual trial number)
    for label, color, marker in [
        (type_a, color_a, 'o'),
        (type_b, color_b, 's'),
    ]:
        trials = result['trial_numbers'][label]
        own = result['corr_to_own'][label]
        other = result['corr_to_other'][label]

        ax_own.scatter(trials, own, c=color, marker=marker, s=30,
                       alpha=0.7, label=label, edgecolors='white', linewidth=0.3)
        ax_cross.scatter(trials, other, c=color, marker=marker, s=30,
                         alpha=0.7, label=label, edgecolors='white', linewidth=0.3)

        x_fit = np.linspace(trials.min(), trials.max(), 50)

        # linear trend line for own-type corr
        valid_own = ~np.isnan(own)
        if valid_own.sum() > 2:
            z = np.polyfit(trials[valid_own], own[valid_own], 1)
            ax_own.plot(x_fit, np.polyval(z, x_fit), color=color,
                        linewidth=1.5, alpha=0.5, linestyle='--')
        # linear trend line for cross-type corr
        valid_cross = ~np.isnan(other)
        if valid_cross.sum() > 2:
            z = np.polyfit(trials[valid_cross], other[valid_cross], 1)
            ax_cross.plot(x_fit, np.polyval(z, x_fit), color=color,
                          linewidth=1.5, alpha=0.5, linestyle='--')

    # Format left panel (own-type LOO)
    ax_own.set_xlabel('Trial number', fontsize=11)
    ax_own.set_ylabel('Mean PV correlation (r)', fontsize=11)
    ax_own.set_title('Correlation to own type (LOO)', fontsize=11, fontweight='bold')
    ax_own.legend(frameon=False, fontsize=9)
    ax_own.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_own.spines['top'].set_visible(False)
    ax_own.spines['right'].set_visible(False)
    ax_own.grid(alpha=0.2, axis='y')

    # Format right panel (cross-type)
    ax_cross.set_xlabel('Trial number', fontsize=11)
    ax_cross.set_title('Correlation to other type', fontsize=11, fontweight='bold')
    ax_cross.legend(frameon=False, fontsize=9)
    ax_cross.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_cross.spines['top'].set_visible(False)
    ax_cross.spines['right'].set_visible(False)
    ax_cross.grid(alpha=0.2, axis='y')

    fig.suptitle(
        pfmt.build_title(f'Within-Session Learning Curve — {segment} segment',
                         animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# SAVE

def _save_figures(figs: dict[str, Figure],
                  save_dir: str | Path | None):
    """Save all figures to directory as 150-dpi PNGs..

    Args:
        figs: Dict of {name: Figure}.
        save_dir: Directory path, or None to skip.
    """
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        path = save_dir / f'{name}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")


# ─── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from df_processing import (find_session_dir, load_session_context,
                               get_session_paths, load_processed_session, load_multiday_sessions)

    mouse_id = '26'
    date = '2025-09-03'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])

    # plot
    # fig = plot_pv_correlation_matrix(data, exp_config, meta,
    #                                           type_a='ABC', type_b='ABDC',
    #                                           animal_id=mouse_id, date=date)
    #
    # for s in ['shared', 'divergent', 'all']:
    #     result = within_session_learning_curve(
    #         data, exp_config, meta, signal_col='multi_day_spikes', segment=s,
    #     )
    #     fig = plot_within_session_learning_curve(result, animal_id=mouse_id, date=date)
    #
    # figs = run_within_session_analysis(data, exp_config, meta,
    # signal_col='multi_day_spikes', animal_id=mouse_id,
    #                                  date=date, show=True)


#     # ── Multiday analysis ──
    # b3, b5, e1, e4, e7 --> 14
    # sessions = load_multiday_sessions(
    #     mouse_dir, dates=['2025-08-18', '2025-08-20', '2025-08-22',
    #                       '2025-08-27', '2025-09-03', '2025-09-05' ],
    #     auto_process=True,
    #     signal_cols=['multi_day_spikes'],
    # )

    sessions = load_multiday_sessions(
        mouse_dir, dates=['2025-08-25', '2025-08-30', '2025-09-03',
                          '2025-09-08', '2025-09-12', '2025-09-16'],
        auto_process=True,
        signal_cols=['multi_day_spikes'],
    )

    # sessions = load_multiday_sessions(
    #     mouse_dir, dates=('2025-08-18', '2025-08-20'),
    #     auto_process=True,
    #     signal_cols=['multi_day_spikes'],
    # )

    # All pairs + autocorrelation for ABC
    #figs = run_multiday_analysis(sessions, trial_type='ABC', signal_col='multi_day_spikes', show=True)

    # Can use specific pairs: day 1 vs day 5, day 1 vs day 1 OR will do all days; I recommend day pairs or calling
    # the functions individually otehrwise it will make >30 individual plots (depends on num days)

    # Option 1: via run_multiday_analysis with grid=True (individual plots + grids)
    #   grid_trial_type_b puts ABC in upper triangle, ABDC in lower triangle
    # dates = sorted(sessions.keys())
    # if len(dates) >= 2:
    #     figs = run_multiday_analysis(
    #         sessions, trial_type=None, signal_col='multi_day_spikes',
    #         animal_id=mouse_id, day_pairs=None,
    #         grid=True, grid_trial_type_b='ABDC', show=True,
    #     )

    # Option 2: standalone grid calls (grids only, no individual plots)
    # trial_type_b=None for single-type mode (upper triangle only)

    # Option A: N×N single-type grids
    # fig = plot_multiday_pv_heatmap_grid(
    #     sessions, trial_type='ABC', animal_id=mouse_id,
    # )
    # fig = plot_multiday_pv_position_grid(
    #     sessions, trial_type='ABC', animal_id=mouse_id,
    # )
    # fig = plot_multiday_histogram_grid(
    #     sessions, trial_type='ABC', animal_id=mouse_id,
    # )

    # Option B: 2N×2N cross-condition grids (all conditions × all days)
    # fig = plot_multiday_pv_heatmap_grid_cross(
    #     sessions, type_a='ABC', type_b='ABDC', animal_id=mouse_id,
    # )
    # fig = plot_multiday_pv_position_grid_cross(
    #     sessions, type_a='ABC', type_b='ABDC', animal_id=mouse_id,
    # )
    # fig = plot_multiday_histogram_grid_cross(
    #     sessions, type_a='ABC', type_b='ABDC', animal_id=mouse_id,
    # )

    fig = plot_multiday_pv_heatmap_grid_a_vs_b(sessions, type_a='ABC', type_b='ABDC', animal_id=mouse_id)

    fig = plot_multiday_combined_grid(sessions, trial_type='ABC', animal_id=mouse_id)