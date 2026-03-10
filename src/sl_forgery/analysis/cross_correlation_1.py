"""
Cross-correlation analysis for place cell spatial tuning curves

Computes population vector (PV) correlations and per-cell spatial correlations
across track types and across days.

Within-session:
    - Split-half reliability (odd vs even trials)
    - Cross-track PV correlation and per-cell correlation
    - PV correlation matrix (bin × bin heatmap)

Across-session (multiday):
    - Per-cell tuning curve correlation between any two days (or same day = split-half)
    - PV correlation across days
    - Full pairwise day matrix (autocorrelation on diagonal)

Standard analyses following Leutgeb et al. 2005, Colgin et al. 2008, Sun et al. 2025.
"""

from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import sys
from df_processing import (compute_session_averages, get_track_length, get_cue_regions, get_bin_size)
import plot_utils as pfmt



# TUNING CURVE EXTRACTION

def get_mean_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
) -> dict[str, np.ndarray]:
    """Compute session-averaged tuning curves per cell per trial type.

    Wrapper around compute_session_averages (from df_processing) that returns just the
    mean tuning curves in (n_bins, n_cells) format.

    Args:
        df: Frame-level DataFrame with position/bin columns.
        metadata: From df_processings, has bin size in cm
        config: Experiment configuration dict.
        signal_col: Column containing neural signals.

    Returns:
        Dict mapping trial_type -> (n_bins, n_cells) array.
    """
    bin_size_cm = get_bin_size(df, metadata)

    stats = compute_session_averages(
        df, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
    )
    return {tt: s['session_avg'] for tt, s in stats.items()}


def get_split_half_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> tuple[np.ndarray, np.ndarray]:
    """Split trials into odd/even halves and compute mean tuning curves.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        metadata: From df_processings, has bin size in cm
        trial_type: Which trial type to split.
        signal_col: Column containing neural signals.

    Returns:
        Tuple of (even_avg, odd_avg), each (n_bins, n_cells).
    """
    tt_df = df.filter(pl.col('trial_type') == trial_type)
    trials = tt_df['trial'].to_numpy()
    unique_trials = np.unique(trials)

    even_trials = unique_trials[::2]
    odd_trials = unique_trials[1::2]

    bin_size_cm = get_bin_size(df, metadata)

    n_bins = int(get_track_length(config, trial_type) / bin_size_cm)
    n_cells = len(df[signal_col][0])


    def _avg_for_trials(trial_set):
        """Bin and average signals for a subset of trials."""
        mask_pl = pl.col('trial').is_in(trial_set.tolist())
        sub = tt_df.filter(mask_pl)

        signals = np.vstack(sub[signal_col].to_list())
        sub_trials = sub['trial'].to_numpy()
        sub_bins = sub['distance_bin'].to_numpy()

        u_trials = np.unique(sub_trials)
        n_t = len(u_trials)
        trial_idx = np.searchsorted(u_trials, sub_trials)
        bin_idx = sub_bins.clip(0, n_bins - 1)

        sums = np.zeros((n_t, n_bins, n_cells))
        counts = np.zeros((n_t, n_bins, 1))
        np.add.at(sums, (trial_idx, bin_idx), signals)
        np.add.at(counts, (trial_idx, bin_idx, 0), 1)

        with np.errstate(invalid='ignore'):
            per_trial = sums / counts

        return np.nanmean(per_trial, axis=0)  # (n_bins, n_cells)

    return _avg_for_trials(even_trials), _avg_for_trials(odd_trials)


# CORE ANALYSES

def per_cell_spatial_correlation(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
    min_bins: int | None = None,
) -> np.ndarray:
    """Pearson correlation of each cell's tuning curve between two conditions (i.e. trial types).

    Args:
        tuning_a: Mean tuning curves, shape (n_bins, n_cells).
        tuning_b: Mean tuning curves, shape (n_bins, n_cells).
        min_bins: Restrict to first min_bins bins (for shared-segment comparison).

    Returns:
        Per-cell Pearson r, shape (n_cells,). NaN for cells with zero variance.
    """
    if min_bins is not None:
        tuning_a = tuning_a[:min_bins]
        tuning_b = tuning_b[:min_bins]

    n_bins = min(tuning_a.shape[0], tuning_b.shape[0])
    a = tuning_a[:n_bins]
    b = tuning_b[:n_bins]
    n_cells = a.shape[1]

    # Vectorized: mask NaNs, compute correlation per cell
    valid = ~(np.isnan(a) | np.isnan(b))  # (n_bins, n_cells)
    valid_count = valid.sum(axis=0)

    corrs = np.full(n_cells, np.nan)
    for c in np.where(valid_count >= 3)[0]:
        m = valid[:, c]
        ac, bc = a[m, c], b[m, c]
        if np.std(ac) == 0 or np.std(bc) == 0:
            continue
        corrs[c] = np.corrcoef(ac, bc)[0, 1]
    return corrs


def population_vector_correlation(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
    min_bins: int | None = None,
) -> np.ndarray:
    """PV correlation across positions: at each spatial bin, correlate the
    population vector (all cells' activity) between two conditions.

    Args:
        tuning_a: Shape (n_bins, n_cells).
        tuning_b: Shape (n_bins, n_cells).
        min_bins: Restrict to first min_bins bins.

    Returns:
        Pearson r at each spatial bin, shape (n_shared_bins,). NaN where undefined.
    """
    if min_bins is not None:
        tuning_a = tuning_a[:min_bins]
        tuning_b = tuning_b[:min_bins]

    n_bins = min(tuning_a.shape[0], tuning_b.shape[0])
    a = tuning_a[:n_bins]
    b = tuning_b[:n_bins]

    pv_corrs = np.full(n_bins, np.nan)
    for i in range(n_bins):
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

    Entry (i, j) = correlation of PV at bin i in condition A with bin j in condition B.

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
) -> int:
    """Number of shared spatial bins between two track types.

    Counts bins from position 0 until the cue sequences diverge.

    Args:
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        metadata: metadata from df_processing, ahs bin size
    Returns:
        Number of shared bins.
    """
    ts = config.get('trial_structures', {})
    seq_a = ts.get(type_a, {}).get('cue_sequence', [])
    seq_b = ts.get(type_b, {}).get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    bin_size_cm = get_bin_size(None, metadata)


    shared_cm = 0.0
    for ca, cb in zip(seq_a, seq_b):
        if ca != cb:
            break
        shared_cm += cue_widths[ca]

    return int(shared_cm / bin_size_cm)


def get_divergence_point(
    config: dict,
    type_a: str,
    type_b: str,
) -> float:
    """Position (cm) where two track types diverge.

    Args:
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.

    Returns:
        Divergence position in cm.
    """
    ts = config.get('trial_structures', {})
    seq_a = ts.get(type_a, {}).get('cue_sequence', [])
    seq_b = ts.get(type_b, {}).get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    pos = 0.0
    for ca, cb in zip(seq_a, seq_b):
        if ca != cb:
            break
        pos += cue_widths[ca]
    return pos


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
    High median r>.5 == stable spatial tuning within session.

    Per-cell Pearson r between odd and even trial tuning curves.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        trial_type: Which trial type to test.
        metadata: metadata from df_processing, has bin size
        signal_col: Column containing neural signals.
        figsize: Figure size.
        animal_id: Which animal to plot.
        date: Expeirment date.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    even_avg, odd_avg = get_split_half_tuning_curves(
        df, config, metadata, trial_type, signal_col=signal_col,
    )
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
    bin_size_cm = get_bin_size(df, metadata)

    avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    shared_bins = get_shared_bins(config, type_a, type_b, metadata)
    diverge_cm = get_divergence_point(config, type_a, type_b)

    pv_shared = population_vector_correlation(avg_a, avg_b, min_bins=shared_bins)
    x_shared = np.arange(shared_bins) * bin_size_cm + bin_size_cm / 2

    len_a = avg_a.shape[0] * bin_size_cm
    len_b = avg_b.shape[0] * bin_size_cm

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_shared, pv_shared, color='black', linewidth=2, zorder=4)
    ax.fill_between(x_shared, pv_shared, alpha=0.15, color='black', zorder=3)

    ax.axvline(diverge_cm, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Tracks diverge ({diverge_cm:.0f} cm)', zorder=5)
    ax.axvspan(diverge_cm, max(len_a, len_b), alpha=0.06, color='red', zorder=0,
               label='Non-shared region')

    if config:
        pfmt.add_cue_shading(ax, config, type_a, alpha=0.05)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    ax.set_title(pfmt.build_title(f'PV Correlation — {type_a} vs {type_b}',
                              animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    ax.set_xlim(0, shared_bins * bin_size_cm)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
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
    bin_size_cm = get_bin_size(df, metadata)

    avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    min_bins = None
    if segment == 'shared':
        min_bins = get_shared_bins(config, type_a, type_b, metadata)

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
    type_a: str,
    type_b: str,
    metadata: dict,
    signal_col: str = 'multi_day_spikes',
    figsize: tuple = (8, 7),
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
) -> Figure:
    """Full bin×bin PV correlation heatmap between two track types.

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
    bin_size_cm = get_bin_size(df, metadata)

    avgs = get_mean_tuning_curves(df, config, metadata, signal_col=signal_col)
    matrix = pv_correlation_matrix(avgs[type_a], avgs[type_b])
    diverge = get_divergence_point(config, type_a, type_b)

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

    Generates split-half reliability, cross-track PV correlation (divergence point), per-cell
    correlation, and PV correlation matrix for all trial type combinations.

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

            fig = plot_per_cell_cross_correlation(
                df, config, metadata, type_a, type_b, signal_col=signal_col,
                segment='shared', animal_id=animal_id, date=date, show=show)
            figs[f'cell_corr_shared_{type_a}_vs_{type_b}'] = fig

            fig = plot_pv_correlation_matrix(
                df, config, type_a, type_b, metadata, signal_col=signal_col,
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
    """PV correlation across position between two days.

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

    bin_size_cm = get_bin_size(None, sessions[day_x]['metadata'])
    x = np.arange(len(pv)) * bin_size_cm + bin_size_cm / 2
    track_len = len(pv) * bin_size_cm

    label_a = day_x[5:]
    label_b = day_y[5:]
    is_auto = day_x == day_y
    pair_label = f'{label_a} split-half' if is_auto else f'{label_a} vs {label_b}'

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, pv, color=color, linewidth=2, zorder=4)
    ax.fill_between(x, pv, alpha=0.15, color=color, zorder=3)

    pfmt.add_cue_shading(ax, config, trial_type, alpha=0.05)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    ax.set_title(pfmt.build_title('PV Correlation', trial_type=trial_type, animal_id=animal_id, day_x=day_x,
                              day_y=day_y), fontsize=13, fontweight='bold')
    ax.set_xlim(0, track_len)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    ax.text(0.02, 0.95,
            f'mean r = {np.nanmean(pv):.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

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
    """Bin×bin PV correlation heatmap for one trial type across two days.

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

    bin_size_cm = get_bin_size(None, sessions[day_x]['metadata'])

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
    """Day × day SUMMARY correlation heatmap.

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


# MULTIDAY WRAPPER

def run_multiday_analysis(
    sessions: dict[str, dict],
    trial_type: str | None = None,
    signal_col: str = 'multi_day_spikes',
    animal_id: str | None = None,
    cell_indices: np.ndarray | None = None,
    day_pairs: list[tuple[str, str]] | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Runs all multiday cross-correlation analysis.

    By default, computes all pairwise day combinations plus autocorrelations.
    Generates per-cell histograms for each pair and a day×day summary matrix.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type. If None, uses first type found.
        signal_col: Column containing neural signals.
        animal_id: Animal name for title. Extracted from sessions if None.
        cell_indices: Subset of cells. None = all.
        day_pairs: Specific (day_x, day_y) pairs to analyze. None = all pairs + diagonal.
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

    _save_figures(figs, save_dir)
    print(f"\nGenerated {len(figs)} figures.")
    return figs


# SAVE

def _save_figures(figs: dict[str, Figure], save_dir: str | Path | None):
    """Save all figures to directory.

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


# WITHIN-SESSION LEARNING CURVE

def _build_per_trial_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    trial_type: str,
    signal_col: str = 'multi_day_spikes',
) -> tuple[np.ndarray, np.ndarray]:
    """Build tuning curve for each individual trial using scatter-add binning.
    Basically average the deconvolved spike values for all frames where the mouse was in that bin during that one
    trial. Result is a vector of shape (n_bins, n_cells) — each cell's spatial activity profile on that one lap.

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

    bin_size_cm = get_bin_size(df, metadata)

    n_bins = int(get_track_length(config, trial_type) / bin_size_cm)
    n_cells = signals.shape[1]
    unique_trials = np.unique(trials)
    n_trials = len(unique_trials)

    trial_idx = np.searchsorted(unique_trials, trials)
    bin_idx = bins.clip(0, n_bins - 1)

    sums = np.zeros((n_trials, n_bins, n_cells))
    counts = np.zeros((n_trials, n_bins, 1))
    np.add.at(sums, (trial_idx, bin_idx), signals)
    np.add.at(counts, (trial_idx, bin_idx, 0), 1)

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

    For each trial, computes its tuning curve and correlates it against
    the mean of all *other* trials (leave-one-out), for each type. This
    shows whether ABC and ABDC representations separate gradually or
    abruptly across trials within a session.

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
    n_shared = get_shared_bins(config, type_a, type_b, metadata)
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

    Left panel: correlation to own-type template (does representation
    stabilize?). Right panel: correlation to other-type template (do
    representations diverge?). Both use leave-one-out templates.

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

        # Trend lines
        valid_own = ~np.isnan(own)
        if valid_own.sum() > 2:
            z = np.polyfit(trials[valid_own], own[valid_own], 1)
            ax_own.plot(x_fit, np.polyval(z, x_fit), color=color,
                        linewidth=1.5, alpha=0.5, linestyle='--')

        valid_cross = ~np.isnan(other)
        if valid_cross.sum() > 2:
            z = np.polyfit(trials[valid_cross], other[valid_cross], 1)
            ax_cross.plot(x_fit, np.polyval(z, x_fit), color=color,
                          linewidth=1.5, alpha=0.5, linestyle='--')

    # Format left panel
    ax_own.set_xlabel('Trial number', fontsize=11)
    ax_own.set_ylabel('Mean PV correlation (r)', fontsize=11)
    ax_own.set_title('Correlation to own type (LOO)', fontsize=11, fontweight='bold')
    ax_own.legend(frameon=False, fontsize=9)
    ax_own.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_own.spines['top'].set_visible(False)
    ax_own.spines['right'].set_visible(False)
    ax_own.grid(alpha=0.2, axis='y')

    # Format right panel
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


# ─── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from df_processing import (find_session_dir, load_session_context,
                               get_session_paths, load_processed_session, load_multiday_sessions)

    mouse_id = '26'
    date = '2025-09-08'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])

    # # plot
    # fig = plot_pv_correlation_across_position(data, exp_config, type_a='ABC', type_b='ABDC',
    #                                     animal_id=mouse_id, date=date)

    for s in ['shared', 'divergent', 'all']:
        result = within_session_learning_curve(
            data, exp_config, meta, signal_col='multi_day_spikes', segment=s,
        )
        fig = plot_within_session_learning_curve(result, animal_id=mouse_id, date=date)

    figs = run_within_session_analysis(data, exp_config, meta, signal_col='multi_day_spikes', animal_id=mouse_id,
                                       date=date, show=True)


#     # ── Multiday analysis ──
    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-03', '2025-09-08'), auto_process=True,
    )

    # All pairs + autocorrelation for ABC
    #figs = run_multiday_analysis(sessions, trial_type='ABC', signal_col='multi_day_spikes', show=True)


    #Specific pairs: day 1 vs day 5, day 1 vs day 1 (
    dates = sorted(sessions.keys())
    if len(dates) >= 2:
        figs = run_multiday_analysis(
            sessions, trial_type=None, signal_col='multi_day_spikes',
            animal_id=mouse_id, day_pairs=None, show=True,
        )

# try just the pv for 2 days
    dates = sorted(sessions.keys())
    if len(dates) >= 2:
        fig = plot_multiday_pv_correlation_matrix(
            sessions, trial_type='ABC',
            signal_col='multi_day_spikes',show=True,
        )