"""
Cross-Correlation Analysis for Place Cell Spatial Tuning Curves

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

Dependencies: numpy, polars, matplotlib, df_processing.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')
from df_processing import (compute_session_averages, get_track_length, get_cue_regions)


# ─── COLORS ──────────────────────────────────────────────────────────

TRIAL_TYPE_COLORS = {
    'ABC': '#2E86AB',
    'ABDC': '#A23B72',
    'ABCD': '#A23B72',
}


# ─── TUNING CURVE EXTRACTION ────────────────────────────────────────

def get_mean_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
) -> dict[str, np.ndarray]:
    """Compute session-averaged tuning curves per trial type.

    Wrapper around compute_session_averages that returns just the
    mean tuning curves in (n_bins, n_cells) format.

    Args:
        df: Frame-level DataFrame with position/bin columns.
        config: Experiment configuration dict.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        Dict mapping trial_type -> (n_bins, n_cells) array.
    """
    stats = compute_session_averages(
        df, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
    )
    return {tt: s['session_avg'] for tt, s in stats.items()}


def get_split_half_tuning_curves(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split trials into odd/even halves and compute mean tuning curves.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        trial_type: Which trial type to split.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        Tuple of (even_avg, odd_avg), each (n_bins, n_cells).
    """
    tt_df = df.filter(pl.col('trial_type') == trial_type)
    trials = tt_df['trial'].to_numpy()
    unique_trials = np.unique(trials)

    even_trials = unique_trials[::2]
    odd_trials = unique_trials[1::2]

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


# ─── CORE COMPUTATIONS ──────────────────────────────────────────────

def per_cell_spatial_correlation(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
    min_bins: int | None = None,
) -> np.ndarray:
    """Pearson correlation of each cell's tuning curve between two conditions.

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


# ─── BIN ALIGNMENT ──────────────────────────────────────────────────

def get_shared_bins(
    config: dict,
    type_a: str,
    type_b: str,
    bin_size_cm: int = 5,
) -> int:
    """Number of shared spatial bins between two track types.

    Counts bins from position 0 until the cue sequences diverge.

    Args:
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        Number of shared bins.
    """
    ts = config.get('trial_structures', {})
    seq_a = ts.get(type_a, {}).get('cue_sequence', [])
    seq_b = ts.get(type_b, {}).get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    shared_cm = 0.0
    for ca, cb in zip(seq_a, seq_b):
        if ca != cb:
            break
        shared_cm += cue_widths.get(ca, 30.0)

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
        pos += cue_widths.get(ca, 30.0)
    return pos


# ─── WITHIN-SESSION PLOTTING ────────────────────────────────────────

def _add_cue_shading(ax: Axes, config: dict, trial_type: str, alpha: float = 0.08):
    """Add light cue region shading to an axis.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        alpha: Shading transparency.
    """
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})
    pos = 0.0
    for cue_id in seq:
        w = cue_widths.get(cue_id, 30.0)
        if cue_id != 0:
            ax.axvspan(pos, pos + w, alpha=alpha, color='gray', zorder=0)
        pos += w


def plot_split_half(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    figsize: tuple = (8, 5),
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
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    even_avg, odd_avg = get_split_half_tuning_curves(
        df, config, trial_type, signal_col=signal_col, bin_size_cm=bin_size_cm,
    )
    corrs = per_cell_spatial_correlation(even_avg, odd_avg)
    valid = corrs[~np.isnan(corrs)]

    color = TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(valid, bins=np.linspace(-1, 1, 41), color=color, alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(valid), color='black', linestyle='--', linewidth=1.5,
               label=f'median = {np.nanmedian(valid):.3f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Pearson r (odd vs even trials)', fontsize=11)
    ax.set_ylabel('Number of cells', fontsize=11)
    ax.set_title(f'Split-Half Reliability — {trial_type}', fontsize=13, fontweight='bold')
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
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """PV correlation at each shared spatial bin between two track types.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    avgs = get_mean_tuning_curves(df, config, signal_col=signal_col, bin_size_cm=bin_size_cm)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    shared_bins = get_shared_bins(config, type_a, type_b, bin_size_cm)
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
        _add_cue_shading(ax, config, type_a, alpha=0.05)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    ax.set_title(f'Population Vector Correlation — {type_a} vs {type_b}',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(len_a, len_b))
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
    type_a: str,
    type_b: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    segment: str = 'shared',
    figsize: tuple = (8, 5),
    show: bool = True,
) -> Figure:
    """Histogram of per-cell spatial correlations between two track types.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        segment: 'shared' for shared bins only, 'full' for shorter track length.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    avgs = get_mean_tuning_curves(df, config, signal_col=signal_col, bin_size_cm=bin_size_cm)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    min_bins = None
    if segment == 'shared':
        min_bins = get_shared_bins(config, type_a, type_b, bin_size_cm)

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
    ax.set_title(f'Per-Cell Spatial Correlation — {type_a} vs {type_b}',
                 fontsize=13, fontweight='bold')
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
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    figsize: tuple = (8, 7),
    show: bool = True,
) -> Figure:
    """Full bin×bin PV correlation heatmap between two track types.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    avgs = get_mean_tuning_curves(df, config, signal_col=signal_col, bin_size_cm=bin_size_cm)
    avg_a, avg_b = avgs[type_a], avgs[type_b]

    matrix = pv_correlation_matrix(avg_a, avg_b)

    len_a = avg_a.shape[0] * bin_size_cm
    len_b = avg_b.shape[0] * bin_size_cm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.T, origin='lower', aspect='auto',
        cmap='RdBu_r', vmin=-0.3, vmax=1.0,
        extent=[0, len_a, 0, len_b],
    )
    plt.colorbar(im, ax=ax, label='PV Correlation (r)', shrink=0.85)

    diverge = get_divergence_point(config, type_a, type_b)
    ax.axvline(diverge, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(diverge, color='red', linestyle='--', linewidth=1, alpha=0.7)

    max_len = min(len_a, len_b)
    ax.plot([0, max_len], [0, max_len], color='white', linewidth=0.8,
            linestyle='--', alpha=0.5)

    ax.set_xlabel(f'{type_a} position (cm)', fontsize=11)
    ax.set_ylabel(f'{type_b} position (cm)', fontsize=11)
    ax.set_title(f'PV Correlation Matrix — {type_a} vs {type_b}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# ─── WITHIN-SESSION WRAPPER ─────────────────────────────────────────

def run_within_session_analysis(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Run full within-session cross-correlation analysis.

    Generates split-half reliability, cross-track PV correlation (divergence point), per-cell
    correlation, and PV correlation matrix for all trial type combinations.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
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
        fig = plot_split_half(df, config, tt, signal_col=signal_col,
                              bin_size_cm=bin_size_cm, show=show)
        figs[f'split_half_{tt}'] = fig

    # Cross-track comparisons
    if len(trial_types) >= 2:
        for type_a, type_b in combinations(trial_types, 2):
            print(f"\nCross-correlation: {type_a} vs {type_b}...")

            fig = plot_pv_correlation_across_position(
                df, config, type_a, type_b, signal_col=signal_col,
                bin_size_cm=bin_size_cm, show=show)
            figs[f'pv_position_{type_a}_vs_{type_b}'] = fig

            fig = plot_per_cell_cross_correlation(
                df, config, type_a, type_b, signal_col=signal_col,
                bin_size_cm=bin_size_cm, segment='shared', show=show)
            figs[f'cell_corr_shared_{type_a}_vs_{type_b}'] = fig

            fig = plot_pv_correlation_matrix(
                df, config, type_a, type_b, signal_col=signal_col,
                bin_size_cm=bin_size_cm, show=show)
            figs[f'pv_matrix_{type_a}_vs_{type_b}'] = fig
    else:
        print("Only one trial type — skipping cross-track comparisons.")

    _save_figures(figs, save_dir)
    print(f"\nGenerated {len(figs)} figures.")
    return figs


# ─── MULTIDAY CORRELATIONS ──────────────────────────────────────────

def multiday_tuning_curves(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
) -> dict[str, np.ndarray]:
    """Compute session-averaged tuning curves for one trial type across days.

    Args:
        sessions: From load_multiday_sessions(). Keys are date strings.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.

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

        avgs = get_mean_tuning_curves(
            df, config, signal_col=signal_col, bin_size_cm=bin_size_cm,
        )
        if trial_type in avgs:
            result[date] = avgs[trial_type]

    return result


def multiday_per_cell_correlation(
    sessions: dict[str, dict],
    day_a: str,
    day_b: str,
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    cell_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Per-cell tuning curve correlation between two days.

    When day_a == day_b, computes split-half correlation (autocorrelation).

    Args:
        sessions: From load_multiday_sessions().
        day_a: First date string.
        day_b: Second date string.
        trial_type: Which trial type to compare.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        cell_indices: Subset of cells to compute. None = all cells.

    Returns:
        Per-cell Pearson r, shape (n_cells,) or (len(cell_indices),).
    """
    if day_a == day_b:
        # Autocorrelation: split-half within the same session
        s = sessions[day_a]
        even_avg, odd_avg = get_split_half_tuning_curves(
            s['data'], s['config'], trial_type,
            signal_col=signal_col, bin_size_cm=bin_size_cm,
        )
        corrs = per_cell_spatial_correlation(even_avg, odd_avg)
    else:
        # Cross-day: full session averages
        tc = multiday_tuning_curves(
            {day_a: sessions[day_a], day_b: sessions[day_b]},
            trial_type, signal_col=signal_col, bin_size_cm=bin_size_cm,
        )
        if day_a not in tc or day_b not in tc:
            n_cells = len(sessions[day_a]['data'][signal_col][0])
            return np.full(n_cells, np.nan)
        corrs = per_cell_spatial_correlation(tc[day_a], tc[day_b])

    if cell_indices is not None:
        corrs = corrs[cell_indices]
    return corrs


def multiday_pv_correlation(
    sessions: dict[str, dict],
    day_a: str,
    day_b: str,
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
) -> np.ndarray:
    """PV correlation across positions between two days for one trial type.

    When day_a == day_b, uses split-half.

    Args:
        sessions: From load_multiday_sessions().
        day_a: First date string.
        day_b: Second date string.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        PV correlation per bin, shape (n_bins,).
    """
    if day_a == day_b:
        s = sessions[day_a]
        even_avg, odd_avg = get_split_half_tuning_curves(
            s['data'], s['config'], trial_type,
            signal_col=signal_col, bin_size_cm=bin_size_cm,
        )
        return population_vector_correlation(even_avg, odd_avg)
    else:
        tc = multiday_tuning_curves(
            {day_a: sessions[day_a], day_b: sessions[day_b]},
            trial_type, signal_col=signal_col, bin_size_cm=bin_size_cm,
        )
        if day_a not in tc or day_b not in tc:
            return np.array([])
        return population_vector_correlation(tc[day_a], tc[day_b])


def multiday_correlation_matrix(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
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
        bin_size_cm: Spatial bin size in cm.
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

    for i, day_a in enumerate(dates):
        for j, day_b in enumerate(dates):
            if j < i:
                matrix[i, j] = matrix[j, i]  # symmetric
                continue

            if metric.startswith('per_cell'):
                corrs = multiday_per_cell_correlation(
                    sessions, day_a, day_b, trial_type,
                    signal_col=signal_col, bin_size_cm=bin_size_cm,
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
                    sessions, day_a, day_b, trial_type,
                    signal_col=signal_col, bin_size_cm=bin_size_cm,
                )
                valid = pv[~np.isnan(pv)]
                if len(valid) > 0:
                    matrix[i, j] = np.mean(valid)

    return matrix, dates


# ─── MULTIDAY PLOTTING ──────────────────────────────────────────────

def plot_multiday_per_cell_histogram(
    sessions: dict[str, dict],
    day_a: str,
    day_b: str,
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    cell_indices: np.ndarray | None = None,
    figsize: tuple = (8, 5),
    show: bool = True,
) -> Figure:
    """Histogram of per-cell correlations between two days.

    Args:
        sessions: From load_multiday_sessions().
        day_a: First date string.
        day_b: Second date string.
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        cell_indices: Subset of cells. None = all.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    corrs = multiday_per_cell_correlation(
        sessions, day_a, day_b, trial_type,
        signal_col=signal_col, bin_size_cm=bin_size_cm,
        cell_indices=cell_indices,
    )
    valid = corrs[~np.isnan(corrs)]

    label_a = day_a[5:]  # drop year prefix for cleaner labels
    label_b = day_b[5:]
    is_auto = day_a == day_b
    pair_label = f'{label_a} split-half' if is_auto else f'{label_a} vs {label_b}'

    color = TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(valid, bins=np.linspace(-1, 1, 41), color=color, alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(np.nanmedian(valid), color='black', linestyle='--', linewidth=1.5,
               label=f'median = {np.nanmedian(valid):.3f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Pearson r', fontsize=11)
    ax.set_ylabel('Number of cells', fontsize=11)
    ax.set_title(f'Per-Cell Correlation — {trial_type} — {pair_label}',
                 fontsize=13, fontweight='bold')
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
    day_a: str,
    day_b: str,
    trial_type: str,
    config: dict | None = None,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """PV correlation across position between two days.

    Args:
        sessions: From load_multiday_sessions().
        day_a: First date string.
        day_b: Second date string.
        trial_type: Which trial type.
        config: Experiment config for cue shading. Uses day_a's config if None.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    pv = multiday_pv_correlation(
        sessions, day_a, day_b, trial_type,
        signal_col=signal_col, bin_size_cm=bin_size_cm,
    )
    if len(pv) == 0:
        print(f"No data for {trial_type} on {day_a} or {day_b}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return fig

    if config is None:
        config = sessions[day_a]['config']

    x = np.arange(len(pv)) * bin_size_cm + bin_size_cm / 2
    track_len = len(pv) * bin_size_cm

    label_a = day_a[5:]
    label_b = day_b[5:]
    is_auto = day_a == day_b
    pair_label = f'{label_a} split-half' if is_auto else f'{label_a} vs {label_b}'

    color = TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, pv, color=color, linewidth=2, zorder=4)
    ax.fill_between(x, pv, alpha=0.15, color=color, zorder=3)

    _add_cue_shading(ax, config, trial_type, alpha=0.05)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('PV Correlation (Pearson r)', fontsize=11)
    ax.set_title(f'PV Correlation — {trial_type} — {pair_label}',
                 fontsize=13, fontweight='bold')
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


def plot_multiday_correlation_matrix(
    sessions: dict[str, dict],
    trial_type: str,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    cell_indices: np.ndarray | None = None,
    metric: str = 'per_cell_median',
    figsize: tuple = (7, 6),
    show: bool = True,
) -> Figure:
    """Day × day correlation heatmap.

    Diagonal = autocorrelation (split-half). Off-diagonal = cross-day.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        cell_indices: Subset of cells. None = all.
        metric: Summary metric ('per_cell_median', 'per_cell_mean', 'pv_mean').
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    matrix, dates = multiday_correlation_matrix(
        sessions, trial_type, signal_col=signal_col, bin_size_cm=bin_size_cm,
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
    ax.set_title(f'Day × Day Correlation — {trial_type} ({metric})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# ─── MULTIDAY WRAPPER ───────────────────────────────────────────────

def run_multiday_analysis(
    sessions: dict[str, dict],
    trial_type: str | None = None,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    cell_indices: np.ndarray | None = None,
    day_pairs: list[tuple[str, str]] | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Run multiday cross-correlation analysis.

    By default, computes all pairwise day combinations plus autocorrelations.
    Generates per-cell histograms for each pair and a day×day summary matrix.

    Args:
        sessions: From load_multiday_sessions().
        trial_type: Which trial type. If None, uses first type found.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        cell_indices: Subset of cells. None = all.
        day_pairs: Specific (day_a, day_b) pairs to analyze. None = all pairs + diagonal.
        show: Call plt.show().
        save_dir: Directory to save figures (optional).

    Returns:
        Dict of {name: Figure}.
    """
    dates = sorted(sessions.keys())
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
    for day_a, day_b in day_pairs:
        label = f'{day_a[5:]}_vs_{day_b[5:]}'
        if day_a == day_b:
            label = f'{day_a[5:]}_auto'

        print(f"  {label}...")

        fig = plot_multiday_per_cell_histogram(
            sessions, day_a, day_b, trial_type,
            signal_col=signal_col, bin_size_cm=bin_size_cm,
            cell_indices=cell_indices, show=show,
        )
        figs[f'multiday_cell_{trial_type}_{label}'] = fig

        fig = plot_multiday_pv_across_position(
            sessions, day_a, day_b, trial_type,
            signal_col=signal_col, bin_size_cm=bin_size_cm, show=show,
        )
        figs[f'multiday_pv_{trial_type}_{label}'] = fig

    # Day × day matrix
    if len(dates) >= 2:
        print(f"\n  Day × day matrix...")
        fig = plot_multiday_correlation_matrix(
            sessions, trial_type, signal_col=signal_col, bin_size_cm=bin_size_cm,
            cell_indices=cell_indices, show=show,
        )
        figs[f'multiday_matrix_{trial_type}'] = fig

    _save_figures(figs, save_dir)
    print(f"\nGenerated {len(figs)} figures.")
    return figs


# ─── UTILITIES ───────────────────────────────────────────────────────

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


# ─── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from df_processing import load_session_dir, get_session_prefix, load_processed_session, load_multiday_sessions

    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')

    # ── Single-session analysis ──
    date = '2025-09-15'
    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    data, meta = load_processed_session(behavior_path.parent / f'{prefix}_processed.parquet')

    #figs = run_within_session_analysis(data, config, signal_col='multi_day_dff', show=True)

    # ── Multiday analysis ──
    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-03', '2025-09-24'), auto_process=False,
    )

    # All pairs + autocorrelation for ABC
    # figs = run_multiday_analysis(
    #     sessions, trial_type='ABC', signal_col='multi_day_dff', show=True,
    # )

    # Specific pairs: day 1 vs day 5, day 1 vs day 1 (
    # dates = sorted(sessions.keys())
    # if len(dates) >= 2:
    #     figs = run_multiday_analysis(
    #         sessions, trial_type='ABC',
    #         day_pairs=None,
    #         signal_col='multi_day_dff', show=True,
    #     )
    import matplotlib.pyplot as plt
# try just the pv for 2 days
    fig = plot_multiday_pv_across_position(
        sessions,
        signal_col='multi_day_dff', show=True,
    )
    figs[f'multiday_pv_{trial_type}_{label}'] = fig
    plt.show(figs)
