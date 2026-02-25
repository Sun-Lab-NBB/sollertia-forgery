"""
Prospective decoding analysis for hippocampal place cell data.

Asks: can we predict the upcoming trial type from neural activity at shared
track positions (before the physical divergence)?

Analyses:
    1. Sliding-window decoder — SVM accuracy at each spatial bin.
    2. Trial-by-trial PV distance — how far is each trial's B-region PV from
       each trial-type centroid?
    3. Region-restricted PV & per-cell correlation — zoom into one or more cue
       regions, with subplot grid for multiple cues.
    4. Splitter cell index — per-cell selectivity at shared positions.

All functions operate on frame-level pl.DataFrames from df_processing.

Dependencies: numpy, polars, scikit-learn, matplotlib, df_processing,
              cross_correlation_1, trial_plotting.
"""

from pathlib import Path

import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from df_processing import compute_session_averages, get_track_length
from plot_utils import build_title, add_cue_shading, add_cue_bar
from cross_correlation_1 import (
    get_mean_tuning_curves, per_cell_spatial_correlation,
    population_vector_correlation, get_shared_bins, get_divergence_point,
)
import plot_utils as pfmt


# HELPERS

def _get_bin_range_for_cue(
    df: pl.DataFrame,
    cue_id: int | str,
    trial_type: str | None = None,
) -> tuple[int, int]:
    """Get (start_bin, end_bin) for a cue region from the DataFrame.

    Accepts either integer cue IDs (from 'cue' column) or string cue IDs
    (from 'cue_id' column). If trial_type is None, uses the first trial
    type that contains the specified cue.

    Args:
        df: Frame-level DataFrame with cue/cue_id and distance_bin columns.
        cue_id: Cue identifier — int (e.g. 2) uses 'cue' column,
            str (e.g. 'B', '0a') uses 'cue_id' column.
        trial_type: Trial type to filter to. If None, auto-selects first
            trial type containing this cue.

    Returns:
        Tuple of (start_bin_inclusive, end_bin_exclusive).
    """
    cue_col = 'cue' if isinstance(cue_id, int) else 'cue_id'

    if trial_type is None:
        # Find first trial type containing this cue
        match = df.filter(pl.col(cue_col) == cue_id)
        if len(match) == 0:
            available = df[cue_col].unique().sort().to_list()
            raise ValueError(f"Cue {cue_id!r} not found. Available: {available}")
        trial_type = match['trial_type'].first()

    sub = df.filter(
        (pl.col('trial_type') == trial_type)
        & (pl.col(cue_col) == cue_id)
    )
    if len(sub) == 0:
        available = df.filter(
            pl.col('trial_type') == trial_type
        )[cue_col].unique().sort().to_list()
        raise ValueError(f"Cue {cue_id!r} not found in {trial_type}. "
                         f"Available: {available}")

    start_bin = sub['distance_bin'].min()
    end_bin = sub['distance_bin'].max() + 1
    return start_bin, end_bin


def _extract_trial_population_vectors(
    df: pl.DataFrame,
    signal_col: str,
    trial_type: str,
    bin_range: tuple[int, int],
) -> np.ndarray:
    """Extract per-trial population vectors for a range of spatial bins.

    For each trial of the given type, averages the signal across frames
    within the bin range to get one population vector per trial.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        signal_col: Column containing neural signals (list per frame).
        trial_type: Filter to this trial type.
        bin_range: (start_bin, end_bin) inclusive/exclusive.

    Returns:
        Population vectors, shape (n_trials, n_cells).
    """
    start_bin, end_bin = bin_range

    sub = df.filter(
        (pl.col('trial_type') == trial_type)
        & (pl.col('distance_bin') >= start_bin)
        & (pl.col('distance_bin') < end_bin)
    )

    if len(sub) == 0:
        return np.empty((0, 0))

    trials = sub['trial'].to_numpy()
    signals = np.vstack(sub[signal_col].to_list())
    unique_trials = np.unique(trials)

    n_cells = signals.shape[1]
    pvs = np.zeros((len(unique_trials), n_cells))

    for i, t in enumerate(unique_trials):
        mask = trials == t
        pvs[i] = np.nanmean(signals[mask], axis=0)

    return pvs


def _extract_per_bin_trial_vectors(
    df: pl.DataFrame,
    signal_col: str,
    trial_type: str,
    bin_idx: int,
) -> np.ndarray:
    """Extract population vector at a single bin for each trial.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        signal_col: Column containing neural signals.
        trial_type: Filter to this trial type.
        bin_idx: Spatial bin index.

    Returns:
        Shape (n_trials, n_cells).
    """
    sub = df.filter(
        (pl.col('trial_type') == trial_type)
        & (pl.col('distance_bin') == bin_idx)
    )
    if len(sub) == 0:
        return np.empty((0, 0))

    trials = sub['trial'].to_numpy()
    signals = np.vstack(sub[signal_col].to_list())
    unique_trials = np.unique(trials)

    n_cells = signals.shape[1]
    pvs = np.zeros((len(unique_trials), n_cells))
    for i, t in enumerate(unique_trials):
        mask = trials == t
        pvs[i] = np.nanmean(signals[mask], axis=0)

    return pvs


# SLIDING-WINDOW DECODER

def sliding_decoder(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_dff',
    classifier: str = 'svm',
    n_splits: int = 5,
    n_shuffles: int = 100,
    seed: int = 42,
) -> dict:
    """Decode upcoming trial type at each spatial bin using cross-validated SVM.

    At each bin in the shared track segment, trains a classifier on the
    population vector to predict ABC vs ABDC. Reports accuracy at each bin
    plus a shuffle-based chance distribution.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        metadata: Metadata from the processed dataframe, contains 'bin_size_cm' value
        signal_col: Column containing neural signals.
        classifier: 'svm' or 'logistic'.
        n_splits: Stratified K-fold splits.
        n_shuffles: Number of label shuffles for chance distribution.
        seed: Random seed.

    Returns:
        Dict with keys: 'bin_centers', 'accuracy', 'chance_mean', 'chance_95',
        'n_shared_bins', 'trial_types'.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(seed)
    bin_size_cm = metadata['bin_size_cm']

    trial_types = sorted(df['trial_type'].unique().to_list())
    if len(trial_types) < 2:
        raise ValueError("Need at least 2 trial types for decoding")
    type_a, type_b = trial_types[0], trial_types[1]

    # Shared bins
    n_shared = get_shared_bins(config, type_a, type_b, bin_size_cm)

    # Also decode a few bins past divergence for comparison
    max_bins_a = int(get_track_length(config, type_a) / bin_size_cm)
    max_bins_b = int(get_track_length(config, type_b) / bin_size_cm)
    n_total_bins = min(max_bins_a, max_bins_b)

    accuracies = np.full(n_total_bins, np.nan)
    chance_mean = np.full(n_total_bins, np.nan)
    chance_95 = np.full(n_total_bins, np.nan)

    for b in range(n_total_bins):
        pv_a = _extract_per_bin_trial_vectors(df, signal_col, type_a, b)
        pv_b = _extract_per_bin_trial_vectors(df, signal_col, type_b, b)

        if pv_a.shape[0] < 3 or pv_b.shape[0] < 3:
            continue

        X = np.vstack([pv_a, pv_b])
        y = np.array([0] * len(pv_a) + [1] * len(pv_b))

        # Remove cells with zero variance
        var = np.nanvar(X, axis=0)
        good = var > 0
        if good.sum() < 2:
            continue
        X = X[:, good]

        # Replace any remaining NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        n_cv = min(n_splits, min(len(pv_a), len(pv_b)))
        if n_cv < 2:
            continue

        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)

        if classifier == 'svm':
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

        # Real accuracy
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            scores.append(clf.score(X[test_idx], y[test_idx]))
        accuracies[b] = np.mean(scores)

        # Shuffle distribution
        shuf_scores = np.zeros(n_shuffles)
        for s in range(n_shuffles):
            y_shuf = rng.permutation(y)
            fold_scores = []
            for train_idx, test_idx in skf.split(X, y_shuf):
                clf.fit(X[train_idx], y_shuf[train_idx])
                fold_scores.append(clf.score(X[test_idx], y_shuf[test_idx]))
            shuf_scores[s] = np.mean(fold_scores)
        chance_mean[b] = shuf_scores.mean()
        chance_95[b] = np.percentile(shuf_scores, 95)

    bin_centers = np.arange(n_total_bins) * bin_size_cm + bin_size_cm / 2

    return {
        'bin_centers': bin_centers,
        'accuracy': accuracies,
        'chance_mean': chance_mean,
        'chance_95': chance_95,
        'n_shared_bins': n_shared,
        'trial_types': (type_a, type_b),
    }


def plot_sliding_decoder(
    result: dict,
    config: dict | None = None,
    metadata: dict | None = None,
    trial_type_for_cues: str | None = None,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Plot decoder accuracy across position with cue shading and divergence line.

    Args:
        result: Output from sliding_decoder().
        config: For cue shading (optional).
        metadata: Metadata from the processed dataframe, contains 'bin_size_cm' value
        trial_type_for_cues: Which trial type's cue layout to shade.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    x = result['bin_centers']
    acc = result['accuracy']
    chance_95 = result['chance_95']
    n_shared = result['n_shared_bins']
    type_a, type_b = result['trial_types']
    bin_size_cm = _get_bin_size(df, metadata)
    if bin_size_cm is None:
        raise ValueError("Cannot determine bin size — no distance_bin column or metadata")
    diverge_cm = n_shared * bin_size_cm

    fig, ax = plt.subplots(figsize=figsize)

    # Chance band
    ax.fill_between(x, 0.5, chance_95, alpha=0.15, color='gray', label='95% shuffle')
    ax.axhline(0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Accuracy
    valid = ~np.isnan(acc)
    ax.plot(x[valid], acc[valid], color='black', linewidth=2, zorder=4)
    ax.scatter(x[valid], acc[valid], color='black', s=20, zorder=5)

    # Divergence
    ax.axvline(diverge_cm, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Tracks diverge ({diverge_cm:.0f} cm)')
    ax.axvspan(diverge_cm, x.max() + bin_size_cm, alpha=0.06, color='red', zorder=0)

    # Cue shading
    if config and trial_type_for_cues:
        pfmt.add_cue_shading(ax, config, trial_type_for_cues, alpha=0.05)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('Decoder Accuracy', fontsize=11)
    ax.set_title(
        pfmt.build_title(f'Trial Type Decoder — {type_a} vs {type_b}',
                     animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    ax.set_xlim(0, x.max() + bin_size_cm)
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=False, fontsize=10, loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, axis='y')
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# TRIAL-BY-TRIAL PV DISTANCE

def trial_pv_distance(
    df: pl.DataFrame,
    config: dict,
    cue_id: int | str = 'B',
    signal_col: str = 'multi_day_dff',
    trial_type_for_cue: str | None = None,
    metric: str = 'cosine',
) -> dict:
    """Compute per-trial distance from each trial-type centroid at a cue region.

    For each trial (regardless of type), computes PV at the specified cue
    region and measures distance to the means (ex: mean ABC PV and mean ABDC PV)

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        cue_id: Cue to analyze. Int uses 'cue' column, str uses 'cue_id' column.
        signal_col: Column containing neural signals.
        trial_type_for_cue: Trial type for cue region lookup. None auto-selects.
        metric: 'cosine' or 'euclidean'.

    Returns:
        Dict with keys: 'dist_to_a', 'dist_to_b', 'labels', 'trial_types',
        'cue_id'.
    """
    from scipy.spatial.distance import cosine, euclidean

    dist_fn = cosine if metric == 'cosine' else euclidean

    trial_types = sorted(df['trial_type'].unique().to_list())
    type_a, type_b = trial_types[0], trial_types[1]

    bin_range = _get_bin_range_for_cue(df, cue_id, trial_type_for_cue)

    pv_a = _extract_trial_population_vectors(df, signal_col, type_a, bin_range)
    pv_b = _extract_trial_population_vectors(df, signal_col, type_b, bin_range)

    centroid_a = np.nanmean(pv_a, axis=0)
    centroid_b = np.nanmean(pv_b, axis=0)

    all_pvs = np.vstack([pv_a, pv_b])
    labels = np.array([0] * len(pv_a) + [1] * len(pv_b))

    dist_to_a = np.array([dist_fn(pv, centroid_a) for pv in all_pvs])
    dist_to_b = np.array([dist_fn(pv, centroid_b) for pv in all_pvs])

    return {
        'dist_to_a': dist_to_a,
        'dist_to_b': dist_to_b,
        'labels': labels,
        'trial_types': (type_a, type_b),
        'cue_id': cue_id,
    }


def plot_trial_pv_distance(
    result: dict,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple = (6, 6),
    show: bool = True,
) -> Figure:
    """Scatter plot: distance to centroid A vs centroid B per trial.

    Points above the diagonal are closer to their own centroid (correct
    clustering). Color-coded by trial type using standard palette.

    Args:
        result: Output from trial_pv_distance().
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    d_a = result['dist_to_a']
    d_b = result['dist_to_b']
    labels = result['labels']
    type_a, type_b = result['trial_types']

    color_a = pfmt.TRIAL_TYPE_COLORS.get(type_a, '#2E86AB')
    color_b = pfmt.TRIAL_TYPE_COLORS.get(type_b, '#A23B72')

    fig, ax = plt.subplots(figsize=figsize)

    for label, tt, color, marker in [
        (0, type_a, color_a, 'o'),
        (1, type_b, color_b, 's'),
    ]:
        mask = labels == label
        ax.scatter(d_a[mask], d_b[mask], c=color, marker=marker,
                   alpha=0.7, s=40, label=tt, edgecolors='white', linewidth=0.5)

    # Diagonal
    lim = max(d_a.max(), d_b.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel(f'Distance to {type_a} centroid', fontsize=11)
    ax.set_ylabel(f'Distance to {type_b} centroid', fontsize=11)
    ax.set_title(
        pfmt.build_title(f'Trial PV Distance at Cue {result["cue_id"]}',
                     animal_id=animal_id, date=date), fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_aspect('equal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


# REGION-RESTRICTED CORRELATION

def region_correlation(
    df: pl.DataFrame,
    config: dict,
    type_a: str,
    type_b: str,
    cue_id: str | None = None,
    bin_range: tuple[int, int] | None = None,
    signal_col: str = 'multi_day_dff',
    trial_type_for_cue: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """PV and per-cell correlation restricted to a cue region or bin range.

    Provide either cue_id or bin_range. Computes population vector correlation
    across bins and per-cell Pearson r within the specified spatial region.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        cue_id: Cue to restrict to (looks up bin range from config).
        bin_range: Explicit (start_bin, end_bin). Overrides cue_id.
        signal_col: Column containing neural signals.
        trial_type_for_cue: Trial type for cue region lookup. Defaults to type_a.
        metadata: Metadata from the processed dataframe, contains 'bin_size_cm' value.


    Returns:
        Dict with keys: 'per_cell_corr', 'pv_corr', 'bin_range', 'region_cm',
        'cue_id'.
    """
    if bin_range is None:
        if cue_id is None:
            raise ValueError("Provide either cue_id or bin_range")
        bin_range = _get_bin_range_for_cue(df, cue_id, trial_type_for_cue)

    bin_size_cm = _get_bin_size(df, metadata)
    if bin_size_cm is None:
        raise ValueError("Cannot determine bin size — no distance_bin column or metadata")

    start_bin, end_bin = bin_range

    avgs = compute_session_averages(
        df, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
    )
    avg_a = avgs[type_a]['session_avg'][start_bin:end_bin]  # (n_bins_region, n_cells)
    avg_b = avgs[type_b]['session_avg'][start_bin:end_bin]

    per_cell = per_cell_spatial_correlation(avg_a, avg_b)
    pv = population_vector_correlation(avg_a, avg_b)

    return {
        'per_cell_corr': per_cell,
        'pv_corr': pv,
        'bin_range': bin_range,
        'region_cm': (start_bin * bin_size_cm, end_bin * bin_size_cm),
        'cue_id': cue_id,
    }


def region_correlation_multi_cue(
    df: pl.DataFrame,
    config: dict,
    type_a: str,
    type_b: str,
    cue_ids: list[str],
    signal_col: str = 'multi_day_dff',
    metadata: dict | None = None,
    trial_type_for_cue: str | None = None,
) -> dict[int, dict]:
    """PV and per-cell correlation for multiple cue regions.

    Convenience wrapper around region_correlation() for a list of cues.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        cue_ids: List of cue IDs to analyze (e.g. [1, 2, 3]).
        signal_col: Column containing neural signals.
        metadata: Metadata from the processed dataframe, contains 'bin_size_cm' value used in region_correlation
        trial_type_for_cue: Trial type for cue region lookup. Defaults to type_a.

    Returns:
        Dict mapping cue_id -> region_correlation() result dict.
    """
    results = {}
    for cue_id in cue_ids:
        results[cue_id] = region_correlation(
            df, config, type_a, type_b,
            cue_id=cue_id, signal_col=signal_col,
            metadata=metadata,
            trial_type_for_cue=trial_type_for_cue,
        )
    return results


def plot_region_correlation_at_cues(
    df: pl.DataFrame,
    config: dict,
    type_a: str,
    type_b: str,
    cue_ids: list[int],
    signal_col: str = 'multi_day_dff',
    metadata: dict | None = None,
    trial_type_for_cue: str | None = None,
    animal_id: str | None = None,
    date: str | None = None,
    figsize_per_cue: tuple = (5, 4),
    show: bool = True,
) -> tuple[Figure, dict[int, dict]]:
    """Plot PV correlation at each cue region as subplots, for state representation comparison.

    Two-row layout: top row shows PV correlation across spatial bins within
    each cue region, bottom row shows per-cell correlation histograms. One
    column per cue, colored by cue palette.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        type_a: First trial type.
        type_b: Second trial type.
        cue_ids: List of cue IDs to plot (e.g. [1, 2, 3]).
        signal_col: Column containing neural signals.
        metadata: Metadata from the processed dataframe, contains 'bin_size_cm' value
        trial_type_for_cue: Trial type for cue region lookup. Defaults to type_a.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize_per_cue: (width, height) per subplot panel.
        show: Call plt.show().

    Returns:
        Tuple of (Figure, dict mapping cue_id -> region_correlation result).
    """
    cue_colors = pfmt.get_cue_colors(config)
    results = region_correlation_multi_cue(
        df, config, type_a, type_b, cue_ids,
        signal_col=signal_col, metadata=metadata,
        trial_type_for_cue=trial_type_for_cue,
    )
    bin_size_cm = _get_bin_size(df, metadata)
    if bin_size_cm is None:
        raise ValueError("Cannot determine bin size — no distance_bin column or metadata")

    n_cues = len(cue_ids)
    fig_w = figsize_per_cue[0] * n_cues
    fig_h = figsize_per_cue[1] * 2  # two rows: PV corr + per-cell hist
    fig, axes = plt.subplots(2, n_cues, figsize=(fig_w, fig_h), squeeze=False)

    for col, cue_id in enumerate(cue_ids):
        rc = results[cue_id]
        pv_corr = rc['pv_corr']
        per_cell = rc['per_cell_corr']
        start_cm, end_cm = rc['region_cm']
        color = cue_colors.get(cue_id, '#555555')

        # ── Top row: PV correlation across bins within cue ──
        ax_pv = axes[0, col]
        x_bins = np.linspace(start_cm, end_cm, len(pv_corr), endpoint=False)
        x_bins += bin_size_cm / 2

        ax_pv.plot(x_bins, pv_corr, color=color, linewidth=2, zorder=4)
        ax_pv.fill_between(x_bins, pv_corr, alpha=0.2, color=color, zorder=3)
        ax_pv.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

        mean_pv = np.nanmean(pv_corr)
        ax_pv.axhline(mean_pv, color='black', linestyle='--', linewidth=1,
                       alpha=0.6, label=f'mean = {mean_pv:.3f}')

        ax_pv.set_xlabel('Position (cm)', fontsize=10)
        ax_pv.set_ylabel('PV Correlation (r)', fontsize=10)
        ax_pv.set_title(f'Cue {cue_id} ({start_cm:.0f}–{end_cm:.0f} cm)',
                        fontsize=11, fontweight='bold')
        ax_pv.set_ylim(-0.3, 1.05)
        ax_pv.legend(frameon=False, fontsize=8)
        ax_pv.spines['top'].set_visible(False)
        ax_pv.spines['right'].set_visible(False)
        ax_pv.grid(alpha=0.2, axis='y')

        # Bottom row: per-cell correlation histogram
        ax_hist = axes[1, col]
        valid_corrs = per_cell[~np.isnan(per_cell)]

        ax_hist.hist(valid_corrs, bins=np.linspace(-1, 1, 31), color=color,
                     alpha=0.7, edgecolor='white', linewidth=0.5)
        median_r = np.nanmedian(valid_corrs)
        ax_hist.axvline(median_r, color='black', linestyle='--', linewidth=1.5,
                        label=f'median = {median_r:.3f}')
        ax_hist.axvline(0, color='gray', linewidth=0.8, alpha=0.5)

        ax_hist.set_xlabel('Pearson r (per cell)', fontsize=10)
        ax_hist.set_ylabel('Cells', fontsize=10)
        ax_hist.set_title(f'Per-Cell Corr — Cue {cue_id}', fontsize=11)
        ax_hist.legend(frameon=False, fontsize=8)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

        n_valid = len(valid_corrs)
        n_total = len(per_cell)
        ax_hist.text(0.02, 0.95, f'n = {n_valid}/{n_total}',
                     transform=ax_hist.transAxes, fontsize=8, va='top',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.suptitle(
        pfmt.build_title(f'Region Correlation — {type_a} vs {type_b}',
                     animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    if show:
        plt.show()
    return fig, results


# SPLITTER CELL INDEX

def splitter_cell_index(
    df: pl.DataFrame,
    config: dict,
    cue_id: int | str = 'B',
    signal_col: str = 'multi_day_dff',
    trial_type_for_cue: str = 'ABC',
    n_shuffles: int = 500,
    seed: int = 42,
) -> dict:
    """Compute a selectivity index for each cell at a cue region.

    For each cell, computes mean activity at the cue region on type_a vs
    type_b trials, then computes:
        SI = (mean_a - mean_b) / (mean_a + mean_b + eps)

    Significance tested via trial-label shuffle (two-sided).

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        cue_id: Cue region to analyze (2=B).
        signal_col: Column containing neural signals.
        trial_type_for_cue: Trial type for cue region lookup.
        n_shuffles: Shuffle iterations for p-values.
        seed: Random seed.

    Returns:
        Dict with keys: 'selectivity', 'p_values', 'mean_a', 'mean_b',
        'significant', 'trial_types', 'cue_id', 'n_cells'.
    """
    rng = np.random.default_rng(seed)

    trial_types = sorted(df['trial_type'].unique().to_list())
    type_a, type_b = trial_types[0], trial_types[1]

    bin_range = _get_bin_range_for_cue(df, trial_type_for_cue, cue_id)

    pv_a = _extract_trial_population_vectors(df, signal_col, type_a, bin_range)
    pv_b = _extract_trial_population_vectors(df, signal_col, type_b, bin_range)

    mean_a = np.nanmean(pv_a, axis=0)
    mean_b = np.nanmean(pv_b, axis=0)

    eps = 1e-10
    observed_si = (mean_a - mean_b) / (mean_a + mean_b + eps)

    # Shuffle test
    all_pvs = np.vstack([pv_a, pv_b])
    n_a = len(pv_a)
    n_total = len(all_pvs)
    n_cells = all_pvs.shape[1]

    shuffle_si = np.zeros((n_shuffles, n_cells))
    for s in range(n_shuffles):
        perm = rng.permutation(n_total)
        shuf_a = np.nanmean(all_pvs[perm[:n_a]], axis=0)
        shuf_b = np.nanmean(all_pvs[perm[n_a:]], axis=0)
        shuffle_si[s] = (shuf_a - shuf_b) / (shuf_a + shuf_b + eps)

    # Two-sided p-value
    p_values = np.mean(np.abs(shuffle_si) >= np.abs(observed_si)[np.newaxis, :], axis=0)

    return {
        'selectivity': observed_si,
        'p_values': p_values,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'significant': p_values < 0.05,
        'trial_types': (type_a, type_b),
        'cue_id': cue_id,
        'n_cells': n_cells,
    }


def plot_splitter_cells(
    result: dict,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Plot splitter cell analysis: selectivity histogram + mean activity scatter.

    Left panel: histogram of selectivity index, significant cells highlighted.
    Right panel: scatter of mean ΔF/F on type_a vs type_b.

    Args:
        result: Output from splitter_cell_index().
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    si = result['selectivity']
    sig = result['significant']
    type_a, type_b = result['trial_types']
    mean_a = result['mean_a']
    mean_b = result['mean_b']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(si[~sig], bins=np.linspace(-1, 1, 41), color='gray', alpha=0.6,
             edgecolor='white', linewidth=0.5, label=f'NS ({(~sig).sum()})')
    ax1.hist(si[sig], bins=np.linspace(-1, 1, 41), color='#E63946', alpha=0.8,
             edgecolor='white', linewidth=0.5, label=f'Sig ({sig.sum()})')
    ax1.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Selectivity Index', fontsize=11)
    ax1.set_ylabel('Number of cells', fontsize=11)
    ax1.set_title(f'Splitter Cells at Cue {result["cue_id"]}', fontsize=11, fontweight='bold')
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Scatter: mean activity type_a vs type_b
    ax2.scatter(mean_a[~sig], mean_b[~sig], c='gray', alpha=0.4, s=15, label='NS')
    ax2.scatter(mean_a[sig], mean_b[sig], c='#E63946', alpha=0.7, s=25,
                edgecolors='black', linewidth=0.3, label='Significant')
    lim = max(mean_a.max(), mean_b.max()) * 1.1
    ax2.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.8)
    ax2.set_xlabel(f'Mean ΔF/F ({type_a})', fontsize=11)
    ax2.set_ylabel(f'Mean ΔF/F ({type_b})', fontsize=11)
    ax2.set_title('Mean Activity per Trial Type', fontsize=11, fontweight='bold')
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(
        pfmt.build_title(f'Splitter Cell Analysis — {type_a} vs {type_b}',
                     animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# WRAPPER

def run_decoding_analysis(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = 'multi_day_dff',
    cue_ids: list[int | str] | int | str = 'B',
    n_shuffles_decoder: int = 100,
    n_shuffles_splitter: int = 500,
    animal_id: str | None = None,
    date: str | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict:
    """Run full decoding analysis pipeline.

    Runs sliding decoder, trial PV distance, region correlation (single or
    multi-cue subplot), and splitter cell index. Generates all figures.

    Args:
        df: Frame-level DataFrame with distance_bin column.
        config: Experiment configuration dict.
        metadata: Processed dataframe metadata dict.
        signal_col: Column containing neural signals.
        cue_ids: Cue region(s) for detailed analysis. Single int or list of ints.
        n_shuffles_decoder: Shuffles for decoder chance distribution.
        n_shuffles_splitter: Shuffles for splitter cell p-values.
        animal_id: Animal identifier for plot titles.
        date: Session date for plot titles.
        show: Call plt.show().
        save_dir: Directory to save figures (optional).

    Returns:
        Dict with 'decoder', 'pv_distance', 'region_corr', 'splitter', 'figures'.
    """
    # Normalize cue_ids to list
    if isinstance(cue_ids, (int, str)):
        cue_ids = [cue_ids]

    trial_types = sorted(df['trial_type'].unique().to_list())
    if len(trial_types) < 2:
        raise ValueError("Need ≥2 trial types for prospective analysis")

    type_a, type_b = trial_types[0], trial_types[1]
    results = {}
    figs = {}

    # 1. Sliding decoder
    print("Running sliding decoder...")
    dec = sliding_decoder(
        df, config, metadata, signal_col=signal_col,
        n_shuffles=n_shuffles_decoder,
    )
    results['decoder'] = dec
    fig = plot_sliding_decoder(
        dec, config=config, trial_type_for_cues=trial_types[0],
        metadata=metadata, animal_id=animal_id, date=date, show=show,
    )
    figs['decoder'] = fig

    # 2. Trial-by-trial PV distance (uses first cue)
    primary_cue = cue_ids[0]
    print(f"Computing trial PV distances at cue {primary_cue}...")
    pvd = trial_pv_distance(
        df, config, cue_id=primary_cue, signal_col=signal_col,
    )
    results['pv_distance'] = pvd
    fig = plot_trial_pv_distance(pvd, animal_id=animal_id, date=date, show=show)
    figs['pv_distance'] = fig

    # 3. Region-restricted correlation (multi-cue subplot)
    print(f"Computing region correlation at cues {cue_ids}...")
    fig, rc_results = plot_region_correlation_at_cues(
        df, config, type_a, type_b, cue_ids,
        signal_col=signal_col, metadata=metadata,
        animal_id=animal_id, date=date, show=show,
    )
    results['region_corr'] = rc_results
    figs['region_corr'] = fig
    for cue_id, rc in rc_results.items():
        print(f"  Cue {cue_id}: PV mean={np.nanmean(rc['pv_corr']):.3f}, "
              f"per-cell median={np.nanmedian(rc['per_cell_corr']):.3f}")

    # 4. Splitter cells (uses first cue)
    print(f"Computing splitter cell index at cue {primary_cue}...")
    sp = splitter_cell_index(
        df, config, cue_id=primary_cue, signal_col=signal_col,
        n_shuffles=n_shuffles_splitter,
    )
    results['splitter'] = sp
    fig = plot_splitter_cells(sp, animal_id=animal_id, date=date, show=show)
    figs['splitter'] = fig
    print(f"  {sp['significant'].sum()}/{sp['n_cells']} significant splitter cells")

    results['figures'] = figs

    # Save
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            path = save_dir / f'prospective_{name}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

    return results


if __name__ == "__main__":
    from df_processing import load_session_dir, get_session_prefix, load_processed_session

    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    date = '2025-09-15'

    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    data, meta = load_processed_session(behavior_path.parent / f'{prefix}_processed.parquet')
    print(meta)

    animal_id = session_data.get('animal_id', '')

    results = run_decoding_analysis(
        data, config, metadata=meta, signal_col='multi_day_dff',
        cue_ids=['A', '0a', 'B', '0b', 'C'],
        animal_id=animal_id, date=date,
        show=True,
    )
