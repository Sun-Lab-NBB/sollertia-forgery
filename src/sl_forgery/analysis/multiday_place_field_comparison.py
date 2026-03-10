"""
Multiday place field comparison for tracking cell recruitment and tuning changes.

Answers: what happens to place cell representations before and after a new task
variant is introduced (e.g., base (ABC-only) days → extension (ABC+ABDC) days)?

Analyses:
    1. Cell recruitment classification — for each cell across days, categorize as
       gained/lost/stable/shifted place field, per trial type.
    2. ABC tuning stability pre vs post extension — does ABC remapping increase
       after ABDC is introduced, beyond normal day-to-day drift?
    3. Bifurcation tuning tracking — per-cell activity at the bifurcation cue (B/0b)
       across days, flagging emergence of "splitter"-like selectivity.
    4. Population summary — place cell counts, field counts, and recruitment
       category proportions across days.

All functions operate on the sessions dict from load_multiday_sessions() and
PlaceFieldResult objects from place_field_detection.

Dependencies: numpy, polars, matplotlib, df_processing, place_field_detection,
              cross_correlation_1, decoding_analysis.
"""

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')

from df_processing import (get_bin_size,
    compute_session_averages, get_track_length, get_cue_regions,
    load_multiday_sessions,
)
from place_field_detection import (
    detect_place_fields, PlaceFieldResult, PlaceFields1d,
    DetectionParams, get_place_cell_indices,
)
from cross_correlation_1 import (
    get_mean_tuning_curves, per_cell_spatial_correlation,
    get_divergence_point,
)
from decoding_analysis import (
    splitter_cell_index, _get_bin_range_for_cue,
    _extract_trial_population_vectors,
)
import plot_utils as pfmt


# PLACE FIELD DETECTION ACROSS DAYS

def detect_fields_multiday(
    sessions: dict[str, dict],
    signal_col: str = 'multi_day_dff',
    params: DetectionParams | None = None,
) -> dict[str, PlaceFieldResult]:
    """Run place field detection on each session independently.

    Uses detect_place_fields() from place_field_detection.py

    Args:
        sessions: From load_multiday_sessions(). Keys are date strings.
        signal_col: Column containing neural signals.
        params: Detection parameters. Uses defaults if None.

    Returns:
        Dict mapping date -> PlaceFieldResult.
    """
    results = {}
    for date in sorted(sessions):
        s = sessions[date]
        print(f"Detecting fields: {date}...")
        bin_size_cm = get_bin_size(None, s['metadata'])
        result = detect_place_fields(
            s['data'], s['config'],
            signal_col=signal_col, bin_size_cm=bin_size_cm, params=params,
        )
        results[date] = result
        print(f"  {result.summary()}\n")
    return results


# CELL RECRUITMENT CLASSIFICATION

def _get_primary_field_centers(
    pf: PlaceFields1d,
    n_cells: int,
) -> np.ndarray:
    """Get the center position (cm) of each cell's primary (strongest) field.

    Args:
        pf: PlaceFields1d object.
        n_cells: Total number of cells.

    Returns:
        Array of shape (n_cells,) with field center in cm. NaN if no field.
    """
    centers = np.full(n_cells, np.nan)

    if pf.n_fields == 0:
        return centers

    cell_ids = pf.cell_id
    field_centers = pf.centers[:, 1] * pf.bin_size  # convert to cm
    intensities = pf.mean_intensity

    for cell_idx in range(n_cells):
        field_mask = cell_ids == cell_idx
        if not field_mask.any():
            continue
        # Pick field with highest mean intensity
        best = np.argmax(intensities[field_mask])
        field_indices = np.where(field_mask)[0]
        centers[cell_idx] = field_centers[field_indices[best]]

    return centers


def classify_cell_recruitment(
    pf_results: dict[str, PlaceFieldResult],
    trial_type: str,
    field_center_tolerance_cm: float | None = None,
) -> dict[str, dict]:
    """Classify each cell's place field fate across consecutive day pairs.

    For each consecutive pair of days where the trial type exists, each cell
    is classified as one of:
        - 'stable': has a field on both days, center within tolerance
        - 'shifted': has a field on both days, center moved beyond tolerance
        - 'gained': no field on day N, has field on day N+1
        - 'lost': has field on day N, no field on day N+1
        - 'absent': no field on either day

    Args:
        pf_results: From detect_fields_multiday(). Keys are date strings.
        trial_type: Which trial type to compare.
        field_center_tolerance_cm: Max distance (cm) between field centers
            to classify as 'stable' vs 'shifted'.  Rn 10 cm (2 bins)
            If None, cells with fields on both days are all classified as 'both' (use center_shift_cm
            for continuous analysis).

    Returns:
        Dict keyed by 'dayN_vs_dayN+1' strings, each containing:
            'categories': array of str, shape (n_cells,)
            'day_a': str, 'day_b': str
            'center_a': array (n_cells,) field center on day A (NaN if absent)
            'center_b': array (n_cells,) field center on day B (NaN if absent)
            'center_shift_cm': array (n_cells,) signed shift (NaN if absent either day)
    """
    dates = sorted(d for d in pf_results if trial_type in pf_results[d].fields)
    if len(dates) < 2:
        raise ValueError(f"Need ≥2 days with {trial_type}, got {len(dates)}")

    results = {}

    for i in range(len(dates) - 1):
        day_a, day_b = dates[i], dates[i + 1]
        pf_a = pf_results[day_a].fields[trial_type]
        pf_b = pf_results[day_b].fields[trial_type]
        n_cells = pf_a.binF.shape[0]

        has_field_a = pf_a.has_place_field  # (n_cells,)
        has_field_b = pf_b.has_place_field

        # Get primary field center per cell (highest mean intensity field)
        center_a = _get_primary_field_centers(pf_a, n_cells)
        center_b = _get_primary_field_centers(pf_b, n_cells)

        center_shift = center_b - center_a  # NaN where either is NaN

        categories = np.full(n_cells, 'absent', dtype='<U10')
        both = has_field_a & has_field_b
        shift_dist = np.abs(center_shift)

        categories[has_field_a & ~has_field_b] = 'lost'
        categories[~has_field_a & has_field_b] = 'gained'

        if field_center_tolerance_cm is not None:
            categories[both & (shift_dist <= field_center_tolerance_cm)] = 'stable'
            categories[both & (shift_dist > field_center_tolerance_cm)] = 'shifted'
        else:
            categories[both] = 'both'

        shift_dist = np.abs(center_shift)

        pair_key = f'{day_a}_vs_{day_b}'
        results[pair_key] = {
            'categories': categories,
            'day_a': day_a,
            'day_b': day_b,
            'center_a': center_a,
            'center_b': center_b,
            'center_shift_cm': center_shift,
            'has_field_a': has_field_a,
            'has_field_b': has_field_b,
        }

        # Summary
        for cat in ['stable', 'shifted', 'gained', 'lost', 'absent']:
            n = (categories == cat).sum()
            print(f"  {pair_key} | {cat}: {n}/{n_cells}")

    return results


def compute_rate_remapping(
    tuning_a: np.ndarray,
    tuning_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute rate remapping metrics between two tuning curve arrays.

    For each cell, compares peak amplitude and mean activity between
    conditions. Pearson r misses these changes because it's scale-invariant (consders shape)

    Args:
        tuning_a: Tuning curves day A, shape (n_bins, n_cells).
        tuning_b: Tuning curves day B, shape (n_bins, n_cells).

    Returns:
        Dict with:
            'peak_ratio': peak_b / peak_a per cell (>1 = gained amplitude)
            'mean_ratio': mean_b / mean_a per cell
            'peak_diff': peak_b - peak_a per cell (raw change)
            'mean_diff': mean_b - mean_a per cell
            'log2_peak_ratio': log2(peak_b / peak_a), centered at 0
    """
    n_bins = min(tuning_a.shape[0], tuning_b.shape[0])
    a = tuning_a[:n_bins]
    b = tuning_b[:n_bins]

    peak_a = np.nanmax(a, axis=0)
    peak_b = np.nanmax(b, axis=0)
    mean_a = np.nanmean(a, axis=0)
    mean_b = np.nanmean(b, axis=0)

    eps = 1e-10
    peak_ratio = peak_b / (peak_a + eps)
    mean_ratio = mean_b / (mean_a + eps)

    with np.errstate(divide='ignore', invalid='ignore'):
        log2_peak_ratio = np.log2(peak_ratio)
        log2_peak_ratio[~np.isfinite(log2_peak_ratio)] = np.nan

    return {
        'peak_ratio': peak_ratio,
        'mean_ratio': mean_ratio,
        'peak_diff': peak_b - peak_a,
        'mean_diff': mean_b - mean_a,
        'log2_peak_ratio': log2_peak_ratio,
    }


# POPULATION SUMMARY ACROSS DAYS

def population_summary(
    pf_results: dict[str, PlaceFieldResult],
    trial_type: str,
) -> dict[str, dict]:
    """Summarize place cell and field counts per day for one trial type.

    Args:
        pf_results: From detect_fields_multiday().
        trial_type: Which trial type.

    Returns:
        Dict keyed by date, each containing:
            'n_place_cells': int
            'n_fields': int
            'n_cells_total': int
            'fraction_place_cells': float
    """
    summary = {}
    for date in sorted(pf_results):
        result = pf_results[date]
        if trial_type not in result.fields:
            continue
        pf = result.fields[trial_type]
        n_pc = pf.has_place_field.sum()
        n_total = result.n_cells
        summary[date] = {
            'n_place_cells': int(n_pc),
            'n_fields': int(pf.n_fields),
            'n_cells_total': n_total,
            'fraction_place_cells': n_pc / max(n_total, 1),
        }
    return summary


# ABC TUNING STABILITY PRE VS POST


def abc_tuning_pre_vs_post(
    sessions: dict[str, dict],
    pf_results: dict[str, PlaceFieldResult],
    introduction_day: str,
    trial_type: str = 'ABC',
    signal_col: str = 'multi_day_dff',
) -> dict:
    """Compare ABC tuning curve stability before vs after ABDC introduction.

    Computes per-cell spatial correlation for:
        - Pre-pre: last two ABC-only days (control drift)
        - Pre-post: last ABC-only day vs first extension day
        - Post-post: consecutive extension days (if available)
    Log peak ratio 0=same, 1=double, -1=halved

    Args:
        sessions: From load_multiday_sessions().
        pf_results: From detect_fields_multiday().
        introduction_day: Date string of first session with the new trial type.
        trial_type: Trial type to track (usually 'ABC').
        signal_col: Column containing neural signals.

    Returns:
        Dict with keys 'pre_pre', 'pre_post', 'post_post', each containing:
            'corrs': per-cell correlations (n_cells,)
            'day_a': str, 'day_b': str
            'place_cell_mask': bool array from day_a
        Returns None for comparisons that lack sufficient days.
    """
    dates = sorted(sessions.keys())
    intro_idx = dates.index(introduction_day)

    pre_dates = [d for d in dates[:intro_idx]
                 if trial_type in sessions[d]['data']['trial_type'].unique().to_list()]
    post_dates = [d for d in dates[intro_idx:]
                  if trial_type in sessions[d]['data']['trial_type'].unique().to_list()]

    def _compute_pair(day_a, day_b):
        """Compute per-cell correlation, rate remapping, and place cell mask for a day pair."""
        avg_a = get_mean_tuning_curves(
            sessions[day_a]['data'], sessions[day_a]['config'],
            sessions[day_a]['metadata'], signal_col=signal_col,
        )[trial_type]
        avg_b = get_mean_tuning_curves(
            sessions[day_b]['data'], sessions[day_b]['config'],
            sessions[day_b]['metadata'], signal_col=signal_col,
        )[trial_type]

        corrs = per_cell_spatial_correlation(avg_a, avg_b)
        rate = compute_rate_remapping(avg_a, avg_b)

        pc_a = pf_results[day_a].is_place_cell.get(trial_type, np.zeros(len(corrs), dtype=bool))
        pc_b = pf_results[day_b].is_place_cell.get(trial_type, np.zeros(len(corrs), dtype=bool))

        return {
            'corrs': corrs,
            'rate_remapping': rate,
            'day_a': day_a,
            'day_b': day_b,
            'pc_both': pc_a & pc_b,
            'pc_day_a_only': pc_a & ~pc_b,
            'pc_day_b_only': ~pc_a & pc_b,
        }

    result = {'pre_pre': None, 'pre_post': None, 'post_post': None}

    if len(pre_dates) >= 2:
        result['pre_pre'] = _compute_pair(pre_dates[-2], pre_dates[-1])
        print(f"Pre-pre: {pre_dates[-2]} vs {pre_dates[-1]}, "
              f"median r={np.nanmedian(result['pre_pre']['corrs']):.3f}, ",
              f"median log2 peak ratio={np.nanmedian(result['pre_pre']['rate_remapping']['log2_peak_ratio']):.3f}")

    if pre_dates and post_dates:
        result['pre_post'] = _compute_pair(pre_dates[-1], post_dates[0])
        print(f"Pre-post: {pre_dates[-1]} vs {post_dates[0]}, "
              f"median r={np.nanmedian(result['pre_post']['corrs']):.3f}, ",
              f"median log2 peak ratio={np.nanmedian(result['pre_post']['rate_remapping']['log2_peak_ratio']):.3f}")

    if len(post_dates) >= 2:
        result['post_post'] = _compute_pair(post_dates[0], post_dates[1])
        print(f"Post-post: {post_dates[0]} vs {post_dates[1]}, "
              f"median r={np.nanmedian(result['post_post']['corrs']):.3f}, ",
              f"median log2 peak ratio={np.nanmedian(result['post_post']['rate_remapping']['log2_peak_ratio']):.3f}")

    return result


# BIFURCATION TUNING ACROSS DAYS


def track_bifurcation_selectivity(
    sessions: dict[str, dict],
    cue_id: int | str = '0b',
    signal_col: str = 'multi_day_dff',
    n_shuffles: int = 200,
    seed: int = 42,
) -> dict[str, dict]:
    """Track splitter cell selectivity at the bifurcation cue across days.

    Only runs on sessions that have ≥2 trial types (i.e., post-extension).
    Returns the splitter_cell_index result per day so you can track when
    individual cells become selective.

    Args:
        sessions: From load_multiday_sessions().
        cue_id: Cue region to analyze (e.g., 'B' or '0b' for bifurcation).
        signal_col: Column containing neural signals.
        n_shuffles: Shuffle iterations for p-values.
        seed: Random seed.

    Returns:
        Dict keyed by date, each containing the output of splitter_cell_index():
            'selectivity', 'p_values', 'significant', 'mean_a', 'mean_b',
            'trial_types', 'cue_id', 'n_cells'.
        Skips single-trial-type sessions.
    """
    results = {}
    for date in sorted(sessions):
        s = sessions[date]
        df = s['data']
        config = s['config']

        trial_types = sorted(df['trial_type'].unique().to_list())
        if len(trial_types) < 2:
            print(f"  {date}: only {trial_types}, skipping splitter analysis")
            continue

        print(f"  {date}: computing splitter index at cue {cue_id}...")
        result = splitter_cell_index(
            df, config, cue_id=cue_id, signal_col=signal_col,
            n_shuffles=n_shuffles, seed=seed,
        )
        results[date] = result
        n_sig = result['significant'].sum()
        print(f"    {n_sig}/{result['n_cells']} significant splitters")

    return results

#TODO consider either rethinking ior removing this, and the plotting func
def track_bifurcation_activity(
    sessions: dict[str, dict],
    cue_id: int | str = '0b',
    signal_col: str = 'multi_day_dff',
) -> dict[str, np.ndarray]:
    """Get mean activity at the bifurcation cue per cell across all days.

    Unlike track_bifurcation_selectivity, this runs on ALL sessions including
    single-trial-type days. Useful for seeing baseline activity before the
    new trial type is introduced.

    Args:
        sessions: From load_multiday_sessions().
        cue_id: Cue region to analyze.
        signal_col: Column containing neural signals.

    Returns:
        Dict keyed by date, each containing mean activity array (n_cells,).
    """
    results = {}
    for date in sorted(sessions):
        s = sessions[date]
        df = s['data']
        config = s['config']

        trial_types = sorted(df['trial_type'].unique().to_list())
        # Use first trial type that contains the cue
        bin_range = _get_bin_range_for_cue(df, cue_id, config=config)

        # Average across all trials regardless of type
        pvs = []
        for tt in trial_types:
            pv = _extract_trial_population_vectors(df, signal_col, tt, bin_range)
            if pv.size > 0:
                pvs.append(pv)

        if pvs:
            all_pvs = np.vstack(pvs)
            results[date] = np.nanmean(all_pvs, axis=0)
        else:
            print(f"  {date}: no data at cue {cue_id}")

    return results


# PLOTTING


def plot_population_summary(
    pf_results: dict[str, PlaceFieldResult],
    trial_type: str,
    introduction_day: str | None = None,
    animal_id: str | None = None,
    figsize: tuple = (10, 4),
    show: bool = True,
) -> Figure:
    """Plot place cell count and fraction across days.

    Args:
        pf_results: From detect_fields_multiday().
        trial_type: Which trial type to plot.
        introduction_day: Date of new trial type introduction (vertical line).
        animal_id: Animal identifier for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    summary = population_summary(pf_results, trial_type)
    dates = sorted(summary.keys())
    n_pc = [summary[d]['n_place_cells'] for d in dates]
    frac_pc = [summary[d]['fraction_place_cells'] for d in dates]
    n_fields = [summary[d]['n_fields'] for d in dates]
    date_labels = [d[5:] for d in dates]

    color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: place cell count
    ax1.plot(range(len(dates)), n_pc, 'o-', color=color, linewidth=2, markersize=6)
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Place cell count', fontsize=10)
    ax1.set_title('Place cells', fontsize=11, fontweight='bold')

    # Right: fraction + field count
    ax2.plot(range(len(dates)), frac_pc, 'o-', color=color, linewidth=2,
             markersize=6, label='Fraction PC')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(dates)), n_fields, 's--', color=color, alpha=0.5,
                  linewidth=1.5, markersize=5, label='Field count')
    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Fraction place cells', fontsize=10)
    ax2_twin.set_ylabel('Total fields', fontsize=10, alpha=0.6)
    ax2.set_title('Place cell fraction & fields', fontsize=11, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2, axis='y')
        if introduction_day and introduction_day in dates:
            idx = dates.index(introduction_day)
            ax.axvline(idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2_twin.spines['top'].set_visible(False)

    fig.suptitle(
        pfmt.build_title('Place Cell Population', trial_type=trial_type, animal_id=animal_id),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_recruitment_categories(
    recruitment: dict[str, dict],
    introduction_day: str | None = None,
    animal_id: str | None = None,
    trial_type: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Stacked bar chart of cell recruitment categories across day pairs.

    Args:
        recruitment: From classify_cell_recruitment().
        introduction_day: Date of new trial type introduction (highlight bar).
        animal_id: Animal identifier for title.
        trial_type: Trial type label for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    pair_keys = sorted(recruitment.keys())
    category_names = ['stable', 'shifted', 'gained', 'lost', 'absent']
    category_colors = {
        'stable': '#59A14F',   # green
        'shifted': '#EDC948',  # gold
        'gained': '#4E79A7',   # blue
        'lost': '#E15759',     # red
        'absent': '#FFFFFF',   # white w black outline        #'#BAB0AC',   # gray
    }

    counts = {cat: [] for cat in category_names}
    for key in pair_keys:
        cats = recruitment[key]['categories']
        for cat in category_names:
            counts[cat].append((cats == cat).sum())

    x = np.arange(len(pair_keys))
    bar_labels = [k.replace('_vs_', '\nto\n') for k in pair_keys]

    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(pair_keys))
    for cat in category_names:
        # if cat == 'absent':
        #     continue  # skip absent for cleaner plot
        vals = np.array(counts[cat])
        ax.bar(x, vals, bottom=bottom, label=cat,
               color=category_colors[cat], edgecolor=None, linewidth=0.5)
        bottom += vals

    # Highlight introduction boundary
    if introduction_day:
        for i, key in enumerate(pair_keys):
            day_b = recruitment[key]['day_b']
            if day_b == introduction_day:
                ax.axvline(i + .5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                           label='ABDC introduced')
                break

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Number of cells', fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(
        pfmt.build_title('Cell Recruitment', trial_type=trial_type, animal_id=animal_id),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_abc_stability_comparison(
    stability: dict,
    animal_id: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Compare ABC tuning stability: pre-pre vs pre-post vs post-post.
    *Where pre==pre-extension, and post==post-extension.
    If ABC representations are disrupted by ABDC introduction, the pre-post median r should be lower than the
    pre-pre median r (which represents baseline drift). If pre-post ≈ pre-pre, then the introduction didn't cause
    extra remapping.

    Left: overlaid histograms of per-cell correlation for each comparison for ALL cells. Each cell gets 1 r-value
    Right: box/strip plot of place-cell-only correlations. "Do existing fields remap?"

    Args:
        stability: From abc_tuning_pre_vs_post().
        animal_id: Animal identifier for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    labels_colors = {
        'pre_pre': ('#59A14F', 'Pre–Pre (drift)'),
        'pre_post': ('#E15759', 'Pre–Post (introduction)'),
        'post_post': ('#4E79A7', 'Post–Post'),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: overlaid histograms
    bins_hist = np.linspace(-1, 1, 41)
    for key, (color, label) in labels_colors.items():
        if stability[key] is None:
            continue
        corrs = stability[key]['corrs']
        valid = corrs[~np.isnan(corrs)]
        ax1.hist(valid, bins=bins_hist, color=color, alpha=0.4, edgecolor='white',
                 linewidth=0.5, label=f'{label} (med={np.nanmedian(valid):.2f})')

    ax1.axvline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Pearson r (per-cell tuning correlation)', fontsize=10)
    ax1.set_ylabel('Number of cells', fontsize=10)
    ax1.set_title('All cells', fontsize=11, fontweight='bold')
    ax1.legend(frameon=False, fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: grouped by place cell group
    pc_groups = {
        'Both days': 'pc_both',
        'Day A only': 'pc_day_a_only',
        'Day B only': 'pc_day_b_only',
    }
    group_colors = {
        'Both days': '#4E79A7',
        'Day A only': '#E15759',
        'Day B only': '#59A14F',
    }

    comparison_keys = ['pre_pre', 'pre_post', 'post_post']
    comparison_labels = ['Pre–Pre', 'Pre–Post', 'Post–Post']

    x_pos = 0
    tick_positions = []
    tick_labels = []
    group_spacing = 1.0
    bar_width = 0.6

    for comp_key, comp_label in zip(comparison_keys, comparison_labels):
        if stability[comp_key] is None:
            continue
        corrs = stability[comp_key]['corrs']

        for group_label, mask_key in pc_groups.items():
            mask = stability[comp_key][mask_key]
            vals = corrs[mask & ~np.isnan(corrs)]

            if len(vals) == 0:
                x_pos += 1
                continue

            bp = ax2.boxplot(
                [vals], positions=[x_pos], widths=bar_width,
                patch_artist=True, showfliers=False,
            )
            bp['boxes'][0].set_facecolor(group_colors[group_label])
            bp['boxes'][0].set_alpha(0.4)
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color('black')
                    line.set_alpha(0.6)

            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax2.scatter(
                np.full(len(vals), x_pos) + jitter, vals,
                c=group_colors[group_label], alpha=0.3, s=8, zorder=3,
            )

            tick_positions.append(x_pos)
            tick_labels.append(f'{group_label}\n({len(vals)})')
            x_pos += 1

        x_pos += group_spacing  # gap between comparison conditions

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=6.5, rotation=45, ha='right')

    # Add comparison condition labels above groups
    group_start = 0
    for comp_key, comp_label in zip(comparison_keys, comparison_labels):
        if stability[comp_key] is None:
            continue
        n_groups = sum(
            1 for mk in pc_groups.values()
            if stability[comp_key][mk].any()
        )
        if n_groups > 0:
            mid = group_start + (n_groups - 1) / 2
            ax2.text(mid, ax2.get_ylim()[1] * 0.95, comp_label,
                     ha='center', va='top', fontsize=8, fontweight='bold')
            group_start += n_groups + group_spacing




    ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Pearson r', fontsize=10)
    ax2.set_title('Place cells only', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(
        pfmt.build_title('ABC Tuning Stability', trial_type='ABC', animal_id=animal_id),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_stability_by_recruitment(
    stability: dict,
    recruitment: dict[str, dict],
    comparison: str = 'pre_post',
    animal_id: str | None = None,
    figsize: tuple = (8, 5),
    show: bool = True,
) -> Figure:
    """Strip plot of per-cell tuning correlation colored by recruitment category.

    Shows how stable/shifted/gained/lost cells contribute to overall correlation changes at a specific comparison (
    e.g., pre vs post extension).
    How to interpret:
    A) stable/both cells have high r with gain/loss low r == turnover, ABC stable (map expanded to add D)
    B) stable/both cells have lower r, shape of tuning curve changed == ABC remapping after introduction of D
    *need to evaluate behaviroal data to ensure comparison

    Args:
        stability: From abc_tuning_pre_vs_post().
        recruitment: From classify_cell_recruitment().
        comparison: Which comparison to plot ('pre_pre', 'pre_post', 'post_post').
        animal_id: Animal identifier for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    if stability[comparison] is None:
        raise ValueError(f"No data for comparison '{comparison}'")

    corrs = stability[comparison]['corrs']
    day_a = stability[comparison]['day_a']
    day_b = stability[comparison]['day_b']

    # Find matching recruitment pair
    pair_key = f'{day_a}_vs_{day_b}'
    if pair_key not in recruitment:
        raise ValueError(
            f"No recruitment data for {pair_key}. "
            f"Available: {list(recruitment.keys())}"
        )

    categories = recruitment[pair_key]['categories']

    category_colors = {
        'stable': '#59A14F',
        'shifted': '#EDC948',
        'gained': '#4E79A7',
        'lost': '#E15759',
        'both': '#76B7B2',
        'absent': '#BAB0AC',
    }
    # Plot order: categories with fields first
    plot_order = [c for c in ['stable', 'shifted', 'both', 'gained', 'lost']
                  if (categories == c).any()]

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)

    box_data = []
    box_positions = []
    box_colors_list = []

    for i, cat in enumerate(plot_order):
        mask = (categories == cat) & ~np.isnan(corrs)
        vals = corrs[mask]
        if len(vals) == 0:
            continue

        bp = ax.boxplot(
            [vals], positions=[i], widths=0.5,
            patch_artist=True, showfliers=False,
        )
        color = category_colors.get(cat, '#999999')
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.4)
        for element in ['whiskers', 'caps', 'medians']:
            for line in bp[element]:
                line.set_color('black')
                line.set_alpha(0.6)

        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            c=color, alpha=0.4, s=12, zorder=3,
            edgecolors='black', linewidth=0.2,
        )

        box_positions.append(i)
        box_colors_list.append(color)

    ax.set_xticks(range(len(plot_order)))
    ax.set_xticklabels(
        [f'{cat}\n(n={int((categories == cat).sum())})'
         for cat in plot_order],
        fontsize=9,
    )
    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Pearson r (tuning correlation)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    comp_labels = {
        'pre_pre': 'Pre–Pre',
        'pre_post': 'Pre–Post',
        'post_post': 'Post–Post',
    }
    ax.set_title(
        pfmt.build_title(
            f'Tuning Stability by Recruitment ({comp_labels.get(comparison, comparison)})',
            trial_type='ABC', animal_id=animal_id,
        ),
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_bifurcation_selectivity_across_days(
    splitter_results: dict[str, dict],
    animal_id: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Track splitter cell emergence across post-extension days.

    Top: fraction of significant splitters per day.
    Bottom: heatmap of per-cell selectivity index across days.

    Args:
        splitter_results: From track_bifurcation_selectivity().
        animal_id: Animal identifier for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(splitter_results.keys())
    if not dates:
        raise ValueError("No splitter results to plot")

    n_cells = splitter_results[dates[0]]['n_cells']
    cue_id = splitter_results[dates[0]]['cue_id']

    # Collect selectivity and significance across days
    si_matrix = np.full((n_cells, len(dates)), np.nan)
    sig_fractions = []
    for j, date in enumerate(dates):
        r = splitter_results[date]
        si_matrix[:, j] = r['selectivity']
        sig_fractions.append(r['significant'].sum() / r['n_cells'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': [1, 3]})
    date_labels = [d[5:] for d in dates]

    # Top: fraction significant
    ax1.bar(range(len(dates)), sig_fractions, color='#E15759', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels([])
    ax1.set_ylabel('Fraction\nsig. splitters', fontsize=9)
    ax1.set_ylim(0, max(sig_fractions) * 1.3 if sig_fractions else 0.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.2, axis='y')

    # Bottom: heatmap of selectivity index
    # Sort cells by mean |SI| for visual clarity
    mean_abs_si = np.nanmean(np.abs(si_matrix), axis=1)
    sort_order = np.argsort(mean_abs_si)[::-1]
    sorted_si = si_matrix[sort_order]

    im = ax2.imshow(sorted_si, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                    interpolation='none')
    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Cells (sorted by |SI|)', fontsize=10)
    ax2.set_xlabel('Session', fontsize=10)
    plt.colorbar(im, ax=ax2, label='Selectivity Index', shrink=0.8)

    fig.suptitle(
        pfmt.build_title(f'Bifurcation Selectivity at Cue {cue_id}', animal_id=animal_id),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_bifurcation_activity_across_days(
    activity: dict[str, np.ndarray],
    pf_results: dict[str, PlaceFieldResult] | None = None,
    trial_type: str = 'ABC',
    introduction_day: str | None = None,
    animal_id: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> Figure:
    """Heatmap of per-cell mean activity at bifurcation cue across all days.

    Args:
        activity: From track_bifurcation_activity().
        pf_results: Optional, used to mark place cells on y-axis.
        trial_type: Trial type for place cell lookup.
        introduction_day: Date of new trial type introduction (vertical line).
        animal_id: Animal identifier for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(activity.keys())
    n_cells = len(activity[dates[0]])

    act_matrix = np.column_stack([activity[d] for d in dates])  # (n_cells, n_days)

    # Sort by peak day for visual structure
    # peak_day = np.nanargmax(act_matrix, axis=1)
    # sort_order = np.argsort(peak_day)
    mean_activity = np.nanmean(act_matrix, axis=1)
    # Sort by activity on first day (highest at top)
    sort_order = np.argsort(act_matrix[:, 0])[::-1]  # highest activity at top
    sorted_act = act_matrix[sort_order]

    has_pc_info = pf_results is not None
    if has_pc_info:
        fig, (ax_pc, ax) = plt.subplots(
            1, 2, figsize=figsize, sharey=True,
            gridspec_kw={'width_ratios': [0.15, 1], 'wspace': 0.02},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    vmax = np.nanpercentile(sorted_act, 95)
    im = ax.imshow(sorted_act, aspect='auto', cmap='magma', vmin=0, vmax=vmax,
                  interpolation='none')

    date_labels = [d[5:] for d in dates]
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cells (sorted by mean activity)', fontsize=10)
    ax.set_xlabel('Session', fontsize=10)
    plt.colorbar(im, ax=ax, label='Mean ΔF/F at bifurcation', shrink=0.8)

    if introduction_day and introduction_day in dates:
        idx = dates.index(introduction_day)
        ax.axvline(idx - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Place cell indicator panel
    if has_pc_info:
        # Binary matrix: is this cell a place cell on each day?
        pc_matrix = np.zeros((n_cells, len(dates)), dtype=float)
        for j, date in enumerate(dates):
            if date in pf_results and trial_type in pf_results[date].fields:
                pc_matrix[:, j] = pf_results[date].fields[trial_type].has_place_field.astype(float)
        sorted_pc = pc_matrix[sort_order]

        ax_pc.imshow(sorted_pc, aspect='auto', cmap='Greens', vmin=0, vmax=1,
                     interpolation='none', alpha=0.8)
        ax_pc.set_xticks(range(len(dates)))
        ax_pc.set_xticklabels([d[-2:] for d in date_labels], rotation=0, fontsize=5)
        if not has_pc_info:
            ax.set_ylabel('Cells (sorted by peak day)', fontsize=10)
        ax_pc.set_title('Place\ncell', fontsize=8, fontweight='bold')
        ax_pc.tick_params(axis='y', labelsize=6)

        if introduction_day and introduction_day in dates:
            idx = dates.index(introduction_day)
            ax_pc.axvline(idx - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title(
        pfmt.build_title('Bifurcation Point Activity Across Days', animal_id=animal_id),
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# WRAPPER


def run_recruitment_analysis(
    sessions: dict[str, dict],
    introduction_day: str,
    trial_type: str = 'ABC',
    bifurcation_cue: int | str = '0b',
    signal_col: str = 'multi_day_dff',
    detection_params: DetectionParams | None = None,
    n_shuffles_splitter: int = 500,
    field_center_tolerance_cm: float = 10.0,
    animal_id: str | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict:
    """Run full multiday recruitment analysis pipeline.

    Steps:
        1. Detect place fields on each session
        2. Classify cell recruitment across consecutive days
        3. Compare ABC tuning stability pre vs post extension
        4. Track bifurcation selectivity and activity across days

    Args:
        sessions: From load_multiday_sessions().
        introduction_day: Date of first session with new trial type.
        trial_type: Primary trial type to track (usually 'ABC').
        bifurcation_cue: Cue for bifurcation analysis (e.g., 'B' or '0b').
        signal_col: Column containing neural signals.
        detection_params: Place field detection parameters. Uses defaults if None.
        n_shuffles_splitter: Shuffle iterations for splitter p-values.
        field_center_tolerance_cm: Tolerance for stable vs shifted classification.
        animal_id: Animal identifier for titles.
        show: Call plt.show().
        save_dir: Directory to save figures (optional).

    Returns:
        Dict with keys:
            'pf_results': per-day PlaceFieldResult
            'recruitment': cell classification per day pair
            'abc_stability': pre/psot extension tuning comparison
            'splitter_tracking': per-day splitter results
            'bifurcation_activity': per-day mean activity at cue
            'figures': dict of all generated Figures
    """
    if animal_id is None:
        animal_id = next(iter(sessions.values()))['session_data'].get('animal_id', '')

    figs = {}
    results = {}

    # 1. Place field detection
    print("=" * 60)
    print("1. Place field detection across days")
    print("=" * 60)
    pf_results = detect_fields_multiday(
        sessions, signal_col=signal_col,
        params=detection_params,
    )
    results['pf_results'] = pf_results

    fig = plot_population_summary(
        pf_results, trial_type, introduction_day=introduction_day,
        animal_id=animal_id, show=show,
    )
    figs['population_summary'] = fig

    # 2. Cell recruitment classification
    print("=" * 60)
    print("2. Cell recruitment classification")
    print("=" * 60)
    recruitment = classify_cell_recruitment(
        pf_results, trial_type,
        field_center_tolerance_cm=field_center_tolerance_cm,
    )
    results['recruitment'] = recruitment

    fig = plot_recruitment_categories(
        recruitment, introduction_day=introduction_day,
        animal_id=animal_id, trial_type=trial_type, show=show,
    )
    figs['recruitment_categories'] = fig

    # 3. ABC tuning stability
    print("\n" + "=" * 60)
    print("3. ABC tuning stability pre vs post extension")
    print("=" * 60)
    stability = abc_tuning_pre_vs_post(
        sessions, pf_results, introduction_day,
        trial_type=trial_type, signal_col=signal_col,
    )
    results['abc_stability'] = stability

    fig = plot_abc_stability_comparison(stability, animal_id=animal_id, show=show)
    figs['abc_stability'] = fig

    # Stability broken down by recruitment category (pre-post only)
    if stability['pre_post'] is not None:
        # Find the matching recruitment pair
        day_a = stability['pre_post']['day_a']
        day_b = stability['pre_post']['day_b']
        pair_key = f'{day_a}_vs_{day_b}'
        if pair_key in recruitment:
            fig = plot_stability_by_recruitment(
                stability, recruitment, comparison='pre_post',
                animal_id=animal_id, show=show,
            )
            figs['stability_by_recruitment'] = fig

    # 4. Bifurcation tracking
    print("\n" + "=" * 60)
    print("4. Bifurcation selectivity tracking")
    print("=" * 60)
    splitter_tracking = track_bifurcation_selectivity(
        sessions, cue_id=bifurcation_cue, signal_col=signal_col,
        n_shuffles=n_shuffles_splitter,
    )
    results['splitter_tracking'] = splitter_tracking

    if splitter_tracking:
        fig = plot_bifurcation_selectivity_across_days(
            splitter_tracking, animal_id=animal_id, show=show,
        )
        figs['bifurcation_selectivity'] = fig

    bif_activity = track_bifurcation_activity(
        sessions, cue_id=bifurcation_cue, signal_col=signal_col,
    )
    results['bifurcation_activity'] = bif_activity

    if bif_activity:
        fig = plot_bifurcation_activity_across_days(
            bif_activity, pf_results=pf_results, trial_type=trial_type,
            introduction_day=introduction_day, animal_id=animal_id, show=show,
        )
        figs['bifurcation_activity'] = fig

    results['figures'] = figs

    # Save
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            path = save_dir / f'recruitment_{name}.png'
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

    print(f"\nDone. Generated {len(figs)} figures.")
    return results



if __name__ == '__main__':

    from df_processing import load_multiday_sessions

    mouse_id = '26'
    date = '2025-09-08'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)


    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-02', '2025-09-10'), auto_process=False,
    )

    results = run_recruitment_analysis(
        sessions,
        introduction_day='2025-09-08',  # first day with ABDC (9-15 is actually last day, first is 9-08)
        trial_type='ABC',
        bifurcation_cue='0b',   #0b is gray zone before C, B is the cue before C
        animal_id=mouse_id,
    )