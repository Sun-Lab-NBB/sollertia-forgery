"""
Bifurcation tuning across days — visualize how spatial tuning curves
at the B–0b region change across recording sessions.

The main question here is do cells in the "bifurcation zone" change their tuning once the extension is added?
i.e. once the bifurcation exists.  Until the extension there is no bifurcation; then it switches to a  probabilistic
state transition

Standalone module for testing. Can be integrated into
multiday_place_field_comparison.py later.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')

from df_processing import get_cue_regions, load_multiday_sessions
from place_field_detection import (
    detect_place_fields, PlaceFieldResult, PlaceFields1d, DetectionParams,
)
from cross_correlation_1 import get_mean_tuning_curves
import plot_utils as pfmt


# HELPERS


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
        best = np.argmax(intensities[field_mask])
        field_indices = np.where(field_mask)[0]
        centers[cell_idx] = field_centers[field_indices[best]]

    return centers


def get_bifurcation_bin_range(
    config: dict,
    trial_type: str = 'ABC',
    bin_size_cm: int = 5,
) -> tuple[int, int]:
    """Get bin range spanning cue B through the end of gap 0b.

    For ABC with sequence [1, 0, 2, 0, 3, 0], B is cue 2 and 0b is the
    gap immediately after B. This is the bifurcation zone — shared between
    ABC and ABDC before physical divergence.

    Args:
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout lookup.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        (start_bin, end_bin) inclusive/exclusive.
    """
    regions = get_cue_regions(config, trial_type)
    cue_sequence = config['trial_structures'][trial_type]['cue_sequence']
    cue_widths = config.get('cue_map', {})

    # B start from regions
    b_start_cm = regions[2][0][0]

    # Walk sequence to find the gap (0) immediately after cue B (2)
    position = 0.0
    found_b = False
    ob_end_cm = None
    for cue_id in cue_sequence:
        width = cue_widths[cue_id]
        if found_b and cue_id == 0:
            ob_end_cm = position + width
            break
        if cue_id == 2:
            found_b = True
        position += width

    if ob_end_cm is None:
        raise ValueError("Could not find gap after cue B in sequence")

    return int(b_start_cm / bin_size_cm), int(ob_end_cm / bin_size_cm)


# MAIN PLOT


def plot_bifurcation_tuning_across_days(
    sessions: dict[str, dict],
    pf_results: dict[str, PlaceFieldResult],
    trial_type: str = 'ABC',
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    cell_selection: str = 'overlap',
    sort_by_day: str | None = None,
    introduction_day: str | None = None,
    animal_id: str | None = None,
    figsize_per_day: tuple[float, float] = (2.5, 6),
    show: bool = True,
) -> Figure:
    """Heatmap of spatial tuning at the bifurcation zone (B–0b) across days.

    One column per day, rows = cells, x-axis = spatial bins in B–0b.
    Cells sorted by field center on first day they have a field.

    Args:
        sessions: From load_multiday_sessions().
        pf_results: From detect_fields_multiday().
        trial_type: Trial type to extract tuning curves from.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        cell_selection: How to select cells:
            'center' — primary field center within B–0b on any day.
            'overlap' — any field boundary overlapping B–0b on any day.
        sort_by_day: Set which day to be the reference for cell sorting
        introduction_day: Date of new trial type introduction (red title).
        animal_id: Animal identifier for title.
        figsize_per_day: (width, height) per day panel.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    dates = sorted(
        d for d in sessions
        if d in pf_results and trial_type in pf_results[d].fields
    )
    if len(dates) < 2:
        raise ValueError(f"Need ≥2 days with {trial_type}, got {len(dates)}")

    # Bifurcation bin range from first session's config
    config_ref = sessions[dates[0]]['config']
    bif_start_bin, bif_end_bin = get_bifurcation_bin_range(
        config_ref, trial_type, bin_size_cm,
    )
    n_bif_bins = bif_end_bin - bif_start_bin
    bif_start_cm = bif_start_bin * bin_size_cm
    bif_end_cm = bif_end_bin * bin_size_cm

    # Extract tuning curves at bifurcation zone per day
    tuning_per_day = {}
    for date in dates:
        tc = get_mean_tuning_curves(
            sessions[date]['data'], sessions[date]['config'],
            sessions[date]['metadata'], signal_col=signal_col,
        )
        if trial_type in tc:
            tuning_per_day[date] = tc[trial_type][bif_start_bin:bif_end_bin]

    if not tuning_per_day:
        raise ValueError("No tuning curves extracted")

    n_cells = tuning_per_day[dates[0]].shape[1]

    # Select cells with fields in bifurcation zone on any day
    selected_mask = np.zeros(n_cells, dtype=bool)

    for date in dates:
        pf = pf_results[date].fields[trial_type]

        if cell_selection == 'center':
            centers = _get_primary_field_centers(pf, n_cells)
            selected_mask |= (centers >= bif_start_cm) & (centers < bif_end_cm)

        elif cell_selection == 'overlap':
            label_region = pf.label_im[:, bif_start_bin:bif_end_bin]
            selected_mask |= np.any(label_region > 0, axis=1)

        else:
            raise ValueError(f"Unknown cell_selection: {cell_selection!r}")

    selected_indices = np.where(selected_mask)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        print("No cells with fields in bifurcation zone on any day.")
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, 'No cells with fields in bifurcation zone',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    print(f"Selected {n_selected}/{n_cells} cells ({cell_selection} mode)")

    # Sort by field center on first day each cell has a field
    sort_key = np.full(n_selected, np.inf)
    if sort_by_day is not None:
        if sort_by_day not in pf_results or trial_type not in pf_results[sort_by_day].fields:
            raise ValueError(f"sort_by_day={sort_by_day!r} not in pf_results")
        pf = pf_results[sort_by_day].fields[trial_type]
        centers = _get_primary_field_centers(pf, n_cells)
        for i, cell_idx in enumerate(selected_indices):
            sort_key[i] = centers[cell_idx]  # NaN → inf via np.full default
    else:
        for i, cell_idx in enumerate(selected_indices):
            for date in dates:
                pf = pf_results[date].fields[trial_type]
                centers = _get_primary_field_centers(pf, n_cells)
                if not np.isnan(centers[cell_idx]):
                    sort_key[i] = centers[cell_idx]
                    break
    sort_order = np.argsort(sort_key)

    # Build figure
    n_days = len(dates)
    fig_width = figsize_per_day[0] * n_days + 2.5
    fig_height = figsize_per_day[1]
    # fig, axes = plt.subplots(
    #     1, n_days, figsize=(fig_width, fig_height), sharey=True,
    #     gridspec_kw={'wspace': 0.15},
    # )
    fig, axes = plt.subplots(
        1, n_days, figsize=(fig_width, fig_height), sharey=True,
        constrained_layout=True,
    )
    if n_days == 1:
        axes = [axes]

    # Global color scale
    all_vals = np.concatenate([
        tuning_per_day[d][:, selected_indices].ravel()
        for d in dates if d in tuning_per_day
    ])
    vmax = np.nanpercentile(all_vals, 95)

    for j, (date, ax) in enumerate(zip(dates, axes)):
        if date in tuning_per_day:
            tc_subset = tuning_per_day[date][:, selected_indices]
            tc_sorted = tc_subset[:, sort_order].T  # (n_selected, n_bif_bins)
        else:
            tc_sorted = np.full((n_selected, n_bif_bins), np.nan)

        im = ax.imshow(
            tc_sorted, aspect='auto', cmap='magma', vmin=0, vmax=vmax,
            interpolation='none',
            extent=[bif_start_cm, bif_end_cm, n_selected, 0],
        )

        ax.set_xlabel('Position (cm)', fontsize=9)

        # Mark B | 0b boundary
        b_end_cm = get_cue_regions(config_ref, trial_type)[2][0][1]
        ax.axvline(b_end_cm, color='white', linestyle='--',
                   linewidth=0.8, alpha=0.7)

        # Title: red for introduction day
        title_color = 'red' if (introduction_day and date == introduction_day) else 'black'
        ax.set_title(date[5:], fontsize=10, fontweight='bold', color=title_color)

        if j == 0:
            ax.set_ylabel(
                f'Cells (n={n_selected}, sorted by field center)', fontsize=9,
            )

    fig.colorbar(im, ax=list(axes), label='Mean ΔF/F', shrink=0.5, pad=0.02)

    selection_label = 'center in' if cell_selection == 'center' else 'overlapping'
    fig.suptitle(
        pfmt.build_title(
            f'Bifurcation Tuning B–0b ({selection_label})',
            trial_type=trial_type, animal_id=animal_id,
        ),
        fontsize=13, fontweight='bold',
    )
    #plt.tight_layout()


    if show:
        plt.show()
    return fig




if __name__ == '__main__':
    from pathlib import Path
    from multiday_place_field_comparison import detect_fields_multiday

    mouse_id = '14'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-08-18', '2025-08-22'), auto_process=False,
        signal_cols=['multi_day_dff'],
    )

    params = DetectionParams(signal_type='dff', signal_threshold=.5, min_peak=.1)
    pf_results = detect_fields_multiday(sessions, params=params)

    for type in ['ABC', 'ABDC']:
        fig = plot_bifurcation_tuning_across_days(
                sessions, pf_results,
                trial_type=type,
                cell_selection='center',        #could also use overlap but more liberal
                introduction_day='2025-08-22',
                animal_id=mouse_id,
            )
