"""
Place Field Plotting
====================
Visualization for place field results: per-trial-type heatmaps and combined heatmaps.

Free functions; PlaceFields1d.plot() is a thin wrapper around plot_place_fields() for
backwards compatibility.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops

from place_field_detection import PlaceFields1d, PlaceFieldResult
from df_processing import get_track_length, get_cue_regions
import plot_utils as pfmt


# SINGLE TRIAL TYPE HEATMAP

def plot_place_fields(
    pf: PlaceFields1d,
    title: str | None = None,
    sort: bool = True,
    cells: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
    config: dict | None = None,
    trial_type: str | None = None,
    animal_id: str | None = None,
    date: str | None = None,
    show_cue_boundaries: bool = True,
    normalize_rows: bool = True,
) -> plt.Figure:
    """Plot place field activity as a sorted heatmap.

    Args:
        pf: PlaceFields1d to plot.
        title: Plot title description. Passed to build_title if config provided.
        sort: Sort cells by field position.
        cells: Boolean mask or indices of cells to include.
        vmin: Min color scale value (default: 0 if normalize_rows else 10th percentile).
        vmax: Max color scale value (default: 1 if normalize_rows else 99th percentile).
        dpi: Figure resolution.
        config: Experiment config dict. Required for cue bar and build_title.
        trial_type: Trial type string. Required for cue bar.
        animal_id: Animal ID for title. Used only if config provided.
        date: Session date string for title. Used only if config provided.
        show_cue_boundaries: If True, draws dashed vertical lines at cue boundaries.
        normalize_rows: If True, scale each cell's tuning curve to [0, 1] before plotting
            so every cell's peak is at the colorbar maximum.

    Returns:
        Matplotlib Figure.
    """
    if config is None or trial_type is None:
        raise ValueError("config and trial_type are required for plot_place_fields()")

    data = pf.binF
    order = pf.order if sort else np.arange(data.shape[0])

    if cells is not None:
        cells = np.atleast_1d(cells)
        if cells.dtype == bool:
            cells = np.where(cells)[0]
        order = order[np.isin(order, cells)]

    data = data[order, :]

    if normalize_rows:
        row_max = np.nanmax(data, axis=1, keepdims=True)
        data = data / np.where(row_max > 0, row_max, 1.0)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
    else:
        if vmin is None:
            vmin = np.nanquantile(data, 0.1)
        if vmax is None:
            vmax = np.nanquantile(data, 0.99)

    track_len_cm = pf.bin_size_cm * data.shape[1]
    figsize = (4.8, 6.5)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 30],
        width_ratios=[20, 1],
        hspace=0.02,
        wspace=0.03,
    )
    ax_cue = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])

    extent = [0, track_len_cm, 1, data.shape[0] + 1]
    im = ax.imshow(data, cmap='magma', extent=extent, interpolation='none',
                   vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Cell #')
    ax.set_xlim(0, track_len_cm)
    fig.colorbar(im, cax=ax_cb, label='Normalized ΔF/F' if normalize_rows else 'ΔF/F')

    cue_colors = pfmt.get_cue_colors(config)
    cue_labels_map = pfmt.get_cue_labels(config)
    regions = get_cue_regions(config, trial_type)

    ax_cue.set_xlim(0, track_len_cm)
    ax_cue.set_yticks([])
    ax_cue.tick_params(bottom=False, labelbottom=False)
    ax_cue.spines[:].set_visible(False)

    for cue_id, spans in regions.items():
        color = cue_colors.get(cue_id, '#CCCCCC')
        label = cue_labels_map.get(cue_id, '')
        for start, end in spans:
            ax_cue.axvspan(start, end, color=color, alpha=0.8)
            if cue_id != 0:
                ax_cue.text((start + end) / 2, 0.5, label,
                            ha='center', va='center', fontsize=9,
                            fontweight='bold', color='white')

    ts = config.get('trial_structures', {}).get(trial_type, {})
    cue_widths = config.get('cue_map', {})
    tick_positions = [0.0]
    pos = 0.0
    for cue_id in ts.get('cue_sequence', []):
        pos += cue_widths[cue_id]
        tick_positions.append(pos)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t)) for t in tick_positions], fontsize=8)

    if show_cue_boundaries:
        pfmt.add_cue_boundary_lines(ax, config, trial_type, axis='x', alpha=0.6)

    title_str = pfmt.build_title(
        title or 'Place Fields',
        trial_type=trial_type,
        animal_id=animal_id,
        date=date,
    )
    ax_cue.set_title(title_str, fontsize=11, fontweight='bold', pad=6)
    plt.subplots_adjust(left=0.18, right=0.83, bottom=0.1, top=0.93)

    return fig


# COMBINED HEATMAP (all trial types side-by-side)

def plot_combined_heatmap(
    result: PlaceFieldResult,
    config: dict,
    session_data: dict,
    trial_types: list[str] | None = None,
    cells: np.ndarray | None = None,
    sort_by: str | None = None,
    bin_size_cm: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple | None = None,
    dpi: int = 150,
    show_cue_boundaries: bool = True,
    show: bool = True,
    normalize_rows: bool = True,
    drop_unsorted: bool = True,
) -> plt.Figure:
    """Plot heatmap with trial types concatenated horizontally.

    Concatenates tuning curves from multiple trial types side by side, with a
    dashed line at each boundary. Cells sorted by field position. Cue regions
    annotated across the full width.

    Args:
        result: PlaceFieldResult from detect_place_fields().
        config: Experiment configuration dict.
        session_data: Session metadata dict (for title).
        trial_types: Which trial types to include, in order. Default: all, sorted.
        cells: Cell indices or boolean mask to include. Default: all place cells.
        sort_by: Trial type to sort cells by field position. Default: first trial type.
        bin_size_cm: Spatial bin size in cm.
        vmin: Min color scale (default: 0 if normalize_rows else 10th percentile).
        vmax: Max color scale (default: 1 if normalize_rows else 99th percentile).
        figsize: Figure size. Auto-scaled if None.
        dpi: Figure resolution.
        show_cue_boundaries: If false, suppresses dashed lines at cue boundaries within each trial type.
        show: Call plt.show().
        normalize_rows: If True, scale each cell's tuning curve to [0, 1] before plotting.
        drop_unsorted: If True, drop cells with no detected field in sort_by so they don't
            appear as an unsorted noise band at the bottom.

    Returns:
        Matplotlib Figure.
    """
    if trial_types is None:
        trial_types = sorted(result.fields.keys())
    if sort_by is None:
        sort_by = trial_types[0]

    segments = []
    boundaries = [0]
    for tt in trial_types:
        binF = result.fields[tt].binF
        segments.append(binF)
        boundaries.append(boundaries[-1] + binF.shape[1])

    combined = np.concatenate(segments, axis=1)

    if cells is not None:
        cells = np.atleast_1d(cells)
        if cells.dtype == bool:
            cells = np.where(cells)[0]
    else:
        cells = np.where(result.is_place_cell_any)[0]

    cells_set = set(cells)

    sort_pf = result.fields[sort_by]
    sort_key = np.full(combined.shape[0], np.inf)
    best_intensity = np.full(combined.shape[0], -np.inf)

    if sort_pf.label_im.max() > 0:
        for prop in regionprops(sort_pf.label_im, sort_pf.binF, cache=False):
            cell_idx = prop['coords'][0, 0]
            if cell_idx in cells_set:
                mean_int = prop['mean_intensity']
                if mean_int > best_intensity[cell_idx]:
                    best_intensity[cell_idx] = mean_int
                    sort_key[cell_idx] = prop['weighted_centroid'][1]

    if drop_unsorted:
        cells = cells[np.isfinite(sort_key[cells])]
    order = cells[np.argsort(sort_key[cells])]
    data = combined[order, :]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    if normalize_rows:
        row_max = np.nanmax(data, axis=1, keepdims=True)
        data = data / np.where(row_max > 0, row_max, 1.0)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
    else:
        if vmin is None:
            vmin = np.nanquantile(data, 0.1)
        if vmax is None:
            vmax = np.nanquantile(data, 0.99)

    total_cm = sum(get_track_length(config, tt) for tt in trial_types)
    n_cells_plot = len(order)
    if figsize is None:
        figsize = (8, 6)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 30],
        hspace=0.02,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_cue = fig.add_subplot(gs[0, 0])

    extent = [0, total_cm, n_cells_plot, 0]
    im = ax.imshow(data, cmap='magma', extent=extent, interpolation='none',
                   vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Neuron #')

    ax_cb = ax.inset_axes([1.02, 0.0, 0.02, 1.0])
    plt.colorbar(im, cax=ax_cb, label='Normalized ΔF/F' if normalize_rows else 'ΔF/F')

    for b in boundaries[1:-1]:
        x = b * bin_size_cm
        ax.axvline(x, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax_cue.axvline(x, color='black', linestyle='--', linewidth=1, alpha=0.5)

    if show_cue_boundaries:
        x_offset = 0
        for tt in trial_types:
            ts = config.get('trial_structures', {}).get(tt, {})
            seq = ts.get('cue_sequence', [])
            cue_widths = config.get('cue_map', {})
            pos = float(x_offset)
            for cue_id in seq[:-1]:
                pos += cue_widths[cue_id]
                ax.axvline(pos, color='white', linestyle='--', linewidth=0.8, alpha=0.5, zorder=5)
                ax_cue.axvline(pos, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
            x_offset += get_track_length(config, tt)

    cue_colors = pfmt.get_cue_colors(config)
    cue_labels_map = pfmt.get_cue_labels(config)
    x_offset = 0
    for tt in trial_types:
        regions = get_cue_regions(config, tt)
        for cue_id, spans in regions.items():
            color = cue_colors.get(cue_id, '#CCCCCC')
            label = cue_labels_map.get(cue_id, '')
            for start, end in spans:
                x0 = x_offset + start
                x1 = x_offset + end
                ax_cue.axvspan(x0, x1, color=color, alpha=0.8)
                if cue_id != 0:
                    ax_cue.text((x0 + x1) / 2, 0.5, label,
                                ha='center', va='center', fontsize=9,
                                fontweight='bold', color='white')
        x_offset += get_track_length(config, tt)

    tick_positions = []
    tick_labels = []
    x_offset = 0
    for tt in trial_types:
        ts = config.get('trial_structures', {}).get(tt, {})
        seq = ts.get('cue_sequence', [])
        cue_widths = config.get('cue_map', {})
        pos = float(x_offset)
        tick_positions.append(pos)
        tick_labels.append('0')
        for cue_id in seq:
            pos += cue_widths[cue_id]
            tick_positions.append(pos)
            tick_labels.append(str(int(pos - x_offset)))
        x_offset += get_track_length(config, tt)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlim(0, total_cm)
    ax_cue.set_xlim(0, total_cm)

    fig.canvas.draw()
    ax_pos = ax.get_position()
    cue_pos = ax_cue.get_position()
    ax_cue.set_position([ax_pos.x0, cue_pos.y0, ax_pos.width, cue_pos.height])
    ax_cue.set_autoscalex_on(False)
    ax_cue.set_yticks([])
    ax_cue.tick_params(bottom=False, labelbottom=False)
    ax_cue.spines[:].set_visible(False)

    animal_id = session_data.get('animal_id', '') if session_data else ''
    date = session_data.get('session_name', '')[:10] if session_data else ''
    ax_cue.set_title(
        pfmt.build_title(
            f'Place Fields ({n_cells_plot} cells, sorted by {sort_by})',
            animal_id=animal_id, date=date,
        ),
        fontsize=11, fontweight='bold', pad=6,
    )

    fig.subplots_adjust(top=0.93, bottom=0.1, left=0.11, right=0.87)

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    from df_processing import find_session_dir, load_session_context
    from place_field_detection import DetectionParams
    from experiment_place_cells import (
        detect_place_fields_for_session,
        compute_experiment_place_cells,
        load_experiment_place_cells,
        CACHE_FILENAME,
    )

    mouse_id = "26"
    mouse_dir = Path("/Users/cs963/Desktop/sun_lab_projects/datasets", mouse_id)
    date = "2025-09-16"

    # Toggle to rebuild caches from scratch.
    force_recompute = True

    params=None
    #params = DetectionParams(smooth_sigma=1, signal_threshold=0.2)
    signal_col = "multi_day_dff"

    # Per-session detection — loads from {session_dir}/*_place_fields_*.pkl if present.
    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)

    result = detect_place_fields_for_session(
        session_dir, signal_col=signal_col, params=params,
        force_recompute=force_recompute,
    )

    # Experiment-wide PCs across all sessions — auto-loads from
    # {mouse_dir}/experiment_place_cells.npz if present and params match.
    exp_cache = mouse_dir / CACHE_FILENAME
    if exp_cache.exists() and not force_recompute:
        try:
            exp_pcs = load_experiment_place_cells(mouse_dir)
            print(f"Loaded experiment PCs cache: {exp_cache.name}")
        except Exception as e:
            print(f"Cache load failed ({e}); recomputing.")
            exp_pcs = compute_experiment_place_cells(
                mouse_dir, signal_col=signal_col, params=params, force=True,
            )
    else:
        exp_pcs = compute_experiment_place_cells(
            mouse_dir, signal_col=signal_col, params=params,
            force=force_recompute,
        )

    union_cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)
    print(f"Plotting {len(union_cells)} cross-day place cells (≥2 days, any trial type)")

    # Per-trial-type heatmap restricted to cross-day PCs
    for tt in result.fields:
        plot_place_fields(
            result.fields[tt], config=exp_config, trial_type=tt,
            cells=union_cells, animal_id=mouse_id, date=date,
        )
        plt.show()

    # Combined heatmap, cross-day PCs only, sorted each way
    for sort_by in ('ABC', 'ABDC'):
        if sort_by in result.fields:
            plot_combined_heatmap(
                result, exp_config, session_data,
                cells=union_cells, sort_by=sort_by,
            )
