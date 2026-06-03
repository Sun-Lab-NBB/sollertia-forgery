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
    normalize_rows: bool = False,
) -> plt.Figure:
    """Plot place field activity as a sorted heatmap.

    Args:
        pf: PlaceFields1d to plot.
        title: Plot title description. Passed to build_title if config provided.
        sort: Sort cells by field position.
        cells: Boolean mask or indices of cells to include.
        vmin: Min color scale value (default: 0 if normalize_rows else 50th percentile).
        vmax: Max color scale value (default: 1 if normalize_rows else 90th percentile).
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
            vmin = np.nanquantile(data, 0.5)
        if vmax is None:
            vmax = np.nanquantile(data, 0.9)

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

    extent = [0, track_len_cm, data.shape[0], 0]
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
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple | None = None,
    dpi: int = 150,
    show_cue_boundaries: bool = True,
    show: bool = True,
    normalize_rows: bool = False,
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
        vmin: Min color scale (default: 0 if normalize_rows else 50th percentile).
        vmax: Max color scale (default: 1 if normalize_rows else 90th percentile).
        figsize: Figure size. Auto-scaled if None.
        dpi: Figure resolution.
        show_cue_boundaries: If false, suppresses dashed lines at cue boundaries within each trial type.
        show: Call plt.show().
        normalize_rows: If True, scale each cell's tuning curve to [0, 1] before plotting.
        drop_unsorted: If True, drop cells with no detected field in any of the
            displayed trial types. Cells with a field only in a non-``sort_by``
            trial type are kept and sort to the bottom — useful for seeing, e.g.,
            ABDC-only cells when sort_by='ABC'.

    Returns:
        Matplotlib Figure.
    """
    if trial_types is None:
        trial_types = sorted(result.fields.keys())
    if sort_by is None:
        sort_by = trial_types[0]

    bin_size_cm = float(result.fields[trial_types[0]].bin_size_cm)

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
        # Drop cells with no detected field in ANY displayed trial type.
        # Cells with a field in some displayed trial type other than sort_by
        # keep an inf sort_key here, so they sort to the bottom — letting you
        # see (e.g.) ABDC-only cells when sort_by='ABC'.
        has_any_field = np.zeros(combined.shape[0], dtype=bool)
        for tt in trial_types:
            has_any_field |= result.fields[tt].has_place_field
        cells = cells[has_any_field[cells]]
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
            vmin = np.nanquantile(data, 0.5)
        if vmax is None:
            vmax = np.nanquantile(data, 0.9)

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


# MULTIDAY GRID

def _panel_sort_key(pf: PlaceFields1d, cells: np.ndarray) -> np.ndarray:
    """For each input cell index, return field centroid (in bins) of its best field, or +inf."""
    sort_key = np.full(pf.binF.shape[0], np.inf)
    best = np.full(pf.binF.shape[0], -np.inf)
    if pf.label_im.max() > 0:
        cells_set = set(int(c) for c in cells)
        for prop in regionprops(pf.label_im, pf.binF, cache=False):
            cell_idx = prop['coords'][0, 0]
            if cell_idx in cells_set and prop['mean_intensity'] > best[cell_idx]:
                best[cell_idx] = prop['mean_intensity']
                sort_key[cell_idx] = prop['weighted_centroid'][1]
    return sort_key


def _draw_heatmap_panel(
    ax: plt.Axes,
    binF: np.ndarray,
    row_order: np.ndarray,
    track_cm: float,
    normalize_rows: bool,
    vmin: float | None,
    vmax: float | None,
    boundaries_cm: list[float] | None = None,
    row_label: str | None = None,
):
    """Render one heatmap panel inside the multiday grid."""
    data = binF[row_order, :].astype(float)
    if normalize_rows:
        row_max = np.nanmax(data, axis=1, keepdims=True)
        data = data / np.where(row_max > 0, row_max, 1.0)
        v0 = 0.0 if vmin is None else vmin
        v1 = 1.0 if vmax is None else vmax
    else:
        v0 = float(np.nanquantile(data, 0.5)) if vmin is None else vmin
        v1 = float(np.nanquantile(data, 0.9)) if vmax is None else vmax

    extent = [0, track_cm, data.shape[0], 0]
    ax.imshow(data, cmap='magma', extent=extent, interpolation='none',
              vmin=v0, vmax=v1, aspect='auto')
    ax.set_xlim(0, track_cm)
    ax.tick_params(labelsize=7)
    if boundaries_cm:
        for x in boundaries_cm:
            ax.axvline(x, color='white', linestyle='--', linewidth=1.0, alpha=0.8)
    if row_label:
        ax.set_ylabel(row_label, fontsize=8)


def plot_multiday_heatmap_grid(
    mouse_dir: Path,
    dates: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    signal_col: str = 'multi_day_dff',
    params=None,
    cell_selection: str = 'per_day',
    sort_by: str = 'ABDC',
    trial_types: tuple[str, str] = ('ABC', 'ABDC'),
    min_days: int = 2,
    sort_reference_date: str | None = None,
    max_days_per_row: int = 3,
    combined_only: bool = False,
    force_recompute: bool = False,
    normalize_rows: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple | None = None,
    dpi: int = 130,
    show: bool = True,
) -> plt.Figure:
    """Grid of per-day heatmaps: three panels per day (single ABC, single ABDC, combined).

    Days flow left-to-right with at most ``max_days_per_row`` per row, then stack.
    Each day cell contains three stacked heatmaps (ABC on top, ABDC middle, combined bottom).

    Args:
        mouse_dir: Mouse root directory passed to session loaders.
        dates: Explicit list of session dates (YYYY-MM-DD). Mutually exclusive with date_range.
        date_range: (start, end) inclusive range to auto-discover sessions.
        signal_col: Neural signal column for detection.
        params: DetectionParams. Defaults if None.
        cell_selection: One of:
            - 'per_day': each panel independently shows that day's place cells
              (cells with a field in that panel's trial type), sorted by that panel.
            - 'union': cells that are place cells in ANY trial type on >= ``min_days`` days
              across the experiment. Same cell order in every panel and every day.
            - 'intersection': cells that are place cells in ``sort_by`` on every day where
              ``sort_by`` ran. Same cell order in every panel and every day.
        sort_by: Trial type used for cell ordering (and intersection criterion).
        trial_types: The two single-trial-type panels to plot. The combined panel
            concatenates them in this order.
        min_days: For 'union' selection.
        sort_reference_date: For 'union'/'intersection', the day whose ``sort_by``
            field positions define cell order. Defaults to the first day where
            ``sort_by`` ran.
        max_days_per_row: Number of day columns before wrapping to a new row.
        combined_only: If True, draw only the combined (ABC|ABDC) panel per day and
            drop any day that doesn't have both trial types. ``sort_by`` still
            controls cell ordering of the combined heatmap.
        force_recompute: If True, ignore cached per-session detection results.
        normalize_rows: Row-normalize each cell's tuning curve to [0, 1].
        vmin, vmax: Color scale. Defaults: 50th/90th percentile (or 0/1 if normalize_rows).
        figsize: Defaults to (4 * n_cols, 3 * n_day_rows).
        dpi: Figure resolution.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    from experiment_place_cells import (
        detect_place_fields_for_session, compute_experiment_place_cells,
    )
    from df_processing import find_session_dir, load_session_context
    from place_field_detection import DetectionParams

    if cell_selection not in {'per_day', 'union', 'intersection'}:
        raise ValueError(f"cell_selection must be 'per_day'|'union'|'intersection', got {cell_selection!r}")

    if params is None:
        params = DetectionParams()

    mouse_dir = Path(mouse_dir)

    # Resolve session dates
    if dates is not None and date_range is not None:
        raise ValueError("Provide either dates or date_range, not both.")
    if date_range is not None:
        start, end = date_range
        dates = sorted(
            d.name[:10] for d in mouse_dir.iterdir()
            if d.is_dir() and len(d.name) >= 10 and start <= d.name[:10] <= end
        )
        dates = sorted(set(dates))
    if not dates:
        raise ValueError("No session dates resolved for the grid.")

    # Per-session detection (cached)
    per_day_results: dict[str, PlaceFieldResult] = {}
    config_ref = None
    for date in dates:
        session_dir = find_session_dir(mouse_dir, date)
        result = detect_place_fields_for_session(
            session_dir, signal_col=signal_col, params=params,
            force_recompute=force_recompute,
        )
        per_day_results[date] = result
        if config_ref is None:
            _, config_ref = load_session_context(session_dir)

    tt_a, tt_b = trial_types

    if combined_only:
        dropped = [d for d in dates
                   if tt_a not in per_day_results[d].fields
                   or tt_b not in per_day_results[d].fields]
        if dropped:
            print(f"combined_only: skipping {len(dropped)} day(s) missing "
                  f"{tt_a} or {tt_b}: {dropped}")
        dates = [d for d in dates if d not in dropped]
        per_day_results = {d: per_day_results[d] for d in dates}
        if not dates:
            raise ValueError(
                f"No selected day has both {tt_a} and {tt_b} for combined_only."
            )

    def _track_cm_for(tt: str) -> float | None:
        cm = get_track_length(config_ref, tt)
        if cm is not None:
            return float(cm)
        for r in per_day_results.values():
            pf = r.fields.get(tt)
            if pf is not None:
                return float(pf.binF.shape[1] * pf.bin_size_cm)
        return None

    track_a = _track_cm_for(tt_a)
    track_b = _track_cm_for(tt_b)
    if track_a is None or track_b is None:
        missing = [tt for tt, v in [(tt_a, track_a), (tt_b, track_b)] if v is None]
        raise ValueError(
            f"Trial type(s) {missing} not found in config or any selected day's results."
        )
    track_comb = track_a + track_b

    # Cell set + shared sort order for union/intersection
    shared_cells: np.ndarray | None = None
    shared_order: np.ndarray | None = None
    if cell_selection in {'union', 'intersection'}:
        exp_pcs = compute_experiment_place_cells(
            mouse_dir, signal_col=signal_col, params=params, force=force_recompute, save=True,
        )
        if cell_selection == 'union':
            shared_cells = exp_pcs.place_cells_in_any_trial_type(min_days=min_days)
        else:
            if sort_by not in exp_pcs.presence:
                raise ValueError(f"sort_by={sort_by!r} not seen in experiment trial types")
            existed = exp_pcs.trial_type_existed[sort_by]
            shared_cells = np.where(exp_pcs.presence[sort_by][:, existed].all(axis=1))[0]

        ref_date = sort_reference_date
        if ref_date is None:
            for d in dates:
                if sort_by in per_day_results[d].fields:
                    ref_date = d
                    break
        if ref_date is None or sort_by not in per_day_results[ref_date].fields:
            raise ValueError(f"No reference day has trial type {sort_by!r} for sorting.")
        ref_key = _panel_sort_key(per_day_results[ref_date].fields[sort_by], shared_cells)
        shared_cells = shared_cells[np.isfinite(ref_key[shared_cells])]
        shared_order = shared_cells[np.argsort(ref_key[shared_cells])]

    # Layout
    n_days = len(dates)
    n_cols = min(n_days, max_days_per_row)
    n_day_rows = (n_days + n_cols - 1) // n_cols
    panels_per_day = 1 if combined_only else 3
    if figsize is None:
        per_day_h = 2.6 if combined_only else 3.3
        figsize = (4.2 * n_cols, per_day_h * n_day_rows)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        n_day_rows * panels_per_day, n_cols,
        hspace=0.45 if not combined_only else 0.55,
        wspace=0.3,
    )

    for d_idx, date in enumerate(dates):
        grid_row = (d_idx // n_cols) * panels_per_day
        grid_col = d_idx % n_cols

        result = per_day_results[date]
        pf_a = result.fields.get(tt_a)
        pf_b = result.fields.get(tt_b)

        # Determine row order for this day
        if cell_selection == 'per_day':
            cells_any = np.where(result.is_place_cell_any)[0]
            if pf_a is not None:
                key_a = _panel_sort_key(pf_a, cells_any)
                cells_a = cells_any[np.isfinite(key_a[cells_any])]
                order_a = cells_a[np.argsort(key_a[cells_a])]
            else:
                order_a = np.array([], dtype=int)
            if pf_b is not None:
                key_b = _panel_sort_key(pf_b, cells_any)
                cells_b = cells_any[np.isfinite(key_b[cells_any])]
                order_b = cells_b[np.argsort(key_b[cells_b])]
            else:
                order_b = np.array([], dtype=int)
            sort_pf = result.fields.get(sort_by)
            if sort_pf is not None and len(cells_any) > 0:
                key_c = _panel_sort_key(sort_pf, cells_any)
                cells_c = cells_any[np.isfinite(key_c[cells_any])]
                order_c = cells_c[np.argsort(key_c[cells_c])]
            else:
                order_c = cells_any
        else:
            order_a = order_b = order_c = shared_order

        if combined_only:
            ax_c = fig.add_subplot(gs[grid_row, grid_col])
            if pf_a is not None and pf_b is not None and len(order_c) > 0:
                combined = np.concatenate([pf_a.binF, pf_b.binF], axis=1)
                _draw_heatmap_panel(
                    ax_c, combined, order_c, track_comb,
                    normalize_rows, vmin, vmax,
                    boundaries_cm=[track_a],
                    row_label=f'{tt_a}|{tt_b}',
                )
            else:
                ax_c.set_axis_off()
                ax_c.text(0.5, 0.5, 'no combined', ha='center', va='center',
                          fontsize=8, transform=ax_c.transAxes)
            ax_c.set_title(date, fontsize=10, fontweight='bold', pad=4)
            ax_c.set_xlabel('Position (cm)', fontsize=8)
        else:
            ax_a = fig.add_subplot(gs[grid_row, grid_col])
            ax_b = fig.add_subplot(gs[grid_row + 1, grid_col])
            ax_c = fig.add_subplot(gs[grid_row + 2, grid_col])

            if pf_a is not None and len(order_a) > 0:
                _draw_heatmap_panel(
                    ax_a, pf_a.binF, order_a, track_a,
                    normalize_rows, vmin, vmax, row_label=tt_a,
                )
            else:
                ax_a.set_axis_off()
                ax_a.text(0.5, 0.5, f'no {tt_a}', ha='center', va='center',
                          fontsize=8, transform=ax_a.transAxes)

            if pf_b is not None and len(order_b) > 0:
                _draw_heatmap_panel(
                    ax_b, pf_b.binF, order_b, track_b,
                    normalize_rows, vmin, vmax, row_label=tt_b,
                )
            else:
                ax_b.set_axis_off()
                ax_b.text(0.5, 0.5, f'no {tt_b}', ha='center', va='center',
                          fontsize=8, transform=ax_b.transAxes)

            if pf_a is not None and pf_b is not None and len(order_c) > 0:
                combined = np.concatenate([pf_a.binF, pf_b.binF], axis=1)
                _draw_heatmap_panel(
                    ax_c, combined, order_c, track_comb,
                    normalize_rows, vmin, vmax,
                    boundaries_cm=[track_a],
                    row_label=f'{tt_a}|{tt_b}',
                )
            else:
                ax_c.set_axis_off()
                ax_c.text(0.5, 0.5, 'no combined', ha='center', va='center',
                          fontsize=8, transform=ax_c.transAxes)

            ax_a.set_title(date, fontsize=10, fontweight='bold', pad=4)
            ax_a.set_xticklabels([])
            ax_b.set_xticklabels([])
            ax_c.set_xlabel('Position (cm)', fontsize=8)

    suptitle_kind = f'{tt_a}|{tt_b} only' if combined_only else 'place fields'
    fig.suptitle(
        f'Multiday {suptitle_kind} ({cell_selection}, sort={sort_by}, n_days={n_days})',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    return fig


# MULTIDAY DAY-BY-DAY MATRIX FOR ONE TRIAL TYPE

def _resolve_dates(mouse_dir: Path, dates, date_range) -> list[str]:
    """Resolve dates from explicit list or inclusive range."""
    if dates is not None and date_range is not None:
        raise ValueError("Provide either dates or date_range, not both.")
    if date_range is not None:
        start, end = date_range
        dates = sorted(
            d.name[:10] for d in mouse_dir.iterdir()
            if d.is_dir() and len(d.name) >= 10 and start <= d.name[:10] <= end
        )
        dates = sorted(set(dates))
    if not dates:
        raise ValueError("No session dates resolved.")
    return list(dates)


def _select_multiday_cells(
    per_day: dict[str, PlaceFieldResult],
    trial_type: str,
    cell_selection: str,
    sort_keys: dict[str, np.ndarray],
    sort_reference_date: str,
    min_days: int,
) -> np.ndarray:
    """Pick cells based on selection strategy and per-day sort keys.

    sort_keys[date][cell] is the field centroid (bins) on that day, or +inf if
    the cell has no field in trial_type that day.
    """
    n_cells = next(iter(per_day.values())).n_cells
    has_field = {d: np.isfinite(sort_keys[d]) for d in per_day}

    if cell_selection == 'drop_unsorted':
        return np.where(has_field[sort_reference_date])[0]

    # Only count days where the trial type actually ran
    valid_days = [d for d, r in per_day.items() if trial_type in r.fields]
    if not valid_days:
        return np.array([], dtype=int)
    stacked = np.stack([has_field[d] for d in valid_days], axis=1)  # (n_cells, n_valid_days)

    if cell_selection == 'union':
        return np.where(stacked.sum(axis=1) >= min_days)[0]
    if cell_selection == 'intersection':
        return np.where(stacked.all(axis=1))[0]
    raise ValueError(f"cell_selection must be 'union'|'intersection'|'drop_unsorted', got {cell_selection!r}")


def _ordered_cells(cells: np.ndarray, sort_key: np.ndarray) -> np.ndarray:
    """Return cells sorted by sort_key (inf last)."""
    return cells[np.argsort(sort_key[cells], kind='stable')]


def plot_multiday_trial_type_matrix(
    mouse_dir: Path,
    dates: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    trial_type: str = 'ABDC',
    layout: str = 'matrix',
    cell_selection: str = 'union',
    sort_reference_date: str | None = None,
    min_days: int = 2,
    signal_col: str = 'multi_day_dff',
    params=None,
    force_recompute: bool = False,
    normalize_rows: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple | None = None,
    dpi: int = 130,
    show: bool = True,
) -> plt.Figure:
    """Day-by-day heatmaps for a single trial type, in three possible layouts.

    All layouts share the same cell-selection logic and per-day sort keys
    (field centroid on each day, +inf for cells with no field that day).

    Args:
        mouse_dir: Mouse root directory.
        dates: Explicit session dates. Mutually exclusive with date_range.
        date_range: (start, end) inclusive range to auto-discover sessions.
        trial_type: Which trial type's tuning curves to plot.
        layout:
            - 'concat': single axis, N day panels concatenated horizontally,
              sorted once by ``sort_reference_date``.
            - 'matrix': N x N grid. Row r sorted by day r; column c plots day c
              data. Diagonal is the within-day sort. Off-diagonal shows how that
              sort generalizes.
            - 'stack': N concat rows, each sorted by a different reference day.
              ``sort_reference_date`` is ignored.
        cell_selection:
            - 'union': cell is a PC in this trial type on >= ``min_days`` of the
              selected days.
            - 'intersection': cell is a PC in this trial type on every selected
              day where the trial type ran.
            - 'drop_unsorted': cell has a detected field on ``sort_reference_date``.
              For 'stack', each row applies drop_unsorted to its own reference day.
        sort_reference_date: Day used to sort cells for 'concat'. For 'matrix' and
            'stack' the per-day sorts are used instead, but for cell selection
            (drop_unsorted) and the title, this is the reference. Defaults to
            first selected day where ``trial_type`` ran.
        min_days: For 'union' selection.
        signal_col: Neural signal column.
        params: DetectionParams. Defaults if None.
        force_recompute: Ignore cached per-session detection.
        normalize_rows: Row-normalize each cell's tuning curve to [0, 1].
        vmin, vmax: Color scale. Defaults: 50th/90th percentile (or 0/1 normalized).
        figsize: Auto-scaled if None.
        dpi: Figure resolution.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    from experiment_place_cells import detect_place_fields_for_session
    from df_processing import find_session_dir, load_session_context
    from place_field_detection import DetectionParams

    if layout not in {'concat', 'matrix', 'stack'}:
        raise ValueError(f"layout must be 'concat'|'matrix'|'stack', got {layout!r}")

    if params is None:
        params = DetectionParams()
    mouse_dir = Path(mouse_dir)
    dates = _resolve_dates(mouse_dir, dates, date_range)

    # Per-session detection (cached)
    per_day: dict[str, PlaceFieldResult] = {}
    config_ref = None
    for date in dates:
        session_dir = find_session_dir(mouse_dir, date)
        result = detect_place_fields_for_session(
            session_dir, signal_col=signal_col, params=params,
            force_recompute=force_recompute,
        )
        per_day[date] = result
        if config_ref is None:
            _, config_ref = load_session_context(session_dir)

    # Drop days where this trial type didn't run
    dropped = [d for d in dates if trial_type not in per_day[d].fields]
    if dropped:
        print(f"Skipping {len(dropped)} day(s) without {trial_type}: {dropped}")
    dates = [d for d in dates if trial_type in per_day[d].fields]
    per_day = {d: per_day[d] for d in dates}
    if not dates:
        raise ValueError(f"No selected day has trial type {trial_type!r}.")

    # Sort keys per day for this trial type
    n_cells = per_day[dates[0]].n_cells
    sort_keys: dict[str, np.ndarray] = {}
    for d in dates:
        pf = per_day[d].fields.get(trial_type)
        if pf is None:
            sort_keys[d] = np.full(n_cells, np.inf)
        else:
            sort_keys[d] = _panel_sort_key(pf, np.arange(n_cells))

    # Default sort_reference_date
    if sort_reference_date is None:
        for d in dates:
            if trial_type in per_day[d].fields:
                sort_reference_date = d
                break
    if sort_reference_date is None:
        raise ValueError(f"No selected day has trial type {trial_type!r}.")

    track_cm = get_track_length(config_ref, trial_type)
    if track_cm is None:
        for r in per_day.values():
            pf = r.fields.get(trial_type)
            if pf is not None:
                track_cm = float(pf.binF.shape[1] * pf.bin_size_cm)
                break
    if track_cm is None:
        raise ValueError(f"Trial type {trial_type!r} not found in config or any selected day.")
    n_days = len(dates)

    def _cells_for(ref_date: str) -> np.ndarray:
        return _select_multiday_cells(
            per_day, trial_type, cell_selection, sort_keys, ref_date, min_days,
        )

    # LAYOUT DISPATCH

    if layout == 'concat':
        cells = _cells_for(sort_reference_date)
        order = _ordered_cells(cells, sort_keys[sort_reference_date])
        if figsize is None:
            figsize = (max(8, 2.2 * n_days), 6)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        panels = []
        for d in dates:
            pf = per_day[d].fields.get(trial_type)
            panels.append(pf.binF if pf is not None else np.full((n_cells, int(track_cm / 5)), np.nan))
        combined = np.concatenate(panels, axis=1)
        total_cm = track_cm * n_days

        _draw_heatmap_panel(
            ax, combined, order, total_cm,
            normalize_rows, vmin, vmax,
            boundaries_cm=[track_cm * (i + 1) for i in range(n_days - 1)],
        )
        ax.set_xlabel('Position (cm), concatenated across days')
        ax.set_ylabel('Cell #')
        # Day labels along the top
        for i, d in enumerate(dates):
            ax.text((i + 0.5) * track_cm, -0.02 * len(order), d,
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    transform=ax.transData)
        fig.suptitle(
            f'{trial_type} across days · sort={sort_reference_date} · '
            f'{cell_selection} ({len(order)} cells)',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    elif layout == 'matrix':
        cells = _cells_for(sort_reference_date)
        if figsize is None:
            figsize = (1.8 * n_days, 1.8 * n_days)
        fig, axes = plt.subplots(n_days, n_days, figsize=figsize, dpi=dpi,
                                 sharex=True, sharey=False)
        if n_days == 1:
            axes = np.array([[axes]])

        for r, ref_date in enumerate(dates):
            order = _ordered_cells(cells, sort_keys[ref_date])
            for c, data_date in enumerate(dates):
                ax = axes[r, c]
                pf = per_day[data_date].fields.get(trial_type)
                if pf is None or len(order) == 0:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                            fontsize=7, transform=ax.transAxes)
                    continue
                _draw_heatmap_panel(
                    ax, pf.binF, order, track_cm,
                    normalize_rows, vmin, vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(data_date, fontsize=8, pad=2)
                if c == 0:
                    ax.set_ylabel(f'sort:\n{ref_date}', fontsize=7, rotation=0,
                                  labelpad=24, ha='right', va='center')

        fig.suptitle(
            f'{trial_type} day x day · {cell_selection} ({len(cells)} cells)',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    else:  # 'stack'
        if figsize is None:
            figsize = (max(8, 2.2 * n_days), 2.2 * n_days)
        fig, axes = plt.subplots(n_days, 1, figsize=figsize, dpi=dpi, squeeze=False)
        for r, ref_date in enumerate(dates):
            cells_r = _cells_for(ref_date)
            order = _ordered_cells(cells_r, sort_keys[ref_date])
            panels = []
            for d in dates:
                pf = per_day[d].fields.get(trial_type)
                panels.append(pf.binF if pf is not None else np.full((n_cells, int(track_cm / 5)), np.nan))
            combined = np.concatenate(panels, axis=1)
            total_cm = track_cm * n_days

            ax = axes[r, 0]
            if len(order) == 0:
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'no cells', ha='center', va='center',
                        fontsize=8, transform=ax.transAxes)
                continue
            _draw_heatmap_panel(
                ax, combined, order, total_cm,
                normalize_rows, vmin, vmax,
                boundaries_cm=[track_cm * (i + 1) for i in range(n_days - 1)],
                row_label=f'sort: {ref_date}',
            )
            ax.set_xticks([])
            if r == 0:
                for i, d in enumerate(dates):
                    ax.text((i + 0.5) * track_cm, -0.05 * len(order), d,
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            transform=ax.transData)
            if r == n_days - 1:
                ax.set_xticks([(i + 0.5) * track_cm for i in range(n_days)])
                ax.set_xticklabels(dates, rotation=45, fontsize=7, ha='right')

        fig.suptitle(
            f'{trial_type} across days · stacked sorts · {cell_selection}',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

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
    date = "2025-09-09"

    # Toggle to rebuild caches from scratch.
    force_recompute = False

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
    print(f"Cross-day PCs (≥2 days, any trial type): {len(union_cells)}")

    # Single-day filter: PC in EITHER trial type on this date (not cross-day)
    single_day_any = np.where(result.is_place_cell_any)[0]
    print(f"Single-day PCs ({date}, any trial type): {len(single_day_any)}")

    # Per-trial-type heatmap restricted to single-day PCs (either ABC or ABDC)
    for tt in result.fields:
        plot_place_fields(
            result.fields[tt], config=exp_config, trial_type=tt,
            cells=single_day_any, animal_id=mouse_id, date=date,
        )
        plt.show()

    # Combined heatmap, cross-day PCs only, sorted each way
    for sort_by in ('ABC', 'ABDC'):
        if sort_by in result.fields:
            plot_combined_heatmap(
                result, exp_config, session_data,
                cells=union_cells, sort_by=sort_by,
            )

    # Multiday selections — set the date window you want to plot across days
    multiday_dates = [
        '2025-08-20', '2025-08-25',
        '2025-09-02', '2025-09-03',
        '2025-09-08', '2025-09-09',
        '2025-09-11',
        '2025-09-15', '2025-09-16',
    ]

    # Grid of per-day heatmaps: 3 panels per day (ABC, ABDC, ABC|ABDC)
    plot_multiday_heatmap_grid(
        mouse_dir=mouse_dir,
        dates=multiday_dates,
        signal_col=signal_col,
        params=params,
        cell_selection='per_day',     # 'per_day' | 'union' | 'intersection'
        sort_by='ABC',
        trial_types=('ABC', 'ABDC'),
        min_days=2,
        max_days_per_row=3,
        force_recompute=force_recompute,
    )

    # Single trial type across days — three layouts
    for layout in ('concat', 'matrix', 'stack'):
        plot_multiday_trial_type_matrix(
            mouse_dir=mouse_dir,
            dates=multiday_dates,
            trial_type='ABC',
            layout=layout,                # 'concat' | 'matrix' | 'stack'
            cell_selection='union',       # 'union' | 'intersection' | 'drop_unsorted'
            min_days=2,
            signal_col=signal_col,
            params=params,
            force_recompute=force_recompute,
        )

    plot_multiday_heatmap_grid(
        mouse_dir=mouse_dir,
        dates=multiday_dates,
        cell_selection='per_day',  # or 'union' / 'intersection'
        sort_by='ABC',  # or 'ABC' — flips the ordering
        trial_types=('ABC', 'ABDC'),
        combined_only=True,
        signal_col=signal_col,
        params=params,
        force_recompute=force_recompute,
    )