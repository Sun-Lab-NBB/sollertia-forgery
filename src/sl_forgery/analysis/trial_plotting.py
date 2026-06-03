"""
Trial Plotting Module -- plots ind and avg place fields and compares trial types

Place field and behavioral visualizations for trial-indexed calcium imaging data.
Works with frame-level DataFrames from df_processing.py.
"""

from pathlib import Path
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import polars as pl

from df_processing import (
    get_cue_regions,
    get_bin_size,
    get_track_length,
    compute_binned_average,
    compute_session_averages,
)

import plot_utils as pfmt  #pfmt == "plot formatting", better than putil (or pu) but we could change it


#Helper function for plotting

def _resolve_cell_idx(cell_idx: int, metadata: dict | None) -> int | None:
    """Translate an original (identity) cell index into its local position via
    metadata's 'cell_index_map'.

    Background: load_processed_session() supports a `cell_indices` arg that loads
    only a subset of cells to bound memory. After subsetting, cells live at new
    local positions 0..N-1 within the loaded arrays, but we want the user-facing
    API to keep using original indices (which are cell IDENTITY — what the cell
    is in the imaging dataset). The map handles the translation.

    Returns:
        - cell_idx unchanged if no map (no subsetting was done — original==local).
        - The local index if cell_idx is in the map.
        - None if a map exists but cell_idx is not in it (cell wasn't loaded for
          this session). Callers should handle None by skipping or showing a
          'not loaded' placeholder.
    """
    if metadata is None:
        return cell_idx
    cmap = metadata.get('cell_index_map')
    if cmap is None:
        return cell_idx
    return cmap.get(cell_idx)


def shared_params(
        ax: Axes,
        trial_type: str,
        config: dict = None,
        show_cues: bool = True,
        show_reward_zone: bool = True,
        show_labels: bool = True,
        alpha: float = 0.15,
        label_y: float = 0.98,
        font_scale: float = 1.0,
):
    """Add cue regions, reward zone, and x-ticks to a track axis."""

    if config is None:
        return

    ts = config.get('trial_structures', {}).get(trial_type, {})

    # Cue shading
    if show_cues:
        cue_regions = get_cue_regions(config, trial_type)
        pfmt.add_cue_shading_with_labels(ax, config, trial_type,
                                         alpha=alpha, label_y=label_y,
                                         font_scale=font_scale)

    # Reward zone
    if show_reward_zone and 'reward_zone_start_cm' in ts:
            rz_start = ts['reward_zone_start_cm']
            rz_end = ts['reward_zone_end_cm']
            ax.axvline(rz_start, color='#1B9AAA', linestyle='--',
                       linewidth=1.5 * font_scale, alpha=0.7, zorder=5)
            ax.axvline(rz_end, color='#1B9AAA', linestyle='--',
                       linewidth=1.5 * font_scale, alpha=0.7, zorder=5)
            ax.text((rz_start + rz_end) / 2, label_y - 0.08, 'reward',
                    ha='center', va='top', fontsize=6 * font_scale, fontstyle='italic',
                    color='#1B9AAA',
                    transform=ax.get_xaxis_transform())

    # X-ticks at cue boundaries
    track_length = get_track_length(config, trial_type)
    cue_sequence = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    ticks = [0.0]
    for cue_id in cue_sequence:
        ticks.append(ticks[-1] + cue_widths.get(cue_id, 30.0))
    if ticks[-1] != track_length:
        ticks.append(track_length)
    ticks = sorted(set(int(t) for t in ticks))

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9 * font_scale)

    # Common styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, axis='y', zorder=0)


# PLACE FIELD PLOTS

def _mark_place_fields(
    ax: Axes,
    pf,
    metadata,
    cell_idx: int,
    avg_trace: np.ndarray,
    color: str = 'k',
    offset_points: float = 14.0,
):
    """Mark detected place field peaks for one cell with an asterisk above each peak.

    Args:
        ax: matplotlib axis.
        pf: PlaceFields1d object.
        metadata: from processed df, has bin size.
        cell_idx: cell index.
        avg_trace: 1D session-averaged trace for this cell, length n_bins.
        color: asterisk color.
        offset_points: vertical offset of asterisk above peak, in display points.

    """
    labels = pf.label_im[cell_idx]  # int array, length n_bins; 0 = no field
    if not (labels > 0).any():
        return

    bin_size_cm = get_bin_size(metadata)
    field_ids = np.unique(labels[labels > 0])

    for fid in field_ids:
        bins_in_field = np.where(labels == fid)[0]
        # Peak within the field, evaluated on the (smoothed) average trace
        peak_local = bins_in_field[np.argmax(avg_trace[bins_in_field])]
        peak_x = peak_local * bin_size_cm + (bin_size_cm / 2)
        peak_y = avg_trace[peak_local]
        ax.annotate(
            '*', xy=(peak_x, peak_y),
            xytext=(0, offset_points), textcoords='offset points',
            ha='center', va='bottom', color=color,
            fontsize=14, fontweight='bold', zorder=5,
        )


def _get_trial_traces(
    df: pl.DataFrame,
    metadata: dict,
    signal_col: str,
    cell_idx: int,
    trial_type: str,
) -> list[tuple[np.ndarray, np.ndarray]]:

    """
    Extract per-trial spatial tuning traces for one cell from frame-level data.
    Calls compute_binned_average() on the frame-level DataFrame, grouping by [trial, distance_bin] to get per-trial binned traces.

    This is good for plotting single cells; if for some reason we wanted to plot 1/2 or all of the cells, this would
    take mins to run.  Instead call compute_session_averages() once and pass the result to the plotting function

    """
    type_df = df.filter(pl.col('trial_type') == trial_type)
    if len(type_df) == 0:
        return []
    binned = compute_binned_average(
        type_df, signal_col=signal_col, cell_idx=cell_idx,
        group_cols=['trial', 'distance_bin'],
    )
    bin_size_cm = get_bin_size(metadata, df)

    traces = []
    for trial_num in binned['trial'].unique().sort().to_list():
        trial_data = binned.filter(pl.col('trial') == trial_num).sort('distance_bin')
        bins = trial_data['distance_bin'].to_numpy()
        signal = trial_data['mean_signal'].to_numpy()
        x = bins * bin_size_cm + (bin_size_cm / 2)
        traces.append((x, signal))

    return traces


def _plot_tuning_on_axis(
    ax: Axes,
    session_stats: dict,
    cell_idx: int,
    trial_type: str,
    config: dict,
    metadata: dict,
    signal_col: str,
    data: pl.DataFrame | None = None,
    smooth_sigma: float = 0.0,
    show_trials: bool = True,
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    font_scale: float = 1.0,
    place_fields: dict | None = None,
    pf_cell_idx: int | None = None,
):
    """Plot avg ± SEM (and optional trial traces) for one cell/trial type on an axis.
    This function removes redundant code blocks


    Args:
        ax: matplotlib axis to plot on.
        session_stats: from compute_session_averages().
        cell_idx: cell index used to slice into session_stats and data. When the
            data has been subsetted via load_processed_session(cell_indices=...),
            this is the LOCAL index. Otherwise it's the same as the original.
        trial_type: trial type string.
        config: experiment config.
        metadata: experiment metadata (needed for bin size)
        signal_col: neural signal column name.
        data: frame-level df, needed only if show_trials=True.
        smooth_sigma: Gaussian smoothing sigma (0 to disable).
        show_trials: plot individual trial traces. If False, plots the average +/- SEM
        alpha_trials: transparency for trial traces.
        show_cues: show cue region shading.
        font_scale: font size scaling to accommodate multiple subplots
        place_fields: dict mapping trial_type -> PlaceFields1d. If provided and
            trial_type is present, shades detected field regions for the cell.
        pf_cell_idx: index used to slice into place_fields' label_im. Place field
            detection is run on FULL data and cached, so its label_im is indexed
            by ORIGINAL cell index. When data has been subsetted, pf_cell_idx
            differs from cell_idx. Defaults to cell_idx (correct when no subsetting).

    """
    if pf_cell_idx is None:
        pf_cell_idx = cell_idx

    tt_colors, tt_colors_dark = pfmt.get_trial_type_colors(config) if config else ({}, {})
    shared_params(ax, trial_type, config, show_cues, font_scale=font_scale)

    # Individual trial traces
    if show_trials and data is not None:
        trial_color = tt_colors.get(trial_type, '#2E86AB')
        traces = _get_trial_traces(data, metadata, signal_col, cell_idx, trial_type)
        for x, signal in traces:
            ax.plot(x, signal, color=trial_color, linewidth=1,
                    alpha=alpha_trials, zorder=2)
    bin_size_cm = get_bin_size(metadata, data)

    # Session average ± SEM, with smoothing
    avg = None
    if trial_type in session_stats:
        avg = session_stats[trial_type]['session_avg'][:, cell_idx]
        sem = session_stats[trial_type]['session_sem'][:, cell_idx]
        if smooth_sigma > 0:
            avg = gaussian_filter1d(avg, sigma=smooth_sigma)
            sem = gaussian_filter1d(sem, sigma=smooth_sigma)
        x_avg = np.arange(len(avg)) * bin_size_cm + (bin_size_cm / 2)
        n_trials = session_stats[trial_type]['n_trials']

        avg_color = tt_colors_dark.get(trial_type, '#0A4D68')
        ax.plot(x_avg, avg, color=avg_color, linewidth=3.5,
                label=f'Avg', zorder=4)
        ax.fill_between(x_avg, avg - sem, avg + sem,
                        color=avg_color, alpha=0.3, zorder=3)

    # Mark detected place field peaks with an asterisk (uses pf_cell_idx — original index)
    if place_fields is not None and trial_type in place_fields and avg is not None:
        _mark_place_fields(ax, place_fields[trial_type], metadata, pf_cell_idx, avg)

    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.legend(frameon=False, fontsize=8*font_scale, loc='center right')


def plot_single_cell(
    data: pl.DataFrame,
    cell_idx: int,
    metadata: dict,
    signal_col: str = 'single_day_f',
    trial_type: str = None,
    session_stats: dict = None,
    config: dict = None,
    smooth_sigma: float = 1.0,
    figsize: tuple | None = None,
    show_trials: bool = True,
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    place_fields: dict | None = None,
    show: bool = True
) -> Figure:
    """Plot tuning curves for a single cell. Stacks subplots if multiple trial types. Exploratory.

    Args:
        data: frame-level df from process_session().
        cell_idx: cell index to plot. ALWAYS in original (identity) terms — the same
            ID used in your imaging dataset. If `data` was loaded with cell subsetting
            (metadata has 'cell_index_map'), this is translated to the local position
            internally.
         metadata: session metadata dict (from load_processed_session).
        trial_type: specific trial type, or None for all types stacked.
        signal_col: neural signal column name.
        session_stats: from compute_session_averages(); computed if None.
        config: experiment config.
        smooth_sigma: Gaussian smoothing sigma param (0 to disable).
        figsize: figure size; auto-scaled if None.
        show_trials: plot individual trial traces. If False, plots the average +/- SEM and uses a global ylimit <--
            this could be a separate param if needed i.e. not tied to show_trials
        alpha_trials: transparency for trial traces.
        show_cues: show cue region shading.
        show: call plt.show().

    Returns:
        Matplotlib Figure, or None if cell_idx wasn't loaded for this session.

    """

    # Translate original cell_idx -> local idx for indexing into the loaded data.
    # If data was loaded without subsetting, local_idx == cell_idx.
    local_idx = _resolve_cell_idx(cell_idx, metadata)
    if local_idx is None:
        print(f"Cell {cell_idx} was not loaded for this session "
              f"(not in cell_indices). Skipping plot.")
        return None

    if trial_type is not None:
        trial_types = [trial_type]
    else:
        trial_types = sorted(data['trial_type'].unique().to_list())

    if not trial_types:
        print("No trial types found")
        return None

    if figsize is None:
        figsize = (12, 5 * len(trial_types))

    bin_size_cm = get_bin_size(metadata)

    if session_stats is None and config is not None:
        session_stats = compute_session_averages(
            data, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
        )

    # Global ylim when not showing trials (avg+SEM only)
    if not show_trials and session_stats:
        all_maxes = []
        for tt in trial_types:
            if tt in session_stats:
                avg = session_stats[tt]['session_avg'][:, local_idx]
                sem = session_stats[tt]['session_sem'][:, local_idx]
                if smooth_sigma > 0:
                    avg = gaussian_filter1d(avg, sigma=smooth_sigma)
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
                all_maxes.append(np.nanmax(avg + sem))
        if all_maxes:
            global_ymax = max(all_maxes) * 1.15
            global_ylim = (-global_ymax * 0.03, global_ymax)
        else:
            global_ylim = None
    else:
        global_ylim = None

    fig, axes = plt.subplots(len(trial_types), 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, tt in zip(axes, trial_types):
        _plot_tuning_on_axis(
            ax, session_stats, local_idx, tt, config, metadata, signal_col,
            data=data if show_trials else None,
            smooth_sigma=smooth_sigma,
            show_trials=show_trials, alpha_trials=alpha_trials,
            show_cues=show_cues, place_fields=place_fields,
            pf_cell_idx=cell_idx,
        )
        n_trials = session_stats[tt]['n_trials'] if tt in session_stats else 0
        n_cells = session_stats[tt]['session_avg'].shape[1] if tt in session_stats else 0
        ax.set_title(f'{tt} — {n_trials} trials, {n_cells} cells',
                     fontsize=12, fontweight='bold', loc='left')

        if global_ylim:
            ax.set_ylim(global_ylim)

    axes[-1].set_xlabel('Position (cm)', fontsize=12)
    fig.suptitle(f'Cell {cell_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show and fig is not None:
        plt.show()

    return fig


def plot_multiday_cell(
    sessions: dict[str, dict],
    cell_idx: int,
    signal_col: str = 'multi_day_dff',
    figsize: tuple = (14, 9),
    alpha_trials: float = 0.3,
    show_cues: bool = True,
) -> None:
    """
    Scroll through daily place field plots for one cell across sessions.
    Left/right arrow keys to navigate days. Works standalone and Jupyter.

    Pre-renders all days on init (may take a few seconds), then navigation is instant.
    Each page shows stacked subplots (one per trial type, union across all days).
    Missing trial types show "No data". Title includes animal ID and date.

    Args:
        sessions: dict[str, dict], from load_multiday_sessions().
            Each value has keys: 'data', 'config', 'session_data', 'metadata'
        cell_idx: int, cell index (consistent across days)
        signal_col: str, neural signal column name
        figsize: tuple, figure size
        alpha_trials: float, transparency for individual trial traces
        show_cues: bool, show cue region shading

    Returns:
        None (displays interactive figure)
    """
    import io
    from matplotlib.image import imread

    dates = sorted(sessions.keys())
    if not dates:
        print("No sessions to plot.")
        return

    all_trial_types = sorted(set(
        tt
        for s in sessions.values()
        for tt in s['data']['trial_type'].unique().to_list()
    ))
    n_types = len(all_trial_types)
    animal_id = sessions[dates[0]]['session_data'].get('animal_id', '??')

    # --- Pre-render all days as images ---
    print(f"Pre-rendering {len(dates)} days...", end=' ', flush=True)
    day_images = []

    for di, date in enumerate(dates):
        s = sessions[date]
        data = s['data']
        config = s['config']
        bin_size_cm = get_bin_size(s['metadata'])
        day_trial_types = sorted(data['trial_type'].unique().to_list())

        session_stats = compute_session_averages(
            data, signal_col=signal_col,
            config=config, bin_size_cm=bin_size_cm,
        )

        tmp_fig = plt.figure(figsize=figsize)
        gs = tmp_fig.add_gridspec(n_types, 1, hspace=0.4, top=0.92, bottom=0.06)
        tmp_axes = [tmp_fig.add_subplot(gs[i, 0]) for i in range(n_types)]

        for i, tt in enumerate(all_trial_types):
            ax = tmp_axes[i]

            if tt not in day_trial_types:
                ax.text(
                    0.5, 0.5, f'{tt} — no data',
                    ha='center', va='center',
                    fontsize=14, color='#999999', fontstyle='italic',
                    transform=ax.transAxes,
                )
                ax.set_ylabel('ΔF/F', fontsize=12)
                ax.set_title(f'{tt}', fontsize=12, fontweight='bold', loc='left')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                continue

            _plot_tuning_on_axis(
                ax, session_stats, cell_idx, tt, config, signal_col,
                data=data,
                show_trials=True, alpha_trials=alpha_trials,
                show_cues=show_cues,
            )

        tmp_axes[-1].set_xlabel('Position (cm)', fontsize=12)
        tmp_fig.suptitle(
            f'Mouse {animal_id} — {date} — Cell {cell_idx}'
            f'    [{di + 1}/{len(dates)}]',
            fontsize=14, fontweight='bold',
        )

        # Render to image buffer (to help with speed)
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format='png', dpi=150)
        plt.close(tmp_fig)
        buf.seek(0)
        day_images.append(imread(buf))
        buf.close()

        print(f"{di + 1}", end=' ', flush=True)

    print("done.")

    # Scroll through daily plots using the arrow keys; kind of slow
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    img_display = ax.imshow(day_images[0])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    state = {'idx': 0}

    def _on_key(event):
        '''Arrow key navigation — instant image swap.'''
        if event.key == 'right' and state['idx'] < len(dates) - 1:
            state['idx'] += 1
        elif event.key == 'left' and state['idx'] > 0:
            state['idx'] -= 1
        else:
            return
        img_display.set_data(day_images[state['idx']])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', _on_key)
    fig._multiday_state = state
    fig._multiday_key_handler = _on_key
    fig._multiday_images = day_images  # prevent GC

    print(f"Use ← → arrow keys to navigate {len(dates)} days.")
    plt.show()



def plot_multiday_comparison(
    sessions: dict[str, dict],
    cell_idx: int,
    signal_col: str = 'multi_day_f',
    smooth_sigma: float = 1.0,
    global_ylim: bool = False,
    show_cues: bool = True,
    place_fields: dict | None = None,
    show: bool = True,
) -> Figure:
    """
    Grid of place field plots: columns = days, rows = trial types.
    Shows session average ± SEM only (no individual trials).
    Y-axes shared within each row, scaled from average + SEM across all days.

    Args:
        sessions: dict[str, dict], from load_multiday_sessions().
        cell_idx: int, ORIGINAL cell index (cell identity — same as in your
            imaging dataset, consistent across days). If sessions were loaded
            with cell_indices subsetting, this is translated per-session via
            each session's cell_index_map. Sessions where the cell wasn't
            loaded show 'cell not loaded' subplots.
        signal_col: str, neural signal column name.
        smooth_sigma: float, Gaussian smoothing on binned avg w/ small kernel (Dombeck et al 2010).
        global_ylim: bool, have all plots share the same y-axis limits; If false, each trial type has their own ylims
        show_cues: bool, show cue region shading.
        show: bool, call plt.show().

    Returns:
        fig: Figure.

    """
    dates = sorted(sessions.keys())
    if not dates:
        print("No sessions to plot.")
        return None

    all_trial_types = sorted(set(
        tt
        for s in sessions.values()
        for tt in s['data']['trial_type'].unique().to_list()
    ))
    n_types = len(all_trial_types)
    n_days = len(dates)

    animal_id = sessions[dates[0]]['session_data'].get('animal_id', '??')

    # Per-session cell_idx translation (original -> local). None if not loaded
    # for that session — that subplot will show a 'cell not loaded' message.
    local_idx_per_date = {
        date: _resolve_cell_idx(cell_idx, s.get('metadata'))
        for date, s in sessions.items()
    }
    if all(v is None for v in local_idx_per_date.values()):
        print(f"Cell {cell_idx} was not loaded for any session.")
        return None

    # Pre-compute session averages
    precomputed = {}
    for date, s in sessions.items():
        bin_size_cm = get_bin_size(s['metadata'])

        precomputed[date] = compute_session_averages(
            s['data'], signal_col=signal_col,
            config=s['config'], bin_size_cm=bin_size_cm,
        )

    # Ylims from avg + SEM across days where the cell is loaded
    ylims = {}
    for tt in all_trial_types:
        all_maxes = []
        for date in dates:
            local_idx = local_idx_per_date[date]
            if local_idx is None:
                continue
            if tt in precomputed[date]:
                avg = precomputed[date][tt]['session_avg'][:, local_idx]
                sem = precomputed[date][tt]['session_sem'][:, local_idx]
                if smooth_sigma > 0:
                    avg = gaussian_filter1d(avg, sigma=smooth_sigma)
                    sem = gaussian_filter1d(sem, sigma=smooth_sigma)
                all_maxes.append(np.nanmax(avg + sem))

        ymax = max(all_maxes) * 1.3 if all_maxes else 1.0
        ylims[tt] = (-ymax * 0.03, ymax)

    # Global ylim: same scale for all trial types
    if global_ylim:
        global_ymax = max(yl[1] for yl in ylims.values())
        ylims = {tt: (-global_ymax * 0.03, global_ymax) for tt in all_trial_types}

    if signal_col.endswith('dff'):
        y_label = 'ΔF/F'
    else:
        y_label = 'Deconvolved Signal'

    # Figure sizing
    col_width = max(3.5, 14 / n_days)
    row_height = 2.5
    fig = plt.figure(figsize=(col_width * n_days, row_height * n_types + 1.5))
    gs = fig.add_gridspec(
        n_types, n_days,
        hspace=0.2, wspace=0.1,
        top=0.88, bottom=0.06, left=0.03, right=0.98,
    )

    axes = np.empty((n_types, n_days), dtype=object)
    for row in range(n_types):
        for col in range(n_days):
            axes[row, col] = fig.add_subplot(
                gs[row, col],
                sharey=axes[row, 0] if col > 0 else None,
            )

    fscale = min(1.0, 3.5 / col_width) if n_days > 3 else 1.0

    # Date headers
    for col, date in enumerate(dates):
        short_date = date[5:]
        bbox = axes[0, col].get_position()
        fig.text(
            (bbox.x0 + bbox.x1) / 2, 0.91,
            short_date, ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )

    for col, date in enumerate(dates):
        s = sessions[date]
        data = s['data']
        config = s['config']
        session_stats = precomputed[date]
        day_trial_types = sorted(data['trial_type'].unique().to_list())
        day_fields = place_fields.get(date) if place_fields else None
        local_idx = local_idx_per_date[date]

        for row, tt in enumerate(all_trial_types):
            ax = axes[row, col]

            if tt in session_stats and tt in day_trial_types:
                n_trials = session_stats[tt]['n_trials']
                n_cells = session_stats[tt]['session_avg'].shape[1]
                ax.set_title(
                    f'{tt} — {n_trials} trials, {n_cells} cells',
                    fontsize=8 * fscale, fontweight='bold', loc='left',
                )
            else:
                ax.set_title(f'{tt}', fontsize=8 * fscale,
                             fontweight='bold', loc='left')

            # Cell wasn't loaded for this session — show placeholder.
            if local_idx is None:
                ax.text(
                    0.5, 0.5, 'cell not loaded',
                    ha='center', va='center',
                    fontsize=10 * fscale, color='#999999', fontstyle='italic',
                    transform=ax.transAxes,
                )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(ylims[tt])
                if col == 0:
                    ax.set_ylabel(y_label, fontsize=9 * fscale)
                ax.tick_params(labelsize=7 * fscale)
                continue

            if tt not in day_trial_types:
                ax.text(
                    0.5, 0.5, 'no data',
                    ha='center', va='center',
                    fontsize=10 * fscale, color='#999999', fontstyle='italic',
                    transform=ax.transAxes,
                )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(ylims[tt])
                if col == 0:
                    ax.set_ylabel(y_label, fontsize=9 * fscale)
                ax.tick_params(labelsize=7 * fscale)
                continue

            day_fields = place_fields.get(date) if place_fields else None

            _plot_tuning_on_axis(
                ax, session_stats, local_idx, tt, config, s['metadata'],
                signal_col,
                smooth_sigma=smooth_sigma,
                show_trials=False, show_cues=show_cues,
                font_scale=fscale, place_fields=day_fields,
                pf_cell_idx=cell_idx,
            )

            ax.set_ylim(ylims[tt])

            if col == 0:
                ax.set_ylabel('ΔF/F', fontsize=9 * fscale)
            else:
                ax.set_ylabel('')

            ax.tick_params(labelsize=7 * fscale)

        axes[-1, col].set_xlabel('cm', fontsize=8 * fscale)

    fig.suptitle(
        f'Mouse {animal_id} — Cell {cell_idx}',
        fontsize=14, fontweight='bold',
    )

    if show:
        plt.show()

    return fig



# SAVE

def save_figure(fig: Figure, path: Path, dpi: int = 150):
    """Save figure to file (supports .png, .pdf, .svg)."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")



if __name__ == "__main__":
    from place_field_detection import detect_place_fields, get_place_cell_indices, DetectionParams
    from experiment_place_cells import detect_multiday_place_fields, detect_multiday_place_fields_cached
    from df_processing import (
        find_session_dir, load_session_context, get_session_paths,
        load_processed_session, load_multiday_sessions
    )
    from place_field_plotting import plot_combined_heatmap

    mouse_id = '26'
    date = '2025-09-16'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    # Single-day ABC | ABDC heatmap, cells sorted by ABC field position.
    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])

    result = detect_place_fields(data, exp_config, signal_col='multi_day_dff')

    plot_combined_heatmap(
        result, exp_config, session_data,
        trial_types=['ABC', 'ABDC'],
        sort_by='ABDC',
    )

    # # #plot multiple sessions for a single cell;  Date range — auto-discovers all sessions between these dates
    # sessions = load_multiday_sessions(mouse_dir, date_range=('2025-08-01', '2025-09-20'), auto_process=False)

    # sessions = load_multiday_sessions(mouse_dir, dates=['2025-08-20', '2025-08-25',  # pre ext
    #                                                     '2025-09-02', '2025-09-03'  # pre ext
    #                                                     '2025-09-08', '2025-09-09',  # add ext
    #                                                     '2025-09-11',
    #                                                     '2025-09-15', '2025-09-16'],  # last 2 days
    #                                   auto_process=False)
    # #
    # # #filter for place cells
    # multiday = detect_multiday_place_fields(sessions, signal_col='multi_day_dff')
    # #
    # #
    # # # Build the place_fields dict from per-day results
    # pf_by_date = {date: r.fields for date, r in multiday.per_day.items()}
    # #
    # #
    # plot_multiday_comparison(sessions, cell_idx=5, signal_col='multi_day_dff', global_ylim=True)
    # #
    # #
    # for i in multiday.union_indices[:5]:
    #     plot_multiday_comparison(
    #         sessions, cell_idx=i+5, signal_col='multi_day_spikes',
    #         global_ylim=True, place_fields=None,
    #     )

#______________________________

    # plot single day activity for a single cell
    # Load processed df (run epoch only) + raw feather (all epochs)
    # session_dir = find_session_dir(mouse_dir, date)
    # session_data, exp_config = load_session_context(session_dir)
    # paths = get_session_paths(session_dir, session_data)
    # data, meta = load_processed_session(paths['parquet'])


    #filter place cells
    # 1. Run detection (once)
    # result = detect_place_fields(data, exp_config, signal_col='multi_day_dff')

    # 2. Get place cell indices
    # pc_indices = get_place_cell_indices(result)  # any trial type
    # or: trial_type='ABC'
    # or: require_all=True --> gives both types

    # for i in pc_indices:
    #     plot_single_cell(data, cell_idx=i, metadata=meta, signal_col='multi_day_dff',
    #                      show_trials=False, config=exp_config)

