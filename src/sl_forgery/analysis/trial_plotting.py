"""
Trial Plotting Module

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
    get_track_length,
    compute_binned_average,
    compute_session_averages,
)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Poppins', 'Liberation Sans', 'DejaVu Sans']
#'Helvetica', 'Arial', 'Poppins', 'Liberation Sans', 'DejaVu Sans'


# COLOR CONFIGURATION

# Tableau 10 - colorblind friendly, fixed by cue ID across all experiments
CUE_COLOR_PALETTE = [
    '#4E79A7',  # Muted blue
    '#F28E2B',  # Warm orange
    '#59A14F',  # Forest green
    '#E15759',  # Soft red
    '#B07AA1',  # Dusty purple
    '#9C755F',  # Warm brown
    '#EDC948',  # Golden yellow
    '#76B7B2',  # Dusty teal
    '#FF9DA7',  # Soft pink
    '#BAB0AC',  # Warm gray
]

SPECIAL_CUE_COLORS = {
    0: '#D3D3D3',   # Light gray (gray zones)
    255: '#2D2D2D', # Charcoal (dark periods)
}

SPECIAL_CUE_LABELS = {
    0: 'Gray',
    255: 'Dark',
}

# Trial type palettes: assigned by sorted order of trial_structures keys. Accommodates up to 5 trial types
# Light = individual traces, Dark = session average
TRIAL_TYPE_PALETTE = [
    '#2E86AB',  # Blue
    '#A23B72',  # Red/magenta
    '#59A14F',  # Green
    '#E15759',  # Coral
    '#B07AA1',  # Purple
]

TRIAL_TYPE_PALETTE_DARK = [
    '#0A4D68',  #Dark blue, etc
    '#6B0848',
    '#2D6A2E',
    '#9E2B2D',
    '#7A4E7A',
]

def get_trial_type_colors(config: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Auto-assign trial type colors from config trial_structures keys."""
    trial_types = sorted(config.get('trial_structures', {}).keys())
    colors = {}
    colors_dark = {}
    for i, tt in enumerate(trial_types):
        colors[tt] = TRIAL_TYPE_PALETTE[i % len(TRIAL_TYPE_PALETTE)]
        colors_dark[tt] = TRIAL_TYPE_PALETTE_DARK[i % len(TRIAL_TYPE_PALETTE_DARK)]
    return colors, colors_dark

TRIAL_TYPE_COLORS = {
    'ABC': TRIAL_TYPE_PALETTE[0],
    'ABDC': TRIAL_TYPE_PALETTE[1],
}


def get_cue_colors(config: dict = None, max_cue_id: int = 20) -> dict[int, str]:
    """Get cue ID to color mapping."""
    colors = SPECIAL_CUE_COLORS.copy()
    
    for cue_id in range(1, max_cue_id + 1):
        idx = (cue_id - 1) % len(CUE_COLOR_PALETTE)
        colors[cue_id] = CUE_COLOR_PALETTE[idx]
    
    # Allow config override
    if config and 'cue_colors' in config:
        colors.update(config['cue_colors'])
    
    return colors


def get_cue_labels(config: dict = None, max_cue_id: int = 20) -> dict[int, str]:
    """Get cue ID to label mapping (A, B, C, ...)."""
    labels = SPECIAL_CUE_LABELS.copy()
    
    for cue_id in range(1, max_cue_id + 1):
        if cue_id <= 26:
            labels[cue_id] = chr(ord('A') + cue_id - 1)
        else:
            labels[cue_id] = 'A' + chr(ord('A') + (cue_id - 27) % 26)
    
    if config and 'cue_labels' in config:
        labels.update(config['cue_labels'])
    
    return labels

#Helper function for plotting

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
        plot_cue_regions(ax, cue_regions, config, show_labels=show_labels,
                         alpha=alpha, label_y=label_y)

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


# CUE REGION PLOTTING

def plot_cue_regions(
    ax: Axes,
    cue_regions: dict[int, tuple],
    config: dict = None,
    show_labels: bool = True,
    alpha: float = 0.15,
    label_y: float = 0.98,
    font_scale: float = 1.0,
):
    """
    Add cue region shading to a matplotlib axis.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    cue_regions : dict
        {cue_id: (start_cm, end_cm)} from get_cue_regions()
    config : dict, optional
        Experiment config for custom colors/labels
    show_labels : bool
        Show cue labels at top
    alpha : float
        Shading transparency
    label_y : float
        Y position for labels (in axis transform coords)
    fontscale
    """
    colors = get_cue_colors(config)
    labels = get_cue_labels(config)
    
    for cue_id, region in cue_regions.items():
        color = colors.get(cue_id, '#CCCCCC')
        
        # Handle single region or multiple
        if isinstance(region, tuple) and len(region) == 2:
            regions = [region]
        elif isinstance(region, list):
            regions = region
        else:
            continue
        
        for i, (start, end) in enumerate(regions):
            start = max(start, 0)  # Clip to track start
            if end <= 0:
                continue
            ax.axvspan(start, end, alpha=alpha, color=color, zorder=1)
            
            # Label only first occurrence
            if show_labels:
                mid = (start + end) / 2
                label = labels.get(cue_id, f'Cue {cue_id}')
                ax.text(
                    mid, label_y, label,
                    ha='center', va='top',
                    fontsize=9 * font_scale, fontweight='bold',
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=color, alpha=0.6, edgecolor='none'
                    )
                )



# PLACE FIELD PLOTS
#TODO add thresholding for place cells
def _get_trial_traces(
    df: pl.DataFrame,
    signal_col: str,
    cell_idx: int,
    trial_type: str,
    bin_size_cm: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:

    '''
    Extract per-trial spatial tuning traces for one cell from frame-level data.
    Calls compute_binned_average() on the frame-level DataFrame, grouping by [trial, distance_bin] to get per-trial binned traces.

    This is good for plotting single cells; if for some reason we wanted to plot 1/2 or all of the cells, this would
    takes mins to run.  Instead call compute_session_averages() once and pass the result to the plotting function

    Args

    '''
    type_df = df.filter(pl.col('trial_type') == trial_type)
    if len(type_df) == 0:
        return []
    binned = compute_binned_average(
        type_df, signal_col=signal_col, cell_idx=cell_idx,
        group_cols=['trial', 'distance_bin'],
    )
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
    signal_col: str,
    data: pl.DataFrame | None = None,
    bin_size_cm: int = 5,
    smooth_sigma: float = 0.0,
    show_trials: bool = True,
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    font_scale: float = 1.0,
):
    """Plot avg ± SEM (and optional trial traces) for one cell/trial type on an axis.
    This function removes redundant code blocks


    Args:
        ax: matplotlib axis to plot on.
        session_stats: from compute_session_averages().
        cell_idx: cell index.
        trial_type: trial type string.
        config: experiment config.
        signal_col: neural signal column name.
        data: frame-level df, needed only if show_trials=True.
        bin_size_cm: spatial bin size in cm.
        smooth_sigma: Gaussian smoothing sigma (0 to disable).
        show_trials: plot individual trial traces. If False, plots the average +/- SEM
        alpha_trials: transparency for trial traces.
        show_cues: show cue region shading.
        font_scale: font size scaling to accommodate multiple subplots

    """
    tt_colors, tt_colors_dark = get_trial_type_colors(config) if config else ({}, {})
    shared_params(ax, trial_type, config, show_cues, font_scale=font_scale)

    # Individual trial traces
    if show_trials and data is not None:
        trial_color = tt_colors.get(trial_type, '#2E86AB')
        traces = _get_trial_traces(data, signal_col, cell_idx, trial_type, bin_size_cm)
        for x, signal in traces:
            ax.plot(x, signal, color=trial_color, linewidth=1,
                    alpha=alpha_trials, zorder=2)

    # Session average ± SEM
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

    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.legend(frameon=False, fontsize=8*font_scale, loc='center right')


def plot_single_cell(
    data: pl.DataFrame,
    cell_idx: int,
    signal_col: str = 'single_day_f',
    trial_type: str = None,
    session_stats: dict = None,
    config: dict = None,
    bin_size_cm: int = 5,
    smooth_sigma: float = 1.0,
    figsize: tuple | None = None,
    show_trials: bool = True,
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    show: bool = True
) -> Figure:
    """Plot tuning curves for a single cell. Stacks subplots if multiple trial types. Exploratory.

    Args:
        data: frame-level df from process_session().
        cell_idx: cell index to plot.
        trial_type: specific trial type, or None for all types stacked.
        signal_col: neural signal column name.
        session_stats: from compute_session_averages(); computed if None.
        config: experiment config.
        bin_size_cm: spatial bin size in cm.
        smooth_sigma: Gaussian smoothing sigma param (0 to disable).
        figsize: figure size; auto-scaled if None.
        show_trials: plot individual trial traces. If False, plots the average +/- SEM and uses a global ylimit <--
            this could be a separate param if needed i.e. not tied to show_trials
        alpha_trials: transparency for trial traces.
        show_cues: show cue region shading.
        show: call plt.show().

    Returns:
        Matplotlib Figure.

    """

    if trial_type is not None:
        trial_types = [trial_type]
    else:
        trial_types = sorted(data['trial_type'].unique().to_list())

    if not trial_types:
        print("No trial types found")
        return None

    if figsize is None:
        figsize = (12, 5 * len(trial_types))

    if session_stats is None and config is not None:
        session_stats = compute_session_averages(
            data, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
        )

    # Global ylim when not showing trials (avg+SEM only)
    if not show_trials and session_stats:
        all_maxes = []
        for tt in trial_types:
            if tt in session_stats:
                avg = session_stats[tt]['session_avg'][:, cell_idx]
                sem = session_stats[tt]['session_sem'][:, cell_idx]
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
            ax, session_stats, cell_idx, tt, config, signal_col,
            data=data if show_trials else None,
            bin_size_cm=bin_size_cm, smooth_sigma=smooth_sigma,
            show_trials=show_trials, alpha_trials=alpha_trials,
            show_cues=show_cues,
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
    bin_size_cm: int = 5,
    figsize: tuple = (14, 9),
    alpha_trials: float = 0.3,
    show_cues: bool = True,
) -> None:
    '''
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
        bin_size_cm: int, spatial bin size in cm
        figsize: tuple, figure size
        alpha_trials: float, transparency for individual trial traces
        show_cues: bool, show cue region shading

    Returns:
        None (displays interactive figure)
    '''
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
                data=data, bin_size_cm=bin_size_cm,
                show_trials=True, alpha_trials=alpha_trials,
                show_cues=show_cues,
            )

        tmp_axes[-1].set_xlabel('Position (cm)', fontsize=12)
        tmp_fig.suptitle(
            f'Mouse {animal_id} — {date} — Cell {cell_idx}'
            f'    [{di + 1}/{len(dates)}]',
            fontsize=14, fontweight='bold',
        )

        # Render to image buffer
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format='png', dpi=150)
        plt.close(tmp_fig)
        buf.seek(0)
        day_images.append(imread(buf))
        buf.close()

        print(f"{di + 1}", end=' ', flush=True)

    print("done.")

    # --- Display with instant arrow key switching ---
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
    bin_size_cm: int = 5,
    smooth_sigma: float = 1.0,
    global_ylim: bool = False,
    show_cues: bool = True,
    show: bool = True,
) -> Figure:
    '''
    Grid of place field plots: columns = days, rows = trial types.
    Shows session average ± SEM only (no individual trials).
    Y-axes shared within each row, scaled from average + SEM across all days.

    Args:
        sessions: dict[str, dict], from load_multiday_sessions().
        cell_idx: int, cell index (consistent across days).
        signal_col: str, neural signal column name.
        bin_size_cm: int, spatial bin size in cm.
        smooth_sigma: float, Gaussian smoothing on binned avg w/ small kernel (Dombeck et al 2010).
        global_ylim: bool, have all plots share the same y-axis limits; If false, each trial type has their own ylims
        show_cues: bool, show cue region shading.
        show: bool, call plt.show().

    Returns:
        fig: Figure.

    '''
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

    # Pre-compute session averages
    precomputed = {}
    for date, s in sessions.items():
        precomputed[date] = compute_session_averages(
            s['data'], signal_col=signal_col,
            config=s['config'], bin_size_cm=bin_size_cm,
        )

    # Ylims from avg + SEM across all days
    ylims = {}
    for tt in all_trial_types:
        all_maxes = []
        for date in dates:
            if tt in precomputed[date]:
                avg = precomputed[date][tt]['session_avg'][:, cell_idx]
                sem = precomputed[date][tt]['session_sem'][:, cell_idx]
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

    # Figure sizing
    col_width = max(3.5, 14 / n_days)
    row_height = 2.5
    fig = plt.figure(figsize=(col_width * n_days, row_height * n_types + 1.5))
    gs = fig.add_gridspec(
        n_types, n_days,
        hspace=0.2, wspace=0.1,
        top=0.88, bottom=0.06, left=0.06, right=0.98,
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
                    ax.set_ylabel('ΔF/F', fontsize=9 * fscale)
                ax.tick_params(labelsize=7 * fscale)
                continue

            _plot_tuning_on_axis(
                ax, session_stats, cell_idx, tt, config, signal_col,
                bin_size_cm=bin_size_cm, smooth_sigma=smooth_sigma,
                show_trials=False, show_cues=show_cues,
                font_scale=fscale,
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
    from df_processing import (load_session_dir, get_session_prefix, load_processed_session, save_processed_session,
                               process_session, load_multiday_sessions)
    from place_field_detection import detect_place_fields, get_place_cell_indices, detect_multiday_place_fields

    # load all the data
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    date = '2025-09-15'  # again, the .feather file in this is actually from 9-16, too slow to download at my house.
    # ***DO NOT GET MISTAKEN

    #plot multiple sessions for a single cell;  Date range — auto-discovers all sessions between these dates
    sessions = load_multiday_sessions(mouse_dir, date_range=('2025-09-03', '2025-09-24'), auto_process=False)
    ###FTR I added a fake file into the 9-12 day bc again the server is slow.  It is really from 9-03

    #filter for place cells
    multiday = detect_multiday_place_fields(sessions, signal_col='multi_day_dff')
    for i in multiday.union_indices[:5]:
        plot_multiday_comparison(sessions, cell_idx=i, signal_col='multi_day_dff')

#______________________________

    # plot single day activity for a single cell
    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    parquet_path = behavior_path.parent / f'{prefix}_processed.parquet'

    if parquet_path.exists():
        print(f"Loading: {parquet_path}")
        data, metadata = load_processed_session(parquet_path)
    else:
        print("No processed file found, processing from raw...")
        behavior_df = pl.read_ipc(behavior_path)
        data, metadata = process_session(behavior_df, config)
        save_processed_session(data, behavior_path.parent, session_data, metadata)

    #filter place cells
    # 1. Run detection (once)
    result = detect_place_fields(data, config, signal_col='multi_day_dff')

    # 2. Get place cell indices
    pc_indices = get_place_cell_indices(result)  # any trial type
    # or: trial_type='ABC'
    # or: require_all=True --> gives both types

    for i in pc_indices:
        plot_single_cell(data, cell_idx=i, signal_col='multi_day_dff', show_trials=False, config=config)

