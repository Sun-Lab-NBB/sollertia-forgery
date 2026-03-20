"""
Behavioral Analysis Module

Visualizations and metrics for behavioral data (licking, speed, reward, trial duration)
across sessions and trial types. Works with processed frame-level DataFrames
from df_processing.py.

Lick rasters:
    - plot_session_lick_raster(): Auto-detects trial types, creates 1+ panels
    - plot_lick_raster(): Single trial-type raster (building block)
    - plot_lick_raster_grid(): Multi-day comparison grid
    - plot_multiday_lick_raster(): Auto-layout multiday grid

Speed analyses:
    - plot_speed_profile(): Mean ± SEM speed vs position, overlaid by trial type
    - plot_speed_raster(): Trial × position speed heatmap
    - plot_session_speed_raster(): Multi-panel speed raster per session
    - plot_multiday_speed_profile(): Speed learning curves across sessions

Reward / water:
    - compute_reward_metrics(): Per-trial hit rate, water consumed, duration
    - compute_session_reward_summary(): Session-level reward stats
    - plot_reward_metrics(): Single-session reward panels
    - plot_multiday_reward_summary(): Cross-session hit rate and water

Trial duration:
    - plot_trial_duration(): Duration across trials within a session
    - plot_multiday_trial_duration(): Mean duration across sessions
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')

from df_processing import get_track_length, get_cue_regions
import plot_utils as pfmt


# LICK RASTER

def _extract_lick_positions(
    df: pl.DataFrame,
    trial_type: str | None = None,
) -> pl.DataFrame:
    """Extract lick event positions and trial numbers from a processed session df.

    Filters to frames where lick == 1, optionally filtered by trial type.
    Renumbers trials to sequential 0-indexed for plotting.

    Args:
        df: Processed frame-level DataFrame with 'lick', 'position', 'trial',
            'trial_type' columns.
        trial_type: If provided, filter to this trial type only.

    Returns:
        DataFrame with columns ['position', 'trial_idx'] where trial_idx
        is 0-indexed sequential trial number within the filtered set.
    """
    filtered = df
    if trial_type is not None:
        filtered = filtered.filter(pl.col('trial_type') == trial_type)

    lick_frames = filtered.filter(pl.col('lick') == 1)

    if lick_frames.is_empty():
        return pl.DataFrame({'position': [], 'trial_idx': []},
                            schema={'position': pl.Float64, 'trial_idx': pl.Int32})

    # Map original trial numbers to sequential indices
    unique_trials = filtered.select('trial').unique().sort('trial')
    trial_map = {
        trial: idx
        for idx, trial in enumerate(unique_trials['trial'].to_list())
    }

    lick_positions = lick_frames.select(
        pl.col('position'),
        pl.col('trial').replace_strict(trial_map).cast(pl.Int32).alias('trial_idx'),
    )

    return lick_positions


def _add_cue_bar_top(
    ax: Axes,
    config: dict,
    trial_type: str,
    bar_width: float = 0.04,
    font_scale: float = 1.0,
):
    """Add a color-coded cue bar at the TOP of an axes.

    Unlike plot_utils.add_cue_bar which places at bottom (ymin=0), this
    places above the axes (ymin=1) to avoid x-tick overlap.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        bar_width: Fraction of axes height for bar thickness.
        font_scale: Scale factor for label font size.
    """
    cue_colors = pfmt.get_cue_colors(config)
    cue_labels = pfmt.get_cue_labels(config)
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        color = cue_colors.get(cue_id, '#D3D3D3')
        mid = pos + w / 2

        ax.axvspan(pos, pos + w, ymin=1, ymax=1 + bar_width,
                   color=color, alpha=0.9, clip_on=False, zorder=10)
        if cue_id != 0:
            ax.text(mid, 1 + bar_width / 2, cue_labels.get(cue_id, ''),
                    ha='center', va='center', fontsize=6 * font_scale,
                    fontweight='bold', color='white',
                    transform=ax.get_xaxis_transform(),
                    clip_on=False, zorder=11)

        pos += w


def plot_lick_raster(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    ax: Axes | None = None,
    label: str | None = None,
    dot_size: float = 4.0,
    dot_color: str | None = None,
    dot_alpha: float = 0.5,
    show_reward_zone: bool = True,
    show_cue_bar: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    font_scale: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot lick positions vs trial number as a scatter raster with cue shading.

    Each dot is one lick event. Background shows cue region colors.
    Optionally shows reward zone boundaries and a cue color bar at top.
    This is the single-panel building block — use plot_session_lick_raster()
    as the main entry point.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        trial_type: Trial type to plot (e.g., 'ABC'). Required.
        ax: Matplotlib Axes to plot on. Created if None.
        label: Subplot title string (e.g., 'Day 1', '09-15').
        dot_size: Scatter point size.
        dot_color: Hex color for lick dots. Defaults to trial type color.
        dot_alpha: Scatter point transparency.
        show_reward_zone: Show dashed reward zone boundaries.
        show_cue_bar: Show colored cue bar at top of plot.
        show_xlabel: Show 'Position (cm)' x-axis label.
        show_ylabel: Show 'Trial' y-axis label.
        font_scale: Scale factor for all font sizes.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))
    else:
        fig = ax.get_figure()

    track_length = get_track_length(config, trial_type)

    # Extract lick positions
    lick_data = _extract_lick_positions(df, trial_type=trial_type)

    # Count total trials for y-axis range
    n_trials = df.filter(pl.col('trial_type') == trial_type)['trial'].n_unique()

    # Cue shading
    cue_regions = get_cue_regions(config, trial_type)
    cue_colors = pfmt.get_cue_colors(config)
    for cue_id, regions in cue_regions.items():
        color = cue_colors.get(cue_id, '#D3D3D3')
        if isinstance(regions, tuple):
            regions = [regions]
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)

    # Reward zone
    if show_reward_zone:
        ts = config.get('trial_structures', {}).get(trial_type, {})
        if 'reward_zone_start_cm' in ts:
            rz_start = ts['reward_zone_start_cm']
            rz_end = ts['reward_zone_end_cm']
            ax.axvspan(rz_start, rz_end, alpha=0.15, color='#1B9AAA', zorder=1)
            ax.axvline(rz_start, color='#1B9AAA', linestyle='--',
                       linewidth=1.0 * font_scale, alpha=0.6, zorder=5)
            ax.axvline(rz_end, color='#1B9AAA', linestyle='--',
                       linewidth=1.0 * font_scale, alpha=0.6, zorder=5)

    # Scatter licks
    if dot_color is None:
        dot_color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#333333')

    if not lick_data.is_empty():
        ax.scatter(
            lick_data['position'].to_numpy(),
            lick_data['trial_idx'].to_numpy(),
            s=dot_size,
            c=dot_color,
            alpha=dot_alpha,
            edgecolors='none',
            zorder=3,
            rasterized=True,
        )

    # Cue bar at top (above axes, no tick overlap)
    if show_cue_bar:
        _add_cue_bar_top(ax, config, trial_type, bar_width=.03, font_scale=font_scale)

    # X-ticks at cue boundaries
    ts = config.get('trial_structures', {}).get(trial_type, {})
    cue_sequence = ts.get('cue_sequence', [])
    cue_widths_map = config.get('cue_map', {})
    ticks = [0.0]
    for cue_id in cue_sequence:
        ticks.append(ticks[-1] + cue_widths_map.get(cue_id, 30.0))
    if ticks[-1] != track_length:
        ticks.append(track_length)
    ticks = sorted(set(int(t) for t in ticks))

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7 * font_scale)

    # Axis formatting
    ax.set_xlim(0, track_length)
    ax.set_ylim(-0.5, n_trials - 0.5)
    ax.invert_yaxis()

    if show_xlabel:
        ax.set_xlabel('Position (cm)', fontsize=9 * font_scale)
    if show_ylabel:
        ax.set_ylabel('Trial', fontsize=9 * font_scale)

    ax.tick_params(labelsize=7 * font_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if label:
        ax.set_title(label, fontsize=10 * font_scale, fontweight='bold', pad=20)

    return fig, ax


def plot_session_lick_raster(
    df: pl.DataFrame,
    config: dict,
    label: str | None = None,
    dot_size: float = 4.0,
    dot_alpha: float = 0.5,
    figsize_per_panel: tuple[float, float] = (4, 6),
    font_scale: float = 1.0,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Plot lick rasters for all trial types in a session.

    Auto-detects trial types from the DataFrame and creates one subplot
    per trial type, side by side. Use this as the primary entry point
    for single-session lick raster plots.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        label: Base title for the figure (trial type is appended per panel).
        dot_size: Scatter point size.
        dot_alpha: Scatter point transparency.
        figsize_per_panel: (width, height) per subplot panel.
        font_scale: Scale factor for all font sizes.
        **kwargs: Additional keyword arguments passed to plot_lick_raster.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    trial_types = sorted(df['trial_type'].unique().to_list())
    n_panels = len(trial_types)

    fig_width = figsize_per_panel[0] * n_panels
    fig_height = figsize_per_panel[1]

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]  # flatten from (1, n) to (n,)

    for i, trial_type in enumerate(trial_types):
        panel_label = f"{label} — {trial_type}" if label else trial_type

        plot_lick_raster(
            df,
            config,
            trial_type=trial_type,
            ax=axes[i],
            label=panel_label,
            dot_size=dot_size,
            dot_alpha=dot_alpha,
            show_ylabel=(i == 0),
            font_scale=font_scale,
            **kwargs,
        )

    return fig, list(axes)


def plot_multiday_lick_raster(
    sessions: dict[str, dict],
    dot_size: float = 4.0,
    dot_alpha: float = 0.5,
    figsize_per_panel: tuple[float, float] = (2.5, 3.0),
    labels: list[str] | None = None,
    suptitle: str | None = None,
    label_format: str = 'day_number',
    font_scale: float = 1.0,
    **kwargs,
) -> Figure:
    """Plot a multi-day lick raster grid, auto-detecting trial types per session.

    Derives the grid layout from the data: one row per unique trial type found
    across all sessions, one column per session day. Panels where a trial type
    doesn't exist in a session are left blank with a 'No X trials' message.

    Args:
        sessions: Dict from load_multiday_sessions(), keyed by date string,
            each value containing 'data' and 'config' keys.
        dot_size: Scatter point size.
        dot_alpha: Scatter point transparency.
        figsize_per_panel: (width, height) per subplot panel.
        labels: Manually add labels for different days (i.e. Base 1, Extended day 1; or even just day 2, day 4)
            overrides label_format if provided
        suptitle: Figure super-title. Auto-generated if None.
        label_format: 'day_number' for 'Day 1', 'Day 2', etc.
            'date' for the session date string (MM-DD).
        font_scale: Scale factor for all font sizes.
        **kwargs: Additional keyword arguments passed to plot_lick_raster.

    Returns:
        The matplotlib Figure.
    """
    sorted_dates = sorted(sessions.keys())
    n_cols = len(sorted_dates)

    # Discover all trial types across all sessions, sorted
    all_trial_types = set()
    for date in sorted_dates:
        session_types = sessions[date]['data']['trial_type'].unique().to_list()
        all_trial_types.update(session_types)
    trial_types = sorted(all_trial_types)
    n_rows = len(trial_types)

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    for col, date in enumerate(sorted_dates):
        session = sessions[date]
        session_df = session['data']
        session_config = session['config']
        available_types = session_df['trial_type'].unique().to_list()

        # Column label
        if labels is not None:
            col_label = labels[col]
        elif label_format == 'day_number':
            col_label = f"Day {col + 1}"
        else:
            col_label = date[5:]  # MM-DD

        for row, trial_type in enumerate(trial_types):
            ax = axes[row, col]

            if trial_type not in available_types:
                ax.text(0.5, 0.5, f'No {trial_type}\ntrials',
                        ha='center', va='center', fontsize=8 * font_scale,
                        color='#999999', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # Still show column label on top row
                if row == 0:
                    ax.set_title(col_label, fontsize=10 * font_scale,
                                 fontweight='bold', pad=15)
                continue

            # Only show labels on edges
            show_xlabel = (row == n_rows - 1)
            show_ylabel = (col == 0)
            # Only show column label on top row
            panel_label = col_label if row == 0 else None

            plot_lick_raster(
                session_df,
                session_config,
                trial_type=trial_type,
                ax=ax,
                label=panel_label,
                dot_size=dot_size,
                dot_alpha=dot_alpha,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                font_scale=font_scale,

                **kwargs,
            )

    # Row labels on left
    for row, trial_type in enumerate(trial_types):
        axes[row, 0].annotate(
            trial_type,
            xy=(-0.3, 0.5),
            xycoords='axes fraction',
            fontsize=11 * font_scale,
            fontweight='bold',
            ha='right',
            va='center',
            rotation=90,
        )

    if suptitle is None:
        suptitle = f"Lick Raster — {n_cols} Sessions"
    fig.suptitle(suptitle, fontsize=13 * font_scale, fontweight='bold')

    return fig


def plot_lick_raster_grid(
    sessions: list[tuple[pl.DataFrame, dict, str]],
    trial_types: list[str],
    n_cols: int | None = None,
    dot_size: float = 0.8,
    dot_alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    row_labels: list[str] | None = None,
    font_scale: float = 1.0,
) -> Figure:
    """Plot a multi-day, multi-trial-type lick raster grid.

    Creates a grid where each column is one session/day and each row is one
    trial type. Useful for comparing lick behavior across learning stages.

    Example layout (3 rows × 7 cols):
        Row 0: Days 1-7, trial_type='ABC'
        Row 1: Days 8-14, trial_type='ABC'
        Row 2: Days 8-14, trial_type='ABDC'

    Args:
        sessions: List of (DataFrame, config_dict, label_str) tuples,
            one per session/day. Sessions are laid out left-to-right in columns.
        trial_types: List of trial type strings, one per row. If a row's
            trial type doesn't exist in a session, that panel is left blank.
        n_cols: Number of columns per row. Defaults to len(sessions) / n_rows.
        dot_size: Scatter point size.
        dot_alpha: Scatter point transparency.
        figsize: Figure size (width, height). Auto-scaled if None.
        suptitle: Figure super-title.
        row_labels: Labels for each row (shown on left side). Defaults to
            trial type names.
        font_scale: Scale factor for all font sizes.

    Returns:
        The matplotlib Figure.
    """
    n_rows = len(trial_types)
    if n_cols is None:
        n_cols = len(sessions) // n_rows if n_rows > 1 else len(sessions)

    if figsize is None:
        figsize = (2.5 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )

    if row_labels is None:
        row_labels = trial_types

    session_idx = 0
    for row in range(n_rows):
        trial_type = trial_types[row]

        for col in range(n_cols):
            ax = axes[row, col]

            if session_idx >= len(sessions):
                ax.set_visible(False)
                continue

            session_df, session_config, session_label = sessions[session_idx]

            # Check if this trial type exists in this session
            available_types = session_df['trial_type'].unique().to_list()
            if trial_type not in available_types:
                ax.text(0.5, 0.5, f'No {trial_type}\ntrials',
                        ha='center', va='center', fontsize=8 * font_scale,
                        color='#999999', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                session_idx += 1
                continue

            show_xlabel = (row == n_rows - 1)
            show_ylabel = (col == 0)

            plot_lick_raster(
                session_df,
                session_config,
                trial_type=trial_type,
                ax=ax,
                label=session_label,
                dot_size=dot_size,
                dot_alpha=dot_alpha,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                font_scale=font_scale,
            )

            session_idx += 1

        # Row label on left
        if row_labels:
            axes[row, 0].annotate(
                row_labels[row],
                xy=(-0.3, 0.5),
                xycoords='axes fraction',
                fontsize=11 * font_scale,
                fontweight='bold',
                ha='right',
                va='center',
                rotation=90,
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13 * font_scale, fontweight='bold')

    return fig


# ══════════════════════════════════════════════════════════════════════
# SPEED PROFILES
# ══════════════════════════════════════════════════════════════════════


def _compute_speed_by_position(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    bin_size_cm: int = 5,
) -> dict:
    """Compute mean and SEM of running speed at each spatial bin for one trial type.

    Args:
        df: Processed frame-level DataFrame with 'speed_cm_s', 'distance_bin',
            'trial', 'trial_type' columns.
        config: Experiment configuration dict.
        trial_type: Trial type to compute for.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        Dict with 'bin_centers', 'mean_speed', 'sem_speed', 'per_trial_speed',
        'n_trials' keys. per_trial_speed is shape (n_trials, n_bins).
    """
    track_length = get_track_length(config, trial_type)
    n_bins = int(track_length / bin_size_cm)

    sub = df.filter(pl.col('trial_type') == trial_type)
    trials = sub['trial'].to_numpy()
    bins = sub['distance_bin'].to_numpy().clip(0, n_bins - 1)
    speeds = sub['speed_cm_s'].to_numpy()

    unique_trials = np.unique(trials)
    n_trials = len(unique_trials)
    trial_idx = np.searchsorted(unique_trials, trials)

    # Accumulate per (trial, bin)
    sums = np.zeros((n_trials, n_bins))
    counts = np.zeros((n_trials, n_bins))
    np.add.at(sums, (trial_idx, bins), speeds)
    np.add.at(counts, (trial_idx, bins), 1)

    with np.errstate(invalid='ignore'):
        per_trial_speed = sums / np.where(counts > 0, counts, np.nan)

    mean_speed = np.nanmean(per_trial_speed, axis=0)
    sem_speed = np.nanstd(per_trial_speed, axis=0, ddof=1) / np.sqrt(n_trials)
    bin_centers = np.arange(n_bins) * bin_size_cm + bin_size_cm / 2

    return {
        'bin_centers': bin_centers,
        'mean_speed': mean_speed,
        'sem_speed': sem_speed,
        'per_trial_speed': per_trial_speed,
        'n_trials': n_trials,
    }

# TODO not using the right position axis, cant plot both trials
def plot_speed_profile(
    df: pl.DataFrame,
    config: dict,
    bin_size_cm: int = 5,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> Figure:
    """Plot mean running speed vs position for each trial type, overlaid.

    Shows mean ± SEM speed profile with cue shading and reward zones.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        bin_size_cm: Spatial bin size in cm.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    trial_types = sorted(df['trial_type'].unique().to_list())
    trial_type_colors, _ = pfmt.get_trial_type_colors(config)

    fig, ax = plt.subplots(figsize=figsize)

    # Cue shading for longest track
    longest_tt = max(trial_types, key=lambda tt: get_track_length(config, tt))
    pfmt.add_cue_shading(ax, config, longest_tt, alpha=0.12)

    for tt in trial_types:
        stats = _compute_speed_by_position(df, config, tt, bin_size_cm)
        color = trial_type_colors.get(tt, '#999999')
        x = stats['bin_centers']

        ax.plot(x, stats['mean_speed'], color=color, linewidth=2, label=tt, zorder=3)
        ax.fill_between(
            x,
            stats['mean_speed'] - stats['sem_speed'],
            stats['mean_speed'] + stats['sem_speed'],
            color=color, alpha=0.2, zorder=2,
        )

    # Reward zones
    for tt in trial_types:
        ts = config.get('trial_structures', {}).get(tt, {})
        if 'reward_zone_start_cm' in ts:
            color = trial_type_colors.get(tt, '#1B9AAA')
            ax.axvspan(ts['reward_zone_start_cm'], ts['reward_zone_end_cm'],
                       alpha=0.1, color=color, zorder=0)

    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('Speed (cm/s)', fontsize=11)
    ax.set_title(
        pfmt.build_title('Speed Profile', animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, get_track_length(config, longest_tt))

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ── SPEED RASTER ─────────────────────────────────────────────────────


def plot_speed_raster(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    bin_size_cm: int = 5,
    ax: Axes | None = None,
    label: str | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap: str = 'viridis',
    show_reward_zone: bool = True,
    show_cue_bar: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    font_scale: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot a trial × position heatmap of running speed.

    Each row is one trial, columns are spatial bins, color is mean speed
    in that trial/bin. Analogous to lick raster but continuous-valued.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        trial_type: Trial type to plot.
        bin_size_cm: Spatial bin size in cm.
        ax: Matplotlib Axes. Created if None.
        label: Subplot title string.
        vmin: Colorbar minimum speed.
        vmax: Colorbar maximum speed. Auto-scaled if None.
        cmap: Colormap name.
        show_reward_zone: Show dashed reward zone boundaries.
        show_cue_bar: Show colored cue bar at top.
        show_xlabel: Show x-axis label.
        show_ylabel: Show y-axis label.
        font_scale: Scale factor for all font sizes.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        fig = ax.get_figure()

    stats = _compute_speed_by_position(df, config, trial_type, bin_size_cm)
    speed_matrix = stats['per_trial_speed']  # (n_trials, n_bins)
    track_length = get_track_length(config, trial_type)
    n_trials = stats['n_trials']

    if vmax is None:
        vmax = np.nanpercentile(speed_matrix, 95)

    im = ax.imshow(
        speed_matrix,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0, track_length, n_trials - 0.5, -0.5],
        interpolation='nearest',
    )
    fig.colorbar(im, ax=ax, label='Speed (cm/s)', shrink=0.7, pad=0.02)

    # Reward zone
    if show_reward_zone:
        ts = config.get('trial_structures', {}).get(trial_type, {})
        if 'reward_zone_start_cm' in ts:
            ax.axvline(ts['reward_zone_start_cm'], color='white', linestyle='--',
                       linewidth=1.0 * font_scale, alpha=0.7, zorder=5)
            ax.axvline(ts['reward_zone_end_cm'], color='white', linestyle='--',
                       linewidth=1.0 * font_scale, alpha=0.7, zorder=5)

    # Cue bar at top
    if show_cue_bar:
        _add_cue_bar_top(ax, config, trial_type, bar_width=0.03, font_scale=font_scale)

    ax.set_xlim(0, track_length)
    if show_xlabel:
        ax.set_xlabel('Position (cm)', fontsize=9 * font_scale)
    if show_ylabel:
        ax.set_ylabel('Trial', fontsize=9 * font_scale)

    ax.tick_params(labelsize=7 * font_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if label:
        ax.set_title(label, fontsize=10 * font_scale, fontweight='bold', pad=20)

    return fig, ax


def plot_session_speed_raster(
    df: pl.DataFrame,
    config: dict,
    bin_size_cm: int = 5,
    label: str | None = None,
    figsize_per_panel: tuple[float, float] = (5, 6),
    font_scale: float = 1.0,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Plot speed rasters for all trial types in a session, side by side.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        bin_size_cm: Spatial bin size in cm.
        label: Base title for the figure.
        figsize_per_panel: (width, height) per subplot panel.
        font_scale: Scale factor for all font sizes.
        **kwargs: Additional keyword arguments passed to plot_speed_raster.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    trial_types = sorted(df['trial_type'].unique().to_list())
    n_panels = len(trial_types)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]

    for i, trial_type in enumerate(trial_types):
        panel_label = f"{label} — {trial_type}" if label else trial_type
        plot_speed_raster(
            df, config, trial_type=trial_type, bin_size_cm=bin_size_cm,
            ax=axes[i], label=panel_label,
            show_ylabel=(i == 0), font_scale=font_scale, **kwargs,
        )

    return fig, list(axes)


# ── SPEED LEARNING CURVES (MULTIDAY) ────────────────────────────────


def plot_multiday_speed_profile(
    sessions: dict[str, dict],
    bin_size_cm: int = 5,
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    animal_id: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.5, 3.0),
    font_scale: float = 1.0,
    show: bool = True,
) -> Figure:
    """Plot speed profiles across sessions, one column per day, one row per trial type.

    Shows how spatial speed profile evolves with learning. Early sessions
    should show relatively uniform speed; late sessions should show
    anticipatory slowing near reward zones.

    Args:
        sessions: Dict from load_multiday_sessions().
        bin_size_cm: Spatial bin size in cm.
        labels: Manual column labels. Overrides label_format.
        label_format: 'day_number' or 'date'.
        animal_id: Animal identifier for suptitle.
        figsize_per_panel: (width, height) per panel.
        font_scale: Scale factor for all font sizes.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    sorted_dates = sorted(sessions.keys())
    n_cols = len(sorted_dates)

    all_trial_types = set()
    for date in sorted_dates:
        all_trial_types.update(sessions[date]['data']['trial_type'].unique().to_list())
    trial_types = sorted(all_trial_types)
    n_rows = len(trial_types)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False, constrained_layout=True,
    )

    # Compute global y-max across all sessions for consistent scaling
    global_max_speed = 0.0
    for date in sorted_dates:
        session = sessions[date]
        for tt in trial_types:
            if tt in session['data']['trial_type'].unique().to_list():
                stats = _compute_speed_by_position(
                    session['data'], session['config'], tt, bin_size_cm,
                )
                peak = np.nanmax(stats['mean_speed'] + stats['sem_speed'])
                global_max_speed = max(global_max_speed, peak)

    trial_type_colors = {}
    if sessions:
        first_config = next(iter(sessions.values()))['config']
        trial_type_colors, _ = pfmt.get_trial_type_colors(first_config)

    for col, date in enumerate(sorted_dates):
        session = sessions[date]
        session_df = session['data']
        session_config = session['config']
        available_types = session_df['trial_type'].unique().to_list()

        if labels is not None:
            col_label = labels[col]
        elif label_format == 'day_number':
            col_label = f"Day {col + 1}"
        else:
            col_label = date[5:]

        for row, tt in enumerate(trial_types):
            ax = axes[row, col]

            if tt not in available_types:
                ax.text(0.5, 0.5, f'No {tt}\ntrials', ha='center', va='center',
                        fontsize=8 * font_scale, color='#999999',
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(col_label, fontsize=10 * font_scale,
                                 fontweight='bold')
                continue

            pfmt.add_cue_shading(ax, session_config, tt, alpha=0.1)
            stats = _compute_speed_by_position(
                session_df, session_config, tt, bin_size_cm,
            )
            color = trial_type_colors.get(tt, '#999999')
            x = stats['bin_centers']

            ax.plot(x, stats['mean_speed'], color=color, linewidth=1.5)
            ax.fill_between(
                x,
                stats['mean_speed'] - stats['sem_speed'],
                stats['mean_speed'] + stats['sem_speed'],
                color=color, alpha=0.2,
            )

            # Reward zone
            ts = session_config.get('trial_structures', {}).get(tt, {})
            if 'reward_zone_start_cm' in ts:
                ax.axvspan(ts['reward_zone_start_cm'], ts['reward_zone_end_cm'],
                           alpha=0.1, color='#1B9AAA', zorder=0)

            track_length = get_track_length(session_config, tt)
            ax.set_xlim(0, track_length)
            ax.set_ylim(0, global_max_speed * 1.05)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == n_rows - 1:
                ax.set_xlabel('Position (cm)', fontsize=8 * font_scale)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel('Speed (cm/s)', fontsize=8 * font_scale)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(col_label, fontsize=10 * font_scale, fontweight='bold')

            ax.tick_params(labelsize=6 * font_scale)

    # Row labels on left
    for row, tt in enumerate(trial_types):
        axes[row, 0].annotate(
            tt, xy=(-0.3, 0.5), xycoords='axes fraction',
            fontsize=11 * font_scale, fontweight='bold',
            ha='right', va='center', rotation=90,
        )

    suptitle = pfmt.build_title('Speed Profile Learning', animal_id=animal_id)
    fig.suptitle(suptitle, fontsize=13 * font_scale, fontweight='bold')

    if show:
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════
# REWARD / WATER METRICS
# ══════════════════════════════════════════════════════════════════════


def compute_reward_metrics(
    df: pl.DataFrame,
    config: dict,
) -> pl.DataFrame:
    """Compute per-trial reward metrics: whether reward was collected and water volume.

    A trial counts as 'rewarded' if any frame has reward == 'yes'.
    Water consumed per trial is max(water_uL) - min(water_uL) within that trial.

    Args:
        df: Processed frame-level DataFrame with 'reward', 'water_uL',
            'trial', 'trial_type' columns.
        config: Experiment configuration dict.

    Returns:
        DataFrame with columns: trial, trial_type, rewarded (bool),
        water_consumed_uL, trial_duration_s.
    """
    return (
        df
        .group_by('trial', maintain_order=True)
        .agg(
            pl.col('trial_type').first(),
            (pl.col('reward') == 'yes').any().alias('rewarded'),
            (pl.col('water_uL').max() - pl.col('water_uL').min()).alias('water_consumed_uL'),
            ((pl.col('time_us').max() - pl.col('time_us').min()) / 1_000_000).alias('trial_duration_s'),
        )
    )


def compute_session_reward_summary(
    df: pl.DataFrame,
    config: dict,
) -> dict[str, dict]:
    """Compute session-level reward summary stats per trial type.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.

    Returns:
        Dict mapping trial_type -> {'hit_rate', 'mean_water_uL',
        'total_water_uL', 'n_trials', 'n_rewarded'}.
    """
    trial_metrics = compute_reward_metrics(df, config)
    result = {}

    for tt in sorted(trial_metrics['trial_type'].unique().to_list()):
        sub = trial_metrics.filter(pl.col('trial_type') == tt)
        n_trials = len(sub)
        n_rewarded = sub['rewarded'].sum()
        total_water = sub['water_consumed_uL'].sum()

        result[tt] = {
            'hit_rate': n_rewarded / n_trials if n_trials > 0 else 0.0,
            'mean_water_uL': total_water / n_trials if n_trials > 0 else 0.0,
            'total_water_uL': total_water,
            'n_trials': n_trials,
            'n_rewarded': n_rewarded,
        }

    return result


def plot_reward_metrics(
    df: pl.DataFrame,
    config: dict,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> Figure:
    """Plot per-trial reward metrics: hit rate, water per trial, cumulative water.

    Three panels:
        1. Hit rate (rolling fraction of rewarded trials)
        2. Water consumed per trial
        3. Cumulative water over trials

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    trial_metrics = compute_reward_metrics(df, config)
    trial_type_colors, _ = pfmt.get_trial_type_colors(config)

    fig, (ax_hit, ax_water, ax_cum) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for tt in sorted(trial_metrics['trial_type'].unique().to_list()):
        sub = trial_metrics.filter(pl.col('trial_type') == tt).sort('trial')
        color = trial_type_colors.get(tt, '#999999')
        trials = sub['trial'].to_numpy()
        rewarded = sub['rewarded'].to_numpy().astype(float)
        water = sub['water_consumed_uL'].to_numpy()

        # Rolling hit rate (window=10 or n_trials, whichever smaller)
        window = min(10, len(rewarded))
        if window > 0:
            rolling_hit = np.convolve(rewarded, np.ones(window) / window, mode='valid')
            x_rolling = trials[window - 1:]
            ax_hit.plot(x_rolling, rolling_hit, color=color, linewidth=1.5, label=tt)

        # Water per trial
        ax_water.scatter(trials, water, color=color, s=12, alpha=0.6, label=tt)

        # Cumulative water
        ax_cum.plot(trials, np.cumsum(water), color=color, linewidth=1.5, label=tt)

    ax_hit.set_ylabel('Hit Rate (rolling)', fontsize=10)
    ax_hit.set_ylim(-0.05, 1.05)
    ax_hit.axhline(1.0, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    ax_hit.legend(frameon=False, fontsize=9)
    ax_hit.spines['top'].set_visible(False)
    ax_hit.spines['right'].set_visible(False)

    ax_water.set_ylabel('Water (µL/trial)', fontsize=10)
    ax_water.spines['top'].set_visible(False)
    ax_water.spines['right'].set_visible(False)

    ax_cum.set_ylabel('Cumulative Water (µL)', fontsize=10)
    ax_cum.set_xlabel('Trial', fontsize=10)
    ax_cum.spines['top'].set_visible(False)
    ax_cum.spines['right'].set_visible(False)

    fig.suptitle(
        pfmt.build_title('Reward Metrics', animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_multiday_reward_summary(
    sessions: dict[str, dict],
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    animal_id: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    font_scale: float = 1.0,
    show: bool = True,
) -> Figure:
    """Plot reward hit rate and total water across sessions.

    Two panels: hit rate per session and total water per session,
    separated by trial type.

    Args:
        sessions: Dict from load_multiday_sessions().
        labels: Manual x-axis labels. Overrides label_format.
        label_format: 'day_number' or 'date'.
        animal_id: Animal identifier for suptitle.
        figsize: Figure size.
        font_scale: Scale factor.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    sorted_dates = sorted(sessions.keys())
    n_sessions = len(sorted_dates)

    if labels is None:
        if label_format == 'day_number':
            labels = [f"Day {i + 1}" for i in range(n_sessions)]
        else:
            labels = [d[5:] for d in sorted_dates]

    # Collect all trial types across sessions
    all_trial_types = set()
    for date in sorted_dates:
        all_trial_types.update(
            sessions[date]['data']['trial_type'].unique().to_list()
        )
    trial_types = sorted(all_trial_types)

    trial_type_colors = {}
    if sessions:
        first_config = next(iter(sessions.values()))['config']
        trial_type_colors, _ = pfmt.get_trial_type_colors(first_config)

    fig, (ax_hit, ax_water) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
    )

    x = np.arange(n_sessions)
    bar_width = 0.8 / len(trial_types)

    for i_tt, tt in enumerate(trial_types):
        hit_rates = []
        total_waters = []
        color = trial_type_colors.get(tt, '#999999')

        for date in sorted_dates:
            session = sessions[date]
            summary = compute_session_reward_summary(session['data'], session['config'])
            if tt in summary:
                hit_rates.append(summary[tt]['hit_rate'])
                total_waters.append(summary[tt]['total_water_uL'])
            else:
                hit_rates.append(np.nan)
                total_waters.append(np.nan)

        offsets = x + (i_tt - len(trial_types) / 2 + 0.5) * bar_width
        ax_hit.bar(offsets, hit_rates, width=bar_width, color=color,
                   alpha=0.7, label=tt, edgecolor='white', linewidth=0.5)
        ax_water.bar(offsets, total_waters, width=bar_width, color=color,
                     alpha=0.7, label=tt, edgecolor='white', linewidth=0.5)

    ax_hit.set_ylabel('Hit Rate', fontsize=10 * font_scale)
    ax_hit.set_ylim(0, 1.05)
    ax_hit.axhline(1.0, color='gray', linewidth=0.5, alpha=0.3, linestyle='--')
    ax_hit.legend(frameon=False, fontsize=9 * font_scale)
    ax_hit.spines['top'].set_visible(False)
    ax_hit.spines['right'].set_visible(False)

    ax_water.set_ylabel('Total Water (µL)', fontsize=10 * font_scale)
    ax_water.set_xlabel('Session', fontsize=10 * font_scale)
    ax_water.set_xticks(x)
    ax_water.set_xticklabels(labels, fontsize=8 * font_scale, rotation=45, ha='right')
    ax_water.spines['top'].set_visible(False)
    ax_water.spines['right'].set_visible(False)

    fig.suptitle(
        pfmt.build_title('Reward Summary', animal_id=animal_id),
        fontsize=13 * font_scale, fontweight='bold',
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════
# TRIAL DURATION
# ══════════════════════════════════════════════════════════════════════


def plot_trial_duration(
    df: pl.DataFrame,
    config: dict,
    animal_id: str | None = None,
    date: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> Figure:
    """Plot trial duration across trials, colored by trial type.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    trial_metrics = compute_reward_metrics(df, config)
    trial_type_colors, _ = pfmt.get_trial_type_colors(config)

    fig, ax = plt.subplots(figsize=figsize)

    for tt in sorted(trial_metrics['trial_type'].unique().to_list()):
        sub = trial_metrics.filter(pl.col('trial_type') == tt).sort('trial')
        color = trial_type_colors.get(tt, '#999999')
        trials = sub['trial'].to_numpy()
        durations = sub['trial_duration_s'].to_numpy()

        ax.scatter(trials, durations, color=color, s=15, alpha=0.6, label=tt)

        # Smoothed trend (rolling median, window=5)
        window = min(5, len(durations))
        if window >= 3:
            from scipy.ndimage import median_filter
            smoothed = median_filter(durations, size=window)
            ax.plot(trials, smoothed, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Duration (s)', fontsize=11)
    ax.set_title(
        pfmt.build_title('Trial Duration', animal_id=animal_id, date=date),
        fontsize=13, fontweight='bold',
    )
    ax.legend(frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_multiday_trial_duration(
    sessions: dict[str, dict],
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    animal_id: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    font_scale: float = 1.0,
    show: bool = True,
) -> Figure:
    """Plot mean trial duration per session across days, by trial type.

    Shows box-like summary (mean ± std) for each session/trial type combination.

    Args:
        sessions: Dict from load_multiday_sessions().
        labels: Manual x-axis labels.
        label_format: 'day_number' or 'date'.
        animal_id: Animal identifier for suptitle.
        figsize: Figure size.
        font_scale: Scale factor.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    sorted_dates = sorted(sessions.keys())
    n_sessions = len(sorted_dates)

    if labels is None:
        if label_format == 'day_number':
            labels = [f"Day {i + 1}" for i in range(n_sessions)]
        else:
            labels = [d[5:] for d in sorted_dates]

    all_trial_types = set()
    for date in sorted_dates:
        all_trial_types.update(
            sessions[date]['data']['trial_type'].unique().to_list()
        )
    trial_types = sorted(all_trial_types)

    trial_type_colors = {}
    if sessions:
        first_config = next(iter(sessions.values()))['config']
        trial_type_colors, _ = pfmt.get_trial_type_colors(first_config)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_sessions)
    offset_width = 0.8 / len(trial_types)

    for i_tt, tt in enumerate(trial_types):
        means = []
        stds = []
        color = trial_type_colors.get(tt, '#999999')

        for date in sorted_dates:
            session = sessions[date]
            trial_metrics = compute_reward_metrics(session['data'], session['config'])
            sub = trial_metrics.filter(pl.col('trial_type') == tt)
            if len(sub) > 0:
                durations = sub['trial_duration_s'].to_numpy()
                means.append(np.mean(durations))
                stds.append(np.std(durations))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        offsets = x + (i_tt - len(trial_types) / 2 + 0.5) * offset_width
        ax.bar(offsets, means, width=offset_width, color=color, alpha=0.7,
               label=tt, edgecolor='white', linewidth=0.5)
        ax.errorbar(offsets, means, yerr=stds, fmt='none', ecolor='black',
                    elinewidth=0.8, capsize=2)

    ax.set_ylabel('Mean Trial Duration (s)', fontsize=10 * font_scale)
    ax.set_xlabel('Session', fontsize=10 * font_scale)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8 * font_scale, rotation=45, ha='right')
    ax.legend(frameon=False, fontsize=9 * font_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(
        pfmt.build_title('Trial Duration', animal_id=animal_id),
        fontsize=13 * font_scale, fontweight='bold',
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# HELPERS

def sessions_to_raster_input(
    sessions: dict[str, dict],
    label_format: str = 'day_number',
    day_offset: int = 0,
) -> list[tuple[pl.DataFrame, dict, str]]:
    """Convert load_multiday_sessions() output to plot_lick_raster_grid input.

    Args:
        sessions: Dict from load_multiday_sessions(), keyed by date string,
            each value containing 'data', 'config', 'session_data' keys.
        label_format: 'day_number' for 'Day 1', 'Day 2', etc.
            'date' for the session date string (MM-DD).
        day_offset: Starting day number offset (e.g., 7 for second phase).

    Returns:
        List of (DataFrame, config, label) tuples sorted chronologically.
    """
    result = []
    for i, date in enumerate(sorted(sessions.keys())):
        session = sessions[date]
        if label_format == 'day_number':
            label = f"Day {i + 1 + day_offset}"
        else:
            label = date[5:]  # MM-DD
        result.append((session['data'], session['config'], label))
    return result



if __name__ == '__main__':

    from pathlib import Path
    from df_processing import (find_session_dir, get_session_paths, load_processed_session,
                               load_session_context, load_multiday_sessions)

    mouse_id = '26'
    date = '2025-09-08'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)

    data, meta = load_processed_session(paths['parquet'])

    # Single session (auto-splits trial types)
    fig, axes = plot_session_lick_raster(data, exp_config, label=date)
    plt.show()

    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-02', '2025-09-08'), auto_process=False,
    )
    fig = plot_multiday_lick_raster(sessions)
    plt.show()

    fig2 = plot_speed_profile(data, exp_config)
    plt.show()

    fig3 = plot_speed_raster(data, exp_config, trial_type = 'ABC')
    plt.show()

    fig4 = plot_session_speed_raster(data, exp_config)
    plt.show()

