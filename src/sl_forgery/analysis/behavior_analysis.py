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

from df_processing import get_track_length, get_cue_regions, get_bin_size
import plot_utils as pfmt


# LICK RASTER

def _extract_lick_positions(
    df: pl.DataFrame,
    trial_type: str | None = None,
) -> pl.DataFrame:
    """Extracts lick event positions and trial numbers from a processed session DataFrame.

    Pulls every frame where the lick sensor fired (lick == 1) and pairs the track position
    with a zero-indexed trial counter. Trial indices are renumbered sequentially so that
    filtered subsets (e.g. a single trial type) still produce a contiguous y-axis in raster
    plots. The output is ready to scatter directly as x=position, y=trial_idx.

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
    """Adds a color-coded cue bar at the top of an axes.

    Draws a thin horizontal strip above the plot area where each segment is colored
    according to its cue identity, with letter labels (A, B, C, ...) centered inside.
    Unlike plot_utils.add_cue_bar which places the bar at the bottom (ymin=0), this
    version renders above the axes (ymin=1) so it does not overlap with x-tick labels
    on raster-style plots where the x-axis sits at the bottom.

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
    """Plots lick positions vs trial number as a scatter raster with cue shading.

    Each dot represents a single lick event, positioned at its track location (x-axis)
    and the trial it occurred in (y-axis). Background color bands show cue region
    identities so the viewer can immediately see where the animal is licking relative
    to the cue structure. Dashed lines mark the reward zone boundaries when enabled.
    In a well-trained animal, lick dots should cluster tightly around the reward zone
    and possibly at learned cue transitions, while naive animals will show scattered
    or uniformly distributed licks. This is the single-panel building block; use
    plot_session_lick_raster() as the main entry point for multi-panel figures.

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
    pfmt.set_cue_boundary_ticks(ax, config, trial_type)
    ax.tick_params(axis='x', labelsize=7 * font_scale)

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
    animal_id: str,
    date: str,
    label: str | None = None,
    dot_size: float = 4.0,
    dot_alpha: float = 0.5,
    figsize_per_panel: tuple[float, float] = (4, 6),
    font_scale: float = 1.0,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Plots lick rasters for all trial types in a session, side by side.

    Auto-detects the trial types present in the DataFrame and creates one subplot
    per type. Dots are lick events across position. This is the primary entry point for single-session lick raster
    visualization. Comparing panels reveals whether the animal discriminates between
    trial types: a trained animal should show spatially focused licking near the
    reward zone on rewarded trial types and suppressed or absent licking on
    unrewarded types.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        animal_id: Animal identifier for figure suptitle.
        date: Session date for figure suptitle.
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
        constrained_layout={'w_pad': 0.15, 'h_pad': 0.15},
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

    fig.suptitle(
        pfmt.build_title('Lick Raster', animal_id=animal_id, date=date),
        fontsize=13 * font_scale, fontweight='bold',
    )

    return fig, list(axes)


def plot_multiday_lick_raster(
    sessions: dict[str, dict],
    animal_id: str,
    dot_size: float = 4.0,
    dot_alpha: float = 0.5,
    figsize_per_panel: tuple[float, float] = (2.5, 3.0),
    labels: list[str] | None = None,
    suptitle: str | None = None,
    label_format: str = 'day_number',
    font_scale: float = 1.0,
    **kwargs,
) -> Figure:
    """Plots a multi-day lick raster grid, auto-detecting trial types per session.

    Derives the grid layout from the data: one row per unique trial type found
    across all sessions, one column per session day. Panels where a trial type
    does not exist in a session are left blank. Reading left to right shows how
    lick behavior evolves over training days. Early columns should show diffuse
    licking across the track, while later columns should show progressive
    tightening of lick clusters around the reward zone as the animal learns the
    cue-reward association.

    Args:
        sessions: Dict from load_multiday_sessions(), keyed by date string,
            each value containing 'data' and 'config' keys.
        dot_size: Scatter point size.
        dot_alpha: Scatter point transparency.
        figsize_per_panel: (width, height) per subplot panel.
        animal_id: Animal identifier for suptitle.
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
        constrained_layout={'w_pad': 0.3, 'h_pad': 0.4},
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
        suptitle = pfmt.build_title(f"Lick Raster — {n_cols} Sessions", animal_id=animal_id)
    fig.suptitle(suptitle, fontsize=13 * font_scale, fontweight='bold')

    return fig


def plot_lick_raster_grid(
    sessions: list[tuple[pl.DataFrame, dict, str]],
    trial_types: list[str],
    animal_id: str,
    n_cols: int | None = None,
    dot_size: float = 0.8,
    dot_alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
    row_labels: list[str] | None = None,
    font_scale: float = 1.0,
) -> Figure:
    """Plots a multi-day, multi-trial-type lick raster grid with explicit layout control.

    Creates a grid where each column is one session/day and each row is one trial type,
    with the caller specifying exactly which sessions fill which slots. This is useful
    when the experiment has distinct training phases (e.g. baseline then extended) that
    require different row structures, or when sessions need to be reordered or grouped
    manually rather than auto-detected. Comparing rows reveals whether the animal
    generalizes or discriminates between trial types, while comparing columns within
    a row tracks learning progression.

    Args:
        sessions: List of (DataFrame, config_dict, label_str) tuples,
            one per session/day. Sessions are laid out left-to-right in columns.
        trial_types: List of trial type strings, one per row. If a row's
            trial type doesn't exist in a session, that panel is left blank.
        n_cols: Number of columns per row. Defaults to len(sessions) / n_rows.
        dot_size: Scatter point size.
        dot_alpha: Scatter point transparency.
        animal_id: Animal identifier for suptitle.
        figsize: Figure size (width, height). Auto-scaled if None.
        suptitle: Figure super-title. Auto-generated if None.
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
        constrained_layout={'w_pad': 0.3, 'h_pad': 0.4},
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

    if suptitle is None:
        suptitle = pfmt.build_title('Lick Raster Grid', animal_id=animal_id)
    fig.suptitle(suptitle, fontsize=13 * font_scale, fontweight='bold')

    return fig



# SPEED PROFILES

def _compute_speed_by_position(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    bin_size_cm: int | None = None,
    metadata: dict | None = None,
) -> dict:
    """Computes mean and SEM of running speed at each spatial bin for one trial type.

    Groups frame-level speed measurements into spatial bins along the track, averages
    within each trial to get a per-trial speed profile, then computes the cross-trial
    mean and SEM at each bin. The resulting arrays drive both the speed profile line
    plots (mean +/- SEM) and the speed raster heatmaps (per-trial matrix).

    Args:
        df: Processed frame-level DataFrame with 'speed_cm_s', 'distance_bin',
            'trial', 'trial_type' columns.
        config: Experiment configuration dict.
        trial_type: Trial type to compute for.
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if ``bin_size_cm`` is not supplied.

    Returns:
        Dict with 'bin_centers', 'mean_speed', 'sem_speed', 'per_trial_speed',
        'n_trials' keys. per_trial_speed is shape (n_trials, n_bins).
    """
    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)
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

# TODO not using the right position axis, shouldnt plot both trials
def plot_speed_profile(
    df: pl.DataFrame,
    config: dict,
    animal_id: str,
    date: str,
    bin_size_cm: int | None = None,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
    metadata: dict | None = None,
) -> Figure:
    """Plots mean running speed vs position for each trial type, overlaid on one axes.

    Shows the trial-averaged speed as a function of track position with a shaded SEM
    envelope, cue region coloring, and reward zone highlights. Each trial type is drawn
    as a separate colored trace. In a well-trained animal, the speed profile should dip
    near the reward zone (anticipatory slowing) and may show acceleration or deceleration
    at cue transitions. Flat or noisy profiles indicate the animal has not yet learned
    the spatial structure of the task. Look for reduced speed at intro of D cue

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        animal_id: Animal identifier for plot title.
        date: Session date for plot title.
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        figsize: Figure size.
        show: Call plt.show().
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if ``bin_size_cm`` is not supplied.

    Returns:
        Matplotlib Figure.
    """
    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)
    trial_types = sorted(df['trial_type'].unique().to_list())
    trial_type_colors, _ = pfmt.get_trial_type_colors(config)

    fig, ax = plt.subplots(figsize=figsize)

    longest_tt = max(trial_types, key=lambda tt: get_track_length(config, tt))
    track_length = get_track_length(config, longest_tt)

    # Shade shared cue prefix with cue colors, diverging region with trial type colors
    cue_colors = pfmt.get_cue_colors(config)
    cue_widths = config.get('cue_map', {})
    sequences = {
        tt: config.get('trial_structures', {}).get(tt, {}).get('cue_sequence', [])
        for tt in trial_types
    }
    min_len = min(len(seq) for seq in sequences.values())
    shared_prefix_len = 0
    for i in range(min_len):
        cue_ids_at_i = {seq[i] for seq in sequences.values()}
        if len(cue_ids_at_i) == 1:
            shared_prefix_len = i + 1
        else:
            break

    # Shade shared cues
    ref_seq = sequences[trial_types[0]]
    pos = 0.0
    for i, cue_id in enumerate(ref_seq[:shared_prefix_len]):
        w = cue_widths[cue_id]
        ax.axvspan(pos, pos + w, alpha=0.12, color=cue_colors.get(cue_id, '#D3D3D3'), zorder=0)
        pos += w

    # Shade diverging region with each trial type's line color
    for tt in trial_types:
        seq = sequences[tt]
        tt_pos = sum(cue_widths[c] for c in seq[:shared_prefix_len])
        color = trial_type_colors.get(tt, '#999999')
        for cue_id in seq[shared_prefix_len:]:
            w = cue_widths[cue_id]
            ax.axvspan(tt_pos, tt_pos + w, alpha=0.08, color=color, zorder=0)
            tt_pos += w

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

    # Reward zones per trial type
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
    ax.set_xlim(0, track_length)
    pfmt.set_cue_boundary_ticks(ax, config, longest_tt)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


# ── SPEED RASTER ─────────────────────────────────────────────────────

def plot_speed_raster(
    df: pl.DataFrame,
    config: dict,
    trial_type: str,
    bin_size_cm: int | None = None,
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
    metadata: dict | None = None,
) -> tuple[Figure, Axes]:
    """Plots a trial-by-position heatmap of running speed.

    Each row is one trial, each column is a spatial bin, and color encodes the mean
    speed within that trial and bin. This is the continuous-valued analog of the lick
    raster: instead of binary dot presence, color intensity reveals the full speed
    landscape. Look for horizontal bands of low speed (cool colors) near the reward
    zone that strengthen across trials as the animal learns. Vertical streaks of
    uniform color across all trials indicate consistent behavior at particular track
    positions, such as always slowing at a cue boundary.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        trial_type: Trial type to plot.
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
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
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if ``bin_size_cm`` is not supplied.

    Returns:
        Tuple of (Figure, Axes).
    """
    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)
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
    pfmt.set_cue_boundary_ticks(ax, config, trial_type)
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
    animal_id: str,
    date: str,
    bin_size_cm: int | None = None,
    label: str | None = None,
    figsize_per_panel: tuple[float, float] = (5, 6),
    font_scale: float = 1.0,
    metadata: dict | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """Plots speed rasters for all trial types in a session, side by side.

    Creates one speed heatmap panel per trial type found in the data. Placing them
    adjacent makes it easy to compare speed modulation between rewarded and unrewarded
    trial types within the same session. A trained animal should show a clear cool
    band near the reward zone on rewarded trials that is absent or weaker on
    unrewarded trials.

    Args:
        df: Processed frame-level DataFrame.
        config: Experiment configuration dict.
        animal_id: Animal identifier for figure suptitle.
        date: Session date for figure suptitle.
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        label: Base title for the figure.
        figsize_per_panel: (width, height) per subplot panel.
        font_scale: Scale factor for all font sizes.
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if ``bin_size_cm`` is not supplied.
        **kwargs: Additional keyword arguments passed to plot_speed_raster.

    Returns:
        Tuple of (Figure, list of Axes).
    """
    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)
    trial_types = sorted(df['trial_type'].unique().to_list())
    n_panels = len(trial_types)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        squeeze=False,
        constrained_layout={'w_pad': 0.15, 'h_pad': 0.15},
    )
    axes = axes[0]

    for i, trial_type in enumerate(trial_types):
        panel_label = f"{label} — {trial_type}" if label else trial_type
        plot_speed_raster(
            df, config, trial_type=trial_type, bin_size_cm=bin_size_cm,
            ax=axes[i], label=panel_label,
            show_ylabel=(i == 0), font_scale=font_scale, **kwargs,
        )

    fig.suptitle(
        pfmt.build_title('Speed Raster', animal_id=animal_id, date=date),
        fontsize=13 * font_scale, fontweight='bold',
    )

    return fig, list(axes)


# ── SPEED LEARNING CURVES (MULTIDAY) ────────────────────────────────

def plot_multiday_speed_profile(
    sessions: dict[str, dict],
    animal_id: str,
    bin_size_cm: int | None = None,
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    figsize_per_panel: tuple[float, float] = (3.5, 3.0),
    font_scale: float = 1.0,
    show: bool = True,
    metadata: dict | None = None,
) -> Figure:
    """Plots speed profiles across sessions, one column per day, one row per trial type.

    Arranges mean +/- SEM speed-vs-position traces in a grid so that each column is a
    training day and each row is a trial type. All panels share the same y-axis scale
    for direct visual comparison. This is the primary plot for tracking the emergence
    of anticipatory slowing: early columns should show relatively flat speed profiles,
    while later columns should develop a clear dip near the reward zone. If the animal
    discriminates trial types, the dip should appear on rewarded rows but remain absent
    on unrewarded rows.

    Args:
        sessions: Dict from load_multiday_sessions().
        animal_id: Animal identifier for suptitle.
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        labels: Manual column labels. Overrides label_format.
        label_format: 'day_number' or 'date'.
        figsize_per_panel: (width, height) per panel.
        font_scale: Scale factor for all font sizes.
        show: Call plt.show().
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if ``bin_size_cm`` is not supplied.

    Returns:
        Matplotlib Figure.
    """
    if bin_size_cm is None:
        if metadata is None and sessions:
            metadata = next(iter(sessions.values())).get('metadata')
        bin_size_cm = get_bin_size(metadata)
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
        squeeze=False, constrained_layout={'w_pad': 0.3, 'h_pad': 0.4},
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
            pfmt.set_cue_boundary_ticks(ax, session_config, tt)
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
    """Computes per-trial reward metrics from frame-level data.

    Groups the DataFrame by trial and extracts three quantities: whether the animal
    collected a reward (any frame with reward == 'yes'), how much water was dispensed
    (max minus min of the cumulative water_uL column within the trial), and trial
    duration in seconds. These per-trial metrics feed into the reward and duration
    plots and are also aggregated by compute_session_reward_summary() for cross-session
    comparisons.

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
    """Computes session-level reward summary statistics per trial type.

    Aggregates per-trial reward metrics into session-wide hit rate, mean and total
    water consumed, and trial counts for each trial type. These summary numbers
    are the data behind the multiday reward bar charts and are also useful for
    quick programmatic checks (e.g. verifying that the animal reached criterion
    hit rate before advancing to the next training phase).

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
    animal_id: str,
    date: str,
    figsize: tuple[float, float] = (10, 8),
    show: bool = True,
) -> Figure:
    """Plots per-trial reward metrics across a single session in three stacked panels.

    The top panel shows a rolling hit rate (fraction of trials rewarded over a sliding
    window of 10 trials, with an expanding window for the first few trials so the trace
    starts at trial 1). This reveals whether the animal is consistently finding the
    reward zone or only succeeding sporadically. The middle panel scatters water
    dispensed per trial, useful for spotting hardware issues (zero-water trials) or
    unusual consumption patterns. The bottom panel plots cumulative water over trials,
    giving a quick read on total fluid intake for the session. All three panels are
    colored by trial type so differences in reward collection between trial types are
    immediately visible.

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

        # Expanding-then-rolling hit rate so the line starts at trial 1.
        # Uses an expanding window for the first few trials, then switches to a
        # fixed window once enough trials are available.
        window = min(10, len(rewarded))
        if len(rewarded) > 0:
            cumulative_sum = np.cumsum(rewarded)
            rolling_hit = np.empty_like(rewarded)
            for i in range(len(rewarded)):
                current_window = min(window, i + 1)
                if i < window:
                    rolling_hit[i] = cumulative_sum[i] / (i + 1)
                else:
                    rolling_hit[i] = (cumulative_sum[i] - cumulative_sum[i - window]) / window
            ax_hit.plot(trials, rolling_hit, color=color, linewidth=1.5, label=tt)

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
    animal_id: str,
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    figsize: tuple[float, float] = (10, 6),
    font_scale: float = 1.0,
    show: bool = True,
) -> Figure:
    """Plots reward hit rate and total water consumption across sessions.

    Two stacked bar chart panels summarize reward performance over training days.
    The top panel shows hit rate (fraction of trials where the animal collected a
    reward) per session, grouped by trial type. A rising trend toward 1.0 indicates
    the animal is learning to locate the reward zone. The bottom panel shows total
    water consumed per session, which should increase with hit rate and serves as a
    welfare check (animals that are not drinking enough may need supplemental water).
    Bars are grouped by trial type so that differential learning across rewarded vs
    unrewarded conditions is easy to assess.

    Args:
        sessions: Dict from load_multiday_sessions().
        animal_id: Animal identifier for suptitle.
        labels: Manual x-axis labels. Overrides label_format.
        label_format: 'day_number' or 'date'.
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

    water_bottom = np.zeros(n_sessions)

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

        water_arr = np.array(total_waters, dtype=float)
        water_arr = np.where(np.isnan(water_arr), 0.0, water_arr)
        ax_water.bar(x, water_arr, width=0.6, bottom=water_bottom, color=color,
                     alpha=0.7, label=tt, edgecolor='white', linewidth=0.5)
        water_bottom += water_arr

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
    animal_id: str,
    date: str,
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> Figure:
    """Plots trial duration across trials within a single session, colored by trial type.

    Scatter points show the duration of each trial with a smoothed rolling-median
    trend line overlaid. Trial duration reflects how long the animal takes to
    traverse the virtual track. A decreasing trend over trials suggests the animal
    is becoming more comfortable running, while unusually long trials may indicate
    the animal stopped or disengaged. Comparing trial types reveals whether the
    animal runs differently on rewarded vs unrewarded tracks (e.g. slower on
    rewarded trials due to anticipatory licking near the reward zone).

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
    animal_id: str,
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    figsize: tuple[float, float] = (10, 5),
    font_scale: float = 1.0,
    show: bool = True,
) -> Figure:
    """Plots mean trial duration per session across training days, grouped by trial type.

    Bar height is the mean duration across all trials in that session, with error bars
    showing +/- one standard deviation. Over the course of training, mean duration
    typically decreases as the animal learns the task and runs more confidently. Large
    standard deviations suggest high trial-to-trial variability, which can indicate
    intermittent disengagement. Comparing trial types reveals whether the animal
    consistently spends more time on certain track configurations, which may reflect
    differential familiarity or reward-seeking behavior.

    Args:
        sessions: Dict from load_multiday_sessions().
        animal_id: Animal identifier for suptitle.
        labels: Manual x-axis labels.
        label_format: 'day_number' or 'date'.
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


def plot_behavior_analysis(
    data: pl.DataFrame,
    exp_config: dict,
    animal_id: str,
    date: str,
    show: bool = True,
) -> list[Figure]:
    """Plots all single-day behavioral analyses for one session.

    Generates the full set of within-session behavioral figures: lick raster (spatial
    lick pattern per trial type), speed profile (mean speed vs position), speed raster
    (trial-by-position speed heatmap), reward metrics (hit rate, water per trial,
    cumulative water), and trial duration. Together these provide a comprehensive
    snapshot of the animal's behavioral state on a given day, covering spatial
    discrimination, locomotor strategy, reward collection, and engagement.

    Args:
        data: Processed frame-level DataFrame.
        exp_config: Experiment configuration dict.
        animal_id: Animal identifier for plot titles.
        date: Session date for plot titles.
        show: Determines whether to call plt.show() after each figure.

    Returns:
        List of generated matplotlib Figures.
    """
    figures = []

    fig_lick, _ = plot_session_lick_raster(data, exp_config, animal_id=animal_id, date=date)
    figures.append(fig_lick)

    fig_speed = plot_speed_profile(data, exp_config, animal_id=animal_id, date=date, show=False)
    figures.append(fig_speed)

    fig_speed_raster, _ = plot_session_speed_raster(
        data, exp_config, animal_id=animal_id, date=date,
    )
    figures.append(fig_speed_raster)

    fig_reward = plot_reward_metrics(data, exp_config, animal_id=animal_id, date=date, show=False)
    figures.append(fig_reward)

    fig_duration = plot_trial_duration(data, exp_config, animal_id=animal_id, date=date, show=False)
    figures.append(fig_duration)

    if show:
        plt.show()

    return figures


def plot_multiday_behavior_analysis(
    sessions: dict[str, dict],
    animal_id: str,
    labels: list[str] | None = None,
    label_format: str = 'day_number',
    font_scale: float = 1.0,
    show: bool = True,
) -> list[Figure]:
    """Plots all multiday behavioral analyses across sessions.

    Generates the full set of cross-session behavioral figures: multiday lick raster
    (lick spatial pattern evolution over days), multiday speed profile (emergence of
    anticipatory slowing), reward summary (hit rate and water intake trends), and
    trial duration (engagement over training). Together these track the arc of
    learning from naive to trained, making it easy to identify when the animal
    reached criterion, when a new trial type was introduced, or when performance
    plateaued or regressed.

    Args:
        sessions: Dict from load_multiday_sessions(), keyed by date string,
            each value containing 'data' and 'config' keys.
        animal_id: Animal identifier for plot titles.
        labels: Manual day labels. Overrides label_format if provided.
        label_format: 'day_number' for 'Day 1', 'Day 2', etc.
            'date' for the session date string (MM-DD).
        font_scale: Scale factor for all font sizes.
        show: Determines whether to call plt.show() after all figures.

    Returns:
        List of generated matplotlib Figures.
    """
    figures = []

    fig_lick = plot_multiday_lick_raster(
        sessions, animal_id=animal_id, labels=labels,
        label_format=label_format, font_scale=font_scale,
    )
    figures.append(fig_lick)

    fig_speed = plot_multiday_speed_profile(
        sessions, animal_id=animal_id, labels=labels,
        label_format=label_format, font_scale=font_scale, show=False,
    )
    figures.append(fig_speed)

    fig_reward = plot_multiday_reward_summary(
        sessions, animal_id=animal_id, labels=labels,
        label_format=label_format, font_scale=font_scale, show=False,
    )
    figures.append(fig_reward)

    fig_duration = plot_multiday_trial_duration(
        sessions, animal_id=animal_id, labels=labels,
        label_format=label_format, font_scale=font_scale, show=False,
    )
    figures.append(fig_duration)

    if show:
        plt.show()

    return figures


# HELPERS

def sessions_to_raster_input(
    sessions: dict[str, dict],
    label_format: str = 'day_number',
    day_offset: int = 0,
) -> list[tuple[pl.DataFrame, dict, str]]:
    """Converts load_multiday_sessions() output to plot_lick_raster_grid input format.

    Transforms the date-keyed sessions dictionary into a flat list of
    (DataFrame, config, label) tuples sorted chronologically. This adapter is
    needed because plot_lick_raster_grid takes an explicit ordered list (allowing
    the caller to control which sessions map to which grid slots), whereas the
    multiday loader returns a dictionary keyed by date string.

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
    date = '2025-09-16'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)

    data, meta = load_processed_session(paths['parquet'])

    # Single session (auto-splits trial types)
    figs = plot_behavior_analysis(data, exp_config, animal_id=mouse_id, date=date)

    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-08-30', '2025-09-11'), auto_process=False,
    )
    multiday_figs = plot_multiday_behavior_analysis(sessions, animal_id=mouse_id)

