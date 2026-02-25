"""
UMAP Plotting Module for Frame-Level Neural Data

Visualization functions for analyzing neural manifolds using UMAP.
Works with frame-level data (after process_session or fix_cue_offset) — NOT spatially binned trial data.

Core workflow:
    1. prepare_umap_data() - Frame df → neural array + metadata dict
    2. compute_umap() - Neural array → embedding
    3. plot_umap_*() - Embedding + metadata → visualization

Main plotting function:
    - plot_umap: Handles 1D (matplotlib), 2D (matplotlib), 3D (interactive Plotly).
        Supports all coloring strategies, trial-type toggles, dropdown to switch
        between multiple strategies, clickable legends, hover info.

Specialized plots (kept separate):
    - plot_umap_2d_density: KDE contours per trial type
    - plot_umap_3d_single_trial_trajectory: Lines connecting consecutive frames within a trial
    - plot_umap_3d_position_matched: Side-by-side ABC vs ABDC for shared position range
"""


import numpy as np
import polars as pl
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Literal, Any
from enum import Enum
from pathlib import Path

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import plot_utils as pfmt



# COLORING STRATEGY

class ColoringStrategy(Enum):
    """Defines different strategies for coloring UMAP points."""
    CUE = "cue"
    POSITION = "position"
    TRIAL_TYPE = "trial_type"
    SPEED = "speed"
    SESSION_PROGRESS = "session_progress"


# Shared Plotly axis settings (clean, no grid)
_PLOTLY_AXIS = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)


# DATA PREPARATION

def prepare_umap_data(
        df: pl.DataFrame,
        signal_column: str = "single_day_f",
        min_speed: float | None = 2.0,
        max_speed: float | None = None,
        cues_to_include: list[int] | None = None,
        cues_to_exclude: list[int] | None = None,
        trial_types_to_include: list[str] | None = None,
        state_filters: dict[str, Any] | None = None,
        max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Prepare frame-level data for UMAP.

    Extracts neural activity arrays and behavioral metadata from a processed
    frame-level DataFrame. Handles filtering, subsampling, and position
    normalization.

    Uses 'position' and 'nominal_track_length' columns if present (from
    process_session), otherwise computes them from 'distance_cm'.

    Args:
        df: Frame-level DataFrame (after process_session or fix_cue_offset).
        signal_column: Column containing neural activity arrays per frame.
        min_speed: Minimum speed threshold; filters out stationary periods.
        max_speed: Maximum speed threshold.
        cues_to_include: Only include frames in these cue zones.
        cues_to_exclude: Exclude frames in these cue zones (e.g., [0] to drop gray).
        trial_types_to_include: Filter to specific trial types.
        state_filters: {column_name: value} for additional filtering.
        max_frames: Subsample using stride if exceeded (preserves temporal ordering).

    Returns:
        neural_data: Array of shape (n_frames, n_cells).
        metadata: Dict with keys 'distance', 'cue', 'trial_type', 'speed',
            'trial', 'track_length'.
    """
    filtered = df

    if min_speed is not None:
        filtered = filtered.filter(pl.col('speed_cm_s') >= min_speed)
    if max_speed is not None:
        filtered = filtered.filter(pl.col('speed_cm_s') <= max_speed)
    if cues_to_include is not None:
        filtered = filtered.filter(pl.col('cue').is_in(cues_to_include))
    if cues_to_exclude is not None:
        filtered = filtered.filter(~pl.col('cue').is_in(cues_to_exclude))
    if trial_types_to_include is not None:
        filtered = filtered.filter(pl.col('trial_type').is_in(trial_types_to_include))
    if state_filters:
        for col, value in state_filters.items():
            if col in filtered.columns:
                filtered = filtered.filter(pl.col(col) == value)

    # Stride-based subsampling (preserves temporal structure)
    if max_frames is not None and len(filtered) > max_frames:
        stride = len(filtered) // max_frames
        filtered = filtered.gather_every(stride)

    neural_data = np.vstack(filtered[signal_column].to_list())

    # Use existing 'position' column if available (from process_session),
    # otherwise compute normalized position from distance_cm
    if 'position' in filtered.columns:
        position = filtered['position'].to_numpy()
    else:
        position = (
            filtered.with_columns(
                (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
                .alias('_norm_pos')
            )['_norm_pos'].to_numpy()
        )

    # Use existing 'nominal_track_length' if available, otherwise estimate
    if 'nominal_track_length' in filtered.columns:
        track_lengths = filtered['nominal_track_length'].to_numpy().astype(float)
    else:
        track_lengths = (
            filtered.with_columns(
                (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
                .max().over('trial').alias('_track_len')
            )['_track_len'].to_numpy()
        )

    metadata = {
        'distance': position,
        'cue': filtered['cue'].to_numpy(),
        'trial_type': filtered['trial_type'].to_numpy(),
        'speed': filtered['speed_cm_s'].to_numpy(),
        'trial': filtered['trial'].to_numpy(),
        'track_length': track_lengths,
    }

    n_filtered = len(df) - len(filtered)
    print(f"Prepared {neural_data.shape[0]} frames × {neural_data.shape[1]} cells")
    if n_filtered > 0:
        print(f"  Filtered out {n_filtered} frames")

    return neural_data, metadata


# UMAP COMPUTATION

def compute_umap(
        neural_data: np.ndarray,
        n_components: int = 3,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = 'correlation',
        random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP embedding. Check API for more param options.
        https://umap-learn.readthedocs.io/en/latest/api.html

    Args:
        neural_data: Array of shape (n_frames, n_cells).
        n_components: Embedding dimensions (1, 2, or 3). Though theoretically could go up to 100
        n_neighbors: UMAP nearest neighbors parameter.  50 was used in OSM paper
        min_dist: UMAP minimum distance between embedded points
        metric: Distance metric for UMAP. Best options: 'correlation', 'euclidean', 'cosine'.
        random_state: Random seed for reproducibility.

    Returns:
        Embedding array of shape (n_frames, n_components).
    """
    print(f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    # Fit X into an embedded space and return the transformed output
    embedding = reducer.fit_transform(neural_data)
    print(f"Done! Embedding shape: {embedding.shape}")
    return embedding


# MATPLOTLIB HELPERS

def get_colors_for_strategy(
        strategy: ColoringStrategy,
        metadata: dict[str, np.ndarray],
        cmap_name: str = 'viridis',
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate colors and colormap info for a coloring strategy.

    Args:
        strategy: How to color the points.
        metadata: Dict with arrays from prepare_umap_data.
        cmap_name: Colormap name for continuous variables (not used for all strategies).

    Returns:
        colors: Array of RGBA values or hex strings.
        color_info: Dict with 'type' ('categorical'/'continuous') and rendering info.
    """
    if strategy == ColoringStrategy.CUE:
        cue_color_map = pfmt.get_cue_colors()
        cue_label_map = pfmt.get_cue_labels()
        unique_cues = np.unique(metadata['cue'])
        colors = np.array([cue_color_map.get(int(c), '#D3D3D3') for c in metadata['cue']])
        legend = {}
        for cue in unique_cues:
            label = cue_label_map.get(int(cue), f'Cue {cue}')
            legend[label] = cue_color_map.get(int(cue), '#D3D3D3')
        color_info = {
            'type': 'categorical', 'legend': legend, 'label': 'Cue Zone',
            'raw_cue_map': {int(c): cue_label_map.get(int(c), f'Cue {c}') for c in unique_cues},
        }

    elif strategy == ColoringStrategy.POSITION:
        cmap = plt.cm.get_cmap('twilight')
        vmax = metadata['track_length'].max() if 'track_length' in metadata else metadata['distance'].max()
        norm = Normalize(vmin=0, vmax=vmax)
        colors = cmap(norm(metadata['distance']))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Position (cm)'}

    elif strategy == ColoringStrategy.TRIAL_TYPE:
        unique_types = np.unique(metadata['trial_type'])
        colors = np.array([pfmt.TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in metadata['trial_type']])
        legend = {tt: pfmt.TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in unique_types}
        color_info = {'type': 'categorical', 'legend': legend, 'label': 'Trial Type'}

    elif strategy == ColoringStrategy.SPEED:
        cmap = plt.cm.get_cmap('plasma')
        norm = Normalize(vmin=metadata['speed'].min(), vmax=metadata['speed'].max())
        colors = cmap(norm(metadata['speed']))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Speed (cm/s)'}

    elif strategy == ColoringStrategy.SESSION_PROGRESS:
        cmap = plt.cm.get_cmap('YlOrBr')
        norm = Normalize(vmin=metadata['trial'].min(), vmax=metadata['trial'].max())
        colors = cmap(norm(metadata['trial']))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Session Progress'}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return colors, color_info


def _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha=0.6, s=20):
    """Scatter for categorical strategies (1D or 2D)."""
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1
    for label, color in color_info['legend'].items():
        if strategy == ColoringStrategy.CUE:
            raw_ids = [k for k, v in color_info['raw_cue_map'].items() if v == label]
            mask = np.isin(metadata['cue'], raw_ids)
        else:
            mask = metadata['trial_type'] == label

        if n_dims == 1:
            y_vals = np.random.normal(0, 0.02, size=mask.sum())
            ax.scatter(embedding[mask], y_vals, c=color, label=label, alpha=alpha, s=s)
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=label, alpha=alpha, s=s)
    ax.legend(title=color_info['label'], bbox_to_anchor=(1.05, 1), loc='upper left')


def _plot_matplotlib(embedding, metadata, strategy, title, save_path, alpha, s):
    """Matplotlib path for 1D and 2D embeddings."""
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1
    colors, color_info = get_colors_for_strategy(strategy, metadata)

    if n_dims == 1:
        fig, ax = plt.subplots(figsize=(12, 2))
        y_vals = np.random.normal(0, 0.02, size=len(embedding))
        if color_info['type'] == 'categorical':
            _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
        else:
            ax.scatter(embedding, y_vals, c=colors, alpha=alpha, s=s)
            plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                         ax=ax).set_label(color_info['label'])
        ax.set_xlabel('UMAP 1')
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
    else:  # 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        if color_info['type'] == 'categorical':
            _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=alpha, s=s)
            plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                         ax=ax, shrink=0.8).set_label(color_info.get('label', ''))
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax.set_title(title or f'UMAP — {strategy.value}')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# PLOTLY HELPERS
def _check_plotly():
    """Raise ImportError if Plotly is not installed."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly required. Install with: pip install plotly")


def _show_and_save(fig: 'go.Figure', save_path: Path | None = None):
    """Save to html/image and open in browser.

    Args:
        fig: Plotly figure.
        save_path: If .html, writes interactive HTML; otherwise writes image.
    """
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == '.html':
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path))
    fig.show(renderer='browser')


# PLOTLY 3D TRACE BUILDERS
# Each builder returns a list of traces for ONE strategy view.
# The caller handles visibility toggling across views.

def _build_cue_traces(embedding, metadata, point_size, opacity):
    """Build per-cue, per-trial-type traces with clickable legend.

    Returns list of trace dicts and list of legend group names.
    """
    traces = []
    trial_types = sorted(np.unique(metadata['trial_type']))
    base_colors = pfmt.get_cue_colors()
    cue_labels = pfmt.get_cue_labels()

    for tt_idx, trial_type in enumerate(trial_types):
        tt_mask = metadata['trial_type'] == trial_type
        cues = metadata['cue'][tt_mask]
        positions = metadata['distance'][tt_mask]
        emb = embedding[tt_mask]

        # Shade cues lighter/darker per trial type for visual distinction
        factor = 1.3 - tt_idx * 0.3
        cue_colors = {cid: pfmt.scale_color(c, factor=factor)
                      for cid, c in base_colors.items()}

        suffix = f' ({trial_type})' if len(trial_types) > 1 else ''

        for cue_id in sorted(set(cues)):
            cue_mask = cues == cue_id
            if not cue_mask.any():
                continue
            cue_int = int(cue_id)
            label = cue_labels.get(cue_int, f'Cue {cue_int}')
            color = cue_colors.get(cue_int, '#AAAAAA')

            traces.append(go.Scatter3d(
                x=emb[cue_mask, 0], y=emb[cue_mask, 1], z=emb[cue_mask, 2],
                mode='markers', name=f'{label}{suffix}',
                marker=dict(size=point_size, opacity=opacity, color=color),
                showlegend=(cue_int != 0),  # hide gray zone from legend
                customdata=np.column_stack([
                    positions[cue_mask],
                    metadata['trial'][tt_mask][cue_mask],
                ]),
                hovertemplate=(
                    f'{trial_type}<br>'
                    'Pos: %{customdata[0]:.1f} cm<br>'
                    f'Cue: {label}<br>'
                    'Trial: %{customdata[1]:.0f}'
                    '<extra></extra>'
                ),
            ))
    return traces


def _build_position_traces(embedding, metadata, point_size, opacity):
    """Build per-trial-type traces colored by track position (separate colorscales).

    Returns list of traces.
    """
    traces = []
    trial_types = sorted(np.unique(metadata['trial_type']))
    cue_labels = pfmt.get_cue_labels()

    for i, trial_type in enumerate(trial_types):
        tt_mask = metadata['trial_type'] == trial_type
        positions = metadata['distance'][tt_mask]
        cues = metadata['cue'][tt_mask]
        track_len = metadata['track_length'][tt_mask].max()
        colorscale = pfmt.trial_type_colorscale(trial_type)

        # Stagger colorbars so they don't overlap
        cb_x = 1.0 + i * 0.15
        cb_y = 0.75 - i * 0.45

        traces.append(go.Scatter3d(
            x=embedding[tt_mask, 0], y=embedding[tt_mask, 1], z=embedding[tt_mask, 2],
            mode='markers', name=f'{trial_type}',
            marker=dict(
                size=point_size, opacity=opacity, color=positions,
                colorscale=colorscale, cmin=0, cmax=track_len,
                colorbar=dict(title=f'{trial_type} pos (cm)', len=0.4,
                              x=cb_x, y=cb_y),
            ),
            text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
            customdata=metadata['trial'][tt_mask],
            hovertemplate=(
                f'{trial_type}<br>'
                'Pos: %{marker.color:.1f} cm<br>'
                'Cue: %{text}<br>'
                'Trial: %{customdata:.0f}'
                '<extra></extra>'
            ),
        ))
    return traces


def _build_trial_type_traces(embedding, metadata, point_size, opacity):
    """Build per-trial-type traces with flat color per type. Clickable legend."""
    traces = []
    cue_labels = pfmt.get_cue_labels()

    for trial_type in sorted(np.unique(metadata['trial_type'])):
        tt_mask = metadata['trial_type'] == trial_type
        color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#D3D3D3')
        positions = metadata['distance'][tt_mask]
        cues = metadata['cue'][tt_mask]

        traces.append(go.Scatter3d(
            x=embedding[tt_mask, 0], y=embedding[tt_mask, 1], z=embedding[tt_mask, 2],
            mode='markers', name=str(trial_type),
            marker=dict(size=point_size, opacity=opacity, color=color),
            customdata=np.column_stack([positions, metadata['trial'][tt_mask]]),
            text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
            hovertemplate=(
                f'{trial_type}<br>'
                'Pos: %{customdata[0]:.1f} cm<br>'
                'Cue: %{text}<br>'
                'Trial: %{customdata[1]:.0f}'
                '<extra></extra>'
            ),
        ))
    return traces


def _build_continuous_traces(embedding, metadata, strategy, point_size, opacity):
    """Build a single trace with continuous colorscale (speed, session progress)."""
    if strategy == ColoringStrategy.SPEED:
        values, cscale, label = metadata['speed'], 'Plasma', 'Speed (cm/s)'
        cmin, cmax = values.min(), values.max()
    elif strategy == ColoringStrategy.SESSION_PROGRESS:
        values, cscale, label = metadata['trial'], 'YlOrBr', 'Session Progress'
        cmin, cmax = values.min(), values.max()
    else:
        raise ValueError(f"No continuous builder for {strategy}")

    cue_labels = pfmt.get_cue_labels()
    cues = metadata['cue']
    positions = metadata['distance']

    trace = go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode='markers', showlegend=False, name=label,
        marker=dict(
            size=point_size, opacity=opacity, color=values,
            colorscale=cscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=label),
        ),
        customdata=np.column_stack([positions, metadata['trial']]),
        text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
        hovertemplate=(
            'Pos: %{customdata[0]:.1f} cm<br>'
            'Cue: %{text}<br>'
            'Trial: %{customdata[1]:.0f}<br>'
            f'{label}: %{{marker.color:.1f}}'
            '<extra></extra>'
        ),
    )
    return [trace]


def _get_trace_builder(strategy: ColoringStrategy):
    """Return the appropriate trace builder for a strategy."""
    if strategy == ColoringStrategy.CUE:
        return _build_cue_traces
    elif strategy == ColoringStrategy.POSITION:
        return _build_position_traces
    elif strategy == ColoringStrategy.TRIAL_TYPE:
        return _build_trial_type_traces
    elif strategy in (ColoringStrategy.SPEED, ColoringStrategy.SESSION_PROGRESS):
        return lambda emb, meta, ps, op: _build_continuous_traces(emb, meta, strategy, ps, op)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------------------------
# MAIN PLOTTING FUNCTION
def plot_umap(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        strategy: str | ColoringStrategy | list[str | ColoringStrategy] = 'cue',
        point_size: int | float = 2,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | str | None = None,
        alpha: float = 0.6,
        s: float = 20,
):
    """Unified UMAP visualization.

    Handles 1D, 2D (matplotlib) and 3D (interactive Plotly) embeddings.

    For 3D plots:
        - Single strategy → one coloring mode with clickable legend / colorbar.
        - List of strategies → dropdown menu to switch between coloring modes.
        - All 3D plots include hover info (position, cue, trial, speed).
        - Categorical strategies (cue, trial_type) get clickable legend entries.
        - The 'position' strategy shows per-trial-type colorscales.

    Args:
        embedding: UMAP embedding of shape (n_frames, n_components).
        metadata: Dict with arrays from prepare_umap_data.
        strategy: Coloring strategy or list of strategies for toggle dropdown.
            Valid values: 'cue', 'position', 'trial_type', 'speed',
            'session_progress', or their ColoringStrategy equivalents.
            Pass a list (e.g. ['cue', 'position']) for a dropdown toggle.
        point_size: Marker size (Plotly for 3D, ignored for matplotlib — use `s`).
        opacity: Marker opacity (Plotly for 3D, use `alpha` for matplotlib).
        title: Plot title. Auto-generated if None.
        save_path: Save as .html (interactive) or image format. None to skip.
        alpha: Matplotlib scatter alpha (1D/2D only).
        s: Matplotlib scatter point size (1D/2D only).

    Returns:
        tuple of (figure, plot_info) where:
            - figure: plt.Figure (1D/2D) or go.Figure (3D)
            - plot_info: dict with 'n_components', 'strategy', 'n_frames',
              'n_trial_types', 'trial_types'
    """
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1

    # Normalize strategy input
    if not isinstance(strategy, list):
        strategy = [strategy]
    strategies = [ColoringStrategy(s) if isinstance(s, str) else s for s in strategy]

    # Build plot_info return dict
    trial_types = sorted(np.unique(metadata['trial_type']).tolist())
    plot_info = {
        'n_components': n_dims,
        'strategy': [s_.value for s_ in strategies],
        'n_frames': len(embedding),
        'n_trial_types': len(trial_types),
        'trial_types': trial_types,
    }

    # ── 1D / 2D: matplotlib ──────────────────────────────────────────────
    if n_dims <= 2:
        if len(strategies) > 1:
            print("Warning: Multiple strategies only supported for 3D. Using first strategy.")
        fig = _plot_matplotlib(embedding, metadata, strategies[0], title, save_path, alpha, s)
        return fig, plot_info

    # ── 3D: Plotly ───────────────────────────────────────────────────────
    _check_plotly()
    fig = go.Figure()

    if len(strategies) == 1:
        # Single strategy — just add traces directly
        builder = _get_trace_builder(strategies[0])
        traces = builder(embedding, metadata, point_size, opacity)
        for t in traces:
            fig.add_trace(t)

        auto_title = f'3D UMAP — {strategies[0].value}'

    else:
        # Multiple strategies — build all trace groups, wire up dropdown
        trace_groups = []  # list of (strategy, traces)
        for strat in strategies:
            builder = _get_trace_builder(strat)
            traces = builder(embedding, metadata, point_size, opacity)
            trace_groups.append((strat, traces))

        # Add all traces, only first group visible
        for group_idx, (strat, traces) in enumerate(trace_groups):
            for t in traces:
                t.visible = (group_idx == 0)
                fig.add_trace(t)

        # Build visibility arrays for each dropdown button
        buttons = []
        offset = 0
        group_sizes = [len(traces) for _, traces in trace_groups]
        total_traces = sum(group_sizes)

        for group_idx, (strat, traces) in enumerate(trace_groups):
            vis = [False] * total_traces
            start = sum(group_sizes[:group_idx])
            for j in range(group_sizes[group_idx]):
                vis[start + j] = True
            buttons.append(dict(
                label=f'Color by {strat.value.replace("_", " ").title()}',
                method='update',
                args=[{'visible': vis}],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type='dropdown', direction='down',
                x=0.0, y=1.15, xanchor='left', yanchor='top',
                buttons=buttons,
            )],
        )

        labels = [s_.value for s_ in strategies]
        auto_title = f'3D UMAP — toggle: {" / ".join(labels)}'

    fig.update_layout(
        title=title or auto_title,
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant', groupclick='toggleitem'),
    )
    _show_and_save(fig, save_path)
    return fig, plot_info

# SPECLIAIZED PLOTS ---------------------

# 2D PLOT WITH KDE OVERLAY
#TODO impose cues on plots with legend so we know what we're actually looking at
def plot_umap_2d_density(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        figsize: tuple[float, float] = (14, 6),
        n_levels: int = 6,
        bandwidth: float | None = None,
        alpha_scatter: float = 0.15,
        alpha_contour: float = 0.8,
        s: float = 5,
        save_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D UMAP with KDE density contours per trial type.

    Shows where the manifold is concentrated for each trial type. Useful for
    seeing whether trial types occupy overlapping or distinct regions.

    Args:
        embedding: 2D UMAP embedding array.
        metadata: Dict with metadata arrays from prepare_umap_data.
        figsize: Figure size.
        n_levels: Number of contour levels.
        bandwidth: KDE bandwidth (None = auto via Scott's rule).
        alpha_scatter: Transparency for scatter points.
        alpha_contour: Transparency for contour lines.
        s: Scatter point size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    from scipy.stats import gaussian_kde

    unique_types = np.unique(metadata['trial_type'])
    fig, axes = plt.subplots(1, len(unique_types) + 1, figsize=figsize)

    # Per-trial-type panels
    for ax, trial_type in zip(axes[:-1], unique_types):
        mask = metadata['trial_type'] == trial_type
        color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#999999')
        emb = embedding[mask]

        ax.scatter(emb[:, 0], emb[:, 1], c=color, alpha=alpha_scatter, s=s)

        # KDE contours
        if emb.shape[0] > 50:
            kde = gaussian_kde(emb.T, bw_method=bandwidth)
            x_grid = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 100)
            y_grid = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 100)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=n_levels, colors=[color], alpha=alpha_contour,
                       linewidths=1.5)

        ax.set_title(f'{trial_type}', fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Overlay panel
    ax_overlay = axes[-1]
    for trial_type in unique_types:
        mask = metadata['trial_type'] == trial_type
        color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#999999')
        emb = embedding[mask]

        ax_overlay.scatter(emb[:, 0], emb[:, 1], c=color, alpha=alpha_scatter * 0.5, s=s,
                           label=trial_type)
        if emb.shape[0] > 50:
            kde = gaussian_kde(emb.T, bw_method=bandwidth)
            x_grid = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 100)
            y_grid = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 100)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax_overlay.contour(xx, yy, zz, levels=n_levels, colors=[color],
                               alpha=alpha_contour, linewidths=1.5)

    ax_overlay.set_title('Overlay', fontweight='bold')
    ax_overlay.set_xlabel('UMAP 1')
    ax_overlay.set_ylabel('UMAP 2')
    ax_overlay.set_aspect('equal')
    ax_overlay.spines['top'].set_visible(False)
    ax_overlay.spines['right'].set_visible(False)
    ax_overlay.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# SINGLE-TRIAL TRAJECTORY OVERLAY UMAP
def plot_umap_3d_single_trial_trajectory(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        trial_ids: list[int] | None = None,
        n_trials_per_type: int = 5,
        trial_selection: Literal['middle', 'spaced'] = 'spaced',
        point_size: int = 3,
        line_width: float = 2.5,
        opacity: float = 0.7,
        show_background: bool = True,
        background_opacity: float = 0.3,
        save_path: Path | None = None,
) -> 'go.Figure':
    """3D UMAP with lines connecting consecutive frames within individual trials.

    Reveals the ring-like topology expected for a looping track — the neural
    state should trace a loop through the manifold on each trial. Useful for
    checking whether the UMAP embedding preserves the spatial structure.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        trial_ids: Specific trial numbers to plot. If None, auto-selects.
        n_trials_per_type: Trials per type if trial_ids is None.
        trial_selection: How to pick trials when trial_ids is None.
            'middle' — consecutive trials from the middle of the session.
            'spaced' — evenly spaced across the full session.
        point_size: Marker size for trajectory points.
        line_width: Width of connecting lines.
        opacity: Trajectory opacity.
        show_background: Show all other points as faint background.
        background_opacity: Opacity for background points.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()

    unique_types = np.unique(metadata['trial_type'])

    # Auto-select trials: pick n from middle of session for each type
    if trial_ids is None:
        trial_ids = []
        for trial_type in unique_types:
            type_mask = metadata['trial_type'] == trial_type
            type_trials = np.unique(metadata['trial'][type_mask])
            n_select = min(n_trials_per_type, len(type_trials))

            if trial_selection == 'middle':
                mid = len(type_trials) // 2
                start = max(0, mid - n_select // 2)
                selected = type_trials[start:start + n_select]
            else:  # 'spaced'
                indices = np.linspace(0, len(type_trials) - 1, n_select, dtype=int)
                selected = type_trials[indices]

            trial_ids.extend(selected.tolist())

    fig = go.Figure()

#TODO fix background coloration for cues (rn only D showing up)
    # Background: all points faintly colored by cue for spatial context
    if show_background:
        cue_colors_map = pfmt.get_cue_colors()
        cue_labels = pfmt.get_cue_labels()
        unique_cues = np.unique(metadata['cue'])
        for cue_id in unique_cues:
            cue_mask = metadata['cue'] == cue_id
            cue_int = int(cue_id)
            label = cue_labels.get(cue_int, f'Cue {cue_int}')
            color = cue_colors_map.get(cue_int, '#CCCCCC')
    fig.add_trace(go.Scatter3d(
        x=embedding[cue_mask, 0], y=embedding[cue_mask, 1], z=embedding[cue_mask, 2],
        mode='markers', name=f'{label} (bg)',
        marker=dict(size=2, opacity=background_opacity, color=color),
        legendgroup='background',
        legendgrouptitle_text='Cue zones',
        showlegend=True,
        hoverinfo='skip',
    ))

    # Sort trial_ids so color gradient matches temporal order
    trial_ids = sorted(trial_ids)

    # Sequential colorscale for early → late trials
    n_trials = len(trial_ids)
    if n_trials <= 1:
        trial_colors = ['#FF6B6B']
    else:
        cmap = plt.cm.get_cmap('cool', n_trials)
        trial_colors = [
            f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}'
            for c in [cmap(i / (n_trials - 1)) for i in range(n_trials)]
        ]

    # Plot each selected trial as a connected line
    for i, trial_id in enumerate(trial_ids):
        trial_mask = metadata['trial'] == trial_id
        if not trial_mask.any():
            continue

        trial_emb = embedding[trial_mask]
        trial_pos = metadata['distance'][trial_mask]
        trial_cues = metadata['cue'][trial_mask]
        trial_type = metadata['trial_type'][trial_mask][0]
        color = trial_colors[i]

        # Line connecting consecutive frames
        fig.add_trace(go.Scatter3d(
            x=trial_emb[:, 0], y=trial_emb[:, 1], z=trial_emb[:, 2],
            mode='lines+markers',
            name=f'Trial {trial_id} ({trial_type})',
            line=dict(color=color, width=line_width),
            marker=dict(size=point_size, color=color, opacity=opacity),
            legendgroup='trajectories',
            legendgrouptitle_text='Trials (early→late)',
            customdata=np.column_stack([trial_pos, trial_cues]),
            hovertemplate=(
                f'Trial {trial_id} ({trial_type})<br>'
                'Pos: %{customdata[0]:.1f} cm<br>'
                'Cue: %{text}<extra></extra>'
            ),
            text=[pfmt.get_cue_labels().get(int(c), f'Cue {c}') for c in trial_cues],
        ))

        # Mark trial start with a larger marker
        fig.add_trace(go.Scatter3d(
            x=[trial_emb[0, 0]], y=[trial_emb[0, 1]], z=[trial_emb[0, 2]],
            mode='markers', showlegend=False,
            marker=dict(size=point_size + 4, color=color, symbol='diamond',
                        opacity=1.0, line=dict(color='black', width=1)),
            hovertemplate=f'Trial {trial_id} START<extra></extra>',
        ))

    fig.update_layout(
        title='UMAP: Single-Trial Trajectories (◆ = start, cool→warm = early→late)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant', groupclick='toggleitem'),
    )
    _show_and_save(fig, save_path)
    return fig


# POSITION-MATCHED COMPARISON
#TODO increase onshared dot size, also consider what the point of this is
def plot_umap_3d_position_matched(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        max_position: float | None = None,
        point_size: int = 2,
        opacity: float = 0.7,
        save_path: Path | None = None,
) -> 'go.Figure':
    """Side-by-side trial types for shared position range only.

    Clips longer trials to the shorter track length so you can directly compare
    where the manifolds overlap vs diverge for the same spatial positions.
    Frames beyond max_position are shown as faint gray for context.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        max_position: Upper position cutoff. Defaults to ABC track length.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()

    # Auto-detect cutoff from shortest trial type
    if max_position is None:
        trial_types = sorted(np.unique(metadata['trial_type']))
        max_lengths = {tt: metadata['track_length'][metadata['trial_type'] == tt].max()
                       for tt in trial_types}
        max_position = min(max_lengths.values())

    fig = go.Figure()

    # Shared colorscale for matched positions
    trial_types = sorted(np.unique(metadata['trial_type']))

    for i, trial_type in enumerate(trial_types):
        type_mask = metadata['trial_type'] == trial_type
        positions = metadata['distance'][type_mask]
        emb = embedding[type_mask]

        # In-range points (shared position)
        in_range = positions <= max_position
        if in_range.any():
            fig.add_trace(go.Scatter3d(
                x=emb[in_range, 0], y=emb[in_range, 1], z=emb[in_range, 2],
                mode='markers', name=f'{trial_type} (0–{max_position:.0f} cm)',
                marker=dict(
                    size=point_size, opacity=opacity,
                    color=positions[in_range],
                    colorscale=pfmt.trial_type_colorscale(trial_type),
                    cmin=0, cmax=max_position,
                    colorbar=dict(
                        title=f'{trial_type} pos (cm)', len=0.4,
                        x=1.0 + i * 0.15,
                        y=0.75 - i * 0.5 if i < 2 else 0.5,
                    ),
                ),
                hovertemplate=f'{trial_type}<br>Pos: %{{marker.color:.1f}} cm<extra></extra>',
            ))

        # Out-of-range (beyond shared region) — faint gray
        out_of_range = positions > max_position
        if out_of_range.any():
            fig.add_trace(go.Scatter3d(
                x=emb[out_of_range, 0], y=emb[out_of_range, 1], z=emb[out_of_range, 2],
                mode='markers', name=f'{trial_type} (>{max_position:.0f} cm)',
                marker=dict(size=point_size - 1, opacity=0.15, color='#AAAAAA'),
                hovertemplate=f'{trial_type} (beyond shared)<br>Pos: %{{customdata:.1f}} cm<extra></extra>',
                customdata=positions[out_of_range],
            ))

    fig.update_layout(
        title=f'UMAP: Position-Matched Comparison (0–{max_position:.0f} cm)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=0, y=1),
    )
    _show_and_save(fig, save_path)
    return fig

###############
#Deleted the natural separation function bc it didn't tell much, btu might be useful again if I do the merging project


