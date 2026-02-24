"""
UMAP Plotting Module for Frame-Level Neural Data

Visualization functions for analyzing neural manifolds using UMAP.
Works with frame-level data (after process_session or fix_cue_offset) — NOT spatially binned trial data.

Core workflow:
    1. prepare_umap_data() - Frame df → neural array + metadata dict
    2. compute_umap() - Neural array → embedding
    3. plot_umap_*() - Embedding + metadata → visualization

Interactive Plotly functions (3D):
    - plot_umap_3d_interactive: Single coloring strategy
    - plot_umap_3d_overlay_trial_types: ABC=blue, ABDC=red position gradients
    - plot_umap_3d_overlay_by_cue: Per-cue clickable legend
    - plot_umap_3d_with_toggle: Dropdown to switch position ↔ cue
    - plot_umap_3d_natural_separation: Light/dark cue colors by trial type
    - compare_trial_types_umap_3d: One-liner convenience function

    - plot_umap_3d_single_trial_trajectory: Lines connecting consecutive frames within a trial
    - plot_umap_3d_position_matched: Side-by-side ABC vs ABDC for shared position range

Static matplotlib (1D/2D):
    - plot_umap_1d, plot_umap_2d
    - plot_umap_2d_density: KDE contours per trial type
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

import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')
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
        signal_column: str = "single_day_dff",
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
        metric: str = 'euclidean',
        random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP embedding.

    Args:
        neural_data: Array of shape (n_frames, n_cells).
        n_components: Embedding dimensions (1, 2, or 3).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: Distance metric for UMAP.
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
    embedding = reducer.fit_transform(neural_data)
    print(f"Done! Embedding shape: {embedding.shape}")
    return embedding


# COLOR HELPERS (for matplotlib plots)

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


# MATPLOTLIB PLOTS (1D, 2D — static)

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


def plot_umap_1d(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        cmap_name: str = 'viridis',
        figsize: tuple[float, float] = (12, 2),
        alpha: float = 0.6,
        s: float = 20,
) -> plt.Figure:
    """Plot 1D UMAP embedding. Tbh interesting view of the session

    Args:
        embedding: 1D UMAP embedding array.
        metadata: Dict with metadata arrays from prepare_umap_data.
        strategy: Coloring strategy.
        cmap_name: Colormap name for continuous variables.
        figsize: Figure size.
        alpha: Point transparency.
        s: Point size.

    Returns:
        Matplotlib figure.
    """
    colors, color_info = get_colors_for_strategy(strategy, metadata, cmap_name)
    fig, ax = plt.subplots(figsize=figsize)

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
    plt.tight_layout()
    return fig


def plot_umap_2d(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        cmap_name: str = 'viridis',
        figsize: tuple[float, float] = (10, 8),
        alpha: float = 0.6,
        s: float = 20,
        title: str | None = None,
        save_path: Path | None = None,
) -> plt.Figure:
    """Plot 2D UMAP with coloring strategy.

    Args:
        embedding: 2D UMAP embedding array.
        metadata: Dict with metadata arrays from prepare_umap_data.
        strategy: Coloring strategy.
        cmap_name: Colormap name for continuous variables.
        figsize: Figure size.
        alpha: Point transparency.
        s: Point size.
        title: Optional plot title.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    colors, color_info = get_colors_for_strategy(strategy, metadata, cmap_name)
    fig, ax = plt.subplots(figsize=figsize)

    if color_info['type'] == 'categorical':
        _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=alpha, s=s)
        plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                     ax=ax, shrink=0.8).set_label(color_info.get('label', ''))

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title or f'UMAP — {strategy.value}')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

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


# PLOTLY 3D PLOTS (interactive)

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


def plot_umap_3d_interactive(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        point_size: int = 2,
        opacity: float = 0.8,
        title: str | None = None,
        save_path: Path | None = None,
) -> 'go.Figure':
    """Interactive 3D UMAP with Plotly — single coloring strategy (position, trial, etc)
    Great basic plot for exploration

    Clickable legend for categorical, colorbar for continuous.

    Args:
        embedding: 3D UMAP embedding array.
        metadata: Dict with metadata arrays from prepare_umap_data.
        strategy: Coloring strategy.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        title: Plot title.
        save_path: Save as .html or image.

    Returns:
        Plotly figure.
    """
    _check_plotly()
    fig = go.Figure()

    if strategy == ColoringStrategy.CUE:
        unique_cues = np.unique(metadata['cue'])
        cue_colors = pfmt.get_cue_colors()
        for cue in unique_cues:
            mask = metadata['cue'] == cue
            label = pfmt.get_cue_labels().get(int(cue), f'Cue {cue}')
            color = cue_colors.get(int(cue), '#CCCCCC')
            fig.add_trace(go.Scatter3d(
                x=embedding[mask, 0], y=embedding[mask, 1], z=embedding[mask, 2],
                mode='markers', name=label,
                marker=dict(size=point_size, opacity=opacity, color=color),
                hovertemplate=f'{label}<br>Pos: %{{customdata:.1f}} cm<extra></extra>',
                customdata=metadata['distance'][mask],
            ))

    elif strategy == ColoringStrategy.TRIAL_TYPE:
        for trial_type in np.unique(metadata['trial_type']):
            mask = metadata['trial_type'] == trial_type
            color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#D3D3D3')
            fig.add_trace(go.Scatter3d(
                x=embedding[mask, 0], y=embedding[mask, 1], z=embedding[mask, 2],
                mode='markers', name=str(trial_type),
                marker=dict(size=point_size, opacity=opacity, color=color),
            ))

    else:
        # Continuous strategies
        if strategy == ColoringStrategy.POSITION:
            values, cscale, label = metadata['distance'], 'Twilight', 'Position (cm)'
            cmin, cmax = 0, metadata['track_length'].max()
        elif strategy == ColoringStrategy.SPEED:
            values, cscale, label = metadata['speed'], 'Plasma', 'Speed (cm/s)'
            cmin, cmax = values.min(), values.max()
        elif strategy == ColoringStrategy.SESSION_PROGRESS:
            values, cscale, label = metadata['trial'], 'YlOrBr', 'Session Progress'
            cmin, cmax = values.min(), values.max()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode='markers', showlegend=False,
            marker=dict(
                size=point_size, opacity=opacity, color=values,
                colorscale=cscale, cmin=cmin, cmax=cmax,
                colorbar=dict(title=label),
            ),
        ))

    fig.update_layout(
        title=title or f'3D UMAP — {strategy.value}',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant'),
    )
    _show_and_save(fig, save_path)
    return fig


def plot_umap_3d_overlay_trial_types(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        point_size: int = 2,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | None = None,
) -> 'go.Figure':
    """3D UMAP with trial types colored by track position on separate color scale

    Dual colorbars show position within each trial type.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        title: Plot title.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()
    fig = go.Figure()

    trial_types = sorted(np.unique(metadata['trial_type']))
    for i, trial_type in enumerate(trial_types):
        mask = metadata['trial_type'] == trial_type
        positions = metadata['distance'][mask]
        track_len = metadata['track_length'][mask].max()

        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0], y=embedding[mask, 1], z=embedding[mask, 2],
            mode='markers', name=f'{trial_type} trials',
            marker=dict(
                size=point_size, opacity=opacity, color=positions,
                colorscale=pfmt.trial_type_colorscale(trial_type), cmin=0, cmax=track_len,
                colorbar=dict(
                    title=f'{trial_type} (cm)', len=0.4,
                    x=1.0 + i * 0.15,
                    y=0.75 - i * 0.5 if i < 2 else 0.5,
                ),
            ),
            hovertemplate=f'{trial_type}<br>Position: %{{marker.color:.1f}} cm<extra></extra>',
        ))

    fig.update_layout(
        title=title or f'UMAP: Position by Trial Type ({", ".join(trial_types)})',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=0, y=1),
    )
    _show_and_save(fig, save_path)
    return fig


def plot_umap_3d_overlay_by_cue(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        point_size: int = 2,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | None = None,
) -> 'go.Figure':
    """3D UMAP colored by cue, split by trial type (light=ABC, dark=ABDC).

    Click legend to isolate individual cues.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        title: Plot title.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()
    fig = go.Figure()

    trial_types = sorted(np.unique(metadata['trial_type']))
    for trial_type in trial_types:
        mask = metadata['trial_type'] == trial_type
        cues = metadata['cue'][mask]
        positions = metadata['distance'][mask]
        emb = embedding[mask]
        type = metadata['trial_type'][mask]

        base_colors = pfmt.get_cue_colors()
        tt_idx = trial_types.index(trial_type)
        factor = 1.3 - tt_idx * 0.3  # 1.3, 1.0, 0.7, ... for successive types esp if more than 1
        cue_colors = {cid: pfmt.scale_color(c, factor=factor) for cid, c in base_colors.items()}
        suffix = f' ({trial_type})'

        for cue_id in sorted(set(cues)):
            cue_mask = cues == cue_id
            if not cue_mask.any():
                continue
            cue_int = int(cue_id)
            label = pfmt.get_cue_labels.get(cue_int, f'Cue {cue_int}')
            color = pfmt.cue_colors.get(cue_int, '#D3D3D3')
#TODO this fallback might be an issue bc it's the same as the gray zone; need to rethink

            fig.add_trace(go.Scatter3d(
                x=emb[cue_mask, 0], y=emb[cue_mask, 1], z=emb[cue_mask, 2],
                mode='markers', name=f'{label}{suffix}',
                marker=dict(size=point_size, opacity=opacity, color=color),
                showlegend=bool(cue_int != 0),
                customdata=positions[cue_mask],
                hovertemplate=f'{trial_type}<br>Pos: %{{customdata:.1f}} cm<br>Cue: {label}<extra></extra>',
            ))
#TODO need to adjust for additional trial types
    fig.update_layout(
        title=title or f'UMAP: Cue Overlay ({", ".join(trial_types)})',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, font=dict(size=10), itemsizing='constant'),
    )
    _show_and_save(fig, save_path)
    return fig


def plot_umap_3d_natural_separation(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        point_size: int = 2,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | None = None,
) -> 'go.Figure':
    """Natural manifold positions — see if trial types separate in UMAP space.

        Uses lightness-scaled cue colors to visually distinguish trial types.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        title: Plot title.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()

    fig = go.Figure()
    trial_types = sorted(np.unique(metadata['trial_type']))

    for trial_type in trial_types:
        mask = metadata['trial_type'] == trial_type
        cues = metadata['cue'][mask]
        positions = metadata['distance'][mask]
        emb = embedding[mask]

        base_colors = pfmt.get_cue_colors()
        tt_idx = trial_types.index(trial_type)
        factor = 1.3 - tt_idx * 0.3
        cue_colors = {cid: pfmt.scale_color(c, lightness=factor) for cid, c in base_colors.items()}
        suffix = f' ({trial_type})'

        for cue_id in sorted(set(cues)):
            cue_mask = cues == cue_id
            if not cue_mask.any():
                continue
            cue_int = int(cue_id)
            label = pfmt.get_cue_labels.get(cue_int, f'Cue {cue_int}')
            color = cue_colors.get(cue_int, '#CCCCCC')
        #TODO is this right? ^

            fig.add_trace(go.Scatter3d(
                x=emb[cue_mask, 0], y=emb[cue_mask, 1], z=emb[cue_mask, 2],
                mode='markers', name=f'{label}{suffix}',
                marker=dict(size=point_size, opacity=opacity, color=color),
                showlegend=bool(cue_int != 0),
                customdata=positions[cue_mask],
                hovertemplate=f'{trial_type}<br>Cue: {label}<br>Pos: %{{customdata:.1f}} cm<extra></extra>',
            ))

    fig.update_layout(
        title=title or 'UMAP: Natural Manifold Positions (click legend to isolate)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant', font=dict(size=10)),
    )
    _show_and_save(fig, save_path)
    return fig


def plot_umap_3d_with_toggle(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        point_size: int = 2,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | None = None,
) -> 'go.Figure':
    """3D UMAP with dropdown to toggle between POSITION and CUE coloring.

    Position mode: Each trial type gets its own colorscale.
    Cue mode: Per-cue per-trial-type traces with clickable legend.

    Args:
        embedding: 3D UMAP embedding.
        metadata: Metadata dict from prepare_umap_data.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        title: Plot title.
        save_path: Save path.

    Returns:
        Plotly figure.
    """
    _check_plotly()
    fig = go.Figure()
    trial_types = sorted(np.unique(metadata['trial_type']))

    # Position traces (visible by default)
    for trial_type in trial_types:
        mask = metadata['trial_type'] == trial_type
        positions = metadata['distance'][mask]
        cues = metadata['cue'][mask]
        track_len = metadata['track_length'][mask].max()
        colorscale = pfmt.trial_type_colorscale(trial_type)

        cb_kwargs = dict(
            title=f'{trial_type} pos (cm)', len=0.4,
            x=1.0 if trial_type == trial_types[0] else 1.15,
            y=0.75 if trial_type == trial_types[0] else 0.25,
        ) if trial_type in [trial_types[0], trial_types[-1]] else None

        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0], y=embedding[mask, 1], z=embedding[mask, 2],
            mode='markers', name=trial_type, visible=True,
            marker=dict(
                size=point_size, opacity=opacity, color=positions,
                colorscale=colorscale, cmin=0, cmax=track_len,
                colorbar=cb_kwargs,
            ),
            text=[pfmt.get_cue_labels().get(int(c), f'Cue {c}') for c in cues],
            hovertemplate=f'{trial_type}<br>Pos: %{{marker.color:.1f}} cm<br>Cue: %{{text}}<extra></extra>',
        ))
    n_position_traces = len(trial_types)

    # Cue traces (hidden by default)
    cue_trace_count = 0
    for trial_type in trial_types:
        mask = metadata['trial_type'] == trial_type
        cues = metadata['cue'][mask]
        positions = metadata['distance'][mask]
        emb = embedding[mask]

        base_colors = pfmt.get_cue_colors()
        tt_idx = trial_types.index(trial_type)
        factor = 1.3 - tt_idx * 0.3
        cue_colors = {cid: pfmt.scale_color(c, factor=factor) for cid, c in base_colors.items()}
        suffix = f' ({trial_type})'

        for cue_id in sorted(set(cues)):
            cue_mask = cues == cue_id
            if not cue_mask.any():
                continue
            cue_int = int(cue_id)
            label = pfmt.get_cue_labels().get(cue_int, f'Cue {cue_int}')
            color = cue_colors.get(cue_int, '#CCCCCC')
        #TODO again consider fallback color

            fig.add_trace(go.Scatter3d(
                x=emb[cue_mask, 0], y=emb[cue_mask, 1], z=emb[cue_mask, 2],
                mode='markers', name=f'{label}{suffix}', visible=False,
                marker=dict(size=point_size, opacity=opacity, color=color),
                showlegend=bool(cue_int != 0),
                customdata=positions[cue_mask],
                hovertemplate=f'{trial_type}<br>Pos: %{{customdata:.1f}} cm<br>Cue: {label}<extra></extra>',
            ))
            cue_trace_count += 1

    # Dropdown toggle
    pos_visible = [True] * n_position_traces + [False] * cue_trace_count
    cue_visible = [False] * n_position_traces + [True] * cue_trace_count

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down", x=0.0, y=1.15, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Color by Position", method="update", args=[{"visible": pos_visible}]),
                dict(label="Color by Cue", method="update", args=[{"visible": cue_visible}]),
            ],
        )],
        title=title or f'UMAP: {" vs ".join(trial_types)} (dropdown to toggle)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant'),
    )
#TODO again need to adjust for multiple trial types
    _show_and_save(fig, save_path)
    return fig



# SINGLE-TRIAL TRAJECTORY

def plot_umap_3d_single_trial_trajectory(
        embedding: np.ndarray,
        metadata: dict[str, np.ndarray],
        trial_ids: list[int] | None = None,
        n_trials_per_type: int = 5,
        point_size: int = 3,
        line_width: float = 4,
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

    unique_trials = np.unique(metadata['trial'])
    unique_types = np.unique(metadata['trial_type'])

    # Auto-select trials: pick n from middle of session for each type
    if trial_ids is None:
        trial_ids = []
        for trial_type in unique_types:
            type_mask = metadata['trial_type'] == trial_type
            type_trials = np.unique(metadata['trial'][type_mask])
            # Pick from the middle of the session (more stable behavior)
            mid = len(type_trials) // 2
            start = max(0, mid - n_trials_per_type // 2)
            selected = type_trials[start:start + n_trials_per_type]
            trial_ids.extend(selected.tolist())

    # Color palette for individual trials
    trial_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F0B27A', '#82E0AA',
    ]

    fig = go.Figure()
#TODO impose cues on background; the trials should increase in color or something so we can see the temporal relation
    # Background: all points faintly
    if show_background:
        fig.add_trace(go.Scatter3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            mode='markers', name='all frames',
            marker=dict(size=2, opacity=background_opacity, color='#888888'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Plot each selected trial as a connected line
    for i, trial_id in enumerate(trial_ids):
        trial_mask = metadata['trial'] == trial_id
        if not trial_mask.any():
            continue

        trial_emb = embedding[trial_mask]
        trial_pos = metadata['distance'][trial_mask]
        trial_cues = metadata['cue'][trial_mask]
        trial_type = metadata['trial_type'][trial_mask][0]
        color = trial_palette[i % len(trial_palette)]

        # Line connecting consecutive frames
        fig.add_trace(go.Scatter3d(
            x=trial_emb[:, 0], y=trial_emb[:, 1], z=trial_emb[:, 2],
            mode='lines+markers',
            name=f'Trial {trial_id} ({trial_type})',
            line=dict(color=color, width=line_width),
            marker=dict(size=point_size, color=color, opacity=opacity),
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
        title='UMAP: Single-Trial Trajectories (◆ = trial start)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant'),
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


# CONVENIENCE FUNCTIONS

def compare_trial_types_umap_3d(
        df: pl.DataFrame,
        signal_column: str = "single_day_spikes",
        color_by: Literal['position', 'cue', 'toggle', 'natural'] = 'toggle',
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        min_speed: float = 2.0,
        max_frames: int | None = None,
        point_size: int = 2,
        opacity: float = 0.7,
        save_path: Path | None = None,
) -> tuple['go.Figure', np.ndarray, dict[str, np.ndarray]]:
    """One-liner: frame_df → 3D UMAP comparing trial types.

    Args:
        df: Frame-level data (after process_session).
        signal_column: Column with neural signals.
        color_by: 'position' (per-type colorscale gradient), 'cue' (per-cue
            clickable legend), 'toggle' (dropdown to switch), 'natural'
            (see if trial types naturally separate).
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        min_speed: Filter out slow frames.
        max_frames: Subsample for speed.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        save_path: Save as .html or image.

    Returns:
        Tuple of (plotly_figure, embedding, metadata).
    """
    neural_data, metadata = prepare_umap_data(
        df, signal_column=signal_column, min_speed=min_speed, max_frames=max_frames,
    )

    trial_types = np.unique(metadata['trial_type'])
    print(f"Trial types: {list(trial_types)}")
    for trial_type in trial_types:
        print(f"  {trial_type}: {(metadata['trial_type'] == trial_type).sum()} frames")

    embedding = compute_umap(neural_data, n_components=3, n_neighbors=n_neighbors, min_dist=min_dist)

    plot_fn = {
        'position': plot_umap_3d_overlay_trial_types,
        'cue': plot_umap_3d_overlay_by_cue,
        'toggle': plot_umap_3d_with_toggle,
        'natural': plot_umap_3d_natural_separation,
    }
    fig = plot_fn[color_by](embedding, metadata, point_size, opacity, save_path=save_path)
    return fig, embedding, metadata


def quick_umap_plot(
        df: pl.DataFrame,
        signal_column: str = "single_day_spikes",
        n_components: int = 3,
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        interactive: bool = True,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        min_speed: float = 2.0,
        max_frames: int | None = None,
        save_path: Path | None = None,
        **prep_kwargs,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One-liner: frame_df → UMAP visualization.

    Args:
        df: Frame-level data (after process_session).
        signal_column: Neural signal column.
        n_components: Embedding dimensions (1, 2, or 3).
        strategy: Coloring strategy.
        interactive: Use Plotly for 3D (True) or matplotlib (False).
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        min_speed: Speed filter.
        max_frames: Subsample limit.
        save_path: Save output.
        **prep_kwargs: Additional args to prepare_umap_data.

    Returns:
        Tuple of (embedding, metadata).
    """
    neural_data, metadata = prepare_umap_data(
        df, signal_column=signal_column, min_speed=min_speed,
        max_frames=max_frames, **prep_kwargs,
    )
    embedding = compute_umap(neural_data, n_components, n_neighbors, min_dist)

    if n_components == 1:
        plot_umap_1d(embedding, metadata, strategy)
        plt.show()
    elif n_components == 2:
        plot_umap_2d(embedding, metadata, strategy, save_path=save_path)
        plt.show()
    elif n_components == 3 and interactive and HAS_PLOTLY:
        plot_umap_3d_interactive(embedding, metadata, strategy, save_path=save_path)
    elif n_components == 3:
        # Fallback to matplotlib 3D
        colors, _ = get_colors_for_strategy(strategy, metadata)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=colors, s=2, alpha=0.6)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        plt.tight_layout()
        plt.show()

    return embedding, metadata
