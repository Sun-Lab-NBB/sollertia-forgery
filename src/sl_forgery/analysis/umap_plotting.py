"""
UMAP Plotting Module for Frame-Level Neural Data
Provides visualization functions for analyzing neural manifolds using UMAP dimensionality reduction.
Adapted for frame-level data where neural activity is stored in arrays per frame.
"""

import numpy as np
import polars as pl
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objects as go
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from trial_plotting import get_cue_colors, get_cue_labels, TRIAL_TYPE_COLORS


class ColoringStrategy(Enum):
    """Defines different strategies for coloring UMAP points"""
    CUE = "cue"
    POSITION = "position"
    TRIAL_TYPE = "trial_type"
    SPEED = "speed"
    SESSION_PROGRESS = "session_progress"


def prepare_umap_data_frame_level(
        df: pl.DataFrame,
        signal_column: str = "single_day_spikes",
        min_speed: Optional[float] = 2.0,
        max_speed: Optional[float] = None,
        cues_to_include: Optional[List[int]] = None,
        trial_types_to_include: Optional[List[str]] = None,
        state_filters: Optional[Dict[str, Any]] = None,
        max_frames: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare frame-level data for UMAP analysis.

    Args:
        df: Polars DataFrame with frame-level data
        signal_column: Column containing neural activity arrays
        min_speed: Minimum speed threshold (inclusive); avoid periods where mouse is stationary
        max_speed: Maximum speed threshold (inclusive)
        cues_to_include: List of cue zones to include (None = all); possible uses -- looking at D separately,
            reward zone, gray zones
        trial_types_to_include: List of trial types to include (None = all)
        state_filters: Dict of {column_name: value} for additional filtering
        max_frames: Maximum number of frames to use (use stride if exceeded); None includes the entire session,
            but you can specify less for speed/quick view of a plot/checking code

    Returns:
        neural_data: Array of shape (n_frames, n_cells)
        metadata: Dict with arrays for 'distance', 'cue', 'trial_type', 'speed', 'trial'
    """
    # Apply filters
    filtered_df = df

    # Speed filter
    if min_speed is not None:
        filtered_df = filtered_df.filter(pl.col('speed_cm_s') >= min_speed)
    if max_speed is not None:
        filtered_df = filtered_df.filter(pl.col('speed_cm_s') <= max_speed)

    # Cue filter
    if cues_to_include is not None:
        filtered_df = filtered_df.filter(pl.col('cue').is_in(cues_to_include))

    # Trial type filter
    if trial_types_to_include is not None:
        filtered_df = filtered_df.filter(pl.col('trial_type').is_in(trial_types_to_include))

    # State filters
    if state_filters:
        for col, value in state_filters.items():
            if col in filtered_df.columns:
                filtered_df = filtered_df.filter(pl.col(col) == value)

    # Enforce max_frames by subsampling
    if max_frames is not None and len(filtered_df) > max_frames:
        stride = len(filtered_df) // max_frames     #use stride instead of random sampling, uses every Nth frame
        filtered_df = filtered_df.gather_every(stride)

    # Extract neural data by stacking spike arrays
    neural_data = np.vstack(filtered_df[signal_column].to_list())

    # Extract metadata
    metadata = {
        'distance': filtered_df['distance_cm'].to_numpy(),
        'cue': filtered_df['cue'].to_numpy(),
        'trial_type': filtered_df['trial_type'].to_numpy(),
        'speed': filtered_df['speed_cm_s'].to_numpy(),
        'trial': filtered_df['trial'].to_numpy()
    }

    return neural_data, metadata


def get_colors_for_strategy(
        strategy: ColoringStrategy,
        metadata: Dict[str, np.ndarray],
        cmap_name: str = 'viridis'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate colors based on the specified strategy.

    Args:
        strategy: ColoringStrategy enum value
        metadata: Dictionary containing metadata arrays
        cmap_name: Colormap name for continuous variables

    Returns:
        colors: Array of colors (RGBA or category labels)
        color_info: Dict with colormap/legend information
    """
    if strategy == ColoringStrategy.CUE:
        # Categorical coloring by cue
        cue_color_map = get_cue_colors()
        cue_label_map = get_cue_labels()
        unique_cues = np.unique(metadata['cue'])
        legend = {}
        color_list = []
        for cue in metadata['cue']:
            hex_color = cue_color_map.get(int(cue), '#CCCCCC')
            color_list.append(hex_color)
        for cue in unique_cues:
            label = cue_label_map.get(int(cue), f'Cue {cue}')
            legend[label] = cue_color_map.get(int(cue), '#CCCCCC')
        colors = np.array(color_list)
        color_info = {
            'type': 'categorical',
            'legend': legend,
            'label': 'Cue Zone',
            'raw_cue_map': {int(c): cue_label_map.get(int(c), f'Cue {c}') for c in unique_cues},
        }

    elif strategy == ColoringStrategy.POSITION:
        # Continuous coloring by position
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=metadata['distance'].min(), vmax=metadata['distance'].max())
        colors = cmap(norm(metadata['distance']))
        color_info = {
            'type': 'continuous',
            'cmap': cmap,
            'norm': norm,
            'label': 'Position (cm)'
        }

    elif strategy == ColoringStrategy.TRIAL_TYPE:
        # Categorical coloring by trial type
        unique_types = np.unique(metadata['trial_type'])
        legend = {tt: TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in unique_types}
        colors = np.array([TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in metadata['trial_type']])
        color_info = {
            'type': 'categorical',
            'legend': legend,
            'label': 'Trial Type'
        }

    elif strategy == ColoringStrategy.SPEED:
        # Continuous coloring by speed
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=metadata['speed'].min(), vmax=metadata['speed'].max())
        colors = cmap(norm(metadata['speed']))
        color_info = {
            'type': 'continuous',
            'cmap': cmap,
            'norm': norm,
            'label': 'Speed (cm/s)'
        }

    elif strategy == ColoringStrategy.SESSION_PROGRESS:
        # Continuous coloring by session progress
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=metadata['trial'].min(), vmax=metadata['trial'].max())
        colors = cmap(norm(metadata['trial']))
        color_info = {
            'type': 'continuous',
            'cmap': cmap,
            'norm': norm,
            'label': 'Trial Number'
        }

    return colors, color_info


def _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha=0.6, s=20):
    """Scatter plot for categorical coloring strategies (1D, 2D, or 3D)."""
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
        elif n_dims == 2:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=label, alpha=alpha, s=s)
        elif n_dims == 3:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                       c=color, label=label, alpha=alpha, s=s)

    ax.legend(title=color_info['label'], bbox_to_anchor=(1.05, 1), loc='upper left')


def quick_umap(
        neural_data: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP on neural data.

    Args:
        neural_data: Array of shape (n_samples, n_features)
        n_components: Number of UMAP dimensions (1, 2, or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric
        random_state: Random seed

    Returns:
        embedding: UMAP embedding of shape (n_samples, n_components)
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(neural_data)
    return embedding


def plot_umap_1d(
        embedding: np.ndarray,
        metadata: Dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        cmap_name: str = 'viridis',
        figsize: Tuple[float, float] = (12, 2),
        alpha: float = 0.6,
        s: float = 20
) -> plt.Figure:
    """
    Plot 1D UMAP embedding.

    Args:
        embedding: 1D UMAP embedding
        metadata: Dictionary with metadata arrays
        strategy: ColoringStrategy to use
        cmap_name: Colormap name for continuous variables
        figsize: Figure size
        alpha: Point transparency
        s: Point size

    Returns:
        fig: Matplotlib figure
    """
    colors, color_info = get_colors_for_strategy(strategy, metadata, cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    # Create y-values with small jitter for visibility
    y_vals = np.random.normal(0, 0.02, size=len(embedding))

    if color_info['type'] == 'categorical':
        _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
    else:
        # Continuous coloring
        scatter = ax.scatter(embedding, y_vals, c=colors, alpha=alpha, s=s)
        cbar = plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                            ax=ax)
        cbar.set_label(color_info['label'])

    ax.set_xlabel('UMAP 1')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_umap_2d(
        embedding: np.ndarray,
        metadata: Dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        cmap_name: str = 'viridis',
        figsize: Tuple[float, float] = (10, 8),
        alpha: float = 0.6,
        s: float = 20
) -> plt.Figure:
    """
    Plot 2D UMAP embedding with specified coloring strategy.

    Args:
        embedding: 2D UMAP embedding
        metadata: Dictionary with metadata arrays
        strategy: ColoringStrategy to use
        cmap_name: Colormap name for continuous variables
        figsize: Figure size
        alpha: Point transparency
        s: Point size

    Returns:
        fig: Matplotlib figure
    """
    colors, color_info = get_colors_for_strategy(strategy, metadata, cmap_name)

    fig, ax = plt.subplots(figsize=figsize)

    if color_info['type'] == 'categorical':
        _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
    else:
        # Continuous coloring
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=alpha, s=s)
        cbar = plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                            ax=ax)
        cbar.set_label(color_info['label'])

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_umap_3d(
        embedding: np.ndarray,
        metadata: Dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        cmap_name: str = 'viridis',
        figsize: Tuple[float, float] = (12, 10),
        alpha: float = 0.6,
        s: float = 20
) -> plt.Figure:
    """
    Plot 3D UMAP embedding with specified coloring strategy.

    Args:
        embedding: 3D UMAP embedding
        metadata: Dictionary with metadata arrays
        strategy: ColoringStrategy to use
        cmap_name: Colormap name for continuous variables
        figsize: Figure size
        alpha: Point transparency
        s: Point size

    Returns:
        fig: Matplotlib figure
    """
    colors, color_info = get_colors_for_strategy(strategy, metadata, cmap_name)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if color_info['type'] == 'categorical':
        _scatter_categorical(ax, embedding, metadata, color_info, strategy, alpha, s)
    else:
        # Continuous coloring
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                             c=colors, alpha=alpha, s=s)
        cbar = plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                            ax=ax, shrink=0.5)
        cbar.set_label(color_info['label'])

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')

    plt.tight_layout()
    return fig


def plot_umap_interactive_2d(
        embedding: np.ndarray,
        metadata: Dict[str, np.ndarray],
        strategy: ColoringStrategy = ColoringStrategy.CUE,
        title: str = "Interactive 2D UMAP"
) -> go.Figure:
    """
    Create interactive 2D UMAP plot with Plotly.

    Args:
        embedding: 2D UMAP embedding
        metadata: Dictionary with metadata arrays
        strategy: ColoringStrategy to use
        title: Plot title

    Returns:
        fig: Plotly figure
    """
    fig = go.Figure()

    if strategy == ColoringStrategy.CUE:
        cue_color_map = get_cue_colors()
        cue_label_map = get_cue_labels()
        unique_cues = np.unique(metadata['cue'])

        for cue in unique_cues:
            mask = metadata['cue'] == cue
            hex_color = cue_color_map.get(int(cue), '#CCCCCC')
            label = cue_label_map.get(int(cue), f'Cue {cue}')
            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=label,
                marker=dict(size=5, color=hex_color, opacity=0.8),
                legendgroup=label
            ))

    elif strategy == ColoringStrategy.TRIAL_TYPE:
        unique_types = np.unique(metadata['trial_type'])

        for tt in unique_types:
            mask = metadata['trial_type'] == tt
            color = TRIAL_TYPE_COLORS.get(tt, '#999999')

            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=str(tt),
                marker=dict(size=5, color=color, opacity=0.8),
                legendgroup=str(tt)
            ))

    else:
        # Continuous coloring
        if strategy == ColoringStrategy.POSITION:
            color_values = metadata['distance']
            colorbar_title = 'Position (cm)'
        elif strategy == ColoringStrategy.SPEED:
            color_values = metadata['speed']
            colorbar_title = 'Speed (cm/s)'
        else:  # SESSION_PROGRESS
            color_values = metadata['trial']
            colorbar_title = 'Trial Number'

        fig.add_trace(go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        hovermode='closest',
        width=800,
        height=600
    )

    return fig


def plot_trial_type_comparison(
        df: pl.DataFrame,
        n_components: int = 2,
        separate_plots: bool = True,
        **umap_kwargs
) -> Tuple[np.ndarray, Dict[str, np.ndarray], plt.Figure]:
    """
    Compare trial types in UMAP space.

    Args:
        df: Frame-level DataFrame
        n_components: UMAP dimensions (2 or 3)
        separate_plots: If True, create side-by-side plots; else overlay
        **umap_kwargs: Additional arguments for prepare_umap_data_frame_level

    Returns:
        embedding: UMAP embedding
        metadata: Metadata dictionary
        fig: Matplotlib figure
    """
    # Prepare data and run UMAP
    neural_data, metadata = prepare_umap_data_frame_level(df, **umap_kwargs)
    embedding = quick_umap(neural_data, n_components=n_components)

    unique_types = np.unique(metadata['trial_type'])
    type_to_color = {tt: TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in unique_types}

    if separate_plots:
        fig, axes = plt.subplots(1, len(unique_types), figsize=(8 * len(unique_types), 6))
        if len(unique_types) == 1:
            axes = [axes]

        for ax, trial_type in zip(axes, unique_types):
            mask = metadata['trial_type'] == trial_type

            if n_components == 2:
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[type_to_color[trial_type]], alpha=0.6, s=20)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
            else:  # 3D
                ax = fig.add_subplot(1, len(unique_types), list(axes).index(ax) + 1, projection='3d')
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                           c=[type_to_color[trial_type]], alpha=0.6, s=20)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_zlabel('UMAP 3')

            ax.set_title(f'{trial_type} trials')
    else:
        # Overlay plot
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            for trial_type in unique_types:
                mask = metadata['trial_type'] == trial_type
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[type_to_color[trial_type]], label=trial_type,
                           alpha=0.6, s=20)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.legend()
        else:  # 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            for trial_type in unique_types:
                mask = metadata['trial_type'] == trial_type
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                           c=[type_to_color[trial_type]], label=trial_type,
                           alpha=0.6, s=20)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            ax.legend()

    plt.tight_layout()
    return embedding, metadata, fig


if __name__ == '__main__':
    from pathlib import Path

    session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    behavior_df = pl.read_ipc(session_root / '2025-09-16-18-44-32-476061.feather')

    neural_data, metadata = prepare_umap_data_frame_level(
        behavior_df,
        signal_column="single_day_spikes",
        max_frames=20000  #defaults to all frames (>42k), use less and subsample for speed
    )

    embedding = quick_umap(neural_data, n_components=2)
    fig = plot_umap_2d(embedding, metadata, strategy=ColoringStrategy.CUE)