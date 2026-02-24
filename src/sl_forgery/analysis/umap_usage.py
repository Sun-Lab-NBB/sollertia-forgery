"""
UMAP usage examples for frame-level neural data.

Workflow:
    1. Load feather + config, run process_session (or load processed parquet)
    2. prepare_umap_data() → neural array + metadata
    3. compute_umap() → embedding
    4. Plot with interactive Plotly (3D) or matplotlib (1D/2D)
"""

from pathlib import Path
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

import umap_plotting as uplot


def _save(fig, save_path: Path | None, filename: str):
    """Save matplotlib figure if path provided.

    Args:
        fig: Matplotlib figure.
        save_path: Directory to save into.
        filename: Output filename.
    """
    if save_path:
        fig.savefig(save_path / filename, dpi=300, bbox_inches='tight')


# INTERACTIVE 3D PLOTLY EXAMPLES

def example_compare_trial_types(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compare trial typse with multiple coloring modes.

    Generates four interactive 3D plots:
        - Toggle dropdown (position ↔ cue)
        - Position only (blue=ABC, red=ABDC)
        - Cue only (clickable legend)
        - Natural separation

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving .html files.

    Returns:
        Tuple of (embedding, metadata).
    """
    # Toggle dropdown (position ↔ cue)
    fig, embedding, metadata = uplot.compare_trial_types_umap_3d(
        frame_df,
        color_by='toggle',
        point_size=2,
        save_path=save_path / 'umap_toggle.html' if save_path else None,
    )

    # Position only
    fig2, _, _ = uplot.compare_trial_types_umap_3d(
        frame_df, color_by='position',
        save_path=save_path / 'umap_position.html' if save_path else None,
    )

    # Cue only
    fig3, _, _ = uplot.compare_trial_types_umap_3d(
        frame_df, color_by='cue',
        save_path=save_path / 'umap_cue.html' if save_path else None,
    )

    # Natural separation
    fig4, _, _ = uplot.compare_trial_types_umap_3d(
        frame_df, color_by='natural',
        save_path=save_path / 'umap_natural.html' if save_path else None,
    )

    return embedding, metadata


def example_quick_plots(frame_df: pl.DataFrame):
    """Quick one-liners with different coloring strategies.

    Args:
        frame_df: Processed frame-level DataFrame.
    """
    uplot.quick_umap_plot(frame_df, n_components=3, strategy=uplot.ColoringStrategy.CUE, interactive=True)
    uplot.quick_umap_plot(frame_df, n_components=3, strategy=uplot.ColoringStrategy.SPEED, interactive=True)
    uplot.quick_umap_plot(frame_df, n_components=3, strategy=uplot.ColoringStrategy.SESSION_PROGRESS, interactive=True)
    uplot.quick_umap_plot(frame_df, n_components=1, strategy=uplot.ColoringStrategy.SESSION_PROGRESS)


def example_single_trial_trajectories(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
):
    """Plot individual trial trajectories through the manifold.

    Shows the ring-like topology: neural state traces a loop on each trial.
    Diamond markers indicate trial start positions.

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving output.
    """
    neural_data, metadata = uplot.prepare_umap_data(frame_df)
    embedding = uplot.compute_umap(neural_data, n_components=3)
    uplot.plot_umap_3d_single_trial_trajectory(
        embedding, metadata,
        n_trials_per_type=5,
        save_path=save_path / 'umap_trajectories.html' if save_path else None,
    )


def example_position_matched(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
):
    """Compare trial types over the shared position range only.

    Clips longer track length so you can see where the manifolds
    overlap vs diverge for the same spatial positions.

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving output.
    """
    neural_data, metadata = uplot.prepare_umap_data(frame_df)
    embedding = uplot.compute_umap(neural_data, n_components=3)
    uplot.plot_umap_3d_position_matched(
        embedding, metadata,
        save_path=save_path / 'umap_position_matched.html' if save_path else None,
    )


# STATIC MATPLOTLIB EXAMPLES

def example_2d_comparison(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
):
    """2D UMAP with multiple coloring strategies.

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving .png files.
    """
    neural_data, metadata = uplot.prepare_umap_data(frame_df, max_frames=20000)
    embedding = uplot.compute_umap(neural_data, n_components=2, n_neighbors=20)

    for strategy in [uplot.ColoringStrategy.CUE, uplot.ColoringStrategy.POSITION, uplot.ColoringStrategy.SPEED]:
        fig = uplot.plot_umap_2d(embedding, metadata, strategy=strategy)
        _save(fig, save_path, f'umap_2d_{strategy.value}.png')
        plt.show()


def example_2d_density(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
):
    """2D UMAP with KDE density contours per trial type.

    Shows where the manifold is concentrated for different trial types, plus an
    overlay panel to see overlap.

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving .png files.
    """
    neural_data, metadata = uplot.prepare_umap_data(frame_df, max_frames=20000)
    embedding = uplot.compute_umap(neural_data, n_components=2, n_neighbors=20)

    fig = uplot.plot_umap_2d_density(embedding, metadata)
    _save(fig, save_path, 'umap_2d_density.png')
    plt.show()


def example_umap_params_sweep(
        frame_df: pl.DataFrame,
        save_path: Path | None = None,
):
    """Try different n_neighbors values to see effect on manifold structure.

    Args:
        frame_df: Processed frame-level DataFrame.
        save_path: Directory for saving .png files.
    """
    neural_data, metadata = uplot.prepare_umap_data(frame_df, max_frames=20000)

    for n_neighbors in [5, 15, 30, 50]:
        embedding = uplot.compute_umap(neural_data, n_components=2, n_neighbors=n_neighbors)
        fig = uplot.plot_umap_2d(embedding, metadata, strategy=uplot.ColoringStrategy.CUE,
                           title=f'n_neighbors={n_neighbors}')
        _save(fig, save_path, f'umap_nn_{n_neighbors}.png')
        plt.show()


if __name__ == "__main__":
    from df_processing import (load_session_dir, get_session_prefix, load_processed_session, save_processed_session,
                               process_session, load_multiday_sessions)

    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    date = '2025-09-15'  # again, the .feather file in this is actually from 9-16, too slow to download at my house.
    # ***DO NOT GET MISTAKEN

#TODO need some way to specify which signal you want (spikes vs dff, single vs multi (potentially for DSA!)
    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    parquet_path = behavior_path.parent / f'{prefix}_processed.parquet'

    if parquet_path.exists():
        print(f"Loading: {parquet_path}")
        data, metadata = load_processed_session(parquet_path)
    else:
        print("No processed file found, processing from raw...")        #OR if you want to process the session with
        # other system states, bc the default is to process by run
        behavior_df = pl.read_ipc(behavior_path)
        data, metadata = process_session(behavior_df, config)
        save_processed_session(data, behavior_path.parent, session_data, metadata)

    save_path = None  # Set to a Path to save figures

    # Main interactive plots
    # print("=== Trial type comparison (Plotly) ===")
    # example_compare_trial_types(data, save_path)
    #
    # # Quick one-liners
    # print("\n=== Quick plots ===")
    # example_quick_plots(data)

    # Single-trial trajectories
    print("\n=== Single-trial trajectories ===")
    example_single_trial_trajectories(data, save_path)

    # Position-matched comparison
    print("\n=== Position-matched comparison ===")
    example_position_matched(data, save_path)

    # Static 2D
    print("\n=== 2D comparison ===")
    example_2d_comparison(data, save_path)

    # Density contours
    print("\n=== 2D density contours ===")
    example_2d_density(data, save_path)
