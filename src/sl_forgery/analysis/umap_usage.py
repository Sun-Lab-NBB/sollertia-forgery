"""
Example usage of UMAP plotting functions with frame-level neural data.
Demonstrates various visualization strategies for analyzing neural manifolds.
"""

import polars as pl
import numpy as np

from pathlib import Path
from create_trial_based_df import fix_cue_offset, load_experiment_config
import umap_plotting as uplot


def _save(fig, save_path: Path | None, filename: str):
    """Helper function to save figure if path is provided."""
    if save_path:
        fig.savefig(save_path / filename, dpi=300, bbox_inches='tight')


#Basic 2D UMAP with different coloring strategies

def example_basic_2d_umap(df: pl.DataFrame,
                          save_path: Path | None = None):
    """
    Create basic 2D UMAP plots with different coloring strategies.

    add docstring
    """
    # Prepare data with optional filtering
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        signal_column="single_day_spikes",
        min_speed=2.0,  # This is the default but can make higher (up to threshold of 5 seems appropriate)
        max_frames=10000,  # Subsample for computational efficiency
    )

    # Run UMAP
    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    # Plot with different coloring strategies
    fig_cue = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.CUE)
    _save(fig_cue, save_path, 'umap_2d_by_cue.png')

    fig_pos = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.POSITION)
    _save(fig_pos, save_path, 'umap_2d_by_position.png')

    fig_speed = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.SPEED)
    _save(fig_speed, save_path, 'umap_2d_by_speed.png')

    return embedding_2d, metadata



# 3D UMAP visualization

def example_3d_umap(df: pl.DataFrame,
                    save_path: Path | None = None):
    """
    Create 3D UMAP visualization with cue coloring.
    """
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        max_frames=20000
    )

    # Run 3D UMAP
    embedding_3d = uplot.quick_umap(neural_data, n_components=3, n_neighbors=30)

    # Plot
    fig = uplot.plot_umap_3d(embedding_3d, metadata, strategy=uplot.ColoringStrategy.CUE)
    _save(fig, save_path, 'umap_3d_by_cue.png')

    return embedding_3d, metadata


# 1D UMAP visualization

def example_1d_umap(df: pl.DataFrame,
                    save_path: Path | None = None):
    """
    Create 1D UMAP visualization to see ordering of neural states.
    """
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        max_frames=20000
    )

    # Run 1D UMAP
    embedding_1d = uplot.quick_umap(neural_data, n_components=1)

    # Plot with different coloring strategies
    fig_cue = uplot.plot_umap_1d(embedding_1d, metadata, strategy=uplot.ColoringStrategy.CUE)
    _save(fig_cue, save_path, 'umap_1d_by_cue.png')

    fig_pos = uplot.plot_umap_1d(embedding_1d, metadata, strategy=uplot.ColoringStrategy.POSITION)
    _save(fig_pos, save_path, 'umap_2d_by_position.png')

    return embedding_1d, metadata



# Interactive Plotly visualization based on Jacob's code


def example_interactive_umap(df: pl.DataFrame):
    """
    Create interactive Plotly visualization with clickable legend.
    """
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        max_frames=20000
    )

    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    # Create interactive plots with different colorings
    fig_cue = uplot.plot_umap_interactive_2d(
        embedding_2d, metadata,
        strategy=uplot.ColoringStrategy.CUE,
        title="2D UMAP colored by Cue"
    )
    fig_cue.write_html('umap_interactive_cue.html')

    fig_pos = uplot.plot_umap_interactive_2d(
        embedding_2d, metadata,
        strategy=uplot.uplot.ColoringStrategy.POSITION,
        title="2D UMAP colored by Position"
    )
    fig_pos.write_html('umap_interactive_position.html')

    return fig_cue, fig_pos


# Trial type comparison

def example_trial_type_comparison(df: pl.DataFrame,
                                  save_path: Path | None = None):
    """
    Compare ABC vs ABDC trials in UMAP space.
    """
    # Side-by-side comparison
    embedding, metadata, fig_separate = uplot.plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=True,
        max_frames=20000
    )
    _save(fig_separate, save_path,'umap_trial_types_separate.png')

    # Overlay comparison
    embedding, metadata, fig_overlay = uplot.plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=False,
        max_frames=20000
    )
    _save(fig_overlay, save_path, 'umap_trial_types_overlay.png')

    return embedding, metadata


# Filtering by specific cue zones

def example_cue_specific_analysis(df: pl.DataFrame,
                                  save_path: Path | None = None):
    """
    Analyze neural manifold for specific cue zones.
    """
    # Focus on reward zones (Cue 3)
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        cues_to_include=[3],
        max_frames=20000
    )

    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    fig = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.POSITION)
    fig.suptitle('UMAP for Reward Zones Only', fontsize=14)
    _save(fig, save_path, 'umap_reward_zones.png')

    return embedding_2d, metadata


#  Session progress analysis

def example_session_progress(df: pl.DataFrame,
                             save_path: Path | None = None):
    """
    Analyze how neural representations change over the session.
    """
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        max_frames=20000
    )

    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    fig = uplot.plot_umap_2d(
        embedding_2d, metadata,
        strategy=uplot.ColoringStrategy.SESSION_PROGRESS,
        cmap_name='plasma'
    )
    fig.suptitle('UMAP colored by Session Progress', fontsize=14)
    _save(fig, save_path, 'umap_session_progress.png')

    return embedding_2d, metadata


# State-filtered analysis

def example_state_filtered_analysis(df: pl.DataFrame,
                                    save_path: Path | None = None):
    """
    Analyze neural activity during specific behavioral states.
    """
    # Filter for specific experimental state
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        state_filters={
            'experiment_state': 'run',  # Example: specific task state; baseline, run, rest (for Ivan)
            'guided': True  # Example: guided trials only, not sure if we'd want this but maybe helpful to compare
        },
        max_frames=20000
    )

    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    fig = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.CUE)
    fig.suptitle('UMAP for Guided Trials Only', fontsize=14)
    _save(fig, save_path, 'umap_guided_trials.png')

    return embedding_2d, metadata


#  Custom UMAP parameters

def example_custom_umap_params(df: pl.DataFrame,
                               save_path: Path | None = None):
    """
    Experiment with different UMAP parameters.
    """
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        max_frames=20000
    )

    # Try different neighbor sizes
    for n_neighbors in [5, 15, 30, 50]:
        embedding = uplot.quick_umap(
            neural_data,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1
        )

        fig = uplot.plot_umap_2d(embedding, metadata, strategy=uplot.ColoringStrategy.CUE)
        fig.suptitle(f'UMAP with n_neighbors={n_neighbors}', fontsize=14)
        _save(fig, save_path, f'umap_neighbors_{n_neighbors}.png')


# Speed-filtered comparison
#
# def example_speed_filtered_comparison(df: pl.DataFrame,
#                                       save_path: Path | None = None):
#     """
#     Compare neural manifolds at different speeds.
#     """
#   Deleted this function but could add back later, if we want to look at how speed affects manifold topo;logy


if __name__ == "__main__":
    # Load your frame-level data
# TODO will need ot make this work with server and directory structure --> project, mouse, date. rn hacky.  Also want
    #  it all to work from the CL

    session_root = Path('/Users/cs963/Desktop/sun_lab_projects')
    #mouse_id = '26_explore'   #something like this, but able to be called from CLI
    experiment_config_path = session_root / '26_explore/experiment_configuration.yaml'

    config = load_experiment_config(experiment_config_path)
    behavior_df = pl.read_ipc(session_root / '26_explore/2025-09-16-18-44-32-476061.feather')
    frame_df = fix_cue_offset(behavior_df, config, system_state='run')
    save_path = None        #specifiy path for saving figs; should this use mouse id or is enough that it's in the
    # folder?


    # Run examples
    print("Running Example 1: Basic 2D UMAP...")
    embedding_2d, metadata = example_basic_2d_umap(frame_df)

    print("Running Example 2: 3D UMAP...")
    embedding_3d, metadata_3d = example_3d_umap(frame_df)

    print("Running Example 3: 1D UMAP...")
    embedding_1d, metadata_1d = example_1d_umap(frame_df)

    print("Running Example 4: Interactive UMAP...")
    fig_cue, fig_pos = example_interactive_umap(frame_df)

    print("Running Example 5: Trial Type Comparison...")
    embedding_comp, metadata_comp = example_trial_type_comparison(frame_df)

    print("All examples completed! Check output files.")