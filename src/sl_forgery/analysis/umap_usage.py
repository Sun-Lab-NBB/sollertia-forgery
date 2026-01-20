"""
Example usage of UMAP plotting functions with frame-level neural data.
Demonstrates various visualization strategies for analyzing neural manifolds.
"""

import polars as pl
import numpy as np

from pathlib import Path
from trial_based_structure import full_pipeline
from umap_plotting_cs import (
    prepare_umap_data_frame_level,
    quick_umap,
    plot_umap_1d,
    plot_umap_2d,
    plot_umap_3d,
    plot_umap_interactive_2d,
    plot_trial_type_comparison,
    ColoringStrategy
)



#Basic 2D UMAP with different coloring strategies

def example_basic_2d_umap(df: pl.DataFrame):
    """
    Create basic 2D UMAP plots with different coloring strategies.
    """
    # Prepare data with optional filtering
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        spike_column="single_day_spikes",
        distance_column="distance_cm",
        cue_column="cue",
        trial_type_column="trial_type",
        speed_column="speed_cm_s",
        trial_column="trial",
        min_speed=2.0,  # Filter for active running
        max_frames=10000  # Subsample for computational efficiency
    )

    # Run UMAP
    embedding_2d = quick_umap(neural_data, n_components=2)

    # Plot with different coloring strategies
    fig_cue = plot_umap_2d(embedding_2d, metadata, strategy=ColoringStrategy.CUE)
    fig_cue.savefig('umap_2d_by_cue.png', dpi=300, bbox_inches='tight')

    fig_pos = plot_umap_2d(embedding_2d, metadata, strategy=ColoringStrategy.POSITION)
    fig_pos.savefig('umap_2d_by_position.png', dpi=300, bbox_inches='tight')

    fig_speed = plot_umap_2d(embedding_2d, metadata, strategy=ColoringStrategy.SPEED)
    fig_speed.savefig('umap_2d_by_speed.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata



# 3D UMAP visualization

def example_3d_umap(df: pl.DataFrame):
    """
    Create 3D UMAP visualization with cue coloring.
    """
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        cues_to_include=['Cue 1', 'Cue 2', 'Cue 3'],  # Focus on specific cues
        max_frames=10000
    )

    # Run 3D UMAP
    embedding_3d = quick_umap(neural_data, n_components=3, n_neighbors=30)

    # Plot
    fig = plot_umap_3d(embedding_3d, metadata, strategy=ColoringStrategy.CUE)
    fig.savefig('umap_3d_by_cue.png', dpi=300, bbox_inches='tight')

    return embedding_3d, metadata


# 1D UMAP visualization

def example_1d_umap(df: pl.DataFrame):
    """
    Create 1D UMAP visualization to see ordering of neural states.
    """
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        max_frames=10000
    )

    # Run 1D UMAP
    embedding_1d = quick_umap(neural_data, n_components=1)

    # Plot with different coloring strategies
    fig_cue = plot_umap_1d(embedding_1d, metadata, strategy=ColoringStrategy.CUE)
    fig_cue.savefig('umap_1d_by_cue.png', dpi=300, bbox_inches='tight')

    fig_pos = plot_umap_1d(embedding_1d, metadata, strategy=ColoringStrategy.POSITION)
    fig_pos.savefig('umap_1d_by_position.png', dpi=300, bbox_inches='tight')

    return embedding_1d, metadata



# Interactive Plotly visualization based on Jacob's code


def example_interactive_umap(df: pl.DataFrame):
    """
    Create interactive Plotly visualization with clickable legend.
    """
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        max_frames=10000
    )

    embedding_2d = quick_umap(neural_data, n_components=2)

    # Create interactive plots with different colorings
    fig_cue = plot_umap_interactive_2d(
        embedding_2d, metadata,
        strategy=ColoringStrategy.CUE,
        title="2D UMAP colored by Cue"
    )
    fig_cue.write_html('umap_interactive_cue.html')

    fig_pos = plot_umap_interactive_2d(
        embedding_2d, metadata,
        strategy=ColoringStrategy.POSITION,
        title="2D UMAP colored by Position"
    )
    fig_pos.write_html('umap_interactive_position.html')

    return fig_cue, fig_pos


# Trial type comparison

def example_trial_type_comparison(df: pl.DataFrame):
    """
    Compare ABC vs ABDC trials in UMAP space.
    """
    # Side-by-side comparison
    embedding, metadata, fig_separate = plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=True,
        min_speed=5.0,
        max_frames=10000
    )
    fig_separate.savefig('umap_trial_types_separate.png', dpi=300, bbox_inches='tight')

    # Overlay comparison
    embedding, metadata, fig_overlay = plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=False,
        min_speed=5.0,
        max_frames=10000
    )
    fig_overlay.savefig('umap_trial_types_overlay.png', dpi=300, bbox_inches='tight')

    return embedding, metadata


# Filtering by specific cue zones

def example_cue_specific_analysis(df: pl.DataFrame):
    """
    Analyze neural manifold for specific cue zones.
    """
    # Focus on reward zones (e.g., Cue 3 and Cue 4)
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        cues_to_include=['Cue 3', 'Cue 4'],
        min_speed=5.0,
        max_frames=10000
    )

    embedding_2d = quick_umap(neural_data, n_components=2)

    fig = plot_umap_2d(embedding_2d, metadata, strategy=ColoringStrategy.POSITION)
    fig.suptitle('UMAP for Reward Zones Only', fontsize=14)
    fig.savefig('umap_reward_zones.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


#  Session progress analysis

def example_session_progress(df: pl.DataFrame):
    """
    Analyze how neural representations change over the session.
    """
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        max_frames=10000
    )

    embedding_2d = quick_umap(neural_data, n_components=2)

    fig = plot_umap_2d(
        embedding_2d, metadata,
        strategy=ColoringStrategy.SESSION_PROGRESS,
        cmap_name='plasma'
    )
    fig.suptitle('UMAP colored by Session Progress', fontsize=14)
    fig.savefig('umap_session_progress.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


# State-filtered analysis

def example_state_filtered_analysis(df: pl.DataFrame):
    """
    Analyze neural activity during specific behavioral states.
    """
    # Filter for specific experimental state (if you have such a column)
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        state_filters={
            'experiment_state': 1,  # Example: specific task state
            'guided': True  # Example: guided trials only
        },
        max_frames=10000
    )

    embedding_2d = quick_umap(neural_data, n_components=2)

    fig = plot_umap_2d(embedding_2d, metadata, strategy=ColoringStrategy.CUE)
    fig.suptitle('UMAP for Guided Trials Only', fontsize=14)
    fig.savefig('umap_guided_trials.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


#  Custom UMAP parameters

def example_custom_umap_params(df: pl.DataFrame):
    """
    Experiment with different UMAP parameters.
    """
    neural_data, metadata = prepare_umap_data_frame_level(
        df,
        min_speed=5.0,
        max_frames=10000
    )

    # Try different neighbor sizes
    for n_neighbors in [5, 15, 30, 50]:
        embedding = quick_umap(
            neural_data,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1
        )

        fig = plot_umap_2d(embedding, metadata, strategy=ColoringStrategy.CUE)
        fig.suptitle(f'UMAP with n_neighbors={n_neighbors}', fontsize=14)
        fig.savefig(f'umap_neighbors_{n_neighbors}.png', dpi=300, bbox_inches='tight')


# Speed-filtered comparison

def example_speed_filtered_comparison(df: pl.DataFrame):
    """
    Compare neural manifolds at different speeds.
    """
    import matplotlib.pyplot as plt

    # Low speed
    neural_low, meta_low = prepare_umap_data_frame_level(
        df, min_speed=0, max_speed=10, max_frames=5000
    )
    embed_low = quick_umap(neural_low, n_components=2)

    # High speed
    neural_high, meta_high = prepare_umap_data_frame_level(
        df, min_speed=20, max_speed=100, max_frames=5000
    )
    embed_high = quick_umap(neural_high, n_components=2)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Low speed plot
    unique_cues = np.unique(meta_low['cue'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cues)))
    cue_to_color = {cue: colors[i] for i, cue in enumerate(unique_cues)}

    for cue in unique_cues:
        mask = meta_low['cue'] == cue
        axes[0].scatter(embed_low[mask, 0], embed_low[mask, 1],
                        c=[cue_to_color[cue]], label=cue, alpha=0.6, s=20)
    axes[0].set_title('Low Speed (0-10 cm/s)')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].legend()

    # High speed plot
    for cue in unique_cues:
        mask = meta_high['cue'] == cue
        axes[1].scatter(embed_high[mask, 0], embed_high[mask, 1],
                        c=[cue_to_color[cue]], label=cue, alpha=0.6, s=20)
    axes[1].set_title('High Speed (20-100 cm/s)')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig('umap_speed_comparison.png', dpi=300, bbox_inches='tight')

    return fig



if __name__ == "__main__":
    # Load your frame-level data
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')
    behavior_df = pl.read_ipc(session_root / '2025-09-16-18-44-32-476061.feather')

    print("Running data processing pipeline...")
    df, track_lengths, original_df = full_pipeline(
        behavior_df,
        system_state='run',
        signal_col='single_day_spikes',
        binning=False
    )


    # Run examples
    print("Running Example 1: Basic 2D UMAP...")
    embedding_2d, metadata = example_basic_2d_umap(df)

    print("Running Example 2: 3D UMAP...")
    embedding_3d, metadata_3d = example_3d_umap(df)

    print("Running Example 3: 1D UMAP...")
    embedding_1d, metadata_1d = example_1d_umap(df)

    print("Running Example 4: Interactive UMAP...")
    fig_cue, fig_pos = example_interactive_umap(df)

    print("Running Example 5: Trial Type Comparison...")
    embedding_comp, metadata_comp = example_trial_type_comparison(df)

    print("All examples completed! Check output files.")