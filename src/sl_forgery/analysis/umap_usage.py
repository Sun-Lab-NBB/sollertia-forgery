"""
Example usage of UMAP plotting functions with frame-level neural data.
Demonstrates various visualization strategies for analyzing neural manifolds.
"""

import polars as pl
import numpy as np

from pathlib import Path
#TODO could make this a single "import _ as _"
from create_trial_based_df import fix_cue_offset, load_experiment_config
import umap_plotting as uplot



#Basic 2D UMAP with different coloring strategies

def example_basic_2d_umap(df: pl.DataFrame):
#TODO need ot make this match the colors used in the trial plotting; should all be coherent
    """
    Create basic 2D UMAP plots with different coloring strategies.
    """
    # Prepare data with optional filtering
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        signal_column="single_day_spikes",
        min_speed=2.0,  # This is the default but can make higher (up to threshold of 5 seems appropriate)
        max_frames=10000  # Subsample for computational efficiency
    )

    # Run UMAP
    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    # Plot with different coloring strategies
    fig_cue = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.CUE)
    fig_cue.savefig('umap_2d_by_cue.png', dpi=300, bbox_inches='tight')

    fig_pos = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.POSITION)
    fig_pos.savefig('umap_2d_by_position.png', dpi=300, bbox_inches='tight')

    fig_speed = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.SPEED)
    fig_speed.savefig('umap_2d_by_speed.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata



# 3D UMAP visualization

def example_3d_umap(df: pl.DataFrame):
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
    fig.savefig('umap_3d_by_cue.png', dpi=300, bbox_inches='tight')

    return embedding_3d, metadata


# 1D UMAP visualization

def example_1d_umap(df: pl.DataFrame):
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
    fig_cue.savefig('umap_1d_by_cue.png', dpi=300, bbox_inches='tight')

    fig_pos = uplot.plot_umap_1d(embedding_1d, metadata, strategy=uplot.ColoringStrategy.POSITION)
    fig_pos.savefig('umap_1d_by_position.png', dpi=300, bbox_inches='tight')

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

    fig_pos = plot_umap_interactive_2d(
        embedding_2d, metadata,
        strategy=uplot.uplot.ColoringStrategy.POSITION,
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
    embedding, metadata, fig_separate = uplot.plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=True,
        max_frames=20000
    )
    fig_separate.savefig('umap_trial_types_separate.png', dpi=300, bbox_inches='tight')

    # Overlay comparison
    embedding, metadata, fig_overlay = uplot.plot_trial_type_comparison(
        df,
        n_components=2,
        separate_plots=False,
        max_frames=20000
    )
    fig_overlay.savefig('umap_trial_types_overlay.png', dpi=300, bbox_inches='tight')

    return embedding, metadata


# Filtering by specific cue zones

def example_cue_specific_analysis(df: pl.DataFrame):
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
    fig.savefig('umap_reward_zones.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


#  Session progress analysis

def example_session_progress(df: pl.DataFrame):
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
    fig.savefig('umap_session_progress.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


# State-filtered analysis

def example_state_filtered_analysis(df: pl.DataFrame):
    """
    Analyze neural activity during specific behavioral states.
    """
    # Filter for specific experimental state
    neural_data, metadata = uplot.prepare_umap_data_frame_level(
        df,
        state_filters={
            'experiment_state': 'baseline',  # Example: specific task state
            'guided': True  # Example: guided trials only, not sure if we'd want this but maybe helpful to compare
        },
        max_frames=20000
    )

    embedding_2d = uplot.quick_umap(neural_data, n_components=2)

    fig = uplot.plot_umap_2d(embedding_2d, metadata, strategy=uplot.ColoringStrategy.CUE)
    fig.suptitle('UMAP for Guided Trials Only', fontsize=14)
    fig.savefig('umap_guided_trials.png', dpi=300, bbox_inches='tight')

    return embedding_2d, metadata


#  Custom UMAP parameters

def example_custom_umap_params(df: pl.DataFrame):
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
        fig.savefig(f'umap_neighbors_{n_neighbors}.png', dpi=300, bbox_inches='tight')


# Speed-filtered comparison

def example_speed_filtered_comparison(df: pl.DataFrame):
    """
    Compare neural manifolds at different speeds.
    """
    import matplotlib.pyplot as plt

    # Low speed
    neural_low, meta_low = uplot.prepare_umap_data_frame_level(
        df, min_speed=0, max_speed=10, max_frames=5000
    )
    embed_low = uplot.quick_umap(neural_low, n_components=2)

    # High speed
    neural_high, meta_high = uplot.prepare_umap_data_frame_level(
        df, min_speed=20, max_speed=100, max_frames=5000
    )
    embed_high = uplot.quick_umap(neural_high, n_components=2)

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
# TODO will need ot make this work with server and directory structure --> project, mouse, date. rn hacky.  Also want
    #  it all to work from the CL

    session_root = Path('/Users/cs963/Desktop/sun_lab_projects')
    #mouse_id = '26_explore'   #something like this, but able to be called from CLI
    experiment_config_path = session_root / '26_explore/experiment_configuration.yaml'

    config = load_experiment_config(experiment_config_path)
    behavior_df = pl.read_ipc(session_root / '26_explore/2025-09-16-18-44-32-476061.feather')
    frame_df = fix_cue_offset(behavior_df, config, system_state='run')


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