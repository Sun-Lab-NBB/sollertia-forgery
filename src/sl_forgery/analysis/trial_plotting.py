"""
Trial Plotting Module

Place field and behavioral visualizations for trial-indexed calcium imaging data.
Works with TrialData from trial_based_df.py.
"""

from pathlib import Path
from typing import Optional, List, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import polars as pl

from create_trial_based_df import load_trial_data


# COLOR CONFIGURATION

# Tableau 10 - colorblind friendly
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

TRIAL_TYPE_COLORS = {
    'ABC': '#2E86AB',
    'ABDC': '#A23B72',
    'ABCD': '#A23B72',
}

TRIAL_TYPE_COLORS_DARK = {
    'ABC': '#0A4D68',
    'ABDC': '#6B0848',
    'ABCD': '#6B0848',
}


def get_cue_colors(config: dict = None, max_cue_id: int = 20) -> Dict[int, str]:
    """Get cue ID to color mapping."""
    colors = SPECIAL_CUE_COLORS.copy()
    
    for cue_id in range(1, max_cue_id + 1):
        idx = (cue_id - 1) % len(CUE_COLOR_PALETTE)
        colors[cue_id] = CUE_COLOR_PALETTE[idx]
    
    # Allow config override
    if config and 'cue_colors' in config:
        colors.update(config['cue_colors'])
    
    return colors


def get_cue_labels(config: dict = None, max_cue_id: int = 20) -> Dict[int, str]:
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
):
    """Add cue regions, reward zone, and x-ticks to a track axis."""
    from create_trial_based_df import get_cue_regions

    if config is None:
        return

    # Cue shading
    if show_cues:
        cue_regions = get_cue_regions(config, trial_type)
        plot_cue_regions(ax, cue_regions, config, show_labels=show_labels,
                         alpha=alpha, label_y=label_y)

    # Reward zone
    if show_reward_zone:
        ts = config.get('trial_structures', {}).get(trial_type, {})
        if 'reward_zone_start_cm' in ts:
            rz_start = ts['reward_zone_start_cm']
            rz_end = ts['reward_zone_end_cm']
            ax.axvline(rz_start, color='#1B9AAA', linestyle='--',
                       linewidth=1.5, alpha=0.7, zorder=5)
            ax.axvline(rz_end, color='#1B9AAA', linestyle='--',
                       linewidth=1.5, alpha=0.7, zorder=5)
            ax.text((rz_start + rz_end) / 2, label_y - 0.08, 'reward',
                    ha='center', va='top', fontsize=8, fontstyle='italic',
                    color='#1B9AAA',
                    transform=ax.get_xaxis_transform())

    # X-ticks at cue boundaries
    cue_width = 30
    if 'cue_map' in config:
        cue_width = list(config['cue_map'].values())[0]
    track_length = ts.get('trial_length_cm', 180) if config else 180
    ticks = list(range(0, int(track_length) + 1, int(cue_width)))
    if ticks[-1] != int(track_length):
        ticks.append(int(track_length))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)

    # Common styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, axis='y', zorder=0)


# CUE REGION PLOTTING


def plot_cue_regions(
    ax: Axes,
    cue_regions: Dict[int, tuple],
    config: dict = None,
    show_labels: bool = True,
    alpha: float = 0.15,
    label_y: float = 0.98,
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
                    fontsize=9, fontweight='bold',
                    transform=ax.get_xaxis_transform(),
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=color, alpha=0.6, edgecolor='none'
                    )
                )



# PLACE FIELD PLOTS

def plot_single_cell(
    data,  # TrialData or pl.DataFrame
    cell_idx: int,
    trial_type: str,
    session_stats: dict = None,
    cue_regions: dict = None,
    config: dict = None,
    bin_size_cm: int = 5,
    figsize: tuple = (12, 6),
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    show: bool = True
) -> Figure:
    """
    Plot all trials + session average for a single cell.
    
    Parameters
    ----------
    data : TrialData or pl.DataFrame
        Trial-indexed data with 'binned_signals'
    cell_idx : int
        Cell index to plot
    trial_type : str
        Filter to this trial type ('ABC', 'ABDC', etc.)
    session_stats : dict, optional
        From compute_session_averages() - if None, computed here
    cue_regions : dict, optional
        From get_cue_regions() - if None and show_cues=True, extracted here
    config : dict, optional
        Experiment config
    bin_size_cm : int
        Bin size for x-axis
    figsize : tuple
        Figure size
    alpha_trials : float
        Transparency for individual trials
    show_cues : bool
        Show cue region shading
    
    Returns
    -------
    Figure
    """
    import polars as pl
    
    # Handle TrialData or DataFrame input
    if hasattr(data, 'trial_df'):
        trial_df = data.trial_df
        if config is None:
            config = data.config
    else:
        trial_df = data
    
    trials = trial_df.filter(pl.col('trial_type') == trial_type)
    
    if len(trials) == 0:
        print(f"No trials found for type '{trial_type}'")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    shared_params(ax, trial_type, config, show_cues)

    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'Cell {cell_idx} - {trial_type}', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()

    if show and fig is not None:
        plt.show()

    return fig


def plot_cell_comparison(
    data,  # TrialData or pl.DataFrame
    cell_idx: int,
    session_stats: dict = None,
    config: dict = None,
    bin_size_cm: int = 5,
    figsize: tuple = (14, 10),
    alpha_trials: float = 0.3,
    show_cues: bool = True,
    show: bool = True,
) -> Figure:
    """
    Compare all trial types for a single cell (stacked subplots).
    
    Parameters
    ----------
    data : TrialData or pl.DataFrame
        Trial-indexed data
    cell_idx : int
        Cell index
    session_stats : dict, optional
        From compute_session_averages()
    config : dict, optional
        Experiment config
    bin_size_cm : int
        Bin size
    figsize : tuple
        Figure size
    alpha_trials : float
        Individual trial transparency
    show_cues : bool
        Show cue regions
    
    Returns
    -------
    Figure
    """
    import polars as pl
    from create_trial_based_df import compute_session_averages
    
    # Handle TrialData or DataFrame input
    if hasattr(data, 'trial_df'):
        trial_df = data.trial_df
        if config is None:
            config = data.config
    else:
        trial_df = data
    
    trial_types = sorted(trial_df['trial_type'].unique().to_list())
    
    if len(trial_types) == 0:
        print("No trial types found")
        return None
    
    if len(trial_types) == 1:
        figsize = (figsize[0], figsize[1] // 2)
    
    fig, axes = plt.subplots(len(trial_types), 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    if session_stats is None:
        session_stats = compute_session_averages(trial_df, by_trial_type=True)
    
    for ax, trial_type in zip(axes, trial_types):
        trials = trial_df.filter(pl.col('trial_type') == trial_type)
        
        shared_params(ax, trial_type, config, show_cues)
        
        # Individual trials
        trial_color = TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')
        for row in trials.iter_rows(named=True):
            binned = row['binned_signals']
            if binned is None or len(binned) == 0:
                continue
            cell_activity = binned[:, cell_idx]
            x = np.arange(len(cell_activity)) * bin_size_cm + (bin_size_cm/2)  #to plot the average activity at the
            # center of the bins, not the edges
            ax.plot(x, cell_activity, color=trial_color, linewidth=1,
                    alpha=alpha_trials, zorder=2)
        
        # Session average
        if trial_type in session_stats:
            avg = session_stats[trial_type]['session_avg'][:, cell_idx]
            sem = session_stats[trial_type]['session_sem'][:, cell_idx]
            x_avg = np.arange(len(avg)) * bin_size_cm + (bin_size_cm/2)
            n_trials = session_stats[trial_type]['n_trials']
            
            avg_color = TRIAL_TYPE_COLORS_DARK.get(trial_type, '#0A4D68')
            ax.plot(x_avg, avg, color=avg_color, linewidth=3.5,
                    label=f'Average (n={n_trials})', zorder=4)
            ax.fill_between(x_avg, avg - sem, avg + sem,
                            color=avg_color, alpha=0.3, zorder=3)


        ax.set_ylabel('ΔF/F', fontsize=12)
        ax.set_title(f'{trial_type}', fontsize=12, fontweight='bold', loc='left')
        ax.legend(frameon=False, fontsize=10, loc='upper right',
                  bbox_to_anchor=(1.0, 1.1))

    
    axes[-1].set_xlabel('Position (cm)', fontsize=12)
    
    plt.suptitle(f'Cell {cell_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show and fig is not None:
        plt.show()

    return fig



# QUICK PLOTTING

def quick_plot(
    data,  # TrialData
    cell_idx: int,
    plot_type: str = 'auto',
    show: bool = True,
) -> Optional[Figure]:
    """
    Quick plotting function for exploring cells.
    
    Parameters
    ----------
    data : TrialData
        From process_session() or load_trial_data()
    cell_idx : int
        Cell index
    plot_type : str
        'auto': Choose based on available trial types
        'comparison': Stacked subplots for all trial types
        'abc', 'abdc', etc.: Single trial type
    show : bool
        Call plt.show()
    
    Returns
    -------
    Figure
    """
    from create_trial_based_df import compute_session_averages
    
    trial_types = data.trial_types
    print(f"Available trial types: {trial_types}")
    
    session_stats = compute_session_averages(data.trial_df, by_trial_type=True)
    
    if plot_type == 'auto':
        if len(trial_types) >= 2:
            plot_type = 'comparison'
        else:
            plot_type = trial_types[0].lower()
    
    if plot_type == 'comparison':
        fig = plot_cell_comparison(
            data.trial_df, cell_idx,
            session_stats=session_stats,
            config=data.config
        )
    elif plot_type.upper() in trial_types:
        fig = plot_single_cell(
            data.trial_df, cell_idx,
            trial_type=plot_type.upper(),
            session_stats=session_stats,
            config=data.config
        )
    else:
        print(f"Unknown plot_type: {plot_type}")
        print(f"Options: 'auto', 'comparison', or trial type name ({trial_types})")
        return None
    
    if show and fig is not None:
        plt.show()
    
    return fig



# SAVE

def save_figure(fig: Figure, path: Path, dpi: int = 150):
    """Save figure to file (supports .png, .pdf, .svg)."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")



if __name__ == "__main__":
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects')

    experiment_config_path = session_root / '26_explore/experiment_configuration.yaml'

    data = load_trial_data(session_root / '26_explore',
                           config_path=experiment_config_path)  #the trial data class, includes the metadata (config)

    for i in range(10):
        quick_plot(data, cell_idx=i)






    #sanity check for offset correction
    cell_idx = 0
    colors = {'ABC': '#2E86AB', 'ABDC': '#A23B72'}

    # Load original frame-level data
    original_df = pl.read_ipc('/Users/cs963/Desktop/sun_lab_projects/26_explore/2025-09-16-18-44-32-476061.feather')
    original_df = original_df.filter(pl.col('system_state') == 'run').sort('frame')



    fig, ax = plt.subplots(figsize=(20, 6))

    # Original: raw distance_cm vs signal
    for trial_num in original_df['trial'].unique().sort():
        trial_data = original_df.filter(pl.col('trial') == trial_num).sort('frame')
        distance = trial_data['distance_cm'].to_numpy()
        signals = np.vstack(trial_data['single_day_f'].to_list())
        ax.plot(distance, signals[:, cell_idx], color='black', linewidth=0.5, alpha=0.5)

    # Corrected: also use distance_cm (from the frame-level corrected_df, not trial_df)
    # Run just step 1 to get the corrected frame-level data:
    from create_trial_based_df import fix_cue_offset, load_experiment_config

    corrected_df = fix_cue_offset(original_df, data.config, system_state='run')

    original_count = original_df.shape[0]
    corrected_count = corrected_df.shape[0]
    expected_loss = original_df.filter(
        (pl.col('trial') == original_df['trial'].unique().sort()[0]) |
        (pl.col('trial') == original_df['trial'].unique().sort()[-1])
    ).shape[0]

    print(f"Original frames: {original_count}")
    print(f"Corrected frames: {corrected_count}")
    print(f"Expected loss (first+last trial): {expected_loss}")
    print(f"Actual loss: {original_count - corrected_count}")
    print(f"Unaccounted missing: {(original_count - corrected_count) - expected_loss}")
    for trial_num in corrected_df['trial'].unique().sort():
        trial_data = corrected_df.filter(pl.col('trial') == trial_num).sort('frame')
        distance = trial_data['distance_cm'].to_numpy()
        signals = np.vstack(trial_data['single_day_f'].to_list())
        trial_type = trial_data['trial_type'][0]
        ax.plot(distance, signals[:, cell_idx], color=colors.get(trial_type, 'gray'),
                linewidth=0.5)
        # Mark corrected trial boundaries
        ax.axvline(distance[0], color='red', linewidth=0.5, alpha=0.5, linestyle='--')

    # Mark original trial boundaries
    for trial_num in original_df['trial'].unique().sort():
        trial_data = original_df.filter(pl.col('trial') == trial_num).sort('frame')
        ax.axvline(trial_data['distance_cm'][0], color='black', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Cumulative distance (cm)')
    ax.set_ylabel('ΔF/F')
    ax.set_title(f'Cell {cell_idx} - Black lines: original boundaries, Red dashed: corrected boundaries')
    plt.tight_layout()
    plt.show()

    #
    # plot_single_cell(data, cell_idx=0, trial_type='ABC')