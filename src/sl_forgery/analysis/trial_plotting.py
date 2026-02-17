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


# =============================================================================
# COLOR CONFIGURATION
# =============================================================================

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


# =============================================================================
# CUE REGION PLOTTING
# =============================================================================

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
            ax.axvspan(start, end, alpha=alpha, color=color, zorder=1)
            
            # Label only first occurrence
            if show_labels and i == 0:
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


# =============================================================================
# PLACE FIELD PLOTS
# =============================================================================

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
    
    # Cue regions
    if show_cues:
        if cue_regions is None:
            from trial_based_df import get_cue_regions
            cue_regions = get_cue_regions(trial_df, trial_type)
        plot_cue_regions(ax, cue_regions, config)
    
    # Individual trials
    for row in trials.iter_rows(named=True):
        binned = row['binned_signals']
        if binned is None or len(binned) == 0:
            continue
        cell_activity = binned[:, cell_idx]
        x = np.arange(len(cell_activity)) * bin_size_cm
        ax.plot(x, cell_activity, color='gray', linewidth=1, alpha=alpha_trials, zorder=2)
    
    # Session average
    if session_stats is None:
        from trial_based_df import compute_session_averages
        session_stats = compute_session_averages(trial_df, by_trial_type=True)
    
    if trial_type in session_stats:
        avg = session_stats[trial_type]['session_avg'][:, cell_idx]
        sem = session_stats[trial_type]['session_sem'][:, cell_idx]
        x_avg = np.arange(len(avg)) * bin_size_cm
        n_trials = session_stats[trial_type]['n_trials']
        
        ax.plot(x_avg, avg, color='#E63946', linewidth=3,
                label=f'Session average (n={n_trials})', zorder=4)
        ax.fill_between(x_avg, avg - sem, avg + sem,
                        color='#E63946', alpha=0.3, zorder=3)
    
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('ΔF/F', fontsize=12)
    ax.set_title(f'Cell {cell_idx} - {trial_type}', fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, axis='y', zorder=0)
    
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
    from trial_based_df import get_cue_regions, compute_session_averages
    
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
        
        # Cue regions with labels
        if show_cues:
            cue_regions = get_cue_regions(trial_df, trial_type)
            plot_cue_regions(ax, cue_regions, config, show_labels=True, alpha=0.15)
            
            # Set x-ticks at nominal cue boundaries (from config)
            cue_width = 30  # default
            if config and 'cue_map' in config:
                # All cues have same width in current config
                cue_width = list(config['cue_map'].values())[0]
            
            # Get track length
            track_length = 180  # default
            if config and trial_type in config.get('trial_structures', {}):
                track_length = config['trial_structures'][trial_type]['trial_length_cm']
            elif trial_type in session_stats:
                track_length = session_stats[trial_type]['session_avg'].shape[0] * bin_size_cm
            
            # Generate nominal ticks: 0, 30, 60, 90, ... up to track length
            tick_positions = list(range(0, int(track_length) + 1, int(cue_width)))
            if tick_positions[-1] != track_length:
                tick_positions.append(int(track_length))
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(t) for t in tick_positions], fontsize=9)
        
        # Individual trials
        trial_color = TRIAL_TYPE_COLORS.get(trial_type, '#2E86AB')
        for row in trials.iter_rows(named=True):
            binned = row['binned_signals']
            if binned is None or len(binned) == 0:
                continue
            cell_activity = binned[:, cell_idx]
            x = np.arange(len(cell_activity)) * bin_size_cm
            ax.plot(x, cell_activity, color=trial_color, linewidth=1,
                    alpha=alpha_trials, zorder=2)
        
        # Session average
        if trial_type in session_stats:
            avg = session_stats[trial_type]['session_avg'][:, cell_idx]
            sem = session_stats[trial_type]['session_sem'][:, cell_idx]
            x_avg = np.arange(len(avg)) * bin_size_cm
            n_trials = session_stats[trial_type]['n_trials']
            
            avg_color = TRIAL_TYPE_COLORS_DARK.get(trial_type, '#0A4D68')
            ax.plot(x_avg, avg, color=avg_color, linewidth=3.5,
                    label=f'Average (n={n_trials})', zorder=4)
            ax.fill_between(x_avg, avg - sem, avg + sem,
                            color=avg_color, alpha=0.3, zorder=3)
        
        ax.set_ylabel('ΔF/F', fontsize=12)
        ax.set_title(f'{trial_type}', fontsize=12, fontweight='bold', loc='left')
        ax.legend(frameon=False, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, axis='y', zorder=0)
    
    axes[-1].set_xlabel('Position (cm)', fontsize=12)
    
    plt.suptitle(f'Cell {cell_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show and fig is not None:
        plt.show()

    return fig


# =============================================================================
# QUICK PLOTTING
# =============================================================================

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
    from trial_based_df import compute_session_averages
    
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


# =============================================================================
# SAVE UTILITIES
# =============================================================================

def save_figure(fig: Figure, path: Path, dpi: int = 150):
    """Save figure to file (supports .png, .pdf, .svg)."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")



if __name__ == "__main__":
    session_root = Path('/Users/cs963/Desktop/sun_lab_projects')

    experiment_config_path = Path(session + '/26_explore.meta.yaml')

    data = load_trial_data('/Users/cs963/Desktop/sun_lab_projects/26_explore/26_explore',
                           config_path=experiment_config_path)  #the trial data class, includes the metadata (config)

    # Re-bin with config for consistent sizes
    from trial_based_df import add_binned_signals, load_experiment_config, get_cue_regions

    config = load_experiment_config(experiment_config_path)

    print(data.trial_df.columns)
    with pl.Config(tbl_cols=100, tbl_rows=50):
        print(data.trial_df.head())

    cue_regions = get_cue_regions(data.trial_df, 'ABC', verbose=True)

    #quick_plot(data, cell_idx=0)
    #plot_single_cell(data, cell_idx=0, trial_type='ABC')