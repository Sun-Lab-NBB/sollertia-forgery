"""
Transition Dynamics Analysis for UMAP Embeddings

Analyzes how neural trajectories behave at divergence points (bifurcations, or in the future connections/merges)
and how this changes over learning.

Designed to work with embedding and metadata from umap_plotting.py:
    - embedding: (n_frames, n_dims) UMAP coordinates
    - metadata: dict with 'trial', 'trial_type', 'distance', 'cue', 'speed', 'track_length'

Key analyses:
    1. Velocity/curvature at divergence region
    2. Trial-to-trial variability over learning
    3. Divergence point between trial types
    4. Recurrence structure
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from typing import Literal
from pathlib import Path


import sys
sys.path.insert(0, '/Users/cs963/Desktop/sun_lab/sl-forgery/src/sl_forgery/analysis/')


# CORE METRICS

def compute_velocity(trajectory: np.ndarray, dt: float = 0.1, smooth_sigma: float = 2
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity (speed and direction) at each timepoint.
    
    Parameters
    ----------
    trajectory : array (n_points, n_dims)
        Embedding coordinates over time
    dt : float
        Time step between frames (0.1 for 10Hz)
    smooth_sigma : float
        Gaussian smoothing sigma (in frames) to reduce noise
    
    Returns
    -------
    speed : array (n_points,)
        Scalar speed at each point (units: embedding units / second)
    velocity : array (n_points, n_dims)
        Velocity vector at each point
    """
    n_dims = trajectory.shape[1]
    
    # Smooth trajectory first
    traj_smooth = np.column_stack([
        gaussian_filter1d(trajectory[:, d], smooth_sigma) 
        for d in range(n_dims)
    ])
    
    # Velocity as finite difference
    velocity = np.gradient(traj_smooth, dt, axis=0)
    
    # Speed is magnitude
    speed = np.linalg.norm(velocity, axis=1)
    
    return speed, velocity


def compute_curvature(trajectory: np.ndarray, dt: float = 0.1, smooth_sigma: float = 2
                      ) -> np.ndarray:
    """
    Compute curvature (how sharply the trajectory bends) at each point.
    
    For 2D: kappa = |v x a| / |v|^3
    For 3D: kappa = |v x a| / |v|^3 (cross product magnitude)
    
    Parameters
    ----------
    trajectory : array (n_points, n_dims)
    dt : float
        Time step
    smooth_sigma : float
        Smoothing sigma
    
    Returns
    -------
    curvature : array (n_points,)
        Curvature at each point (higher = sharper turn)
    """
    n_dims = trajectory.shape[1]
    
    # Smooth
    traj_smooth = np.column_stack([
        gaussian_filter1d(trajectory[:, d], smooth_sigma)
        for d in range(n_dims)
    ])
    
    # Velocity (first derivative)
    v = np.gradient(traj_smooth, dt, axis=0)
    
    # Acceleration (second derivative)
    a = np.gradient(v, dt, axis=0)
    
    # Speed
    speed = np.linalg.norm(v, axis=1)
    
    if n_dims == 2:
        # 2D curvature: |v_x * a_y - v_y * a_x| / speed^3
        cross = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0])
    else:
        # 3D curvature: |v x a| / speed^3
        cross = np.linalg.norm(np.cross(v, a), axis=1)
    
    # Avoid division by zero
    curvature = np.divide(cross, speed**3, 
                          out=np.zeros_like(cross),
                          where=speed > 1e-10)
    
    return curvature


def interpolate_trial(trial_embedding, trial_distance, common_positions):
    """Interpolate embedding onto common position grid."""
    order = np.argsort(trial_distance)
    pos_sorted = trial_distance[order]
    emb_sorted = trial_embedding[order]

    n_dims = emb_sorted.shape[1]
    interpolated = np.column_stack([
        np.interp(common_positions, pos_sorted, emb_sorted[:, d])
        for d in range(n_dims)
    ])
    return interpolated


# TRIAL SEGMENTATION

def segment_trials(embedding: np.ndarray, metadata: dict
                   ) -> dict[int, dict]:
    """
    Segment continuous embedding into individual trials.
    
    Parameters
    ----------
    embedding : array (n_frames, n_dims)
    metadata : dict with 'trial', 'trial_type', 'distance', 'cue', 'speed'
    
    Returns
    -------
    trials : dict
        {trial_id: {
            'embedding': array (n_frames_in_trial, n_dims),
            'trial_type': str,
            'distance': array,
            'cue': array,
            'speed': array,
            'indices': array (original indices in full embedding)
        }}
    """
    trials = {}
    unique_trials = np.unique(metadata['trial'])
    
    for trial_id in unique_trials:
        mask = metadata['trial'] == trial_id
        indices = np.where(mask)[0]
        
        trials[trial_id] = {
            'embedding': embedding[mask],
            'trial_type': metadata['trial_type'][mask][0],  # all same within trial
            'distance': metadata['distance'][mask],
            'cue': metadata['cue'][mask],
            'cue_id': metadata['cue_id'][mask],
            'speed': metadata['speed'][mask],
            'indices': indices,
        }
    
    return trials


def get_decision_region(trial_data: dict, 
                        decision_cues: list | None = None,
                        position_range: tuple[float, float] | None = None,
                        cue_id: str = 'cue'
                        ) -> dict | None:
    """
    Extract the divergence region from a single trial.
    
    Parameters
    ----------
    trial_data : dict
        Single trial from segment_trials()
    decision_cues : list
        Cue IDs that define the divergence region (e.g., [2, 0] for B and gray)
    position_range : tuple
        (min_pos, max_pos) to define divergence region by position
    cue_id: the string label (B, 0b)
    
    Returns
    -------
    region : dict or None
        Same structure as trial_data but only for divergence region frames.
        Returns None if region is empty.
    """
    if decision_cues is not None:
        mask = np.isin(trial_data[cue_id], decision_cues)
    elif position_range is not None:
        mask = ((trial_data['distance'] >= position_range[0]) & 
                (trial_data['distance'] <= position_range[1]))
    else:
        raise ValueError("Must specify decision_cues or position_range")
    
    if not mask.any():
        return None
    
    return {
        'embedding': trial_data['embedding'][mask],
        'trial_type': trial_data['trial_type'],
        'distance': trial_data['distance'][mask],
        'cue': trial_data['cue'][mask],
        'speed': trial_data['speed'][mask],
        'indices': trial_data['indices'][mask],
    }


# DECISION REGION METRICS

def extract_decision_metrics(embedding: np.ndarray, 
                             metadata: dict,
                             decision_cues: list | None = None,
                             position_range: tuple[float, float] | None = None,
                             dt: float = 0.1,
                             smooth_sigma: float = 2,
                             cue_id: str = 'cue',
                             ) -> dict:
    """
    Extract dynamics metrics from the divergence region for all trials.
    
    Parameters
    ----------
    embedding : array (n_frames, n_dims)
    metadata : dict
    decision_cues : list
        Cue IDs defining divergence region
    position_range : tuple
        Alternative: (min_pos, max_pos) for divergence region
    dt : float
        Frame interval (seconds)
    smooth_sigma : float
        Smoothing for velocity/curvature
    
    Returns
    -------
    metrics : dict
        {
            'trial_id': array,
            'trial_type': array,
            'trial_number': array (temporal order, for learning analysis),
            'mean_velocity': array,
            'max_velocity': array,
            'mean_curvature': array,
            'max_curvature': array,
            'path_length': array,
            'direct_distance': array,
            'tortuosity': array,
            'mean_speed_behavioral': array,
            'n_frames': array,
        }
    """
    trials = segment_trials(embedding, metadata)
    
    metrics = {
        'trial_id': [],
        'trial_type': [],
        'trial_number': [],
        'mean_velocity': [],
        'max_velocity': [],
        'mean_curvature': [],
        'max_curvature': [],
        'path_length': [],
        'direct_distance': [],
        'tortuosity': [],
        'mean_speed_behavioral': [],
        'n_frames': [],
    }
    
    # Sort trials by ID to get temporal order
    sorted_trial_ids = sorted(trials.keys())
    
    for trial_num, trial_id in enumerate(sorted_trial_ids):
        trial_data = trials[trial_id]
        
        # Get divergence region
        region = get_decision_region(trial_data, decision_cues, position_range, cue_id)
        
        if region is None or len(region['embedding']) < 3:
            # Skip trials with insufficient data in divergence region
            continue
        
        traj = region['embedding']
        
        # Velocity metrics
        speed, velocity = compute_velocity(traj, dt, smooth_sigma)
        
        # Curvature metrics
        curvature = compute_curvature(traj, dt, smooth_sigma)
        
        # Path metrics
        diffs = np.diff(traj, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        direct_distance = np.linalg.norm(traj[-1] - traj[0])
        tortuosity = path_length / direct_distance if direct_distance > 1e-10 else np.nan
        
        # Store
        metrics['trial_id'].append(trial_id)
        metrics['trial_type'].append(region['trial_type'])
        metrics['trial_number'].append(trial_num)
        metrics['mean_velocity'].append(np.mean(speed))
        metrics['max_velocity'].append(np.max(speed))
        metrics['mean_curvature'].append(np.mean(curvature))
        metrics['max_curvature'].append(np.max(curvature))
        metrics['path_length'].append(path_length)
        metrics['direct_distance'].append(direct_distance)
        metrics['tortuosity'].append(tortuosity)
        metrics['mean_speed_behavioral'].append(np.mean(region['speed']))
        metrics['n_frames'].append(len(traj))
    
    # Convert to arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    return metrics


# TRIAL-TO-TRIAL VARIABILITY

def compute_trajectory_variability(embedding: np.ndarray,
                                   metadata: dict,
                                   decision_cues: list | None = None,
                                   position_range: tuple[float, float] | None = None,
                                   window_size: int = 10,
                                   step_size: int = 5,
                                   cue_id: str = 'cue',
                                   ) -> dict:
    """
    Compute trial-to-trial variability in sliding windows.
    
    For each window, computes the variance of trajectories around the mean
    trajectory, separately for each trial type.
    
    Parameters
    ----------
    embedding, metadata : as usual
    decision_cues, position_range : divergence region definition
    window_size : int
        Number of trials per window
    step_size : int
        Trials to advance between windows
    
    Returns
    -------
    variability : dict
        {
            'window_centers': array (trial numbers),
            'ABC_variability': array,
            'ABDC_variability': array,
            'ABC_n_trials': array,
            'ABDC_n_trials': array,
        }
    """
    trials = segment_trials(embedding, metadata)
    sorted_trial_ids = sorted(trials.keys())
    
    results = {
        'window_centers': [],
        'ABC_variability': [],
        'ABDC_variability': [],
        'ABC_n_trials': [],
        'ABDC_n_trials': [],
    }
    
    for start in range(0, len(sorted_trial_ids) - window_size + 1, step_size):
        end = start + window_size
        window_ids = sorted_trial_ids[start:end]
        
        results['window_centers'].append((start + end) / 2)
        
        # Separate by trial type
        for trial_type in ['ABC', 'ABDC']:
            # Get divergence region trajectories for this type
            trajs = []
            for tid in window_ids:
                if trials[tid]['trial_type'] != trial_type:
                    continue
                region = get_decision_region(trials[tid], decision_cues, position_range, cue_id)
                if region is not None and len(region['embedding']) >= 3:
                    trajs.append(region['embedding'])
            
            if len(trajs) >= 2:
                # Align to same length (truncate to shortest)
                min_len = min(len(t) for t in trajs)
                aligned = np.array([t[:min_len] for t in trajs])
                
                # Mean trajectory
                mean_traj = np.mean(aligned, axis=0)
                
                # Variability: mean distance from mean trajectory
                deviations = aligned - mean_traj
                variability = np.mean(np.linalg.norm(deviations, axis=2))
                
                results[f'{trial_type}_variability'].append(variability)
                results[f'{trial_type}_n_trials'].append(len(trajs))
            else:
                results[f'{trial_type}_variability'].append(np.nan)
                results[f'{trial_type}_n_trials'].append(len(trajs))
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


# =============================================================================
# DIVERGENCE ANALYSIS
# =============================================================================

def compute_divergence_over_learning(embedding: np.ndarray,
                                     metadata: dict,
                                     window_size: int = 10,
                                     step_size: int = 5,
                                     threshold: float = 0.3
                                     ) -> dict:
    """
    Track when ABC and ABDC trajectories diverge, over learning.
    
    For each window, computes the mean trajectory for each trial type
    and finds the timepoint (position) where they diverge.
    
    Parameters
    ----------
    embedding, metadata : as usual
    window_size, step_size : window parameters
    threshold : float
        Distance threshold for considering trajectories "diverged"
    
    Returns
    -------
    divergence : dict
        {
            'window_centers': array,
            'divergence_position': array (position in cm where divergence occurs),
            'divergence_timepoint': array (frame index within trial),
            'max_divergence': array (maximum separation achieved),
        }
    """
    trials = segment_trials(embedding, metadata)
    sorted_trial_ids = sorted(trials.keys())
    
    results = {
        'window_centers': [],
        'divergence_position': [],
        'divergence_timepoint': [],
        'max_divergence': [],
    }
    
    for start in range(0, len(sorted_trial_ids) - window_size + 1, step_size):
        end = start + window_size
        window_ids = sorted_trial_ids[start:end]
        
        results['window_centers'].append((start + end) / 2)
        
        # Get all trials by type
        abc_trajs, abc_pos = [], []
        abdc_trajs, abdc_pos = [], []
        
        for tid in window_ids:
            t = trials[tid]
            if t['trial_type'] == 'ABC':
                abc_trajs.append(t['embedding'])
                abc_pos.append(t['distance'])
            else:
                abdc_trajs.append(t['embedding'])
                abdc_pos.append(t['distance'])
        
        if len(abc_trajs) < 2 or len(abdc_trajs) < 2:
            results['divergence_position'].append(np.nan)
            results['divergence_timepoint'].append(np.nan)
            results['max_divergence'].append(np.nan)
            continue
        
        # Align by truncating to shortest
        min_len = min(
            min(len(t) for t in abc_trajs),
            min(len(t) for t in abdc_trajs)
        )
        
        abc_aligned = np.array([t[:min_len] for t in abc_trajs])
        abdc_aligned = np.array([t[:min_len] for t in abdc_trajs])
        abc_pos_aligned = np.array([p[:min_len] for p in abc_pos])
        
        # Mean trajectories
        mean_abc = np.mean(abc_aligned, axis=0)
        mean_abdc = np.mean(abdc_aligned, axis=0)
        mean_pos = np.mean(abc_pos_aligned, axis=0)
        
        # Distance between means over time
        distances = np.linalg.norm(mean_abc - mean_abdc, axis=1)
        
        # Find divergence point
        diverged = np.where(distances > threshold)[0]
        
        if len(diverged) > 0:
            div_idx = diverged[0]
            results['divergence_position'].append(mean_pos[div_idx])
            results['divergence_timepoint'].append(div_idx)
        else:
            results['divergence_position'].append(np.nan)
            results['divergence_timepoint'].append(np.nan)
        
        results['max_divergence'].append(np.max(distances))
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


# RECURRENCE ANALYSIS
# recurrence is part of dynamic systems theory, asks when does the system revist states

def compute_recurrence_plot(trajectory: np.ndarray,
                            threshold_percentile: float = 20
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute recurrence plot from trajectory.
    
    Parameters
    ----------
    trajectory : array (n_points, n_dims)
    threshold_percentile : float
        Distances below this percentile are marked as recurrent
    
    Returns
    -------
    recurrence : array (n_points, n_points)
        Binary recurrence matrix
    distances : array (n_points, n_points)
        Raw distance matrix
    """
    distances = squareform(pdist(trajectory))
    threshold = np.percentile(distances, threshold_percentile)
    recurrence = distances < threshold
    
    return recurrence, distances


def compute_recurrence_metrics(recurrence: np.ndarray) -> dict:
    """
    Extract summary metrics from a recurrence plot.
    
    Parameters
    ----------
    recurrence : array (n, n)
        Binary recurrence matrix
    
    Returns
    -------
    metrics : dict
        'recurrence_rate': fraction of recurrent points
        'determinism': fraction of recurrent points in diagonal lines
        'mean_diagonal_length': average length of diagonal structures
    """
    n = recurrence.shape[0]
    
    # Recurrence rate (excluding main diagonal)
    mask = ~np.eye(n, dtype=bool)
    recurrence_rate = recurrence[mask].sum() / mask.sum()
    
    # Determinism: fraction of recurrent points forming diagonal lines (length >= 2)
    # This is a simplified version - full RQA would use proper line detection
    diag_count = 0
    total_recurrent = recurrence[mask].sum()
    
    for offset in range(1, n):
        diag = np.diag(recurrence, k=offset)
        # Count points that are part of a line (preceded or followed by recurrence)
        if len(diag) >= 2:
            is_line = diag[:-1] & diag[1:]
            diag_count += is_line.sum() * 2  # Both points in the pair
    
    determinism = diag_count / total_recurrent if total_recurrent > 0 else 0
    
    return {
        'recurrence_rate': recurrence_rate,
        'determinism': min(determinism, 1.0),  # Cap at 1
    }


def compute_recurrence_over_learning(embedding: np.ndarray,
                                     metadata: dict,
                                     window_size: int = 15,
                                     step_size: int = 5,
                                     threshold_percentile: float = 20
                                     ) -> dict:
    """
    Track recurrence structure over learning.
    
    Computes recurrence plots for sliding windows and extracts metrics.
    """
    trials = segment_trials(embedding, metadata)
    sorted_trial_ids = sorted(trials.keys())
    
    results = {
        'window_centers': [],
        'recurrence_rate': [],
        'determinism': [],
    }
    
    for start in range(0, len(sorted_trial_ids) - window_size + 1, step_size):
        end = start + window_size
        window_ids = sorted_trial_ids[start:end]
        
        results['window_centers'].append((start + end) / 2)
        
        # Concatenate trials in window
        trajs = [trials[tid]['embedding'] for tid in window_ids]
        combined = np.vstack(trajs)
        
        # Compute recurrence
        recurrence, _ = compute_recurrence_plot(combined, threshold_percentile)
        metrics = compute_recurrence_metrics(recurrence)
        
        results['recurrence_rate'].append(metrics['recurrence_rate'])
        results['determinism'].append(metrics['determinism'])
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_decision_metrics_over_learning(metrics: dict,
                                        save_path: Path | str | None = None
                                        ) -> plt.Figure:
    """
    Plot divergence region metrics over learning (trial number).
    
    Parameters
    ----------
    metrics : dict
        Output from extract_decision_metrics()
    save_path : Path, optional
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    trial_nums = metrics['trial_number']
    is_abdc = metrics['trial_type'] == 'ABDC'
    
    # Color by trial type
    colors = np.where(is_abdc, 'coral', 'steelblue')
    
    # 1. Velocity
    ax = axes[0, 0]
    ax.scatter(trial_nums, metrics['mean_velocity'], c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Velocity')
    ax.set_title('Velocity at Decision Point')
    _add_trendline(ax, trial_nums, metrics['mean_velocity'])
    
    # 2. Curvature
    ax = axes[0, 1]
    ax.scatter(trial_nums, metrics['mean_curvature'], c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Curvature')
    ax.set_title('Curvature at Decision Point')
    _add_trendline(ax, trial_nums, metrics['mean_curvature'])
    
    # 3. Tortuosity
    ax = axes[0, 2]
    valid = ~np.isnan(metrics['tortuosity'])
    ax.scatter(trial_nums[valid], metrics['tortuosity'][valid], 
               c=colors[valid], alpha=0.6, s=30)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Straight line')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Tortuosity')
    ax.set_title('Path Tortuosity (1 = direct)')
    _add_trendline(ax, trial_nums[valid], metrics['tortuosity'][valid])
    
    # 4. Velocity by trial type (separate trends)
    ax = axes[1, 0]
    ax.scatter(trial_nums[~is_abdc], metrics['mean_velocity'][~is_abdc], 
               c='steelblue', alpha=0.6, s=30, label='ABC')
    ax.scatter(trial_nums[is_abdc], metrics['mean_velocity'][is_abdc],
               c='coral', alpha=0.6, s=30, label='ABDC')
    _add_trendline(ax, trial_nums[~is_abdc], metrics['mean_velocity'][~is_abdc], 'steelblue')
    _add_trendline(ax, trial_nums[is_abdc], metrics['mean_velocity'][is_abdc], 'coral')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Velocity')
    ax.set_title('Velocity by Trial Type')
    ax.legend()
    
    # 5. Path length
    ax = axes[1, 1]
    ax.scatter(trial_nums, metrics['path_length'], c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Path Length')
    ax.set_title('Path Length in Decision Region')
    _add_trendline(ax, trial_nums, metrics['path_length'])
    
    # 6. Legend/summary
    ax = axes[1, 2]
    ax.scatter([], [], c='steelblue', s=50, label='ABC trials')
    ax.scatter([], [], c='coral', s=50, label='ABDC trials')
    ax.legend(loc='center', fontsize=12)
    ax.axis('off')
    
    # Summary text
    n_abc = (~is_abdc).sum()
    n_abdc = is_abdc.sum()
    summary = f"Total trials: {len(trial_nums)}\nABC: {n_abc}, ABDC: {n_abdc}"
    ax.text(0.5, 0.3, summary, ha='center', fontsize=11, transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_variability_over_learning(variability: dict,
                                   save_path: Path | str | None = None
                                   ) -> plt.Figure:
    """Plot trial-to-trial variability over learning."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    centers = variability['window_centers']
    
    ax.plot(centers, variability['ABC_variability'], 
            'o-', color='steelblue', label='ABC', markersize=6)
    ax.plot(centers, variability['ABDC_variability'],
            'o-', color='coral', label='ABDC', markersize=6)
    
    ax.set_xlabel('Trial Number (window center)')
    ax.set_ylabel('Trajectory Variability')
    ax.set_title('Trial-to-Trial Variability Over Learning')
    ax.legend()
    
    # Add trend lines
    _add_trendline(ax, centers, variability['ABC_variability'], 'steelblue')
    _add_trendline(ax, centers, variability['ABDC_variability'], 'coral')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_divergence_over_learning(divergence: dict,
                                  save_path: Path | str | None = None
                                  ) -> plt.Figure:
    """Plot divergence analysis over learning."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    centers = divergence['window_centers']
    
    # Divergence position
    ax = axes[0]
    valid = ~np.isnan(divergence['divergence_position'])
    ax.plot(centers[valid], divergence['divergence_position'][valid],
            'ko-', markersize=6)
    ax.set_xlabel('Trial Number (window center)')
    ax.set_ylabel('Position (cm)')
    ax.set_title('Position Where Trajectories Diverge')
    ax.invert_yaxis()  # Earlier (lower position) = better separation
    
    # Max divergence
    ax = axes[1]
    valid = ~np.isnan(divergence['max_divergence'])
    ax.plot(centers[valid], divergence['max_divergence'][valid],
            'ko-', markersize=6)
    ax.set_xlabel('Trial Number (window center)')
    ax.set_ylabel('Max Distance')
    ax.set_title('Maximum Separation Between Trial Types')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_recurrence_comparison(embedding: np.ndarray,
                               metadata: dict,
                               early_trials: int = 10,
                               late_trials: int = 10,
                               threshold_percentile: float = 20,
                               save_path: Path | str | None = None
                               ) -> plt.Figure:
    """
    Compare recurrence plots: early vs late in session.
    """
    trials = segment_trials(embedding, metadata)
    sorted_trial_ids = sorted(trials.keys())
    
    # Early trials
    early_ids = sorted_trial_ids[:early_trials]
    early_trajs = [trials[tid]['embedding'] for tid in early_ids]
    early_combined = np.vstack(early_trajs)

    # Middle trials
    mid_start = len(sorted_trial_ids) // 2 - early_trials // 2
    mid_ids = sorted_trial_ids[mid_start:mid_start + early_trials]
    mid_trajs = [trials[tid]['embedding'] for tid in mid_ids]
    mid_combined = np.vstack(mid_trajs)
    
    # Late trials
    late_ids = sorted_trial_ids[-late_trials:]
    late_trajs = [trials[tid]['embedding'] for tid in late_ids]
    late_combined = np.vstack(late_trajs)
    
    # Compute recurrence
    rec_early, _ = compute_recurrence_plot(early_combined, threshold_percentile)
    rec_mid, _ = compute_recurrence_plot(mid_combined, threshold_percentile)
    rec_late, _ = compute_recurrence_plot(late_combined, threshold_percentile)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(rec_early, cmap='binary', origin='lower', aspect='auto')
    axes[0].set_title(f'Early Learning (trials {early_ids[0]}-{early_ids[-1]})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Time')

    axes[1].imshow(rec_mid, cmap='binary', origin='lower', aspect='auto')
    axes[1].set_title(f'Mid Learning (trials {mid_ids[0]}-{mid_ids[-1]})')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Time')

    axes[2].imshow(rec_late, cmap='binary', origin='lower', aspect='auto')
    axes[2].set_title(f'Late Learning (trials {late_ids[0]}-{late_ids[-1]})')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Time')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_recurrence_with_umap(
    sessions: list[dict],
    color_by: Literal['cue', 'trial_type'] = 'cue',
    cue_colors: dict | None = None,
    trial_type_colors: dict | None = None,
    threshold_percentile: float = 20,
    umap_dims: tuple[int, int] = (0, 1),
    figsize_per_row: tuple[float, float] = (14, 5),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """
    Side-by-side UMAP trajectories and recurrence plots for N sessions.

    Parameters
    ----------
    sessions : list of dict
        Each dict must have:
            'embedding': array (n_frames, n_dims)
            'metadata': dict with 'cue', 'trial_type', 'trial'
            'label': str (e.g., 'Day 1', 'Day 3 - Late Learning')
    color_by : 'cue' or 'trial_type'
        How to color the UMAP scatter.
    cue_colors : dict, optional
        {cue_value: color}. If None, uses a default palette.
    trial_type_colors : dict, optional
        {trial_type: color}. If None, defaults to {'ABC': 'steelblue', 'ABDC': 'coral'}.
    threshold_percentile : float
        Percentile for recurrence threshold.
    umap_dims : tuple of int
        Which two UMAP dimensions to plot (default: first two).
    figsize_per_row : tuple
        (width, height) per session row.
    save_path : Path or str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    n_sessions = len(sessions)

    if trial_type_colors is None:
        trial_type_colors = {'ABC': 'steelblue', 'ABDC': 'coral'}

    fig, axes = plt.subplots(
        n_sessions, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_sessions),
        squeeze=False,
    )

    d0, d1 = umap_dims

    for i, session in enumerate(sessions):
        emb = session['embedding']
        meta = session['metadata']
        label = session.get('label', f'Session {i + 1}')

        # --- Left: UMAP scatter with trajectory lines ---
        ax_umap = axes[i, 0]

        # Draw trial trajectories as light gray lines
        trials = segment_trials(emb, meta)
        for tid in sorted(trials.keys()):
            t = trials[tid]['embedding']
            ax_umap.plot(t[:, d0], t[:, d1], color='lightgray', alpha=0.3, linewidth=0.5, zorder=1)

        # Scatter points
        if color_by == 'cue':
            cue_vals = meta['cue']
            unique_cues = np.unique(cue_vals)

            if cue_colors is None:
                default_cmap = plt.cm.tab10
                cue_colors_used = {c: default_cmap(j / max(len(unique_cues) - 1, 1))
                                   for j, c in enumerate(unique_cues)}
            else:
                cue_colors_used = cue_colors

            for cue in unique_cues:
                mask = cue_vals == cue
                ax_umap.scatter(
                    emb[mask, d0], emb[mask, d1],
                    c=[cue_colors_used.get(cue, 'gray')],
                    label=str(cue), s=10, alpha=0.6, zorder=2,
                )
            ax_umap.legend(fontsize=8, markerscale=2, loc='best')

        elif color_by == 'trial_type':
            tt = meta['trial_type']
            for ttype, color in trial_type_colors.items():
                mask = tt == ttype
                if mask.any():
                    ax_umap.scatter(
                        emb[mask, d0], emb[mask, d1],
                        c=color, label=ttype, s=10, alpha=0.6, zorder=2,
                    )
            ax_umap.legend(fontsize=8, markerscale=2, loc='best')

        ax_umap.set_xlabel(f'Dim {d0 + 1}')
        ax_umap.set_ylabel(f'Dim {d1 + 1}')
        ax_umap.set_title(f'{label}\nState Space Trajectory')

        # --- Right: Recurrence plot ---
        ax_rec = axes[i, 1]
        recurrence, _ = compute_recurrence_plot(emb, threshold_percentile)
        ax_rec.imshow(recurrence, cmap='binary', origin='lower', aspect='auto')
        ax_rec.set_xlabel('Time')
        ax_rec.set_ylabel('Time')
        ax_rec.set_title(f'{label}\nRecurrence Plot')

        # Add metrics as text
        rec_metrics = compute_recurrence_metrics(recurrence)
        ax_rec.text(
            0.02, 0.97,
            f"RR={rec_metrics['recurrence_rate']:.3f}\nDET={rec_metrics['determinism']:.2f}",
            transform=ax_rec.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _add_trendline(ax, x, y, color='red', alpha=0.5):
    """Add linear trend line to axis."""
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return
    x_valid, y_valid = np.array(x)[valid], np.array(y)[valid]
    z = np.polyfit(x_valid, y_valid, 1)
    p = np.poly1d(z)
    ax.plot(x_valid, p(x_valid), '--', color=color, alpha=alpha, linewidth=2)



# MAIN ANALYSIS FUNCTION (WRAPPER)

def run_full_analysis(embedding: np.ndarray,
                      metadata: dict,
                      decision_cues: list,
                      output_dir: Path | str | None = None,
                      window_size: int = 10,
                      step_size: int = 5,
                      dt: float = 0.1,
                      cue_id: str = 'cue',
                      ) -> dict:
    """
    Run complete transition dynamics analysis.
    
    Parameters
    ----------
    embedding : array (n_frames, n_dims)
    metadata : dict
    decision_cues : list
        Cue IDs defining divergence region (e.g., [2, 0] for B and gray)
    output_dir : Path, optional
        Directory to save figures
    window_size : int
        Trials per sliding window
    step_size : int
        Step between windows
    dt : float
        Frame interval (seconds)
    
    Returns
    -------
    results : dict
        All computed metrics and analysis results
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    print("Extracting divergence region metrics...")
    metrics = extract_decision_metrics(
        embedding, metadata, 
        decision_cues=decision_cues,
        dt=dt,
        cue_id=cue_id,
    )
    print(f"  Analyzed {len(metrics['trial_id'])} trials")
    
    print("Computing trial-to-trial variability...")
    variability = compute_trajectory_variability(
        embedding, metadata,
        decision_cues=decision_cues,
        window_size=window_size,
        step_size=step_size,
        cue_id=cue_id,
    )
    
    print("Computing divergence over learning...")
    divergence = compute_divergence_over_learning(
        embedding, metadata,
        window_size=window_size,
        step_size=step_size
    )
    
    print("Computing recurrence over learning...")
    recurrence = compute_recurrence_over_learning(
        embedding, metadata,
        window_size=window_size,
        step_size=step_size
    )
    
    # Generate plots
    print("Generating figures...")
    
    fig1 = plot_decision_metrics_over_learning(
        metrics, 
        save_path=output_dir / 'decision_metrics.png' if output_dir else None
    )
    plt.show()
    
    fig2 = plot_variability_over_learning(
        variability,
        save_path=output_dir / 'variability.png' if output_dir else None
    )
    plt.show()
    
    fig3 = plot_divergence_over_learning(
        divergence,
        save_path=output_dir / 'divergence.png' if output_dir else None
    )
    plt.show()
    
    fig4 = plot_recurrence_comparison(
        embedding, metadata,
        save_path=output_dir / 'recurrence_comparison.png' if output_dir else None
    )
    plt.show()
    
    print("Done!")
    
    return {
        'metrics': metrics,
        'variability': variability,
        'divergence': divergence,
        'recurrence': recurrence,
        'figures': [fig1, fig2, fig3, fig4],
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':

    from df_processing import (load_session_dir, get_session_prefix, load_processed_session, save_processed_session,
                               process_session, load_multiday_sessions)

    from umap_plotting import prepare_umap_data, compute_umap

    # import the cue-aligned data
    mouse_id = '26'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)
    date = '2025-09-11'

    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    parquet_path = behavior_path.parent / f'{prefix}_processed.parquet'

    if parquet_path.exists():
        print(f"Loading: {parquet_path}")
        data, metadata = load_processed_session(parquet_path)
    else:
        print("No processed file found, processing from raw...")  # OR if you want to process the session with
        # other system states, bc the default is to process by run
        behavior_df = pl.read_ipc(behavior_path)
        data, metadata = process_session(behavior_df, config)
        save_processed_session(data, behavior_path.parent, session_data, metadata)

    save_path = None  # Set to a Path to save figures

    # prepare data
    neural_data, metadata = prepare_umap_data(data, signal_column='multi_day_dff', max_frames=None)

    # compute umap
    embedding = compute_umap(neural_data, n_components=3, n_neighbors=50)

    # Debug: check what cues exist and if they match
    print("Unique cues in data:", np.unique(metadata['cue']))

    # Check if any frames match
    mask = np.isin(metadata['cue'], ['B', '0b'])
    print(f"Frames matching decision_cues: {mask.sum()} / {len(mask)}")

    # Check per trial
    from transition_dynamics import segment_trials, get_decision_region

    trials = segment_trials(embedding, metadata)

    # Look at first few trials
    for tid in list(trials.keys())[:5]:
        region = get_decision_region(trials[tid], decision_cues=['B', '0b'])
        n_frames = len(region['embedding']) if region else 0
        print(f"Trial {tid} ({trials[tid]['trial_type']}): {n_frames} frames in divergence region")

    
    # Define divergence region by cue IDs
    # (check your cue mapping - these are placeholder values)
    decision_cues = ['B', '0b']  # B and gray zone
    
    # Run analysis
    results = run_full_analysis(
        embedding, 
        metadata,
        decision_cues=decision_cues,
        cue_id='cue_id',
        output_dir='./transition_analysis',
        dt=0.1,  # 10 Hz
    )
    
    # Access results
    metrics = results['metrics']
    print(f"Mean velocity trend: {metrics['mean_velocity']}")
