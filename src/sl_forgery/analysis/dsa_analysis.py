"""
Dynamical Similarity Analysis (DSA) Module

Compares neural dynamics between trial types, sessions, and animals using
fast Dynamical Similarity Analysis (Behrad et al., 2025; Ostrow et al.,
NeurIPS 2023). fastDSA embeds neural time series into a linear Koopman
space via delay embeddings + DMD with automatic rank selection (SVHT),
then compares linear operators with optimized Procrustes over Vector Fields.

Key analyses:
    1. Within-session: ABC vs ABDC dynamics (full trial or position-restricted)
    2. Cross-session: How dynamics evolve across days of learning
    3. Cross-animal: Compare dynamical regimes between mice at matched stages

Input: Frame-level processed DataFrames with multi_day_dff signal columns.
DSA operates on raw neural time series — NOT UMAP embeddings.

Dependencies: numpy, polars, matplotlib, scikit-learn, fastDSA.
Install fastDSA:
    git clone https://github.com/CMC-lab/fastDSA.git
    cd fastDSA && pip install -e .
"""

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

import plot_utils as pfmt
from df_processing import (
    find_session_dir,
    load_session_context,
    get_session_paths,
    load_processed_session,
    get_track_length,
)

try:
    from fastDSA.simdist import SimDistConfig, FastDSASimilarity
    HAS_FASTDSA = True
except ImportError:
    HAS_FASTDSA = False


def _check_fastdsa_installed():
    """Raise ImportError with install instructions if fastDSA is missing."""
    if not HAS_FASTDSA:
        raise ImportError(
            "fastDSA not installed. Install with:\n"
            "  git clone https://github.com/CMC-lab/fastDSA.git\n"
            "  cd fastDSA && pip install -e ."
        )


# CONFIGURATION

def make_dsa_config(
    n_delays: int = 15,
    delay_interval: int = 1,
    rank: int | None = None,
    method: str = 'ro',
    iters: int = 200,
    lr: float = 1e-2,
    device: str = 'cpu',
    verbose: bool = False,
) -> 'SimDistConfig':
    """Create a fastDSA configuration.

    Args:
        n_delays: Number of time delays for Hankel embedding.
        delay_interval: Interval between delays (in frames).
        rank: DMD rank. None = automatic SVHT selection (recommended).
        method: Optimizer — 'ro' (regularized), 'rim' (Riemannian),
            'land' (Landing), 'kw' (kernel Wasserstein).
        iters: Optimization iterations for similarity transform.
        lr: Learning rate.
        device: 'cpu' or 'cuda'.
        verbose: Print progress.

    Returns:
        SimDistConfig instance.
    """
    _check_fastdsa_installed()
    return SimDistConfig(
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=rank,
        method=method,
        iters=iters,
        lr=lr,
        device=device,
        verbose=verbose,
    )

# GET SESSIONS

# need these functions ot avoid the file crashing with load_multiday_sessions (all parquets are in RAM)
def extract_sessions(
    mouse_dir: Path,
    dates: list[str],
    signal_col: str = 'multi_day_dff',
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    trial_type: str | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load sessions one at a time, extract trial signals, discard raw data.

    Memory-efficient alternative to load_multiday_sessions() for DSA.
    Only keeps PCA-reduced trial matrices in RAM.

    Args:
        mouse_dir: Mouse-level directory path.
        dates: List of session date strings.
        signal_col: Column with neural signals.
        n_pca_components: PCA dimensions.
        min_speed: Speed filter.
        trial_type: Filter to one trial type. None = pool all.

    Returns:
        Dict mapping date -> list of trial arrays (channels, timepoints).
    """
    result = {}
    _columns = ['frame', 'trial', 'trial_type', 'experiment_state',
                'position', 'speed_cm_s', 'water_uL', signal_col]

    for date in sorted(dates):
        try:
            session_dir = find_session_dir(mouse_dir, date)
            session_data, _ = load_session_context(session_dir)
            paths = get_session_paths(session_dir, session_data)

            data = pl.read_parquet(paths['parquet'], columns=_columns)

            trials, _ = extract_trial_signals(
                data, signal_col=signal_col, trial_type=trial_type,
                n_pca_components=n_pca_components, min_speed=min_speed,
            )
            result[date] = trials
            del data
            print(f"  {date}: {len(trials)} trials extracted")

        except Exception as e:
            print(f"  WARNING: {date}: {e}")

    return result


# SIGNAL EXTRACTION

def extract_trial_signals(
    df: pl.DataFrame,
    signal_col: str = 'multi_day_dff',
    trial_type: str | None = None,
    min_speed: float = 0.0,
    exclude_unrewarded: bool = False,
    position_range: tuple[float, float] | None = None,
    n_pca_components: int = 10,
    pca_model: PCA | None = None,
) -> tuple[list[np.ndarray], PCA | None]:
    """Extract per-trial neural signal matrices, PCA-reduced.

    Each trial yields a (channels, timepoints) array shaped for fastDSA.
    Trials with fewer than 2 frames after filtering are dropped.

    Args:
        df: Frame-level DataFrame with signal and behavioral columns.
        signal_col: Column containing neural signal vectors (list per frame).
        trial_type: Filter to this trial type. None = all trials.
        min_speed: Minimum speed threshold (cm/s). 0 = no filtering.
        exclude_unrewarded: Drop trials where reward was never delivered.
        position_range: (min_cm, max_cm) to restrict frames by position.
        n_pca_components: Number of PCA dimensions. 0 = no PCA.
        pca_model: Pre-fit PCA model. If None, fits on this data.

    Returns:
        Tuple of (trial_matrices, pca_model).
        trial_matrices: List of arrays, each (n_dims, n_frames_in_trial).
        pca_model: Fitted PCA (or None if n_pca_components == 0).
    """
    filtered = df.filter(pl.col('experiment_state') == 'run')

    if trial_type is not None:
        filtered = filtered.filter(pl.col('trial_type') == trial_type)

    if min_speed > 0:
        filtered = filtered.filter(pl.col('speed_cm_s') >= min_speed)

    if exclude_unrewarded:
        rewarded_trials = (
            filtered.group_by('trial')
            .agg(pl.col('water_uL').sum().alias('total_reward'))
            .filter(pl.col('total_reward') > 0)
            ['trial']
        )
        filtered = filtered.filter(pl.col('trial').is_in(rewarded_trials))

    if position_range is not None:
        lo, hi = position_range
        filtered = filtered.filter(
            (pl.col('position') >= lo) & (pl.col('position') < hi)
        )

    if len(filtered) == 0:
        return [], pca_model

    # Extract full signal matrix for PCA fitting
    all_signals = np.vstack(filtered[signal_col].to_list())

    # Drop NaN columns (cells absent in multi_day matching)
    valid_cols = ~np.all(np.isnan(all_signals), axis=0)
    all_signals = all_signals[:, valid_cols]

    # Fit or apply PCA
    if n_pca_components > 0:
        n_pca_components = min(n_pca_components, all_signals.shape[1],
                               all_signals.shape[0])
        if pca_model is None:
            pca_model = PCA(n_components=n_pca_components)
            pca_model.fit(all_signals)
        all_reduced = pca_model.transform(all_signals)
    else:
        all_reduced = all_signals
        pca_model = None

    # Split into per-trial matrices, transposed to (channels, timepoints)
    trials = filtered['trial'].to_numpy()
    unique_trials = np.unique(trials)

    trial_matrices = []
    for t in unique_trials:
        mask = trials == t
        trial_data = all_reduced[mask]  # (frames, dims)
        if trial_data.shape[0] >= 2:
            trial_matrices.append(trial_data.T)  # -> (dims, frames)

    return trial_matrices, pca_model


# DSA COMPUTATION

def compute_fastdsa(
    trials_a: list[np.ndarray],
    trials_b: list[np.ndarray],
    cfg: 'SimDistConfig | None' = None,
    **cfg_overrides,
) -> tuple[float, int]:
    """Compute fastDSA score between two sets of trial data.

    Args:
        trials_a: List of arrays each (channels, timepoints) for system A.
        trials_b: List of arrays each (channels, timepoints) for system B.
        cfg: SimDistConfig. If None, creates default config.
        **cfg_overrides: Override specific config fields (e.g. device='cuda').

    Returns:
        Tuple of (dsa_score, rank_used).
        dsa_score: Angular distance (0 = identical dynamics, π/2 = maximally different).
        rank_used: DMD rank selected by SVHT (or fixed rank if specified).
    """
    _check_fastdsa_installed()

    if cfg is None:
        cfg = make_dsa_config(**cfg_overrides)

    sim = FastDSASimilarity(cfg)
    score, used_rank = sim.fit_score(trials_a, trials_b)
    return float(score), int(used_rank)


# WITHIN-SESSION: ABC vs ABDC

def within_session_dsa(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    position_range: tuple[float, float] | None = None,
    dsa_cfg: 'SimDistConfig | None' = None,
) -> dict:
    """Compare ABC vs ABDC dynamics within a single session.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        signal_col: Column with neural signals.
        n_pca_components: PCA dimensions before DMD.
        min_speed: Speed filter (cm/s).
        position_range: Restrict to position range (min_cm, max_cm).
        dsa_cfg: fastDSA config. None = default.

    Returns:
        Dict with 'dsa_score', 'rank_used', 'trial_types',
        'n_trials_*', 'pca_variance_explained'.
    """
    trial_types = sorted(df.filter(
        pl.col('experiment_state') == 'run'
    )['trial_type'].unique().to_list())

    if len(trial_types) < 2:
        raise ValueError(f"Need >= 2 trial types, found: {trial_types}")

    # Fit PCA on all trial types combined
    _, pca_model = extract_trial_signals(
        df, signal_col=signal_col, trial_type=None,
        min_speed=min_speed, n_pca_components=n_pca_components,
        position_range=position_range,
    )

    # Extract per trial type using shared PCA
    trials_by_type = {}
    results = {}

    for tt in trial_types:
        trials, _ = extract_trial_signals(
            df, signal_col=signal_col, trial_type=tt,
            min_speed=min_speed, n_pca_components=n_pca_components,
            pca_model=pca_model, position_range=position_range,
        )
        trials_by_type[tt] = trials
        results[f'n_trials_{tt}'] = len(trials)

        if len(trials) < 2:
            print(f"  WARNING: {tt} has {len(trials)} trials — need >= 2")
            results['dsa_score'] = np.nan
            return results

    type_a, type_b = trial_types[0], trial_types[1]
    score, rank_used = compute_fastdsa(
        trials_by_type[type_a], trials_by_type[type_b], cfg=dsa_cfg,
    )

    results['dsa_score'] = score
    results['rank_used'] = rank_used
    results['trial_types'] = trial_types
    results['pca_model'] = pca_model
    results['pca_variance_explained'] = pca_model.explained_variance_ratio_.sum()

    return results


def within_session_dsa_across_days(
    mouse_dir: Path,
    dates: list[str],
    signal_col: str = 'multi_day_dff',
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    dsa_cfg: 'SimDistConfig | None' = None,
) -> dict:
    """Run within-session ABC vs ABDC DSA for each session.

    Memory-efficient: loads one session at a time.

    Args:
        mouse_dir: Mouse-level directory path.
        dates: List of session date strings.
        signal_col: Column with neural signals.
        n_pca_components: PCA dimensions.
        min_speed: Speed filter.
        dsa_cfg: fastDSA config. None = default.

    Returns:
        Dict with 'dates', 'dsa_scores', 'ranks_used',
        'n_trials' (dict per date).
    """
    _columns = ['frame', 'trial', 'trial_type', 'experiment_state',
                'position', 'speed_cm_s', 'water_uL', signal_col]

    valid_dates = []
    scores = []
    ranks = []
    trial_info = {}

    for date in sorted(dates):
        try:
            session_dir = find_session_dir(mouse_dir, date)
            session_data, config = load_session_context(session_dir)
            paths = get_session_paths(session_dir, session_data)
            data = pl.read_parquet(paths['parquet'], columns=_columns)

            result = within_session_dsa(
                data, config,
                signal_col=signal_col,
                n_pca_components=n_pca_components,
                min_speed=min_speed,
                dsa_cfg=dsa_cfg,
            )
            valid_dates.append(date)
            scores.append(result['dsa_score'])
            ranks.append(result['rank_used'])
            trial_info[date] = {
                tt: result.get(f'n_trials_{tt}', 0)
                for tt in result.get('trial_types', [])
            }
            print(f"  {date}: DSA={result['dsa_score']:.4f}, rank={result['rank_used']}")

            del data

        except Exception as e:
            print(f"  WARNING: {date}: {e}")

    return {
        'dates': valid_dates,
        'dsa_scores': np.array(scores),
        'ranks_used': np.array(ranks),
        'n_trials': trial_info,
    }


def position_resolved_dsa(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    window_size_cm: float = 30.0,
    step_cm: float = 10.0,
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    dsa_cfg: 'SimDistConfig | None' = None,
) -> dict:
    """Compute DSA between trial types in sliding position windows.

    Sweeps a spatial window along the shared segment of the track to
    identify where dynamics begin to diverge.

    Args:
        df: Frame-level DataFrame.
        config: Experiment configuration dict.
        signal_col: Column with neural signals.
        window_size_cm: Width of sliding window in cm.
        step_cm: Step size between windows in cm.
        n_pca_components: PCA dimensions.
        min_speed: Speed filter.
        dsa_cfg: fastDSA config. None = default with reduced n_delays=10.

    Returns:
        Dict with 'positions' (window centers), 'dsa_scores',
        'ranks_used', 'n_trials_per_window', 'window_size_cm'.
    """
    if dsa_cfg is None:
        dsa_cfg = make_dsa_config(n_delays=10)

    trial_types = sorted(df.filter(
        pl.col('experiment_state') == 'run'
    )['trial_type'].unique().to_list())

    if len(trial_types) < 2:
        raise ValueError(f"Need >= 2 trial types, found: {trial_types}")

    # Shared track extent
    shared_end = min(get_track_length(config, tt) for tt in trial_types)

    # Fit global PCA on all data
    _, pca_model = extract_trial_signals(
        df, signal_col=signal_col, trial_type=None,
        min_speed=min_speed, n_pca_components=n_pca_components,
    )

    positions = []
    dsa_scores = []
    ranks_used = []
    trial_counts = []

    center = window_size_cm / 2
    while center + window_size_cm / 2 <= shared_end:
        lo = center - window_size_cm / 2
        hi = center + window_size_cm / 2
        pos_range = (lo, hi)

        trials_by_type = {}
        min_trials = float('inf')
        skip = False

        min_frames_needed = max(dsa_cfg.n_delays + 2, 5)

        for tt in trial_types:
            trials, _ = extract_trial_signals(
                df, signal_col=signal_col, trial_type=tt,
                min_speed=min_speed, n_pca_components=n_pca_components,
                pca_model=pca_model, position_range=pos_range,
            )
            # Filter trials with too few frames for delay embedding
            # trial shape is (channels, timepoints)
            trials = [t for t in trials if t.shape[1] >= min_frames_needed]
            min_trials = min(min_trials, len(trials))

            if len(trials) < 2:
                skip = True
                break

            trials_by_type[tt] = trials

        if skip:
            positions.append(center)
            dsa_scores.append(np.nan)
            ranks_used.append(0)
            trial_counts.append(0)
        else:
            score, rank = compute_fastdsa(
                trials_by_type[trial_types[0]],
                trials_by_type[trial_types[1]],
                cfg=dsa_cfg,
            )
            positions.append(center)
            dsa_scores.append(score)
            ranks_used.append(rank)
            trial_counts.append(int(min_trials))

        center += step_cm

    return {
        'positions': np.array(positions),
        'dsa_scores': np.array(dsa_scores),
        'ranks_used': np.array(ranks_used),
        'n_trials_per_window': np.array(trial_counts),
        'window_size_cm': window_size_cm,
        'trial_types': trial_types,
    }


# CROSS-SESSION (single animal)

def cross_session_dsa(
    mouse_dir: Path,
    dates: list[str],
    signal_col: str = 'multi_day_dff',
    trial_type: str | None = None,
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    dsa_cfg: 'SimDistConfig | None' = None,
) -> dict:
    """Compute pairwise DSA across sessions. Memory-efficient.

    Args:
        mouse_dir: Mouse-level directory path.
        dates: List of session date strings.
        signal_col: Column with neural signals.
        trial_type: Filter to one trial type. None = pool all.
        n_pca_components: PCA dimensions.
        min_speed: Speed filter.
        dsa_cfg: fastDSA config. None = default.

    Returns:
        Dict with 'dates', 'distance_matrix', 'trial_data'.
    """
    trial_data = extract_sessions(
        mouse_dir, dates, signal_col=signal_col, trial_type=trial_type,
        n_pca_components=n_pca_components, min_speed=min_speed,
    )

    valid_dates = sorted(trial_data.keys())
    n = len(valid_dates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            s, _ = compute_fastdsa(
                trial_data[valid_dates[i]],
                trial_data[valid_dates[j]],
                cfg=dsa_cfg,
            )
            dist_matrix[i, j] = dist_matrix[j, i] = s
            print(f"  {valid_dates[i]} vs {valid_dates[j]}: {s:.4f}")

    return {
        'dates': valid_dates,
        'distance_matrix': dist_matrix,
        'trial_data': trial_data,
        'trial_type': trial_type,
    }

# CROSS-ANIMAL

def cross_animal_dsa(
    mouse_dir_a: Path,
    dates_a: list[str],
    mouse_dir_b: Path,
    dates_b: list[str],
    animal_id_a: str,
    animal_id_b: str,
    signal_col: str = 'multi_day_dff',
    trial_type: str | None = None,
    n_pca_components: int = 10,
    min_speed: float = 2.0,
    dsa_cfg: 'SimDistConfig | None' = None,
) -> dict:
    """Memory-efficient cross-animal DSA. Loads one session at a time.

    Args:
        mouse_dir_a: Mouse directory for animal A.
        dates_a: Session dates for animal A.
        mouse_dir_b: Mouse directory for animal B.
        dates_b: Session dates for animal B.
        animal_id_a: Identifier for animal A.
        animal_id_b: Identifier for animal B.
        signal_col: Column with neural signals.
        trial_type: Filter to one trial type. None = pool all.
        n_pca_components: PCA dimensions.
        min_speed: Speed filter.
        dsa_cfg: fastDSA config. None = default.

    Returns:
        Dict with 'within_a_matrix', 'within_b_matrix',
        'cross_matrix', 'dates_a', 'dates_b'.
    """
    extract_kwargs = dict(signal_col=signal_col, trial_type=trial_type,
                          n_pca_components=n_pca_components, min_speed=min_speed)

    print(f"Extracting mouse {animal_id_a}...")
    trials_a = extract_sessions(mouse_dir_a, dates_a, **extract_kwargs)

    print(f"Extracting mouse {animal_id_b}...")
    trials_b = extract_sessions(mouse_dir_b, dates_b, **extract_kwargs)

    valid_a = sorted(trials_a.keys())
    valid_b = sorted(trials_b.keys())
    n_a, n_b = len(valid_a), len(valid_b)

    # Within-animal A
    within_a = np.zeros((n_a, n_a))
    for i in range(n_a):
        for j in range(i + 1, n_a):
            s, _ = compute_fastdsa(trials_a[valid_a[i]], trials_a[valid_a[j]], cfg=dsa_cfg)
            within_a[i, j] = within_a[j, i] = s
            print(f"  A {valid_a[i]} vs {valid_a[j]}: {s:.4f}")

    # Within-animal B
    within_b = np.zeros((n_b, n_b))
    for i in range(n_b):
        for j in range(i + 1, n_b):
            s, _ = compute_fastdsa(trials_b[valid_b[i]], trials_b[valid_b[j]], cfg=dsa_cfg)
            within_b[i, j] = within_b[j, i] = s
            print(f"  B {valid_b[i]} vs {valid_b[j]}: {s:.4f}")

    # Cross-animal
    cross = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            s, _ = compute_fastdsa(trials_a[valid_a[i]], trials_b[valid_b[j]], cfg=dsa_cfg)
            cross[i, j] = s
            print(f"  A:{valid_a[i]} vs B:{valid_b[j]}: {s:.4f}")

    return {
        'animal_ids': (animal_id_a, animal_id_b),
        'dates_a': valid_a,
        'dates_b': valid_b,
        'within_a_matrix': within_a,
        'within_b_matrix': within_b,
        'cross_matrix': cross,
    }


# PLOTTING

def plot_within_session_dsa(
    result: dict,
    config: dict,
    animal_id: str = '',
    figsize: tuple[float, float] = (8, 4),
    show: bool = True,
) -> Figure:
    """Plot ABC vs ABDC DSA score across sessions.

    Args:
        result: Output of within_session_dsa_across_days().
        config: Experiment configuration dict.
        animal_id: For title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = result['dates']
    scores = result['dsa_scores']
    short_dates = [d[5:] for d in dates]

    ax.plot(range(len(scores)), scores, 'o-', color='#2E86AB',
            markersize=7, linewidth=2)

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(short_dates, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('DSA distance (angular)', fontsize=10)
    ax.set_xlabel('Session', fontsize=10)
    max_score = max(scores) if len(scores) > 0 else 0.5
    ax.set_ylim(0, max(max_score * 1.3, 0.5))
    ax.axhline(np.pi / 4, ls='--', color='gray', alpha=0.4, label='π/4')
    ax.legend(fontsize=9)

    ax.set_title(
        pfmt.build_title(animal_id, 'ABC vs ABDC dynamics across days'),
        fontsize=11,
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_position_resolved_dsa(
    result: dict,
    config: dict,
    session_label: str = '',
    figsize: tuple[float, float] = (10, 4),
    show: bool = True,
) -> Figure:
    """Plot DSA score as a function of track position.

    Args:
        result: Output of position_resolved_dsa().
        config: Experiment configuration dict.
        session_label: Label for title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = result['positions']
    scores = result['dsa_scores']
    trial_types = result['trial_types']

    valid = ~np.isnan(scores)
    ax.plot(positions[valid], scores[valid], 'o-', color='#2E86AB',
            markersize=4, linewidth=1.5, label='DSA distance')
    ax.fill_between(positions[valid], 0, scores[valid], alpha=0.15,
                    color='#2E86AB')

    # Mark bifurcation point
    bifurcation_cm = _find_bifurcation_cm(config)
    if bifurcation_cm is not None:
        ax.axvline(bifurcation_cm, ls='--', color='red', alpha=0.7,
                   label=f'Bifurcation ({bifurcation_cm:.0f} cm)')

    # Cue shading for the shorter track
    shorter_tt = min(trial_types, key=lambda tt: get_track_length(config, tt))
    pfmt.add_cue_shading(ax, config, shorter_tt, alpha=0.1)

    ax.set_xlabel('Track position (cm)', fontsize=10)
    ax.set_ylabel('DSA distance (angular)', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

    title = pfmt.build_title(
        session_label,
        f'Position-resolved DSA — {trial_types[0]} vs {trial_types[1]}',
    )
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_cross_animal_dsa(
    result: dict,
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
) -> Figure:
    """Plot cross-animal DSA from cross_animal_dsa_lean output.

    Args:
        result: Output of cross_animal_dsa_lean().
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    animal_a, animal_b = result['animal_ids']
    dates_a = result['dates_a']
    dates_b = result['dates_b']
    within_a = result['within_a_matrix']
    within_b = result['within_b_matrix']
    cross = result['cross_matrix']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Within-animal consecutive-day distances
    ax = axes[0]
    n_a, n_b = len(dates_a), len(dates_b)
    if n_a >= 2:
        consec_a = [within_a[i, i + 1] for i in range(n_a - 1)]
        ax.plot(range(len(consec_a)), consec_a, 'o-', color='#2E86AB',
                label=f'Mouse {animal_a}', markersize=7, linewidth=1.5)
    if n_b >= 2:
        consec_b = [within_b[i, i + 1] for i in range(n_b - 1)]
        ax.plot(range(len(consec_b)), consec_b, 'o-', color='#A23B72',
                label=f'Mouse {animal_b}', markersize=7, linewidth=1.5)

    ax.set_xlabel('Session transition', fontsize=10)
    ax.set_ylabel('DSA distance', fontsize=10)
    ax.set_title('Within-animal dynamics change', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Panel 2: Cross-animal matched-session distances (diagonal)
    ax2 = axes[1]
    min_n = min(n_a, n_b)
    if min_n > 0:
        matched = [cross[i, i] for i in range(min_n)]
        ax2.plot(range(min_n), matched, 's-', color='#666',
                 markersize=6, linewidth=1.5,
                 label='Cross-animal (matched session)')
        ax2.set_xlabel('Session index', fontsize=10)
        ax2.set_ylabel('DSA distance', fontsize=10)
        ax2.set_title('Cross-animal at matched sessions', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.set_ylim(bottom=0)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        f'Mouse {animal_a} vs Mouse {animal_b} — fastDSA',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig


# HELPERS

def _find_bifurcation_cm(config: dict) -> float | None:
    """Find position where ABC and ABDC cue sequences first diverge.

    Args:
        config: Experiment configuration dict.

    Returns:
        Bifurcation position in cm, or None if not determinable.
    """
    trial_structures = config.get('trial_structures', {})
    if 'ABC' not in trial_structures or 'ABDC' not in trial_structures:
        return None

    seq_abc = trial_structures['ABC']['cue_sequence']
    seq_abdc = trial_structures['ABDC']['cue_sequence']
    cue_widths = config.get('cue_map', {})

    position = 0.0
    for c_abc, c_abdc in zip(seq_abc, seq_abdc):
        if c_abc != c_abdc:
            return position
        position += cue_widths.get(c_abc, 0)

    return position



# MAIN
if __name__ == '__main__':
    mouse_id = '26'
    date = '2025-09-10'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])

    print(f"Loaded {date}: {len(data)} frames, {data['trial'].n_unique()} trials")

#TODO use load_mulitsession and loop through

    # # --- Build config ---
    dsa_cfg = make_dsa_config(
        n_delays=15,
        rank=None,       # SVHT auto-rank
        method='ro',
        iters=300,
        device='cpu',    # switch to 'cuda' if available
    )
    #
    # --- Within-session DSA ---
    # print("\n--- Within-session DSA (ABC vs ABDC) ---")
    # result = within_session_dsa(
    #     data, exp_config,
    #     signal_col='multi_day_dff',
    #     n_pca_components=10,
    #     min_speed=2.0,
    #     dsa_cfg=dsa_cfg,
    # )
    # print(f"DSA score: {result['dsa_score']:.4f}")
    # print(f"SVHT rank: {result['rank_used']}")
    # print(f"PCA variance explained: {result['pca_variance_explained']:.2%}")

    #plot_within_session_dsa(result, exp_config, session_label=date)
    #

    dates_26 = ['2025-09-08', '2025-09-10', '2025-09-15', '2025-09-16']
    result = within_session_dsa_across_days(
        mouse_dir, dates_26,
        signal_col='multi_day_dff',
        n_pca_components=10,
        min_speed=2.0,
        dsa_cfg=dsa_cfg,
    )
    plot_within_session_dsa(result, exp_config, animal_id=mouse_id)



    # # --- Position-resolved DSA ---
    # print("\n--- Position-resolved DSA ---")
    # pos_cfg = make_dsa_config(n_delays=10, rank=None, method='ro',
    #                           iters=200, device='cpu')
    # pos_result = position_resolved_dsa(
    #     data, exp_config,
    #     signal_col='multi_day_dff',
    #     window_size_cm=30.0,
    #     step_cm=10.0,
    #     n_pca_components=10,
    #     min_speed=2.0,
    #     dsa_cfg=pos_cfg,
    # )
    # plot_position_resolved_dsa(pos_result, exp_config, session_label=date)

   # --- Cross-session DSA  ---
    print("\n--- Cross-session DSA ---")

    mouse_id_2 = '14'
    mouse_dir_b = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id_2)

# days b1, b5, e1, e3, e7
    # 26 '2025-08-20', '2025-08-25', '2025-08-30', '2025-09-03',
    #14 '2025-08-12', '2025-08-14', '2025-08-18', '2025-08-20',
    cross_animal_result = cross_animal_dsa(
        mouse_dir, [ '2025-09-08', '2025-09-10',
                    '2025-09-11', '2025-09-16'],
        mouse_dir_b, [ '2025-08-22', '2025-08-27',
                      '2025-09-03', '2025-09-05'],
        animal_id_a=mouse_id, animal_id_b=mouse_id_2,
        trial_type='ABC',
        dsa_cfg=dsa_cfg,
    )

    plot_cross_animal_dsa(cross_animal_result)
