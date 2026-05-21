"""
UMAP Plotting Module for Frame-Level Neural Data

Visualization functions for analyzing neural manifolds using UMAP.
Works with frame-level data (after process_session or fix_cue_offset) — NOT spatially binned trial data.

Core workflow:
    1. prepare_umap_data() - Frame df → neural array + filtered_df (signal columns dropped)
    2. compute_umap() - Neural array → embedding + umap_params as config file
    3. plot_umap_*() - Embedding + filtered_df → visualization

Main plotting function:
    - plot_umap: Handles 1D (matplotlib), 2D (matplotlib), 3D (interactive Plotly).
        Supports all coloring strategies, trial-type toggles, dropdown to switch
        between multiple strategies, clickable legends, hover info.

Specialized plots (kept separate):
    - plot_umap_2d_density: KDE contours per trial type
    - plot_umap_3d_single_trial_trajectory: Lines connecting consecutive frames within a trial
    - plot_umap_3d_position_matched: Side-by-side ABC vs ABDC for shared position range
"""


import hashlib
import json

import numpy as np
import polars as pl
import umap
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from typing import Literal, Any
from enum import Enum
from pathlib import Path

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import plot_utils as pfmt



# COLORING STRATEGY

class ColoringStrategy(Enum):
    """Defines different strategies for coloring UMAP points."""
    CUE = "cue"
    CUE_ID = "cue_id"
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
) -> tuple[np.ndarray, pl.DataFrame]:
    """Prepare frame-level data for UMAP.

    Filters the DataFrame, extracts neural activity as a numpy array, and
    returns the lightweight filtered DataFrame (signal columns dropped) for
    downstream use. The filtered DataFrame is frame-aligned with neural_data.

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
        filtered_df: Filtered DataFrame with signal columns dropped. Frame-aligned
            with neural_data; use columns directly for coloring/segmenting.
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

    # Memory-efficient stack: pre-allocate float32 and fill row-by-row to avoid
    # the float64 vstack peak (list-of-arrays + 2x sized output copy).
    col = filtered[signal_column]
    n_frames = len(col)
    first = np.asarray(col[0], dtype=np.float32)
    n_cells = first.shape[0]
    print(f"  Allocating neural_data: {n_frames} × {n_cells} float32 "
          f"(~{n_frames * n_cells * 4 / 1e9:.2f} GB)...")
    neural_data = np.empty((n_frames, n_cells), dtype=np.float32)
    neural_data[0] = first
    for i in range(1, n_frames):
        neural_data[i] = col[i]
    print(f"  Stack OK: shape={neural_data.shape}, dtype={neural_data.dtype}, "
          f"size={neural_data.nbytes / 1e9:.2f} GB")

    # Add 'position' if not present (normalized within-trial distance)
    if 'position' not in filtered.columns:
        filtered = filtered.with_columns(
            (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
            .alias('position')
        )

    # Add 'nominal_track_length' if not present
    if 'nominal_track_length' not in filtered.columns:
        filtered = filtered.with_columns(
            (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
            .max().over('trial').alias('nominal_track_length')
        )

    # Within-(day-or-session) normalized progress [0, 1]. For single-day data
    # this is equivalent to normalizing the 'trial' column directly; for
    # multiday it makes trials comparable across sessions of different lengths.
    group_col = 'session_date' if 'session_date' in filtered.columns else None
    if group_col is not None:
        filtered = filtered.with_columns(
            ((pl.col('trial') - pl.col('trial').min().over(group_col))
             / (pl.col('trial').max().over(group_col) - pl.col('trial').min().over(group_col)).clip(lower_bound=1))
            .alias('session_progress')
        )
    else:
        t_min = filtered['trial'].min()
        t_max = filtered['trial'].max()
        denom = max(1, t_max - t_min)
        filtered = filtered.with_columns(
            ((pl.col('trial') - t_min) / denom).alias('session_progress')
        )

    # Across-day normalized progress [0, 1]: dense-rank over (session_date, trial)
    # so day-1 trial-1 = 0, last day's last trial = 1. For single-day data this
    # equals session_progress.
    if group_col is not None:
        rank_expr = pl.struct([group_col, 'trial']).rank('dense').cast(pl.Float64)
    else:
        rank_expr = pl.col('trial').rank('dense').cast(pl.Float64)
    filtered = filtered.with_columns(
        ((rank_expr - 1) / (rank_expr.max() - 1).clip(lower_bound=1))
        .alias('experiment_progress')
    )

    # Drop signal columns — neural_data already extracted as numpy array
    signal_cols = [c for c in filtered.columns if c.startswith(('single_day_', 'multi_day_'))]
    filtered_df = filtered.drop(signal_cols)

    n_filtered = len(df) - len(filtered_df)
    print(f"Prepared {neural_data.shape[0]} frames × {neural_data.shape[1]} cells")
    if n_filtered > 0:
        print(f"  Filtered out {n_filtered} frames")

    cues = filtered_df['cue'].to_numpy()  # uint8
    for cue_val in sorted(set(cues), key=str):
        print(type(cue_val), cue_val, (cues == cue_val).sum())

    return neural_data, filtered_df


"""
Patch for umap_plotting.py — adds position-binned PV preparation.

Paste `prepare_umap_data_position_binned` into umap_plotting.py
(e.g. right after `prepare_umap_data`), and update the `__main__`
block with the example at the bottom.
"""

import numpy as np
import polars as pl
from typing import Any


def prepare_umap_data_position_binned(
        df: pl.DataFrame,
        signal_column: str = "multi_day_dff",
        bin_width_cm: float = 5.0,
        min_speed: float | None = 2.0,
        max_speed: float | None = None,
        cues_to_include: list[int] | None = None,
        cues_to_exclude: list[int] | None = None,
        trial_types_to_include: list[str] | None = None,
        state_filters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Prepare position-binned PVs for UMAP.

    One row per (trial, position_bin): neural activity averaged across frames
    in that spatial bin. Removes within-bin temporal context so any UMAP
    separation must come from instantaneous PV differences rather than
    trajectory/temporal dynamics.

    Frame-level filters (speed, cue, trial_type, state) are applied BEFORE
    binning. After filtering, frames are grouped by (trial, position_bin)
    and the signal is averaged element-wise across frames within each group.

    Args:
        df: Frame-level DataFrame (after process_session or fix_cue_offset).
        signal_column: Column containing per-frame neural activity arrays.
        bin_width_cm: Position bin width in cm.
        min_speed: Min speed threshold (frame-level, applied before binning).
        max_speed: Max speed threshold (frame-level).
        cues_to_include: Frame-level cue filter — keep only these cue zones.
        cues_to_exclude: Frame-level cue filter — drop these cue zones.
        trial_types_to_include: Frame-level trial type filter.
        state_filters: {col: val} for extra frame-level filtering.

    Returns:
        neural_data: Array of shape (n_bins_total, n_cells). One row per
            (trial, position_bin), neural activity averaged across frames.
        binned_df: One row per bin, frame-aligned with neural_data. Columns:
            trial, trial_type, position_bin (int), position (mean cm in bin),
            cue (modal), speed_cm_s (mean), nominal_track_length,
            n_frames_in_bin.
    """
    filtered = df

    # ── frame-level filters (same as prepare_umap_data) ──
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

    # ── ensure position / nominal_track_length columns exist ──
    if 'position' not in filtered.columns:
        filtered = filtered.with_columns(
            (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
            .alias('position')
        )
    if 'nominal_track_length' not in filtered.columns:
        filtered = filtered.with_columns(
            (pl.col('distance_cm') - pl.col('distance_cm').min().over('trial'))
            .max().over('trial').alias('nominal_track_length')
        )

    # ── assign position bin per frame ──
    filtered = filtered.with_columns(
        (pl.col('position') // bin_width_cm).cast(pl.Int32).alias('position_bin')
    )

    # ── aggregate neural signal per (trial, position_bin) ──
    # Extract frame-level neural as numpy, then average using groupby indices.
    # Doing the aggregation in numpy is cleaner than dealing with polars
    # element-wise list-column means.
    frame_neural = np.vstack(filtered[signal_column].to_list())  # (n_frames, n_cells)

    groups = np.stack(
        [filtered['trial'].to_numpy(), filtered['position_bin'].to_numpy()],
        axis=1,
    )
    unique_groups, inverse, counts = np.unique(
        groups, axis=0, return_inverse=True, return_counts=True
    )
    n_bins, n_cells = len(unique_groups), frame_neural.shape[1]

    binned_neural = np.zeros((n_bins, n_cells), dtype=np.float64)
    np.add.at(binned_neural, inverse, frame_neural)
    binned_neural /= counts[:, None]
    binned_neural = binned_neural.astype(frame_neural.dtype)

    # ── metadata aggregation per bin ──
    binned_df = (
        filtered
        .group_by(['trial', 'position_bin'], maintain_order=False)
        .agg([
            pl.col('trial_type').first(),
            pl.col('position').mean().alias('position'),
            pl.col('cue').mode().first().alias('cue'),
            pl.col('speed_cm_s').mean().alias('speed_cm_s'),
            pl.col('nominal_track_length').first(),
            pl.len().alias('n_frames_in_bin'),
        ])
        .sort(['trial', 'position_bin'])
    )

    # ── align neural_data rows with binned_df row order ──
    # `unique_groups` is lex-sorted by (trial, position_bin); `binned_df` is
    # also sorted the same way, so the alignment is row-for-row. Double-check
    # to fail loudly if that ever drifts.
    md_pairs = np.stack(
        [binned_df['trial'].to_numpy(), binned_df['position_bin'].to_numpy()],
        axis=1,
    )
    assert np.array_equal(md_pairs, unique_groups), \
        "Row order mismatch between numpy unique_groups and binned_df"

    # ── add cue_id if present in original df (preserves coloring support) ──
    if 'cue_id' in df.columns:
        cue_id_lookup = (
            filtered
            .group_by(['trial', 'position_bin'], maintain_order=False)
            .agg(pl.col('cue_id').mode().first().alias('cue_id'))
            .sort(['trial', 'position_bin'])
        )
        binned_df = binned_df.join(cue_id_lookup, on=['trial', 'position_bin'])

    print(f"Position-binned: {binned_neural.shape[0]} bins × {binned_neural.shape[1]} cells")
    print(f"  Bin width: {bin_width_cm} cm")
    print(f"  Frames per bin: mean={counts.mean():.1f}, min={counts.min()}, max={counts.max()}")

    cues = binned_df['cue'].to_numpy()
    for cue_val in sorted(set(cues), key=str):
        print(type(cue_val), cue_val, (cues == cue_val).sum())

    return binned_neural, binned_df



# UMAP COMPUTATION

def compute_umap(
        neural_data: np.ndarray,
        n_components: int = 3,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = 'correlation',
        random_state: int = 42,
        n_pca_components: int | None = None,
) -> np.ndarray:
    """Compute UMAP embedding. Check API for more param options.
        https://umap-learn.readthedocs.io/en/latest/api.html

    Args:
        neural_data: Array of shape (n_frames, n_cells).
        n_components: Embedding dimensions (1, 2, or 3). Though theoretically could go up to 100
        n_neighbors: UMAP nearest neighbors parameter.  50 was used in OSM paper
        min_dist: UMAP minimum distance between embedded points
        metric: Distance metric for UMAP. Best options: 'correlation', 'euclidean', 'cosine'.
        random_state: Random seed for reproducibility.
        n_pca_components: If set, pre-reduce neural_data with PCA to this many components
            before running UMAP. Recommended for high cell counts (e.g., >3000) to cut
            memory/runtime. 50–100 typically preserves structure. None disables PCA.

    Returns:
        embedding: array of shape (n_frames, n_components).
        umap_params: Dict of hyperparameters used for reproducibility/saving.
    """
    # Cast to float32 to halve memory; UMAP upcasts internally anyway.
    neural_data = np.asarray(neural_data, dtype=np.float32)

    pca_explained_variance = None
    if n_pca_components is not None:
        n_pcs = min(n_pca_components, *neural_data.shape)
        print(f"PCA pre-reduction: {neural_data.shape[1]} cells → {n_pcs} PCs...")
        pca = PCA(n_components=n_pcs, random_state=random_state)
        neural_data = pca.fit_transform(neural_data).astype(np.float32)
        pca_explained_variance = float(pca.explained_variance_ratio_.sum())
        print(f"  PCA explained variance: {pca_explained_variance:.3f}")

    print(f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    # Fit X into an embedded space and return the transformed output
    embedding = reducer.fit_transform(neural_data)

    umap_params = {
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
        'random_state': random_state,
        'n_pca_components': n_pca_components,
        'pca_explained_variance': pca_explained_variance,
        'n_frames': neural_data.shape[0],
        'n_cells_or_pcs_in': neural_data.shape[1],
    }

    print(f"Done! Embedding shape: {embedding.shape}")
    return embedding, umap_params


def load_or_compute_umap(
        neural_data: np.ndarray,
        cache_dir: Path,
        cache_name: str,
        filtered_df: pl.DataFrame | None = None,
        n_components: int = 3,
        n_neighbors: int = 50,
        min_dist: float = 0.1,
        metric: str = 'correlation',
        random_state: int = 42,
        n_pca_components: int | None = None,
        force_recompute: bool = False,
) -> tuple[np.ndarray, pl.DataFrame | None, dict]:
    """Compute UMAP with disk caching, or load a previously cached embedding.

    Cache layout (all in cache_dir):
        {cache_name}_{hash}.npy      - embedding
        {cache_name}_{hash}.yaml     - umap_params (human-readable)
        {cache_name}_{hash}.parquet  - filtered_df (only if provided at save time)

    The hash digest covers UMAP params (and PCA setting), so changing any of them
    yields a new cache file. cache_name should encode dataset identity (animal,
    days, signal_column, frame filters) since that is NOT hashed — pick a name
    that uniquely identifies the input neural_data.

    Args:
        neural_data: Array of shape (n_frames, n_cells). Only used on cache miss.
        cache_dir: Directory for cached embeddings (created if missing).
        cache_name: Dataset-identifying name (e.g. 'M26_2025-09-10_to_12_multidff').
        filtered_df: Optional frame-aligned DataFrame to save alongside the embedding
            so downstream plotting can re-load without re-running prepare_umap_data.
        n_components, n_neighbors, min_dist, metric, random_state, n_pca_components:
            Forwarded to compute_umap.
        force_recompute: If True, ignore any existing cache and recompute + overwrite.

    Returns:
        embedding: Array of shape (n_frames, n_components).
        filtered_df: The cached parquet (if found) or the one passed in, else None.
        umap_params: Hyperparameters dict.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    params_for_hash = {
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'metric': metric,
        'random_state': random_state,
        'n_pca_components': n_pca_components,
    }
    digest = hashlib.md5(json.dumps(params_for_hash, sort_keys=True).encode()).hexdigest()[:8]
    base = cache_dir / f'{cache_name}_{digest}'
    npy_path = base.with_suffix('.npy')
    yaml_path = base.with_suffix('.yaml')
    parquet_path = base.with_suffix('.parquet')

    if not force_recompute and npy_path.exists() and yaml_path.exists():
        embedding = np.load(npy_path)
        with open(yaml_path, 'r') as f:
            umap_params = yaml.safe_load(f)
        df_out = pl.read_parquet(parquet_path) if parquet_path.exists() else filtered_df
        print(f"Loaded cached UMAP: {npy_path.name} (shape {embedding.shape})")
        return embedding, df_out, umap_params

    embedding, umap_params = compute_umap(
        neural_data,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_pca_components=n_pca_components,
    )

    np.save(npy_path, embedding)
    with open(yaml_path, 'w') as f:
        yaml.dump(umap_params, f, default_flow_style=False)
    if filtered_df is not None:
        filtered_df.write_parquet(parquet_path)
        print(f"Saved: {npy_path.name}, {yaml_path.name}, {parquet_path.name}")
    else:
        print(f"Saved: {npy_path.name}, {yaml_path.name}")

    return embedding, filtered_df, umap_params


# MATPLOTLIB HELPERS

def get_colors_for_strategy(
        strategy: ColoringStrategy,
        filtered_df: pl.DataFrame,
        cmap_name: str = 'viridis',
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate colors and colormap info for a coloring strategy.

    Args:
        strategy: How to color the points.
        filtered_df: Filtered DataFrame from prepare_umap_data.
        cmap_name: Colormap name for continuous variables (not used for all strategies).

    Returns:
        colors: Array of RGBA values or hex strings.
        color_info: Dict with 'type' ('categorical'/'continuous') and rendering info.
    """
    if strategy == ColoringStrategy.CUE:
        cue_color_map = pfmt.get_cue_colors()
        cue_label_map = pfmt.get_cue_labels()
        unique_cues = np.unique(filtered_df['cue'].to_numpy())
        colors = np.array([cue_color_map.get(int(c), '#D3D3D3') for c in filtered_df['cue'].to_numpy()])
        legend = {}
        for cue in unique_cues:
            label = cue_label_map.get(int(cue), f'Cue {cue}')
            legend[label] = cue_color_map.get(int(cue), '#D3D3D3')
        color_info = {
            'type': 'categorical', 'legend': legend, 'label': 'Cue Zone',
            'raw_cue_map': {int(c): cue_label_map.get(int(c), f'Cue {c}') for c in unique_cues},
        }

    elif strategy == ColoringStrategy.CUE_ID:
        cue_color_map = pfmt.get_cue_colors()
        cue_ids = filtered_df['cue_id'].to_numpy()
        unique_cue_ids = sorted(set(cue_ids), key=str)
        colors = np.array([cue_color_map.get(c, '#D3D3D3') for c in cue_ids])
        legend = {c: cue_color_map.get(c, '#D3D3D3') for c in unique_cue_ids}
        color_info = {
            'type': 'categorical', 'legend': legend, 'label': 'Cue ID',
            'column': 'cue_id',
        }

    elif strategy == ColoringStrategy.POSITION:
        cmap = plt.cm.get_cmap('twilight')
        vmax = filtered_df['nominal_track_length'].to_numpy().max()
        norm = Normalize(vmin=0, vmax=vmax)
        colors = cmap(norm(filtered_df['position'].to_numpy()))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Position (cm)'}

    elif strategy == ColoringStrategy.TRIAL_TYPE:
        unique_types = np.unique(filtered_df['trial_type'].to_numpy())
        colors = np.array([pfmt.TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in filtered_df['trial_type'].to_numpy()])
        legend = {tt: pfmt.TRIAL_TYPE_COLORS.get(tt, '#999999') for tt in unique_types}
        color_info = {'type': 'categorical', 'legend': legend, 'label': 'Trial Type'}

    elif strategy == ColoringStrategy.SPEED:
        cmap = plt.cm.get_cmap('plasma')
        norm = Normalize(vmin=filtered_df['speed_cm_s'].to_numpy().min(), vmax=filtered_df['speed_cm_s'].to_numpy().max())
        colors = cmap(norm(filtered_df['speed_cm_s'].to_numpy()))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Speed (cm/s)'}

    elif strategy == ColoringStrategy.SESSION_PROGRESS:
        cmap = plt.cm.get_cmap('YlOrBr')
        norm = Normalize(vmin=filtered_df['trial'].to_numpy().min(), vmax=filtered_df['trial'].to_numpy().max())
        colors = cmap(norm(filtered_df['trial'].to_numpy()))
        color_info = {'type': 'continuous', 'cmap': cmap, 'norm': norm, 'label': 'Session Progress'}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return colors, color_info


def _scatter_categorical(ax, embedding, filtered_df, color_info, strategy, alpha=0.6, s=20):
    """Scatter for categorical strategies (1D or 2D)."""
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1
    for label, color in color_info['legend'].items():
        if strategy == ColoringStrategy.CUE:
            raw_ids = [k for k, v in color_info['raw_cue_map'].items() if v == label]
            mask = np.isin(filtered_df['cue'].to_numpy(), raw_ids)
        elif strategy == ColoringStrategy.CUE_ID:
            mask = filtered_df['cue_id'].to_numpy() == label
        else:
            mask = filtered_df['trial_type'].to_numpy() == label

        if n_dims == 1:
            y_vals = np.random.normal(0, 0.02, size=mask.sum())
            ax.scatter(embedding[mask], y_vals, c=color, label=label, alpha=alpha, s=s)
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=label, alpha=alpha, s=s)
    ax.legend(title=color_info['label'], bbox_to_anchor=(1.05, 1), loc='upper left')


def _plot_matplotlib(embedding, filtered_df, strategy, title, save_path, alpha, s):
    """Matplotlib path for 1D and 2D embeddings."""
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1
    colors, color_info = get_colors_for_strategy(strategy, filtered_df)

    if n_dims == 1:
        fig, ax = plt.subplots(figsize=(12, 2))
        y_vals = np.random.normal(0, 0.02, size=len(embedding))
        if color_info['type'] == 'categorical':
            _scatter_categorical(ax, embedding, filtered_df, color_info, strategy, alpha, s)
        else:
            ax.scatter(embedding, y_vals, c=colors, alpha=alpha, s=s)
            plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                         ax=ax).set_label(color_info['label'])
        ax.set_xlabel('UMAP 1')
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
    else:  # 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        if color_info['type'] == 'categorical':
            _scatter_categorical(ax, embedding, filtered_df, color_info, strategy, alpha, s)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=alpha, s=s)
            plt.colorbar(ScalarMappable(norm=color_info['norm'], cmap=color_info['cmap']),
                         ax=ax, shrink=0.8).set_label(color_info.get('label', ''))
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax.set_title(title or f'UMAP — {strategy.value}')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# PLOTLY HELPERS
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


# PLOTLY 3D TRACE BUILDERS
# Each builder returns a list of traces for ONE strategy view.
# The caller handles visibility toggling across views.

def _build_cue_traces(embedding, filtered_df, point_size, opacity, use_cue_id=False):
    """Build per-cue, per-trial-type traces with clickable legend.

    Args:
        embedding: UMAP embedding array.
        filtered_df: Filtered DataFrame from prepare_umap_data.
        point_size: Plotly marker size.
        opacity: Marker opacity.
        use_cue_id: If True, color by 'cue_id' (unique gray zones).
            If False, color by 'cue' (int, gray zones merged).

    Returns:
        List of Plotly Scatter3d traces (trace dicts and list of legend group names.
    """
    traces = []
    trial_types = sorted(np.unique(filtered_df['trial_type'].to_numpy()))
    base_colors = pfmt.get_cue_colors()
    cue_labels = pfmt.get_cue_labels()
    col = 'cue_id' if use_cue_id else 'cue'


    for tt_idx, trial_type in enumerate(trial_types):
        tt_mask = filtered_df['trial_type'].to_numpy() == trial_type
        cues = filtered_df[col].to_numpy()[tt_mask]
        positions = filtered_df['position'].to_numpy()[tt_mask]
        emb = embedding[tt_mask]

        # Shade cues lighter/darker per trial type for visual distinction
        factor = 1.3 - tt_idx * 0.4
        cue_colors = {cid: pfmt.scale_color(c, factor=factor)
        if cid not in pfmt.SPECIAL_CUE_COLORS else c
                      for cid, c in base_colors.items()}

        suffix = f' ({trial_type})' if len(trial_types) > 1 else ''

        for cue_val in sorted(set(cues), key=str):
            cue_mask = cues == cue_val
            if not cue_mask.any():
                continue

            # For int cues, use int key; for cue_id strings, use string key directly
            color_key = int(cue_val) if not use_cue_id else cue_val
            label = cue_labels.get(color_key, str(cue_val)) if not use_cue_id else str(cue_val)
            color = cue_colors.get(color_key, '#AAAAAA')
            is_gray = (color_key == 0) if not use_cue_id else str(cue_val).startswith('0')

            print(f"Adding trace: cue_val={cue_val}, color={color}, n_points={cue_mask.sum()}")

            traces.append(go.Scatter3d(
                    x=emb[cue_mask, 0], y=emb[cue_mask, 1], z=emb[cue_mask, 2],
                    mode='markers', name=f'{label}{suffix}',
                    marker=dict(size=point_size, opacity=opacity, color=color),
                    showlegend=True,  # hide gray zone from legend
                    customdata=np.column_stack([
                        positions[cue_mask],
                        filtered_df['trial'].to_numpy()[tt_mask][cue_mask],
                    ]),
                    hovertemplate=(
                        f'{trial_type}<br>'
                        'Pos: %{customdata[0]:.1f} cm<br>'
                        f'Cue: {label}<br>'
                        'Trial: %{customdata[1]:.0f}'
                        '<extra></extra>'
                    ),
            ))
    return traces


def _build_position_traces(embedding, filtered_df, point_size, opacity):
    """Build per-trial-type traces colored by track position (separate colorscales).

    Returns list of traces.
    """
    traces = []
    trial_types = sorted(np.unique(filtered_df['trial_type']))
    cue_labels = pfmt.get_cue_labels()

    for i, trial_type in enumerate(trial_types):
        tt_mask = filtered_df['trial_type'].to_numpy() == trial_type
        positions = filtered_df['position'].to_numpy()[tt_mask]
        cues = filtered_df['cue'].to_numpy()[tt_mask]
        track_len = filtered_df['nominal_track_length'].to_numpy()[tt_mask].max()
        colorscale = pfmt.trial_type_colorscale(trial_type)

        # Stagger colorbars so they don't overlap
        cb_x = 1.0 + i * 0.15
        cb_y = 0.75 - i * 0.45

        traces.append(go.Scatter3d(
            x=embedding[tt_mask, 0], y=embedding[tt_mask, 1], z=embedding[tt_mask, 2],
            mode='markers', name=f'{trial_type}',
            marker=dict(
                size=point_size, opacity=opacity, color=positions,
                colorscale=colorscale, cmin=0, cmax=track_len,
                colorbar=dict(title=f'{trial_type} pos (cm)', len=0.4,
                              x=cb_x, y=cb_y),
            ),
            text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
            customdata=filtered_df['trial'][tt_mask],
            hovertemplate=(
                f'{trial_type}<br>'
                'Pos: %{marker.color:.1f} cm<br>'
                'Cue: %{text}<br>'
                'Trial: %{customdata:.0f}'
                '<extra></extra>'
            ),
        ))
    return traces


def _build_trial_type_traces(embedding, filtered_df, point_size, opacity):
    """Build per-trial-type traces with flat color per type. Clickable legend."""
    traces = []
    cue_labels = pfmt.get_cue_labels()

    for trial_type in sorted(np.unique(filtered_df['trial_type'])):
        tt_mask = filtered_df['trial_type'].to_numpy() == trial_type
        color = pfmt.TRIAL_TYPE_COLORS.get(trial_type, '#D3D3D3')
        positions = filtered_df['position'].to_numpy()[tt_mask]
        cues = filtered_df['cue'].to_numpy()[tt_mask]

        traces.append(go.Scatter3d(
            x=embedding[tt_mask, 0], y=embedding[tt_mask, 1], z=embedding[tt_mask, 2],
            mode='markers', name=str(trial_type),
            marker=dict(size=point_size, opacity=opacity, color=color),
            customdata=np.column_stack([positions, filtered_df['trial'].to_numpy()[tt_mask]]),
            text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
            hovertemplate=(
                f'{trial_type}<br>'
                'Pos: %{customdata[0]:.1f} cm<br>'
                'Cue: %{text}<br>'
                'Trial: %{customdata[1]:.0f}'
                '<extra></extra>'
            ),
        ))
    return traces


def _build_continuous_traces(embedding, filtered_df, strategy, point_size, opacity):
    """Build a single trace with continuous colorscale (speed, session progress)."""
    if strategy == ColoringStrategy.SPEED:
        values, cscale, label = filtered_df['speed_cm_s'].to_numpy(), 'Plasma', 'Speed (cm/s)'
        cmin, cmax = values.min(), values.max()
    elif strategy == ColoringStrategy.SESSION_PROGRESS:
        values, cscale, label = filtered_df['trial'].to_numpy(), 'YlOrBr', 'Session Progress'
        cmin, cmax = values.min(), values.max()
    else:
        raise ValueError(f"No continuous builder for {strategy}")

    cue_labels = pfmt.get_cue_labels()
    cues = filtered_df['cue'].to_numpy()
    positions = filtered_df['position'].to_numpy()

    trace = go.Scatter3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
        mode='markers', showlegend=False, name=label,
        marker=dict(
            size=point_size, opacity=opacity, color=values,
            colorscale=cscale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=label),
        ),
        customdata=np.column_stack([positions, filtered_df['trial'].to_numpy()]),
        text=[cue_labels.get(int(c), f'Cue {c}') for c in cues],
        hovertemplate=(
            'Pos: %{customdata[0]:.1f} cm<br>'
            'Cue: %{text}<br>'
            'Trial: %{customdata[1]:.0f}<br>'
            f'{label}: %{{marker.color:.1f}}'
            '<extra></extra>'
        ),
    )
    return [trace]


def _get_trace_builder(strategy: ColoringStrategy):
    """Return the appropriate trace builder for a strategy.
        - emb: embedding
        - df: dataframe
        - ps: point size
        - opacity: opacity

    """
    if strategy == ColoringStrategy.CUE:
        return _build_cue_traces
    elif strategy == ColoringStrategy.CUE_ID:
        return lambda emb, df, ps, op: _build_cue_traces(emb, df, ps, op, use_cue_id=True)
    elif strategy == ColoringStrategy.POSITION:
        return _build_position_traces
    elif strategy == ColoringStrategy.TRIAL_TYPE:
        return _build_trial_type_traces
    elif strategy in (ColoringStrategy.SPEED, ColoringStrategy.SESSION_PROGRESS):
        return lambda emb, df, ps, op: _build_continuous_traces(emb, df, strategy, ps, op)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------------------------
# MAIN PLOTTING FUNCTION
def plot_umap(
        embedding: np.ndarray,
        filtered_df:pl.DataFrame,
        strategy: str | ColoringStrategy | list[str | ColoringStrategy] = 'cue',
        point_size: int | float = 3,
        opacity: float = 0.7,
        title: str | None = None,
        save_path: Path | str | None = None,
        alpha: float = 0.6,
        s: float = 20,
):
    """Unified UMAP visualization.

    Handles 1D, 2D (matplotlib) and 3D (interactive Plotly) embeddings.

    For 3D plots:
        - Single strategy → one coloring mode with clickable legend / colorbar.
        - List of strategies → dropdown menu to switch between coloring modes.
        - All 3D plots include hover info (position, cue, trial, speed).
        - Categorical strategies (cue, trial_type) get clickable legend entries.
        - The 'position' strategy shows per-trial-type colorscales.

    Args:
        embedding: UMAP embedding of shape (n_frames, n_components).
        filtered_df: Filtered DataFrame from prepare_umap_data.
        strategy: Coloring strategy or list of strategies for toggle dropdown.
            Valid values: 'cue', 'position', 'trial_type', 'speed',
            'session_progress', or their ColoringStrategy equivalents.
            Pass a list (e.g. ['cue', 'position']) for a dropdown toggle.
        point_size: Marker size (Plotly for 3D, ignored for matplotlib — use `s`).
        opacity: Marker opacity (Plotly for 3D, use `alpha` for matplotlib).
        title: Plot title. Auto-generated if None.
        save_path: Save as .html (interactive) or image format. None to skip.
        alpha: Matplotlib scatter alpha (1D/2D only).
        s: Matplotlib scatter point size (1D/2D only).

    Returns:
        tuple of (figure, plot_info) where:
            - figure: plt.Figure (1D/2D) or go.Figure (3D)
            - plot_info: dict with 'n_components', 'strategy', 'n_frames',
              'n_trial_types', 'trial_types'
    """
    n_dims = embedding.shape[1] if embedding.ndim > 1 else 1

    # Normalize strategy input
    if not isinstance(strategy, list):
        strategy = [strategy]
    strategies = [ColoringStrategy(s) if isinstance(s, str) else s for s in strategy]

    # Build plot_info return dict
    trial_types = sorted(np.unique(filtered_df['trial_type'].to_numpy()).tolist())
    plot_info = {
        'n_components': n_dims,
        'strategy': [s_.value for s_ in strategies],
        'n_frames': len(embedding),
        'n_trial_types': len(trial_types),
        'trial_types': trial_types,
    }

    # ── 1D / 2D: matplotlib ──────────────────────────────────────────────
    if n_dims <= 2:
        if len(strategies) > 1:
            print("Warning: Multiple strategies only supported for 3D. Using first strategy.")
        fig = _plot_matplotlib(embedding, filtered_df, strategies[0], title, save_path, alpha, s)
        return fig, plot_info

    # ── 3D: Plotly ───────────────────────────────────────────────────────
    _check_plotly()
    fig = go.Figure()

    if len(strategies) == 1:
        # Single strategy — just add traces directly
        builder = _get_trace_builder(strategies[0])
        traces = builder(embedding, filtered_df, point_size, opacity)
        for t in traces:
            fig.add_trace(t)

        auto_title = f'3D UMAP — {strategies[0].value}'

    else:
        # Multiple strategies — build all trace groups, wire up dropdown
        trace_groups = []  # list of (strategy, traces)
        for strat in strategies:
            builder = _get_trace_builder(strat)
            traces = builder(embedding, filtered_df, point_size, opacity)
            trace_groups.append((strat, traces))

        # Add all traces, only first group visible
        for group_idx, (strat, traces) in enumerate(trace_groups):
            for t in traces:
                t.visible = (group_idx == 0)
                fig.add_trace(t)

        # Build visibility arrays for each dropdown button
        buttons = []
        offset = 0
        group_sizes = [len(traces) for _, traces in trace_groups]
        total_traces = sum(group_sizes)

        for group_idx, (strat, traces) in enumerate(trace_groups):
            vis = [False] * total_traces
            start = sum(group_sizes[:group_idx])
            for j in range(group_sizes[group_idx]):
                vis[start + j] = True
            buttons.append(dict(
                label=f'Color by {strat.value.replace("_", " ").title()}',
                method='update',
                args=[{'visible': vis}],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type='dropdown', direction='down',
                x=0.0, y=1.15, xanchor='left', yanchor='top',
                buttons=buttons,
            )],
        )

        labels = [s_.value for s_ in strategies]
        auto_title = f'3D UMAP — toggle: {" / ".join(labels)}'

    fig.update_layout(
        title=title or auto_title,
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant', groupclick='toggleitem'),
    )
    _show_and_save(fig, save_path)
    return fig, plot_info

# SPECLIAIZED PLOTS ---------------------

# 2D PLOT WITH KDE OVERLAY
#TODO impose cues on plots with legend so we know what we're actually looking at
def plot_umap_2d_density(
        embedding: np.ndarray,
        filtered_df: pl.DataFrame,
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
        filtered_df: Filtered DataFrame from prepare_umap_data.
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

    unique_types = np.unique(filtered_df['trial_type'].to_numpy())
    fig, axes = plt.subplots(1, len(unique_types) + 1, figsize=figsize)

    # Per-trial-type panels
    for ax, trial_type in zip(axes[:-1], unique_types):
        mask = filtered_df['trial_type'].to_numpy() == trial_type
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
        mask = filtered_df['trial_type'].to_numpy() == trial_type
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


# SINGLE-TRIAL TRAJECTORY OVERLAY UMAP
def plot_umap_3d_single_trial_trajectory(
        embedding: np.ndarray,
        filtered_df: pl.DataFrame,
        trial_ids: list[int] | None = None,
        n_trials_per_type: int = 5,
        trial_selection: Literal['middle', 'spaced'] = 'spaced',
        point_size: int = 3,
        line_width: float = 2.5,
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
        filtered_df: Filtered DataFrame from prepare_umap_data.
        trial_ids: Specific trial numbers to plot. If None, auto-selects.
        n_trials_per_type: Trials per type if trial_ids is None.
        trial_selection: How to pick trials when trial_ids is None.
            'middle' — consecutive trials from the middle of the session.
            'spaced' — evenly spaced across the full session.
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

    unique_types = np.unique(filtered_df['trial_type'].to_numpy())

    # Auto-select trials: pick n from middle of session for each type
    if trial_ids is None:
        trial_ids = []
        for trial_type in unique_types:
            type_mask = filtered_df['trial_type'].to_numpy() == trial_type
            type_trials = np.unique(filtered_df['trial'].to_numpy()[type_mask])
            n_select = min(n_trials_per_type, len(type_trials))

            if trial_selection == 'middle':
                mid = len(type_trials) // 2
                start = max(0, mid - n_select // 2)
                selected = type_trials[start:start + n_select]
            else:  # 'spaced'
                indices = np.linspace(0, len(type_trials) - 1, n_select, dtype=int)
                selected = type_trials[indices]

            trial_ids.extend(selected.tolist())

    fig = go.Figure()

#TODO fix background coloration for cues (rn only D showing up)
    # Background: all points faintly colored by cue for spatial context
    if show_background:
        cue_colors_map = pfmt.get_cue_colors()
        cue_labels = pfmt.get_cue_labels()
        unique_cues = np.unique(filtered_df['cue'].to_numpy())
        for cue_id in unique_cues:
            cue_mask = filtered_df['cue'] == cue_id
            cue_int = int(cue_id)
            label = cue_labels.get(cue_int, f'Cue {cue_int}')
            color = cue_colors_map.get(cue_int, '#CCCCCC')

            fig.add_trace(go.Scatter3d(
                x=embedding[cue_mask, 0], y=embedding[cue_mask, 1], z=embedding[cue_mask, 2],
                mode='markers', name=f'{label}',
                marker=dict(size=2, opacity=background_opacity, color=color),
                legendgroup='background',
                legendgrouptitle_text='Cue zones',
                showlegend=True,
                hoverinfo='skip',
            ))

    # Sort trial_ids so color gradient matches temporal order
    trial_ids = sorted(trial_ids)

    # Sequential colorscale for early → late trials
    n_trials = len(trial_ids)
    if n_trials <= 1:
        trial_colors = ['#FF6B6B']
    else:
        cmap = plt.cm.get_cmap('cool', n_trials)
        trial_colors = [
            f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}'
            for c in [cmap(i / (n_trials - 1)) for i in range(n_trials)]
        ]

    # Plot each selected trial as a connected line
    for i, trial_id in enumerate(trial_ids):
        trial_mask = filtered_df['trial'].to_numpy() == trial_id
        if not trial_mask.any():
            continue

        trial_emb = embedding[trial_mask]
        trial_pos = filtered_df['position'].to_numpy()[trial_mask]
        trial_cues = filtered_df['cue'].to_numpy()[trial_mask]
        trial_type = filtered_df['trial_type'].to_numpy()[trial_mask][0]
        color = trial_colors[i]

        # Line connecting consecutive frames
        fig.add_trace(go.Scatter3d(
            x=trial_emb[:, 0], y=trial_emb[:, 1], z=trial_emb[:, 2],
            mode='lines+markers',
            name=f'Trial {trial_id} ({trial_type})',
            line=dict(color=color, width=line_width),
            marker=dict(size=point_size, color=color, opacity=opacity),
            legendgroup='trajectories',
            legendgrouptitle_text='Trials (early→late)',
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
        title='UMAP: Single-Trial Trajectories (◆ = start, cool→warm = early→late)',
        scene=dict(xaxis=_PLOTLY_AXIS, yaxis=_PLOTLY_AXIS, zaxis=_PLOTLY_AXIS),
        legend=dict(x=1, y=0.9, itemsizing='constant', groupclick='toggleitem'),
    )
    _show_and_save(fig, save_path)
    return fig


# POSITION-MATCHED COMPARISON
#TODO increase onshared dot size, also consider what the point of this is
def plot_umap_3d_position_matched(
        embedding: np.ndarray,
        filtered_df: pl.DataFrame,
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
        filtered_df: Filtered DataFrame from prepare_umap_data.
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
        trial_types = sorted(np.unique(filtered_df['trial_type'].to_numpy()))
        max_lengths = {tt: filtered_df['nominal_track_length'].to_numpy()[filtered_df['trial_type'].to_numpy() == tt].max()
                       for tt in trial_types}
        max_position = min(max_lengths.values())

    fig = go.Figure()

    # Shared colorscale for matched positions
    trial_types = sorted(np.unique(filtered_df['trial_type'].to_numpy()))

    for i, trial_type in enumerate(trial_types):
        type_mask = filtered_df['trial_type'].to_numpy() == trial_type
        positions = filtered_df['position'].to_numpy()[type_mask]
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







###############
#Deleted the natural separation function bc it didn't tell much, btu might be useful again if I do the merging project


if __name__ == '__main__':
    from df_processing import (find_session_dir, get_session_paths, process_session,
                               load_session_context, load_processed_session, save_processed_session)

    mouse_id = '26'
    date = '2025-09-10'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, experiment_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)

    if paths['parquet'].exists():
        print(f"Loading: {paths['parquet']}")
        data, metadata = load_processed_session(paths['parquet'])
    else:
        print("No processed file found, processing from raw...")  # OR if you want to process the session with
        # other system states, bc the default is to process by run
        behavior_df = pl.read_ipc(paths['feather'])
        data, metadata = process_session(behavior_df, experiment_config)
        save_processed_session(data, session_dir, session_data, metadata)

    save_path = None  # Set to a Path to save figures


    # prepare data
    neural_data, filtered_df = prepare_umap_data(data, signal_column='multi_day_dff', max_frames=None)
    # compute umap
    embedding, _ = compute_umap(neural_data, n_components=3, n_neighbors=50)       #3D embedding

    # plot
    fig, meta = plot_umap(embedding, filtered_df, strategy=['trial_type', 'cue']) #basic plot, 3D

    fig1= plot_umap_3d_single_trial_trajectory(embedding, filtered_df, n_trials_per_type=10) # individual rtial plot

    #fig2D = plot_umap_2d_density(embedding, filtered_df) # 2D with KDE, needs a 2D embedding
