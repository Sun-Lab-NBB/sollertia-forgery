"""
Place Field Detection Module
Replaces the old place_1d module to work with our data structure

Detects and validates place fields from frame-level calcium imaging data.
Interfaces with df_processing.py — uses compute_session_averages() for binned tuning curves and frame-level data for
shuffle validation.

Core pipeline:
    1. detect_place_fields() — Threshold + connected components on session-averaged tuning curves
    2. validate_place_fields() — Chunk-shuffle test at frame level (no dask dependency)

Detection algorithm (Tank protocol):
    - Gaussian smooth binned tuning curves
    - Quantile-based thresholding per cell
    - Circular connected component detection (track wraps)
    - Filter by min field width, outside-field ratio, min peak amplitude

Output:
    PlaceFieldResult with per-trial-type PlaceFields1d objects and boolean is_place_cell arrays.

Dependencies: numpy, scipy, polars, scikit-image (regionprops). No vr2p or dask.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import label, gaussian_filter1d
from skimage.measure import regionprops

from df_processing import compute_session_averages, get_track_length


# DETECTION PARAMETERS
@dataclass
class DetectionParams:
    """Parameters for place field detection.

    Args:
        smooth_sigma: Gaussian smoothing sigma in bins (0 to disable).
        base_quantile: Quantile for per-cell baseline estimate.
        signal_threshold: Fraction of (peak - baseline) above baseline to threshold.
        min_bins: Minimum contiguous bins for a valid place field.
        outside_threshold: In-field mean must exceed outside-field mean × this factor.
        min_peak: Minimum peak ΔF/F within a field.
        sig_threshold: p-value cutoff for shuffle validation.
        n_shuffles: Number of shuffle iterations for validation.
        n_chunks: Number of chunks for temporal shuffle.
    """
    smooth_sigma: float = 1.0
    base_quantile: float = 0.25
    signal_threshold: float = 0.25
    min_bins: int = 3
    outside_threshold: float = 3.0
    min_peak: float = 0.1
    sig_threshold: float = 0.05
    n_shuffles: int = 500
    n_chunks: int = 100


# PLACE FIELDS CONTAINER

class PlaceFields1d:
    """Container for detected place fields on a 1D track.

    Holds labeled field image, binned fluorescence, and field centers.
    Provides properties for field statistics and cell ordering.

    Args:
        label_im: Labeled image of detected fields, shape (n_cells, n_bins).
            0 = no field, 1..N = field IDs.
        binF: Binned fluorescence data, shape (n_cells, n_bins).
        centers: Weighted centroids of fields, shape (n_fields, 2).
            Each row is [cell_idx, position_cm].
        bin_size: Spatial bin size in cm.
    """

    def __init__(
        self,
        label_im: np.ndarray,
        binF: np.ndarray,
        centers: np.ndarray | None = None,
        bin_size: float = 5.0,
    ):
        self.bin_size = bin_size
        self.label_im = label_im.astype(int)
        self.binF = binF.astype(float)

        if centers is not None and len(centers) > 0:
            self.centers = centers
        else:
            props = regionprops(self.label_im, self.binF, cache=False)
            if props:
                self.centers = np.array(
                    [prop["weighted_centroid"] * np.array([1, bin_size]) for prop in props]
                )
            else:
                self.centers = np.empty((0, 2))

    @property
    def n_fields(self) -> int:
        """Total number of detected fields."""
        return self.label_im.max()

    @property
    def mean_intensity(self) -> np.ndarray:
        """Mean intensity for each detected field.

        Returns:
            Array of shape (n_fields,).
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([p["mean_intensity"] for p in props]) if props else np.array([])

    @property
    def max_intensity(self) -> np.ndarray:
        """Max intensity for each detected field.

        Returns:
            Array of shape (n_fields,).
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([p["max_intensity"] for p in props]) if props else np.array([])

    @property
    def cell_id(self) -> np.ndarray:
        """Cell index for each detected field.

        Returns:
            Array of shape (n_fields,), int.
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([p["coords"][0, 0] for p in props], dtype=int) if props else np.array([], dtype=int)

    @property
    def has_place_field(self) -> np.ndarray:
        """Boolean mask: which cells have at least one detected field.

        Returns:
            Array of shape (n_cells,), bool.
        """
        return np.any(self.label_im > 0, axis=1)

    @property
    def order(self) -> np.ndarray:
        """Sort indices to order cells by place field center position.

        Cells with multiple fields are ordered by the field with highest mean intensity.
        Cells without fields are sorted to the end.

        Returns:
            Array of shape (n_cells,), int indices.
        """
        n_cells = self.binF.shape[0]
        sort_key = np.full(n_cells, np.inf)

        if self.centers.size == 0:
            return np.argsort(sort_key)

        intensity = self.mean_intensity
        cell_ids = self.cell_id

        for icell in range(n_cells):
            cell_mask = cell_ids == icell
            if cell_mask.any():
                best = np.argmax(intensity[cell_mask])
                field_indices = np.where(cell_mask)[0]
                sort_key[icell] = self.centers[field_indices[best], 1]

        return np.argsort(sort_key)

    def remove_fields(self, ind: np.ndarray) -> PlaceFields1d:
        """Remove specified fields by index.

        Args:
            ind: Indices of fields to remove (0-based, into the list of detected fields).

        Returns:
            New PlaceFields1d with fields removed and labels renumbered.
        """
        pf = deepcopy(self)
        if len(ind) == 0:
            return pf

        ind = np.atleast_1d(ind).flatten()

        # Zero out removed fields, renumber remaining
        pf.label_im[np.isin(pf.label_im, ind + 1)] = 0
        for counter, value in enumerate(np.unique(pf.label_im)):
            if value != 0:
                pf.label_im[pf.label_im == value] = counter

        pf.centers = np.delete(pf.centers, ind, axis=0)
        return pf

    def filter_cells(self, ind: np.ndarray) -> PlaceFields1d:
        """Keep only specified cells.

        Args:
            ind: Cell indices to keep.

        Returns:
            New PlaceFields1d containing only the specified cells.
        """
        pf = deepcopy(self)
        all_idx = np.arange(pf.label_im.shape[0])
        cell_ids = pf.cell_id

        pf.label_im[~np.isin(all_idx, ind), :] = 0

        for counter, value in enumerate(np.unique(pf.label_im)):
            if value != 0:
                pf.label_im[pf.label_im == value] = counter

        pf.centers = pf.centers[np.isin(cell_ids, ind), :]
        return pf

    def plot(
        self,
        title: str | None = None,
        sort: bool = True,
        cells: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        figsize: tuple = (4, 6),
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot place field activity as a sorted heatmap.

        Args:
            title: Plot title.
            sort: Sort cells by field position.
            cells: Boolean mask or indices of cells to include.
            vmin: Min color scale value (default: 50th percentile).
            vmax: Max color scale value (default: 90th percentile).
            figsize: Figure size.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure.
        """
        data = self.binF
        order = self.order if sort else np.arange(data.shape[0])

        if cells is not None:
            cells = np.atleast_1d(cells)
            if cells.dtype == bool:
                cells = np.where(cells)[0]
            order = order[np.isin(order, cells)]

        data = data[order, :]

        if vmin is None:
            vmin = np.nanquantile(data, 0.5)
        if vmax is None:
            vmax = np.nanquantile(data, 0.9)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        extent = [0, self.bin_size * data.shape[1], 1, data.shape[0] + 1]
        im = ax.imshow(data, cmap='magma', extent=extent, interpolation='none',
                        vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Cell #')
        if title:
            ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax, label='ΔF/F', shrink=0.8)
        plt.tight_layout()
        return fig


# RESULT CONTAINER

@dataclass
class PlaceFieldResult:
    """Results from place field detection across trial types.

    Args:
        fields: PlaceFields1d object per trial type.
        is_place_cell: Boolean array per trial type, shape (n_cells,).
        params: Detection parameters used.
        n_cells: Total number of cells.
        p_values: Per-cell p-values from shuffle test (None if not validated).
        sig_cells: Indices of significant cells after validation (None if not validated).
    """
    fields: dict[str, PlaceFields1d] = field(default_factory=dict)
    is_place_cell: dict[str, np.ndarray] = field(default_factory=dict)
    params: DetectionParams | None = None
    n_cells: int = 0
    p_values: dict[str, np.ndarray] = field(default_factory=dict)
    sig_cells: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_place_cell_any(self) -> np.ndarray:
        """Boolean mask: cell is a place cell in ANY trial type.

        Returns:
            Array of shape (n_cells,), bool.
        """
        if not self.is_place_cell:
            return np.zeros(self.n_cells, dtype=bool)
        return np.any(np.stack(list(self.is_place_cell.values())), axis=0)

    @property
    def is_place_cell_all(self) -> np.ndarray:
        """Boolean mask: cell is a place cell in ALL trial types.

        Returns:
            Array of shape (n_cells,), bool.
        """
        if not self.is_place_cell:
            return np.zeros(self.n_cells, dtype=bool)
        return np.all(np.stack(list(self.is_place_cell.values())), axis=0)

    def summary(self) -> str:
        """Print summary of detection results.

        Returns:
            Formatted summary string.
        """
        lines = [f"Place field detection: {self.n_cells} cells"]
        for tt, mask in self.is_place_cell.items():
            n = mask.sum()
            pct = 100 * n / self.n_cells if self.n_cells > 0 else 0
            n_fields = self.fields[tt].n_fields if tt in self.fields else 0
            lines.append(f"  {tt}: {n}/{self.n_cells} place cells ({pct:.1f}%), {n_fields} fields")
        n_any = self.is_place_cell_any.sum()
        lines.append(f"  Any type: {n_any}/{self.n_cells} ({100 * n_any / self.n_cells:.1f}%)")
        if self.p_values:
            lines.append(f"  Validated with {self.params.n_shuffles} shuffles (p < {self.params.sig_threshold})")
        return '\n'.join(lines)


# SIGNAL PROCESSING (replaces vr2p)

def _quantile_threshold(
    binF: np.ndarray,
    base_quantile: float = 0.25,
    signal_threshold: float = 0.25,
) -> np.ndarray:
    """Quantile-based thresholding per cell. Replaces vr2p.quantile_max_threshold.

    For each cell, computes baseline as the base_quantile quantile of its tuning curve.
    A bin passes threshold if: value > baseline + signal_threshold × (peak - baseline).

    Args:
        binF: Binned fluorescence, shape (n_cells, n_bins).
        base_quantile: Quantile for baseline (0-1).
        signal_threshold: Fraction of dynamic range above baseline.

    Returns:
        Binary array, shape (n_cells, n_bins). 1 = above threshold.
    """
    baseline = np.nanquantile(binF, base_quantile, axis=1, keepdims=True)
    peak = np.nanmax(binF, axis=1, keepdims=True)
    threshold = baseline + signal_threshold * (peak - baseline)
    return (binF > threshold).astype(int)


# CORE DETECTION ALGORITHMS

def circular_connected_placefields(
    thres_im: np.ndarray,
    binF: np.ndarray,
    min_bins: int = 3,
) -> PlaceFields1d:
    """Detect place fields using circular connected components.

    Handles wrap-around connectivity (track is a loop) by padding the binary image,
    running connected component labeling, then mapping back to original coordinates.

    Args:
        thres_im: Binary thresholded image, shape (n_cells, n_bins).
        binF: Binned fluorescence, shape (n_cells, n_bins).
        min_bins: Minimum contiguous bins for a valid field.

    Returns:
        PlaceFields1d with detected fields.
    """
    num_bins = thres_im.shape[1]

    # Circular padding
    pad_thres = np.pad(thres_im, ((0, 0), (num_bins, num_bins)), mode='wrap')
    F_padded = np.pad(binF, ((0, 0), (num_bins, num_bins)), mode='wrap')

    # Connected components (horizontal connectivity only — each row is independent)
    label_im, _ = label(pad_thres, [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    props = np.array(regionprops(label_im, F_padded, cache=False))

    if len(props) == 0:
        return PlaceFields1d(
            np.zeros(thres_im.shape, dtype=np.uint32), binF,
            centers=np.empty((0, 2)),
        )

    centers = np.array([p['weighted_centroid'] for p in props])
    area = np.array([p['area'] for p in props], dtype=np.uint32)

    # Keep components with center in the original (non-padded) region + min size
    valid = (centers[:, 1] >= num_bins) & (centers[:, 1] < num_bins * 2) & (area >= min_bins)

    result_label = np.zeros(thres_im.shape, dtype=np.uint32)
    adj_centers = []

    for counter, prop in enumerate(props[valid]):
        coords = prop['coords']
        wrapped_bins = np.take(np.arange(num_bins), coords[:, 1], mode='wrap')
        result_label[coords[:, 0], wrapped_bins] = counter + 1

        center = np.array(prop['weighted_centroid'])
        center[1] -= num_bins
        adj_centers.append(center)

    centers_out = np.vstack(adj_centers) if adj_centers else np.empty((0, 2))
    return PlaceFields1d(result_label, binF, centers=centers_out)


def outside_field_threshold(
    pf: PlaceFields1d,
    threshold_factor: float = 3.0,
) -> PlaceFields1d:
    """Remove fields where in-field activity is not sufficiently above baseline.

    For each cell, computes mean activity outside all its fields. A field is removed
    if its mean intensity < threshold_factor × outside mean.

    Args:
        pf: PlaceFields1d with detected fields.
        threshold_factor: Required ratio of in-field to outside-field activity.

    Returns:
        Filtered PlaceFields1d.
    """
    if pf.n_fields == 0:
        return pf

    outside_im = pf.binF.copy()
    outside_im[pf.label_im != 0] = np.nan
    outside_values = np.nanmean(outside_im, axis=1)

    cell_ids = pf.cell_id
    threshold_values = outside_values[cell_ids] * threshold_factor
    invalid = np.where(pf.mean_intensity < threshold_values)[0]

    return pf.remove_fields(invalid)


def _detect_on_tuning_curves(
    binF: np.ndarray,
    params: DetectionParams,
    bin_size: float = 5.0,
) -> PlaceFields1d:
    """Run detection pipeline on pre-computed tuning curves.

    Internal function used by both detect_place_fields and the shuffle test.

    Args:
        binF: Binned fluorescence, shape (n_cells, n_bins).
        params: Detection parameters.
        bin_size: Spatial bin size in cm.

    Returns:
        PlaceFields1d with detected fields.
    """
    # Smooth
    if params.smooth_sigma > 0:
        smoothed = np.apply_along_axis(
            gaussian_filter1d, axis=1, arr=binF, sigma=params.smooth_sigma,
            mode='wrap',
        )
    else:
        smoothed = binF

    # Threshold
    thres_im = _quantile_threshold(smoothed, params.base_quantile, params.signal_threshold)

    # Connected components (circular)
    pf = circular_connected_placefields(thres_im, smoothed, min_bins=params.min_bins)
    pf.bin_size = bin_size

    # Filter: outside-field ratio
    pf = outside_field_threshold(pf, params.outside_threshold)

    # Filter: minimum peak amplitude
    if pf.n_fields > 0:
        weak = np.where(pf.max_intensity < params.min_peak)[0]
        pf = pf.remove_fields(weak)

    return pf


# MAIN

def detect_place_fields(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    params: DetectionParams | None = None,
) -> PlaceFieldResult:
    """Detect place fields from frame-level data.

    Computes session-averaged tuning curves via compute_session_averages(), then runs
    threshold + connected component detection per trial type.

    Args:
        df: Frame-level DataFrame from process_session() (after fix_cue_offset).
        config: Experiment configuration dict.
        signal_col: Column containing neural signals (list per frame).
        bin_size_cm: Spatial bin size in cm.
        params: Detection parameters. Uses defaults if None.

    Returns:
        PlaceFieldResult with per-trial-type detection results.
    """
    if params is None:
        params = DetectionParams()

    # Get session-averaged tuning curves: {trial_type: {'session_avg': (n_bins, n_cells), ...}}
    stats = compute_session_averages(
        df, signal_col=signal_col, config=config, bin_size_cm=bin_size_cm,
    )

    # Determine n_cells from first trial type
    first_tt = next(iter(stats))
    n_cells = stats[first_tt]['session_avg'].shape[1]

    result = PlaceFieldResult(params=params, n_cells=n_cells)

    for tt, tt_stats in stats.items():
        # session_avg is (n_bins, n_cells) — transpose to (n_cells, n_bins)
        binF = tt_stats['session_avg'].T

        print(f"Detecting place fields for {tt}: {binF.shape[0]} cells × {binF.shape[1]} bins...")

        pf = _detect_on_tuning_curves(binF, params, bin_size=bin_size_cm)

        result.fields[tt] = pf
        result.is_place_cell[tt] = pf.has_place_field

        n_pc = pf.has_place_field.sum()
        print(f"  {tt}: {n_pc} place cells ({pf.n_fields} fields)")

    return result


def validate_place_fields(
    df: pl.DataFrame,
    config: dict,
    result: PlaceFieldResult | None = None,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int = 5,
    params: DetectionParams | None = None,
    seed: int = 42,
) -> PlaceFieldResult:
    """Validate place fields with a chunk-shuffle test.  This slows down processing significantly.

    For each shuffle iteration:
        1. Split frame-level signals into temporal chunks
        2. Randomly permute chunk order (breaks position-signal mapping)
        3. Re-bin into tuning curves
        4. Run detection on shuffled tuning curves
        5. Record which cells have fields

    p-value = fraction of shuffles producing a field for each cell.
    Cells with p < sig_threshold AND a detected field are marked significant.

    Args:
        df: Frame-level DataFrame from process_session().
        config: Experiment configuration dict.
        result: Existing PlaceFieldResult to update. If None, runs detection first.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        params: Detection parameters. Uses result.params or defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        Updated PlaceFieldResult with p_values and sig_cells populated.
    """
    if params is None:
        params = result.params if result is not None else DetectionParams()

    # Run detection if not provided
    if result is None:
        result = detect_place_fields(df, config, signal_col, bin_size_cm, params)

    rng = np.random.default_rng(seed)

    # Extract all signals once
    all_signals = np.vstack(df[signal_col].to_list())  # (n_frames, n_cells)
    trials = df['trial'].to_numpy()
    bins = df['distance_bin'].to_numpy()
    trial_types = df['trial_type'].to_numpy()
    n_cells = all_signals.shape[1]

    def _bin_signals(signals, mask, n_bins):
        """Bin signals for a subset of frames, same logic as compute_session_averages."""
        sub_signals = signals[mask]
        sub_trials = trials[mask]
        sub_bins = bins[mask]

        unique_trials = np.unique(sub_trials)
        n_trials = len(unique_trials)
        trial_idx = np.searchsorted(unique_trials, sub_trials)
        bin_idx = sub_bins.clip(0, n_bins - 1)

        sums = np.zeros((n_trials, n_bins, n_cells))
        counts = np.zeros((n_trials, n_bins, 1))
        np.add.at(sums, (trial_idx, bin_idx), sub_signals)
        np.add.at(counts, (trial_idx, bin_idx, 0), 1)

        with np.errstate(invalid='ignore'):
            per_trial = sums / counts

        # Average across trials → (n_bins, n_cells), then transpose → (n_cells, n_bins)
        return np.nanmean(per_trial, axis=0).T

    # Shuffle and re-detect per trial type
    for tt in result.fields:
        mask = trial_types == tt
        n_frames_tt = mask.sum()
        n_bins = int(get_track_length(config, tt) / bin_size_cm)

        print(f"Validating {tt}: {params.n_shuffles} shuffles, {n_frames_tt} frames...")

        observed = result.fields[tt].has_place_field  # (n_cells,)
        shuffle_counts = np.zeros(n_cells, dtype=int)

        # Frame indices for this trial type
        tt_indices = np.where(mask)[0]
        n_per_chunk = max(1, len(tt_indices) // params.n_chunks)

        for i in range(params.n_shuffles):
            # Chunk-shuffle: split frame indices into chunks, permute chunk order
            chunks = [tt_indices[j:j + n_per_chunk] for j in range(0, len(tt_indices), n_per_chunk)]
            perm = rng.permutation(len(chunks))
            shuffled_indices = np.concatenate([chunks[p] for p in perm])

            # Create shuffled signal array: signals are reordered, positions stay fixed
            shuffled_signals = all_signals.copy()
            shuffled_signals[tt_indices] = all_signals[shuffled_indices]

            # Re-bin and detect
            shuffled_binF = _bin_signals(shuffled_signals, mask, n_bins)
            shuf_pf = _detect_on_tuning_curves(shuffled_binF, params, bin_size=bin_size_cm)
            shuffle_counts += shuf_pf.has_place_field.astype(int)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{params.n_shuffles}")

        # p-value = fraction of shuffles where field was detected
        p_values = shuffle_counts / params.n_shuffles
        sig_mask = observed & (p_values < params.sig_threshold)

        result.p_values[tt] = p_values
        result.sig_cells[tt] = np.where(sig_mask)[0]

        # Update is_place_cell to only include validated cells
        result.is_place_cell[tt] = sig_mask

        n_sig = sig_mask.sum()
        n_detected = observed.sum()
        print(f"  {tt}: {n_sig}/{n_detected} cells survived validation "
              f"(p < {params.sig_threshold})")

    return result


# CONVENIENCE

def get_place_cell_indices(
    result: PlaceFieldResult,
    trial_type: str | None = None,
    require_all: bool = False,
) -> np.ndarray:
    """Get indices of place cells.

    Args:
        result: PlaceFieldResult from detect/validate.
        trial_type: Specific trial type, or None for across all types.
        require_all: If True and trial_type is None, cell must be a place cell
            in ALL trial types. If False, in ANY type.

    Returns:
        Array of cell indices (int).
    """
    if trial_type is not None:
        return np.where(result.is_place_cell[trial_type])[0]

    mask = result.is_place_cell_all if require_all else result.is_place_cell_any
    return np.where(mask)[0]

def plot_combined_heatmap(
    result: PlaceFieldResult,
    config: dict,
    session_data: dict,
    trial_types: list[str] | None = None,
    cells: np.ndarray | None = None,
    sort_by: str | None = None,
    bin_size_cm: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple | None = None,
    dpi: int = 150,
    show: bool = True,
) -> plt.Figure:
    """Plot heatmap with trial types concatenated horizontally.

    Concatenates tuning curves from multiple trial types side by side,
    with a dashed line at each boundary. Cells sorted by field position.
    Cue regions annotated across the full width.

    Args:
        result: PlaceFieldResult from detect_place_fields().
        config: Experiment configuration dict.
        trial_types: Which trial types to include, in order. Default: all, sorted.
        cells: Cell indices or boolean mask to include. Default: all place cells.
        sort_by: Trial type to sort cells by field position. Default: first trial type.
        bin_size_cm: Spatial bin size in cm.
        vmin: Min color scale (default: 50th percentile).
        vmax: Max color scale (default: 90th percentile).
        figsize: Figure size. Auto-scaled if None.
        dpi: Figure resolution.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    from trial_plotting import get_cue_colors, get_cue_labels
    from df_processing import get_cue_regions

    if trial_types is None:
        trial_types = sorted(result.fields.keys())
    if sort_by is None:
        sort_by = trial_types[0]

    # Concatenate tuning curves horizontally: (n_cells, total_bins)
    segments = []
    boundaries = [0]
    for tt in trial_types:
        binF = result.fields[tt].binF
        segments.append(binF)
        boundaries.append(boundaries[-1] + binF.shape[1])

    combined = np.concatenate(segments, axis=1)

    # Filter cells
    if cells is not None:
        cells = np.atleast_1d(cells)
        if cells.dtype == bool:
            cells = np.where(cells)[0]
    else:
        cells = np.where(result.is_place_cell_any)[0]

    # Sort by field position in sort_by trial type
    sort_pf = result.fields[sort_by]
    sort_key = np.full(combined.shape[0], np.inf)
    if sort_pf.centers.size > 0:
        cell_ids = sort_pf.cell_id
        intensities = sort_pf.mean_intensity
        for icell in cells:
            cell_mask = cell_ids == icell
            if cell_mask.any():
                best = np.argmax(intensities[cell_mask])
                field_idx = np.where(cell_mask)[0][best]
                sort_key[icell] = sort_pf.centers[field_idx, 1]

    order = cells[np.argsort(sort_key[cells])]
    data = combined[order, :]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Color scale
    if vmin is None:
        vmin = np.nanquantile(data, 0.5)
    if vmax is None:
        vmax = np.nanquantile(data, 0.9)

    # Figure — use gridspec so colorbar doesn't break alignment
    total_cm = sum(get_track_length(config, tt) for tt in trial_types)
    n_cells_plot = len(order)
    if figsize is None:
        figsize = (8, 6)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 30],
        hspace=0.02,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_cue = fig.add_subplot(gs[0, 0])

    # Heatmap
    extent = [0, total_cm, n_cells_plot, 0]
    im = ax.imshow(data, cmap='magma', extent=extent, interpolation='none',
                   vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Neuron #')

    # Colorbar in its own axis — same height as heatmap
    ax_cb = ax.inset_axes([1.02, 0.0, 0.02, 1.0])
    #if 'dff' in signal_col:
        #label = 'ΔF/F'
    #elif 'spikes' in signal_col:
        #label = 'Spikes'   #does that make sense? would we ever use the spike data?
    #else:
        #label = 'Raw fluorescence'
    plt.colorbar(im, cax=ax_cb, label='ΔF/F')

    # Trial type boundaries
    for b in boundaries[1:-1]:
        x = b * bin_size_cm
        ax.axvline(x, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax_cue.axvline(x, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Cue bar
    cue_colors = get_cue_colors(config)
    cue_labels_map = get_cue_labels(config)
    x_offset = 0

    for tt in trial_types:
        regions = get_cue_regions(config, tt)
        for cue_id, spans in regions.items():
            color = cue_colors.get(cue_id, '#CCCCCC')
            label = cue_labels_map.get(cue_id, f'Cue {cue_id}')

            for start, end in spans:
                x0 = x_offset + start
                x1 = x_offset + end
                ax_cue.axvspan(x0, x1, color=color, alpha=0.6)

                if cue_id != 0:
                    ax_cue.text((x0 + x1) / 2, 0.5, label,
                                ha='center', va='center', fontsize=9,
                                fontweight='bold')

        x_offset += get_track_length(config, tt)

    # X-ticks: reset to per-track position labels
    tick_positions = []
    tick_labels = []
    x_offset = 0
    for tt in trial_types:
        track_len = get_track_length(config, tt)
        # Ticks every 30cm within each track
        local_ticks = np.arange(0, track_len + 1, 30)
        for t in local_ticks:
            tick_positions.append(x_offset + t)
            tick_labels.append(str(int(t)))
        x_offset += track_len

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    # Force cue bar to match heatmap position exactly
    fig.canvas.draw()
    ax_pos = ax.get_position()
    cue_pos = ax_cue.get_position()
    ax_cue.set_position([ax_pos.x0, cue_pos.y0, ax_pos.width, cue_pos.height])
    ax_cue.set_xlim(ax.get_xlim())
    ax_cue.set_autoscalex_on(False)
    ax_cue.set_yticks([])
    ax_cue.tick_params(bottom=False, labelbottom=False)
    ax_cue.spines[:].set_visible(False)

    prefix = ''
    if session_data:
        prefix = f'Mouse {session_data["animal_id"]} — {session_data["session_name"][:10]} — '

    ax_cue.set_title(
        f'{prefix}Place Fields ({n_cells_plot} cells, sorted by {sort_by} trials)',
        fontsize=11, fontweight='bold', pad=6,
    )

    plt.subplots_adjust(top=0.95, bottom=0.08)

    if show:
        plt.show()

    return fig

# MULTIDAY DETECTION

@dataclass
class MultidayPlaceFieldResult:
    """Results from place field detection across multiple sessions.

    Args:
        per_day: Per-date PlaceFieldResult.
        dates: Sorted list of session dates.
        trial_types: Union of all trial types across days.
        n_cells: Total number of registered cells.
        presence: Boolean presence matrix per trial type, shape (n_cells, n_days).
        centers: Field center position per trial type, shape (n_cells, n_days). NaN if no field.
        union_indices: Cell indices that are place cells on >= min_days.
        params: Detection parameters used.
    """
    per_day: dict[str, PlaceFieldResult] = field(default_factory=dict)
    dates: list[str] = field(default_factory=list)
    trial_types: list[str] = field(default_factory=list)
    n_cells: int = 0
    presence: dict[str, np.ndarray] = field(default_factory=dict)
    centers: dict[str, np.ndarray] = field(default_factory=dict)
    union_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    params: DetectionParams | None = None

    def stability_score(self, trial_type: str) -> np.ndarray:
        """Fraction of days each cell has a place field.

        Args:
            trial_type: Trial type to compute stability for.

        Returns:
            Array of shape (n_cells,), float 0-1.
        """
        return self.presence[trial_type].mean(axis=1)

    def summary(self) -> str:
        """Print summary of multiday detection results.

        Returns:
            Formatted summary string.
        """
        lines = [f"Multiday place field detection: {self.n_cells} cells, {len(self.dates)} days"]
        for tt in self.trial_types:
            p = self.presence[tt]
            n_ever = np.any(p, axis=1).sum()
            n_all = np.all(p, axis=1).sum()
            mean_stab = self.stability_score(tt).mean()
            lines.append(
                f"  {tt}: {n_ever} cells with field on ≥1 day, "
                f"{n_all} on all days, mean stability={mean_stab:.2f}"
            )

            # Per-day counts
            day_counts = [f"{d[5:]}: {int(p[:, i].sum())}" for i, d in enumerate(self.dates)]
            lines.append(f"    Per day: {', '.join(day_counts)}")

        lines.append(f"  Union (any type, any day): {len(self.union_indices)} cells")
        return '\n'.join(lines)


def detect_multiday_place_fields(
    sessions: dict[str, dict],
    signal_col: str = 'single_day_dff',
    bin_size_cm: int = 5,
    params: DetectionParams | None = None,
    min_days: int = 1,
) -> MultidayPlaceFieldResult:
    """Detect place fields independently per session, then combine across days.

    Runs detect_place_fields() on each session. Builds a presence matrix showing
    which cells have fields on which days, and a union of all place cell indices.
    No day is privileged — a cell appearing on only the last day is included.

    Args:
        sessions: From load_multiday_sessions(). Each value has keys:
            'data', 'config', 'session_data', 'metadata'.
        signal_col: Column containing neural signals.
        bin_size_cm: Spatial bin size in cm.
        params: Detection parameters. Uses defaults if None.
        min_days: Minimum number of days a cell must have a field to be
            included in union_indices. Default 1 (any day).

    Returns:
        MultidayPlaceFieldResult with per-day results and cross-day summaries.
    """
    if params is None:
        params = DetectionParams()

    dates = sorted(sessions.keys())
    n_days = len(dates)

    # Detect per day
    per_day = {}
    for date in dates:
        s = sessions[date]
        print(f"\n--- {date} ---")
        per_day[date] = detect_place_fields(
            s['data'], s['config'],
            signal_col=signal_col,
            bin_size_cm=bin_size_cm,
            params=params,
        )

    # Get n_cells and trial types from first day (registered cells = same count)
    n_cells = per_day[dates[0]].n_cells
    all_trial_types = sorted(set(
        tt for r in per_day.values() for tt in r.fields
    ))

    # Build presence matrix and center trajectories per trial type
    presence = {}
    centers_mat = {}

    for tt in all_trial_types:
        pres = np.zeros((n_cells, n_days), dtype=bool)
        ctrs = np.full((n_cells, n_days), np.nan)

        for di, date in enumerate(dates):
            result = per_day[date]
            if tt not in result.fields:
                continue

            pf = result.fields[tt]
            pres[:, di] = pf.has_place_field

            # Extract field center for each cell (use strongest field if multiple)
            if pf.centers.size > 0:
                cell_ids = pf.cell_id
                intensities = pf.mean_intensity
                for icell in range(n_cells):
                    cell_mask = cell_ids == icell
                    if cell_mask.any():
                        best = np.argmax(intensities[cell_mask])
                        field_idx = np.where(cell_mask)[0][best]
                        ctrs[icell, di] = pf.centers[field_idx, 1]

        presence[tt] = pres
        centers_mat[tt] = ctrs

    # Union indices: place cell on >= min_days in ANY trial type
    any_type_days = np.zeros(n_cells, dtype=int)
    for tt in all_trial_types:
        any_type_days = np.maximum(any_type_days, presence[tt].sum(axis=1))
    union_idx = np.where(any_type_days >= min_days)[0]

    multiday_result = MultidayPlaceFieldResult(
        per_day=per_day,
        dates=dates,
        trial_types=all_trial_types,
        n_cells=n_cells,
        presence=presence,
        centers=centers_mat,
        union_indices=union_idx,
        params=params,
    )

    print(f"\n{multiday_result.summary()}")
    return multiday_result


def save_multiday_result(result: MultidayPlaceFieldResult, path: Path):
    """Save multiday detection results to .npz.

    Saves the cross-day summary (presence matrix, centers, union indices)
    and detection params. Does NOT save per-day PlaceFields1d objects — re-run
    detection if you need those.

    Args:
        result: MultidayPlaceFieldResult to save.
        path: Output path (.npz).
    """
    path = Path(path)
    save_dict = {
        'dates': np.array(result.dates),
        'trial_types': np.array(result.trial_types),
        'n_cells': result.n_cells,
        'union_indices': result.union_indices,
    }

    for tt in result.trial_types:
        save_dict[f'{tt}_presence'] = result.presence[tt]
        save_dict[f'{tt}_centers'] = result.centers[tt]

    np.savez(path, **save_dict)
    print(f"Saved: {path}")


def load_multiday_result(path: Path) -> MultidayPlaceFieldResult:
    """Load multiday detection summary from .npz.

    Note: per_day PlaceFieldResult objects will be empty. Re-run detection
    if you need full per-day fields.

    Args:
        path: Path to .npz file.

    Returns:
        MultidayPlaceFieldResult (without per_day populated).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    trial_types = [str(tt) for tt in data['trial_types']]

    presence = {}
    centers = {}
    for tt in trial_types:
        presence[tt] = data[f'{tt}_presence']
        centers[tt] = data[f'{tt}_centers']

    return MultidayPlaceFieldResult(
        dates=[str(d) for d in data['dates']],
        trial_types=trial_types,
        n_cells=int(data['n_cells']),
        presence=presence,
        centers=centers,
        union_indices=data['union_indices'],
    )





if __name__ == '__main__':
    from df_processing import load_session_dir, get_session_prefix, load_processed_session, load_multiday_sessions
    from trial_plotting import plot_multiday_comparison

    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/26_explore')

    # ── Single-day detection ──
    date = '2025-09-15'
    session_data, config, behavior_path = load_session_dir(mouse_dir, date)
    prefix = get_session_prefix(session_data)
    data, meta = load_processed_session(behavior_path.parent / f'{prefix}_processed.parquet')

    result = detect_place_fields(data, config, signal_col='multi_day_dff')
    print(result.summary())

    # Sorted heatmap per trial type
    # for tt, pf in result.fields.items():
    #     pf.plot(title=f'{tt} Place Fields — {date}', sort=True)
    #     plt.show()

    # All place cells, sorted by ABC field position
    plot_combined_heatmap(result, config, session_data)

    # ── Multiday detection ──
    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-03', '2025-09-24'), auto_process=False,
    )

    multiday = detect_multiday_place_fields(sessions, signal_col='multi_day_dff')
    save_multiday_result(multiday, mouse_dir / 'place_field_results.npz')

    # Heatmap per day for union cells
    for date, day_result in multiday.per_day.items():
        for tt, pf in day_result.fields.items():
            pf.plot(
                title=f'{tt} — {date}', sort=True,
                cells=multiday.union_indices,
            )
            plt.show()

    # Only union cells from multiday, sorted by ABDC
    plot_combined_heatmap(result, config, cells=multiday.union_indices, sort_by='ABDC')
    plot_combined_heatmap(result, config, cells=multiday.union_indices, sort_by='ABC')

    # Multiday comparison for top place cells
    for i in multiday.union_indices[:5]:
        plot_multiday_comparison(sessions, cell_idx=i, signal_col='multi_day_dff')