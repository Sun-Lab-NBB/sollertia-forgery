"""
Place Field Detection
=====================
Single-session place field detection from frame-level calcium imaging data.

Pipeline:
    1. detect_place_fields() — threshold + connected components on session-averaged tuning curves
    2. validate_place_fields() — chunk-shuffle test at frame level

Output: PlaceFieldResult with per-trial-type PlaceFields1d objects and is_place_cell arrays.

This module knows nothing about plotting, multi-session aggregation, or remapping.
For those, see place_field_plotting.py, experiment_place_cells.py, remapping_analysis.py.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import label, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from skimage.measure import regionprops

from df_processing import compute_session_averages, get_track_length


# DETECTION PARAMETERS
@dataclass
class DetectionParams:
    """Parameters for place field detection.

    Args:
        smooth_sigma: Gaussian smoothing sigma in bins (0 to disable).
        base_quantile: Quantile used to identify sub-threshold bins for dff baseline
            estimation. Ignored for spikes.
        signal_threshold: For spikes: fraction of peak rate. For dff: fraction of
            (peak - baseline) above baseline.
        min_bins: Minimum contiguous bins for a valid place field.
        outside_threshold: In-field mean must exceed outside-field mean × this factor.
        min_peak: Minimum peak value within a field. If None (default), no peak
            filter is applied — detection relies on outside_threshold instead.
        min_speed_cm_s: Minimum speed threshold in cm/s. Frames below this speed are
            excluded before binning. Set to None to disable speed filtering.
        sig_threshold: p-value cutoff for shuffle validation.
        n_shuffles: Number of shuffle iterations for validation.
        n_chunks: Number of chunks for temporal shuffle.
    """
    method: str = 'threshold'
    smooth_sigma: float = 1.0
    base_quantile: float = 0.25
    signal_threshold: float = 0.20
    min_bins: int = 2
    outside_threshold: float = 3.0
    min_peak: float | None = None
    min_speed_cm_s: float | None = 5.0
    sig_threshold: float = 0.05
    n_shuffles: int = 500
    n_chunks: int = 100
    # Prominence-detection params (used only when method='prominence').
    # Independence by construction: each peak is evaluated against its own local
    # valleys via scipy.signal.find_peaks(prominence=...) rather than a global
    # per-cell threshold. A tall peak in one location cannot suppress detection
    # of a separate peak elsewhere on the track.
    min_prominence_frac: float = 0.20
    field_rel_height: float = 0.5


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
        bin_size_cm: Spatial bin size in cm.
    """
# TODO change the bin size to match the metadata
    def __init__(
        self,
        label_im: np.ndarray,
        binF: np.ndarray,
        bin_size_cm: float,
        centers: np.ndarray | None = None,
    ):
        self.bin_size_cm = bin_size_cm
        self.label_im = label_im.astype(int)
        self.binF = binF.astype(float)

        if centers is not None and len(centers) > 0:
            self.centers = centers * np.array([1, bin_size_cm])
        else:
            props = regionprops(self.label_im, self.binF, cache=False)
            if props:
                self.centers = np.array(
                    [prop["weighted_centroid"] * np.array([1, bin_size_cm]) for prop in props]
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
        best_intensity = np.full(n_cells, -np.inf)

        props = regionprops(self.label_im, self.binF, cache=False)
        if not props:
            return np.argsort(sort_key)

        for prop in props:
            cell_idx = prop['coords'][0, 0]
            mean_int = prop['mean_intensity']
            if mean_int > best_intensity[cell_idx]:
                best_intensity[cell_idx] = mean_int
                sort_key[cell_idx] = prop['weighted_centroid'][1]

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

        # Map field-index -> actual label number via the sorted unique-labels
        # list. regionprops returns props in ascending label order, so the i-th
        # unique non-zero label IS the label for field-index i. This holds
        # whether labels are contiguous (1..N) or have gaps (e.g. prominence
        # detection can leave label gaps when overlapping fields overwrite each
        # other's bins). The old `ind + 1` arithmetic assumed contiguous labels
        # and silently wiped the wrong field whenever a gap was present.
        unique_labels = np.unique(pf.label_im)
        unique_labels = unique_labels[unique_labels > 0]
        labels_to_remove = unique_labels[ind]
        pf.label_im[np.isin(pf.label_im, labels_to_remove)] = 0

        # Renumber surviving labels to contiguous 1..K so downstream consumers
        # can keep using simple counters.
        remaining = np.unique(pf.label_im)
        remaining = remaining[remaining > 0]
        new_label_im = np.zeros_like(pf.label_im)
        for new_id, old_id in enumerate(remaining, start=1):
            new_label_im[pf.label_im == old_id] = new_id
        pf.label_im = new_label_im

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

    def plot(self, *args, **kwargs):
        """Plot place field heatmap. Backwards-compat wrapper.

        Implementation lives in place_field_plotting.plot_place_fields() to keep
        this module free of plotting dependencies.
        """
        from place_field_plotting import plot_place_fields
        return plot_place_fields(self, *args, **kwargs)

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
    signal_type: str = 'spikes',
    base_quantile: float = 0.25,
    signal_threshold: float = 0.25,
) -> np.ndarray:
    """Threshold binned activity per cell to identify place field candidates.

    For spikes: threshold = signal_threshold × peak. Baseline is assumed to be
    zero since most bins in a sparse spike map will be at or near zero.

    For dff: baseline is computed as the mean of bins at or below the base_quantile
    quantile (not the quantile value itself). Threshold = baseline + signal_threshold
    × (peak - baseline). This matches the Tank lab protocol from place_cell_analysis.py
    and avoids the quantile-as-baseline error where high activity inflates the baseline.

    Args:
        binF: Binned activity, shape (n_cells, n_bins).
        signal_type: Either 'spikes' or 'dff'.
        base_quantile: Quantile for identifying sub-threshold bins. Only used for dff.
        signal_threshold: Fraction of dynamic range to threshold at.

    Returns:
        Binary array, shape (n_cells, n_bins). 1 = above threshold.
    """
    if signal_type == 'spikes':
        peak = np.nanmax(binF, axis=1, keepdims=True)
        threshold = signal_threshold * peak

    else:
        quantile_values = np.nanquantile(binF, base_quantile, axis=1, keepdims=True)
        # Mean of bins at or below the quantile — lower and more accurate than using
        # the quantile value itself as the baseline.
        sub_threshold_mask = binF <= quantile_values
        masked = np.where(sub_threshold_mask, binF, np.nan)
        baseline = np.nanmean(masked, axis=1, keepdims=True)
        # Robust peak (99th percentile) so a single extreme bin doesn't lift the
        # threshold above legitimate secondary fields. Using nanmax here caused
        # cross-day inconsistency: a one-bin noise spike on day B would raise the
        # threshold and reject the same real field that passed on day A.
        peak = np.nanpercentile(binF, 99, axis=1, keepdims=True)
        threshold = baseline + signal_threshold * (peak - baseline)

    return (binF > threshold).astype(int)


# CORE DETECTION ALGORITHMS


def circular_connected_placefields(
    thres_im: np.ndarray,
    binF: np.ndarray,
    min_bins: int,
    bin_size_cm: float,
) -> PlaceFields1d:
    """Detect place fields with optional wrap-around merging.

    First detects fields on the unpadded array (no cross-copy merging).
    Then checks if fields at the first and last bins of the same cell
    should be merged (true wrap-around). Computes weighted centroids
    using circular mean for wrapped fields.

    Args:
        thres_im: Binary thresholded image, shape (n_cells, n_bins).
        binF: Binned fluorescence, shape (n_cells, n_bins).
        min_bins: Minimum contiguous bins for a valid field (3 == 15 cm)

    Returns:
        PlaceFields1d with detected fields.
    """
    num_bins = thres_im.shape[1]
    struct = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    # Step 1: detect on unpadded array
    label_im, n_labels = label(thres_im, struct)

    if n_labels == 0:
        return PlaceFields1d(
            np.zeros(thres_im.shape, dtype=np.uint32), binF,
            centers=np.empty((0, 2)),
            bin_size_cm = bin_size_cm,
        )

    # Step 2: merge wrap-around fields per cell
    # Only merge if the combined field is a plausible place field (< half the track)
    max_merge_bins = num_bins // 2
    for row in range(label_im.shape[0]):
        left_label = label_im[row, 0]
        right_label = label_im[row, num_bins - 1]
        if left_label > 0 and right_label > 0 and left_label != right_label:
            left_size = (label_im[row] == left_label).sum()
            right_size = (label_im[row] == right_label).sum()
            if left_size + right_size <= max_merge_bins:
                label_im[row, label_im[row] == left_label] = right_label

    # Renumber labels to be contiguous 1..N
    unique_labels = np.unique(label_im)
    unique_labels = unique_labels[unique_labels > 0]
    new_label_im = np.zeros_like(label_im)
    for new_id, old_id in enumerate(unique_labels, start=1):
        new_label_im[label_im == old_id] = new_id

    # Step 3: filter by min_bins and compute centers
    props = regionprops(new_label_im, binF, cache=False)
    result_label = np.zeros(thres_im.shape, dtype=np.uint32)
    centers_list = []
    counter = 1

    for prop in props:
        if prop['area'] < min_bins:
            continue

        coords = prop['coords']
        cell_idx = coords[0, 0]
        bin_indices = coords[:, 1]
        result_label[cell_idx, bin_indices] = counter

        # Weighted centroid — use circular mean for wrapped fields
        weights = binF[cell_idx, bin_indices]
        weight_sum = weights.sum()
        if weight_sum == 0:
            center_bin = bin_indices.mean()
        else:
            touches_start = 0 in bin_indices
            touches_end = (num_bins - 1) in bin_indices
            is_wrapped = touches_start and touches_end

            if is_wrapped:
                # Circular weighted mean: shift bins so the gap is at the far side
                shifted = (bin_indices + num_bins // 2) % num_bins
                center_shifted = np.average(shifted, weights=weights)
                center_bin = (center_shifted - num_bins // 2) % num_bins
            else:
                center_bin = np.average(bin_indices, weights=weights)

        centers_list.append([float(cell_idx), float(center_bin)])
        counter += 1

    centers_out = np.array(centers_list) if centers_list else np.empty((0, 2))
    return PlaceFields1d(result_label, binF, centers=centers_out, bin_size_cm=bin_size_cm)


def prominence_placefields(
    smoothed: np.ndarray,
    binF: np.ndarray,
    min_prominence_frac: float,
    base_quantile: float,
    min_bins: int,
    rel_height: float,
    bin_size_cm: float,
) -> PlaceFields1d:
    """Detect place fields via local peak prominence.

    Each cell's tuning curve is scanned with `scipy.signal.find_peaks`, where
    every peak is required to clear a prominence threshold computed PER-CELL
    from the cell's own dynamic range. Prominence is the topographic height of
    a peak above its surrounding valleys — it does NOT depend on taller peaks
    elsewhere in the curve. This gives field-independent detection: a tall peak
    cannot shadow a smaller but still locally-prominent peak on the same cell.

    Field extent for each detected peak is taken from `peak_widths` at
    `rel_height` of the prominence (default 0.5 == half-prominence width,
    standard in the place-cell literature).

    Args:
        smoothed: Smoothed binned activity, shape (n_cells, n_bins). Used for
            peak finding; field extent and statistics are computed from binF.
        binF: Raw binned activity, shape (n_cells, n_bins). Stored on the result.
        min_prominence_frac: Required peak prominence as a fraction of the cell's
            (peak - baseline) dynamic range. 0.20 mirrors signal_threshold.
        base_quantile: Quantile for baseline estimation (mean of bins at or below
            this quantile of the cell's curve).
        min_bins: Minimum field width in bins. Fields narrower than this are
            dropped.
        rel_height: Fractional height at which field width is measured, passed
            to `scipy.signal.peak_widths`. 0.5 = half-prominence (standard).
        bin_size_cm: Spatial bin size in cm. Passed to PlaceFields1d.

    Returns:
        PlaceFields1d with one field per accepted peak.
    """
    n_cells, num_bins = smoothed.shape
    result_label = np.zeros((n_cells, num_bins), dtype=np.uint32)
    centers_list = []
    counter = 1

    # Per-cell baseline (same definition as _quantile_threshold dff branch).
    quantile_values = np.nanquantile(smoothed, base_quantile, axis=1, keepdims=True)
    sub_threshold_mask = smoothed <= quantile_values
    masked_low = np.where(sub_threshold_mask, smoothed, np.nan)
    baseline = np.nanmean(masked_low, axis=1)  # shape (n_cells,)
    # Robust peak (matches threshold branch) so a single extreme bin doesn't
    # set an unreachable prominence requirement.
    peak = np.nanpercentile(smoothed, 99, axis=1)  # shape (n_cells,)

    for cell_idx in range(n_cells):
        curve = smoothed[cell_idx]
        dynamic_range = peak[cell_idx] - baseline[cell_idx]
        if not np.isfinite(dynamic_range) or dynamic_range <= 0:
            continue
        min_prominence = min_prominence_frac * dynamic_range

        tiled = np.concatenate([curve, curve, curve])
        peaks_tiled, _ = find_peaks(tiled, prominence=min_prominence)
        if len(peaks_tiled) == 0:
            continue

        in_middle = (peaks_tiled >= num_bins) & (peaks_tiled < 2 * num_bins)
        peaks_tiled = peaks_tiled[in_middle]
        if len(peaks_tiled) == 0:
            continue

        widths, _, left_ips, right_ips = peak_widths(
            tiled, peaks_tiled, rel_height=rel_height,
        )

        for peak_pos_tiled, left, right in zip(peaks_tiled, left_ips, right_ips):
            # Map tiled coords back to original [0, num_bins). The peak
            # position in the original curve is simply peak_pos_tiled - num_bins.
            peak_pos = peak_pos_tiled - num_bins
            left_bin = int(np.floor(left)) - num_bins
            right_bin = int(np.ceil(right)) - num_bins
            n_bins_field = right_bin - left_bin + 1
            if n_bins_field < min_bins:
                continue

            # Wrap into [0, num_bins). A field crossing the seam produces two
            # contiguous segments in the original frame; np.unique handles
            # duplicate bins if the field is wider than the track (rare).
            bin_indices = np.arange(left_bin, right_bin + 1) % num_bins
            bin_indices = np.unique(bin_indices)

            result_label[cell_idx, bin_indices] = counter
            weights = binF[cell_idx, bin_indices]
            weight_sum = weights.sum()
            if weight_sum == 0:
                center_bin = float(peak_pos % num_bins)
            else:
                # Circular weighted mean for fields that wrap the seam.
                touches_start = 0 in bin_indices
                touches_end = (num_bins - 1) in bin_indices
                is_wrapped = touches_start and touches_end and (n_bins_field < num_bins)
                if is_wrapped:
                    shifted = (bin_indices + num_bins // 2) % num_bins
                    center_shifted = np.average(shifted, weights=weights)
                    center_bin = float((center_shifted - num_bins // 2) % num_bins)
                else:
                    center_bin = float(np.average(bin_indices, weights=weights))
            centers_list.append([float(cell_idx), center_bin])
            counter += 1

    # Consolidate label numbering. When a later field's window overlaps an
    # earlier field's bins, the later assignment overwrites the earlier label.
    # If the earlier label is fully overwritten, the counter still advanced
    # past it AND centers_list got a row for it — but no bins in result_label
    # carry that label. Those phantom centers must be dropped, or downstream
    # regionprops-based properties (cell_id, mean_intensity) will desynchronize
    # from centers and silently mis-pair fields.
    unique_surviving = np.unique(result_label)
    unique_surviving = unique_surviving[unique_surviving > 0]
    new_result_label = np.zeros_like(result_label)
    new_centers_list: list[list[float]] = []
    for new_id, old_id in enumerate(unique_surviving, start=1):
        new_result_label[result_label == old_id] = new_id
        new_centers_list.append(centers_list[int(old_id) - 1])
    result_label = new_result_label
    centers_list = new_centers_list

    centers_out = np.array(centers_list) if centers_list else np.empty((0, 2))
    return PlaceFields1d(result_label, binF, centers=centers_out, bin_size_cm=bin_size_cm)


def outside_field_threshold(
    pf: PlaceFields1d,
    threshold_factor: float = 3.0,
    base_quantile: float = 0.25,
) -> PlaceFields1d:
    """Remove fields whose in-field activity isn't sufficiently above the cell's quiescent state.

    The quiescent reference for each cell is computed two ways and the more
    permissive (lower) of the two is used as the denominator:

      1. **Quiescent baseline** — mean of bins at or below the `base_quantile`
         quantile of the cell's tuning curve. This estimates the cell's true
         resting state and is independent of how many fields the cell has or
         how broad they are. It's the same baseline used by
         `_quantile_threshold` for the initial detection step.

      2. **Outside-field median** — median (NOT mean) of all bins not assigned
         to any detected field. The median is robust to "field shoulder" bins
         that fell just below the candidate threshold but are still elevated
         from the smoothing kernel; the original `mean` was inflated by those
         shoulders, which made multi-peak cells unfairly fail the test
         (the more peaks a cell had, the higher the outside mean climbed,
         even after excluding the detected field bins themselves).

    Using `min(baseline, outside_median)` means we trust whichever estimate is
    closer to the noise floor. In practice the two are very close for cells
    with one tight field; they diverge (and median wins) when smoothed peaks
    have wide flanks, or when a cell has multiple fields elevating much of the
    track.

    A field is removed if `mean(in-field) < threshold_factor × reference`.

    Args:
        pf: PlaceFields1d with detected fields.
        threshold_factor: Required ratio of in-field mean to the quiescent
            reference. With the new (median-or-baseline) reference, the
            denominator is more conservative than the old outside-mean
            denominator, so this factor can stay at the original 3.0 without
            being too strict in practice.
        base_quantile: Quantile cutoff for the quiescent baseline calculation.
            Bins at or below this quantile are averaged to estimate quiet state.
            0.25 matches the default used for thresholding in _quantile_threshold.

    Returns:
        Filtered PlaceFields1d.
    """
    if pf.n_fields == 0:
        return pf

    # Reference 1: quiescent baseline = mean of the bottom-`base_quantile` bins.
    # Robust to multiplicity of fields — adding more fields doesn't change the
    # bottom of the distribution.
    quantile_values = np.nanquantile(pf.binF, base_quantile, axis=1, keepdims=True)
    sub_threshold_mask = pf.binF <= quantile_values
    masked_low = np.where(sub_threshold_mask, pf.binF, np.nan)
    baseline = np.nanmean(masked_low, axis=1)  # shape (n_cells,)

    # Reference 2: median of bins outside all detected fields. Median is
    # robust to elevated flanks/shoulders that the original mean-based test
    # was sensitive to.
    outside_im = pf.binF.copy().astype(float)
    outside_im[pf.label_im != 0] = np.nan
    outside_median = np.nanmedian(outside_im, axis=1)  # shape (n_cells,)

    # Take the lower of the two references — both estimate quiet state, and we
    # trust the smaller (closer to true baseline). np.fmin ignores NaNs.
    reference = np.fmin(baseline, outside_median)

    cell_ids = pf.cell_id
    threshold_values = reference[cell_ids] * threshold_factor
    invalid = np.where(pf.mean_intensity < threshold_values)[0]

    return pf.remove_fields(invalid)


# TODO fix bin size/meta
def _detect_on_tuning_curves(
    binF: np.ndarray,
    params: DetectionParams,
    signal_type: str,
    bin_size_cm: float,
) -> PlaceFields1d:
    """Run detection pipeline on pre-computed tuning curves.

    Internal function used by both detect_place_fields and the shuffle test.

    Args:
        binF: Binned fluorescence, shape (n_cells, n_bins).
        params: Detection parameters.
        signal_type: Either 'spikes' or 'dff', inferred from signal_col.
        bin_size_cm: Spatial bin size in cm.

    Returns:
        PlaceFields1d with detected fields.
    """
    # Gaussian smoothing along the position axis. Wraps for circular tracks.
    if params.smooth_sigma > 0:
        smoothed = gaussian_filter1d(
            binF, sigma=params.smooth_sigma, axis=1, mode='wrap',
        )
    else:
        smoothed = binF

    if params.method == 'prominence':
        # Field-independent detection: each peak evaluated by local prominence,
        # not against a per-cell global threshold dominated by the tallest peak.
        pf = prominence_placefields(
            smoothed, binF,
            min_prominence_frac=params.min_prominence_frac,
            base_quantile=params.base_quantile,
            min_bins=params.min_bins,
            rel_height=params.field_rel_height,
            bin_size_cm=bin_size_cm,
        )
    else:
        # Threshold on smoothed data for robust detection, but store raw binF for plotting/filtering
        thres_im = _quantile_threshold(
            smoothed, signal_type, params.base_quantile, params.signal_threshold,
        )
        # Connected components (circular) — use raw binF so heatmaps aren't blurred
        pf = circular_connected_placefields(thres_im, binF, min_bins=params.min_bins, bin_size_cm=bin_size_cm)

    # Filter: outside-field ratio
    pf = outside_field_threshold(pf, params.outside_threshold)

    # Filter: minimum peak amplitude (only if explicitly set)
    if params.min_peak is not None and pf.n_fields > 0:
        weak = np.where(pf.max_intensity < params.min_peak)[0]
        pf = pf.remove_fields(weak)

    return pf


def detect_place_fields(
    df: pl.DataFrame,
    config: dict,
    signal_col: str = 'multi_day_dff',
    bin_size_cm: int | None = None,
    params: DetectionParams | None = None,
    metadata: dict | None = None,
) -> PlaceFieldResult:
    """Detect place fields from frame-level data.

    Computes session-averaged tuning curves via compute_session_averages(), then runs
    threshold + connected component detection per trial type.

    Args:
        df: Frame-level DataFrame from process_session() (after fix_cue_offset).
        config: Experiment configuration dict.
        signal_col: Column containing neural signals (list per frame).
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        params: Detection parameters. Uses defaults if None.
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if
            ``bin_size_cm`` is not supplied.

    Returns:
        PlaceFieldResult with per-trial-type detection results.
    """
    from df_processing import get_bin_size

    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)

    if params is None:
        params = DetectionParams()

    # Infer signal type from column name
    signal_type = 'spikes' if 'spikes' in signal_col else 'dff'

    # Speed filter — exclude stationary frames that inflate occupancy at rest positions
    if params.min_speed_cm_s is not None and 'speed_cm_s' in df.columns:
        n_before = len(df)
        df = df.filter(pl.col('speed_cm_s') >= params.min_speed_cm_s)
        print(f"Speed filter: {n_before - len(df)}/{n_before} frames removed "
              f"(< {params.min_speed_cm_s} cm/s)")

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

        pf = _detect_on_tuning_curves(binF, params, signal_type, bin_size_cm=bin_size_cm)

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
    bin_size_cm: int | None = None,
    params: DetectionParams | None = None,
    seed: int = 42,
    metadata: dict | None = None,
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
        bin_size_cm: Spatial bin size in cm. If None, resolved from ``metadata``.
        params: Detection parameters. Uses result.params or defaults if None.
        seed: Random seed for reproducibility.
        metadata: Session metadata dict carrying ``bin_size_cm``. Required if
            ``bin_size_cm`` is not supplied.

    Returns:
        Updated PlaceFieldResult with p_values and sig_cells populated.
    """
    from df_processing import get_bin_size

    if bin_size_cm is None:
        bin_size_cm = get_bin_size(metadata, df)

    if params is None:
        params = result.params if result is not None else DetectionParams()

    # Infer signal type from column name
    signal_type = 'spikes' if 'spikes' in signal_col else 'dff'

    if params.min_speed_cm_s is not None and 'speed_cm_s' in df.columns:
        n_before = len(df)
        df = df.filter(pl.col('speed_cm_s') >= params.min_speed_cm_s)
        print(f"Speed filter: {n_before - len(df)}/{n_before} frames removed "
              f"(< {params.min_speed_cm_s} cm/s)")

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
            shuf_pf = _detect_on_tuning_curves(shuffled_binF, params, signal_type, bin_size_cm=bin_size_cm)
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


if __name__ == "__main__":
    from df_processing import find_session_dir, get_session_paths, load_session_context, load_processed_session

    mouse_id = "26"
    mouse_dir = Path("/Users/cs963/Desktop/sun_lab_projects/datasets", mouse_id)
    date = "2025-09-08"

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths["parquet"])

    params = DetectionParams(smooth_sigma=0, signal_threshold=0.3)
    result = detect_place_fields(
        data, exp_config, signal_col="multi_day_dff",
        bin_size_cm=meta["bin_size_cm"], params=params,
    )
    print(result.summary())
