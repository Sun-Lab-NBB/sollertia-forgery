"""Provides functionality for detecting and analyzing spatial tuning and place fields in neural recordings on a linear
track.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import field, dataclass, replace
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from numba import njit, prange
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter1d,
    minimum_filter1d,
    uniform_filter1d,
)
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_NO_TRIAL_SENTINEL: int = 255
"""Sentinel trial id used by the acquisition pipeline to mark frames outside of any trial."""
_WORKER_RESERVE: int = 4
"""Number of CPU cores reserved for the OS when worker_count=-1 selects an automatic worker count."""
_PLOT_TICK_INTERVAL_CM: float = 25.0
"""Spacing in centimeters between x-axis ticks on the position-ordered heatmap."""


class BaselineMethod(StrEnum):
    """Defines the baseline (F0) calculation method for dF/F0 normalization."""

    MAXIMIN = "maximin"
    """Applies Gaussian smoothing along the time axis followed by minimum then maximum filtering to extract a slow
    drifting baseline."""
    AVERAGE = "average"
    """Uses the per-cell mean across all timepoints as a constant baseline."""


@dataclass(frozen=True)
class PlaceFieldDetectionConfiguration:
    """Defines configuration parameters for Tank lab place field detection algorithm."""

    minimum_speed: float = 5.0
    """Minimum speed threshold in cm/s for including timepoints in analysis."""
    smooth_size: int = 3
    """Size of the smoothing kernel in bins for the moving average filter."""
    base_quantile: float = 0.25
    """Quantile (0-1) of the binned fluorescence distribution. Values at or below this quantile are averaged to compute
    the baseline activity level used in place field thresholding."""
    signal_threshold: float = 0.25
    """Signal threshold factor applied to the difference between max and baseline."""
    minimum_bins: int = 3
    """Minimum number of contiguous bins required for a valid place field."""
    outside_threshold: float = 3.0
    """Factor by which in-field activity must exceed out-of-field activity."""
    maximum_intensity_threshold: float = 0.1
    """Minimum peak intensity required for a valid place field."""
    chunk_count: int = 100
    """Number of temporal chunks used for shuffle-based validation."""
    significance_threshold: float = 0.05
    """P-value threshold for determining statistically significant place fields."""


@dataclass(frozen=True)
class PlaceFields:
    """Stores detected place fields in one-dimensional space."""

    label_image: NDArray[np.int32]
    """Labeled image of detected place fields with dimensions (cell_count, bin_count)."""
    binned_fluorescence: NDArray[np.float32]
    """Binned fluorescence data with dimensions (cell_count, bin_count)."""
    binned_fluorescence_per_trial: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=np.float32),
    )
    """Per-lap binned fluorescence with dimensions (cell_count, trial_count, bin_count). Empty when per-trial
    binning was not computed."""
    centers: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    """Centers of detected place fields with dimensions (field_count, 2)."""
    bin_size: float = 5.0
    """Size of spatial bins in centimeters."""

    @property
    def mean_intensity(self) -> NDArray[np.float32]:
        """Returns the mean intensity for each detected place field."""
        return _compute_mean_intensity(label_image=self.label_image, intensity_image=self.binned_fluorescence)

    @property
    def max_intensity(self) -> NDArray[np.float32]:
        """Returns the maximum intensity for each detected place field."""
        return _compute_max_intensity(label_image=self.label_image, intensity_image=self.binned_fluorescence)

    @property
    def cell_id(self) -> NDArray[np.int32]:
        """Returns the cell ID for each detected place field."""
        # noinspection PyTypeChecker
        return self.centers[:, 0].astype(np.int32)

    @property
    def has_place_field(self) -> NDArray[np.bool_]:
        """Returns a boolean array indicating whether each cell has a detected place field."""
        return np.any(self.label_image > 0, axis=1)

    @property
    def order(self) -> NDArray[np.int32]:
        """Returns cell ordering indices based on place field centers.

        For cells with multiple place fields, the field with the highest mean intensity is used for ordering.
        """
        cell_count = self.binned_fluorescence.shape[0]

        # With no detected fields, falls back to the natural cell order so callers can rely on a length-cell_count
        # permutation regardless of detection results.
        if self.centers.size == 0:
            # noinspection PyTypeChecker
            return np.arange(cell_count, dtype=np.int32)

        # Initializes sort order with infinity to ensure cells without place fields are sorted to the end.
        sort_order = np.full(cell_count, np.inf, dtype=np.float32)
        intensity = self.mean_intensity
        field_centers = self.centers
        # noinspection PyTypeChecker
        field_cell_id: NDArray[np.int32] = field_centers[:, 0].astype(np.int32)

        # Orders cells with multiple place fields based on the field with the highest mean intensity.
        for cell_index in range(cell_count):
            cell_field_mask = field_cell_id == cell_index
            cell_field_indices = np.flatnonzero(cell_field_mask)
            if cell_field_indices.size > 0:
                max_intensity_index = np.argmax(intensity[cell_field_mask])
                # Assigns the position coordinate of the highest intensity field as the sort key.
                sort_order[cell_index] = field_centers[cell_field_indices[max_intensity_index], 1]

        # Returns indices that would sort cells by their place field position along the track.
        # noinspection PyTypeChecker
        return np.argsort(sort_order).astype(np.int32)

    def remove_fields(self, indices: NDArray[np.int32]) -> PlaceFields:
        """Removes specified place fields and returns a new PlaceFields object.

        Args:
            indices: Indices of place fields to remove.

        Returns:
            A new PlaceFields object with the specified fields removed.
        """
        # Zeros out the labels for the fields to be removed on a fresh copy so the source instance remains immutable.
        new_label_image = self.label_image.copy()
        new_label_image[np.isin(new_label_image, indices + 1)] = 0
        _renumber_labels(new_label_image)

        return replace(
            self,
            label_image=new_label_image,
            centers=_compute_centers_from_labels(
                label_image=new_label_image,
                binned_fluorescence=self.binned_fluorescence,
                bin_size=self.bin_size,
            ),
        )

    def filter_cells(self, indices: NDArray[np.int32]) -> PlaceFields:
        """Filters to keep only specified cells and returns a new PlaceFields object.

        Args:
            indices: Indices of cells to keep.

        Returns:
            A new PlaceFields object containing only the specified cells.
        """
        # Zeros out labels for cells that are not in the list of cells to keep on a fresh copy so the source instance
        # remains immutable.
        new_label_image = self.label_image.copy()
        all_indices = np.arange(new_label_image.shape[0])
        new_label_image[~np.isin(all_indices, indices), :] = 0
        _renumber_labels(new_label_image)

        return replace(
            self,
            label_image=new_label_image,
            centers=_compute_centers_from_labels(
                label_image=new_label_image,
                binned_fluorescence=self.binned_fluorescence,
                bin_size=self.bin_size,
            ),
        )


@njit(cache=True)
def _compute_label_centers(
    label_image: NDArray[np.int32],
    intensity_image: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Computes cell index and intensity-weighted centroid for each labeled place field region.

    Args:
        label_image: Labeled image with dimensions (cell_count, bin_count).
        intensity_image: Intensity image with same dimensions.

    Returns:
        A tuple of (cell_indices, weighted_centroids) arrays.
    """
    region_count = int(np.max(label_image))
    if region_count == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    cell_indices = np.zeros(region_count, dtype=np.int32)
    intensity_sums = np.zeros(region_count, dtype=np.float32)
    weighted_sums = np.zeros(region_count, dtype=np.float32)

    # Accumulates intensity and position-weighted intensity for each labeled region.
    for cell_index in range(label_image.shape[0]):
        for bin_index in range(label_image.shape[1]):
            label_value = label_image[cell_index, bin_index]
            if label_value > 0:
                region_index = label_value - 1
                intensity = intensity_image[cell_index, bin_index]
                cell_indices[region_index] = cell_index
                intensity_sums[region_index] += intensity
                weighted_sums[region_index] += bin_index * intensity

    # Computes intensity-weighted centroid by dividing position-weighted sum by total intensity.
    weighted_centroids = np.zeros(region_count, dtype=np.float32)
    for region_index in range(region_count):
        if intensity_sums[region_index] > 0:
            weighted_centroids[region_index] = weighted_sums[region_index] / intensity_sums[region_index]

    return cell_indices, weighted_centroids


@njit(cache=True)
def _compute_mean_intensity(
    label_image: NDArray[np.int32],
    intensity_image: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Computes the mean fluorescence intensity for each labeled place field region.

    Args:
        label_image: Labeled image with dimensions (cell_count, bin_count).
        intensity_image: Intensity image with same dimensions.

    Returns:
        Array of mean intensities with length equal to the number of labeled regions.
    """
    region_count = int(np.max(label_image))
    if region_count == 0:
        return np.zeros(0, dtype=np.float32)

    intensity_sums = np.zeros(region_count, dtype=np.float32)
    pixel_counts = np.zeros(region_count, dtype=np.int32)

    # Accumulates intensity and pixel count for each labeled region.
    for cell_index in range(label_image.shape[0]):
        for bin_index in range(label_image.shape[1]):
            label_value = label_image[cell_index, bin_index]
            if label_value > 0:
                region_index = label_value - 1
                intensity_sums[region_index] += intensity_image[cell_index, bin_index]
                pixel_counts[region_index] += 1

    # Computes mean intensity by dividing total intensity by pixel count.
    mean_intensities = np.zeros(region_count, dtype=np.float32)
    for region_index in range(region_count):
        if pixel_counts[region_index] > 0:
            mean_intensities[region_index] = intensity_sums[region_index] / pixel_counts[region_index]

    return mean_intensities


@njit(cache=True)
def _compute_max_intensity(
    label_image: NDArray[np.int32],
    intensity_image: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Computes the maximum fluorescence intensity for each labeled place field region.

    Args:
        label_image: Labeled image with dimensions (cell_count, bin_count).
        intensity_image: Intensity image with same dimensions.

    Returns:
        Array of maximum intensities with length equal to the number of labeled regions.
    """
    region_count = int(np.max(label_image))
    if region_count == 0:
        return np.zeros(0, dtype=np.float32)

    max_intensities = np.zeros(region_count, dtype=np.float32)

    # Tracks the maximum intensity value encountered for each labeled region.
    for cell_index in range(label_image.shape[0]):
        for bin_index in range(label_image.shape[1]):
            label_value = label_image[cell_index, bin_index]
            if label_value > 0:
                region_index = label_value - 1
                intensity = intensity_image[cell_index, bin_index]
                max_intensities[region_index] = max(max_intensities[region_index], intensity)

    return max_intensities


@njit(cache=True, parallel=True)
def _accumulate_binned_fluorescence(
    fluorescence: NDArray[np.float32],
    bin_indices: NDArray[np.int32],
    bin_count: int,
    sample_counts: NDArray[np.int32],
    use_mean: bool,
    output: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Accumulates neural fluorescence values into spatial position bins for each cell.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        bin_indices: Bin index for each frame with length frame_count.
        bin_count: Total number of bins.
        sample_counts: Number of samples per bin with length bin_count.
        use_mean: If True, computes the mean fluorescence per bin by dividing the accumulated fluorescence sum by
            sample count; otherwise returns the raw sum of fluorescence values per bin.
        output: Pre-allocated output array with dimensions (cell_count, bin_count).

    Returns:
        The output array filled with binned fluorescence values.
    """
    cell_count = fluorescence.shape[0]
    frame_count = fluorescence.shape[1]

    for cell_index in prange(cell_count):
        # Initializes a temporary array to accumulate fluorescence values for each spatial bin.
        bin_sums = np.zeros(bin_count, dtype=np.float32)

        # Adds each fluorescence value to its corresponding spatial bin based on the animal's position at that frame.
        for frame_index in range(frame_count):
            bin_index = bin_indices[frame_index]
            bin_sums[bin_index] += fluorescence[cell_index, frame_index]

        # Converts accumulated sums to mean values (if requested) by dividing by the number of samples in each bin.
        for bin_index in range(bin_count):
            if sample_counts[bin_index] > 0:
                if use_mean:
                    output[cell_index, bin_index] = bin_sums[bin_index] / sample_counts[bin_index]
                else:
                    output[cell_index, bin_index] = bin_sums[bin_index]
            else:
                output[cell_index, bin_index] = np.nan

    return output


@njit(cache=True, parallel=True)
def _apply_place_field_threshold(
    fluorescence: NDArray[np.float32],
    quantile_values: NDArray[np.float32],
    max_values: NDArray[np.float32],
    threshold_factor: float,
    output: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Applies activity threshold to binned fluorescence data for place field candidate detection.

    Args:
        fluorescence: Binned dF/F0 fluorescence data with dimensions (cell_count, bin_count).
        quantile_values: The fluorescence value at the specified quantile for each cell.
        max_values: The maximum fluorescence value for each cell.
        threshold_factor: Fraction of (max - baseline) to add to baseline for threshold.
        output: Pre-allocated output boolean array with dimensions (cell_count, bin_count).

    Returns:
        The output array with True where fluorescence exceeds the computed threshold.
    """
    cell_count = fluorescence.shape[0]
    bin_count = fluorescence.shape[1]

    for cell_index in prange(cell_count):
        quantile_threshold = quantile_values[cell_index]

        # Computes the baseline fluorescence for the current cell as the mean of bins at or below the quantile
        # threshold.
        total = 0.0
        count = 0
        for bin_index in range(bin_count):
            value = fluorescence[cell_index, bin_index]
            if not np.isnan(value) and value <= quantile_threshold:
                total += value
                count += 1

        baseline = total / count if count > 0 else 0.0
        cell_threshold = baseline + (max_values[cell_index] - baseline) * threshold_factor

        # Marks spatial bins with fluorescence exceeding the cell-specific activity threshold as candidate place fields.
        for bin_index in range(bin_count):
            value = fluorescence[cell_index, bin_index]
            output[cell_index, bin_index] = not np.isnan(value) and value > cell_threshold

    return output


@njit(cache=True, parallel=True)
def _label_place_field_regions(
    thresholded: NDArray[np.bool_],
    intensity_image: NDArray[np.float32],
    labels: NDArray[np.int32],
    cell_indices: NDArray[np.int32],
    areas: NDArray[np.int32],
    centroids: NDArray[np.float32],
    region_counts: NDArray[np.int32],
) -> None:
    """Labels contiguous above-threshold bins as place field regions and extracts their area and centroid.

    Args:
        thresholded: Binary (bool) image with dimensions (cell_count, bin_count).
        intensity_image: Intensity image for weighted centroid calculation.
        labels: Pre-allocated output array for component labels with dimensions (cell_count, bin_count).
        cell_indices: Pre-allocated output for cell indices with dimensions (cell_count, max_regions_per_row).
        areas: Pre-allocated output for region areas with dimensions (cell_count, max_regions_per_row).
        centroids: Pre-allocated output for centroids with dimensions (cell_count, max_regions_per_row).
        region_counts: Pre-allocated output for region count per cell with length cell_count.
    """
    cell_count = thresholded.shape[0]
    bin_count = thresholded.shape[1]

    for cell_index in prange(cell_count):
        current_label = 0
        local_count = 0
        bin_index = 0

        # Scans each bin to detect contiguous place field regions.
        while bin_index < bin_count:
            if thresholded[cell_index, bin_index]:
                current_label += 1
                start_bin = bin_index
                weighted_sum = 0.0
                intensity_sum = 0.0

                # Accumulates bin count and intensity-weighted position for each place field region.
                while bin_index < bin_count and thresholded[cell_index, bin_index]:
                    labels[cell_index, bin_index] = current_label
                    weighted_sum += bin_index * intensity_image[cell_index, bin_index]
                    intensity_sum += intensity_image[cell_index, bin_index]
                    bin_index += 1

                # Stores region properties once the contiguous region ends.
                cell_indices[cell_index, local_count] = cell_index
                areas[cell_index, local_count] = bin_index - start_bin
                centroids[cell_index, local_count] = weighted_sum / intensity_sum if intensity_sum > 0 else 0.0
                local_count += 1

            else:
                bin_index += 1

        region_counts[cell_index] = local_count


@njit(cache=True)
def _flatten_region_properties(
    cell_indices: NDArray[np.int32],
    areas: NDArray[np.int32],
    centroids: NDArray[np.float32],
    region_counts: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Flattens per-cell place field region property arrays into contiguous 1D arrays.

    Args:
        cell_indices: Cell indices with dimensions (cell_count, max_regions_per_row).
        areas: Region areas with dimensions (cell_count, max_regions_per_row).
        centroids: Region centroids with dimensions (cell_count, max_regions_per_row).
        region_counts: Number of regions per cell with length cell_count.

    Returns:
        A tuple containing flattened cell indices, areas, and centroids arrays.
    """
    # Pre-allocates 1D output arrays with total region count across all cells.
    total_regions = np.sum(region_counts)
    flat_cell_indices = np.zeros(total_regions, dtype=np.int32)
    flat_areas = np.zeros(total_regions, dtype=np.int32)
    flat_centroids = np.zeros(total_regions, dtype=np.float32)

    # Copies valid regions from each cell row into contiguous 1D arrays.
    index = 0
    for cell_index in range(len(region_counts)):
        for region_index in range(region_counts[cell_index]):
            flat_cell_indices[index] = cell_indices[cell_index, region_index]
            flat_areas[index] = areas[cell_index, region_index]
            flat_centroids[index] = centroids[cell_index, region_index]
            index += 1

    return flat_cell_indices, flat_areas, flat_centroids


def _renumber_labels(label_image: NDArray[np.int32]) -> None:
    """Renumbers label_image values sequentially starting from 1 in place after removing or filtering fields."""
    new_label = 1
    for value in np.unique(label_image):
        if value != 0:
            label_image[label_image == value] = new_label
            new_label += 1


def _compute_centers_from_labels(
    label_image: NDArray[np.int32],
    binned_fluorescence: NDArray[np.float32],
    bin_size: float,
) -> NDArray[np.float32]:
    """Computes the (cell_index, position_cm) centers array from a label image and its intensity image.

    Args:
        label_image: Labeled image with dimensions (cell_count, bin_count).
        binned_fluorescence: Intensity image with the same dimensions used to weight centroid positions.
        bin_size: Size of spatial bins in centimeters used to convert bin indices to centimeters.

    Returns:
        Centers array with dimensions (field_count, 2) where each row is (cell_index, position_cm). Returns an empty
        (0, 2) array when no labeled regions are present.
    """
    cell_indices, weighted_centroids = _compute_label_centers(
        label_image=label_image, intensity_image=binned_fluorescence
    )

    if len(cell_indices) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Converts bin indices to centimeters and combines with cell indices into a (field_count, 2) array.
    # noinspection PyTypeChecker
    return np.column_stack((cell_indices.astype(np.float32), weighted_centroids * bin_size))


def _compute_baseline_fluorescence(
    fluorescence: NDArray[np.float32],
    method: BaselineMethod = BaselineMethod.MAXIMIN,
    gaussian_sigma: float = 20.0,
    filter_window_size: int = 600,
) -> NDArray[np.float32]:
    """Computes the baseline (F0) of neural fluorescence traces for dF/F0 normalization.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        method: The baseline calculation method. See BaselineMethod for valid values.
        gaussian_sigma: Standard deviation of the Gaussian filter for the MAXIMIN method.
        filter_window_size: Window size for min/max filtering in the MAXIMIN method.

    Returns:
        Baseline array with the same shape as fluorescence.

    Raises:
        ValueError: If an unknown method is provided or fluorescence has invalid dimensions.
    """
    if method == BaselineMethod.MAXIMIN:
        # Applies Gaussian smoothing along the time axis only (sigma=0 for cell axis).
        if fluorescence.ndim == 2:
            baseline = gaussian_filter(input=fluorescence, sigma=[0.0, gaussian_sigma])
        elif fluorescence.ndim == 1:
            baseline = gaussian_filter(input=fluorescence, sigma=[gaussian_sigma])
        else:
            message = (
                f"Unable to compute baseline fluorescence. Expected fluorescence to be 1D or 2D, "
                f"but got {fluorescence.ndim}D."
            )
            console.error(message=message, error=ValueError)

        # Applies minimum then maximum filtering to extract the slow-varying baseline fluorescence (F0) used for
        # computing dF/F0.
        baseline = minimum_filter1d(input=baseline, size=filter_window_size)
        baseline = maximum_filter1d(input=baseline, size=filter_window_size)

    elif method == BaselineMethod.AVERAGE:
        # Computes a constant baseline fluorescence (F0) per cell using the mean across all timepoints.
        baseline = np.broadcast_to(np.mean(fluorescence, axis=1)[:, np.newaxis], fluorescence.shape).copy()

    else:
        message = (
            f"Unable to compute baseline fluorescence. The method must be one of "
            f"{[entry.value for entry in BaselineMethod]}, but got '{method}'."
        )
        console.error(message=message, error=ValueError)

    # noinspection PyTypeChecker
    return baseline


def _compute_delta_fluorescence(
    fluorescence: NDArray[np.float32],
    baseline_method: BaselineMethod = BaselineMethod.MAXIMIN,
    subtract_minimum: bool = False,
    gaussian_sigma: float = 20.0,
    filter_window_size: int = 600,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the relative fluorescence change (ΔF/F0) normalized to baseline fluorescence.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        baseline_method: Baseline calculation method. See _compute_baseline_fluorescence() for valid methods.
        subtract_minimum: Whether to subtract the minimum fluorescence from each row before baseline calculation.
        gaussian_sigma: Standard deviation of the Gaussian filter for 'maximin' baseline method.
        filter_window_size: Window size for min/max filtering in 'maximin' baseline method.

    Returns:
        A tuple containing the delta fluorescence array (F - F0) / F0 and the baseline F0 array,
        both with the same shape as the input fluorescence.
    """
    # Removes any negative offset by shifting each cell's trace so its minimum is zero.
    if subtract_minimum:
        fluorescence = fluorescence - np.min(fluorescence, axis=1)[..., np.newaxis]

    # Estimates the baseline fluorescence (F0) representing the resting or non-active state of each cell.
    baseline = _compute_baseline_fluorescence(
        fluorescence=fluorescence,
        method=baseline_method,
        gaussian_sigma=gaussian_sigma,
        filter_window_size=filter_window_size,
    )

    # Computes the relative fluorescence change.
    delta_fluorescence = (fluorescence - baseline) / baseline

    return delta_fluorescence, baseline


def _bin_fluorescence_by_position(
    fluorescence: NDArray[np.float32],
    position: NDArray[np.float32],
    bin_edges: NDArray[np.float32],
    compute_mean: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Bins neural fluorescence data by animal position along the linear track.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        position: Position values used for binning with length matching frame_count.
        bin_edges: Bin edges for spatial binning.
        compute_mean: Determines whether to compute mean or sum for each bin.

    Returns:
        A tuple containing the binned fluorescence array with dimensions (cell_count, bin_count)
        and the sample count per bin with length bin_count.
    """
    # Assigns each position to a spatial bin and clips to the range [0, bin_count - 1].
    bin_indices = np.searchsorted(bin_edges, position, side="right") - 1
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int32] = np.clip(bin_indices, 0, len(bin_edges) - 2).astype(np.int32)

    bin_count = len(bin_edges) - 1
    cell_count = fluorescence.shape[0]
    # noinspection PyTypeChecker
    sample_counts: NDArray[np.int32] = np.bincount(bin_indices, minlength=bin_count).astype(np.int32)

    output = np.full((cell_count, bin_count), np.nan, dtype=np.float32)

    # Accumulates fluorescence values into spatial bins for each cell.
    _accumulate_binned_fluorescence(
        fluorescence=fluorescence,
        bin_indices=bin_indices,
        bin_count=bin_count,
        sample_counts=sample_counts,
        use_mean=compute_mean,
        output=output,
    )

    return output, sample_counts


def _compute_quantile_max_threshold(
    fluorescence: NDArray[np.float32],
    base_quantile: float = 0.25,
    threshold_factor: float = 0.25,
) -> NDArray[np.bool_]:
    """Thresholds binned fluorescence using a fractional difference between baseline quantile and peak activity.

    Args:
        fluorescence: Binned fluorescence data with dimensions (cell_count, bin_count).
        base_quantile: Quantile of the per-cell fluorescence distribution used as the baseline reference.
        threshold_factor: Fraction of (max - baseline) added to the baseline to form the per-cell threshold.

    Returns:
        Boolean mask of the same shape as fluorescence. Returns True if values exceed the computed threshold.
    """
    # noinspection PyTypeChecker
    max_values: NDArray[np.float32] = np.nanmax(fluorescence, axis=1).astype(np.float32)
    # noinspection PyTypeChecker
    quantile_values: NDArray[np.float32] = np.nanquantile(fluorescence, base_quantile, axis=1).astype(np.float32)
    output = np.empty(fluorescence.shape, dtype=np.bool_)

    return _apply_place_field_threshold(
        fluorescence=fluorescence,
        quantile_values=quantile_values,
        max_values=max_values,
        threshold_factor=threshold_factor,
        output=output,
    )


def _compute_circular_connected_place_fields(
    thresholded_image: NDArray[np.bool_],
    binned_fluorescence: NDArray[np.float32],
    minimum_bins: int = 3,
    bin_size: float = 5.0,
) -> PlaceFields:
    """Detects place fields as circularly connected above-threshold regions along the linear track.

    Args:
        thresholded_image: Thresholded binary image of binned place field activity with dimensions
            (cell_count, bin_count).
        binned_fluorescence: Binned fluorescence data with dimensions (cell_count, bin_count).
        minimum_bins: Minimal required size of a connected region in bins.
        bin_size: Size of spatial bins in centimeters propagated into the returned PlaceFields.

    Returns:
        The detected one-dimensional place fields.
    """
    cell_count, bin_count = thresholded_image.shape
    padded_bin_count = bin_count * 3
    max_regions_per_row = padded_bin_count // 2 + 1

    # Pads the arrays by wrapping bins from both track edges to treat activity spanning track boundaries as a single
    # place field.
    padded_threshold = np.pad(thresholded_image, ((0, 0), (bin_count, bin_count)), mode="wrap")
    padded_fluorescence = np.pad(binned_fluorescence, ((0, 0), (bin_count, bin_count)), mode="wrap")

    # Pre-allocates output arrays for labeling and property computation.
    padded_labels = np.zeros(padded_threshold.shape, dtype=np.int32)
    cell_indices = np.zeros((cell_count, max_regions_per_row), dtype=np.int32)
    areas = np.zeros((cell_count, max_regions_per_row), dtype=np.int32)
    centroids = np.zeros((cell_count, max_regions_per_row), dtype=np.float32)
    region_counts = np.zeros(cell_count, dtype=np.int32)

    # Labels connected components and extracts properties in a single pass.
    _label_place_field_regions(
        thresholded=padded_threshold,
        intensity_image=padded_fluorescence,
        labels=padded_labels,
        cell_indices=cell_indices,
        areas=areas,
        centroids=centroids,
        region_counts=region_counts,
    )

    # Flattens per-cell 2D arrays into 1D arrays for centroid and area filtering.
    cell_indices, areas, centroids = _flatten_region_properties(
        cell_indices=cell_indices,
        areas=areas,
        centroids=centroids,
        region_counts=region_counts,
    )

    # Filters to components whose centroids fall within the original track region and that span at least the minimum
    # number of contiguous bins.
    valid_mask = (centroids >= bin_count) & (centroids < bin_count * 2) & (areas >= minimum_bins)
    valid_indices = np.where(valid_mask)[0]

    result_label_image = np.zeros(thresholded_image.shape, dtype=np.int32)
    adjusted_centers = []

    # Maps valid components back to original coordinate space.
    for counter, region_index in enumerate(valid_indices):
        cell_index = cell_indices[region_index]
        target_centroid = centroids[region_index]

        # Retrieves the label value at the centroid position.
        centroid_bin = int(target_centroid)
        target_label = padded_labels[cell_index, centroid_bin]

        # Copies the labeled region to the result image with wrapped coordinates.
        for bin_index in range(padded_bin_count):
            if padded_labels[cell_index, bin_index] == target_label:
                wrapped_bin = bin_index % bin_count
                result_label_image[cell_index, wrapped_bin] = counter + 1

        adjusted_centers.append([float(cell_index), target_centroid - bin_count])

    # Converts adjusted centers list to a 2D array, or creates an empty array if no valid regions were found.
    # noinspection PyTypeChecker
    centers: NDArray[np.float32] = (
        np.array(adjusted_centers, dtype=np.float32) if adjusted_centers else np.zeros((0, 2), dtype=np.float32)
    )

    return PlaceFields(
        label_image=result_label_image,
        binned_fluorescence=binned_fluorescence,
        centers=centers,
        bin_size=bin_size,
    )


def _outside_field_threshold(place_fields: PlaceFields, threshold_factor: float = 3.0) -> PlaceFields:
    """Filters place fields by requiring in-field activity to exceed out-of-field baseline by a threshold factor.

    Removes false positives by requiring that detected place fields have significantly higher activity than the
    baseline outside the field. In cases where a cell has multiple fields, both fields are excluded from the
    outside field calculation.

    Args:
        place_fields: PlaceFields object with previously detected place fields.
        threshold_factor: Scalar factor of the outside field signal to set the threshold.

    Returns:
        The filtered PlaceFields object.
    """
    # Creates a mask to exclude place field pixels and compute mean fluorescence outside the fields.
    outside_image = place_fields.binned_fluorescence.copy()
    outside_image[place_fields.label_image != 0] = np.nan
    outside_values = np.nanmean(outside_image, axis=1)

    # Identifies fields where in-field activity does not exceed the outside-field baseline by the threshold factor.
    threshold_values = outside_values[place_fields.cell_id] * threshold_factor
    # noinspection PyTypeChecker
    invalid_regions: NDArray[np.int32] = np.flatnonzero(place_fields.mean_intensity < threshold_values).astype(
        np.int32
    )

    return place_fields.remove_fields(indices=invalid_regions)


class PlaceFieldDetector:
    """Detects, validates, and visualizes 1D place fields using thresholding and connected component analysis."""

    def __init__(
        self,
        session_path: Path,
        track_length: float,
        fluorescence_column: str = "single_day_dff",
        trial_type: str | None = None,
        bin_size: float = 5.0,
        configuration: PlaceFieldDetectionConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, position, speed, and trial data from a memory-mapped feather file for place field
        detection. Filters to 'run' state frames and converts cumulative distance to track position via modulo.

        Args:
            session_path: Path to the session feather file.
            track_length: Length of the track in centimeters.
            fluorescence_column: Name of the fluorescence column to use.
            trial_type: Trial type to filter by (e.g. "ABC", "ABCD"). If None, includes all trial types.
            bin_size: Size of spatial bins in centimeters.
            configuration: Configuration parameters for place field detection. Uses defaults if None.
        """
        df = pl.read_ipc(
            session_path,
            columns=["system_state", "trial_type", fluorescence_column, "distance_cm", "speed_cm_s", "trial"],
        )
        df = df.filter(pl.col("system_state") == "run")
        if trial_type is not None:
            df = df.filter(pl.col("trial_type") == trial_type)

        # Extracts fluorescence data and transposes from (frame, cell) to (cell, frame).
        # noinspection PyTypeChecker
        self.fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column].to_list(), dtype=np.float32).T

        # Converts the cumulative distance to track position using modulus to wrap within a single lap.
        # noinspection PyTypeChecker
        self.position: NDArray[np.float32] = df["distance_cm"].to_numpy().astype(np.float32) % track_length
        # noinspection PyTypeChecker
        self.speed: NDArray[np.float32] = df["speed_cm_s"].to_numpy().astype(np.float32)
        # noinspection PyTypeChecker
        self.trial_ids: NDArray[np.int32] = df["trial"].to_numpy().astype(np.int32)
        self.track_length = track_length
        self.bin_size = bin_size
        self.configuration = configuration if configuration is not None else PlaceFieldDetectionConfiguration()

        # Caches the bin edges shared by `_run_detection` and `_bin_fluorescence_per_trial`.
        self._bin_edges = np.arange(0, track_length + bin_size, bin_size, dtype=np.float32)

    def detect(self, run_shuffle: bool = False) -> PlaceFields:
        """Detects place fields from the original fluorescence and position data.

        Args:
            run_shuffle: Determines whether to run shuffle significance testing and filter results to only include
                cells with statistically significant place fields.

        Returns:
            A PlaceFields instance containing the labeled regions, pooled and per-lap binned fluorescence, and centers
            of detected place fields. If run_shuffle is True, only significant cells are included.
        """
        # Computes dF/F0 to normalize fluorescence relative to baseline.
        fluorescence, _ = _compute_delta_fluorescence(fluorescence=self.fluorescence)

        # Bins fluorescence by spatial position, applies thresholding, and detects connected regions as place fields.
        place_fields = self._run_detection(
            fluorescence=fluorescence,
            position=self.position,
            speed=self.speed,
        )

        # Filters to only include cells with statistically significant place fields based on shuffle testing.
        if run_shuffle:
            significant_cells, _ = self.compute_shuffle_significance(repeat_count=self.configuration.chunk_count)
            place_fields = place_fields.filter_cells(indices=significant_cells)

        # Computes per-lap binned fluorescence using the same speed filter and smoothing as the pooled computation.
        return replace(
            place_fields,
            binned_fluorescence_per_trial=self._bin_fluorescence_per_trial(fluorescence=fluorescence),
        )

    def _bin_fluorescence_per_trial(self, fluorescence: NDArray[np.float32]) -> NDArray[np.float32]:
        """Bins dF/F0 fluorescence per lap into a (cell_count, trial_count, bin_count) array.

        Notes:
            Excludes the trial id sentinel that the acquisition pipeline uses to mark "no trial" frames. Applies the
            same speed filter and uniform_filter1d smoothing used for the pooled binned fluorescence so that averaging
            the returned array across the trial axis reproduces the pooled binned_fluorescence within numerical
            rounding.

        Args:
            fluorescence: Pre-normalized dF/F0 fluorescence with dimensions (cell_count, frame_count).

        Returns:
            Per-lap binned fluorescence array with dimensions (cell_count, trial_count, bin_count). Bins and lap
            slices with no valid speed-filtered frames are filled with NaN.
        """
        # Collects valid trial identifiers, excluding the sentinel that marks "no trial" frames.
        valid_trial_mask = self.trial_ids != _NO_TRIAL_SENTINEL
        unique_trials = np.unique(self.trial_ids[valid_trial_mask])
        trial_count = len(unique_trials)

        cell_count = fluorescence.shape[0]
        bin_count = len(self._bin_edges) - 1

        output = np.full((cell_count, trial_count, bin_count), np.nan, dtype=np.float32)

        # Bins each lap independently, applying the same speed filter and smoothing as the pooled computation.
        for trial_index, trial_id in enumerate(unique_trials):
            trial_mask = (self.trial_ids == trial_id) & (self.speed > self.configuration.minimum_speed)
            if not np.any(trial_mask):
                continue

            trial_position = self.position[trial_mask]
            trial_fluorescence = fluorescence[:, trial_mask]

            trial_binned, _ = _bin_fluorescence_by_position(
                fluorescence=trial_fluorescence,
                position=trial_position,
                bin_edges=self._bin_edges,
            )

            # noinspection PyTypeChecker
            trial_binned: NDArray[np.float32] = uniform_filter1d(
                input=trial_binned,
                size=self.configuration.smooth_size,
                axis=1,
                mode="wrap",
            ).astype(np.float32)

            output[:, trial_index, :] = trial_binned

        return output

    def compute_shuffle_significance(
        self,
        repeat_count: int = 100,
        worker_count: int = -1,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Validates place fields by comparing observed fields against shuffled data via per-cell p-values.

        Args:
            repeat_count: Number of shuffles to perform.
            worker_count: Number of parallel workers for shuffle iterations. If -1, uses all CPU cores minus a
                small reserve.

        Returns:
            A tuple containing the significant cell indices and p-values arrays.
        """
        fluorescence, _ = _compute_delta_fluorescence(fluorescence=self.fluorescence)

        # Replaces NaN speed values with 0 so they fall below the minimum-speed gate without raising in comparisons.
        speed = self.speed.copy()
        speed[np.isnan(speed)] = 0

        # Detects place fields in the original dataset.
        observed = self._run_detection(
            fluorescence=fluorescence,
            position=self.position,
            speed=speed,
        ).has_place_field

        if worker_count == -1:
            worker_count = max(1, os.cpu_count() - _WORKER_RESERVE)

        # Spawns a thread for each shuffle iteration to parallelize detection.
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._shuffle_iteration_has_field, fluorescence, speed, iteration)
                for iteration in range(repeat_count)
            ]
            shuffled_results = [future.result() for future in tqdm(futures, desc="Shuffle significance", unit="iter")]

        shuffled_results = np.vstack(shuffled_results).T

        # Computes p-values as the proportion of shuffles where a place field was detected by chance.
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = (np.sum(shuffled_results, axis=1) / shuffled_results.shape[1]).astype(
            np.float32
        )

        # Selects cells with an observed place field and a p-value below the significance threshold.
        # noinspection PyTypeChecker
        significant_cells: NDArray[np.int32] = np.flatnonzero(
            observed & (p_values < self.configuration.significance_threshold)
        ).astype(np.int32)

        return significant_cells, p_values

    def plot(
        self,
        place_fields: PlaceFields,
        show_color_bar: bool = True,
        title: str | None = None,
        sort_by_position: bool = True,
        show_only_place_cells: bool = True,
        cell_mask: NDArray[np.bool_] | None = None,
        figure_dpi: int = 150,
        minimum_percentile: float = 0.5,
        maximum_percentile: float = 0.9,
        cmap: str = "gray_r",
    ) -> plt.Figure:
        """Plots binned fluorescence activity across cells as a position-ordered heatmap.

        Args:
            place_fields: A PlaceFields instance containing the binned fluorescence data to visualize.
            show_color_bar: Whether to display a color bar alongside the heatmap.
            title: Optional title displayed at the top of the figure.
            sort_by_position: Whether to order cells by their place field center location along the track.
            show_only_place_cells: Whether to display only cells that have detected place fields.
            cell_mask: Boolean mask with length cell_count specifying which cells to include in the plot. If provided,
                this overrides show_only_place_cells.
            figure_dpi: Resolution of the figure in dots per inch.
            minimum_percentile: Percentile of the data used to set the lower bound of the color scale.
            maximum_percentile: Percentile of the data used to set the upper bound of the color scale.
            cmap: Matplotlib colormap name for the heatmap. Defaults to grayscale.

        Returns:
            The matplotlib Figure object containing the heatmap.
        """
        data = place_fields.binned_fluorescence

        # Determines cell ordering based on place field position or original order.
        sort_order = place_fields.order if sort_by_position else np.arange(data.shape[0])

        # Filters to include only cells with place fields or those specified in the mask.
        if cell_mask is not None:
            sort_order = sort_order[np.isin(sort_order, np.flatnonzero(cell_mask))]
        elif show_only_place_cells:
            sort_order = sort_order[np.isin(sort_order, np.flatnonzero(place_fields.has_place_field))]

        data = data[sort_order, :]

        # Computes color scale limits from data percentiles to handle outliers.
        minimum_value = np.nanquantile(data, minimum_percentile)
        maximum_value = np.nanquantile(data, maximum_percentile)

        figure, axes = plt.subplots(1, 1, figsize=(8, 4), facecolor="white", dpi=figure_dpi)

        if title is not None:
            axes.set_title(title, fontsize=8)

        # Sets axis extent where x-axis shows the position in centimeters and y-axis shows the cell number.
        extent = [0, place_fields.bin_size * data.shape[1], data.shape[0], 0]
        image = axes.imshow(
            data,
            cmap=cmap,
            extent=extent,
            interpolation="none",
            vmin=minimum_value,
            vmax=maximum_value,
            origin="upper",
        )

        axes.set_aspect("auto")
        axes.set_xlabel("Position (cm)")
        axes.set_ylabel("Cell number")

        # Sets x-axis ticks at fixed centimeter intervals for consistent position labeling.
        track_length = place_fields.bin_size * data.shape[1]
        x_ticks = np.arange(0, track_length + 1, _PLOT_TICK_INTERVAL_CM)
        axes.set_xticks(x_ticks)

        if show_color_bar:
            cbar = figure.colorbar(image, ax=axes)
            cbar.set_label("ΔF/F₀")

            # Sets colorbar ticks at 0.5 ΔF/F₀ intervals for consistent fluorescence labeling.
            cbar_min = np.floor(minimum_value / 0.5) * 0.5
            cbar_max = np.ceil(maximum_value / 0.5) * 0.5
            cbar_ticks = np.arange(cbar_min, cbar_max, 0.5)
            cbar.set_ticks(cbar_ticks)

        return figure

    def _run_detection(
        self,
        fluorescence: NDArray[np.float32],
        position: NDArray[np.float32],
        speed: NDArray[np.float32],
    ) -> PlaceFields:
        """Runs the place field detection pipeline on dF/F0 normalized fluorescence data.

        Notes:
            Expects fluorescence data that has already been converted to dF/F0. This method is shared by both the
            public detect() method for original data and compute_shuffle_significance() for shuffled data.

        Args:
            fluorescence: Pre-normalized dF/F0 fluorescence data with dimensions (cell_count, timepoint_count).
            position: Position data with length timepoint_count.
            speed: Speed data with length timepoint_count.

        Returns:
            A PlaceFields instance containing the labeled regions, binned fluorescence, and centers of detected place
            fields.
        """
        # Excludes timepoints where the animal is moving below the minimum speed threshold.
        speed_mask = speed > self.configuration.minimum_speed
        position = position[speed_mask]
        fluorescence = fluorescence[:, speed_mask]

        # Bins fluorescence by spatial position along the track.
        binned_fluorescence, _ = _bin_fluorescence_by_position(
            fluorescence=fluorescence,
            position=position,
            bin_edges=self._bin_edges,
        )

        # Applies a moving average filter to smooth binned fluorescence across spatial bins. Casts back to float32
        # because uniform_filter1d promotes to float64.
        # noinspection PyTypeChecker
        binned_fluorescence: NDArray[np.float32] = uniform_filter1d(
            input=binned_fluorescence, size=self.configuration.smooth_size, axis=1, mode="wrap"
        ).astype(np.float32)

        # Creates a binary mask by thresholding bins that exceed the baseline-to-max activity level.
        thresholded_fluorescence = _compute_quantile_max_threshold(
            fluorescence=binned_fluorescence,
            base_quantile=self.configuration.base_quantile,
            threshold_factor=self.configuration.signal_threshold,
        )

        # Detects place fields as horizontally connected regions in the thresholded binary mask.
        place_fields = _compute_circular_connected_place_fields(
            thresholded_image=thresholded_fluorescence,
            binned_fluorescence=binned_fluorescence,
            minimum_bins=self.configuration.minimum_bins,
            bin_size=self.bin_size,
        )

        # Removes fields where in-field activity does not sufficiently exceed outside-field activity.
        place_fields = _outside_field_threshold(
            place_fields=place_fields,
            threshold_factor=self.configuration.outside_threshold,
        )

        # Removes fields with peak intensity below the minimum threshold.
        # noinspection PyTypeChecker
        invalid_indices: NDArray[np.int32] = np.flatnonzero(
            place_fields.max_intensity < self.configuration.maximum_intensity_threshold
        ).astype(np.int32)

        return place_fields.remove_fields(indices=invalid_indices)

    def _shuffle_iteration_has_field(
        self,
        fluorescence: NDArray[np.float32],
        speed: NDArray[np.float32],
        iteration: int,
    ) -> NDArray[np.bool_]:
        """Runs detection on a single shuffled fluorescence trace and returns the per-cell place field flag."""
        return self._run_detection(
            fluorescence=self._shuffle(data=fluorescence, iteration=iteration),
            position=self.position,
            speed=speed,
        ).has_place_field

    def _shuffle(self, data: NDArray[np.float32], iteration: int) -> NDArray[np.float32]:
        """Shuffles fluorescence traces by circular time-shifting to disrupt spatial tuning for significance testing.

        Args:
            data: Fluorescence data to be shuffled with dimensions (cell_count, timepoint_count).
            iteration: Shuffle iteration used as random seed.

        Returns:
            The shuffled fluorescence data with the same dimensions as input.
        """
        random_generator = np.random.default_rng(iteration)

        # Computes the minimum shift as a fraction of total frames based on chunk_count configuration.
        total_frames = data.shape[1]
        minimum_shift = total_frames // self.configuration.chunk_count

        # Generates a random shift amount that ensures at least minimum_shift displacement in either direction.
        shift_amount = random_generator.integers(minimum_shift, total_frames - minimum_shift)

        # Applies circular shift along the time axis to disrupt position-fluorescence correlations.
        return np.roll(data, shift=shift_amount, axis=1)
