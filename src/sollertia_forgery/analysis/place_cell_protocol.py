"""Provides functionality for detecting and analyzing spatial tuning and place fields in neural recordings on a linear
track.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from dataclasses import field, replace, dataclass
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from numba import njit, prange
import numpy as np
from scipy.ndimage import uniform_filter1d

from ..forging import FluorescenceColumn
from .utilities import assemble_run_session_data, bin_fluorescence_by_position

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_NO_TRIAL_SENTINEL: int = 255
"""Sentinel trial id used by the acquisition pipeline to mark samples outside of any trial."""
_WORKER_RESERVE: int = 4
"""Number of CPU cores reserved for the OS when worker_count=-1 selects an automatic worker count."""


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
        # noinspection PyTypeChecker
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
        # noinspection PyTypeChecker
        sort_order: NDArray[np.float32] = np.full(cell_count, np.inf, dtype=np.float32)
        intensity = self.mean_intensity
        field_centers = self.centers
        # noinspection PyTypeChecker
        field_cell_id: NDArray[np.int32] = field_centers[:, 0].astype(np.int32)

        # Orders cells with multiple place fields based on the field with the highest mean intensity.
        for cell_index in range(cell_count):
            # noinspection PyTypeChecker
            cell_field_mask: NDArray[np.bool_] = field_cell_id == cell_index
            # noinspection PyTypeChecker
            cell_field_indices: NDArray[np.int64] = np.flatnonzero(cell_field_mask)
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
        # noinspection PyTypeChecker
        new_label_image: NDArray[np.int32] = self.label_image.copy()
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
        # noinspection PyTypeChecker
        new_label_image: NDArray[np.int32] = self.label_image.copy()
        # noinspection PyTypeChecker
        all_indices: NDArray[np.int64] = np.arange(new_label_image.shape[0])
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


class PlaceFieldDetector:
    """Detects, validates, and visualizes 1D place fields using thresholding and connected component analysis."""

    def __init__(
        self,
        session_path: Path,
        trial_type: str,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
        bin_size: float = 5.0,
        configuration: PlaceFieldDetectionConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, position, speed, and trial data from the session feather and trial geometry data file
        for place field detection.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type: Trial type to analyze (e.g. "ABC", "ABCD"). Must match an entry in the session's trial
                geometry data file.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            bin_size: Size of spatial bins in centimeters.
            configuration: Configuration parameters for place field detection. Uses defaults if None.
        """
        session = assemble_run_session_data(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )
        # noinspection PyTypeChecker
        self.fluorescence: NDArray[np.float32] = session.fluorescence
        # noinspection PyTypeChecker
        self.position: NDArray[np.float32] = session.position
        # noinspection PyTypeChecker
        self.speed: NDArray[np.float32] = session.speed
        # noinspection PyTypeChecker
        self.trial_ids: NDArray[np.int32] = session.trial_ids
        self.track_length = session.geometry.trial_length_cm
        self.bin_size = bin_size
        self.configuration = configuration if configuration is not None else PlaceFieldDetectionConfiguration()

        # Caches the bin edges shared by `_run_detection` and `_bin_fluorescence_per_trial`.
        # noinspection PyTypeChecker
        self._bin_edges: NDArray[np.float32] = np.arange(0, self.track_length + bin_size, bin_size, dtype=np.float32)

    def detect(self, *, run_shuffle: bool = False) -> PlaceFields:
        """Detects place fields from the original fluorescence and position data.

        Args:
            run_shuffle: Determines whether to run shuffle significance testing and filter results to only include
                cells with statistically significant place fields.

        Returns:
            A PlaceFields instance containing the labeled regions, pooled and per-lap binned fluorescence, and centers
            of detected place fields. If run_shuffle is True, only significant cells are included.
        """
        # Bins fluorescence by spatial position, applies thresholding, and detects connected regions as place fields.
        place_fields = self._run_detection(
            fluorescence=self.fluorescence,
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
            binned_fluorescence_per_trial=self._bin_fluorescence_per_trial(fluorescence=self.fluorescence),
        )

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
        # Replaces NaN speed values with 0 so they fall below the minimum-speed gate without raising in comparisons.
        # noinspection PyTypeChecker
        speed: NDArray[np.float32] = self.speed.copy()
        speed[np.isnan(speed)] = 0

        # Detects place fields in the original dataset.
        observed = self._run_detection(
            fluorescence=self.fluorescence,
            position=self.position,
            speed=speed,
        ).has_place_field

        if worker_count == -1:
            worker_count = max(1, (os.cpu_count() or 1) - _WORKER_RESERVE)

        # Spawns a thread for each shuffle iteration to parallelize detection.
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._shuffle_iteration_has_field, self.fluorescence, speed, iteration)
                for iteration in range(repeat_count)
            ]
            shuffled_results = [future.result() for future in tqdm(futures, desc="Shuffle significance", unit="iter")]

        # noinspection PyTypeChecker
        stacked_results: NDArray[np.bool_] = np.vstack(shuffled_results).T

        # Computes p-values as the proportion of shuffles where a place field was detected by chance.
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = (np.sum(stacked_results, axis=1) / stacked_results.shape[1]).astype(np.float32)

        # Selects cells with an observed place field and a p-value below the significance threshold.
        # noinspection PyTypeChecker
        significant_cells: NDArray[np.int32] = np.flatnonzero(
            observed & (p_values < self.configuration.significance_threshold)
        ).astype(np.int32)

        return significant_cells, p_values

    def _bin_fluorescence_per_trial(self, fluorescence: NDArray[np.float32]) -> NDArray[np.float32]:
        """Bins dF/F0 fluorescence per lap into a (cell_count, trial_count, bin_count) array.

        Notes:
            Excludes the trial id sentinel that the acquisition pipeline uses to mark "no trial" samples. Applies the
            same speed filter and uniform_filter1d smoothing used for the pooled binned fluorescence so that averaging
            the returned array across the trial axis reproduces the pooled binned_fluorescence within numerical
            rounding.

        Args:
            fluorescence: Pre-normalized dF/F0 fluorescence with dimensions (cell_count, sample_count).

        Returns:
            Per-lap binned fluorescence array with dimensions (cell_count, trial_count, bin_count). Bins and lap
            slices with no valid speed-filtered samples are filled with NaN.
        """
        # Collects valid trial identifiers, excluding the sentinel that marks "no trial" samples.
        # noinspection PyTypeChecker
        valid_trial_mask: NDArray[np.bool_] = self.trial_ids != _NO_TRIAL_SENTINEL
        # noinspection PyTypeChecker
        unique_trials: NDArray[np.int32] = np.unique(self.trial_ids[valid_trial_mask])
        trial_count = len(unique_trials)

        cell_count = fluorescence.shape[0]
        bin_count = len(self._bin_edges) - 1

        # noinspection PyTypeChecker
        output: NDArray[np.float32] = np.full((cell_count, trial_count, bin_count), np.nan, dtype=np.float32)

        # Bins each lap independently, applying the same speed filter and smoothing as the pooled computation.
        for trial_index, trial_id in enumerate(unique_trials):
            # noinspection PyTypeChecker
            trial_mask: NDArray[np.bool_] = (self.trial_ids == trial_id) & (
                self.speed > self.configuration.minimum_speed
            )
            if not np.any(trial_mask):
                continue

            trial_position = self.position[trial_mask]
            trial_fluorescence = fluorescence[:, trial_mask]

            raw_trial_binned, _ = bin_fluorescence_by_position(
                fluorescence=trial_fluorescence,
                position=trial_position,
                position_bin_edges=self._bin_edges,
            )

            # noinspection PyTypeChecker
            smoothed_trial_binned: NDArray[np.float32] = uniform_filter1d(
                input=raw_trial_binned,
                size=self.configuration.smooth_size,
                axis=1,
                mode="wrap",
            ).astype(np.float32)

            output[:, trial_index, :] = smoothed_trial_binned

        return output

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
        # noinspection PyTypeChecker
        speed_mask: NDArray[np.bool_] = speed > self.configuration.minimum_speed
        position = position[speed_mask]
        fluorescence = fluorescence[:, speed_mask]

        # Bins fluorescence by spatial position along the track.
        raw_binned_fluorescence, _ = bin_fluorescence_by_position(
            fluorescence=fluorescence,
            position=position,
            position_bin_edges=self._bin_edges,
        )

        # Applies a moving average filter to smooth binned fluorescence across spatial bins. Casts back to float32
        # because uniform_filter1d promotes to float64.
        # noinspection PyTypeChecker
        binned_fluorescence: NDArray[np.float32] = uniform_filter1d(
            input=raw_binned_fluorescence, size=self.configuration.smooth_size, axis=1, mode="wrap"
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

        # Computes the minimum shift as a fraction of total samples based on chunk_count configuration.
        total_samples = data.shape[1]
        minimum_shift = total_samples // self.configuration.chunk_count

        # Generates a random shift amount that ensures at least minimum_shift displacement in either direction.
        shift_amount = random_generator.integers(minimum_shift, total_samples - minimum_shift)

        # Applies circular shift along the time axis to disrupt position-fluorescence correlations.
        # noinspection PyTypeChecker
        return np.roll(data, shift=shift_amount, axis=1)


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
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    # noinspection PyTypeChecker
    cell_indices: NDArray[np.int32] = np.zeros(region_count, dtype=np.int32)
    # noinspection PyTypeChecker
    intensity_sums: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)
    # noinspection PyTypeChecker
    weighted_sums: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)

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
    # noinspection PyTypeChecker
    weighted_centroids: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)
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
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float32)

    # noinspection PyTypeChecker
    intensity_sums: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)
    # noinspection PyTypeChecker
    pixel_counts: NDArray[np.int32] = np.zeros(region_count, dtype=np.int32)

    # Accumulates intensity and pixel count for each labeled region.
    for cell_index in range(label_image.shape[0]):
        for bin_index in range(label_image.shape[1]):
            label_value = label_image[cell_index, bin_index]
            if label_value > 0:
                region_index = label_value - 1
                intensity_sums[region_index] += intensity_image[cell_index, bin_index]
                pixel_counts[region_index] += 1

    # Computes mean intensity by dividing total intensity by pixel count.
    # noinspection PyTypeChecker
    mean_intensities: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)
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
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float32)

    # noinspection PyTypeChecker
    max_intensities: NDArray[np.float32] = np.zeros(region_count, dtype=np.float32)

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
    # noinspection PyTypeChecker
    flat_cell_indices: NDArray[np.int32] = np.zeros(total_regions, dtype=np.int32)
    # noinspection PyTypeChecker
    flat_areas: NDArray[np.int32] = np.zeros(total_regions, dtype=np.int32)
    # noinspection PyTypeChecker
    flat_centroids: NDArray[np.float32] = np.zeros(total_regions, dtype=np.float32)

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
        # noinspection PyTypeChecker
        return np.zeros((0, 2), dtype=np.float32)

    # Converts bin indices to centimeters and combines with cell indices into a (field_count, 2) array.
    # noinspection PyTypeChecker
    return np.column_stack((cell_indices.astype(np.float32), weighted_centroids * bin_size))


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
    # noinspection PyTypeChecker
    output: NDArray[np.bool_] = np.empty(fluorescence.shape, dtype=np.bool_)

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
    # noinspection PyTypeChecker
    padded_threshold: NDArray[np.bool_] = np.pad(thresholded_image, ((0, 0), (bin_count, bin_count)), mode="wrap")
    # noinspection PyTypeChecker
    padded_fluorescence: NDArray[np.float32] = np.pad(
        binned_fluorescence, ((0, 0), (bin_count, bin_count)), mode="wrap"
    )

    # Pre-allocates output arrays for labeling and property computation.
    # noinspection PyTypeChecker
    padded_labels: NDArray[np.int32] = np.zeros(padded_threshold.shape, dtype=np.int32)
    # noinspection PyTypeChecker
    cell_indices: NDArray[np.int32] = np.zeros((cell_count, max_regions_per_row), dtype=np.int32)
    # noinspection PyTypeChecker
    areas: NDArray[np.int32] = np.zeros((cell_count, max_regions_per_row), dtype=np.int32)
    # noinspection PyTypeChecker
    centroids: NDArray[np.float32] = np.zeros((cell_count, max_regions_per_row), dtype=np.float32)
    # noinspection PyTypeChecker
    region_counts: NDArray[np.int32] = np.zeros(cell_count, dtype=np.int32)

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
    # noinspection PyTypeChecker
    valid_mask: NDArray[np.bool_] = (centroids >= bin_count) & (centroids < bin_count * 2) & (areas >= minimum_bins)
    # noinspection PyTypeChecker
    valid_indices: NDArray[np.int64] = np.where(valid_mask)[0]

    # noinspection PyTypeChecker
    result_label_image: NDArray[np.int32] = np.zeros(thresholded_image.shape, dtype=np.int32)
    adjusted_centers: list[list[float]] = []

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
    # noinspection PyTypeChecker
    outside_image: NDArray[np.float32] = place_fields.binned_fluorescence.copy()
    outside_image[place_fields.label_image != 0] = np.nan
    # noinspection PyTypeChecker
    outside_values: NDArray[np.float32] = np.nanmean(outside_image, axis=1)

    # Identifies fields where in-field activity does not exceed the outside-field baseline by the threshold factor.
    # noinspection PyTypeChecker
    threshold_values: NDArray[np.float32] = outside_values[place_fields.cell_id] * threshold_factor
    # noinspection PyTypeChecker
    invalid_regions: NDArray[np.int32] = np.flatnonzero(place_fields.mean_intensity < threshold_values).astype(np.int32)

    return place_fields.remove_fields(indices=invalid_regions)
