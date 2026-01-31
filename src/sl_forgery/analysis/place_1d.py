"""Provides functionality for detecting and analyzing place fields in neural recordings."""

from copy import deepcopy
from dataclasses import field, dataclass

import dask
from numba import njit, prange
import numpy as np
import colorcet as cc
import dask.array as da
from numpy.typing import NDArray
from scipy.signal import convolve2d
from scipy.ndimage import label, filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from ataraxis_base_utilities import console


@dataclass
class PlaceFieldDetectionParams:
    """Defines configuration parameters for Tank lab place field detection algorithm."""

    minimum_speed: float = 5.0
    """Minimum speed threshold in cm/s for including timepoints in analysis."""
    smooth_size: int = 3
    """Size of the smoothing kernel in bins for the moving average filter."""
    base_quantile: float = 0.25
    """Quantile used as baseline for computing the activity threshold."""
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


@dataclass
class PlaceFields1d:
    """Stores detected place fields in one-dimensional space.

    Attributes:
        label_image: Labeled image of detected place fields with dimensions (cell_count, bin_count).
        binned_fluorescence: Binned fluorescence data with dimensions (cell_count, bin_count).
        centers: Centers of detected place fields with dimensions (field_count, 2).
        bin_size: Size of spatial bins in centimeters.
    """

    label_image: NDArray[np.int32]
    """Labeled image of detected place fields with dimensions (cell_count, bin_count)."""
    binned_fluorescence: NDArray[np.float32]
    """Binned fluorescence data with dimensions (cell_count, bin_count)."""
    centers: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    """Centers of detected place fields with dimensions (field_count, 2)."""
    bin_size: float = 1.0
    """Size of spatial bins in centimeters."""

    def __post_init__(self) -> None:
        """Validates and normalizes field data types, computing centers if not provided."""
        self.label_image = self.label_image.astype(np.int32)
        self.binned_fluorescence = self.binned_fluorescence.astype(np.float32)

        if self.centers is None or len(self.centers) == 0:
            properties = regionprops(
                label_image=self.label_image, intensity_image=self.binned_fluorescence, cache=False
            )
            self.centers = np.array(
                [prop["weighted_centroid"] * np.array([1, self.bin_size]) for prop in properties],
                dtype=np.float32,
            )
        else:
            self.centers = self.centers.astype(np.float32)

    @property
    def mean_intensity(self) -> NDArray[np.float32]:
        """Returns the mean intensity for each detected place field."""
        properties = regionprops(label_image=self.label_image, intensity_image=self.binned_fluorescence, cache=False)
        return np.array([prop["mean_intensity"] for prop in properties], dtype=np.float32)

    @property
    def max_intensity(self) -> NDArray[np.float32]:
        """Returns the maximum intensity for each detected place field."""
        properties = regionprops(label_image=self.label_image, intensity_image=self.binned_fluorescence, cache=False)
        return np.array([prop["max_intensity"] for prop in properties], dtype=np.float32)

    @property
    def cell_id(self) -> NDArray[np.int32]:
        """Returns the cell ID for each detected place field."""
        properties = regionprops(label_image=self.label_image, intensity_image=self.binned_fluorescence, cache=False)
        return np.array([region["coords"][0, 0] for region in properties], dtype=np.int32)

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
        sort_order = np.full(cell_count, np.inf, dtype=np.float32)
        intensity = self.mean_intensity
        field_centers = self.centers

        if not field_centers.any():
            return sort_order.astype(np.int32)

        field_cell_id = field_centers[:, 0].astype(np.int32)

        # For cells with multiple place fields, orders based on the field with highest mean intensity.
        for cell_index in range(cell_count):
            cell_field_indices = np.argwhere(field_cell_id == cell_index)
            if cell_field_indices.size > 0:
                max_intensity_index = np.argmax(intensity[field_cell_id == cell_index])
                sort_order[cell_index] = field_centers[cell_field_indices[max_intensity_index], 1]

        return np.argsort(sort_order).astype(np.int32)

    def remove_fields(self, indices: NDArray[np.int32]) -> PlaceFields1d:
        """Removes specified place fields and returns a new PlaceFields1d object.

        Args:
            indices: Indices of place fields to remove.

        Returns:
            A new PlaceFields1d object with the specified fields removed.
        """
        place_fields = deepcopy(self)

        place_fields.label_image[np.isin(place_fields.label_image, indices + 1)] = 0
        for counter, value in enumerate(np.unique(place_fields.label_image)):
            if value != 0:
                place_fields.label_image[place_fields.label_image == value] = counter

        place_fields.centers = np.delete(place_fields.centers, indices, axis=0)

        return place_fields

    def filter_cells(self, indices: NDArray[np.int32]) -> PlaceFields1d:
        """Filters to keep only specified cells and returns a new PlaceFields1d object.

        Args:
            indices: Indices of cells to keep.

        Returns:
            A new PlaceFields1d object containing only the specified cells.
        """
        place_fields = deepcopy(self)

        all_indices = np.arange(0, place_fields.label_image.shape[0])
        field_cell_id = place_fields.cell_id
        place_fields.label_image[~np.isin(all_indices, indices), :] = 0

        for counter, value in enumerate(np.unique(place_fields.label_image)):
            if value != 0:
                place_fields.label_image[place_fields.label_image == value] = counter

        place_fields.centers = place_fields.centers[np.isin(field_cell_id, indices), :]

        return place_fields


def _compute_baseline_fluorescence(
    fluorescence: NDArray[np.float32],
    method: str = "maximin",
    gaussian_sigma: float = 20.0,
    filter_window_size: int = 600,
) -> NDArray[np.float32]:
    """Computes the baseline of fluorescence data.

    Supported methods:
      - 'maximin': Applies Gaussian filter, followed by min and max filtering.
      - 'average': Computes mean for each row of fluorescence.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        method: The baseline calculation method.
        gaussian_sigma: Sigma of the Gaussian filter for 'maximin' method.
        filter_window_size: Window size for min/max filtering in 'maximin' method.

    Returns:
        Baseline array with the same shape as fluorescence.

    Raises:
        ValueError: If an unknown method is provided or fluorescence has invalid dimensions.
    """
    if method == "maximin":
        if fluorescence.ndim == 2:
            baseline = filters.gaussian_filter(input=fluorescence, sigma=[0.0, gaussian_sigma])
        elif fluorescence.ndim == 1:
            baseline = filters.gaussian_filter(input=fluorescence, sigma=[gaussian_sigma])
        else:
            message = (
                f"Unable to compute baseline fluorescence. Expected fluorescence to be 1D or 2D, "
                f"but got {fluorescence.ndim}D."
            )
            console.error(message=message, error=ValueError)

        baseline = filters.minimum_filter1d(input=baseline, size=filter_window_size)
        baseline = filters.maximum_filter1d(input=baseline, size=filter_window_size)

    elif method == "average":
        baseline = np.tile(np.mean(fluorescence, axis=1), (fluorescence.shape[1], 1)).T

    else:
        message = (
            f"Unable to compute baseline fluorescence. The method must be 'maximin' or 'average', but got '{method}'."
        )
        console.error(message=message, error=ValueError)

    return baseline


def _compute_delta_fluorescence(
    fluorescence: NDArray[np.float32],
    baseline_method: str = "maximin",
    subtract_minimum: bool = False,
    **kwargs,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes delta F over F0 for fluorescence data.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        baseline_method: Baseline calculation method. See _compute_baseline_fluorescence() for valid methods.
        subtract_minimum: Whether to subtract the minimum fluorescence from each row before baseline calculation.

    Returns:
        A tuple containing the delta fluorescence array (F - F0) / F0 and the baseline F0 array,
        both with the same shape as the input fluorescence.
    """
    if subtract_minimum:
        fluorescence = fluorescence - np.min(fluorescence, axis=1)[..., np.newaxis]

    baseline = _compute_baseline_fluorescence(fluorescence=fluorescence, method=baseline_method, **kwargs)
    delta_fluorescence = (fluorescence - baseline) / baseline

    return delta_fluorescence, baseline


@njit(cache=True, parallel=True)
def _bin_fluorescence_worker(
    fluorescence: NDArray[np.float32],
    bin_indices: NDArray[np.int32],
    bin_count: int,
    sample_counts: NDArray[np.int32],
    use_mean: bool,
    output: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Worker function which accumulates fluorescence values into spatial bins across cells.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        bin_indices: Bin index for each frame with length frame_count.
        bin_count: Total number of bins.
        sample_counts: Number of samples per bin with length bin_count.
        use_mean: If True, compute mean; otherwise compute sum.
        output: Pre-allocated output array with dimensions (cell_count, bin_count).

    Returns:
        The output array filled with binned fluorescence values.
    """
    cell_count = fluorescence.shape[0]
    frame_count = fluorescence.shape[1]

    for cell_index in prange(cell_count):
        bin_sums = np.zeros(bin_count, dtype=np.float32)

        for frame_index in range(frame_count):
            bin_idx = bin_indices[frame_index]
            bin_sums[bin_idx] += fluorescence[cell_index, frame_index]

        for bin_index in range(bin_count):
            if sample_counts[bin_index] > 0:
                if use_mean:
                    output[cell_index, bin_index] = bin_sums[bin_index] / sample_counts[bin_index]
                else:
                    output[cell_index, bin_index] = bin_sums[bin_index]
            else:
                output[cell_index, bin_index] = np.nan

    return output


def _bin_fluorescence_by_position(
    fluorescence: NDArray[np.float32],
    position: NDArray[np.float32],
    bin_edges: NDArray[np.float32],
    compute_mean: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Bins fluorescence data according to position values.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        position: Position values used for binning with length matching frame_count.
        bin_edges: Bin edges for spatial binning.
        compute_mean: Determines whether to compute mean or sum for each bin.

    Returns:
        A tuple containing the binned fluorescence array with dimensions (cell_count, bin_count)
        and the sample count per bin with length bin_count.
    """
    bin_indices = np.searchsorted(bin_edges, position, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2).astype(np.int32)

    bin_count = len(bin_edges) - 1
    cell_count = fluorescence.shape[0]
    sample_counts = np.bincount(bin_indices, minlength=bin_count).astype(np.int32)

    output = np.full((cell_count, bin_count), np.nan, dtype=np.float32)

    _bin_fluorescence_worker(
        fluorescence=fluorescence,
        bin_indices=bin_indices,
        bin_count=bin_count,
        sample_counts=sample_counts,
        use_mean=compute_mean,
        output=output,
    )

    return output, sample_counts


@njit(cache=True, parallel=True)
def _compute_base_values(
    fluorescence: NDArray[np.float32],
    quantile_values: NDArray[np.float32],
    base_values: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Worker function which computes the mean of values below the quantile threshold for each cell.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        quantile_values: Quantile value for each cell with length cell_count.
        base_values: Pre-allocated output array for base values with length cell_count.

    Returns:
        The base_values array updated with the mean of values below each cell's quantile threshold.
    """
    cell_count = fluorescence.shape[0]

    for cell_index in prange(cell_count):
        row = fluorescence[cell_index, :]
        quantile_threshold = quantile_values[cell_index]

        total = 0.0
        count = 0

        for frame_index in range(row.shape[0]):
            value = row[frame_index]
            if not np.isnan(value) and value <= quantile_threshold:
                total += value
                count += 1

        if count > 0:
            base_values[cell_index] = total / count
        else:
            base_values[cell_index] = np.nan

    return base_values


@njit(cache=True, parallel=True)
def _apply_threshold(
    fluorescence: NDArray[np.float32],
    threshold: NDArray[np.float32],
    output: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Worker function which applies per-cell thresholds to fluorescence data.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        threshold: Threshold values with length cell_count.
        output: Pre-allocated output boolean array with dimensions (cell_count, frame_count).

    Returns:
        The output array updated with True where fluorescence exceeds the threshold.
    """
    cell_count = fluorescence.shape[0]
    frame_count = fluorescence.shape[1]

    for cell_index in prange(cell_count):
        cell_threshold = threshold[cell_index]
        for frame_index in range(frame_count):
            value = fluorescence[cell_index, frame_index]
            if np.isnan(value):
                output[cell_index, frame_index] = False
            else:
                output[cell_index, frame_index] = value > cell_threshold

    return output


def _quantile_max_threshold(
    fluorescence: NDArray[np.float32],
    base_quantile: float = 0.25,
    threshold_factor: float = 0.25,
) -> NDArray[np.bool_]:
    """Thresholds fluorescence data using a fractional difference between a baseline quantile and the maximum value.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        base_quantile: Quantile used as baseline.
        threshold_factor: Fraction of (max_val - base_val).

    Returns:
        Boolean mask of the same shape as fluorescence, True where values exceed the computed threshold.
    """
    max_values = np.nanmax(fluorescence, axis=1).astype(np.float32)
    quantile_values = np.nanquantile(fluorescence, base_quantile, axis=1).astype(np.float32)

    base_values = np.empty(fluorescence.shape[0], dtype=np.float32)
    base_values = _compute_base_values(
        fluorescence=fluorescence, quantile_values=quantile_values, base_values=base_values
    )

    threshold = (base_values + (max_values - base_values) * threshold_factor).astype(np.float32)

    output = np.empty(fluorescence.shape, dtype=np.bool_)
    return _apply_threshold(fluorescence=fluorescence, threshold=threshold, output=output)


def circular_connected_placefields(
    thresholded_image: NDArray[np.bool_],
    binned_fluorescence: NDArray[np.float32],
    minimum_bins: int = 3,
) -> PlaceFields1d:
    """Creates a labeled image of circularly connected regions within each cell row.

    Args:
        thresholded_image: Thresholded binary image of binned place field activity with dimensions
                           (cell_count, bin_count).
        binned_fluorescence: Binned fluorescence data with dimensions (cell_count, bin_count).
        minimum_bins: Minimal required size of a connected region in bins.

    Returns:
        The detected one-dimensional place fields.
    """
    bin_count = thresholded_image.shape[1]

    padded_threshold = np.pad(thresholded_image, ((0, 0), (bin_count, bin_count)), mode="wrap")
    padded_fluorescence = np.pad(binned_fluorescence, ((0, 0), (bin_count, bin_count)), mode="wrap")

    label_image, _ = label(input=padded_threshold, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    properties = np.array(regionprops(label_image=label_image, intensity_image=padded_fluorescence, cache=False))

    field_centers = np.array([prop["weighted_centroid"] for prop in properties])
    area = np.array([prop["area"] for prop in properties], dtype=np.uint32)

    # Selects components with center within original area (for circularity) and minimum area size.
    valid_indices = (field_centers[:, 1] >= bin_count) & (field_centers[:, 1] < bin_count * 2) & (area >= minimum_bins)

    result_label_image = np.zeros(thresholded_image.shape, dtype=np.int32)
    adjusted_centers = []

    for counter, prop in enumerate(properties[valid_indices]):
        coordinates = prop["coords"]
        wrapped_indices = np.take(np.arange(0, bin_count), coordinates[:, 1], mode="wrap")
        result_label_image[coordinates[:, 0], wrapped_indices] = counter + 1

        center = np.array(prop["weighted_centroid"])
        center[1] -= bin_count
        adjusted_centers.append(center)

    return PlaceFields1d(
        label_image=result_label_image,
        binned_fluorescence=binned_fluorescence,
        centers=np.vstack(adjusted_centers).astype(np.float32),
    )


def outside_field_threshold(place_fields: PlaceFields1d, threshold_factor: float = 3) -> PlaceFields1d:
    """Filters place fields based on the signal-to-baseline ratio.

    Removes false positives by requiring that detected place fields have significantly higher activity than the
    baseline outside the field. In cases where a cell has multiple fields, both fields are excluded from the
    outside field calculation.

    Args:
        place_fields: PlaceFields1d object with previously detected place fields.
        threshold_factor: Scalar factor of the outside field signal to set the threshold.

    Returns:
        The filtered PlaceFields1d object.
    """
    outside_image = place_fields.binned_fluorescence.copy()
    outside_image[place_fields.label_image != 0] = np.nan
    outside_values = np.nanmean(outside_image, axis=1)

    threshold_values = outside_values[place_fields.cell_id] * threshold_factor
    invalid_regions = np.concatenate(np.argwhere(place_fields.mean_intensity < threshold_values))

    return place_fields.remove_fields(indices=invalid_regions)


@dataclass
class PlaceFieldDetector1d:
    """Detects, validates, and visualizes 1D place fields using thresholding and connected component analysis."""

    fluorescence: NDArray[np.float32]
    """Fluorescence data with dimensions (cell_count, timepoint_count)."""
    position: NDArray[np.float32]
    """Position data with length timepoint_count."""
    speed: NDArray[np.float32]
    """Speed data with length timepoint_count."""
    track_length: float
    """Length of the track in centimeters."""
    bin_size: float
    """Size of spatial bins in centimeters."""
    detection_params: PlaceFieldDetectionParams = field(default_factory=PlaceFieldDetectionParams)
    """Configuration parameters for place field detection."""

    def detect(self, calculate_df: bool = True) -> PlaceFields1d:
        """Detects place fields from fluorescence and position data.

        Args:
            calculate_df: Determines whether to calculate dF/F0.

        Returns:
            The detected place fields.
        """
        fluorescence = self.fluorescence.copy()
        if calculate_df:
            fluorescence, _ = _compute_delta_fluorescence(fluorescence=fluorescence)

        speed_indices = self.speed > self.detection_params.minimum_speed
        position = self.position[speed_indices]
        fluorescence = fluorescence[:, speed_indices]

        bin_edges = np.arange(0, self.track_length + self.bin_size, self.bin_size, dtype=np.float32)
        binned_fluorescence, _ = _bin_fluorescence_by_position(
            fluorescence=fluorescence,
            position=position,
            bin_edges=bin_edges,
        )

        binned_fluorescence = convolve2d(
            binned_fluorescence,
            np.ones((1, self.detection_params.smooth_size)) / self.detection_params.smooth_size,
            mode="same",
            boundary="wrap",
        )

        thresholded_fluorescence = _quantile_max_threshold(
            fluorescence=binned_fluorescence.astype(np.float32),
            base_quantile=self.detection_params.base_quantile,
            threshold_factor=self.detection_params.signal_threshold,
        )

        place_fields = circular_connected_placefields(
            thresholded_image=thresholded_fluorescence,
            binned_fluorescence=binned_fluorescence.astype(np.float32),
            minimum_bins=self.detection_params.minimum_bins,
        )
        place_fields.bin_size = self.bin_size

        place_fields = outside_field_threshold(
            place_fields=place_fields,
            threshold_factor=self.detection_params.outside_threshold,
        )

        place_fields = place_fields.remove_fields(
            indices=np.argwhere(place_fields.max_intensity < self.detection_params.maximum_intensity_threshold)
        )

        return place_fields

    def validate_shuffle(self, repeat_count: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Validates place fields using a shuffle test by comparing observed fields with shuffled data to compute
        p-values.

        Args:
            repeat_count: Number of shuffles to perform.

        Returns:
            A tuple containing the significant cell indices and p-values arrays.
        """
        fluorescence, _ = _compute_delta_fluorescence(fluorescence=self.fluorescence)
        fluorescence = da.from_array(fluorescence)

        speed = self.speed.copy()
        speed[np.isnan(speed)] = 0

        results = [
            dask.delayed(self._detect_from_data)(
                fluorescence=fluorescence,
                position=self.position,
                speed=speed,
                calculate_df=False,
            ).has_place_field
        ]

        for iteration in range(repeat_count):
            shuffled = self._shuffle(data=fluorescence, iteration=iteration)
            result = dask.delayed(self._detect_from_data)(
                fluorescence=shuffled,
                position=self.position,
                speed=speed,
                calculate_df=False,
            ).has_place_field
            results.append(result)

        results = dask.compute(results)
        results = np.vstack(results).T

        observed = results[:, 0]
        shuffled_results = results[:, 1:]

        p_values = (np.sum(shuffled_results, axis=1) / shuffled_results.shape[1]).astype(np.float32)

        significant_cells = np.argwhere(
            (observed) & (p_values < self.detection_params.significance_threshold)
        ).flatten().astype(np.int32)

        return significant_cells, p_values

    def plot(
        self,
        place_fields: PlaceFields1d,
        show_color_bar: bool = True,
        title: str | None = None,
        sort_by_position: bool = True,
        cell_mask: NDArray[np.bool_] | None = None,
        figure_dpi: int = 150,
        minimum_percentile: float = 0.5,
        maximum_percentile: float = 0.9,
        **kwargs,
    ) -> plt.Figure:
        """Plots place field activity as a heatmap.

        Args:
            place_fields: The detected place fields to visualize.
            show_color_bar: Whether to display a color bar.
            title: Title for the plot.
            sort_by_position: Whether to sort cells by place field position.
            cell_mask: Boolean mask specifying which cells to plot. All cells are plotted by default.
            figure_dpi: Figure DPI.
            minimum_percentile: Percentile for minimum color scaling value.
            maximum_percentile: Percentile for maximum color scaling value.

        Returns:
            The matplotlib Figure object containing the heatmap.
        """
        data = place_fields.binned_fluorescence

        sort_order = place_fields.order if sort_by_position else np.arange(0, data.shape[0])

        if cell_mask is not None:
            sort_order = sort_order[np.isin(sort_order, np.argwhere(cell_mask))]

        data = data[sort_order, :]

        minimum_value = np.nanquantile(data, minimum_percentile)
        maximum_value = np.nanquantile(data, maximum_percentile)

        figure, axes = plt.subplots(1, 1, figsize=(2, 3), facecolor="white", dpi=figure_dpi)

        if title:
            plt.title(title, fontsize=8)

        extent = [0, place_fields.bin_size * data.shape[1], 1, data.shape[0] + 1]
        plt.imshow(
            data,
            cmap=cc.cm.CET_CBL2,
            extent=extent,
            interpolation="none",
            vmin=minimum_value,
            vmax=maximum_value,
            **kwargs,
        )

        axes.set_aspect("auto")
        plt.xlabel("Position (cm)")
        plt.ylabel("Cell #")

        if show_color_bar:
            plt.colorbar()

        return figure

    def _detect_from_data(
        self,
        fluorescence: NDArray[np.float32],
        position: NDArray[np.float32],
        speed: NDArray[np.float32],
        calculate_df: bool = True,
    ) -> PlaceFields1d:
        """Detects place fields from provided data arrays.

        Args:
            fluorescence: Fluorescence data with dimensions (cell_count, timepoint_count).
            position: Position data with length timepoint_count.
            speed: Speed data with length timepoint_count.
            calculate_df: Determines whether to calculate dF/F0.

        Returns:
            The detected place fields.
        """
        if calculate_df:
            fluorescence, _ = _compute_delta_fluorescence(fluorescence=fluorescence)

        speed_indices = speed > self.detection_params.minimum_speed
        position = position[speed_indices]
        fluorescence = fluorescence[:, speed_indices]

        bin_edges = np.arange(0, self.track_length + self.bin_size, self.bin_size, dtype=np.float32)
        binned_fluorescence, _ = _bin_fluorescence_by_position(
            fluorescence=fluorescence,
            position=position,
            bin_edges=bin_edges,
        )

        binned_fluorescence = convolve2d(
            binned_fluorescence,
            np.ones((1, self.detection_params.smooth_size)) / self.detection_params.smooth_size,
            mode="same",
            boundary="wrap",
        )

        thresholded_fluorescence = _quantile_max_threshold(
            fluorescence=binned_fluorescence.astype(np.float32),
            base_quantile=self.detection_params.base_quantile,
            threshold_factor=self.detection_params.signal_threshold,
        )

        place_fields = circular_connected_placefields(
            thresholded_image=thresholded_fluorescence,
            binned_fluorescence=binned_fluorescence.astype(np.float32),
            minimum_bins=self.detection_params.minimum_bins,
        )
        place_fields.bin_size = self.bin_size

        place_fields = outside_field_threshold(
            place_fields=place_fields,
            threshold_factor=self.detection_params.outside_threshold,
        )

        place_fields = place_fields.remove_fields(
            indices=np.argwhere(place_fields.max_intensity < self.detection_params.maximum_intensity_threshold)
        )

        return place_fields

    @dask.delayed
    def _shuffle(self, data: NDArray[np.float32], iteration: int) -> NDArray[np.float32]:
        """Shuffles fluorescence data for validation by splitting into chunks and reordering.

        Args:
            data: Fluorescence data to be shuffled with dimensions (cell_count, timepoint_count).
            iteration: Shuffle iteration used as random seed.

        Returns:
            The shuffled fluorescence data with the same dimensions as input.
        """
        data_chunks = np.array_split(data, self.detection_params.chunk_count, axis=1)
        rng = np.random.default_rng(iteration)
        shuffle_indices = rng.choice(
            np.arange(self.detection_params.chunk_count),
            self.detection_params.chunk_count,
            replace=False,
        )
        shuffled_data = np.concatenate([data_chunks[index] for index in shuffle_indices], axis=1)

        return shuffled_data
