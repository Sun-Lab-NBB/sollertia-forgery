"""Provides functionality for detecting and analyzing spatial tuning and place fields in neural recordings on
a linear track. Methodological references for the tuning pipeline live on
`..tuning_report.compute_tuning_report`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
from dataclasses import field, replace, dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
from scipy.ndimage import uniform_filter1d

from .utilities import (
    MINIMUM_VALID_BINS_FOR_PEARSON,
    RunSessionData,
    bin_fluorescence_per_trial,
    bin_fluorescence_by_position,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


_MINIMUM_TRIALS_FOR_STABILITY: int = 2
"""Minimum trial count required to compute a per-cell split-half Pearson r for the stability shuffle."""
_STABILITY_SHUFFLE_CHUNK_SIZE: int = 50
"""Number of stability-shuffle iterations dispatched into the numba kernel per tqdm progress tick. Small enough
that the progress bar updates smoothly, large enough that per-call kernel-launch overhead stays negligible."""
_PEAK_SHUFFLE_CHUNK_SIZE: int = 50
"""Number of peak-shuffle iterations dispatched into the numba kernel per tqdm progress tick. Mirrors the
stability-shuffle chunking so the progress bar advances smoothly while keeping per-call dispatch overhead
negligible against the heavier per-iteration work."""


@dataclass(frozen=True, slots=True)
class PlaceFieldDetectionConfiguration:
    """Defines configuration parameters for the place field detection algorithm."""

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
    minimum_lap_coverage: float = 0.33
    """Minimum fraction of laps on which a candidate field must show in-field activity above its out-of-field
    baseline."""
    shuffle_minimum_chunk_count: int = 100
    """Sets the minimum circular shift in the shuffle to ``total_samples / shuffle_minimum_chunk_count``
    samples. Used as the chunk-granularity floor when ``minimum_shift_seconds`` would either fall below one
    sample or exceed the safe upper bound (``total_samples / 4``)."""
    shuffle_repeat_count: int = 1000
    """Number of shuffle iterations for significance testing."""
    minimum_shift_seconds: float = 10.0
    """Minimum circular shift expressed in seconds. Set above the GCaMP6 autocorrelation timescale (roughly
    1.2-2 s) so the null distribution is not contaminated by indicator decay."""
    peak_percentile: float = 0.99
    """Percentile of the per-cell shuffled peak distribution above which the observed peak rate is classified
    as peak-significant."""


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


class PlaceFieldDetector:
    """Detects, validates, and visualizes 1D place fields using thresholding and connected component analysis."""

    def __init__(
        self,
        run_session: RunSessionData,
        *,
        bin_size: float = 5.0,
        configuration: PlaceFieldDetectionConfiguration | None = None,
    ) -> None:
        """Constructs the detector from already-loaded session data.

        Notes:
            Takes the canonical data dependency (a `RunSessionData` from
            `assemble_run_session_data`) so the same loaded session can feed both place- and reward-cell
            detectors without re-reading the feather.

        Args:
            run_session: Pre-loaded session data from `assemble_run_session_data`.
            bin_size: Size of spatial bins in centimeters.
            configuration: Configuration parameters for place field detection. Uses defaults if None.
        """
        # noinspection PyTypeChecker
        self.fluorescence: NDArray[np.float32] = run_session.fluorescence
        # noinspection PyTypeChecker
        self.position: NDArray[np.float32] = run_session.position
        # noinspection PyTypeChecker
        self.speed: NDArray[np.float32] = run_session.speed
        # noinspection PyTypeChecker
        self.trial_ids: NDArray[np.int32] = run_session.trial_ids
        self.track_length = run_session.geometry.trial_length_cm
        self.bin_size = bin_size
        self.sampling_rate_hz = run_session.sampling_rate_hz
        self.configuration = configuration if configuration is not None else PlaceFieldDetectionConfiguration()

        # Caches the bin edges shared by `_run_detection` and `_bin_fluorescence_per_trial`.
        # noinspection PyTypeChecker
        self._bin_edges: NDArray[np.float32] = np.arange(0, self.track_length + bin_size, bin_size, dtype=np.float32)

        # Caches shuffle-invariant arrays so each shuffle iteration only computes the indirection array and the
        # rate-map accumulation, not the speed mask or bin assignments. Computed at init time because the
        # configuration is stable for the lifetime of the detector.
        self._build_shuffle_invariants()

    def _build_shuffle_invariants(self) -> None:
        """Builds the shuffle-invariant arrays (speed-filtered sample indices, bin indices, occupancy, minimum shift)
        and caches them on self.

        Notes:
            Called once from ``__init__``. Replaces NaN speed values with 0 so they fall below the minimum-speed gate
            without raising in comparisons. Resolves the minimum shift the same way ``_shuffle`` does:
            seconds-based when in range, falling back to chunk-granularity floor on degenerate short sessions.
        """
        # noinspection PyTypeChecker
        speed_safe: NDArray[np.float32] = self.speed.copy()
        speed_safe[np.isnan(speed_safe)] = 0
        # noinspection PyTypeChecker
        speed_mask: NDArray[np.bool_] = speed_safe > self.configuration.minimum_speed

        bin_count = len(self._bin_edges) - 1
        # noinspection PyTypeChecker
        self._shuffle_filtered_sample_indices: NDArray[np.int32] = np.nonzero(speed_mask)[0].astype(np.int32)
        # noinspection PyTypeChecker
        filtered_position: NDArray[np.float32] = self.position[speed_mask]
        # noinspection PyTypeChecker
        self._shuffle_filtered_bin_indices: NDArray[np.int32] = np.clip(
            np.searchsorted(self._bin_edges, filtered_position, side="right") - 1, 0, bin_count - 1
        ).astype(np.int32)
        # noinspection PyTypeChecker
        self._shuffle_sample_counts: NDArray[np.int32] = np.bincount(
            self._shuffle_filtered_bin_indices, minlength=bin_count
        ).astype(np.int32)

        total_samples = int(self.fluorescence.shape[1])
        seconds_based_shift = int(self.sampling_rate_hz * self.configuration.minimum_shift_seconds)
        chunk_based_shift = total_samples // self.configuration.shuffle_minimum_chunk_count
        upper_bound = total_samples // 4
        self._shuffle_minimum_shift: int = (
            seconds_based_shift if 0 < seconds_based_shift <= upper_bound else max(chunk_based_shift, 1)
        )
        self._shuffle_total_samples: int = total_samples
        self._shuffle_bin_count: int = bin_count

    def detect(self) -> PlaceFields:
        """Detects place fields from the original fluorescence and position data.

        Notes:
            Pipeline ordering: threshold-based detection -> per-trial binning -> lap-coverage filter.
            Significance is reported by the multi-criterion shuffles in
            `compute_multi_criterion_significance`; the legacy combined "did a place field appear under
            shuffle?" filter is intentionally not applied here so downstream consumers can decide on a
            population using each criterion's per-cell p-value rather than a single conflated cutoff.

        Returns:
            A PlaceFields instance containing the labeled regions, pooled and per-lap binned fluorescence, and centers
            of detected place fields.
        """
        # Bins fluorescence by spatial position, applies thresholding, and detects connected regions as place fields.
        place_fields = self._run_detection(
            fluorescence=self.fluorescence,
            position=self.position,
            speed=self.speed,
        )

        # Computes per-lap binned fluorescence using the same speed filter and smoothing as the pooled computation.
        # noinspection PyTypeChecker
        per_trial_binned: NDArray[np.float32] = bin_fluorescence_per_trial(
            fluorescence=self.fluorescence,
            position=self.position,
            speed=self.speed,
            trial_ids=self.trial_ids,
            bin_edges=self._bin_edges,
            minimum_speed=self.configuration.minimum_speed,
            smooth_size=self.configuration.smooth_size,
        )

        # Drops fields with insufficient lap coverage on the per-trial binned matrix, which is independent of the
        # shuffle nulls used by the multi-criterion classifier.
        place_fields = _lap_coverage_filter(
            place_fields=place_fields,
            binned_fluorescence_per_trial=per_trial_binned,
            minimum_lap_coverage=self.configuration.minimum_lap_coverage,
        )

        return replace(
            place_fields,
            binned_fluorescence_per_trial=per_trial_binned,
        )

    def compute_multi_criterion_significance(
        self,
        observed_pooled_rate_map: NDArray[np.float32],
        observed_per_trial_rate_map: NDArray[np.float32],
        observed_split_half_r: NDArray[np.float32],
        repeat_count: int = 1000,
        peak_percentile: float = 0.99,
        stability_percentile: float = 0.95,
        *,
        display_progress: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]]:
        """Computes the per-cell IS_STABLE and IS_PEAK_SIGNIFICANT flags via per-cell shuffle distributions, and
        returns the corresponding per-cell p-values.

        Notes:
            The peak null uses a time-domain circular-shift shuffle, captures the per-cell peak of each shuffled
            smoothed rate map, and classifies cells whose observed peak exceeds the per-cell ``peak_percentile`` of
            the shuffled distribution. The stability null is a per-trial shuffle that circularly shifts each trial's
            rate map by an independent random bin offset and recomputes the per-cell split-half Pearson r; cells
            whose observed r exceeds the per-cell ``stability_percentile`` of the shuffled distribution are
            classified as stable. The p-value for each measure is the fraction of shuffles whose statistic is greater
            than or equal to the observed value (NaN where the observed statistic is NaN or the shuffled distribution
            is empty).

            Both nulls are produced by single ``@njit(parallel=True)`` kernels dispatched in chunks so the
            tqdm progress bar advances smoothly while every CPU core stays saturated. The kernels parallelize
            over ``(iteration * cell_count)`` so the per-iteration cell loop never serializes the heavy work.

        Args:
            observed_pooled_rate_map: Observed pooled rate map with dimensions (cell_count, bin_count).
            observed_per_trial_rate_map: Observed per-trial rate map with dimensions (cell_count, trial_count,
                bin_count). May be empty for sessions with too few trials; the stability flag falls back to all-False
                and the stability p-values to all-NaN.
            observed_split_half_r: Observed per-cell split-half Pearson r with length cell_count. NaN entries are
                classified as not-stable and receive a NaN p-value.
            repeat_count: Number of shuffle iterations.
            peak_percentile: Percentile cutoff (0-1) for the peak-method classifier; default 0.99.
            stability_percentile: Percentile cutoff (0-1) for the stability classifier; default 0.95.
            display_progress: When True, render the inner Peak / Stability shuffle tqdm bars; set to False by
                ``..tuning_report.run_tuning_analysis`` when its session-level progress bar is the active visual
                signal so the bars stay quiet.

        Returns:
            A tuple of (is_stable, is_peak_significant, stability_p_values, peak_p_values). All four arrays have
            length cell_count; the booleans share the percentile cutoff, and the p-values report the per-cell rank
            of the observed value within its shuffled distribution.
        """
        cell_count = observed_pooled_rate_map.shape[0]

        # Pre-generates one shift per iteration with ``default_rng(iteration)`` so the kernel reproduces the
        # original per-iteration seeding exactly. Doing this in numpy keeps reproducibility intact while letting
        # the kernel stay seed-agnostic and avoid contending on numpy's global random state from worker threads.
        # noinspection PyTypeChecker
        peak_shifts: NDArray[np.int32] = np.empty(repeat_count, dtype=np.int32)
        upper_bound = self._shuffle_total_samples - self._shuffle_minimum_shift
        for iteration in range(repeat_count):
            generator = np.random.default_rng(iteration)
            peak_shifts[iteration] = int(
                generator.integers(low=self._shuffle_minimum_shift, high=upper_bound, dtype=np.int64)
            )

        # noinspection PyTypeChecker
        shuffled_peaks: NDArray[np.float32] = np.full((repeat_count, cell_count), np.nan, dtype=np.float32)
        smooth_size = int(self.configuration.smooth_size)
        peak_chunk_starts = list(range(0, repeat_count, _PEAK_SHUFFLE_CHUNK_SIZE))
        for chunk_start in tqdm(peak_chunk_starts, desc="Peak shuffle", unit="chunk", disable=not display_progress):
            chunk_end = min(chunk_start + _PEAK_SHUFFLE_CHUNK_SIZE, repeat_count)
            _peak_shuffle_kernel(
                fluorescence=self.fluorescence,
                filtered_sample_indices=self._shuffle_filtered_sample_indices,
                bin_indices=self._shuffle_filtered_bin_indices,
                sample_counts=self._shuffle_sample_counts,
                shift_amounts=peak_shifts[chunk_start:chunk_end],
                smooth_size=smooth_size,
                sample_count=self._shuffle_total_samples,
                output=shuffled_peaks[chunk_start:chunk_end],
            )

        # noinspection PyTypeChecker
        observed_peaks: NDArray[np.float32] = np.nanmax(observed_pooled_rate_map, axis=1)
        with np.errstate(invalid="ignore"):
            # noinspection PyTypeChecker
            peak_thresholds: NDArray[np.float32] = np.nanquantile(shuffled_peaks, peak_percentile, axis=0).astype(
                np.float32
            )
        # noinspection PyTypeChecker
        is_peak_significant: NDArray[np.bool_] = observed_peaks > peak_thresholds
        peak_p_values = _per_cell_p_values(observed=observed_peaks, shuffled=shuffled_peaks)

        # Runs the per-trial stability shuffle. Generates a null by circularly shifting each trial's rate map by
        # a random bin offset (independent per trial), then recomputes split-half r per cell.
        # noinspection PyTypeChecker
        is_stable: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)
        # noinspection PyTypeChecker
        stability_p_values: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
        if (
            observed_per_trial_rate_map.size > 0
            and observed_per_trial_rate_map.shape[1] >= _MINIMUM_TRIALS_FOR_STABILITY
        ):
            stability_thresholds, stability_p_values = self._stability_shuffle_threshold(
                per_trial_rate_map=observed_per_trial_rate_map,
                repeat_count=repeat_count,
                stability_percentile=stability_percentile,
                observed_split_half_r=observed_split_half_r,
                display_progress=display_progress,
            )
            # noinspection PyTypeChecker
            valid_observed: NDArray[np.bool_] = ~np.isnan(observed_split_half_r)
            is_stable = valid_observed & (observed_split_half_r > stability_thresholds)

        return is_stable, is_peak_significant, stability_p_values, peak_p_values

    def _stability_shuffle_threshold(
        self,
        per_trial_rate_map: NDArray[np.float32],
        repeat_count: int,
        stability_percentile: float,
        observed_split_half_r: NDArray[np.float32],
        *,
        display_progress: bool = True,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Computes the per-cell stability threshold and per-cell stability p-value from a per-trial circular-shift
        null distribution.

        Notes:
            Each shuffle iteration shifts every trial's rate map by an independent random bin offset, then computes
            the per-cell split-half Pearson r between the first-half and second-half means of the shuffled per-trial
            matrix. Returns the per-cell ``stability_percentile`` quantile of the shuffle distribution alongside the
            per-cell p-value (fraction of shuffles whose r is greater than or equal to the observed r).

            Hands the heavy work to ``_stability_shuffle_kernel`` (numba ``parallel=True`` over iterations) so the
            entire shuffle distribution materializes in one shot rather than a Python loop with per-iteration
            ``np.roll``, ``np.nanmean``, and ``per_cell_pearson_safe`` calls. The kernel is dispatched in chunks
            of ``_STABILITY_SHUFFLE_CHUNK_SIZE`` so a tqdm progress bar can advance as iterations complete; per-cell
            shifts are pre-generated with ``np.random.default_rng(iteration)`` to preserve the original
            seed-per-iteration reproducibility.

        Args:
            per_trial_rate_map: Per-trial rate map with dimensions (cell_count, trial_count, bin_count).
            repeat_count: Number of shuffle iterations.
            stability_percentile: Percentile cutoff (0-1).
            observed_split_half_r: Observed per-cell split-half Pearson r with length cell_count. Used to compute the
                per-cell p-value against the shuffled distribution.
            display_progress: When True, render the inner Stability shuffle tqdm bar; set to False when the
                session-level progress bar of ``..tuning_report.run_tuning_analysis`` is active.

        Returns:
            A tuple of (thresholds, p_values), each with length cell_count.
        """
        cell_count, trial_count, bin_count = per_trial_rate_map.shape
        half_index = trial_count // 2

        # Pre-generates one shift vector per iteration with ``default_rng(iteration)`` so the kernel reproduces
        # the original per-iteration seeding exactly; doing this in numpy keeps reproducibility intact while
        # letting the kernel stay seed-agnostic.
        # noinspection PyTypeChecker
        all_shifts: NDArray[np.int32] = np.empty((repeat_count, trial_count), dtype=np.int32)
        for iteration in range(repeat_count):
            generator = np.random.default_rng(iteration)
            all_shifts[iteration] = generator.integers(low=0, high=bin_count, size=trial_count, dtype=np.int32)

        # noinspection PyTypeChecker
        shuffled_split_half: NDArray[np.float32] = np.full((repeat_count, cell_count), np.nan, dtype=np.float32)

        chunk_starts = list(range(0, repeat_count, _STABILITY_SHUFFLE_CHUNK_SIZE))
        for chunk_start in tqdm(chunk_starts, desc="Stability shuffle", unit="chunk", disable=not display_progress):
            chunk_end = min(chunk_start + _STABILITY_SHUFFLE_CHUNK_SIZE, repeat_count)
            _stability_shuffle_kernel(
                per_trial_rate_map=per_trial_rate_map,
                shifts=all_shifts[chunk_start:chunk_end],
                half_index=half_index,
                output=shuffled_split_half[chunk_start:chunk_end],
            )

        with np.errstate(invalid="ignore"):
            # noinspection PyTypeChecker
            thresholds: NDArray[np.float32] = np.nanquantile(shuffled_split_half, stability_percentile, axis=0).astype(
                np.float32
            )
        p_values = _per_cell_p_values(observed=observed_split_half_r, shuffled=shuffled_split_half)
        return thresholds, p_values

    def _run_detection(
        self,
        fluorescence: NDArray[np.float32],
        position: NDArray[np.float32],
        speed: NDArray[np.float32],
    ) -> PlaceFields:
        """Runs the place field detection pipeline on dF/F0 normalized fluorescence data.

        Notes:
            Speed-filters, bins by position, smooths, then hands off to `_detect_from_smoothed_rate_map` for
            the threshold + label + outside-field + peak filter steps.

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

        return self._detect_from_smoothed_rate_map(binned_fluorescence=binned_fluorescence)

    def _detect_from_smoothed_rate_map(self, binned_fluorescence: NDArray[np.float32]) -> PlaceFields:
        """Runs the threshold + label + outside-field + peak filter steps on a precomputed smoothed rate map.

        Notes:
            Called from `_run_detection` after the speed-mask + bin-by-position + smooth steps. Does not
            apply the lap-coverage filter (which requires per-trial binning); `detect` applies it after
            this method returns.

        Args:
            binned_fluorescence: Smoothed per-cell rate map with dimensions (cell_count, bin_count).

        Returns:
            A PlaceFields instance containing the labeled regions, the input rate map, and centers.
        """
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

    Notes:
        Per cell, baseline is the mean of bins at or below the ``base_quantile``-th percentile, and the
        threshold is ``baseline + threshold_factor * (peak - baseline)`` -- the canonical "25% of
        (peak - baseline)" place-field criterion.

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


@njit(cache=True, parallel=True)
def _stability_shuffle_kernel(
    per_trial_rate_map: NDArray[np.float32],
    shifts: NDArray[np.int32],
    half_index: int,
    output: NDArray[np.float32],
) -> None:
    """Computes per-cell shuffled split-half Pearson r values for a chunk of stability-shuffle iterations.

    Notes:
        Parallelism is flattened over ``prange(chunk_count * cell_count)`` so every (iteration, cell) pair is an
        independent task. The previous version paralleled only over iterations (``chunk_count`` ≤ 50); with cell
        counts in the thousands, that left most cores idle on per-iteration cell loops. The flat axis spawns
        ``chunk_count * cell_count`` tasks, which numba schedules across all available cores for full saturation.

        The trial-outer / bin-inner loop ordering replaces the prior bin-outer / trial-inner traversal so that for
        every (cell, trial) the inner sweep walks ``per_trial_rate_map[cell, trial, :]`` contiguously (after a
        single circular wrap), which is cache-friendly. Per-task ``first_map`` / ``second_map`` scratch buffers
        sized to ``bin_count`` stay small enough that numba's allocator pools them across tasks per worker thread.

        Replicates ``per_cell_pearson_safe`` semantics inline so the kernel fuses the mean accumulation and Pearson
        computation into one walk over the cell's bins; matches ``MINIMUM_VALID_BINS_FOR_PEARSON`` and the
        zero-variance guard of the standalone helper. Output entries whose mean or variance fail the validity
        checks are left at their caller-supplied initial value (NaN).

    Args:
        per_trial_rate_map: Per-trial rate map with dimensions (cell_count, trial_count, bin_count), C-contiguous
            fp32.
        shifts: Random per-trial circular shifts for this chunk with dimensions (chunk_count, trial_count). Each
            row is one iteration's full shift vector.
        half_index: Number of trials in the first half (``trial_count // 2``).
        output: Pre-allocated (chunk_count, cell_count) fp32 buffer pre-filled with NaN. The kernel writes the
            per-iteration per-cell Pearson r in place.
    """
    chunk_count = shifts.shape[0]
    cell_count = per_trial_rate_map.shape[0]
    trial_count = per_trial_rate_map.shape[1]
    bin_count = per_trial_rate_map.shape[2]
    total_tasks = chunk_count * cell_count

    for task in prange(total_tasks):
        iteration = task // cell_count
        cell_index = task % cell_count

        # Per-task scratch sized to bin_count. Numba's allocator pools and reuses these across the tasks each
        # worker thread receives, so allocation overhead is amortized even at hundreds of thousands of tasks.
        # noinspection PyTypeChecker
        first_sum: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
        # noinspection PyTypeChecker
        first_count: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)
        # noinspection PyTypeChecker
        second_sum: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
        # noinspection PyTypeChecker
        second_count: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)

        # Phase 1a: accumulate first-half contributions. Trial-outer / bin-inner walks
        # ``per_trial_rate_map[cell, trial, :]`` contiguously after one circular wrap.
        for trial_index in range(half_index):
            shift = shifts[iteration, trial_index]
            for bin_index in range(bin_count):
                source_bin = bin_index - shift
                if source_bin < 0:
                    source_bin += bin_count
                elif source_bin >= bin_count:
                    source_bin -= bin_count
                value = per_trial_rate_map[cell_index, trial_index, source_bin]
                if not np.isnan(value):
                    first_sum[bin_index] += value
                    first_count[bin_index] += 1

        # Phase 1b: accumulate second-half contributions identically.
        for trial_index in range(half_index, trial_count):
            shift = shifts[iteration, trial_index]
            for bin_index in range(bin_count):
                source_bin = bin_index - shift
                if source_bin < 0:
                    source_bin += bin_count
                elif source_bin >= bin_count:
                    source_bin -= bin_count
                value = per_trial_rate_map[cell_index, trial_index, source_bin]
                if not np.isnan(value):
                    second_sum[bin_index] += value
                    second_count[bin_index] += 1

        # Phase 2: NaN-safe Pearson r between the two split-half mean maps. Mirrors
        # ``per_cell_pearson_safe``: skip bins with no valid contribution in either half; require at least
        # ``MINIMUM_VALID_BINS_FOR_PEARSON`` valid bins; guard against zero variance. Bins are validated and
        # converted to means inline so we do not need a separate scratch traversal.
        valid_count = 0
        sum_a = np.float32(0.0)
        sum_b = np.float32(0.0)
        for bin_index in range(bin_count):
            count_a = first_count[bin_index]
            count_b = second_count[bin_index]
            if count_a > 0 and count_b > 0:
                valid_count += 1
                sum_a += first_sum[bin_index] / np.float32(count_a)
                sum_b += second_sum[bin_index] / np.float32(count_b)
        if valid_count < MINIMUM_VALID_BINS_FOR_PEARSON:
            continue
        mean_a = sum_a / np.float32(valid_count)
        mean_b = sum_b / np.float32(valid_count)

        var_a = np.float32(0.0)
        var_b = np.float32(0.0)
        cov = np.float32(0.0)
        for bin_index in range(bin_count):
            count_a = first_count[bin_index]
            count_b = second_count[bin_index]
            if count_a > 0 and count_b > 0:
                value_a = first_sum[bin_index] / np.float32(count_a)
                value_b = second_sum[bin_index] / np.float32(count_b)
                diff_a = value_a - mean_a
                diff_b = value_b - mean_b
                var_a += diff_a * diff_a
                var_b += diff_b * diff_b
                cov += diff_a * diff_b
        if var_a <= 0.0 or var_b <= 0.0:
            continue
        output[iteration, cell_index] = cov / np.sqrt(var_a * var_b)


@njit(cache=True, parallel=True)
def _peak_shuffle_kernel(
    fluorescence: NDArray[np.float32],
    filtered_sample_indices: NDArray[np.int32],
    bin_indices: NDArray[np.int32],
    sample_counts: NDArray[np.int32],
    shift_amounts: NDArray[np.int32],
    smooth_size: int,
    sample_count: int,
    output: NDArray[np.float32],
) -> None:
    """Computes the per-cell peak of one circularly-shifted, smoothed shuffled rate map for each iteration.

    Notes:
        Parallelism is flattened over ``prange(chunk_count * cell_count)``: every (iteration, cell) pair is an
        independent task that builds its own rate map, smooths it inline with a wrap-around uniform kernel, and
        writes the per-cell peak to ``output``. Replaces the legacy ThreadPoolExecutor-driven shuffle that
        dispatched 1000 Python futures (each calling several numba kernels with their own parallel-over-cells
        ``prange``); fanning out at the (iteration, cell) granularity keeps every core busy without nested
        parallelism overhead.

        Inlines the equivalent of ``compute_shuffle_source_indices(chunk_count=1)`` plus
        ``accumulate_shuffled_rate_maps`` plus ``uniform_filter1d(mode="wrap")`` plus ``np.nanmax`` so the kernel
        does the entire single-iteration pipeline without intermediate allocations beyond the per-task
        ``bin_count``-sized scratch.

    Args:
        fluorescence: Pre-normalized dF/F0 fluorescence with dimensions (cell_count, sample_count).
        filtered_sample_indices: Speed-filtered destination sample indices with length filtered_count.
        bin_indices: Spatial bin assignment for each filtered sample with length filtered_count.
        sample_counts: Per-bin occupancy counts with length bin_count (matches ``bincount`` on
            ``bin_indices``).
        shift_amounts: Per-iteration circular shift amounts with length chunk_count.
        smooth_size: Width of the wrap-around uniform smoothing kernel applied along the bin axis. Must be odd.
        sample_count: Total number of samples in the original fluorescence time series.
        output: Pre-allocated (chunk_count, cell_count) fp32 buffer; the kernel writes per-cell peaks in place.
    """
    chunk_count = shift_amounts.shape[0]
    cell_count = fluorescence.shape[0]
    filtered_count = filtered_sample_indices.shape[0]
    bin_count = sample_counts.shape[0]
    half_smooth = smooth_size // 2
    total_tasks = chunk_count * cell_count

    for task in prange(total_tasks):
        iteration = task // cell_count
        cell_index = task % cell_count
        shift = shift_amounts[iteration]

        # Per-task scratch for the per-cell rate map. Allocated once per task and zeroed in place.
        # noinspection PyTypeChecker
        bin_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)

        # Accumulate fluorescence into bins via the indirection-array shuffle. ``filtered_sample_indices`` and
        # ``bin_indices`` are shuffle-invariant and shared across all tasks.
        for filtered_index in range(filtered_count):
            destination = filtered_sample_indices[filtered_index]
            source = destination - shift
            if source < 0:
                source += sample_count
            elif source >= sample_count:
                source -= sample_count
            bin_sums[bin_indices[filtered_index]] += fluorescence[cell_index, source]

        # Convert to per-bin means in place. Empty bins keep their zero value (matches the existing convention
        # in ``accumulate_shuffled_rate_maps`` so smoothing remains numerically stable).
        for bin_index in range(bin_count):
            if sample_counts[bin_index] > 0:
                bin_sums[bin_index] = bin_sums[bin_index] / np.float32(sample_counts[bin_index])
            else:
                bin_sums[bin_index] = np.float32(0.0)

        # Inline circular uniform smoothing followed by per-cell peak. Matches
        # ``uniform_filter1d(mode="wrap")`` semantics: each output bin averages a centered window of size
        # ``smooth_size`` with wrap-around at the track boundary. Tracking the running maximum avoids
        # materializing a separate smoothed buffer.
        peak = np.float32(-np.inf)
        for bin_index in range(bin_count):
            total = np.float32(0.0)
            for offset in range(-half_smooth, half_smooth + 1):
                neighbor = bin_index + offset
                if neighbor < 0:
                    neighbor += bin_count
                elif neighbor >= bin_count:
                    neighbor -= bin_count
                total += bin_sums[neighbor]
            smoothed = total / np.float32(smooth_size)
            peak = max(peak, smoothed)
        output[iteration, cell_index] = peak


def _per_cell_p_values(observed: NDArray[np.float32], shuffled: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes per-cell p-values as the fraction of shuffles whose statistic exceeds or equals the observed value.

    Notes:
        Returns NaN for cells whose observed value is NaN or whose shuffled column contains no finite entries. NaN
        entries inside the shuffled distribution are excluded from the denominator so that fragmented per-trial
        shuffles (e.g., on cells with frequent all-NaN trial halves) do not bias the p-value toward zero.

    Args:
        observed: Per-cell observed values with length cell_count.
        shuffled: Per-shuffle, per-cell values with dimensions (repeat_count, cell_count).

    Returns:
        Per-cell p-values with length cell_count.
    """
    cell_count = int(observed.shape[0])
    # noinspection PyTypeChecker
    p_values: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if shuffled.shape[1] != cell_count or shuffled.shape[0] == 0:
        return p_values
    for cell_index in range(cell_count):
        observed_value = float(observed[cell_index])
        if np.isnan(observed_value):
            continue
        cell_shuffled = shuffled[:, cell_index]
        # noinspection PyTypeChecker
        valid_mask: NDArray[np.bool_] = ~np.isnan(cell_shuffled)
        valid_count = int(np.sum(valid_mask))
        if valid_count == 0:
            continue
        hit_count = int(np.sum(cell_shuffled[valid_mask] >= observed_value))
        p_values[cell_index] = np.float32(hit_count / valid_count)
    return p_values


def _lap_coverage_filter(
    place_fields: PlaceFields,
    binned_fluorescence_per_trial: NDArray[np.float32],
    minimum_lap_coverage: float = 0.33,
) -> PlaceFields:
    """Filters place fields by requiring per-lap in-field activity to exceed the cell's out-of-field baseline on at
    least ``minimum_lap_coverage`` of laps with valid in-field samples.

    Notes:
        Computes the per-cell out-of-field baseline from the pooled rate map (matching the convention in
        `_outside_field_threshold`). For each detected field, walks every lap, computes the mean in-field
        fluorescence for that lap from the per-trial binned matrix, and counts laps where that mean exceeds
        the baseline. Laps where every in-field bin is NaN (no samples landed in the field that lap) are
        excluded from both the numerator and the denominator. Fields are dropped when the resulting coverage
        fraction is below ``minimum_lap_coverage``.

    Args:
        place_fields: PlaceFields object with previously detected fields.
        binned_fluorescence_per_trial: Per-lap binned fluorescence with dimensions (cell_count, trial_count,
            bin_count). NaN-filled bins/laps with no valid speed-filtered samples are treated as missing.
        minimum_lap_coverage: Minimum fraction of laps on which in-field activity must exceed the out-of-field
            baseline.

    Returns:
        The filtered PlaceFields object.
    """
    if binned_fluorescence_per_trial.size == 0:
        return place_fields

    label_image = place_fields.label_image
    region_count = int(np.max(label_image)) if label_image.size > 0 else 0
    if region_count == 0:
        return place_fields

    # Recomputes the per-cell out-of-field baseline; matches _outside_field_threshold so coverage and ratio filters
    # share the same reference value. Cells whose every bin is labeled as a place field, and trials whose every
    # in-field bin is NaN, both produce empty slices for nanmean; the resulting NaNs are handled downstream via the
    # valid_lap_mask, but nanmean still emits "Mean of empty slice" for those rows. The catch_warnings block below
    # silences only that specific message so unrelated RuntimeWarnings continue to surface.
    # noinspection PyTypeChecker
    outside_image: NDArray[np.float32] = place_fields.binned_fluorescence.copy()
    outside_image[label_image != 0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        # noinspection PyTypeChecker
        outside_values: NDArray[np.float32] = np.nanmean(outside_image, axis=1)

        invalid_regions: list[int] = []
        cell_ids = place_fields.cell_id
        for region_index in range(region_count):
            label = region_index + 1
            cell_index = int(cell_ids[region_index])
            # noinspection PyTypeChecker
            in_field_bin_mask: NDArray[np.bool_] = label_image[cell_index] == label
            # noinspection PyTypeChecker
            per_trial_in_field: NDArray[np.float32] = binned_fluorescence_per_trial[cell_index][:, in_field_bin_mask]
            # noinspection PyTypeChecker
            in_field_means: NDArray[np.float32] = np.nanmean(per_trial_in_field, axis=1)
            # noinspection PyTypeChecker
            valid_lap_mask: NDArray[np.bool_] = ~np.isnan(in_field_means)
            valid_lap_count = int(np.sum(valid_lap_mask))
            if valid_lap_count == 0:
                invalid_regions.append(region_index)
                continue
            active_lap_count = int(np.sum(in_field_means[valid_lap_mask] > outside_values[cell_index]))
            if active_lap_count / valid_lap_count < minimum_lap_coverage:
                invalid_regions.append(region_index)

    if not invalid_regions:
        return place_fields
    # noinspection PyTypeChecker
    return place_fields.remove_fields(indices=np.asarray(invalid_regions, dtype=np.int32))


def _outside_field_threshold(place_fields: PlaceFields, threshold_factor: float = 3.0) -> PlaceFields:
    """Filters place fields by requiring in-field activity to exceed out-of-field baseline by a threshold factor.

    Notes:
        Removes false positives by requiring that detected place fields have significantly higher activity
        than the baseline outside the field. In cases where a cell has multiple fields, both fields are
        excluded from the outside field calculation.

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
