"""Provides functionality for detecting Synchronous Calcium Events (SCEs) in neural recordings."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ..forging import FluorescenceColumn
from .utilities import trim_acquisition_warmup, compute_within_trial_position
from ..shared_assets import DatasetFiles, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from sollertia_forgery.analysis.place_cell_protocol import PlaceFields


_MINIMUM_STABLE_FRACTION: float = 0.5
"""Minimum fraction of stable torque samples required for a rest period to be included in SCE analysis."""
_MINIMUM_STABLE_SAMPLE_COUNT: int = 10
"""Minimum number of stable torque samples required for a rest period to be included in SCE analysis."""


class PeriodType(StrEnum):
    """Defines the analysis period types for SCE detection."""

    REST = "rest"
    """Indicates a rest period where the animal is stationary."""
    RUN = "run"
    """Indicates a run period where the animal is actively locomoting."""


@dataclass(frozen=True, slots=True)
class SCEDetectionConfiguration:
    """Defines configuration parameters for synchronous calcium event detection."""

    smoothing_window_seconds: float = 0.5
    """Window length in seconds for the smoothing filter applied to each cell's fluorescence trace."""
    smoothing_order: int = 3
    """Polynomial order for the smoothing filter."""
    derivative_threshold_scale: float = 3.0
    """Number of standard deviations above the mean derivative for transient onset detection."""
    minimum_inter_event_seconds: float = 1.0
    """Minimum interval in seconds between consecutive calcium transients for the same cell."""
    coactivation_window_seconds: float = 0.2
    """Width of the sliding window in seconds used to count co-active cells for SCE detection."""
    shuffle_count: int = 1000
    """Number of temporal shuffles used to generate the null distribution for SCE significance testing."""
    significance_scale: float = 3.0
    """Number of standard deviations above the shuffled mean required to classify a time bin as an SCE."""
    minimum_cell_count: int = 5
    """Minimum number of co-active cells required for a valid SCE."""
    torque_stability_window_seconds: float = 5.0
    """Window length in seconds for computing the rolling standard deviation of torque during rest periods."""
    torque_stability_threshold: float = 0.1
    """Maximum allowable rolling standard deviation of torque (in N*cm) for a rest sample to be considered stable."""


@dataclass(slots=True)
class SCEResult:
    """Stores the results of SCE detection for a single analysis period.

    Attributes:
        period_type: Identifies this result as belonging to a rest or run period.
        onset_matrix: Binary matrix of calcium transient onsets with dimensions (cell_count, sample_count). Each entry
            is True if the corresponding cell has a transient onset at that sample.
        smoothed_fluorescence: Smoothed fluorescence traces with dimensions (cell_count, sample_count), used for
            plotting continuous activity traces.
        coactive_counts: Number of co-active cells at each sample with length sample_count.
        sce_mask: Boolean mask indicating samples that belong to a detected SCE with length sample_count.
        sce_labels: Integer labels assigning each SCE sample to an SCE event index (1-indexed) with length sample_count.
            Samples outside SCEs have label 0.
        threshold: Significance threshold (mean + significance_scale * SD of shuffled distribution) used for SCE
            detection.
        sampling_rate: Sampling rate in Hz used for the analysis.
        timestamps: Timestamps in minutes for each sample with length sample_count.
    """

    period_type: PeriodType
    """Identifies this result as belonging to a rest or run period."""
    onset_matrix: NDArray[np.bool_]
    """Binary matrix of transient onsets with dimensions (cell_count, sample_count)."""
    smoothed_fluorescence: NDArray[np.float32]
    """Smoothed fluorescence traces with dimensions (cell_count, sample_count)."""
    coactive_counts: NDArray[np.int32]
    """Number of co-active cells per sample with length sample_count."""
    sce_mask: NDArray[np.bool_]
    """Boolean mask for SCE samples with length sample_count."""
    sce_labels: NDArray[np.int32]
    """Integer SCE event labels (1-indexed) with length sample_count."""
    threshold: float
    """Significance threshold used for SCE detection."""
    sampling_rate: float
    """Sampling rate in Hz."""
    timestamps: NDArray[np.float32]
    """Timestamps in minutes for each sample with length sample_count."""


@njit(cache=True, parallel=True)
def _enforce_minimum_interval(
    above_threshold: NDArray[np.bool_],
    minimum_inter_event_samples: int,
) -> NDArray[np.bool_]:
    """Suppresses threshold crossings that fall within the refractory period of a preceding onset so that each
    accepted onset represents a distinct calcium transient rather than repeated crossings from the same event.

    Args:
        above_threshold: Binary matrix where True indicates the trace exceeds the adaptive threshold, with dimensions
            (cell_count, sample_count).
        minimum_inter_event_samples: Refractory period in samples after an accepted onset during which subsequent
            crossings are suppressed.

    Returns:
        Binary onset matrix with dimensions (cell_count, sample_count) where consecutive onsets are separated by at
        least minimum_inter_event_samples.
    """
    cell_count = above_threshold.shape[0]
    sample_count = above_threshold.shape[1]
    # noinspection PyTypeChecker
    onsets: NDArray[np.bool_] = np.zeros((cell_count, sample_count), dtype=np.bool_)

    for cell_index in prange(cell_count):
        # Offsets the last onset beyond the refractory period so the first threshold crossing is always accepted.
        last_onset_sample = -minimum_inter_event_samples - 1

        # Accepts each crossing only if enough samples have elapsed since the last accepted onset.
        for sample_index in range(sample_count):
            if (
                above_threshold[cell_index, sample_index]
                and (sample_index - last_onset_sample) >= minimum_inter_event_samples
            ):
                onsets[cell_index, sample_index] = True
                last_onset_sample = sample_index

    return onsets


@njit(cache=True)
def _label_contiguous_regions(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Assigns sequential integer labels to contiguous True regions in a boolean mask.

    Args:
        mask: Boolean mask with length sample_count.

    Returns:
        Integer label array with length sample_count, where each contiguous True region receives a unique label
        starting from 1.
    """
    # noinspection PyTypeChecker
    labels: NDArray[np.int32] = np.zeros(len(mask), dtype=np.int32)
    current_label = 0

    for index in range(len(mask)):
        if mask[index]:
            if index == 0 or not mask[index - 1]:
                current_label += 1
            labels[index] = current_label

    return labels


@njit(cache=True, parallel=True)
def _compute_shuffled_max_counts(
    onset_positions: NDArray[np.int32],
    onset_offsets: NDArray[np.int32],
    cell_count: int,
    sample_count: int,
    half_window: int,
    shift_amounts: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Computes the peak co-active cell count for each temporal shuffle iteration using sparse onset positions.

    Args:
        onset_positions: Flat array of onset sample indices for all cells, ordered by cell.
        onset_offsets: Array of length cell_count + 1 where onset_offsets[i]:onset_offsets[i+1] indexes into
            onset_positions for cell i.
        cell_count: Number of cells.
        sample_count: Number of samples in the trace.
        half_window: Half-width of the co-activation sliding window in samples.
        shift_amounts: Pre-generated random shift amounts with dimensions (shuffle_count, cell_count).

    Returns:
        Array of peak co-active counts with length shuffle_count.
    """
    shuffle_count = shift_amounts.shape[0]
    # noinspection PyTypeChecker
    max_counts: NDArray[np.float32] = np.empty(shuffle_count, dtype=np.float32)

    for shuffle_index in prange(shuffle_count):
        # Builds a difference array from sparse onset positions so that a prefix sum recovers the co-active count.
        # noinspection PyTypeChecker
        diff: NDArray[np.int32] = np.zeros(sample_count + 1, dtype=np.int32)

        # Applies a circular shift to each cell's onsets and marks the affected window in the difference array.
        for cell_index in range(cell_count):
            onset_start = onset_offsets[cell_index]
            onset_end = onset_offsets[cell_index + 1]
            shift = shift_amounts[shuffle_index, cell_index]

            for onset_index in range(onset_start, onset_end):
                shifted_sample = (onset_positions[onset_index] + shift) % sample_count
                win_start = max(0, shifted_sample - half_window)
                win_end = min(sample_count - 1, shifted_sample + half_window)
                diff[win_start] += 1
                diff[win_end + 1] -= 1

        # Recovers the co-active count via prefix sum and tracks the peak across all samples.
        running = 0
        peak = 0
        for sample_index in range(sample_count):
            running += diff[sample_index]
            peak = max(peak, running)
        max_counts[shuffle_index] = peak

    return max_counts


def _detect_transient_onsets(
    smoothed: NDArray[np.float32],
    derivative_threshold_scale: float,
    minimum_inter_event_samples: int,
) -> NDArray[np.bool_]:
    """Detects calcium transient onsets as samples where the first derivative of the smoothed trace exceeds a per-cell
    threshold defined as mean + derivative_threshold_scale * standard deviation.

    Args:
        smoothed: Filtered fluorescence with dimensions (cell_count, sample_count).
        derivative_threshold_scale: Number of standard deviations above the mean derivative for the threshold.
        minimum_inter_event_samples: Minimum number of samples between consecutive transient onsets for the same cell.

    Returns:
        Binary onset matrix with dimensions (cell_count, sample_count).
    """
    # Computes the first derivative and pads to preserve the original sample count.
    # noinspection PyTypeChecker
    derivative: NDArray[np.float32] = np.diff(smoothed, axis=1)
    # noinspection PyTypeChecker
    derivative = np.concatenate([np.zeros((smoothed.shape[0], 1), dtype=smoothed.dtype), derivative], axis=1)

    # Thresholds each cell's derivative at mean + scale * std.
    cell_mean = np.mean(derivative, axis=1, keepdims=True)
    cell_std = np.std(derivative, axis=1, keepdims=True)
    # noinspection PyTypeChecker
    above_threshold: NDArray[np.bool_] = derivative > (cell_mean + derivative_threshold_scale * cell_std)

    # Suppresses repeated crossings within the refractory period.
    return _enforce_minimum_interval(
        above_threshold=above_threshold,
        minimum_inter_event_samples=minimum_inter_event_samples,
    )


def _count_coactive_cells(
    onsets: NDArray[np.bool_],
    window_samples: int,
) -> NDArray[np.int32]:
    """Counts the number of cells with at least one transient onset within a sliding window at each sample.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, sample_count).
        window_samples: Width of the sliding window in samples.

    Returns:
        Array of co-active cell counts with length sample_count.
    """
    effective_window = 2 * (window_samples // 2) + 1
    # noinspection PyTypeChecker
    has_onset_in_window: NDArray[np.bool_] = (
        maximum_filter1d(input=onsets.view(np.uint8), size=effective_window, axis=1) > 0
    )
    # np.sum overloads return scalar | ndarray; the axis=0 path always yields an array, so the cast is safe.
    # noinspection PyTypeChecker
    coactive_counts: NDArray[np.int32] = np.asarray(np.sum(has_onset_in_window, axis=0, dtype=np.int32), dtype=np.int32)
    return coactive_counts


def _compute_shuffled_threshold(
    onsets: NDArray[np.bool_],
    window_samples: int,
    shuffle_count: int,
    significance_scale: float,
) -> float:
    """Computes the SCE significance threshold by circularly shifting each cell's onset trace by a random amount per
    shuffle iteration and recording the peak co-active count.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, sample_count).
        window_samples: Width of the co-activation sliding window in samples.
        shuffle_count: Number of shuffle iterations.
        significance_scale: Number of standard deviations above the shuffled mean for the threshold.

    Returns:
        The significance threshold for SCE detection.
    """
    cell_count, sample_count = onsets.shape
    half_window = window_samples // 2

    # Packs per-cell onset sample indices into a flat array with offsets for sparse iteration.
    onset_lists: list[NDArray[np.int32]] = []
    # noinspection PyTypeChecker
    onset_offsets: NDArray[np.int32] = np.zeros(cell_count + 1, dtype=np.int32)

    for cell_index in range(cell_count):
        # noinspection PyTypeChecker
        cell_onsets: NDArray[np.int32] = np.nonzero(onsets[cell_index])[0].astype(np.int32)
        onset_lists.append(cell_onsets)
        onset_offsets[cell_index + 1] = onset_offsets[cell_index] + len(cell_onsets)
    # noinspection PyTypeChecker
    onset_positions: NDArray[np.int32] = (
        np.concatenate(onset_lists).astype(np.int32) if onset_lists else np.empty(0, dtype=np.int32)
    )

    rng = np.random.default_rng(seed=42)
    # noinspection PyTypeChecker
    shift_amounts: NDArray[np.int32] = rng.integers(low=1, high=sample_count, size=(shuffle_count, cell_count)).astype(
        np.int32
    )

    shuffled_max_counts = _compute_shuffled_max_counts(
        onset_positions=onset_positions,
        onset_offsets=onset_offsets,
        cell_count=cell_count,
        sample_count=sample_count,
        half_window=half_window,
        shift_amounts=shift_amounts,
    )

    # Derives the threshold from the shuffled null distribution.
    return float(np.mean(shuffled_max_counts) + significance_scale * np.std(shuffled_max_counts))


def _detect_sces(
    fluorescence: NDArray[np.float32],
    sampling_rate: float,
    timestamps: NDArray[np.float32],
    configuration: SCEDetectionConfiguration,
    period_type: PeriodType,
) -> SCEResult:
    """Runs the full SCE detection pipeline on fluorescence data for a single analysis period.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, sample_count).
        sampling_rate: Sampling rate in Hz.
        timestamps: Timestamps in minutes for each sample with length sample_count.
        configuration: SCE detection parameters.
        period_type: Identifies this period as rest or run.

    Returns:
        An SCEResult containing the detected SCEs, onset matrix, and co-active counts.
    """
    # Converts time-based configuration parameters to sample counts.
    smoothing_window_samples = int(configuration.smoothing_window_seconds * sampling_rate)
    if smoothing_window_samples % 2 == 0:
        smoothing_window_samples += 1
    smoothing_window_samples = max(smoothing_window_samples, configuration.smoothing_order + 2)

    minimum_inter_event_samples = int(configuration.minimum_inter_event_seconds * sampling_rate)
    coactivation_window_samples = max(1, int(configuration.coactivation_window_seconds * sampling_rate))

    # Applies polynomial smoothing to each cell's fluorescence trace.
    smoothed = savgol_filter(
        x=fluorescence,
        window_length=smoothing_window_samples,
        polyorder=configuration.smoothing_order,
        axis=1,
    )

    # Detects calcium transient onsets from the first derivative of the smoothed trace.
    onsets = _detect_transient_onsets(
        smoothed=smoothed,
        derivative_threshold_scale=configuration.derivative_threshold_scale,
        minimum_inter_event_samples=minimum_inter_event_samples,
    )

    # Counts co-active cells per sample using the sliding window.
    coactive_counts = _count_coactive_cells(onsets=onsets, window_samples=coactivation_window_samples)

    # Computes the significance threshold from temporal shuffling.
    threshold = _compute_shuffled_threshold(
        onsets=onsets,
        window_samples=coactivation_window_samples,
        shuffle_count=configuration.shuffle_count,
        significance_scale=configuration.significance_scale,
    )

    # Marks samples where the co-active count exceeds both the shuffled threshold and the minimum cell count.
    # noinspection PyTypeChecker
    sce_mask: NDArray[np.bool_] = (coactive_counts > threshold) & (coactive_counts >= configuration.minimum_cell_count)

    # Assigns sequential labels to contiguous SCE regions.
    sce_labels = _label_contiguous_regions(mask=sce_mask)

    return SCEResult(
        period_type=period_type,
        onset_matrix=onsets,
        smoothed_fluorescence=smoothed,
        coactive_counts=coactive_counts,
        sce_mask=sce_mask,
        sce_labels=sce_labels,
        threshold=threshold,
        sampling_rate=sampling_rate,
        timestamps=timestamps,
    )


def _identify_stable_rest_samples(
    torque: NDArray[np.float32],
    sampling_rate: float,
    stability_window_seconds: float,
    stability_threshold: float,
) -> NDArray[np.bool_]:
    """Identifies rest samples with stable torque baseline by computing a rolling standard deviation.

    Notes:
        Samples where the rolling standard deviation of the torque signal exceeds the stability threshold are excluded
        from rest-period analysis. This removes periods where the animal is fidgeting or producing inconsistent torque
        outputs that could introduce noise into SCE detection.

    Args:
        torque: Torque signal in N*cm with length sample_count.
        sampling_rate: Sampling rate in Hz.
        stability_window_seconds: Window length in seconds for computing rolling standard deviation.
        stability_threshold: Maximum allowable rolling standard deviation for a sample to be considered stable.

    Returns:
        Boolean mask with length sample_count, True for samples with stable torque.
    """
    window_samples = max(1, int(stability_window_seconds * sampling_rate))

    # Computes rolling standard deviation from the rolling mean and mean of squares.
    rolling_mean = uniform_filter1d(input=torque, size=window_samples, mode="nearest")
    rolling_mean_sq = uniform_filter1d(input=torque**2, size=window_samples, mode="nearest")
    rolling_variance = rolling_mean_sq - rolling_mean**2

    # Clamps negative values caused by floating-point precision before taking the square root.
    rolling_std = np.sqrt(np.maximum(rolling_variance, 0.0))

    return rolling_std <= stability_threshold


class SCEDetector:
    """Detects Synchronous Calcium Events (SCEs) separately in rest and run periods.

    Separates the session into rest and run epochs, applies torque-based stability filtering for rest periods, and
    masks place field activity at the animal's current position during run periods.

    Args:
        session_path: Path to the session's dataset directory containing the data feather.
        track_length: Length of the track in centimeters, used to convert distance to position for place field masking.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to read from the data
            feather.
        place_fields: Detected place fields from PlaceFieldDetector, used to mask place field activity at the animal's
            current position during run periods. If None, no masking is applied during run.
        configuration: SCE detection parameters. Uses defaults if None.
    """

    def __init__(
        self,
        session_path: Path,
        track_length: float,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        place_fields: PlaceFields | None = None,
        configuration: SCEDetectionConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, torque, distance, and system state data from the session's data feather for SCE
        detection.

        Args:
            session_path: Path to the session's dataset directory containing the data feather.
            track_length: Length of the track in centimeters.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            place_fields: Detected place fields for run-period masking. If None, no masking is applied.
            configuration: SCE detection parameters. Uses defaults if None.
        """
        df = pl.read_ipc(
            source=session_path.joinpath(DatasetFiles.DATA),
            columns=[
                DatasetColumn.SYSTEM_STATE.value,
                DatasetColumn.TIME_US.value,
                fluorescence_column.value,
                DatasetColumn.TORQUE_N_CM.value,
                DatasetColumn.DISTANCE_CM.value,
                DatasetColumn.TRIAL.value,
            ],
            memory_map=True,
        )
        # Drops the acquisition warmup window before extracting any per-sample arrays so SCE detection runs on
        # stabilized samples without needing its own warmup-aware logic.
        df = trim_acquisition_warmup(df)

        # Estimates the sampling rate from the median inter-sample interval.
        time_us = df[DatasetColumn.TIME_US.value].to_numpy()
        median_interval_us = np.median(np.diff(time_us))
        self._sampling_rate: float = 1_000_000.0 / float(median_interval_us)

        self._system_state = df[DatasetColumn.SYSTEM_STATE.value].to_list()
        self._elapsed_minutes = (time_us - time_us[0]).astype(np.float32) / np.float32(60_000_000.0)

        # Extracts fluorescence data and transposes from (sample, cell) to (cell, sample).
        # noinspection PyTypeChecker
        self._fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T
        self._torque = df[DatasetColumn.TORQUE_N_CM.value].to_numpy()

        # Precomputes the within-trial position once so per-period slicing matches the rest of the analysis
        # surface. Replaces the prior modulo-based wrap that drifted with per-lap encoder noise.
        distance = df[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32)
        trial_ids = df[DatasetColumn.TRIAL.value].to_numpy().astype(np.int32)
        self._position = compute_within_trial_position(
            distance=distance, trial_ids=trial_ids, track_length=track_length
        )

        self._track_length: float = track_length
        self._place_fields: PlaceFields | None = place_fields
        self._configuration: SCEDetectionConfiguration = (
            configuration if configuration is not None else SCEDetectionConfiguration()
        )
        self._results: list[SCEResult] = []

    @property
    def rest_results(self) -> list[SCEResult]:
        """Returns the subset of results belonging to rest periods in temporal order."""
        return [r for r in self._results if r.period_type == PeriodType.REST]

    @property
    def run_results(self) -> list[SCEResult]:
        """Returns the subset of results belonging to run periods in temporal order."""
        return [r for r in self._results if r.period_type == PeriodType.RUN]

    @property
    def results(self) -> list[SCEResult]:
        """Returns every detected SCE result in temporal order, with rest and run periods interleaved."""
        return list(self._results)

    @property
    def sampling_rate_hz(self) -> float:
        """Returns the sampling rate estimated from the median inter-sample interval at construction time."""
        return self._sampling_rate

    def detect_events(self, *, progress: bool = True) -> list[SCEResult]:
        """Detects SCEs separately in rest and run periods across the session.

        Notes:
            Segments the session into alternating rest and run periods, applies torque stability filtering for rest and
            place field exclusion for run, then runs the SCE detection pipeline on each period independently.

        Args:
            progress: Displays a tqdm progress bar tracking period completion when True.

        Returns:
            A list of SCEResult objects in temporal session order, each tagged with its PeriodType.
        """
        # Segments the session into contiguous rest and run periods and prepares fluorescence data for each.
        pending = []
        current_state = None
        period_start = 0

        for sample_index in range(len(self._system_state) + 1):
            state = self._system_state[sample_index] if sample_index < len(self._system_state) else None

            if state != current_state:
                if current_state in (PeriodType.REST, PeriodType.RUN) and (sample_index - period_start) > 0:
                    period_fluorescence = self._fluorescence[:, period_start:sample_index]
                    period_timestamps = self._elapsed_minutes[period_start:sample_index]

                    if current_state == PeriodType.REST:
                        period_torque = self._torque[period_start:sample_index]
                        stable_mask = _identify_stable_rest_samples(
                            torque=period_torque,
                            sampling_rate=self._sampling_rate,
                            stability_window_seconds=self._configuration.torque_stability_window_seconds,
                            stability_threshold=self._configuration.torque_stability_threshold,
                        )

                        stable_count = int(np.sum(stable_mask))
                        stable_fraction = stable_count / len(stable_mask)

                        # Skips rest periods where fewer than half the samples have stable torque.
                        if stable_fraction > _MINIMUM_STABLE_FRACTION and stable_count > _MINIMUM_STABLE_SAMPLE_COUNT:
                            pending.append(
                                (
                                    period_fluorescence[:, stable_mask],
                                    period_timestamps[stable_mask],
                                    PeriodType.REST,
                                )
                            )

                    elif current_state == PeriodType.RUN:
                        # noinspection PyTypeChecker
                        run_fluorescence: NDArray[np.float32] = period_fluorescence.copy()

                        # Masks place field activity at the animal's current position for each sample.
                        if self._place_fields is not None:
                            period_position = self._position[period_start:sample_index]
                            # noinspection PyTypeChecker
                            valid_position: NDArray[np.bool_] = ~np.isnan(period_position)

                            if valid_position.any():
                                bin_size = self._place_fields.bin_size
                                bin_count = self._place_fields.label_image.shape[1]
                                # noinspection PyTypeChecker
                                bin_edges: NDArray[np.float32] = np.arange(
                                    0,
                                    self._track_length + bin_size,
                                    bin_size,
                                    dtype=np.float32,
                                )
                                # Substitutes 0 for NaN positions so searchsorted does not push them to the last bin;
                                # the corresponding mask columns are reset to False below.
                                # noinspection PyTypeChecker
                                safe_position: NDArray[np.float32] = np.where(
                                    valid_position, period_position, np.float32(0.0)
                                )
                                position_bins = np.clip(
                                    np.searchsorted(bin_edges, safe_position, side="right") - 1,
                                    0,
                                    bin_count - 1,
                                ).astype(np.int32)

                                # Builds a (cell_count, sample_count) mask from the label image indexed by each
                                # sample's position bin, then zeros out all masked entries in a single vectorized
                                # operation. Samples belonging to incomplete trials carry no place-field gating.
                                # noinspection PyTypeChecker
                                place_field_mask: NDArray[np.bool_] = (
                                    self._place_fields.label_image[:, position_bins] > 0
                                )
                                place_field_mask[:, ~valid_position] = False
                                run_fluorescence[place_field_mask] = 0.0

                        pending.append((run_fluorescence, period_timestamps, PeriodType.RUN))

                current_state = state
                period_start = sample_index

        # Runs SCE detection sequentially so each period's numba kernel gets full CPU access.
        periods = tqdm(pending, desc="SCE detection", unit="period") if progress else pending
        self._results = [
            _detect_sces(
                fluorescence=fluorescence,
                sampling_rate=self._sampling_rate,
                timestamps=timestamps,
                configuration=self._configuration,
                period_type=period_type,
            )
            for fluorescence, timestamps, period_type in periods
        ]

        return self._results
