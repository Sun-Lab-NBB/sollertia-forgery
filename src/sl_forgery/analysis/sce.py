"""Provides functionality for detecting and visualizing Synchronous Calcium Events (SCEs) in neural recordings."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sl_forgery.analysis.place_cell_analysis import PlaceFields


_MINIMUM_STABLE_FRACTION: float = 0.5
"""Minimum fraction of stable torque frames required for a rest period to be included in SCE analysis."""
_MINIMUM_STABLE_FRAME_COUNT: int = 10
"""Minimum number of stable torque frames required for a rest period to be included in SCE analysis."""


class PeriodType(IntEnum):
    """Defines the analysis period types for SCE detection."""

    REST = 0
    """Indicates a rest period where the animal is stationary."""
    RUN = 1
    """Indicates a run period where the animal is actively locomoting."""


@dataclass(frozen=True)
class SCEDetectionConfiguration:
    """Defines configuration parameters for synchronous calcium event detection."""

    smoothing_window_seconds: float = 0.5
    """Window length in seconds for the smoothing filter applied to each cell's fluorescence trace."""
    smoothing_order: int = 3
    """Polynomial order for the smoothing filter."""
    threshold_window_seconds: float = 2.0
    """Half-width of the sliding window in seconds used to compute the adaptive calcium transient threshold."""
    threshold_scale: float = 3.0
    """Number of IQR units above the median for the adaptive transient detection threshold."""
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
    """Maximum allowable rolling standard deviation of torque (in N*cm) for a rest frame to be considered stable."""


@dataclass
class SCEResult:
    """Stores the results of SCE detection for a single analysis period.

    Attributes:
        period_type: Identifies this result as belonging to a rest or run period.
        onset_matrix: Binary matrix of calcium transient onsets with dimensions (cell_count, frame_count). Each entry
            is True if the corresponding cell has a transient onset at that frame.
        smoothed_fluorescence: Smoothed fluorescence traces with dimensions (cell_count, frame_count), used for plotting
            continuous activity traces.
        coactive_counts: Number of co-active cells at each frame with length frame_count.
        sce_mask: Boolean mask indicating frames that belong to a detected SCE with length frame_count.
        sce_labels: Integer labels assigning each SCE frame to an SCE event index (1-indexed) with length frame_count.
            Frames outside SCEs have label 0.
        threshold: Significance threshold (mean + significance_scale * SD of shuffled distribution) used for SCE
            detection.
        frame_rate: Sampling rate in Hz used for the analysis.
        timestamps: Timestamps in minutes for each frame with length frame_count.
    """

    period_type: PeriodType
    """Identifies this result as belonging to a rest or run period."""
    onset_matrix: NDArray[np.bool_]
    """Binary matrix of transient onsets with dimensions (cell_count, frame_count)."""
    smoothed_fluorescence: NDArray[np.float32]
    """Smoothed fluorescence traces with dimensions (cell_count, frame_count)."""
    coactive_counts: NDArray[np.int32]
    """Number of co-active cells per frame with length frame_count."""
    sce_mask: NDArray[np.bool_]
    """Boolean mask for SCE frames with length frame_count."""
    sce_labels: NDArray[np.int32]
    """Integer SCE event labels (1-indexed) with length frame_count."""
    threshold: float
    """Significance threshold used for SCE detection."""
    frame_rate: float
    """Sampling rate in Hz."""
    timestamps: NDArray[np.float32]
    """Timestamps in minutes for each frame with length frame_count."""


@njit(cache=True, parallel=True)
def _detect_transient_onsets(
    smoothed: NDArray[np.float32],
    threshold_half_width: int,
    threshold_scale: float,
    minimum_inter_event_frames: int,
) -> NDArray[np.bool_]:
    """Detects calcium transient onsets using an adaptive threshold computed from a sliding window.

    Notes:
        For each cell and each frame, computes a threshold as median + threshold_scale * IQR within a window of
        +/- threshold_half_width frames. A transient onset is recorded at frames where the smoothed trace exceeds the
        threshold, subject to the minimum inter-event interval constraint. Cells are processed in parallel via prange.

    Args:
        smoothed: Filtered fluorescence with dimensions (cell_count, frame_count).
        threshold_half_width: Half-width of the sliding window in frames for adaptive threshold computation.
        threshold_scale: Number of IQR units above the median for the threshold.
        minimum_inter_event_frames: Minimum number of frames between consecutive transient onsets for the same cell.

    Returns:
        Binary onset matrix with dimensions (cell_count, frame_count).
    """
    cell_count = smoothed.shape[0]
    frame_count = smoothed.shape[1]
    onsets = np.zeros((cell_count, frame_count), dtype=np.bool_)

    for cell_index in prange(cell_count):
        last_onset_frame = -minimum_inter_event_frames - 1

        for frame_index in range(frame_count):
            # Computes the window boundaries, clamped to the array bounds.
            window_start = max(0, frame_index - threshold_half_width)
            window_end = min(frame_count, frame_index + threshold_half_width + 1)
            window_size = window_end - window_start

            # Copies the window values into a temporary array for percentile computation.
            window_values = np.empty(window_size, dtype=np.float32)
            for window_index in range(window_size):
                window_values[window_index] = smoothed[cell_index, window_start + window_index]

            # Sorts the window values to compute median and quartiles in a single pass.
            window_values.sort()
            median_value = window_values[window_size // 2]
            interquartile_range = window_values[(3 * window_size) // 4] - window_values[window_size // 4]
            adaptive_threshold = median_value + threshold_scale * interquartile_range

            # Marks a transient onset if the trace exceeds the threshold and the minimum inter-event interval has
            # elapsed.
            if smoothed[cell_index, frame_index] > adaptive_threshold:
                if (frame_index - last_onset_frame) >= minimum_inter_event_frames:
                    onsets[cell_index, frame_index] = True
                    last_onset_frame = frame_index

    return onsets


@njit(cache=True, parallel=True)
def _count_coactive_cells(
    onsets: NDArray[np.bool_],
    window_frames: int,
) -> NDArray[np.int32]:
    """Counts the number of cells with transient onsets within a sliding window at each frame.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, frame_count).
        window_frames: Width of the sliding window in frames.

    Returns:
        Array of co-active cell counts with length frame_count.
    """
    cell_count = onsets.shape[0]
    frame_count = onsets.shape[1]
    half_window = window_frames // 2
    counts = np.zeros(frame_count, dtype=np.int32)

    for frame_index in prange(frame_count):
        window_start = max(0, frame_index - half_window)
        window_end = min(frame_count, frame_index + half_window + 1)

        active_count = 0
        for cell_index in range(cell_count):
            # Checks whether the cell has any transient onset within the window.
            has_onset = False
            for window_frame in range(window_start, window_end):
                if onsets[cell_index, window_frame]:
                    has_onset = True
                    break
            if has_onset:
                active_count += 1

        counts[frame_index] = active_count

    return counts


@njit(cache=True)
def _label_contiguous_regions(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Assigns sequential integer labels to contiguous True regions in a boolean mask.

    Args:
        mask: Boolean mask with length frame_count.

    Returns:
        Integer label array with length frame_count, where each contiguous True region receives a unique label
        starting from 1.
    """
    labels = np.zeros(len(mask), dtype=np.int32)
    current_label = 0

    for index in range(len(mask)):
        if mask[index]:
            if index == 0 or not mask[index - 1]:
                current_label += 1
            labels[index] = current_label

    return labels


@njit(cache=True, parallel=True)
def _circular_shift_and_count(
    onsets: NDArray[np.bool_],
    window_frames: int,
    shift_amounts: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Performs temporal shuffles and computes the maximum co-active count for each iteration.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, frame_count).
        window_frames: Width of the co-activation sliding window in frames.
        shift_amounts: Pre-generated random shift amounts with dimensions (shuffle_count, cell_count).

    Returns:
        Array of maximum co-active counts with length shuffle_count.
    """
    cell_count = onsets.shape[0]
    frame_count = onsets.shape[1]
    shuffle_count = shift_amounts.shape[0]
    half_window = window_frames // 2
    shuffled_max_counts = np.empty(shuffle_count, dtype=np.float32)

    for shuffle_index in prange(shuffle_count):
        # Circularly shifts each cell's onset trace by a random amount.
        shuffled_onsets = np.empty_like(onsets)
        for cell_index in range(cell_count):
            shift = shift_amounts[shuffle_index, cell_index]
            for frame_index in range(frame_count):
                source_index = (frame_index - shift) % frame_count
                shuffled_onsets[cell_index, frame_index] = onsets[cell_index, source_index]

        # Computes the maximum co-active count across all frames for this shuffle.
        max_count = 0
        for frame_index in range(frame_count):
            window_start = max(0, frame_index - half_window)
            window_end = min(frame_count, frame_index + half_window + 1)

            active_count = 0
            for cell_index in range(cell_count):
                has_onset = False
                for window_frame in range(window_start, window_end):
                    if shuffled_onsets[cell_index, window_frame]:
                        has_onset = True
                        break
                if has_onset:
                    active_count += 1

            max_count = max(max_count, active_count)

        shuffled_max_counts[shuffle_index] = max_count

    return shuffled_max_counts


def _compute_shuffled_threshold(
    onsets: NDArray[np.bool_],
    window_frames: int,
    shuffle_count: int,
    significance_scale: float,
) -> float:
    """Computes the SCE significance threshold by temporally shuffling cell onset times.

    Notes:
        For each shuffle iteration, circularly shifts each cell's onset trace by a random amount and recomputes the
        co-active cell count. The threshold is set as the mean + significance_scale * SD of the maximum co-active count
        across all shuffles.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, frame_count).
        window_frames: Width of the co-activation sliding window in frames.
        shuffle_count: Number of shuffle iterations.
        significance_scale: Number of standard deviations above the shuffled mean for the threshold.

    Returns:
        The significance threshold for SCE detection.
    """
    cell_count = onsets.shape[0]
    frame_count = onsets.shape[1]

    # Pre-generates all shift amounts.
    rng = np.random.default_rng(seed=42)
    shift_amounts = rng.integers(low=1, high=frame_count, size=(shuffle_count, cell_count)).astype(np.int32)

    shuffled_max_counts = _circular_shift_and_count(
        onsets=onsets,
        window_frames=window_frames,
        shift_amounts=shift_amounts,
    )

    shuffled_mean = np.mean(shuffled_max_counts)
    shuffled_std = np.std(shuffled_max_counts)

    return float(shuffled_mean + significance_scale * shuffled_std)


def _detect_sces(
    fluorescence: NDArray[np.float32],
    frame_rate: float,
    timestamps: NDArray[np.float32],
    configuration: SCEDetectionConfiguration,
    period_type: PeriodType,
) -> SCEResult:
    """Runs the full SCE detection pipeline on fluorescence data for a single analysis period.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        frame_rate: Sampling rate in Hz.
        timestamps: Timestamps in minutes for each frame with length frame_count.
        configuration: SCE detection parameters.
        period_type: Identifies this period as rest or run.

    Returns:
        An SCEResult containing the detected SCEs, onset matrix, and co-active counts.
    """
    # Converts time-based configuration parameters to frame counts.
    smoothing_window_frames = int(configuration.smoothing_window_seconds * frame_rate)
    if smoothing_window_frames % 2 == 0:
        smoothing_window_frames += 1
    smoothing_window_frames = max(smoothing_window_frames, configuration.smoothing_order + 2)

    threshold_half_width = int(configuration.threshold_window_seconds * frame_rate)
    minimum_inter_event_frames = int(configuration.minimum_inter_event_seconds * frame_rate)
    coactivation_window_frames = max(1, int(configuration.coactivation_window_seconds * frame_rate))

    # Applies polynomial smoothing to each cell's fluorescence trace.
    smoothed = savgol_filter(
        x=fluorescence,
        window_length=smoothing_window_frames,
        polyorder=configuration.smoothing_order,
        axis=1,
    ).astype(np.float32)

    # Detects calcium transient onsets using the adaptive threshold method.
    onsets = _detect_transient_onsets(
        smoothed=smoothed,
        threshold_half_width=threshold_half_width,
        threshold_scale=configuration.threshold_scale,
        minimum_inter_event_frames=minimum_inter_event_frames,
    )

    # Counts co-active cells per frame using the sliding window.
    coactive_counts = _count_coactive_cells(onsets=onsets, window_frames=coactivation_window_frames)

    # Computes the significance threshold from temporal shuffling.
    threshold = _compute_shuffled_threshold(
        onsets=onsets,
        window_frames=coactivation_window_frames,
        shuffle_count=configuration.shuffle_count,
        significance_scale=configuration.significance_scale,
    )

    # Marks frames where the co-active count exceeds both the shuffled threshold and the minimum cell count.
    sce_mask = (coactive_counts > threshold) & (coactive_counts >= configuration.minimum_cell_count)

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
        frame_rate=frame_rate,
        timestamps=timestamps,
    )


def _identify_stable_rest_frames(
    torque: NDArray[np.float32],
    frame_rate: float,
    stability_window_seconds: float,
    stability_threshold: float,
) -> NDArray[np.bool_]:
    """Identifies rest frames with stable torque baseline by computing a rolling standard deviation.

    Notes:
        Frames where the rolling standard deviation of the torque signal exceeds the stability threshold are excluded
        from rest-period analysis. This removes periods where the animal is fidgeting or producing inconsistent torque
        outputs that could introduce noise into SCE detection.

    Args:
        torque: Torque signal in N*cm with length frame_count.
        frame_rate: Sampling rate in Hz.
        stability_window_seconds: Window length in seconds for computing rolling standard deviation.
        stability_threshold: Maximum allowable rolling standard deviation for a frame to be considered stable.

    Returns:
        Boolean mask with length frame_count, True for frames with stable torque.
    """
    window_frames = max(1, int(stability_window_seconds * frame_rate))

    # Computes rolling variance as E[x^2] - E[x]^2 using uniform filters, avoiding per-frame std calls.
    rolling_mean = uniform_filter1d(input=torque, size=window_frames, mode="nearest")
    rolling_mean_sq = uniform_filter1d(input=torque**2, size=window_frames, mode="nearest")
    rolling_variance = rolling_mean_sq - rolling_mean**2

    # Clamps negative values caused by floating-point precision before taking the square root.
    rolling_std = np.sqrt(np.maximum(rolling_variance, 0.0))

    return rolling_std <= stability_threshold


class SCEDetector:
    """Detects Synchronous Calcium Events (SCEs) separately in rest and run periods.

    Separates the session into rest and run epochs, applies torque-based stability filtering for rest periods, and
    masks place field activity at the animal's current position during run periods. Provides plotting utilities for
    visualizing the temporal sequence of rest-run-rest periods and SCE raster plots.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters, used to convert distance to position for place field masking.
        fluorescence_column: Name of the fluorescence column to read from the feather file.
        place_fields: Detected place fields from PlaceFieldDetector, used to mask place field activity at the animal's
            current position during run periods. If None, no masking is applied during run.
        configuration: SCE detection parameters. Uses defaults if None.

    Attributes:
        _session_path: Cached path to the session feather file.
        _track_length: Cached track length in centimeters.
        _fluorescence_column: Cached fluorescence column name.
        _place_fields: Cached place fields for run-period masking.
        _configuration: Cached SCE detection configuration.
        _frame_rate: Estimated sampling rate in Hz.
        _results: List of SCEResult objects in temporal session order, each tagged with its PeriodType.
    """

    def __init__(
        self,
        session_path: Path,
        track_length: float,
        fluorescence_column: str = "single_day_dff",
        place_fields: PlaceFields | None = None,
        configuration: SCEDetectionConfiguration | None = None,
    ) -> None:
        self._session_path: Path = session_path
        self._track_length: float = track_length
        self._fluorescence_column: str = fluorescence_column
        self._place_fields: PlaceFields | None = place_fields
        self._configuration: SCEDetectionConfiguration = (
            configuration if configuration is not None else SCEDetectionConfiguration()
        )
        self._frame_rate: float = 0.0
        self._results: list[SCEResult] = []

    @property
    def rest_results(self) -> list[SCEResult]:
        """Returns the subset of results belonging to rest periods in temporal order."""
        return [r for r in self._results if r.period_type == PeriodType.REST]

    @property
    def run_results(self) -> list[SCEResult]:
        """Returns the subset of results belonging to run periods in temporal order."""
        return [r for r in self._results if r.period_type == PeriodType.RUN]

    def detect(self) -> list[SCEResult]:
        """Detects SCEs separately in rest and run periods across the session.

        Notes:
            Loads the session data, estimates the frame rate from timestamps, segments the session into alternating
            rest and run periods, applies torque stability filtering for rest and place field exclusion for run, then
            runs the SCE detection pipeline on each period independently.

        Returns:
            A list of SCEResult objects in temporal session order, each tagged with its PeriodType.
        """
        columns_to_load = [
            "system_state",
            "time_us",
            self._fluorescence_column,
            "torque_N_cm",
            "distance_cm",
        ]
        df = pl.read_ipc(source=self._session_path, columns=columns_to_load)

        # Estimates frame rate from the median inter-frame interval.
        time_us = df["time_us"].to_numpy().astype(np.float64)
        median_interval_seconds = np.median(np.diff(time_us)) / 1_000_000.0
        self._frame_rate = 1.0 / median_interval_seconds

        # Extracts the system state column to identify rest and run periods.
        system_state = df["system_state"].to_list()
        elapsed_minutes = ((time_us - time_us[0]) / 1_000_000.0 / 60.0).astype(np.float32)

        # Extracts fluorescence data and transposes from (frame, cell) to (cell, frame).
        fluorescence_all = np.vstack(df[self._fluorescence_column].to_list()).T.astype(np.float32)
        torque_all = df["torque_N_cm"].to_numpy().astype(np.float32)
        distance_all = df["distance_cm"].to_numpy().astype(np.float32)

        # Segments the session into contiguous rest and run periods.
        self._results = []
        current_state = None
        period_start = 0

        for frame_index in range(len(system_state) + 1):
            state = system_state[frame_index] if frame_index < len(system_state) else None

            if state != current_state:
                if current_state in ("rest", "run") and (frame_index - period_start) > 0:
                    period_fluorescence = fluorescence_all[:, period_start:frame_index]
                    period_timestamps = elapsed_minutes[period_start:frame_index]

                    if current_state == "rest":
                        period_torque = torque_all[period_start:frame_index]
                        stable_mask = _identify_stable_rest_frames(
                            torque=period_torque,
                            frame_rate=self._frame_rate,
                            stability_window_seconds=self._configuration.torque_stability_window_seconds,
                            stability_threshold=self._configuration.torque_stability_threshold,
                        )

                        stable_count = int(np.sum(stable_mask))
                        stable_fraction = stable_count / len(stable_mask)

                        # Skips rest periods where fewer than half the frames have stable torque.
                        if stable_fraction > _MINIMUM_STABLE_FRACTION and stable_count > _MINIMUM_STABLE_FRAME_COUNT:
                            result = _detect_sces(
                                fluorescence=period_fluorescence[:, stable_mask],
                                frame_rate=self._frame_rate,
                                timestamps=period_timestamps[stable_mask],
                                configuration=self._configuration,
                                period_type=PeriodType.REST,
                            )
                            self._results.append(result)

                    elif current_state == "run":
                        run_fluorescence = period_fluorescence.copy()

                        # Masks place field activity at the animal's current position for each frame.
                        if self._place_fields is not None:
                            period_distance = distance_all[period_start:frame_index]
                            period_position = (period_distance % self._track_length).astype(np.float32)

                            bin_size = self._place_fields.bin_size
                            bin_count = self._place_fields.label_image.shape[1]
                            bin_edges = np.arange(
                                0,
                                self._track_length + bin_size,
                                bin_size,
                                dtype=np.float32,
                            )
                            position_bins = np.clip(
                                np.searchsorted(bin_edges, period_position, side="right") - 1,
                                0,
                                bin_count - 1,
                            ).astype(np.int32)

                            # Builds a (cell_count, frame_count) mask from the label image indexed by each frame's
                            # position bin, then zeros out all masked entries in a single vectorized operation.
                            place_field_mask = self._place_fields.label_image[:, position_bins] > 0
                            run_fluorescence[place_field_mask] = 0.0

                        result = _detect_sces(
                            fluorescence=run_fluorescence,
                            frame_rate=self._frame_rate,
                            timestamps=period_timestamps,
                            configuration=self._configuration,
                            period_type=PeriodType.RUN,
                        )
                        self._results.append(result)

                current_state = state
                period_start = frame_index

        return self._results

    def plot_rest_run_rest_sequence(
        self,
        cell_indices: NDArray[np.int32] | list[int] | None = None,
        cell_count: int = 5,
        trial_index: int = 0,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots the rest-run-rest sequence showing calcium activity for selected cells across one trial cycle.

        Notes:
            The top row contains three labeled panels (Rest, Run, Rest) indicating the session phase. Below, each row
            shows the smoothed fluorescence trace for one cell, with detected transient onsets marked as red dots.
            The background is color-coded to distinguish rest (blue) and run (yellow) periods.

        Args:
            cell_indices: Specific cell indices to plot. If None, selects the cells with the most detected transient
                onsets across all periods.
            cell_count: Number of cells to plot when cell_indices is not provided.
            trial_index: Which trial cycle (rest-run-rest group) to display, starting from 0.
            figure_dpi: Resolution of the figure in dots per inch.

        Returns:
            The matplotlib Figure object containing the rest-run-rest sequence plots.
        """
        # Identifies the periods belonging to the requested trial cycle (rest-run-rest pattern).
        rest = self.rest_results
        run = self.run_results

        if trial_index >= len(run):
            trial_index = 0

        # Selects the rest-run-rest sequence: pre-run rest, run, post-run rest.
        sequence_results: list[SCEResult] = []

        if trial_index < len(rest):
            sequence_results.append(rest[trial_index])
        sequence_results.append(run[trial_index])
        if trial_index + 1 < len(rest):
            sequence_results.append(rest[trial_index + 1])

        # Selects cells with the highest transient onset counts if not specified.
        if cell_indices is None:
            total_onsets = np.zeros(sequence_results[0].onset_matrix.shape[0], dtype=np.int32)
            for result in sequence_results:
                total_onsets += np.sum(result.onset_matrix, axis=1).astype(np.int32)
            cell_indices = np.argsort(total_onsets)[-cell_count:][::-1]
        else:
            cell_indices = np.asarray(cell_indices, dtype=np.int32)

        # Computes the concatenated time boundaries for each period.
        period_boundaries = []
        time_offset = 0.0
        for result in sequence_results:
            period_duration = result.timestamps[-1] - result.timestamps[0]
            period_boundaries.append((time_offset, time_offset + period_duration, result.period_type))
            time_offset += period_duration

        total_time = time_offset

        # Creates figure with a label row on top and one trace row per cell.
        height_ratios = [0.4] + [1.0] * len(cell_indices)
        figure, all_axes = plt.subplots(
            nrows=1 + len(cell_indices),
            ncols=1,
            figsize=(14, 1.5 * len(cell_indices) + 1),
            facecolor="white",
            dpi=figure_dpi,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )

        # Draws the period label bar in the top row.
        label_axis = all_axes[0]
        period_colors = {PeriodType.REST: "#A8D8EA", PeriodType.RUN: "#FFE0A0"}
        rest_label_count = 0

        for start, end, period_type in period_boundaries:
            face_color = period_colors[period_type]

            # Distinguishes pre-run and post-run rest labels.
            if period_type == PeriodType.REST:
                rest_label_count += 1
                display_label = f"Rest {rest_label_count}"
            else:
                display_label = "Run"

            label_axis.axvspan(xmin=start, xmax=end, color=face_color, alpha=0.8)
            label_axis.text(
                x=(start + end) / 2,
                y=0.5,
                s=display_label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        label_axis.set_xlim(0, total_time)
        label_axis.set_ylim(0, 1)
        label_axis.set_yticks([])
        label_axis.spines["top"].set_visible(False)
        label_axis.spines["right"].set_visible(False)
        label_axis.spines["left"].set_visible(False)
        label_axis.spines["bottom"].set_visible(False)
        label_axis.set_title(f"Rest-Run-Rest Sequence (Trial {trial_index + 1})", fontsize=11)

        # Draws each cell's fluorescence trace across the concatenated periods.
        trace_axes = all_axes[1:]
        for axis_index, cell_index in enumerate(cell_indices):
            axis = trace_axes[axis_index]
            current_offset = 0.0

            for result in sequence_results:
                period_time = result.timestamps - result.timestamps[0] + current_offset
                fluorescence_trace = result.smoothed_fluorescence[cell_index]

                # Shades the background to match the period label bar.
                background_color = period_colors[result.period_type]
                axis.axvspan(xmin=period_time[0], xmax=period_time[-1], alpha=0.15, color=background_color)

                # Computes the z-score for the trace per-period so that each cell's activity fills its subplot
                # vertically.
                trace_mean = np.mean(fluorescence_trace)
                trace_std = np.std(fluorescence_trace)
                if trace_std > 0:
                    normalized_trace = (fluorescence_trace - trace_mean) / trace_std
                else:
                    normalized_trace = fluorescence_trace - trace_mean

                # Plots the normalized fluorescence trace as a continuous line.
                axis.plot(period_time, normalized_trace, color="black", linewidth=0.5, alpha=0.8)

                current_offset = period_time[-1]

            axis.set_ylabel(f"Cell {cell_index}", fontsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        trace_axes[-1].set_xlabel("Time (minutes)")

        figure.tight_layout()
        return figure

    def plot_sce_raster(
        self,
        period_type: str = "rest",
        period_index: int = 0,
        run_period_index: int = 0,
        window_ms: float = 300.0,
        max_display: int | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots individual SCE events side-by-side as separate raster columns.

        Notes:
            Each detected SCE is displayed as its own panel, with the x-axis showing time relative to SCE onset
            (0 to window_ms) and the y-axis showing cells ordered by their mean activation onset during the
            corresponding run period. Black dots mark transient onsets within the SCE window. This layout matches the
            raster format used to visualize sequential reactivation patterns during rest.

        Args:
            period_type: Which period type to plot, either "rest" or "run".
            period_index: Index of the specific period to plot within the selected period type.
            run_period_index: Index of the run period used to determine cell ordering. Cells are sorted by their
                mean transient onset time during this run period so that sequential run patterns are visible.
            window_ms: Width of each SCE panel in milliseconds.
            max_display: Maximum number of SCEs to display in the plot. If None, all SCEs are shown. The title
                always reports the total number of detected SCEs regardless of this limit.
            figure_dpi: Resolution of the figure in dots per inch.

        Returns:
            The matplotlib Figure object containing the raster plot.
        """
        results = self.rest_results if period_type == "rest" else self.run_results
        result = results[period_index]

        cell_count = result.onset_matrix.shape[0]
        total_sce_count = int(np.max(result.sce_labels))

        if total_sce_count == 0:
            figure, axis = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
            axis.text(0.5, 0.5, "No SCEs detected", ha="center", va="center", fontsize=12)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.axis("off")
            return figure

        # Limits the number of displayed SCEs if requested.
        sce_count = min(total_sce_count, max_display) if max_display is not None else total_sce_count

        # Extracts the onset frames within a fixed window around each SCE onset.
        window_frames = max(1, int(window_ms / 1000.0 * result.frame_rate))

        # Identifies cells that participate in at least one displayed SCE by checking for onsets within any SCE window.
        participating_mask = np.zeros(cell_count, dtype=np.bool_)
        for sce_label in range(1, sce_count + 1):
            sce_frames = np.where(result.sce_labels == sce_label)[0]
            sce_onset_frame = sce_frames[0]
            window_end_frame = min(sce_onset_frame + window_frames, result.onset_matrix.shape[1])
            active_in_window = np.any(result.onset_matrix[:, sce_onset_frame:window_end_frame], axis=1)
            participating_mask |= active_in_window

        participating_count = int(np.sum(participating_mask))

        # Computes cell ordering from the run period, filtered to only participating cells.
        full_sort_order = self._compute_run_sequence_order(run_period_index=run_period_index, cell_count=cell_count)
        sort_order = np.array([idx for idx in full_sort_order if participating_mask[idx]], dtype=np.int32)

        # Uses a fixed column width per SCE so panels are never squished.
        column_width = 1.2
        figure_width = column_width * sce_count + 1.5
        figure_height = max(4, participating_count * 0.025 + 2)

        figure, axes = plt.subplots(
            nrows=1,
            ncols=sce_count,
            figsize=(figure_width, figure_height),
            facecolor="white",
            dpi=figure_dpi,
            sharey=True,
        )

        if sce_count == 1:
            axes = [axes]

        for sce_index in range(sce_count):
            axis = axes[sce_index]
            sce_label = sce_index + 1

            # Finds the first frame of this SCE to use as the time reference.
            sce_frames = np.where(result.sce_labels == sce_label)[0]
            sce_onset_frame = sce_frames[0]
            window_end_frame = min(sce_onset_frame + window_frames, result.onset_matrix.shape[1])

            # Extracts the onset sub-matrix for this SCE window, filtered and reordered by run sequence.
            window_onsets = result.onset_matrix[sort_order, sce_onset_frame:window_end_frame]

            # Converts frame indices to milliseconds relative to SCE onset.
            ms_per_frame = 1000.0 / result.frame_rate

            for cell_row in range(participating_count):
                onset_indices = np.where(window_onsets[cell_row])[0]
                if len(onset_indices) > 0:
                    onset_times_ms = onset_indices * ms_per_frame
                    axis.scatter(
                        x=onset_times_ms,
                        y=np.full(len(onset_indices), cell_row + 1),
                        color="black",
                        s=13,
                        marker=".",
                        linewidths=0,
                    )

            axis.set_xlim(0, window_ms)
            axis.set_ylim(participating_count + 0.5, 0.5)
            axis.set_xticks([])

            # Draws a box around each SCE panel so boundaries between adjacent events are clearly visible.
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("blue")

            if sce_index > 0:
                axis.tick_params(axis="y", labelleft=False)

        # Labels the first panel y-axis with cell numbers visible.
        axes[0].set_ylabel("Cell position in sequence")

        # Adds the corresponding SCE number below each panel.
        for sce_index in range(sce_count):
            axes[sce_index].set_xlabel(f"{sce_index + 1}", fontsize=7, color="red")

        # Reports total detected SCEs in the title even when display is limited.
        display_note = f" (showing first {sce_count})" if sce_count < total_sce_count else ""
        figure.suptitle(
            f"SCE Raster — {period_type.capitalize()} Period {period_index + 1}  "
            f"({total_sce_count} SCEs){display_note}",
            fontsize=11,
        )
        figure.text(0.5, 0.01, "SCE number", ha="center", fontsize=10)

        figure.subplots_adjust(wspace=0.1, left=0.06, right=0.98, top=0.94, bottom=0.06)
        return figure

    def _compute_run_sequence_order(
        self,
        run_period_index: int,
        cell_count: int,
    ) -> NDArray[np.int32]:
        """Computes cell ordering based on mean activation onset time during a run period.

        Notes:
            Cells are sorted by the mean frame index of their first transient onset within each run, so that cells
            that fire early in the run sequence appear at the top of the raster. Cells with no onsets during the run
            period are placed at the bottom.

        Args:
            run_period_index: Index of the run period to use for computing the ordering.
            cell_count: Total number of cells.

        Returns:
            Array of cell indices sorted by their mean run-period activation onset.
        """
        run = self.run_results
        if len(run) == 0 or run_period_index >= len(run):
            return np.arange(cell_count, dtype=np.int32)

        run_result = run[run_period_index]

        # Computes the mean onset frame for each cell using vectorized operations. Multiplies the onset matrix by a
        # frame index array, sums per cell, and divides by onset count to obtain the mean.
        onset_matrix = run_result.onset_matrix
        frame_indices = np.arange(onset_matrix.shape[1], dtype=np.float32)
        onset_counts = np.sum(onset_matrix, axis=1).astype(np.float32)

        # Avoids division by zero for cells with no onsets by setting their count to 1 and replacing with inf after.
        has_onsets = onset_counts > 0
        safe_counts = np.where(has_onsets, onset_counts, 1.0)
        mean_onset_frame = np.sum(onset_matrix * frame_indices[np.newaxis, :], axis=1) / safe_counts
        mean_onset_frame[~has_onsets] = np.inf

        return np.argsort(mean_onset_frame).astype(np.int32)
