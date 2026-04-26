"""Provides functionality for detecting and visualizing Synchronous Calcium Events (SCEs) in neural recordings."""

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
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

from .utilities import compute_canonical_position

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from sollertia_forgery.analysis.place_cell_analysis import PlaceFields


_MINIMUM_STABLE_FRACTION: float = 0.5
"""Minimum fraction of stable torque frames required for a rest period to be included in SCE analysis."""
_MINIMUM_STABLE_FRAME_COUNT: int = 10
"""Minimum number of stable torque frames required for a rest period to be included in SCE analysis."""


class PeriodType(StrEnum):
    """Defines the analysis period types for SCE detection."""

    REST = "rest"
    """Indicates a rest period where the animal is stationary."""
    RUN = "run"
    """Indicates a run period where the animal is actively locomoting."""


@dataclass(frozen=True)
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


@dataclass
class SCEAssembly:
    """Represents a group of cells that frequently co-activate during SCEs.

    Attributes:
        cell_indices: Indices of cells belonging to this assembly.
        activation_sce_indices: 1-indexed SCE labels where this assembly was active.
        activation_count: Number of SCEs where this assembly was active.
    """

    cell_indices: NDArray[np.int32]
    """Indices of cells belonging to this assembly."""
    activation_sce_indices: NDArray[np.int32]
    """SCE labels (1-indexed) where this assembly was active."""
    activation_count: int
    """Number of SCEs where this assembly was active."""


@njit(cache=True, parallel=True)
def _enforce_minimum_interval(
    above_threshold: NDArray[np.bool_],
    minimum_inter_event_frames: int,
) -> NDArray[np.bool_]:
    """Suppresses threshold crossings that fall within the refractory period of a preceding onset so that each
    accepted onset represents a distinct calcium transient rather than repeated crossings from the same event.

    Args:
        above_threshold: Binary matrix where True indicates the trace exceeds the adaptive threshold, with dimensions
            (cell_count, frame_count).
        minimum_inter_event_frames: Refractory period in frames after an accepted onset during which subsequent
            crossings are suppressed.

    Returns:
        Binary onset matrix with dimensions (cell_count, frame_count) where consecutive onsets are separated by at
        least minimum_inter_event_frames.
    """
    cell_count = above_threshold.shape[0]
    frame_count = above_threshold.shape[1]
    onsets = np.zeros((cell_count, frame_count), dtype=np.bool_)

    for cell_index in prange(cell_count):
        # Offsets the last onset beyond the refractory period so the first threshold crossing is always accepted.
        last_onset_frame = -minimum_inter_event_frames - 1

        # Accepts each crossing only if enough frames have elapsed since the last accepted onset.
        for frame_index in range(frame_count):
            if (
                above_threshold[cell_index, frame_index]
                and (frame_index - last_onset_frame) >= minimum_inter_event_frames
            ):
                onsets[cell_index, frame_index] = True
                last_onset_frame = frame_index

    return onsets


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
def _compute_shuffled_max_counts(
    onset_positions: NDArray[np.int32],
    onset_offsets: NDArray[np.int32],
    cell_count: int,
    frame_count: int,
    half_window: int,
    shift_amounts: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Computes the peak co-active cell count for each temporal shuffle iteration using sparse onset positions.

    Args:
        onset_positions: Flat array of onset frame indices for all cells, ordered by cell.
        onset_offsets: Array of length cell_count + 1 where onset_offsets[i]:onset_offsets[i+1] indexes into
            onset_positions for cell i.
        cell_count: Number of cells.
        frame_count: Number of frames in the trace.
        half_window: Half-width of the co-activation sliding window in frames.
        shift_amounts: Pre-generated random shift amounts with dimensions (shuffle_count, cell_count).

    Returns:
        Array of peak co-active counts with length shuffle_count.
    """
    shuffle_count = shift_amounts.shape[0]
    max_counts = np.empty(shuffle_count, dtype=np.float32)

    for shuffle_index in prange(shuffle_count):
        # Builds a difference array from sparse onset positions so that a prefix sum recovers the co-active count.
        diff = np.zeros(frame_count + 1, dtype=np.int32)

        # Applies a circular shift to each cell's onsets and marks the affected window in the difference array.
        for cell_index in range(cell_count):
            onset_start = onset_offsets[cell_index]
            onset_end = onset_offsets[cell_index + 1]
            shift = shift_amounts[shuffle_index, cell_index]

            for onset_index in range(onset_start, onset_end):
                shifted_frame = (onset_positions[onset_index] + shift) % frame_count
                win_start = max(0, shifted_frame - half_window)
                win_end = min(frame_count - 1, shifted_frame + half_window)
                diff[win_start] += 1
                diff[win_end + 1] -= 1

        # Recovers the co-active count via prefix sum and tracks the peak across all frames.
        running = 0
        peak = 0
        for frame_index in range(frame_count):
            running += diff[frame_index]
            peak = max(peak, running)
        max_counts[shuffle_index] = peak

    return max_counts


def _detect_transient_onsets(
    smoothed: NDArray[np.float32],
    derivative_threshold_scale: float,
    minimum_inter_event_frames: int,
) -> NDArray[np.bool_]:
    """Detects calcium transient onsets as frames where the first derivative of the smoothed trace exceeds a per-cell
    threshold defined as mean + derivative_threshold_scale * standard deviation.

    Args:
        smoothed: Filtered fluorescence with dimensions (cell_count, frame_count).
        derivative_threshold_scale: Number of standard deviations above the mean derivative for the threshold.
        minimum_inter_event_frames: Minimum number of frames between consecutive transient onsets for the same cell.

    Returns:
        Binary onset matrix with dimensions (cell_count, frame_count).
    """
    # Computes the first derivative and pads to preserve the original frame count.
    derivative = np.diff(smoothed, axis=1)
    derivative = np.concatenate([np.zeros((smoothed.shape[0], 1), dtype=smoothed.dtype), derivative], axis=1)

    # Thresholds each cell's derivative at mean + scale * std.
    cell_mean = np.mean(derivative, axis=1, keepdims=True)
    cell_std = np.std(derivative, axis=1, keepdims=True)
    above_threshold = derivative > (cell_mean + derivative_threshold_scale * cell_std)

    # Suppresses repeated crossings within the refractory period.
    return _enforce_minimum_interval(
        above_threshold=above_threshold,
        minimum_inter_event_frames=minimum_inter_event_frames,
    )


def _count_coactive_cells(
    onsets: NDArray[np.bool_],
    window_frames: int,
) -> NDArray[np.int32]:
    """Counts the number of cells with at least one transient onset within a sliding window at each frame.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, frame_count).
        window_frames: Width of the sliding window in frames.

    Returns:
        Array of co-active cell counts with length frame_count.
    """
    effective_window = 2 * (window_frames // 2) + 1
    has_onset_in_window = maximum_filter1d(input=onsets.view(np.uint8), size=effective_window, axis=1) > 0
    return np.sum(has_onset_in_window, axis=0, dtype=np.int32)


def _compute_shuffled_threshold(
    onsets: NDArray[np.bool_],
    window_frames: int,
    shuffle_count: int,
    significance_scale: float,
) -> float:
    """Computes the SCE significance threshold by circularly shifting each cell's onset trace by a random amount per
    shuffle iteration and recording the peak co-active count.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, frame_count).
        window_frames: Width of the co-activation sliding window in frames.
        shuffle_count: Number of shuffle iterations.
        significance_scale: Number of standard deviations above the shuffled mean for the threshold.

    Returns:
        The significance threshold for SCE detection.
    """
    cell_count, frame_count = onsets.shape
    half_window = window_frames // 2

    # Packs per-cell onset frame indices into a flat array with offsets for sparse iteration.
    onset_lists: list[NDArray[np.int32]] = []
    onset_offsets = np.zeros(cell_count + 1, dtype=np.int32)

    for cell_index in range(cell_count):
        cell_onsets = np.nonzero(onsets[cell_index])[0]
        onset_lists.append(cell_onsets)
        onset_offsets[cell_index + 1] = onset_offsets[cell_index] + len(cell_onsets)
    onset_positions = np.concatenate(onset_lists).astype(np.int32) if onset_lists else np.empty(0, dtype=np.int32)

    rng = np.random.default_rng(seed=42)
    shift_amounts = rng.integers(low=1, high=frame_count, size=(shuffle_count, cell_count)).astype(np.int32)

    shuffled_max_counts = _compute_shuffled_max_counts(
        onset_positions=onset_positions,
        onset_offsets=onset_offsets,
        cell_count=cell_count,
        frame_count=frame_count,
        half_window=half_window,
        shift_amounts=shift_amounts,
    )

    # Derives the threshold from the shuffled null distribution.
    return float(np.mean(shuffled_max_counts) + significance_scale * np.std(shuffled_max_counts))


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

    minimum_inter_event_frames = int(configuration.minimum_inter_event_seconds * frame_rate)
    coactivation_window_frames = max(1, int(configuration.coactivation_window_seconds * frame_rate))

    # Applies polynomial smoothing to each cell's fluorescence trace.
    smoothed = savgol_filter(
        x=fluorescence,
        window_length=smoothing_window_frames,
        polyorder=configuration.smoothing_order,
        axis=1,
    )

    # Detects calcium transient onsets from the first derivative of the smoothed trace.
    onsets = _detect_transient_onsets(
        smoothed=smoothed,
        derivative_threshold_scale=configuration.derivative_threshold_scale,
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

    # Computes rolling standard deviation from the rolling mean and mean of squares.
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
    """

    def __init__(
        self,
        session_path: Path,
        track_length: float,
        fluorescence_column: str = "single_day_dff",
        place_fields: PlaceFields | None = None,
        configuration: SCEDetectionConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, torque, distance, and system state data from a feather file for SCE detection.

        Args:
            session_path: Path to the session feather file.
            track_length: Length of the track in centimeters.
            fluorescence_column: Name of the fluorescence column to use.
            place_fields: Detected place fields for run-period masking. If None, no masking is applied.
            configuration: SCE detection parameters. Uses defaults if None.
        """
        df = pl.read_ipc(
            source=session_path,
            columns=["system_state", "time_us", fluorescence_column, "torque_N_cm", "distance_cm", "trial"],
            memory_map=True,
        )

        # Estimates the frame rate from the median inter-frame interval.
        time_us = df["time_us"].to_numpy()
        median_interval_us = np.median(np.diff(time_us))
        self._frame_rate: float = 1_000_000.0 / float(median_interval_us)

        self._system_state = df["system_state"].to_list()
        self._elapsed_minutes = (time_us - time_us[0]).astype(np.float32) / np.float32(60_000_000.0)

        # Extracts fluorescence data and transposes from (frame, cell) to (cell, frame).
        self._fluorescence = np.array(df[fluorescence_column].to_list(), dtype=np.float32).T
        self._torque = df["torque_N_cm"].to_numpy()

        # Precomputes the canonical per-trial position once so per-period slicing matches the rest of the analysis
        # surface. Replaces the prior modulo-based wrap that drifted with per-lap encoder noise.
        distance = df["distance_cm"].to_numpy().astype(np.float32)
        trial_ids = df["trial"].to_numpy().astype(np.int32)
        self._position = compute_canonical_position(
            distance=distance, trial_ids=trial_ids, canonical_track_length=track_length
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

    def detect_events(self, progress: bool = True) -> list[SCEResult]:
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

        for frame_index in range(len(self._system_state) + 1):
            state = self._system_state[frame_index] if frame_index < len(self._system_state) else None

            if state != current_state:
                if current_state in (PeriodType.REST, PeriodType.RUN) and (frame_index - period_start) > 0:
                    period_fluorescence = self._fluorescence[:, period_start:frame_index]
                    period_timestamps = self._elapsed_minutes[period_start:frame_index]

                    if current_state == PeriodType.REST:
                        period_torque = self._torque[period_start:frame_index]
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
                            pending.append(
                                (
                                    period_fluorescence[:, stable_mask],
                                    period_timestamps[stable_mask],
                                    PeriodType.REST,
                                )
                            )

                    elif current_state == PeriodType.RUN:
                        run_fluorescence = period_fluorescence.copy()

                        # Masks place field activity at the animal's current position for each frame.
                        if self._place_fields is not None:
                            period_position = self._position[period_start:frame_index]
                            valid_position = ~np.isnan(period_position)

                            if valid_position.any():
                                bin_size = self._place_fields.bin_size
                                bin_count = self._place_fields.label_image.shape[1]
                                bin_edges = np.arange(
                                    0,
                                    self._track_length + bin_size,
                                    bin_size,
                                    dtype=np.float32,
                                )
                                # Substitutes 0 for NaN positions so searchsorted does not push them to the last bin;
                                # the corresponding mask columns are reset to False below.
                                safe_position = np.where(valid_position, period_position, np.float32(0.0))
                                position_bins = np.clip(
                                    np.searchsorted(bin_edges, safe_position, side="right") - 1,
                                    0,
                                    bin_count - 1,
                                ).astype(np.int32)

                                # Builds a (cell_count, frame_count) mask from the label image indexed by each
                                # frame's position bin, then zeros out all masked entries in a single vectorized
                                # operation. Frames belonging to incomplete trials carry no place-field gating.
                                place_field_mask = self._place_fields.label_image[:, position_bins] > 0
                                place_field_mask[:, ~valid_position] = False
                                run_fluorescence[place_field_mask] = 0.0

                        pending.append((run_fluorescence, period_timestamps, PeriodType.RUN))

                current_state = state
                period_start = frame_index

        # Runs SCE detection sequentially so each period's numba kernel gets full CPU access.
        periods = tqdm(pending, desc="SCE detection", unit="period") if progress else pending
        self._results = [
            _detect_sces(
                fluorescence=fluorescence,
                frame_rate=self._frame_rate,
                timestamps=timestamps,
                configuration=self._configuration,
                period_type=period_type,
            )
            for fluorescence, timestamps, period_type in periods
        ]

        return self._results

    def detect_cell_assemblies(
        self,
        period_type: PeriodType = PeriodType.REST,
        period_index: int = 0,
        max_clusters: int = 15,
        activation_threshold: float = 0.3,
        minimum_assembly_size: int = 3,
    ) -> list[SCEAssembly]:
        """Detects cell assemblies from SCE participation patterns using hierarchical clustering.

        Args:
            period_type: Which period type to analyze.
            period_index: Index of the specific period within the selected period type.
            max_clusters: Maximum number of clusters to generate from hierarchical clustering.
            activation_threshold: Minimum fraction of assembly members that must participate in an SCE for that SCE to
                count as an activation of the assembly.
            minimum_assembly_size: Minimum number of cells required for a valid assembly.

        Returns:
            A list of SCEAssembly objects sorted by activation count in descending order.
        """
        results = self.rest_results if period_type == PeriodType.REST else self.run_results
        result = results[period_index]
        total_sce_count = int(np.max(result.sce_labels))

        if total_sce_count < 2:
            return []

        participation = self._build_participation_matrix(result=result)

        # Filters to cells that participate in at least one SCE.
        cell_participation_count = np.sum(participation, axis=0)
        active_cell_mask = cell_participation_count > 0
        active_cell_indices = np.where(active_cell_mask)[0].astype(np.int32)

        if len(active_cell_indices) < minimum_assembly_size:
            return []

        # Computes pairwise Jaccard distance between cells based on SCE participation patterns.
        cell_vectors = participation[:, active_cell_mask].T.astype(np.float64)
        distances = pdist(X=cell_vectors, metric="jaccard")
        distances = np.nan_to_num(distances, nan=0.0)

        # Clusters cells using average-linkage hierarchical clustering.
        n_clusters = min(max_clusters, len(active_cell_indices) // minimum_assembly_size)
        n_clusters = max(2, n_clusters)

        linkage_matrix = linkage(distances, method="average")
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

        # Builds assemblies from clusters that meet the minimum size requirement.
        assemblies: list[SCEAssembly] = []
        for cluster_id in range(1, n_clusters + 1):
            member_mask = cluster_labels == cluster_id
            if int(np.sum(member_mask)) < minimum_assembly_size:
                continue

            member_original_indices = active_cell_indices[member_mask]
            member_participation = participation[:, member_original_indices]

            # Marks SCEs where enough assembly members were co-active as activations.
            active_fraction = np.mean(member_participation, axis=1)
            activating_sces = np.where(active_fraction >= activation_threshold)[0].astype(np.int32)

            if len(activating_sces) == 0:
                continue

            assemblies.append(
                SCEAssembly(
                    cell_indices=member_original_indices,
                    activation_sce_indices=activating_sces + 1,
                    activation_count=len(activating_sces),
                )
            )

        assemblies.sort(key=lambda a: a.activation_count, reverse=True)
        return assemblies

    def plot_rest_run_rest_sequence(
        self,
        cell_indices: NDArray[np.int32] | list[int] | None = None,
        cell_count: int = 5,
        trial_index: int = 0,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots smoothed fluorescence traces, detected onsets, and co-activation counts for selected cells across one
        rest-run-rest trial cycle.

        Args:
            cell_indices: Specific cell indices to plot. If None, selects the cells with the most detected transient
                onsets across all periods.
            cell_count: Number of cells to plot when cell_indices is not provided.
            trial_index: Which trial cycle (rest-run-rest group) to display, starting from 0.
            figure_dpi: Resolution of the figure in dots per inch.

        Returns:
            The matplotlib Figure object containing the rest-run-rest sequence plots.
        """
        # Identifies the periods belonging to the requested trial cycle.
        rest = self.rest_results
        run = self.run_results

        if trial_index >= len(run):
            trial_index = 0

        # Builds the rest-run-rest sequence for the requested trial cycle.
        sequence_results: list[SCEResult] = []

        if trial_index < len(rest):
            sequence_results.append(rest[trial_index])
        sequence_results.append(run[trial_index])
        if trial_index + 1 < len(rest):
            sequence_results.append(rest[trial_index + 1])

        # Chooses a mix of rest-active and rest-quiet cells that are also active during run.
        if cell_indices is None:
            cell_total = sequence_results[0].onset_matrix.shape[0]
            rest_onsets = np.zeros(cell_total, dtype=np.int32)
            run_onsets = np.zeros(cell_total, dtype=np.int32)

            for result in sequence_results:
                counts = np.sum(result.onset_matrix, axis=1).astype(np.int32)
                if result.period_type == PeriodType.REST:
                    rest_onsets += counts
                else:
                    run_onsets += counts

            # Filters cells with at least one transient onset during run to ensure visible activity in traces.
            run_active_mask = run_onsets > 0
            run_active_indices = np.where(run_active_mask)[0]

            if len(run_active_indices) == 0:
                run_active_indices = np.arange(cell_total)

            # Splits run-active cells into the most and least rest-active halves.
            rest_spikers = cell_count // 2
            rest_calm = cell_count - rest_spikers

            rest_onsets_subset = rest_onsets[run_active_indices]
            sorted_by_rest = np.argsort(rest_onsets_subset)

            calm_indices = run_active_indices[sorted_by_rest[:rest_calm]]
            spiker_indices = run_active_indices[sorted_by_rest[-rest_spikers:]]

            cell_indices = np.unique(np.concatenate([calm_indices, spiker_indices])).astype(np.int32)
            cell_indices = cell_indices[:cell_count]

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

    def plot_assemblies(
        self,
        period_type: PeriodType = PeriodType.REST,
        period_index: int = 0,
        top_n: int = 5,
        max_clusters: int = 15,
        activation_threshold: float = 0.3,
        minimum_assembly_size: int = 3,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Detects and plots the most frequent SCE cell assemblies as raster panels.

        Args:
            period_type: Which period type to analyze.
            period_index: Index of the specific period within the selected period type.
            top_n: Maximum number of assemblies to display, selected by highest activation count.
            max_clusters: Maximum number of clusters for hierarchical clustering.
            activation_threshold: Minimum fraction of assembly members that must participate for an SCE to count as an
                activation.
            minimum_assembly_size: Minimum number of cells required for a valid assembly.
            title: Optional title displayed at the top of the figure.
            figure_dpi: Resolution of the figure in dots per inch.

        Returns:
            The matplotlib Figure object containing the assembly raster panels.
        """
        results = self.rest_results if period_type == PeriodType.REST else self.run_results
        result = results[period_index]
        total_sce_count = int(np.max(result.sce_labels))

        assemblies = self.detect_cell_assemblies(
            period_type=period_type,
            period_index=period_index,
            max_clusters=max_clusters,
            activation_threshold=activation_threshold,
            minimum_assembly_size=minimum_assembly_size,
        )

        if len(assemblies) == 0:
            figure, axis = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
            axis.text(0.5, 0.5, "No assemblies detected", ha="center", va="center", fontsize=12)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.axis("off")
            return figure

        display_count = min(top_n, len(assemblies))
        displayed_assemblies = assemblies[:display_count]

        total_cells = result.onset_matrix.shape[0]
        column_width = 1.8
        figure_width = column_width * display_count + 1.5
        figure_height = max(5, min(12, total_cells * 0.003 + 2))

        figure, axes = plt.subplots(
            nrows=1,
            ncols=display_count,
            figsize=(figure_width, figure_height),
            facecolor="white",
            dpi=figure_dpi,
        )

        if display_count == 1:
            axes = [axes]

        # Assigns a distinct color to each assembly for both dots and border.
        assembly_colors = ["black", "red", "blue", "green", "magenta"]

        for assembly_index, assembly in enumerate(displayed_assemblies):
            axis = axes[assembly_index]
            color = assembly_colors[assembly_index % len(assembly_colors)]

            member_cells = np.sort(assembly.cell_indices)

            # Plots dots at actual cell number positions on the Y-axis.
            axis.scatter(
                x=np.zeros(len(member_cells)),
                y=member_cells,
                color=color,
                s=13,
                marker=".",
                linewidths=0,
            )

            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(total_cells - 0.5, -0.5)
            axis.set_xticks([])

            # Places tick marks at every 500 cells for positional reference.
            yticks = list(range(0, total_cells, 500))
            axis.set_yticks(yticks)
            axis.set_yticklabels([str(t) for t in yticks], fontsize=6)

            # Draws a colored box around each assembly panel.
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color(color)

            axis.set_xlabel(
                f"Assembly {assembly_index + 1}",
                fontsize=7,
                color=color,
            )

        if assembly_index > 0:
            axis.tick_params(axis="y", labelleft=False)

        axes[0].set_ylabel("Cell number")

        if title is None:
            title = (
                f"SCE Assemblies — {period_type.title()} Period {period_index + 1}  "
                f"({total_sce_count} total SCEs, showing top {display_count} assemblies)"
            )
        figure.suptitle(title, fontsize=11)
        figure.subplots_adjust(wspace=0.1, left=0.06, right=0.98, top=0.94, bottom=0.06)

        return figure

    def _build_participation_matrix(self, result: SCEResult) -> NDArray[np.bool_]:
        """Builds a binary matrix indicating which cells participated in each SCE.

        Args:
            result: SCE detection result containing the onset matrix and SCE labels.

        Returns:
            Boolean matrix with dimensions (total_sce_count, cell_count), where entry (i, j) is True if cell j had a
            transient onset during SCE i+1.
        """
        total_sce_count = int(np.max(result.sce_labels))
        frame_count = result.onset_matrix.shape[1]

        # Maps each frame to its SCE label via scatter indexing, then uses a matrix multiply to determine which cells
        # had at least one onset during each SCE.
        sce_frame_mask = result.sce_labels > 0
        sce_frame_indices = np.where(sce_frame_mask)[0]

        frame_to_sce = np.zeros((frame_count, total_sce_count), dtype=np.float32)
        frame_to_sce[sce_frame_indices, result.sce_labels[sce_frame_indices] - 1] = 1.0

        return (result.onset_matrix.astype(np.float32) @ frame_to_sce > 0).T
