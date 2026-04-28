"""Provides functionality for detecting Synchronous Calcium Events (SCEs) in neural recordings.

References:
    - Malvache, Reichinnek, Villette, Haimerl & Cossart (2016). Awake hippocampal reactivations project onto
      orthogonal neuronal assemblies. Science. https://doi.org/10.1126/science.aaf3319 -- the canonical SCE
      detection pipeline (~250 ms co-activation window, >=5 cells, shuffle-derived peak-coactive threshold) that
      this module mirrors. The percentile-based threshold default tracks the original report rather than the
      ``mean + k*sigma`` parametric variant.
    - Villette, Malvache, Tressard, Dupuy & Cossart (2015). Internally Recurring Hippocampal Sequences as a
      Population Template of Spatiotemporal Information. Neuron. https://doi.org/10.1016/j.neuron.2015.09.052
      -- per-cell onset-rank-within-SCE motivation; consumed downstream by the rank-correlation analysis in
      :mod:`sollertia_forgery.analysis.cell_tuning.cell_analysis`.
    - Modol, Sousa, Malvache, Tressard et al. (2020). Hippocampal hub neurons maintain distinct connectivity
      throughout their lifetime. Nat Commun. https://doi.org/10.1038/s41467-020-18432-6 -- per-cell
      SCE-recruitment significance test ("super-rich" cells) implemented here as the per-cell participation
      jitter null at fixed SCE event times.
    - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of cells
      identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- percentile-against-shuffle
      idiom adopted across the analysis package, reused here for the SCE peak-coactive threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ...forging import FluorescenceColumn
from ..utilities import trim_acquisition_warmup
from ...shared_assets import DatasetFiles, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MINIMUM_STABLE_FRACTION: float = 0.5
"""Minimum fraction of stable torque samples required for a rest period to be included in SCE analysis."""
_MINIMUM_STABLE_SAMPLE_COUNT: int = 10
"""Minimum number of stable torque samples required for a rest period to be included in SCE analysis."""
_DEFAULT_RNG_SEED: int = 42
"""Seed for the per-period numpy generator used for circular-shift shuffles. Fixed for reproducibility."""


@dataclass(frozen=True, slots=True)
class SCEDetectionConfiguration:
    """Defines configuration parameters for synchronous calcium event detection."""

    smoothing_window_seconds: float = 0.5
    """Window length in seconds for the Savitzky-Golay filter applied to each cell's fluorescence trace."""
    smoothing_order: int = 3
    """Polynomial order for the Savitzky-Golay filter."""
    derivative_threshold_scale: float = 3.0
    """Per-cell derivative threshold expressed as ``mean + scale * std`` of the smoothed-trace first derivative.
    A candidate onset must clear this gate AND the amplitude floor below before it is accepted."""
    derivative_amplitude_floor: float = 2.0
    """Per-cell amplitude floor for accepted onsets, expressed in MAD units of the smoothed dF/F0 baseline.
    Decouples a true rising transient from any noise excursion that happens to have positive slope. Set to 0.0
    to disable the amplitude gate and recover the legacy derivative-only behavior (Malvache 2016)."""
    minimum_inter_event_seconds: float = 0.5
    """Minimum interval in seconds between consecutive calcium transients for the same cell. Default is matched
    to GCaMP6f decay (~0.3-0.5 s); raise for slower indicators such as GCaMP6s."""
    coactivation_window_seconds: float = 0.2
    """Width of the sliding window in seconds used to count co-active cells for SCE detection."""
    shuffle_count: int = 1000
    """Number of temporal shuffles used to generate the null distribution for SCE peak-coactive significance."""
    significance_percentile: float = 99.0
    """Percentile of the shuffled peak-coactive distribution used as the SCE-detection threshold. The
    distribution of the population peak under independent circular shifts is bounded, integer-valued, and
    right-skewed; a percentile cutoff is calibrated and distribution-agnostic, unlike the legacy
    ``mean + k*sigma`` rule (Malvache 2016; Climer & Dombeck 2021)."""
    minimum_shift_seconds: float = 1.0
    """Lower bound in seconds on the absolute circular shift drawn per cell per shuffle. Prevents trivial
    near-identity shifts from contaminating the null. Should exceed the dominant calcium-trace autocorrelation
    timescale."""
    minimum_cell_count: int = 5
    """Minimum number of co-active cells required for a valid SCE."""
    participation_shuffle_count: int = 500
    """Number of per-cell circular shifts used to compute the SCE-recruitment significance jitter null. Lower
    than ``shuffle_count`` because the per-cell test is run cell-by-cell at fixed observed SCE times rather
    than at the population level."""
    participation_significance_percentile: float = 95.0
    """Percentile of the per-cell participation jitter null at which a cell is flagged as an SCE-recruited cell
    ("super-rich" sense of Modol et al. 2020). A cell whose observed participation rate exceeds this percentile
    of its own jitter null is recorded in the per-period ``is_sce_cell`` mask."""
    torque_stability_window_seconds: float = 5.0
    """Window length in seconds for computing the rolling standard deviation of torque during rest-state
    periods. Torque is the canonical stationarity indicator when the animal is in a designated rest state on
    the wheel: zero net force exerted means the animal is not preparing to run."""
    torque_stability_threshold: float = 0.1
    """Maximum allowable rolling standard deviation of torque (in N*cm) for a rest-state sample to be
    considered stationary."""
    encoder_stability_window_seconds: float = 2.0
    """Window length in seconds for computing the rolling standard deviation of wheel speed during non-rest
    states. Set to 2 seconds to match the canonical quiet-wakefulness floor used in awake-replay / immobility
    studies (Foster & Wilson 2006; Diba & Buzsaki 2007; Davidson, Kloosterman & Wilson 2009): >=2 s of
    sustained immobility distinguishes genuine pauses from deceleration phases of ongoing locomotion."""
    encoder_stability_threshold: float = 0.1
    """Maximum allowable rolling standard deviation of wheel speed (in cm/s) for a non-rest sample to be
    considered stationary. Strict ~0 cm/s cutoff -- the animal must be completely still on the wheel for the
    sample to qualify, in line with the SCE / replay literature's "complete immobility" requirement."""


@dataclass(slots=True)
class SCEResult:
    """Stores the results of SCE detection for a single stationary period.

    Attributes:
        period_state: ``DatasetColumn.SYSTEM_STATE`` value identifying the experiment-protocol state this
            stationary block sat inside (e.g. ``"rest"``, ``"run"``, or any custom protocol state).
        onset_matrix: Binary matrix of calcium transient onsets with dimensions (cell_count, sample_count).
        smoothed_fluorescence: Smoothed fluorescence traces with dimensions (cell_count, sample_count).
        coactive_counts: Number of co-active cells at each sample with length sample_count.
        sce_mask: Boolean mask indicating samples that belong to a detected SCE with length sample_count.
        sce_labels: Integer labels assigning each SCE sample to an SCE event index (1-indexed) with length
            sample_count. Samples outside SCEs have label 0.
        threshold: Percentile-derived peak-coactive threshold used for SCE detection.
        sampling_rate: Sampling rate in Hz used for the analysis.
        timestamps: Timestamps in minutes for each sample with length sample_count.
        trial_ids: Per-sample trial identifier with length sample_count; matches the dataset ``trial`` column
            (``-1`` for samples that fall outside any complete trial). Persisted alongside SCEs so reward- and
            position-aligned analyses can locate every event without re-loading the data feather.
        sce_size: Number of distinct cells participating in each detected SCE; length equal to the SCE count.
        sce_width_samples: Duration in samples of each detected SCE; length equal to the SCE count.
        sce_peak_coactive: Peak co-active cell count within each SCE; length equal to the SCE count.
        sce_inter_event_intervals_samples: Sample gap between consecutive SCEs in this period; length equal to
            the SCE count minus one (empty for periods with at most one SCE).
        sce_rate_hz: Detected SCE rate in events per second over the period.
        participation_p_values: Per-cell jitter-null p-value for SCE recruitment with length cell_count;
            ``1.0`` for cells with zero observed participation, ``NaN`` only when the period has no SCEs.
        is_sce_cell: Per-cell boolean indicating whether observed participation exceeds the configured
            participation-significance percentile of its jitter null with length cell_count.
    """

    period_state: str
    """``DatasetColumn.SYSTEM_STATE`` value of the protocol epoch this period was extracted from."""
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
    """Percentile-derived peak-coactive threshold used for SCE detection."""
    sampling_rate: float
    """Sampling rate in Hz."""
    timestamps: NDArray[np.float32]
    """Timestamps in minutes for each sample with length sample_count."""
    trial_ids: NDArray[np.int32]
    """Per-sample trial id with length sample_count (-1 outside any complete trial)."""
    sce_size: NDArray[np.int32]
    """Per-SCE participating-cell count with length sce_count."""
    sce_width_samples: NDArray[np.int32]
    """Per-SCE duration in samples with length sce_count."""
    sce_peak_coactive: NDArray[np.int32]
    """Per-SCE peak co-active count with length sce_count."""
    sce_inter_event_intervals_samples: NDArray[np.int32]
    """Per-SCE sample gap between consecutive events with length max(sce_count - 1, 0)."""
    sce_rate_hz: float
    """Detected SCE rate in events per second."""
    participation_p_values: NDArray[np.float32]
    """Per-cell SCE-recruitment jitter-null p-value with length cell_count."""
    is_sce_cell: NDArray[np.bool_]
    """Per-cell SCE-recruitment significance flag with length cell_count."""


@njit(cache=True, parallel=True)
def _enforce_minimum_interval(
    above_threshold: NDArray[np.bool_],
    minimum_inter_event_samples: int,
) -> NDArray[np.bool_]:
    """Suppresses threshold crossings that fall within the refractory period of a preceding onset so that each
    accepted onset represents a distinct calcium transient rather than repeated crossings from the same event.

    Args:
        above_threshold: Binary matrix where True indicates the trace exceeds the adaptive threshold, with
            dimensions (cell_count, sample_count).
        minimum_inter_event_samples: Refractory period in samples after an accepted onset during which
            subsequent crossings are suppressed.

    Returns:
        Binary onset matrix with dimensions (cell_count, sample_count) where consecutive onsets are separated
        by at least minimum_inter_event_samples.
    """
    cell_count = above_threshold.shape[0]
    sample_count = above_threshold.shape[1]
    # noinspection PyTypeChecker
    onsets: NDArray[np.bool_] = np.zeros((cell_count, sample_count), dtype=np.bool_)

    for cell_index in prange(cell_count):
        last_onset_sample = -minimum_inter_event_samples - 1
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
        Integer label array with length sample_count, where each contiguous True region receives a unique
        label starting from 1.
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
        sample_count: Number of samples in the period.
        half_window: Half-width of the co-activation sliding window in samples.
        shift_amounts: Pre-generated random shift amounts with dimensions (shuffle_count, cell_count).

    Returns:
        Array of peak co-active counts with length shuffle_count.
    """
    shuffle_count = shift_amounts.shape[0]
    # noinspection PyTypeChecker
    max_counts: NDArray[np.float32] = np.empty(shuffle_count, dtype=np.float32)

    for shuffle_index in prange(shuffle_count):
        # noinspection PyTypeChecker
        diff: NDArray[np.int32] = np.zeros(sample_count + 1, dtype=np.int32)

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

        running = 0
        peak = 0
        for sample_index in range(sample_count):
            running += diff[sample_index]
            peak = max(peak, running)
        max_counts[shuffle_index] = peak

    return max_counts


@njit(cache=True, parallel=True)
def _compute_shuffled_participation(
    onset_positions: NDArray[np.int32],
    onset_offsets: NDArray[np.int32],
    sample_count: int,
    sce_window_starts: NDArray[np.int32],
    sce_window_ends: NDArray[np.int32],
    shift_amounts: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Counts each cell's SCE participation under independent circular shifts of its onset trace, holding the
    SCE event-window boundaries fixed at the observed values.

    Args:
        onset_positions: Flat array of onset sample indices for all cells, ordered by cell.
        onset_offsets: Array of length cell_count + 1 indexing onset_positions per cell.
        sample_count: Number of samples in the period.
        sce_window_starts: First sample index (inclusive) of each detected SCE; length sce_count.
        sce_window_ends: Last sample index (inclusive) of each detected SCE; length sce_count.
        shift_amounts: Pre-generated random shifts with dimensions (shuffle_count, cell_count).

    Returns:
        Per-cell shuffled participation counts with dimensions (shuffle_count, cell_count).
    """
    shuffle_count = shift_amounts.shape[0]
    cell_count = onset_offsets.shape[0] - 1
    sce_count = sce_window_starts.shape[0]
    # noinspection PyTypeChecker
    counts: NDArray[np.int32] = np.zeros((shuffle_count, cell_count), dtype=np.int32)

    if sce_count == 0:
        return counts

    for shuffle_index in prange(shuffle_count):
        for cell_index in range(cell_count):
            onset_start = onset_offsets[cell_index]
            onset_end = onset_offsets[cell_index + 1]
            shift = shift_amounts[shuffle_index, cell_index]

            for onset_index in range(onset_start, onset_end):
                shifted_sample = (onset_positions[onset_index] + shift) % sample_count
                # Each shuffled onset contributes at most one participation count per cell across the SCE
                # series; ties (an onset landing inside two adjacent SCE windows after shifting) are impossible
                # because windows are by definition non-overlapping.
                for sce_index in range(sce_count):
                    if sce_window_starts[sce_index] <= shifted_sample <= sce_window_ends[sce_index]:
                        counts[shuffle_index, cell_index] += 1
                        break

    return counts


def _compute_baseline_mad(smoothed: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns a per-cell baseline MAD estimate computed on the lower half of each smoothed trace.

    Notes:
        The lower-half conditioning excludes the largest fluorescence excursions (transient peaks) so the
        resulting MAD reflects baseline noise rather than event amplitude. MAD is multiplied by 1.4826 to
        approximate the standard deviation under a Gaussian baseline so the amplitude floor scales naturally
        with sigma units.

    Args:
        smoothed: Smoothed fluorescence with dimensions (cell_count, sample_count).

    Returns:
        Per-cell baseline scale estimate with length cell_count.
    """
    cell_count = smoothed.shape[0]
    if smoothed.shape[1] == 0:
        # noinspection PyTypeChecker
        return np.zeros(cell_count, dtype=np.float32)
    medians = np.median(smoothed, axis=1, keepdims=True).astype(np.float32, copy=False)
    deviations = np.abs(smoothed - medians).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    median_threshold: NDArray[np.float32] = np.median(deviations, axis=1).astype(np.float32, copy=False)
    return (median_threshold * np.float32(1.4826)).astype(np.float32, copy=False)


def _detect_transient_onsets(
    smoothed: NDArray[np.float32],
    derivative_threshold_scale: float,
    derivative_amplitude_floor: float,
    minimum_inter_event_samples: int,
) -> NDArray[np.bool_]:
    """Detects calcium transient onsets as samples where the first derivative of the smoothed trace exceeds a
    per-cell ``mean + scale * std`` threshold AND the smoothed dF/F0 amplitude exceeds the per-cell baseline
    MAD floor.

    Args:
        smoothed: Filtered fluorescence with dimensions (cell_count, sample_count).
        derivative_threshold_scale: Number of standard deviations above the mean derivative required for a
            candidate onset.
        derivative_amplitude_floor: Per-cell baseline MAD-units floor on the smoothed dF/F0 amplitude required
            for an accepted onset. ``0.0`` disables the gate.
        minimum_inter_event_samples: Minimum number of samples between consecutive accepted onsets per cell.

    Returns:
        Binary onset matrix with dimensions (cell_count, sample_count).
    """
    # noinspection PyTypeChecker
    derivative: NDArray[np.float32] = np.diff(smoothed, axis=1)
    # noinspection PyTypeChecker
    derivative = np.concatenate([np.zeros((smoothed.shape[0], 1), dtype=smoothed.dtype), derivative], axis=1)

    cell_mean = np.mean(derivative, axis=1, keepdims=True)
    cell_std = np.std(derivative, axis=1, keepdims=True)
    # noinspection PyTypeChecker
    derivative_pass: NDArray[np.bool_] = derivative > (cell_mean + derivative_threshold_scale * cell_std)

    if derivative_amplitude_floor > 0.0:
        baseline_mad = _compute_baseline_mad(smoothed=smoothed).reshape(-1, 1)
        baseline_median = np.median(smoothed, axis=1, keepdims=True).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        amplitude_pass: NDArray[np.bool_] = smoothed > (baseline_median + derivative_amplitude_floor * baseline_mad)
        # noinspection PyTypeChecker
        above_threshold: NDArray[np.bool_] = derivative_pass & amplitude_pass
    else:
        above_threshold = derivative_pass

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
    # noinspection PyTypeChecker
    coactive_counts: NDArray[np.int32] = np.asarray(np.sum(has_onset_in_window, axis=0, dtype=np.int32), dtype=np.int32)
    return coactive_counts


def _pack_onsets_for_shuffle(onsets: NDArray[np.bool_]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Packs the dense per-cell onset matrix into (positions, offsets) sparse arrays for the shuffle kernels."""
    cell_count = onsets.shape[0]
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
    return onset_positions, onset_offsets


def _draw_circular_shifts(
    rng: np.random.Generator,
    *,
    sample_count: int,
    cell_count: int,
    shuffle_count: int,
    minimum_shift_samples: int,
) -> NDArray[np.int32]:
    """Draws per-cell circular shifts uniformly from ``[minimum_shift, sample_count - minimum_shift)`` so
    trivially small shifts that barely perturb the trace cannot weaken the null distribution.
    """
    if sample_count <= 1 or shuffle_count == 0 or cell_count == 0:
        # noinspection PyTypeChecker
        return np.zeros((shuffle_count, cell_count), dtype=np.int32)
    floor = max(1, int(minimum_shift_samples))
    ceil = max(floor + 1, sample_count - floor)
    # noinspection PyTypeChecker
    shift_amounts: NDArray[np.int32] = rng.integers(low=floor, high=ceil, size=(shuffle_count, cell_count)).astype(
        np.int32
    )
    return shift_amounts


def _compute_shuffled_threshold(
    onsets: NDArray[np.bool_],
    *,
    window_samples: int,
    shuffle_count: int,
    significance_percentile: float,
    minimum_shift_samples: int,
    rng: np.random.Generator,
) -> float:
    """Computes the SCE significance threshold by circularly shifting each cell's onset trace by a random
    amount per shuffle iteration and reporting the configured percentile of the shuffled peak-coactive
    distribution (Malvache 2016 / Climer & Dombeck 2021 idiom).

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, sample_count).
        window_samples: Width of the co-activation sliding window in samples.
        shuffle_count: Number of shuffle iterations.
        significance_percentile: Percentile cutoff applied to the shuffled peak-coactive distribution.
        minimum_shift_samples: Lower bound on the absolute circular shift drawn per cell per shuffle.
        rng: Per-period numpy random generator. Passing the generator explicitly keeps detection deterministic
            and lets callers reuse the same stream for the per-cell participation jitter null below.

    Returns:
        The SCE peak-coactive threshold used for detection.
    """
    cell_count, sample_count = onsets.shape
    half_window = window_samples // 2

    onset_positions, onset_offsets = _pack_onsets_for_shuffle(onsets=onsets)
    shift_amounts = _draw_circular_shifts(
        rng=rng,
        sample_count=sample_count,
        cell_count=cell_count,
        shuffle_count=shuffle_count,
        minimum_shift_samples=minimum_shift_samples,
    )
    shuffled_max_counts = _compute_shuffled_max_counts(
        onset_positions=onset_positions,
        onset_offsets=onset_offsets,
        cell_count=cell_count,
        sample_count=sample_count,
        half_window=half_window,
        shift_amounts=shift_amounts,
    )
    if shuffled_max_counts.size == 0:
        return 0.0
    return float(np.percentile(shuffled_max_counts, significance_percentile))


def _compute_per_sce_descriptors(
    *,
    onsets: NDArray[np.bool_],
    sce_labels: NDArray[np.int32],
    coactive_counts: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Computes per-SCE size (distinct participating cells), width (samples), peak co-active count, and the
    inter-event interval series.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, sample_count).
        sce_labels: Per-sample SCE labels (1-indexed; 0 outside any SCE) with length sample_count.
        coactive_counts: Per-sample co-active counts with length sample_count.

    Returns:
        ``(sce_size, sce_width_samples, sce_peak_coactive, sce_inter_event_intervals_samples)``.
    """
    sce_count = int(np.max(sce_labels)) if sce_labels.size > 0 else 0
    if sce_count == 0:
        # noinspection PyTypeChecker
        empty_int: NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        return empty_int, empty_int, empty_int, empty_int

    # noinspection PyTypeChecker
    sce_size: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)
    # noinspection PyTypeChecker
    sce_width: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)
    # noinspection PyTypeChecker
    sce_peak: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)
    # noinspection PyTypeChecker
    starts: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)

    for sce_label in range(1, sce_count + 1):
        # noinspection PyTypeChecker
        sample_indices: NDArray[np.int64] = np.where(sce_labels == sce_label)[0]
        starts[sce_label - 1] = int(sample_indices[0])
        sce_width[sce_label - 1] = int(sample_indices.size)
        sce_peak[sce_label - 1] = int(np.max(coactive_counts[sample_indices]))
        # Distinct participating cells = cells with at least one onset somewhere inside the SCE window.
        window = onsets[:, sample_indices]
        # noinspection PyTypeChecker
        any_onset: NDArray[np.bool_] = np.any(window, axis=1)
        sce_size[sce_label - 1] = int(np.sum(any_onset))

    if sce_count > 1:
        # noinspection PyTypeChecker
        intervals: NDArray[np.int32] = np.diff(starts).astype(np.int32, copy=False)
    else:
        # noinspection PyTypeChecker
        intervals = np.zeros(0, dtype=np.int32)

    return sce_size, sce_width, sce_peak, intervals


def _compute_per_cell_participation_significance(
    *,
    onsets: NDArray[np.bool_],
    sce_labels: NDArray[np.int32],
    shuffle_count: int,
    significance_percentile: float,
    minimum_shift_samples: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Computes per-cell SCE-recruitment p-values and significance flags using a fixed-event jitter null.

    Notes:
        Each cell's onsets are circularly shifted independently while the observed SCE event windows are held
        fixed; the cell's shuffled participation count is the number of SCE windows containing at least one
        shifted onset. The p-value is the fraction of shuffles whose count is >= the observed count
        (Modol et al. 2020 super-rich cell logic). Cells with zero observed participation always receive
        ``p == 1.0``. With zero SCEs in the period, the routine returns NaN p-values and an all-False mask.

    Args:
        onsets: Binary onset matrix with dimensions (cell_count, sample_count).
        sce_labels: Per-sample SCE labels.
        shuffle_count: Number of per-cell jitter iterations.
        significance_percentile: Percentile of the per-cell jitter null at which a cell is flagged significant.
        minimum_shift_samples: Lower bound on the absolute circular shift per cell per shuffle.
        rng: Per-period numpy random generator.

    Returns:
        ``(participation_p_values, is_sce_cell)`` arrays each with length cell_count.
    """
    cell_count, sample_count = onsets.shape
    sce_count = int(np.max(sce_labels)) if sce_labels.size > 0 else 0

    if sce_count == 0:
        # noinspection PyTypeChecker
        nan_p: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        flags: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)
        return nan_p, flags

    # noinspection PyTypeChecker
    sce_window_starts: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)
    # noinspection PyTypeChecker
    sce_window_ends: NDArray[np.int32] = np.zeros(sce_count, dtype=np.int32)
    for sce_label in range(1, sce_count + 1):
        # noinspection PyTypeChecker
        samples: NDArray[np.int64] = np.where(sce_labels == sce_label)[0]
        sce_window_starts[sce_label - 1] = int(samples[0])
        sce_window_ends[sce_label - 1] = int(samples[-1])

    # Observed participation per cell against the same fixed event windows.
    # noinspection PyTypeChecker
    observed: NDArray[np.int32] = np.zeros(cell_count, dtype=np.int32)
    for sce_label in range(1, sce_count + 1):
        # noinspection PyTypeChecker
        samples = np.where(sce_labels == sce_label)[0]
        # noinspection PyTypeChecker
        any_onset: NDArray[np.bool_] = np.any(onsets[:, samples], axis=1)
        observed += any_onset.astype(np.int32, copy=False)

    onset_positions, onset_offsets = _pack_onsets_for_shuffle(onsets=onsets)
    shift_amounts = _draw_circular_shifts(
        rng=rng,
        sample_count=sample_count,
        cell_count=cell_count,
        shuffle_count=shuffle_count,
        minimum_shift_samples=minimum_shift_samples,
    )
    if shift_amounts.size == 0:
        # noinspection PyTypeChecker
        nan_p = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        flags = np.zeros(cell_count, dtype=np.bool_)
        return nan_p, flags

    shuffled = _compute_shuffled_participation(
        onset_positions=onset_positions,
        onset_offsets=onset_offsets,
        sample_count=sample_count,
        sce_window_starts=sce_window_starts,
        sce_window_ends=sce_window_ends,
        shift_amounts=shift_amounts,
    )

    # P-value: fraction of shuffles meeting or exceeding the observed count. Use a +1/+1 add-one smoother so
    # cells with very high observed participation still receive a positive lower-bound p-value.
    # noinspection PyTypeChecker
    ge_counts: NDArray[np.int32] = np.sum(shuffled >= observed[np.newaxis, :], axis=0).astype(np.int32)
    # noinspection PyTypeChecker
    p_values: NDArray[np.float32] = ((ge_counts + 1).astype(np.float32) / np.float32(shuffle_count + 1)).astype(
        np.float32, copy=False
    )
    p_values[observed == 0] = np.float32(1.0)

    # Significance: observed > the configured-percentile of the per-cell jitter null. Computed in the rate
    # space (counts / sce_count) to mirror how downstream metrics are reported.
    # noinspection PyTypeChecker
    threshold_counts: NDArray[np.float32] = np.percentile(shuffled, significance_percentile, axis=0).astype(
        np.float32, copy=False
    )
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = observed.astype(np.float32) > threshold_counts

    return p_values, is_significant


def _detect_sces(
    *,
    fluorescence: NDArray[np.float32],
    sampling_rate: float,
    timestamps: NDArray[np.float32],
    trial_ids: NDArray[np.int32],
    configuration: SCEDetectionConfiguration,
    period_state: str,
    rng: np.random.Generator,
) -> SCEResult:
    """Runs the full SCE detection pipeline on fluorescence data for a single stationary period.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, sample_count).
        sampling_rate: Sampling rate in Hz.
        timestamps: Timestamps in minutes for each sample with length sample_count.
        trial_ids: Per-sample trial id with length sample_count (-1 outside any complete trial).
        configuration: SCE detection parameters.
        period_state: System-state label (e.g. ``"rest"``, ``"run"``) preserved on the result.
        rng: Per-period numpy random generator (shared between threshold and per-cell significance shuffles).

    Returns:
        A populated SCEResult.
    """
    smoothing_window_samples = int(configuration.smoothing_window_seconds * sampling_rate)
    if smoothing_window_samples % 2 == 0:
        smoothing_window_samples += 1
    smoothing_window_samples = max(smoothing_window_samples, configuration.smoothing_order + 2)

    minimum_inter_event_samples = int(configuration.minimum_inter_event_seconds * sampling_rate)
    coactivation_window_samples = max(1, int(configuration.coactivation_window_seconds * sampling_rate))
    minimum_shift_samples = max(1, int(configuration.minimum_shift_seconds * sampling_rate))

    smoothed = savgol_filter(
        x=fluorescence,
        window_length=smoothing_window_samples,
        polyorder=configuration.smoothing_order,
        axis=1,
    ).astype(np.float32, copy=False)

    onsets = _detect_transient_onsets(
        smoothed=smoothed,
        derivative_threshold_scale=configuration.derivative_threshold_scale,
        derivative_amplitude_floor=configuration.derivative_amplitude_floor,
        minimum_inter_event_samples=minimum_inter_event_samples,
    )

    coactive_counts = _count_coactive_cells(onsets=onsets, window_samples=coactivation_window_samples)

    threshold = _compute_shuffled_threshold(
        onsets=onsets,
        window_samples=coactivation_window_samples,
        shuffle_count=configuration.shuffle_count,
        significance_percentile=configuration.significance_percentile,
        minimum_shift_samples=minimum_shift_samples,
        rng=rng,
    )

    # noinspection PyTypeChecker
    sce_mask: NDArray[np.bool_] = (coactive_counts > threshold) & (coactive_counts >= configuration.minimum_cell_count)
    sce_labels = _label_contiguous_regions(mask=sce_mask)

    sce_size, sce_width, sce_peak, intervals = _compute_per_sce_descriptors(
        onsets=onsets,
        sce_labels=sce_labels,
        coactive_counts=coactive_counts,
    )
    sce_count = int(sce_size.size)
    duration_seconds = float(timestamps[-1] - timestamps[0]) * 60.0 if timestamps.size > 1 else 0.0
    sce_rate_hz = float(sce_count / duration_seconds) if duration_seconds > 0.0 else 0.0

    participation_p_values, is_sce_cell = _compute_per_cell_participation_significance(
        onsets=onsets,
        sce_labels=sce_labels,
        shuffle_count=configuration.participation_shuffle_count,
        significance_percentile=configuration.participation_significance_percentile,
        minimum_shift_samples=minimum_shift_samples,
        rng=rng,
    )

    return SCEResult(
        period_state=period_state,
        onset_matrix=onsets,
        smoothed_fluorescence=smoothed,
        coactive_counts=coactive_counts,
        sce_mask=sce_mask,
        sce_labels=sce_labels,
        threshold=threshold,
        sampling_rate=sampling_rate,
        timestamps=timestamps,
        trial_ids=trial_ids,
        sce_size=sce_size,
        sce_width_samples=sce_width,
        sce_peak_coactive=sce_peak,
        sce_inter_event_intervals_samples=intervals,
        sce_rate_hz=sce_rate_hz,
        participation_p_values=participation_p_values,
        is_sce_cell=is_sce_cell,
    )


def _identify_stable_rest_samples(
    torque: NDArray[np.float32],
    sampling_rate: float,
    stability_window_seconds: float,
    stability_threshold: float,
) -> NDArray[np.bool_]:
    """Identifies rest samples with stable torque baseline by computing a rolling standard deviation.

    Notes:
        Samples where the rolling standard deviation of the torque signal exceeds the stability threshold are
        excluded from rest-period analysis, removing fidgeting epochs that would inject noise into SCE
        detection.

    Args:
        torque: Torque signal in N*cm with length sample_count.
        sampling_rate: Sampling rate in Hz.
        stability_window_seconds: Window length in seconds for computing rolling standard deviation.
        stability_threshold: Maximum allowable rolling standard deviation for a sample to be considered stable.

    Returns:
        Boolean mask with length sample_count, True for samples with stable torque.
    """
    window_samples = max(1, int(stability_window_seconds * sampling_rate))

    rolling_mean = uniform_filter1d(input=torque, size=window_samples, mode="nearest")
    rolling_mean_sq = uniform_filter1d(input=torque**2, size=window_samples, mode="nearest")
    rolling_variance = rolling_mean_sq - rolling_mean**2
    rolling_std = np.sqrt(np.maximum(rolling_variance, 0.0))

    return rolling_std <= stability_threshold


_REST_STATE: str = "rest"
"""``DatasetColumn.SYSTEM_STATE`` value that uses the torque-based stationarity filter. Every other state value
falls through to the speed/encoder-based filter. Both filters are stationarity gates: SCEs are restricted to
moments when the animal is motionless, regardless of which protocol epoch the moment is in (Malvache 2016
canonical rest replay + Buzsaki two-stage immobility framing)."""


def _identify_stable_run_samples(
    speed: NDArray[np.float32],
    sampling_rate: float,
    stability_window_seconds: float,
    stability_threshold: float,
) -> NDArray[np.bool_]:
    """Identifies non-rest samples where the wheel encoder is stable, marking pauses-within-run.

    Notes:
        Mirrors :func:`_identify_stable_rest_samples` but uses wheel speed as the stationarity indicator. During
        designated run epochs the animal is mostly locomoting, but brief pauses (encoder coasts to zero) are
        windows where SCE-class population synchrony can occur; this mask admits those samples.

    Args:
        speed: Wheel speed in cm/s with length sample_count.
        sampling_rate: Sampling rate in Hz.
        stability_window_seconds: Window length in seconds for the rolling speed standard deviation.
        stability_threshold: Maximum allowable rolling standard deviation of speed for a sample to be considered
            stationary.

    Returns:
        Boolean mask with length sample_count, True for samples where the wheel is stationary.
    """
    window_samples = max(1, int(stability_window_seconds * sampling_rate))
    rolling_mean = uniform_filter1d(input=speed, size=window_samples, mode="nearest")
    rolling_mean_sq = uniform_filter1d(input=speed**2, size=window_samples, mode="nearest")
    rolling_variance = rolling_mean_sq - rolling_mean**2
    rolling_std = np.sqrt(np.maximum(rolling_variance, 0.0))
    return rolling_std <= stability_threshold


class SCEDetector:
    """Detects Synchronous Calcium Events (SCEs) during stationary samples across every protocol epoch.

    Walks every contiguous ``DatasetColumn.SYSTEM_STATE`` block, applies a state-appropriate stationarity gate
    (torque-stability for ``"rest"``-state samples, encoder/speed-stability for every other state), and runs the
    SCE detection pipeline on each surviving stationary chunk. SCEs are by definition a quiet-wakefulness
    phenomenon (Malvache 2016 lineage; Buzsaki two-stage model) so the gate is animal stationarity, not which
    protocol epoch the sample sits in. Pauses-within-run survive and contribute their own SCE periods, tagged
    with the originating ``period_state`` so post-hoc analyses can split events by epoch.

    References:
        - Malvache, Reichinnek, Villette, Haimerl & Cossart (2016). Awake hippocampal reactivations project
          onto orthogonal neuronal assemblies. Science. https://doi.org/10.1126/science.aaf3319 -- canonical
          rest-state SCE detection pipeline (~250 ms window, >=5 cells, percentile-against-shuffle threshold).
        - Modol et al. (2020). Hippocampal hub neurons maintain distinct connectivity throughout their
          lifetime. Nat Commun. https://doi.org/10.1038/s41467-020-18432-6 -- per-cell SCE-recruitment
          significance test against a per-cell jitter null.

    Args:
        session_path: Path to the session's dataset directory containing the data feather.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to read from the
            data feather.
        configuration: SCE detection parameters. Uses defaults if None.
        rng_seed: Seed for the per-period numpy random generator. Fixed for reproducibility.
    """

    def __init__(
        self,
        session_path: Path,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        configuration: SCEDetectionConfiguration | None = None,
        rng_seed: int = _DEFAULT_RNG_SEED,
    ) -> None:
        """Loads fluorescence, torque, speed, trial-id, and system-state data from the session's data feather.

        Args:
            session_path: Path to the session's dataset directory.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use.
            configuration: SCE detection parameters. Uses defaults if None.
            rng_seed: Seed for the per-period numpy random generator.
        """
        df = pl.read_ipc(
            source=session_path.joinpath(DatasetFiles.DATA),
            columns=[
                DatasetColumn.SYSTEM_STATE.value,
                DatasetColumn.TIME_US.value,
                fluorescence_column.value,
                DatasetColumn.TORQUE_N_CM.value,
                DatasetColumn.SPEED_CM_S.value,
                DatasetColumn.TRIAL.value,
            ],
            memory_map=True,
        )
        df = trim_acquisition_warmup(df)

        time_us = df[DatasetColumn.TIME_US.value].to_numpy()
        median_interval_us = np.median(np.diff(time_us))
        self._sampling_rate: float = 1_000_000.0 / float(median_interval_us)

        self._system_state = df[DatasetColumn.SYSTEM_STATE.value].to_list()
        self._elapsed_minutes = (time_us - time_us[0]).astype(np.float32) / np.float32(60_000_000.0)

        # noinspection PyTypeChecker
        self._fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T
        # noinspection PyTypeChecker
        self._torque: NDArray[np.float32] = df[DatasetColumn.TORQUE_N_CM.value].to_numpy().astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        self._speed: NDArray[np.float32] = df[DatasetColumn.SPEED_CM_S.value].to_numpy().astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        self._trial_ids: NDArray[np.int32] = df[DatasetColumn.TRIAL.value].to_numpy().astype(np.int32, copy=False)

        self._configuration: SCEDetectionConfiguration = (
            configuration if configuration is not None else SCEDetectionConfiguration()
        )
        self._rng_seed: int = int(rng_seed)
        self._results: list[SCEResult] = []

    @property
    def results(self) -> list[SCEResult]:
        """Returns every detected SCE result in temporal session order. Each result carries the originating
        ``period_state`` so callers can split events by protocol epoch downstream.
        """
        return list(self._results)

    @property
    def sampling_rate_hz(self) -> float:
        """Returns the sampling rate estimated from the median inter-sample interval at construction time."""
        return self._sampling_rate

    def detect_events(self, *, progress: bool = True) -> list[SCEResult]:
        """Detects SCEs in every stationary chunk across every protocol epoch in the session.

        Notes:
            Walks every contiguous ``SYSTEM_STATE`` block. Inside ``"rest"``-state blocks the torque-stability
            mask is applied to drop fidgeting samples; inside every other state block the speed-stability mask
            is applied to retain only pauses-within-run. Each surviving stationary chunk feeds a separate
            ``SCEResult`` whose ``period_state`` carries the originating block's state name.

        Args:
            progress: Displays a tqdm progress bar tracking period completion when True.

        Returns:
            A list of SCEResult objects in temporal session order.
        """
        pending: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], str]] = []
        current_state: str | None = None
        period_start = 0

        for sample_index in range(len(self._system_state) + 1):
            state = self._system_state[sample_index] if sample_index < len(self._system_state) else None

            if state != current_state:
                if current_state is not None and (sample_index - period_start) > 0:
                    period_state_str = str(current_state)
                    if period_state_str == _REST_STATE:
                        stable_mask = _identify_stable_rest_samples(
                            torque=self._torque[period_start:sample_index],
                            sampling_rate=self._sampling_rate,
                            stability_window_seconds=self._configuration.torque_stability_window_seconds,
                            stability_threshold=self._configuration.torque_stability_threshold,
                        )
                    else:
                        stable_mask = _identify_stable_run_samples(
                            speed=self._speed[period_start:sample_index],
                            sampling_rate=self._sampling_rate,
                            stability_window_seconds=self._configuration.encoder_stability_window_seconds,
                            stability_threshold=self._configuration.encoder_stability_threshold,
                        )

                    stable_count = int(np.sum(stable_mask))
                    stable_fraction = stable_count / len(stable_mask)

                    if stable_fraction > _MINIMUM_STABLE_FRACTION and stable_count > _MINIMUM_STABLE_SAMPLE_COUNT:
                        pending.append(
                            (
                                self._fluorescence[:, period_start:sample_index][:, stable_mask],
                                self._elapsed_minutes[period_start:sample_index][stable_mask],
                                self._trial_ids[period_start:sample_index][stable_mask],
                                period_state_str,
                            )
                        )

                current_state = state
                period_start = sample_index

        periods = tqdm(pending, desc="SCE detection", unit="period") if progress else pending
        # Each period gets its own rng-stream derived from the master seed so detection remains deterministic
        # even when periods are dropped or reordered upstream.
        master_rng = np.random.default_rng(seed=self._rng_seed)
        period_seeds = master_rng.integers(low=0, high=np.iinfo(np.int64).max, size=len(pending))
        self._results = [
            _detect_sces(
                fluorescence=fluorescence,
                sampling_rate=self._sampling_rate,
                timestamps=timestamps,
                trial_ids=trial_ids,
                configuration=self._configuration,
                period_state=period_state,
                rng=np.random.default_rng(seed=int(period_seeds[index])),
            )
            for index, (fluorescence, timestamps, trial_ids, period_state) in enumerate(periods)
        ]

        return self._results
