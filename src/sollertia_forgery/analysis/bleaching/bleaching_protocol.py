"""Per-session computation kernel for the chronic photobleaching evaluation protocol.

Computes the per-session inputs that the cross-session bleaching analysis aggregates: per-cell session-median
baseline fluorescence (estimated as a low percentile of the raw trace within a baseline window), per-cell
signal-to-noise contrast (transient amplitude over MAD noise floor), and the within-session FOV-mean baseline
trace used to quantify acute single-session bleaching. The cross-session aggregates (decay fit, paired Wilcoxon
SNR test, combined flag mask) live alongside in :mod:`.bleaching_analysis`; per-session and cross-session plots
live in :mod:`.plotting`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, interval_to_rate

from ..shared_utilities import trim_acquisition_warmup
from ...shared_assets import DatasetFiles, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MAD_TO_STD_SCALE: np.float32 = np.float32(1.4826)
"""Scaling that maps the median absolute deviation of Gaussian noise to its standard deviation."""
_MINIMUM_SAMPLES_FOR_RATE_ESTIMATE: int = 2
"""Minimum number of timestamp samples required to estimate the inter-sample sampling rate."""


@dataclass(frozen=True, slots=True)
class BleachingConfiguration:
    """Defines configuration parameters for the chronic photobleaching evaluation protocol."""

    baseline_percentile: int = 8
    """Per-cell percentile (0-100) of the fluorescence values within each baseline window taken as the baseline
    fluorescence. Low percentiles approximate the resting trace below transient calcium events; the percentile is
    reused for the across-session trend and for detrending the trace prior to the SNR estimate."""
    cell_baseline_window_seconds: int = 60
    """Width of each non-overlapping window in seconds over which ``baseline_percentile`` is evaluated per cell to
    produce the per-cell baseline fluorescence trace used for the across-session trend and for SNR detrending."""
    session_baseline_window_seconds: int = 10
    """Width of each non-overlapping window in seconds over which ``baseline_percentile`` is evaluated on the FOV-mean
    trace (fluorescence averaged across all cells first) to produce the within-session baseline curve used to quantify
    acute, single-session bleaching."""
    snr_signal_percentile: int = 95
    """Per-cell percentile (0-100) of the detrended trace (raw minus baseline) treated as the typical calcium-event
    amplitude — the upper-tail counterpart to ``baseline_percentile`` and the SNR numerator. The denominator is the
    median absolute deviation (MAD) of the same trace as a transient-robust noise floor; ``snr_loss_threshold`` and
    ``snr_significance_threshold`` use the resulting SNR to flag sessions where events lose contrast against the
    noise."""
    baseline_fluorescence_loss_threshold: float = 0.30
    """Fractional drop in population-median baseline fluorescence from the first session above which the session
    is flagged as chronically bleached."""
    within_session_loss_threshold: float = 0.20
    """Fractional drop in the within-session FOV-mean baseline from the first to the last bin above which the session
    is flagged as acutely bleaching within itself."""
    snr_loss_threshold: float = 0.30
    """Fractional drop in population-median per-cell SNR (transient amplitude over noise floor) from the first session
    above which the session is flagged, provided the paired Wilcoxon comparison is also significant at
    ``snr_significance_threshold``."""
    snr_significance_threshold: float = 0.01
    """Significance level for the paired Wilcoxon signed-rank test comparing each session's per-cell SNR distribution
    to the first session, applied alongside ``snr_loss_threshold`` as the second criterion for SNR-based flagging."""


@dataclass(frozen=True, slots=True)
class BleachingSessionResult:
    """Stores the per-session analysis values consumed by the cross-session bleaching report constructor."""

    sampling_rate_hz: float
    """Effective fluorescence sampling rate in Hz, derived from the median inter-sample period."""
    cell_baseline_fluorescence: NDArray[np.float32]
    """Per-cell session-median baseline fluorescence with length cell_count."""
    cell_snr: NDArray[np.float32]
    """Per-cell signal-to-noise ratio with length cell_count."""
    within_session_time_seconds: NDArray[np.float32]
    """Bin-center timestamps for the within-session FOV-mean baseline trace, in seconds."""
    within_session_baseline: NDArray[np.float32]
    """Within-session FOV-mean baseline values, parallel to ``within_session_time_seconds``."""
    within_session_fractional_drop: float
    """Fraction by which the within-session FOV-mean baseline trace drops from its first to its last bin. NaN when
    the trace is empty or its first bin is non-positive."""


def compute_session_metrics(
    session_path: Path,
    configuration: BleachingConfiguration,
) -> BleachingSessionResult:
    """Loads a single session's multi-recording raw fluorescence and computes the per-session bleaching metrics.

    Notes:
        Operates exclusively on ``DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE``. The protocol's across-session
        per-cell comparisons (paired Wilcoxon SNR test, per-cell baseline trend, decay fit on the population
        median) require that cell index N denote the same neuron across every session in the evaluation set.
        Only the multi-recording cindra column carries that information. Single-recording fluorescence carries no
        cell correspondence across days, so it is not exposed as an option here.

        Per-cell baseline fluorescence is estimated as a low percentile of the raw trace within non-overlapping
        windows, after the Suite2p convention from Pachitariu et al. (2017). Per-cell SNR uses median-absolute-
        deviation noise estimation as a transient-robust noise floor (Pnevmatikakis et al., 2016; Hampel, 1974),
        and the GCaMP signal-to-noise characterization that informs the ~30% degradation threshold in the
        cross-session flag mask comes from Dana et al. (2019) and Zhang et al. (2023). Within-session FOV-mean
        baseline tracking follows the Dombeck/Tank chronic-imaging lineage (Sheffield & Dombeck, 2015;
        Driscoll et al., 2017). All numba kernels and helpers in this module inherit these references through
        this accessor.

    References:
        Suite2p baseline convention (8th-percentile baseline within a 60-second window):
            Pachitariu et al. (2017). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. bioRxiv.
            https://doi.org/10.1101/061507
        Robust MAD-based noise estimation underlying the per-cell SNR computation:
            Pnevmatikakis et al. (2016). Simultaneous denoising, deconvolution, and demixing of calcium imaging
            data. Neuron. https://doi.org/10.1016/j.neuron.2015.11.037
            Hampel (1974). The influence curve and its role in robust estimation. Journal of the American
            Statistical Association. https://doi.org/10.2307/2285666
        GCaMP signal-to-noise characterization informing the ~30% per-cell SNR degradation threshold:
            Dana et al. (2019). High-performance calcium sensors for imaging activity in neuronal populations and
            microcompartments. Nature Methods. https://doi.org/10.1038/s41592-019-0435-6
            Zhang et al. (2023). Fast and sensitive GCaMP calcium indicators for imaging neural populations.
            Nature. https://doi.org/10.1038/s41586-023-05828-9
        Within-session bleaching control common to the Dombeck/Tank chronic-imaging lineage:
            Sheffield & Dombeck (2015). Calcium transient prevalence across the dendritic arbour predicts place
            field properties. Nature. https://doi.org/10.1038/nature14066
            Driscoll et al. (2017). Dynamic reorganization of neuronal activity patterns in parietal cortex.
            Cell. https://doi.org/10.1016/j.cell.2017.05.021

    Args:
        session_path: Path to the forged session directory containing the data feather.
        configuration: Bleaching evaluation parameters that drive window sizes and percentile choices.

    Returns:
        A :class:`BleachingSessionResult` holding the per-cell baseline, per-cell SNR, and within-session baseline
        trace plus the sampling rate and within-session fractional drop scalar.
    """
    # Routes through an annotated local so PyCharm narrows the unpacked elements to the declared fp32/int64 pair.
    raw_session: tuple[NDArray[np.float32], NDArray[np.int64]] = _load_session_raw(session_path=session_path)
    fluorescence, time_us = raw_session
    sampling_rate_hz = _estimate_sampling_rate_hz(time_us=time_us)

    baseline_window_samples = max(round(configuration.cell_baseline_window_seconds * sampling_rate_hz), 1)
    cell_count = fluorescence.shape[0]
    bin_count = fluorescence.shape[1] // baseline_window_samples

    # Declares the array locals up front with explicit fp32 types so PyCharm narrows the constructor call below
    # regardless of which branch produced them.
    cell_baseline_fluorescence: NDArray[np.float32]
    cell_snr: NDArray[np.float32]

    if bin_count == 0:
        # noinspection PyTypeChecker
        cell_baseline_fluorescence = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        cell_snr = np.zeros(cell_count, dtype=np.float32)
    else:
        # noinspection PyTypeChecker
        binned_baseline: NDArray[np.float32] = _compute_binned_baseline(
            fluorescence=fluorescence,
            bin_size_samples=baseline_window_samples,
            percentile=configuration.baseline_percentile,
        )
        # noinspection PyTypeChecker
        cell_baseline_fluorescence = np.empty(cell_count, dtype=np.float32)
        # noinspection PyTypeChecker
        _per_cell_median_along_axis1(matrix=binned_baseline, output=cell_baseline_fluorescence)
        # noinspection PyTypeChecker
        cell_snr = _compute_cell_snr(
            fluorescence=fluorescence,
            binned_baseline=binned_baseline,
            bin_size_samples=baseline_window_samples,
            signal_percentile=configuration.snr_signal_percentile,
        )

    within_session_bin_samples = max(round(configuration.session_baseline_window_seconds * sampling_rate_hz), 1)
    # Routes through an annotated local so PyCharm narrows the unpacked elements to the declared fp32 NDArray pair.
    within_session_result: tuple[NDArray[np.float32], NDArray[np.float32]] = _compute_within_session_baseline(
        fluorescence=fluorescence,
        sampling_rate_hz=sampling_rate_hz,
        bin_size_samples=within_session_bin_samples,
        percentile=configuration.baseline_percentile,
    )
    within_session_time_seconds, within_session_baseline = within_session_result

    if within_session_baseline.size > 0 and within_session_baseline[0] > 0:
        within_session_fractional_drop = float(
            (within_session_baseline[0] - within_session_baseline[-1]) / within_session_baseline[0]
        )
    else:
        within_session_fractional_drop = float("nan")

    return BleachingSessionResult(
        sampling_rate_hz=sampling_rate_hz,
        cell_baseline_fluorescence=cell_baseline_fluorescence,
        cell_snr=cell_snr,
        within_session_time_seconds=within_session_time_seconds,
        within_session_baseline=within_session_baseline,
        within_session_fractional_drop=within_session_fractional_drop,
    )


def _load_session_raw(session_path: Path) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Loads the multi-recording raw per-cell fluorescence trace and per-sample timestamps from a forged session
    feather, with the leading acquisition-warmup window trimmed off both arrays.

    Notes:
        Loads ``DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE`` exclusively because the protocol's across-session
        per-cell comparisons require registered cell correspondence across days, which only the multi-recording
        cindra column carries. The fluorescence column is stored as a polars list-of-float32, one list per sample.
        Converting via ``Series.to_list()`` and ``np.array`` materializes a Python list of lists for every sample,
        which is single-threaded, GIL-bound, and dominates load time for multi-thousand-cell sessions. Exploding
        the list column to a flat fp32 series and reshaping in NumPy stays in compiled code and runs roughly an
        order of magnitude faster while producing the same (cell_count, sample_count) C-contiguous layout.

        Drops the acquisition warmup window at the dataframe level via ``trim_acquisition_warmup`` so every
        downstream kernel — per-cell baseline percentile, SNR, within-session bleaching — operates on stabilized
        data without needing its own warmup-aware logic. Sessions that contain no samples past the warmup window
        are returned as empty arrays; existing length guards in the per-session pipeline produce NaN sentinels for
        such degenerate sessions.

    Args:
        session_path: Path to the forged session directory containing the data feather.

    Returns:
        A tuple containing the (cell_count, sample_count) fp32 fluorescence array and the per-sample int64
        microsecond timestamps, both already trimmed of the acquisition-warmup window.
    """
    df = pl.read_ipc(
        source=session_path.joinpath(DatasetFiles.DATA),
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE.value],
    )
    df = trim_acquisition_warmup(df=df)
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy().astype(np.int64, copy=False)

    sample_count = df.height
    # noinspection PyTypeChecker
    flat: NDArray[np.float32] = df[DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE.value].explode().to_numpy()
    if flat.dtype != np.float32:
        # noinspection PyTypeChecker
        flat = flat.astype(np.float32, copy=False)
    cell_count = flat.size // sample_count
    # Reshapes the flattened (sample_count * cell_count) buffer into (sample_count, cell_count) and transposes to
    # the analysis-canonical (cell_count, sample_count) layout. The transpose is a non-contiguous view, so a single
    # ascontiguousarray copy materializes the C-contiguous result that downstream reshapes need.
    # noinspection PyTypeChecker
    fluorescence: NDArray[np.float32] = np.ascontiguousarray(flat.reshape(sample_count, cell_count).T)

    return fluorescence, time_us


def _estimate_sampling_rate_hz(time_us: NDArray[np.int64]) -> float:
    """Estimates the sampling rate in Hz from the median inter-sample interval of the timestamp array.

    Args:
        time_us: Per-sample acquisition timestamps in microseconds.

    Returns:
        The estimated sampling rate in Hz. NaN when fewer than two samples are supplied or the median inter-sample
        interval is non-positive.
    """
    if time_us.size < _MINIMUM_SAMPLES_FOR_RATE_ESTIMATE:
        return float("nan")
    # Diffs the int64 timestamps directly (intervals are positive and small enough to never overflow) and lets the
    # median materialize as a Python float for the final scalar division.
    # noinspection PyTypeChecker
    deltas_us: NDArray[np.int64] = np.diff(time_us)
    median_delta_us = float(np.median(deltas_us))
    if median_delta_us <= 0:
        return float("nan")
    return float(interval_to_rate(interval=median_delta_us, from_units=TimeUnits.MICROSECOND, as_float=True))


def _compute_binned_baseline(
    fluorescence: NDArray[np.float32],
    bin_size_samples: int,
    percentile: int,
) -> NDArray[np.float32]:
    """Computes a per-cell, per-bin percentile baseline using non-overlapping windows along the time axis.

    Notes:
        Replaces the per-sample rolling-window percentile of the original Suite2p convention with a non-overlapping
        binned percentile of the same window size. Dispatches to a numba parallel-over-cells kernel that performs
        the percentile via in-place quickselect; the previous ``np.percentile`` call was single-threaded and
        dominated session compute on multi-thousand-cell traces.

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count).
        bin_size_samples: Width of each non-overlapping bin in samples.
        percentile: Percentile (0-100) evaluated within each bin.

    Returns:
        Per-cell, per-bin baseline values with dimensions (cell_count, bin_count). Returns a (cell_count, 0) array
        when fewer than one full bin fits in the input.
    """
    cell_count, sample_count = fluorescence.shape
    bin_count = sample_count // bin_size_samples
    if bin_count == 0:
        # noinspection PyTypeChecker
        empty_binned: NDArray[np.float32] = np.zeros((cell_count, 0), dtype=np.float32)
        return empty_binned

    # noinspection PyTypeChecker
    binned: NDArray[np.float32] = np.empty((cell_count, bin_count), dtype=np.float32)
    _binned_percentile_kernel(
        fluorescence=fluorescence,
        bin_size_samples=np.int64(bin_size_samples),
        bin_count=np.int64(bin_count),
        percentile=np.float32(percentile),
        output=binned,
    )
    return binned


def _compute_cell_snr(
    fluorescence: NDArray[np.float32],
    binned_baseline: NDArray[np.float32],
    bin_size_samples: int,
    signal_percentile: int,
) -> NDArray[np.float32]:
    """Computes per-cell SNR by detrending the raw trace with the per-bin baseline.

    Notes:
        Hands the heavy work to a single numba parallel-over-cells kernel that computes the detrended trace into a
        thread-local scratch buffer, runs three quickselects (median for the noise center, MAD median, signal
        percentile), and writes the SNR. Avoiding the explicit ``np.repeat`` upsample saves a (cell_count *
        sample_count) allocation and a memory-bound pass over it, and replacing the three single-threaded
        ``np.median`` / ``np.percentile`` reductions with one parallel kernel scales the operation across cores.

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count).
        binned_baseline: Per-cell, per-bin baseline with dimensions (cell_count, bin_count) returned by
            ``_compute_binned_baseline``.
        bin_size_samples: Width of each baseline bin in samples, used to map sample indices to bin indices when
            detrending on the fly.
        signal_percentile: Upper percentile of the detrended trace treated as the per-cell event amplitude.

    Returns:
        Per-cell SNR with length cell_count. Cells with zero estimated noise are reported as 0 to avoid division
        by zero rather than NaN or infinity.
    """
    cell_count = fluorescence.shape[0]
    if binned_baseline.shape[1] == 0:
        # noinspection PyTypeChecker
        empty_snr: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        return empty_snr

    # noinspection PyTypeChecker
    snr: NDArray[np.float32] = np.empty(cell_count, dtype=np.float32)
    _cell_snr_kernel(
        fluorescence=fluorescence,
        binned_baseline=binned_baseline,
        bin_size_samples=np.int64(bin_size_samples),
        signal_percentile=np.float32(signal_percentile),
        output=snr,
    )
    return snr


def _compute_within_session_baseline(
    fluorescence: NDArray[np.float32],
    sampling_rate_hz: float,
    bin_size_samples: int,
    percentile: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the within-session FOV-mean baseline percentile across non-overlapping time bins.

    Notes:
        Reduces what was a Python loop over per-bin masks to a single contiguous reshape plus one vectorized
        ``np.percentile`` call along the bin axis, which is orders of magnitude faster on multi-thousand-sample
        traces.

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count).
        sampling_rate_hz: Per-sample sampling rate used to convert bin indices to seconds.
        bin_size_samples: Width of each non-overlapping bin in samples.
        percentile: Percentile (0-100) evaluated within each bin of the FOV-mean trace.

    Returns:
        A tuple of bin-center timestamps (seconds) and per-bin FOV-mean baseline values, both with length bin_count.
        Returns two empty arrays when the trace contains fewer than one full bin or the sampling rate is unknown.
    """
    sample_count = fluorescence.shape[1]
    bin_count = sample_count // bin_size_samples
    if sample_count < _MINIMUM_SAMPLES_FOR_RATE_ESTIMATE or not np.isfinite(sampling_rate_hz) or bin_count == 0:
        # noinspection PyTypeChecker
        empty_time: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_baseline: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        return empty_time, empty_baseline

    # noinspection PyTypeChecker
    fov_mean: NDArray[np.float32] = np.mean(fluorescence, axis=0).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    trimmed: NDArray[np.float32] = fov_mean[: bin_count * bin_size_samples]
    # noinspection PyTypeChecker
    reshaped: NDArray[np.float32] = trimmed.reshape(bin_count, bin_size_samples)
    # noinspection PyTypeChecker
    baseline: NDArray[np.float32] = np.percentile(reshaped, percentile, axis=1).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    bin_centers: NDArray[np.float32] = (
        (np.arange(bin_count, dtype=np.float32) + np.float32(0.5))
        * np.float32(bin_size_samples)
        / np.float32(sampling_rate_hz)
    )
    return bin_centers, baseline


@njit(cache=True, parallel=True)
def _binned_percentile_kernel(
    fluorescence: NDArray[np.float32],
    bin_size_samples: int,
    bin_count: int,
    percentile: float,
    output: NDArray[np.float32],
) -> None:
    """Computes per-cell, per-bin percentile of the fluorescence trace via in-place quickselect.

    Notes:
        Parallelizes over cells. Each thread allocates a single scratch buffer sized to one bin and reuses it
        across that cell's bins; the buffer fits in L2 for typical 60-second windows and avoids per-bin
        allocations that would otherwise dominate kernel time.

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count), C-contiguous fp32.
        bin_size_samples: Width of each non-overlapping bin in samples.
        bin_count: Number of complete bins that fit in the input.
        percentile: Percentile (0-100) evaluated within each bin via NumPy linear interpolation.
        output: Pre-allocated (cell_count, bin_count) fp32 array. Modified in place.
    """
    cell_count = fluorescence.shape[0]
    fractional_position = (percentile / np.float32(100.0)) * np.float32(bin_size_samples - 1)
    lower_index = int(np.floor(fractional_position))
    lower_index = max(lower_index, 0)
    lower_index = min(lower_index, bin_size_samples - 1)
    upper_index = lower_index + 1 if lower_index < bin_size_samples - 1 else lower_index
    weight = np.float32(fractional_position - lower_index)

    for cell_index in prange(cell_count):
        # noinspection PyTypeChecker
        scratch: NDArray[np.float32] = np.empty(bin_size_samples, dtype=np.float32)
        for bin_index in range(bin_count):
            offset = bin_index * bin_size_samples
            for sample_index in range(bin_size_samples):
                scratch[sample_index] = fluorescence[cell_index, offset + sample_index]

            lower_value = _quickselect_inplace(buffer=scratch, target_index=lower_index)
            if upper_index == lower_index:
                output[cell_index, bin_index] = lower_value
            else:
                # The values strictly above the lower-rank pivot are concentrated in scratch[lower_index + 1:];
                # quickselect did not fully sort, so the upper rank is the minimum of that suffix.
                upper_value = scratch[lower_index + 1]
                for scan_index in range(lower_index + 2, bin_size_samples):
                    # noinspection PyTypeChecker
                    upper_value = min(upper_value, scratch[scan_index])
                # noinspection PyTypeChecker
                output[cell_index, bin_index] = lower_value + weight * (upper_value - lower_value)


@njit(cache=True, parallel=True)
def _per_cell_median_along_axis1(
    matrix: NDArray[np.float32],
    output: NDArray[np.float32],
) -> None:
    """Computes the median of each row of ``matrix`` and writes it to ``output``.

    Notes:
        Replaces ``np.median(matrix, axis=1)``, which is single-threaded. Quickselect is O(n) per row and runs in
        parallel across rows.

    Args:
        matrix: Input array with dimensions (row_count, column_count), C-contiguous fp32.
        output: Pre-allocated fp32 array with length row_count. Modified in place.
    """
    row_count = matrix.shape[0]
    column_count = matrix.shape[1]
    if column_count == 0:
        output[:] = np.float32(np.nan)
        return
    lower_index = (column_count - 1) // 2
    is_even = column_count % 2 == 0
    for row_index in prange(row_count):
        # noinspection PyTypeChecker
        scratch: NDArray[np.float32] = np.empty(column_count, dtype=np.float32)
        for column_index in range(column_count):
            scratch[column_index] = matrix[row_index, column_index]
        lower_value = _quickselect_inplace(buffer=scratch, target_index=lower_index)
        if not is_even:
            output[row_index] = lower_value
        else:
            upper_value = scratch[lower_index + 1]
            for scan_index in range(lower_index + 2, column_count):
                # noinspection PyTypeChecker
                upper_value = min(upper_value, scratch[scan_index])
            output[row_index] = np.float32(0.5) * (lower_value + upper_value)


@njit(cache=True, parallel=True)
def _cell_snr_kernel(
    fluorescence: NDArray[np.float32],
    binned_baseline: NDArray[np.float32],
    bin_size_samples: int,
    signal_percentile: float,
    output: NDArray[np.float32],
) -> None:
    """Computes per-cell SNR with detrending fused into one parallel-over-cells pass.

    Notes:
        Each cell allocates a single scratch buffer sized to ``sample_count`` and reuses it for the detrended
        trace, then for the absolute deviations from the median. The baseline is referenced from ``binned_baseline``
        on the fly so the (cell_count * sample_count) upsampled buffer that the previous implementation built is
        never materialized. Three quickselects (median for noise center, MAD median, signal percentile) cost O(n)
        each instead of three full sorts at O(n log n).

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count), C-contiguous fp32.
        binned_baseline: Per-cell, per-bin baseline with dimensions (cell_count, bin_count) from the binned-baseline
            kernel. Bins indexed past ``bin_count - 1`` reuse the last bin so the trailing samples that do not fill
            a complete window stay aligned with the input length.
        bin_size_samples: Width of each baseline bin in samples.
        signal_percentile: Upper percentile of the detrended trace treated as the per-cell event amplitude.
        output: Pre-allocated fp32 array with length cell_count. Modified in place.
    """
    cell_count = fluorescence.shape[0]
    sample_count = fluorescence.shape[1]
    bin_count = binned_baseline.shape[1]
    last_bin_index = bin_count - 1

    median_lower = (sample_count - 1) // 2
    median_is_even = sample_count % 2 == 0

    signal_position = (signal_percentile / np.float32(100.0)) * np.float32(sample_count - 1)
    signal_lower_index = int(np.floor(signal_position))
    signal_lower_index = max(signal_lower_index, 0)
    signal_lower_index = min(signal_lower_index, sample_count - 1)
    signal_upper_index = signal_lower_index + 1 if signal_lower_index < sample_count - 1 else signal_lower_index
    signal_weight = np.float32(signal_position - signal_lower_index)

    mad_scale = _MAD_TO_STD_SCALE

    for cell_index in prange(cell_count):
        # noinspection PyTypeChecker
        scratch: NDArray[np.float32] = np.empty(sample_count, dtype=np.float32)

        # Detrends on the fly; bins beyond bin_count - 1 reuse the last bin to mirror the previous repeat-and-pad
        # behavior of the materialized upsample.
        for sample_index in range(sample_count):
            bin_index = min(sample_index // bin_size_samples, last_bin_index)
            scratch[sample_index] = fluorescence[cell_index, sample_index] - binned_baseline[cell_index, bin_index]

        # Computes the median of the detrended trace via quickselect.
        median_lower_value = _quickselect_inplace(buffer=scratch, target_index=median_lower)
        if not median_is_even:
            detrended_median = median_lower_value
        else:
            upper_value = scratch[median_lower + 1]
            for scan_index in range(median_lower + 2, sample_count):
                # noinspection PyTypeChecker
                upper_value = min(upper_value, scratch[scan_index])
            detrended_median = np.float32(0.5) * (median_lower_value + upper_value)

        # Refills scratch with the detrended values; quickselect partially sorted them, and recomputing on the fly
        # is cheaper than holding a second copy.
        for sample_index in range(sample_count):
            bin_index = min(sample_index // bin_size_samples, last_bin_index)
            scratch[sample_index] = fluorescence[cell_index, sample_index] - binned_baseline[cell_index, bin_index]

        # Signal percentile via quickselect with linear interpolation between the two bracketing ranks.
        signal_lower_value = _quickselect_inplace(buffer=scratch, target_index=signal_lower_index)
        if signal_upper_index == signal_lower_index:
            signal = signal_lower_value
        else:
            upper_value = scratch[signal_lower_index + 1]
            for scan_index in range(signal_lower_index + 2, sample_count):
                # noinspection PyTypeChecker
                upper_value = min(upper_value, scratch[scan_index])
            # noinspection PyTypeChecker
            signal = signal_lower_value + signal_weight * (upper_value - signal_lower_value)

        # Reuses scratch for the absolute deviations from the detrended median, then quickselects the MAD.
        for sample_index in range(sample_count):
            bin_index = min(sample_index // bin_size_samples, last_bin_index)
            value = fluorescence[cell_index, sample_index] - binned_baseline[cell_index, bin_index]
            deviation = value - detrended_median
            scratch[sample_index] = deviation if deviation >= 0 else -deviation

        mad_lower_value = _quickselect_inplace(buffer=scratch, target_index=median_lower)
        if not median_is_even:
            mad = mad_lower_value
        else:
            upper_value = scratch[median_lower + 1]
            for scan_index in range(median_lower + 2, sample_count):
                # noinspection PyTypeChecker
                upper_value = min(upper_value, scratch[scan_index])
            mad = np.float32(0.5) * (mad_lower_value + upper_value)

        noise_std = mad * mad_scale
        if noise_std > 0:
            output[cell_index] = signal / noise_std
        else:
            output[cell_index] = np.float32(0.0)


@njit(cache=True)
def _quickselect_inplace(buffer: NDArray[np.float32], target_index: int) -> np.float32:
    """Partitions ``buffer`` in place so the element with ordinal ``target_index`` lands at that index.

    Notes:
        Lomuto-partition quickselect with median-of-three pivot selection. Average O(n), worst-case O(n^2);
        median-of-three keeps the worst case from showing up on monotone or nearly-sorted inputs that the
        baseline-percentile workload sees. After return, every element at index < target_index is <=
        ``buffer[target_index]`` and every element at index > target_index is >= ``buffer[target_index]``, so the
        caller can recover the next-larger value as ``min(buffer[target_index + 1:])`` for percentile interpolation.

    Args:
        buffer: 1D fp32 array. Modified in place.
        target_index: 0-based index whose ordinal value should land at ``buffer[target_index]``.

    Returns:
        The value that ends up at ``buffer[target_index]`` after partitioning, equivalent to the
        ``target_index``-th order statistic.
    """
    left = 0
    right = buffer.shape[0] - 1
    while left < right:
        # Sorts (left, mid, right) so buffer[left] <= buffer[mid] <= buffer[right] for median-of-three pivot
        # selection. Then stages the pivot at the right end so the Lomuto scan can run over [left, right - 1] with
        # the pivot value held constant in buffer[right].
        mid = (left + right) // 2
        if buffer[left] > buffer[mid]:
            buffer[left], buffer[mid] = buffer[mid], buffer[left]
        if buffer[left] > buffer[right]:
            buffer[left], buffer[right] = buffer[right], buffer[left]
        if buffer[mid] > buffer[right]:
            buffer[mid], buffer[right] = buffer[right], buffer[mid]
        buffer[mid], buffer[right] = buffer[right], buffer[mid]
        pivot = buffer[right]

        store_index = left
        for scan_index in range(left, right):
            if buffer[scan_index] < pivot:
                buffer[scan_index], buffer[store_index] = buffer[store_index], buffer[scan_index]
                store_index += 1
        # Swaps the pivot into its final position. Everything in [left, store_index) is < pivot, everything in
        # (store_index, right] is >= pivot, and buffer[store_index] is the pivot value.
        buffer[store_index], buffer[right] = buffer[right], buffer[store_index]

        if store_index == target_index:
            return buffer[store_index]
        if store_index < target_index:
            left = store_index + 1
        else:
            right = store_index - 1
    return buffer[target_index]
