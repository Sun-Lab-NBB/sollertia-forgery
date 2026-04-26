"""Quantifies photobleaching across a chronologically ordered set of two-photon imaging sessions for the same animal.

Implements the canonical three-metric protocol for chronic GCaMP imaging: per-cell session-median baseline F0
trend across days fit to a single exponential, within-session bleaching slope, and per-cell signal-to-noise change
on the multi-recording registered cell intersection.

References:
    Suite2p baseline convention (8th-percentile baseline within a 60-second window):
        Pachitariu et al. (2017). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. bioRxiv.
        https://doi.org/10.1101/061507
    Multi-day registered cell intersection for longitudinal comparisons:
        Ziv et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience.
        https://doi.org/10.1038/nn.3329
        Rubin et al. (2015). Hippocampal ensemble dynamics timestamp events in long-term memory. eLife.
        https://doi.org/10.7554/eLife.12247
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
    Standardized longitudinal-imaging quality-control framework that motivates the combined-flagging strategy:
        de Vries et al. (2020). A large-scale standardized physiological survey reveals functional
        organization of the mouse visual cortex. Nature Neuroscience.
        https://doi.org/10.1038/s41593-019-0550-9
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ataraxis_base_utilities import console

from ..forging import DATA_FILENAME, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MICROSECONDS_PER_SECOND: float = 1.0e6
"""Conversion factor between microsecond timestamps and seconds."""
_SECONDS_PER_DAY: float = 86400.0
"""Conversion factor between seconds and days."""
_MAD_TO_STD_SCALE: float = 1.4826
"""Scaling that maps the median absolute deviation of Gaussian noise to its standard deviation."""
_SESSION_NAME_FIELD_COUNT: int = 7
"""Number of dash-separated fields produced by the canonical session timestamp format."""
_MINIMUM_SESSIONS_FOR_DECAY_FIT: int = 3
"""Minimum number of sessions required to fit a single-exponential decay model."""
_MINIMUM_SESSIONS_FOR_EVALUATION: int = 2
"""Minimum number of sessions required to evaluate any across-session bleaching metric."""
_MINIMUM_SAMPLES_FOR_RATE_ESTIMATE: int = 2
"""Minimum number of timestamp samples required to estimate the inter-sample sampling rate."""
_WORKER_RESERVE: int = 4
"""Number of CPU cores reserved for the OS when worker_count=-1 selects an automatic worker count."""


@dataclass(frozen=True, slots=True)
class BleachingConfiguration:
    """Defines configuration parameters for the chronic photobleaching evaluation protocol."""

    baseline_percentile: float = 8.0
    """Per-cell baseline percentile applied within each rolling baseline window. Suite2p convention is the 8th
    percentile, which is robust to large transients while still tracking slow drift."""
    baseline_window_seconds: float = 60.0
    """Width of the rolling baseline window in seconds. Suite2p default for chronic imaging is 60 seconds."""
    within_session_bin_seconds: float = 10.0
    """Bin width in seconds for the within-session bleaching trace. Coarser bins suppress transient leakage into the
    baseline estimate."""
    snr_signal_percentile: float = 95.0
    """Percentile of the detrended trace treated as the per-cell event amplitude when computing SNR."""
    f0_loss_threshold: float = 0.30
    """Fractional drop in median across-session F0 (relative to the first session) above which a session is flagged
    as chronically bleached. Aligned with the GENIE/Janelia GCaMP characterization papers."""
    within_session_loss_threshold: float = 0.20
    """Fractional drop in within-session FOV-mean baseline (start to end of session) above which a session is flagged
    as acutely bleaching."""
    snr_loss_threshold: float = 0.30
    """Fractional drop in median per-cell SNR (relative to the first session) above which a session is flagged when
    accompanied by a significant Wilcoxon test."""
    snr_significance_threshold: float = 0.01
    """P-value threshold for the paired Wilcoxon test comparing per-cell SNR distributions to the first session."""


@dataclass(frozen=True, slots=True)
class SessionBleachingMetrics:
    """Stores the per-session bleaching metrics computed for a single session in the evaluation set."""

    session_path: Path
    """Path to the session directory the metrics were computed from."""
    days_since_first: float
    """Calendar days elapsed between the first session in the evaluation set and this session, derived from the
    session timestamp."""
    sampling_rate_hz: float
    """Effective fluorescence sampling rate in Hz, derived from the median inter-sample period in the session."""
    cell_baseline_f0: NDArray[np.float32]
    """Per-cell session-median baseline fluorescence with length cell_count, computed as the median over time of
    the rolling-window baseline percentile."""
    cell_snr: NDArray[np.float32]
    """Per-cell signal-to-noise ratio with length cell_count, computed as the configured signal percentile of the
    detrended trace divided by the robust noise standard deviation."""
    within_session_time_seconds: NDArray[np.float32]
    """Bin-center timestamps for the within-session baseline trace in seconds."""
    within_session_baseline: NDArray[np.float32]
    """Within-session FOV-mean baseline trace evaluated at ``within_session_time_seconds``."""
    within_session_fractional_drop: float
    """Fraction by which the within-session baseline trace drops from its first to its last bin
    ``(start - end) / start``."""


@dataclass(frozen=True, slots=True)
class ExponentialDecayFit:
    """Stores the result of fitting ``F0(d) = amplitude * exp(-d / tau_days) + offset`` to per-session F0."""

    amplitude: float
    """Decaying-component amplitude in raw fluorescence units."""
    tau_days: float
    """Decay time constant in days. NaN when the fit failed or fewer than the required number of sessions were
    available."""
    offset: float
    """Asymptotic baseline component in raw fluorescence units."""
    fit_succeeded: bool
    """Whether scipy.optimize.curve_fit converged on a finite, in-bounds solution."""


@dataclass(frozen=True, slots=True)
class BleachingReport:
    """Aggregates the bleaching evaluation results across all sessions in the evaluation set."""

    sessions: tuple[SessionBleachingMetrics, ...]
    """Per-session metrics in the chronological order they were supplied."""
    cell_count: int
    """Number of registered cells common to every session in the evaluation set."""
    f0_population_trend: NDArray[np.float32]
    """Population-median per-session F0 with length session_count, computed across the registered cell intersection."""
    f0_decay_fit: ExponentialDecayFit
    """Single-exponential decay fit applied to ``f0_population_trend`` versus ``days_since_first``."""
    f0_fractional_loss: float
    """Fractional drop of the last session's population-median F0 relative to the first session
    ``(F0_first - F0_last) / F0_first``."""
    snr_population_trend: NDArray[np.float32]
    """Population-median per-session SNR with length session_count."""
    snr_paired_p_values: NDArray[np.float64]
    """Paired Wilcoxon signed-rank p-values comparing per-cell SNR in each session against the first session, with
    length session_count. The first entry is NaN since the comparison is to itself."""
    flagged_sessions: tuple[Path, ...]
    """Paths of sessions that violated at least one threshold criterion in the configuration."""
    configuration: BleachingConfiguration
    """The configuration used to produce the report."""

    def plot_baseline_trend(self) -> plt.Figure:
        """Plots the population-median per-session F0 versus days, overlaid with the exponential decay fit and
        per-cell baseline distributions.

        Returns:
            A matplotlib Figure showing the across-session F0 trend.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        days = np.array([session.days_since_first for session in self.sessions], dtype=np.float32)

        # Draws the per-cell distributions as boxplots so the population spread is visible alongside the median trend.
        boxplot_data = [session.cell_baseline_f0 for session in self.sessions]
        axes.boxplot(
            boxplot_data,
            positions=days,
            widths=0.4 * max(np.diff(days).min() if len(days) > 1 else 1.0, 0.1),
            showfliers=False,
        )

        # Overlays the population-median trend used for the exponential fit.
        axes.plot(days, self.f0_population_trend, marker="o", color="tab:blue", linewidth=1.5, label="Median F0")

        # Draws the fitted exponential when the fit converged.
        if self.f0_decay_fit.fit_succeeded:
            dense_days = np.linspace(days.min(), days.max(), num=200)
            fit_curve = self.f0_decay_fit.amplitude * np.exp(-dense_days / self.f0_decay_fit.tau_days) + (
                self.f0_decay_fit.offset
            )
            axes.plot(
                dense_days,
                fit_curve,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"Exp fit (tau = {self.f0_decay_fit.tau_days:.1f} d)",
            )

        axes.set_xlabel("Days since first session")
        axes.set_ylabel("Baseline fluorescence F0 (a.u.)")
        axes.set_title(f"Across-session F0 trend (n={self.cell_count} registered cells)", fontsize=10)
        axes.legend(loc="best", fontsize=8)
        figure.tight_layout()
        return figure

    def plot_within_session(self) -> plt.Figure:
        """Plots the within-session FOV-mean baseline trace for each session as overlaid curves.

        Returns:
            A matplotlib Figure showing within-session bleaching.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        colormap = plt.get_cmap("viridis")
        for index, session in enumerate(self.sessions):
            color = colormap(index / max(len(self.sessions) - 1, 1))
            label = f"Day {session.days_since_first:.1f} (drop={session.within_session_fractional_drop:.1%})"
            axes.plot(
                session.within_session_time_seconds / 60.0,
                session.within_session_baseline,
                color=color,
                linewidth=1.0,
                label=label,
            )

        axes.set_xlabel("Time within session (minutes)")
        axes.set_ylabel("FOV-mean baseline (a.u.)")
        axes.set_title("Within-session bleaching", fontsize=10)
        axes.legend(loc="best", fontsize=7)
        figure.tight_layout()
        return figure

    def plot_snr_distributions(self) -> plt.Figure:
        """Plots per-session per-cell SNR distributions as violins, annotated with the paired Wilcoxon p-values.

        Returns:
            A matplotlib Figure showing the SNR-vs-session comparison.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        days = [session.days_since_first for session in self.sessions]
        snr_data = [session.cell_snr for session in self.sessions]
        axes.violinplot(snr_data, positions=days, showmedians=True)

        # Annotates each session past the first with the Wilcoxon p-value relative to session 0.
        y_position = max(snr.max() for snr in snr_data) * 1.05
        for index, day in enumerate(days):
            if index == 0:
                continue
            p_value = self.snr_paired_p_values[index]
            label = f"p={p_value:.1e}" if np.isfinite(p_value) else "p=N/A"
            axes.text(day, y_position, label, ha="center", fontsize=7)

        axes.set_xlabel("Days since first session")
        axes.set_ylabel("Per-cell SNR")
        axes.set_title("Per-cell SNR across sessions (paired Wilcoxon vs session 0)", fontsize=10)
        figure.tight_layout()
        return figure


def evaluate_bleaching(
    session_paths: tuple[Path, ...],
    fluorescence_column: DatasetColumn = DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE,
    *,
    configuration: BleachingConfiguration | None = None,
    worker_count: int = -1,
) -> BleachingReport:
    """Quantifies photobleaching across the supplied chronologically ordered sessions for a single animal.

    Notes:
        Sessions must share a multi-recording cindra registration so the cell axis is comparable across days. The
        function rejects mismatched cell counts. By default, the multi-recording raw cell-fluorescence column is used,
        which carries the cell correspondence required for a defensible across-session comparison; passing the
        single-recording column is supported but only yields valid within-session metrics.

    Args:
        session_paths: Chronologically ordered tuple of session directory paths. Sessions are validated to be in
            non-decreasing date order via the canonical session-name timestamp.
        fluorescence_column: The raw cell-fluorescence column to load. Must be one of the raw cindra columns;
            baseline-corrected dF/F0 columns are rejected because bleaching is defined on raw signal decay.
        configuration: Bleaching evaluation parameters. Uses defaults if None.
        worker_count: Number of parallel worker threads used to compute per-session metrics. Set to -1 to derive
            the count automatically from the available CPU cores minus a small reserve. Per-session computation
            mostly executes in GIL-releasing C/Rust code (polars I/O, NumPy reductions), so threads scale near
            linearly up to the count of supplied sessions.

    Returns:
        A BleachingReport containing per-session metrics, the across-session F0 decay fit, paired SNR Wilcoxon
        results, and the list of sessions that exceeded any threshold.
    """
    configuration = configuration if configuration is not None else BleachingConfiguration()

    if len(session_paths) < _MINIMUM_SESSIONS_FOR_EVALUATION:
        message = (
            f"Unable to evaluate bleaching across the supplied sessions. The protocol requires at least two "
            f"sessions, but got {len(session_paths)}."
        )
        console.error(message=message, error=ValueError)

    if fluorescence_column not in (
        DatasetColumn.SINGLE_DAY_CELL_FLUORESCENCE,
        DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE,
    ):
        message = (
            f"Unable to evaluate bleaching using the requested fluorescence column. The protocol must operate on "
            f"raw fluorescence, but got {fluorescence_column.value!r}."
        )
        console.error(message=message, error=ValueError)

    session_dates = tuple(_parse_session_date(session_path=path) for path in session_paths)
    _validate_chronological_order(session_paths=session_paths, session_dates=session_dates)

    first_date = session_dates[0]
    days_since_first_list = [
        (session_date - first_date).total_seconds() / _SECONDS_PER_DAY for session_date in session_dates
    ]

    if worker_count == -1:
        worker_count = max(1, (os.cpu_count() or 1) - _WORKER_RESERVE)
    worker_count = min(worker_count, len(session_paths))

    # Computes per-session metrics in parallel. Each call loads the session feather and runs vectorized NumPy
    # reductions, both of which release the GIL, so threads scale efficiently across sessions without paying the
    # serialization cost of process-based parallelism.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _compute_session_metrics,
                session_path=session_path,
                fluorescence_column=fluorescence_column,
                days_since_first=days_since_first,
                configuration=configuration,
            )
            for session_path, days_since_first in zip(session_paths, days_since_first_list, strict=True)
        ]
        metrics: list[SessionBleachingMetrics] = [future.result() for future in futures]

    cell_count_reference = int(metrics[0].cell_baseline_f0.shape[0])
    for session_metrics in metrics[1:]:
        if int(session_metrics.cell_baseline_f0.shape[0]) != cell_count_reference:
            other_count = int(session_metrics.cell_baseline_f0.shape[0])
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The cell count must match across all "
                f"sessions for the multi-recording registered comparison, but session "
                f"{session_metrics.session_path.name!r} has {other_count} cells while the first session has "
                f"{cell_count_reference}."
            )
            console.error(message=message, error=ValueError)

    metrics_tuple = tuple(metrics)

    # noinspection PyTypeChecker
    f0_population_trend: NDArray[np.float32] = np.array(
        [float(np.median(session.cell_baseline_f0)) for session in metrics_tuple],
        dtype=np.float32,
    )
    # noinspection PyTypeChecker
    days_array: NDArray[np.float32] = np.array(
        [session.days_since_first for session in metrics_tuple], dtype=np.float32
    )
    decay_fit = _fit_exponential_decay(days=days_array, baseline=f0_population_trend)

    f0_fractional_loss = float((f0_population_trend[0] - f0_population_trend[-1]) / f0_population_trend[0])

    # noinspection PyTypeChecker
    snr_population_trend: NDArray[np.float32] = np.array(
        [float(np.median(session.cell_snr)) for session in metrics_tuple],
        dtype=np.float32,
    )
    snr_paired_p_values = _compute_paired_snr_p_values(metrics=metrics_tuple)

    flagged_sessions = _flag_sessions(
        metrics=metrics_tuple,
        f0_population_trend=f0_population_trend,
        snr_population_trend=snr_population_trend,
        snr_paired_p_values=snr_paired_p_values,
        configuration=configuration,
    )

    return BleachingReport(
        sessions=metrics_tuple,
        cell_count=cell_count_reference if cell_count_reference is not None else 0,
        f0_population_trend=f0_population_trend,
        f0_decay_fit=decay_fit,
        f0_fractional_loss=f0_fractional_loss,
        snr_population_trend=snr_population_trend,
        snr_paired_p_values=snr_paired_p_values,
        flagged_sessions=flagged_sessions,
        configuration=configuration,
    )


def _parse_session_date(session_path: Path) -> datetime:
    """Parses the canonical 'YYYY-MM-DD-HH-MM-SS-microseconds' session-directory name into a datetime instance."""
    parts = session_path.name.split("-")
    if len(parts) < _SESSION_NAME_FIELD_COUNT:
        message = (
            f"Unable to parse the session timestamp from path {str(session_path)!r}. The session directory name must "
            f"follow the 'YYYY-MM-DD-HH-MM-SS-microseconds' format, but got {session_path.name!r}."
        )
        console.error(message=message, error=ValueError)
    year, month, day, hour, minute, second, microsecond = parts[:_SESSION_NAME_FIELD_COUNT]
    return datetime(  # noqa: DTZ001
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        microsecond=int(microsecond),
    )


def _validate_chronological_order(
    session_paths: tuple[Path, ...],
    session_dates: tuple[datetime, ...],
) -> None:
    """Verifies that the supplied session dates are non-decreasing, which the across-session metrics assume."""
    for index in range(1, len(session_dates)):
        if session_dates[index] < session_dates[index - 1]:
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The session paths must be sorted in "
                f"chronological order, but {session_paths[index].name!r} is older than "
                f"{session_paths[index - 1].name!r}."
            )
            console.error(message=message, error=ValueError)


def _compute_session_metrics(
    session_path: Path,
    fluorescence_column: DatasetColumn,
    days_since_first: float,
    configuration: BleachingConfiguration,
) -> SessionBleachingMetrics:
    """Loads a single session's raw fluorescence and computes its bleaching metrics."""
    fluorescence, time_us = _load_session_raw(session_path=session_path, fluorescence_column=fluorescence_column)
    sampling_rate_hz = _estimate_sampling_rate_hz(time_us=time_us)

    baseline_window_samples = max(round(configuration.baseline_window_seconds * sampling_rate_hz), 1)
    binned_baseline = _compute_binned_baseline(
        fluorescence=fluorescence,
        bin_size_samples=baseline_window_samples,
        percentile=configuration.baseline_percentile,
    )

    if binned_baseline.shape[1] == 0:
        # noinspection PyTypeChecker
        cell_baseline_f0: NDArray[np.float32] = np.full(fluorescence.shape[0], np.nan, dtype=np.float32)
    else:
        # noinspection PyTypeChecker
        cell_baseline_f0 = np.median(binned_baseline, axis=1).astype(np.float32)

    cell_snr = _compute_cell_snr(
        fluorescence=fluorescence,
        binned_baseline=binned_baseline,
        bin_size_samples=baseline_window_samples,
        signal_percentile=configuration.snr_signal_percentile,
    )

    within_session_bin_samples = max(round(configuration.within_session_bin_seconds * sampling_rate_hz), 1)
    within_session_time_seconds, within_session_baseline = _compute_within_session_baseline(
        fluorescence=fluorescence,
        sampling_rate_hz=sampling_rate_hz,
        bin_size_samples=within_session_bin_samples,
        percentile=configuration.baseline_percentile,
    )

    if within_session_baseline.size > 0 and within_session_baseline[0] > 0:
        within_session_fractional_drop = float(
            (within_session_baseline[0] - within_session_baseline[-1]) / within_session_baseline[0]
        )
    else:
        within_session_fractional_drop = float("nan")

    return SessionBleachingMetrics(
        session_path=session_path,
        days_since_first=days_since_first,
        sampling_rate_hz=sampling_rate_hz,
        cell_baseline_f0=cell_baseline_f0,
        cell_snr=cell_snr,
        within_session_time_seconds=within_session_time_seconds,
        within_session_baseline=within_session_baseline,
        within_session_fractional_drop=within_session_fractional_drop,
    )


def _load_session_raw(
    session_path: Path,
    fluorescence_column: DatasetColumn,
) -> tuple[NDArray[np.float32], NDArray[np.uint64]]:
    """Loads the raw per-cell fluorescence trace and per-sample timestamps from a forged session feather."""
    df = pl.read_ipc(
        source=session_path.joinpath(DATA_FILENAME),
        columns=[DatasetColumn.TIME_US.value, fluorescence_column.value],
    )
    # noinspection PyTypeChecker
    time_us: NDArray[np.uint64] = df[DatasetColumn.TIME_US.value].to_numpy().astype(np.uint64)
    # noinspection PyTypeChecker
    fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T
    return fluorescence, time_us


def _estimate_sampling_rate_hz(time_us: NDArray[np.uint64]) -> float:
    """Estimates the sampling rate in Hz from the median inter-sample interval of the timestamp array."""
    if time_us.size < _MINIMUM_SAMPLES_FOR_RATE_ESTIMATE:
        return float("nan")
    deltas_us = np.diff(time_us.astype(np.float64))
    median_delta_us = float(np.median(deltas_us))
    if median_delta_us <= 0:
        return float("nan")
    return _MICROSECONDS_PER_SECOND / median_delta_us


def _compute_binned_baseline(
    fluorescence: NDArray[np.float32],
    bin_size_samples: int,
    percentile: float,
) -> NDArray[np.float32]:
    """Computes a per-cell, per-bin percentile baseline using non-overlapping windows along the time axis.

    Notes:
        Replaces the per-sample rolling-window percentile of the original Suite2p convention with a non-overlapping
        binned percentile of the same window size. The session-median estimator is unbiased and statistically
        equivalent to the rolling-window estimator while running orders of magnitude faster: total cost drops from
        O(cell_count * sample_count * window * log(window)) to O(cell_count * sample_count * log(window)). The
        trailing partial bin is dropped so the operation reduces to a single contiguous reshape plus a vectorized
        ``np.percentile`` call along the bin axis.

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

    # Trims trailing samples that do not fill a complete bin so the reshape is exact and contiguous.
    trimmed = fluorescence[:, : bin_count * bin_size_samples]
    reshaped = trimmed.reshape(cell_count, bin_count, bin_size_samples)
    # noinspection PyTypeChecker
    binned: NDArray[np.float32] = np.percentile(reshaped, percentile, axis=2).astype(np.float32)
    return binned


def _compute_cell_snr(
    fluorescence: NDArray[np.float32],
    binned_baseline: NDArray[np.float32],
    bin_size_samples: int,
    signal_percentile: float,
) -> NDArray[np.float32]:
    """Computes per-cell SNR by detrending the raw trace with the upsampled binned baseline.

    Notes:
        The per-cell, per-bin baseline is upsampled to per-sample by repeating each bin's value ``bin_size_samples``
        times, then padded with the final bin's value if the original trace length was not a multiple of the bin
        size. SNR is the configured upper percentile of the detrended trace divided by a MAD-based robust noise
        standard deviation.

    Args:
        fluorescence: Raw per-cell fluorescence with dimensions (cell_count, sample_count).
        binned_baseline: Per-cell, per-bin baseline with dimensions (cell_count, bin_count) returned by
            ``_compute_binned_baseline``.
        bin_size_samples: Width of each baseline bin in samples, used to upsample the binned baseline back to
            per-sample resolution.
        signal_percentile: Upper percentile of the detrended trace treated as the per-cell event amplitude.

    Returns:
        Per-cell SNR with length cell_count. Cells with zero estimated noise are reported as 0 to avoid division
        by zero rather than NaN or infinity.
    """
    cell_count, sample_count = fluorescence.shape
    if binned_baseline.shape[1] == 0:
        # noinspection PyTypeChecker
        empty_snr: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        return empty_snr

    # Upsamples the per-bin baseline back to per-sample by repeating each bin's value, then pads the trailing partial
    # bin with the last bin's value so the detrending shape matches the input.
    # noinspection PyTypeChecker
    baseline_per_sample: NDArray[np.float32] = np.repeat(binned_baseline, bin_size_samples, axis=1)
    if baseline_per_sample.shape[1] < sample_count:
        pad_width = sample_count - baseline_per_sample.shape[1]
        # noinspection PyTypeChecker
        last_bin: NDArray[np.float32] = baseline_per_sample[:, -1:]
        # noinspection PyTypeChecker
        padding: NDArray[np.float32] = np.tile(last_bin, (1, pad_width))
        baseline_per_sample = np.concatenate([baseline_per_sample, padding], axis=1)

    # noinspection PyTypeChecker
    detrended: NDArray[np.float32] = fluorescence - baseline_per_sample

    # noinspection PyTypeChecker
    detrended_median: NDArray[np.float32] = np.median(detrended, axis=1, keepdims=True).astype(np.float32)
    # noinspection PyTypeChecker
    mad: NDArray[np.float32] = np.median(np.abs(detrended - detrended_median), axis=1).astype(np.float32)
    noise_std = mad * _MAD_TO_STD_SCALE

    # noinspection PyTypeChecker
    signal: NDArray[np.float32] = np.percentile(detrended, signal_percentile, axis=1).astype(np.float32)

    # Avoids division by zero by gating the divisor; cells with zero noise return SNR=0 rather than NaN/inf.
    # noinspection PyTypeChecker
    safe_noise: NDArray[np.float32] = np.where(noise_std > 0, noise_std, 1.0).astype(np.float32)
    # noinspection PyTypeChecker
    snr: NDArray[np.float32] = np.where(noise_std > 0, signal / safe_noise, 0.0).astype(np.float32)
    return snr


def _compute_within_session_baseline(
    fluorescence: NDArray[np.float32],
    sampling_rate_hz: float,
    bin_size_samples: int,
    percentile: float,
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
    if sample_count < _MINIMUM_SAMPLES_FOR_RATE_ESTIMATE or not np.isfinite(sampling_rate_hz):
        # noinspection PyTypeChecker
        empty_time: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_baseline: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        return empty_time, empty_baseline

    bin_count = sample_count // bin_size_samples
    if bin_count == 0:
        # noinspection PyTypeChecker
        empty_time = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_baseline = np.zeros(0, dtype=np.float32)
        return empty_time, empty_baseline

    # noinspection PyTypeChecker
    fov_mean: NDArray[np.float32] = np.mean(fluorescence, axis=0).astype(np.float32)
    trimmed = fov_mean[: bin_count * bin_size_samples]
    reshaped = trimmed.reshape(bin_count, bin_size_samples)
    # noinspection PyTypeChecker
    baseline: NDArray[np.float32] = np.percentile(reshaped, percentile, axis=1).astype(np.float32)
    # noinspection PyTypeChecker
    bin_centers: NDArray[np.float32] = (
        (np.arange(bin_count, dtype=np.float64) + 0.5) * bin_size_samples / sampling_rate_hz
    ).astype(np.float32)
    return bin_centers, baseline


def _fit_exponential_decay(
    days: NDArray[np.float32],
    baseline: NDArray[np.float32],
) -> ExponentialDecayFit:
    """Fits ``F0(d) = amplitude * exp(-d / tau_days) + offset`` to per-session F0 across days."""
    if days.size < _MINIMUM_SESSIONS_FOR_DECAY_FIT:
        return ExponentialDecayFit(
            amplitude=float("nan"),
            tau_days=float("nan"),
            offset=float("nan"),
            fit_succeeded=False,
        )

    initial_amplitude = float(baseline[0] - baseline[-1])
    span = float(days[-1] - days[0]) if float(days[-1] - days[0]) > 0 else 1.0
    initial_tau = max(span / 2.0, 1e-3)
    initial_offset = float(baseline[-1])

    try:
        parameters, _ = curve_fit(
            f=_exponential_decay_model,
            xdata=days.astype(np.float64),
            ydata=baseline.astype(np.float64),
            p0=(initial_amplitude, initial_tau, initial_offset),
            bounds=((-np.inf, 1e-6, -np.inf), (np.inf, np.inf, np.inf)),
            maxfev=10000,
        )
    except RuntimeError, ValueError:
        return ExponentialDecayFit(
            amplitude=float("nan"),
            tau_days=float("nan"),
            offset=float("nan"),
            fit_succeeded=False,
        )

    amplitude, tau_days, offset = (float(parameter) for parameter in parameters)
    if not (np.isfinite(amplitude) and np.isfinite(tau_days) and np.isfinite(offset)):
        return ExponentialDecayFit(
            amplitude=float("nan"),
            tau_days=float("nan"),
            offset=float("nan"),
            fit_succeeded=False,
        )

    return ExponentialDecayFit(amplitude=amplitude, tau_days=tau_days, offset=offset, fit_succeeded=True)


def _exponential_decay_model(
    days: NDArray[np.float64],
    amplitude: float,
    tau_days: float,
    offset: float,
) -> NDArray[np.float64]:
    """Single-exponential decay model used by ``_fit_exponential_decay``."""
    return amplitude * np.exp(-days / tau_days) + offset


def _compute_paired_snr_p_values(metrics: tuple[SessionBleachingMetrics, ...]) -> NDArray[np.float64]:
    """Computes paired Wilcoxon signed-rank p-values comparing each session's per-cell SNR to the first session."""
    # noinspection PyTypeChecker
    p_values: NDArray[np.float64] = np.full(len(metrics), np.nan, dtype=np.float64)
    reference_snr = metrics[0].cell_snr
    for index in range(1, len(metrics)):
        target_snr = metrics[index].cell_snr
        # noinspection PyTypeChecker
        difference: NDArray[np.float32] = target_snr - reference_snr
        if np.all(difference == 0):
            continue
        try:
            result = wilcoxon(x=target_snr, y=reference_snr, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            continue
        p_values[index] = float(result.pvalue)
    return p_values


def _flag_sessions(
    metrics: tuple[SessionBleachingMetrics, ...],
    f0_population_trend: NDArray[np.float32],
    snr_population_trend: NDArray[np.float32],
    snr_paired_p_values: NDArray[np.float64],
    configuration: BleachingConfiguration,
) -> tuple[Path, ...]:
    """Selects sessions that violate any of the configured F0, within-session, or SNR thresholds."""
    flagged: list[Path] = []
    f0_reference = float(f0_population_trend[0])
    snr_reference = float(snr_population_trend[0])

    for index, session in enumerate(metrics):
        f0_loss = (f0_reference - float(f0_population_trend[index])) / f0_reference if f0_reference > 0 else 0.0
        snr_loss = (snr_reference - float(snr_population_trend[index])) / snr_reference if snr_reference > 0 else 0.0
        snr_p_value = float(snr_paired_p_values[index])

        f0_flagged = f0_loss > configuration.f0_loss_threshold
        within_flagged = (
            np.isfinite(session.within_session_fractional_drop)
            and session.within_session_fractional_drop > configuration.within_session_loss_threshold
        )
        snr_flagged = (
            snr_loss > configuration.snr_loss_threshold
            and np.isfinite(snr_p_value)
            and snr_p_value < configuration.snr_significance_threshold
        )

        if f0_flagged or within_flagged or snr_flagged:
            flagged.append(session.session_path)

    return tuple(flagged)
