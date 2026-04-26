"""Quantifies photobleaching across a chronologically ordered set of two-photon imaging sessions for the same animal.

Implements the canonical three-metric protocol for chronic GCaMP imaging: per-cell session-median baseline F0
trend across days fit to a single exponential, within-session bleaching slope, and per-cell signal-to-noise change
on the multi-recording registered cell intersection. Per-step methodological references are attached to the
top-level functions that implement each step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import datetime
from itertools import pairwise
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ataraxis_base_utilities import LogLevel, console

from ..forging import DATA_FILENAME, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MICROSECONDS_PER_SECOND: float = 1.0e6
"""Conversion factor between microsecond timestamps and seconds."""
_SECONDS_PER_DAY: float = 86400.0
"""Conversion factor between seconds and days."""
_MAD_TO_STD_SCALE: np.float32 = np.float32(1.4826)
"""Scaling that maps the median absolute deviation of Gaussian noise to its standard deviation."""
_SESSION_NAME_FIELD_COUNT: int = 7
"""Number of dash-separated fields produced by the canonical session timestamp format."""
_MINIMUM_SESSIONS_FOR_DECAY_FIT: int = 3
"""Minimum number of sessions required to fit a single-exponential decay model."""
_MINIMUM_SESSIONS_FOR_EVALUATION: int = 2
"""Minimum number of sessions required to evaluate any across-session bleaching metric."""
_MINIMUM_SAMPLES_FOR_RATE_ESTIMATE: int = 2
"""Minimum number of timestamp samples required to estimate the inter-sample sampling rate."""


@dataclass(frozen=True, slots=True)
class BleachingConfiguration:
    """Defines configuration parameters for the chronic photobleaching evaluation protocol."""

    baseline_percentile: float = 8.0
    """Per-cell baseline percentile applied within each non-overlapping baseline window. Suite2p convention is the 8th
    percentile, which is robust to large transients while still tracking slow drift."""
    baseline_window_seconds: float = 60.0
    """Width of each non-overlapping baseline window in seconds. Suite2p default for chronic imaging is 60 seconds."""
    within_session_bin_seconds: float = 10.0
    """Bin width in seconds for the within-session bleaching trace. Coarser bins suppress transient leakage into the
    baseline estimate."""
    snr_signal_percentile: float = 95.0
    """Percentile of the detrended trace treated as the per-cell event amplitude when computing SNR."""
    f0_loss_threshold: float = 0.30
    """Fractional drop in median across-session F0 (relative to the first session) above which a session is flagged
    as chronically bleached."""
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
    the per-bin baseline percentile."""
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
    """Determines whether scipy.optimize.curve_fit converged on a finite, in-bounds solution."""


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
        """Plots the population-median per-session F0 trend, the exponential fit, and per-cell baseline distributions.

        Returns:
            A matplotlib Figure showing the across-session F0 trend.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        days = np.array([session.days_since_first for session in self.sessions], dtype=np.float32)

        # Computes a non-zero box width that scales with the smallest day step so adjacent boxes do not overlap.
        minimum_day_step = float(np.diff(days).min()) if len(days) > 1 else 1.0
        box_width = 0.4 * max(minimum_day_step, 0.1)

        # Draws the per-cell distributions as boxplots so the population spread is visible alongside the median trend.
        boxplot_data = [session.cell_baseline_f0 for session in self.sessions]
        axes.boxplot(boxplot_data, positions=days, widths=box_width, showfliers=False)

        # Overlays the population-median trend used for the exponential fit.
        axes.plot(days, self.f0_population_trend, marker="o", color="tab:blue", linewidth=1.5, label="Median F0")

        # Draws the fitted exponential when the fit converged.
        if self.f0_decay_fit.fit_succeeded:
            dense_days = np.linspace(days.min(), days.max(), num=200)
            fit_curve = (
                self.f0_decay_fit.amplitude * np.exp(-dense_days / self.f0_decay_fit.tau_days)
                + self.f0_decay_fit.offset
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
            # Guards against division by zero when the report contains a single session.
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
) -> BleachingReport:
    """Quantifies photobleaching across the supplied chronologically ordered sessions for a single animal.

    Notes:
        Sessions must share a multi-recording cindra registration so the cell axis is comparable across days. The
        function rejects mismatched cell counts. By default, the multi-recording raw cell-fluorescence column is used,
        which carries the cell correspondence required for a defensible across-session comparison; passing the
        single-recording column is supported but only yields valid within-session metrics. Sessions are processed
        sequentially so the heavy per-cell percentile and SNR kernels can saturate every available CPU core via
        Numba's thread pool; a per-session progress line is emitted as each session completes.

    References:
        Multi-day registered cell intersection for longitudinal comparisons:
            Ziv et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience.
            https://doi.org/10.1038/nn.3329
            Rubin et al. (2015). Hippocampal ensemble dynamics timestamp events in long-term memory. eLife.
            https://doi.org/10.7554/eLife.12247
        Standardized longitudinal-imaging quality-control framework that motivates the combined-flagging strategy:
            de Vries et al. (2020). A large-scale standardized physiological survey reveals functional
            organization of the mouse visual cortex. Nature Neuroscience.
            https://doi.org/10.1038/s41593-019-0550-9

    Args:
        session_paths: Chronologically ordered tuple of session directory paths. Sessions are validated to be in
            non-decreasing date order via the canonical session-name timestamp.
        fluorescence_column: The raw cell-fluorescence column to load. Must be one of the raw cindra columns;
            baseline-corrected dF/F0 columns are rejected because bleaching is defined on raw signal decay.
        configuration: Bleaching evaluation parameters. Uses defaults if None.

    Returns:
        A BleachingReport containing per-session metrics, the across-session F0 decay fit, paired SNR Wilcoxon
        results, and the list of sessions that exceeded any threshold.
    """
    if configuration is None:
        configuration = BleachingConfiguration()

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

    # Processes sessions sequentially so the per-cell numba kernels can use every available core for one session at
    # a time. Spreading sessions across threads here would just oversubscribe the CPU and starve the kernels of
    # threads, while also blocking progress reporting until every session finished.
    session_count = len(session_paths)
    metrics: list[SessionBleachingMetrics] = []
    for index, (session_path, days_since_first) in enumerate(
        zip(session_paths, days_since_first_list, strict=True),
    ):
        console.echo(
            message=f"Evaluating bleaching for session {index + 1}/{session_count}: {session_path.name}",
            level=LogLevel.INFO,
        )
        session_metrics = _compute_session_metrics(
            session_path=session_path,
            fluorescence_column=fluorescence_column,
            days_since_first=days_since_first,
            configuration=configuration,
        )
        metrics.append(session_metrics)
        console.echo(
            message=(
                f"  session {index + 1}/{session_count} done: "
                f"cells={session_metrics.cell_baseline_f0.shape[0]}, "
                f"sampling_rate={session_metrics.sampling_rate_hz:.2f} Hz, "
                f"within-session drop={session_metrics.within_session_fractional_drop:.1%}"
            ),
            level=LogLevel.INFO,
        )

    cell_count_reference = int(metrics[0].cell_baseline_f0.shape[0])
    for session_metrics in metrics[1:]:
        other_count = int(session_metrics.cell_baseline_f0.shape[0])
        if other_count != cell_count_reference:
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The cell count must match across all "
                f"sessions for the multi-recording registered comparison, but session "
                f"{session_metrics.session_path.name!r} has {other_count} cells while the first session has "
                f"{cell_count_reference}."
            )
            console.error(message=message, error=ValueError)

    metrics_tuple = tuple(metrics)

    # noinspection PyTypeChecker
    f0_population_trend: NDArray[np.float32] = np.fromiter(
        (np.median(session.cell_baseline_f0) for session in metrics_tuple),
        dtype=np.float32,
        count=len(metrics_tuple),
    )
    # noinspection PyTypeChecker
    days_array: NDArray[np.float32] = np.fromiter(
        (session.days_since_first for session in metrics_tuple),
        dtype=np.float32,
        count=len(metrics_tuple),
    )
    decay_fit = _fit_exponential_decay(days=days_array, baseline=f0_population_trend)

    f0_fractional_loss = float((f0_population_trend[0] - f0_population_trend[-1]) / f0_population_trend[0])

    # noinspection PyTypeChecker
    snr_population_trend: NDArray[np.float32] = np.fromiter(
        (np.median(session.cell_snr) for session in metrics_tuple),
        dtype=np.float32,
        count=len(metrics_tuple),
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
        cell_count=cell_count_reference,
        f0_population_trend=f0_population_trend,
        f0_decay_fit=decay_fit,
        f0_fractional_loss=f0_fractional_loss,
        snr_population_trend=snr_population_trend,
        snr_paired_p_values=snr_paired_p_values,
        flagged_sessions=flagged_sessions,
        configuration=configuration,
    )


def _parse_session_date(session_path: Path) -> datetime:
    """Parses the canonical 'YYYY-MM-DD-HH-MM-SS-microseconds' session-directory name into a datetime instance.

    Args:
        session_path: Path whose final component encodes the session timestamp in the canonical dash-separated format.

    Returns:
        A datetime built from the parsed year, month, day, hour, minute, second, and microsecond fields.
    """
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
    """Verifies that the supplied session dates are non-decreasing, which the across-session metrics assume.

    Args:
        session_paths: Session directory paths in the order they were supplied; used in the error message.
        session_dates: Parsed session timestamps in the same order as ``session_paths``.
    """
    for previous_index, (previous_date, current_date) in enumerate(pairwise(session_dates)):
        if current_date < previous_date:
            current_index = previous_index + 1
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The session paths must be sorted in "
                f"chronological order, but {session_paths[current_index].name!r} is older than "
                f"{session_paths[previous_index].name!r}."
            )
            console.error(message=message, error=ValueError)


def _compute_session_metrics(
    session_path: Path,
    fluorescence_column: DatasetColumn,
    days_since_first: float,
    configuration: BleachingConfiguration,
) -> SessionBleachingMetrics:
    """Loads a single session's raw fluorescence and computes its bleaching metrics.

    Args:
        session_path: Path to the forged session directory containing the data feather.
        fluorescence_column: The raw cell-fluorescence column to load from the feather.
        days_since_first: Calendar days elapsed since the first session in the evaluation set.
        configuration: Bleaching evaluation parameters that drive window sizes and percentile choices.

    Returns:
        A SessionBleachingMetrics instance populated with the per-cell baseline, per-cell SNR, and within-session
        baseline trace for the session.
    """
    fluorescence, time_us = _load_session_raw(session_path=session_path, fluorescence_column=fluorescence_column)
    sampling_rate_hz = _estimate_sampling_rate_hz(time_us=time_us)

    baseline_window_samples = max(round(configuration.baseline_window_seconds * sampling_rate_hz), 1)
    cell_count = fluorescence.shape[0]
    bin_count = fluorescence.shape[1] // baseline_window_samples

    if bin_count == 0:
        # noinspection PyTypeChecker
        cell_baseline_f0: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        cell_snr: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
    else:
        binned_baseline = _compute_binned_baseline(
            fluorescence=fluorescence,
            bin_size_samples=baseline_window_samples,
            percentile=configuration.baseline_percentile,
        )
        # noinspection PyTypeChecker
        cell_baseline_f0 = np.empty(cell_count, dtype=np.float32)
        _per_cell_median_along_axis1(matrix=binned_baseline, output=cell_baseline_f0)
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
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Loads the raw per-cell fluorescence trace and per-sample timestamps from a forged session feather.

    Notes:
        The fluorescence column is stored as a polars list-of-float32, one list per sample. Converting via
        ``Series.to_list()`` and ``np.array`` materializes a Python list of lists for every sample, which is
        single-threaded, GIL-bound, and dominates load time for multi-thousand-cell sessions. Exploding the list
        column to a flat fp32 series and reshaping in NumPy stays in compiled code and runs roughly an order of
        magnitude faster while producing the same (cell_count, sample_count) C-contiguous layout.

    Args:
        session_path: Path to the forged session directory containing the data feather.
        fluorescence_column: The raw cell-fluorescence column to load.

    Returns:
        A tuple containing the (cell_count, sample_count) fp32 fluorescence array and the per-sample int64
        microsecond timestamps.
    """
    df = pl.read_ipc(
        source=session_path.joinpath(DATA_FILENAME),
        columns=[DatasetColumn.TIME_US.value, fluorescence_column.value],
    )
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy().astype(np.int64, copy=False)

    sample_count = df.height
    flat = df[fluorescence_column.value].explode().to_numpy()
    if flat.dtype != np.float32:
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
    deltas_us = np.diff(time_us)
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
        binned percentile of the same window size. Dispatches to a numba parallel-over-cells kernel that performs
        the percentile via in-place quickselect; the previous ``np.percentile`` call was single-threaded and
        dominated session compute on multi-thousand-cell traces.

    References:
        Suite2p baseline convention (8th-percentile baseline within a 60-second window):
            Pachitariu et al. (2017). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. bioRxiv.
            https://doi.org/10.1101/061507

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
    signal_percentile: float,
) -> NDArray[np.float32]:
    """Computes per-cell SNR by detrending the raw trace with the per-bin baseline.

    Notes:
        Hands the heavy work to a single numba parallel-over-cells kernel that computes the detrended trace into a
        thread-local scratch buffer, runs three quickselects (median for the noise center, MAD median, signal
        percentile), and writes the SNR. Avoiding the explicit ``np.repeat`` upsample saves a (cell_count *
        sample_count) allocation and a memory-bound pass over it, and replacing the three single-threaded
        ``np.median`` / ``np.percentile`` reductions with one parallel kernel scales the operation across cores.

    References:
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
    percentile: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the within-session FOV-mean baseline percentile across non-overlapping time bins.

    Notes:
        Reduces what was a Python loop over per-bin masks to a single contiguous reshape plus one vectorized
        ``np.percentile`` call along the bin axis, which is orders of magnitude faster on multi-thousand-sample
        traces.

    References:
        Within-session bleaching control common to the Dombeck/Tank chronic-imaging lineage:
            Sheffield & Dombeck (2015). Calcium transient prevalence across the dendritic arbour predicts place
            field properties. Nature. https://doi.org/10.1038/nature14066
            Driscoll et al. (2017). Dynamic reorganization of neuronal activity patterns in parietal cortex.
            Cell. https://doi.org/10.1016/j.cell.2017.05.021

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

    fov_mean = np.mean(fluorescence, axis=0).astype(np.float32, copy=False)
    trimmed = fov_mean[: bin_count * bin_size_samples]
    reshaped = trimmed.reshape(bin_count, bin_size_samples)
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
        scratch = np.empty(bin_size_samples, dtype=np.float32)
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
                    upper_value = min(upper_value, scratch[scan_index])
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
        scratch = np.empty(column_count, dtype=np.float32)
        for column_index in range(column_count):
            scratch[column_index] = matrix[row_index, column_index]
        lower_value = _quickselect_inplace(buffer=scratch, target_index=lower_index)
        if not is_even:
            output[row_index] = lower_value
        else:
            upper_value = scratch[lower_index + 1]
            for scan_index in range(lower_index + 2, column_count):
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
        scratch = np.empty(sample_count, dtype=np.float32)

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
                upper_value = min(upper_value, scratch[scan_index])
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
        # selection, then stages the pivot at the right end so the Lomuto scan can run over [left, right - 1] with
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


def _fit_exponential_decay(
    days: NDArray[np.float32],
    baseline: NDArray[np.float32],
) -> ExponentialDecayFit:
    """Fits ``F0(d) = amplitude * exp(-d / tau_days) + offset`` to per-session F0 across days.

    Notes:
        Returns a sentinel ExponentialDecayFit with ``fit_succeeded=False`` and NaN parameters when fewer than the
        required number of sessions are supplied, when ``curve_fit`` raises, or when any fitted parameter is
        non-finite.

    Args:
        days: Per-session day offsets relative to the first session.
        baseline: Per-session population-median F0 values aligned with ``days``.

    Returns:
        An ExponentialDecayFit holding the fitted amplitude, tau, and offset, or the failure sentinel described
        above.
    """
    if days.size < _MINIMUM_SESSIONS_FOR_DECAY_FIT:
        return _failed_decay_fit()

    initial_amplitude = float(baseline[0] - baseline[-1])
    day_span = float(days[-1] - days[0])
    if day_span <= 0:
        day_span = 1.0
    initial_tau = max(day_span / 2.0, 1e-3)
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
        return _failed_decay_fit()

    amplitude, tau_days, offset = (float(parameter) for parameter in parameters)
    if not (np.isfinite(amplitude) and np.isfinite(tau_days) and np.isfinite(offset)):
        return _failed_decay_fit()

    return ExponentialDecayFit(amplitude=amplitude, tau_days=tau_days, offset=offset, fit_succeeded=True)


def _failed_decay_fit() -> ExponentialDecayFit:
    """Returns the sentinel ExponentialDecayFit used when the decay fit cannot be produced."""
    return ExponentialDecayFit(
        amplitude=float("nan"),
        tau_days=float("nan"),
        offset=float("nan"),
        fit_succeeded=False,
    )


def _exponential_decay_model(
    days: NDArray[np.float64],
    amplitude: float,
    tau_days: float,
    offset: float,
) -> NDArray[np.float64]:
    """Single-exponential decay model used by ``_fit_exponential_decay``.

    Args:
        days: Day offsets at which to evaluate the model.
        amplitude: Decaying-component amplitude.
        tau_days: Decay time constant in days.
        offset: Asymptotic baseline component.

    Returns:
        The model values at each day in ``days``.
    """
    return amplitude * np.exp(-days / tau_days) + offset


def _compute_paired_snr_p_values(metrics: tuple[SessionBleachingMetrics, ...]) -> NDArray[np.float64]:
    """Computes paired Wilcoxon signed-rank p-values comparing each session's per-cell SNR to the first session.

    Args:
        metrics: Per-session metrics in chronological order; the first entry serves as the reference.

    Returns:
        An array with length len(metrics) holding the per-session paired Wilcoxon p-values. The first entry is NaN
        because it would compare the reference to itself, and entries are also NaN for sessions whose SNR is
        identical to the reference or for which the test raises.
    """
    # noinspection PyTypeChecker
    p_values: NDArray[np.float64] = np.full(len(metrics), np.nan, dtype=np.float64)
    reference_snr = metrics[0].cell_snr
    for index in range(1, len(metrics)):
        target_snr = metrics[index].cell_snr
        if np.all(target_snr == reference_snr):
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
    """Selects sessions that violate any of the configured F0, within-session, or SNR thresholds.

    Args:
        metrics: Per-session metrics in chronological order.
        f0_population_trend: Population-median per-session F0 aligned with ``metrics``.
        snr_population_trend: Population-median per-session SNR aligned with ``metrics``.
        snr_paired_p_values: Paired Wilcoxon p-values aligned with ``metrics``; the first entry is NaN.
        configuration: Bleaching evaluation parameters that supply the threshold values.

    Returns:
        Paths of every session that exceeded at least one configured threshold, in chronological order.
    """
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
