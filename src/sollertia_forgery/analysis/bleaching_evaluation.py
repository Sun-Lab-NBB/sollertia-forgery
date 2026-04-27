"""Quantifies photobleaching across a chronologically ordered set of two-photon imaging sessions for the same animal.

Implements the canonical three-metric protocol for chronic GCaMP imaging: per-cell session-median baseline
fluorescence (estimated as a low percentile of the raw trace within a baseline window) trend across days fit to a
single exponential, within-session bleaching slope, and per-cell signal-to-noise change on the multi-recording
registered cell intersection. Per-step methodological references are attached to the top-level functions that
implement each step.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple
from itertools import pairwise
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl
from scipy.stats import wilcoxon
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp, interval_to_rate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig

from ..shared_assets import DatasetData, DatasetFiles, DatasetAnimal, DatasetColumn

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MAD_TO_STD_SCALE: np.float32 = np.float32(1.4826)
"""Scaling that maps the median absolute deviation of Gaussian noise to its standard deviation."""
_MINIMUM_SESSIONS_FOR_DECAY_FIT: int = 3
"""Minimum number of sessions required to fit a single-exponential decay model."""
_MINIMUM_SESSIONS_FOR_EVALUATION: int = 2
"""Minimum number of sessions required to evaluate any across-session bleaching metric."""
_MINIMUM_SAMPLES_FOR_RATE_ESTIMATE: int = 2
"""Minimum number of timestamp samples required to estimate the inter-sample sampling rate."""
_SESSION_TIMESTAMP_FORMAT: str = "%Y-%m-%d-%H-%M-%S-%f"
"""``strptime`` format string for the canonical session-directory timestamp."""


class BleachingColumn(StrEnum):
    """Defines every column written to the per-session ``bleaching.feather`` table inside an animal directory."""

    SESSION = "session"
    """Session directory name, the canonical 'YYYY-MM-DD-HH-MM-SS-microseconds' timestamp. Primary key."""
    DAYS_SINCE_FIRST = "days_since_first"
    """Calendar days elapsed between the first session in the evaluation set and this session."""
    SAMPLING_RATE_HZ = "sampling_rate_hz"
    """Effective fluorescence sampling rate in Hz, derived from the median inter-sample period."""
    CELL_BASELINE_FLUORESCENCE = "cell_baseline_fluorescence"
    """Per-cell session-median baseline fluorescence stored as a list of cell_count fp32 values."""
    CELL_SNR = "cell_snr"
    """Per-cell signal-to-noise ratio stored as a list of cell_count fp32 values."""
    WITHIN_SESSION_TIME_SECONDS = "within_session_time_seconds"
    """Bin-center timestamps for the within-session baseline trace in seconds. Variable length per session."""
    WITHIN_SESSION_BASELINE = "within_session_baseline"
    """Within-session FOV-mean baseline trace, parallel to ``within_session_time_seconds``."""
    WITHIN_SESSION_FRACTIONAL_DROP = "within_session_fractional_drop"
    """Fraction by which the within-session baseline trace drops from its first to its last bin. NaN allowed."""
    POPULATION_BASELINE_FLUORESCENCE = "population_baseline_fluorescence"
    """Precomputed population-median baseline fluorescence; equals the median of ``cell_baseline_fluorescence``."""
    POPULATION_SNR = "population_snr"
    """Precomputed population-median SNR; equals the median of ``cell_snr``."""
    SNR_PAIRED_P_VALUE = "snr_paired_p_value"
    """Paired Wilcoxon signed-rank p-value comparing this session's per-cell SNR distribution to the first session.
    The first row is NaN by construction (self-comparison) and is not a test failure."""
    FLAGGED = "flagged"
    """Boolean flag combining the baseline-fluorescence-loss, within-session, and SNR criteria evaluated at the
    configuration thresholds active when the report was generated."""


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


class _SessionRow(NamedTuple):
    """Internal per-session computed values used to assemble one row of the bleaching feather.

    Used only inside ``BleachingReport.evaluate``; not exposed in the public API. Values produced by
    ``_compute_session_row`` flow directly into the polars DataFrame without an intermediate dataclass wrapper.
    """

    sampling_rate_hz: float
    cell_baseline_fluorescence: NDArray[np.float32]
    cell_snr: NDArray[np.float32]
    within_session_time_seconds: NDArray[np.float32]
    within_session_baseline: NDArray[np.float32]
    within_session_fractional_drop: float


@dataclass(frozen=True, slots=True)
class ExponentialDecayFit:
    """Stores the result of fitting ``amplitude * exp(-d / tau_days) + offset`` (with ``d`` in days since the first
    session) to the per-session population-median baseline fluorescence.
    """

    amplitude: float
    """Decaying-component amplitude in raw fluorescence units."""
    tau_days: float
    """Decay time constant in days. NaN when the fit failed or fewer than the required number of sessions were
    available."""
    offset: float
    """Asymptotic baseline component in raw fluorescence units."""
    fit_succeeded: bool
    """Determines whether scipy.optimize.curve_fit converged on a finite, in-bounds solution."""


@dataclass
class BleachingSummary(YamlConfig):
    """Animal-level YAML companion to ``bleaching.feather`` carrying only fields not derivable from the table.

    The per-session list-typed arrays, precomputed scalar trends, the paired Wilcoxon p-values, and the combined
    flag column all live in ``bleaching.feather``. Quantities derivable from those columns (cell count,
    fractional loss) are exposed as properties on ``BleachingReport`` and are not duplicated here.
    """

    configuration: BleachingConfiguration
    """The configuration that produced the report. Load-bearing because the flag column was computed against its
    threshold values; reinterpreting the flags requires the configuration that produced them."""
    baseline_fluorescence_decay_fit: ExponentialDecayFit
    """Single-exponential decay fit applied to the population-median per-session baseline fluorescence. Persisted
    rather than rederived because ``scipy.optimize.curve_fit`` is non-trivial and may fail; the failure sentinel
    is itself information that has to be preserved."""


@dataclass(frozen=True, slots=True)
class BleachingReport:
    """Top-level container for a chronic photobleaching evaluation.

    Holds the per-session feather as a polars DataFrame (``table``) and the animal-level YAML as a
    ``BleachingSummary`` (``summary``). Every consumer access — figure regeneration, agent queries, threshold
    inspection, downstream analysis — goes through this class via typed property accessors over those two
    fields. The DataFrame is the per-session source of truth; the summary is the cross-session source of truth.
    No per-session dataclass intermediary exists; per-session arrays are read on demand from list-typed columns.
    """

    table: pl.DataFrame
    """The per-session table. One row per session in chronological order; columns enumerated by BleachingColumn.
    List-typed columns hold the per-cell and within-session arrays; scalar columns hold precomputed population
    trends, the paired SNR p-value, and the combined flag."""
    summary: BleachingSummary
    """The animal-level YAML wrapper holding configuration, the cross-session decay fit, the cross-session
    fractional loss, and audit metadata."""

    @classmethod
    def evaluate(
        cls,
        session_paths: tuple[Path, ...],
        *,
        configuration: BleachingConfiguration | None = None,
    ) -> BleachingReport:
        """Quantifies photobleaching across the supplied chronologically ordered sessions for a single animal.

        Notes:
            Operates exclusively on ``DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE``. The protocol's across-session
            per-cell comparisons (paired Wilcoxon SNR test, per-cell baseline trend, decay fit on the population
            median) require that cell index N denote the same neuron across every session in the evaluation set,
            which only the multi-recording cindra column carries. Single-recording fluorescence carries no cell
            correspondence across days and would silently produce mathematically valid but biologically meaningless
            paired statistics, so it is not exposed as an option. Sessions are processed sequentially so the heavy
            per-cell percentile and SNR kernels can saturate every available CPU core via Numba's thread pool; a
            per-session progress line is emitted as each session completes.

        References:
            Multi-day registered cell intersection for longitudinal comparisons:
                Ziv et al. (2013). Long-term dynamics of CA1 hippocampal place codes. Nature Neuroscience.
                https://doi.org/10.1038/nn.3329
                Rubin et al. (2015). Hippocampal ensemble dynamics timestamp events in long-term memory. eLife.
                https://doi.org/10.7554/eLife.12247
            Standardized longitudinal-imaging quality-control framework that motivates the combined-flagging
            strategy:
                de Vries et al. (2020). A large-scale standardized physiological survey reveals functional
                organization of the mouse visual cortex. Nature Neuroscience.
                https://doi.org/10.1038/s41593-019-0550-9

        Args:
            session_paths: Chronologically ordered tuple of session directory paths. Sessions are validated to be
                in non-decreasing date order via the canonical session-name timestamp.
            configuration: Bleaching evaluation parameters. Uses defaults if None.

        Returns:
            A BleachingReport whose ``table`` holds one row per session and whose ``summary`` holds the
            cross-session aggregates and configuration.
        """
        # Resolves the optional configuration into a non-None local with an explicit type so PyCharm narrows the
        # type downstream; the parameter itself stays Optional for the public signature.
        resolved_configuration: BleachingConfiguration = (
            configuration if configuration is not None else BleachingConfiguration()
        )

        if len(session_paths) < _MINIMUM_SESSIONS_FOR_EVALUATION:
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The protocol requires at least two "
                f"sessions, but got {len(session_paths)}."
            )
            console.error(message=message, error=ValueError)

        session_microseconds = tuple(_parse_session_microseconds(session_path=path) for path in session_paths)
        _validate_chronological_order(session_paths=session_paths, session_microseconds=session_microseconds)

        first_microseconds = session_microseconds[0]
        days_since_first_values = [
            float(
                convert_time(
                    time=session_us - first_microseconds,
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.DAY,
                    as_float=True,
                )
            )
            for session_us in session_microseconds
        ]

        # Accumulators for the per-session columns of the resulting feather. Sessions are processed sequentially
        # so the per-cell numba kernels can use every available core for one session at a time; spreading sessions
        # across threads here would just oversubscribe the CPU and starve the kernels of threads, while also
        # blocking progress reporting until every session finished.
        session_count = len(session_paths)
        session_names: list[str] = []
        sampling_rates: list[float] = []
        cell_baseline_arrays: list[NDArray[np.float32]] = []
        cell_snr_arrays: list[NDArray[np.float32]] = []
        within_session_time_arrays: list[NDArray[np.float32]] = []
        within_session_baseline_arrays: list[NDArray[np.float32]] = []
        within_session_drops: list[float] = []

        cell_count_reference: int | None = None
        with console.progress(
            total=session_count,
            description="Evaluating bleaching",
            unit="session",
        ) as progress_bar:
            for session_path in session_paths:
                row = _compute_session_row(
                    session_path=session_path,
                    configuration=resolved_configuration,
                )
                row_cell_count = int(row.cell_baseline_fluorescence.shape[0])
                if cell_count_reference is None:
                    cell_count_reference = row_cell_count
                elif row_cell_count != cell_count_reference:
                    message = (
                        f"Unable to evaluate bleaching across the supplied sessions. The cell count must match "
                        f"across all sessions for the multi-recording registered comparison, but session "
                        f"{session_path.name!r} has {row_cell_count} cells while the first session has "
                        f"{cell_count_reference}."
                    )
                    console.error(message=message, error=ValueError)

                session_names.append(session_path.name)
                sampling_rates.append(row.sampling_rate_hz)
                cell_baseline_arrays.append(row.cell_baseline_fluorescence)
                cell_snr_arrays.append(row.cell_snr)
                within_session_time_arrays.append(row.within_session_time_seconds)
                within_session_baseline_arrays.append(row.within_session_baseline)
                within_session_drops.append(row.within_session_fractional_drop)

                progress_bar.update()

        # Cross-session aggregates derived from the per-session arrays.
        population_baseline_values = [float(np.median(arr)) for arr in cell_baseline_arrays]
        population_snr_values = [float(np.median(arr)) for arr in cell_snr_arrays]

        # noinspection PyTypeChecker
        population_baseline_array: NDArray[np.float32] = np.array(population_baseline_values, dtype=np.float32)
        # noinspection PyTypeChecker
        population_snr_array: NDArray[np.float32] = np.array(population_snr_values, dtype=np.float32)
        # noinspection PyTypeChecker
        within_session_drop_array: NDArray[np.float32] = np.array(within_session_drops, dtype=np.float32)
        # noinspection PyTypeChecker
        days_array: NDArray[np.float32] = np.array(days_since_first_values, dtype=np.float32)

        decay_fit = _fit_exponential_decay(days=days_array, baseline=population_baseline_array)

        snr_paired_p_values = _compute_paired_snr_p_values(cell_snr_arrays=cell_snr_arrays)
        flagged_mask = _compute_flag_mask(
            population_baseline=population_baseline_array,
            population_snr=population_snr_array,
            snr_paired_p_values=snr_paired_p_values,
            within_session_drops=within_session_drop_array,
            configuration=resolved_configuration,
        )

        # Assembles the per-session feather. Equal-length cell columns are promoted by polars to
        # Array(Float32, cell_count); ragged within-session columns stay List(Float32). Each Series is constructed
        # with an explicit dtype so the schema is stable rather than inferred from data.
        table = pl.DataFrame(
            [
                pl.Series(name=BleachingColumn.SESSION.value, values=session_names, dtype=pl.Utf8),
                pl.Series(
                    name=BleachingColumn.DAYS_SINCE_FIRST.value, values=days_since_first_values, dtype=pl.Float32
                ),
                pl.Series(name=BleachingColumn.SAMPLING_RATE_HZ.value, values=sampling_rates, dtype=pl.Float64),
                pl.Series(
                    name=BleachingColumn.CELL_BASELINE_FLUORESCENCE.value,
                    values=cell_baseline_arrays,
                    dtype=pl.List(pl.Float32),
                ),
                pl.Series(
                    name=BleachingColumn.CELL_SNR.value,
                    values=cell_snr_arrays,
                    dtype=pl.List(pl.Float32),
                ),
                pl.Series(
                    name=BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value,
                    values=within_session_time_arrays,
                    dtype=pl.List(pl.Float32),
                ),
                pl.Series(
                    name=BleachingColumn.WITHIN_SESSION_BASELINE.value,
                    values=within_session_baseline_arrays,
                    dtype=pl.List(pl.Float32),
                ),
                pl.Series(
                    name=BleachingColumn.WITHIN_SESSION_FRACTIONAL_DROP.value,
                    values=within_session_drops,
                    dtype=pl.Float32,
                ),
                pl.Series(
                    name=BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value,
                    values=population_baseline_values,
                    dtype=pl.Float32,
                ),
                pl.Series(
                    name=BleachingColumn.POPULATION_SNR.value,
                    values=population_snr_values,
                    dtype=pl.Float32,
                ),
                pl.Series(
                    name=BleachingColumn.SNR_PAIRED_P_VALUE.value,
                    values=snr_paired_p_values,
                    dtype=pl.Float64,
                ),
                pl.Series(
                    name=BleachingColumn.FLAGGED.value,
                    values=flagged_mask,
                    dtype=pl.Boolean,
                ),
            ]
        )

        summary = BleachingSummary(
            configuration=resolved_configuration,
            baseline_fluorescence_decay_fit=decay_fit,
        )

        return cls(table=table, summary=summary)

    @classmethod
    def load(cls, animal: DatasetAnimal) -> BleachingReport:
        """Loads a previously saved bleaching report from the animal directory.

        Args:
            animal: The DatasetAnimal whose directory holds ``bleaching.yaml`` and ``bleaching.feather``.

        Returns:
            A BleachingReport equivalent to the one produced by ``evaluate``.
        """
        summary: BleachingSummary = BleachingSummary.from_yaml(file_path=animal.bleaching_path)
        table = pl.read_ipc(source=animal.bleaching_table_path, memory_map=True)
        return cls(table=table, summary=summary)

    def save(self, animal: DatasetAnimal) -> None:
        """Persists the report to ``bleaching.yaml`` and ``bleaching.feather`` inside the animal directory.

        Args:
            animal: The DatasetAnimal whose directory will hold the two artifacts.
        """
        self.summary.to_yaml(file_path=animal.bleaching_path)
        self.table.write_ipc(file=animal.bleaching_table_path)

    def summarize(self) -> str:
        """Returns a multi-line text summary of the bleaching evaluation.

        Notes:
            Covers all three protocol metrics — across-session baseline trend (chronic), within-session bleaching
            (acute), per-cell SNR change — with the configured thresholds and pass/fail status for each, plus a
            per-session detail table. Together with ``plot_baseline_trend``, ``plot_within_session``, and
            ``plot_snr_distributions``, this is jointly sufficient for scientific presentation, discussion, and
            publication of the animal's photobleaching state. Designed to be human-readable and parseable by
            downstream agents.

        Returns:
            A multi-line string. Use ``print_summary`` for direct console output.
        """
        configuration = self.summary.configuration
        decay_fit = self.summary.baseline_fluorescence_decay_fit
        table = self.table

        # noinspection PyTypeChecker
        days: NDArray[np.float32] = (
            table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        population_baseline: NDArray[np.float32] = (
            table[BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        population_snr: NDArray[np.float32] = (
            table[BleachingColumn.POPULATION_SNR.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        snr_p_values: NDArray[np.float64] = (
            table[BleachingColumn.SNR_PAIRED_P_VALUE.value].to_numpy().astype(np.float64, copy=False)
        )
        # noinspection PyTypeChecker
        within_drops: NDArray[np.float32] = (
            table[BleachingColumn.WITHIN_SESSION_FRACTIONAL_DROP.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        flagged_mask: NDArray[np.bool_] = table[BleachingColumn.FLAGGED.value].to_numpy()
        session_names = table[BleachingColumn.SESSION.value].to_list()

        session_count = len(days)
        cell_count = len(table[BleachingColumn.CELL_BASELINE_FLUORESCENCE.value][0]) if session_count > 0 else 0

        # Cross-session aggregates.
        f0_first = float(population_baseline[0]) if session_count > 0 else float("nan")
        f0_last = float(population_baseline[-1]) if session_count > 0 else float("nan")
        f0_loss = (f0_first - f0_last) / f0_first if f0_first > 0 else float("nan")
        snr_first = float(population_snr[0]) if session_count > 0 else float("nan")
        snr_last = float(population_snr[-1]) if session_count > 0 else float("nan")
        snr_loss = (snr_first - snr_last) / snr_first if snr_first > 0 else float("nan")
        finite_drops = within_drops[np.isfinite(within_drops)]
        drop_median = float(np.median(finite_drops)) if finite_drops.size > 0 else float("nan")
        drop_max = float(np.max(finite_drops)) if finite_drops.size > 0 else float("nan")

        # Per-criterion violation counts. Mirrors ``_compute_flag_mask``.
        within_violations = int(
            np.sum(np.isfinite(within_drops) & (within_drops > configuration.within_session_loss_threshold))
        )
        snr_loss_per_session = (
            (snr_first - population_snr) / snr_first if snr_first > 0 else np.full_like(population_snr, np.nan)
        )
        snr_violations_mask = (
            np.isfinite(snr_loss_per_session)
            & (snr_loss_per_session > configuration.snr_loss_threshold)
            & np.isfinite(snr_p_values)
            & (snr_p_values < configuration.snr_significance_threshold)
        )
        snr_violations = int(np.sum(snr_violations_mask))
        flagged_count = int(np.sum(flagged_mask))

        f0_status = (
            "pass" if np.isfinite(f0_loss) and f0_loss <= configuration.baseline_fluorescence_loss_threshold else "FAIL"
        )
        within_status = "pass" if within_violations == 0 else "FAIL"
        snr_status = "pass" if snr_violations == 0 else "FAIL"

        lines: list[str] = []
        lines.append("Chronic Photobleaching Evaluation")
        lines.append("=================================")
        lines.append("")
        lines.append("Overview")
        lines.append("--------")
        if session_count > 0:
            lines.append(
                f"Sessions:         {session_count} spanning {float(days[-1] - days[0]):.1f} days "
                f"(day {float(days[0]):.1f} to {float(days[-1]):.1f})"
            )
        else:
            lines.append("Sessions:         0")
        lines.append(f"Registered cells: {cell_count}")
        lines.append("")
        lines.append("Configuration")
        lines.append("-------------")
        lines.append(
            f"Baseline percentile:   {configuration.baseline_percentile} "
            f"(per-cell window: {configuration.cell_baseline_window_seconds} s, "
            f"FOV-mean window: {configuration.session_baseline_window_seconds} s)"
        )
        lines.append(f"SNR signal percentile: {configuration.snr_signal_percentile}")
        lines.append(
            f"Thresholds:            F0 loss > {configuration.baseline_fluorescence_loss_threshold:.0%}, "
            f"within-session drop > {configuration.within_session_loss_threshold:.0%},"
        )
        lines.append(
            f"                       SNR loss > {configuration.snr_loss_threshold:.0%} combined "
            f"with paired p < {configuration.snr_significance_threshold}"
        )
        lines.append("")
        lines.append("Across-session baseline fluorescence (chronic photobleaching)")
        lines.append("-------------------------------------------------------------")
        lines.append(f"Population F0 (first -> last):  {f0_first:.2f} -> {f0_last:.2f}")
        lines.append(f"Fractional loss:                {f0_loss:.1%}  [{f0_status}]")
        if decay_fit.fit_succeeded:
            lines.append(
                f"Decay fit:                      amplitude={decay_fit.amplitude:.2f}, "
                f"tau={decay_fit.tau_days:.2f} days, offset={decay_fit.offset:.2f}  [converged]"
            )
        else:
            lines.append("Decay fit:                      did not converge")
        lines.append("")
        lines.append("Within-session baseline (acute bleaching)")
        lines.append("-----------------------------------------")
        if finite_drops.size > 0:
            lines.append(f"Per-session drop median / max:  {drop_median:.1%} / {drop_max:.1%}")
        else:
            lines.append("Per-session drop median / max:  not available")
        lines.append(f"Sessions above threshold:       {within_violations} / {session_count}  [{within_status}]")
        lines.append("")
        lines.append("Per-cell SNR (signal-to-noise contrast across sessions)")
        lines.append("-------------------------------------------------------")
        lines.append(f"Population SNR (first -> last): {snr_first:.2f} -> {snr_last:.2f}")
        lines.append(f"Fractional loss:                {snr_loss:.1%}  [{snr_status}]")
        lines.append(f"Sessions failing both criteria: {snr_violations} / {session_count}")
        lines.append("")
        lines.append("Per-session detail")
        lines.append("------------------")
        lines.append(
            f"{'day':>6}  {'F0_pop':>8}  {'F0_loss':>8}  {'within_drop':>11}  "
            f"{'SNR_pop':>7}  {'SNR_p_vs_d0':>11}  {'flag':>4}  session"
        )
        for index in range(session_count):
            f0_pop_value = float(population_baseline[index])
            f0_loss_session = (f0_first - f0_pop_value) / f0_first if f0_first > 0 else float("nan")
            f0_loss_str = "       -" if index == 0 else f"{f0_loss_session:>8.1%}"
            within_value = float(within_drops[index])
            within_str = f"{within_value:>11.1%}" if np.isfinite(within_value) else f"{'N/A':>11}"
            snr_p = float(snr_p_values[index])
            if index == 0:
                snr_p_str = f"{'-':>11}"
            elif np.isfinite(snr_p):
                snr_p_str = f"{snr_p:>11.2e}"
            else:
                snr_p_str = f"{'N/A':>11}"
            flag_str = "yes" if bool(flagged_mask[index]) else "-"
            lines.append(
                f"{float(days[index]):>6.2f}  "
                f"{f0_pop_value:>8.2f}  "
                f"{f0_loss_str}  "
                f"{within_str}  "
                f"{float(population_snr[index]):>7.2f}  "
                f"{snr_p_str}  "
                f"{flag_str:>4}  "
                f"{session_names[index]}"
            )
        lines.append("")
        lines.append(f"Combined flag: {flagged_count} / {session_count} sessions exceeded any threshold criterion.")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Prints the summary text to the terminal via the ataraxis console.

        Notes:
            Equivalent to ``console.echo(self.summarize(), raw=True)``; provided so the print form is discoverable
            from the report's API surface alongside ``summarize`` (which returns a string for programmatic use).
        """
        console.echo(message=self.summarize(), raw=True)

    def plot_baseline_trend(self) -> plt.Figure:
        """Plots the population-median per-session baseline fluorescence trend, the exponential fit, and per-cell
        baseline fluorescence distributions.

        Returns:
            A matplotlib Figure showing the across-session baseline fluorescence trend.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        table = self.table
        # noinspection PyTypeChecker
        days: NDArray[np.float32] = (
            table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        population_baseline: NDArray[np.float32] = (
            table[BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value].to_numpy().astype(np.float32, copy=False)
        )
        cell_baseline_distributions = [
            np.asarray(values, dtype=np.float32)
            for values in table[BleachingColumn.CELL_BASELINE_FLUORESCENCE.value].to_list()
        ]
        cell_count = len(cell_baseline_distributions[0]) if cell_baseline_distributions else 0

        # Computes a non-zero box width that scales with the smallest day step so adjacent boxes do not overlap.
        minimum_day_step = float(np.diff(days).min()) if len(days) > 1 else 1.0
        box_width = 0.4 * max(minimum_day_step, 0.1)

        # Draws the per-cell distributions as boxplots so the population spread is visible alongside the median trend.
        axes.boxplot(cell_baseline_distributions, positions=days, widths=box_width, showfliers=False)

        # Overlays the population-median trend used for the exponential fit.
        axes.plot(
            days,
            population_baseline,
            marker="o",
            color="tab:blue",
            linewidth=1.5,
            label="Population median",
        )

        # Draws the fitted exponential when the fit converged.
        decay_fit = self.summary.baseline_fluorescence_decay_fit
        if decay_fit.fit_succeeded:
            # noinspection PyTypeChecker
            dense_days: NDArray[np.float32] = np.linspace(days.min(), days.max(), num=200, dtype=np.float32)
            # noinspection PyTypeChecker
            fit_curve: NDArray[np.float32] = (
                decay_fit.amplitude * np.exp(-dense_days / decay_fit.tau_days) + decay_fit.offset
            )
            axes.plot(
                dense_days,
                fit_curve,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=f"Exp fit (tau = {decay_fit.tau_days:.1f} d)",
            )

        axes.set_xlabel("Days since first session")
        axes.set_ylabel("Baseline fluorescence (a.u.)")
        axes.set_title(
            f"Across-session baseline fluorescence trend (n={cell_count} registered cells)",
            fontsize=10,
        )
        axes.legend(loc="best", fontsize=8)
        figure.tight_layout()
        return figure

    def plot_within_session(self) -> plt.Figure:
        """Plots the within-session FOV-mean baseline trace for each session as overlaid curves.

        Returns:
            A matplotlib Figure showing within-session bleaching.
        """
        figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

        table = self.table
        # noinspection PyTypeChecker
        days: NDArray[np.float32] = (
            table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
        )
        time_seconds_list = [
            np.asarray(values, dtype=np.float32)
            for values in table[BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value].to_list()
        ]
        baseline_list = [
            np.asarray(values, dtype=np.float32)
            for values in table[BleachingColumn.WITHIN_SESSION_BASELINE.value].to_list()
        ]
        # noinspection PyTypeChecker
        drops: NDArray[np.float32] = (
            table[BleachingColumn.WITHIN_SESSION_FRACTIONAL_DROP.value].to_numpy().astype(np.float32, copy=False)
        )
        session_count = len(days)

        colormap = plt.get_cmap("viridis")
        for index in range(session_count):
            # Guards against division by zero when the report contains a single session.
            color = colormap(index / max(session_count - 1, 1))
            label = f"Day {days[index]:.1f} (drop={drops[index]:.1%})"
            axes.plot(
                time_seconds_list[index] / 60.0,
                baseline_list[index],
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

        table = self.table
        # noinspection PyTypeChecker
        days: NDArray[np.float32] = (
            table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
        )
        snr_data = [np.asarray(values, dtype=np.float32) for values in table[BleachingColumn.CELL_SNR.value].to_list()]
        # noinspection PyTypeChecker
        p_values: NDArray[np.float64] = (
            table[BleachingColumn.SNR_PAIRED_P_VALUE.value].to_numpy().astype(np.float64, copy=False)
        )

        axes.violinplot(snr_data, positions=days, showmedians=True)

        # Annotates each session past the first with the Wilcoxon p-value relative to session 0.
        y_position = max(snr.max() for snr in snr_data) * 1.05
        for index in range(1, len(days)):
            p_value = p_values[index]
            label = f"p={p_value:.1e}" if np.isfinite(p_value) else "p=N/A"
            axes.text(days[index], y_position, label, ha="center", fontsize=7)

        axes.set_xlabel("Days since first session")
        axes.set_ylabel("Per-cell SNR")
        axes.set_title("Per-cell SNR across sessions (paired Wilcoxon vs session 0)", fontsize=10)
        figure.tight_layout()
        return figure


def evaluate_and_save_bleaching(
    dataset: DatasetData,
    animal: str,
    *,
    configuration: BleachingConfiguration | None = None,
) -> BleachingReport:
    """Resolves the animal's sessions from the dataset, evaluates bleaching, and persists the report.

    Notes:
        Sorts the animal's sessions chronologically before invoking ``BleachingReport.evaluate`` (lexicographic
        order on the canonical 'YYYY-MM-DD-HH-MM-SS-microseconds' session name is equivalent to chronological
        order). The report is written to ``<dataset>/<animal>/bleaching.yaml`` and
        ``<dataset>/<animal>/bleaching.feather`` and returned to the caller for in-process figure rendering.

    Args:
        dataset: The DatasetData instance describing the dataset that contains the animal.
        animal: The unique identifier of the animal to evaluate.
        configuration: Bleaching evaluation parameters. Uses defaults if None.

    Returns:
        The BleachingReport produced for the animal, with both files persisted under the animal directory.
    """
    dataset_animal = dataset.get_animal(animal=animal)
    animal_sessions = dataset.get_sessions_for_animal(animal=animal)
    sorted_sessions = sorted(animal_sessions, key=lambda dataset_session: dataset_session.session)
    session_paths = tuple(dataset_session.session_path for dataset_session in sorted_sessions)
    report = BleachingReport.evaluate(
        session_paths=session_paths,
        configuration=configuration,
    )
    report.save(animal=dataset_animal)
    return report


def _parse_session_microseconds(session_path: Path) -> int:
    """Parses the canonical 'YYYY-MM-DD-HH-MM-SS-microseconds' session-directory name into UTC microseconds.

    Args:
        session_path: Path whose final component encodes the session timestamp in the canonical dash-separated format.

    Returns:
        Microseconds elapsed since the UTC epoch corresponding to the session timestamp.
    """
    try:
        microseconds = parse_timestamp(
            date_string=session_path.name,
            format_string=_SESSION_TIMESTAMP_FORMAT,
            output_format=TimestampFormats.INTEGER,
        )
    except ValueError:
        message = (
            f"Unable to parse the session timestamp from path {str(session_path)!r}. The session directory name must "
            f"follow the 'YYYY-MM-DD-HH-MM-SS-microseconds' format, but got {session_path.name!r}."
        )
        console.error(message=message, error=ValueError)
    return int(microseconds)


def _validate_chronological_order(
    session_paths: tuple[Path, ...],
    session_microseconds: tuple[int, ...],
) -> None:
    """Verifies that the supplied session timestamps are non-decreasing, which the across-session metrics assume.

    Args:
        session_paths: Session directory paths in the order they were supplied; used in the error message.
        session_microseconds: Parsed session timestamps as UTC microseconds, in the same order as ``session_paths``.
    """
    for previous_index, (previous_us, current_us) in enumerate(pairwise(session_microseconds)):
        if current_us < previous_us:
            current_index = previous_index + 1
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The session paths must be sorted in "
                f"chronological order, but {session_paths[current_index].name!r} is older than "
                f"{session_paths[previous_index].name!r}."
            )
            console.error(message=message, error=ValueError)


def _compute_session_row(
    session_path: Path,
    configuration: BleachingConfiguration,
) -> _SessionRow:
    """Loads a single session's multi-recording raw fluorescence and computes the per-row values written to
    bleaching.feather.

    Args:
        session_path: Path to the forged session directory containing the data feather.
        configuration: Bleaching evaluation parameters that drive window sizes and percentile choices.

    Returns:
        A ``_SessionRow`` named tuple holding the per-cell baseline, per-cell SNR, and within-session baseline
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

    return _SessionRow(
        sampling_rate_hz=sampling_rate_hz,
        cell_baseline_fluorescence=cell_baseline_fluorescence,
        cell_snr=cell_snr,
        within_session_time_seconds=within_session_time_seconds,
        within_session_baseline=within_session_baseline,
        within_session_fractional_drop=within_session_fractional_drop,
    )


def _load_session_raw(session_path: Path) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Loads the multi-recording raw per-cell fluorescence trace and per-sample timestamps from a forged session
    feather.

    Notes:
        Loads ``DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE`` exclusively because the protocol's across-session
        per-cell comparisons require registered cell correspondence across days, which only the multi-recording
        cindra column carries. The fluorescence column is stored as a polars list-of-float32, one list per sample.
        Converting via ``Series.to_list()`` and ``np.array`` materializes a Python list of lists for every sample,
        which is single-threaded, GIL-bound, and dominates load time for multi-thousand-cell sessions. Exploding
        the list column to a flat fp32 series and reshaping in NumPy stays in compiled code and runs roughly an
        order of magnitude faster while producing the same (cell_count, sample_count) C-contiguous layout.

    Args:
        session_path: Path to the forged session directory containing the data feather.

    Returns:
        A tuple containing the (cell_count, sample_count) fp32 fluorescence array and the per-sample int64
        microsecond timestamps.
    """
    df = pl.read_ipc(
        source=session_path.joinpath(DatasetFiles.DATA),
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE.value],
    )
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
    signal_percentile: int,
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
    percentile: int,
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


def _fit_exponential_decay(
    days: NDArray[np.float32],
    baseline: NDArray[np.float32],
) -> ExponentialDecayFit:
    """Fits ``amplitude * exp(-d / tau_days) + offset`` (with ``d`` in days) to the per-session baseline fluorescence.

    Notes:
        Returns a sentinel ExponentialDecayFit with ``fit_succeeded=False`` and NaN parameters when fewer than the
        required number of sessions are supplied, when ``curve_fit`` raises, or when any fitted parameter is
        non-finite.

    Args:
        days: Per-session day offsets relative to the first session.
        baseline: Per-session population-median baseline fluorescence values aligned with ``days``.

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


def _compute_paired_snr_p_values(cell_snr_arrays: list[NDArray[np.float32]]) -> NDArray[np.float64]:
    """Computes paired Wilcoxon signed-rank p-values comparing each session's per-cell SNR to the first session.

    Args:
        cell_snr_arrays: Per-session per-cell SNR arrays in chronological order; the first element is the reference.

    Returns:
        An array with length ``len(cell_snr_arrays)`` holding the per-session paired Wilcoxon p-values. The first
        entry is NaN because it would compare the reference to itself, and entries are also NaN for sessions whose
        SNR is identical to the reference or for which the test raises.
    """
    # noinspection PyTypeChecker
    p_values: NDArray[np.float64] = np.full(len(cell_snr_arrays), np.nan, dtype=np.float64)
    reference_snr = cell_snr_arrays[0]
    for index in range(1, len(cell_snr_arrays)):
        target_snr = cell_snr_arrays[index]
        if np.all(target_snr == reference_snr):
            continue
        try:
            result = wilcoxon(x=target_snr, y=reference_snr, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            continue
        p_values[index] = float(result.pvalue)
    return p_values


def _compute_flag_mask(
    population_baseline: NDArray[np.float32],
    population_snr: NDArray[np.float32],
    snr_paired_p_values: NDArray[np.float64],
    within_session_drops: NDArray[np.float32],
    configuration: BleachingConfiguration,
) -> NDArray[np.bool_]:
    """Returns a boolean mask flagging sessions that violate any configured threshold criterion.

    Args:
        population_baseline: Population-median baseline fluorescence per session.
        population_snr: Population-median SNR per session.
        snr_paired_p_values: Paired Wilcoxon p-values per session; the first entry is NaN.
        within_session_drops: Within-session fractional drop per session (NaN allowed).
        configuration: Bleaching evaluation parameters that supply the threshold values.

    Returns:
        Boolean mask aligned with the per-session arrays. True where the session violated any threshold.
    """
    session_count = population_baseline.shape[0]
    # noinspection PyTypeChecker
    flagged: NDArray[np.bool_] = np.zeros(session_count, dtype=np.bool_)

    baseline_reference = float(population_baseline[0])
    snr_reference = float(population_snr[0])

    for index in range(session_count):
        baseline_loss = (
            (baseline_reference - float(population_baseline[index])) / baseline_reference
            if baseline_reference > 0
            else 0.0
        )
        snr_loss = (snr_reference - float(population_snr[index])) / snr_reference if snr_reference > 0 else 0.0
        snr_p_value = float(snr_paired_p_values[index])
        within_drop = float(within_session_drops[index])

        baseline_flagged = baseline_loss > configuration.baseline_fluorescence_loss_threshold
        within_flagged = np.isfinite(within_drop) and within_drop > configuration.within_session_loss_threshold
        snr_flagged = (
            snr_loss > configuration.snr_loss_threshold
            and np.isfinite(snr_p_value)
            and snr_p_value < configuration.snr_significance_threshold
        )

        flagged[index] = baseline_flagged or within_flagged or snr_flagged

    return flagged
