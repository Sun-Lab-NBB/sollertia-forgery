"""Cross-session photobleaching evaluation across a chronologically ordered set of two-photon imaging sessions
for the same animal.

Implements the canonical three-metric protocol for chronic GCaMP imaging by aggregating per-session inputs from
:mod:`.bleaching_protocol`: per-cell session-median baseline fluorescence (estimated as a low percentile of the
raw trace within a baseline window) trend across days fit to a single exponential, within-session bleaching
slope, and per-cell signal-to-noise change on the multi-recording registered cell intersection. The per-session
compute kernel and its numba implementation live in :mod:`.bleaching_protocol`; per-animal and dataset-level
plots live in :mod:`.plotting`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple
from itertools import pairwise
from contextlib import nullcontext
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from numba import set_num_threads
import numpy as np
import polars as pl
from scipy.stats import wilcoxon
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp
from scipy.optimize import curve_fit
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import YamlConfig

from .bleaching_protocol import BleachingConfiguration, BleachingSessionResult, compute_session_metrics
from ..shared_utilities import resolve_display_units
from ...shared_assets import DatasetData, DatasetAnimal

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MINIMUM_SESSIONS_FOR_DECAY_FIT: int = 3
"""Minimum number of sessions required to fit a single-exponential decay model."""
_MINIMUM_SESSIONS_FOR_EVALUATION: int = 2
"""Minimum number of sessions required to evaluate any across-session bleaching metric."""
_PREFERRED_WORKERS_PER_SESSION: int = 10
"""Preferred number of CPU cores per parallel session subprocess. The saturating allocator targets this width
before spawning additional parallel sessions; a smaller width than cindra's 30 because per-session evaluation is
dominated by per-cell Numba kernels that scale only modestly past ten threads, leaving the remaining budget for
additional concurrent sessions."""
_MINIMUM_WORKERS_PER_SESSION: int = 5
"""Floor on the per-subprocess worker count when running multiple sessions in parallel. Falling below this floor
reduces the across-session parallelism rather than spawning under-resourced subprocesses."""
_WORKER_MULTIPLE: int = 5
"""Worker counts are rounded down to the nearest multiple of this value for clean allocation."""
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
    """True when ``scipy.optimize.curve_fit`` converged on a finite, in-bounds solution; False otherwise."""

    def evaluate(self, days: NDArray[np.floating]) -> NDArray[np.float64]:
        """Evaluates the fitted single-exponential decay at the supplied day offsets.

        Args:
            days: Day offsets relative to the first session. Accepts any floating dtype; the result is fp64 to
                match the precision of ``scipy.optimize.curve_fit``.

        Returns:
            ``amplitude * exp(-days / tau_days) + offset`` evaluated at every entry of ``days``.
        """
        return _exponential_decay_model(
            days=days.astype(np.float64, copy=False),
            amplitude=self.amplitude,
            tau_days=self.tau_days,
            offset=self.offset,
        )


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
    def load(cls, animal: DatasetAnimal) -> BleachingReport:
        """Loads a previously saved bleaching report from the animal directory.

        Args:
            animal: The DatasetAnimal whose directory holds ``bleaching.yaml`` and ``bleaching.feather``.

        Returns:
            A BleachingReport persisted by :func:`run_bleaching_analysis`.
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
            per-session detail table. Together with :func:`.plotting.plot_baseline_trend`,
            :func:`.plotting.plot_within_session`, and :func:`.plotting.plot_snr_distributions`, this is jointly
            sufficient for scientific presentation, discussion, and publication of the animal's photobleaching
            state. Designed to be human-readable and parseable by downstream agents.

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

        # Resolves the integer display unit and ticks once so the overview header and the per-session detail table
        # stay in the same unit. Falls back to a "day" placeholder when the table is empty so the header column label
        # is still well-defined; the per-session loop will not execute in that case.
        if session_count > 0:
            unit, ticks = resolve_display_units(days_since_first=days)
        else:
            unit = "day"
            # noinspection PyTypeChecker
            ticks = np.zeros(0, dtype=np.int64)
        unit_plural = f"{unit}s"

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

        # Per-criterion violation counts come from the same ``_compute_flag_masks`` helper that produced the
        # FLAGGED column when the report was built, so the pass/fail tags here cannot drift from the persisted
        # flag whenever a threshold definition shifts. f0_status keeps its first-to-last semantics as a distinct
        # headline metric (the per-session baseline mask answers a different question — "did any session along
        # the way fall below threshold").
        flag_masks = _compute_flag_masks(
            population_baseline=population_baseline,
            population_snr=population_snr,
            snr_paired_p_values=snr_p_values,
            within_session_drops=within_drops,
            configuration=configuration,
        )
        within_violations = int(np.sum(flag_masks.within))
        snr_violations = int(np.sum(flag_masks.snr))
        flagged_count = int(np.sum(flagged_mask))

        f0_status = (
            "pass" if np.isfinite(f0_loss) and f0_loss <= configuration.baseline_fluorescence_loss_threshold else "FAIL"
        )
        within_status = "pass" if within_violations == 0 else "FAIL"
        snr_status = "pass" if snr_violations == 0 else "FAIL"

        lines: list[str] = [
            "Chronic Photobleaching Evaluation",
            "=================================",
            "",
            "Overview",
            "--------",
        ]
        if session_count > 0:
            lines.append(
                f"Sessions:         {session_count} spanning {int(ticks[-1] - ticks[0])} {unit_plural} "
                f"({unit} {int(ticks[0])} to {int(ticks[-1])})"
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
        # Renders the per-session detail as a pipe-separated ASCII table that mirrors the table style used by the
        # ataraxis-time benchmark report. Header names match the canonical ``BleachingColumn`` enum values where they
        # fit and shorten to descriptive equivalents (``population_F0`` for ``population_baseline_fluorescence``,
        # ``within_session_drop`` for ``within_session_fractional_drop``) where the full names would dominate the
        # table width. The separator row is derived from the header by replacing pipes with plus signs and remaining
        # characters with dashes so the column boundaries stay aligned regardless of how the widths are tuned.
        # Session labels are truncated to the ``YY-MM-DD-HH`` prefix because the protocol's >=1h separation
        # invariant guarantees the hour resolution is sufficient to identify each session uniquely.
        table_header = (
            f"{unit:>4} | {'population_F0':>13} | {'F0_loss':>7} | {'within_session_drop':>19} | "
            f"{'population_SNR':>14} | {'SNR_paired_p_value':>18} | {'flagged':>7} | {'session':<11}"
        )
        table_separator = "".join("+" if character == "|" else "-" for character in table_header)
        lines.append(table_header)
        lines.append(table_separator)
        for index in range(session_count):
            f0_pop_value = float(population_baseline[index])
            f0_loss_session = (f0_first - f0_pop_value) / f0_first if f0_first > 0 else float("nan")
            f0_loss_str = "-" if index == 0 else f"{f0_loss_session:.1%}"
            within_value = float(within_drops[index])
            within_str = f"{within_value:.1%}" if np.isfinite(within_value) else "N/A"
            snr_p = float(snr_p_values[index])
            if index == 0:
                snr_p_str = "-"
            elif np.isfinite(snr_p):
                snr_p_str = f"{snr_p:.2e}"
            else:
                snr_p_str = "N/A"
            flag_str = "yes" if bool(flagged_mask[index]) else "-"
            # Slices the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` directory name to ``YY-MM-DD-HH``; the >=1h
            # separation invariant makes the minute / second / microsecond fields redundant for identification here.
            short_session = session_names[index][2:13]
            lines.append(
                f"{int(ticks[index]):>4d} | "
                f"{f0_pop_value:>13.2f} | "
                f"{f0_loss_str:>7} | "
                f"{within_str:>19} | "
                f"{float(population_snr[index]):>14.2f} | "
                f"{snr_p_str:>18} | "
                f"{flag_str:>7} | "
                f"{short_session:<11}"
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


def _assemble_bleaching_report(
    session_paths: tuple[Path, ...],
    session_results: tuple[BleachingSessionResult, ...],
    configuration: BleachingConfiguration,
) -> BleachingReport:
    """Assembles a :class:`BleachingReport` from pre-computed per-session results.

    Notes:
        Pure aggregation step — no I/O, no per-cell compute. Validates chronological order of ``session_paths``
        and registered cell-count consistency across results before running the cross-session decay fit, paired
        Wilcoxon comparison, and combined-flag computation. Methodological references for the protocol live on
        the public orchestrator :func:`run_bleaching_analysis`; this helper exists so the orchestrator can hand
        off its parallel session-results dispatch to a single deterministic aggregation pass.

    Args:
        session_paths: Chronologically ordered tuple of session directory paths, parallel to
            ``session_results``.
        session_results: Per-session results produced by :func:`.bleaching_protocol.compute_session_metrics`,
            in the same order as ``session_paths``.
        configuration: Bleaching evaluation parameters that produced the results.

    Returns:
        A BleachingReport whose ``table`` holds one row per session and whose ``summary`` holds the
        cross-session aggregates and the configuration.
    """
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

    # Validates that every session sees the same registered cell count before any cross-session reduction
    # touches the per-cell arrays. Mismatched cell counts would silently produce broadcasting errors in the
    # paired Wilcoxon test or the population-median trend.
    cell_count_reference: int | None = None
    for session_path, result in zip(session_paths, session_results, strict=True):
        result_cell_count = int(result.cell_baseline_fluorescence.shape[0])
        if cell_count_reference is None:
            cell_count_reference = result_cell_count
        elif result_cell_count != cell_count_reference:
            message = (
                f"Unable to evaluate bleaching across the supplied sessions. The cell count must match "
                f"across all sessions for the multi-recording registered comparison, but session "
                f"{session_path.name!r} has {result_cell_count} cells while the first session has "
                f"{cell_count_reference}."
            )
            console.error(message=message, error=ValueError)

    session_names = [session_path.name for session_path in session_paths]
    sampling_rates = [result.sampling_rate_hz for result in session_results]
    cell_baseline_arrays = [result.cell_baseline_fluorescence for result in session_results]
    cell_snr_arrays = [result.cell_snr for result in session_results]
    within_session_time_arrays = [result.within_session_time_seconds for result in session_results]
    within_session_baseline_arrays = [result.within_session_baseline for result in session_results]
    within_session_drops = [result.within_session_fractional_drop for result in session_results]

    # Cross-session aggregates derived from the per-session arrays.
    population_baseline_values = [float(np.median(array)) for array in cell_baseline_arrays]
    population_snr_values = [float(np.median(array)) for array in cell_snr_arrays]

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
    flag_masks = _compute_flag_masks(
        population_baseline=population_baseline_array,
        population_snr=population_snr_array,
        snr_paired_p_values=snr_paired_p_values,
        within_session_drops=within_session_drop_array,
        configuration=configuration,
    )
    flagged_mask = flag_masks.combined

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
        configuration=configuration,
        baseline_fluorescence_decay_fit=decay_fit,
    )

    return BleachingReport(table=table, summary=summary)


def run_bleaching_analysis(
    dataset: DatasetData,
    *,
    animal: str | None = None,
    workers: int = -1,
    display_progress: bool = True,
    configuration: BleachingConfiguration | None = None,
) -> tuple[BleachingReport, ...]:
    """Evaluates bleaching for every animal in the dataset (or a single specified animal) and persists each report.

    Notes:
        Sole orchestrator for assembling :class:`BleachingReport` artifacts. Per-session metrics are produced
        by :func:`.bleaching_protocol.compute_session_metrics` and cross-session aggregation is delegated to
        :func:`_assemble_bleaching_report`; methodological references and protocol context live on those
        algorithmic entry-points. Each report is written to ``<dataset>/<animal>/bleaching.yaml`` and
        ``<dataset>/<animal>/bleaching.feather`` and returned to the caller for in-process figure rendering.

        ``workers`` controls the total CPU budget. Sessions are the natural unit of parallelism: the budget is
        split between across-session subprocesses and within-session Numba threads through cindra's saturating
        allocator (:func:`_resolve_saturating_allocation`). A budget of one (or a single session in scope)
        collapses to an in-process run with every thread handed to Numba.

    Args:
        dataset: The DatasetData instance whose animals are evaluated.
        animal: The unique identifier of a single animal to evaluate. When None, every animal in the dataset
            is evaluated and the returned tuple preserves the order of ``DatasetData.animals``.
        workers: The total number of CPU cores to use. A non-positive value requests every available core minus
            the system reserve. ``workers=1`` forces a fully sequential, single-threaded run.
        display_progress: Determines whether to display a progress bar tracking total sessions across every
            in-scope animal as each session row is computed.
        configuration: Bleaching evaluation parameters shared across animals. Uses defaults if None.

    Returns:
        A tuple of BleachingReports in the same order as the resolved animal set, with each report's two
        artifacts persisted under the corresponding animal directory.
    """
    resolved_configuration: BleachingConfiguration = (
        configuration if configuration is not None else BleachingConfiguration()
    )

    # Routes through an explicitly typed local so PyCharm narrows ``DatasetData.get_animal`` (whose trailing
    # ``console.error`` makes the IDE infer an unreachable None branch) and the multi-animal iteration both land
    # as plain ``str``. The single-animal branch reuses the supplied name after ``get_animal`` validates it.
    if animal is not None:
        resolved_animal: DatasetAnimal = dataset.get_animal(animal=animal)
        animal_names: tuple[str, ...] = (resolved_animal.animal,)
    else:
        animal_names = tuple(dataset_animal.animal for dataset_animal in dataset.animals)

    if not animal_names:
        message = (
            f"Unable to run bleaching analysis on dataset {dataset.name!r}. The dataset contains no animals "
            f"to evaluate."
        )
        console.error(message=message, error=ValueError)

    # Resolves chronologically ordered session paths per animal up front so the orchestrator can validate the
    # per-animal session minimum before any compute is dispatched. Sessions are paired with their owning animal
    # in a flat job list that the inner ProcessPoolExecutor can consume without further grouping.
    sessions_by_animal: dict[str, tuple[Path, ...]] = {}
    for animal_name in animal_names:
        animal_sessions = dataset.get_sessions_for_animal(animal=animal_name)
        sorted_sessions = sorted(animal_sessions, key=lambda dataset_session: dataset_session.session)
        animal_session_paths = tuple(dataset_session.session_path for dataset_session in sorted_sessions)
        if len(animal_session_paths) < _MINIMUM_SESSIONS_FOR_EVALUATION:
            message = (
                f"Unable to run bleaching analysis on dataset {dataset.name!r}. The protocol requires at least "
                f"{_MINIMUM_SESSIONS_FOR_EVALUATION} sessions per animal, but animal {animal_name!r} has "
                f"{len(animal_session_paths)}."
            )
            console.error(message=message, error=ValueError)
        sessions_by_animal[animal_name] = animal_session_paths

    session_jobs: tuple[tuple[str, Path], ...] = tuple(
        (animal_name, session_path)
        for animal_name in animal_names
        for session_path in sessions_by_animal[animal_name]
    )
    total_sessions = len(session_jobs)

    # Splits the resolved CPU budget between the across-session process pool (outer) and the per-cell Numba
    # thread pool (inner) using the same saturating allocator that cindra uses for its compute-bound jobs:
    # saturate each subprocess up to the preferred worker count before spawning a new parallel session, and
    # never drop below the per-session minimum when running in parallel.
    total_workers = resolve_worker_count(requested_workers=workers)
    numba_threads_per_session, parallel_sessions = _resolve_saturating_allocation(
        budget=total_workers,
        session_count=total_sessions,
    )

    console.echo(
        message=(
            f"Running bleaching analysis on dataset {dataset.name!r} for "
            f"{len(animal_names)} animal{'s' if len(animal_names) != 1 else ''} "
            f"({total_sessions} session{'s' if total_sessions != 1 else ''} total): "
            f"{parallel_sessions} parallel × {numba_threads_per_session} Numba "
            f"thread{'s' if numba_threads_per_session != 1 else ''} per session "
            f"(total CPU budget: {total_workers})..."
        ),
        level=LogLevel.INFO,
    )

    # Computes per-session results. Each result keys back to its (animal, session_path) pair so the parent can
    # reorder them into chronological per-animal arrays before aggregation.
    results_by_animal: dict[str, dict[Path, BleachingSessionResult]] = {animal_name: {} for animal_name in animal_names}

    if parallel_sessions == 1:
        # Single-session or single-worker fast path: stay in-process and hand every thread to Numba so the
        # per-cell kernels saturate the local thread pool.
        set_num_threads(numba_threads_per_session)
        progress_context = (
            console.progress(total=total_sessions, description="Computing session rows", unit="session")
            if display_progress
            else nullcontext()
        )
        with progress_context as progress_bar:
            for animal_name, session_path in session_jobs:
                results_by_animal[animal_name][session_path] = compute_session_metrics(
                    session_path=session_path,
                    configuration=resolved_configuration,
                )
                if progress_bar is not None:
                    progress_bar.update()
    else:
        # Multi-session path: dispatch session results across a process pool. Each subprocess sets its Numba
        # thread cap via the initializer so the per-cell kernels respect the per-process share, and the parent
        # process surfaces one session-level progress bar.
        progress_context = (
            console.progress(
                total=total_sessions,
                description=f"Computing session rows ({parallel_sessions} sessions in parallel)",
                unit="session",
            )
            if display_progress
            else nullcontext()
        )
        with (
            ProcessPoolExecutor(
                max_workers=parallel_sessions,
                initializer=_configure_subprocess_numba_threads,
                initargs=(numba_threads_per_session,),
            ) as executor,
            progress_context as progress_bar,
        ):
            future_to_job = {
                executor.submit(
                    compute_session_metrics,
                    session_path=session_path,
                    configuration=resolved_configuration,
                ): (animal_name, session_path)
                for animal_name, session_path in session_jobs
            }
            for future in as_completed(future_to_job):
                completed_animal, completed_path = future_to_job[future]
                results_by_animal[completed_animal][completed_path] = future.result()
                if progress_bar is not None:
                    progress_bar.update()

    # Aggregates per animal in the parent. The cross-session step is cheap (population medians, curve_fit,
    # Wilcoxon, flag mask) compared to the per-session compute, and keeping it in the parent avoids round-tripping
    # whole reports through the IPC layer.
    reports: list[BleachingReport] = []
    for animal_name in animal_names:
        ordered_paths = sessions_by_animal[animal_name]
        ordered_results = tuple(results_by_animal[animal_name][session_path] for session_path in ordered_paths)
        report = _assemble_bleaching_report(
            session_paths=ordered_paths,
            session_results=ordered_results,
            configuration=resolved_configuration,
        )
        report.save(animal=dataset.get_animal(animal=animal_name))
        reports.append(report)

    console.echo(
        message=(
            f"Bleaching analysis complete. Persisted {len(animal_names)} "
            f"report{'s' if len(animal_names) != 1 else ''} under "
            f"{dataset.dataset_data_path.parent}."
        ),
        level=LogLevel.SUCCESS,
    )
    return tuple(reports)


def _resolve_saturating_allocation(budget: int, session_count: int) -> tuple[int, int]:
    """Splits a CPU budget between per-session Numba threads and across-session subprocesses.

    Notes:
        Mirrors cindra's compute-bound saturating allocator with bleaching-specific constants: each subprocess is
        filled to ``_PREFERRED_WORKERS_PER_SESSION`` (10) before a new parallel session is added, the
        per-subprocess thread count is rounded down to a multiple of ``_WORKER_MULTIPLE`` for clean allocation,
        and parallelism is reduced one step at a time whenever the per-subprocess share would fall below
        ``_MINIMUM_WORKERS_PER_SESSION`` (5). The single-session path collapses naturally: a budget of N with one
        session returns ``(round_down_to_5(N), 1)`` and hands every thread to Numba.

    Args:
        budget: Total CPU cores available after the system reservation, as returned by ``resolve_worker_count``.
        session_count: Number of sessions scheduled for parallel evaluation, summed across every animal in scope.

    Returns:
        A tuple of ``(numba_threads_per_session, parallel_sessions)`` whose product never exceeds the budget.
    """
    max_at_preferred = max(1, budget // _PREFERRED_WORKERS_PER_SESSION)
    parallel_sessions = min(session_count, max_at_preferred)
    raw_workers = budget // parallel_sessions
    numba_threads = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)

    # Reduces parallelism one session at a time until each subprocess clears the per-session floor. The loop
    # cannot reduce below a single subprocess; with ``parallel_sessions == 1`` the floor stops applying because
    # the in-process branch hands every thread to Numba directly.
    while numba_threads < _MINIMUM_WORKERS_PER_SESSION and parallel_sessions > 1:
        parallel_sessions -= 1
        raw_workers = budget // parallel_sessions
        numba_threads = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)

    return numba_threads, parallel_sessions


def _configure_subprocess_numba_threads(thread_count: int) -> None:
    """Caps the active Numba thread pool inside a worker subprocess to the share assigned by the orchestrator.

    Notes:
        Used as the ``initializer`` for the ``ProcessPoolExecutor`` spawned by ``run_bleaching_analysis`` so that
        every subprocess respects the per-process share of the resolved worker budget rather than defaulting to
        every available core.

    Args:
        thread_count: The maximum number of threads Numba parallel kernels may use inside this subprocess.
    """
    set_num_threads(thread_count)


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
    except (RuntimeError, ValueError):
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


class _FlagMasks(NamedTuple):
    """Per-criterion and combined boolean masks produced by :func:`_compute_flag_masks`.

    Each mask is aligned with the per-session arrays the helper consumed. ``combined`` is the elementwise OR of
    ``baseline``, ``within``, and ``snr`` and is what gets persisted into the FLAGGED column; the per-criterion
    masks are surfaced so the textual summary can count violations per criterion without re-deriving the
    threshold logic.
    """

    baseline: NDArray[np.bool_]
    within: NDArray[np.bool_]
    snr: NDArray[np.bool_]
    combined: NDArray[np.bool_]


def _compute_flag_masks(
    population_baseline: NDArray[np.float32],
    population_snr: NDArray[np.float32],
    snr_paired_p_values: NDArray[np.float64],
    within_session_drops: NDArray[np.float32],
    configuration: BleachingConfiguration,
) -> _FlagMasks:
    """Computes per-criterion and combined per-session flag masks against the configured thresholds.

    Args:
        population_baseline: Population-median baseline fluorescence per session.
        population_snr: Population-median SNR per session.
        snr_paired_p_values: Paired Wilcoxon p-values per session; the first entry is NaN.
        within_session_drops: Within-session fractional drop per session (NaN allowed).
        configuration: Bleaching evaluation parameters that supply the threshold values.

    Returns:
        A :class:`_FlagMasks` whose ``baseline``, ``within``, and ``snr`` fields hold per-criterion masks and
        whose ``combined`` field is the elementwise OR used for the persisted FLAGGED column.
    """
    session_count = population_baseline.shape[0]
    # noinspection PyTypeChecker
    baseline_mask: NDArray[np.bool_] = np.zeros(session_count, dtype=np.bool_)
    # noinspection PyTypeChecker
    within_mask: NDArray[np.bool_] = np.zeros(session_count, dtype=np.bool_)
    # noinspection PyTypeChecker
    snr_mask: NDArray[np.bool_] = np.zeros(session_count, dtype=np.bool_)

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

        baseline_mask[index] = baseline_loss > configuration.baseline_fluorescence_loss_threshold
        within_mask[index] = np.isfinite(within_drop) and within_drop > configuration.within_session_loss_threshold
        snr_mask[index] = (
            snr_loss > configuration.snr_loss_threshold
            and np.isfinite(snr_p_value)
            and snr_p_value < configuration.snr_significance_threshold
        )

    # noinspection PyTypeChecker
    combined: NDArray[np.bool_] = baseline_mask | within_mask | snr_mask
    return _FlagMasks(baseline=baseline_mask, within=within_mask, snr=snr_mask, combined=combined)
