"""Per-session synchronous calcium event (SCE) report container.

Wraps the :class:`SCEDetector` output as a triplet of persisted artifacts: a per-cell participation feather, a
per-period SCE-state feather, and a summary YAML. The report owns persistence (``save`` / ``load``) and the
human-readable summary; per-session and cross-session plots live in :mod:`.plotting` and consume the report.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import chi2
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

from ...forging import FluorescenceColumn
from .sce_protocol import SCEResult, SCEDetector, SCEDetectionConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


class SCECellColumn(StrEnum):
    """Defines every column written to the per-session ``sce_cells.feather`` per-cell participation table."""

    CELL_ID = "cell_id"
    """Contiguous integer cell identifier."""
    SCE_PARTICIPATION_COUNT = "sce_participation_count"
    """Total number of stationary-period SCEs the cell participated in."""
    SCE_PARTICIPATION_RATE = "sce_participation_rate"
    """Fraction of stationary-period SCEs the cell participated in."""
    SCE_EVENTS = "sce_events"
    """Per-cell list of (period_index, sce_label) pairs for every SCE the cell participated in."""
    SCE_PARTICIPATION_P_VALUE = "sce_participation_p_value"
    """Per-cell aggregated p-value for SCE recruitment across all stationary periods. Combined via Fisher's
    method over the per-period jitter-null p-values produced by ``SCEDetector``. NaN when no period contributed
    SCEs."""
    IS_SCE_CELL = "is_sce_cell"
    """True for cells whose participation rate exceeds the per-cell jitter null in at least one stationary
    period at the configured ``participation_significance_percentile``."""


class SCEPeriodColumn(StrEnum):
    """Defines every column written to the per-session ``sce_periods.feather`` per-period table."""

    PERIOD_INDEX = "period_index"
    """0-based stationary-period index in temporal session order, matching the ``(period_index, sce_label)``
    references in :attr:`SCECellColumn.SCE_EVENTS`."""
    PERIOD_STATE = "period_state"
    """``DatasetColumn.SYSTEM_STATE`` value of the protocol epoch this stationary chunk was extracted from
    (e.g. ``"rest"``, ``"run"``, or any custom protocol state)."""
    CELL_COUNT = "cell_count"
    """Cell count active during the period; constant across all rows in a session."""
    SAMPLE_COUNT = "sample_count"
    """Number of samples retained in this period after stability filtering."""
    SAMPLING_RATE_HZ = "sampling_rate_hz"
    """Acquisition sampling rate in Hz; constant across all rows."""
    THRESHOLD = "threshold"
    """Significance threshold (mean + scale * SD of shuffled distribution) used for SCE detection in this period."""
    TIMESTAMPS_MINUTES = "timestamps_minutes"
    """Per-sample elapsed-minutes timestamps recorded directly from the live SCEResult."""
    COACTIVE_COUNTS = "coactive_counts"
    """Per-sample co-active cell counts."""
    SCE_LABELS = "sce_labels"
    """Per-sample SCE event labels (1-indexed); 0 outside any SCE."""
    ONSET_CELL_INDICES = "onset_cell_indices"
    """Sparse encoding of the per-period onset matrix: cell index for each True onset entry."""
    ONSET_SAMPLE_INDICES = "onset_sample_indices"
    """Sparse encoding of the per-period onset matrix: sample index (local to the period) for each True onset
    entry. Same length as ``ONSET_CELL_INDICES``."""
    TRIAL_IDS = "trial_ids"
    """Per-sample trial id matching the dataset ``trial`` column (-1 outside any complete trial)."""
    SCE_SIZE = "sce_size"
    """Per-SCE distinct participating-cell count."""
    SCE_WIDTH_SAMPLES = "sce_width_samples"
    """Per-SCE duration in samples."""
    SCE_PEAK_COACTIVE = "sce_peak_coactive"
    """Per-SCE peak co-active count, the maximum of ``COACTIVE_COUNTS`` over the SCE window."""
    SCE_INTER_EVENT_INTERVALS_SAMPLES = "sce_inter_event_intervals_samples"
    """Per-SCE sample gap between consecutive events; length equal to ``max(sce_count - 1, 0)``."""
    SCE_RATE_HZ = "sce_rate_hz"
    """SCE rate in events per second over the period (post-stability masking)."""


@dataclass
class SCESummary(YamlConfig):
    """Per-session YAML companion to ``sce_cells.feather`` and ``sce_periods.feather``.

    Carries only fields that are not derivable from the two feathers: the SCE detection configuration (load-bearing
    because the per-cell flags depend on its thresholds), the sampling rate, and the period / event totals that
    anchor cross-session aggregates.
    """

    sce_configuration: SCEDetectionConfiguration
    """SCE detection configuration that produced the per-cell SCE columns and the per-period feather."""
    sampling_rate_hz: float
    """Acquisition sampling rate in Hz."""
    cell_count: int
    """Total number of cells in the session."""
    period_count: int
    """Number of stationary periods (across every protocol state) that contributed an SCE row."""
    total_sces: int
    """Total SCEs detected across every stationary period."""
    sce_cell_count: int
    """Number of cells flagged as SCE-recruited during at least one stationary period (Modol 2020 super-rich)."""


@dataclass(frozen=True, slots=True)
class SCEReport:
    """Top-level per-session container for the SCE pipeline output.

    Holds the per-cell participation table (``cells``), the per-period SCE-state table (``periods``), and the YAML
    summary (``summary``). Persistence is co-located here; plot regeneration lives in :mod:`.plotting`.
    """

    cells: pl.DataFrame
    """Per-cell wide table; one row per cell. Schema enumerated by :class:`SCECellColumn`."""
    periods: pl.DataFrame
    """Per-period SCE state table; one row per stationary period that yielded SCEs. Schema enumerated by
    :class:`SCEPeriodColumn`."""
    summary: SCESummary
    """YAML wrapper holding the configuration and session-level scalars."""

    @classmethod
    def evaluate(
        cls,
        session_path: Path,
        *,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        configuration: SCEDetectionConfiguration | None = None,
    ) -> SCEReport:
        """Runs the SCE detection pipeline and assembles an in-memory report.

        Args:
            session_path: Path to the session's dataset directory.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            configuration: SCE detection parameters. Uses defaults if None.

        Returns:
            An in-memory SCEReport ready to be saved or plotted.
        """
        resolved_configuration = configuration if configuration is not None else SCEDetectionConfiguration()

        console.echo(message="Running SCE detection...", level=LogLevel.INFO)
        sce_detector = SCEDetector(
            session_path=session_path,
            fluorescence_column=fluorescence_column,
            configuration=resolved_configuration,
        )
        sce_detector.detect_events()
        sce_results: list[SCEResult] = sce_detector.results
        cell_count = int(sce_detector.cell_count)
        sampling_rate_hz = float(sce_detector.sampling_rate_hz)
        period_count = len(sce_results)
        total_sces = int(sum(int(np.max(result.sce_labels)) for result in sce_results))
        console.echo(
            message=f"SCE detection complete: {period_count} stationary periods ({total_sces} SCEs).",
            level=LogLevel.SUCCESS,
        )

        cells = _build_sce_cells_table(cell_count=cell_count, sce_results=sce_results)
        periods = _build_sce_periods_table(
            sampling_rate_hz=sampling_rate_hz, results=sce_results, cell_count=cell_count
        )

        # noinspection PyTypeChecker
        sce_cell_flag: NDArray[np.bool_] = cells[SCECellColumn.IS_SCE_CELL.value].to_numpy()

        summary = SCESummary(
            sce_configuration=resolved_configuration,
            sampling_rate_hz=sampling_rate_hz,
            cell_count=cell_count,
            period_count=period_count,
            total_sces=total_sces,
            sce_cell_count=int(np.sum(sce_cell_flag)),
        )
        return cls(cells=cells, periods=periods, summary=summary)

    @classmethod
    def load(cls, session: DatasetSession) -> SCEReport:
        """Loads a previously saved report from the session directory.

        Args:
            session: The DatasetSession whose directory holds the three artifacts.

        Returns:
            An SCEReport whose feathers are memory-mapped against the on-disk files.
        """
        summary: SCESummary = SCESummary.from_yaml(file_path=session.sce_summary_path)
        cells = pl.read_ipc(source=session.sce_cells_path, memory_map=True)
        periods = pl.read_ipc(source=session.sce_periods_path, memory_map=True)
        return cls(cells=cells, periods=periods, summary=summary)

    def save(self, session: DatasetSession) -> None:
        """Persists the report to the three per-session artifacts inside the session directory.

        Args:
            session: The DatasetSession whose directory will hold the three artifacts.
        """
        self.summary.to_yaml(file_path=session.sce_summary_path)
        self.cells.write_ipc(file=session.sce_cells_path)
        self.periods.write_ipc(file=session.sce_periods_path)

    def summarize(self) -> str:
        """Returns a multi-line human-readable summary of the SCE pipeline output."""
        summary = self.summary
        cell_count = summary.cell_count
        sce_pct = 100.0 * summary.sce_cell_count / cell_count if cell_count > 0 else 0.0
        return "\n".join(
            [
                "SCE report",
                "==========",
                f"Cells: {cell_count}",
                f"  SCE-recruited cells: {summary.sce_cell_count} ({sce_pct:.1f}%)",
                "",
                "SCE detection (stationary samples across every protocol epoch):",
                f"  Stationary periods: {summary.period_count} ({summary.total_sces} SCEs)",
                "",
                "Sampling:",
                f"  Sampling rate: {summary.sampling_rate_hz:.2f} Hz",
            ]
        )


def evaluate_and_save_sce_report(
    session: DatasetSession,
    *,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    configuration: SCEDetectionConfiguration | None = None,
) -> SCEReport:
    """Evaluates the SCE pipeline for a single session and persists the report to disk.

    Args:
        session: The DatasetSession to analyze.
        fluorescence_column: Fluorescence column to use as the analysis input.
        configuration: SCE detection parameters. Uses defaults if None.

    Returns:
        The SCEReport produced for the session, with all three artifacts persisted under the session directory.
    """
    report = SCEReport.evaluate(
        session_path=session.session_path,
        fluorescence_column=fluorescence_column,
        configuration=configuration,
    )
    report.save(session=session)
    return report


# ===== Private helpers ==========================================================================================


_SCE_CELLS_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    SCECellColumn.CELL_ID.value: pl.Int32,
    SCECellColumn.SCE_PARTICIPATION_COUNT.value: pl.Int32,
    SCECellColumn.SCE_PARTICIPATION_RATE.value: pl.Float32,
    SCECellColumn.SCE_EVENTS.value: pl.List(pl.List(pl.Int32)),
    SCECellColumn.SCE_PARTICIPATION_P_VALUE.value: pl.Float32,
    SCECellColumn.IS_SCE_CELL.value: pl.Boolean,
}


_SCE_PERIODS_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    SCEPeriodColumn.PERIOD_INDEX.value: pl.Int32,
    SCEPeriodColumn.PERIOD_STATE.value: pl.Utf8,
    SCEPeriodColumn.CELL_COUNT.value: pl.Int32,
    SCEPeriodColumn.SAMPLE_COUNT.value: pl.Int32,
    SCEPeriodColumn.SAMPLING_RATE_HZ.value: pl.Float32,
    SCEPeriodColumn.THRESHOLD.value: pl.Float32,
    SCEPeriodColumn.TIMESTAMPS_MINUTES.value: pl.List(pl.Float32),
    SCEPeriodColumn.COACTIVE_COUNTS.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_LABELS.value: pl.List(pl.Int32),
    SCEPeriodColumn.ONSET_CELL_INDICES.value: pl.List(pl.Int32),
    SCEPeriodColumn.ONSET_SAMPLE_INDICES.value: pl.List(pl.Int32),
    SCEPeriodColumn.TRIAL_IDS.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_SIZE.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_WIDTH_SAMPLES.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_PEAK_COACTIVE.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_INTER_EVENT_INTERVALS_SAMPLES.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_RATE_HZ.value: pl.Float32,
}


def _combine_pvalues_fisher(values: list[float]) -> float:
    """Combines a list of independent p-values via Fisher's method, returning the combined survival function.

    Notes:
        Uses an epsilon floor of 1e-10 to avoid ``log(0)`` for cells whose per-period jitter null returned
        ``p == 0``; the resulting combined p-value is bounded but conservative. Returns NaN for an empty input.
    """
    if not values:
        return float("nan")
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-10, 1.0)
    chi2_stat = float(-2.0 * np.sum(np.log(arr)))
    return float(chi2.sf(chi2_stat, df=2 * len(values)))


def _aggregate_sce_cell_columns(cell_count: int, sce_results: list[SCEResult]) -> dict:
    """Computes per-cell SCE participation and recruitment-significance metrics across every stationary period.

    References:
        - Modol et al. (2020). Hippocampal hub neurons. Nat Commun.
          https://doi.org/10.1038/s41467-020-18432-6 -- per-cell SCE recruitment significance ("super-rich"
          cells); per-period p-values combined here via Fisher's method (Fisher 1925).
    """
    # noinspection PyTypeChecker
    participation: NDArray[np.int32] = np.zeros(cell_count, dtype=np.int32)
    total_sces: int = 0
    sce_events: list[list[list[int]]] = [[] for _ in range(cell_count)]
    p_value_lists: list[list[float]] = [[] for _ in range(cell_count)]
    # noinspection PyTypeChecker
    is_sce_cell: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)

    for period_index, result in enumerate(sce_results):
        sce_count = int(np.max(result.sce_labels))
        total_sces += sce_count

        is_sce_cell |= result.is_sce_cell
        if not bool(np.all(np.isnan(result.participation_p_values))):
            for cell_index in range(cell_count):
                p_value = float(result.participation_p_values[cell_index])
                if not np.isnan(p_value):
                    p_value_lists[cell_index].append(p_value)

        if sce_count == 0:
            continue

        # noinspection PyTypeChecker
        sce_sample_indices: NDArray[np.int64] = np.where(result.sce_labels > 0)[0]
        # noinspection PyTypeChecker
        sample_to_sce: NDArray[np.float32] = np.zeros((result.onset_matrix.shape[1], sce_count), dtype=np.float32)
        sample_to_sce[sce_sample_indices, result.sce_labels[sce_sample_indices] - 1] = 1.0
        # noinspection PyTypeChecker
        cell_sce_participation: NDArray[np.bool_] = (result.onset_matrix.astype(np.float32) @ sample_to_sce) > 0
        participation += cell_sce_participation.sum(axis=1).astype(np.int32)

        for sce_label in range(1, sce_count + 1):
            # noinspection PyTypeChecker
            participating_indices: NDArray[np.int64] = np.where(cell_sce_participation[:, sce_label - 1])[0]
            for cell in participating_indices:
                sce_events[cell].append([period_index, sce_label])

    # noinspection PyTypeChecker
    rate: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if total_sces > 0:
        rate = (participation / total_sces).astype(np.float32)
    # noinspection PyTypeChecker
    combined_p: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    for cell_index in range(cell_count):
        combined_p[cell_index] = _combine_pvalues_fisher(values=p_value_lists[cell_index])

    return {
        "participation_count": participation,
        "participation_rate": rate,
        "sce_events": sce_events,
        "participation_p_value": combined_p,
        "is_sce_cell": is_sce_cell,
    }


def _build_sce_cells_table(cell_count: int, sce_results: list[SCEResult]) -> pl.DataFrame:
    """Assembles the per-cell SCE participation feather from the live detector outputs."""
    if cell_count == 0:
        return pl.DataFrame(schema=_SCE_CELLS_EMPTY_SCHEMA)

    columns = _aggregate_sce_cell_columns(cell_count=cell_count, sce_results=sce_results)
    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)
    return pl.DataFrame(
        {
            SCECellColumn.CELL_ID.value: pl.Series(values=cell_ids, dtype=pl.Int32),
            SCECellColumn.SCE_PARTICIPATION_COUNT.value: pl.Series(
                values=columns["participation_count"], dtype=pl.Int32
            ),
            SCECellColumn.SCE_PARTICIPATION_RATE.value: pl.Series(
                values=columns["participation_rate"], dtype=pl.Float32
            ),
            SCECellColumn.SCE_EVENTS.value: pl.Series(
                name=SCECellColumn.SCE_EVENTS.value,
                values=columns["sce_events"],
                dtype=pl.List(pl.List(pl.Int32)),
            ),
            SCECellColumn.SCE_PARTICIPATION_P_VALUE.value: pl.Series(
                values=columns["participation_p_value"], dtype=pl.Float32
            ),
            SCECellColumn.IS_SCE_CELL.value: pl.Series(values=columns["is_sce_cell"], dtype=pl.Boolean),
        },
    ).sort(SCECellColumn.CELL_ID.value)


def _build_sce_periods_table(
    sampling_rate_hz: float,
    results: list[SCEResult],
    cell_count: int,
) -> pl.DataFrame:
    """Assembles the per-period SCE feather, encoding the dense onset matrix as sparse cell/sample index lists
    and persisting the per-SCE descriptors.

    Args:
        sampling_rate_hz: Sampling rate in Hz; constant across all rows.
        results: List of stationary-period ``SCEResult`` instances in temporal session order.
        cell_count: Total number of cells in the session.

    Returns:
        A polars DataFrame following :class:`SCEPeriodColumn`.
    """
    if not results:
        return pl.DataFrame(schema=_SCE_PERIODS_EMPTY_SCHEMA)

    period_index_column: list[int] = []
    period_state_column: list[str] = []
    cell_count_column: list[int] = []
    sample_count_column: list[int] = []
    sampling_rate_column: list[float] = []
    threshold_column: list[float] = []
    timestamps_column: list[list[float]] = []
    coactive_counts_column: list[list[int]] = []
    sce_labels_column: list[list[int]] = []
    onset_cell_indices_column: list[list[int]] = []
    onset_sample_indices_column: list[list[int]] = []
    trial_ids_column: list[list[int]] = []
    sce_size_column: list[list[int]] = []
    sce_width_column: list[list[int]] = []
    sce_peak_column: list[list[int]] = []
    sce_inter_interval_column: list[list[int]] = []
    sce_rate_column: list[float] = []

    for period_index, result in enumerate(results):
        onset_cells, onset_samples = np.nonzero(result.onset_matrix)
        period_index_column.append(period_index)
        period_state_column.append(result.period_state)
        cell_count_column.append(cell_count)
        sample_count_column.append(int(result.onset_matrix.shape[1]))
        sampling_rate_column.append(sampling_rate_hz)
        threshold_column.append(float(result.threshold))
        timestamps_column.append([float(value) for value in result.timestamps.tolist()])
        coactive_counts_column.append([int(value) for value in result.coactive_counts.tolist()])
        sce_labels_column.append([int(value) for value in result.sce_labels.tolist()])
        onset_cell_indices_column.append([int(value) for value in onset_cells.tolist()])
        onset_sample_indices_column.append([int(value) for value in onset_samples.tolist()])
        trial_ids_column.append([int(value) for value in result.trial_ids.tolist()])
        sce_size_column.append([int(value) for value in result.sce_size.tolist()])
        sce_width_column.append([int(value) for value in result.sce_width_samples.tolist()])
        sce_peak_column.append([int(value) for value in result.sce_peak_coactive.tolist()])
        sce_inter_interval_column.append([int(value) for value in result.sce_inter_event_intervals_samples.tolist()])
        sce_rate_column.append(float(result.sce_rate_hz))

    return pl.DataFrame(
        {
            SCEPeriodColumn.PERIOD_INDEX.value: pl.Series(values=period_index_column, dtype=pl.Int32),
            SCEPeriodColumn.PERIOD_STATE.value: pl.Series(values=period_state_column, dtype=pl.Utf8),
            SCEPeriodColumn.CELL_COUNT.value: pl.Series(values=cell_count_column, dtype=pl.Int32),
            SCEPeriodColumn.SAMPLE_COUNT.value: pl.Series(values=sample_count_column, dtype=pl.Int32),
            SCEPeriodColumn.SAMPLING_RATE_HZ.value: pl.Series(values=sampling_rate_column, dtype=pl.Float32),
            SCEPeriodColumn.THRESHOLD.value: pl.Series(values=threshold_column, dtype=pl.Float32),
            SCEPeriodColumn.TIMESTAMPS_MINUTES.value: pl.Series(values=timestamps_column, dtype=pl.List(pl.Float32)),
            SCEPeriodColumn.COACTIVE_COUNTS.value: pl.Series(values=coactive_counts_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_LABELS.value: pl.Series(values=sce_labels_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.ONSET_CELL_INDICES.value: pl.Series(
                values=onset_cell_indices_column, dtype=pl.List(pl.Int32)
            ),
            SCEPeriodColumn.ONSET_SAMPLE_INDICES.value: pl.Series(
                values=onset_sample_indices_column, dtype=pl.List(pl.Int32)
            ),
            SCEPeriodColumn.TRIAL_IDS.value: pl.Series(values=trial_ids_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_SIZE.value: pl.Series(values=sce_size_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_WIDTH_SAMPLES.value: pl.Series(values=sce_width_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_PEAK_COACTIVE.value: pl.Series(values=sce_peak_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_INTER_EVENT_INTERVALS_SAMPLES.value: pl.Series(
                values=sce_inter_interval_column, dtype=pl.List(pl.Int32)
            ),
            SCEPeriodColumn.SCE_RATE_HZ.value: pl.Series(values=sce_rate_column, dtype=pl.Float32),
        }
    )
