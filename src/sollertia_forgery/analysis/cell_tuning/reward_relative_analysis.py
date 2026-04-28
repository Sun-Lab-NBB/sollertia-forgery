"""Within-session reward-relative analysis comparing two trial types whose reward locations differ.

Implements the Sosa, Plitt & Giocomo (2025) framework for distinguishing reward-anchored from track-anchored
cells. The session must contain at least two trial types whose reward zones sit at different track positions
(e.g., the ABC vs ABDC interleaved-trial design). For each cell whose tuning is statistically significant in
both trial types, the analysis compares peak position in track-aligned coordinates against peak position in
reward-aligned coordinates and classifies the cell by whichever shift is smaller.

References:
    - Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to reward.
      Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4 -- the reward-aligned coordinate framework, the
      ±50 cm circular peak-shift threshold, and the random-remapping cell-ID shuffle null.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import polars as pl
from ataraxis_data_structures import YamlConfig

from ...forging import FluorescenceColumn
from .cell_analysis import (
    CellAnalysisColumn,
    CellAnalysisReport,
    CellAnalysisConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_REWARD_RELATIVE_TABLE_TEMPLATE: str = "reward_relative_{a}_vs_{b}.feather"
"""Filename template for the persisted per-cell cross-trial-type table. Substitution variables are the two trial
type names; ordering matches the constructor."""
_REWARD_RELATIVE_SUMMARY_TEMPLATE: str = "reward_relative_{a}_vs_{b}.yaml"
"""Filename template for the persisted summary YAML."""


class RewardRelativeColumn(StrEnum):
    """Defines every column written to the per-session reward-relative per-cell table."""

    CELL_ID = "cell_id"
    """Contiguous integer cell identifier; matches ``CellAnalysisColumn.CELL_ID`` in both source reports."""
    PEAK_POSITION_TRACK_A_CM = "peak_position_track_a_cm"
    """Trial-type-A peak position in track-aligned coordinates (centimeters from track start)."""
    PEAK_POSITION_TRACK_B_CM = "peak_position_track_b_cm"
    """Trial-type-B peak position in track-aligned coordinates."""
    PEAK_POSITION_REWARD_A_CM = "peak_position_reward_a_cm"
    """Trial-type-A peak position in reward-aligned coordinates: signed circular offset from the trial-type-A reward
    midpoint, in ``[-track_length_a/2, track_length_a/2)``."""
    PEAK_POSITION_REWARD_B_CM = "peak_position_reward_b_cm"
    """Trial-type-B peak position in reward-aligned coordinates: signed circular offset from the trial-type-B reward
    midpoint, in ``[-track_length_b/2, track_length_b/2)``."""
    PEAK_SHIFT_TRACK_CM = "peak_shift_track_cm"
    """Absolute difference between trial-A and trial-B peaks in track-aligned coordinates. Small for cells whose
    activity is locked to a fixed track position regardless of reward location (track-anchored)."""
    PEAK_SHIFT_REWARD_CM = "peak_shift_reward_cm"
    """Absolute difference between trial-A and trial-B peaks in reward-aligned coordinates. Small for cells whose
    activity tracks the reward position across trial types (reward-anchored)."""
    PEAK_SHIFT_TRACK_P_VALUE = "peak_shift_track_p_value"
    """Random-remapping shuffle p-value for the track-aligned peak shift: fraction of cell-ID permutations whose
    shuffled shift is at or below the observed shift. Small p-values indicate the observed shift is unusually small
    relative to random pairings, supporting the track-anchored classification."""
    PEAK_SHIFT_REWARD_P_VALUE = "peak_shift_reward_p_value"
    """Random-remapping shuffle p-value for the reward-aligned peak shift; small p-values support the
    reward-anchored classification."""
    IS_REWARD_ANCHORED = "is_reward_anchored"
    """True when the reward-aligned peak shift is within the configured threshold AND the random-remapping p-value
    falls below ``significance_threshold``. Cells that fail the test in either trial type carry False."""
    IS_TRACK_ANCHORED = "is_track_anchored"
    """True when the track-aligned peak shift is within the configured threshold AND the random-remapping p-value
    falls below ``significance_threshold``."""
    IS_BOTH_ANCHORED = "is_both_anchored"
    """True when a cell satisfies both ``IS_REWARD_ANCHORED`` and ``IS_TRACK_ANCHORED`` simultaneously. Rare unless
    the reward shift is small relative to the threshold."""
    IS_TESTED = "is_tested"
    """True when the cell entered the cross-trial-type comparison: both per-trial-type reports flagged it as
    spatially significant. Cells with ``IS_TESTED=False`` carry NaN peaks and shifts."""


@dataclass(frozen=True, slots=True)
class RewardRelativeConfiguration:
    """Configuration for the within-session cross-trial-type reward-relative analysis."""

    peak_shift_threshold_cm: float = 50.0
    """Maximum peak shift (in centimeters) for a cell to be classified as reward- or track-anchored. Sosa, Plitt &
    Giocomo (2025) use 50 cm on a 450 cm track; scale this with track length if your design differs substantially."""
    shuffle_count: int = 1000
    """Number of cell-ID permutations for the random-remapping null distribution. Sosa et al. (2025) use 1000."""
    significance_threshold: float = 0.05
    """Per-cell p-value cutoff for the anchored-cell flags. The full classification rule is ``shift <= threshold AND
    p_value < significance_threshold``; the threshold prunes large-shift cells from the random-remapping tail."""
    require_significant_in_both: bool = True
    """When True (default), only cells flagged ``IS_SPATIALLY_SIGNIFICANT`` in both trial-type reports enter the
    test. When False, cells significant in either trial type are tested with NaN peaks for missing trial types."""


@dataclass
class RewardRelativeSummary(YamlConfig):
    """Per-session reward-relative summary persisted alongside the table.

    Notes:
        Holds the trial-type pair, geometry, count statistics, and the configuration so the analysis is
        self-describing on disk. Fields parallel the ``CellAnalysisSummary`` style.
    """

    trial_type_a: str
    """First trial type name (e.g., 'ABC')."""
    trial_type_b: str
    """Second trial type name (e.g., 'ABCD'); reward must sit at a different track position than trial type A for
    the comparison to be informative."""
    track_length_a_cm: float
    """Trial-type-A canonical track length in centimeters."""
    track_length_b_cm: float
    """Trial-type-B canonical track length in centimeters."""
    reward_position_a_cm: float
    """Trial-type-A reward midpoint in centimeters."""
    reward_position_b_cm: float
    """Trial-type-B reward midpoint in centimeters."""
    reward_shift_cm: float
    """Signed shift of the reward midpoint from trial type A to trial type B (B minus A)."""
    cell_count: int
    """Total cells in the session (matches both source reports)."""
    tested_count: int
    """Cells admitted to the cross-trial-type comparison."""
    reward_anchored_count: int
    """Cells classified as reward-anchored."""
    track_anchored_count: int
    """Cells classified as track-anchored."""
    both_anchored_count: int
    """Cells classified as both reward- and track-anchored simultaneously."""
    neither_anchored_count: int
    """Tested cells that satisfied neither anchored classification."""
    configuration: RewardRelativeConfiguration
    """The configuration used to produce this summary."""


@dataclass(frozen=True, slots=True)
class RewardRelativeReport:
    """Per-session container for the cross-trial-type reward-relative analysis."""

    table: pl.DataFrame
    """Per-cell wide table; one row per cell. Schema enumerated by :class:`RewardRelativeColumn`."""
    summary: RewardRelativeSummary
    """Summary YAML wrapper holding configuration, trial-type pair, and aggregate counts."""

    @classmethod
    def evaluate(
        cls,
        session_path: Path,
        *,
        trial_type_a: str,
        trial_type_b: str,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        configuration: RewardRelativeConfiguration | None = None,
        cell_analysis_configuration: CellAnalysisConfiguration | None = None,
    ) -> RewardRelativeReport:
        """Runs both per-trial-type cell analyses and assembles the cross-trial-type reward-relative report.

        Notes:
            Calls ``CellAnalysisReport.evaluate`` once per trial type and forwards both reports to
            :meth:`from_reports`. The two per-trial-type reports are not persisted; callers who want them on disk
            should run :meth:`CellAnalysisReport.evaluate` themselves and use :meth:`from_reports`.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type_a: First trial type to analyze; must match an entry in the session's trial geometry data file.
            trial_type_b: Second trial type to analyze; must match an entry in the session's trial geometry data file
                and should have a different reward position than trial type A for the comparison to be informative.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input. Applied uniformly to both trial-type analyses.
            configuration: Reward-relative configuration; uses defaults if None.
            cell_analysis_configuration: Cell-analysis configuration forwarded to both per-trial-type evaluations;
                uses defaults if None.

        Returns:
            A RewardRelativeReport instance with the per-cell cross-trial-type table and summary.
        """
        report_a = CellAnalysisReport.evaluate(
            session_path=session_path,
            trial_type=trial_type_a,
            fluorescence_column=fluorescence_column,
            configuration=cell_analysis_configuration,
        )
        report_b = CellAnalysisReport.evaluate(
            session_path=session_path,
            trial_type=trial_type_b,
            fluorescence_column=fluorescence_column,
            configuration=cell_analysis_configuration,
        )
        return cls.from_reports(
            report_a=report_a,
            report_b=report_b,
            trial_type_a=trial_type_a,
            trial_type_b=trial_type_b,
            configuration=configuration,
        )

    @classmethod
    def from_reports(
        cls,
        report_a: CellAnalysisReport,
        report_b: CellAnalysisReport,
        *,
        trial_type_a: str,
        trial_type_b: str,
        configuration: RewardRelativeConfiguration | None = None,
    ) -> RewardRelativeReport:
        """Assembles the cross-trial-type report from two pre-computed per-trial-type cell-analysis reports.

        Args:
            report_a: Cell-analysis report for trial type A.
            report_b: Cell-analysis report for trial type B.
            trial_type_a: Name of trial type A; recorded in the summary.
            trial_type_b: Name of trial type B.
            configuration: Reward-relative configuration; uses defaults if None.

        Returns:
            A RewardRelativeReport instance.

        Raises:
            ValueError: When the two reports report different cell counts.
        """
        resolved_configuration = configuration if configuration is not None else RewardRelativeConfiguration()
        cell_count_a = report_a.summary.cell_count
        cell_count_b = report_b.summary.cell_count
        if cell_count_a != cell_count_b:
            message = (
                f"Cannot build a cross-trial-type reward-relative report from reports with mismatched cell counts: "
                f"trial type '{trial_type_a}' has {cell_count_a} cells while trial type '{trial_type_b}' has "
                f"{cell_count_b}. Both reports must come from the same session and the same fluorescence input so "
                f"cell IDs match across trial types."
            )
            raise ValueError(message)

        table, anchored_counts = _build_table(
            report_a=report_a,
            report_b=report_b,
            configuration=resolved_configuration,
        )

        summary = RewardRelativeSummary(
            trial_type_a=trial_type_a,
            trial_type_b=trial_type_b,
            track_length_a_cm=float(report_a.summary.track_length_cm),
            track_length_b_cm=float(report_b.summary.track_length_cm),
            reward_position_a_cm=float(report_a.summary.reward_position_cm),
            reward_position_b_cm=float(report_b.summary.reward_position_cm),
            reward_shift_cm=float(report_b.summary.reward_position_cm - report_a.summary.reward_position_cm),
            cell_count=int(cell_count_a),
            tested_count=int(anchored_counts["tested"]),
            reward_anchored_count=int(anchored_counts["reward_anchored"]),
            track_anchored_count=int(anchored_counts["track_anchored"]),
            both_anchored_count=int(anchored_counts["both_anchored"]),
            neither_anchored_count=int(anchored_counts["neither_anchored"]),
            configuration=resolved_configuration,
        )
        return cls(table=table, summary=summary)

    @classmethod
    def load(cls, session_path: Path, *, trial_type_a: str, trial_type_b: str) -> RewardRelativeReport:
        """Loads a previously saved cross-trial-type report from the session directory.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type_a: First trial type name used at save time.
            trial_type_b: Second trial type name used at save time.

        Returns:
            A RewardRelativeReport with the table memory-mapped and the summary loaded.
        """
        table_path = session_path.joinpath(_REWARD_RELATIVE_TABLE_TEMPLATE.format(a=trial_type_a, b=trial_type_b))
        summary_path = session_path.joinpath(_REWARD_RELATIVE_SUMMARY_TEMPLATE.format(a=trial_type_a, b=trial_type_b))
        summary = RewardRelativeSummary.from_yaml(file_path=summary_path)
        table = pl.read_ipc(source=table_path, memory_map=True)
        return cls(table=table, summary=summary)

    def save(self, session_path: Path) -> None:
        """Persists the report to two artifacts inside the session directory using trial-type-suffixed filenames.

        Args:
            session_path: Path to the session's dataset directory.
        """
        table_path = session_path.joinpath(
            _REWARD_RELATIVE_TABLE_TEMPLATE.format(a=self.summary.trial_type_a, b=self.summary.trial_type_b)
        )
        summary_path = session_path.joinpath(
            _REWARD_RELATIVE_SUMMARY_TEMPLATE.format(a=self.summary.trial_type_a, b=self.summary.trial_type_b)
        )
        self.summary.to_yaml(file_path=summary_path)
        self.table.write_ipc(file=table_path)

    def summarize(self) -> str:
        """Returns a multi-line human-readable summary of the cross-trial-type analysis."""
        summary = self.summary
        cell_count = summary.cell_count
        tested_pct = 100.0 * summary.tested_count / cell_count if cell_count > 0 else 0.0
        reward_pct = 100.0 * summary.reward_anchored_count / cell_count if cell_count > 0 else 0.0
        track_pct = 100.0 * summary.track_anchored_count / cell_count if cell_count > 0 else 0.0
        both_pct = 100.0 * summary.both_anchored_count / cell_count if cell_count > 0 else 0.0
        neither_pct = 100.0 * summary.neither_anchored_count / cell_count if cell_count > 0 else 0.0
        return "\n".join(
            [
                f"Reward-relative analysis: {summary.trial_type_a} vs {summary.trial_type_b}",
                "=" * 60,
                f"Track lengths:   A={summary.track_length_a_cm:.1f} cm, B={summary.track_length_b_cm:.1f} cm",
                f"Reward position: A={summary.reward_position_a_cm:.1f} cm, "
                f"B={summary.reward_position_b_cm:.1f} cm (shift {summary.reward_shift_cm:+.1f} cm)",
                "",
                f"Cells: {cell_count}",
                f"  Tested:           {summary.tested_count} ({tested_pct:.1f}%)",
                f"  Reward-anchored:  {summary.reward_anchored_count} ({reward_pct:.1f}%)",
                f"  Track-anchored:   {summary.track_anchored_count} ({track_pct:.1f}%)",
                f"  Both anchored:    {summary.both_anchored_count} ({both_pct:.1f}%)",
                f"  Neither anchored: {summary.neither_anchored_count} ({neither_pct:.1f}%)",
                "",
                f"Configuration: peak_shift_threshold={summary.configuration.peak_shift_threshold_cm:.0f} cm, "
                f"shuffle_count={summary.configuration.shuffle_count}, "
                f"p<{summary.configuration.significance_threshold:.2f}",
            ]
        )


@dataclass(frozen=True, slots=True)
class CrossBlockClassification:
    """Per-cell peak positions, shifts, p-values, and anchored flags from the cross-block analysis kernel.

    Notes:
        Shared output struct for the within-session (Phase 3) and cross-session (Phase 4) reward-relative analyses.
        All arrays have length cell_count. NaN is propagated for cells that did not enter the test (e.g., not
        significant in both blocks); the corresponding boolean flags are False.
    """

    peaks_track_a: NDArray[np.float32]
    peaks_track_b: NDArray[np.float32]
    peaks_reward_a: NDArray[np.float32]
    peaks_reward_b: NDArray[np.float32]
    peak_shift_track: NDArray[np.float32]
    peak_shift_reward: NDArray[np.float32]
    p_values_track: NDArray[np.float32]
    p_values_reward: NDArray[np.float32]
    is_reward_anchored: NDArray[np.bool_]
    is_track_anchored: NDArray[np.bool_]
    is_both_anchored: NDArray[np.bool_]
    is_tested: NDArray[np.bool_]


def compute_cross_block_classification(
    rate_maps_a: NDArray[np.float32],
    rate_maps_b: NDArray[np.float32],
    *,
    is_significant_a: NDArray[np.bool_],
    is_significant_b: NDArray[np.bool_],
    track_length_a_cm: float,
    track_length_b_cm: float,
    reward_position_a_cm: float,
    reward_position_b_cm: float,
    bin_size_a_cm: float,
    bin_size_b_cm: float,
    configuration: RewardRelativeConfiguration,
) -> CrossBlockClassification:
    """Computes per-cell peak positions, shifts, random-remapping p-values, and anchored-cell flags.

    Notes:
        Shared kernel for the within-session (Phase 3) and cross-session (Phase 4) reward-relative analyses. Takes
        rate maps and metadata directly so callers can build the rate maps from a single trial type, average them
        across sessions, or assemble them from any other source as long as cell IDs align across the two inputs.

    Args:
        rate_maps_a: Per-cell rate maps for block A with dimensions (cell_count, bin_count_a).
        rate_maps_b: Per-cell rate maps for block B with dimensions (cell_count, bin_count_b).
        is_significant_a: Per-cell spatial significance for block A with length cell_count.
        is_significant_b: Per-cell spatial significance for block B with length cell_count.
        track_length_a_cm: Track length for block A in centimeters.
        track_length_b_cm: Track length for block B in centimeters.
        reward_position_a_cm: Reward midpoint for block A in centimeters.
        reward_position_b_cm: Reward midpoint for block B in centimeters.
        bin_size_a_cm: Spatial bin size for block A in centimeters.
        bin_size_b_cm: Spatial bin size for block B in centimeters.
        configuration: Reward-relative configuration.

    Returns:
        A CrossBlockClassification dataclass holding every per-cell array.
    """
    cell_count = rate_maps_a.shape[0]

    # Peaks in track-aligned coordinates: bin index of the per-cell argmax converted to centimeters at the bin
    # center. Cells whose rate map sums to zero (uniformly silent) get NaN peaks and are excluded downstream.
    # noinspection PyTypeChecker
    peaks_track_a: NDArray[np.float32] = _resolve_peak_positions(rate_maps=rate_maps_a, bin_size_cm=bin_size_a_cm)
    # noinspection PyTypeChecker
    peaks_track_b: NDArray[np.float32] = _resolve_peak_positions(rate_maps=rate_maps_b, bin_size_cm=bin_size_b_cm)

    # Reward-aligned coordinates: signed circular offset from the block-specific reward midpoint, in
    # ``[-track_length/2, track_length/2)``. Cells with NaN track peaks propagate NaN.
    # noinspection PyTypeChecker
    peaks_reward_a: NDArray[np.float32] = _signed_circular_offset(
        positions=peaks_track_a, anchor=reward_position_a_cm, track_length=track_length_a_cm
    )
    # noinspection PyTypeChecker
    peaks_reward_b: NDArray[np.float32] = _signed_circular_offset(
        positions=peaks_track_b, anchor=reward_position_b_cm, track_length=track_length_b_cm
    )

    # Per-cell observed peak shifts in each coordinate system. Track-aligned uses absolute centimeters; the two
    # tracks may have different lengths but the leading segment (e.g., AB in ABC vs ABDC, or the entire track for
    # MF reward-shift sessions) is the same physical environment, so absolute cm is the right scale. Reward-aligned
    # uses absolute centimeters of the signed circular offset; cells that fire on the same side of reward in both
    # blocks stay on the same scale, and cells that flip sides get a large shift (correct behavior — those are not
    # reward-anchored).
    # noinspection PyTypeChecker
    peak_shift_track: NDArray[np.float32] = np.abs(peaks_track_a - peaks_track_b).astype(np.float32)
    # noinspection PyTypeChecker
    peak_shift_reward: NDArray[np.float32] = np.abs(peaks_reward_a - peaks_reward_b).astype(np.float32)

    # Random-remapping null: per shuffle iteration, permute block-B cell IDs and recompute peak shifts. The per-cell
    # p-value is the fraction of permutations whose shuffled shift is <= observed.
    if configuration.require_significant_in_both:
        is_tested_mask = is_significant_a & is_significant_b
    else:
        is_tested_mask = is_significant_a | is_significant_b

    # noinspection PyTypeChecker
    valid_for_shuffle: NDArray[np.bool_] = is_tested_mask & ~np.isnan(peaks_track_a) & ~np.isnan(peaks_track_b)

    p_track = np.full(cell_count, np.nan, dtype=np.float32)
    p_reward = np.full(cell_count, np.nan, dtype=np.float32)
    if int(np.sum(valid_for_shuffle)) >= 2:
        p_track[valid_for_shuffle], p_reward[valid_for_shuffle] = _compute_random_remapping_p_values(
            peaks_track_a=peaks_track_a[valid_for_shuffle],
            peaks_track_b=peaks_track_b[valid_for_shuffle],
            peaks_reward_a=peaks_reward_a[valid_for_shuffle],
            peaks_reward_b=peaks_reward_b[valid_for_shuffle],
            shuffle_count=configuration.shuffle_count,
        )

    is_reward_anchored = (
        valid_for_shuffle
        & (peak_shift_reward <= configuration.peak_shift_threshold_cm)
        & (p_reward < configuration.significance_threshold)
    )
    is_track_anchored = (
        valid_for_shuffle
        & (peak_shift_track <= configuration.peak_shift_threshold_cm)
        & (p_track < configuration.significance_threshold)
    )
    is_both_anchored = is_reward_anchored & is_track_anchored

    return CrossBlockClassification(
        peaks_track_a=peaks_track_a,
        peaks_track_b=peaks_track_b,
        peaks_reward_a=peaks_reward_a,
        peaks_reward_b=peaks_reward_b,
        peak_shift_track=peak_shift_track,
        peak_shift_reward=peak_shift_reward,
        p_values_track=p_track,
        p_values_reward=p_reward,
        is_reward_anchored=is_reward_anchored,
        is_track_anchored=is_track_anchored,
        is_both_anchored=is_both_anchored,
        is_tested=valid_for_shuffle,
    )


def _build_table(
    report_a: CellAnalysisReport,
    report_b: CellAnalysisReport,
    configuration: RewardRelativeConfiguration,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Builds the per-cell cross-trial-type table and returns aggregate anchored-cell counts."""
    cell_count = report_a.summary.cell_count
    summary_a = report_a.summary
    summary_b = report_b.summary

    rate_maps_a = stack_rate_maps_from_table(report_a.table, target_length=int(summary_a.bin_count))
    rate_maps_b = stack_rate_maps_from_table(report_b.table, target_length=int(summary_b.bin_count))

    # noinspection PyTypeChecker
    is_significant_a: NDArray[np.bool_] = report_a.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_significant_b: NDArray[np.bool_] = report_b.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()

    classification = compute_cross_block_classification(
        rate_maps_a=rate_maps_a,
        rate_maps_b=rate_maps_b,
        is_significant_a=is_significant_a,
        is_significant_b=is_significant_b,
        track_length_a_cm=float(summary_a.track_length_cm),
        track_length_b_cm=float(summary_b.track_length_cm),
        reward_position_a_cm=float(summary_a.reward_position_cm),
        reward_position_b_cm=float(summary_b.reward_position_cm),
        bin_size_a_cm=float(summary_a.bin_size_cm),
        bin_size_b_cm=float(summary_b.bin_size_cm),
        configuration=configuration,
    )

    table = build_cross_block_table(cell_count=cell_count, classification=classification)
    counts = aggregate_cross_block_counts(classification=classification)
    return table, counts


def build_cross_block_table(
    *,
    cell_count: int,
    classification: CrossBlockClassification,
) -> pl.DataFrame:
    """Assembles the per-cell wide-format DataFrame from a CrossBlockClassification.

    Notes:
        Shared between the within-session and cross-session reports so both produce the same column schema. The
        table is sorted by cell ID for stable downstream consumers.
    """
    return pl.DataFrame(
        {
            RewardRelativeColumn.CELL_ID.value: np.arange(cell_count, dtype=np.int32),
            RewardRelativeColumn.PEAK_POSITION_TRACK_A_CM.value: pl.Series(
                values=classification.peaks_track_a, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_POSITION_TRACK_B_CM.value: pl.Series(
                values=classification.peaks_track_b, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_POSITION_REWARD_A_CM.value: pl.Series(
                values=classification.peaks_reward_a, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_POSITION_REWARD_B_CM.value: pl.Series(
                values=classification.peaks_reward_b, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_SHIFT_TRACK_CM.value: pl.Series(
                values=classification.peak_shift_track, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_SHIFT_REWARD_CM.value: pl.Series(
                values=classification.peak_shift_reward, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_SHIFT_TRACK_P_VALUE.value: pl.Series(
                values=classification.p_values_track, dtype=pl.Float32
            ),
            RewardRelativeColumn.PEAK_SHIFT_REWARD_P_VALUE.value: pl.Series(
                values=classification.p_values_reward, dtype=pl.Float32
            ),
            RewardRelativeColumn.IS_REWARD_ANCHORED.value: pl.Series(
                values=classification.is_reward_anchored, dtype=pl.Boolean
            ),
            RewardRelativeColumn.IS_TRACK_ANCHORED.value: pl.Series(
                values=classification.is_track_anchored, dtype=pl.Boolean
            ),
            RewardRelativeColumn.IS_BOTH_ANCHORED.value: pl.Series(
                values=classification.is_both_anchored, dtype=pl.Boolean
            ),
            RewardRelativeColumn.IS_TESTED.value: pl.Series(values=classification.is_tested, dtype=pl.Boolean),
        }
    ).sort(RewardRelativeColumn.CELL_ID.value)


def aggregate_cross_block_counts(*, classification: CrossBlockClassification) -> dict[str, int]:
    """Returns aggregate anchored-cell counts from a CrossBlockClassification."""
    is_neither = classification.is_tested & ~classification.is_reward_anchored & ~classification.is_track_anchored
    return {
        "tested": int(np.sum(classification.is_tested)),
        "reward_anchored": int(np.sum(classification.is_reward_anchored)),
        "track_anchored": int(np.sum(classification.is_track_anchored)),
        "both_anchored": int(np.sum(classification.is_both_anchored)),
        "neither_anchored": int(np.sum(is_neither)),
    }


def stack_rate_maps_from_table(table: pl.DataFrame, target_length: int) -> NDArray[np.float32]:
    """Stacks the per-cell list-of-bins rate-map column into a contiguous (cell_count, bin_count) array.

    Notes:
        Pads or truncates each row to ``target_length`` so callers can rely on a uniform second axis even when the
        list column carries variable-length entries from older saved reports.
    """
    rows = table[CellAnalysisColumn.RATE_MAP.value].to_list()
    cell_count = len(rows)
    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.zeros((cell_count, target_length), dtype=np.float32)
    for cell_index, row in enumerate(rows):
        values = np.asarray(row, dtype=np.float32) if row is not None else np.zeros(0, dtype=np.float32)
        clipped = values[:target_length]
        output[cell_index, : clipped.shape[0]] = clipped
    return output


def _resolve_peak_positions(rate_maps: NDArray[np.float32], bin_size_cm: float) -> NDArray[np.float32]:
    """Returns the per-cell peak position in centimeters at the bin center; NaN for cells with all-zero rate maps."""
    cell_count = rate_maps.shape[0]
    # noinspection PyTypeChecker
    peaks: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    for cell_index in range(cell_count):
        # noinspection PyTypeChecker
        row: NDArray[np.float32] = rate_maps[cell_index]
        if row.size == 0 or not np.isfinite(row).any() or float(np.nanmax(row)) <= 0.0:
            continue
        peak_bin = int(np.nanargmax(row))
        peaks[cell_index] = (peak_bin + 0.5) * bin_size_cm
    return peaks


def _signed_circular_offset(positions: NDArray[np.float32], anchor: float, track_length: float) -> NDArray[np.float32]:
    """Returns the signed circular offset of each position from ``anchor`` in ``[-track_length/2, track_length/2)``.

    Notes:
        NaN positions propagate. The result preserves direction: negative values are upstream of the anchor,
        positive values are downstream.
    """
    half = track_length / 2.0
    # noinspection PyTypeChecker
    offset: NDArray[np.float32] = ((positions - anchor + half) % track_length - half).astype(np.float32)
    offset[np.isnan(positions)] = np.nan
    return offset


def _compute_random_remapping_p_values(
    peaks_track_a: NDArray[np.float32],
    peaks_track_b: NDArray[np.float32],
    peaks_reward_a: NDArray[np.float32],
    peaks_reward_b: NDArray[np.float32],
    shuffle_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-cell p-values from the random-remapping cell-ID shuffle (Sosa, Plitt & Giocomo 2025).

    Notes:
        For each shuffle iteration, the trial-B peaks are permuted across cells. The p-value reports the fraction
        of shuffles whose shuffled shift is at or below the observed shift, so cells whose true shift is unusually
        small relative to random pairings receive small p-values. Track-aligned and reward-aligned shifts use the
        same permutation per iteration so the two p-values are directly comparable across cells.

    Args:
        peaks_track_a: Trial-A peaks in track-aligned coords (length tested_count).
        peaks_track_b: Trial-B peaks in track-aligned coords (length tested_count).
        peaks_reward_a: Trial-A peaks in reward-aligned coords (length tested_count).
        peaks_reward_b: Trial-B peaks in reward-aligned coords (length tested_count).
        shuffle_count: Number of cell-ID permutations.

    Returns:
        A tuple of (p_values_track, p_values_reward) per-cell arrays of length tested_count.
    """
    tested_count = peaks_track_a.shape[0]
    observed_track = np.abs(peaks_track_a - peaks_track_b)
    observed_reward = np.abs(peaks_reward_a - peaks_reward_b)

    # noinspection PyTypeChecker
    le_count_track: NDArray[np.int64] = np.zeros(tested_count, dtype=np.int64)
    # noinspection PyTypeChecker
    le_count_reward: NDArray[np.int64] = np.zeros(tested_count, dtype=np.int64)

    for iteration in tqdm(range(shuffle_count), desc="Reward-relative shuffle", unit="iter", leave=False):
        generator = np.random.default_rng(seed=iteration)
        permutation = generator.permutation(tested_count)
        # noinspection PyTypeChecker
        shuffled_track: NDArray[np.float32] = np.abs(peaks_track_a - peaks_track_b[permutation])
        # noinspection PyTypeChecker
        shuffled_reward: NDArray[np.float32] = np.abs(peaks_reward_a - peaks_reward_b[permutation])
        le_count_track += (shuffled_track <= observed_track).astype(np.int64)
        le_count_reward += (shuffled_reward <= observed_reward).astype(np.int64)

    # noinspection PyTypeChecker
    p_track: NDArray[np.float32] = (le_count_track.astype(np.float32) / float(shuffle_count)).astype(np.float32)
    # noinspection PyTypeChecker
    p_reward: NDArray[np.float32] = (le_count_reward.astype(np.float32) / float(shuffle_count)).astype(np.float32)
    return p_track, p_reward
