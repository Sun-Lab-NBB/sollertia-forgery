"""Cross-session reward-shift analysis comparing pre-shift and post-shift sessions in a longitudinal cohort.

Implements the longitudinal counterpart to :mod:`reward_relative_analysis` for animals whose reward zone is moved
mid-cohort (e.g., MF11, MF15 — last five days at a shifted reward location). Consumes per-session
:class:`CellAnalysisReport` instances, partitions them into pre-shift and post-shift groups by reward midpoint,
averages rate maps within each group, and runs the shared cross-block kernel
(:func:`compute_cross_block_classification`) to classify each longitudinally-tracked cell as reward-anchored or
track-anchored.

Requires multi-day cell registration so cell IDs are stable across sessions; cindra's ``MULTI_DAY_*`` fluorescence
columns provide this by construction. Cells absent from a session are expected to carry NaN rate maps from the
upstream multi-day pipeline; per-cell session-coverage statistics are reported alongside the classification.

References:
    - Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to reward.
      Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4 -- the reward-aligned coordinate framework, the
      ±50 cm circular peak-shift threshold, and the random-remapping cell-ID shuffle null. Their reward-switch
      manipulation is the closest published analog to the MF cross-day reward-shift design.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
import polars as pl
from ataraxis_data_structures import YamlConfig

from .cell_analysis import CellAnalysisColumn, CellAnalysisReport
from .reward_relative_analysis import (
    RewardRelativeColumn,
    CrossBlockClassification,
    RewardRelativeConfiguration,
    build_cross_block_table,
    stack_rate_maps_from_table,
    aggregate_cross_block_counts,
    compute_cross_block_classification,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from numpy.typing import NDArray


_LONGITUDINAL_TABLE_FILENAME: str = "longitudinal_reward_shift.feather"
"""Filename for the persisted per-cell cross-session table inside the animal directory."""
_LONGITUDINAL_SUMMARY_FILENAME: str = "longitudinal_reward_shift.yaml"
"""Filename for the persisted summary YAML inside the animal directory."""


class LongitudinalRewardShiftColumn(StrEnum):
    """Defines every column written to the per-animal longitudinal reward-shift per-cell table.

    Notes:
        Inherits the cross-block columns from :class:`RewardRelativeColumn` and adds per-cell session-coverage
        statistics so downstream consumers can filter by how many sessions each cell was actually detected in.
    """

    CELL_ID = RewardRelativeColumn.CELL_ID.value
    PEAK_POSITION_TRACK_PRE_CM = RewardRelativeColumn.PEAK_POSITION_TRACK_A_CM.value
    PEAK_POSITION_TRACK_POST_CM = RewardRelativeColumn.PEAK_POSITION_TRACK_B_CM.value
    PEAK_POSITION_REWARD_PRE_CM = RewardRelativeColumn.PEAK_POSITION_REWARD_A_CM.value
    PEAK_POSITION_REWARD_POST_CM = RewardRelativeColumn.PEAK_POSITION_REWARD_B_CM.value
    PEAK_SHIFT_TRACK_CM = RewardRelativeColumn.PEAK_SHIFT_TRACK_CM.value
    PEAK_SHIFT_REWARD_CM = RewardRelativeColumn.PEAK_SHIFT_REWARD_CM.value
    PEAK_SHIFT_TRACK_P_VALUE = RewardRelativeColumn.PEAK_SHIFT_TRACK_P_VALUE.value
    PEAK_SHIFT_REWARD_P_VALUE = RewardRelativeColumn.PEAK_SHIFT_REWARD_P_VALUE.value
    IS_REWARD_ANCHORED = RewardRelativeColumn.IS_REWARD_ANCHORED.value
    IS_TRACK_ANCHORED = RewardRelativeColumn.IS_TRACK_ANCHORED.value
    IS_BOTH_ANCHORED = RewardRelativeColumn.IS_BOTH_ANCHORED.value
    IS_TESTED = RewardRelativeColumn.IS_TESTED.value
    SIGNIFICANT_IN_PRE_FRACTION = "significant_in_pre_fraction"
    """Fraction of pre-shift sessions in which this cell was flagged ``IS_SPATIALLY_SIGNIFICANT``."""
    SIGNIFICANT_IN_POST_FRACTION = "significant_in_post_fraction"
    """Fraction of post-shift sessions in which this cell was flagged ``IS_SPATIALLY_SIGNIFICANT``."""


@dataclass(frozen=True, slots=True)
class LongitudinalRewardShiftConfiguration:
    """Configuration for the per-animal cross-session reward-shift analysis."""

    peak_shift_threshold_cm: float = 50.0
    """Maximum peak shift (in centimeters) for a cell to be classified as reward- or track-anchored. Sosa, Plitt &
    Giocomo (2025) use 50 cm."""
    shuffle_count: int = 1000
    """Number of cell-ID permutations for the random-remapping null distribution."""
    significance_threshold: float = 0.05
    """Per-cell p-value cutoff for the anchored-cell flags."""
    require_significant_in_both: bool = True
    """When True (default), only cells that meet the per-block significance fraction in both blocks are tested."""
    minimum_sessions_per_block: int = 1
    """Minimum number of sessions in each (pre, post) block for the analysis to proceed."""
    minimum_significance_fraction: float = 0.5
    """Per-cell significance fraction required for a cell to count as significant in a block. With the default 0.5,
    a cell is significant in a block when it passes the per-session ``IS_SPATIALLY_SIGNIFICANT`` flag in at least
    half of that block's sessions."""
    reward_position_tolerance_cm: float = 1.0
    """Centimeter tolerance for grouping sessions by reward position. Differences within this tolerance are treated
    as the same reward location."""

    def to_reward_relative_configuration(self) -> RewardRelativeConfiguration:
        """Returns a :class:`RewardRelativeConfiguration` with the matching subset of fields."""
        return RewardRelativeConfiguration(
            peak_shift_threshold_cm=self.peak_shift_threshold_cm,
            shuffle_count=self.shuffle_count,
            significance_threshold=self.significance_threshold,
            require_significant_in_both=self.require_significant_in_both,
        )


@dataclass
class LongitudinalRewardShiftSummary(YamlConfig):
    """Per-animal longitudinal reward-shift summary persisted alongside the table."""

    animal: str
    """Animal identifier."""
    pre_shift_session_names: list[str] = field(default_factory=list)
    """Chronologically-ordered names of the sessions assigned to the pre-shift block."""
    post_shift_session_names: list[str] = field(default_factory=list)
    """Chronologically-ordered names of the sessions assigned to the post-shift block."""
    track_length_pre_cm: float = 0.0
    """Track length used in the pre-shift sessions (centimeters); the analysis assumes track length is constant
    within each block but allows it to differ between blocks."""
    track_length_post_cm: float = 0.0
    """Track length used in the post-shift sessions (centimeters)."""
    reward_position_pre_cm: float = 0.0
    """Reward midpoint in pre-shift sessions (centimeters)."""
    reward_position_post_cm: float = 0.0
    """Reward midpoint in post-shift sessions (centimeters)."""
    reward_shift_cm: float = 0.0
    """Signed shift of the reward midpoint from pre-shift to post-shift (post minus pre)."""
    cell_count: int = 0
    """Total cells in the multi-day registration."""
    tested_count: int = 0
    """Cells admitted to the cross-block comparison."""
    reward_anchored_count: int = 0
    """Cells classified as reward-anchored across the shift."""
    track_anchored_count: int = 0
    """Cells classified as track-anchored across the shift."""
    both_anchored_count: int = 0
    """Cells classified as both reward- and track-anchored simultaneously."""
    neither_anchored_count: int = 0
    """Tested cells that satisfied neither anchored classification."""
    configuration: LongitudinalRewardShiftConfiguration = field(default_factory=LongitudinalRewardShiftConfiguration)
    """The configuration used to produce this summary."""


@dataclass(frozen=True, slots=True)
class LongitudinalRewardShiftReport:
    """Per-animal container for the cross-session reward-shift analysis."""

    table: pl.DataFrame
    """Per-cell wide table; one row per longitudinally-tracked cell. Schema enumerated by
    :class:`LongitudinalRewardShiftColumn`."""
    summary: LongitudinalRewardShiftSummary
    """Summary YAML wrapper holding configuration, session lists, geometry, and aggregate counts."""

    @classmethod
    def evaluate(
        cls,
        *,
        animal: str,
        reports: Sequence[CellAnalysisReport],
        session_names: Sequence[str],
        configuration: LongitudinalRewardShiftConfiguration | None = None,
    ) -> LongitudinalRewardShiftReport:
        """Detects the reward shift across a chronological session sequence and applies the cross-block kernel.

        Notes:
            Sessions are sorted by ``session_names`` lexicographically (Sollertia session names are timestamp-based
            and so sort chronologically). The shift boundary is detected as the first session whose reward midpoint
            differs from the first session's reward midpoint by more than ``reward_position_tolerance_cm``. Sessions
            before the boundary form the pre-shift block; sessions from the boundary onward form the post-shift
            block. Multi-shift sequences (more than one boundary) are not supported.

            Per-cell rate maps are averaged with ``np.nanmean`` across sessions in each block so cells absent from a
            session (NaN row from the upstream multi-day pipeline) do not dilute the average. A cell counts as
            significant in a block when it passes ``IS_SPATIALLY_SIGNIFICANT`` in at least
            ``minimum_significance_fraction`` of that block's sessions.

        Args:
            animal: Animal identifier; persisted in the summary.
            reports: Per-session :class:`CellAnalysisReport` instances; must share a common cell-ID space (multi-day
                registration). Length must equal ``len(session_names)``.
            session_names: Session identifiers aligned with ``reports``. Used both for chronological ordering and for
                recording the pre/post split in the summary.
            configuration: Longitudinal-reward-shift configuration; uses defaults if None.

        Returns:
            A LongitudinalRewardShiftReport instance.

        Raises:
            ValueError: When inputs are malformed or no reward shift is detectable.
        """
        resolved_configuration = configuration if configuration is not None else LongitudinalRewardShiftConfiguration()
        if len(reports) != len(session_names):
            message = (
                f"reports and session_names must have the same length; got {len(reports)} reports vs. "
                f"{len(session_names)} session names."
            )
            raise ValueError(message)
        if len(reports) < 2:
            message = f"Need at least two sessions to detect a reward shift; got {len(reports)}."
            raise ValueError(message)

        # Sort chronologically by session name (Sollertia session names are timestamp-prefixed).
        sorted_indices = sorted(range(len(reports)), key=lambda index: session_names[index])
        sorted_reports = [reports[index] for index in sorted_indices]
        sorted_names = [session_names[index] for index in sorted_indices]

        # noinspection PyTypeChecker
        reward_positions: NDArray[np.float64] = np.asarray(
            [report.summary.reward_position_cm for report in sorted_reports], dtype=np.float64
        )
        initial_reward = float(reward_positions[0])
        # noinspection PyTypeChecker
        differs_mask: NDArray[np.bool_] = (
            np.abs(reward_positions - initial_reward) > resolved_configuration.reward_position_tolerance_cm
        )
        if not bool(np.any(differs_mask)):
            message = (
                f"No reward shift detected for animal '{animal}': all {len(sorted_reports)} sessions have reward "
                f"position within {resolved_configuration.reward_position_tolerance_cm} cm of "
                f"{initial_reward:.1f} cm. The longitudinal-reward-shift analysis requires the reward midpoint to "
                f"move between two distinct positions across the session sequence."
            )
            raise ValueError(message)

        boundary_index = int(np.argmax(differs_mask))
        pre_indices = list(range(boundary_index))
        post_indices = list(range(boundary_index, len(sorted_reports)))
        if len(pre_indices) < resolved_configuration.minimum_sessions_per_block:
            message = (
                f"Pre-shift block for animal '{animal}' has only {len(pre_indices)} sessions; configuration "
                f"requires at least {resolved_configuration.minimum_sessions_per_block}."
            )
            raise ValueError(message)
        if len(post_indices) < resolved_configuration.minimum_sessions_per_block:
            message = (
                f"Post-shift block for animal '{animal}' has only {len(post_indices)} sessions; configuration "
                f"requires at least {resolved_configuration.minimum_sessions_per_block}."
            )
            raise ValueError(message)

        # Verify cell counts match across sessions; multi-day registration guarantees this when the same cohort is
        # tracked, but a misaligned report set should fail loudly rather than silently truncate.
        cell_count = int(sorted_reports[0].summary.cell_count)
        for index, report in enumerate(sorted_reports):
            if int(report.summary.cell_count) != cell_count:
                message = (
                    f"Session '{sorted_names[index]}' has {report.summary.cell_count} cells but the first session "
                    f"has {cell_count}. Cross-session reward-shift analysis requires multi-day registration so cell "
                    f"IDs are aligned; the input reports must come from the multi-day cindra pipeline."
                )
                raise ValueError(message)

        # Aggregate rate maps and significance per block.
        pre_reports = [sorted_reports[index] for index in pre_indices]
        post_reports = [sorted_reports[index] for index in post_indices]
        pre_rate_maps, pre_sig_fraction = _aggregate_block(reports=pre_reports, cell_count=cell_count)
        post_rate_maps, post_sig_fraction = _aggregate_block(reports=post_reports, cell_count=cell_count)

        # noinspection PyTypeChecker
        is_significant_pre: NDArray[np.bool_] = pre_sig_fraction >= resolved_configuration.minimum_significance_fraction
        # noinspection PyTypeChecker
        is_significant_post: NDArray[np.bool_] = (
            post_sig_fraction >= resolved_configuration.minimum_significance_fraction
        )

        # Pull per-block geometry from the first session of each block (track length and bin size are stable within
        # a block by assumption).
        pre_summary = sorted_reports[pre_indices[0]].summary
        post_summary = sorted_reports[post_indices[0]].summary

        classification = compute_cross_block_classification(
            rate_maps_a=pre_rate_maps,
            rate_maps_b=post_rate_maps,
            is_significant_a=is_significant_pre,
            is_significant_b=is_significant_post,
            track_length_a_cm=float(pre_summary.track_length_cm),
            track_length_b_cm=float(post_summary.track_length_cm),
            reward_position_a_cm=float(pre_summary.reward_position_cm),
            reward_position_b_cm=float(post_summary.reward_position_cm),
            bin_size_a_cm=float(pre_summary.bin_size_cm),
            bin_size_b_cm=float(post_summary.bin_size_cm),
            configuration=resolved_configuration.to_reward_relative_configuration(),
        )

        table = _build_longitudinal_table(
            cell_count=cell_count,
            classification=classification,
            pre_significance_fraction=pre_sig_fraction,
            post_significance_fraction=post_sig_fraction,
        )
        counts = aggregate_cross_block_counts(classification=classification)
        summary = LongitudinalRewardShiftSummary(
            animal=animal,
            pre_shift_session_names=[sorted_names[index] for index in pre_indices],
            post_shift_session_names=[sorted_names[index] for index in post_indices],
            track_length_pre_cm=float(pre_summary.track_length_cm),
            track_length_post_cm=float(post_summary.track_length_cm),
            reward_position_pre_cm=float(pre_summary.reward_position_cm),
            reward_position_post_cm=float(post_summary.reward_position_cm),
            reward_shift_cm=float(post_summary.reward_position_cm - pre_summary.reward_position_cm),
            cell_count=cell_count,
            tested_count=int(counts["tested"]),
            reward_anchored_count=int(counts["reward_anchored"]),
            track_anchored_count=int(counts["track_anchored"]),
            both_anchored_count=int(counts["both_anchored"]),
            neither_anchored_count=int(counts["neither_anchored"]),
            configuration=resolved_configuration,
        )
        return cls(table=table, summary=summary)

    @classmethod
    def load(cls, animal_path: Path) -> LongitudinalRewardShiftReport:
        """Loads a previously saved per-animal report from the animal directory."""
        summary_path = animal_path.joinpath(_LONGITUDINAL_SUMMARY_FILENAME)
        table_path = animal_path.joinpath(_LONGITUDINAL_TABLE_FILENAME)
        summary = LongitudinalRewardShiftSummary.from_yaml(file_path=summary_path)
        table = pl.read_ipc(source=table_path, memory_map=True)
        return cls(table=table, summary=summary)

    def save(self, animal_path: Path) -> None:
        """Persists the report to two artifacts inside the animal directory."""
        summary_path = animal_path.joinpath(_LONGITUDINAL_SUMMARY_FILENAME)
        table_path = animal_path.joinpath(_LONGITUDINAL_TABLE_FILENAME)
        self.summary.to_yaml(file_path=summary_path)
        self.table.write_ipc(file=table_path)

    def summarize(self) -> str:
        """Returns a multi-line human-readable summary of the longitudinal analysis."""
        summary = self.summary
        cell_count = summary.cell_count
        tested_pct = 100.0 * summary.tested_count / cell_count if cell_count > 0 else 0.0
        reward_pct = 100.0 * summary.reward_anchored_count / cell_count if cell_count > 0 else 0.0
        track_pct = 100.0 * summary.track_anchored_count / cell_count if cell_count > 0 else 0.0
        both_pct = 100.0 * summary.both_anchored_count / cell_count if cell_count > 0 else 0.0
        neither_pct = 100.0 * summary.neither_anchored_count / cell_count if cell_count > 0 else 0.0
        return "\n".join(
            [
                f"Longitudinal reward-shift analysis: animal {summary.animal}",
                "=" * 60,
                f"Pre-shift sessions:  {len(summary.pre_shift_session_names)} "
                f"(track {summary.track_length_pre_cm:.1f} cm, reward {summary.reward_position_pre_cm:.1f} cm)",
                f"Post-shift sessions: {len(summary.post_shift_session_names)} "
                f"(track {summary.track_length_post_cm:.1f} cm, reward {summary.reward_position_post_cm:.1f} cm)",
                f"Reward shift: {summary.reward_shift_cm:+.1f} cm",
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
                f"min_sig_fraction={summary.configuration.minimum_significance_fraction:.2f}, "
                f"p<{summary.configuration.significance_threshold:.2f}",
            ]
        )


def _aggregate_block(
    reports: Sequence[CellAnalysisReport],
    cell_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Returns the per-cell mean rate map and per-cell significance fraction across a block of sessions.

    Notes:
        Stacks each session's per-cell rate map (padded to the bin count of the first session) and averages with
        ``np.nanmean`` so cells absent from a session (NaN row from the upstream multi-day pipeline) do not dilute
        the average. The significance fraction is the proportion of sessions in which each cell was flagged
        ``IS_SPATIALLY_SIGNIFICANT``.
    """
    bin_count = int(reports[0].summary.bin_count)
    # noinspection PyTypeChecker
    stacked_maps: NDArray[np.float32] = np.full((len(reports), cell_count, bin_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    significance_counts: NDArray[np.int32] = np.zeros(cell_count, dtype=np.int32)
    for index, report in enumerate(reports):
        rate_maps = stack_rate_maps_from_table(table=report.table, target_length=bin_count)
        stacked_maps[index] = rate_maps
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = report.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        significance_counts += is_significant.astype(np.int32)

    # NaN-aware mean across the session axis; cells that are NaN in every session collapse to NaN here, which the
    # downstream cross-block kernel handles by excluding them from the test.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            # noinspection PyTypeChecker
            mean_rate_maps: NDArray[np.float32] = np.nanmean(stacked_maps, axis=0).astype(np.float32)

    # noinspection PyTypeChecker
    significance_fraction: NDArray[np.float32] = significance_counts.astype(np.float32) / float(len(reports))
    return mean_rate_maps, significance_fraction


def _build_longitudinal_table(
    *,
    cell_count: int,
    classification: CrossBlockClassification,
    pre_significance_fraction: NDArray[np.float32],
    post_significance_fraction: NDArray[np.float32],
) -> pl.DataFrame:
    """Builds the per-cell longitudinal table by composing the cross-block table with the session-coverage columns."""
    base_table = build_cross_block_table(cell_count=cell_count, classification=classification)
    return base_table.with_columns(
        [
            pl.Series(
                name=LongitudinalRewardShiftColumn.SIGNIFICANT_IN_PRE_FRACTION.value,
                values=pre_significance_fraction,
                dtype=pl.Float32,
            ),
            pl.Series(
                name=LongitudinalRewardShiftColumn.SIGNIFICANT_IN_POST_FRACTION.value,
                values=post_significance_fraction,
                dtype=pl.Float32,
            ),
        ]
    )
