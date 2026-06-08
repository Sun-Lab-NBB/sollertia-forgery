"""Per-session tuning-report container that consolidates place-field and reward-cell detection.

The `TuningReport` pair (per-cell long-format feather + summary YAML) is the analysis counterpart to
`BleachingReport`; the report owns persistence and per-trial-type population-mask resolution so loading
a saved report is sufficient to reproduce every figure without rerunning detection. Plot regeneration lives in
`.plotting`. SCE-related analyses live alongside in `..sce` and produce their own per-session
report. Methodological references for the assembly pipeline are attached to `compute_tuning_report`.

The persisted ``tuning_cells.feather`` is **long-format**: one row per ``(cell_id, trial_type)`` pair with a
``trial_type: Utf8`` column. Cell IDs are stable across trial types because the upstream multi-day cindra
pipeline registers cells once per session, so cross-trial-type queries collapse to ``polars`` filters or
joins on ``cell_id``. The summary YAML carries one `TuningTrialSummary` entry per trial type alongside
session-level fields shared across trial types (sub-pipeline configurations, sampling rate, cell count).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
import warnings
from dataclasses import field, dataclass

from numba import set_num_threads
import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import YamlConfig

from ...forging import FluorescenceColumn
from .utilities import per_cell_pearson_safe, assemble_run_session_data
from ...shared_assets import DatasetFiles, TrialGeometry, delay_terminal
from ..shared_utilities import resolve_session_selection
from .place_tuning_protocol import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_tuning_protocol import RewardCellResults, RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ...shared_assets import DatasetData, DatasetSession


_PLACE_FIELD_BIN_SIZE_CM: float = 5.0
"""Default spatial bin size in centimeters used by ``PlaceFieldDetector`` and persisted in the per-trial
fluorescence matrix. Matches the ``PlaceFieldDetector`` constructor default."""


class TuningColumn(StrEnum):
    """Defines every column written to the per-session ``tuning_cells.feather`` long-format per-cell table."""

    CELL_ID = "cell_id"
    """Contiguous integer cell identifier; stable across trial types within a session."""
    TRIAL_TYPE = "trial_type"
    """Trial type the row's per-cell flags and rate maps were computed against; matches an entry in the
    session's ``trial_geometry.yaml``."""
    IS_PLACE = "is_place"
    """True for cells with at least one detected place field."""
    IS_SPATIALLY_SIGNIFICANT = "is_spatially_significant"
    """True for cells whose Skaggs spatial information is significant under shuffle testing."""
    IS_REWARD_PROXIMAL = "is_reward_proximal"
    """True for cells whose circular center of mass falls within the reward zone. Alias of ``IS_ZONE``
    preserved for backwards compatibility; bit-identical to it."""
    IS_APPROACH = "is_approach"
    """True for cells whose center of mass falls in the approach (anticipatory) band immediately upstream of
    the reward zone (default 40 cm)."""
    IS_ZONE = "is_zone"
    """True for cells whose center of mass falls inside the reward zone (configured by ``reward_zone_width``)."""
    IS_DEPARTURE = "is_departure"
    """True for cells whose center of mass falls in the post-reward (departure) band immediately downstream
    of the reward zone (default 40 cm)."""
    IS_REWARD_CELL = "is_reward_cell"
    """True for cells that are both spatially significant and reward-proximal (zone-band)."""
    IS_POSITION_GLM_SIGNIFICANT = "is_position_glm_significant"
    """True for cells whose 5-fold CV ΔR² of position over speed+acceleration exceeds the trial-label
    permutation null at the configured ``glm_significance_threshold``."""
    PF_START_CM = "pf_start_cm"
    """Per-field place-field start position in centimeters."""
    PF_END_CM = "pf_end_cm"
    """Per-field place-field end position in centimeters."""
    PF_CENTER_CM = "pf_center_cm"
    """Per-field intensity-weighted place-field centroid in centimeters."""
    PF_MEAN_INTENSITY = "pf_mean_intensity"
    """Per-field mean fluorescence intensity within each place field."""
    PF_MAX_INTENSITY = "pf_max_intensity"
    """Per-field peak fluorescence intensity within each place field."""
    PF_WIDTH_CM = "pf_width_cm"
    """Per-field spatial width in centimeters."""
    BINNED_FLUORESCENCE_PER_TRIAL = "binned_fluorescence_per_trial"
    """Per-cell trial x bin fluorescence matrix; null for cells lacking any per-trial binning."""
    RATE_MAP = "rate_map"
    """Pooled, speed-filtered, smoothed cell tuning curve as a length-bin_count vector."""
    CENTER_OF_MASS_CM = "center_of_mass_cm"
    """Circular center-of-mass position in centimeters; -1 for cells with invalid COM."""
    SPATIAL_INFORMATION_BITS = "spatial_information_bits"
    """Skaggs spatial information content in bits per event."""
    SPATIAL_INFORMATION_Z = "spatial_information_z"
    """Z-scored Skaggs spatial information against the same circular-shift null used for
    ``SPATIAL_P_VALUE``."""
    SPATIAL_P_VALUE = "spatial_p_value"
    """Shuffle-derived p-value for the spatial information statistic."""
    SPATIAL_FDR_SURVIVED = "spatial_fdr_survived"
    """True for cells whose ``SPATIAL_P_VALUE`` survives Benjamini-Hochberg FDR correction at the configured
    ``fdr_q`` level."""
    SPATIAL_SPLIT_HALF_R = "spatial_split_half_r"
    """Per-cell even/odd-lap Pearson r used by the reward-cell pipeline as the lap-reliability gate. With
    matching configuration (the default) bit-identical to ``STABILITY_EVEN_ODD``."""
    REWARD_RELATIVITY_SCORE = "reward_relativity_score"
    """Per-cell ``zone_peak / overall_peak`` of the smoothed rate map; in [0, 1]."""
    CV_POSITION_PARTIAL_R2 = "cv_position_partial_r2"
    """Per-cell 5-fold CV ΔR² of position over speed+acceleration in the pre-reward window."""
    POSITION_GLM_P_VALUE = "position_glm_p_value"
    """Per-cell trial-label permutation p-value for the ``CV_POSITION_PARTIAL_R2`` statistic."""
    STABILITY_EVEN_ODD = "stability_even_odd"
    """Pearson r between the even-trial and odd-trial mean rate maps."""
    STABILITY_SPLIT_HALF = "stability_split_half"
    """Pearson r between the first-half and second-half mean rate maps."""
    IS_RELIABLE = "is_reliable"
    """True for cells with at least one place field that passes the lap-coverage criterion."""
    IS_STABLE = "is_stable"
    """True for cells whose split-half stability r exceeds the 95th percentile of a per-cell shuffled null."""
    STABILITY_P_VALUE = "stability_p_value"
    """Per-cell p-value for the Stability shuffle."""
    IS_PEAK_SIGNIFICANT = "is_peak_significant"
    """True for cells whose observed pooled-rate-map peak exceeds the 99th percentile of the shuffled
    per-cell peak distribution."""
    PEAK_P_VALUE = "peak_p_value"
    """Per-cell p-value for the Peak shuffle."""
    IS_STRICT_PLACE = "is_strict_place"
    """True for cells that pass all three of ``IS_PLACE``, ``IS_STABLE``, and ``IS_PEAK_SIGNIFICANT``
    simultaneously."""


@dataclass(frozen=True, slots=True)
class TuningConfiguration:
    """Wraps the place-field and reward-cell sub-configurations together so a single object drives evaluate()."""

    place: PlaceFieldDetectionConfiguration
    """Place-field detection parameters."""
    reward: RewardCellConfiguration
    """Reward-cell detection parameters."""

    @classmethod
    def default(cls) -> TuningConfiguration:
        """Returns a TuningConfiguration whose two sub-configurations all use library defaults."""
        return cls(
            place=PlaceFieldDetectionConfiguration(),
            reward=RewardCellConfiguration(),
        )


@dataclass
class TuningTrialSummary(YamlConfig):
    """Per-trial-type entry inside the session-level `TuningSummary`.

    Carries fields that vary per trial type: track / reward geometry resolved from the trial geometry data
    file, the rate-map bin axis, the four-component mixture-model fit, and pre-aggregated per-cell counts
    that ``summarize()`` reports without rehydrating the cells feather.
    """

    track_length_cm: float
    """Track length in centimeters resolved from ``trial_geometry.yaml``."""
    reward_position_cm: float
    """Reward position in centimeters (midpoint of the stimulus trigger zone)."""
    bin_size_cm: float
    """Spatial bin size in centimeters used for rate maps and per-cell place-field detection."""
    bin_count: int
    """Number of spatial bins along the track."""

    place_cell_count: int
    """Number of cells with at least one detected place field."""
    spatially_significant_count: int
    """Number of cells whose Skaggs spatial information passes the shuffle threshold."""
    reward_cell_count: int
    """Number of cells that are both spatially significant and reward-proximal."""
    reward_predictive_count: int
    """Number of cells that are reward-associated and pass the position-vs-speed GLM."""
    reliable_count: int
    """Number of cells whose detected fields pass the lap-coverage criterion."""
    stable_count: int
    """Number of cells whose split-half stability r exceeds the 95th percentile of a per-cell shuffled null."""
    peak_significant_count: int
    """Number of cells whose pooled-rate-map peak exceeds the 99th percentile of a per-cell shuffled null."""
    strict_place_cell_count: int
    """Number of cells that simultaneously pass IS_PLACE, IS_STABLE, and IS_PEAK_SIGNIFICANT."""
    place_only_count: int
    """Number of cells flagged ``IS_PLACE`` but not ``IS_REWARD_CELL``."""
    strict_place_only_count: int
    """Number of cells passing ``IS_PLACE & IS_STABLE & IS_PEAK_SIGNIFICANT`` but not ``IS_REWARD_CELL``."""

    mixture_weight: float
    """Reward-component weight from the four-component mixture model fit to the spatially significant COMs."""
    gaussian_mean_cm: float
    """Reward-component Gaussian mean in centimeters."""
    gaussian_std_cm: float
    """Reward-component Gaussian standard deviation in centimeters."""
    track_start_weight: float
    """Track-start landmark Gaussian weight (centered at 0 cm)."""
    track_end_weight: float
    """Track-end landmark Gaussian weight (centered at ``track_length_cm``)."""
    track_start_std_cm: float
    """Track-start landmark Gaussian standard deviation in centimeters."""
    track_end_std_cm: float
    """Track-end landmark Gaussian standard deviation in centimeters."""


@dataclass
class TuningSummary(YamlConfig):
    """Per-session YAML companion to ``tuning_cells.feather``.

    Carries session-level fields shared across trial types (sub-pipeline configurations, sampling rate, total
    cell count) plus a `TuningTrialSummary` entry for every trial type the session was evaluated
    against. Cell IDs are stable across the entries because the upstream multi-day pipeline registers cells
    once per session, so any cross-trial-type tabulation collapses to a join on ``cell_id`` against the
    long-format cells feather.
    """

    place_configuration: PlaceFieldDetectionConfiguration
    """Place-field detection configuration that produced the place-field columns."""
    reward_configuration: RewardCellConfiguration
    """Reward-cell detection configuration that produced the reward and slowing columns."""

    sampling_rate_hz: float
    """Acquisition sampling rate in Hz."""
    cell_count: int
    """Total number of cells in the session; constant across every trial-type entry."""

    trial_types: list[str] = field(default_factory=list)
    """Trial-type names in chronological / configuration order; keys into ``trial_type_summaries``."""
    trial_type_summaries: dict[str, TuningTrialSummary] = field(default_factory=dict)
    """Per-trial-type summary entries keyed by trial-type name."""


@dataclass(frozen=True, slots=True)
class TuningReport:
    """Top-level per-session container for the place-field + reward-cell tuning analysis.

    Holds the long-format per-cell tuning table (``cells``) and the YAML summary (``summary``). Persistence
    and per-trial-type population-mask resolution stay on the report; plot regeneration lives in
    `.plotting`.
    """

    cells: pl.DataFrame
    """Long-format per-cell wide table; one row per ``(cell_id, trial_type)`` pair. Schema enumerated by
    `TuningColumn`."""
    summary: TuningSummary
    """YAML wrapper holding the configurations, mixture-model fit, and per-trial-type session-level scalars."""

    @classmethod
    def load(cls, session: DatasetSession) -> TuningReport:
        """Loads a previously saved report from the session directory."""
        summary: TuningSummary = TuningSummary.from_yaml(file_path=session.tuning_summary_path)
        cells = pl.read_ipc(source=session.tuning_cells_path, memory_map=True)
        return cls(cells=cells, summary=summary)

    def save(self, session: DatasetSession) -> None:
        """Persists the report to the two per-session artifacts inside the session directory."""
        self.summary.to_yaml(file_path=session.tuning_summary_path)
        self.cells.write_ipc(file=session.tuning_cells_path)

    def trial_summary(self, trial_type: str) -> TuningTrialSummary:
        """Returns the `TuningTrialSummary` entry for ``trial_type``.

        Raises:
            KeyError: When ``trial_type`` is not present in ``summary.trial_type_summaries``.
        """
        return self.summary.trial_type_summaries[trial_type]

    def trial_cells(self, trial_type: str) -> pl.DataFrame:
        """Returns the per-cell rows for ``trial_type`` with the ``trial_type`` column dropped.

        Notes:
            The returned slice is a view of the long-format feather restricted to the requested trial type
            and ordered by ``cell_id``. Downstream consumers can treat it as the wide single-trial-type table
            the previous report layout exposed.
        """
        return (
            self.cells.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == trial_type)
            .drop(TuningColumn.TRIAL_TYPE.value)
            .sort(TuningColumn.CELL_ID.value)
        )

    def place_mask(
        self,
        *,
        trial_type: str,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
    ) -> NDArray[np.bool_]:
        """Returns the per-cell boolean mask for cells in ``trial_type`` passing every requested place-cell
        criterion simultaneously.

        Notes:
            Defaults to the strict triple-AND of place / stable / peak-significant. When every kwarg is False
            the method returns an all-True mask, treating "no criteria" as "no filter". Skaggs spatial
            significance is intentionally not exposed here; use `resolve_population_masks` instead when
            you need place / reward populations that respect mutual exclusion.

        Args:
            trial_type: Trial type to extract; must match an entry in ``summary.trial_type_summaries``.
            require_place: Require ``IS_PLACE`` (place-field morphology + lap coverage).
            require_stable: Require ``IS_STABLE`` (split-half stability shuffle).
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (per-cell peak shuffle).

        Returns:
            Per-cell boolean mask with length ``cell_count`` aligned with the persisted cell-id ordering.
        """
        cells = self.trial_cells(trial_type=trial_type)
        # noinspection PyTypeChecker
        mask: NDArray[np.bool_] = np.ones(cells.height, dtype=np.bool_)
        if require_place:
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = cells[TuningColumn.IS_PLACE.value].to_numpy()
            mask = mask & place_flag
        if require_stable:
            # noinspection PyTypeChecker
            stable_flag: NDArray[np.bool_] = cells[TuningColumn.IS_STABLE.value].to_numpy()
            mask = mask & stable_flag
        if require_peak_significant:
            # noinspection PyTypeChecker
            peak_flag: NDArray[np.bool_] = cells[TuningColumn.IS_PEAK_SIGNIFICANT.value].to_numpy()
            mask = mask & peak_flag
        return mask

    def resolve_population_masks(
        self,
        *,
        trial_type: str,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
        mutually_exclusive: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Returns ``(place_mask, reward_mask)`` for ``trial_type`` honoring the mutual-exclusion option.

        Notes:
            The reward mask is always ``IS_REWARD_CELL`` (i.e., spatially significant cells whose COM lies in
            the zone band). When ``mutually_exclusive=True``, cells flagged as both place and reward are
            subtracted from the place mask so the two populations are disjoint at the presentation layer; when
            ``False``, the place mask is the unfiltered place population and a cell may appear in both. The
            persisted feather always keeps both flags independently — mutual exclusion is purely a
            presentation-layer convention so the distinction is reversible without re-running detection.

        Args:
            trial_type: Trial type to extract.
            require_place: Require ``IS_PLACE`` for the place population.
            require_stable: Require ``IS_STABLE`` for the place population.
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` for the place population.
            mutually_exclusive: When True, subtract ``IS_REWARD_CELL`` cells from the place mask. Default True
                for visualization clarity.

        Returns:
            A tuple of (place_mask, reward_mask) per-cell boolean arrays each with length ``cell_count``.
        """
        place_population = self.place_mask(
            trial_type=trial_type,
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
        )
        cells = self.trial_cells(trial_type=trial_type)
        # noinspection PyTypeChecker
        reward_mask: NDArray[np.bool_] = cells[TuningColumn.IS_REWARD_CELL.value].to_numpy()
        place_mask = place_population & ~reward_mask if mutually_exclusive else place_population
        return place_mask, reward_mask

    def summarize(self, *, mutually_exclusive: bool = True) -> str:
        """Returns a multi-line human-readable summary of the per-cell statistics for every trial type.

        Notes:
            With ``mutually_exclusive=True`` (default), the place-cell count subtracts cells also flagged as
            ``IS_REWARD_CELL`` and is reported as "place-only"; the reward count is unchanged. With
            ``mutually_exclusive=False``, the raw counts persisted in the summary YAML are reported instead
            and a cell may contribute to both totals. Each trial type is summarized in its own block.

        Args:
            mutually_exclusive: When True (default), report ``IS_PLACE & ~IS_REWARD_CELL`` for the place
                count.
        """
        summary = self.summary
        cell_count = summary.cell_count
        lines: list[str] = [
            "Tuning report",
            "=============",
            f"Cells: {cell_count}",
            f"Trial types: {', '.join(summary.trial_types) if summary.trial_types else '<none>'}",
            f"Sampling rate: {summary.sampling_rate_hz:.2f} Hz",
        ]
        for trial_type in summary.trial_types:
            entry = summary.trial_type_summaries[trial_type]
            lines.extend(
                self._format_trial_type_block(
                    trial_type=trial_type,
                    entry=entry,
                    mutually_exclusive=mutually_exclusive,
                )
            )
        return "\n".join(lines)

    def _format_trial_type_block(
        self,
        *,
        trial_type: str,
        entry: TuningTrialSummary,
        mutually_exclusive: bool,
    ) -> list[str]:
        """Formats a per-trial-type detail block for ``summarize``."""
        cell_count = self.summary.cell_count
        if mutually_exclusive and cell_count > 0:
            cells = self.trial_cells(trial_type=trial_type)
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = cells[TuningColumn.IS_PLACE.value].to_numpy()
            # noinspection PyTypeChecker
            reward_flag: NDArray[np.bool_] = cells[TuningColumn.IS_REWARD_CELL.value].to_numpy()
            place_count = int(np.sum(place_flag & ~reward_flag))
            place_label = "Place-only:"
        else:
            place_count = entry.place_cell_count
            place_label = "Place cells:"

        spatially_pct = 100.0 * entry.spatially_significant_count / cell_count if cell_count > 0 else 0.0
        place_pct = 100.0 * place_count / cell_count if cell_count > 0 else 0.0
        reward_pct = 100.0 * entry.reward_cell_count / cell_count if cell_count > 0 else 0.0
        predictive_pct = 100.0 * entry.reward_predictive_count / cell_count if cell_count > 0 else 0.0
        reliable_pct = 100.0 * entry.reliable_count / cell_count if cell_count > 0 else 0.0
        stable_pct = 100.0 * entry.stable_count / cell_count if cell_count > 0 else 0.0
        peak_pct = 100.0 * entry.peak_significant_count / cell_count if cell_count > 0 else 0.0
        strict_pct = 100.0 * entry.strict_place_cell_count / cell_count if cell_count > 0 else 0.0

        return [
            "",
            f"Trial type: {trial_type}",
            "-" * (len(trial_type) + len("Trial type: ")),
            f"  Track length:      {entry.track_length_cm:.1f} cm",
            f"  Reward position:   {entry.reward_position_cm:.1f} cm",
            f"  Bin size:          {entry.bin_size_cm:.1f} cm ({entry.bin_count} bins)",
            f"  {place_label:<22} {place_count} ({place_pct:.1f}%)",
            f"  Reliable (lap cov.):   {entry.reliable_count} ({reliable_pct:.1f}%)",
            f"  Stable (split-half):   {entry.stable_count} ({stable_pct:.1f}%)",
            f"  Peak-significant:      {entry.peak_significant_count} ({peak_pct:.1f}%)",
            f"  Strict place cells:    {entry.strict_place_cell_count} ({strict_pct:.1f}%)",
            f"  Spatially significant: {entry.spatially_significant_count} ({spatially_pct:.1f}%)",
            f"  Reward cells:          {entry.reward_cell_count} ({reward_pct:.1f}%)",
            f"  Reward-predictive:     {entry.reward_predictive_count} ({predictive_pct:.1f}%)",
            f"  Reward mixture: weight {entry.mixture_weight:.3f}, "
            f"center {entry.gaussian_mean_cm:.1f} cm (SD {entry.gaussian_std_cm:.1f} cm)",
            f"  Track-start wt: {entry.track_start_weight:.3f} (SD {entry.track_start_std_cm:.1f} cm)",
            f"  Track-end wt:   {entry.track_end_weight:.3f} (SD {entry.track_end_std_cm:.1f} cm)",
        ]


def run_tuning_analysis(
    dataset: DatasetData,
    *,
    animal: str | tuple[str, ...] | None = None,
    session: str | tuple[str, ...] | None = None,
    trial_types: tuple[str, ...] | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    workers: int = -1,
    aggregate_progress: bool = True,
    configuration: TuningConfiguration | None = None,
) -> tuple[TuningReport, ...]:
    """Evaluates the tuning pipeline for every session matching the (animal, session) filter and persists each
    report.

    Notes:
        Sole orchestrator for assembling `TuningReport` artifacts. Per-session compute and methodological
        references live on `compute_tuning_report`; this function resolves the in-scope DatasetSession set,
        configures the per-cell Numba thread pool to the resolved CPU budget, and walks the sessions
        sequentially because the per-cell kernels already saturate the available threads on a single session.
        Each report is written to ``<session>/tuning_summary.yaml`` and ``<session>/tuning_cells.feather`` and
        returned to the caller for downstream plotting.

        ``animal`` and ``session`` accept None (no filter), a single identifier, or a tuple of identifiers and
        compose as a logical AND through `..shared_utilities.resolve_session_selection`. Feedback is
        non-optional: ``aggregate_progress`` only chooses the form of feedback (a single session-level
        progress bar versus the per-session per-stage echoes emitted by `compute_tuning_report`).

    Args:
        dataset: The DatasetData whose sessions are evaluated.
        animal: Animal-identifier filter; None evaluates every animal in the dataset.
        session: Session-identifier filter; None evaluates every session within the resolved animal scope.
        trial_types: Optional explicit tuple of trial types to evaluate per session; default evaluates every
            entry in each session's ``trial_geometry.yaml``.
        fluorescence_column: Fluorescence column to use as the analysis input across every session.
        workers: The total number of CPU cores to use. A non-positive value requests every available core
            minus the system reserve. The full budget is handed to the per-cell Numba thread pool because
            sessions are processed sequentially.
        aggregate_progress: When True, render a single session-level progress bar and silence per-stage
            compute echoes so the bar is the only visual signal. When False, skip the bar and let
            `compute_tuning_report` emit per-trial-type and per-detector echoes for each session as it is
            processed.
        configuration: Wrapper holding the two sub-pipeline configurations shared across every session. Uses
            defaults if None.

    Returns:
        A tuple of TuningReports in the same order as the resolved DatasetSession set, with each report's two
        artifacts persisted under the corresponding session directory.
    """
    sessions = resolve_session_selection(dataset=dataset, animal=animal, session=session)

    total_workers = resolve_worker_count(requested_workers=workers)
    set_num_threads(total_workers)

    animal_count = len({dataset_session.animal for dataset_session in sessions})
    console.echo(
        message=(
            f"Running tuning analysis on dataset {dataset.name!r} for "
            f"{animal_count} animal{'s' if animal_count != 1 else ''} "
            f"({len(sessions)} session{'s' if len(sessions) != 1 else ''} total): "
            f"sequential session loop x {total_workers} Numba "
            f"thread{'s' if total_workers != 1 else ''} per session."
        ),
        level=LogLevel.INFO,
    )
    delay_terminal()

    reports: list[TuningReport] = []
    if aggregate_progress:
        with console.progress(
            total=len(sessions), description="Running tuning analysis", unit="session"
        ) as progress_bar:
            for dataset_session in sessions:
                report = compute_tuning_report(
                    session_path=dataset_session.session_path,
                    trial_types=trial_types,
                    fluorescence_column=fluorescence_column,
                    configuration=configuration,
                    verbose=False,
                )
                report.save(session=dataset_session)
                reports.append(report)
                progress_bar.update()
    else:
        for dataset_session in sessions:
            report = compute_tuning_report(
                session_path=dataset_session.session_path,
                trial_types=trial_types,
                fluorescence_column=fluorescence_column,
                configuration=configuration,
                verbose=True,
            )
            report.save(session=dataset_session)
            reports.append(report)

    console.echo(
        message=(
            f"Tuning analysis complete. Persisted {len(reports)} "
            f"report{'s' if len(reports) != 1 else ''} under {dataset.dataset_data_path.parent}."
        ),
        level=LogLevel.SUCCESS,
    )
    delay_terminal()
    return tuple(reports)


def compute_tuning_report(
    session_path: Path,
    *,
    trial_types: tuple[str, ...] | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    configuration: TuningConfiguration | None = None,
    verbose: bool = True,
) -> TuningReport:
    """Computes the per-session tuning report by running the place-field and reward-cell detectors per trial
    type.

    Notes:
        Per-session algorithmic entry-point for the tuning pipeline; consolidates every place- and reward-
        cell methodology that gates a column in the persisted feather. For each requested trial type, loads
        a single `RunSessionData` and shares it across `PlaceFieldDetector` and
        `RewardCellDetector` so the two flag sets operate on identical speed-filtered samples and
        bit-identical rate maps. Place-field detection uses the thresholding-plus-connected-component
        pipeline of Dombeck et al. (2010) followed by the lap-coverage gate of Climer et al. (2025). The
        ``IS_STABLE`` and ``IS_PEAK_SIGNIFICANT`` flags follow the multi-criterion framework of Climer &
        Dombeck (2021): per-cell circular-shift nulls computed inside `PlaceFieldDetector` yield the
        Stability (95th percentile of split-half r) and Peak (99th percentile of pooled-rate-map peak)
        classifiers. The reward-cell pipeline contributes ``IS_SPATIALLY_SIGNIFICANT`` via Skaggs spatial
        information (Skaggs et al. 1996) z-scored against the circular-shift null in the variant of Souza &
        Tort (2018), gated by Benjamini-Hochberg FDR correction and an even/odd-lap split-half r reliability
        check (Krishnan & Sheffield 2024). Reward-relative classification follows the Gauthier & Tank (2018)
        mixture-model framework with the Issa, Radvansky, Xuan & Dombeck (2024) approach / zone / departure
        decomposition; the per-cell ``IS_POSITION_GLM_SIGNIFICANT`` flag implements the Sosa, Plitt &
        Giocomo (2025) and Hardcastle et al. (2017) cross-validated partial-variance test of position over
        speed and acceleration with a trial-label permutation null. All numba kernels and helpers in
        `.place_tuning_protocol` and `.reward_tuning_protocol` inherit these references through
        this accessor.

        The cells produced for each trial type are concatenated into a single long-format
        `polars.DataFrame` keyed by ``(cell_id, trial_type)``. Cell IDs are stable across trial types
        (and across sessions) because the upstream multi-day cindra pipeline registers cells once per
        session, so cross-trial-type and cross-session aggregations downstream collapse to ``polars`` filters
        / joins on ``cell_id``. For paired-frame statistical tests (e.g. peak-shift comparisons across trial
        types or sessions), `.utilities.random_remapping_peak_shift_p_values` provides a reusable
        cell-ID-shuffle helper.

    References:
        Place-field morphology, in-/out-of-field ratio, peak intensity, and lap-coverage criterion:
            Dombeck, Harvey, Tian, Looger & Tank (2010). Functional imaging of hippocampal place cells at
            cellular resolution during virtual navigation. Nat Neurosci. https://doi.org/10.1038/nn.2648
            Climer, Davoudi, Oh & Dombeck (2025). Hippocampal representations drift in stable multisensory
            environments. Nature. https://doi.org/10.1038/s41586-025-09245-y
        Multi-criterion place-cell classification (Stability and Peak shuffles):
            Climer & Dombeck (2021). Choice of method of place cell classification determines the population
            of cells identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835
        Skaggs spatial information and the z-scored sensitivity variant against the circular-shift null:
            Skaggs, McNaughton, Wilson & Barnes (1996). Theta phase precession in hippocampal neuronal
            populations and the compression of temporal sequences. Hippocampus.
            https://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K
            Souza, Pavão, Belchior & Tort (2018). On information metrics for spatial coding. Neuroscience.
            https://doi.org/10.1016/j.neuroscience.2018.01.066
        Lap-reliability split-half r as the within-session reliability gate:
            Krishnan & Sheffield (2024). Mechanisms underlying the development and maintenance of stable
            spatial representations in the mouse hippocampus. https://doi.org/10.1038/s41467-024-50596-3
            Hainmueller & Bartos (2018). Parallel emergence of stable and dynamic memory engrams in the
            hippocampus. Nature. https://doi.org/10.1038/s41586-018-0191-2
        Reward-cell mixture-model framework and approach / zone / departure decomposition:
            Gauthier & Tank (2018). A dedicated population for reward coding in the hippocampus. Neuron.
            https://doi.org/10.1016/j.neuron.2018.06.008
            Issa, Radvansky, Xuan & Dombeck (2024). Lateral entorhinal cortex subpopulations represent
            experiential epochs surrounding reward. Nat Neurosci.
            https://doi.org/10.1038/s41593-023-01557-4
        Position-vs-speed partial-variance GLM with trial-label permutation null:
            Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to
            reward. Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4
            Hardcastle, Maheswaranathan, Ganguli & Giocomo (2017). A multiplexed, heterogeneous, and adaptive
            code for navigation in MEC. Neuron. https://doi.org/10.1016/j.neuron.2017.03.025

    Args:
        session_path: Path to the session's dataset directory.
        trial_types: Optional explicit tuple of trial types to evaluate. Default evaluates every entry in
            ``trial_geometry.yaml``.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
            analysis input.
        configuration: Wrapper holding the two sub-pipeline configurations. Uses defaults if None.
        verbose: When True, emit per-trial-type and per-detector progress echoes via the ataraxis console.
            Set to False by `run_tuning_analysis` when its session-level progress bar is the active visual
            signal so the bar stays clean.

    Returns:
        An in-memory `TuningReport` with one long-format row per ``(cell_id, trial_type)`` pair.
    """
    resolved_configuration = configuration if configuration is not None else TuningConfiguration.default()

    geometry = TrialGeometry.from_yaml(file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY))
    resolved_trial_types = tuple(geometry.entries.keys()) if trial_types is None else tuple(trial_types)
    if not resolved_trial_types:
        message = (
            f"Unable to compute tuning report: no trial types resolved from {session_path}. The "
            f"trial_geometry.yaml lists {list(geometry.entries.keys())}; trial_types argument was "
            f"{trial_types!r}."
        )
        console.error(message=message, error=ValueError)

    per_trial_cells: list[pl.DataFrame] = []
    trial_type_summaries: dict[str, TuningTrialSummary] = {}
    cell_count_reference: int | None = None
    sampling_rate_hz: float = float("nan")

    for trial_type in resolved_trial_types:
        if verbose:
            console.echo(message=f"Evaluating tuning for trial type {trial_type!r}...", level=LogLevel.INFO)
        cells_frame, trial_summary, cell_count, trial_sampling_rate_hz = _compute_trial_type(
            session_path=session_path,
            trial_type=trial_type,
            geometry=geometry,
            fluorescence_column=fluorescence_column,
            configuration=resolved_configuration,
            verbose=verbose,
        )
        per_trial_cells.append(cells_frame)
        trial_type_summaries[trial_type] = trial_summary
        if cell_count_reference is None:
            cell_count_reference = cell_count
            sampling_rate_hz = trial_sampling_rate_hz
        elif cell_count != cell_count_reference:
            message = (
                f"Cell count mismatch across trial types in session {session_path}: trial type "
                f"{trial_type!r} has {cell_count} cells but the first evaluated trial type had "
                f"{cell_count_reference}. Multi-day registered cell IDs must align across trial types; "
                f"check that ``MULTI_DAY_*`` fluorescence columns are loaded."
            )
            console.error(message=message, error=ValueError)

    cells = pl.concat(per_trial_cells, how="vertical_relaxed")

    summary = TuningSummary(
        place_configuration=resolved_configuration.place,
        reward_configuration=resolved_configuration.reward,
        sampling_rate_hz=sampling_rate_hz,
        cell_count=int(cell_count_reference if cell_count_reference is not None else 0),
        trial_types=list(resolved_trial_types),
        trial_type_summaries=trial_type_summaries,
    )
    return TuningReport(cells=cells, summary=summary)


def _compute_trial_type(
    session_path: Path,
    trial_type: str,
    *,
    geometry: TrialGeometry,
    fluorescence_column: FluorescenceColumn,
    configuration: TuningConfiguration,
    verbose: bool = True,
) -> tuple[pl.DataFrame, TuningTrialSummary, int, float]:
    """Runs the place-field and reward-cell detectors on a single trial type and returns the long-format rows.

    Args:
        session_path: Path to the session's dataset directory.
        trial_type: Trial type name to evaluate.
        geometry: Pre-loaded trial geometry covering ``trial_type``.
        fluorescence_column: Fluorescence column used as the analysis input.
        configuration: Sub-pipeline configurations to pass to both detectors.
        verbose: When True, emit per-detector progress echoes via the ataraxis console.

    Returns:
        A tuple of ``(cells_frame, trial_summary, cell_count, sampling_rate_hz)``. ``cells_frame`` already
        carries a `TuningColumn.TRIAL_TYPE` column populated with ``trial_type`` for every row.
    """
    geometry_entry = geometry.entries[trial_type]
    track_length_cm = float(geometry_entry.trial_length_cm)
    reward_position_cm = float(
        (geometry_entry.stimulus_trigger_zone_start_cm + geometry_entry.stimulus_trigger_zone_end_cm) / 2.0
    )

    run_session = assemble_run_session_data(
        session_path=session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
    )
    sampling_rate_hz = float(run_session.sampling_rate_hz)

    if verbose:
        console.echo(message="Running place field detection...", level=LogLevel.INFO)
    place_detector = PlaceFieldDetector(
        run_session=run_session,
        bin_size=_PLACE_FIELD_BIN_SIZE_CM,
        configuration=configuration.place,
    )
    place_fields = place_detector.detect()
    cell_count = int(place_fields.binned_fluorescence.shape[0])
    place_cell_count = int(place_fields.has_place_field.sum())
    if verbose:
        console.echo(
            message=f"Place field detection complete: {place_cell_count}/{cell_count} place cells.",
            level=LogLevel.SUCCESS,
        )

    if verbose:
        console.echo(message="Running reward cell detection...", level=LogLevel.INFO)
    reward_detector = RewardCellDetector(
        run_session=run_session,
        configuration=configuration.reward,
    )
    reward_results = reward_detector.detect(display_progress=verbose)
    spatially_significant_count = int(np.sum(reward_results.spatial_results.is_significant))
    reward_cell_count = int(reward_results.reward_cell_count)
    reward_predictive_count = len(reward_results.reward_predictive_indices)
    if verbose:
        console.echo(
            message=(
                f"Reward cell detection complete: {reward_cell_count} reward cells, "
                f"{reward_predictive_count} reward-predictive."
            ),
            level=LogLevel.SUCCESS,
        )

    rate_map_bin_count = int(reward_results.spatial_results.rate_maps.shape[1])

    stability_even_odd, stability_split_half = _compute_stability_metrics(
        binned_fluorescence_per_trial=place_fields.binned_fluorescence_per_trial,
        cell_count=cell_count,
    )
    if verbose:
        console.echo(
            message=(
                f"Running multi-criterion shuffles "
                f"({configuration.place.shuffle_repeat_count} iterations each for Peak and Stability)..."
            ),
            level=LogLevel.INFO,
        )
    is_stable, is_peak_significant, stability_p_values, peak_p_values = _compute_multi_criterion_flags(
        place_detector=place_detector,
        place_fields=place_fields,
        stability_split_half=stability_split_half,
        configuration=configuration.place,
        display_progress=verbose,
    )
    if verbose:
        console.echo(
            message=(
                f"Multi-criterion shuffles complete: {int(np.sum(is_stable))} stable, "
                f"{int(np.sum(is_peak_significant))} peak-significant."
            ),
            level=LogLevel.SUCCESS,
        )
    # noinspection PyTypeChecker
    is_strict_place: NDArray[np.bool_] = place_fields.has_place_field & is_stable & is_peak_significant
    # noinspection PyTypeChecker
    is_reward_cell_array: NDArray[np.bool_] = reward_results.spatial_results.is_significant & reward_results.is_zone
    # noinspection PyTypeChecker
    is_place_only: NDArray[np.bool_] = place_fields.has_place_field & ~is_reward_cell_array
    # noinspection PyTypeChecker
    is_strict_place_only: NDArray[np.bool_] = is_strict_place & ~is_reward_cell_array

    cells_frame = _build_cell_table(
        cell_count=cell_count,
        trial_type=trial_type,
        place_fields=place_fields,
        reward_results=reward_results,
        stability_even_odd=stability_even_odd,
        stability_split_half=stability_split_half,
        is_stable=is_stable,
        is_peak_significant=is_peak_significant,
        stability_p_values=stability_p_values,
        peak_p_values=peak_p_values,
        is_strict_place=is_strict_place,
    )

    trial_summary = TuningTrialSummary(
        track_length_cm=track_length_cm,
        reward_position_cm=reward_position_cm,
        bin_size_cm=float(configuration.reward.bin_size),
        bin_count=rate_map_bin_count,
        place_cell_count=place_cell_count,
        spatially_significant_count=spatially_significant_count,
        reward_cell_count=reward_cell_count,
        reward_predictive_count=reward_predictive_count,
        reliable_count=int(place_fields.has_place_field.sum()),
        stable_count=int(np.sum(is_stable)),
        peak_significant_count=int(np.sum(is_peak_significant)),
        strict_place_cell_count=int(np.sum(is_strict_place)),
        place_only_count=int(np.sum(is_place_only)),
        strict_place_only_count=int(np.sum(is_strict_place_only)),
        mixture_weight=float(reward_results.mixture_weight),
        gaussian_mean_cm=float(reward_results.gaussian_mean),
        gaussian_std_cm=float(reward_results.gaussian_std),
        track_start_weight=float(reward_results.track_start_weight),
        track_end_weight=float(reward_results.track_end_weight),
        track_start_std_cm=float(reward_results.track_start_std),
        track_end_std_cm=float(reward_results.track_end_std),
    )

    return cells_frame, trial_summary, cell_count, sampling_rate_hz


def _compute_stability_metrics(
    binned_fluorescence_per_trial: NDArray[np.float32],
    cell_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes per-cell even/odd and split-half Pearson correlations from a per-trial binned rate-map matrix.

    Notes:
        Both metrics return NaN for cells with fewer than two trials, fewer than three valid bins in either
        half, or zero variance in either half. Bins where either half has NaN are excluded pairwise from the
        correlation.

    Args:
        binned_fluorescence_per_trial: Per-trial binned fluorescence with dimensions (cell_count, trial_count,
            bin_count). NaN entries are treated as missing.
        cell_count: Number of cells in the session; used to size the output arrays when the per-trial matrix
            is empty.

    Returns:
        A tuple of (even_odd_r, split_half_r), each with length cell_count.
    """
    # noinspection PyTypeChecker
    even_odd: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    split_half: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)

    if binned_fluorescence_per_trial.size == 0:
        return even_odd, split_half
    trial_count = binned_fluorescence_per_trial.shape[1]
    minimum_trials_for_split = 2
    if trial_count < minimum_trials_for_split:
        return even_odd, split_half

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # noinspection PyTypeChecker
        even_map: NDArray[np.float32] = np.nanmean(binned_fluorescence_per_trial[:, 0::2, :], axis=1).astype(
            np.float32, copy=False
        )
        # noinspection PyTypeChecker
        odd_map: NDArray[np.float32] = np.nanmean(binned_fluorescence_per_trial[:, 1::2, :], axis=1).astype(
            np.float32, copy=False
        )
    even_odd = per_cell_pearson_safe(first_matrix=even_map, second_matrix=odd_map)

    half_index = trial_count // 2
    if half_index > 0 and trial_count - half_index > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # noinspection PyTypeChecker
            first_map: NDArray[np.float32] = np.nanmean(
                binned_fluorescence_per_trial[:, :half_index, :], axis=1
            ).astype(np.float32, copy=False)
            # noinspection PyTypeChecker
            second_map: NDArray[np.float32] = np.nanmean(
                binned_fluorescence_per_trial[:, half_index:, :], axis=1
            ).astype(np.float32, copy=False)
        split_half = per_cell_pearson_safe(first_matrix=first_map, second_matrix=second_map)

    return even_odd, split_half


def _compute_multi_criterion_flags(
    place_detector: PlaceFieldDetector,
    place_fields: PlaceFields,
    stability_split_half: NDArray[np.float32],
    configuration: PlaceFieldDetectionConfiguration,
    *,
    display_progress: bool = True,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]]:
    """Computes the IS_STABLE / IS_PEAK_SIGNIFICANT per-cell flags and their per-cell shuffle p-values.

    Args:
        place_detector: PlaceFieldDetector instance used to access shuffle infrastructure and pooled rate map.
        place_fields: PlaceFields output containing the pooled and per-trial rate maps.
        stability_split_half: Observed per-cell split-half Pearson r from ``_compute_stability_metrics``.
        configuration: Place-field configuration carrying ``shuffle_repeat_count`` and ``peak_percentile``.
        display_progress: When True, render the inner Peak / Stability shuffle tqdm bars; set to False by
            `_compute_trial_type` when its caller silenced per-stage echoes.

    Returns:
        A tuple of (is_stable, is_peak_significant, stability_p_values, peak_p_values), each with length
        cell_count.
    """
    cell_count = place_fields.binned_fluorescence.shape[0]
    is_stable, is_peak_significant, stability_p_values, peak_p_values = (
        place_detector.compute_multi_criterion_significance(
            observed_pooled_rate_map=place_fields.binned_fluorescence,
            observed_per_trial_rate_map=place_fields.binned_fluorescence_per_trial,
            observed_split_half_r=stability_split_half,
            repeat_count=configuration.shuffle_repeat_count,
            peak_percentile=configuration.peak_percentile,
            display_progress=display_progress,
        )
    )
    if is_stable.size != cell_count:
        # noinspection PyTypeChecker
        is_stable = np.zeros(cell_count, dtype=np.bool_)
    if is_peak_significant.size != cell_count:
        # noinspection PyTypeChecker
        is_peak_significant = np.zeros(cell_count, dtype=np.bool_)
    if stability_p_values.size != cell_count:
        # noinspection PyTypeChecker
        stability_p_values = np.full(cell_count, np.nan, dtype=np.float32)
    if peak_p_values.size != cell_count:
        # noinspection PyTypeChecker
        peak_p_values = np.full(cell_count, np.nan, dtype=np.float32)
    return is_stable, is_peak_significant, stability_p_values, peak_p_values


def _build_cell_table(
    cell_count: int,
    trial_type: str,
    place_fields: PlaceFields,
    reward_results: RewardCellResults,
    *,
    stability_even_odd: NDArray[np.float32],
    stability_split_half: NDArray[np.float32],
    is_stable: NDArray[np.bool_],
    is_peak_significant: NDArray[np.bool_],
    stability_p_values: NDArray[np.float32],
    peak_p_values: NDArray[np.float32],
    is_strict_place: NDArray[np.bool_],
) -> pl.DataFrame:
    """Assembles the per-trial-type slice of the long-format DataFrame from the live detector outputs.

    The returned DataFrame already carries a `TuningColumn.TRIAL_TYPE` column populated with
    ``trial_type`` for every row. The caller concatenates per-trial-type slices into the session-level
    long-format ``cells`` feather.
    """
    spatial = reward_results.spatial_results
    place_rows = _build_place_field_rows(cell_count=cell_count, place_fields=place_fields)

    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)
    is_reward_cell = spatial.is_significant & reward_results.is_reward_proximal

    keep_per_trial_mask = place_fields.has_place_field | spatial.is_significant
    binned_fluorescence_per_trial: list[list[list[float]] | None] = []
    for cell_index in range(cell_count):
        if keep_per_trial_mask[cell_index] and place_fields.binned_fluorescence_per_trial.size > 0:
            binned_fluorescence_per_trial.append(place_fields.binned_fluorescence_per_trial[cell_index].tolist())
        else:
            binned_fluorescence_per_trial.append(None)

    rate_maps_list = [spatial.rate_maps[cell_index].tolist() for cell_index in range(cell_count)]

    return pl.DataFrame(
        {
            TuningColumn.CELL_ID.value: cell_ids,
            TuningColumn.TRIAL_TYPE.value: pl.Series(values=[trial_type] * cell_count, dtype=pl.Utf8),
            TuningColumn.IS_PLACE.value: pl.Series(values=place_fields.has_place_field, dtype=pl.Boolean),
            TuningColumn.IS_SPATIALLY_SIGNIFICANT.value: pl.Series(values=spatial.is_significant, dtype=pl.Boolean),
            TuningColumn.IS_REWARD_PROXIMAL.value: pl.Series(values=reward_results.is_zone, dtype=pl.Boolean),
            TuningColumn.IS_APPROACH.value: pl.Series(values=reward_results.is_approach, dtype=pl.Boolean),
            TuningColumn.IS_ZONE.value: pl.Series(values=reward_results.is_zone, dtype=pl.Boolean),
            TuningColumn.IS_DEPARTURE.value: pl.Series(values=reward_results.is_departure, dtype=pl.Boolean),
            TuningColumn.IS_REWARD_CELL.value: pl.Series(values=is_reward_cell, dtype=pl.Boolean),
            TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value: pl.Series(
                values=reward_results.is_position_glm_significant, dtype=pl.Boolean
            ),
            TuningColumn.IS_RELIABLE.value: pl.Series(values=place_fields.has_place_field, dtype=pl.Boolean),
            TuningColumn.IS_STABLE.value: pl.Series(values=is_stable, dtype=pl.Boolean),
            TuningColumn.STABILITY_P_VALUE.value: pl.Series(values=stability_p_values, dtype=pl.Float32),
            TuningColumn.IS_PEAK_SIGNIFICANT.value: pl.Series(values=is_peak_significant, dtype=pl.Boolean),
            TuningColumn.PEAK_P_VALUE.value: pl.Series(values=peak_p_values, dtype=pl.Float32),
            TuningColumn.IS_STRICT_PLACE.value: pl.Series(values=is_strict_place, dtype=pl.Boolean),
            TuningColumn.PF_START_CM.value: pl.Series(values=place_rows["pf_start_cm"], dtype=pl.List(pl.Float32)),
            TuningColumn.PF_END_CM.value: pl.Series(values=place_rows["pf_end_cm"], dtype=pl.List(pl.Float32)),
            TuningColumn.PF_CENTER_CM.value: pl.Series(values=place_rows["pf_center_cm"], dtype=pl.List(pl.Float32)),
            TuningColumn.PF_MEAN_INTENSITY.value: pl.Series(
                values=place_rows["pf_mean_intensity"], dtype=pl.List(pl.Float32)
            ),
            TuningColumn.PF_MAX_INTENSITY.value: pl.Series(
                values=place_rows["pf_max_intensity"], dtype=pl.List(pl.Float32)
            ),
            TuningColumn.PF_WIDTH_CM.value: pl.Series(values=place_rows["pf_width_cm"], dtype=pl.List(pl.Float32)),
            TuningColumn.BINNED_FLUORESCENCE_PER_TRIAL.value: pl.Series(
                name=TuningColumn.BINNED_FLUORESCENCE_PER_TRIAL.value,
                values=binned_fluorescence_per_trial,
                dtype=pl.List(pl.List(pl.Float32)),
            ),
            TuningColumn.RATE_MAP.value: pl.Series(values=rate_maps_list, dtype=pl.List(pl.Float32)),
            TuningColumn.CENTER_OF_MASS_CM.value: pl.Series(values=spatial.centers_of_mass, dtype=pl.Float32),
            TuningColumn.SPATIAL_INFORMATION_Z.value: pl.Series(values=spatial.spatial_information_z, dtype=pl.Float32),
            TuningColumn.SPATIAL_FDR_SURVIVED.value: pl.Series(values=spatial.fdr_survived, dtype=pl.Boolean),
            TuningColumn.SPATIAL_SPLIT_HALF_R.value: pl.Series(values=spatial.split_half_r, dtype=pl.Float32),
            TuningColumn.SPATIAL_INFORMATION_BITS.value: pl.Series(
                values=spatial.spatial_information, dtype=pl.Float32
            ),
            TuningColumn.SPATIAL_P_VALUE.value: pl.Series(values=spatial.p_values, dtype=pl.Float32),
            TuningColumn.STABILITY_EVEN_ODD.value: pl.Series(values=stability_even_odd, dtype=pl.Float32),
            TuningColumn.STABILITY_SPLIT_HALF.value: pl.Series(values=stability_split_half, dtype=pl.Float32),
            TuningColumn.REWARD_RELATIVITY_SCORE.value: pl.Series(
                values=spatial.reward_relativity_score, dtype=pl.Float32
            ),
            TuningColumn.CV_POSITION_PARTIAL_R2.value: pl.Series(
                values=reward_results.cv_position_partial_r2, dtype=pl.Float32
            ),
            TuningColumn.POSITION_GLM_P_VALUE.value: pl.Series(
                values=reward_results.position_glm_p_values, dtype=pl.Float32
            ),
        },
    ).sort(TuningColumn.CELL_ID.value)


def _build_place_field_rows(cell_count: int, place_fields: PlaceFields) -> dict[str, list[list[float]]]:
    """Builds the per-cell list-of-fields columns for the cell feather from a PlaceFields instance."""
    label_image = place_fields.label_image
    bin_size = place_fields.bin_size
    region_count = int(np.max(label_image)) if label_image.size > 0 else 0

    field_cell_ids = place_fields.cell_id
    centers = place_fields.centers
    mean_intensities = place_fields.mean_intensity
    max_intensities = place_fields.max_intensity

    pf_start_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_end_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_center_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_mean_intensity: list[list[float]] = [[] for _ in range(cell_count)]
    pf_max_intensity: list[list[float]] = [[] for _ in range(cell_count)]
    pf_width_cm: list[list[float]] = [[] for _ in range(cell_count)]

    for field_index in range(region_count):
        label = field_index + 1
        cell_index = int(field_cell_ids[field_index])
        # noinspection PyTypeChecker
        bins: NDArray[np.int64] = np.where(label_image[cell_index, :] == label)[0]
        # noinspection PyTypeChecker
        gap_indices: NDArray[np.int64] = np.where(np.diff(bins) > 1)[0]
        if gap_indices.size > 0:
            gap = gap_indices[0]
            start_bin = bins[gap + 1]
            end_bin = bins[gap]
        else:
            start_bin = bins[0]
            end_bin = bins[-1]

        pf_start_cm[cell_index].append(float(start_bin * bin_size))
        pf_end_cm[cell_index].append(float(end_bin * bin_size))
        pf_center_cm[cell_index].append(float(centers[field_index, 1]))
        pf_mean_intensity[cell_index].append(float(mean_intensities[field_index]))
        pf_max_intensity[cell_index].append(float(max_intensities[field_index]))
        pf_width_cm[cell_index].append(float(len(bins) * bin_size))

    return {
        "pf_start_cm": pf_start_cm,
        "pf_end_cm": pf_end_cm,
        "pf_center_cm": pf_center_cm,
        "pf_mean_intensity": pf_mean_intensity,
        "pf_max_intensity": pf_max_intensity,
        "pf_width_cm": pf_width_cm,
    }
