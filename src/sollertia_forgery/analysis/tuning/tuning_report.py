"""Per-session tuning-report container that consolidates place-field and reward-cell detection.

The :class:`TuningReport` pair (per-cell feather + summary YAML) is the analysis counterpart to
:class:`BleachingReport`; the report owns persistence and population-mask resolution so loading a saved report
is sufficient to reproduce every figure without rerunning detection. Plot regeneration lives in
:mod:`.plotting`. SCE-related analyses live alongside in :mod:`..sce` and produce their own per-session report.

References:
    - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of cells
      identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- Stability and Peak methods
      that gate ``IS_STABLE`` and ``IS_PEAK_SIGNIFICANT``.
    - Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to reward.
      Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4 -- the position-vs-speed GLM ΔR² that gates
      ``IS_POSITION_GLM_SIGNIFICANT``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

from ...forging import FluorescenceColumn
from .utilities import per_cell_pearson_safe, assemble_run_session_data
from ...shared_assets import DatasetFiles, TrialGeometry
from .place_tuning_protocol import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_tuning_protocol import RewardCellResults, RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


_PLACE_FIELD_BIN_SIZE_CM: float = 5.0
"""Default spatial bin size in centimeters used by ``PlaceFieldDetector`` and persisted in the per-trial
fluorescence matrix. Matches the ``PlaceFieldDetector`` constructor default."""


class TuningColumn(StrEnum):
    """Defines every column written to the per-session ``tuning_cells.feather`` per-cell table."""

    CELL_ID = "cell_id"
    """Contiguous integer cell identifier."""
    IS_PLACE = "is_place"
    """True for cells with at least one detected place field."""
    IS_SPATIALLY_SIGNIFICANT = "is_spatially_significant"
    """True for cells whose Skaggs spatial information is significant under shuffle testing."""
    IS_REWARD_PROXIMAL = "is_reward_proximal"
    """True for cells whose circular center of mass falls within the reward zone. Alias of ``IS_ZONE`` preserved
    for backwards compatibility; bit-identical to it."""
    IS_APPROACH = "is_approach"
    """True for cells whose center of mass falls in the approach band immediately upstream of the reward zone
    (Issa, Radvansky, Xuan & Dombeck 2024 anticipatory band; default 40 cm)."""
    IS_ZONE = "is_zone"
    """True for cells whose center of mass falls inside the reward zone (Issa et al. 2024 zone band; configured
    by ``reward_zone_width``)."""
    IS_DEPARTURE = "is_departure"
    """True for cells whose center of mass falls in the departure band immediately downstream of the reward zone
    (Issa et al. 2024 post-reward band; default 40 cm)."""
    IS_REWARD_CELL = "is_reward_cell"
    """True for cells that are both spatially significant and reward-proximal (zone-band)."""
    IS_POSITION_GLM_SIGNIFICANT = "is_position_glm_significant"
    """True for cells whose 5-fold CV ΔR² of position over speed+acceleration exceeds the trial-label permutation
    null at the configured ``glm_significance_threshold`` (Sosa, Plitt & Giocomo 2025; Hardcastle et al. 2017)."""
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
    """Z-scored Skaggs spatial information against the same circular-shift null used for ``SPATIAL_P_VALUE``
    (Souza & Tort 2018; Sheintuch et al. 2022)."""
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
    """True for cells with at least one place field that passes the lap-coverage criterion (Climer 2025)."""
    IS_STABLE = "is_stable"
    """True for cells whose split-half stability r exceeds the 95th percentile of a per-cell shuffled null
    (Climer & Dombeck 2021 Stability method)."""
    STABILITY_P_VALUE = "stability_p_value"
    """Per-cell p-value for the Stability shuffle."""
    IS_PEAK_SIGNIFICANT = "is_peak_significant"
    """True for cells whose observed pooled-rate-map peak exceeds the 99th percentile of the shuffled per-cell
    peak distribution (Climer & Dombeck 2021 Peak method)."""
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
class TuningSummary(YamlConfig):
    """Per-session YAML companion to ``tuning_cells.feather``.

    Carries only fields that are not derivable from the cells feather: the two sub-configurations (load-bearing
    because the per-cell flag columns were computed against their thresholds), the mixture-model fit (cannot be
    rederived without rerunning the EM step), session-level geometry and sampling metadata, and precomputed
    cell-population counts that ``summarize()`` reports.
    """

    place_configuration: PlaceFieldDetectionConfiguration
    """Place-field detection configuration that produced the place-field columns."""
    reward_configuration: RewardCellConfiguration
    """Reward-cell detection configuration that produced the reward and slowing columns."""

    track_length_cm: float
    """Track length in centimeters resolved from ``trial_geometry.yaml``."""
    reward_position_cm: float
    """Reward position in centimeters (midpoint of the stimulus trigger zone)."""
    sampling_rate_hz: float
    """Acquisition sampling rate in Hz."""
    bin_size_cm: float
    """Spatial bin size in centimeters used for rate maps and per-cell place-field detection."""
    bin_count: int
    """Number of spatial bins along the track."""

    cell_count: int
    """Total number of cells in the session."""
    place_cell_count: int
    """Number of cells with at least one detected place field (Dombeck-style threshold + lap-coverage filter)."""
    spatially_significant_count: int
    """Number of cells whose Skaggs spatial information passes the shuffle threshold."""
    reward_cell_count: int
    """Number of cells that are both spatially significant and reward-proximal."""
    reward_predictive_count: int
    """Number of cells that are reward-associated and slowing-correlated."""
    reliable_count: int
    """Number of cells whose detected fields pass the lap-coverage criterion (Climer 2025)."""
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
    """Reward-component weight from the four-component (uniform + reward + track-start + track-end) mixture
    model fit to the spatially significant COMs."""
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


@dataclass(frozen=True, slots=True)
class TuningReport:
    """Top-level per-session container for the place-field + reward-cell tuning analysis.

    Holds the per-cell tuning table (``cells``) and the YAML summary (``summary``). Persistence and
    population-mask resolution stay on the report; plot regeneration lives in :mod:`.plotting`.
    """

    cells: pl.DataFrame
    """Per-cell wide table; one row per cell. Schema enumerated by :class:`TuningColumn`."""
    summary: TuningSummary
    """YAML wrapper holding the configurations, mixture-model fit, and session-level scalars."""

    @classmethod
    def evaluate(
        cls,
        session_path: Path,
        *,
        trial_type: str = "ABC",
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        configuration: TuningConfiguration | None = None,
    ) -> TuningReport:
        """Runs the place-field and reward-cell pipelines sequentially and assembles an in-memory report.

        Notes:
            Returns the report unsaved so callers can inspect or plot before persisting.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type: Trial type to analyze. Must match an entry in ``trial_geometry.yaml``.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            configuration: Wrapper holding the two sub-pipeline configurations. Uses defaults if None.

        Returns:
            An in-memory TuningReport ready to be saved or plotted.
        """
        resolved_configuration = configuration if configuration is not None else TuningConfiguration.default()

        # Resolves canonical track length and reward position from the session's trial geometry data file.
        geometry_entry = TrialGeometry.from_yaml(file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY)).entries[
            trial_type
        ]
        track_length_cm = float(geometry_entry.trial_length_cm)
        reward_position_cm = float(
            (geometry_entry.stimulus_trigger_zone_start_cm + geometry_entry.stimulus_trigger_zone_end_cm) / 2.0
        )

        # Loads the session data once and shares it across both detectors via from_run_session, so the place
        # and reward flags operate on identical speed-filtered samples and bit-identical rate maps.
        run_session = assemble_run_session_data(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )
        sampling_rate_hz = float(run_session.sampling_rate_hz)

        console.echo(message="Running place field detection...", level=LogLevel.INFO)
        place_detector = PlaceFieldDetector(
            run_session=run_session,
            bin_size=_PLACE_FIELD_BIN_SIZE_CM,
            configuration=resolved_configuration.place,
        )
        place_fields = place_detector.detect()
        cell_count = int(place_fields.binned_fluorescence.shape[0])
        place_cell_count = int(place_fields.has_place_field.sum())
        console.echo(
            message=f"Place field detection complete: {place_cell_count}/{cell_count} place cells.",
            level=LogLevel.SUCCESS,
        )

        console.echo(message="Running reward cell detection...", level=LogLevel.INFO)
        reward_detector = RewardCellDetector(
            run_session=run_session,
            configuration=resolved_configuration.reward,
        )
        reward_results = reward_detector.detect()
        spatially_significant_count = int(np.sum(reward_results.spatial_results.is_significant))
        reward_cell_count = int(reward_results.reward_cell_count)
        reward_predictive_count = len(reward_results.reward_predictive_indices)
        console.echo(
            message=(
                f"Reward cell detection complete: {reward_cell_count} reward cells, "
                f"{reward_predictive_count} reward-predictive."
            ),
            level=LogLevel.SUCCESS,
        )

        rate_map_bin_count = int(reward_results.spatial_results.rate_maps.shape[1])

        # Computes split-half and even/odd stability r per cell from the per-trial binned fluorescence already
        # produced during place-field detection.
        stability_even_odd, stability_split_half = _compute_stability_metrics(
            binned_fluorescence_per_trial=place_fields.binned_fluorescence_per_trial,
            cell_count=cell_count,
        )
        console.echo(
            message=(
                f"Running multi-criterion shuffles "
                f"({resolved_configuration.place.shuffle_repeat_count} iterations each for Peak and Stability)..."
            ),
            level=LogLevel.INFO,
        )
        is_stable, is_peak_significant, stability_p_values, peak_p_values = _compute_multi_criterion_flags(
            place_detector=place_detector,
            place_fields=place_fields,
            stability_split_half=stability_split_half,
            configuration=resolved_configuration.place,
        )
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
        is_reward_cell_array: NDArray[np.bool_] = (
            reward_results.spatial_results.is_significant & reward_results.is_zone
        )
        # noinspection PyTypeChecker
        is_place_only: NDArray[np.bool_] = place_fields.has_place_field & ~is_reward_cell_array
        # noinspection PyTypeChecker
        is_strict_place_only: NDArray[np.bool_] = is_strict_place & ~is_reward_cell_array

        cells = _build_cell_table(
            cell_count=cell_count,
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

        summary = TuningSummary(
            place_configuration=resolved_configuration.place,
            reward_configuration=resolved_configuration.reward,
            track_length_cm=track_length_cm,
            reward_position_cm=reward_position_cm,
            sampling_rate_hz=sampling_rate_hz,
            bin_size_cm=float(resolved_configuration.reward.bin_size),
            bin_count=rate_map_bin_count,
            cell_count=cell_count,
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
        return cls(cells=cells, summary=summary)

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

    def place_mask(
        self,
        *,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
    ) -> NDArray[np.bool_]:
        """Returns the per-cell boolean mask for cells passing every requested place-cell criterion simultaneously.

        Notes:
            Defaults to the strict triple-AND of place / stable / peak-significant recommended by Climer &
            Dombeck (2021). When every kwarg is False the method returns an all-True mask, treating "no
            criteria" as "no filter". Skaggs spatial significance is intentionally not exposed here: it is the
            reward-cell pipeline's broader spatial filter and overlaps heavily with Dombeck place-field
            morphology. Use :meth:`resolve_population_masks` instead when you need place / reward populations
            that respect mutual exclusion.

        Args:
            require_place: Require ``IS_PLACE`` (Dombeck 2010 morphology + lap coverage).
            require_stable: Require ``IS_STABLE`` (Climer & Dombeck 2021 Stability method).
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (Climer & Dombeck 2021 Peak method).

        Returns:
            Per-cell boolean mask with length cell_count.
        """
        # noinspection PyTypeChecker
        mask: NDArray[np.bool_] = np.ones(self.cells.height, dtype=np.bool_)
        if require_place:
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = self.cells[TuningColumn.IS_PLACE.value].to_numpy()
            mask = mask & place_flag
        if require_stable:
            # noinspection PyTypeChecker
            stable_flag: NDArray[np.bool_] = self.cells[TuningColumn.IS_STABLE.value].to_numpy()
            mask = mask & stable_flag
        if require_peak_significant:
            # noinspection PyTypeChecker
            peak_flag: NDArray[np.bool_] = self.cells[TuningColumn.IS_PEAK_SIGNIFICANT.value].to_numpy()
            mask = mask & peak_flag
        return mask

    def resolve_population_masks(
        self,
        *,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
        mutually_exclusive: bool = True,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Returns ``(place_mask, reward_mask)`` honoring the mutual-exclusion option.

        Notes:
            The reward mask is always ``IS_REWARD_CELL`` (i.e., spatially significant cells whose COM lies in
            the zone band). When ``mutually_exclusive=True``, cells flagged as both place and reward are
            subtracted from the place mask so the two populations are disjoint at the presentation layer; when
            ``False``, the place mask is the unfiltered place population and a cell may appear in both. The
            persisted feather always keeps both flags independently — mutual exclusion is purely a
            presentation-layer convention so the distinction is reversible without re-running detection.

        Args:
            require_place: Require ``IS_PLACE`` for the place population.
            require_stable: Require ``IS_STABLE`` for the place population.
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` for the place population.
            mutually_exclusive: When True, subtract ``IS_REWARD_CELL`` cells from the place mask. Default True
                for visualization clarity; set False to keep both populations as recorded in the table.

        Returns:
            A tuple of (place_mask, reward_mask) per-cell boolean arrays each with length cell_count.
        """
        place_population = self.place_mask(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
        )
        # noinspection PyTypeChecker
        reward_mask: NDArray[np.bool_] = self.cells[TuningColumn.IS_REWARD_CELL.value].to_numpy()
        place_mask = place_population & ~reward_mask if mutually_exclusive else place_population
        return place_mask, reward_mask

    def summarize(self, *, mutually_exclusive: bool = True) -> str:
        """Returns a multi-line human-readable summary of the report's per-cell statistics.

        Notes:
            With ``mutually_exclusive=True`` (default), the place-cell count subtracts cells also flagged as
            ``IS_REWARD_CELL`` and is reported as "place-only"; the reward count is unchanged. With
            ``mutually_exclusive=False``, the raw counts persisted in the summary YAML are reported instead and
            a cell may contribute to both totals.

        Args:
            mutually_exclusive: When True (default), report ``IS_PLACE & ~IS_REWARD_CELL`` for the place count.
        """
        summary = self.summary
        cell_count = summary.cell_count

        if mutually_exclusive and cell_count > 0:
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = self.cells[TuningColumn.IS_PLACE.value].to_numpy()
            # noinspection PyTypeChecker
            reward_flag: NDArray[np.bool_] = self.cells[TuningColumn.IS_REWARD_CELL.value].to_numpy()
            place_count = int(np.sum(place_flag & ~reward_flag))
            place_label = "Place-only (Dombeck):"
        else:
            place_count = summary.place_cell_count
            place_label = "Place cells (Dombeck):"

        spatially_pct = 100.0 * summary.spatially_significant_count / cell_count if cell_count > 0 else 0.0
        place_pct = 100.0 * place_count / cell_count if cell_count > 0 else 0.0
        reward_pct = 100.0 * summary.reward_cell_count / cell_count if cell_count > 0 else 0.0
        predictive_pct = 100.0 * summary.reward_predictive_count / cell_count if cell_count > 0 else 0.0
        reliable_pct = 100.0 * summary.reliable_count / cell_count if cell_count > 0 else 0.0
        stable_pct = 100.0 * summary.stable_count / cell_count if cell_count > 0 else 0.0
        peak_pct = 100.0 * summary.peak_significant_count / cell_count if cell_count > 0 else 0.0
        strict_pct = 100.0 * summary.strict_place_cell_count / cell_count if cell_count > 0 else 0.0

        lines = [
            "Tuning report",
            "=============",
            f"Cells: {cell_count}",
            f"  {place_label:<22} {place_count} ({place_pct:.1f}%)",
            f"  Reliable (lap cov.):   {summary.reliable_count} ({reliable_pct:.1f}%)",
            f"  Stable (split-half):   {summary.stable_count} ({stable_pct:.1f}%)",
            f"  Peak-significant:      {summary.peak_significant_count} ({peak_pct:.1f}%)",
            f"  Strict place cells:    {summary.strict_place_cell_count} ({strict_pct:.1f}%)",
            f"  Spatially significant: {summary.spatially_significant_count} ({spatially_pct:.1f}%)",
            f"  Reward cells:          {summary.reward_cell_count} ({reward_pct:.1f}%)",
            f"  Reward-predictive:     {summary.reward_predictive_count} ({predictive_pct:.1f}%)",
            "",
            "Reward mixture model (uniform + reward + track-start + track-end):",
            f"  Reward weight:     {summary.mixture_weight:.3f}",
            f"  Reward Gaussian:   {summary.gaussian_mean_cm:.1f} cm (SD {summary.gaussian_std_cm:.1f} cm)",
            f"  Track-start wt:    {summary.track_start_weight:.3f} (SD {summary.track_start_std_cm:.1f} cm)",
            f"  Track-end wt:      {summary.track_end_weight:.3f} (SD {summary.track_end_std_cm:.1f} cm)",
            f"  Reward position:   {summary.reward_position_cm:.1f} cm",
            "",
            "Geometry / sampling:",
            f"  Track length:      {summary.track_length_cm:.1f} cm",
            f"  Bin size:          {summary.bin_size_cm:.1f} cm ({summary.bin_count} bins)",
            f"  Sampling rate:     {summary.sampling_rate_hz:.2f} Hz",
        ]
        return "\n".join(lines)


def evaluate_and_save_tuning_report(
    session: DatasetSession,
    *,
    trial_type: str = "ABC",
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    configuration: TuningConfiguration | None = None,
) -> TuningReport:
    """Evaluates the tuning pipeline for a single session and persists the report to disk.

    Args:
        session: The DatasetSession to analyze.
        trial_type: Trial type to analyze.
        fluorescence_column: Fluorescence column to use as the analysis input.
        configuration: Wrapper holding the two sub-pipeline configurations. Uses defaults if None.

    Returns:
        The TuningReport produced for the session, with both artifacts persisted under the session directory.
    """
    report = TuningReport.evaluate(
        session_path=session.session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
        configuration=configuration,
    )
    report.save(session=session)
    return report


# ===== Private helpers ==========================================================================================


def _compute_stability_metrics(
    binned_fluorescence_per_trial: NDArray[np.float32],
    cell_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes per-cell even/odd and split-half Pearson correlations from a per-trial binned rate-map matrix.

    Notes:
        Both metrics return NaN for cells with fewer than two trials, fewer than three valid bins in either
        half, or zero variance in either half. Bins where either half has NaN are excluded pairwise from the
        correlation.

    References:
        - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of
          cells identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- Stability method.
        - Hainmueller & Bartos (2018). Parallel emergence of stable and dynamic memory engrams in the
          hippocampus. Nature. https://doi.org/10.1038/s41586-018-0191-2 -- split-half stability r as a
          place-cell criterion.

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
    even_odd = per_cell_pearson_safe(a=even_map, b=odd_map)

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
        split_half = per_cell_pearson_safe(a=first_map, b=second_map)

    return even_odd, split_half


def _compute_multi_criterion_flags(
    place_detector: PlaceFieldDetector,
    place_fields: PlaceFields,
    stability_split_half: NDArray[np.float32],
    configuration: PlaceFieldDetectionConfiguration,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]]:
    """Computes the IS_STABLE / IS_PEAK_SIGNIFICANT per-cell flags and their per-cell shuffle p-values.

    References:
        - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of
          cells identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- Peak method (99th
          percentile) and Stability method (95th percentile).

    Args:
        place_detector: PlaceFieldDetector instance used to access shuffle infrastructure and pooled rate map.
        place_fields: PlaceFields output containing the pooled and per-trial rate maps.
        stability_split_half: Observed per-cell split-half Pearson r from ``_compute_stability_metrics``.
        configuration: Place-field configuration carrying ``shuffle_repeat_count`` and ``peak_percentile``.

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
    """Assembles the per-cell wide-format DataFrame from the live detector outputs."""
    spatial = reward_results.spatial_results
    place_rows = _build_place_field_rows(cell_count=cell_count, place_fields=place_fields)

    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)
    is_reward_cell = spatial.is_significant & reward_results.is_reward_proximal

    # Persists the per-trial binned fluorescence for any cell that is either a place cell or spatially
    # significant so reward-cell plotting paths can use the same column without an extra rebinning pass.
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
            TuningColumn.PF_CENTER_CM.value: pl.Series(
                values=place_rows["pf_center_cm"], dtype=pl.List(pl.Float32)
            ),
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
            TuningColumn.SPATIAL_INFORMATION_Z.value: pl.Series(
                values=spatial.spatial_information_z, dtype=pl.Float32
            ),
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
