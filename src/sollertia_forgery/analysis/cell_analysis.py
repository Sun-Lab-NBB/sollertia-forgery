"""Per-session cell-analysis pipeline that consolidates place-field, reward-cell, and SCE detection into a single
report container.

The :class:`CellAnalysisReport` triplet (per-cell feather + summary YAML + per-period SCE feather) is the analysis
counterpart to :class:`BleachingReport`; the report owns persistence and plot regeneration so loading a saved report
is sufficient to reproduce every figure without rerunning detection. Module-level ``plot_dataset_*`` helpers walk a
:class:`DatasetData` instance and stitch per-session reports into across-animal aggregates following the same
silent-skip / IQR-shading pattern as the chronic photobleaching dataset plot.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from ataraxis_base_utilities import LogLevel, console
from scipy.cluster.hierarchy import linkage, fcluster
from ataraxis_data_structures import YamlConfig

from ..forging import FluorescenceColumn
from .utilities import resolve_display_units, trim_acquisition_warmup, assemble_run_session_data
from .sce_protocol import SCEResult, PeriodType, SCEDetector, SCEDetectionConfiguration
from ..shared_assets import (
    DatasetData,
    DatasetFiles,
    DatasetColumn,
    TrialGeometry,
    DatasetSession,
)
from .place_cell_protocol import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_cell_protocol import RewardCellResults, RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray


_PLOT_TICK_INTERVAL_CM: float = 25.0
"""Spacing in centimeters between x-axis ticks on track-position plots."""
_PLACE_FIELD_BIN_SIZE_CM: float = 5.0
"""Default spatial bin size in centimeters used by ``PlaceFieldDetector`` and persisted in the per-trial fluorescence
matrix. Matches the ``PlaceFieldDetector`` constructor default."""
_MINIMUM_SCE_COUNT_FOR_ASSEMBLY: int = 2
"""Minimum number of SCEs required in a period to attempt cell-assembly detection."""
_MINIMUM_PERIODS_FOR_REST_RUN_REST: int = 1
"""Minimum number of run periods required to render a rest-run-rest plot."""


class CellAnalysisColumn(StrEnum):
    """Defines every column written to the per-session ``cell_analysis.feather`` per-cell table."""

    CELL_ID = "cell_id"
    """Contiguous integer cell identifier."""
    IS_PLACE = "is_place"
    """True for cells with at least one detected place field."""
    IS_SPATIALLY_SIGNIFICANT = "is_spatially_significant"
    """True for cells whose Skaggs spatial information is significant under shuffle testing."""
    IS_REWARD_PROXIMAL = "is_reward_proximal"
    """True for cells whose circular center of mass falls within the reward zone."""
    IS_REWARD_CELL = "is_reward_cell"
    """True for cells that are both spatially significant and reward-proximal."""
    IS_SLOWING_CORRELATED = "is_slowing_correlated"
    """True for cells with significant negative speed-activity correlation in the pre-reward window."""
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
    SPATIAL_P_VALUE = "spatial_p_value"
    """Shuffle-derived p-value for the spatial information statistic."""
    SPEED_ACTIVITY_CORRELATION = "speed_activity_correlation"
    """Pearson correlation between binned speed and activity in the pre-reward window."""
    SCE_PARTICIPATION_COUNT_REST = "sce_participation_count_rest"
    """Total number of rest-period SCEs the cell participated in."""
    SCE_PARTICIPATION_COUNT_RUN = "sce_participation_count_run"
    """Total number of run-period SCEs the cell participated in."""
    SCE_PARTICIPATION_RATE_REST = "sce_participation_rate_rest"
    """Fraction of rest-period SCEs the cell participated in."""
    SCE_PARTICIPATION_RATE_RUN = "sce_participation_rate_run"
    """Fraction of run-period SCEs the cell participated in."""
    SCE_MEAN_ONSET_RANK_REST = "sce_mean_onset_rank_rest"
    """Mean normalized onset rank within rest-period SCEs the cell participated in."""
    SCE_MEAN_ONSET_RANK_RUN = "sce_mean_onset_rank_run"
    """Mean normalized onset rank within run-period SCEs the cell participated in."""
    SCE_EVENTS_REST = "sce_events_rest"
    """Per-cell list of (period_index, sce_label) pairs for every rest-period SCE the cell participated in."""
    SCE_EVENTS_RUN = "sce_events_run"
    """Per-cell list of (period_index, sce_label) pairs for every run-period SCE the cell participated in."""


class SCEPeriodColumn(StrEnum):
    """Defines every column written to the per-session ``sce_periods.feather`` per-period table."""

    PERIOD_TYPE = "period_type"
    """Period kind: ``"rest"`` or ``"run"``."""
    PERIOD_INDEX = "period_index"
    """Index within the period type, 0-based, matching the ``(period_index, sce_label)`` references in
    ``CellAnalysisColumn.SCE_EVENTS_*``."""
    CELL_COUNT = "cell_count"
    """Cell count active during the period; constant across all rows in a session and equal to the cell-feather
    height."""
    SAMPLE_COUNT = "sample_count"
    """Number of samples retained in this period after stability or place-field masking."""
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


@dataclass(frozen=True, slots=True)
class CellAnalysisConfiguration:
    """Wraps the three sub-pipeline configurations together so a single object drives all of evaluate()."""

    place: PlaceFieldDetectionConfiguration
    """Place-field detection parameters."""
    reward: RewardCellConfiguration
    """Reward-cell detection parameters."""
    sce: SCEDetectionConfiguration
    """Synchronous calcium event detection parameters."""

    @classmethod
    def default(cls) -> CellAnalysisConfiguration:
        """Returns a CellAnalysisConfiguration whose three sub-configurations all use library defaults."""
        return cls(
            place=PlaceFieldDetectionConfiguration(),
            reward=RewardCellConfiguration(),
            sce=SCEDetectionConfiguration(),
        )


@dataclass
class CellAnalysisSummary(YamlConfig):
    """Per-session YAML companion to ``cell_analysis.feather`` and ``sce_periods.feather``.

    Carries only fields that are not derivable from the two feathers: the three sub-configurations (load-bearing
    because the per-cell flag columns were computed against their thresholds), the mixture-model fit (cannot be
    rederived without rerunning the EM step), session-level geometry and sampling metadata, precomputed cell-population
    counts that ``summarize()`` reports, and the SCE per-period counts that anchor cross-session aggregates.
    """

    place_configuration: PlaceFieldDetectionConfiguration
    """Place-field detection configuration that produced the place-field columns."""
    reward_configuration: RewardCellConfiguration
    """Reward-cell detection configuration that produced the reward and slowing columns."""
    sce_configuration: SCEDetectionConfiguration
    """SCE detection configuration that produced the per-cell SCE columns and the per-period feather."""

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
    """Number of cells with at least one detected place field."""
    spatially_significant_count: int
    """Number of cells whose Skaggs spatial information passes the shuffle threshold."""
    reward_cell_count: int
    """Number of cells that are both spatially significant and reward-proximal."""
    reward_predictive_count: int
    """Number of cells that are reward-associated and slowing-correlated."""

    mixture_weight: float
    """Reward-component weight from the uniform + Gaussian mixture model fit to the spatially significant COMs."""
    gaussian_mean_cm: float
    """Reward-component Gaussian mean in centimeters."""
    gaussian_std_cm: float
    """Reward-component Gaussian standard deviation in centimeters."""

    rest_period_count: int
    """Number of rest periods that survived torque-stability filtering and contributed an SCE row."""
    run_period_count: int
    """Number of run periods that contributed an SCE row."""
    total_rest_sces: int
    """Total SCEs detected across all rest periods."""
    total_run_sces: int
    """Total SCEs detected across all run periods."""


@dataclass(frozen=True, slots=True)
class CellAnalysisReport:
    """Top-level per-session container for the unified place / reward / SCE analysis.

    Holds the per-cell table (``table``), the per-period SCE table (``sce_periods``), and the YAML summary
    (``summary``). Every plot method touches only these three fields plus, where noted, the session's
    ``data.feather`` opened memory-mapped at plot time. All filenames are declared on
    :class:`DatasetSession`; this class never spells a filename literally.
    """

    table: pl.DataFrame
    """Per-cell wide table; one row per cell. Schema enumerated by :class:`CellAnalysisColumn`."""
    sce_periods: pl.DataFrame
    """Per-period SCE state table; one row per detected rest or run period. Schema enumerated by
    :class:`SCEPeriodColumn`."""
    summary: CellAnalysisSummary
    """YAML wrapper holding the configurations, mixture-model fit, and session-level scalars."""

    @classmethod
    def evaluate(
        cls,
        session_path: Path,
        *,
        trial_type: str = "ABC",
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
        configuration: CellAnalysisConfiguration | None = None,
    ) -> CellAnalysisReport:
        """Runs the place / reward / SCE pipelines sequentially and assembles an in-memory report.

        Notes:
            Passes the live ``PlaceFields`` instance directly to ``SCEDetector`` rather than rehydrating it from
            disk; place-field reconstruction from the cell feather is no longer needed. Returns the report unsaved
            so callers can inspect or plot before persisting.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type: Trial type to analyze. Must match an entry in ``trial_geometry.yaml``.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            configuration: Wrapper holding the three sub-pipeline configurations. Uses defaults if None.

        Returns:
            An in-memory CellAnalysisReport ready to be saved or plotted.
        """
        resolved_configuration = configuration if configuration is not None else CellAnalysisConfiguration.default()

        # Resolves canonical track length and reward position from the session's trial geometry data file.
        geometry_entry = TrialGeometry.from_yaml(file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY)).entries[
            trial_type
        ]
        track_length_cm = float(geometry_entry.trial_length_cm)
        reward_position_cm = float(
            (geometry_entry.stimulus_trigger_zone_start_cm + geometry_entry.stimulus_trigger_zone_end_cm) / 2.0
        )

        # Runs place-field detection.
        console.echo(message="Running place field detection...", level=LogLevel.INFO)
        place_detector = PlaceFieldDetector(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
            bin_size=_PLACE_FIELD_BIN_SIZE_CM,
            configuration=resolved_configuration.place,
        )
        place_fields = place_detector.detect(run_shuffle=False)
        cell_count = int(place_fields.binned_fluorescence.shape[0])
        place_cell_count = int(place_fields.has_place_field.sum())
        console.echo(
            message=f"Place field detection complete: {place_cell_count}/{cell_count} place cells.",
            level=LogLevel.SUCCESS,
        )

        # Runs reward-cell detection.
        console.echo(message="Running reward cell detection...", level=LogLevel.INFO)
        reward_detector = RewardCellDetector(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
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

        # Runs SCE detection. The detector exposes per-period sample-index arrays so plot regeneration can
        # re-slice ``data.feather`` without persisting the full smoothed-fluorescence trace.
        console.echo(message="Running SCE detection...", level=LogLevel.INFO)
        sce_detector = SCEDetector(
            session_path=session_path,
            track_length=track_length_cm,
            fluorescence_column=fluorescence_column,
            place_fields=place_fields,
            configuration=resolved_configuration.sce,
        )
        sce_detector.detect_events()
        rest_period_count = len(sce_detector.rest_results)
        run_period_count = len(sce_detector.run_results)
        total_rest_sces = int(sum(int(np.max(r.sce_labels)) for r in sce_detector.rest_results))
        total_run_sces = int(sum(int(np.max(r.sce_labels)) for r in sce_detector.run_results))
        console.echo(
            message=(
                f"SCE detection complete: {rest_period_count} rest periods ({total_rest_sces} SCEs), "
                f"{run_period_count} run periods ({total_run_sces} SCEs)."
            ),
            level=LogLevel.SUCCESS,
        )

        sce_results: list[SCEResult] = sce_detector.results
        sampling_rate_hz = float(sce_detector.sampling_rate_hz)
        rate_map_bin_count = int(reward_results.spatial_results.rate_maps.shape[1])

        # Assembles the per-cell wide table.
        table = _build_cell_table(
            cell_count=cell_count,
            place_fields=place_fields,
            reward_results=reward_results,
            sce_results=sce_results,
        )

        # Assembles the per-period SCE table.
        sce_periods = _build_sce_periods_table(
            sampling_rate_hz=sampling_rate_hz,
            results=sce_results,
            cell_count=cell_count,
        )

        summary = CellAnalysisSummary(
            place_configuration=resolved_configuration.place,
            reward_configuration=resolved_configuration.reward,
            sce_configuration=resolved_configuration.sce,
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
            mixture_weight=float(reward_results.mixture_weight),
            gaussian_mean_cm=float(reward_results.gaussian_mean),
            gaussian_std_cm=float(reward_results.gaussian_std),
            rest_period_count=rest_period_count,
            run_period_count=run_period_count,
            total_rest_sces=total_rest_sces,
            total_run_sces=total_run_sces,
        )

        return cls(table=table, sce_periods=sce_periods, summary=summary)

    @classmethod
    def load(cls, session: DatasetSession) -> CellAnalysisReport:
        """Loads a previously saved report from the session directory.

        Args:
            session: The DatasetSession whose directory holds the three artifacts.

        Returns:
            A CellAnalysisReport whose feathers are memory-mapped against the on-disk files.
        """
        summary: CellAnalysisSummary = CellAnalysisSummary.from_yaml(file_path=session.cell_analysis_summary_path)
        table = pl.read_ipc(source=session.cell_analysis_table_path, memory_map=True)
        sce_periods = pl.read_ipc(source=session.sce_periods_table_path, memory_map=True)
        return cls(table=table, sce_periods=sce_periods, summary=summary)

    def save(self, session: DatasetSession) -> None:
        """Persists the report to the three per-session artifacts inside the session directory.

        Args:
            session: The DatasetSession whose directory will hold the three artifacts.
        """
        self.summary.to_yaml(file_path=session.cell_analysis_summary_path)
        self.table.write_ipc(file=session.cell_analysis_table_path)
        self.sce_periods.write_ipc(file=session.sce_periods_table_path)

    def summarize(self) -> str:
        """Returns a multi-line human-readable summary of the report's per-cell and per-period statistics."""
        summary = self.summary
        cell_count = summary.cell_count
        spatially_pct = 100.0 * summary.spatially_significant_count / cell_count if cell_count > 0 else 0.0
        place_pct = 100.0 * summary.place_cell_count / cell_count if cell_count > 0 else 0.0
        reward_pct = 100.0 * summary.reward_cell_count / cell_count if cell_count > 0 else 0.0
        predictive_pct = 100.0 * summary.reward_predictive_count / cell_count if cell_count > 0 else 0.0

        lines = [
            "Cell analysis report",
            "====================",
            f"Cells: {cell_count}",
            f"  Spatially significant: {summary.spatially_significant_count} ({spatially_pct:.1f}%)",
            f"  Place cells:           {summary.place_cell_count} ({place_pct:.1f}%)",
            f"  Reward cells:          {summary.reward_cell_count} ({reward_pct:.1f}%)",
            f"  Reward-predictive:     {summary.reward_predictive_count} ({predictive_pct:.1f}%)",
            "",
            "Reward mixture model:",
            f"  Mixture weight:    {summary.mixture_weight:.3f}",
            f"  Gaussian mean:     {summary.gaussian_mean_cm:.1f} cm",
            f"  Gaussian std:      {summary.gaussian_std_cm:.1f} cm",
            f"  Reward position:   {summary.reward_position_cm:.1f} cm",
            "",
            "SCE detection:",
            f"  Rest periods: {summary.rest_period_count} ({summary.total_rest_sces} SCEs)",
            f"  Run periods:  {summary.run_period_count} ({summary.total_run_sces} SCEs)",
            "",
            "Geometry / sampling:",
            f"  Track length:      {summary.track_length_cm:.1f} cm",
            f"  Bin size:          {summary.bin_size_cm:.1f} cm ({summary.bin_count} bins)",
            f"  Sampling rate:     {summary.sampling_rate_hz:.2f} Hz",
        ]
        return "\n".join(lines)

    # ---- Plot methods -----------------------------------------------------------------------------------------

    def plot_place_cell_heatmap(
        self,
        *,
        title: str | None = None,
        sort_by_position: bool = True,
        show_only_place_cells: bool = True,
        figure_dpi: int = 150,
        minimum_percentile: float = 0.5,
        maximum_percentile: float = 0.9,
        cmap: str = "gray_r",
        show_color_bar: bool = True,
    ) -> plt.Figure:
        """Plots the position-ordered binned-fluorescence heatmap from the persisted per-cell rate maps.

        Args:
            title: Optional title displayed at the top of the figure.
            sort_by_position: Order cells by their place-field center along the track before plotting.
            show_only_place_cells: Display only cells whose ``is_place`` flag is True.
            figure_dpi: Figure resolution in dots per inch.
            minimum_percentile: Percentile used as the lower bound of the color scale.
            maximum_percentile: Percentile used as the upper bound of the color scale.
            cmap: Matplotlib colormap name.
            show_color_bar: Render a color bar alongside the heatmap.

        Returns:
            The matplotlib Figure containing the heatmap.
        """
        bin_size_cm = self.summary.bin_size_cm
        bin_count = self.summary.bin_count

        # Reconstructs the per-cell pooled rate map and per-cell place-field center used for ordering.
        rate_maps = _stack_list_column(table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=bin_count)
        # noinspection PyTypeChecker
        is_place: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_PLACE.value].to_numpy()
        order = _resolve_place_cell_order(table=self.table)

        if not sort_by_position:
            # noinspection PyTypeChecker
            order = np.arange(rate_maps.shape[0], dtype=np.int64)

        if show_only_place_cells:
            order = order[np.isin(order, np.flatnonzero(is_place))]

        sorted_data = rate_maps[order, :]
        if sorted_data.size == 0:
            sorted_data = rate_maps[:0, :]

        minimum_value = float(np.nanquantile(sorted_data, minimum_percentile)) if sorted_data.size > 0 else 0.0
        maximum_value = float(np.nanquantile(sorted_data, maximum_percentile)) if sorted_data.size > 0 else 1.0

        figure, axes = plt.subplots(1, 1, figsize=(8, 4), facecolor="white", dpi=figure_dpi)
        if title is not None:
            axes.set_title(title, fontsize=8)

        extent: tuple[float, float, float, float] = (
            0.0,
            float(bin_size_cm * bin_count),
            float(sorted_data.shape[0]),
            0.0,
        )
        image = axes.imshow(
            sorted_data,
            cmap=cmap,
            extent=extent,
            interpolation="none",
            vmin=minimum_value,
            vmax=maximum_value,
            origin="upper",
        )
        axes.set_aspect("auto")
        axes.set_xlabel("Position (cm)")
        axes.set_ylabel("Cell number")

        track_length_cm = self.summary.track_length_cm
        # noinspection PyTypeChecker
        x_ticks: NDArray[np.float64] = np.arange(0, track_length_cm + 1, _PLOT_TICK_INTERVAL_CM)
        axes.set_xticks(x_ticks)

        if show_color_bar:
            color_bar = figure.colorbar(image, ax=axes)
            color_bar.set_label("ΔF/F₀")
            cbar_min = np.floor(minimum_value / 0.5) * 0.5
            cbar_max = np.ceil(maximum_value / 0.5) * 0.5
            # noinspection PyTypeChecker
            cbar_ticks: NDArray[np.float64] = np.arange(cbar_min, cbar_max, 0.5)
            color_bar.set_ticks(cbar_ticks.tolist())

        return figure

    def plot_reward_com_histogram(
        self,
        *,
        bin_count: int = 20,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots the spatially significant COM histogram with the fitted uniform + Gaussian mixture overlay."""
        summary = self.summary
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )
        valid_centers = centers_of_mass[is_significant & (centers_of_mass >= 0.0)]

        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        # noinspection PyTypeChecker
        hist_bins: NDArray[np.float64] = np.linspace(0, summary.track_length_cm, bin_count + 1)
        axes.hist(
            valid_centers, bins=hist_bins.tolist(), color="0.7", edgecolor="0.5", density=True, label="Observed COMs"
        )

        # noinspection PyTypeChecker
        positions: NDArray[np.float64] = np.linspace(0, summary.track_length_cm, 200)
        # noinspection PyTypeChecker
        uniform_density: NDArray[np.float64] = np.full_like(positions, 1.0 / summary.track_length_cm)
        gaussian_std = max(summary.gaussian_std_cm, 1.0)
        gaussian_density = np.exp(-0.5 * ((positions - summary.gaussian_mean_cm) / gaussian_std) ** 2) / (
            gaussian_std * np.sqrt(2.0 * np.pi)
        )
        mixture_density = (1.0 - summary.mixture_weight) * uniform_density + summary.mixture_weight * gaussian_density

        axes.fill_between(
            positions,
            0,
            (1.0 - summary.mixture_weight) * uniform_density,
            alpha=0.3,
            color="lightblue",
            label="Uniform (place cells)",
        )
        axes.fill_between(
            positions,
            (1.0 - summary.mixture_weight) * uniform_density,
            mixture_density,
            alpha=0.4,
            color="mediumpurple",
            label="Gaussian (reward cells)",
        )
        axes.plot(positions, mixture_density, color="black", linewidth=1.5, label="Mixture fit")
        axes.axvline(
            x=summary.reward_position_cm,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Reward location",
        )

        axes.set_xlabel("Track Position (cm)")
        axes.set_ylabel("Density")
        axes.legend(fontsize=7, loc="upper left")
        annotation_text = (
            f"Significant: {summary.spatially_significant_count}/{summary.cell_count} cells\n"
            f"Mixture weight: {summary.mixture_weight:.1%} reward\n"
            f"Gaussian center: {summary.gaussian_mean_cm:.0f} cm (SD {summary.gaussian_std_cm:.0f} cm)"
        )
        axes.text(
            0.98,
            0.95,
            annotation_text,
            transform=axes.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )
        if title:
            axes.set_title(title, fontsize=9)
        figure.tight_layout()
        return figure

    def plot_rate_map_heatmap(
        self,
        *,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots row-normalized rate maps for reward cells and non-reward place cells side by side, sorted by COM."""
        summary = self.summary
        rate_maps = _stack_list_column(
            table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=summary.bin_count
        )
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward_proximal: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_PROXIMAL.value].to_numpy()
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )

        reward_mask = is_significant & is_reward_proximal
        place_mask = is_significant & ~is_reward_proximal

        reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
        reward_left = summary.reward_position_cm - reward_zone_half
        reward_right = summary.reward_position_cm + reward_zone_half

        figure, (axes_reward, axes_place) = plt.subplots(
            1, 2, figsize=(12, 6), facecolor="white", dpi=figure_dpi, sharey=False
        )

        for axes, mask, panel_title in [
            (axes_reward, reward_mask, "Reward Cells"),
            (axes_place, place_mask, "Place Cells"),
        ]:
            maps = rate_maps[mask]
            coms = centers_of_mass[mask]
            # noinspection PyTypeChecker
            sort_order: NDArray[np.int64] = np.argsort(coms)
            sorted_maps = maps[sort_order]

            row_maxima = sorted_maps.max(axis=1, keepdims=True)
            row_maxima[row_maxima == 0] = 1.0
            normalized_maps = sorted_maps / row_maxima

            extent = [0, summary.bin_size_cm * sorted_maps.shape[1], normalized_maps.shape[0], 0]
            axes.imshow(
                normalized_maps,
                cmap="gray_r",
                extent=extent,
                interpolation="none",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                aspect="auto",
            )
            axes.axvline(x=reward_left, color="red", linestyle="--", linewidth=1, alpha=0.7)
            axes.axvline(x=reward_right, color="red", linestyle="--", linewidth=1, alpha=0.7)
            axes.set_xlabel("Track Position (cm)")
            axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
            axes.set_title(f"{panel_title} (n={int(np.sum(mask))})", fontsize=9)

        axes_reward.set_ylabel("Neuron (sorted by COM)")
        if title:
            figure.suptitle(title, fontsize=9)
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            figure.tight_layout()
        return figure

    def plot_population_activity_by_position(
        self,
        *,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots mean population fluorescence vs track position, contrasting reward-predictive against all
        spatially modulated cells.
        """
        summary = self.summary
        rate_maps = _stack_list_column(
            table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=summary.bin_count
        )
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward_proximal: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_PROXIMAL.value].to_numpy()
        # noinspection PyTypeChecker
        is_slowing_correlated: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SLOWING_CORRELATED.value].to_numpy()

        bin_centers = (np.arange(summary.bin_count) + 0.5) * summary.bin_size_cm
        all_significant = is_significant
        reward_predictive_mask = all_significant & is_reward_proximal & is_slowing_correlated

        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        if int(np.sum(all_significant)) > 0:
            mean_all = np.mean(rate_maps[all_significant], axis=0)
            axes.fill_between(bin_centers, 0, mean_all, color="0.8", alpha=0.6)
            axes.plot(
                bin_centers,
                mean_all,
                color="0.5",
                linewidth=1.5,
                label=f"All spatially modulated (n={int(np.sum(all_significant))})",
            )

        if int(np.sum(reward_predictive_mask)) > 0:
            mean_predictive = np.mean(rate_maps[reward_predictive_mask], axis=0)
            axes.fill_between(bin_centers, 0, mean_predictive, color="mediumpurple", alpha=0.3)
            axes.plot(
                bin_centers,
                mean_predictive,
                color="darkviolet",
                linewidth=2.5,
                label=f"Slowing-correlated (n={int(np.sum(reward_predictive_mask))})",
            )

        axes.axvline(
            x=summary.reward_position_cm,
            color="red",
            linestyle="-",
            linewidth=2.0,
            alpha=0.8,
            label="Reward location",
        )
        axes.set_xlabel("Track Position (cm)")
        axes.set_ylabel("Average Fluorescence (dF/F)")
        axes.legend(fontsize=7, loc="upper left")
        axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
        if title:
            axes.set_title(title, fontsize=9)
        figure.tight_layout()
        return figure

    def plot_speed_and_activity_by_position(
        self,
        *,
        session: DatasetSession,
        title: str | None = None,
        figure_dpi: int = 150,
        position_sigma_cm: float = 5.0,
    ) -> plt.Figure:
        """Plots binned-speed alongside reward-predictive cell activity. Reads ``data.feather`` to bin speed by
        position; reward-predictive activity comes from the persisted rate maps.
        """
        summary = self.summary
        rate_maps = _stack_list_column(
            table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=summary.bin_count
        )
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward_proximal: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_PROXIMAL.value].to_numpy()
        # noinspection PyTypeChecker
        is_slowing_correlated: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SLOWING_CORRELATED.value].to_numpy()
        reward_predictive_mask = is_significant & is_reward_proximal & is_slowing_correlated

        if int(np.sum(reward_predictive_mask)) == 0:
            figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
            axes.text(
                0.5,
                0.5,
                "No reward-predictive cells found",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            return figure

        binned_speed = _bin_speed_by_position(
            session=session,
            track_length_cm=summary.track_length_cm,
            bin_size_cm=summary.bin_size_cm,
            bin_count=summary.bin_count,
        )
        sigma_bins = position_sigma_cm / summary.bin_size_cm
        # noinspection PyTypeChecker
        smoothed_speed: NDArray[np.float32] = gaussian_filter1d(input=binned_speed, sigma=sigma_bins, mode="wrap")

        bin_centers = (np.arange(summary.bin_count) + 0.5) * summary.bin_size_cm
        mean_activity = np.mean(rate_maps[reward_predictive_mask], axis=0)

        figure, axes_speed = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes_activity = axes_speed.twinx()
        axes_speed.plot(bin_centers, smoothed_speed, color="0.4", linewidth=1.5, label="Mean speed")
        axes_speed.set_xlabel("Track Position (cm)")
        axes_speed.set_ylabel("Speed (cm/s)", color="0.4")
        axes_speed.tick_params(axis="y", labelcolor="0.4")

        predictive_count = int(np.sum(reward_predictive_mask))
        axes_activity.plot(
            bin_centers,
            mean_activity,
            color="darkviolet",
            linewidth=2.0,
            label=f"Reward-predictive (n={predictive_count})",
        )
        axes_activity.set_ylabel("Mean Fluorescence (dF/F)", color="darkviolet")
        axes_activity.tick_params(axis="y", labelcolor="darkviolet")

        reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
        axes_speed.axvspan(
            summary.reward_position_cm - reward_zone_half,
            summary.reward_position_cm + reward_zone_half,
            alpha=0.1,
            color="red",
            label="Reward zone",
        )
        axes_speed.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
        lines_speed, labels_speed = axes_speed.get_legend_handles_labels()
        lines_activity, labels_activity = axes_activity.get_legend_handles_labels()
        axes_speed.legend(lines_speed + lines_activity, labels_speed + labels_activity, fontsize=7, loc="upper left")
        if title:
            axes_speed.set_title(title, fontsize=9)
        figure.tight_layout()
        return figure

    def plot_per_trial_activity(
        self,
        *,
        session: DatasetSession,
        trial_type: str = "ABC",
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
        title: str | None = None,
        figure_dpi: int = 150,
        position_bin_size_cm: float = 2.0,
        position_sigma_cm: float = 3.0,
        slowing_threshold_cm_s: float = 10.0,
    ) -> plt.Figure:
        """Plots per-trial activity heatmaps for an example reward-predictive cell and an example non-reward place
        cell, with slowing-onset markers overlaid. Reads ``data.feather`` for the raw fluorescence and per-trial
        speed time series.
        """
        summary = self.summary
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward_proximal: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_PROXIMAL.value].to_numpy()
        # noinspection PyTypeChecker
        is_slowing_correlated: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SLOWING_CORRELATED.value].to_numpy()
        # noinspection PyTypeChecker
        speed_correlations: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.SPEED_ACTIVITY_CORRELATION.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )

        predictive_mask = is_significant & is_reward_proximal & is_slowing_correlated
        # noinspection PyTypeChecker
        predictive_indices: NDArray[np.int64] = np.argwhere(predictive_mask).flatten()
        place_mask = is_significant & ~is_reward_proximal
        # noinspection PyTypeChecker
        place_indices: NDArray[np.int64] = np.argwhere(place_mask).flatten()

        if predictive_indices.size == 0 or place_indices.size == 0:
            figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
            axes.text(
                0.5,
                0.5,
                "Insufficient cells for comparison",
                transform=axes.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            return figure

        best_predictive = int(predictive_indices[np.argmin(speed_correlations[predictive_indices])])
        track_midpoint = summary.track_length_cm / 2.0
        place_distances = np.abs(centers_of_mass[place_indices] - track_midpoint)
        best_place = int(place_indices[np.argmin(place_distances)])

        # Loads the session's run-state arrays via the shared utility so position and speed match the
        # detection-time view exactly.
        run_session = assemble_run_session_data(
            session_path=session.session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )

        # noinspection PyTypeChecker
        unique_trials: NDArray[np.int64] = np.unique(run_session.trial_ids)
        trial_count = int(unique_trials.size)
        # noinspection PyTypeChecker
        bin_edges: NDArray[np.float32] = np.arange(
            0, summary.track_length_cm + position_bin_size_cm, position_bin_size_cm, dtype=np.float32
        )
        bin_count = len(bin_edges) - 1
        sigma_bins = position_sigma_cm / position_bin_size_cm

        reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
        reward_left = summary.reward_position_cm - reward_zone_half
        reward_right = summary.reward_position_cm + reward_zone_half
        pre_reward_start = summary.reward_position_cm - summary.reward_configuration.pre_reward_window

        figure, (axes_predictive, axes_place) = plt.subplots(1, 2, figsize=(14, 8), facecolor="white", dpi=figure_dpi)

        cells = [
            (axes_predictive, best_predictive, "Reward-predictive", "Purples"),
            (axes_place, best_place, "Place cell", "Blues"),
        ]
        for axes, cell_index, label, colormap in cells:
            # noinspection PyTypeChecker
            activity_image: NDArray[np.float32] = np.zeros((trial_count, bin_count), dtype=np.float32)
            # noinspection PyTypeChecker
            slowing_onsets: NDArray[np.float32] = np.full(trial_count, np.nan, dtype=np.float32)

            for trial_index, trial_id in enumerate(unique_trials):
                # noinspection PyTypeChecker
                trial_mask: NDArray[np.bool_] = run_session.trial_ids == trial_id
                trial_positions = run_session.position[trial_mask]
                trial_speeds = run_session.speed[trial_mask]
                trial_fluorescence = run_session.fluorescence[cell_index, trial_mask]

                # noinspection PyTypeChecker
                trial_bin_indices: NDArray[np.int64] = np.clip(
                    np.searchsorted(bin_edges, trial_positions, side="right") - 1, 0, bin_count - 1
                )
                # noinspection PyTypeChecker
                activity_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
                # noinspection PyTypeChecker
                activity_counts: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)
                np.add.at(activity_sums, trial_bin_indices, trial_fluorescence)
                np.add.at(activity_counts, trial_bin_indices, 1)
                # noinspection PyTypeChecker
                valid_bins: NDArray[np.bool_] = activity_counts > 0
                activity_image[trial_index, valid_bins] = activity_sums[valid_bins] / activity_counts[valid_bins]

                # noinspection PyTypeChecker
                pre_reward: NDArray[np.bool_] = (trial_positions >= pre_reward_start) & (
                    trial_positions < summary.reward_position_cm
                )
                # noinspection PyTypeChecker
                below_threshold: NDArray[np.bool_] = pre_reward & (trial_speeds < slowing_threshold_cm_s)
                if np.any(below_threshold):
                    slowing_onsets[trial_index] = trial_positions[below_threshold][0]

            activity_image = gaussian_filter1d(input=activity_image, sigma=sigma_bins, axis=1, mode="wrap")
            row_maxima = activity_image.max(axis=1, keepdims=True)
            row_maxima[row_maxima == 0] = 1.0
            normalized_activity = activity_image / row_maxima

            axes.imshow(
                normalized_activity,
                cmap=colormap,
                extent=[0, summary.track_length_cm, trial_count, 0],
                interpolation="none",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                aspect="auto",
                alpha=0.85,
            )

            # noinspection PyTypeChecker
            onset_trials: NDArray[np.int64] = np.argwhere(~np.isnan(slowing_onsets)).flatten()
            for marker_index, trial_index in enumerate(onset_trials):
                onset_label = f"Slowing onset (<{slowing_threshold_cm_s:.0f} cm/s)" if marker_index == 0 else None
                axes.plot(
                    slowing_onsets[trial_index],
                    trial_index + 0.5,
                    marker="|",
                    color="black",
                    markersize=6,
                    markeredgewidth=1.5,
                    label=onset_label,
                )

            axes.axvline(x=reward_left, color="red", linestyle="--", linewidth=1, alpha=0.7)
            axes.axvline(x=reward_right, color="red", linestyle="--", linewidth=1, alpha=0.7)
            axes.legend(fontsize=6, loc="upper left")
            axes.set_xlabel("Track Position (cm)")
            axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
            cell_com = centers_of_mass[cell_index]
            cell_corr = speed_correlations[cell_index]
            axes.set_title(f"{label} (cell {cell_index}, COM={cell_com:.0f} cm, r={cell_corr:.2f})", fontsize=9)

        axes_predictive.set_ylabel("Trial")
        if title:
            figure.suptitle(title, fontsize=9)
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            figure.tight_layout()
        return figure

    def plot_sce_rest_run_rest_sequence(
        self,
        *,
        session: DatasetSession,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
        cell_indices: NDArray[np.int32] | list[int] | None = None,
        cell_count: int = 5,
        trial_index: int = 0,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots smoothed fluorescence traces, onsets, and co-active counts for selected cells across one
        rest-run-rest sequence. Reads ``data.feather`` to recompute Savitzky-Golay smoothing on the period slice.
        """
        rest_periods = _filter_period_rows(table=self.sce_periods, period_type=PeriodType.REST)
        run_periods = _filter_period_rows(table=self.sce_periods, period_type=PeriodType.RUN)

        if len(run_periods) < _MINIMUM_PERIODS_FOR_REST_RUN_REST:
            figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
            axes.text(0.5, 0.5, "No run periods detected", transform=axes.transAxes, ha="center", va="center")
            return figure

        if trial_index >= len(run_periods):
            trial_index = 0

        sequence_rows: list[int] = []
        if trial_index < len(rest_periods):
            sequence_rows.append(rest_periods[trial_index])
        sequence_rows.append(run_periods[trial_index])
        if trial_index + 1 < len(rest_periods):
            sequence_rows.append(rest_periods[trial_index + 1])

        # Reloads raw fluorescence and re-applies Savitzky-Golay smoothing for each period slice.
        sequence_data = [
            _reconstruct_period_state(
                session=session,
                fluorescence_column=fluorescence_column,
                sce_configuration=self.summary.sce_configuration,
                period_row=self.sce_periods.row(row_index, named=True),
            )
            for row_index in sequence_rows
        ]

        if cell_indices is None:
            # Picks a mix of rest-active and rest-quiet cells that are also active during run, falling back to
            # the cells with the most onsets across the sequence when run activity is empty.
            picked = _select_rest_run_cells(
                sequence_data=sequence_data,
                cell_count=cell_count,
                summary_cell_count=self.summary.cell_count,
            )
            # noinspection PyTypeChecker
            cell_indices = picked
        else:
            cell_indices = np.asarray(cell_indices, dtype=np.int32)

        period_boundaries = []
        time_offset = 0.0
        for state in sequence_data:
            period_duration = float(state.timestamps[-1] - state.timestamps[0])
            period_boundaries.append((time_offset, time_offset + period_duration, state.period_type))
            time_offset += period_duration
        total_time = time_offset

        height_ratios = [0.4] + [1.0] * len(cell_indices)
        figure, all_axes = plt.subplots(
            nrows=1 + len(cell_indices),
            ncols=1,
            figsize=(14, 1.5 * len(cell_indices) + 1),
            facecolor="white",
            dpi=figure_dpi,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )

        label_axis = all_axes[0]
        period_colors = {PeriodType.REST: "#A8D8EA", PeriodType.RUN: "#FFE0A0"}
        rest_label_count = 0
        for start, end, period_type in period_boundaries:
            face_color = period_colors[period_type]
            if period_type == PeriodType.REST:
                rest_label_count += 1
                display_label = f"Rest {rest_label_count}"
            else:
                display_label = "Run"
            label_axis.axvspan(xmin=start, xmax=end, color=face_color, alpha=0.8)
            label_axis.text(
                x=(start + end) / 2,
                y=0.5,
                s=display_label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        label_axis.set_xlim(0, total_time)
        label_axis.set_ylim(0, 1)
        label_axis.set_yticks([])
        for spine_name in ("top", "right", "left", "bottom"):
            label_axis.spines[spine_name].set_visible(False)
        label_axis.set_title(f"Rest-Run-Rest Sequence (Trial {trial_index + 1})", fontsize=11)

        trace_axes = all_axes[1:]
        for axis_index, cell_index in enumerate(cell_indices):
            axis = trace_axes[axis_index]
            current_offset = 0.0
            for state in sequence_data:
                period_time = state.timestamps - state.timestamps[0] + current_offset
                fluorescence_trace = state.smoothed_fluorescence[cell_index]
                axis.axvspan(
                    xmin=period_time[0], xmax=period_time[-1], alpha=0.15, color=period_colors[state.period_type]
                )
                trace_mean = float(np.mean(fluorescence_trace))
                trace_std = float(np.std(fluorescence_trace))
                if trace_std > 0:
                    normalized_trace = (fluorescence_trace - trace_mean) / trace_std
                else:
                    normalized_trace = fluorescence_trace - trace_mean
                axis.plot(period_time, normalized_trace, color="black", linewidth=0.5, alpha=0.8)
                current_offset = period_time[-1]
            axis.set_ylabel(f"Cell {int(cell_index)}", fontsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        trace_axes[-1].set_xlabel("Time (minutes)")
        figure.tight_layout()
        return figure

    def plot_sce_assemblies(
        self,
        *,
        period_type: PeriodType = PeriodType.REST,
        period_index: int = 0,
        top_n: int = 5,
        max_clusters: int = 15,
        activation_threshold: float = 0.3,
        minimum_assembly_size: int = 3,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Detects and plots the most frequent SCE cell assemblies as raster panels. The clustering is recomputed
        from the persisted onset matrix on every call so no live detector is required.
        """
        period_rows = _filter_period_rows(table=self.sce_periods, period_type=period_type)
        if period_index >= len(period_rows):
            figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
            axes.text(
                0.5,
                0.5,
                f"Period {period_index + 1} not found for {period_type.value}",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes.axis("off")
            return figure

        row_index = period_rows[period_index]
        row = self.sce_periods.row(row_index, named=True)
        sce_labels = np.asarray(row[SCEPeriodColumn.SCE_LABELS.value], dtype=np.int32)
        onset_matrix = _reconstruct_onset_matrix(
            cell_count=int(row[SCEPeriodColumn.CELL_COUNT.value]),
            sample_count=int(row[SCEPeriodColumn.SAMPLE_COUNT.value]),
            onset_cell_indices=row[SCEPeriodColumn.ONSET_CELL_INDICES.value],
            onset_sample_indices=row[SCEPeriodColumn.ONSET_SAMPLE_INDICES.value],
        )
        total_sce_count = int(np.max(sce_labels)) if sce_labels.size > 0 else 0

        assemblies = _detect_assemblies_from_state(
            sce_labels=sce_labels,
            onset_matrix=onset_matrix,
            max_clusters=max_clusters,
            activation_threshold=activation_threshold,
            minimum_assembly_size=minimum_assembly_size,
        )
        if not assemblies:
            figure, axis = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
            axis.text(0.5, 0.5, "No assemblies detected", ha="center", va="center", fontsize=12)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.axis("off")
            return figure

        display_count = min(top_n, len(assemblies))
        displayed = assemblies[:display_count]
        total_cells = onset_matrix.shape[0]

        column_width = 1.8
        figure_width = column_width * display_count + 1.5
        figure_height = max(5, min(12, total_cells * 0.003 + 2))
        figure, axes_array = plt.subplots(
            nrows=1, ncols=display_count, figsize=(figure_width, figure_height), facecolor="white", dpi=figure_dpi
        )
        axes_list = [axes_array] if display_count == 1 else list(axes_array)

        assembly_colors = ["black", "red", "blue", "green", "magenta"]
        last_index = 0
        for assembly_index, member_cells in enumerate(displayed):
            axis = axes_list[assembly_index]
            color = assembly_colors[assembly_index % len(assembly_colors)]
            sorted_members = np.sort(member_cells)
            axis.scatter(
                x=np.zeros(len(sorted_members)),
                y=sorted_members,
                color=color,
                s=13,
                marker=".",
                linewidths=0,
            )
            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(total_cells - 0.5, -0.5)
            axis.set_xticks([])
            yticks = list(range(0, total_cells, 500))
            axis.set_yticks(yticks)
            axis.set_yticklabels([str(t) for t in yticks], fontsize=6)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color(color)
            axis.set_xlabel(f"Assembly {assembly_index + 1}", fontsize=7, color=color)
            last_index = assembly_index

        if last_index > 0:
            axes_list[last_index].tick_params(axis="y", labelleft=False)
        axes_list[0].set_ylabel("Cell number")

        if title is None:
            title = (
                f"SCE Assemblies — {period_type.value.title()} Period {period_index + 1}  "
                f"({total_sce_count} total SCEs, showing top {display_count} assemblies)"
            )
        figure.suptitle(title, fontsize=11)
        figure.subplots_adjust(wspace=0.1, left=0.06, right=0.98, top=0.94, bottom=0.06)
        return figure


# ===== Module-level helpers ====================================================================================


def evaluate_and_save_cell_analysis(
    session: DatasetSession,
    *,
    trial_type: str = "ABC",
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
    configuration: CellAnalysisConfiguration | None = None,
) -> CellAnalysisReport:
    """Evaluates the cell analysis pipeline for a single session and persists the report to disk.

    Args:
        session: The DatasetSession to analyze.
        trial_type: Trial type to analyze.
        fluorescence_column: Fluorescence column to use as the analysis input.
        configuration: Wrapper holding the three sub-pipeline configurations. Uses defaults if None.

    Returns:
        The CellAnalysisReport produced for the session, with all three artifacts persisted under the session
        directory.
    """
    report = CellAnalysisReport.evaluate(
        session_path=session.session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
        configuration=configuration,
    )
    report.save(session=session)
    return report


def plot_dataset_place_cell_fraction(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal place-cell fraction trend overlaid for every animal, with the across-animal median +
    IQR rendered on top. Animals without saved reports are silently skipped.
    """
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: summary.place_cell_count / summary.cell_count if summary.cell_count else 0.0,
        y_label="Place cell fraction",
        figure_title="Across-animal place cell fraction",
    )


def plot_dataset_reward_cell_fraction(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal reward-cell fraction trend; same aggregation pattern as the place-cell variant."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: summary.reward_cell_count / summary.cell_count if summary.cell_count else 0.0,
        y_label="Reward cell fraction",
        figure_title="Across-animal reward cell fraction",
    )


def plot_dataset_sce_rate(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal total SCE count (rest + run) trend; same aggregation pattern."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: float(summary.total_rest_sces + summary.total_run_sces),
        y_label="Total SCEs (rest + run)",
        figure_title="Across-animal SCE counts",
    )


def plot_dataset_cell_count(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal cell-count trend; useful for spotting registration drift across days."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: float(summary.cell_count),
        y_label="Cell count",
        figure_title="Across-animal cell counts",
    )


# ===== Private helpers ==========================================================================================


@dataclass(frozen=True, slots=True)
class _ReconstructedPeriod:
    """Plot-time view of a reconstructed SCE period with smoothed fluorescence."""

    period_type: PeriodType
    """Period kind (REST or RUN)."""
    timestamps: NDArray[np.float32]
    """Per-sample elapsed-minutes timestamps."""
    smoothed_fluorescence: NDArray[np.float32]
    """Savitzky-Golay smoothed cell x sample fluorescence trace."""


def _build_cell_table(
    cell_count: int,
    place_fields: PlaceFields,
    reward_results: RewardCellResults,
    sce_results: list,
) -> pl.DataFrame:
    """Assembles the per-cell wide-format DataFrame from the live detector outputs."""
    spatial = reward_results.spatial_results

    place_rows = _build_place_field_rows(cell_count=cell_count, place_fields=place_fields)
    sce_columns = _aggregate_sce_columns(cell_count=cell_count, sce_results=sce_results)

    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)
    is_reward_cell = spatial.is_significant & reward_results.is_reward_proximal

    # Persists the per-trial binned fluorescence for any cell that is either a place cell or spatially significant
    # so reward-cell plotting paths can use the same column without an extra rebinning pass.
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
            CellAnalysisColumn.CELL_ID.value: cell_ids,
            CellAnalysisColumn.IS_PLACE.value: pl.Series(values=place_fields.has_place_field, dtype=pl.Boolean),
            CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value: pl.Series(
                values=spatial.is_significant, dtype=pl.Boolean
            ),
            CellAnalysisColumn.IS_REWARD_PROXIMAL.value: pl.Series(
                values=reward_results.is_reward_proximal, dtype=pl.Boolean
            ),
            CellAnalysisColumn.IS_REWARD_CELL.value: pl.Series(values=is_reward_cell, dtype=pl.Boolean),
            CellAnalysisColumn.IS_SLOWING_CORRELATED.value: pl.Series(
                values=reward_results.is_slowing_correlated, dtype=pl.Boolean
            ),
            CellAnalysisColumn.PF_START_CM.value: pl.Series(
                values=place_rows["pf_start_cm"], dtype=pl.List(pl.Float32)
            ),
            CellAnalysisColumn.PF_END_CM.value: pl.Series(values=place_rows["pf_end_cm"], dtype=pl.List(pl.Float32)),
            CellAnalysisColumn.PF_CENTER_CM.value: pl.Series(
                values=place_rows["pf_center_cm"], dtype=pl.List(pl.Float32)
            ),
            CellAnalysisColumn.PF_MEAN_INTENSITY.value: pl.Series(
                values=place_rows["pf_mean_intensity"], dtype=pl.List(pl.Float32)
            ),
            CellAnalysisColumn.PF_MAX_INTENSITY.value: pl.Series(
                values=place_rows["pf_max_intensity"], dtype=pl.List(pl.Float32)
            ),
            CellAnalysisColumn.PF_WIDTH_CM.value: pl.Series(
                values=place_rows["pf_width_cm"], dtype=pl.List(pl.Float32)
            ),
            CellAnalysisColumn.BINNED_FLUORESCENCE_PER_TRIAL.value: pl.Series(
                name=CellAnalysisColumn.BINNED_FLUORESCENCE_PER_TRIAL.value,
                values=binned_fluorescence_per_trial,
                dtype=pl.List(pl.List(pl.Float32)),
            ),
            CellAnalysisColumn.RATE_MAP.value: pl.Series(values=rate_maps_list, dtype=pl.List(pl.Float32)),
            CellAnalysisColumn.CENTER_OF_MASS_CM.value: pl.Series(values=spatial.centers_of_mass, dtype=pl.Float32),
            CellAnalysisColumn.SPATIAL_INFORMATION_BITS.value: pl.Series(
                values=spatial.spatial_information, dtype=pl.Float32
            ),
            CellAnalysisColumn.SPATIAL_P_VALUE.value: pl.Series(values=spatial.p_values, dtype=pl.Float32),
            CellAnalysisColumn.SPEED_ACTIVITY_CORRELATION.value: pl.Series(
                values=reward_results.speed_activity_correlations, dtype=pl.Float32
            ),
            CellAnalysisColumn.SCE_PARTICIPATION_COUNT_REST.value: sce_columns["participation_count_rest"],
            CellAnalysisColumn.SCE_PARTICIPATION_COUNT_RUN.value: sce_columns["participation_count_run"],
            CellAnalysisColumn.SCE_PARTICIPATION_RATE_REST.value: sce_columns["participation_rate_rest"],
            CellAnalysisColumn.SCE_PARTICIPATION_RATE_RUN.value: sce_columns["participation_rate_run"],
            CellAnalysisColumn.SCE_MEAN_ONSET_RANK_REST.value: sce_columns["mean_onset_rank_rest"],
            CellAnalysisColumn.SCE_MEAN_ONSET_RANK_RUN.value: sce_columns["mean_onset_rank_run"],
            CellAnalysisColumn.SCE_EVENTS_REST.value: pl.Series(
                name=CellAnalysisColumn.SCE_EVENTS_REST.value,
                values=sce_columns["sce_events_rest"],
                dtype=pl.List(pl.List(pl.Int32)),
            ),
            CellAnalysisColumn.SCE_EVENTS_RUN.value: pl.Series(
                name=CellAnalysisColumn.SCE_EVENTS_RUN.value,
                values=sce_columns["sce_events_run"],
                dtype=pl.List(pl.List(pl.Int32)),
            ),
        },
    ).sort(CellAnalysisColumn.CELL_ID.value)


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


def _aggregate_sce_columns(cell_count: int, sce_results: list) -> dict:
    """Computes per-cell SCE participation and timing metrics across all detected periods."""
    # noinspection PyTypeChecker
    participation: NDArray[np.int32] = np.zeros((2, cell_count), dtype=np.int32)
    # noinspection PyTypeChecker
    rank_sum: NDArray[np.float32] = np.zeros((2, cell_count), dtype=np.float32)
    # noinspection PyTypeChecker
    rank_count: NDArray[np.int32] = np.zeros((2, cell_count), dtype=np.int32)
    # noinspection PyTypeChecker
    total_sces: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    # noinspection PyTypeChecker
    period_counter: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    sce_events: list[list[list[list[int]]]] = [[[] for _ in range(cell_count)] for _ in range(2)]

    for result in sce_results:
        period = 0 if result.period_type == PeriodType.REST else 1
        period_index = int(period_counter[period])
        period_counter[period] += 1
        sce_count = int(np.max(result.sce_labels))
        total_sces[period] += sce_count
        if sce_count == 0:
            continue

        # noinspection PyTypeChecker
        sce_sample_indices: NDArray[np.int64] = np.where(result.sce_labels > 0)[0]
        # noinspection PyTypeChecker
        sample_to_sce: NDArray[np.float32] = np.zeros((result.onset_matrix.shape[1], sce_count), dtype=np.float32)
        sample_to_sce[sce_sample_indices, result.sce_labels[sce_sample_indices] - 1] = 1.0
        # noinspection PyTypeChecker
        cell_sce_participation: NDArray[np.bool_] = (result.onset_matrix.astype(np.float32) @ sample_to_sce) > 0
        participation[period] += cell_sce_participation.sum(axis=1).astype(np.int32)

        for sce_label in range(1, sce_count + 1):
            # noinspection PyTypeChecker
            participating_indices: NDArray[np.int64] = np.where(cell_sce_participation[:, sce_label - 1])[0]
            participant_count = participating_indices.size
            for cell in participating_indices:
                sce_events[period][cell].append([period_index, sce_label])
            if participant_count > 1:
                # noinspection PyTypeChecker
                sce_samples: NDArray[np.int64] = np.where(result.sce_labels == sce_label)[0]
                onset_window = result.onset_matrix[participating_indices][:, sce_samples]
                first_onset = np.argmax(onset_window, axis=1)
                # noinspection PyTypeChecker
                normalized_ranks: NDArray[np.float32] = (
                    np.argsort(np.argsort(first_onset)).astype(np.float32) / (participant_count - 1)
                ).astype(np.float32)
                rank_sum[period, participating_indices] += normalized_ranks
                rank_count[period, participating_indices] += 1
            elif participant_count == 1:
                rank_sum[period, participating_indices] += 0.5
                rank_count[period, participating_indices] += 1

    # noinspection PyTypeChecker
    rate: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    mean_rank: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    for period in range(2):
        if total_sces[period] > 0:
            rate[period] = (participation[period] / total_sces[period]).astype(np.float32)
        # noinspection PyTypeChecker
        has_ranks: NDArray[np.bool_] = rank_count[period] > 0
        mean_rank[period, has_ranks] = rank_sum[period, has_ranks] / rank_count[period, has_ranks]

    return {
        "participation_count_rest": participation[0],
        "participation_count_run": participation[1],
        "participation_rate_rest": rate[0],
        "participation_rate_run": rate[1],
        "mean_onset_rank_rest": mean_rank[0],
        "mean_onset_rank_run": mean_rank[1],
        "sce_events_rest": sce_events[0],
        "sce_events_run": sce_events[1],
    }


def _build_sce_periods_table(
    sampling_rate_hz: float,
    results: list,
    cell_count: int,
) -> pl.DataFrame:
    """Assembles the per-period SCE feather, encoding the dense onset matrix as sparse cell/sample index lists."""
    if not results:
        # Empty schema-conforming dataframe so downstream readers do not need to special-case missing files.
        return pl.DataFrame(
            schema={
                SCEPeriodColumn.PERIOD_TYPE.value: pl.Utf8,
                SCEPeriodColumn.PERIOD_INDEX.value: pl.Int32,
                SCEPeriodColumn.CELL_COUNT.value: pl.Int32,
                SCEPeriodColumn.SAMPLE_COUNT.value: pl.Int32,
                SCEPeriodColumn.SAMPLING_RATE_HZ.value: pl.Float32,
                SCEPeriodColumn.THRESHOLD.value: pl.Float32,
                SCEPeriodColumn.TIMESTAMPS_MINUTES.value: pl.List(pl.Float32),
                SCEPeriodColumn.COACTIVE_COUNTS.value: pl.List(pl.Int32),
                SCEPeriodColumn.SCE_LABELS.value: pl.List(pl.Int32),
                SCEPeriodColumn.ONSET_CELL_INDICES.value: pl.List(pl.Int32),
                SCEPeriodColumn.ONSET_SAMPLE_INDICES.value: pl.List(pl.Int32),
            }
        )

    period_type_column: list[str] = []
    period_index_column: list[int] = []
    cell_count_column: list[int] = []
    sample_count_column: list[int] = []
    sampling_rate_column: list[float] = []
    threshold_column: list[float] = []
    timestamps_column: list[list[float]] = []
    coactive_counts_column: list[list[int]] = []
    sce_labels_column: list[list[int]] = []
    onset_cell_indices_column: list[list[int]] = []
    onset_sample_indices_column: list[list[int]] = []

    rest_counter = 0
    run_counter = 0

    for result in results:
        if result.period_type == PeriodType.REST:
            period_index = rest_counter
            rest_counter += 1
        else:
            period_index = run_counter
            run_counter += 1

        onset_cells, onset_samples = np.nonzero(result.onset_matrix)
        period_type_column.append(result.period_type.value)
        period_index_column.append(period_index)
        cell_count_column.append(cell_count)
        sample_count_column.append(int(result.onset_matrix.shape[1]))
        sampling_rate_column.append(sampling_rate_hz)
        threshold_column.append(float(result.threshold))
        timestamps_column.append([float(value) for value in result.timestamps.tolist()])
        coactive_counts_column.append([int(value) for value in result.coactive_counts.tolist()])
        sce_labels_column.append([int(value) for value in result.sce_labels.tolist()])
        onset_cell_indices_column.append([int(value) for value in onset_cells.tolist()])
        onset_sample_indices_column.append([int(value) for value in onset_samples.tolist()])

    return pl.DataFrame(
        {
            SCEPeriodColumn.PERIOD_TYPE.value: pl.Series(values=period_type_column, dtype=pl.Utf8),
            SCEPeriodColumn.PERIOD_INDEX.value: pl.Series(values=period_index_column, dtype=pl.Int32),
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
        }
    )


def _stack_list_column(table: pl.DataFrame, column: CellAnalysisColumn, target_length: int) -> NDArray[np.float32]:
    """Materializes a List(Float32) column into a (cell_count, target_length) numpy array, padding with NaN rows
    for nulls.
    """
    cell_count = table.height
    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.full((cell_count, target_length), np.nan, dtype=np.float32)
    values = table[column.value].to_list()
    for cell_index, vector in enumerate(values):
        if vector is None:
            continue
        as_array = np.asarray(vector, dtype=np.float32)
        clipped_length = min(target_length, as_array.size)
        output[cell_index, :clipped_length] = as_array[:clipped_length]
    return output


def _resolve_place_cell_order(table: pl.DataFrame) -> NDArray[np.int64]:
    """Returns a length-cell_count permutation that sorts cells by their place-field center along the track,
    placing cells without place fields after the sorted block.
    """
    cell_count = table.height
    # noinspection PyTypeChecker
    sort_keys: NDArray[np.float32] = np.full(cell_count, np.inf, dtype=np.float32)

    pf_centers = table[CellAnalysisColumn.PF_CENTER_CM.value].to_list()
    pf_intensities = table[CellAnalysisColumn.PF_MEAN_INTENSITY.value].to_list()
    for cell_index in range(cell_count):
        centers = pf_centers[cell_index]
        intensities = pf_intensities[cell_index]
        if not centers:
            continue
        # Picks the center whose mean intensity is highest for cells with multiple fields.
        intensity_array = np.asarray(intensities, dtype=np.float32)
        sort_keys[cell_index] = float(centers[int(np.argmax(intensity_array))])
    # noinspection PyTypeChecker
    return np.argsort(sort_keys, kind="stable").astype(np.int64)


def _filter_period_rows(table: pl.DataFrame, period_type: PeriodType) -> list[int]:
    """Returns the row indices in ``sce_periods`` that belong to the given period type, ordered by period_index."""
    if table.height == 0:
        return []
    period_types = table[SCEPeriodColumn.PERIOD_TYPE.value].to_list()
    period_indices = table[SCEPeriodColumn.PERIOD_INDEX.value].to_list()
    matched = [
        (period_indices[row_index], row_index)
        for row_index in range(table.height)
        if period_types[row_index] == period_type.value
    ]
    matched.sort()
    return [row_index for _, row_index in matched]


def _bin_speed_by_position(
    session: DatasetSession,
    track_length_cm: float,
    bin_size_cm: float,
    bin_count: int,
) -> NDArray[np.float32]:
    """Returns the per-bin mean running speed for the session, computed off ``data.feather``."""
    df = pl.read_ipc(
        source=session.data_path,
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.DISTANCE_CM.value, DatasetColumn.SPEED_CM_S.value],
        memory_map=True,
    )
    df = trim_acquisition_warmup(df)

    # noinspection PyTypeChecker
    distance: NDArray[np.float32] = df[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    speed: NDArray[np.float32] = df[DatasetColumn.SPEED_CM_S.value].to_numpy().astype(np.float32, copy=False)

    # Maps cumulative distance into a within-track position by modulo on track_length so that speed bins regardless
    # of which trial each sample belongs to. Avoids requiring trial information for this lightweight binning.
    # noinspection PyTypeChecker
    position: NDArray[np.float32] = (distance % np.float32(track_length_cm)).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    bin_edges: NDArray[np.float32] = np.arange(0.0, track_length_cm + bin_size_cm, bin_size_cm, dtype=np.float32)
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int64] = np.clip(np.searchsorted(bin_edges, position, side="right") - 1, 0, bin_count - 1)
    # noinspection PyTypeChecker
    speed_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
    # noinspection PyTypeChecker
    sample_counts: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)
    np.add.at(speed_sums, bin_indices, speed)
    np.add.at(sample_counts, bin_indices, 1)
    # noinspection PyTypeChecker
    mean_speed: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
    # noinspection PyTypeChecker
    valid: NDArray[np.bool_] = sample_counts > 0
    mean_speed[valid] = speed_sums[valid] / sample_counts[valid]
    return mean_speed


def _reconstruct_period_state(
    session: DatasetSession,
    fluorescence_column: FluorescenceColumn,
    sce_configuration: SCEDetectionConfiguration,
    period_row: dict,
) -> _ReconstructedPeriod:
    """Rebuilds a smoothed-fluorescence view for one persisted SCE period by reloading ``data.feather`` and
    re-applying Savitzky-Golay smoothing matching the persisted sce_configuration parameters.
    """
    timestamps = np.asarray(period_row[SCEPeriodColumn.TIMESTAMPS_MINUTES.value], dtype=np.float32)
    sampling_rate_hz = float(period_row[SCEPeriodColumn.SAMPLING_RATE_HZ.value])

    df = pl.read_ipc(
        source=session.data_path,
        columns=[DatasetColumn.TIME_US.value, fluorescence_column.value],
        memory_map=True,
    )
    df = trim_acquisition_warmup(df)
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy()
    elapsed_minutes = (time_us - time_us[0]).astype(np.float32) / np.float32(60_000_000.0)

    # Locates the trimmed-row index for each persisted timestamp via searchsorted; ties to the closest match per
    # sample so unstable rest samples that were dropped during detection are excluded from the reload too.
    # noinspection PyTypeChecker
    sample_indices: NDArray[np.int64] = np.searchsorted(elapsed_minutes, timestamps, side="left")
    sample_indices = np.clip(sample_indices, 0, elapsed_minutes.size - 1)

    # noinspection PyTypeChecker
    fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T[
        :, sample_indices
    ]

    smoothing_window_samples = int(sce_configuration.smoothing_window_seconds * sampling_rate_hz)
    if smoothing_window_samples % 2 == 0:
        smoothing_window_samples += 1
    smoothing_window_samples = max(smoothing_window_samples, sce_configuration.smoothing_order + 2)
    smoothing_window_samples = min(smoothing_window_samples, fluorescence.shape[1])
    if smoothing_window_samples <= sce_configuration.smoothing_order:
        # noinspection PyTypeChecker
        smoothed: NDArray[np.float32] = fluorescence.astype(np.float32, copy=False)
    else:
        # noinspection PyTypeChecker
        smoothed = savgol_filter(
            x=fluorescence,
            window_length=smoothing_window_samples,
            polyorder=sce_configuration.smoothing_order,
            axis=1,
        ).astype(np.float32, copy=False)

    return _ReconstructedPeriod(
        period_type=PeriodType(period_row[SCEPeriodColumn.PERIOD_TYPE.value]),
        timestamps=timestamps,
        smoothed_fluorescence=smoothed,
    )


def _reconstruct_onset_matrix(
    cell_count: int,
    sample_count: int,
    onset_cell_indices: list[int] | None,
    onset_sample_indices: list[int] | None,
) -> NDArray[np.bool_]:
    """Materializes the dense (cell_count, sample_count) bool onset matrix from the persisted sparse encoding."""
    # noinspection PyTypeChecker
    matrix: NDArray[np.bool_] = np.zeros((cell_count, sample_count), dtype=np.bool_)
    if onset_cell_indices is None or onset_sample_indices is None:
        return matrix
    cells = np.asarray(onset_cell_indices, dtype=np.int64)
    samples = np.asarray(onset_sample_indices, dtype=np.int64)
    if cells.size != samples.size or cells.size == 0:
        return matrix
    matrix[cells, samples] = True
    return matrix


def _select_rest_run_cells(
    sequence_data: list,
    cell_count: int,
    summary_cell_count: int,
) -> NDArray[np.int32]:
    """Picks a mix of rest-active and rest-quiet cells that are also active during run for the trace plot."""
    # noinspection PyTypeChecker
    rest_onsets: NDArray[np.int32] = np.zeros(summary_cell_count, dtype=np.int32)
    # noinspection PyTypeChecker
    run_onsets: NDArray[np.int32] = np.zeros(summary_cell_count, dtype=np.int32)
    # The sequence_data list does not currently carry the dense onset matrix; using non-zero smoothed-fluorescence
    # variance per cell as a proxy for activity within each period.
    for state in sequence_data:
        per_cell_variance = np.std(state.smoothed_fluorescence, axis=1)
        # noinspection PyTypeChecker
        active_cells: NDArray[np.bool_] = per_cell_variance > 0
        if state.period_type == PeriodType.REST:
            rest_onsets[active_cells] += 1
        else:
            run_onsets[active_cells] += 1

    # noinspection PyTypeChecker
    run_active_indices: NDArray[np.int64] = np.where(run_onsets > 0)[0]
    if run_active_indices.size == 0:
        # noinspection PyTypeChecker
        run_active_indices = np.arange(summary_cell_count, dtype=np.int64)

    rest_spikers = cell_count // 2
    rest_calm = cell_count - rest_spikers
    rest_subset = rest_onsets[run_active_indices]
    # noinspection PyTypeChecker
    sorted_by_rest: NDArray[np.int64] = np.argsort(rest_subset)
    calm_indices = run_active_indices[sorted_by_rest[:rest_calm]]
    spiker_indices = run_active_indices[sorted_by_rest[-rest_spikers:]]
    # noinspection PyTypeChecker
    selected: NDArray[np.int32] = np.unique(np.concatenate([calm_indices, spiker_indices])).astype(np.int32)
    return selected[:cell_count]


def _detect_assemblies_from_state(
    sce_labels: NDArray[np.int32],
    onset_matrix: NDArray[np.bool_],
    max_clusters: int,
    activation_threshold: float,
    minimum_assembly_size: int,
) -> list[NDArray[np.int32]]:
    """Reproduces the live SCEDetector cell-assembly clustering from the persisted onset matrix and SCE labels."""
    total_sce_count = int(np.max(sce_labels)) if sce_labels.size > 0 else 0
    if total_sce_count < _MINIMUM_SCE_COUNT_FOR_ASSEMBLY:
        return []

    sample_count = onset_matrix.shape[1]
    # noinspection PyTypeChecker
    sce_sample_mask: NDArray[np.bool_] = sce_labels > 0
    # noinspection PyTypeChecker
    sce_sample_indices: NDArray[np.int64] = np.where(sce_sample_mask)[0]
    # noinspection PyTypeChecker
    sample_to_sce: NDArray[np.float32] = np.zeros((sample_count, total_sce_count), dtype=np.float32)
    sample_to_sce[sce_sample_indices, sce_labels[sce_sample_indices] - 1] = 1.0
    # noinspection PyTypeChecker
    participation: NDArray[np.bool_] = (onset_matrix.astype(np.float32) @ sample_to_sce > 0).T

    cell_participation_count = np.sum(participation, axis=0)
    # noinspection PyTypeChecker
    active_cell_mask: NDArray[np.bool_] = cell_participation_count > 0
    # noinspection PyTypeChecker
    active_cell_indices: NDArray[np.int32] = np.where(active_cell_mask)[0].astype(np.int32)
    if active_cell_indices.size < minimum_assembly_size:
        return []

    cell_vectors = participation[:, active_cell_mask].T.astype(np.float64)
    distances = pdist(X=cell_vectors, metric="jaccard")
    distances = np.nan_to_num(distances, nan=0.0)

    cluster_target = min(max_clusters, active_cell_indices.size // minimum_assembly_size)
    cluster_target = max(2, cluster_target)
    linkage_matrix = linkage(distances, method="average")
    cluster_labels = fcluster(linkage_matrix, t=cluster_target, criterion="maxclust")

    assemblies: list[tuple[int, NDArray[np.int32]]] = []
    for cluster_id in range(1, cluster_target + 1):
        # noinspection PyTypeChecker
        member_mask: NDArray[np.bool_] = cluster_labels == cluster_id
        if int(np.sum(member_mask)) < minimum_assembly_size:
            continue
        member_indices = active_cell_indices[member_mask]
        member_participation = participation[:, member_indices]
        active_fraction = np.mean(member_participation, axis=1)
        # noinspection PyTypeChecker
        activations: NDArray[np.int32] = np.where(active_fraction >= activation_threshold)[0].astype(np.int32)
        if activations.size == 0:
            continue
        assemblies.append((int(activations.size), member_indices))

    assemblies.sort(key=lambda entry: entry[0], reverse=True)
    return [member_indices for _, member_indices in assemblies]


def _plot_dataset_metric(
    dataset: DatasetData,
    metric_extractor: Callable[[CellAnalysisSummary], float],
    y_label: str,
    figure_title: str,
) -> plt.Figure:
    """Walks ``dataset.sessions`` per animal, loads each session's CellAnalysisReport, and renders a per-animal
    line + across-animal median + IQR aggregate following the bleaching dataset-trend template.
    """
    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    animal_traces: list[tuple[NDArray[np.int64], NDArray[np.float32]]] = []
    for dataset_animal in dataset.animals:
        animal_sessions = sorted(
            dataset.get_sessions_for_animal(animal=dataset_animal.animal),
            key=lambda dataset_session: dataset_session.session,
        )
        per_session_values: list[float] = []
        per_session_session_names: list[str] = []
        for session in animal_sessions:
            try:
                report = CellAnalysisReport.load(session=session)
            except FileNotFoundError:
                continue
            per_session_values.append(float(metric_extractor(report.summary)))
            per_session_session_names.append(session.session)

        if not per_session_values:
            continue

        days_float = _session_names_to_days_since_first(session_names=per_session_session_names)
        if days_float.size == 0:
            continue
        # noinspection PyTypeChecker
        unit, ticks = resolve_display_units(days_since_first=days_float)
        del unit  # tick units are int64; the dataset-level x-axis is always integer days for stable alignment.
        # noinspection PyTypeChecker
        days_int: NDArray[np.int64] = np.round(days_float).astype(np.int64, copy=False) if ticks is None else ticks
        # noinspection PyTypeChecker
        values: NDArray[np.float32] = np.asarray(per_session_values, dtype=np.float32)
        animal_traces.append((days_int, values))

    if not animal_traces:
        axes.set_xlabel("Days since first session")
        axes.set_ylabel(y_label)
        axes.set_title(f"{figure_title} (no reports found)", fontsize=10)
        figure.tight_layout()
        return figure

    for days_int, values in animal_traces:
        axes.plot(days_int, values, color="grey", alpha=0.5, linewidth=1.0, marker="o", markersize=3)

    max_day = int(max(days_int.max() for days_int, _ in animal_traces))
    # noinspection PyTypeChecker
    matrix: NDArray[np.float32] = np.full((len(animal_traces), max_day + 1), np.nan, dtype=np.float32)
    for animal_index, (days_int, values) in enumerate(animal_traces):
        matrix[animal_index, days_int] = values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # noinspection PyTypeChecker
        median_trace: NDArray[np.float32] = np.nanmedian(matrix, axis=0).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        lower_quartile: NDArray[np.float32] = np.nanpercentile(matrix, 25, axis=0).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        upper_quartile: NDArray[np.float32] = np.nanpercentile(matrix, 75, axis=0).astype(np.float32, copy=False)

    valid_mask = np.isfinite(median_trace)
    grid = np.arange(max_day + 1, dtype=np.int64)
    axes.fill_between(
        grid[valid_mask],
        lower_quartile[valid_mask],
        upper_quartile[valid_mask],
        color="black",
        alpha=0.15,
        linewidth=0,
        label="IQR (25-75%)",
    )
    axes.plot(grid[valid_mask], median_trace[valid_mask], color="black", linewidth=2.5, label="Across-animal median")

    axes.set_xlabel("Days since first session")
    axes.set_ylabel(y_label)
    axes.set_title(f"{figure_title} (n={len(animal_traces)} animals)", fontsize=10)
    axes.legend(fontsize=8, loc="best")
    figure.tight_layout()
    return figure


def _session_names_to_days_since_first(session_names: list[str]) -> NDArray[np.float32]:
    """Converts a list of canonical YYYY-MM-DD-HH-MM-SS-microseconds session names into float days since the first
    session.
    """
    session_format = "%Y-%m-%d-%H-%M-%S-%f"
    microseconds = [
        int(parse_timestamp(date_string=name, format_string=session_format, output_format=TimestampFormats.INTEGER))
        for name in session_names
    ]
    if not microseconds:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float32)
    base = microseconds[0]
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = np.asarray(
        [
            float(convert_time(time=value - base, from_units=TimeUnits.MICROSECOND, to_units=TimeUnits.DAY))
            for value in microseconds
        ],
        dtype=np.float32,
    )
    return days
