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
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import polars as pl
from scipy.stats import chi2
from scipy.signal import savgol_filter
from threadpoolctl import threadpool_limits
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import YamlConfig

from ..forging import FluorescenceColumn
from .utilities import (
    resolve_display_units,
    per_cell_pearson_safe,
    trim_acquisition_warmup,
    assemble_run_session_data,
)
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
_PLACE_STRIP_WIDTH_RATIO: float = 0.04
"""Per-strip width ratio (relative to the main heatmap) used by the per-cell significance strips."""
_PLACE_PVALUE_DISPLAY_FLOOR: float = 1e-4
"""P-value floor used when computing the ``-log10(p)`` color scale; keeps the dynamic range bounded."""
_PLACE_FIELD_BIN_SIZE_CM: float = 5.0
"""Default spatial bin size in centimeters used by ``PlaceFieldDetector`` and persisted in the per-trial fluorescence
matrix. Matches the ``PlaceFieldDetector`` constructor default."""
_MINIMUM_SCE_COUNT_FOR_ASSEMBLY: int = 2
"""Minimum number of SCEs required in a period to attempt cell-assembly detection."""
_MINIMUM_PERIODS_FOR_REST_RUN_REST: int = 1
"""Minimum number of run periods required to render a rest-run-rest plot."""
_MINIMUM_OBSERVATIONS_FOR_VARIANCE: int = 2
"""Minimum number of samples or cells required for a variance-, PCA-, or rank-correlation-based step to
produce a defined output. Below this threshold the corresponding helper short-circuits to NaN."""
_ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE: int = 10
"""Preferred BLAS thread count per ICA-CS / reactivation shuffle worker. Mirrors the bleaching analyzer's
``_PREFERRED_WORKERS_PER_SESSION = 10`` constant: the Lanczos matvec and the per-period reactivation GEMMs
are BLAS-bound and stop scaling cleanly past ten threads, so the shuffle-level allocator
(:func:`_resolve_ica_shuffle_allocation`) targets this width and uses the remaining budget to spawn more
parallel workers. On a 128-core host the allocator returns ``(10, 12)`` — twelve concurrent shuffles each
with a ten-thread BLAS pool, totalling 120 cores in flight."""
_ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE: int = 5
"""Floor on per-worker BLAS threads. Falling below this floor reduces parallel-shuffle count one worker at a
time rather than spawning under-resourced workers whose matvec performance would collapse."""
_ICA_BLAS_THREAD_MULTIPLE: int = 5
"""Per-worker BLAS thread counts are rounded down to this multiple for clean allocation. Mirrors the
bleaching/cindra worker-multiple convention."""


class CellAnalysisColumn(StrEnum):
    """Defines every column written to the per-session ``cell_analysis.feather`` per-cell table."""

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
    """True for cells whose center of mass falls inside the reward zone (Issa et al. 2024 zone band; configured by
    ``reward_zone_width``)."""
    IS_DEPARTURE = "is_departure"
    """True for cells whose center of mass falls in the departure band immediately downstream of the reward zone
    (Issa et al. 2024 post-reward band; default 40 cm)."""
    IS_REWARD_CELL = "is_reward_cell"
    """True for cells that are both spatially significant and reward-proximal (zone-band)."""
    IS_POSITION_GLM_SIGNIFICANT = "is_position_glm_significant"
    """True for cells whose 5-fold CV ΔR² of position over speed+acceleration exceeds the trial-label permutation
    null at the configured ``glm_significance_threshold``. Replaces the legacy ``IS_SLOWING_CORRELATED`` flag with a
    rigorously decoupled position-vs-speed test (Sosa, Plitt & Giocomo 2025; Hardcastle et al. 2017)."""
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
    """Z-scored Skaggs spatial information ``(I_obs - mean(I_shuf)) / std(I_shuf)`` against the same circular-shift
    null used for ``SPATIAL_P_VALUE``. Reported alongside the raw bits/event because raw Skaggs is biased at low
    rates with calcium imaging (Souza & Tort 2018; Sheintuch et al. 2022); the z-score is calibrated to the per-cell
    shuffled null and correlates better with Bayesian-decoder accuracy."""
    SPATIAL_P_VALUE = "spatial_p_value"
    """Shuffle-derived p-value for the spatial information statistic."""
    SPATIAL_FDR_SURVIVED = "spatial_fdr_survived"
    """True for cells whose ``SPATIAL_P_VALUE`` survives Benjamini-Hochberg FDR correction at the configured
    ``fdr_q`` level. ``IS_SPATIALLY_SIGNIFICANT`` ANDs this with the lap-reliability gate; this column exposes the
    raw FDR result for downstream consumers that want only the FDR contribution."""
    SPATIAL_SPLIT_HALF_R = "spatial_split_half_r"
    """Per-cell even/odd-lap Pearson r used by the reward-cell pipeline as the lap-reliability gate. Distinct from
    ``STABILITY_EVEN_ODD`` only insofar as the reward pipeline computes it against its own bin edges; with matching
    config (the default) the two columns are bit-identical. NaN when the trial count is below two or either half-map
    has zero variance."""
    REWARD_RELATIVITY_SCORE = "reward_relativity_score"
    """Per-cell ``zone_peak / overall_peak`` of the smoothed rate map; in [0, 1]. 1.0 means the cell's peak falls
    inside the reward zone; values < 1.0 measure how much weaker the in-zone signal is than the cell's overall
    peak. Used by the Phase-2 mutual-exclusion winner-take-all and naturally extends to the multi-block reward-shift
    analysis in Phase 3 (where it would be computed in reward-aligned coordinates per block)."""
    CV_POSITION_PARTIAL_R2 = "cv_position_partial_r2"
    """Per-cell 5-fold CV ΔR² of position over speed+acceleration in the pre-reward window. The principal A1
    statistic; positive values indicate that position adds explanatory power beyond running-speed and acceleration,
    decoupling reward-proximal cells from cells that simply slow at the zone (Sosa, Plitt & Giocomo 2025)."""
    POSITION_GLM_P_VALUE = "position_glm_p_value"
    """Per-cell trial-label permutation p-value for the ``CV_POSITION_PARTIAL_R2`` statistic. NaN where the cell
    did not enter the GLM (insufficient active trials, untested band, etc.)."""
    STABILITY_EVEN_ODD = "stability_even_odd"
    """Pearson r between the even-trial and odd-trial mean rate maps. NaN when fewer than two trials are available
    or either half-map has zero variance."""
    STABILITY_SPLIT_HALF = "stability_split_half"
    """Pearson r between the first-half and second-half mean rate maps. NaN when the trial count is below 2 or
    either half-map has zero variance."""
    IS_RELIABLE = "is_reliable"
    """True for cells with at least one place field that passes the lap-coverage criterion. Equivalent to ``IS_PLACE``
    today because the lap-coverage filter is applied during place-field detection (Climer et al., 2025); persisted as
    a separate column so downstream analysts can keep the criteria explicit."""
    IS_STABLE = "is_stable"
    """True for cells whose split-half stability r exceeds the 95th percentile of a per-cell shuffled null. From the
    Stability method of Climer & Dombeck (2021)."""
    STABILITY_P_VALUE = "stability_p_value"
    """Per-cell p-value for the Stability shuffle: fraction of shuffles whose split-half r is greater than or equal
    to the observed value. NaN when the observed r is NaN or the shuffle distribution is empty (e.g., fewer than two
    trials)."""
    IS_PEAK_SIGNIFICANT = "is_peak_significant"
    """True for cells whose observed pooled-rate-map peak exceeds the 99th percentile of the shuffled per-cell peak
    distribution. From the Peak method of Climer & Dombeck (2021)."""
    PEAK_P_VALUE = "peak_p_value"
    """Per-cell p-value for the Peak shuffle: fraction of shuffles whose smoothed-rate-map peak is greater than or
    equal to the observed peak. NaN when the observed peak is NaN."""
    IS_STRICT_PLACE = "is_strict_place"
    """True for cells that pass all three of ``IS_PLACE``, ``IS_STABLE``, and ``IS_PEAK_SIGNIFICANT`` simultaneously.
    Convenience column for downstream consumers; computed by AND-ing the three independent flags during table
    assembly."""
    SCE_PARTICIPATION_COUNT_REST = "sce_participation_count_rest"
    """Total number of rest-period SCEs the cell participated in."""
    SCE_PARTICIPATION_COUNT_RUN = "sce_participation_count_run"
    """Total number of run-period SCEs the cell participated in."""
    SCE_PARTICIPATION_RATE_REST = "sce_participation_rate_rest"
    """Fraction of rest-period SCEs the cell participated in."""
    SCE_PARTICIPATION_RATE_RUN = "sce_participation_rate_run"
    """Fraction of run-period SCEs the cell participated in."""
    SCE_EVENTS_REST = "sce_events_rest"
    """Per-cell list of (period_index, sce_label) pairs for every rest-period SCE the cell participated in."""
    SCE_EVENTS_RUN = "sce_events_run"
    """Per-cell list of (period_index, sce_label) pairs for every run-period SCE the cell participated in."""
    SCE_PARTICIPATION_P_VALUE_REST = "sce_participation_p_value_rest"
    """Per-cell aggregated p-value for SCE recruitment across all rest periods. Combined via Fisher's method
    over the per-period jitter-null p-values produced by ``SCEDetector``. NaN when no rest period contributed
    SCEs (Modol et al. 2020 super-rich-cell logic; Fisher 1925 p-value combination)."""
    SCE_PARTICIPATION_P_VALUE_RUN = "sce_participation_p_value_run"
    """Per-cell aggregated p-value for SCE recruitment across all run periods. NaN when no run period
    contributed SCEs."""
    IS_SCE_CELL_REST = "is_sce_cell_rest"
    """True for cells whose participation rate exceeds the per-cell jitter null in at least one rest period at
    the configured ``participation_significance_percentile``."""
    IS_SCE_CELL_RUN = "is_sce_cell_run"
    """True for cells whose participation rate exceeds the per-cell jitter null in at least one run period at
    the configured ``participation_significance_percentile``."""


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
    TRIAL_IDS = "trial_ids"
    """Per-sample trial id matching the dataset ``trial`` column (-1 outside any complete trial). Persisted so
    reward- and position-aligned SCE analyses can locate every event without re-loading ``data.feather``."""
    SCE_SIZE = "sce_size"
    """Per-SCE distinct participating-cell count. Length equal to the per-period SCE count."""
    SCE_WIDTH_SAMPLES = "sce_width_samples"
    """Per-SCE duration in samples. Convert to seconds with ``SAMPLING_RATE_HZ``."""
    SCE_PEAK_COACTIVE = "sce_peak_coactive"
    """Per-SCE peak co-active count, the maximum of ``COACTIVE_COUNTS`` over the SCE window."""
    SCE_INTER_EVENT_INTERVALS_SAMPLES = "sce_inter_event_intervals_samples"
    """Per-SCE sample gap between consecutive events; length equal to ``max(sce_count - 1, 0)``."""
    SCE_RATE_HZ = "sce_rate_hz"
    """SCE rate in events per second over the period (post-stability or post-rate-map masking)."""
    REACTIVATION_STRENGTHS = "reactivation_strengths"
    """Per-SCE population reactivation strength against the run-derived assembly templates. Computed via the
    Peyrache 2009 / Lopes-dos-Santos 2013 diagonal-removed projection on z-scored binary onsets. Length equal
    to the per-period SCE count; empty for run periods (templates are derived from run, evaluated on rest)."""
    MEAN_REACTIVATION_STRENGTH = "mean_reactivation_strength"
    """Per-period mean reactivation strength across the period's SCEs; NaN when no SCEs were detected."""
    REACTIVATING_SCE_FRACTION = "reactivating_sce_fraction"
    """Per-period fraction of SCEs whose reactivation strength exceeds the per-period 95th-percentile of the
    null distribution computed by independent circular shifts on the run-template projections."""


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
    """Number of cells with at least one detected place field (Dombeck-style threshold + lap-coverage filter)."""
    spatially_significant_count: int
    """Number of cells whose Skaggs spatial information passes the shuffle threshold."""
    reward_cell_count: int
    """Number of cells that are both spatially significant and reward-proximal."""
    reward_predictive_count: int
    """Number of cells that are reward-associated and slowing-correlated."""
    reliable_count: int
    """Number of cells whose detected fields pass the lap-coverage criterion (Climer 2025). Equivalent to
    ``place_cell_count`` while lap-coverage is applied during detection; persisted separately so the criterion is
    explicit in the summary."""
    stable_count: int
    """Number of cells whose split-half stability r exceeds the 95th percentile of a per-cell shuffled null
    (Climer & Dombeck 2021 Stability method)."""
    peak_significant_count: int
    """Number of cells whose pooled-rate-map peak exceeds the 99th percentile of a per-cell shuffled null
    (Climer & Dombeck 2021 Peak method)."""
    strict_place_cell_count: int
    """Number of cells that simultaneously pass IS_PLACE, IS_STABLE, and IS_PEAK_SIGNIFICANT. Used as the working
    place-cell population for plotting; downstream consumers can still recover any single criterion from the
    per-cell table."""
    place_only_count: int
    """Number of cells flagged ``IS_PLACE`` but not ``IS_REWARD_CELL``. Cached so dataset-level place-fraction plots
    can render the mutually-exclusive view without rehydrating per-cell feathers."""
    strict_place_only_count: int
    """Number of cells passing ``IS_PLACE & IS_STABLE & IS_PEAK_SIGNIFICANT`` but not ``IS_REWARD_CELL``. Cached
    counterpart of ``strict_place_cell_count`` for the mutually-exclusive view."""

    mixture_weight: float
    """Reward-component weight from the four-component (uniform + reward + track-start + track-end) mixture model
    fit to the spatially significant COMs."""
    gaussian_mean_cm: float
    """Reward-component Gaussian mean in centimeters."""
    gaussian_std_cm: float
    """Reward-component Gaussian standard deviation in centimeters."""
    track_start_weight: float
    """Track-start landmark Gaussian weight (centered at 0 cm). Captures over-representation of trajectory start
    cells that would otherwise contaminate ``mixture_weight`` (Hainmueller & Bartos 2018; Sato et al. 2020)."""
    track_end_weight: float
    """Track-end landmark Gaussian weight (centered at ``track_length_cm``). Captures trajectory-endpoint cells
    (Frank, Brown & Wilson 2000)."""
    track_start_std_cm: float
    """Track-start landmark Gaussian standard deviation in centimeters."""
    track_end_std_cm: float
    """Track-end landmark Gaussian standard deviation in centimeters."""

    rest_period_count: int
    """Number of rest periods that survived torque-stability filtering and contributed an SCE row."""
    run_period_count: int
    """Number of run periods that contributed an SCE row."""
    total_rest_sces: int
    """Total SCEs detected across all rest periods."""
    total_run_sces: int
    """Total SCEs detected across all run periods."""
    rest_sce_cell_count: int
    """Per-session count of cells flagged as SCE-recruited during at least one rest period (Modol 2020
    super-rich)."""
    run_sce_cell_count: int
    """Per-session count of cells flagged as SCE-recruited during at least one run period."""
    mean_rest_reactivation_strength: float
    """Mean reactivation strength of rest-period SCEs against the run-derived assembly templates (C2a in the
    Peyrache 2009 / Lopes-dos-Santos 2013 sense). NaN when no rest period contributed SCEs or when too few
    significant assemblies survived the Marchenko-Pastur cutoff to define a template."""
    rest_reactivating_sce_fraction: float
    """Fraction of rest-period SCEs whose reactivation strength exceeds the 95th percentile of a per-SCE
    independent-shuffle null. NaN under the same conditions as ``mean_rest_reactivation_strength``."""


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
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
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

        # Loads the session data once and shares it across both detectors via from_run_session, so the place and
        # reward flags operate on identical speed-filtered samples and bit-identical rate maps.
        run_session = assemble_run_session_data(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )

        # Runs place-field detection.
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

        # Runs reward-cell detection.
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

        # Computes split-half and even/odd stability r per cell from the per-trial binned fluorescence already
        # produced during place-field detection. The IS_STABLE and IS_PEAK_SIGNIFICANT booleans are computed by
        # ``_compute_multi_criterion_flags`` from shuffle distributions; see Climer & Dombeck (2021) for the
        # multi-criterion framework.
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
        is_reward_cell_array: NDArray[np.bool_] = reward_results.spatial_results.is_significant & reward_results.is_zone
        # noinspection PyTypeChecker
        is_place_only: NDArray[np.bool_] = place_fields.has_place_field & ~is_reward_cell_array
        # noinspection PyTypeChecker
        is_strict_place_only: NDArray[np.bool_] = is_strict_place & ~is_reward_cell_array

        # Assembles the per-cell wide table.
        table = _build_cell_table(
            cell_count=cell_count,
            place_fields=place_fields,
            reward_results=reward_results,
            sce_results=sce_results,
            stability_even_odd=stability_even_odd,
            stability_split_half=stability_split_half,
            is_stable=is_stable,
            is_peak_significant=is_peak_significant,
            stability_p_values=stability_p_values,
            peak_p_values=peak_p_values,
            is_strict_place=is_strict_place,
        )

        # Computes population reactivation strength (Peyrache 2009 / Lopes-dos-Santos 2013): templates from
        # concatenated RUN onsets via ICA-CS, projected onto each REST period's SCE windows.
        console.echo(
            message=(
                f"Running population reactivation analysis ({rest_period_count} rest periods, "
                f"{run_period_count} run periods)..."
            ),
            level=LogLevel.INFO,
        )
        templates, per_period_reactivation = _compute_population_reactivation(
            run_results=sce_detector.run_results,
            rest_results=sce_detector.rest_results,
            cell_count=cell_count,
        )
        rest_indices_in_results = [i for i, r in enumerate(sce_results) if r.period_type == PeriodType.REST]
        reactivation_columns: dict[int, dict[str, object]] = {
            rest_indices_in_results[rest_position]: payload
            for rest_position, payload in per_period_reactivation.items()
        }
        if templates.shape[0] > 0 and per_period_reactivation:
            mean_strength_value = float(
                np.mean([float(entry["mean_reactivation_strength"]) for entry in per_period_reactivation.values()])
            )
            reactivating_fraction_value = float(
                np.mean([float(entry["reactivating_sce_fraction"]) for entry in per_period_reactivation.values()])
            )
        else:
            mean_strength_value = float("nan")
            reactivating_fraction_value = float("nan")
        console.echo(
            message=(
                f"Reactivation analysis complete: {templates.shape[0]} run-derived templates, "
                f"mean strength {mean_strength_value:.3f}, reactivating fraction {reactivating_fraction_value:.3f}."
            ),
            level=LogLevel.INFO,
        )

        # Assembles the per-period SCE table with the reactivation columns folded in.
        sce_periods = _build_sce_periods_table(
            sampling_rate_hz=sampling_rate_hz,
            results=sce_results,
            cell_count=cell_count,
            reactivation_columns=reactivation_columns,
        )

        # noinspection PyTypeChecker
        rest_sce_cell_flag: NDArray[np.bool_] = table[CellAnalysisColumn.IS_SCE_CELL_REST.value].to_numpy()
        # noinspection PyTypeChecker
        run_sce_cell_flag: NDArray[np.bool_] = table[CellAnalysisColumn.IS_SCE_CELL_RUN.value].to_numpy()

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
            rest_period_count=rest_period_count,
            run_period_count=run_period_count,
            total_rest_sces=total_rest_sces,
            total_run_sces=total_run_sces,
            rest_sce_cell_count=int(np.sum(rest_sce_cell_flag)),
            run_sce_cell_count=int(np.sum(run_sce_cell_flag)),
            mean_rest_reactivation_strength=mean_strength_value,
            rest_reactivating_sce_fraction=reactivating_fraction_value,
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

    def place_mask(
        self,
        *,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
    ) -> NDArray[np.bool_]:
        """Returns the per-cell boolean mask for cells passing every requested place-cell criterion simultaneously.

        Notes:
            Defaults to the strict triple-AND of place / stable / peak-significant recommended by Climer & Dombeck
            (2021). When every kwarg is False the method returns an all-True mask, treating "no criteria" as "no
            filter". Skaggs spatial significance is intentionally not exposed here: it is the reward-cell pipeline's
            broader spatial filter and overlaps heavily with Dombeck place-field morphology. Use
            :meth:`resolve_population_masks` instead when you need place / reward populations that respect mutual
            exclusion.

        Args:
            require_place: Require ``IS_PLACE`` (Dombeck 2010 morphology + lap coverage).
            require_stable: Require ``IS_STABLE`` (Climer & Dombeck 2021 Stability method).
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (Climer & Dombeck 2021 Peak method).

        Returns:
            Per-cell boolean mask with length cell_count.
        """
        # noinspection PyTypeChecker
        mask: NDArray[np.bool_] = np.ones(self.table.height, dtype=np.bool_)
        if require_place:
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_PLACE.value].to_numpy()
            mask = mask & place_flag
        if require_stable:
            # noinspection PyTypeChecker
            stable_flag: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_STABLE.value].to_numpy()
            mask = mask & stable_flag
        if require_peak_significant:
            # noinspection PyTypeChecker
            peak_flag: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_PEAK_SIGNIFICANT.value].to_numpy()
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
            The reward mask is always ``IS_REWARD_CELL`` (i.e., spatially significant cells whose COM lies in the
            zone band). When ``mutually_exclusive=True``, cells flagged as both place and reward are subtracted from
            the place mask so the two populations are disjoint at the presentation layer; when ``False``, the place
            mask is the unfiltered place population and a cell may appear in both. The persisted feather always
            keeps both flags independently — mutual exclusion is purely a presentation-layer convention so the
            distinction is reversible without re-running detection.

        Args:
            require_place: Require ``IS_PLACE`` for the place population.
            require_stable: Require ``IS_STABLE`` for the place population.
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` for the place population.
            mutually_exclusive: When True, subtract ``IS_REWARD_CELL`` cells from the place mask. Default True for
                visualization clarity; set False to keep both populations as recorded in the table.

        Returns:
            A tuple of (place_mask, reward_mask) per-cell boolean arrays each with length cell_count.
        """
        place_population = self.place_mask(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
        )
        # noinspection PyTypeChecker
        reward_mask: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_CELL.value].to_numpy()
        place_mask = place_population & ~reward_mask if mutually_exclusive else place_population
        return place_mask, reward_mask

    def summarize(self, *, mutually_exclusive: bool = True) -> str:
        """Returns a multi-line human-readable summary of the report's per-cell and per-period statistics.

        Notes:
            With ``mutually_exclusive=True`` (default), the place-cell count subtracts cells also flagged as
            ``IS_REWARD_CELL`` and is reported as "place-only"; the reward count is unchanged. With
            ``mutually_exclusive=False``, the raw counts persisted in the summary YAML are reported instead and a
            cell may contribute to both totals. The persisted summary YAML always stores raw counts so this
            distinction is purely a display-time choice.

        Args:
            mutually_exclusive: When True (default), report ``IS_PLACE & ~IS_REWARD_CELL`` for the place count.
        """
        summary = self.summary
        cell_count = summary.cell_count

        if mutually_exclusive and cell_count > 0:
            # noinspection PyTypeChecker
            place_flag: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_PLACE.value].to_numpy()
            # noinspection PyTypeChecker
            reward_flag: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_CELL.value].to_numpy()
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
            "Cell analysis report",
            "====================",
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
            "SCE detection:",
            f"  Rest periods: {summary.rest_period_count} ({summary.total_rest_sces} SCEs)",
            f"  Run periods:  {summary.run_period_count} ({summary.total_run_sces} SCEs)",
            f"  SCE-recruited cells (rest): {summary.rest_sce_cell_count}",
            f"  SCE-recruited cells (run):  {summary.run_sce_cell_count}",
            "",
            "Population reactivation (REST SCEs vs. RUN-derived templates):",
            f"  Mean strength:           {summary.mean_rest_reactivation_strength:.3f}",
            f"  Reactivating fraction:   {summary.rest_reactivating_sce_fraction:.3f}",
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
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
        mutually_exclusive: bool = True,
        show_significance_strip: bool = True,
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
            show_only_place_cells: Display only cells that pass every requested criterion (the AND of the three
                ``require_*`` flags below).
            require_place: Require ``IS_PLACE`` (Dombeck morphology + lap coverage).
            require_stable: Require ``IS_STABLE`` (Climer & Dombeck 2021 Stability shuffle).
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (Climer & Dombeck 2021 Peak shuffle).
            mutually_exclusive: When True (default), cells also flagged as ``IS_REWARD_CELL`` are removed from the
                place population so the panel shows only place cells that are not reward cells. Set False to display
                the unfiltered place population.
            show_significance_strip: When True, renders one thin per-cell ``-log10(p)`` strip to the left of the main
                heatmap for each active p-value-bearing criterion (Stable / Peak).
            figure_dpi: Figure resolution in dots per inch.
            minimum_percentile: Percentile used as the lower bound of the color scale.
            maximum_percentile: Percentile used as the upper bound of the color scale.
            cmap: Matplotlib colormap name for the rate-map panel.
            show_color_bar: Render a color bar alongside the heatmap.

        Returns:
            The matplotlib Figure containing the heatmap.
        """
        bin_size_cm = self.summary.bin_size_cm
        bin_count = self.summary.bin_count

        # Reconstructs the per-cell pooled rate map and per-cell place-field center used for ordering.
        rate_maps = _stack_list_column(table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=bin_count)
        cell_population, _ = self.resolve_population_masks(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
            mutually_exclusive=mutually_exclusive,
        )
        order = _resolve_place_cell_order(table=self.table)

        if not sort_by_position:
            # noinspection PyTypeChecker
            order = np.arange(rate_maps.shape[0], dtype=np.int64)

        if show_only_place_cells:
            order = order[np.isin(order, np.flatnonzero(cell_population))]

        sorted_data = rate_maps[order, :]
        if sorted_data.size == 0:
            sorted_data = rate_maps[:0, :]

        minimum_value = float(np.nanquantile(sorted_data, minimum_percentile)) if sorted_data.size > 0 else 0.0
        maximum_value = float(np.nanquantile(sorted_data, maximum_percentile)) if sorted_data.size > 0 else 1.0

        # Builds figure with optional p-value significance strips on the left edge.
        strip_columns = _active_significance_columns(
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
            table=self.table,
        )
        strip_count = len(strip_columns) if show_significance_strip else 0
        figure = _make_heatmap_figure(strip_count=strip_count, figure_dpi=figure_dpi)
        strip_axes, axes, colorbar_axes = _layout_heatmap_axes(
            figure=figure,
            strip_count=strip_count,
            include_colorbar=show_color_bar,
        )

        population_label = _compose_population_label(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
        )
        if title is not None:
            axes.set_title(f"{title} — {population_label} (n={order.size})", fontsize=8)
        elif show_only_place_cells:
            axes.set_title(f"{population_label} (n={order.size})", fontsize=8)

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
        if show_significance_strip and strip_axes:
            _render_significance_strips(
                figure=figure,
                strip_axes=strip_axes,
                strip_columns=strip_columns,
                table=self.table,
                ordered_indices=order,
            )

        track_length_cm = self.summary.track_length_cm
        # noinspection PyTypeChecker
        x_ticks: NDArray[np.float64] = np.arange(0, track_length_cm + 1, _PLOT_TICK_INTERVAL_CM)
        axes.set_xticks(x_ticks)

        if show_color_bar and colorbar_axes is not None:
            color_bar = figure.colorbar(image, cax=colorbar_axes)
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
        gaussian_density_reward = np.exp(-0.5 * ((positions - summary.gaussian_mean_cm) / gaussian_std) ** 2) / (
            gaussian_std * np.sqrt(2.0 * np.pi)
        )
        track_start_std = max(summary.track_start_std_cm, 1.0)
        gaussian_density_start = np.exp(-0.5 * ((positions - 0.0) / track_start_std) ** 2) / (
            track_start_std * np.sqrt(2.0 * np.pi)
        )
        track_end_std = max(summary.track_end_std_cm, 1.0)
        gaussian_density_end = np.exp(-0.5 * ((positions - summary.track_length_cm) / track_end_std) ** 2) / (
            track_end_std * np.sqrt(2.0 * np.pi)
        )
        uniform_weight = max(1.0 - summary.mixture_weight - summary.track_start_weight - summary.track_end_weight, 0.0)

        # Stacks the four mixture components in plotting order so each band is filled cumulatively without overlap.
        uniform_band = uniform_weight * uniform_density
        landmark_band = uniform_band + (
            summary.track_start_weight * gaussian_density_start + summary.track_end_weight * gaussian_density_end
        )
        mixture_density = landmark_band + summary.mixture_weight * gaussian_density_reward

        axes.fill_between(
            positions,
            0,
            uniform_band,
            alpha=0.3,
            color="lightblue",
            label="Uniform (place cells)",
        )
        axes.fill_between(
            positions,
            uniform_band,
            landmark_band,
            alpha=0.3,
            color="khaki",
            label="Track-end Gaussians",
        )
        axes.fill_between(
            positions,
            landmark_band,
            mixture_density,
            alpha=0.4,
            color="mediumpurple",
            label="Reward Gaussian",
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
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
        mutually_exclusive: bool = True,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots row-normalized rate maps for reward cells and place cells side by side, sorted by COM.

        Args:
            title: Optional figure-level title.
            require_place: Require ``IS_PLACE`` for the place-cell panel population.
            require_stable: Require ``IS_STABLE`` for the place-cell panel population.
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` for the place-cell panel population.
            mutually_exclusive: When True (default), cells flagged as ``IS_REWARD_CELL`` are subtracted from the
                place panel so the two side-by-side populations are disjoint. Set False to keep both panels showing
                their unfiltered populations.
            figure_dpi: Figure resolution in dots per inch.

        Notes:
            The reward panel always uses ``IS_REWARD_CELL`` (= ``IS_SPATIALLY_SIGNIFICANT & IS_ZONE``) and is
            unaffected by the ``require_*`` flags; those flags govern only the place-cell panel population.
        """
        summary = self.summary
        rate_maps = _stack_list_column(
            table=self.table, column=CellAnalysisColumn.RATE_MAP, target_length=summary.bin_count
        )
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )

        place_mask, reward_mask = self.resolve_population_masks(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
            mutually_exclusive=mutually_exclusive,
        )

        reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
        reward_left = summary.reward_position_cm - reward_zone_half
        reward_right = summary.reward_position_cm + reward_zone_half

        figure, (axes_reward, axes_place) = plt.subplots(
            1, 2, figsize=(12, 6), facecolor="white", dpi=figure_dpi, sharey=False
        )

        place_label = "Place Cells — " + _compose_population_label(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
        )
        for axes, mask, panel_title in [
            (axes_reward, reward_mask, "Reward Cells"),
            (axes_place, place_mask, place_label),
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
        is_position_glm_significant: NDArray[np.bool_] = self.table[
            CellAnalysisColumn.IS_POSITION_GLM_SIGNIFICANT.value
        ].to_numpy()

        bin_centers = (np.arange(summary.bin_count) + 0.5) * summary.bin_size_cm
        all_significant = is_significant
        reward_predictive_mask = all_significant & is_reward_proximal & is_position_glm_significant

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
                label=f"Reward-predictive (GLM, n={int(np.sum(reward_predictive_mask))})",
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
        is_position_glm_significant: NDArray[np.bool_] = self.table[
            CellAnalysisColumn.IS_POSITION_GLM_SIGNIFICANT.value
        ].to_numpy()
        reward_predictive_mask = is_significant & is_reward_proximal & is_position_glm_significant

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
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        title: str | None = None,
        require_place: bool = True,
        require_stable: bool = True,
        require_peak_significant: bool = True,
        mutually_exclusive: bool = True,
        figure_dpi: int = 150,
        position_bin_size_cm: float = 2.0,
        position_sigma_cm: float = 3.0,
        slowing_threshold_cm_s: float = 10.0,
    ) -> plt.Figure:
        """Plots per-trial activity heatmaps for an example reward-predictive cell and an example place cell, with
        slowing-onset markers overlaid. Reads ``data.feather`` for the raw fluorescence and per-trial speed time
        series.

        Args:
            session: DatasetSession backing the on-disk data.feather.
            trial_type: Trial type to load.
            fluorescence_column: Column to load from data.feather.
            title: Optional figure-level title.
            require_place: Require ``IS_PLACE`` for the place-cell pool from which the example place cell is picked.
            require_stable: Require ``IS_STABLE`` for the place-cell pool.
            require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` for the place-cell pool.
            mutually_exclusive: When True (default), cells flagged as ``IS_REWARD_CELL`` are subtracted from the
                place-cell pool so the example place cell is never also a reward cell. Set False to allow the
                example place-cell pick to come from the unfiltered place population.
            figure_dpi: Figure resolution in dots per inch.
            position_bin_size_cm: Spatial bin size used to build the per-trial heatmaps.
            position_sigma_cm: Gaussian smoothing sigma applied along the position axis.
            slowing_threshold_cm_s: Speed cutoff (cm/s) used to mark per-trial slowing-onset locations.

        Notes:
            The reward-predictive panel is unaffected by the ``require_*`` and ``mutually_exclusive`` flags; those
            flags govern only the place-cell pool.
        """
        summary = self.summary
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward_proximal: NDArray[np.bool_] = self.table[CellAnalysisColumn.IS_REWARD_PROXIMAL.value].to_numpy()
        # noinspection PyTypeChecker
        is_position_glm_significant: NDArray[np.bool_] = self.table[
            CellAnalysisColumn.IS_POSITION_GLM_SIGNIFICANT.value
        ].to_numpy()
        # noinspection PyTypeChecker
        cv_partial_r2: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CV_POSITION_PARTIAL_R2.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = (
            self.table[CellAnalysisColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )

        place_mask, _ = self.resolve_population_masks(
            require_place=require_place,
            require_stable=require_stable,
            require_peak_significant=require_peak_significant,
            mutually_exclusive=mutually_exclusive,
        )
        predictive_mask = is_significant & is_reward_proximal & is_position_glm_significant
        # noinspection PyTypeChecker
        predictive_indices: NDArray[np.int64] = np.argwhere(predictive_mask).flatten()
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

        # Picks the candidate with the largest CV ΔR² of position over speed+accel as the most clearly position-tuned
        # cell beyond the speed/acceleration covariates; replaces the legacy "most-negative speed correlation" pick.
        best_predictive = int(predictive_indices[np.argmax(cv_partial_r2[predictive_indices])])
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
            cell_partial_r2 = float(cv_partial_r2[cell_index]) if not np.isnan(cv_partial_r2[cell_index]) else 0.0
            axes.set_title(f"{label} (cell {cell_index}, COM={cell_com:.0f} cm, ΔR²={cell_partial_r2:.2f})", fontsize=9)

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
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
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
        shuffle_count: int = 200,
        eigenvalue_significance_percentile: float = 99.0,
        membership_z_threshold: float = 2.0,
        minimum_assembly_size: int = 3,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Detects and plots the most prominent SCE cell assemblies as raster panels using ICA-CS
        (Lopes-dos-Santos 2013) on the persisted per-period SCE participation matrix. Recomputes assemblies
        on every call so no live detector is required.
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

        # Build the per-SCE participation matrix (cells x sce_count) for ICA-CS, mirroring the upstream
        # cell-feather aggregation but kept local to the plot so callers can rerun with different thresholds.
        if total_sce_count < _MINIMUM_SCE_COUNT_FOR_ASSEMBLY:
            participation_matrix: NDArray[np.float32] = np.empty((onset_matrix.shape[0], 0), dtype=np.float32)
        else:
            sample_count = onset_matrix.shape[1]
            # noinspection PyTypeChecker
            sce_sample_indices: NDArray[np.int64] = np.where(sce_labels > 0)[0]
            # noinspection PyTypeChecker
            sample_to_sce: NDArray[np.float32] = np.zeros((sample_count, total_sce_count), dtype=np.float32)
            sample_to_sce[sce_sample_indices, sce_labels[sce_sample_indices] - 1] = 1.0
            # noinspection PyTypeChecker
            participation_matrix = (onset_matrix.astype(np.float32) @ sample_to_sce > 0).astype(np.float32)

        _, assemblies = _detect_assemblies_ica_cs(
            activity_matrix=participation_matrix,
            shuffle_count=shuffle_count,
            eigenvalue_significance_percentile=eigenvalue_significance_percentile,
            membership_z_threshold=membership_z_threshold,
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
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
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


def plot_dataset_place_cell_fraction(
    dataset: DatasetData,
    *,
    require_place: bool = True,
    require_stable: bool = True,
    require_peak_significant: bool = True,
    mutually_exclusive: bool = True,
) -> plt.Figure:
    """Plots the per-animal place-cell fraction trend for the criterion combination chosen by the ``require_*`` flags.

    Notes:
        Animals without saved reports are silently skipped. The y-axis label and figure title encode the active
        criterion combination so several invocations with different criteria can be saved alongside one another
        without ambiguity. Single-criterion combinations and the strict triple-AND read directly from precomputed
        summary fields; other combinations are not currently cached and raise. With ``mutually_exclusive=True``
        (default) and either the IS_PLACE-only or strict-triple combination, the plot uses the cached
        ``place_only_count`` / ``strict_place_only_count`` summary fields so reward cells are not double-counted.

    Args:
        dataset: DatasetData root used to enumerate per-animal sessions.
        require_place: Require ``IS_PLACE`` (Dombeck morphology + lap coverage).
        require_stable: Require ``IS_STABLE`` (Climer & Dombeck 2021 Stability shuffle).
        require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (Climer & Dombeck 2021 Peak shuffle).
        mutually_exclusive: When True, subtract reward cells from the place fraction. Only the IS_PLACE-only and
            strict-triple combinations have cached mutually-exclusive counts; other combinations require
            ``mutually_exclusive=False``.
    """
    label = _compose_population_label(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
    )
    extractor = _resolve_dataset_place_metric_extractor(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )
    suffix = " (excl. reward)" if mutually_exclusive else ""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=extractor,
        y_label=f"{label}{suffix} fraction",
        figure_title=f"Across-animal {label.lower()}{suffix} fraction",
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


def plot_dataset_stable_cell_fraction(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal stable-cell fraction trend (Climer & Dombeck 2021 Stability method)."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: summary.stable_count / summary.cell_count if summary.cell_count else 0.0,
        y_label="Stable cell fraction",
        figure_title="Across-animal stable cell fraction",
    )


def plot_dataset_peak_significant_cell_fraction(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal peak-significant cell fraction trend (Climer & Dombeck 2021 Peak method)."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: (
            summary.peak_significant_count / summary.cell_count if summary.cell_count else 0.0
        ),
        y_label="Peak-significant cell fraction",
        figure_title="Across-animal peak-significant cell fraction",
    )


def plot_dataset_reliable_cell_fraction(dataset: DatasetData) -> plt.Figure:
    """Plots the per-animal reliable-cell fraction trend (Climer 2025 lap-coverage criterion)."""
    return _plot_dataset_metric(
        dataset=dataset,
        metric_extractor=lambda summary: summary.reliable_count / summary.cell_count if summary.cell_count else 0.0,
        y_label="Reliable cell fraction",
        figure_title="Across-animal reliable cell fraction",
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


def _compute_stability_metrics(
    binned_fluorescence_per_trial: NDArray[np.float32],
    cell_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes per-cell even/odd and split-half Pearson correlations from a per-trial binned rate-map matrix.

    Notes:
        Both metrics return NaN for cells with fewer than two trials, fewer than three valid bins in either half,
        or zero variance in either half. Bins where either half has NaN are excluded pairwise from the correlation.

    References:
        - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of cells
          identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- Stability method.
        - Hainmueller & Bartos (2018). Parallel emergence of stable and dynamic memory engrams in the hippocampus.
          Nature. https://doi.org/10.1038/s41586-018-0191-2 -- split-half stability r as a place-cell criterion.

    Args:
        binned_fluorescence_per_trial: Per-trial binned fluorescence with dimensions (cell_count, trial_count,
            bin_count). NaN entries are treated as missing.
        cell_count: Number of cells in the session; used to size the output arrays when the per-trial matrix is
            empty.

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
    # Need at least one trial in each half to compute even/odd and split-half Pearson r.
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
            first_map: NDArray[np.float32] = np.nanmean(binned_fluorescence_per_trial[:, :half_index, :], axis=1).astype(
                np.float32, copy=False
            )
            # noinspection PyTypeChecker
            second_map: NDArray[np.float32] = np.nanmean(binned_fluorescence_per_trial[:, half_index:, :], axis=1).astype(
                np.float32, copy=False
            )
        split_half = per_cell_pearson_safe(a=first_map, b=second_map)

    return even_odd, split_half


def _compute_multi_criterion_flags(
    place_detector: PlaceFieldDetector,
    place_fields: PlaceFields,
    stability_split_half: NDArray[np.float32],
    configuration: PlaceFieldDetectionConfiguration,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]]:
    """Computes the IS_STABLE / IS_PEAK_SIGNIFICANT per-cell flags and their per-cell shuffle p-values.

    Notes:
        The Peak-method null distribution is generated by re-running threshold-based detection on the same shuffles
        already used by the Peak shuffle in ``compute_multi_criterion_significance``. The Stability null is generated
        by circularly shifting each
        trial's rate map by a random bin offset and recomputing the split-half correlation. The p-values report the
        per-cell rank of the observed statistic within its shuffled distribution and are NaN where the observed value
        is NaN or the shuffle distribution is empty.

    References:
        - Climer & Dombeck (2021). Choice of method of place cell classification determines the population of cells
          identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835 -- Peak method (99th percentile)
          and Stability method (95th percentile).

    Args:
        place_detector: PlaceFieldDetector instance used to access shuffle infrastructure and pooled rate map.
        place_fields: PlaceFields output containing the pooled and per-trial rate maps.
        stability_split_half: Observed per-cell split-half Pearson r from ``_compute_stability_metrics``.
        configuration: Place-field configuration carrying ``shuffle_repeat_count`` and ``peak_percentile``.

    Returns:
        A tuple of (is_stable, is_peak_significant, stability_p_values, peak_p_values), each with length cell_count.
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
    sce_results: list,
    *,
    stability_even_odd: NDArray[np.float32],
    stability_split_half: NDArray[np.float32],
    is_stable: NDArray[np.bool_],
    is_peak_significant: NDArray[np.bool_],
    stability_p_values: NDArray[np.float32],
    peak_p_values: NDArray[np.float32],
    is_strict_place: NDArray[np.bool_],
) -> pl.DataFrame:
    """Assembles the per-cell wide-format DataFrame from the live detector outputs.

    Args:
        cell_count: Total number of cells in the session.
        place_fields: PlaceFields output from PlaceFieldDetector.detect.
        reward_results: RewardCellResults from RewardCellDetector.detect.
        sce_results: List of SCEResult instances from SCEDetector.
        stability_even_odd: Per-cell Pearson r between even-trial and odd-trial mean rate maps with length cell_count.
        stability_split_half: Per-cell Pearson r between first-half and second-half mean rate maps with length
            cell_count.
        is_stable: Per-cell boolean from the stability shuffle (Climer & Dombeck 2021 Stability method) with length
            cell_count.
        is_peak_significant: Per-cell boolean from the peak-method shuffle (Climer & Dombeck 2021 Peak method) with
            length cell_count.
        stability_p_values: Per-cell p-value from the stability shuffle with length cell_count.
        peak_p_values: Per-cell p-value from the peak shuffle with length cell_count.
        is_strict_place: Per-cell boolean equal to ``has_place_field & is_stable & is_peak_significant`` with length
            cell_count.

    Returns:
        Per-cell wide-format polars DataFrame with one row per cell.
    """
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
            CellAnalysisColumn.IS_REWARD_PROXIMAL.value: pl.Series(values=reward_results.is_zone, dtype=pl.Boolean),
            CellAnalysisColumn.IS_APPROACH.value: pl.Series(values=reward_results.is_approach, dtype=pl.Boolean),
            CellAnalysisColumn.IS_ZONE.value: pl.Series(values=reward_results.is_zone, dtype=pl.Boolean),
            CellAnalysisColumn.IS_DEPARTURE.value: pl.Series(values=reward_results.is_departure, dtype=pl.Boolean),
            CellAnalysisColumn.IS_REWARD_CELL.value: pl.Series(values=is_reward_cell, dtype=pl.Boolean),
            CellAnalysisColumn.IS_POSITION_GLM_SIGNIFICANT.value: pl.Series(
                values=reward_results.is_position_glm_significant, dtype=pl.Boolean
            ),
            CellAnalysisColumn.IS_RELIABLE.value: pl.Series(values=place_fields.has_place_field, dtype=pl.Boolean),
            CellAnalysisColumn.IS_STABLE.value: pl.Series(values=is_stable, dtype=pl.Boolean),
            CellAnalysisColumn.STABILITY_P_VALUE.value: pl.Series(values=stability_p_values, dtype=pl.Float32),
            CellAnalysisColumn.IS_PEAK_SIGNIFICANT.value: pl.Series(values=is_peak_significant, dtype=pl.Boolean),
            CellAnalysisColumn.PEAK_P_VALUE.value: pl.Series(values=peak_p_values, dtype=pl.Float32),
            CellAnalysisColumn.IS_STRICT_PLACE.value: pl.Series(values=is_strict_place, dtype=pl.Boolean),
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
            CellAnalysisColumn.SPATIAL_INFORMATION_Z.value: pl.Series(
                values=spatial.spatial_information_z, dtype=pl.Float32
            ),
            CellAnalysisColumn.SPATIAL_FDR_SURVIVED.value: pl.Series(values=spatial.fdr_survived, dtype=pl.Boolean),
            CellAnalysisColumn.SPATIAL_SPLIT_HALF_R.value: pl.Series(values=spatial.split_half_r, dtype=pl.Float32),
            CellAnalysisColumn.SPATIAL_INFORMATION_BITS.value: pl.Series(
                values=spatial.spatial_information, dtype=pl.Float32
            ),
            CellAnalysisColumn.SPATIAL_P_VALUE.value: pl.Series(values=spatial.p_values, dtype=pl.Float32),
            CellAnalysisColumn.STABILITY_EVEN_ODD.value: pl.Series(values=stability_even_odd, dtype=pl.Float32),
            CellAnalysisColumn.STABILITY_SPLIT_HALF.value: pl.Series(values=stability_split_half, dtype=pl.Float32),
            CellAnalysisColumn.REWARD_RELATIVITY_SCORE.value: pl.Series(
                values=spatial.reward_relativity_score, dtype=pl.Float32
            ),
            CellAnalysisColumn.CV_POSITION_PARTIAL_R2.value: pl.Series(
                values=reward_results.cv_position_partial_r2, dtype=pl.Float32
            ),
            CellAnalysisColumn.POSITION_GLM_P_VALUE.value: pl.Series(
                values=reward_results.position_glm_p_values, dtype=pl.Float32
            ),
            CellAnalysisColumn.SCE_PARTICIPATION_COUNT_REST.value: sce_columns["participation_count_rest"],
            CellAnalysisColumn.SCE_PARTICIPATION_COUNT_RUN.value: sce_columns["participation_count_run"],
            CellAnalysisColumn.SCE_PARTICIPATION_RATE_REST.value: sce_columns["participation_rate_rest"],
            CellAnalysisColumn.SCE_PARTICIPATION_RATE_RUN.value: sce_columns["participation_rate_run"],
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
            CellAnalysisColumn.SCE_PARTICIPATION_P_VALUE_REST.value: pl.Series(
                values=sce_columns["participation_p_value_rest"], dtype=pl.Float32
            ),
            CellAnalysisColumn.SCE_PARTICIPATION_P_VALUE_RUN.value: pl.Series(
                values=sce_columns["participation_p_value_run"], dtype=pl.Float32
            ),
            CellAnalysisColumn.IS_SCE_CELL_REST.value: pl.Series(
                values=sce_columns["is_sce_cell_rest"], dtype=pl.Boolean
            ),
            CellAnalysisColumn.IS_SCE_CELL_RUN.value: pl.Series(
                values=sce_columns["is_sce_cell_run"], dtype=pl.Boolean
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


def _aggregate_sce_columns(cell_count: int, sce_results: list) -> dict:
    """Computes per-cell SCE participation and recruitment-significance metrics across all detected periods.

    References:
        - Modol et al. (2020). Hippocampal hub neurons. Nat Commun.
          https://doi.org/10.1038/s41467-020-18432-6 -- per-cell SCE recruitment significance ("super-rich"
          cells); per-period p-values combined here via Fisher's method (Fisher 1925).
    """
    # noinspection PyTypeChecker
    participation: NDArray[np.int32] = np.zeros((2, cell_count), dtype=np.int32)
    # noinspection PyTypeChecker
    total_sces: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    # noinspection PyTypeChecker
    period_counter: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    sce_events: list[list[list[list[int]]]] = [[[] for _ in range(cell_count)] for _ in range(2)]
    p_value_lists: list[list[list[float]]] = [[[] for _ in range(cell_count)] for _ in range(2)]
    # noinspection PyTypeChecker
    is_sce_cell: NDArray[np.bool_] = np.zeros((2, cell_count), dtype=np.bool_)

    for result in sce_results:
        period = 0 if result.period_type == PeriodType.REST else 1
        period_index = int(period_counter[period])
        period_counter[period] += 1
        sce_count = int(np.max(result.sce_labels))
        total_sces[period] += sce_count

        # Per-period participation significance flags OR-merge across periods of the same type.
        is_sce_cell[period] |= result.is_sce_cell
        if not bool(np.all(np.isnan(result.participation_p_values))):
            for cell_index in range(cell_count):
                p_value = float(result.participation_p_values[cell_index])
                if not np.isnan(p_value):
                    p_value_lists[period][cell_index].append(p_value)

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
            for cell in participating_indices:
                sce_events[period][cell].append([period_index, sce_label])

    # noinspection PyTypeChecker
    rate: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    combined_p: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    for period in range(2):
        if total_sces[period] > 0:
            rate[period] = (participation[period] / total_sces[period]).astype(np.float32)
        for cell_index in range(cell_count):
            combined_p[period, cell_index] = _combine_pvalues_fisher(values=p_value_lists[period][cell_index])

    return {
        "participation_count_rest": participation[0],
        "participation_count_run": participation[1],
        "participation_rate_rest": rate[0],
        "participation_rate_run": rate[1],
        "sce_events_rest": sce_events[0],
        "sce_events_run": sce_events[1],
        "participation_p_value_rest": combined_p[0],
        "participation_p_value_run": combined_p[1],
        "is_sce_cell_rest": is_sce_cell[0],
        "is_sce_cell_run": is_sce_cell[1],
    }


_SCE_PERIODS_EMPTY_SCHEMA: dict[str, pl.DataType] = {
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
    SCEPeriodColumn.TRIAL_IDS.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_SIZE.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_WIDTH_SAMPLES.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_PEAK_COACTIVE.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_INTER_EVENT_INTERVALS_SAMPLES.value: pl.List(pl.Int32),
    SCEPeriodColumn.SCE_RATE_HZ.value: pl.Float32,
    SCEPeriodColumn.REACTIVATION_STRENGTHS.value: pl.List(pl.Float32),
    SCEPeriodColumn.MEAN_REACTIVATION_STRENGTH.value: pl.Float32,
    SCEPeriodColumn.REACTIVATING_SCE_FRACTION.value: pl.Float32,
}


def _build_sce_periods_table(
    sampling_rate_hz: float,
    results: list,
    cell_count: int,
    reactivation_columns: dict[int, dict[str, object]] | None = None,
) -> pl.DataFrame:
    """Assembles the per-period SCE feather, encoding the dense onset matrix as sparse cell/sample index lists
    and persisting the per-SCE descriptors and the population reactivation strength columns.

    Args:
        sampling_rate_hz: Sampling rate in Hz; constant across all rows.
        results: List of ``SCEResult`` instances in temporal session order.
        cell_count: Total number of cells in the session.
        reactivation_columns: Optional mapping from result index in ``results`` to a dict carrying
            ``{"reactivation_strengths": list[float], "mean_reactivation_strength": float,
            "reactivating_sce_fraction": float}``. Rows without an entry receive empty/NaN reactivation
            columns. Run-period rows always receive empty/NaN values because the templates are derived from
            run and evaluated on rest.

    Returns:
        A polars DataFrame following :class:`SCEPeriodColumn`.
    """
    if not results:
        return pl.DataFrame(schema=_SCE_PERIODS_EMPTY_SCHEMA)

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
    trial_ids_column: list[list[int]] = []
    sce_size_column: list[list[int]] = []
    sce_width_column: list[list[int]] = []
    sce_peak_column: list[list[int]] = []
    sce_inter_interval_column: list[list[int]] = []
    sce_rate_column: list[float] = []
    reactivation_strengths_column: list[list[float]] = []
    mean_reactivation_column: list[float] = []
    reactivating_fraction_column: list[float] = []

    rest_counter = 0
    run_counter = 0

    for result_index, result in enumerate(results):
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
        trial_ids_column.append([int(value) for value in result.trial_ids.tolist()])
        sce_size_column.append([int(value) for value in result.sce_size.tolist()])
        sce_width_column.append([int(value) for value in result.sce_width_samples.tolist()])
        sce_peak_column.append([int(value) for value in result.sce_peak_coactive.tolist()])
        sce_inter_interval_column.append([int(value) for value in result.sce_inter_event_intervals_samples.tolist()])
        sce_rate_column.append(float(result.sce_rate_hz))

        entry = reactivation_columns.get(result_index) if reactivation_columns is not None else None
        if entry is not None:
            strengths = list(entry.get("reactivation_strengths", []))  # type: ignore[arg-type]
            reactivation_strengths_column.append([float(value) for value in strengths])
            mean_reactivation_column.append(float(entry.get("mean_reactivation_strength", float("nan"))))  # type: ignore[arg-type]
            reactivating_fraction_column.append(float(entry.get("reactivating_sce_fraction", float("nan"))))  # type: ignore[arg-type]
        else:
            reactivation_strengths_column.append([])
            mean_reactivation_column.append(float("nan"))
            reactivating_fraction_column.append(float("nan"))

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
            SCEPeriodColumn.TRIAL_IDS.value: pl.Series(values=trial_ids_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_SIZE.value: pl.Series(values=sce_size_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_WIDTH_SAMPLES.value: pl.Series(values=sce_width_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_PEAK_COACTIVE.value: pl.Series(values=sce_peak_column, dtype=pl.List(pl.Int32)),
            SCEPeriodColumn.SCE_INTER_EVENT_INTERVALS_SAMPLES.value: pl.Series(
                values=sce_inter_interval_column, dtype=pl.List(pl.Int32)
            ),
            SCEPeriodColumn.SCE_RATE_HZ.value: pl.Series(values=sce_rate_column, dtype=pl.Float32),
            SCEPeriodColumn.REACTIVATION_STRENGTHS.value: pl.Series(
                values=reactivation_strengths_column, dtype=pl.List(pl.Float32)
            ),
            SCEPeriodColumn.MEAN_REACTIVATION_STRENGTH.value: pl.Series(
                values=mean_reactivation_column, dtype=pl.Float32
            ),
            SCEPeriodColumn.REACTIVATING_SCE_FRACTION.value: pl.Series(
                values=reactivating_fraction_column, dtype=pl.Float32
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


def _z_score_along_samples(activity: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Z-scores each row of ``activity`` along the sample axis and returns the (z, has_variance) pair.

    Cells with zero sample-axis variance get a zeroed row in the returned z-matrix and a False entry in the
    mask, so downstream consumers can either drop them or fall back to the all-zero contribution.
    """
    cell_mean = activity.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    cell_std = activity.std(axis=1, keepdims=True).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    has_variance: NDArray[np.bool_] = cell_std[:, 0] > 0
    safe_std = np.where(cell_std > 0.0, cell_std, np.float32(1.0))
    # noinspection PyTypeChecker
    z: NDArray[np.float32] = ((activity - cell_mean) / safe_std).astype(np.float32, copy=False)
    z[~has_variance, :] = np.float32(0.0)
    return z, has_variance


def _resolve_ica_shuffle_allocation(budget: int, shuffle_count: int) -> tuple[int, int]:
    """Splits a CPU budget between per-shuffle BLAS threads and concurrent shuffle workers.

    Notes:
        Mirrors the saturating allocator used by bleaching / cindra: each worker is filled to
        ``_ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE`` BLAS threads before a new parallel shuffle is spawned, the
        per-worker thread count is rounded down to a multiple of ``_ICA_BLAS_THREAD_MULTIPLE`` for clean
        allocation, and parallelism is reduced one worker at a time whenever the per-worker share would fall
        below ``_ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE``. Single-worker configurations collapse naturally: a
        budget of N with one shuffle returns ``(round_down_to_multiple(N), 1)`` and hands every thread to BLAS.

    Args:
        budget: Total CPU cores available after the system reservation, as returned by ``resolve_worker_count``.
        shuffle_count: Number of shuffle iterations to run.

    Returns:
        A tuple of ``(blas_threads_per_shuffle, parallel_shuffles)`` whose product never exceeds the budget.
    """
    if shuffle_count <= 1:
        return max(1, budget), 1
    max_at_preferred = max(1, budget // _ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE)
    parallel_shuffles = min(shuffle_count, max_at_preferred)
    raw_threads = budget // parallel_shuffles
    blas_threads = max(1, (raw_threads // _ICA_BLAS_THREAD_MULTIPLE) * _ICA_BLAS_THREAD_MULTIPLE)

    # Reduces parallelism one worker at a time until each worker clears the per-shuffle floor. Cannot reduce
    # below a single worker; with ``parallel_shuffles == 1`` the floor stops applying because the entire
    # budget collapses onto BLAS.
    while blas_threads < _ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE and parallel_shuffles > 1:
        parallel_shuffles -= 1
        raw_threads = budget // parallel_shuffles
        blas_threads = max(1, (raw_threads // _ICA_BLAS_THREAD_MULTIPLE) * _ICA_BLAS_THREAD_MULTIPLE)

    return blas_threads, parallel_shuffles


def _shuffle_max_eigenvalue(
    z: NDArray[np.float32],
    *,
    shuffle_count: int,
    rng: np.random.Generator,
    minimum_shift_samples: int,
    progress_description: str = "Assembly null shuffle",
    requested_workers: int = 0,
) -> NDArray[np.float32]:
    """Returns the shuffled-distribution maximum eigenvalue per shuffle for the cell-by-cell correlation
    matrix obtained after independent circular shifts of each cell's z-scored activity (Lopes-dos-Santos 2013
    ICA-CS null).

    Notes:
        Each iteration draws independent per-cell shifts and gathers the shifted z-matrix. The largest eigenvalue
        of ``(shuffled @ shuffled.T) / N`` is recovered via ARPACK Lanczos with ``k=1`` on a ``LinearOperator``
        whose ``matvec`` is ``shuffled @ (shuffled.T @ v) / N`` -- so the (cell_count, cell_count) correlation
        matrix is never materialised and the eigendecomp's ``O(cell_count^3)`` cost collapses to roughly
        ``Lanczos_iters * 2 * cell_count * sample_count``. With cell counts in the thousands this is one to two
        orders of magnitude faster than the previous ``np.linalg.eigvalsh`` call that computed every eigenvalue
        only to read the last one. Falls back to ``eigvalsh`` on the rare ARPACK convergence failure (degenerate
        spectrum) so the null distribution always has a defined value.

        The shuffles run concurrently on a ``ThreadPoolExecutor``: the saturating allocator
        :func:`_resolve_ica_shuffle_allocation` splits the resolved CPU budget into ``parallel_shuffles``
        workers each granted ``blas_threads_per_shuffle`` BLAS threads via
        ``threadpoolctl.threadpool_limits`` in the worker initializer. On 128-core hosts with the default
        constants the allocator returns ``(30, 4)``: four Lanczos shuffles run concurrently, each saturating
        thirty BLAS threads, for a total footprint of one hundred twenty cores. Per-shuffle shifts are
        pre-drawn from the input rng in serial order so the rng state advances exactly as in the previous
        sequential implementation -- downstream consumers that share the rng (FastICA in
        :func:`_detect_assemblies_ica_cs`) see no behavior change despite the parallelism.

    Args:
        z: Z-scored activity with dimensions (cell_count, sample_count).
        shuffle_count: Number of shuffle iterations.
        rng: Per-call numpy random generator used to draw per-cell shifts.
        minimum_shift_samples: Lower bound on the absolute circular shift per cell per shuffle.
        progress_description: tqdm progress-bar description.
        requested_workers: Optional cap on the total CPU budget passed to ``resolve_worker_count``. Non-positive
            values request all available cores after the default system reservation.
    """
    cell_count, sample_count = z.shape
    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.zeros(shuffle_count, dtype=np.float32)
    if cell_count == 0 or sample_count < _MINIMUM_OBSERVATIONS_FOR_VARIANCE or shuffle_count == 0:
        return output
    floor = max(1, int(minimum_shift_samples))
    ceil = max(floor + 1, sample_count - floor)
    # noinspection PyTypeChecker
    sample_index_arange: NDArray[np.int64] = np.arange(sample_count, dtype=np.int64)
    # noinspection PyTypeChecker
    cell_arange: NDArray[np.int64] = np.arange(cell_count, dtype=np.int64)[:, np.newaxis]
    inverse_sample_count = np.float32(1.0 / float(sample_count))

    # Pre-draws every shuffle's shift vector from the input rng in the same order a serial implementation
    # would consume them. The rng state after this call advances by ``shuffle_count * cell_count`` int64
    # draws -- identical to the previous per-iteration loop -- so downstream consumers (FastICA) sharing the
    # generator observe no behavior change.
    # noinspection PyTypeChecker
    all_shifts: NDArray[np.int64] = rng.integers(low=floor, high=ceil, size=(shuffle_count, cell_count)).astype(
        np.int64
    )

    total_budget = resolve_worker_count(requested_workers=requested_workers)
    blas_threads_per_shuffle, parallel_shuffles = _resolve_ica_shuffle_allocation(
        budget=total_budget,
        shuffle_count=shuffle_count,
    )

    def _one_shuffle(shuffle_index: int) -> tuple[int, float]:
        # noinspection PyTypeChecker
        gather: NDArray[np.int64] = (
            sample_index_arange[np.newaxis, :] - all_shifts[shuffle_index][:, np.newaxis]
        ) % sample_count
        shuffled = z[cell_arange, gather]
        return shuffle_index, _largest_zzt_eigenvalue(z=shuffled, inverse_sample_count=inverse_sample_count)

    if parallel_shuffles <= 1:
        # Serial fallback: hands the entire budget to BLAS so the single Lanczos call saturates the worker.
        with threadpool_limits(limits=blas_threads_per_shuffle):
            for shuffle_index in tqdm(
                range(shuffle_count),
                desc=progress_description,
                unit="iter",
                leave=False,
            ):
                _, eigval = _one_shuffle(shuffle_index)
                output[shuffle_index] = eigval
        return output

    def _init_worker() -> None:
        # Pins each worker's BLAS thread pool so the per-shuffle Lanczos matvec stays inside its slice of the
        # CPU budget instead of fighting other workers for cores. ``threadpool_limits`` is invoked at worker
        # startup so subsequent BLAS calls in that thread inherit the cap automatically.
        threadpool_limits(limits=blas_threads_per_shuffle)

    with ThreadPoolExecutor(max_workers=parallel_shuffles, initializer=_init_worker) as executor:
        futures = [executor.submit(_one_shuffle, index) for index in range(shuffle_count)]
        for future in tqdm(
            as_completed(futures),
            total=shuffle_count,
            desc=progress_description,
            unit="iter",
            leave=False,
        ):
            shuffle_index, eigval = future.result()
            output[shuffle_index] = eigval
    return output


def _largest_zzt_eigenvalue(
    z: NDArray[np.float32],
    inverse_sample_count: float | np.float32,
) -> float:
    """Returns the largest eigenvalue of ``(z @ z.T) / sample_count`` without materialising the correlation
    matrix.

    Notes:
        Uses ARPACK Lanczos via ``scipy.sparse.linalg.eigsh`` with ``k=1`` and a matrix-free ``LinearOperator``
        whose ``matvec`` is ``z @ (z.T @ v) * inverse_sample_count``. Each Lanczos iteration costs two BLAS
        matvecs (``O(cell_count * sample_count)``); typical convergence is a few dozen iterations, dwarfing the
        ``O(cell_count^3)`` cost of a full eigendecomposition for large cell counts. Falls back to a full
        ``np.linalg.eigvalsh`` call on the rare ``ArpackNoConvergence`` (degenerate spectrum) so callers always
        receive a defined value.
    """
    cell_count, sample_count = z.shape
    if cell_count == 0 or sample_count == 0:
        return 0.0

    def matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        # noinspection PyTypeChecker
        projected: NDArray[np.float32] = z.T @ vector.astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        result: NDArray[np.float32] = z @ projected
        return (result.astype(np.float64, copy=False)) * float(inverse_sample_count)

    operator = LinearOperator(shape=(cell_count, cell_count), matvec=matvec, dtype=np.float64)
    try:
        eigvals = eigsh(operator, k=1, which="LA", tol=1e-3, return_eigenvectors=False)
        return float(eigvals[0])
    except ArpackNoConvergence as failure:
        # ARPACK occasionally fails to converge on degenerate spectra; reuse whatever Ritz values it produced
        # before falling back to a full dense decomposition so the null still has a defined entry.
        if failure.eigenvalues.size > 0:
            return float(np.max(failure.eigenvalues.real))
        correlation = (z @ z.T) * float(inverse_sample_count)
        return float(np.linalg.eigvalsh(correlation)[-1])


def _fast_ica_deflation(
    whitened: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    maximum_iterations: int = 200,
    tolerance: float = 1e-4,
) -> NDArray[np.float64]:
    """Runs deflation FastICA with the ``tanh`` non-linearity on a whitened ``(n_components, n_samples)``
    matrix and returns the unmixing matrix ``W`` with shape ``(n_components, n_components)``.

    Notes:
        Implements the standard one-component-at-a-time fixed-point iteration of Hyvärinen 1999 with
        Gram-Schmidt deflation. ``whitened`` must already have unit-variance, decorrelated rows (the caller
        whitens via the significant-PC eigendecomposition).
    """
    n_components, n_samples = whitened.shape
    # noinspection PyTypeChecker
    unmixing: NDArray[np.float64] = np.zeros((n_components, n_components), dtype=np.float64)
    for component_index in range(n_components):
        # noinspection PyTypeChecker
        candidate: NDArray[np.float64] = rng.standard_normal(n_components).astype(np.float64, copy=False)
        candidate /= np.linalg.norm(candidate) + 1e-12
        # Project out previously-found components.
        for previous in range(component_index):
            candidate -= float(candidate @ unmixing[previous]) * unmixing[previous]
        candidate /= np.linalg.norm(candidate) + 1e-12

        for _ in range(maximum_iterations):
            projection = candidate @ whitened
            g_value = np.tanh(projection)
            g_derivative = np.float64(1.0) - g_value * g_value
            updated = (whitened @ g_value) / float(n_samples) - g_derivative.mean() * candidate
            for previous in range(component_index):
                updated -= float(updated @ unmixing[previous]) * unmixing[previous]
            updated /= np.linalg.norm(updated) + 1e-12

            cos_similarity = float(np.abs(updated @ candidate))
            candidate = updated
            if abs(cos_similarity - 1.0) < tolerance:
                break

        unmixing[component_index] = candidate
    return unmixing


def _detect_assemblies_ica_cs(
    activity_matrix: NDArray[np.bool_] | NDArray[np.float32],
    *,
    shuffle_count: int = 200,
    eigenvalue_significance_percentile: float = 99.0,
    membership_z_threshold: float = 2.0,
    minimum_assembly_size: int = 3,
    minimum_shift_samples: int = 5,
    rng_seed: int = 0,
) -> tuple[NDArray[np.float32], list[NDArray[np.int32]]]:
    """Detects neural assemblies via the Lopes-dos-Santos 2013 ICA-CS pipeline.

    Notes:
        Z-scores activity along the sample axis, computes the cell-by-cell correlation matrix, retains
        principal components whose eigenvalues exceed the configured percentile of a circular-shift null
        distribution, whitens the data via these PCs, runs deflation FastICA, sign-corrects each independent
        component so its peak weight is positive, and reports cells whose weight magnitudes exceed the
        configured z-threshold as assembly members. Outperforms hierarchical Jaccard clustering on calcium
        imaging benchmarks (Mölter, Avitan & Goodhill 2018, BMC Biol).

    References:
        - Lopes-dos-Santos, Ribeiro & Tort (2013). Detecting cell assemblies in large neuronal populations.
          J Neurosci Methods. https://doi.org/10.1016/j.jneumeth.2013.04.010 -- the ICA-CS algorithm.
        - Mölter, Avitan & Goodhill (2018). Detecting neural assemblies in calcium imaging data. BMC Biol.
          https://doi.org/10.1186/s12915-018-0606-4 -- comparative benchmark recommending ICA-CS over
          hierarchical clustering.
        - Hyvärinen (1999). Fast and robust fixed-point algorithms for ICA. IEEE Trans Neural Netw.
          https://doi.org/10.1109/72.761722 -- deflation FastICA fixed-point iteration used here.

    Args:
        activity_matrix: Cell-by-time-bin activity matrix. Boolean inputs are cast to float32; float inputs
            are used as-is. Each column is one observation (time bin or SCE event).
        shuffle_count: Number of circular-shift shuffles for the eigenvalue significance test.
        eigenvalue_significance_percentile: Percentile cutoff applied to the shuffled max-eigenvalue
            distribution.
        membership_z_threshold: Per-component z-threshold (in units of the component-weight standard
            deviation) above which a cell is reported as an assembly member.
        minimum_assembly_size: Drop assemblies with fewer than this many member cells.
        minimum_shift_samples: Lower bound on the absolute circular shift per cell per shuffle.
        rng_seed: Seed for the per-call numpy random generator.

    Returns:
        A tuple of (templates, member_lists) where ``templates`` has shape ``(n_assemblies, cell_count)`` in
        the original cell-index space (zero-padded for cells that contributed no variance) and
        ``member_lists`` is a list of length ``n_assemblies`` each containing the member cell indices.
    """
    cell_count = int(activity_matrix.shape[0])
    rng = np.random.default_rng(seed=rng_seed)

    # noinspection PyTypeChecker
    activity: NDArray[np.float32] = np.asarray(activity_matrix, dtype=np.float32)
    if activity.shape[1] < _MINIMUM_OBSERVATIONS_FOR_VARIANCE:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    z, has_variance = _z_score_along_samples(activity=activity)
    active_cell_indices = np.where(has_variance)[0].astype(np.int32)
    if active_cell_indices.size < minimum_assembly_size:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    z_active = z[active_cell_indices, :]
    sample_count = z_active.shape[1]

    correlation = (z_active @ z_active.T) / float(sample_count)
    eigvals_all, eigvecs_all = np.linalg.eigh(correlation)

    shuffled_max = _shuffle_max_eigenvalue(
        z=z_active,
        shuffle_count=shuffle_count,
        rng=rng,
        minimum_shift_samples=minimum_shift_samples,
        progress_description="ICA-CS template shuffle",
    )
    threshold = float(np.percentile(shuffled_max, eigenvalue_significance_percentile))
    # noinspection PyTypeChecker
    significant_mask: NDArray[np.bool_] = eigvals_all > threshold
    if not significant_mask.any():
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    # noinspection PyTypeChecker
    significant_eigvals: NDArray[np.float64] = eigvals_all[significant_mask].astype(np.float64, copy=False)
    significant_eigvecs = eigvecs_all[:, significant_mask].astype(np.float64, copy=False)

    # Whitener carries the data into a unit-variance, decorrelated PC subspace as the FastICA prerequisite.
    whitener = significant_eigvecs / np.sqrt(significant_eigvals)[np.newaxis, :]
    projected = whitener.T @ z_active.astype(np.float64, copy=False)

    unmixing = _fast_ica_deflation(whitened=projected, rng=rng)

    # Templates in active-cell space: V_active = whitener @ W.T -> (active_cells, n_components)
    templates_active = (whitener @ unmixing.T).astype(np.float32, copy=False)

    # Sign-correct so each template's largest-magnitude weight is positive.
    for component_index in range(templates_active.shape[1]):
        peak_index = int(np.argmax(np.abs(templates_active[:, component_index])))
        if templates_active[peak_index, component_index] < 0.0:
            templates_active[:, component_index] *= np.float32(-1.0)

    # Lift back to the full cell-index space.
    # noinspection PyTypeChecker
    templates: NDArray[np.float32] = np.zeros((templates_active.shape[1], cell_count), dtype=np.float32)
    templates[:, active_cell_indices] = templates_active.T

    member_lists: list[NDArray[np.int32]] = []
    surviving: list[NDArray[np.float32]] = []
    for component_index in range(templates.shape[0]):
        # Compute the membership threshold from the active-cell weights only so silent cells do not deflate
        # the per-component standard deviation toward zero.
        active_weights = templates[component_index, active_cell_indices]
        weight_std = float(np.std(active_weights))
        if weight_std == 0.0:
            continue
        # noinspection PyTypeChecker
        member_mask: NDArray[np.bool_] = templates[component_index] > membership_z_threshold * weight_std
        # noinspection PyTypeChecker
        member_indices: NDArray[np.int32] = np.where(member_mask)[0].astype(np.int32)
        if member_indices.size < minimum_assembly_size:
            continue
        member_lists.append(member_indices)
        surviving.append(templates[component_index])

    if not surviving:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []
    # noinspection PyTypeChecker
    surviving_templates: NDArray[np.float32] = np.stack(surviving, axis=0).astype(np.float32, copy=False)
    return surviving_templates, member_lists


def _per_sce_reactivation_strength(
    z: NDArray[np.float32],
    templates: NDArray[np.float32],
    sce_labels: NDArray[np.int32],
    sce_count: int,
    *,
    templates_squared: NDArray[np.float32] | None = None,
    z_squared: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Returns the per-SCE mean diagonal-removed reactivation strength under the Peyrache 2009 / Lopes-dos-
    Santos 2013 quadratic projection ``R(t) = sum_k ((P_k @ z(:,t))^2 - sum_i P_k[i]^2 * z(i,t)^2)``.

    Notes:
        Reformulates the Peyrache / Lopes-dos-Santos quadratic projection as two BLAS GEMMs so the per-sample
        strength becomes a vectorised reduction. Lets ``A = templates @ z`` and ``B = (templates ** 2) @ (z ** 2)``;
        the per-sample strength is ``(A ** 2 - B).sum(axis=0)``. Aggregating across SCEs is one ``np.bincount``
        with weights. Replaces the previous numba-kernel implementation that walked
        ``sample_count * n_assemblies * cell_count`` strided loads on a column-major access pattern (z is
        C-contiguous (cell, sample)) and accumulated per-SCE sums via a non-atomic
        ``sums[label - 1] += per_sample_strength`` scatter under ``prange`` -- the new path is roughly an
        order of magnitude faster and is race-free.

        ``templates_squared`` and ``z_squared`` are optional pre-computed elementwise squares. Inside the
        shuffle loop the templates are shuffle-invariant so ``templates_squared`` is hoisted once per period;
        ``z_squared`` is also reused per shuffle because the circular-shift gather commutes with elementwise
        squaring (``shuffled_z ** 2 == shuffled(z ** 2)`` when both use identical per-cell shifts), so the
        caller pre-computes ``z * z`` once per period and gathers it with the same indices used for ``z``.

    Args:
        z: Z-scored activity with dimensions (cell_count, sample_count).
        templates: Assembly templates with dimensions (n_assemblies, cell_count).
        sce_labels: Per-sample SCE labels (1-indexed; 0 outside any SCE) with length sample_count.
        sce_count: Number of detected SCEs (the maximum value in ``sce_labels``).
        templates_squared: Optional pre-computed ``templates * templates`` array; when provided the per-call
            elementwise squaring is skipped. Useful inside shuffle loops where templates are invariant.
        z_squared: Optional pre-computed ``z * z`` array (same shape and dtype as ``z``). When provided the
            per-call ``z ** 2`` materialisation -- the dominant elementwise allocation at the typical
            ``cell_count * sample_count`` of a rest period -- is skipped.

    Returns:
        Per-SCE mean reactivation strength with length sce_count; entries for SCEs whose sample slice is empty
        remain at zero.
    """
    sample_count = z.shape[1]
    n_assemblies = templates.shape[0]
    # noinspection PyTypeChecker
    means: NDArray[np.float32] = np.zeros(sce_count, dtype=np.float32)
    if sce_count == 0 or n_assemblies == 0 or sample_count == 0:
        return means

    # noinspection PyTypeChecker
    projection: NDArray[np.float32] = (templates @ z).astype(np.float32, copy=False)
    if templates_squared is None:
        # noinspection PyTypeChecker
        templates_squared = (templates * templates).astype(np.float32, copy=False)
    if z_squared is None:
        # noinspection PyTypeChecker
        z_squared = (z * z).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    diagonal: NDArray[np.float32] = (templates_squared @ z_squared).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    per_sample: NDArray[np.float32] = (projection * projection - diagonal).sum(axis=0).astype(
        np.float32, copy=False
    )

    # ``np.bincount(sce_labels, weights=per_sample, minlength=sce_count + 1)`` puts samples whose label is 0
    # into bin 0 (which we discard) and accumulates per_sample[t] into bin label[t] for valid SCEs. Counts use
    # the same minlength so the per-SCE divisor is computed without a separate pass over the labels.
    minlength = sce_count + 1
    # noinspection PyTypeChecker
    sums: NDArray[np.float64] = np.bincount(sce_labels, weights=per_sample, minlength=minlength)[1:]
    # noinspection PyTypeChecker
    counts: NDArray[np.int64] = np.bincount(sce_labels, minlength=minlength)[1:]
    nonempty = counts > 0
    if np.any(nonempty):
        # noinspection PyTypeChecker
        means[nonempty] = (sums[nonempty] / counts[nonempty]).astype(np.float32, copy=False)
    return means


def _build_run_template_matrix(
    *,
    run_results: list,
    cell_count: int,
    shuffle_count: int,
    eigenvalue_significance_percentile: float,
    minimum_shift_samples: int,
    rng_seed: int,
) -> NDArray[np.float32]:
    """Concatenates RUN-period onset matrices column-wise and runs ICA-CS to derive the population-assembly
    templates used for rest-period reactivation projection.
    """
    if not run_results:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32)

    # noinspection PyTypeChecker
    run_concat: NDArray[np.float32] = np.concatenate(
        [result.onset_matrix.astype(np.float32, copy=False) for result in run_results], axis=1
    )
    if run_concat.shape[1] < max(2, minimum_shift_samples * 2):
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32)

    templates, _ = _detect_assemblies_ica_cs(
        activity_matrix=run_concat,
        shuffle_count=shuffle_count,
        eigenvalue_significance_percentile=eigenvalue_significance_percentile,
        minimum_shift_samples=minimum_shift_samples,
        rng_seed=rng_seed,
    )
    return templates


def _compute_population_reactivation(
    *,
    run_results: list,
    rest_results: list,
    cell_count: int,
    shuffle_count: int = 200,
    template_eigenvalue_significance_percentile: float = 99.0,
    sce_significance_percentile: float = 95.0,
    minimum_shift_samples: int = 5,
    rng_seed: int = 0,
) -> tuple[NDArray[np.float32], dict[int, dict[str, object]]]:
    """Computes per-SCE population reactivation strength of REST SCEs against templates derived from the
    concatenated RUN onset matrix (Peyrache 2009 / Lopes-dos-Santos 2013).

    Notes:
        Templates are derived once from RUN onsets via ICA-CS; each REST period's onsets are then
        z-scored along the sample axis and projected onto the templates via the diagonal-removed quadratic
        form so within-cell autocorrelation does not inflate the strength. A per-SCE significance flag is
        derived from a per-period circular-shuffle null on the same projection.

    Args:
        run_results: List of RUN-period ``SCEResult`` instances supplying the onset matrix used for template
            estimation.
        rest_results: List of REST-period ``SCEResult`` instances onto which the templates are projected.
        cell_count: Total number of cells in the session.
        shuffle_count: Number of shuffle iterations both for template selection and for the per-SCE null.
        template_eigenvalue_significance_percentile: Percentile cutoff for selecting significant principal
            components during template estimation.
        sce_significance_percentile: Percentile of the per-SCE shuffle null at which an SCE is flagged as
            reactivating.
        minimum_shift_samples: Lower bound on the absolute circular shift per cell per shuffle (template
            estimation and per-SCE null).
        rng_seed: Master seed for both the template-estimation rng and the per-period shuffle rngs.

    Returns:
        Tuple of (templates, per_period_dict). ``templates`` has shape ``(n_assemblies, cell_count)``;
        ``per_period_dict`` maps result index in ``rest_results`` (also indexable as the position of the rest
        result inside the combined results list when consumed by ``_build_sce_periods_table``) to a dict
        carrying ``reactivation_strengths`` (per-SCE), ``mean_reactivation_strength`` (period mean), and
        ``reactivating_sce_fraction`` (per-period significance fraction).
    """
    templates = _build_run_template_matrix(
        run_results=run_results,
        cell_count=cell_count,
        shuffle_count=shuffle_count,
        eigenvalue_significance_percentile=template_eigenvalue_significance_percentile,
        minimum_shift_samples=minimum_shift_samples,
        rng_seed=rng_seed,
    )
    if templates.shape[0] == 0 or not rest_results:
        return templates, {}

    master_rng = np.random.default_rng(seed=rng_seed + 1)
    period_seeds = master_rng.integers(low=0, high=np.iinfo(np.int64).max, size=len(rest_results))

    # Pre-computes ``templates ** 2`` once across every per-period reactivation call. Templates are
    # shuffle-invariant (and period-invariant) so the elementwise square is hoisted out of the hot path.
    # noinspection PyTypeChecker
    templates_squared: NDArray[np.float32] = (templates * templates).astype(np.float32, copy=False)

    total_budget = resolve_worker_count(requested_workers=0)
    blas_threads_per_shuffle, parallel_shuffles = _resolve_ica_shuffle_allocation(
        budget=total_budget,
        shuffle_count=shuffle_count,
    )

    def _init_reactivation_worker() -> None:
        # Same per-worker BLAS-thread pin used by ``_shuffle_max_eigenvalue``: each ICA-CS reactivation
        # shuffle does two GEMMs (templates @ z and templates_squared @ z_squared), so capping BLAS to its
        # share of the CPU budget prevents 4x oversubscription while still saturating each worker's matvec.
        threadpool_limits(limits=blas_threads_per_shuffle)

    per_period: dict[int, dict[str, object]] = {}
    rest_iterator = tqdm(
        list(enumerate(rest_results)),
        desc="Reactivation per rest period",
        unit="period",
        leave=False,
    )
    for period_position, result in rest_iterator:
        sce_count = int(np.max(result.sce_labels)) if result.sce_labels.size > 0 else 0
        if sce_count == 0:
            continue
        # noinspection PyTypeChecker
        rest_activity: NDArray[np.float32] = result.onset_matrix.astype(np.float32, copy=False)
        z, _ = _z_score_along_samples(activity=rest_activity)
        # Pre-computes ``z * z`` once per period. Circular shifts commute with elementwise squaring, so
        # gathering ``z_squared`` with the same per-cell shifts used for ``z`` yields ``shuffled_z ** 2``
        # without paying the per-shuffle elementwise allocation (the dominant non-GEMM allocation at
        # ``cell_count * sample_count`` of a rest period).
        # noinspection PyTypeChecker
        z_squared_period: NDArray[np.float32] = (z * z).astype(np.float32, copy=False)
        sce_labels_int32 = result.sce_labels.astype(np.int32, copy=False)

        observed = _per_sce_reactivation_strength(
            z=z,
            templates=templates,
            sce_labels=sce_labels_int32,
            sce_count=sce_count,
            templates_squared=templates_squared,
            z_squared=z_squared_period,
        )

        # Per-SCE null: independent circular shifts of each cell, recompute reactivation strength.
        period_rng = np.random.default_rng(seed=int(period_seeds[period_position]))
        sample_count = rest_activity.shape[1]
        floor = max(1, int(minimum_shift_samples))
        ceil = max(floor + 1, sample_count - floor)
        # noinspection PyTypeChecker
        sample_index_arange: NDArray[np.int64] = np.arange(sample_count, dtype=np.int64)
        # noinspection PyTypeChecker
        cell_arange: NDArray[np.int64] = np.arange(cell_count, dtype=np.int64)[:, np.newaxis]
        # noinspection PyTypeChecker
        null_strengths: NDArray[np.float32] = np.zeros((shuffle_count, sce_count), dtype=np.float32)

        # Pre-draws every shuffle's shift vector from the period rng in serial order so the rng state advances
        # exactly as the previous sequential implementation -- preserves bit-for-bit reproducibility against
        # legacy reports.
        # noinspection PyTypeChecker
        all_shifts: NDArray[np.int64] = period_rng.integers(
            low=floor, high=ceil, size=(shuffle_count, cell_count)
        ).astype(np.int64)

        def _one_reactivation_shuffle(shuffle_index: int) -> tuple[int, NDArray[np.float32]]:
            # noinspection PyTypeChecker
            gather: NDArray[np.int64] = (
                sample_index_arange[np.newaxis, :] - all_shifts[shuffle_index][:, np.newaxis]
            ) % sample_count
            shuffled_z = z[cell_arange, gather]
            shuffled_z_squared = z_squared_period[cell_arange, gather]
            strengths = _per_sce_reactivation_strength(
                z=shuffled_z,
                templates=templates,
                sce_labels=sce_labels_int32,
                sce_count=sce_count,
                templates_squared=templates_squared,
                z_squared=shuffled_z_squared,
            )
            return shuffle_index, strengths

        # No inner per-shuffle progress bar -- the outer ``Reactivation per rest period`` bar already advances
        # at period granularity, and at typical shuffle throughputs the inner bar emits hundreds of updates per
        # period that flood non-TTY consoles (PyCharm Run windows, log files) without adding actionable
        # information.
        if parallel_shuffles <= 1:
            with threadpool_limits(limits=blas_threads_per_shuffle):
                for shuffle_index in range(shuffle_count):
                    _, strengths = _one_reactivation_shuffle(shuffle_index)
                    null_strengths[shuffle_index] = strengths
        else:
            with ThreadPoolExecutor(
                max_workers=parallel_shuffles,
                initializer=_init_reactivation_worker,
            ) as executor:
                futures = [
                    executor.submit(_one_reactivation_shuffle, index) for index in range(shuffle_count)
                ]
                for future in as_completed(futures):
                    shuffle_index, strengths = future.result()
                    null_strengths[shuffle_index] = strengths

        # noinspection PyTypeChecker
        sce_threshold: NDArray[np.float32] = np.percentile(null_strengths, sce_significance_percentile, axis=0).astype(
            np.float32, copy=False
        )
        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = observed > sce_threshold
        per_period[period_position] = {
            "reactivation_strengths": observed.tolist(),
            "mean_reactivation_strength": float(np.mean(observed)),
            "reactivating_sce_fraction": float(np.mean(is_significant)),
        }

    return templates, per_period


def _active_significance_columns(
    *,
    require_stable: bool,
    require_peak_significant: bool,
    table: pl.DataFrame,
) -> list[tuple[str, CellAnalysisColumn]]:
    """Returns the active p-value-bearing criteria as (display label, p-value column) pairs in canonical order.

    Notes:
        Skips criteria whose p-value column is missing from the table, so plots remain functional on legacy
        reports persisted before the multi-criterion p-value columns were added.
    """
    candidates: list[tuple[bool, str, CellAnalysisColumn]] = [
        (require_stable, "Stable", CellAnalysisColumn.STABILITY_P_VALUE),
        (require_peak_significant, "Peak", CellAnalysisColumn.PEAK_P_VALUE),
    ]
    return [(label, column) for active, label, column in candidates if active and column.value in table.columns]


def _make_heatmap_figure(strip_count: int, figure_dpi: int) -> plt.Figure:
    """Allocates a figure sized to leave room for the requested number of significance strips."""
    base_width = 8.0
    extra_width = 0.45 * strip_count
    return plt.figure(figsize=(base_width + extra_width, 4), facecolor="white", dpi=figure_dpi)


def _layout_heatmap_axes(
    figure: plt.Figure,
    strip_count: int,
    *,
    include_colorbar: bool,
) -> tuple[list[plt.Axes], plt.Axes, plt.Axes | None]:
    """Builds the gridspec layout for a heatmap with optional left-side significance strips and a right-side colorbar.

    Returns:
        A tuple of (strip_axes, main_axes, colorbar_axes_or_None).
    """
    width_ratios: list[float] = [_PLACE_STRIP_WIDTH_RATIO] * strip_count + [1.0]
    if include_colorbar:
        width_ratios.append(0.05)
    grid = figure.add_gridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.08)
    strip_axes = [figure.add_subplot(grid[0, i]) for i in range(strip_count)]
    main_axes = figure.add_subplot(grid[0, strip_count])
    colorbar_axes = figure.add_subplot(grid[0, strip_count + 1]) if include_colorbar else None
    return strip_axes, main_axes, colorbar_axes


def _render_significance_strips(
    figure: plt.Figure,
    strip_axes: list[plt.Axes],
    strip_columns: list[tuple[str, CellAnalysisColumn]],
    table: pl.DataFrame,
    ordered_indices: NDArray[np.int64],
) -> None:
    """Renders one ``-log10(p)`` strip per (label, column) entry in ``strip_columns`` alongside the main heatmap.

    Notes:
        Cells with NaN p-values render at the floor (treated as p=1.0). The color scale spans 0 to ``-log10`` of
        :data:`_PLACE_PVALUE_DISPLAY_FLOOR`, which keeps very-significant outliers from compressing the visible range
        for the rest of the population. The leftmost strip carries the ``-log₁₀(p)`` y-axis label and a small
        horizontal colorbar at the bottom encodes the standard significance landmarks (p=1, 0.05, 0.01, 0.001).
    """
    if not strip_axes or not strip_columns:
        return
    floor = _PLACE_PVALUE_DISPLAY_FLOOR
    vmax = float(-np.log10(floor))
    cell_count = int(ordered_indices.size)
    image = None
    for axis, (label, column) in zip(strip_axes, strip_columns, strict=True):
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = table[column.value].to_numpy().astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        ordered_p: NDArray[np.float32] = np.empty(0, dtype=np.float32) if cell_count == 0 else p_values[ordered_indices]
        # noinspection PyTypeChecker
        clipped: NDArray[np.float32] = np.clip(ordered_p, floor, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # noinspection PyTypeChecker
            neg_log_p: NDArray[np.float32] = (-np.log10(clipped)).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        cleaned: NDArray[np.float32] = np.where(np.isnan(ordered_p), 0.0, neg_log_p).astype(np.float32, copy=False)
        column_data = cleaned.reshape(-1, 1) if cleaned.size > 0 else np.zeros((1, 1), dtype=np.float32)
        image = axis.imshow(
            column_data,
            cmap="Reds",
            extent=(0.0, 1.0, float(max(cell_count, 1)), 0.0),
            vmin=0.0,
            vmax=vmax,
            interpolation="none",
            origin="upper",
            aspect="auto",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(label, fontsize=7, rotation=0, labelpad=2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("0.5")
    strip_axes[0].set_ylabel("-log₁₀(p)", fontsize=7)
    if image is not None:
        # Inset a thin horizontal colorbar at the bottom of the leftmost strip with significance landmarks.
        anchor = strip_axes[0].get_position()
        bar_height = 0.02
        bar_axes = figure.add_axes(
            (anchor.x0, anchor.y0 - bar_height - 0.04, anchor.width * len(strip_axes), bar_height)
        )
        color_bar = figure.colorbar(image, cax=bar_axes, orientation="horizontal")
        landmark_p_values = [1.0, 0.05, 0.01, 0.001]
        # noinspection PyTypeChecker
        landmark_ticks: list[float] = [float(-np.log10(max(p, floor))) for p in landmark_p_values if p >= floor]
        color_bar.set_ticks(landmark_ticks)
        color_bar.set_ticklabels(
            [f"{p:g}" for p in landmark_p_values if p >= floor],
            fontsize=6,
        )
        color_bar.ax.tick_params(length=2, pad=1)
        color_bar.set_label("p", fontsize=6, labelpad=2)


def _compose_population_label(
    *,
    require_place: bool,
    require_stable: bool,
    require_peak_significant: bool,
) -> str:
    """Returns ``"Place ∩ Stable ∩ Peak"``-style labels from active criterion bools; ``"All cells"`` when all False."""
    parts: list[str] = []
    if require_place:
        parts.append("Place")
    if require_stable:
        parts.append("Stable")
    if require_peak_significant:
        parts.append("Peak")
    return " ∩ ".join(parts) if parts else "All cells"


def _resolve_dataset_place_metric_extractor(
    *,
    require_place: bool,
    require_stable: bool,
    require_peak_significant: bool,
    mutually_exclusive: bool = True,
) -> Callable[[CellAnalysisSummary], float]:
    """Returns a per-summary fraction extractor for the requested criterion combination, using precomputed summary
    fields when available.

    Notes:
        Single-criterion combinations and the default strict triple-AND each have a precomputed count in
        :class:`CellAnalysisSummary`. With ``mutually_exclusive=True``, only the IS_PLACE-only and strict-triple
        combinations have cached "_only_count" companions (cells flagged IS_PLACE but not IS_REWARD_CELL); other
        multi-criterion combinations are not currently cached in the summary and would require loading each
        session's per-cell feather. This helper raises in that case so callers know to either request a cached
        combination or extend the summary schema.
    """
    bools = (require_place, require_stable, require_peak_significant)
    if bools == (True, True, True):
        if mutually_exclusive:
            return lambda summary: summary.strict_place_only_count / summary.cell_count if summary.cell_count else 0.0
        return lambda summary: summary.strict_place_cell_count / summary.cell_count if summary.cell_count else 0.0
    if bools == (True, False, False):
        if mutually_exclusive:
            return lambda summary: summary.place_only_count / summary.cell_count if summary.cell_count else 0.0
        return lambda summary: summary.place_cell_count / summary.cell_count if summary.cell_count else 0.0
    if bools == (False, True, False):
        return lambda summary: summary.stable_count / summary.cell_count if summary.cell_count else 0.0
    if bools == (False, False, True):
        return lambda summary: summary.peak_significant_count / summary.cell_count if summary.cell_count else 0.0
    message = (
        "Unable to plot a dataset-level place-cell fraction for the requested criterion combination. The summary "
        f"caches counts for single criteria and for the strict triple-AND only; got "
        f"(require_place={require_place}, require_stable={require_stable}, "
        f"require_peak_significant={require_peak_significant}). Either pass one of the cached combinations or "
        "extend CellAnalysisSummary to cache the requested combination."
    )
    console.error(message=message, error=ValueError)
    # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
    # noinspection PyUnreachableCode
    raise ValueError(message)  # pragma: no cover


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
