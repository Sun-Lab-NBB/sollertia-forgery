"""Provides functionality for quantifying cross-session tuning drift in chronic two-photon imaging datasets.

The drift pipeline operates **per animal** because the upstream multi-day cindra registration pipeline issues
cell IDs that are stable across an animal's sessions but unrelated across animals. Per-animal session loading,
classification trajectories, and rate-map / peak / COM stability metrics are computed here from the persisted
`TuningReport` artifacts of every session in scope. Methodological references for the drift pipeline live
on `..drift_report.compute_drift_report`.

The protocol consolidates four canonical drift signals:

* Per-cell classification trajectories and consecutive-pair recurrence probability (Ziv 2013 / Hainmueller &
  Bartos 2018 / Mau 2018) for the place-, reward-, and strict-place classifications.
* Per-cell tuning-curve correlation across sessions (Krishnan & Sheffield 2024 / Hainmueller & Bartos 2018) and
  per-cell peak / COM shift distributions with Fisher-combined random-remapping p-values reusing the existing
  `..tuning.utilities.random_remapping_peak_shift_p_values` cell-ID-shuffle null.
* Population-vector correlation as a function of session lag (Sheintuch 2023 / Climer, Davoudi, Oh & Dombeck
  2025) with an exponential-decay fit that summarizes the global drift timescale.
* Per-cell baseline-fluorescence trends and per-session photobleaching flags from the saved `BleachingReport`,
  used to gate stability claims so peak instability that is actually a slow-baseline artifact is not
  misattributed to representational drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit

from .utilities import session_pair_indices, boolean_recurrence_probability
from ..tuning.utilities import per_cell_pearson_safe, random_remapping_peak_shift_p_values

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .utilities import AnimalSessionData


_FISHER_CLIP_FLOOR: float = 1e-10
"""Lower clip applied before the ``log`` step in Fisher's combined-p-value formula. Cells whose per-pair
random-remapping null returned ``p == 0`` would otherwise diverge to infinity."""

_MINIMUM_SESSIONS_FOR_DECAY_FIT: int = 3
"""Minimum number of unique session-lag bins required to fit the population-vector decay model."""

_MINIMUM_FLATTENED_PV_SAMPLES: int = 2
"""Minimum number of paired non-NaN entries required across the two sessions' flattened rate-map vectors
before a population-vector Pearson r is computed. Below this threshold the per-pair PV correlation falls
through to the NaN sentinel."""


@dataclass(frozen=True, slots=True)
class DriftDetectionConfiguration:
    """Defines configuration parameters for cross-session tuning-drift quantification."""

    persistence_fraction_threshold: float = 0.5
    """Fraction-of-sessions threshold above which a cell is flagged as "persistently classified" for a given
    classification (place / reward / strict_place). Default mirrors the Hainmueller & Bartos (2018) "stable
    place cell" cut at 50% of sessions."""
    rate_map_correlation_threshold: float = 0.3
    """Per-cell mean rate-map Pearson r threshold above which a place / reward cell is treated as having a
    stable tuning curve. The 0.3 default is the field-standard cutoff for "stable place fields" from Krishnan
    & Sheffield (2024)."""
    peak_shift_significance_threshold: float = 0.05
    """Significance threshold against which the Fisher-combined random-remapping peak-shift p-value is
    compared. Cells whose combined p-value falls below this threshold are flagged as having significantly
    stable peaks."""
    peak_shift_shuffle_count: int = 1000
    """Number of cell-ID permutations per session pair used by the random-remapping peak-shift null. Aligned
    with the default ``shuffle_count`` of `..tuning.utilities.random_remapping_peak_shift_p_values`."""
    peak_shift_random_seed: int = 42
    """Base seed forwarded to the random-remapping peak-shift null. Each session pair iteration adds an offset
    so the per-pair null streams stay reproducibly distinct."""
    bleaching_high_drift_quartile: float = 0.75
    """Quartile of the absolute per-cell baseline-slope distribution above which a cell is flagged as a
    "high-bleaching-drift" candidate. The 75th-percentile default matches the convention used by
    `..bleaching.bleaching_analysis._compute_flag_masks` for per-cell SNR stratification."""
    use_bleaching_session_mask: bool = True
    """When True, sessions whose persisted `BleachingReport` flag is True are excluded from the per-cell
    drift aggregates. When False, every session contributes to the aggregates regardless of bleaching status.
    The persisted classification trajectory always carries the unmasked flags so the choice is reversible."""
    minimum_pair_count: int = 1
    """Minimum number of unmasked session pairs required for a per-cell aggregate to be reported. Cells with
    fewer surviving pairs receive NaN aggregates; the threshold defaults to one because every cell needs at
    least one pair to compute a Pearson r at all."""
    minimum_classification_sessions: int = 2
    """Minimum number of unmasked sessions required for a per-cell trajectory metric (recurrence probability,
    persistence flag) to be reported. Cells with fewer surviving sessions receive NaN / False aggregates."""

    def __post_init__(self) -> None:
        """Validates that every threshold is in its admissible range so downstream consumers can trust the values."""
        if not 0.0 < self.persistence_fraction_threshold <= 1.0:
            message = (
                f"persistence_fraction_threshold must lie in (0, 1], got "
                f"{self.persistence_fraction_threshold!r}."
            )
            raise ValueError(message)
        if not -1.0 <= self.rate_map_correlation_threshold <= 1.0:
            message = (
                f"rate_map_correlation_threshold must lie in [-1, 1], got "
                f"{self.rate_map_correlation_threshold!r}."
            )
            raise ValueError(message)
        if not 0.0 < self.peak_shift_significance_threshold < 1.0:
            message = (
                f"peak_shift_significance_threshold must lie in (0, 1), got "
                f"{self.peak_shift_significance_threshold!r}."
            )
            raise ValueError(message)
        if self.peak_shift_shuffle_count <= 0:
            message = f"peak_shift_shuffle_count must be positive, got {self.peak_shift_shuffle_count!r}."
            raise ValueError(message)
        if not 0.0 < self.bleaching_high_drift_quartile < 1.0:
            message = (
                f"bleaching_high_drift_quartile must lie in (0, 1), got "
                f"{self.bleaching_high_drift_quartile!r}."
            )
            raise ValueError(message)
        if self.minimum_pair_count <= 0:
            message = f"minimum_pair_count must be positive, got {self.minimum_pair_count!r}."
            raise ValueError(message)
        if self.minimum_classification_sessions <= 0:
            message = (
                f"minimum_classification_sessions must be positive, got "
                f"{self.minimum_classification_sessions!r}."
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class PairwiseDriftMetrics:
    """Stores per-session-pair drift metrics produced by `compute_pairwise_drift_metrics`.

    All arrays use ``pair_count`` along the first axis where ``pair_count = session_count * (session_count -
    1) // 2``. Per-cell columns are 2-D (pair_count, cell_count) and use NaN to encode "metric undefined for
    this cell on this pair" (e.g., the cell was not classified in either session).
    """

    pair_a_indices: NDArray[np.int64]
    """First session index of each ordered pair with ``a < b``, length ``pair_count``."""
    pair_b_indices: NDArray[np.int64]
    """Second session index of each ordered pair, length ``pair_count``."""
    lag_days: NDArray[np.float32]
    """Per-pair calendar-day lag computed as ``days[b] - days[a]``, length ``pair_count``."""
    population_vector_correlation: NDArray[np.float32]
    """Per-pair population-vector Pearson r between the two sessions' per-cell rate-map matrices flattened to
    cell-by-bin vectors. Shape ``(pair_count,)``."""
    place_recurrence_count: NDArray[np.int32]
    """Per-pair count of cells classified place in both sessions, shape ``(pair_count,)``."""
    reward_recurrence_count: NDArray[np.int32]
    """Per-pair count of cells classified reward in both sessions, shape ``(pair_count,)``."""
    strict_place_recurrence_count: NDArray[np.int32]
    """Per-pair count of cells classified strict-place in both sessions, shape ``(pair_count,)``."""
    rate_map_correlation_per_cell: NDArray[np.float32]
    """Per-pair per-cell rate-map Pearson r with shape ``(pair_count, cell_count)``. NaN for cells whose rate
    map has zero variance in either session."""
    peak_shift_cm_per_cell: NDArray[np.float32]
    """Per-pair per-cell |peak_b - peak_a| in centimeters with shape ``(pair_count, cell_count)``. NaN for
    cells whose peak is undefined in either session."""
    com_shift_cm_per_cell: NDArray[np.float32]
    """Per-pair per-cell |COM_b - COM_a| in centimeters with shape ``(pair_count, cell_count)``. NaN for
    cells whose COM is undefined in either session."""
    peak_shift_p_value_per_cell: NDArray[np.float32]
    """Per-pair per-cell random-remapping peak-shift p-value from
    `..tuning.utilities.random_remapping_peak_shift_p_values`, shape ``(pair_count, cell_count)``. NaN for
    cells with undefined peaks in either session."""


@dataclass(frozen=True, slots=True)
class CellDriftMetrics:
    """Stores per-cell aggregate drift metrics produced by `compute_per_cell_drift_metrics`.

    Every array has length ``cell_count`` aligned with the multi-day-registered cell ordering used by the
    upstream tuning reports. Boolean classifications include the bleaching-cleaned variants used by the report
    summary.
    """

    pair_count_per_cell: NDArray[np.int32]
    """Number of session pairs that contributed at least one finite per-cell metric, length ``cell_count``."""
    place_session_count: NDArray[np.int32]
    """Number of unmasked sessions where the cell was flagged ``IS_PLACE``, length ``cell_count``."""
    reward_session_count: NDArray[np.int32]
    """Number of unmasked sessions where the cell was flagged ``IS_REWARD_CELL``, length ``cell_count``."""
    strict_place_session_count: NDArray[np.int32]
    """Number of unmasked sessions where the cell was flagged ``IS_STRICT_PLACE``, length ``cell_count``."""
    place_session_fraction: NDArray[np.float32]
    """Fraction of unmasked sessions where the cell was flagged ``IS_PLACE``, length ``cell_count``."""
    reward_session_fraction: NDArray[np.float32]
    """Fraction of unmasked sessions where the cell was flagged ``IS_REWARD_CELL``, length ``cell_count``."""
    strict_place_session_fraction: NDArray[np.float32]
    """Fraction of unmasked sessions where the cell was flagged ``IS_STRICT_PLACE``, length ``cell_count``."""
    place_recurrence_probability: NDArray[np.float32]
    """Per-cell consecutive-pair P(place_{t+1} | place_t), length ``cell_count``. NaN when the conditioning
    event never fires."""
    reward_recurrence_probability: NDArray[np.float32]
    """Per-cell consecutive-pair P(reward_{t+1} | reward_t), length ``cell_count``."""
    strict_place_recurrence_probability: NDArray[np.float32]
    """Per-cell consecutive-pair P(strict_place_{t+1} | strict_place_t), length ``cell_count``."""
    is_persistent_place: NDArray[np.bool_]
    """True for cells whose ``place_session_fraction`` exceeds ``persistence_fraction_threshold``."""
    is_persistent_reward: NDArray[np.bool_]
    """True for cells whose ``reward_session_fraction`` exceeds ``persistence_fraction_threshold``."""
    is_persistent_strict_place: NDArray[np.bool_]
    """True for cells whose ``strict_place_session_fraction`` exceeds ``persistence_fraction_threshold``."""
    mean_rate_map_correlation: NDArray[np.float32]
    """Per-cell mean Pearson r of rate maps across surviving session pairs."""
    mean_consecutive_rate_map_correlation: NDArray[np.float32]
    """Per-cell mean Pearson r of rate maps restricted to consecutive session pairs (lag = neighboring
    sessions). More sensitive to short-timescale drift than the full pairwise mean."""
    mean_peak_shift_cm: NDArray[np.float32]
    """Per-cell mean absolute peak shift in centimeters across surviving session pairs."""
    mean_com_shift_cm: NDArray[np.float32]
    """Per-cell mean absolute COM shift in centimeters across surviving session pairs."""
    fisher_peak_shift_p_value: NDArray[np.float32]
    """Per-cell Fisher-combined random-remapping peak-shift p-value across surviving session pairs."""
    is_peak_stable: NDArray[np.bool_]
    """True for cells whose ``fisher_peak_shift_p_value`` falls below
    ``peak_shift_significance_threshold``."""
    is_field_stable: NDArray[np.bool_]
    """True for cells whose ``mean_rate_map_correlation`` exceeds
    ``rate_map_correlation_threshold``."""
    cell_baseline_slope: NDArray[np.float32]
    """Per-cell linear baseline-fluorescence slope in fluorescence units per day, length ``cell_count``. NaN
    when bleaching cross-correlation is unavailable for this animal."""
    is_high_bleaching_drift: NDArray[np.bool_]
    """True for cells whose absolute baseline slope exceeds the configured high-drift quartile of the per-cell
    distribution. Always False when bleaching cross-correlation is unavailable."""
    is_stably_tuned_place: NDArray[np.bool_]
    """Composite flag — True for cells that are persistently classified place AND have a stable rate-map
    correlation AND have stable peaks AND are not flagged as high-bleaching-drift candidates."""
    is_stably_tuned_reward: NDArray[np.bool_]
    """Composite flag — True for cells that are persistently classified reward AND have a stable rate-map
    correlation AND are not flagged as high-bleaching-drift candidates."""
    is_stably_tuned_strict_place: NDArray[np.bool_]
    """Composite flag — True for cells that are persistently classified strict-place AND have a stable
    rate-map correlation AND have stable peaks AND are not flagged as high-bleaching-drift candidates."""


@dataclass(frozen=True, slots=True)
class PopulationDriftMetrics:
    """Stores animal-level lag-binned population-vector drift metrics and the decay fit."""

    lag_days: NDArray[np.float32]
    """Per-pair calendar-day lag, length ``pair_count``. Mirrors `PairwiseDriftMetrics.lag_days` for
    convenience."""
    population_vector_correlation: NDArray[np.float32]
    """Per-pair population-vector Pearson r, length ``pair_count``. Mirrors
    `PairwiseDriftMetrics.population_vector_correlation`."""
    decay_amplitude: float
    """Decaying-component amplitude of the ``amplitude * exp(-lag / tau_days) + offset`` model fit to
    ``population_vector_correlation`` versus ``lag_days``. NaN when the fit failed."""
    decay_tau_days: float
    """Decay time constant in days. NaN when the fit failed."""
    decay_offset: float
    """Asymptotic correlation offset. NaN when the fit failed."""
    decay_fit_succeeded: bool
    """True when ``scipy.optimize.curve_fit`` converged on a finite, in-bounds solution."""


def compute_pairwise_drift_metrics(
    animal_data: AnimalSessionData,
    *,
    configuration: DriftDetectionConfiguration,
    rate_map_session_mask: NDArray[np.bool_],
) -> PairwiseDriftMetrics:
    """Computes every per-session-pair drift metric for an animal.

    Notes:
        Per-pair Pearson r between rate maps reuses `..tuning.utilities.per_cell_pearson_safe`, which is
        NaN-safe and zero-variance-safe — cells with all-zero rate maps in either session contribute NaN to
        the pairwise tables. The peak-shift random-remapping null reuses
        `..tuning.utilities.random_remapping_peak_shift_p_values` once per pair with a per-pair seed offset
        so the null streams stay reproducibly distinct. The bleaching session mask is applied at the per-cell
        aggregation step, not here, so the persisted pair table contains every chronologically defined pair
        and the mask choice remains reversible.

    Args:
        animal_data: Per-animal session matrices produced by `..utilities.load_animal_session_data`.
        configuration: Drift configuration carrying the random-remapping shuffle parameters.
        rate_map_session_mask: Per-session boolean mask indicating which sessions are eligible for inclusion
            in the per-cell rate-map / peak / COM aggregates. The pairwise table itself is built across every
            pair regardless of this mask; the mask is propagated as NaN entries on a per-cell-per-pair basis
            when either pair member is masked out.

    Returns:
        A PairwiseDriftMetrics with per-pair per-cell arrays ready for the per-cell aggregation step.
    """
    session_count = animal_data.rate_maps.shape[0]
    cell_count = animal_data.cell_count

    a_indices, b_indices = session_pair_indices(session_count=session_count)
    pair_count = int(a_indices.size)

    # noinspection PyTypeChecker
    lag_days: NDArray[np.float32] = np.zeros(pair_count, dtype=np.float32)
    # noinspection PyTypeChecker
    pv_correlation: NDArray[np.float32] = np.full(pair_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    place_recurrence: NDArray[np.int32] = np.zeros(pair_count, dtype=np.int32)
    # noinspection PyTypeChecker
    reward_recurrence: NDArray[np.int32] = np.zeros(pair_count, dtype=np.int32)
    # noinspection PyTypeChecker
    strict_place_recurrence: NDArray[np.int32] = np.zeros(pair_count, dtype=np.int32)
    # noinspection PyTypeChecker
    rate_map_corr: NDArray[np.float32] = np.full((pair_count, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    peak_shift: NDArray[np.float32] = np.full((pair_count, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    com_shift: NDArray[np.float32] = np.full((pair_count, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    peak_shift_p: NDArray[np.float32] = np.full((pair_count, cell_count), np.nan, dtype=np.float32)

    for pair_index in range(pair_count):
        a_index = int(a_indices[pair_index])
        b_index = int(b_indices[pair_index])
        lag_days[pair_index] = float(
            animal_data.days_since_first[b_index] - animal_data.days_since_first[a_index]
        )

        rate_map_a = animal_data.rate_maps[a_index]
        rate_map_b = animal_data.rate_maps[b_index]
        # ``per_cell_pearson_safe`` returns NaN for cells with zero variance or fewer than three valid bins,
        # which is exactly the desired "metric undefined" sentinel for the cross-session correlation.
        rate_map_corr[pair_index, :] = per_cell_pearson_safe(
            first_matrix=rate_map_a, second_matrix=rate_map_b
        )

        # Population-vector correlation: flatten the (cell, bin) matrices and compute a single Pearson r.
        # Skip the pair when fewer than two non-NaN entries align across the flattened vectors.
        flat_a = rate_map_a.reshape(-1)
        flat_b = rate_map_b.reshape(-1)
        # noinspection PyTypeChecker
        valid_mask: NDArray[np.bool_] = np.isfinite(flat_a) & np.isfinite(flat_b)
        if int(np.sum(valid_mask)) >= _MINIMUM_FLATTENED_PV_SAMPLES:
            valid_a = flat_a[valid_mask].astype(np.float64, copy=False)
            valid_b = flat_b[valid_mask].astype(np.float64, copy=False)
            valid_a -= float(np.mean(valid_a))
            valid_b -= float(np.mean(valid_b))
            denominator = float(np.sqrt(np.sum(valid_a * valid_a) * np.sum(valid_b * valid_b)))
            if denominator > 0.0:
                pv_correlation[pair_index] = float(np.sum(valid_a * valid_b) / denominator)

        # Recurrence counts are computed against the unmasked classification flags. Bleaching masking enters
        # only at the per-cell aggregation level; the persisted pair counts always reflect the full set.
        place_recurrence[pair_index] = int(
            np.sum(np.logical_and(animal_data.is_place[a_index], animal_data.is_place[b_index]))
        )
        reward_recurrence[pair_index] = int(
            np.sum(np.logical_and(animal_data.is_reward_cell[a_index], animal_data.is_reward_cell[b_index]))
        )
        strict_place_recurrence[pair_index] = int(
            np.sum(
                np.logical_and(
                    animal_data.is_strict_place[a_index], animal_data.is_strict_place[b_index]
                )
            )
        )

        # Per-cell shifts default to NaN; populate only where both endpoints are finite. The peak / COM peaks
        # come pre-projected to NaN by the loader for cells whose upstream rate map vanishes or whose COM
        # gate is invalid.
        peaks_a = animal_data.peak_positions_cm[a_index]
        peaks_b = animal_data.peak_positions_cm[b_index]
        # noinspection PyTypeChecker
        peak_valid: NDArray[np.bool_] = np.isfinite(peaks_a) & np.isfinite(peaks_b)
        peak_shift[pair_index, peak_valid] = np.abs(peaks_b[peak_valid] - peaks_a[peak_valid])

        coms_a = animal_data.centers_of_mass_cm[a_index]
        coms_b = animal_data.centers_of_mass_cm[b_index]
        # noinspection PyTypeChecker
        com_valid: NDArray[np.bool_] = np.isfinite(coms_a) & np.isfinite(coms_b)
        com_shift[pair_index, com_valid] = np.abs(coms_b[com_valid] - coms_a[com_valid])

        # Per-pair random-remapping peak-shift null. Reuses the existing helper with a per-pair seed offset
        # so the null streams stay reproducibly distinct across pairs while remaining deterministic.
        peak_shift_p[pair_index, :] = random_remapping_peak_shift_p_values(
            peaks_a=peaks_a,
            peaks_b=peaks_b,
            shuffle_count=configuration.peak_shift_shuffle_count,
            seed=configuration.peak_shift_random_seed + pair_index,
        )

        # Apply the per-session bleaching mask by NaNing out per-cell entries on pairs touching a flagged
        # session. The unmasked recurrence and PV-correlation entries stay populated so cross-session
        # population diagnostics remain available even when bleaching gating is on.
        if not bool(rate_map_session_mask[a_index]) or not bool(rate_map_session_mask[b_index]):
            rate_map_corr[pair_index, :] = np.nan
            peak_shift[pair_index, :] = np.nan
            com_shift[pair_index, :] = np.nan
            peak_shift_p[pair_index, :] = np.nan

    return PairwiseDriftMetrics(
        pair_a_indices=a_indices,
        pair_b_indices=b_indices,
        lag_days=lag_days,
        population_vector_correlation=pv_correlation,
        place_recurrence_count=place_recurrence,
        reward_recurrence_count=reward_recurrence,
        strict_place_recurrence_count=strict_place_recurrence,
        rate_map_correlation_per_cell=rate_map_corr,
        peak_shift_cm_per_cell=peak_shift,
        com_shift_cm_per_cell=com_shift,
        peak_shift_p_value_per_cell=peak_shift_p,
    )


def compute_per_cell_drift_metrics(
    animal_data: AnimalSessionData,
    pairwise: PairwiseDriftMetrics,
    *,
    configuration: DriftDetectionConfiguration,
    classification_session_mask: NDArray[np.bool_],
    cell_baseline_slope: NDArray[np.float32],
) -> CellDriftMetrics:
    """Aggregates per-session-pair drift metrics into per-cell scalar metrics and stability classifications.

    Notes:
        Classification fractions and consecutive-pair recurrence probabilities are computed against the
        bleaching-masked per-session classification trajectory; the unmasked trajectory is persisted on the
        cells feather as list columns for downstream re-analysis. The Fisher-combined p-value uses the chi-2
        survival function with degrees of freedom equal to ``2 * pair_count`` per cell, dropping NaN per-pair
        p-values from the combination so cells with mixed valid / invalid pairs are still scored. The
        composite ``is_stably_tuned_*`` flags follow the multi-criterion convention of Krishnan & Sheffield
        (2024): persistent classification AND stable tuning curve AND (where peaks are well-defined) stable
        peaks AND not flagged as a high-bleaching-drift candidate.

    Args:
        animal_data: Per-animal session matrices.
        pairwise: Pairwise drift metrics produced by `compute_pairwise_drift_metrics`.
        configuration: Drift configuration carrying the threshold values.
        classification_session_mask: Per-session boolean mask indicating which sessions contribute to the
            classification trajectory aggregates. Sessions where the mask is False are excluded from the
            ``*_session_count`` and ``*_recurrence_probability`` totals but stay available on the persisted
            per-session list columns.
        cell_baseline_slope: Per-cell linear baseline-fluorescence slope in fluorescence units per day,
            length ``cell_count``. NaN where bleaching cross-correlation is unavailable.

    Returns:
        A `CellDriftMetrics` ready for serialization into ``drift_cells.feather``.
    """
    cell_count = animal_data.cell_count

    if cell_count == 0:
        return _empty_cell_drift_metrics()

    masked_place = animal_data.is_place[classification_session_mask, :]
    masked_reward = animal_data.is_reward_cell[classification_session_mask, :]
    masked_strict_place = animal_data.is_strict_place[classification_session_mask, :]

    # noinspection PyTypeChecker
    place_session_count: NDArray[np.int32] = np.sum(masked_place, axis=0).astype(np.int32, copy=False)
    # noinspection PyTypeChecker
    reward_session_count: NDArray[np.int32] = np.sum(masked_reward, axis=0).astype(np.int32, copy=False)
    # noinspection PyTypeChecker
    strict_place_session_count: NDArray[np.int32] = np.sum(masked_strict_place, axis=0).astype(
        np.int32, copy=False
    )
    masked_session_count = int(masked_place.shape[0])

    # noinspection PyTypeChecker
    place_session_fraction: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    reward_session_fraction: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    strict_place_session_fraction: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if masked_session_count >= configuration.minimum_classification_sessions:
        place_session_fraction = (place_session_count / masked_session_count).astype(np.float32, copy=False)
        reward_session_fraction = (reward_session_count / masked_session_count).astype(np.float32, copy=False)
        strict_place_session_fraction = (
            strict_place_session_count / masked_session_count
        ).astype(np.float32, copy=False)

    # noinspection PyTypeChecker
    place_recurrence: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    reward_recurrence: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    strict_place_recurrence: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if masked_session_count >= configuration.minimum_classification_sessions:
        for cell_index in range(cell_count):
            place_recurrence[cell_index] = boolean_recurrence_probability(
                per_session_flag=masked_place[:, cell_index]
            )
            reward_recurrence[cell_index] = boolean_recurrence_probability(
                per_session_flag=masked_reward[:, cell_index]
            )
            strict_place_recurrence[cell_index] = boolean_recurrence_probability(
                per_session_flag=masked_strict_place[:, cell_index]
            )

    # noinspection PyTypeChecker
    is_persistent_place: NDArray[np.bool_] = (
        place_session_fraction >= configuration.persistence_fraction_threshold
    ) & np.isfinite(place_session_fraction)
    # noinspection PyTypeChecker
    is_persistent_reward: NDArray[np.bool_] = (
        reward_session_fraction >= configuration.persistence_fraction_threshold
    ) & np.isfinite(reward_session_fraction)
    # noinspection PyTypeChecker
    is_persistent_strict_place: NDArray[np.bool_] = (
        strict_place_session_fraction >= configuration.persistence_fraction_threshold
    ) & np.isfinite(strict_place_session_fraction)

    # Per-cell pair-level aggregates. ``np.nanmean`` over an all-NaN column emits a RuntimeWarning, so guard
    # the reduction with a finite-pair count.
    rate_map_corr = pairwise.rate_map_correlation_per_cell  # (pair_count, cell_count)
    peak_shifts = pairwise.peak_shift_cm_per_cell
    com_shifts = pairwise.com_shift_cm_per_cell
    peak_p_values = pairwise.peak_shift_p_value_per_cell

    # noinspection PyTypeChecker
    pair_count_per_cell: NDArray[np.int32] = np.sum(np.isfinite(rate_map_corr), axis=0).astype(
        np.int32, copy=False
    )
    # noinspection PyTypeChecker
    mean_rate_map_correlation: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    mean_peak_shift: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    mean_com_shift: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    fisher_p: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)

    for cell_index in range(cell_count):
        if int(pair_count_per_cell[cell_index]) >= configuration.minimum_pair_count:
            # noinspection PyTypeChecker
            cell_correlations: NDArray[np.float32] = rate_map_corr[:, cell_index]
            cell_correlations_finite = cell_correlations[np.isfinite(cell_correlations)]
            if cell_correlations_finite.size > 0:
                mean_rate_map_correlation[cell_index] = float(np.mean(cell_correlations_finite))
            cell_peak_shifts = peak_shifts[:, cell_index]
            cell_peak_shifts_finite = cell_peak_shifts[np.isfinite(cell_peak_shifts)]
            if cell_peak_shifts_finite.size > 0:
                mean_peak_shift[cell_index] = float(np.mean(cell_peak_shifts_finite))
            cell_com_shifts = com_shifts[:, cell_index]
            cell_com_shifts_finite = cell_com_shifts[np.isfinite(cell_com_shifts)]
            if cell_com_shifts_finite.size > 0:
                mean_com_shift[cell_index] = float(np.mean(cell_com_shifts_finite))
            cell_peak_p = peak_p_values[:, cell_index]
            cell_peak_p_finite = cell_peak_p[np.isfinite(cell_peak_p)]
            if cell_peak_p_finite.size > 0:
                fisher_p[cell_index] = _combine_pvalues_fisher(values=cell_peak_p_finite)

    # Consecutive-pair rate-map correlation. ``b == a + 1`` selects the chronologically adjacent pairs out of
    # the upper-triangular pair list.
    a_indices = pairwise.pair_a_indices
    b_indices = pairwise.pair_b_indices
    # noinspection PyTypeChecker
    consecutive_mask: NDArray[np.bool_] = (b_indices - a_indices) == 1
    # noinspection PyTypeChecker
    mean_consecutive_corr: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if int(np.sum(consecutive_mask)) > 0:
        consecutive_corr_block = rate_map_corr[consecutive_mask, :]
        for cell_index in range(cell_count):
            cell_consecutive = consecutive_corr_block[:, cell_index]
            finite = cell_consecutive[np.isfinite(cell_consecutive)]
            if finite.size > 0:
                mean_consecutive_corr[cell_index] = float(np.mean(finite))

    # noinspection PyTypeChecker
    is_field_stable: NDArray[np.bool_] = (
        mean_rate_map_correlation >= configuration.rate_map_correlation_threshold
    ) & np.isfinite(mean_rate_map_correlation)
    # noinspection PyTypeChecker
    is_peak_stable: NDArray[np.bool_] = (
        fisher_p < configuration.peak_shift_significance_threshold
    ) & np.isfinite(fisher_p)

    # Bleaching gating: cells whose absolute baseline slope is in the high-drift quartile are flagged. The
    # mask collapses to all-False when bleaching is unavailable so the ``is_stably_tuned_*`` flags reduce to
    # the bleaching-agnostic intersection.
    # noinspection PyTypeChecker
    abs_slopes: NDArray[np.float32] = np.abs(cell_baseline_slope)
    finite_slope_mask = np.isfinite(abs_slopes)
    if bool(np.any(finite_slope_mask)):
        threshold = float(
            np.quantile(abs_slopes[finite_slope_mask], configuration.bleaching_high_drift_quartile)
        )
        # noinspection PyTypeChecker
        is_high_bleaching_drift: NDArray[np.bool_] = (
            (abs_slopes > threshold) & finite_slope_mask
        )
    else:
        # noinspection PyTypeChecker
        is_high_bleaching_drift = np.zeros(cell_count, dtype=np.bool_)

    is_bleaching_clean = ~is_high_bleaching_drift

    # noinspection PyTypeChecker
    is_stably_tuned_place: NDArray[np.bool_] = (
        is_persistent_place & is_field_stable & is_peak_stable & is_bleaching_clean
    )
    # noinspection PyTypeChecker
    is_stably_tuned_reward: NDArray[np.bool_] = (
        is_persistent_reward & is_field_stable & is_bleaching_clean
    )
    # noinspection PyTypeChecker
    is_stably_tuned_strict_place: NDArray[np.bool_] = (
        is_persistent_strict_place & is_field_stable & is_peak_stable & is_bleaching_clean
    )

    return CellDriftMetrics(
        pair_count_per_cell=pair_count_per_cell,
        place_session_count=place_session_count,
        reward_session_count=reward_session_count,
        strict_place_session_count=strict_place_session_count,
        place_session_fraction=place_session_fraction,
        reward_session_fraction=reward_session_fraction,
        strict_place_session_fraction=strict_place_session_fraction,
        place_recurrence_probability=place_recurrence,
        reward_recurrence_probability=reward_recurrence,
        strict_place_recurrence_probability=strict_place_recurrence,
        is_persistent_place=is_persistent_place,
        is_persistent_reward=is_persistent_reward,
        is_persistent_strict_place=is_persistent_strict_place,
        mean_rate_map_correlation=mean_rate_map_correlation,
        mean_consecutive_rate_map_correlation=mean_consecutive_corr,
        mean_peak_shift_cm=mean_peak_shift,
        mean_com_shift_cm=mean_com_shift,
        fisher_peak_shift_p_value=fisher_p,
        is_peak_stable=is_peak_stable,
        is_field_stable=is_field_stable,
        cell_baseline_slope=cell_baseline_slope.astype(np.float32, copy=False),
        is_high_bleaching_drift=is_high_bleaching_drift,
        is_stably_tuned_place=is_stably_tuned_place,
        is_stably_tuned_reward=is_stably_tuned_reward,
        is_stably_tuned_strict_place=is_stably_tuned_strict_place,
    )


def fit_population_decay(
    lag_days: NDArray[np.float32],
    population_vector_correlation: NDArray[np.float32],
) -> PopulationDriftMetrics:
    """Fits ``amplitude * exp(-lag / tau_days) + offset`` to per-pair PV correlation versus session lag.

    Notes:
        Mirrors `..bleaching.bleaching_analysis._fit_exponential_decay` but operates on the cross-session
        population-vector correlation versus calendar lag (Climer et al. 2025): a stable representation has a
        slow tau (large) and a high asymptotic offset; a drifting representation has a fast tau and a low
        asymptotic offset. Returns the failure sentinel when fewer than three unique lag bins are available
        or the fit does not converge.

    Args:
        lag_days: Per-pair calendar-day lag, length ``pair_count``.
        population_vector_correlation: Per-pair population-vector Pearson r, length ``pair_count``.

    Returns:
        A `PopulationDriftMetrics` carrying the per-pair arrays plus the fitted parameters and a
        ``decay_fit_succeeded`` flag.
    """
    finite_mask = np.isfinite(lag_days) & np.isfinite(population_vector_correlation)
    if int(np.sum(finite_mask)) < _MINIMUM_SESSIONS_FOR_DECAY_FIT or int(
        np.unique(lag_days[finite_mask].astype(np.float64)).size
    ) < _MINIMUM_SESSIONS_FOR_DECAY_FIT:
        return PopulationDriftMetrics(
            lag_days=lag_days,
            population_vector_correlation=population_vector_correlation,
            decay_amplitude=float("nan"),
            decay_tau_days=float("nan"),
            decay_offset=float("nan"),
            decay_fit_succeeded=False,
        )

    days = lag_days[finite_mask].astype(np.float64, copy=False)
    correlations = population_vector_correlation[finite_mask].astype(np.float64, copy=False)
    initial_amplitude = float(correlations.max() - correlations.min())
    span_days = float(days.max() - days.min())
    initial_tau = max(span_days / 2.0, 1e-3)
    initial_offset = float(correlations.min())

    try:
        parameters, _ = curve_fit(
            f=_pv_decay_model,
            xdata=days,
            ydata=correlations,
            p0=(initial_amplitude, initial_tau, initial_offset),
            bounds=((-np.inf, 1e-6, -np.inf), (np.inf, np.inf, np.inf)),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return PopulationDriftMetrics(
            lag_days=lag_days,
            population_vector_correlation=population_vector_correlation,
            decay_amplitude=float("nan"),
            decay_tau_days=float("nan"),
            decay_offset=float("nan"),
            decay_fit_succeeded=False,
        )

    amplitude, tau_days, offset = (float(parameter) for parameter in parameters)
    if not (np.isfinite(amplitude) and np.isfinite(tau_days) and np.isfinite(offset)):
        return PopulationDriftMetrics(
            lag_days=lag_days,
            population_vector_correlation=population_vector_correlation,
            decay_amplitude=float("nan"),
            decay_tau_days=float("nan"),
            decay_offset=float("nan"),
            decay_fit_succeeded=False,
        )

    return PopulationDriftMetrics(
        lag_days=lag_days,
        population_vector_correlation=population_vector_correlation,
        decay_amplitude=amplitude,
        decay_tau_days=tau_days,
        decay_offset=offset,
        decay_fit_succeeded=True,
    )


def _combine_pvalues_fisher(values: NDArray[np.float32]) -> float:
    """Combines a finite-valued p-value array via Fisher's method, returning the chi-2 survival function value.

    Notes:
        Mirrors `..sce.sce_report._combine_pvalues_fisher` but accepts a numpy array directly. The shared
        epsilon clip prevents ``log(0)`` divergences for cells whose per-pair null returned exactly zero.

    Args:
        values: Finite per-pair random-remapping peak-shift p-values for one cell.

    Returns:
        The Fisher-combined p-value as a float in ``[0, 1]``. Returns NaN when the input array is empty.
    """
    if values.size == 0:
        return float("nan")
    clipped = np.clip(values.astype(np.float64, copy=False), _FISHER_CLIP_FLOOR, 1.0)
    chi2_stat = float(-2.0 * np.sum(np.log(clipped)))
    return float(chi2.sf(chi2_stat, df=2 * values.size))


def _pv_decay_model(
    days: NDArray[np.float64],
    amplitude: float,
    tau_days: float,
    offset: float,
) -> NDArray[np.float64]:
    """Single-exponential decay model used by `fit_population_decay`.

    Args:
        days: Per-pair calendar-day lag at which to evaluate the model.
        amplitude: Decaying-component amplitude.
        tau_days: Decay time constant in days.
        offset: Asymptotic correlation offset.

    Returns:
        The model values evaluated at every entry of ``days``.
    """
    return amplitude * np.exp(-days / tau_days) + offset


def _empty_cell_drift_metrics() -> CellDriftMetrics:
    """Returns a zero-sized `CellDriftMetrics` used for the ``cell_count == 0`` short-circuit path."""
    # noinspection PyTypeChecker
    int_array: NDArray[np.int32] = np.zeros(0, dtype=np.int32)
    # noinspection PyTypeChecker
    float_array: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
    # noinspection PyTypeChecker
    bool_array: NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
    return CellDriftMetrics(
        pair_count_per_cell=int_array,
        place_session_count=int_array,
        reward_session_count=int_array,
        strict_place_session_count=int_array,
        place_session_fraction=float_array,
        reward_session_fraction=float_array,
        strict_place_session_fraction=float_array,
        place_recurrence_probability=float_array,
        reward_recurrence_probability=float_array,
        strict_place_recurrence_probability=float_array,
        is_persistent_place=bool_array,
        is_persistent_reward=bool_array,
        is_persistent_strict_place=bool_array,
        mean_rate_map_correlation=float_array,
        mean_consecutive_rate_map_correlation=float_array,
        mean_peak_shift_cm=float_array,
        mean_com_shift_cm=float_array,
        fisher_peak_shift_p_value=float_array,
        is_peak_stable=bool_array,
        is_field_stable=bool_array,
        cell_baseline_slope=float_array,
        is_high_bleaching_drift=bool_array,
        is_stably_tuned_place=bool_array,
        is_stably_tuned_reward=bool_array,
        is_stably_tuned_strict_place=bool_array,
    )
