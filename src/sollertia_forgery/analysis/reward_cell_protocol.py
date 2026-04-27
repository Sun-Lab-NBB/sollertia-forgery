"""Identifies reward-associated and reward-predictive neurons from spatial and speed-activity data."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
from dataclasses import dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize

from sollertia_forgery.forging import FluorescenceColumn
from sollertia_forgery.analysis.utilities import (
    RunSessionData,
    bin_fluorescence_per_trial,
    per_cell_pearson_safe,
    assemble_run_session_data,
    bin_fluorescence_by_position,
    accumulate_shuffled_rate_maps,
    compute_shuffle_source_indices,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MINIMUM_PRE_REWARD_BIN_COUNT: int = 2
"""Minimum number of pre-reward spatial bins required to construct a usable GLM design matrix."""
_MINIMUM_TRIALS_FOR_SPLIT_HALF: int = 2
"""Minimum number of trials required to compute an even/odd-lap split-half Pearson r."""


@dataclass(slots=True)
class RewardCellConfiguration:
    """Defines configuration parameters for reward cell detection.

    Notes:
        Bin size and smoothing default to the place-pipeline values (5.0 cm bins, uniform 3-bin smoothing) so that
        ``IS_PLACE`` and ``IS_SPATIALLY_SIGNIFICANT`` are computed against the same rate map and disagreements between
        the two flags reflect biology rather than binning artifacts. The standalone
        :meth:`RewardCellDetector.from_session_path` path re-bins from scratch using these defaults.
    """

    bin_size: float = 5.0
    """Spatial bin size in centimeters for position binning. Matches the place-pipeline default so the two flags
    operate on the same rate map."""
    minimum_speed: float = 5.0
    """Minimum speed threshold in cm/s for including samples in analysis."""
    smooth_size: int = 3
    """Size of the uniform-smoothing kernel in bins applied to the rate map. Matches the place-pipeline smoothing
    (Dombeck-lineage uniform 3-bin moving average) so the two flags share the same smoothing."""
    shuffle_count: int = 1000
    """Number of shuffle iterations for significance testing. Matches the place-pipeline default of 1000 (Climer
    et al., 2025)."""
    minimum_shift_seconds: float = 10.0
    """Minimum circular shift expressed in seconds. Set above the GCaMP6 autocorrelation timescale (roughly 1.2-2 s)
    so the null distribution is not contaminated by indicator decay; matches the place-pipeline default. Climer et
    al. (2025) uses 15 s; Climer & Dombeck (2021) uses 5 s. Resolved to a sample count at runtime via the session's
    ``sampling_rate_hz``."""
    shuffle_minimum_chunk_count: int = 100
    """Sets the minimum circular shift to ``total_samples / shuffle_minimum_chunk_count`` samples when
    ``minimum_shift_seconds`` would either fall below one sample or exceed the safe upper bound
    (``total_samples / 4``). Mirrors the place-pipeline fallback so degenerate short sessions still produce a usable
    null."""
    chunk_count: int = 6
    """Number of chunks for chunk-and-permute shuffle method."""
    significance_threshold: float = 0.01
    """P-value threshold for determining statistically significant spatial information. Climer et al. (2025) uses
    p < 0.01 (99th percentile)."""
    minimum_split_half_r: float = 0.3
    """Minimum even/odd-lap Pearson r required for a cell to count as spatially significant. Gates the shuffle-only
    significance with a within-session reliability check; cells that pass the shuffle but are unreliable across laps
    are excluded so the reward-mixture fit operates on biologically reproducible activity. Krishnan & Sheffield
    (2024) and Mau, Sun, Buzsáki et al. (2020) motivate lap-reliability gating; the specific 0.3 cut is a widely-used
    default in the field."""
    fdr_q: float = 0.05
    """Benjamini-Hochberg FDR target rate applied to the population of shuffle p-values before the
    significance gate. Controls the expected false-discovery proportion across cells; raw per-cell p < 0.01 without
    FDR yields ~1% × N false positives, which materially distorts the mixture-model fit on small reward populations.
    Set to 1.0 to disable FDR (recovers per-cell uncorrected behavior)."""
    reward_zone_width: float = 30.0
    """Width of the reward zone in centimeters for defining the zone band. Centered on the geometry midpoint
    ``(stimulus_trigger_zone_start_cm + stimulus_trigger_zone_end_cm) / 2``. The Issa zone band is bit-identical to
    the legacy ``is_reward_proximal`` definition with this parameter."""
    approach_distance: float = 40.0
    """Length in centimeters of the approach band immediately upstream of the zone. Issa, Radvansky, Xuan & Dombeck
    (2024) Nat Neurosci use 40 cm for the spatial pre-reward window. The approach band is signed-circular
    ``[reward - half_zone - approach_distance, reward - half_zone)``."""
    departure_distance: float = 40.0
    """Length in centimeters of the departure band immediately downstream of the zone. Symmetric counterpart of
    ``approach_distance``; Issa et al. (2024) use 40 cm."""
    pre_reward_window: float = 50.0
    """Distance in centimeters before reward to use as the pre-reward spatial window for the position-vs-speed GLM.
    Defines the window within which (trial, bin) samples are extracted; should cover the deceleration-and-anticipation
    phase. Independent of ``approach_distance`` (band classification) so the two can be tuned separately."""
    minimum_active_trials: int = 10
    """Minimum number of trials a neuron must be active in the pre-reward window to qualify for the position GLM."""
    glm_cv_fold_count: int = 5
    """Number of cross-validation folds (split by trial) for the position-vs-speed partial-variance GLM. Five matches
    the Sosa, Plitt & Giocomo (2025) and Hardcastle et al. (2017) defaults."""
    glm_permutation_count: int = 200
    """Number of trial-label permutations for the GLM partial-variance null. Sosa et al. (2025) report stable
    partial-deviance significance from a few hundred permutations; the permutation cost dominates the GLM runtime."""
    glm_significance_threshold: float = 0.05
    """P-value threshold for the position-vs-speed GLM partial-variance test. Cells whose CV ΔR² (position vs.
    speed+accel) exceeds the upper ``1 - threshold`` percentile of the trial-label permutation null are flagged
    ``IS_POSITION_GLM_SIGNIFICANT``."""


@dataclass(slots=True)
class SpatiallyModulatedNeurons:
    """Stores results of spatial modulation analysis for neurons on a linear or circular track."""

    rate_maps: NDArray[np.float32]
    """Smoothed spatial rate maps with dimensions (cell_count, bin_count)."""
    occupancy: NDArray[np.int32]
    """Per-bin occupancy sample counts with length bin_count (from bin_fluorescence_by_position)."""
    spatial_information: NDArray[np.float32]
    """Spatial information content in bits/event with length cell_count."""
    spatial_information_z: NDArray[np.float32]
    """Z-scored Skaggs spatial information ``(I_obs - mean(I_shuf)) / std(I_shuf)`` with length cell_count. Uses the
    same shuffle distribution as ``p_values`` and provides a decoding-equivalent sensitivity metric (Souza & Tort
    2018) calibrated to the shuffled null rather than to raw bits/event."""
    is_significant: NDArray[np.bool_]
    """Boolean mask indicating neurons with statistically significant spatial information with length cell_count.
    Combines the shuffle p-value gate (``p_values < significance_threshold``) with the lap-reliability gate
    (``split_half_r > minimum_split_half_r``); both must pass for a cell to be admitted."""
    p_values: NDArray[np.float32]
    """P-values from shuffle testing with length cell_count."""
    fdr_survived: NDArray[np.bool_]
    """Boolean mask of cells whose shuffle p-value survives the BH-FDR correction at the configured ``fdr_q`` level
    with length cell_count. Inputs the population-level FDR gate that ``is_significant`` ANDs with the lap-reliability
    gate."""
    split_half_r: NDArray[np.float32]
    """Per-cell even/odd-lap Pearson r with length cell_count. NaN when the trial count is below two or either
    half-map has zero variance; cells with NaN are not admitted by the lap-reliability gate."""
    centers_of_mass: NDArray[np.float32]
    """Circular center-of-mass position in centimeters with length cell_count."""
    reward_relativity_score: NDArray[np.float32]
    """Per-cell continuous reward-relativity score in [0, 1] with length cell_count. Defined as
    ``zone_peak / overall_peak`` of the smoothed rate map; 1.0 means the cell's peak falls inside the reward zone,
    values < 1.0 mean the cell's peak falls outside the zone (with magnitude reflecting how much weaker the in-zone
    signal is). Used by the Phase-2 mutual-exclusion winner-take-all and naturally extends to multi-block analysis
    in Phase 3."""
    bin_size: float
    """Spatial bin size in centimeters."""
    track_length: float
    """Length of the track in centimeters."""

    @property
    def significant_centers(self) -> NDArray[np.float32]:
        """Returns the centers of mass for only the spatially significant neurons."""
        return self.centers_of_mass[self.is_significant]

    @property
    def significant_count(self) -> int:
        """Returns the number of spatially significant neurons."""
        return int(np.sum(self.is_significant))


@dataclass(slots=True)
class RewardCellResults:
    """Stores the complete results of reward cell analysis including spatial modulation and reward classification."""

    spatial_results: SpatiallyModulatedNeurons
    """The underlying spatial modulation analysis results."""
    reward_position: float
    """The reward position in centimeters used for classification."""
    mixture_weight: float
    """Fraction of spatially modulated neurons attributed to the reward Gaussian component of the extended mixture."""
    gaussian_mean: float
    """Fitted reward-Gaussian center position in centimeters from the mixture model."""
    gaussian_std: float
    """Fitted reward-Gaussian standard deviation in centimeters from the mixture model."""
    track_start_weight: float
    """Fraction of spatially modulated neurons attributed to the track-start Gaussian component (centered at 0 cm).
    Captures landmark / start-cell over-representation that would otherwise contaminate ``mixture_weight``
    (Hainmueller & Bartos 2018; Sato et al. 2020)."""
    track_end_weight: float
    """Fraction of spatially modulated neurons attributed to the track-end Gaussian component (centered at
    ``track_length``). Captures trajectory-endpoint over-representation (Frank, Brown & Wilson 2000)."""
    track_start_std: float
    """Fitted track-start Gaussian standard deviation in centimeters."""
    track_end_std: float
    """Fitted track-end Gaussian standard deviation in centimeters."""
    is_approach: NDArray[np.bool_]
    """Boolean mask: COM in the approach band (signed-circular ``[reward - half_zone - approach_distance,
    reward - half_zone)``) with length cell_count. Issa, Radvansky, Xuan & Dombeck (2024) anticipatory band."""
    is_zone: NDArray[np.bool_]
    """Boolean mask: COM in the reward zone band (signed-circular ``[reward - half_zone, reward + half_zone]``) with
    length cell_count. Bit-identical to the legacy ``is_reward_proximal`` definition."""
    is_departure: NDArray[np.bool_]
    """Boolean mask: COM in the departure band (signed-circular ``(reward + half_zone, reward + half_zone +
    departure_distance]``) with length cell_count. Issa et al. (2024) post-reward band."""
    cv_position_partial_r2: NDArray[np.float32]
    """Per-cell 5-fold CV ΔR² of position over speed+acceleration in the pre-reward window (length cell_count). NaN
    where the cell did not enter the GLM (insufficient active trials, untested band, etc.). The principal A1 statistic
    that replaces the legacy ``speed_activity_correlations``."""
    position_glm_p_values: NDArray[np.float32]
    """Per-cell trial-label permutation p-value for the position-vs-speed GLM partial-variance test (length
    cell_count). NaN where the cell did not enter the GLM."""
    is_position_glm_significant: NDArray[np.bool_]
    """Boolean mask: ``cv_position_partial_r2`` exceeds the upper ``1 - glm_significance_threshold`` percentile of the
    trial-label permutation null. Replaces the legacy ``is_slowing_correlated`` flag."""

    @property
    def is_reward_proximal(self) -> NDArray[np.bool_]:
        """Alias of ``is_zone`` preserving the legacy column semantics."""
        return self.is_zone

    @property
    def reward_cell_indices(self) -> NDArray[np.int32]:
        """Returns the indices of neurons classified as reward-associated (significant spatial field in the zone)."""
        mask = self.spatial_results.is_significant & self.is_zone
        return np.argwhere(mask).flatten().astype(np.int32)

    @property
    def reward_cell_count(self) -> int:
        """Returns the number of neurons classified as reward-associated."""
        return len(self.reward_cell_indices)

    @property
    def reward_predictive_indices(self) -> NDArray[np.int32]:
        """Returns the indices of zone-classified, GLM-significant reward-predictive neurons."""
        mask = self.spatial_results.is_significant & self.is_zone & self.is_position_glm_significant
        return np.argwhere(mask).flatten().astype(np.int32)

    @property
    def non_reward_place_cell_indices(self) -> NDArray[np.int32]:
        """Returns the indices of spatially modulated neurons not classified as reward-associated."""
        mask = self.spatial_results.is_significant & ~self.is_zone
        return np.argwhere(mask).flatten().astype(np.int32)


@njit(cache=True, parallel=True)
def _compute_spatial_information(
    rate_maps: NDArray[np.float32],
    occupancy: NDArray[np.int32],
    information: NDArray[np.float32],
) -> None:
    """Computes spatial information content for each neuron from its spatial rate map.

    Notes:
        Implements the Skaggs spatial-information formula in bits per event:
        ``I = sum_x p(x) * (rate[x] / mean_rate) * log2(rate[x] / mean_rate)``, where ``p(x)`` is the per-bin
        occupancy probability. Bins with zero occupancy or zero rate are skipped (the log term diverges).

    References:
        - Skaggs, McNaughton, Wilson & Barnes (1996). Theta phase precession in hippocampal neuronal populations
          and the compression of temporal sequences. Hippocampus.
          https://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K
        - Skaggs, McNaughton & Gothard (1993). An information-theoretic approach to deciphering the hippocampal
          code. NIPS. https://proceedings.neurips.cc/paper/1992/hash/4e4d9c44e7c41a8c0fa5e0c9a47a9e44 -- the
          original Skaggs spatial information measure.

    Args:
        rate_maps: Mean fluorescence rate maps with dimensions (cell_count, bin_count).
        occupancy: Per-bin occupancy sample counts with length bin_count.
        information: Pre-allocated output array with length cell_count for spatial information values in bits/event.
    """
    cell_count = rate_maps.shape[0]
    bin_count = rate_maps.shape[1]

    # Computes the total occupancy for normalizing per-bin occupancy into probability.
    total_occupancy = 0.0
    for bin_index in range(bin_count):
        total_occupancy += occupancy[bin_index]

    for cell_index in prange(cell_count):
        # Computes overall mean firing rate across all occupied bins for this cell.
        mean_rate = 0.0
        for bin_index in range(bin_count):
            if occupancy[bin_index] > 0.0:
                mean_rate += rate_maps[cell_index, bin_index] * (occupancy[bin_index] / total_occupancy)

        # Accumulates spatial information using the Skaggs measure.
        info = 0.0
        if mean_rate > 0.0:
            for bin_index in range(bin_count):
                if occupancy[bin_index] > 0.0 and rate_maps[cell_index, bin_index] > 0.0:
                    probability = occupancy[bin_index] / total_occupancy
                    rate_ratio = rate_maps[cell_index, bin_index] / mean_rate
                    info += probability * rate_ratio * np.log2(rate_ratio)

        information[cell_index] = info


@njit(cache=True, parallel=True)
def _compute_circular_center_of_mass(
    rate_maps: NDArray[np.float32],
    track_length: float,
    centers: NDArray[np.float32],
) -> None:
    """Computes the circular center of mass for each neuron's spatial rate map.

    Notes:
        Transforms spatial bin positions to polar coordinates to handle the circular track topology, computes the 2D
        center of mass (weighted by activity), and converts back to track position. This avoids edge artifacts for
        fields spanning the track wrap-around point.

    Args:
        rate_maps: Smoothed rate maps with dimensions (cell_count, bin_count).
        track_length: Length of the track in centimeters.
        centers: Pre-allocated output array with length cell_count for center-of-mass positions in centimeters.
    """
    cell_count = rate_maps.shape[0]
    bin_count = rate_maps.shape[1]
    bin_size = track_length / bin_count

    for cell_index in prange(cell_count):
        # Accumulates sin and cos components weighted by activity for circular mean computation.
        sum_sin = 0.0
        sum_cos = 0.0
        total_weight = 0.0

        for bin_index in range(bin_count):
            weight = rate_maps[cell_index, bin_index]
            if weight > 0.0:
                # Converts bin center position to angle on the circular track.
                position = (bin_index + 0.5) * bin_size
                angle = 2.0 * np.pi * position / track_length
                sum_sin += weight * np.sin(angle)
                sum_cos += weight * np.cos(angle)
                total_weight += weight

        if total_weight > 0.0:
            # Recovers track position from the mean angle using atan2.
            mean_angle = np.arctan2(sum_sin / total_weight, sum_cos / total_weight)
            if mean_angle < 0.0:
                mean_angle += 2.0 * np.pi
            centers[cell_index] = mean_angle * track_length / (2.0 * np.pi)
        else:
            centers[cell_index] = -1.0


@njit(cache=True, parallel=True)
def _compute_reward_relativity_score(
    rate_maps: NDArray[np.float32],
    bin_centers: NDArray[np.float32],
    track_length: float,
    reward_position: float,
    half_zone: float,
    scores: NDArray[np.float32],
) -> None:
    """Computes the per-cell reward-relativity score as ``zone_peak / overall_peak`` of the smoothed rate map.

    Notes:
        Uses signed circular offset from ``reward_position`` so the zone wraps correctly when it straddles a track
        boundary. Cells whose overall peak is non-positive (uniformly zero rate map) receive 0.0; cells whose zone
        contains no bins (degenerate ``half_zone``) also receive 0.0. The score lies in ``[0, 1]``: 1 means the
        cell's peak falls in the zone, smaller values mean the in-zone signal is weaker than the overall peak.

    Args:
        rate_maps: Smoothed rate maps with dimensions (cell_count, bin_count).
        bin_centers: Spatial bin centers in centimeters with length bin_count.
        track_length: Length of the track in centimeters.
        reward_position: Reward midpoint in centimeters.
        half_zone: Half-width of the reward zone in centimeters.
        scores: Pre-allocated output array with length cell_count.
    """
    cell_count = rate_maps.shape[0]
    bin_count = rate_maps.shape[1]

    for cell_index in prange(cell_count):
        overall_peak = 0.0
        zone_peak = 0.0
        for bin_index in range(bin_count):
            value = rate_maps[cell_index, bin_index]
            if value > overall_peak:
                overall_peak = value
            # Signed circular offset for this bin, in [-track_length/2, track_length/2).
            offset = ((bin_centers[bin_index] - reward_position + track_length / 2.0) % track_length) - (
                track_length / 2.0
            )
            if -half_zone <= offset <= half_zone and value > zone_peak:
                zone_peak = value
        scores[cell_index] = zone_peak / overall_peak if overall_peak > 0.0 else 0.0


def _benjamini_hochberg_fdr(p_values: NDArray[np.float32], q: float) -> NDArray[np.bool_]:
    """Returns the per-cell mask of p-values surviving Benjamini-Hochberg FDR control at level ``q``.

    Notes:
        Standard step-up procedure: sort the p-values in ascending order, find the largest rank ``k`` for which
        ``p_(k) <= q * k / n``, and reject every hypothesis whose rank is at most ``k``. NaN p-values are treated as
        non-significant. Setting ``q >= 1.0`` returns ``p_values < q`` (no FDR adjustment) for diagnostic comparison.

    Args:
        p_values: Per-cell shuffle p-values with length cell_count.
        q: Target false-discovery rate in (0, 1]. Values >= 1 disable the FDR step.

    Returns:
        Boolean mask with length cell_count.
    """
    cell_count = int(p_values.shape[0])
    # noinspection PyTypeChecker
    survived: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)
    if cell_count == 0:
        return survived

    # noinspection PyTypeChecker
    valid_mask: NDArray[np.bool_] = ~np.isnan(p_values)
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return survived

    # noinspection PyTypeChecker
    valid_indices: NDArray[np.int64] = np.flatnonzero(valid_mask)
    valid_p = p_values[valid_indices]
    # noinspection PyTypeChecker
    sort_order: NDArray[np.int64] = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]
    # noinspection PyTypeChecker
    critical: NDArray[np.float64] = q * np.arange(1, valid_count + 1, dtype=np.float64) / float(valid_count)
    # noinspection PyTypeChecker
    survived_sorted: NDArray[np.bool_] = sorted_p <= critical.astype(np.float32)
    if not np.any(survived_sorted):
        return survived

    cutoff_rank = int(np.flatnonzero(survived_sorted).max())
    # ``sort_order[:cutoff_rank + 1]`` gives the positions inside ``valid_p`` whose sorted ranks are at or below the
    # cutoff; mapping through ``valid_indices`` returns their positions in the original (NaN-aware) p-value array.
    survived[valid_indices[sort_order[: cutoff_rank + 1]]] = True
    return survived


@dataclass(frozen=True, slots=True)
class _CvFoldBlock:
    """Pre-computed per-fold matrices for the GLM CV partial-variance test.

    Notes:
        Stores the held-out test mask, the design submatrices, and the Moore-Penrose pseudoinverses of the training
        full and reduced models. Each permutation re-uses these blocks and only pays for ``pinv @ Y_train`` and
        ``X_test @ beta`` matrix products, which are O(n_features × n_samples × n_cells).
    """

    test_mask: NDArray[np.bool_]
    """Per-sample boolean mask selecting held-out (test) (trial, bin) tuples for this fold."""
    train_mask: NDArray[np.bool_]
    """Per-sample boolean mask selecting (trial, bin) tuples used to fit the fold's GLM."""
    pinv_full: NDArray[np.float32]
    """Pseudoinverse of the training-fold full design matrix with dimensions
    (bin_count + 2, train_sample_count)."""
    pinv_reduced: NDArray[np.float32]
    """Pseudoinverse of the training-fold reduced design matrix with dimensions (3, train_sample_count)."""
    x_test_full: NDArray[np.float32]
    """Held-out full design matrix with dimensions (test_sample_count, bin_count + 2)."""
    x_test_reduced: NDArray[np.float32]
    """Held-out reduced design matrix with dimensions (test_sample_count, 3)."""


def _build_cv_fold_blocks(
    x_full: NDArray[np.float32],
    x_reduced: NDArray[np.float32],
    sample_fold: NDArray[np.int32],
    fold_count: int,
) -> list[_CvFoldBlock]:
    """Builds per-fold pseudoinverse blocks for the GLM CV partial-variance test.

    Notes:
        Skips folds that have no test samples or no training samples (degenerate trial assignments). Pseudoinverses
        are computed once and reused across permutations. NumPy's ``linalg.pinv`` uses SVD which is numerically
        robust for the rank-deficient or near-singular design matrices that can arise when a fold is missing some
        position bins.

    Args:
        x_full: Full design matrix with dimensions (sample_count, bin_count + 2).
        x_reduced: Reduced design matrix with dimensions (sample_count, 3).
        sample_fold: Per-sample fold assignment (in [0, fold_count)) with length sample_count.
        fold_count: Total number of CV folds.

    Returns:
        List of ``_CvFoldBlock`` instances; folds without enough train/test samples are omitted.
    """
    blocks: list[_CvFoldBlock] = []
    for fold_index in range(fold_count):
        # noinspection PyTypeChecker
        test_mask: NDArray[np.bool_] = sample_fold == fold_index
        # noinspection PyTypeChecker
        train_mask: NDArray[np.bool_] = ~test_mask
        if not bool(np.any(test_mask)) or not bool(np.any(train_mask)):
            continue
        x_train_full = x_full[train_mask]
        x_train_reduced = x_reduced[train_mask]
        if x_train_full.shape[0] < x_train_full.shape[1]:
            # Underdetermined system; pseudoinverse would give a degenerate fit, skip the fold.
            continue
        # noinspection PyTypeChecker
        pinv_full: NDArray[np.float32] = np.linalg.pinv(x_train_full).astype(np.float32)
        # noinspection PyTypeChecker
        pinv_reduced: NDArray[np.float32] = np.linalg.pinv(x_train_reduced).astype(np.float32)
        blocks.append(
            _CvFoldBlock(
                test_mask=test_mask,
                train_mask=train_mask,
                pinv_full=pinv_full,
                pinv_reduced=pinv_reduced,
                x_test_full=x_full[test_mask],
                x_test_reduced=x_reduced[test_mask],
            )
        )
    return blocks


def _compute_cv_partial_r2(
    fold_blocks: list[_CvFoldBlock],
    y: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Computes per-cell CV ΔR² (full minus reduced model) using pre-computed fold blocks.

    Notes:
        Vectorized across cells. Held-out residuals are accumulated per fold and summed at the end so each (trial,
        bin) tuple contributes to its own fold's CV residual sum exactly once. Cells whose total sum of squares is
        zero (uniformly zero activity) receive 0.0 as a defensible default rather than NaN; the caller's active-trial
        gate excludes them from significance.

    Args:
        fold_blocks: Per-fold pseudoinverse blocks from ``_build_cv_fold_blocks``.
        y: Per-(trial, bin) activity matrix with dimensions (sample_count, cell_count).

    Returns:
        Per-cell CV ΔR² with length cell_count.
    """
    sample_count, cell_count = y.shape
    # noinspection PyTypeChecker
    cv_residuals_full: NDArray[np.float32] = np.zeros((sample_count, cell_count), dtype=np.float32)
    # noinspection PyTypeChecker
    cv_residuals_reduced: NDArray[np.float32] = np.zeros((sample_count, cell_count), dtype=np.float32)

    for block in fold_blocks:
        y_train = y[block.train_mask]
        y_test = y[block.test_mask]
        beta_full = block.pinv_full @ y_train
        beta_reduced = block.pinv_reduced @ y_train
        cv_residuals_full[block.test_mask] = y_test - block.x_test_full @ beta_full
        cv_residuals_reduced[block.test_mask] = y_test - block.x_test_reduced @ beta_reduced

    cv_sse_full = (cv_residuals_full ** 2).sum(axis=0)
    cv_sse_reduced = (cv_residuals_reduced ** 2).sum(axis=0)
    y_mean = y.mean(axis=0, keepdims=True)
    total_ss = ((y - y_mean) ** 2).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # noinspection PyTypeChecker
        full_r2: NDArray[np.float32] = np.where(
            total_ss > 0.0, 1.0 - cv_sse_full / total_ss, 0.0
        ).astype(np.float32)
        # noinspection PyTypeChecker
        reduced_r2: NDArray[np.float32] = np.where(
            total_ss > 0.0, 1.0 - cv_sse_reduced / total_ss, 0.0
        ).astype(np.float32)
    return (full_r2 - reduced_r2).astype(np.float32)


def _compute_even_odd_split_half_r(
    per_trial_rate_map: NDArray[np.float32],
    cell_count: int,
) -> NDArray[np.float32]:
    """Computes per-cell even/odd-lap split-half Pearson r from a per-trial rate map.

    Notes:
        Splits the per-trial axis into even-indexed and odd-indexed laps, averages each half across the trial axis
        with NaN-aware mean, and computes the per-cell Pearson r between the two half-maps. Returns NaN for cells
        when fewer than two trials are available, when either half-map collapses to all-NaN, or when either half has
        zero variance. Even/odd partitioning rather than first/second-half preserves trial-level structure across
        within-session drift, which is the lap-reliability statistic used by Krishnan & Sheffield (2024) and the
        Mau/Sun/Buzsáki (2020) event-cell analysis.

    Args:
        per_trial_rate_map: Per-trial smoothed rate map with dimensions (cell_count, trial_count, bin_count).
        cell_count: Number of cells in the session; used to size the output when the per-trial matrix is empty.

    Returns:
        Per-cell even/odd Pearson r with length cell_count.
    """
    # noinspection PyTypeChecker
    out: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if per_trial_rate_map.size == 0:
        return out
    trial_count = per_trial_rate_map.shape[1]
    if trial_count < _MINIMUM_TRIALS_FOR_SPLIT_HALF:
        return out

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            # noinspection PyTypeChecker
            even_map: NDArray[np.float32] = np.nanmean(per_trial_rate_map[:, 0::2, :], axis=1).astype(np.float32)
            # noinspection PyTypeChecker
            odd_map: NDArray[np.float32] = np.nanmean(per_trial_rate_map[:, 1::2, :], axis=1).astype(np.float32)

    return per_cell_pearson_safe(a=even_map, b=odd_map)


def _apply_uniform_smoothing_wrapped(
    rate_maps: NDArray[np.float32],
    smooth_size: int,
) -> NDArray[np.float32]:
    """Applies uniform-kernel (boxcar) smoothing to rate maps with circular wrapping at track edges.

    Notes:
        Uses the same uniform 3-bin moving average as the place-cell pipeline so the spatial-information rate map and
        the place-field rate map are bit-identical when the bin sizes match.

    References:
        - Dombeck, Harvey, Tian, Looger & Tank (2010). Functional imaging of hippocampal place cells at cellular
          resolution during virtual navigation. Nat Neurosci. https://doi.org/10.1038/nn.2648 -- uniform smoothing
          across spatial bins is the canonical choice in the Dombeck/Tank lineage.

    Args:
        rate_maps: Rate maps with dimensions (cell_count, bin_count).
        smooth_size: Width of the uniform kernel in bins.

    Returns:
        The smoothed rate maps with the same dimensions as input.
    """
    return uniform_filter1d(input=rate_maps, size=smooth_size, axis=1, mode="wrap").astype(np.float32)


def _extended_mixture_negative_log_likelihood(
    parameters: NDArray[np.float64],
    centers: NDArray[np.float32],
    track_length: float,
) -> float:
    """Computes the negative log-likelihood of the four-component reward / track-end mixture model.

    Notes:
        Density model:
            ``p(x) = w_u * U(x) + w_r * N(x; mu_r, sigma_r) + w_s * N(x; 0, sigma_s) + w_e * N(x; L, sigma_e)``,
        where ``U`` is the uniform density on ``[0, L]`` and ``w_u = 1 - w_r - w_s - w_e``. Generalizing the original
        Gauthier & Tank (2018) uniform + Gaussian fit with explicit landmark Gaussians at the track ends prevents the
        reward weight from absorbing trajectory-endpoint over-representation (Hainmueller & Bartos 2018; Sato et al.
        2020 distinguish reward and landmark over-representations as biologically separable).

    Args:
        parameters: Optimization parameters as [w_reward, w_start, w_end, mean_reward, std_reward, std_start, std_end].
        centers: Array of valid center-of-mass positions in centimeters.
        track_length: Length of the track in centimeters.

    Returns:
        The negative log-likelihood of the mixture model given the observed centers.
    """
    w_reward = np.clip(parameters[0], 1e-6, 1.0 - 3e-6)
    w_start = np.clip(parameters[1], 0.0, 1.0 - 3e-6)
    w_end = np.clip(parameters[2], 0.0, 1.0 - 3e-6)
    w_uniform = max(1.0 - w_reward - w_start - w_end, 1e-6)
    mean_reward = parameters[3]
    std_reward = max(parameters[4], 1.0)
    std_start = max(parameters[5], 1.0)
    std_end = max(parameters[6], 1.0)

    uniform_density = 1.0 / track_length
    gaussian_reward = np.exp(-0.5 * ((centers - mean_reward) / std_reward) ** 2) / (
        std_reward * np.sqrt(2.0 * np.pi)
    )
    gaussian_start = np.exp(-0.5 * ((centers - 0.0) / std_start) ** 2) / (std_start * np.sqrt(2.0 * np.pi))
    gaussian_end = np.exp(-0.5 * ((centers - track_length) / std_end) ** 2) / (std_end * np.sqrt(2.0 * np.pi))

    mixture_density = (
        w_uniform * uniform_density
        + w_reward * gaussian_reward
        + w_start * gaussian_start
        + w_end * gaussian_end
    )
    mixture_density = np.clip(mixture_density, 1e-300, None)
    return -np.sum(np.log(mixture_density))


def _compute_extended_mixture(
    centers: NDArray[np.float32],
    track_length: float,
    reward_position: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Fits the four-component (uniform + reward + track-start + track-end) mixture to the COM distribution.

    Notes:
        Optimizes via L-BFGS-B with bounded parameters. The track-start and track-end weights are bounded in
        ``[0, 0.5]`` so they cannot dominate the fit; their Gaussian centers are fixed at 0 and ``track_length``
        respectively (track endpoints are physical landmarks, not free parameters). Returns zero weights and the
        reward seed position when no spatially significant centers are available.

    References:
        - Gauthier & Tank (2018). A dedicated population for reward coding in the hippocampus. Neuron.
          https://doi.org/10.1016/j.neuron.2018.06.008 -- canonical uniform + Gaussian formulation.
        - Hainmueller & Bartos (2018). Parallel emergence of stable and dynamic memory engrams in the hippocampus.
          Nature. https://doi.org/10.1038/s41586-018-0191-2 -- non-uniform place-cell distribution motivates
          explicit landmark components.
        - Sato et al. (2020). Distinct mechanisms of over-representation of landmarks and rewards in the hippocampus.
          Cell Reports. https://doi.org/10.1016/j.celrep.2020.107987 -- landmark vs. reward over-representations
          arise from molecularly separable mechanisms.

    Args:
        centers: Array of center-of-mass positions in centimeters for all spatially modulated neurons.
        track_length: Length of the track in centimeters.
        reward_position: Expected reward position in centimeters used as the initial reward-Gaussian mean.

    Returns:
        A tuple of (mixture_weight, gaussian_mean, gaussian_std, track_start_weight, track_end_weight,
        track_start_std, track_end_std) capturing the reward Gaussian and both track-end Gaussians.
    """
    valid_centers = centers[centers >= 0.0]
    if len(valid_centers) == 0:
        return 0.0, reward_position, 10.0, 0.0, 0.0, 10.0, 10.0

    # Initializes the reward Gaussian at the seed position with seeds for the two landmark components; bounds keep
    # landmark weights modest so they cannot dominate the fit on cohorts without strong endpoint over-representation.
    initial_parameters = np.array([0.15, 0.05, 0.05, reward_position, 15.0, 10.0, 10.0])
    bounds = [
        (0.01, 0.99),  # w_reward
        (0.0, 0.5),  # w_start
        (0.0, 0.5),  # w_end
        (0.0, track_length),  # mean_reward
        (1.0, track_length / 2.0),  # std_reward
        (1.0, track_length / 4.0),  # std_start
        (1.0, track_length / 4.0),  # std_end
    ]

    result = minimize(
        fun=_extended_mixture_negative_log_likelihood,
        x0=initial_parameters,
        args=(valid_centers, track_length),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return (
        float(result.x[0]),
        float(result.x[3]),
        float(result.x[4]),
        float(result.x[1]),
        float(result.x[2]),
        float(result.x[5]),
        float(result.x[6]),
    )


class RewardCellDetector:
    """Detects reward-associated and reward-predictive neurons using spatial information and reward proximity."""

    def __init__(
        self,
        run_session: RunSessionData,
        *,
        configuration: RewardCellConfiguration | None = None,
    ) -> None:
        """Constructs the detector from already-loaded session data.

        Notes:
            Use :meth:`from_session_path` when starting from a session directory; this constructor takes the canonical
            data dependency (a :class:`RunSessionData`) so the same loaded session can feed both detectors without
            re-reading the feather. The reward position is taken as the midpoint of the stimulus trigger zone defined
            in the session's trial geometry data file, since water is delivered wherever in the lick-active zone the
            animal happens to lick rather than at a single point.

        Args:
            run_session: Pre-loaded session data from :func:`assemble_run_session_data`.
            configuration: Configuration parameters for detection thresholds and shuffle testing. Uses defaults if
                None.
        """
        self.fluorescence = run_session.fluorescence
        self.position = run_session.position
        self.speed = run_session.speed
        self.trial_ids = run_session.trial_ids
        self.track_length = run_session.geometry.trial_length_cm
        self.reward_position = (
            run_session.geometry.stimulus_trigger_zone_start_cm + run_session.geometry.stimulus_trigger_zone_end_cm
        ) / 2.0
        self.sampling_rate_hz = run_session.sampling_rate_hz
        self.configuration = configuration if configuration is not None else RewardCellConfiguration()

        # Caches the bin edges shared by spatial-modulation and per-trial-binning paths so split-half r and the pooled
        # rate map use bit-identical bin boundaries.
        # noinspection PyTypeChecker
        self._bin_edges: NDArray[np.float32] = np.arange(
            0, self.track_length + self.configuration.bin_size, self.configuration.bin_size, dtype=np.float32
        )

    @classmethod
    def from_session_path(
        cls,
        session_path: Path,
        trial_type: str,
        *,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
        configuration: RewardCellConfiguration | None = None,
    ) -> RewardCellDetector:
        """Loads fluorescence, position, speed, and trial data from the session feather and constructs the detector.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type: Trial type to analyze (e.g. "ABC", "ABCD"). Must match an entry in the session's trial
                geometry data file.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            configuration: Configuration parameters for detection thresholds and shuffle testing. Uses defaults if
                None.

        Returns:
            A constructed RewardCellDetector ready to call ``detect()`` on.
        """
        run_session = assemble_run_session_data(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )
        return cls(run_session=run_session, configuration=configuration)

    def detect(self) -> RewardCellResults:
        """Runs the full reward cell detection pipeline.

        Notes:
            Computes spatial rate maps with shuffle-based significance testing (gated by BH-FDR on the population
            shuffle p-values and by an even/odd-lap split-half reliability check), classifies each cell into the
            Issa approach / zone / departure bands, fits the four-component (uniform + reward + track-start +
            track-end) mixture to the significant centers of mass, and runs the per-cell Gaussian GLM
            partial-variance test (position vs. speed + acceleration, 5-fold CV with trial-label permutation null)
            for cells in the approach or zone bands.

        References:
            - Gauthier & Tank (2018). A dedicated population for reward coding in the hippocampus. Neuron.
              https://doi.org/10.1016/j.neuron.2018.06.008 -- canonical reward-cell mixture-model framework.
            - Issa, Radvansky, Xuan & Dombeck (2024). Lateral entorhinal cortex subpopulations represent
              experiential epochs surrounding reward. Nat Neurosci. https://doi.org/10.1038/s41593-023-01557-4 --
              approach / zone / departure decomposition.
            - Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to
              reward. Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4 -- partial-variance GLM with speed
              and acceleration covariates that controls for stop-and-lick aliasing.

        Returns:
            A RewardCellResults instance containing spatial modulation results, three-band classification, the
            four-component mixture parameters, and the position-GLM partial-variance test results.
        """
        spatial_results = self._compute_spatial_modulation()

        # Fits the extended (uniform + reward + track-start + track-end) mixture to the significant neurons' COMs.
        (
            mixture_weight,
            gaussian_mean,
            gaussian_std,
            track_start_weight,
            track_end_weight,
            track_start_std,
            track_end_std,
        ) = _compute_extended_mixture(
            centers=spatial_results.significant_centers,
            track_length=self.track_length,
            reward_position=self.reward_position,
        )

        # Classifies neurons into approach / zone / departure bands.
        is_approach, is_zone, is_departure = self._classify_proximity_bands(
            centers_of_mass=spatial_results.centers_of_mass,
        )

        # Runs the position-vs-speed GLM partial-variance test on cells in the approach or zone bands.
        candidate_mask = spatial_results.is_significant & (is_approach | is_zone)
        cv_partial_r2, glm_p_values, is_glm_significant = self._compute_partial_variance(
            candidate_mask=candidate_mask,
        )

        return RewardCellResults(
            spatial_results=spatial_results,
            reward_position=self.reward_position,
            mixture_weight=mixture_weight,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
            track_start_weight=track_start_weight,
            track_end_weight=track_end_weight,
            track_start_std=track_start_std,
            track_end_std=track_end_std,
            is_approach=is_approach,
            is_zone=is_zone,
            is_departure=is_departure,
            cv_position_partial_r2=cv_partial_r2,
            position_glm_p_values=glm_p_values,
            is_position_glm_significant=is_glm_significant,
        )

    def _compute_spatial_modulation(self) -> SpatiallyModulatedNeurons:
        """Computes spatial rate maps, spatial information, and shuffle-based significance for all neurons.

        Notes:
            Significance gates the shuffle p-value with an even/odd-lap split-half Pearson r (Krishnan & Sheffield
            2024 lap-reliability practice) so the downstream mixture-model fit operates on cells whose tuning is
            reproducible across laps, not just statistically distinguishable from a circular-shift null.

        References:
            - Souza, Pavão, Belchior & Tort (2018). On information metrics for spatial coding. Neuroscience.
              https://doi.org/10.1016/j.neuroscience.2018.01.066 -- z-scored spatial information correlates better
              with decoder accuracy than raw bits/event.

        Returns:
            A SpatiallyModulatedNeurons instance with rate maps, spatial information (raw and z-scored), significance
            masks, p-values, split-half r, and centers of mass.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]

        # Applies speed filtering and computes bin assignments.
        # noinspection PyTypeChecker
        speed_mask: NDArray[np.bool_] = self.speed > configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        rate_maps, sample_counts = bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            position_bin_edges=self._bin_edges,
            compute_mean=True,
        )

        # Replaces NaN bins (unvisited) with zero for downstream computation.
        rate_maps = np.nan_to_num(rate_maps, nan=0.0)

        # Applies uniform-kernel smoothing with circular wrapping at track edges, matching the place-pipeline
        # smoothing so the two flags operate on the same rate map.
        smoothed_maps = _apply_uniform_smoothing_wrapped(rate_maps=rate_maps, smooth_size=configuration.smooth_size)

        # Computes spatial information for the observed data.
        # noinspection PyTypeChecker
        observed_information: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        _compute_spatial_information(
            rate_maps=smoothed_maps,
            occupancy=sample_counts,
            information=observed_information,
        )

        # Runs shuffle significance testing and reports both the p-value and the z-scored statistic against the
        # same null distribution.
        p_values, spatial_information_z = self._compute_shuffle_significance(
            filtered_position=filtered_position,
            speed_mask=speed_mask,
            occupancy=sample_counts,
            smooth_size=configuration.smooth_size,
            observed_information=observed_information,
        )

        # Computes per-trial binning + even/odd-lap split-half Pearson r as the lap-reliability statistic that gates
        # admission. Reuses the shared kernel so the place- and reward-cell pipelines compute reliability against
        # bit-identical per-trial rate maps when their bin sizes match.
        per_trial_rate_map = bin_fluorescence_per_trial(
            fluorescence=self.fluorescence,
            position=self.position,
            speed=self.speed,
            trial_ids=self.trial_ids,
            bin_edges=self._bin_edges,
            minimum_speed=configuration.minimum_speed,
            smooth_size=configuration.smooth_size,
        )
        split_half_r = _compute_even_odd_split_half_r(
            per_trial_rate_map=per_trial_rate_map,
            cell_count=cell_count,
        )

        # Applies Benjamini-Hochberg FDR correction on the population shuffle p-values before gating; with thousands
        # of cells per session, raw p < threshold yields ~threshold × N false positives, which materially distorts
        # the mixture-model fit on small reward populations.
        # noinspection PyTypeChecker
        fdr_survived: NDArray[np.bool_] = _benjamini_hochberg_fdr(p_values=p_values, q=configuration.fdr_q)

        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = (
            fdr_survived
            & (split_half_r > configuration.minimum_split_half_r)
            & ~np.isnan(split_half_r)
        )

        # Computes the circular center of mass for each neuron.
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = np.full(cell_count, -1.0, dtype=np.float32)
        _compute_circular_center_of_mass(
            rate_maps=smoothed_maps,
            track_length=self.track_length,
            centers=centers_of_mass,
        )

        # Computes the per-cell continuous reward-relativity score from the smoothed rate maps.
        bin_count = smoothed_maps.shape[1]
        # noinspection PyTypeChecker
        bin_centers: NDArray[np.float32] = (
            self._bin_edges[:bin_count] + configuration.bin_size / 2.0
        ).astype(np.float32)
        # noinspection PyTypeChecker
        reward_relativity_score: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        _compute_reward_relativity_score(
            rate_maps=smoothed_maps,
            bin_centers=bin_centers,
            track_length=self.track_length,
            reward_position=self.reward_position,
            half_zone=configuration.reward_zone_width / 2.0,
            scores=reward_relativity_score,
        )

        return SpatiallyModulatedNeurons(
            rate_maps=smoothed_maps,
            occupancy=sample_counts,
            spatial_information=observed_information,
            spatial_information_z=spatial_information_z,
            is_significant=is_significant,
            p_values=p_values,
            fdr_survived=fdr_survived,
            split_half_r=split_half_r,
            centers_of_mass=centers_of_mass,
            reward_relativity_score=reward_relativity_score,
            bin_size=configuration.bin_size,
            track_length=self.track_length,
        )

    def _compute_shuffle_significance(
        self,
        filtered_position: NDArray[np.float32],
        speed_mask: NDArray[np.bool_],
        occupancy: NDArray[np.int32],
        smooth_size: int,
        observed_information: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Computes shuffle-derived p-values and z-scored spatial information against a circular-shift null.

        Notes:
            For each shuffle iteration, the fluorescence time series is circularly shifted by at least minimum_shift
            samples and then split into chunks that are randomly permuted. Only the speed-filtered subset of
            fluorescence is shuffled and rebinned using the shared indirection-array shuffle helpers hoisted into
            ``analysis.utilities``. The minimum-shift floor is resolved seconds-first against the session's sampling
            rate (matches the place-pipeline pattern); on degenerate short sessions where the seconds-based shift
            would fall outside the safe range the chunk-granularity fallback is used. Z-scored information uses the
            same shuffle distribution as the p-value and reports a decoding-equivalent sensitivity metric (Souza &
            Tort 2018).

        References:
            - Skaggs, McNaughton, Wilson & Barnes (1996). Theta phase precession in hippocampal neuronal populations
              and the compression of temporal sequences. Hippocampus.
              https://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K -- Skaggs spatial information.
            - Climer, Davoudi, Oh & Dombeck (2025). Hippocampal representations drift in stable multisensory
              environments. Nature. https://doi.org/10.1038/s41586-025-09245-y -- circular-shift null with a 15 s
              minimum shift.
            - Souza, Pavão, Belchior & Tort (2018). On information metrics for spatial coding. Neuroscience.
              https://doi.org/10.1016/j.neuroscience.2018.01.066 -- z-scored information vs. raw bits/event.

        Args:
            filtered_position: Speed-filtered position values with length filtered_sample_count.
            speed_mask: Boolean mask indicating speed-filtered samples with length sample_count.
            occupancy: Per-bin occupancy sample counts with length bin_count.
            smooth_size: Width of the uniform smoothing kernel in bins.
            observed_information: Observed spatial information values with length cell_count.

        Returns:
            A tuple of (p_values, spatial_information_z), both with length cell_count.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]
        sample_count = self.fluorescence.shape[1]
        bin_count = len(self._bin_edges) - 1

        # Resolves the minimum shift seconds-first, falling back to the chunk-granularity floor on degenerate short
        # sessions; mirrors the place-pipeline pattern so the two protocols use the same null structure.
        seconds_based_shift = (
            int(self.sampling_rate_hz * configuration.minimum_shift_seconds)
            if not np.isnan(self.sampling_rate_hz)
            else 0
        )
        chunk_based_shift = sample_count // configuration.shuffle_minimum_chunk_count
        upper_bound = sample_count // 4
        minimum_shift = (
            seconds_based_shift if 0 < seconds_based_shift <= upper_bound else max(chunk_based_shift, 1)
        )

        # Precomputes destination-sample indices and their spatial bin assignments; both are invariant across shuffles.
        # noinspection PyTypeChecker
        filtered_sample_indices: NDArray[np.int32] = np.nonzero(speed_mask)[0].astype(np.int32)
        # noinspection PyTypeChecker
        filtered_bin_indices: NDArray[np.int32] = np.clip(
            np.searchsorted(self._bin_edges, filtered_position, side="right") - 1, 0, bin_count - 1
        ).astype(np.int32)

        # Reuses rate-map and information buffers across iterations.
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.empty((cell_count, bin_count), dtype=np.float32)
        # noinspection PyTypeChecker
        shuffled_information: NDArray[np.float32] = np.zeros(
            (configuration.shuffle_count, cell_count), dtype=np.float32
        )

        for iteration in tqdm(range(configuration.shuffle_count), desc="Running shuffling", unit="iter"):
            source_indices = compute_shuffle_source_indices(
                filtered_sample_indices=filtered_sample_indices,
                sample_count=sample_count,
                minimum_shift=minimum_shift,
                chunk_count=configuration.chunk_count,
                seed=iteration,
            )

            accumulate_shuffled_rate_maps(
                fluorescence=self.fluorescence,
                source_indices=source_indices,
                bin_indices=filtered_bin_indices,
                sample_counts=occupancy,
                output=rate_maps,
            )

            smoothed_shuffled = _apply_uniform_smoothing_wrapped(rate_maps=rate_maps, smooth_size=smooth_size)

            _compute_spatial_information(
                rate_maps=smoothed_shuffled,
                occupancy=occupancy,
                information=shuffled_information[iteration],
            )

        # Computes p-values as the fraction of shuffles exceeding observed.
        exceed_count = np.sum(shuffled_information >= observed_information[np.newaxis, :], axis=0)
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = (exceed_count / configuration.shuffle_count).astype(np.float32)

        # Computes the z-scored spatial information against the same shuffled null. Cells whose null distribution
        # has zero standard deviation (degenerate constant trace) receive z = 0 rather than a divide-by-zero NaN.
        shuffle_mean = shuffled_information.mean(axis=0)
        shuffle_std = shuffled_information.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            # noinspection PyTypeChecker
            spatial_information_z: NDArray[np.float32] = np.where(
                shuffle_std > 0.0,
                (observed_information - shuffle_mean) / shuffle_std,
                0.0,
            ).astype(np.float32)

        return p_values, spatial_information_z

    def _classify_proximity_bands(
        self,
        centers_of_mass: NDArray[np.float32],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Classifies neurons into approach, zone, and departure bands by signed circular offset from reward position.

        Notes:
            Uses signed circular offset (in ``[-track_length/2, track_length/2)``) so the bands wrap correctly when
            the zone or its outer bands cross a track boundary. The zone band is bit-identical to the legacy
            ``is_reward_proximal`` definition (``circular_distance <= half_zone``); the approach and departure bands
            extend Issa, Radvansky, Xuan & Dombeck (2024) anticipatory and post-reward decomposition outward by
            ``approach_distance`` and ``departure_distance`` respectively.

        References:
            - Issa, Radvansky, Xuan & Dombeck (2024). Lateral entorhinal cortex subpopulations represent experiential
              epochs surrounding reward. Nat Neurosci. https://doi.org/10.1038/s41593-023-01557-4 -- four-window
              decomposition (approach / zone / consumption / departure); the zone is time-locked to consumption in
              their work, here represented spatially by the task-defined reward zone width.

        Args:
            centers_of_mass: Center-of-mass positions in centimeters with length cell_count.

        Returns:
            A tuple of (is_approach, is_zone, is_departure) boolean masks, each with length cell_count.
        """
        configuration = self.configuration
        half_zone = configuration.reward_zone_width / 2.0
        approach_distance = configuration.approach_distance
        departure_distance = configuration.departure_distance

        # Signed circular offset in [-track_length/2, track_length/2). Negative values are upstream (approach side);
        # positive values are downstream (departure side).
        signed_offset = (
            (centers_of_mass - self.reward_position + self.track_length / 2.0) % self.track_length
        ) - self.track_length / 2.0

        # Marks neurons with invalid COM (-1 sentinel) as outside every band.
        # noinspection PyTypeChecker
        is_valid: NDArray[np.bool_] = centers_of_mass >= 0.0
        # noinspection PyTypeChecker
        is_zone: NDArray[np.bool_] = is_valid & (np.abs(signed_offset) <= half_zone)
        # noinspection PyTypeChecker
        is_approach: NDArray[np.bool_] = (
            is_valid & (signed_offset < -half_zone) & (signed_offset >= -(half_zone + approach_distance))
        )
        # noinspection PyTypeChecker
        is_departure: NDArray[np.bool_] = (
            is_valid & (signed_offset > half_zone) & (signed_offset <= half_zone + departure_distance)
        )
        return is_approach, is_zone, is_departure

    def _bin_trials_in_pre_reward_window(
        self,
        signal: NDArray[np.float32],
        unique_trials: NDArray[np.int32],
        bin_edges: NDArray[np.float32],
        window_start: float,
        window_end: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Bins a per-sample signal into a (trial_count, bin_count) matrix within the pre-reward spatial window.

        Args:
            signal: Per-sample values to bin with length sample_count.
            unique_trials: Sorted unique trial identifiers.
            bin_edges: Spatial bin edges for the pre-reward window with length bin_count + 1.
            window_start: Start of the pre-reward spatial window in centimeters.
            window_end: End of the pre-reward spatial window in centimeters.

        Returns:
            A tuple of (sums, counts) where sums has the accumulated signal per bin and counts has the number of
            samples per bin, both with dimensions (trial_count, bin_count).
        """
        bin_count = len(bin_edges) - 1
        trial_count = len(unique_trials)
        # noinspection PyTypeChecker
        sums: NDArray[np.float32] = np.zeros((trial_count, bin_count), dtype=np.float32)
        # noinspection PyTypeChecker
        counts: NDArray[np.int32] = np.zeros((trial_count, bin_count), dtype=np.int32)

        for trial_index, trial_id in enumerate(unique_trials):
            # noinspection PyTypeChecker
            trial_mask: NDArray[np.bool_] = self.trial_ids == trial_id
            trial_positions = self.position[trial_mask]
            trial_speeds = self.speed[trial_mask]
            trial_signal = signal[trial_mask]

            # Restricts to the pre-reward window and speed-filtered samples.
            # noinspection PyTypeChecker
            window_mask: NDArray[np.bool_] = (
                (trial_positions >= window_start)
                & (trial_positions < window_end)
                & (trial_speeds > self.configuration.minimum_speed)
            )
            window_positions = trial_positions[window_mask]
            window_signal = trial_signal[window_mask]

            # noinspection PyTypeChecker
            bin_indices: NDArray[np.int64] = np.clip(
                np.searchsorted(bin_edges, window_positions, side="right") - 1, 0, bin_count - 1
            )

            for sample_index in range(len(bin_indices)):
                bin_index = bin_indices[sample_index]
                sums[trial_index, bin_index] += window_signal[sample_index]
                counts[trial_index, bin_index] += 1

        return sums, counts

    def _compute_partial_variance(
        self,
        candidate_mask: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Per-cell 5-fold CV partial-variance test for position over speed+acceleration in the pre-reward window.

        Notes:
            Only candidate cells are tested. For each cell:
                1. Bin speed, acceleration, and activity per (trial, bin) tuple in the pre-reward window.
                2. Build a Gaussian GLM with predictors [position one-hot, speed, acceleration].
                3. Build a reduced GLM with only [intercept, speed, acceleration].
                4. Compute 5-fold CV ΔR² (full vs. reduced) using by-trial fold assignments so within-trial
                   autocorrelation is not absorbed into the held-out residuals.
                5. Permute trial labels of the activity matrix and recompute ΔR² to build the null distribution.
                6. p-value = (1 + count of perms with ΔR² >= observed) / (1 + permutation_count) — additive smoothing
                   keeps the p-value strictly positive on small permutation counts.
            Cells with insufficient active trials, degenerate covariate variance, or fewer valid (trial, bin) tuples
            than the design-matrix rank receive NaN ΔR², NaN p-value, and ``False`` significance.

        References:
            - Sosa, Plitt & Giocomo (2025). A flexible hippocampal population code for experience relative to reward.
              Nat Neurosci. https://doi.org/10.1038/s41593-025-01985-4 -- partial-deviance / partial-R² framework
              with position, reward-relative position, speed, acceleration, and licking covariates; the GLM
              architecture this method follows.
            - Hardcastle, Maheswaranathan, Ganguli & Giocomo (2017). A multiplexed, heterogeneous, and adaptive code
              for navigation in MEC. Neuron. https://doi.org/10.1016/j.neuron.2017.03.025 -- canonical 2P-GLM with
              forward selection across position / speed / head-direction predictors.

        Args:
            candidate_mask: Boolean mask of cells to test (typically ``is_significant & (is_approach | is_zone)``)
                with length cell_count.

        Returns:
            A tuple of (cv_position_partial_r2, p_values, is_glm_significant) per cell, each with length cell_count.
            Cells outside the candidate mask carry NaN ΔR², NaN p-value, and ``False`` significance.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]

        # noinspection PyTypeChecker
        cv_partial_r2: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
        # noinspection PyTypeChecker
        is_glm_significant: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)

        # noinspection PyTypeChecker
        candidate_indices: NDArray[np.int64] = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return cv_partial_r2, p_values, is_glm_significant

        # Defines the pre-reward spatial window.
        window_start = self.reward_position - configuration.pre_reward_window
        window_end = self.reward_position
        # noinspection PyTypeChecker
        pre_reward_bin_edges: NDArray[np.float32] = np.arange(
            window_start, window_end + configuration.bin_size, configuration.bin_size, dtype=np.float32
        )
        bin_count = len(pre_reward_bin_edges) - 1
        if bin_count < _MINIMUM_PRE_REWARD_BIN_COUNT:
            return cv_partial_r2, p_values, is_glm_significant

        # noinspection PyTypeChecker
        unique_trials: NDArray[np.int32] = np.unique(self.trial_ids).astype(np.int32)
        trial_count = len(unique_trials)
        if trial_count < configuration.glm_cv_fold_count:
            return cv_partial_r2, p_values, is_glm_significant

        # Computes per-trial gradient of speed for the acceleration covariate; per-trial gradient avoids spurious
        # accelerations across trial boundaries.
        sampling_rate_hz = self.sampling_rate_hz if not np.isnan(self.sampling_rate_hz) else 1.0
        # noinspection PyTypeChecker
        acceleration: NDArray[np.float32] = np.zeros_like(self.speed)
        for trial_id in unique_trials:
            # noinspection PyTypeChecker
            trial_mask: NDArray[np.bool_] = self.trial_ids == trial_id
            if int(np.sum(trial_mask)) < 2:
                continue
            acceleration[trial_mask] = (np.gradient(self.speed[trial_mask]) * sampling_rate_hz).astype(np.float32)

        # Bins speed, acceleration, and per-candidate activity per (trial, bin) tuple in the pre-reward window.
        speed_sums, speed_counts = self._bin_trials_in_pre_reward_window(
            signal=self.speed,
            unique_trials=unique_trials,
            bin_edges=pre_reward_bin_edges,
            window_start=window_start,
            window_end=window_end,
        )
        accel_sums, _ = self._bin_trials_in_pre_reward_window(
            signal=acceleration,
            unique_trials=unique_trials,
            bin_edges=pre_reward_bin_edges,
            window_start=window_start,
            window_end=window_end,
        )

        # noinspection PyTypeChecker
        valid_grid: NDArray[np.bool_] = speed_counts > 0
        # noinspection PyTypeChecker
        speed_grid: NDArray[np.float32] = np.zeros_like(speed_sums)
        # noinspection PyTypeChecker
        accel_grid: NDArray[np.float32] = np.zeros_like(accel_sums)
        speed_grid[valid_grid] = speed_sums[valid_grid] / speed_counts[valid_grid]
        accel_grid[valid_grid] = accel_sums[valid_grid] / speed_counts[valid_grid]

        # Flattens (trial, bin) tuples to a single sample axis and restricts to valid tuples (i.e., those with at
        # least one speed-filtered sample in the bin).
        valid_flat = valid_grid.reshape(-1)
        if int(np.sum(valid_flat)) < bin_count + 3:
            return cv_partial_r2, p_values, is_glm_significant

        # noinspection PyTypeChecker
        trial_indices_grid: NDArray[np.int32] = np.repeat(np.arange(trial_count, dtype=np.int32), bin_count)
        # noinspection PyTypeChecker
        bin_indices_grid: NDArray[np.int32] = np.tile(np.arange(bin_count, dtype=np.int32), trial_count)
        trial_valid = trial_indices_grid[valid_flat]
        bin_valid = bin_indices_grid[valid_flat]
        speed_valid = speed_grid.reshape(-1)[valid_flat]
        accel_valid = accel_grid.reshape(-1)[valid_flat]
        sample_count = int(speed_valid.shape[0])

        # Builds the full and reduced design matrices.
        # noinspection PyTypeChecker
        x_full: NDArray[np.float32] = np.zeros((sample_count, bin_count + 2), dtype=np.float32)
        x_full[np.arange(sample_count), bin_valid] = 1.0
        x_full[:, bin_count] = speed_valid
        x_full[:, bin_count + 1] = accel_valid
        # noinspection PyTypeChecker
        x_reduced: NDArray[np.float32] = np.column_stack(
            (np.ones(sample_count, dtype=np.float32), speed_valid, accel_valid)
        ).astype(np.float32)

        # Fold assignment by trial: trial t goes into fold (t % fold_count). Stable across permutations because the
        # fold index attaches to the trial position in ``unique_trials`` (not to the trial id).
        fold_count = configuration.glm_cv_fold_count
        # noinspection PyTypeChecker
        sample_fold: NDArray[np.int32] = (trial_valid % fold_count).astype(np.int32)

        # Pre-computes per-fold pseudoinverses so each permutation only does cheap matrix-vector products.
        fold_blocks = _build_cv_fold_blocks(
            x_full=x_full, x_reduced=x_reduced, sample_fold=sample_fold, fold_count=fold_count
        )
        if not fold_blocks:
            return cv_partial_r2, p_values, is_glm_significant

        # Builds the per-candidate activity grid and the corresponding flattened-valid matrix.
        candidate_count = int(candidate_indices.size)
        # noinspection PyTypeChecker
        activity_grid: NDArray[np.float32] = np.zeros((trial_count, bin_count, candidate_count), dtype=np.float32)
        active_trial_counts = np.zeros(candidate_count, dtype=np.int32)
        for column_index, neuron_index in enumerate(candidate_indices):
            activity_sums, activity_counts = self._bin_trials_in_pre_reward_window(
                signal=self.fluorescence[neuron_index],
                unique_trials=unique_trials,
                bin_edges=pre_reward_bin_edges,
                window_start=window_start,
                window_end=window_end,
            )
            # noinspection PyTypeChecker
            cell_valid: NDArray[np.bool_] = activity_counts > 0
            # noinspection PyTypeChecker
            activity_means: NDArray[np.float32] = np.zeros_like(activity_sums)
            activity_means[cell_valid] = activity_sums[cell_valid] / activity_counts[cell_valid]
            activity_grid[:, :, column_index] = activity_means
            active_trial_counts[column_index] = int(np.sum(activity_sums.sum(axis=1) > 0))

        # Flattens the activity grid in (trial, bin, candidate) order and restricts to valid (trial, bin) tuples.
        # noinspection PyTypeChecker
        y_valid: NDArray[np.float32] = activity_grid.reshape(trial_count * bin_count, candidate_count)[valid_flat]

        # Computes the observed CV ΔR² per candidate cell.
        observed_partial_r2 = _compute_cv_partial_r2(fold_blocks=fold_blocks, y=y_valid)

        # Generates the trial-label permutation null. Each permutation re-shuffles the trial axis of the activity
        # grid (preserving per-bin marginals) and recomputes the CV ΔR². Pre-computed fold pseudoinverses keep each
        # permutation cheap (matrix-vector multiplies).
        permutation_count = configuration.glm_permutation_count
        # noinspection PyTypeChecker
        null_partial_r2: NDArray[np.float32] = np.zeros((permutation_count, candidate_count), dtype=np.float32)
        for permutation_index in tqdm(
            range(permutation_count), desc="Running GLM null", unit="iter", leave=False
        ):
            generator = np.random.default_rng(seed=permutation_index)
            # noinspection PyTypeChecker
            permutation: NDArray[np.int64] = generator.permutation(trial_count)
            permuted_grid = activity_grid[permutation]
            permuted_valid = permuted_grid.reshape(trial_count * bin_count, candidate_count)[valid_flat]
            null_partial_r2[permutation_index] = _compute_cv_partial_r2(
                fold_blocks=fold_blocks, y=permuted_valid
            )

        # Per-cell p-value with additive smoothing so a permutation count of 200 cannot yield exactly 0.
        exceed_count = np.sum(null_partial_r2 >= observed_partial_r2[np.newaxis, :], axis=0)
        # noinspection PyTypeChecker
        p_value_per_candidate: NDArray[np.float32] = (
            (exceed_count + 1).astype(np.float32) / float(permutation_count + 1)
        )

        # Active-trial gate: cells with too few non-zero-activity trials are not admitted to the test.
        # noinspection PyTypeChecker
        qualified: NDArray[np.bool_] = active_trial_counts >= configuration.minimum_active_trials

        for column_index, neuron_index in enumerate(candidate_indices):
            if not qualified[column_index]:
                continue
            cv_partial_r2[neuron_index] = observed_partial_r2[column_index]
            p_values[neuron_index] = p_value_per_candidate[column_index]
            is_glm_significant[neuron_index] = (
                p_value_per_candidate[column_index] < configuration.glm_significance_threshold
            )

        return cv_partial_r2, p_values, is_glm_significant
