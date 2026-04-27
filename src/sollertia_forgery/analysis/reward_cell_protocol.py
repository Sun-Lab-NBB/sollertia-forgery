"""Identifies reward-associated and reward-predictive neurons from spatial and speed-activity data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from sollertia_forgery.forging import FluorescenceColumn
from sollertia_forgery.analysis.utilities import (
    assemble_run_session_data,
    bin_fluorescence_by_position,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MINIMUM_PRE_REWARD_BIN_COUNT: int = 2
"""Minimum number of pre-reward spatial bins required for slowing-correlation analysis."""
_MINIMUM_VALID_SAMPLE_COUNT: int = 3
"""Minimum number of speed/activity sample pairs required to compute a reliable Pearson correlation."""


@dataclass(slots=True)
class RewardCellConfiguration:
    """Defines configuration parameters for reward cell detection."""

    bin_size: float = 10.0
    """Spatial bin size in centimeters for position binning."""
    minimum_speed: float = 5.0
    """Minimum speed threshold in cm/s for including samples in analysis."""
    gaussian_sigma: float = 20.0
    """Standard deviation in centimeters for Gaussian spatial smoothing of rate maps."""
    shuffle_count: int = 100
    """Number of shuffle iterations for significance testing."""
    minimum_shift_samples: int = 500
    """Minimum circular shift in samples applied during shuffle."""
    chunk_count: int = 6
    """Number of chunks for chunk-and-permute shuffle method."""
    significance_threshold: float = 0.05
    """P-value threshold for determining statistically significant spatial information."""
    reward_zone_width: float = 30.0
    """Width of the reward zone in centimeters for defining reward-proximal fields."""
    pre_reward_window: float = 50.0
    """Distance in centimeters before reward to use as the pre-reward spatial window for slowing analysis."""
    minimum_active_trials: int = 10
    """Minimum number of trials a neuron must be active in the pre-reward window to qualify for slowing analysis."""
    slowing_shuffle_count: int = 100
    """Number of trial-shuffle iterations for slowing-correlation significance testing."""
    slowing_significance_threshold: float = 0.05
    """P-value threshold for slowing-correlation significance testing."""


@dataclass(slots=True)
class SpatiallyModulatedNeurons:
    """Stores results of spatial modulation analysis for neurons on a linear or circular track."""

    rate_maps: NDArray[np.float32]
    """Smoothed spatial rate maps with dimensions (cell_count, bin_count)."""
    occupancy: NDArray[np.int32]
    """Per-bin occupancy sample counts with length bin_count (from bin_fluorescence_by_position)."""
    spatial_information: NDArray[np.float32]
    """Spatial information content in bits/event with length cell_count."""
    is_significant: NDArray[np.bool_]
    """Boolean mask indicating neurons with statistically significant spatial information with length cell_count."""
    p_values: NDArray[np.float32]
    """P-values from shuffle testing with length cell_count."""
    centers_of_mass: NDArray[np.float32]
    """Circular center-of-mass position in centimeters with length cell_count."""
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
    """Fraction of spatially modulated neurons attributed to the reward-associated Gaussian component."""
    gaussian_mean: float
    """Fitted Gaussian center position in centimeters from the mixture model."""
    gaussian_std: float
    """Fitted Gaussian standard deviation in centimeters from the mixture model."""
    is_reward_proximal: NDArray[np.bool_]
    """Boolean mask indicating neurons with center-of-mass within the reward zone with length cell_count."""
    speed_activity_correlations: NDArray[np.float32]
    """Pearson correlation between spatially binned speed and activity with length cell_count."""
    is_slowing_correlated: NDArray[np.bool_]
    """Boolean mask indicating neurons with significant negative speed-activity correlation with length cell_count."""

    @property
    def reward_cell_indices(self) -> NDArray[np.int32]:
        """Returns the indices of neurons classified as reward-associated (significant spatial field near reward)."""
        mask = self.spatial_results.is_significant & self.is_reward_proximal
        return np.argwhere(mask).flatten().astype(np.int32)

    @property
    def reward_cell_count(self) -> int:
        """Returns the number of neurons classified as reward-associated."""
        return len(self.reward_cell_indices)

    @property
    def reward_predictive_indices(self) -> NDArray[np.int32]:
        """Returns the indices of neurons that are both reward-associated and slowing-correlated."""
        mask = self.spatial_results.is_significant & self.is_reward_proximal & self.is_slowing_correlated
        return np.argwhere(mask).flatten().astype(np.int32)

    @property
    def non_reward_place_cell_indices(self) -> NDArray[np.int32]:
        """Returns the indices of spatially modulated neurons not classified as reward-associated."""
        mask = self.spatial_results.is_significant & ~self.is_reward_proximal
        return np.argwhere(mask).flatten().astype(np.int32)


@njit(cache=True, parallel=True)
def _compute_spatial_information(
    rate_maps: NDArray[np.float32],
    occupancy: NDArray[np.int32],
    information: NDArray[np.float32],
) -> None:
    """Computes spatial information content for each neuron from its spatial rate map.

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


@njit(cache=True, nogil=True)
def _compute_shuffled_source_indices(
    filtered_sample_indices: NDArray[np.int32],
    sample_count: int,
    minimum_shift: int,
    chunk_count: int,
    seed: int,
) -> NDArray[np.int32]:
    """Maps each speed-filtered destination sample back to the source sample it pulls from under the shuffle.

    Notes:
        Encodes the circular shift and chunk permutation as an indirection array rather than materializing a full
        shuffled fluorescence matrix.

    Args:
        filtered_sample_indices: Destination-sample indices retained by the speed filter with length
            filtered_sample_count.
        sample_count: Total number of samples in the original fluorescence time series.
        minimum_shift: Minimum number of samples for the circular shift.
        chunk_count: Number of chunks to split the shifted trace into for permutation.
        seed: Random seed for reproducibility.

    Returns:
        Source-sample indices with length filtered_sample_count.
    """
    np.random.seed(seed)  # noqa: NPY002
    shift_amount = np.random.randint(minimum_shift, sample_count - minimum_shift)  # noqa: NPY002
    chunk_size = sample_count // chunk_count
    permutation = np.random.permutation(chunk_count)  # noqa: NPY002

    # Computes cumulative output-chunk start positions so each destination can be located within the permuted layout.
    output_chunk_starts = np.empty(chunk_count + 1, dtype=np.int32)
    output_chunk_starts[0] = 0
    for output_chunk_position in range(chunk_count):
        source_chunk_index = permutation[output_chunk_position]
        if source_chunk_index < chunk_count - 1:
            chunk_size_local = chunk_size
        else:
            chunk_size_local = sample_count - source_chunk_index * chunk_size
        output_chunk_starts[output_chunk_position + 1] = output_chunk_starts[output_chunk_position] + chunk_size_local

    filtered_count = filtered_sample_indices.shape[0]
    source_indices = np.empty(filtered_count, dtype=np.int32)

    # Resolves each destination back through the permutation and shift to its source sample.
    for filtered_index in range(filtered_count):
        destination = filtered_sample_indices[filtered_index]
        output_chunk_position = 0
        while output_chunk_position + 1 < chunk_count and output_chunk_starts[output_chunk_position + 1] <= destination:
            output_chunk_position += 1
        offset_within_chunk = destination - output_chunk_starts[output_chunk_position]
        source_chunk_index = permutation[output_chunk_position]
        shifted_index = source_chunk_index * chunk_size + offset_within_chunk
        source_indices[filtered_index] = (shifted_index - shift_amount) % sample_count

    return source_indices


@njit(cache=True, parallel=True, nogil=True)
def _accumulate_shuffled_rate_maps(
    fluorescence: NDArray[np.float32],
    source_indices: NDArray[np.int32],
    bin_indices: NDArray[np.int32],
    sample_counts: NDArray[np.int32],
    output: NDArray[np.float32],
) -> None:
    """Bins fluorescence into a per-cell rate map by gathering source samples through an indirection array.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, sample_count).
        source_indices: Source-sample indices per filtered destination sample with length filtered_sample_count.
        bin_indices: Spatial bin indices per filtered destination sample with length filtered_sample_count.
        sample_counts: Per-bin occupancy counts with length bin_count.
        output: Pre-allocated output rate maps with dimensions (cell_count, bin_count).
    """
    cell_count = fluorescence.shape[0]
    filtered_count = source_indices.shape[0]
    bin_count = output.shape[1]

    for cell_index in prange(cell_count):
        bin_sums = np.zeros(bin_count, dtype=np.float32)
        for filtered_index in range(filtered_count):
            bin_sums[bin_indices[filtered_index]] += fluorescence[cell_index, source_indices[filtered_index]]

        for bin_index in range(bin_count):
            if sample_counts[bin_index] > 0:
                output[cell_index, bin_index] = bin_sums[bin_index] / sample_counts[bin_index]
            else:
                output[cell_index, bin_index] = 0.0


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


def _apply_smooth_rate_maps_wrapped(
    rate_maps: NDArray[np.float32],
    sigma_bins: float,
) -> NDArray[np.float32]:
    """Applies Gaussian smoothing to rate maps with circular wrapping at track edges.

    Args:
        rate_maps: Rate maps with dimensions (cell_count, bin_count).
        sigma_bins: Standard deviation of the Gaussian kernel in bin units.

    Returns:
        The smoothed rate maps with the same dimensions as input.
    """
    return gaussian_filter1d(input=rate_maps, sigma=sigma_bins, axis=1, mode="wrap").astype(np.float32)


def _compute_negative_log_likelihood(
    parameters: NDArray[np.float64],
    centers: NDArray[np.float32],
    track_length: float,
) -> float:
    """Computes the negative log-likelihood of a uniform + Gaussian mixture model.

    Args:
        parameters: Optimization parameters as [mixture_weight, gaussian_mean, gaussian_std].
        centers: Array of valid center-of-mass positions in centimeters.
        track_length: Length of the track in centimeters.

    Returns:
        The negative log-likelihood of the mixture model given the observed centers.
    """
    weight = np.clip(parameters[0], 1e-6, 1.0 - 1e-6)
    mean = parameters[1]
    standard_deviation = max(parameters[2], 1.0)

    uniform_density = 1.0 / track_length
    gaussian_density = np.exp(-0.5 * ((centers - mean) / standard_deviation) ** 2) / (
        standard_deviation * np.sqrt(2.0 * np.pi)
    )

    # Computes the mixture density as a weighted sum of uniform and Gaussian components.
    mixture_density = (1.0 - weight) * uniform_density + weight * gaussian_density
    mixture_density = np.clip(mixture_density, 1e-300, None)

    return -np.sum(np.log(mixture_density))


def _compute_uniform_gaussian_mixture(
    centers: NDArray[np.float32],
    track_length: float,
    reward_position: float,
) -> tuple[float, float, float]:
    """Fits a uniform + Gaussian mixture model to the distribution of spatial field centers of mass.

    Notes:
        Models the COM distribution as a mixture of a uniform distribution (place cells with evenly distributed fields)
        and a Gaussian centered near the reward location (reward-associated cells). The mixture weight, Gaussian mean,
        and Gaussian standard deviation are optimized via maximum likelihood.

    Args:
        centers: Array of center-of-mass positions in centimeters for all spatially modulated neurons.
        track_length: Length of the track in centimeters.
        reward_position: Expected reward position in centimeters used as the initial Gaussian mean.

    Returns:
        A tuple of (mixture_weight, gaussian_mean, gaussian_std) where mixture_weight is the fraction of neurons
        attributed to the Gaussian (reward) component, gaussian_mean is the fitted center in centimeters, and
        gaussian_std is the fitted standard deviation in centimeters.
    """
    valid_centers = centers[centers >= 0.0]
    if len(valid_centers) == 0:
        return 0.0, reward_position, 10.0

    # Initializes optimization with the reward position as the Gaussian center.
    initial_parameters = np.array([0.15, reward_position, 15.0])
    bounds = [(0.01, 0.99), (0.0, track_length), (1.0, track_length / 2.0)]

    result = minimize(
        fun=_compute_negative_log_likelihood,
        x0=initial_parameters,
        args=(valid_centers, track_length),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return float(result.x[0]), float(result.x[1]), float(result.x[2])


class RewardCellDetector:
    """Detects reward-associated and reward-predictive neurons using spatial information and reward proximity."""

    def __init__(
        self,
        session_path: Path,
        trial_type: str,
        fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
        configuration: RewardCellConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, position, speed, and trial data from the session feather and trial geometry data file
        for reward cell detection.

        Notes:
            The reward position is taken as the midpoint of the stimulus trigger zone defined in the session's
            trial geometry data file, since water is delivered wherever in the lick-active zone the animal happens to
            lick rather than at a single point.

        Args:
            session_path: Path to the session's dataset directory.
            trial_type: Trial type to analyze (e.g. "ABC", "ABCD"). Must match an entry in the session's trial
                geometry data file.
            fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the
                analysis input.
            configuration: Configuration parameters for detection thresholds and shuffle testing. Uses defaults if None.
        """
        session = assemble_run_session_data(
            session_path=session_path,
            trial_type=trial_type,
            fluorescence_column=fluorescence_column,
        )
        self.fluorescence = session.fluorescence
        self.position = session.position
        self.speed = session.speed
        self.trial_ids = session.trial_ids
        self.track_length = session.geometry.trial_length_cm
        self.reward_position = (
            session.geometry.stimulus_trigger_zone_start_cm + session.geometry.stimulus_trigger_zone_end_cm
        ) / 2.0
        self.configuration = configuration if configuration is not None else RewardCellConfiguration()

    def detect(self) -> RewardCellResults:
        """Runs the full reward cell detection pipeline.

        Notes:
            Computes spatial rate maps with shuffle-based significance testing, then fits a uniform + Gaussian mixture
            model to the significant neurons' center-of-mass distribution to identify excess field density near the
            reward zone. Classifies reward-proximal neurons and tests them for speed-activity correlation in the
            pre-reward window using trial-label permutation.

        Returns:
            A RewardCellResults instance containing spatial modulation results, reward classification, mixture model
            parameters, and slowing-correlation analysis.
        """
        spatial_results = self._compute_spatial_modulation()

        # Fits a uniform + Gaussian mixture model to the significant neurons' center-of-mass distribution.
        mixture_weight, gaussian_mean, gaussian_std = _compute_uniform_gaussian_mixture(
            centers=spatial_results.significant_centers,
            track_length=self.track_length,
            reward_position=self.reward_position,
        )

        # Classifies neurons as reward-proximal based on circular distance from reward position.
        is_reward_proximal = self._classify_reward_proximal(
            centers_of_mass=spatial_results.centers_of_mass,
        )

        # Computes speed-activity correlations and identifies slowing-correlated neurons.
        speed_correlations, is_slowing_correlated = self._compute_slowing_correlations(
            spatial_results=spatial_results,
            is_reward_proximal=is_reward_proximal,
        )

        return RewardCellResults(
            spatial_results=spatial_results,
            reward_position=self.reward_position,
            mixture_weight=mixture_weight,
            gaussian_mean=gaussian_mean,
            gaussian_std=gaussian_std,
            is_reward_proximal=is_reward_proximal,
            speed_activity_correlations=speed_correlations,
            is_slowing_correlated=is_slowing_correlated,
        )

    def _compute_spatial_modulation(self) -> SpatiallyModulatedNeurons:
        """Computes spatial rate maps, spatial information, and shuffle-based significance for all neurons.

        Returns:
            A SpatiallyModulatedNeurons instance with rate maps, spatial information, significance masks, and
            centers of mass.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]

        # Applies speed filtering and computes bin assignments.
        # noinspection PyTypeChecker
        speed_mask: NDArray[np.bool_] = self.speed > configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        # Reuses place_1d binning to compute mean fluorescence per spatial bin.
        # noinspection PyTypeChecker
        bin_edges: NDArray[np.float32] = np.arange(
            0, self.track_length + configuration.bin_size, configuration.bin_size, dtype=np.float32
        )

        rate_maps, sample_counts = bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            position_bin_edges=bin_edges,
            compute_mean=True,
        )

        # Replaces NaN bins (unvisited) with zero for downstream computation.
        rate_maps = np.nan_to_num(rate_maps, nan=0.0)

        # Applies Gaussian smoothing with circular wrapping at track edges.
        sigma_bins = configuration.gaussian_sigma / configuration.bin_size
        smoothed_maps = _apply_smooth_rate_maps_wrapped(rate_maps=rate_maps, sigma_bins=sigma_bins)

        # Computes spatial information for the observed data.
        # noinspection PyTypeChecker
        observed_information: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        _compute_spatial_information(
            rate_maps=smoothed_maps,
            occupancy=sample_counts,
            information=observed_information,
        )

        # Runs shuffle significance testing.
        p_values = self._compute_shuffle_significance(
            filtered_position=filtered_position,
            speed_mask=speed_mask,
            bin_edges=bin_edges,
            occupancy=sample_counts,
            sigma_bins=sigma_bins,
            observed_information=observed_information,
        )

        # noinspection PyTypeChecker
        is_significant: NDArray[np.bool_] = p_values < configuration.significance_threshold

        # Computes the circular center of mass for each neuron.
        # noinspection PyTypeChecker
        centers_of_mass: NDArray[np.float32] = np.full(cell_count, -1.0, dtype=np.float32)
        _compute_circular_center_of_mass(
            rate_maps=smoothed_maps,
            track_length=self.track_length,
            centers=centers_of_mass,
        )

        return SpatiallyModulatedNeurons(
            rate_maps=smoothed_maps,
            occupancy=sample_counts,
            spatial_information=observed_information,
            is_significant=is_significant,
            p_values=p_values,
            centers_of_mass=centers_of_mass,
            bin_size=configuration.bin_size,
            track_length=self.track_length,
        )

    def _compute_shuffle_significance(
        self,
        filtered_position: NDArray[np.float32],
        speed_mask: NDArray[np.bool_],
        bin_edges: NDArray[np.float32],
        occupancy: NDArray[np.int32],
        sigma_bins: float,
        observed_information: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Computes p-values by comparing observed spatial information to a null distribution from shuffled data.

        Notes:
            For each shuffle iteration, the fluorescence time series is circularly shifted by at least minimum_shift
            samples and then split into chunks that are randomly permuted. Only the speed-filtered
            subset of fluorescence is shuffled and rebinned using the shared place_1d binning function.

        Args:
            filtered_position: Speed-filtered position values with length filtered_sample_count.
            speed_mask: Boolean mask indicating speed-filtered samples with length sample_count.
            bin_edges: Spatial bin edges with length bin_count + 1.
            occupancy: Per-bin occupancy sample counts with length bin_count.
            sigma_bins: Gaussian smoothing kernel width in bin units.
            observed_information: Observed spatial information values with length cell_count.

        Returns:
            Array of p-values with length cell_count.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]
        sample_count = self.fluorescence.shape[1]
        bin_count = len(bin_edges) - 1

        # Precomputes destination-sample indices and their spatial bin assignments; both are invariant across shuffles.
        # noinspection PyTypeChecker
        filtered_sample_indices: NDArray[np.int32] = np.nonzero(speed_mask)[0].astype(np.int32)
        # noinspection PyTypeChecker
        filtered_bin_indices: NDArray[np.int32] = np.clip(
            np.searchsorted(bin_edges, filtered_position, side="right") - 1, 0, bin_count - 1
        ).astype(np.int32)

        # Reuses rate-map and information buffers across iterations.
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.empty((cell_count, bin_count), dtype=np.float32)
        # noinspection PyTypeChecker
        shuffled_information: NDArray[np.float32] = np.zeros(
            (configuration.shuffle_count, cell_count), dtype=np.float32
        )

        for iteration in tqdm(range(configuration.shuffle_count), desc="Running shuffling", unit="iter"):
            source_indices = _compute_shuffled_source_indices(
                filtered_sample_indices=filtered_sample_indices,
                sample_count=sample_count,
                minimum_shift=configuration.minimum_shift_samples,
                chunk_count=configuration.chunk_count,
                seed=iteration,
            )

            _accumulate_shuffled_rate_maps(
                fluorescence=self.fluorescence,
                source_indices=source_indices,
                bin_indices=filtered_bin_indices,
                sample_counts=occupancy,
                output=rate_maps,
            )

            smoothed_shuffled = _apply_smooth_rate_maps_wrapped(rate_maps=rate_maps, sigma_bins=sigma_bins)

            _compute_spatial_information(
                rate_maps=smoothed_shuffled,
                occupancy=occupancy,
                information=shuffled_information[iteration],
            )

        # Computes p-values as the fraction of shuffles exceeding observed.
        exceed_count = np.sum(shuffled_information >= observed_information[np.newaxis, :], axis=0)
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = (exceed_count / configuration.shuffle_count).astype(np.float32)

        return p_values

    def _classify_reward_proximal(
        self,
        centers_of_mass: NDArray[np.float32],
    ) -> NDArray[np.bool_]:
        """Classifies neurons as reward-proximal based on circular distance of their COM from the reward position.

        Args:
            centers_of_mass: Center-of-mass positions in centimeters with length cell_count.

        Returns:
            Boolean mask with length cell_count indicating reward-proximal neurons.
        """
        half_zone = self.configuration.reward_zone_width / 2.0

        # Computes the minimum circular distance between each COM and the reward position.
        direct_distance = np.abs(centers_of_mass - self.reward_position)
        circular_distance = np.minimum(direct_distance, self.track_length - direct_distance)

        # Marks neurons with invalid COM (-1) as non-reward-proximal.
        # noinspection PyTypeChecker
        is_valid: NDArray[np.bool_] = centers_of_mass >= 0.0
        return is_valid & (circular_distance <= half_zone)

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

    def _compute_slowing_correlations(
        self,
        spatial_results: SpatiallyModulatedNeurons,
        is_reward_proximal: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """Identifies slowing-correlated neurons using per-trial speed-activity correlation in the pre-reward window.

        Notes:
            Only reward-proximal neurons are tested. For each candidate neuron, per-trial vectors of spatially binned
            activity and speed in the pre-reward window are built. The Pearson correlation is computed across the
            flattened trial x bin matrices. Significance is determined by permuting trial labels of the activity matrix
            and comparing the observed correlation to the shuffle distribution.

        Args:
            spatial_results: The spatial modulation results containing rate maps and occupancy.
            is_reward_proximal: Boolean mask indicating reward-proximal neurons with length cell_count.

        Returns:
            A tuple of (correlations, is_slowing_correlated) where correlations has length cell_count and
            is_slowing_correlated is a boolean mask with length cell_count.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]

        # noinspection PyTypeChecker
        observed_correlations: NDArray[np.float32] = np.zeros(cell_count, dtype=np.float32)
        # noinspection PyTypeChecker
        is_slowing_correlated: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)

        # Defines the pre-reward spatial window.
        window_start = self.reward_position - configuration.pre_reward_window
        window_end = self.reward_position
        # noinspection PyTypeChecker
        pre_reward_bin_edges: NDArray[np.float32] = np.arange(
            window_start, window_end + configuration.bin_size, configuration.bin_size, dtype=np.float32
        )

        if len(pre_reward_bin_edges) - 1 < _MINIMUM_PRE_REWARD_BIN_COUNT:
            return observed_correlations, is_slowing_correlated

        # Identifies candidate neurons: reward-proximal and spatially significant.
        candidate_indices = np.argwhere(is_reward_proximal & spatial_results.is_significant).flatten()
        if len(candidate_indices) == 0:
            return observed_correlations, is_slowing_correlated

        # noinspection PyTypeChecker
        unique_trials: NDArray[np.int32] = np.unique(self.trial_ids).astype(np.int32)
        trial_count = len(unique_trials)

        # Builds the shared per-trial speed matrix in the pre-reward window.
        speed_sums, speed_counts = self._bin_trials_in_pre_reward_window(
            signal=self.speed,
            unique_trials=unique_trials,
            bin_edges=pre_reward_bin_edges,
            window_start=window_start,
            window_end=window_end,
        )
        # noinspection PyTypeChecker
        valid_speed: NDArray[np.bool_] = speed_counts > 0
        speed_sums[valid_speed] /= speed_counts[valid_speed]
        speed_flat = speed_sums.flatten()

        # Processes each candidate neuron.
        for neuron_index in candidate_indices:
            activity_sums, activity_counts = self._bin_trials_in_pre_reward_window(
                signal=self.fluorescence[neuron_index],
                unique_trials=unique_trials,
                bin_edges=pre_reward_bin_edges,
                window_start=window_start,
                window_end=window_end,
            )

            # Counts trials with nonzero total activity in the window.
            active_trial_count = int(np.sum(activity_sums.sum(axis=1) > 0))
            if active_trial_count < configuration.minimum_active_trials:
                continue

            # Converts activity sums to means.
            # noinspection PyTypeChecker
            valid_activity: NDArray[np.bool_] = activity_counts > 0
            activity_sums[valid_activity] /= activity_counts[valid_activity]
            activity_flat = activity_sums.flatten()

            # Selects entries where both speed and activity have data.
            # noinspection PyTypeChecker
            valid_mask: NDArray[np.bool_] = (speed_flat != 0) & (activity_flat != 0)
            if np.sum(valid_mask) < _MINIMUM_VALID_SAMPLE_COUNT:
                continue

            speed_valid = speed_flat[valid_mask]
            activity_valid = activity_flat[valid_mask]

            observed_correlation = np.float32(np.corrcoef(speed_valid, activity_valid)[0, 1])
            observed_correlations[neuron_index] = observed_correlation

            # Generates shuffle distribution by permuting trial labels of the activity matrix.
            # noinspection PyTypeChecker
            shuffle_correlations: NDArray[np.float32] = np.zeros(configuration.slowing_shuffle_count, dtype=np.float32)
            for shuffle_index in range(configuration.slowing_shuffle_count):
                random_generator = np.random.default_rng(seed=shuffle_index)
                permuted_flat = activity_sums[random_generator.permutation(trial_count)].flatten()
                permuted_valid = permuted_flat[valid_mask]

                if np.std(permuted_valid) > 0:
                    shuffle_correlations[shuffle_index] = np.corrcoef(speed_valid, permuted_valid)[0, 1]

            # Classifies as slowing-correlated if observed correlation is below the significance percentile.
            threshold = np.quantile(shuffle_correlations, configuration.slowing_significance_threshold)
            is_slowing_correlated[neuron_index] = observed_correlation < threshold

        return observed_correlations, is_slowing_correlated
