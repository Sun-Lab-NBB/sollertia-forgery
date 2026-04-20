"""Identifies reward-associated and reward-predictive neurons from spatial and speed-activity data."""

from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
from numba import njit, prange
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from sl_forgery.analysis.utilities import compute_within_trial_position
from sl_forgery.analysis.place_cell_analysis import _bin_fluorescence_by_position


@dataclass
class RewardCellConfiguration:
    """Defines configuration parameters for reward cell detection."""

    bin_size: float = 10.0
    """Spatial bin size in centimeters for position binning."""
    minimum_speed: float = 5.0
    """Minimum speed threshold in cm/s for including frames in analysis."""
    gaussian_sigma: float = 20.0
    """Standard deviation in centimeters for Gaussian spatial smoothing of rate maps."""
    shuffle_count: int = 100
    """Number of shuffle iterations for significance testing."""
    minimum_shift_frames: int = 500
    """Minimum circular shift in frames applied during shuffle."""
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


@dataclass
class SpatiallyModulatedNeurons:
    """Stores results of spatial modulation analysis for neurons on a linear or circular track."""

    rate_maps: NDArray[np.float32]
    """Smoothed spatial rate maps with dimensions (cell_count, bin_count)."""
    occupancy: NDArray[np.int32]
    """Per-bin occupancy sample counts with length bin_count (from _bin_fluorescence_by_position)."""
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


@dataclass
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
    filtered_frame_indices: NDArray[np.int32],
    frame_count: int,
    minimum_shift: int,
    chunk_count: int,
    seed: int,
) -> NDArray[np.int32]:
    """Maps each speed-filtered destination frame back to the source frame it pulls from under the shuffle.

    Notes:
        Encodes the circular shift and chunk permutation as an indirection array rather than materializing a full
        shuffled fluorescence matrix.

    Args:
        filtered_frame_indices: Destination-frame indices retained by the speed filter with length
            filtered_frame_count.
        frame_count: Total number of frames in the original fluorescence time series.
        minimum_shift: Minimum number of frames for the circular shift.
        chunk_count: Number of chunks to split the shifted trace into for permutation.
        seed: Random seed for reproducibility.

    Returns:
        Source-frame indices with length filtered_frame_count.
    """
    np.random.seed(seed)
    shift_amount = np.random.randint(minimum_shift, frame_count - minimum_shift)
    chunk_size = frame_count // chunk_count
    permutation = np.random.permutation(chunk_count)

    # Computes cumulative output-chunk start positions so each destination can be located within the permuted layout.
    output_chunk_starts = np.empty(chunk_count + 1, dtype=np.int32)
    output_chunk_starts[0] = 0
    for output_chunk_position in range(chunk_count):
        source_chunk_index = permutation[output_chunk_position]
        if source_chunk_index < chunk_count - 1:
            chunk_size_local = chunk_size
        else:
            chunk_size_local = frame_count - source_chunk_index * chunk_size
        output_chunk_starts[output_chunk_position + 1] = output_chunk_starts[output_chunk_position] + chunk_size_local

    filtered_count = filtered_frame_indices.shape[0]
    source_indices = np.empty(filtered_count, dtype=np.int32)

    # Resolves each destination back through the permutation and shift to its source frame.
    for filtered_index in range(filtered_count):
        destination = filtered_frame_indices[filtered_index]
        output_chunk_position = 0
        while output_chunk_position + 1 < chunk_count and output_chunk_starts[output_chunk_position + 1] <= destination:
            output_chunk_position += 1
        offset_within_chunk = destination - output_chunk_starts[output_chunk_position]
        source_chunk_index = permutation[output_chunk_position]
        shifted_index = source_chunk_index * chunk_size + offset_within_chunk
        source_indices[filtered_index] = (shifted_index - shift_amount) % frame_count

    return source_indices


@njit(cache=True, parallel=True, nogil=True)
def _accumulate_shuffled_rate_maps(
    fluorescence: NDArray[np.float32],
    source_indices: NDArray[np.int32],
    bin_indices: NDArray[np.int32],
    sample_counts: NDArray[np.int32],
    output: NDArray[np.float32],
) -> None:
    """Bins fluorescence into a per-cell rate map by gathering source frames through an indirection array.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, frame_count).
        source_indices: Source-frame indices per filtered destination frame with length filtered_frame_count.
        bin_indices: Spatial bin indices per filtered destination frame with length filtered_frame_count.
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
        track_length: float,
        reward_position: float,
        fluorescence_column: str = "single_day_dff",
        trial_type: str | None = None,
        configuration: RewardCellConfiguration | None = None,
    ) -> None:
        """Loads fluorescence, position, speed, and trial data from a memory-mapped feather file for reward cell
        detection. Filters to 'run' state frames and computes within-trial position.

        Args:
            session_path: Path to the session feather file.
            track_length: Length of the track in centimeters.
            reward_position: Position of the reward zone in centimeters.
            fluorescence_column: Name of the fluorescence column to use.
            trial_type: Trial type to filter by (e.g. "ABC", "ABCD"). If None, includes all trial types.
            configuration: Configuration parameters for detection thresholds and shuffle testing. Uses defaults if None.
        """
        df = pl.read_ipc(
            session_path,
            columns=["system_state", "trial_type", fluorescence_column, "distance_cm", "speed_cm_s", "trial"],
        )
        df = df.filter(pl.col("system_state") == "run")
        if trial_type is not None:
            df = df.filter(pl.col("trial_type") == trial_type)

        # Extracts fluorescence data and transposes from (frame, cell) to (cell, frame).
        self.fluorescence = np.array(df[fluorescence_column].to_list(), dtype=np.float32).T

        distance = df["distance_cm"].to_numpy().astype(np.float32)
        self.trial_ids = df["trial"].to_numpy().astype(np.int32)

        # Computes within-trial position to avoid inter-trial drift from global modulo.
        self.position = compute_within_trial_position(distance=distance, trial_ids=self.trial_ids)
        self.speed = df["speed_cm_s"].to_numpy().astype(np.float32)
        self.track_length = track_length
        self.reward_position = reward_position
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
        speed_mask = self.speed > configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        # Reuses place_1d binning to compute mean fluorescence per spatial bin.
        bin_edges = np.arange(0, self.track_length + configuration.bin_size, configuration.bin_size, dtype=np.float32)

        rate_maps, sample_counts = _bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            bin_edges=bin_edges,
            compute_mean=True,
        )

        # Replaces NaN bins (unvisited) with zero for downstream computation.
        rate_maps = np.nan_to_num(rate_maps, nan=0.0)

        # Applies Gaussian smoothing with circular wrapping at track edges.
        sigma_bins = configuration.gaussian_sigma / configuration.bin_size
        smoothed_maps = _apply_smooth_rate_maps_wrapped(rate_maps=rate_maps, sigma_bins=sigma_bins)

        # Computes spatial information for the observed data.
        observed_information = np.zeros(cell_count, dtype=np.float32)
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

        is_significant = p_values < configuration.significance_threshold

        # Computes the circular center of mass for each neuron.
        centers_of_mass = np.full(cell_count, -1.0, dtype=np.float32)
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
            frames and then split into chunks that are randomly permuted. Only the speed-filtered
            subset of fluorescence is shuffled and rebinned using the shared place_1d binning function.

        Args:
            filtered_position: Speed-filtered position values with length filtered_frame_count.
            speed_mask: Boolean mask indicating speed-filtered frames with length frame_count.
            bin_edges: Spatial bin edges with length bin_count + 1.
            occupancy: Per-bin occupancy sample counts with length bin_count.
            sigma_bins: Gaussian smoothing kernel width in bin units.
            observed_information: Observed spatial information values with length cell_count.

        Returns:
            Array of p-values with length cell_count.
        """
        configuration = self.configuration
        cell_count = self.fluorescence.shape[0]
        frame_count = self.fluorescence.shape[1]
        bin_count = len(bin_edges) - 1

        # Precomputes destination-frame indices and their spatial bin assignments; both are invariant across shuffles.
        filtered_frame_indices = np.nonzero(speed_mask)[0].astype(np.int32)
        filtered_bin_indices = np.clip(
            np.searchsorted(bin_edges, filtered_position, side="right") - 1, 0, bin_count - 1
        ).astype(np.int32)

        # Reuses rate-map and information buffers across iterations.
        rate_maps = np.empty((cell_count, bin_count), dtype=np.float32)
        shuffled_information = np.zeros((configuration.shuffle_count, cell_count), dtype=np.float32)

        for iteration in tqdm(range(configuration.shuffle_count), desc="Running shuffling", unit="iter"):
            source_indices = _compute_shuffled_source_indices(
                filtered_frame_indices=filtered_frame_indices,
                frame_count=frame_count,
                minimum_shift=configuration.minimum_shift_frames,
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
        p_values = (exceed_count / configuration.shuffle_count).astype(np.float32)

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
        is_valid = centers_of_mass >= 0.0
        return is_valid & (circular_distance <= half_zone)

    def _bin_trials_in_pre_reward_window(
        self,
        signal: NDArray[np.float32],
        unique_trials: NDArray[np.int32],
        bin_edges: NDArray[np.float32],
        window_start: float,
        window_end: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Bins a per-frame signal into a (trial_count, bin_count) matrix within the pre-reward spatial window.

        Args:
            signal: Per-frame values to bin with length frame_count.
            unique_trials: Sorted unique trial identifiers.
            bin_edges: Spatial bin edges for the pre-reward window with length bin_count + 1.
            window_start: Start of the pre-reward spatial window in centimeters.
            window_end: End of the pre-reward spatial window in centimeters.

        Returns:
            A tuple of (sums, counts) where sums has the accumulated signal per bin and counts has the number of
            frames per bin, both with dimensions (trial_count, bin_count).
        """
        bin_count = len(bin_edges) - 1
        trial_count = len(unique_trials)
        sums = np.zeros((trial_count, bin_count), dtype=np.float32)
        counts = np.zeros((trial_count, bin_count), dtype=np.int32)

        for trial_index, trial_id in enumerate(unique_trials):
            trial_mask = self.trial_ids == trial_id
            trial_positions = self.position[trial_mask]
            trial_speeds = self.speed[trial_mask]
            trial_signal = signal[trial_mask]

            # Restricts to the pre-reward window and speed-filtered frames.
            window_mask = (
                (trial_positions >= window_start)
                & (trial_positions < window_end)
                & (trial_speeds > self.configuration.minimum_speed)
            )
            window_positions = trial_positions[window_mask]
            window_signal = trial_signal[window_mask]

            bin_indices = np.clip(np.searchsorted(bin_edges, window_positions, side="right") - 1, 0, bin_count - 1)

            for frame_index in range(len(bin_indices)):
                bin_index = bin_indices[frame_index]
                sums[trial_index, bin_index] += window_signal[frame_index]
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

        observed_correlations = np.zeros(cell_count, dtype=np.float32)
        is_slowing_correlated = np.zeros(cell_count, dtype=np.bool_)

        # Defines the pre-reward spatial window.
        window_start = self.reward_position - configuration.pre_reward_window
        window_end = self.reward_position
        pre_reward_bin_edges = np.arange(
            window_start, window_end + configuration.bin_size, configuration.bin_size, dtype=np.float32
        )

        if len(pre_reward_bin_edges) - 1 < 2:
            return observed_correlations, is_slowing_correlated

        # Identifies candidate neurons: reward-proximal and spatially significant.
        candidate_indices = np.argwhere(is_reward_proximal & spatial_results.is_significant).flatten()
        if len(candidate_indices) == 0:
            return observed_correlations, is_slowing_correlated

        unique_trials = np.unique(self.trial_ids)
        trial_count = len(unique_trials)

        # Builds the shared per-trial speed matrix in the pre-reward window.
        speed_sums, speed_counts = self._bin_trials_in_pre_reward_window(
            signal=self.speed,
            unique_trials=unique_trials,
            bin_edges=pre_reward_bin_edges,
            window_start=window_start,
            window_end=window_end,
        )
        valid_speed = speed_counts > 0
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
            valid_activity = activity_counts > 0
            activity_sums[valid_activity] /= activity_counts[valid_activity]
            activity_flat = activity_sums.flatten()

            # Selects entries where both speed and activity have data.
            valid_mask = (speed_flat != 0) & (activity_flat != 0)
            if np.sum(valid_mask) < 3:
                continue

            speed_valid = speed_flat[valid_mask]
            activity_valid = activity_flat[valid_mask]

            observed_correlation = np.float32(np.corrcoef(speed_valid, activity_valid)[0, 1])
            observed_correlations[neuron_index] = observed_correlation

            # Generates shuffle distribution by permuting trial labels of the activity matrix.
            shuffle_correlations = np.zeros(configuration.slowing_shuffle_count, dtype=np.float32)
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

    def plot_com_histogram(
        self,
        results: RewardCellResults,
        bin_count: int = 20,
        title: str | None = None,
        figure_dpi: int = 150,
    ) -> plt.Figure:
        """Plots the center-of-mass histogram with the fitted uniform + Gaussian mixture model overlay.

        Args:
            results: The RewardCellResults from a completed detect() call.
            bin_count: Number of histogram bins along the track.
            title: Optional title for the figure.
            figure_dpi: Resolution of the figure in dots per inch.

        Returns:
            The matplotlib Figure object.
        """
        spatial = results.spatial_results
        centers = spatial.significant_centers
        valid_centers = centers[centers >= 0.0]

        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)

        # Plots the histogram of COM positions for spatially significant neurons.
        hist_bins = np.linspace(0, self.track_length, bin_count + 1)
        axes.hist(valid_centers, bins=hist_bins, color="0.7", edgecolor="0.5", density=True, label="Observed COMs")

        # Overlays the fitted mixture model components.
        positions = np.linspace(0, self.track_length, 200)
        uniform_density = np.full_like(positions, 1.0 / self.track_length)
        gaussian_density = np.exp(-0.5 * ((positions - results.gaussian_mean) / results.gaussian_std) ** 2) / (
            results.gaussian_std * np.sqrt(2.0 * np.pi)
        )
        mixture_density = (1.0 - results.mixture_weight) * uniform_density + results.mixture_weight * gaussian_density

        axes.fill_between(
            positions,
            0,
            (1.0 - results.mixture_weight) * uniform_density,
            alpha=0.3,
            color="lightblue",
            label="Uniform (place cells)",
        )
        axes.fill_between(
            positions,
            (1.0 - results.mixture_weight) * uniform_density,
            mixture_density,
            alpha=0.4,
            color="mediumpurple",
            label="Gaussian (reward cells)",
        )
        axes.plot(positions, mixture_density, color="black", linewidth=1.5, label="Mixture fit")

        # Marks the reward position with a vertical line.
        axes.axvline(x=self.reward_position, color="red", linestyle="--", linewidth=1.5, label="Reward location")

        axes.set_xlabel("Track Position (cm)")
        axes.set_ylabel("Density")
        axes.legend(fontsize=7, loc="upper left")

        annotation_text = (
            f"Significant: {spatial.significant_count}/{len(spatial.is_significant)} cells\n"
            f"Mixture weight: {results.mixture_weight:.1%} reward\n"
            f"Gaussian center: {results.gaussian_mean:.0f} cm (SD {results.gaussian_std:.0f} cm)"
        )
        axes.text(
            0.98,
            0.95,
            annotation_text,
            transform=axes.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        if title:
            axes.set_title(title, fontsize=9)

        figure.tight_layout()
        return figure

    def plot_rate_map_heatmap(
        self,
        results: RewardCellResults,
        title: str | None = None,
        figure_dpi: int = 150,
        plot_bin_size: float = 2.0,
        plot_sigma: float = 5.0,
    ) -> plt.Figure:
        """Plots row-normalized rate maps for reward cells and place cells as side-by-side panels sorted by COM.

        Args:
            results: The RewardCellResults from a completed detect() call.
            title: Optional title for the figure.
            figure_dpi: Resolution of the figure in dots per inch.
            plot_bin_size: Spatial bin size in centimeters for the visualization rate maps.
            plot_sigma: Standard deviation in centimeters for Gaussian smoothing of the visualization rate maps.

        Returns:
            The matplotlib Figure object.
        """
        spatial = results.spatial_results
        significant_mask = spatial.is_significant
        reward_mask = significant_mask & results.is_reward_proximal
        place_mask = significant_mask & ~results.is_reward_proximal

        # Recomputes rate maps at finer resolution for visualization.
        speed_mask = self.speed > self.configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        plot_bin_edges = np.arange(0, self.track_length + plot_bin_size, plot_bin_size, dtype=np.float32)
        plot_maps, _ = _bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            bin_edges=plot_bin_edges,
            compute_mean=True,
        )
        plot_maps = np.nan_to_num(plot_maps, nan=0.0)

        plot_sigma_bins = plot_sigma / plot_bin_size
        plot_maps = _apply_smooth_rate_maps_wrapped(rate_maps=plot_maps, sigma_bins=plot_sigma_bins)

        reward_left = self.reward_position - self.configuration.reward_zone_width / 2.0
        reward_right = self.reward_position + self.configuration.reward_zone_width / 2.0

        figure, (axes_reward, axes_place) = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            facecolor="white",
            dpi=figure_dpi,
            sharey=False,
        )

        for axes, mask, panel_title in [
            (axes_reward, reward_mask, "Reward Cells"),
            (axes_place, place_mask, "Place Cells"),
        ]:
            maps = plot_maps[mask]
            coms = spatial.centers_of_mass[mask]

            # Sorts neurons by their COM position along the track.
            sort_order = np.argsort(coms)
            sorted_maps = maps[sort_order]

            # Normalizes each row to [0, 1] so that field structure is visible regardless of absolute rate.
            row_maxima = sorted_maps.max(axis=1, keepdims=True)
            row_maxima[row_maxima == 0] = 1.0
            normalized_maps = sorted_maps / row_maxima

            extent = [0, plot_bin_size * sorted_maps.shape[1], normalized_maps.shape[0], 0]
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
            axes.set_xticks(np.arange(0, self.track_length + 1, 25))
            axes.set_title(f"{panel_title} (n={int(np.sum(mask))})", fontsize=9)

        axes_reward.set_ylabel("Neuron (sorted by COM)")

        if title:
            figure.suptitle(title, fontsize=9)
            figure.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            figure.tight_layout()

        return figure

    def plot_population_activity_by_position(
        self,
        results: RewardCellResults,
        title: str | None = None,
        figure_dpi: int = 150,
        plot_bin_size: float = 2.0,
        plot_sigma: float = 5.0,
    ) -> plt.Figure:
        """Plots mean population fluorescence by track position, comparing reward-predictive cells against all spatially
        modulated cells.

        Args:
            results: The RewardCellResults from a completed detect() call.
            title: Optional title for the figure.
            figure_dpi: Resolution of the figure in dots per inch.
            plot_bin_size: Spatial bin size in centimeters for the visualization rate maps.
            plot_sigma: Standard deviation in centimeters for Gaussian smoothing of the visualization rate maps.

        Returns:
            The matplotlib Figure object.
        """
        spatial = results.spatial_results

        # Recomputes rate maps at fine resolution.
        speed_mask = self.speed > self.configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        plot_bin_edges = np.arange(0, self.track_length + plot_bin_size, plot_bin_size, dtype=np.float32)
        plot_maps, _ = _bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            bin_edges=plot_bin_edges,
            compute_mean=True,
        )
        plot_maps = np.nan_to_num(plot_maps, nan=0.0)
        plot_sigma_bins = plot_sigma / plot_bin_size
        plot_maps = _apply_smooth_rate_maps_wrapped(rate_maps=plot_maps, sigma_bins=plot_sigma_bins)

        bin_centers = (plot_bin_edges[:-1] + plot_bin_edges[1:]) / 2.0

        # Computes population means for the two key classes.
        all_significant = spatial.is_significant
        reward_predictive_mask = all_significant & results.is_reward_proximal & results.is_slowing_correlated

        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)

        # Plots all spatially modulated cells as the reference population (gray).
        if np.sum(all_significant) > 0:
            mean_all = np.mean(plot_maps[all_significant], axis=0)
            axes.fill_between(bin_centers, 0, mean_all, color="0.8", alpha=0.6)
            axes.plot(
                bin_centers,
                mean_all,
                color="0.5",
                linewidth=1.5,
                label=f"All spatially modulated (n={int(np.sum(all_significant))})",
            )

        # Plots reward-predictive (slowing-correlated) cells as the colored trace.
        if np.sum(reward_predictive_mask) > 0:
            mean_predictive = np.mean(plot_maps[reward_predictive_mask], axis=0)
            axes.fill_between(bin_centers, 0, mean_predictive, color="mediumpurple", alpha=0.3)
            axes.plot(
                bin_centers,
                mean_predictive,
                color="darkviolet",
                linewidth=2.5,
                label=f"Slowing-correlated (n={int(np.sum(reward_predictive_mask))})",
            )

        # Marks the reward location with a solid red vertical line.
        axes.axvline(
            x=self.reward_position,
            color="red",
            linestyle="-",
            linewidth=2.0,
            alpha=0.8,
            label="Reward location",
        )

        axes.set_xlabel("Track Position (cm)")
        axes.set_ylabel("Average Fluorescence (dF/F)")
        axes.legend(fontsize=7, loc="upper left")
        axes.set_xticks(np.arange(0, self.track_length + 1, 25))

        if title:
            axes.set_title(title, fontsize=9)

        figure.tight_layout()
        return figure

    def plot_speed_and_activity_by_position(
        self,
        results: RewardCellResults,
        title: str | None = None,
        figure_dpi: int = 150,
        position_bin_size: float = 2.0,
        position_sigma: float = 5.0,
    ) -> plt.Figure:
        """Plots trial-averaged speed and reward-predictive cell activity by track position on dual axes.

        Notes:
            Shows the spatial relationship between the animal's running speed profile and reward-predictive cell
            activity. If reward-predictive cells encode reward proximity rather than speed per se, their activity peak
            should precede the speed minimum and be sharper than the speed trough.

        Args:
            results: The RewardCellResults from a completed detect() call.
            title: Optional title for the figure.
            figure_dpi: Resolution of the figure in dots per inch.
            position_bin_size: Spatial bin size in centimeters for binning speed and activity.
            position_sigma: Standard deviation in centimeters for Gaussian smoothing.

        Returns:
            The matplotlib Figure object.
        """
        spatial = results.spatial_results
        reward_predictive_mask = spatial.is_significant & results.is_reward_proximal & results.is_slowing_correlated

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

        # Bins speed by position across all frames.
        bin_edges = np.arange(0, self.track_length + position_bin_size, position_bin_size, dtype=np.float32)
        bin_count = len(bin_edges) - 1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        speed_sums = np.zeros(bin_count, dtype=np.float32)
        speed_counts = np.zeros(bin_count, dtype=np.int32)

        bin_indices = np.clip(
            np.searchsorted(bin_edges, self.position, side="right") - 1,
            0,
            bin_count - 1,
        )

        for frame_index in range(len(self.position)):
            bin_index = bin_indices[frame_index]
            speed_sums[bin_index] += self.speed[frame_index]
            speed_counts[bin_index] += 1

        mean_speed = np.zeros(bin_count, dtype=np.float32)
        valid = speed_counts > 0
        mean_speed[valid] = speed_sums[valid] / speed_counts[valid]

        sigma_bins = position_sigma / position_bin_size
        mean_speed = gaussian_filter1d(input=mean_speed, sigma=sigma_bins, mode="wrap")

        # Computes mean fluorescence for reward-predictive cells at the same resolution.
        speed_mask = self.speed > self.configuration.minimum_speed
        filtered_position = self.position[speed_mask]
        filtered_fluorescence = self.fluorescence[:, speed_mask]

        plot_maps, _ = _bin_fluorescence_by_position(
            fluorescence=filtered_fluorescence,
            position=filtered_position,
            bin_edges=bin_edges,
            compute_mean=True,
        )
        plot_maps = np.nan_to_num(plot_maps, nan=0.0)
        plot_maps = _apply_smooth_rate_maps_wrapped(rate_maps=plot_maps, sigma_bins=sigma_bins)

        mean_activity = np.mean(plot_maps[reward_predictive_mask], axis=0)

        figure, axes_speed = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes_activity = axes_speed.twinx()

        # Plots speed profile on the primary axis.
        axes_speed.plot(bin_centers, mean_speed, color="0.4", linewidth=1.5, label="Mean speed")
        axes_speed.set_xlabel("Track Position (cm)")
        axes_speed.set_ylabel("Speed (cm/s)", color="0.4")
        axes_speed.tick_params(axis="y", labelcolor="0.4")

        # Plots reward-predictive cell activity on the secondary axis.
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

        # Marks the reward zone.
        reward_left = self.reward_position - self.configuration.reward_zone_width / 2.0
        reward_right = self.reward_position + self.configuration.reward_zone_width / 2.0
        axes_speed.axvspan(reward_left, reward_right, alpha=0.1, color="red", label="Reward zone")

        axes_speed.set_xticks(np.arange(0, self.track_length + 1, 25))

        # Combines legends from both axes.
        lines_speed, labels_speed = axes_speed.get_legend_handles_labels()
        lines_activity, labels_activity = axes_activity.get_legend_handles_labels()
        axes_speed.legend(lines_speed + lines_activity, labels_speed + labels_activity, fontsize=7, loc="upper left")

        if title:
            axes_speed.set_title(title, fontsize=9)

        figure.tight_layout()
        return figure

    def plot_per_trial_activity(
        self,
        results: RewardCellResults,
        title: str | None = None,
        figure_dpi: int = 150,
        position_bin_size: float = 2.0,
        position_sigma: float = 3.0,
        slowing_threshold_cm_s: float = 10.0,
    ) -> plt.Figure:
        """Plots per-trial activity heatmaps for an example reward-predictive and place cell, with slowing onset markers
        overlaid.

        Args:
            results: The RewardCellResults from a completed detect() call.
            title: Optional title for the figure.
            figure_dpi: Resolution of the figure in dots per inch.
            position_bin_size: Spatial bin size in centimeters for per-trial binning.
            position_sigma: Standard deviation in centimeters for Gaussian smoothing of per-trial activity.
            slowing_threshold_cm_s: Speed threshold in cm/s for detecting slowing onset on each trial.

        Returns:
            The matplotlib Figure object.
        """
        spatial = results.spatial_results

        # Selects the reward-predictive cell with the strongest negative correlation.
        predictive_mask = spatial.is_significant & results.is_reward_proximal & results.is_slowing_correlated
        predictive_indices = np.argwhere(predictive_mask).flatten()

        # Selects a place cell near the middle of the track (far from reward).
        place_mask = spatial.is_significant & ~results.is_reward_proximal
        place_indices = np.argwhere(place_mask).flatten()
        place_coms = spatial.centers_of_mass[place_indices]

        if len(predictive_indices) == 0 or len(place_indices) == 0:
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

        # Picks the reward-predictive cell with the most negative correlation.
        predictive_correlations = results.speed_activity_correlations[predictive_indices]
        best_predictive = predictive_indices[np.argmin(predictive_correlations)]

        # Picks a place cell whose COM is closest to track midpoint (farthest from reward).
        track_midpoint = self.track_length / 2.0
        mid_distances = np.abs(place_coms - track_midpoint)
        best_place = place_indices[np.argmin(mid_distances)]

        unique_trials = np.unique(self.trial_ids)
        trial_count = len(unique_trials)
        bin_edges = np.arange(0, self.track_length + position_bin_size, position_bin_size, dtype=np.float32)
        bin_count = len(bin_edges) - 1
        sigma_bins = position_sigma / position_bin_size

        reward_left = self.reward_position - self.configuration.reward_zone_width / 2.0
        reward_right = self.reward_position + self.configuration.reward_zone_width / 2.0

        figure, (axes_predictive, axes_place) = plt.subplots(
            1,
            2,
            figsize=(14, 8),
            facecolor="white",
            dpi=figure_dpi,
        )

        cells = [
            (axes_predictive, best_predictive, "Reward-predictive", "Purples"),
            (axes_place, best_place, "Place cell", "Blues"),
        ]

        pre_reward_start = self.reward_position - self.configuration.pre_reward_window

        for axes, cell_index, label, colormap in cells:
            activity_image = np.zeros((trial_count, bin_count), dtype=np.float32)
            slowing_onsets = np.full(trial_count, np.nan, dtype=np.float32)

            for trial_index, trial_id in enumerate(unique_trials):
                trial_mask = self.trial_ids == trial_id
                trial_positions = self.position[trial_mask]
                trial_speeds = self.speed[trial_mask]
                trial_fluorescence = self.fluorescence[cell_index, trial_mask]

                # Bins fluorescence by position using vectorized accumulation.
                trial_bin_indices = np.clip(
                    np.searchsorted(bin_edges, trial_positions, side="right") - 1, 0, bin_count - 1
                )
                activity_sums = np.zeros(bin_count, dtype=np.float32)
                activity_counts = np.zeros(bin_count, dtype=np.int32)
                np.add.at(activity_sums, trial_bin_indices, trial_fluorescence)
                np.add.at(activity_counts, trial_bin_indices, 1)
                valid = activity_counts > 0
                activity_image[trial_index, valid] = activity_sums[valid] / activity_counts[valid]

                # Detects slowing onset: first frame below threshold in the pre-reward window.
                pre_reward = (trial_positions >= pre_reward_start) & (trial_positions < self.reward_position)
                below = pre_reward & (trial_speeds < slowing_threshold_cm_s)
                if np.any(below):
                    slowing_onsets[trial_index] = trial_positions[below][0]

            # Smooths along the position axis and normalizes each row to [0, 1].
            activity_image = gaussian_filter1d(input=activity_image, sigma=sigma_bins, axis=1, mode="wrap")
            row_maxima = activity_image.max(axis=1, keepdims=True)
            row_maxima[row_maxima == 0] = 1.0
            normalized_activity = activity_image / row_maxima

            axes.imshow(
                normalized_activity,
                cmap=colormap,
                extent=[0, self.track_length, trial_count, 0],
                interpolation="none",
                vmin=0.0,
                vmax=1.0,
                origin="upper",
                aspect="auto",
                alpha=0.85,
            )

            # Overlays slowing onset markers.
            onset_trials = np.argwhere(~np.isnan(slowing_onsets)).flatten()
            for i, trial_index in enumerate(onset_trials):
                onset_label = f"Slowing onset (<{slowing_threshold_cm_s:.0f} cm/s)" if i == 0 else None
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
            axes.set_xticks(np.arange(0, self.track_length + 1, 25))

            cell_com = spatial.centers_of_mass[cell_index]
            cell_corr = results.speed_activity_correlations[cell_index]
            axes.set_title(f"{label} (cell {cell_index}, COM={cell_com:.0f} cm, r={cell_corr:.2f})", fontsize=9)

        axes_predictive.set_ylabel("Trial")

        if title:
            figure.suptitle(title, fontsize=9)
            figure.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            figure.tight_layout()

        return figure
