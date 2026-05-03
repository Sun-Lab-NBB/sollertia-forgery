"""Tuning-pipeline utilities: run-state assembly, per-trial binning, and shuffle-source helpers.

These helpers are package-private to the tuning analysis. The place- and reward-cell detectors share them so
both pipelines operate on bit-identical speed-filtered samples and rate-map binning. Helpers that any analysis
package may need (e.g. acquisition-warmup trimming) live in `..shared_utilities`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, interval_to_rate

from ...forging import FluorescenceColumn
from ...shared_assets import (
    DatasetFiles,
    DatasetColumn,
    TrialGeometry,
    TrialGeometryEntry,
)
from ..shared_utilities import realign_trial_starts_to_first_cue, trim_acquisition_warmup

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


MINIMUM_VALID_BINS_FOR_PEARSON: int = 3
"""Minimum number of pairwise-non-NaN bins required for a numerically stable per-cell Pearson r."""
_NO_TRIAL_SENTINEL: int = 255
"""Sentinel trial id the acquisition pipeline writes for samples outside any trial. Used by
`bin_fluorescence_per_trial` to drop the sentinel slot before the per-trial binning fans out."""


@dataclass(frozen=True, slots=True)
class RunSessionData:
    """Stores run-state arrays and trial geometry resolved from a forged session for a single trial type."""

    fluorescence: NDArray[np.float32]
    """Pre-normalized dF/F0 fluorescence with dimensions (cell_count, sample_count)."""
    position: NDArray[np.float32]
    """Within-trial position in centimeters at each sample."""
    speed: NDArray[np.float32]
    """Animal's speed in cm/s at each sample."""
    trial_ids: NDArray[np.int32]
    """Trial identifier at each sample."""
    trial_type: str
    """The trial type the loaded samples belong to."""
    geometry: TrialGeometryEntry
    """The canonical Virtual Reality environment geometry for the loaded trial type, including track length and
    stimulus trigger zone."""
    sampling_rate_hz: float
    """Acquisition sampling rate in Hz, computed from the median per-sample inter-time interval before any
    run-state filtering. NaN when the session has fewer than two samples."""


def assemble_run_session_data(
    session_path: Path,
    trial_type: str,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
) -> RunSessionData:
    """Assembles run-state arrays and trial geometry from a forged session for the given trial type.

    Notes:
        Resolves the canonical track length from the session's trial geometry data file, drops the leading
        acquisition-warmup window so downstream binning operates on stabilized samples, filters the session's
        data feather to system_state == 'run' and trial_type == trial_type, computes within-trial position, and
        drops samples belonging to incomplete trials so downstream binning never sees NaN positions. All returned
        arrays share the same sample axis and are aligned in lockstep.

    Args:
        session_path: Path to the session's dataset directory containing the data feather and the trial geometry
            data file.
        trial_type: Trial type to load (e.g., "ABC", "ABCD"). Must match an entry in the session's trial geometry
            data file.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to load. Selects
            between single-recording and multi-recording cindra outputs.

    Returns:
        A RunSessionData instance containing the aligned per-sample arrays, the trial type string, and the
        resolved TrialGeometryEntry.
    """
    geometry_entry = TrialGeometry.from_yaml(
        file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY),
    ).entries[trial_type]

    df = pl.read_ipc(
        source=session_path.joinpath(DatasetFiles.DATA),
        columns=[
            DatasetColumn.TIME_US.value,
            DatasetColumn.SYSTEM_STATE.value,
            DatasetColumn.TRIAL_TYPE.value,
            fluorescence_column.value,
            DatasetColumn.DISTANCE_CM.value,
            DatasetColumn.SPEED_CM_S.value,
            DatasetColumn.TRIAL.value,
            DatasetColumn.CUE.value,
        ],
    )
    df = trim_acquisition_warmup(dataframe=df)

    # Resolves the sampling rate from the post-warmup time column before any run-state filtering, so the rate
    # reflects the canonical acquisition cadence rather than the cadence of the (possibly gappy) run-only subset.
    sampling_rate_hz = _resolve_sampling_rate_hz(df=df)

    df = df.filter(
        (pl.col(DatasetColumn.SYSTEM_STATE.value) == "run") & (pl.col(DatasetColumn.TRIAL_TYPE.value) == trial_type),
    )

    # noinspection PyTypeChecker
    fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T
    # noinspection PyTypeChecker
    distance: NDArray[np.float32] = df[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32)
    # noinspection PyTypeChecker
    trial_ids: NDArray[np.int32] = df[DatasetColumn.TRIAL.value].to_numpy().astype(np.int32)
    # noinspection PyTypeChecker
    speed: NDArray[np.float32] = df[DatasetColumn.SPEED_CM_S.value].to_numpy().astype(np.float32)

    # When the runtime starts trials mid-first-cue (cue_offset_cm > 0), re-anchor trial boundaries to
    # cue-aligned positions so the rate-map x-axis at position 0 corresponds to the canonical first-cue start.
    # The first / last realigned trials are typically incomplete and get dropped by the completeness mask in
    # ``compute_within_trial_position``.
    if geometry_entry.cue_offset_cm != 0.0:
        # noinspection PyTypeChecker
        cue: NDArray[np.uint8] = df[DatasetColumn.CUE.value].to_numpy().astype(np.uint8)
        trial_ids = realign_trial_starts_to_first_cue(cue=cue)

    position = compute_within_trial_position(
        distance=distance,
        trial_ids=trial_ids,
        track_length=geometry_entry.trial_length_cm,
    )
    # noinspection PyTypeChecker
    valid: NDArray[np.bool_] = ~np.isnan(position)

    return RunSessionData(
        fluorescence=fluorescence[:, valid],
        position=position[valid],
        speed=speed[valid],
        trial_ids=trial_ids[valid],
        trial_type=trial_type,
        geometry=geometry_entry,
        sampling_rate_hz=sampling_rate_hz,
    )


def compute_within_trial_position(
    distance: NDArray[np.float32],
    trial_ids: NDArray[np.int32],
    track_length: float,
    completeness_threshold: float = 0.9,
) -> NDArray[np.float32]:
    """Rescales a cumulative distance array into per-trial chunks that start at 0 at each trial's first sample.

    Notes:
        Trials whose measured length falls below completeness_threshold * track_length are masked with NaN so
        downstream binning can drop them via ~np.isnan(position). Assumes samples are time-ordered so each
        trial's samples form one contiguous block.

    Args:
        distance: The cumulative distance traveled by the animal at each sample of the session.
        trial_ids: The trial identifier at each sample of the session.
        track_length: The total length of the virtual reality track for the processed type of trials, in
            centimeters.
        completeness_threshold: Minimum fraction of track_length that a trial's measured length must reach to be
            considered complete. Samples in below-threshold trials are returned as NaN.

    Returns:
        Per-sample within-trial position in centimeters, with NaN at samples belonging to incomplete trials.
    """
    # Brackets every contiguous trial block with (start, end) index pairs by detecting trial_ids transitions.
    # noinspection PyTypeChecker
    change_indices: NDArray[np.int64] = np.flatnonzero(np.diff(trial_ids)) + 1
    # noinspection PyTypeChecker
    starts: NDArray[np.int64] = np.concatenate(([0], change_indices))
    # noinspection PyTypeChecker
    ends: NDArray[np.int64] = np.concatenate((change_indices, [distance.size]))

    # Broadcasts each trial's starting distance back to one value per sample for a single vectorized subtraction.
    counts = ends - starts
    per_trial_start = distance[starts]
    per_trial_length = distance[ends - 1] - per_trial_start
    # noinspection PyTypeChecker
    per_sample_start: NDArray[np.float32] = np.repeat(a=per_trial_start, repeats=counts)
    # noinspection PyTypeChecker
    position: NDArray[np.float32] = (distance - per_sample_start).astype(np.float32)

    # Masks samples in below-threshold trials with NaN so downstream consumers can drop them in one step.
    minimum_length = np.float32(completeness_threshold * track_length)
    # noinspection PyTypeChecker
    per_sample_incomplete: NDArray[np.bool_] = np.repeat(a=per_trial_length < minimum_length, repeats=counts)
    position[per_sample_incomplete] = np.float32("nan")
    return position


def bin_fluorescence_by_position(
    fluorescence: NDArray[np.float32],
    position: NDArray[np.float32],
    position_bin_edges: NDArray[np.float32],
    *,
    compute_mean: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Bins neural fluorescence data by animal position along the linear track.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, sample_count).
        position: The animal's position at each fluorescence sample.
        position_bin_edges: Monotonically increasing position-bin boundaries in centimeters with length
            bin_count + 1.
        compute_mean: Determines whether to compute the per-bin mean. When False, the per-bin sum is returned.

    Returns:
        A tuple containing the binned fluorescence array with dimensions (cell_count, bin_count) and the sample
        count per bin with length bin_count.
    """
    # Assigns each position to a spatial bin and clips to the range [0, bin_count - 1].
    # noinspection PyTypeChecker
    raw_bin_indices: NDArray[np.int64] = np.searchsorted(a=position_bin_edges, v=position, side="right") - 1
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int32] = np.clip(
        a=raw_bin_indices,
        a_min=0,
        a_max=len(position_bin_edges) - 2,
    ).astype(np.int32)

    bin_count = len(position_bin_edges) - 1
    cell_count = fluorescence.shape[0]
    # noinspection PyTypeChecker
    sample_counts: NDArray[np.int32] = np.bincount(bin_indices, minlength=bin_count).astype(np.int32)

    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.full(shape=(cell_count, bin_count), fill_value=np.nan, dtype=np.float32)

    _accumulate_binned_fluorescence(
        fluorescence=fluorescence,
        bin_indices=bin_indices,
        sample_counts=sample_counts,
        use_mean=compute_mean,
        output=output,
    )

    return output, sample_counts


@njit(cache=True, nogil=True)
def compute_shuffle_source_indices(
    filtered_sample_indices: NDArray[np.int32],
    sample_count: int,
    minimum_shift: int,
    chunk_count: int,
    seed: int,
) -> NDArray[np.int32]:
    """Maps each speed-filtered destination sample back to the source sample it pulls from under the shuffle.

    Notes:
        Encodes the circular shift (when ``chunk_count == 1``) and circular-shift + chunk-permute (when
        ``chunk_count > 1``) as an indirection array rather than materializing a full shuffled fluorescence
        matrix. Hoisted from the reward-cell pipeline so both protocols share the same shuffle implementation.

    Args:
        filtered_sample_indices: Destination-sample indices retained by the speed filter with length
            filtered_sample_count.
        sample_count: Total number of samples in the original fluorescence time series.
        minimum_shift: Minimum number of samples for the circular shift.
        chunk_count: Number of chunks to split the shifted trace into for permutation. Set to 1 to disable
            chunk-permute and use a pure circular shift.
        seed: Random seed for reproducibility.

    Returns:
        Source-sample indices with length filtered_sample_count.
    """
    np.random.seed(seed)  # noqa: NPY002
    shift_amount = np.random.randint(minimum_shift, sample_count - minimum_shift)  # noqa: NPY002
    chunk_size = sample_count // chunk_count
    permutation = np.random.permutation(chunk_count)  # noqa: NPY002

    # Computes cumulative output-chunk start positions so each destination can be located within the permuted
    # layout.
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
    # noinspection PyTypeChecker
    source_indices: NDArray[np.int32] = np.empty(filtered_count, dtype=np.int32)

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
def accumulate_shuffled_rate_maps(
    fluorescence: NDArray[np.float32],
    source_indices: NDArray[np.int32],
    bin_indices: NDArray[np.int32],
    sample_counts: NDArray[np.int32],
    output: NDArray[np.float32],
) -> None:
    """Bins fluorescence into a per-cell rate map by gathering source samples through an indirection array.

    Notes:
        Shared by the place- and reward-cell pipelines. Empty bins are written as 0.0 (not NaN) so callers can
        apply smoothing without a NaN-aware kernel; smoothing in the calling code uses ``mode="wrap"`` and is
        robust to zero-occupancy bins.

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


@njit(cache=True)
def per_cell_pearson_safe(
    first_matrix: NDArray[np.float32],
    second_matrix: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Computes per-cell Pearson r between two (cell_count, bin_count) matrices, NaN-safe and zero-variance-safe.

    Notes:
        Returns NaN for cells with fewer than three pairwise-valid bins or zero variance in either half.
        Compiled with numba so it runs without GIL contention inside shuffle loops. Hoisted from the place-cell
        pipeline so the place- and reward-cell detectors compute split-half stability against the same kernel.

    Args:
        first_matrix: First matrix with dimensions (cell_count, bin_count).
        second_matrix: Second matrix with dimensions (cell_count, bin_count).

    Returns:
        Per-cell Pearson r with length cell_count.
    """
    cell_count = first_matrix.shape[0]
    bin_count = first_matrix.shape[1]
    output = np.full(cell_count, np.nan, dtype=np.float32)
    for cell_index in range(cell_count):
        valid_count = 0
        first_sum = 0.0
        second_sum = 0.0
        for bin_index in range(bin_count):
            first_value = first_matrix[cell_index, bin_index]
            second_value = second_matrix[cell_index, bin_index]
            if not np.isnan(first_value) and not np.isnan(second_value):
                valid_count += 1
                first_sum += first_value
                second_sum += second_value
        if valid_count < MINIMUM_VALID_BINS_FOR_PEARSON:
            continue
        first_mean = first_sum / valid_count
        second_mean = second_sum / valid_count

        first_variance = 0.0
        second_variance = 0.0
        covariance = 0.0
        for bin_index in range(bin_count):
            first_value = first_matrix[cell_index, bin_index]
            second_value = second_matrix[cell_index, bin_index]
            if not np.isnan(first_value) and not np.isnan(second_value):
                first_diff = first_value - first_mean
                second_diff = second_value - second_mean
                first_variance += first_diff * first_diff
                second_variance += second_diff * second_diff
                covariance += first_diff * second_diff
        if first_variance <= 0.0 or second_variance <= 0.0:
            continue
        output[cell_index] = np.float32(covariance / np.sqrt(first_variance * second_variance))
    return output


def bin_fluorescence_per_trial(
    fluorescence: NDArray[np.float32],
    position: NDArray[np.float32],
    speed: NDArray[np.float32],
    trial_ids: NDArray[np.int32],
    bin_edges: NDArray[np.float32],
    *,
    minimum_speed: float,
    smooth_size: int,
) -> NDArray[np.float32]:
    """Bins per-sample fluorescence per lap into a (cell_count, trial_count, bin_count) array.

    Notes:
        Excludes the ``_NO_TRIAL_SENTINEL`` trial id that the acquisition pipeline uses to mark "no trial"
        samples. Applies the same speed filter and uniform_filter1d smoothing the place- and reward-cell
        detectors apply to their pooled rate maps so averaging the returned array across the trial axis
        reproduces the pooled rate map within numerical rounding. Bins and lap slices with no valid
        speed-filtered samples are filled with NaN so consumers can treat them as missing without downstream
        guards.

        The hot loop is hoisted into ``_bin_fluorescence_per_trial_kernel`` (``@njit(parallel=True)``) which
        fans out across ``(cell, trial)`` pairs. The previous version walked trials sequentially in Python,
        calling a ``bin_fluorescence_by_position`` + ``uniform_filter1d`` pair per trial; the kernel replaces
        both with a single fused pass that accumulates per-bin sums, computes per-bin means, and applies
        wrap-around uniform smoothing in place.

    Args:
        fluorescence: Pre-normalized dF/F0 fluorescence with dimensions (cell_count, sample_count).
        position: The animal's per-sample within-trial position in centimeters with length sample_count.
        speed: The animal's per-sample speed in cm/s with length sample_count.
        trial_ids: The per-sample trial identifier with length sample_count.
        bin_edges: Monotonically increasing position-bin boundaries in centimeters with length bin_count + 1.
        minimum_speed: Minimum speed threshold in cm/s for including samples.
        smooth_size: Width of the uniform smoothing kernel in bins applied across the position axis.

    Returns:
        Per-lap binned fluorescence with dimensions (cell_count, trial_count, bin_count).
    """
    # noinspection PyTypeChecker
    valid_trial_mask: NDArray[np.bool_] = trial_ids != _NO_TRIAL_SENTINEL
    # noinspection PyTypeChecker
    unique_trials: NDArray[np.int32] = np.unique(trial_ids[valid_trial_mask])
    trial_count = len(unique_trials)

    cell_count = fluorescence.shape[0]
    bin_count = len(bin_edges) - 1

    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.full((cell_count, trial_count, bin_count), np.nan, dtype=np.float32)

    if trial_count == 0:
        return output

    # Builds a CSR-style trial->sample lookup so the kernel only walks each trial's samples instead of scanning
    # the full sample axis for every (cell, trial). Computes the trial slot per sample once (samples below the
    # speed cut get slot -1, then drop out), sorts samples by slot to group them, and records per-trial
    # start/end offsets into the sorted index array.
    # noinspection PyTypeChecker
    sample_trial_slot: NDArray[np.int32] = np.full(trial_ids.shape[0], -1, dtype=np.int32)
    # noinspection PyTypeChecker
    speed_mask: NDArray[np.bool_] = speed > minimum_speed
    for slot, trial_id in enumerate(unique_trials):
        # noinspection PyTypeChecker
        slot_mask: NDArray[np.bool_] = (trial_ids == trial_id) & speed_mask
        sample_trial_slot[slot_mask] = slot

    # noinspection PyTypeChecker
    valid_sample_mask: NDArray[np.bool_] = sample_trial_slot >= 0
    # noinspection PyTypeChecker
    valid_sample_indices: NDArray[np.int32] = np.flatnonzero(valid_sample_mask).astype(np.int32)
    # noinspection PyTypeChecker
    valid_slots: NDArray[np.int32] = sample_trial_slot[valid_sample_indices]
    # noinspection PyTypeChecker
    sort_order: NDArray[np.int64] = np.argsort(valid_slots, kind="stable")
    # noinspection PyTypeChecker
    trial_sample_indices: NDArray[np.int32] = valid_sample_indices[sort_order]
    # noinspection PyTypeChecker
    sorted_slots: NDArray[np.int32] = valid_slots[sort_order]
    # noinspection PyTypeChecker
    trial_sample_offsets: NDArray[np.int32] = np.zeros(trial_count + 1, dtype=np.int32)
    # noinspection PyTypeChecker
    counts_per_slot: NDArray[np.int32] = np.bincount(sorted_slots, minlength=trial_count).astype(np.int32)
    trial_sample_offsets[1:] = np.cumsum(counts_per_slot)

    # Per-sample bin indices stay in original sample-index space; the kernel reads them via
    # ``trial_sample_indices`` to fetch only the samples for its trial.
    # noinspection PyTypeChecker
    raw_bin_indices: NDArray[np.int64] = np.searchsorted(bin_edges, position, side="right") - 1
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int32] = np.clip(raw_bin_indices, 0, bin_count - 1).astype(np.int32)

    _bin_fluorescence_per_trial_kernel(
        fluorescence=fluorescence,
        trial_sample_indices=trial_sample_indices,
        trial_sample_offsets=trial_sample_offsets,
        bin_indices=bin_indices,
        trial_count=trial_count,
        smooth_size=int(smooth_size),
        output=output,
    )

    return output


def _resolve_sampling_rate_hz(df: pl.DataFrame) -> float:
    """Computes the acquisition sampling rate in Hz from the ``time_us`` column of a session dataframe.

    Notes:
        Uses the median per-sample inter-time interval to be robust to gaps that arise from system-state
        transitions within the session. Returns NaN when the dataframe has fewer than two samples.

    Args:
        df: Session dataframe loaded from ``DatasetFiles.DATA`` (post-warmup, pre-filter). Must include the
            ``DatasetColumn.TIME_US`` column.

    Returns:
        The acquisition sampling rate in Hz, or NaN when the dataframe has fewer than two samples.
    """
    minimum_samples_for_interval = 2
    if df.height < minimum_samples_for_interval:
        return float("nan")
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy()
    median_interval_us = float(np.median(np.diff(time_us)))
    if median_interval_us <= 0.0:
        return float("nan")
    return float(
        interval_to_rate(
            interval=median_interval_us,
            from_units=TimeUnits.MICROSECOND,
            as_float=True,
        )
    )


@njit(cache=True, parallel=True)
def _bin_fluorescence_per_trial_kernel(
    fluorescence: NDArray[np.float32],
    trial_sample_indices: NDArray[np.int32],
    trial_sample_offsets: NDArray[np.int32],
    bin_indices: NDArray[np.int32],
    trial_count: int,
    smooth_size: int,
    output: NDArray[np.float32],
) -> None:
    """Accumulates, averages, and wrap-smooths per-trial binned fluorescence into ``output`` in place.

    Notes:
        Parallelism fans out across ``prange(cell_count * trial_count)`` so every (cell, trial) is an
        independent task. Each task indexes a CSR-style trial->sample lookup (``trial_sample_indices`` /
        ``trial_sample_offsets``) so it only walks its own trial's samples; per-bin sum and count scratch sized
        to ``bin_count`` stays small enough for numba's allocator to pool across tasks.

    Args:
        fluorescence: Pre-normalized dF/F0 fluorescence with dimensions (cell_count, sample_count).
        trial_sample_indices: Sample indices grouped by trial slot in ``[0, trial_count)`` with length equal to
            the number of speed-filtered, in-trial samples.
        trial_sample_offsets: Per-trial start offsets into ``trial_sample_indices`` with length trial_count + 1.
        bin_indices: Per-sample bin index in ``[0, bin_count)`` indexed in original sample-axis space; the
            kernel reads ``bin_indices[trial_sample_indices[k]]`` to resolve the bin for the k-th sample of a
            trial.
        trial_count: Number of unique retained trials (size of the trial axis of ``output``).
        smooth_size: Width of the wrap-around uniform smoothing kernel applied along the bin axis. Must be odd.
        output: Pre-allocated (cell_count, trial_count, bin_count) buffer pre-filled with NaN.
    """
    cell_count = fluorescence.shape[0]
    bin_count = output.shape[2]
    half_smooth = smooth_size // 2
    total_tasks = cell_count * trial_count

    for task in prange(total_tasks):
        cell_index = task // trial_count
        trial_slot = task % trial_count
        sample_start = trial_sample_offsets[trial_slot]
        sample_end = trial_sample_offsets[trial_slot + 1]
        if sample_start == sample_end:
            continue

        # noinspection PyTypeChecker
        bin_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
        # noinspection PyTypeChecker
        bin_counts: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)

        for offset in range(sample_start, sample_end):
            sample_index = trial_sample_indices[offset]
            bin_index = bin_indices[sample_index]
            bin_sums[bin_index] += fluorescence[cell_index, sample_index]
            bin_counts[bin_index] += 1

        # noinspection PyTypeChecker
        raw_means: NDArray[np.float32] = np.empty(bin_count, dtype=np.float32)
        any_valid = False
        for bin_index in range(bin_count):
            if bin_counts[bin_index] > 0:
                raw_means[bin_index] = bin_sums[bin_index] / np.float32(bin_counts[bin_index])
                any_valid = True
            else:
                raw_means[bin_index] = np.float32(np.nan)
        if not any_valid:
            continue

        # Wrap-around uniform smoothing of size ``smooth_size`` (matches scipy's
        # ``uniform_filter1d(mode="wrap")`` for odd kernels).
        for bin_index in range(bin_count):
            total = np.float32(0.0)
            for window_offset in range(-half_smooth, half_smooth + 1):
                neighbor = bin_index + window_offset
                if neighbor < 0:
                    neighbor += bin_count
                elif neighbor >= bin_count:
                    neighbor -= bin_count
                total += raw_means[neighbor]
            output[cell_index, trial_slot, bin_index] = total / np.float32(smooth_size)


@njit(cache=True, parallel=True)
def _accumulate_binned_fluorescence(
    fluorescence: NDArray[np.float32],
    bin_indices: NDArray[np.int32],
    sample_counts: NDArray[np.int32],
    use_mean: bool,  # noqa: FBT001
    output: NDArray[np.float32],
) -> None:
    """Accumulates cell fluorescence values into spatial position bins for each cell, writing into output in place.

    Notes:
        Only non-empty bins are written. Empty bins are left at the caller's pre-filled value (NaN, set by
        ``bin_fluorescence_by_position``), avoiding a conditional store per empty bin in the hot loop.

    Args:
        fluorescence: Fluorescence data with dimensions (cell_count, sample_count).
        bin_indices: Bin index for each fluorescence data sample.
        sample_counts: Number of samples for each bin.
        use_mean: Determines whether to compute the per-bin mean by dividing the accumulated sum by the sample
            count. When False, the raw per-bin sum is written.
        output: Pre-allocated output array with dimensions (cell_count, bin_count). Empty bins must be pre-filled
            with NaN. Modified in place.
    """
    cell_count = fluorescence.shape[0]
    sample_count = fluorescence.shape[1]
    bin_count = output.shape[1]

    for cell_index in prange(cell_count):
        bin_sums = np.zeros(bin_count, dtype=np.float32)

        for sample_index in range(sample_count):
            bin_index = bin_indices[sample_index]
            bin_sums[bin_index] += fluorescence[cell_index, sample_index]

        for bin_index in range(bin_count):
            count = sample_counts[bin_index]
            if count > 0:
                if use_mean:
                    output[cell_index, bin_index] = bin_sums[bin_index] / count
                else:
                    output[cell_index, bin_index] = bin_sums[bin_index]


def random_remapping_peak_shift_p_values(
    peaks_a: NDArray[np.float32],
    peaks_b: NDArray[np.float32],
    *,
    shuffle_count: int = 1000,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Per-cell p-values from a random-remapping cell-ID shuffle of paired peak positions.

    Notes:
        Reusable building block for cross-frame peak-shift analyses (cross-trial-type reward-relative,
        cross-session reward-shift, or any other paired-peak comparison). For each shuffle iteration the
        ``peaks_b`` vector is permuted across cells; the per-cell p-value is the fraction of shuffles whose
        shuffled ``|peaks_a - peaks_b|`` is at or below the observed ``|peaks_a - peaks_b|``. Cells whose
        observed paired shift is unusually small relative to random pairings receive small p-values. NaN
        entries in either input propagate to NaN p-values.

        Both inputs must already be in the desired coordinate system (e.g. track-aligned cm or signed-circular
        reward-aligned cm). Compute peak positions with `numpy.argmax` on per-cell rate maps and convert
        to centimeters at the bin center; for reward-aligned coordinates apply a signed circular wrap to the
        per-cell reward midpoint before passing in.

    Args:
        peaks_a: First-frame per-cell peak positions, length ``cell_count``. Same units as ``peaks_b``.
        peaks_b: Second-frame per-cell peak positions, length ``cell_count``.
        shuffle_count: Number of cell-ID permutations.
        seed: Base RNG seed; iteration ``i`` uses ``seed + i``. Fixed-seed iteration order makes the result
            reproducible across runs.

    Returns:
        Per-cell p-values with length ``cell_count``. Cells whose observed shift is NaN (either input is NaN)
        receive NaN p-values; all other cells receive a value in ``[0, 1]``.
    """
    cell_count = peaks_a.shape[0]
    # noinspection PyTypeChecker
    p_values: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if cell_count == 0 or shuffle_count <= 0:
        return p_values

    # noinspection PyTypeChecker
    valid_mask: NDArray[np.bool_] = ~np.isnan(peaks_a) & ~np.isnan(peaks_b)
    if not bool(np.any(valid_mask)):
        return p_values

    valid_a = peaks_a[valid_mask]
    valid_b = peaks_b[valid_mask]
    valid_count = int(valid_a.shape[0])
    observed = np.abs(valid_a - valid_b)
    # noinspection PyTypeChecker
    le_count: NDArray[np.int64] = np.zeros(valid_count, dtype=np.int64)

    for iteration in range(shuffle_count):
        generator = np.random.default_rng(seed=seed + iteration)
        permutation = generator.permutation(valid_count)
        # noinspection PyTypeChecker
        shuffled: NDArray[np.float32] = np.abs(valid_a - valid_b[permutation])
        le_count += (shuffled <= observed).astype(np.int64)

    # noinspection PyTypeChecker
    valid_p: NDArray[np.float32] = (le_count.astype(np.float32) / float(shuffle_count)).astype(np.float32)
    p_values[valid_mask] = valid_p
    return p_values
