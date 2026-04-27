"""Provides shared utility assets for other analysis modules."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from numba import njit, prange
import numpy as np
import polars as pl

from ..forging import FluorescenceColumn
from ..shared_assets import (
    DatasetFiles,
    DatasetColumn,
    TrialGeometry,
    TrialGeometryEntry,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


ACQUISITION_WARMUP_SECONDS: float = 60.0
"""Number of leading seconds discarded from every loaded session trace before any analysis runs. Sollertia
experiments include a multi-minute pre-imaging baseline period during which the PMT gain, resonant scanner phase,
shutter, and laser power have not yet stabilized; the resulting initial fluorescence valley would otherwise
contaminate downstream estimates (per-cell baselines, within-session bleaching, SCE statistics, place-field
tuning). Trimming at load time guarantees every analyzer operates on stabilized samples without needing to know
the artifact exists."""


def trim_acquisition_warmup(df: pl.DataFrame) -> pl.DataFrame:
    """Drops the leading ``ACQUISITION_WARMUP_SECONDS`` of samples from a session dataframe based on the
    ``time_us`` column.

    Notes:
        Operates on the polars dataframe directly (rather than the post-explode numpy arrays) so the warmup window
        never enters any subsequent column-level reshape. Sessions whose entire trace falls within the warmup
        window collapse to an empty dataframe; downstream loaders' existing length guards then produce NaN sentinels
        for such degenerate sessions.

    Args:
        df: Session dataframe loaded from ``DatasetFiles.DATA``. Must include ``DatasetColumn.TIME_US`` among the
            selected columns; all other columns are passed through untouched.

    Returns:
        The input dataframe sliced to drop every row whose ``time_us`` value precedes the warmup cutoff.
    """
    if df.height == 0:
        return df
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy()
    warmup_us = int(ACQUISITION_WARMUP_SECONDS * 1_000_000)
    cutoff_us = int(time_us[0]) + warmup_us
    warmup_index = int(np.searchsorted(time_us, cutoff_us, side="left"))
    if warmup_index <= 0:
        return df
    return df.slice(warmup_index)


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


def assemble_run_session_data(
    session_path: Path,
    trial_type: str,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
) -> RunSessionData:
    """Assembles run-state arrays and trial geometry from a forged session for the given trial type.

    Notes:
        Resolves the canonical track length from the session's trial geometry data file, drops the leading
        acquisition-warmup window so downstream binning operates on stabilized samples, filters the session's data
        feather to system_state == 'run' and trial_type == trial_type, computes within-trial position, and drops
        samples belonging to incomplete trials so downstream binning never sees NaN positions. All returned arrays
        share the same sample axis and are aligned in lockstep.

    Args:
        session_path: Path to the session's dataset directory containing the data feather and the trial geometry
            data file.
        trial_type: Trial type to load (e.g. "ABC", "ABCD"). Must match an entry in the session's trial geometry
            data file.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to load. Selects between
            single-recording and multi-recording cindra outputs.

    Returns:
        A RunSessionData instance containing the aligned per-sample arrays, the trial type string, and the resolved
        TrialGeometryEntry.
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
        ],
    )
    df = trim_acquisition_warmup(df)
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
        downstream binning can drop them via ~np.isnan(position). Assumes samples are time-ordered so each trial's
        samples form one contiguous block.

    Args:
        distance: The cumulative distance traveled by the animal at each sample of the session.
        trial_ids: The trial identifier at each sample of the session.
        track_length: The total length of the virtual reality track for the processed type of trials, in centimeters.
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
    per_sample_start: NDArray[np.float32] = np.repeat(per_trial_start, counts)
    # noinspection PyTypeChecker
    position: NDArray[np.float32] = (distance - per_sample_start).astype(np.float32)

    # Masks samples in below-threshold trials with NaN so downstream consumers can drop them in one step.
    minimum_length = np.float32(completeness_threshold * track_length)
    # noinspection PyTypeChecker
    per_sample_incomplete: NDArray[np.bool_] = np.repeat(per_trial_length < minimum_length, counts)
    position[per_sample_incomplete] = np.float32("nan")
    # noinspection PyTypeChecker
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
        A tuple containing the binned fluorescence array with dimensions (cell_count, bin_count) and the sample count
        per bin with length bin_count.
    """
    # Assigns each position to a spatial bin and clips to the range [0, bin_count - 1].
    # noinspection PyTypeChecker
    raw_bin_indices: NDArray[np.int64] = np.searchsorted(position_bin_edges, position, side="right") - 1
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int32] = np.clip(raw_bin_indices, 0, len(position_bin_edges) - 2).astype(np.int32)

    bin_count = len(position_bin_edges) - 1
    cell_count = fluorescence.shape[0]
    # noinspection PyTypeChecker
    sample_counts: NDArray[np.int32] = np.bincount(bin_indices, minlength=bin_count).astype(np.int32)

    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.full((cell_count, bin_count), np.nan, dtype=np.float32)

    # Hands off to vectorized and compiled accumulator.
    _accumulate_binned_fluorescence(
        fluorescence=fluorescence,
        bin_indices=bin_indices,
        sample_counts=sample_counts,
        use_mean=compute_mean,
        output=output,
    )

    return output, sample_counts


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

    # Parallelizes over cells. Each thread processes one cell at a time, so per-cell scratch buffers stay private and
    # the writes to output never collide.
    for cell_index in prange(cell_count):
        # Allocates a thread-private scratch buffer sized to bin_count. At typical bin counts (under ~100 bins) this
        # easily fits in L1, which keeps the random-access scatter in the next loop fast.
        bin_sums = np.zeros(bin_count, dtype=np.float32)

        # Scatter-adds each sample's fluorescence into the bin it falls under. Reads fluorescence row-sequentially
        # (cache-friendly) and bin_indices once per sample (small, stays hot in cache across cells).
        for sample_index in range(sample_count):
            bin_index = bin_indices[sample_index]
            bin_sums[bin_index] += fluorescence[cell_index, sample_index]

        # Writes the per-bin result for this cell. Empty bins are skipped intentionally — the caller pre-fills
        # output with NaN, so leaving those entries untouched gives the correct empty-bin sentinel for free.
        for bin_index in range(bin_count):
            count = sample_counts[bin_index]
            if count > 0:
                if use_mean:
                    output[cell_index, bin_index] = bin_sums[bin_index] / count
                else:
                    output[cell_index, bin_index] = bin_sums[bin_index]
