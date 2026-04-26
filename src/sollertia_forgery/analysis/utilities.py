"""Provides shared utility assets for other analysis modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    change_indices = np.flatnonzero(np.diff(trial_ids)) + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices, [distance.size]))

    # Broadcasts each trial's starting distance back to one value per sample for a single vectorized subtraction.
    counts = ends - starts
    per_trial_start = distance[starts]
    per_trial_length = distance[ends - 1] - per_trial_start
    per_sample_start = np.repeat(per_trial_start, counts)
    position = (distance - per_sample_start).astype(np.float32)

    # Masks samples in below-threshold trials with NaN so downstream consumers can drop them in one step.
    minimum_length = np.float32(completeness_threshold * track_length)
    per_sample_incomplete = np.repeat(per_trial_length < minimum_length, counts)
    position[per_sample_incomplete] = np.float32("nan")
    # noinspection PyTypeChecker
    return position
