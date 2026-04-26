"""Provides utility functions for reading trial geometry data and reconstructing canonical position from forged
session feather files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..forging import TRIAL_GEOMETRY_FILENAME, TrialGeometry

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def compute_track_length(session_path: Path, trial_type: str) -> float:
    """Returns the canonical track length for the given trial type read from the session's trial_geometry.yaml file.

    Args:
        session_path: Path to the session's data.feather file.
        trial_type: Trial type name to look up in the trial geometry sidecar.

    Returns:
        Canonical track length in centimeters.
    """
    geometry = TrialGeometry.from_yaml(file_path=session_path.parent.joinpath(TRIAL_GEOMETRY_FILENAME))
    return geometry.entries[trial_type].trial_length_cm


def compute_stimulus_zone_center(session_path: Path, trial_type: str) -> float:
    """Returns the canonical center of the stimulus trigger zone for the given trial type.

    Notes:
        For LICK-triggered (REWARD) trials this is the center of the lick-active reward zone, used as the
        representative position for reward-cell analysis since water is delivered wherever in the zone the
        animal happens to lick (no single delivery point exists). For OCCUPANCY-triggered (AVERSIVE) trials
        this is the center of the occupancy zone, distinct from the collider boundary at stimulus_location_cm
        where automated puff delivery happens. Callers gate on the relevant stimulus_mode when interpreting
        the returned value.

    Args:
        session_path: Path to the session's data.feather file.
        trial_type: Trial type name to look up in the trial geometry sidecar.

    Returns:
        Center of the stimulus trigger zone in centimeters, in trial-relative canonical coordinates.
    """
    geometry = TrialGeometry.from_yaml(file_path=session_path.parent.joinpath(TRIAL_GEOMETRY_FILENAME))
    entry = geometry.entries[trial_type]
    return (entry.stimulus_trigger_zone_start_cm + entry.stimulus_trigger_zone_end_cm) / 2.0


def compute_canonical_position(
    distance: NDArray[np.float32],
    trial_ids: NDArray[np.int32],
    canonical_track_length: float,
    completeness_threshold: float = 0.9,
) -> NDArray[np.float32]:
    """Computes per-frame canonical position by subtracting each trial's starting cumulative distance, returning
    NaN for frames belonging to incomplete trials.

    Notes:
        Each frame's position is the encoder-honest offset from where its trial began (distance - distance at the
        trial's first frame). The animal's measured per-trial length only deviates from the VR's canonical length
        by sub-sample artifacts from photometry's 10 Hz downsampling, which falls well within the default
        completeness_threshold of 0.9. Trials whose measured length falls below
        completeness_threshold * canonical_track_length emit NaN for every frame so downstream binning can drop
        them via a single ~np.isnan(position) mask. This catches the partial first or last trial of a session
        and any trial where the animal got stuck. Assumes frames are time-ordered so each trial's frames form one
        contiguous block, which holds after the forging pipeline's run-state filtering.

    Args:
        distance: Cumulative distance in centimeters, with one value per frame.
        trial_ids: Trial identity for each frame, with one value per frame.
        canonical_track_length: The canonical track length in centimeters from the trial geometry sidecar, used
            to identify incomplete trials.
        completeness_threshold: Minimum fraction of canonical_track_length that a trial's measured length must
            reach to be considered complete. Frames in below-threshold trials are returned as NaN.

    Returns:
        Per-frame canonical position in centimeters as the trial-relative encoder offset, with NaN at every frame
        belonging to a trial whose measured length is below completeness_threshold * canonical_track_length.
    """
    if distance.size == 0:
        # noinspection PyTypeChecker
        return np.empty(0, dtype=np.float32)

    # Detects each trial's first frame as a transition in trial_ids; brackets every contiguous trial block with
    # (start, end) index pairs that span the whole input.
    change_indices = np.flatnonzero(np.diff(trial_ids)) + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices, [distance.size]))

    # Computes each trial's first cumulative distance and measured length once, then broadcasts the per-trial
    # start back out to one value per frame so the offset subtraction is a single vectorized arithmetic.
    counts = ends - starts
    per_trial_start = distance[starts]
    per_trial_length = distance[ends - 1] - per_trial_start
    per_frame_start = np.repeat(per_trial_start, counts)

    # Computes raw trial-relative position; the animal's encoder reading minus where the trial began.
    position = (distance - per_frame_start).astype(np.float32)

    # Marks frames in below-threshold trials with NaN so downstream binning can drop them with a single
    # ~np.isnan(position) filter. This catches the partial first or last trial of a session and any trial where
    # the animal got stuck or did not complete a lap.
    minimum_length = np.float32(completeness_threshold * canonical_track_length)
    per_frame_incomplete = np.repeat(per_trial_length < minimum_length, counts)
    position[per_frame_incomplete] = np.float32("nan")
    # noinspection PyTypeChecker
    return position
