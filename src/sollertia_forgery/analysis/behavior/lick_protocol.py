"""Per-session and per-animal lick-event extraction utilities.

Module-level functions consume a forged session's persisted ``data.feather`` plus its trial geometry data
file and return a per-animal `LickContext` ready for `.plotting.plot_lick_scatter`. The pipeline is
stateless: no feather or YAML artifact is persisted, because every input lives in already-forged dataset
files and the per-session compute (a polars filter plus a rising-edge ``np.diff``) is cheap enough to
repeat on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
from itertools import groupby
from dataclasses import dataclass

import numpy as np
import polars as pl

from ...shared_assets import (
    DatasetFiles,
    DatasetColumn,
    TrialGeometry,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


_NO_TRIAL_SENTINEL: int = 255
"""Sentinel trial id the acquisition pipeline writes for samples outside any trial. Used to drop these
samples before any per-trial reduction."""

_DEFAULT_TRACK_LENGTH_CM: float = 240.0
"""Fallback track length used when a session's trial type is missing from its geometry layout. Matches the
acquisition default so the lick scatter still renders for sessions whose geometry data file is incomplete."""


@dataclass(frozen=True, slots=True)
class TrialBlock:
    """One block of contiguous trials sharing the same trial type and reward zone.

    Notes:
        Aggregating licks across an animal's sessions cannot assume a single reward zone: the same
        ``trial_type`` can have different reward zones across longitudinally separated sessions
        (mid-recording reward-zone shifts), and a single session can contain multiple trial types
        (intra-session multi-trial-type recordings). Each block therefore carries its own zone bounds
        so the plotting layer renders the *active* zone per row regardless of which dataset is being
        processed. Cumulative trial indices are animal-level (zero at the first trial of the first
        chronologically-ordered session) when the block lives on a `LickContext`.
    """

    cum_trial_start: int
    """Cumulative trial index (across the animal's full session sequence) at which this block starts."""
    cum_trial_end: int
    """Cumulative trial index (exclusive) at which this block ends."""
    session_index: int
    """Position of the owning session within the animal's chronologically ordered session list."""
    trial_type: str
    """Trial type label shared by every trial in the block."""
    trial_length_cm: float
    """Canonical track length for the block's trial type, in centimeters."""
    reward_lo: float
    """Trial-relative start of the stimulus trigger zone for the block, in centimeters. NaN when the
    session's geometry layout does not enumerate this trial type."""
    reward_hi: float
    """Trial-relative end of the stimulus trigger zone for the block, in centimeters. NaN when the
    session's geometry layout does not enumerate this trial type."""


@dataclass(frozen=True, slots=True)
class SessionLickData:
    """Run-state lick events plus per-trial reward-zone geometry resolved from a single forged session.

    Notes:
        The per-trial parallel arrays (``trial_types``, ``trial_lengths_cm``, ``reward_lo_cm``,
        ``reward_hi_cm``) are session-local: trial 0 is the first run-state trial of the session.
        ``aggregate_lick_events`` shifts cumulative indices by the running session-prefix sum to
        produce animal-level `TrialBlock` instances; consumers that only need single-session lick
        events (e.g., a per-session quality plot) can read this dataclass directly without going
        through the aggregator.
    """

    trial_types: tuple[str, ...]
    """Per-trial trial-type labels with length ``n_trials``."""
    trial_lengths_cm: NDArray[np.float32]
    """Per-trial canonical track lengths in centimeters with length ``n_trials``."""
    reward_lo_cm: NDArray[np.float32]
    """Per-trial stimulus trigger zone start in centimeters with length ``n_trials``. NaN for trials
    whose trial type is missing from the session's geometry layout."""
    reward_hi_cm: NDArray[np.float32]
    """Per-trial stimulus trigger zone end in centimeters with length ``n_trials``. NaN for trials
    whose trial type is missing from the session's geometry layout."""
    lick_positions: NDArray[np.float32]
    """Within-trial position of every rising-edge lick event in centimeters."""
    lick_trial_indices: NDArray[np.int64]
    """Session-local trial index for each lick event, parallel to ``lick_positions``."""

    @property
    def n_trials(self) -> int:
        """Returns the number of unique completed trials in the session's run state."""
        return len(self.trial_types)

    @property
    def track_length_cm(self) -> float:
        """Returns the maximum trial track length across every trial type used by this session."""
        if self.n_trials == 0:
            return _DEFAULT_TRACK_LENGTH_CM
        return float(self.trial_lengths_cm.max())


@dataclass(frozen=True, slots=True)
class LickContext:
    """Per-animal lick aggregate spanning every session and trial type the animal saw.

    Notes:
        Cumulative trial indices in ``lick_trials`` count across every session in chronological order
        starting at zero. ``session_boundaries`` records the cumulative trial count at each session
        transition (length ``len(sessions) + 1`` with a leading zero) so the plotting layer can draw
        session separators without re-deriving them.
    """

    lick_positions: NDArray[np.float32]
    """Within-trial position of every rising-edge lick event in centimeters, length ``lick_count``."""
    lick_trials: NDArray[np.int64]
    """Cumulative across-animal trial index for each lick event, parallel to ``lick_positions``."""
    trial_blocks: tuple[TrialBlock, ...]
    """Contiguous-zone trial blocks ordered chronologically across every session."""
    session_boundaries: tuple[int, ...]
    """Cumulative trial counts at each session boundary; ``session_boundaries[i]`` is the cumulative
    trial index at which session ``i`` starts. Length ``len(sessions) + 1``."""
    track_length_cm: float
    """Maximum trial track length in centimeters across every block. Drives the plotting x-axis and
    the per-block "absent track" mask for trials whose own ``trial_length_cm`` is shorter than the
    animal's max."""

    @property
    def total_trials(self) -> int:
        """Returns the total number of trials across every session in the aggregate."""
        return int(self.session_boundaries[-1]) if self.session_boundaries else 0

    @property
    def lick_count(self) -> int:
        """Returns the total number of rising-edge lick events captured across every session."""
        return int(self.lick_positions.size)


def extract_session_lick_events(session_path: Path) -> SessionLickData:
    """Extracts rising-edge lick events and per-trial reward zones from a single forged session.

    Notes:
        Filters the session's ``data.feather`` to ``system_state == "run"`` plus a sentinel-trial cut,
        computes within-trial position by subtracting each trial's first-sample distance from the
        cumulative ``distance_cm`` track, wraps positions by their per-trial ``trial_length_cm`` to
        accommodate cyclic tracks, and resolves each trial's reward zone from the session's trial
        geometry data file. Lick events are the rising edges of the binary ``lick`` sample column.
        Trial types absent from the session's geometry layout get NaN reward bounds and a
        ``_DEFAULT_TRACK_LENGTH_CM`` fallback so the lick scatter still renders.

    Args:
        session_path: Path to the forged session directory containing both ``data.feather`` and the
            trial geometry data file.

    Returns:
        A SessionLickData carrying the per-trial parallel arrays and the rising-edge lick events with
        their session-local trial indices.
    """
    geometry = TrialGeometry.from_yaml(file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY))
    df = pl.read_ipc(
        source=session_path.joinpath(DatasetFiles.DATA),
        columns=[
            DatasetColumn.SYSTEM_STATE.value,
            DatasetColumn.LICK.value,
            DatasetColumn.DISTANCE_CM.value,
            DatasetColumn.TRIAL.value,
            DatasetColumn.TRIAL_TYPE.value,
        ],
        memory_map=True,
    )
    run = df.filter(
        (pl.col(DatasetColumn.SYSTEM_STATE.value) == "run")
        & (pl.col(DatasetColumn.TRIAL.value) < _NO_TRIAL_SENTINEL),
    )
    if run.height == 0:
        # noinspection PyTypeChecker
        empty_positions: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_indices: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        # noinspection PyTypeChecker
        empty_lengths: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        return SessionLickData(
            trial_types=(),
            trial_lengths_cm=empty_lengths,
            reward_lo_cm=empty_lengths,
            reward_hi_cm=empty_lengths,
            lick_positions=empty_positions,
            lick_trial_indices=empty_indices,
        )

    # noinspection PyTypeChecker
    distance: NDArray[np.float64] = run[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float64)
    # noinspection PyTypeChecker
    trials: NDArray[np.int64] = run[DatasetColumn.TRIAL.value].to_numpy().astype(np.int64)
    # noinspection PyTypeChecker
    licks: NDArray[np.int8] = run[DatasetColumn.LICK.value].to_numpy().astype(np.int8)
    trial_types = run[DatasetColumn.TRIAL_TYPE.value].to_list()

    # Walks samples once and groups them into within-session trial slots, records the trial type seen
    # for each slot, and computes the per-sample within-trial position by subtracting each trial's
    # starting distance. The single-pass approach avoids an extra ``np.unique`` + ``searchsorted``
    # pair that would re-walk the trial column.
    sample_count = distance.shape[0]
    # noinspection PyTypeChecker
    position: NDArray[np.float64] = np.zeros(sample_count, dtype=np.float64)
    # noinspection PyTypeChecker
    trial_index_in_session: NDArray[np.int64] = np.zeros(sample_count, dtype=np.int64)
    seen_trial_types: list[str] = [trial_types[0]]
    current_trial = int(trials[0])
    start_distance = float(distance[0])
    for sample_index in range(sample_count):
        if int(trials[sample_index]) != current_trial:
            current_trial = int(trials[sample_index])
            start_distance = float(distance[sample_index])
            seen_trial_types.append(trial_types[sample_index])
        trial_index_in_session[sample_index] = len(seen_trial_types) - 1
        position[sample_index] = float(distance[sample_index]) - start_distance
    n_trials = len(seen_trial_types)

    # Resolves per-trial reward-zone bounds and track lengths from the geometry data file. Missing
    # trial types are surfaced as warnings so partial-geometry sessions render with NaN zone bounds
    # rather than silently dropping the affected trials.
    # noinspection PyTypeChecker
    per_trial_zone_lo: NDArray[np.float32] = np.empty(n_trials, dtype=np.float32)
    # noinspection PyTypeChecker
    per_trial_zone_hi: NDArray[np.float32] = np.empty(n_trials, dtype=np.float32)
    # noinspection PyTypeChecker
    per_trial_length: NDArray[np.float32] = np.empty(n_trials, dtype=np.float32)
    for trial_idx in range(n_trials):
        trial_type_name = seen_trial_types[trial_idx]
        entry = geometry.entries.get(trial_type_name)
        if entry is None:
            warnings.warn(
                message=(
                    f"Trial type {trial_type_name!r} is missing from the geometry layout for session "
                    f"{session_path.name!r}; reward-zone bounds will render as NaN for the affected trials."
                ),
                stacklevel=2,
            )
            per_trial_zone_lo[trial_idx] = np.float32("nan")
            per_trial_zone_hi[trial_idx] = np.float32("nan")
            per_trial_length[trial_idx] = np.float32(_DEFAULT_TRACK_LENGTH_CM)
            continue
        per_trial_zone_lo[trial_idx] = np.float32(entry.stimulus_trigger_zone_start_cm)
        per_trial_zone_hi[trial_idx] = np.float32(entry.stimulus_trigger_zone_end_cm)
        per_trial_length[trial_idx] = np.float32(entry.trial_length_cm)

    # Wraps positions by per-sample trial length so cyclic tracks render at within-trial position
    # rather than continuing to grow unboundedly.
    # noinspection PyTypeChecker
    sample_lengths: NDArray[np.float64] = per_trial_length[trial_index_in_session].astype(np.float64)
    # noinspection PyTypeChecker
    position_wrapped: NDArray[np.float64] = np.mod(position, sample_lengths)

    # Detects rising-edge lick events with a leading-zero diff so the first sample's lick state is
    # treated as a transition from no-lick.
    # noinspection PyTypeChecker
    lick_diff: NDArray[np.int8] = np.diff(np.concatenate(([np.int8(0)], licks))).astype(np.int8)
    # noinspection PyTypeChecker
    rising_edges: NDArray[np.bool_] = lick_diff == 1
    # noinspection PyTypeChecker
    lick_positions: NDArray[np.float32] = position_wrapped[rising_edges].astype(np.float32)
    # noinspection PyTypeChecker
    lick_trial_indices: NDArray[np.int64] = trial_index_in_session[rising_edges].astype(np.int64)

    return SessionLickData(
        trial_types=tuple(seen_trial_types),
        trial_lengths_cm=per_trial_length,
        reward_lo_cm=per_trial_zone_lo,
        reward_hi_cm=per_trial_zone_hi,
        lick_positions=lick_positions,
        lick_trial_indices=lick_trial_indices,
    )


def aggregate_lick_events(sessions: tuple[DatasetSession, ...]) -> LickContext:
    """Walks every session for an animal and assembles a `LickContext` with cumulative trial indices.

    Notes:
        Empty (zero-trial) sessions still appear in ``session_boundaries`` so the plotting layer can
        draw separator lines at every session transition regardless of trial count. Per-block
        cumulative indices are shifted by the cumulative trial count seen across earlier sessions, so
        ``LickContext.lick_trials`` and ``TrialBlock.cum_trial_*`` share a common animal-level zero.

    Args:
        sessions: Chronologically ordered tuple of DatasetSession instances belonging to a single
            animal. Each session's data and trial geometry data files must be present.

    Returns:
        A LickContext aggregating every session's run-state lick events and trial blocks.
    """
    cum_trial_count = 0
    boundaries: list[int] = [0]
    blocks: list[TrialBlock] = []
    lick_position_chunks: list[NDArray[np.float32]] = []
    lick_trial_chunks: list[NDArray[np.int64]] = []
    track_length_max = 0.0

    for session_index, session in enumerate(sessions):
        session_data = extract_session_lick_events(session_path=session.session_path)
        if session_data.n_trials == 0:
            boundaries.append(cum_trial_count)
            continue

        # Composes contiguous-zone trial blocks. Adjacent trials sharing the same
        # (zone_lo, zone_hi, trial_type, trial_length_cm) tuple collapse into one block. Cumulative
        # indices are shifted by ``cum_trial_count`` so they read at animal level.
        zone_keys: list[tuple[float, float, str, float]] = [
            (
                float(session_data.reward_lo_cm[trial_idx]),
                float(session_data.reward_hi_cm[trial_idx]),
                session_data.trial_types[trial_idx],
                float(session_data.trial_lengths_cm[trial_idx]),
            )
            for trial_idx in range(session_data.n_trials)
        ]
        for key, group in groupby(enumerate(zone_keys), key=lambda kv: kv[1]):
            indices_in_block = [item[0] for item in group]
            start_in_session = indices_in_block[0]
            end_in_session = indices_in_block[-1] + 1
            blocks.append(
                TrialBlock(
                    cum_trial_start=cum_trial_count + start_in_session,
                    cum_trial_end=cum_trial_count + end_in_session,
                    session_index=session_index,
                    trial_type=key[2],
                    trial_length_cm=key[3],
                    reward_lo=key[0],
                    reward_hi=key[1],
                )
            )

        track_length_max = max(track_length_max, session_data.track_length_cm)
        if session_data.lick_positions.size > 0:
            lick_position_chunks.append(session_data.lick_positions)
            lick_trial_chunks.append(session_data.lick_trial_indices + cum_trial_count)
        cum_trial_count += session_data.n_trials
        boundaries.append(cum_trial_count)

    if lick_position_chunks:
        # noinspection PyTypeChecker
        lick_positions: NDArray[np.float32] = np.concatenate(lick_position_chunks)
        # noinspection PyTypeChecker
        lick_trials: NDArray[np.int64] = np.concatenate(lick_trial_chunks)
    else:
        # noinspection PyTypeChecker
        lick_positions = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        lick_trials = np.zeros(0, dtype=np.int64)

    return LickContext(
        lick_positions=lick_positions,
        lick_trials=lick_trials,
        trial_blocks=tuple(blocks),
        session_boundaries=tuple(boundaries),
        track_length_cm=track_length_max if track_length_max > 0 else _DEFAULT_TRACK_LENGTH_CM,
    )
