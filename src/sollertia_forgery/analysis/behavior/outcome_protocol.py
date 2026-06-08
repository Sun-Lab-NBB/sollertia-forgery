"""Per-session and per-animal trial-outcome classification utilities.

Module-level functions consume a forged session's persisted ``data.feather`` and produce per-trial
outcome labels (success / failure / guided) and per-animal aggregate counts. The pipeline is stateless:
no feather or YAML artifact is persisted, since every input lives in already-forged dataset files and
the per-session compute is a polars filter plus per-trial reductions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp

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
"""Sentinel trial id the acquisition pipeline writes for samples outside any trial. Dropped before any
per-trial reduction."""

_SESSION_TIMESTAMP_FORMAT: str = "%Y-%m-%d-%H-%M-%S-%f"
"""``strptime`` format string for the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` session-directory name."""


class TrialOutcome(StrEnum):
    """Per-trial outcome label produced by `extract_session_trial_outcomes`.

    Notes:
        Every run-state trial is classified as exactly one of these three values; the categories are
        mutually exclusive by construction. A trial is `SUCCESS` whenever the animal triggered the
        reward via licking inside the stimulus trigger zone (its position at the first rewarded
        sample sits before ``stimulus_location_cm``) — this overrides the guided-trial flag because
        the animal earned the reward before the auto-release fired. A trial is `GUIDED` when the
        guidance flag is set but the animal did not initiate the reward in time; the system
        delivered the reward under auto-release.

        The integer codes (``code`` property) are the storage representation in
        ``SessionTrialOutcomes.trial_outcomes``; the StrEnum members are the human-readable companion
        consumed by visualization and reporting code.
    """

    FAILURE = "failure"
    """Animal completed the trial without earning a reward and without guidance."""
    SUCCESS = "success"
    """Animal initiated the reward via licking inside the stimulus trigger zone before the
    automated release fired (or earned a reward on a non-guided trial)."""
    GUIDED = "guided"
    """Trial ran under reinforcing or aversive guidance and the animal did not initiate the reward
    before the auto-release fired."""

    @property
    def code(self) -> int:
        """Returns the integer code used in `SessionTrialOutcomes.trial_outcomes` for this label."""
        return _OUTCOME_CODE_BY_LABEL[self]


# Stable integer codes paired with the StrEnum members so the per-trial array can stay ``uint8``
# while consumers continue to use the readable label form. Both directions of the mapping are
# materialized once at module-import time so the property accessors are constant-time lookups.
_OUTCOME_CODE_BY_LABEL: dict[TrialOutcome, int] = {
    TrialOutcome.FAILURE: 0,
    TrialOutcome.SUCCESS: 1,
    TrialOutcome.GUIDED: 2,
}
"""Mapping from `TrialOutcome` to its integer code in ``SessionTrialOutcomes.trial_outcomes``."""

_OUTCOME_LABEL_BY_CODE: dict[int, TrialOutcome] = {
    code: label for label, code in _OUTCOME_CODE_BY_LABEL.items()
}
"""Mapping from integer code in ``SessionTrialOutcomes.trial_outcomes`` to its `TrialOutcome`."""

OUTCOME_FAILURE: int = _OUTCOME_CODE_BY_LABEL[TrialOutcome.FAILURE]
"""Integer code for `TrialOutcome.FAILURE` in ``SessionTrialOutcomes.trial_outcomes``."""
OUTCOME_SUCCESS: int = _OUTCOME_CODE_BY_LABEL[TrialOutcome.SUCCESS]
"""Integer code for `TrialOutcome.SUCCESS` in ``SessionTrialOutcomes.trial_outcomes``."""
OUTCOME_GUIDED: int = _OUTCOME_CODE_BY_LABEL[TrialOutcome.GUIDED]
"""Integer code for `TrialOutcome.GUIDED` in ``SessionTrialOutcomes.trial_outcomes``."""


@dataclass(frozen=True, slots=True)
class SessionTrialOutcomes:
    """Per-trial outcome labels for a single forged session.

    Notes:
        Every run-state trial gets exactly one outcome — the three categories ``OUTCOME_FAILURE``,
        ``OUTCOME_SUCCESS``, ``OUTCOME_GUIDED`` are mutually exclusive by construction and partition
        the trial set with no overlap. ``trial_outcomes`` carries the integer code form for compact
        storage; ``outcome_labels`` returns the matching `TrialOutcome` StrEnum form for downstream
        code that prefers readable labels. The aggregate ``*_count`` properties cover consumers that
        only need totals.
    """

    trial_outcomes: NDArray[np.uint8]
    """Per-trial outcome codes with length ``n_trials``. Values are the module-level ``OUTCOME_*``
    constants — equivalently, ``label.code`` for ``label in TrialOutcome``."""

    @property
    def n_trials(self) -> int:
        """Returns the total number of run-state trials in the session."""
        return int(self.trial_outcomes.size)

    @property
    def success_count(self) -> int:
        """Returns the number of trials classified as `TrialOutcome.SUCCESS`."""
        return int(np.sum(self.trial_outcomes == OUTCOME_SUCCESS))

    @property
    def failure_count(self) -> int:
        """Returns the number of trials classified as `TrialOutcome.FAILURE`."""
        return int(np.sum(self.trial_outcomes == OUTCOME_FAILURE))

    @property
    def guided_count(self) -> int:
        """Returns the number of trials classified as `TrialOutcome.GUIDED`."""
        return int(np.sum(self.trial_outcomes == OUTCOME_GUIDED))

    @property
    def outcome_labels(self) -> tuple[TrialOutcome, ...]:
        """Returns the per-trial outcomes as `TrialOutcome` enum values aligned with ``trial_outcomes``."""
        return tuple(_OUTCOME_LABEL_BY_CODE[int(code)] for code in self.trial_outcomes)


@dataclass(frozen=True, slots=True)
class TrialOutcomeContext:
    """Per-animal trial-outcome aggregate spanning every chronologically-ordered session.

    Notes:
        Each per-session count array is aligned with ``session_names`` and ``day_offsets`` along the
        same axis. ``day_offsets`` carries the integer day offset relative to the first session in
        chronological order; sessions whose timestamp fails to parse fall back to a value derived
        from their position in the chronological list.
    """

    session_names: tuple[str, ...]
    """Chronologically ordered session directory names; aligns one-to-one with the count arrays."""
    day_offsets: NDArray[np.int32]
    """Integer day offset relative to the first chronologically ordered session, length ``n_sessions``."""
    success_counts: NDArray[np.int64]
    """Per-session count of ``OUTCOME_SUCCESS`` trials, length ``n_sessions``."""
    failure_counts: NDArray[np.int64]
    """Per-session count of ``OUTCOME_FAILURE`` trials, length ``n_sessions``."""
    guided_counts: NDArray[np.int64]
    """Per-session count of ``OUTCOME_GUIDED`` trials, length ``n_sessions``."""

    @property
    def n_sessions(self) -> int:
        """Returns the number of sessions covered by the aggregate."""
        return len(self.session_names)


def extract_session_trial_outcomes(session_path: Path) -> SessionTrialOutcomes:
    """Classifies each run-state trial in a single forged session as success / failure / guided.

    Notes:
        Filters the session's ``data.feather`` to ``system_state == "run"`` plus a sentinel-trial cut,
        then walks samples once to compute per-trial reductions and assigns each trial exactly one
        outcome from the mutually exclusive set ``{TrialOutcome.FAILURE, TrialOutcome.SUCCESS,
        TrialOutcome.GUIDED}``:

        * ``SUCCESS`` when the animal initiated the reward via licking inside the stimulus trigger
          zone before the auto-release fired. Detected by checking whether the trial-relative
          position at the *first* rewarded sample sits below the trial type's
          ``stimulus_location_cm`` boundary, which the animal must touch to trigger the auto-
          release. Wins precedence — overrides the guided-trial flag because the animal earned
          the reward before the system would have intervened. Also covers regular non-guided
          rewarded trials (no guidance flag, any rewarded sample).
        * ``GUIDED`` when the guidance flag is set on any sample and the trial does not qualify as
          ``SUCCESS``. Captures both the auto-release case (reward fired at or past
          ``stimulus_location_cm``) and the no-reward-but-guidance case.
        * ``FAILURE`` otherwise — the trial completed with neither a lick-earned reward nor any
          guidance event.

        Sessions whose feather lacks either guidance column simply skip that contribution to the
        guided check. Trials whose trial type is missing from the session's geometry layout fall
        back to the original guidance-precedent logic (no position-based override is possible).

    Args:
        session_path: Path to the forged session directory containing ``data.feather`` and the
            trial geometry data file.

    Returns:
        A SessionTrialOutcomes with one entry per unique completed run-state trial.
    """
    geometry = TrialGeometry.from_yaml(file_path=session_path.joinpath(DatasetFiles.TRIAL_GEOMETRY))
    data_path = session_path.joinpath(DatasetFiles.DATA)
    available_columns = pl.scan_ipc(source=data_path).collect_schema().names()
    has_reinforcing = DatasetColumn.REINFORCING_GUIDED.value in available_columns
    has_aversive = DatasetColumn.AVERSIVE_GUIDED.value in available_columns

    columns = [
        DatasetColumn.SYSTEM_STATE.value,
        DatasetColumn.TRIAL.value,
        DatasetColumn.TRIAL_TYPE.value,
        DatasetColumn.REWARD.value,
        DatasetColumn.DISTANCE_CM.value,
    ]
    if has_reinforcing:
        columns.append(DatasetColumn.REINFORCING_GUIDED.value)
    if has_aversive:
        columns.append(DatasetColumn.AVERSIVE_GUIDED.value)

    df = pl.read_ipc(source=data_path, columns=columns, memory_map=True)
    run = df.filter(
        (pl.col(DatasetColumn.SYSTEM_STATE.value) == "run")
        & (pl.col(DatasetColumn.TRIAL.value) < _NO_TRIAL_SENTINEL),
    )
    if run.height == 0:
        # noinspection PyTypeChecker
        return SessionTrialOutcomes(trial_outcomes=np.zeros(0, dtype=np.uint8))

    # noinspection PyTypeChecker
    trials: NDArray[np.int64] = run[DatasetColumn.TRIAL.value].to_numpy().astype(np.int64)
    # The forged ``reward`` column is a polars Enum with values ``"no"`` (no reward event),
    # ``"yes"`` (water dispensed), and ``"tone"`` (the reward tone is still playing). The reward
    # tone outlasts the water-pulse window, so both ``"yes"`` and ``"tone"`` mark a rewarded sample;
    # only ``"no"`` counts as no reward.
    # noinspection PyTypeChecker
    sample_rewarded: NDArray[np.bool_] = (
        (run[DatasetColumn.REWARD.value] != "no").to_numpy()
    )
    # noinspection PyTypeChecker
    distance: NDArray[np.float64] = (
        run[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float64)
    )
    trial_types_per_sample: list[str] = run[DatasetColumn.TRIAL_TYPE.value].to_list()
    # noinspection PyTypeChecker
    sample_count: int = int(trials.size)

    # Builds a per-sample "is guided" mask combining whichever guidance columns are present. Missing
    # columns contribute all-False so absent guidance never marks a trial as guided.
    # noinspection PyTypeChecker
    sample_is_guided: NDArray[np.bool_] = np.zeros(sample_count, dtype=np.bool_)
    if has_reinforcing:
        # noinspection PyTypeChecker
        reinforcing_guided: NDArray[np.uint8] = (
            run[DatasetColumn.REINFORCING_GUIDED.value].to_numpy().astype(np.uint8)
        )
        sample_is_guided |= reinforcing_guided > 0
    if has_aversive:
        # noinspection PyTypeChecker
        aversive_guided: NDArray[np.uint8] = (
            run[DatasetColumn.AVERSIVE_GUIDED.value].to_numpy().astype(np.uint8)
        )
        sample_is_guided |= aversive_guided > 0

    # Assigns each sample a session-local trial slot by walking the trial column once. Samples within
    # one trial share the same slot; switching trial id starts a new slot. Tracks each slot's first-
    # sample index and trial-type label so the lick-earned check below can compute the trial-relative
    # position at first reward without re-walking the trial column.
    # noinspection PyTypeChecker
    sample_slots: NDArray[np.int64] = np.zeros(sample_count, dtype=np.int64)
    slot_first_sample: list[int] = [0]
    slot_trial_types: list[str] = [trial_types_per_sample[0]]
    current_trial = int(trials[0])
    slot_count = 1
    for sample_index in range(1, sample_count):
        if int(trials[sample_index]) != current_trial:
            current_trial = int(trials[sample_index])
            slot_count += 1
            slot_first_sample.append(sample_index)
            slot_trial_types.append(trial_types_per_sample[sample_index])
        sample_slots[sample_index] = slot_count - 1

    # Per-trial reductions: a trial is guided if ANY of its samples has the guidance flag set, and
    # rewarded if ANY of its samples has the reward flag set.
    # noinspection PyTypeChecker
    trial_guided: NDArray[np.bool_] = np.zeros(slot_count, dtype=np.bool_)
    # noinspection PyTypeChecker
    trial_rewarded: NDArray[np.bool_] = np.zeros(slot_count, dtype=np.bool_)
    np.logical_or.at(trial_guided, sample_slots, sample_is_guided)
    np.logical_or.at(trial_rewarded, sample_slots, sample_rewarded)

    # Per-trial lick-earned check: the trial-relative position at the first rewarded sample is
    # below the trial type's ``stimulus_location_cm`` boundary, indicating the animal initiated the
    # reward via licking inside the stimulus trigger zone before the auto-release fired. Trials
    # whose trial type is missing from the geometry layout cannot resolve a boundary and stay False
    # (the original guided-precedent logic still applies via ``trial_guided``).
    # noinspection PyTypeChecker
    trial_lick_earned: NDArray[np.bool_] = np.zeros(slot_count, dtype=np.bool_)
    for slot_index in range(slot_count):
        if not bool(trial_rewarded[slot_index]):
            continue
        slot_start = slot_first_sample[slot_index]
        slot_end = (
            slot_first_sample[slot_index + 1] if slot_index + 1 < slot_count else sample_count
        )
        # noinspection PyTypeChecker
        slot_reward_window: NDArray[np.bool_] = sample_rewarded[slot_start:slot_end]
        first_reward_offset = int(np.argmax(slot_reward_window))
        position_at_reward = float(
            distance[slot_start + first_reward_offset] - distance[slot_start]
        )
        entry = geometry.entries.get(slot_trial_types[slot_index])
        if entry is None:
            continue
        if position_at_reward < float(entry.stimulus_location_cm):
            trial_lick_earned[slot_index] = True

    # Classification: lick-earned trials win unconditionally (the animal triggered the reward via
    # licking before the auto-release fired). Otherwise, guided takes precedence over a "rewarded
    # but not lick-earned" trial because the system delivered the reward via auto-release.
    # noinspection PyTypeChecker
    trial_outcomes: NDArray[np.uint8] = np.full(slot_count, OUTCOME_FAILURE, dtype=np.uint8)
    success_mask = trial_lick_earned | (trial_rewarded & ~trial_guided)
    trial_outcomes[success_mask] = OUTCOME_SUCCESS
    guided_mask = trial_guided & ~success_mask
    trial_outcomes[guided_mask] = OUTCOME_GUIDED

    return SessionTrialOutcomes(trial_outcomes=trial_outcomes)


def aggregate_trial_outcomes(sessions: tuple[DatasetSession, ...]) -> TrialOutcomeContext:
    """Walks every session for an animal and assembles per-session success/failure/guided counts.

    Notes:
        Sessions whose ``data.feather`` is missing or whose run-state filter yields zero trials still
        appear in the returned context with zero counts — the per-session axis stays aligned with
        ``sessions`` so downstream visualization can render gaps explicitly.

    Args:
        sessions: Chronologically ordered DatasetSession instances belonging to a single animal.

    Returns:
        A TrialOutcomeContext aggregating every session's success/failure/guided counts plus the
        per-session day offsets relative to ``sessions[0]``.
    """
    n_sessions = len(sessions)
    # noinspection PyTypeChecker
    success_counts: NDArray[np.int64] = np.zeros(n_sessions, dtype=np.int64)
    # noinspection PyTypeChecker
    failure_counts: NDArray[np.int64] = np.zeros(n_sessions, dtype=np.int64)
    # noinspection PyTypeChecker
    guided_counts: NDArray[np.int64] = np.zeros(n_sessions, dtype=np.int64)

    session_names: list[str] = []
    for session_index, session in enumerate(sessions):
        session_names.append(session.session)
        if not session.data_path.exists():
            continue
        outcomes = extract_session_trial_outcomes(session_path=session.session_path)
        success_counts[session_index] = outcomes.success_count
        failure_counts[session_index] = outcomes.failure_count
        guided_counts[session_index] = outcomes.guided_count

    day_offsets = _resolve_day_offsets(session_names=tuple(session_names))
    return TrialOutcomeContext(
        session_names=tuple(session_names),
        day_offsets=day_offsets,
        success_counts=success_counts,
        failure_counts=failure_counts,
        guided_counts=guided_counts,
    )


def _resolve_day_offsets(session_names: tuple[str, ...]) -> NDArray[np.int32]:
    """Returns integer day offsets relative to the first parseable session timestamp.

    Notes:
        Sessions whose timestamp fails to parse fall back to their chronological index so the bar
        chart can still place them on the x-axis without a tick gap. Mirrors the convention used by
        the per-day rate-map plotting helpers.

    Args:
        session_names: Chronologically ordered session directory names.

    Returns:
        Per-session integer day offsets aligned with ``session_names``.
    """
    n_sessions = len(session_names)
    # noinspection PyTypeChecker
    day_offsets: NDArray[np.int32] = np.zeros(n_sessions, dtype=np.int32)
    if n_sessions == 0:
        return day_offsets
    try:
        first_us = int(parse_timestamp(
            date_string=session_names[0],
            format_string=_SESSION_TIMESTAMP_FORMAT,
            output_format=TimestampFormats.INTEGER,
        ))
    except ValueError:
        first_us = None
    for session_index, name in enumerate(session_names):
        if first_us is None:
            day_offsets[session_index] = np.int32(session_index)
            continue
        try:
            session_us = int(parse_timestamp(
                date_string=name,
                format_string=_SESSION_TIMESTAMP_FORMAT,
                output_format=TimestampFormats.INTEGER,
            ))
        except ValueError:
            day_offsets[session_index] = np.int32(session_index)
            continue
        day_offsets[session_index] = np.int32(round(float(convert_time(
            time=session_us - first_us,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.DAY,
            as_float=True,
        ))))
    return day_offsets
