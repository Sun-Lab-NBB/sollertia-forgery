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
from dataclasses import field, dataclass

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp

from ...shared_assets import (
    DatasetFiles,
    DatasetColumn,
    TrialGeometry,
)
from ..shared_utilities import realign_trial_starts_to_first_cue

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

_SESSION_TIMESTAMP_FORMAT: str = "%Y-%m-%d-%H-%M-%S-%f"
"""``strptime`` format string for the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` session-directory
name. Matches the format used by `..outcome_protocol._resolve_day_offsets`."""


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
class CueSpan:
    """One contiguous span of a single Virtual Reality wall cue within a trial type's cue layout.

    Notes:
        Spans are extracted by walking the per-sample ``cue`` column of the run-state subset of a
        trial: each contiguous run of identical cue codes becomes one span, with start / end positions
        in trial-relative centimeters. Adjacent spans are contiguous (``end_cm`` of one equals
        ``start_cm`` of the next). The first and last spans may be truncated relative to the cue's
        canonical length when the runtime starts mid-cue (``cue_offset_cm > 0``) or when the trial
        ends mid-cue.
    """

    code: int
    """Uint8 cue identifier matching the values written into the data feather's ``cue`` column."""
    start_cm: float
    """Trial-relative start position of the cue span, in centimeters."""
    end_cm: float
    """Trial-relative end position of the cue span, in centimeters."""


@dataclass(frozen=True, slots=True)
class SessionLickData:
    """Run-state lick events plus per-trial reward-zone geometry resolved from a single forged session.

    Notes:
        Trials are canonical-realigned via `realign_trial_starts_to_first_cue` for every trial
        type whose ``cue_offset_cm`` is non-zero, so each entry in the per-trial parallel arrays
        (``trial_types``, ``trial_lengths_cm``, ``reward_lo_cm``, ``reward_hi_cm``) maps to one
        canonical trial whose samples span position 0 to ``trial_length_cm``. Trial 0 is the first
        retained canonical trial of the session; the leading partial trial (whose first sample
        lands mid-first-cue) is dropped during extraction. ``aggregate_lick_events`` shifts
        cumulative indices by the running session-prefix sum to produce animal-level `TrialBlock`
        instances; consumers that only need single-session lick events (e.g., a per-session
        quality plot) can read this dataclass directly without going through the aggregator.
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
    cue_layouts: dict[str, tuple[CueSpan, ...]]
    """Per-trial-type cue layout, mapping each trial type observed in the session to the ordered
    sequence of cue spans the animal traversed within one trial of that type. Empty for sessions with
    no run-state trials."""

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
    cue_layouts: dict[str, tuple[CueSpan, ...]]
    """Per-trial-type cue layout, merged across every session contributing trials of that type.
    Mapping each trial type to the ordered sequence of cue spans the animal traversed within one
    trial of that type. The plotting layer uses these to render the cue-block reference panel."""
    session_indices: tuple[int, ...]
    """One-based session indices in the original chronological ordering used to build the context,
    aligned with ``session_boundaries`` so ``session_indices[i]`` is the original 1-based number of
    the session whose trials lie in the ``[session_boundaries[i], session_boundaries[i + 1])``
    cumulative range. Filtering through ``_filter_context_to_sessions`` preserves the original
    indices, so a context built from sessions ``1..10`` and filtered to ``(2, 5, 7)`` keeps the
    indices ``(2, 5, 7)``. The plotting layer surfaces these in the figure title."""
    true_session_boundaries: tuple[int, ...] = ()
    """Per-session cumulative trial counts using each session's *uncapped* trial total, parallel
    to ``session_boundaries`` and with the same length (``n_sessions + 1``). Aggregated alongside
    ``session_boundaries`` so the rendering layer can keep the trial-axis tick labels referenced
    to the true cumulative count even when ``_cap_lick_context_trials`` clips the rendered set
    into a compact ``0..N`` axis. Filtering and origin-anchoring transformations propagate the
    same shifts that ``session_boundaries`` undergoes (plus the gap collapse the cap intentionally
    hides), so the two boundary tuples stay in lockstep on session count and direction. Defaults
    to an empty tuple when the context was constructed before this field existed; the plotting
    layer treats that as "fall back to ``session_boundaries`` for labels"."""
    day_offsets: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32),
    )
    """Per-session integer day offsets relative to the first chronologically ordered session,
    parallel to ``session_indices``. Sessions whose timestamp fails to parse fall back to their
    chronological index, mirroring `..outcome_protocol._resolve_day_offsets`. Filtering through
    ``_filter_context_to_sessions`` slices this in lockstep with ``session_indices`` so each
    retained session keeps its absolute day offset rather than being renumbered against the
    filter's first entry."""

    @property
    def total_trials(self) -> int:
        """Returns the total number of trials across every session in the aggregate.

        Notes:
            Computes the difference between the leftmost and rightmost boundary so an anchored
            context (where ``session_boundaries[0]`` may be negative) reports a positive count.
            Unanchored contexts have ``session_boundaries[0] == 0`` and the result coincides with
            the cumulative end of the last session, matching the pre-anchoring semantics.
        """
        if not self.session_boundaries:
            return 0
        return int(self.session_boundaries[-1] - self.session_boundaries[0])

    @property
    def lick_count(self) -> int:
        """Returns the total number of rising-edge lick events captured across every session."""
        return int(self.lick_positions.size)


def extract_session_lick_events(session_path: Path) -> SessionLickData:
    """Extracts rising-edge lick events and per-trial reward zones from a single forged session.

    Notes:
        Filters the session's ``data.feather`` to ``system_state == "run"`` plus a sentinel-trial
        cut, then re-anchors trial boundaries to the canonical first-cue start via
        `realign_trial_starts_to_first_cue` for trial types whose ``cue_offset_cm > 0``. After
        realignment each retained trial spans canonical position 0 to ``trial_length_cm`` so the
        lick scatter, cue layout, and reward-zone overlays all share the same coordinate system as
        the tuning rate-map figures. The leading partial trial (whose first sample lands mid-first-
        cue when the runtime starts after a cycle has already begun) is dropped so every retained
        row of the scatter starts at canonical zero.

        Lick events are the rising edges of the binary ``lick`` sample column, mapped to the
        retained trial's canonical position. Trial types absent from the session's geometry layout
        keep their runtime-trial boundaries (no realignment is possible without ``cue_offset_cm``)
        and surface a one-time warning per session.

    Args:
        session_path: Path to the forged session directory containing both ``data.feather`` and the
            trial geometry data file.

    Returns:
        A SessionLickData carrying one entry per retained canonical trial, the rising-edge lick
        events with their session-local canonical-trial indices, and the per-trial-type cue layout
        in canonical coordinates.
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
            DatasetColumn.CUE.value,
        ],
        memory_map=True,
    )
    run = df.filter(
        (pl.col(DatasetColumn.SYSTEM_STATE.value) == "run")
        & (pl.col(DatasetColumn.TRIAL.value) < _NO_TRIAL_SENTINEL),
    )
    if run.height == 0:
        return _empty_session_lick_data()

    # noinspection PyTypeChecker
    distance: NDArray[np.float64] = run[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float64)
    # noinspection PyTypeChecker
    runtime_trials: NDArray[np.int64] = run[DatasetColumn.TRIAL.value].to_numpy().astype(np.int64)
    # noinspection PyTypeChecker
    licks: NDArray[np.int8] = run[DatasetColumn.LICK.value].to_numpy().astype(np.int8)
    # noinspection PyTypeChecker
    cues: NDArray[np.uint8] = run[DatasetColumn.CUE.value].to_numpy().astype(np.uint8)
    trial_types_per_sample: list[str] = run[DatasetColumn.TRIAL_TYPE.value].to_list()
    sample_count = distance.shape[0]

    # Splits run-state samples into contiguous trial-type segments so each segment can be re-
    # anchored against its own geometry independently. Trial-type changes typically follow runtime
    # trial boundaries, so the segment count is on the order of the number of distinct trial types
    # in the session.
    segment_starts, segment_ends = _segment_indices_by_trial_type(
        trial_types_per_sample=trial_types_per_sample,
    )

    canonical_position = np.full(sample_count, np.nan, dtype=np.float64)
    canonical_trial_id = np.full(sample_count, -1, dtype=np.int64)
    canonical_trial_metadata: list[tuple[str, float, float, float]] = []
    warned_trial_types: set[str] = set()
    next_canonical_trial_id = 0

    for segment_start, segment_end in zip(segment_starts, segment_ends, strict=True):
        if segment_end <= segment_start:
            continue
        segment_trial_type = trial_types_per_sample[segment_start]
        entry = geometry.entries.get(segment_trial_type)
        if entry is None:
            if segment_trial_type not in warned_trial_types:
                warnings.warn(
                    message=(
                        f"Trial type {segment_trial_type!r} is missing from the geometry layout "
                        f"for session {session_path.name!r}; reward-zone bounds will render as NaN "
                        f"for the affected trials and canonical realignment is skipped."
                    ),
                    stacklevel=2,
                )
                warned_trial_types.add(segment_trial_type)
            track_length_cm = _DEFAULT_TRACK_LENGTH_CM
            reward_lo_cm = float("nan")
            reward_hi_cm = float("nan")
            cue_offset_cm = 0.0
        else:
            track_length_cm = float(entry.trial_length_cm)
            reward_lo_cm = float(entry.stimulus_trigger_zone_start_cm)
            reward_hi_cm = float(entry.stimulus_trigger_zone_end_cm)
            cue_offset_cm = float(entry.cue_offset_cm)

        segment_distance = distance[segment_start:segment_end]
        segment_cues = cues[segment_start:segment_end]
        segment_runtime_trials = runtime_trials[segment_start:segment_end]

        # Picks the trial-id basis: canonical realignment when the runtime starts mid-first-cue,
        # otherwise the runtime trial column already aligns with cue boundaries.
        if cue_offset_cm != 0.0:
            # noinspection PyTypeChecker
            segment_trial_ids: NDArray[np.int32] = realign_trial_starts_to_first_cue(
                cue=segment_cues,
            )
            drop_leading_partial = True
        else:
            segment_trial_ids = (
                segment_runtime_trials - segment_runtime_trials[0]
            ).astype(np.int32)
            drop_leading_partial = False

        segment_position = _compute_within_trial_position(
            distance=segment_distance,
            trial_ids=segment_trial_ids,
        )

        # Walks each per-segment realigned trial and assigns a session-global canonical id. Trials
        # whose measured length falls below the completeness threshold are dropped so every
        # retained trial spans (close to) the full canonical track length.
        unique_trials = np.unique(segment_trial_ids).tolist()
        for trial_index_in_segment in unique_trials:
            # noinspection PyTypeChecker
            trial_mask: NDArray[np.bool_] = segment_trial_ids == trial_index_in_segment
            if not bool(trial_mask.any()):
                continue
            if drop_leading_partial and int(trial_index_in_segment) == int(unique_trials[0]):
                continue
            trial_position = segment_position[trial_mask]
            measured_length = float(trial_position[-1] - trial_position[0])
            if measured_length < _CANONICAL_COMPLETENESS_THRESHOLD * track_length_cm:
                continue

            # Maps the segment-local samples back into session-global indices and tags them with
            # the new canonical trial id.
            absolute_indices = np.flatnonzero(trial_mask) + segment_start
            canonical_position[absolute_indices] = trial_position
            canonical_trial_id[absolute_indices] = next_canonical_trial_id
            canonical_trial_metadata.append(
                (segment_trial_type, track_length_cm, reward_lo_cm, reward_hi_cm),
            )
            next_canonical_trial_id += 1

    if not canonical_trial_metadata:
        return _empty_session_lick_data()

    # noinspection PyTypeChecker
    valid_mask: NDArray[np.bool_] = canonical_trial_id >= 0
    valid_position = canonical_position[valid_mask]
    valid_trial_id = canonical_trial_id[valid_mask]
    valid_licks = licks[valid_mask]
    valid_cues = cues[valid_mask]

    n_trials = len(canonical_trial_metadata)
    trial_types_out = tuple(metadata[0] for metadata in canonical_trial_metadata)
    # noinspection PyTypeChecker
    trial_lengths_cm: NDArray[np.float32] = np.array(
        [metadata[1] for metadata in canonical_trial_metadata], dtype=np.float32,
    )
    # noinspection PyTypeChecker
    reward_lo_array: NDArray[np.float32] = np.array(
        [metadata[2] for metadata in canonical_trial_metadata], dtype=np.float32,
    )
    # noinspection PyTypeChecker
    reward_hi_array: NDArray[np.float32] = np.array(
        [metadata[3] for metadata in canonical_trial_metadata], dtype=np.float32,
    )

    # Detects rising-edge lick events post-realignment. The leading-zero diff treats the first
    # retained sample's lick state as a transition from no-lick.
    # noinspection PyTypeChecker
    lick_diff: NDArray[np.int8] = np.diff(np.concatenate(([np.int8(0)], valid_licks))).astype(np.int8)
    # noinspection PyTypeChecker
    rising_edges: NDArray[np.bool_] = lick_diff == 1
    # noinspection PyTypeChecker
    lick_positions: NDArray[np.float32] = valid_position[rising_edges].astype(np.float32)
    # noinspection PyTypeChecker
    lick_trial_indices: NDArray[np.int64] = valid_trial_id[rising_edges].astype(np.int64)

    # Extracts one cue layout per unique trial type from the first retained canonical trial of
    # that type. Retained trials all start at canonical position 0, so the layout is canonical.
    # noinspection PyTypeChecker
    valid_trial_starts: NDArray[np.int64] = np.concatenate(
        ([np.int64(0)], np.flatnonzero(np.diff(valid_trial_id)) + 1),
    )
    # noinspection PyTypeChecker
    valid_trial_ends: NDArray[np.int64] = np.concatenate(
        (valid_trial_starts[1:], [np.int64(valid_position.size)]),
    )
    cue_layouts: dict[str, tuple[CueSpan, ...]] = {}
    for trial_index in range(n_trials):
        trial_type_name = trial_types_out[trial_index]
        if trial_type_name in cue_layouts:
            continue
        slice_start = int(valid_trial_starts[trial_index])
        slice_end = int(valid_trial_ends[trial_index])
        layout = _extract_cue_layout(
            cue=valid_cues[slice_start:slice_end],
            position=valid_position[slice_start:slice_end],
            trial_length_cm=float(trial_lengths_cm[trial_index]),
        )
        if layout:
            cue_layouts[trial_type_name] = layout

    return SessionLickData(
        trial_types=trial_types_out,
        trial_lengths_cm=trial_lengths_cm,
        reward_lo_cm=reward_lo_array,
        reward_hi_cm=reward_hi_array,
        lick_positions=lick_positions,
        lick_trial_indices=lick_trial_indices,
        cue_layouts=cue_layouts,
    )


_CANONICAL_COMPLETENESS_THRESHOLD: float = 0.9
"""Minimum fraction of the canonical track length a realigned trial must reach to be retained.
Mirrors `..tuning.utilities.compute_within_trial_position` so the lick scatter and rate-map
figures classify the same trials as complete."""


def _empty_session_lick_data() -> SessionLickData:
    """Returns a `SessionLickData` instance whose every per-trial / per-event array is empty."""
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
        cue_layouts={},
    )


def _segment_indices_by_trial_type(
    trial_types_per_sample: list[str],
) -> tuple[list[int], list[int]]:
    """Returns the (start, end) sample-index pairs for each contiguous trial-type segment."""
    if not trial_types_per_sample:
        return [], []
    segment_starts: list[int] = [0]
    for index in range(1, len(trial_types_per_sample)):
        if trial_types_per_sample[index] != trial_types_per_sample[index - 1]:
            segment_starts.append(index)
    segment_ends: list[int] = [*segment_starts[1:], len(trial_types_per_sample)]
    return segment_starts, segment_ends


def _compute_within_trial_position(
    distance: NDArray[np.float64],
    trial_ids: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Computes per-sample within-trial position by subtracting each trial's first-sample distance.

    Notes:
        Mirrors `..tuning.utilities.compute_within_trial_position` minus the completeness mask:
        callers in this module apply their own per-trial measured-length filter against the
        per-trial-type ``track_length_cm`` after this helper returns.
    """
    if distance.size == 0:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float64)
    # noinspection PyTypeChecker
    change_indices: NDArray[np.int64] = np.flatnonzero(np.diff(trial_ids)) + 1
    # noinspection PyTypeChecker
    starts: NDArray[np.int64] = np.concatenate(([np.int64(0)], change_indices))
    # noinspection PyTypeChecker
    ends: NDArray[np.int64] = np.concatenate((change_indices, [np.int64(distance.size)]))
    counts = ends - starts
    per_trial_start = distance[starts]
    # noinspection PyTypeChecker
    per_sample_start: NDArray[np.float64] = np.repeat(a=per_trial_start, repeats=counts)
    return distance - per_sample_start


def _extract_cue_layout(
    cue: NDArray[np.uint8],
    position: NDArray[np.float64],
    trial_length_cm: float,
) -> tuple[CueSpan, ...]:
    """Returns the ordered cue layout extracted from one trial's per-sample arrays.

    Notes:
        Detects cue transitions with a single ``np.diff`` over the cue codes; each contiguous run of
        the same code becomes one `CueSpan`. The first span starts at ``position[0]`` (typically 0
        but can be slightly positive when the first sample is mid-cue), each subsequent span starts
        at the position of the transition sample, and the final span ends at ``trial_length_cm`` so
        the layout always spans the full trial axis.

    Args:
        cue: Per-sample uint8 cue codes for one trial, length ``sample_count``.
        position: Per-sample trial-relative position in centimeters for the same trial, parallel to
            ``cue``.
        trial_length_cm: The canonical track length the trial is mapped onto. Used as the final
            span's end position so the layout closes at the canonical trial end.

    Returns:
        Ordered tuple of CueSpan instances covering the trial. Empty when the trial has no samples.
    """
    if cue.size == 0:
        return ()
    # noinspection PyTypeChecker
    transitions: NDArray[np.int64] = np.flatnonzero(np.diff(cue.astype(np.int64))) + 1
    # noinspection PyTypeChecker
    starts: NDArray[np.int64] = np.concatenate(([np.int64(0)], transitions))
    # noinspection PyTypeChecker
    ends: NDArray[np.int64] = np.concatenate((transitions, [np.int64(cue.size)]))
    spans: list[CueSpan] = []
    for span_index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        end_position = (
            float(position[end]) if span_index < starts.size - 1 else float(trial_length_cm)
        )
        spans.append(
            CueSpan(
                code=int(cue[start]),
                start_cm=float(position[start]),
                end_cm=end_position,
            )
        )
    return tuple(spans)


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
    cue_layouts: dict[str, tuple[CueSpan, ...]] = {}

    for session_index, session in enumerate(sessions):
        session_data = extract_session_lick_events(session_path=session.session_path)
        for trial_type_name, layout in session_data.cue_layouts.items():
            cue_layouts.setdefault(trial_type_name, layout)
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

    session_names = tuple(session.session for session in sessions)
    day_offsets = _resolve_session_day_offsets(session_names=session_names)

    return LickContext(
        lick_positions=lick_positions,
        lick_trials=lick_trials,
        trial_blocks=tuple(blocks),
        session_boundaries=tuple(boundaries),
        # Pre-cap the true and display boundaries match exactly. Downstream cap / filter / anchor
        # transformations diverge them: the display axis collapses cap-induced gaps so the figure
        # stays compact, while the true tuple keeps each session's uncapped extent so labels can
        # still surface the actual cumulative trial counts.
        true_session_boundaries=tuple(boundaries),
        track_length_cm=track_length_max if track_length_max > 0 else _DEFAULT_TRACK_LENGTH_CM,
        cue_layouts=cue_layouts,
        session_indices=tuple(range(1, len(sessions) + 1)),
        day_offsets=day_offsets,
    )


def _resolve_session_day_offsets(session_names: tuple[str, ...]) -> NDArray[np.int32]:
    """Returns integer day offsets relative to the first parseable session timestamp.

    Notes:
        Mirrors `..outcome_protocol._resolve_day_offsets`: sessions whose name fails to parse fall
        back to their chronological index so consumers can place every session on a continuous day
        axis without gaps. Hoisted into ``lick_protocol`` so the lick aggregate can carry the same
        per-session day annotation as ``TrialOutcomeContext`` without taking a cross-module
        private-helper dependency.

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
