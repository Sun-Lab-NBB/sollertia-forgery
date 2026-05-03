"""Per-animal behavior plots.

Module-level functions consume the per-animal contexts produced by ``.lick_protocol`` and
``.outcome_protocol``. Mirrors the plotting layout of the other ``analysis`` packages so each
subpackage keeps a single file responsible for matplotlib output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from .lick_protocol import TrialBlock, LickContext
from .outcome_protocol import TrialOutcomeContext

if TYPE_CHECKING:
    from numpy.typing import NDArray


_COLOR_LICK: str = "#c2185b"
"""Marker color for individual rising-edge lick events drawn on the main scatter."""
_COLOR_REWARDED: str = "#a6c8e5"
"""Fill color for the reward zone *active* in each trial block."""
_COLOR_NON_REWARDED: str = "#f5b8b8"
"""Fill color for reward zones from other trial types projected onto the current block, rendered so the
operator can see which zones are inactive in each block."""
_COLOR_ABSENT_TRACK: str = "#cccccc"
"""Fill color for the diagonal-hatched mask drawn over the section of the position axis that does not
exist for trials whose ``trial_length_cm`` is shorter than the animal's max track length."""

_COLOR_SUCCESS: str = "#2ca02c"
"""Stack-bar color for trials classified as ``OUTCOME_SUCCESS`` (animal earned the reward)."""
_COLOR_FAILURE: str = "#d62728"
"""Stack-bar color for trials classified as ``OUTCOME_FAILURE`` (animal completed without reward)."""
_COLOR_GUIDED: str = "#7f7f7f"
"""Stack-bar color for trials classified as ``OUTCOME_GUIDED`` (system delivered reward via guidance)."""

_CUE_GRAY_CODE: int = 0
"""Convention used by the project's experiment configurations: cue code 0 is the neutral 'Gray' filler
between named cues, and the cue-block panel renders it in a low-saturation gray."""
_COLOR_CUE_GRAY: str = "#dddddd"
"""Fill color for the neutral 'Gray' cue (code 0) in the top reference panel."""
_COLOR_CUE_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
)
"""Distinct fill colors cycled for non-gray cue codes in the top reference panel. Indexed by
``(code - 1) % len(palette)`` so cue codes 1, 2, 3, ... map to consecutive palette entries."""


def _cue_color(code: int) -> str:
    """Returns the fill color for a cue rectangle based on its uint8 code."""
    if code == _CUE_GRAY_CODE:
        return _COLOR_CUE_GRAY
    return _COLOR_CUE_PALETTE[(code - 1) % len(_COLOR_CUE_PALETTE)]


def plot_lick_scatter(
    context: LickContext,
    *,
    display_sessions: tuple[int, ...] | None = None,
    animal_id: str | None = None,
) -> plt.Figure:
    """Plots discrete lick events for one animal with per-trial reward zones.

    Notes:
        Per-trial reward-zone rectangles are drawn from each block's resolved geometry. Trial-type and
        geometry shifts (longitudinal mid-recording reward-zone shifts or intra-session multi-trial-type
        recordings) appear as different per-trial rectangles with the same color coding (rewarded vs
        non-rewarded relative to the trial's own reward zone). A top reference bar lists every unique
        reward zone present across the rendered sessions, and a hatched mask covers the absent track
        section for blocks whose ``trial_length_cm`` is shorter than the animal's max.

        When ``display_sessions`` is supplied, only the requested sessions render; trial blocks and
        lick events from omitted sessions are dropped, the y-axis renumbers retained trials into a
        contiguous ``0..N`` range in chronological order, and per-block ``session_index`` values are
        re-mapped to the contiguous filtered space. Mirrors the convention used by
        ``..tuning.plotting.plot_sorted_heatmap``.

    Args:
        context: The per-animal lick aggregate whose trial blocks and lick events drive the plot.
        display_sessions: 1-indexed session numbers (matching the original chronological ordering used
            to build ``context``) selecting which sessions to render. Out-of-range entries are
            silently skipped, duplicates are deduplicated, and order is normalized to chronological.
            ``None`` renders every session in ``context``.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.

    Returns:
        A matplotlib Figure showing the lick scatter, the per-trial reward-zone overlay, and the
        unique-reward-zone reference bar.
    """
    if display_sessions is not None:
        context = _filter_context_to_sessions(context=context, display_sessions=display_sessions)

    if context.total_trials == 0:
        figure, axis = plt.subplots(
            1, 1, figsize=(7, 4), facecolor="white", dpi=150, layout="constrained",
        )
        axis.text(0.5, 0.5, "no run-state trials", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.suptitle(_title_with_animal_prefix(animal_id=animal_id, description="lick scatter"))
        return figure

    figure_height = max(8.0, 0.025 * context.total_trials + 4.0)
    # Uses ``constrained_layout`` because the figure mixes a ``GridSpec`` with a per-axis legend
    # anchored outside the axes; ``tight_layout`` warns and produces inconsistent margins for that
    # combination, while ``constrained_layout`` reserves space for the legend, suptitle, and shared
    # x-axis automatically.
    figure = plt.figure(
        figsize=(11, figure_height), facecolor="white", dpi=150, layout="constrained",
    )
    rendered_trial_types = _ordered_rendered_trial_types(context=context)
    cue_panel_weight = max(1.6 * max(len(rendered_trial_types), 1), 1.6)
    grid = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[cue_panel_weight, 22], hspace=0.06)
    ax_zones = figure.add_subplot(grid[0, 0])
    ax_main = figure.add_subplot(grid[1, 0], sharex=ax_zones)

    unique_zones = _collect_unique_reward_zones(context=context)
    _draw_cue_blocks_panel(
        axis=ax_zones, context=context, rendered_trial_types=rendered_trial_types,
    )
    _draw_trial_blocks(axis=ax_main, context=context, unique_zones=unique_zones)

    ax_main.scatter(
        context.lick_positions,
        context.lick_trials,
        s=3,
        color=_COLOR_LICK,
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    for boundary in context.session_boundaries[1:-1]:
        ax_main.axhline(boundary, color="gray", linestyle="--", linewidth=0.6, zorder=1)

    ax_main.set_xlim(0, context.track_length_cm)
    ax_main.set_ylim(context.total_trials, 0)
    ax_main.set_xlabel("Position (cm)", fontsize=11)
    session_count = max(len(context.session_boundaries) - 1, 0)
    y_axis_label = (
        "Trial number (chronological across all sessions)" if session_count > 1
        else "Trial number"
    )
    ax_main.set_ylabel(y_axis_label, fontsize=11)

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="none", markerfacecolor=_COLOR_LICK,
            markersize=7, label="Lick events",
        ),
        Patch(facecolor=_COLOR_REWARDED, edgecolor="black", linewidth=0.6, label="Rewarded location"),
        Patch(facecolor=_COLOR_NON_REWARDED, edgecolor="black", linewidth=0.6, label="Non-rewarded location"),
        Patch(
            facecolor=_COLOR_ABSENT_TRACK, edgecolor="#888888", linewidth=0.6, hatch="///",
            label="Absent track section",
        ),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.7, label="Session separator"),
    ]
    ax_main.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )
    session_label = "session" if session_count == 1 else "sessions"
    figure.suptitle(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=(
                f"discrete lick events across {session_count} {session_label} — "
                f"{context.total_trials} trials, {context.lick_count} rising-edge lick events, "
                f"{len(rendered_trial_types)} trial type(s)"
            ),
        ),
        fontsize=12,
    )
    return figure


def _ordered_rendered_trial_types(context: LickContext) -> tuple[str, ...]:
    """Returns the unique trial types rendered in chronological first-appearance order.

    Notes:
        Iterates ``context.trial_blocks`` (ordered chronologically across sessions) and records each
        trial type the first time it appears, restricted to types whose cue layout is available so
        the cue-block panel only allocates rows it can actually render.
    """
    seen: list[str] = []
    for block in context.trial_blocks:
        if block.trial_type in seen:
            continue
        if block.trial_type not in context.cue_layouts:
            continue
        seen.append(block.trial_type)
    return tuple(seen)


def _collect_unique_reward_zones(context: LickContext) -> dict[tuple[str, float, float], int]:
    """Counts trials per unique (trial_type, reward_lo, reward_hi) tuple across the animal's blocks.

    Notes:
        Skips blocks whose reward bounds are NaN (trial type missing from the session's geometry layout)
        so the reference bar only lists zones the animal actually saw.

    Args:
        context: The per-animal lick aggregate whose blocks drive the inventory.

    Returns:
        Mapping from (trial_type, reward_lo, reward_hi) tuple to the number of trials that fell in
        any block matching that tuple, ordered by insertion (chronological).
    """
    unique_zones: dict[tuple[str, float, float], int] = {}
    for block in context.trial_blocks:
        if not np.isfinite(block.reward_lo) or not np.isfinite(block.reward_hi):
            continue
        key = (block.trial_type, block.reward_lo, block.reward_hi)
        unique_zones[key] = unique_zones.get(key, 0) + (block.cum_trial_end - block.cum_trial_start)
    return unique_zones


def _draw_cue_blocks_panel(
    axis: plt.Axes,
    context: LickContext,
    rendered_trial_types: tuple[str, ...],
) -> None:
    """Renders the top reference panel as one row of cue rectangles per unique trial type.

    Notes:
        Each row reads as a horizontal sequence of colored rectangles spanning the full track length,
        one per `CueSpan` in the trial type's layout. Cue codes drive the fill color so the same code
        gets the same color across rows, with code 0 (the conventional 'Gray' filler) rendered as a
        light gray. Each rectangle is labeled with its cue code so the operator can match the panel
        to the per-sample cue column in the data feather.

    Args:
        axis: The matplotlib axis on which to render the cue rows.
        context: The per-animal lick aggregate whose cue layouts and track length drive the panel.
        rendered_trial_types: Trial types to allocate rows for, in display order.
    """
    n_rows = max(len(rendered_trial_types), 1)
    row_height = 1.0 / n_rows
    axis.set_facecolor("#f8f8f8")
    axis.set_xlim(0, context.track_length_cm)
    axis.set_ylim(0, 1)
    for row_index, trial_type_name in enumerate(rendered_trial_types):
        layout = context.cue_layouts.get(trial_type_name, ())
        y_lo = 1.0 - (row_index + 1) * row_height
        y_hi = 1.0 - row_index * row_height
        rect_y = y_lo + 0.12 * row_height
        rect_height = 0.76 * row_height
        for span in layout:
            width = span.end_cm - span.start_cm
            if width <= 0:
                continue
            axis.add_patch(
                Rectangle(
                    (span.start_cm, rect_y),
                    width,
                    rect_height,
                    facecolor=_cue_color(code=span.code),
                    edgecolor="black",
                    linewidth=0.5,
                )
            )
            if width >= 0.04 * context.track_length_cm:
                axis.text(
                    0.5 * (span.start_cm + span.end_cm),
                    0.5 * (y_lo + y_hi),
                    str(span.code),
                    ha="center", va="center", fontsize=8,
                )
        axis.text(
            -2, 0.5 * (y_lo + y_hi), trial_type_name,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )
    axis.set_yticks([])
    axis.tick_params(labelbottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_trial_blocks(
    axis: plt.Axes,
    context: LickContext,
    unique_zones: dict[tuple[str, float, float], int],
) -> None:
    """Draws the per-trial reward-zone rectangles, non-rewarded rectangles, and the absent-track mask.

    Notes:
        For each block the function draws (i) a hatched mask over the missing section of the position
        axis when ``block.trial_length_cm`` is shorter than the animal's max, (ii) a rewarded rectangle
        spanning the block's own reward zone, and (iii) a non-rewarded rectangle for every other unique
        reward zone whose bounds fall within this block's actual track length. The non-rewarded
        overlay lets the operator see at a glance which alternate-zone licks are mistaken for rewarded
        in the current block.

    Args:
        axis: The matplotlib axis on which to render the rectangles.
        context: The per-animal lick aggregate whose blocks drive the rectangles.
        unique_zones: Mapping returned by `_collect_unique_reward_zones`.
    """
    for block in context.trial_blocks:
        height = block.cum_trial_end - block.cum_trial_start
        if block.trial_length_cm < context.track_length_cm:
            axis.add_patch(
                Rectangle(
                    (block.trial_length_cm, block.cum_trial_start),
                    context.track_length_cm - block.trial_length_cm,
                    height,
                    facecolor=_COLOR_ABSENT_TRACK,
                    edgecolor="#888888",
                    alpha=0.55,
                    hatch="///",
                    linewidth=0.0,
                    zorder=0,
                )
            )
        if not np.isfinite(block.reward_lo) or not np.isfinite(block.reward_hi):
            continue
        axis.add_patch(
            Rectangle(
                (block.reward_lo, block.cum_trial_start),
                block.reward_hi - block.reward_lo,
                height,
                facecolor=_COLOR_REWARDED,
                edgecolor="none",
                alpha=0.45,
                zorder=0,
            )
        )
        # Mirrors non-rewarded zones for *other* trial types this animal sees: every alternate trial
        # type's reward zone gets a non-rewarded rectangle within this block, but only when the
        # alternate zone falls within this block's actual track length.
        for other_key in unique_zones:
            if other_key == (block.trial_type, block.reward_lo, block.reward_hi):
                continue
            other_lo, other_hi = other_key[1], other_key[2]
            if not np.isfinite(other_lo) or not np.isfinite(other_hi):
                continue
            if other_hi > block.trial_length_cm:
                continue
            axis.add_patch(
                Rectangle(
                    (other_lo, block.cum_trial_start),
                    other_hi - other_lo,
                    height,
                    facecolor=_COLOR_NON_REWARDED,
                    edgecolor="none",
                    alpha=0.35,
                    zorder=0,
                )
            )


def _title_with_animal_prefix(*, animal_id: str | None, description: str) -> str:
    """Builds a one-sentence figure title.

    When ``animal_id`` is supplied, prepends ``"Animal {id} "`` so the result reads as a single
    sentence. When ``animal_id`` is ``None``, capitalises the first character of ``description`` so
    the standalone form still reads naturally.
    """
    if animal_id is not None:
        return f"Animal {animal_id} {description}"
    return description[:1].upper() + description[1:]


def _filter_context_to_sessions(
    context: LickContext,
    display_sessions: tuple[int, ...],
) -> LickContext:
    """Returns a new `LickContext` containing only the requested sessions, with re-numbered indices.

    Notes:
        Resolves 1-indexed session numbers against ``context.session_boundaries`` (which carries
        ``len(sessions) + 1`` entries with a leading zero), drops out-of-range and duplicate entries,
        and renumbers retained trials into a contiguous ``0..N`` range in chronological order. Per-block
        ``session_index`` values are re-mapped to the contiguous filtered space so the rendering layer
        can keep its session-boundary logic unchanged.

    Args:
        context: The original per-animal lick aggregate.
        display_sessions: 1-indexed session numbers to retain. Order is normalized to chronological;
            duplicates are deduplicated; out-of-range entries are dropped.

    Returns:
        A LickContext whose trial blocks, lick events, and session boundaries cover only the requested
        sessions and reference a contiguous trial space starting at zero.
    """
    boundaries = context.session_boundaries
    n_sessions = max(len(boundaries) - 1, 0)
    resolved_old_indices: list[int] = []
    for target in display_sessions:
        idx = int(target) - 1
        if 0 <= idx < n_sessions and idx not in resolved_old_indices:
            resolved_old_indices.append(idx)
    resolved_old_indices.sort()

    if not resolved_old_indices:
        # noinspection PyTypeChecker
        empty_positions: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_trials: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        return LickContext(
            lick_positions=empty_positions,
            lick_trials=empty_trials,
            trial_blocks=(),
            session_boundaries=(0,),
            track_length_cm=context.track_length_cm,
            cue_layouts={},
        )

    # Builds the new session_boundaries and the per-session offset used to renumber trials. The
    # ``old_to_new_trial`` lookup maps every retained old trial index to its new contiguous trial
    # index; old trials whose session was filtered out stay at the ``-1`` sentinel and get dropped
    # from the lick events below.
    new_boundaries: list[int] = [0]
    old_to_new_session_index: dict[int, int] = {}
    total_old_trials = boundaries[-1] if boundaries else 0
    # noinspection PyTypeChecker
    old_to_new_trial: NDArray[np.int64] = np.full(total_old_trials, -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(resolved_old_indices):
        old_to_new_session_index[old_idx] = new_idx
        old_start = boundaries[old_idx]
        old_end = boundaries[old_idx + 1]
        n_trials = old_end - old_start
        new_offset = new_boundaries[-1]
        new_boundaries.append(new_offset + n_trials)
        if n_trials > 0:
            old_to_new_trial[old_start:old_end] = (
                np.arange(n_trials, dtype=np.int64) + new_offset
            )

    new_blocks: list[TrialBlock] = []
    for block in context.trial_blocks:
        if block.session_index not in old_to_new_session_index:
            continue
        old_session_start = boundaries[block.session_index]
        new_session_offset = new_boundaries[old_to_new_session_index[block.session_index]]
        shift = new_session_offset - old_session_start
        new_blocks.append(
            TrialBlock(
                cum_trial_start=block.cum_trial_start + shift,
                cum_trial_end=block.cum_trial_end + shift,
                session_index=old_to_new_session_index[block.session_index],
                trial_type=block.trial_type,
                trial_length_cm=block.trial_length_cm,
                reward_lo=block.reward_lo,
                reward_hi=block.reward_hi,
            )
        )

    if context.lick_trials.size > 0:
        # noinspection PyTypeChecker
        new_indices_per_lick: NDArray[np.int64] = old_to_new_trial[context.lick_trials]
        # noinspection PyTypeChecker
        retained_mask: NDArray[np.bool_] = new_indices_per_lick >= 0
        new_lick_positions = context.lick_positions[retained_mask]
        new_lick_trials = new_indices_per_lick[retained_mask]
    else:
        # noinspection PyTypeChecker
        new_lick_positions = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        new_lick_trials = np.zeros(0, dtype=np.int64)

    retained_trial_types = {block.trial_type for block in new_blocks}
    new_cue_layouts = {
        trial_type: layout
        for trial_type, layout in context.cue_layouts.items()
        if trial_type in retained_trial_types
    }

    return LickContext(
        lick_positions=new_lick_positions,
        lick_trials=new_lick_trials,
        trial_blocks=tuple(new_blocks),
        session_boundaries=tuple(new_boundaries),
        track_length_cm=context.track_length_cm,
        cue_layouts=new_cue_layouts,
    )


def plot_trial_outcomes(
    context: TrialOutcomeContext,
    *,
    display_sessions: tuple[int, ...] | None = None,
    animal_id: str | None = None,
) -> plt.Figure:
    """Plots a per-session stacked bar chart of success / failure / guided trial counts.

    Notes:
        Each bar represents one session, positioned along the x-axis at its day offset relative to the
        first session. The stack reads bottom-to-top as success → failure → guided, so the green band
        height directly visualizes the animal's earned-reward count and the total bar height equals
        the trial count for that session.

        When ``display_sessions`` is supplied, only the requested sessions render; their day offsets
        keep the original spacing relative to the animal's first chronological session, matching the
        convention used by ``plot_lick_scatter`` and the tuning per-session figures.

        Sessions whose timestamps fail to parse fall back to chronological-index x-positions to avoid
        leaving gaps in the bar layout.

    Args:
        context: The per-animal trial-outcome aggregate produced by `aggregate_trial_outcomes`.
        display_sessions: 1-indexed session numbers (matching the original chronological ordering used
            to build ``context``) selecting which sessions to render. Out-of-range entries are
            silently skipped, duplicates are deduplicated, and order is normalized to chronological.
            ``None`` renders every session in ``context``.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.

    Returns:
        A matplotlib Figure showing the stacked success / failure / guided counts.
    """
    if display_sessions is not None:
        context = _filter_outcome_context_to_sessions(
            context=context, display_sessions=display_sessions,
        )
    if context.n_sessions == 0:
        figure, axis = plt.subplots(
            1, 1, figsize=(7, 4), facecolor="white", dpi=150, layout="constrained",
        )
        axis.text(0.5, 0.5, "no sessions", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.suptitle(_title_with_animal_prefix(animal_id=animal_id, description="trial outcomes"))
        return figure

    figure, ax_counts = plt.subplots(
        1, 1, figsize=(max(7.0, 0.5 * context.n_sessions + 4.0), 4.0),
        facecolor="white", dpi=150, layout="constrained",
    )

    x_positions = context.day_offsets.astype(np.float64)
    # Picks a bar width that fits even when consecutive sessions span single days; falls back to a
    # fraction of the smallest tick spacing so densely-packed sessions don't overlap.
    if context.n_sessions > 1:
        spacings = np.diff(np.sort(x_positions))
        positive_spacings = spacings[spacings > 0]
        min_step = float(positive_spacings.min()) if positive_spacings.size > 0 else 1.0
    else:
        min_step = 1.0
    bar_width = 0.7 * min_step

    success = context.success_counts.astype(np.int64)
    failure = context.failure_counts.astype(np.int64)
    guided = context.guided_counts.astype(np.int64)

    ax_counts.bar(
        x_positions, success, width=bar_width,
        color=_COLOR_SUCCESS, edgecolor="black", linewidth=0.4, label="Success",
    )
    ax_counts.bar(
        x_positions, failure, width=bar_width, bottom=success,
        color=_COLOR_FAILURE, edgecolor="black", linewidth=0.4, label="Failure",
    )
    ax_counts.bar(
        x_positions, guided, width=bar_width, bottom=success + failure,
        color=_COLOR_GUIDED, edgecolor="black", linewidth=0.4, label="Guided",
    )

    ax_counts.set_xlabel("Days since first session", fontsize=10)
    ax_counts.set_ylabel("Trial count", fontsize=10)
    ax_counts.set_xticks(x_positions)
    ax_counts.set_xticklabels([str(int(d)) for d in context.day_offsets], fontsize=8)
    ax_counts.tick_params(axis="y", labelsize=8)

    legend_handles = [
        Patch(facecolor=_COLOR_SUCCESS, edgecolor="black", linewidth=0.4, label="Success"),
        Patch(facecolor=_COLOR_FAILURE, edgecolor="black", linewidth=0.4, label="Failure"),
        Patch(facecolor=_COLOR_GUIDED, edgecolor="black", linewidth=0.4, label="Guided"),
    ]
    ax_counts.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8)

    total_trials = int(success.sum() + failure.sum() + guided.sum())
    figure.suptitle(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=(
                f"trial outcomes across {context.n_sessions} sessions — "
                f"{int(success.sum())} success / {int(failure.sum())} failure / "
                f"{int(guided.sum())} guided ({total_trials} trials total)"
            ),
        ),
        fontsize=11,
    )
    return figure


def _filter_outcome_context_to_sessions(
    context: TrialOutcomeContext,
    display_sessions: tuple[int, ...],
) -> TrialOutcomeContext:
    """Returns a new `TrialOutcomeContext` containing only the requested sessions.

    Notes:
        Resolves 1-indexed session numbers against ``context.session_names``, drops out-of-range
        and duplicate entries, and normalizes the retained order to chronological. Day offsets are
        preserved verbatim from the source so the rendered bars keep their original temporal spacing
        relative to the animal's first chronological session.
    """
    n_sessions = context.n_sessions
    resolved: list[int] = []
    for target in display_sessions:
        idx = int(target) - 1
        if 0 <= idx < n_sessions and idx not in resolved:
            resolved.append(idx)
    resolved.sort()
    if not resolved:
        # noinspection PyTypeChecker
        empty_int32: NDArray[np.int32] = np.zeros(0, dtype=np.int32)
        # noinspection PyTypeChecker
        empty_int64: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        return TrialOutcomeContext(
            session_names=(),
            day_offsets=empty_int32,
            success_counts=empty_int64,
            failure_counts=empty_int64,
            guided_counts=empty_int64,
        )
    # noinspection PyTypeChecker
    indices: NDArray[np.int64] = np.asarray(resolved, dtype=np.int64)
    return TrialOutcomeContext(
        session_names=tuple(context.session_names[i] for i in resolved),
        day_offsets=context.day_offsets[indices],
        success_counts=context.success_counts[indices],
        failure_counts=context.failure_counts[indices],
        guided_counts=context.guided_counts[indices],
    )
