"""Per-animal behavior plots.

Module-level functions consume the per-animal contexts produced by ``.lick_protocol`` and
``.outcome_protocol``. Mirrors the plotting layout of the other ``analysis`` packages so each
subpackage keeps a single file responsible for matplotlib output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from .lick_protocol import TrialBlock, LickContext
from .outcome_protocol import TrialOutcomeContext
from ..shared_utilities import resolve_figure_style, resolve_figure_width_scale

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
_COLOR_CUE_GRAY: str = "#9e9e9e"
"""Fill color for the neutral 'Gray' cue (code 0) in the top reference panel; chosen darker than the
panel's facecolor so the gray cues read as a distinct fill rather than a transparent gap."""
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
    cue_textures: dict[int, NDArray] | None = None,
    max_trials_per_session: int | None = 50,
    trial_origin_session: int | None = None,
    figure_dpi: int = 150,
    figure_preset: str = "print",
) -> plt.Figure:
    """Plots discrete lick events for one animal with per-trial reward zones.

    Notes:
        Per-trial reward-zone rectangles are drawn from each block's resolved geometry. Trial-type and
        geometry shifts (longitudinal mid-recording reward-zone shifts or intra-session multi-trial-type
        recordings) appear as different per-trial rectangles with the same color coding (rewarded vs
        non-rewarded relative to the trial's own reward zone). A top reference bar lists every unique
        reward zone present across the rendered sessions, and a hatched mask covers the absent track
        section for blocks whose ``trial_length_cm`` is shorter than the animal's max.

        Three transformations are composed before rendering, in this order:

        1. **Per-session cap.** When ``max_trials_per_session`` is set, only the first
           ``max_trials_per_session`` trials of each session are kept; later trials and their lick
           events are dropped. The default of ``50`` matches the typical chronic-imaging session
           length where the first sweep has the cleanest licking; pass ``None`` to disable.
        2. **Display filter.** When ``display_sessions`` is supplied, only the requested sessions
           render and the y-axis renumbers retained trials into a contiguous ``0..N`` range in
           chronological order.
        3. **Origin anchor.** When ``trial_origin_session`` is set, the trial axis is shifted so the
           first trial of that session sits at index 0. Trials of earlier retained sessions become
           negative; trials of later retained sessions remain non-negative. When the origin session
           is not in the retained set, the shift puts the boundary between before-origin and
           at-or-after-origin sessions at 0 so the sign convention still holds.

        A secondary y-axis on the right edge labels each retained session block at its trial-axis
        midpoint as ``Session X (Day Y)``, where ``X`` is the original 1-indexed session number and
        ``Y`` is the integer day offset relative to the animal's first session. The day component is
        omitted for sessions whose timestamp could not be parsed at aggregation time.

        The legend's ``Absent track section`` entry is suppressed when every retained block's
        ``trial_length_cm`` matches the animal's maximum (i.e., no hatched mask was actually drawn).

    Args:
        context: The per-animal lick aggregate whose trial blocks and lick events drive the plot.
        display_sessions: 1-indexed session numbers (matching the original chronological ordering used
            to build ``context``) selecting which sessions to render. Out-of-range entries are
            silently skipped, duplicates are deduplicated, and order is normalized to chronological.
            ``None`` renders every session in ``context``.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.
        cue_textures: Optional mapping from cue uint8 code to an image array (HxW, HxWx3, or HxWx4)
            used to fill that code's rectangles in the reference strip. Codes without an entry fall
            back to the per-code solid color. ``load_cue_textures`` resolves the mapping from a
            session's ``experiment_configuration.yaml`` and the Unity textures directory.
        max_trials_per_session: Maximum number of trials retained per session. Trials beyond this
            cap (and their lick events) are dropped before rendering. Default ``50``; pass ``None``
            to disable the cap, ``0`` to drop every trial.
        trial_origin_session: 1-indexed session number whose first trial becomes trial-axis index
            ``0``. Earlier retained sessions render with negative trial numbers; later retained
            sessions render with non-negative trial numbers. ``None`` keeps the default
            ``0..N`` numbering anchored at the first retained session.
        figure_dpi: Output figure DPI. Threaded through to ``matplotlib.figure.Figure`` and reused
            for the placeholder figure rendered when the context contains no trials.
        figure_preset: ``"print"`` (default) or ``"presentation"``. Resolves the per-element font
            sizes through `..shared_utilities.resolve_figure_style`; the height-proportional
            ``font_scale`` then multiplies these baseline sizes so the lick scatter remains
            consistent with other analysis plotters at either preset.

    Returns:
        A matplotlib Figure showing the lick scatter, the per-trial reward-zone overlay, and the
        unique-reward-zone reference bar.
    """
    style = resolve_figure_style(preset=figure_preset)
    # Resolves the origin's true cumulative trial count BEFORE the display filter; ``filter``
    # drops non-displayed sessions, but the user's origin (e.g., ``trial_origin_session=1`` while
    # displaying sessions 5/10/15) might be one of those dropped sessions. Capturing it from the
    # full pre-filter context lets the anchor honor the true gap between origin and the first
    # retained session even when origin itself isn't rendered.
    origin_true_position = (
        _resolve_origin_true_position(context=context, trial_origin_session=trial_origin_session)
        if trial_origin_session is not None else None
    )
    if max_trials_per_session is not None:
        context = _cap_lick_context_trials(
            context=context, max_trials_per_session=max_trials_per_session,
        )
    if display_sessions is not None:
        context = _filter_context_to_sessions(context=context, display_sessions=display_sessions)
    if trial_origin_session is not None:
        context = _anchor_lick_context_trials(
            context=context,
            trial_origin_session=trial_origin_session,
            true_origin_position=origin_true_position,
        )

    if context.total_trials == 0:
        figure, axis = plt.subplots(
            1, 1, figsize=(7, 4), facecolor="white", dpi=figure_dpi, layout="constrained",
        )
        axis.text(0.5, 0.5, "no run-state trials", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.suptitle(_title_with_animal_prefix(animal_id=animal_id, description="lick scatter"))
        return figure

    figure_height = max(8.0, 0.025 * context.total_trials + 4.0)
    # Scales the in-figure text proportionally to ``figure_height``. Matplotlib renders text in
    # absolute points, but Jupyter rescales the inline figure to fit the cell width — so a taller
    # figure ends up scaled down more, and fixed-point text appears proportionally smaller next to
    # the data. Anchoring the scale to the 8" floor (the smallest ``figure_height`` ever produced)
    # keeps the shortest figure at the historical fontsizes while taller figures grow their text
    # so the visual size stays stable across session counts. Every fontsize and the lick scatter's
    # marker size below derive from this single ``font_scale`` so a future tweak only needs to
    # adjust the baseline rather than chase fontsizes through every helper.
    font_scale = figure_height / 8.0
    # Baseline font sizes come from the ``figure_preset`` resolver — either the print preset
    # (suptitle 12 / xlabel 9 / ylabel 9 / tick 9 / legend 9) which matches
    # ``..tuning.plotting.plot_sorted_heatmap``, or the presentation preset (suptitle 18 /
    # xlabel 16 / ylabel 16 / tick 14 / legend 14) sized to the slide-projector floor.
    # ``font_scale`` then multiplies the baseline so taller figures still render at consistent
    # visual size after Jupyter rescales them.
    title_fontsize = style.suptitle_fontsize * font_scale
    axis_label_fontsize = style.axis_label_fontsize * font_scale
    tick_fontsize = style.tick_label_fontsize * font_scale
    legend_fontsize = style.legend_fontsize * font_scale
    # Markers grow with the preset, but at half the rate of the tick labels — the full
    # tick-label ratio (14/9 ≈ 1.56× diameter on the presentation preset) drowns the lick events
    # against the reward-zone fills, so the diameter is interpolated halfway between the print
    # baseline and the preset ratio. ``font_scale`` (height-proportional) still multiplies on
    # top so taller figures keep their height-driven boost regardless of preset; the scatter
    # ``s`` parameter is an area, hence the square.
    marker_scale = 1.0 + 0.5 * (style.tick_label_fontsize / 9.0 - 1.0)
    legend_marker_size = 7.0 * font_scale * marker_scale
    lick_marker_area = 3.0 * (font_scale * marker_scale) ** 2

    # Uses ``constrained_layout`` because the figure pairs the main scatter with a legend anchored
    # outside the axes; ``tight_layout`` warns and produces inconsistent margins for that
    # combination, while ``constrained_layout`` reserves space for the legend, suptitle, and the
    # cue strip drawn above the axis automatically. The 9-inch print baseline gives the
    # track-position axis enough breathing room; ``resolve_figure_width_scale`` widens the figure
    # for presentation so the bumped-up labels do not compress the data area when
    # ``constrained_layout`` reserves their margin space. ``font_scale`` is a function of
    # ``figure_height`` alone, so widening leaves the lick marker size and every fontsize untouched.
    width_scale = resolve_figure_width_scale(style=style)
    figure = plt.figure(
        figsize=(9 * width_scale, figure_height),
        facecolor="white", dpi=figure_dpi, layout="constrained",
    )
    ax_main = figure.add_subplot(1, 1, 1)
    rendered_trial_types = _ordered_rendered_trial_types(context=context)

    unique_zones = _collect_unique_reward_zones(context=context)
    _draw_trial_blocks(axis=ax_main, context=context, unique_zones=unique_zones)
    # Draws the cue reference strip directly onto the main axis using a blended transform with
    # ``y`` in axes-fraction coordinates above 1.0 and ``clip_on=False``. A separate top-axis
    # gridspec slot was tried first, but ``constrained_layout`` enforces a residual ~3.6%-of-figure
    # gap between adjacent slots that no public knob (``hspace``, ``h_pad``, etc.) collapses to
    # zero. Drawing the cue rectangles on the main axis itself makes them sit at exactly the data
    # axis's top spine, which is what the user wants.
    _draw_cue_blocks_panel(
        axis=ax_main,
        context=context,
        rendered_trial_types=rendered_trial_types,
        cue_textures=cue_textures,
    )

    ax_main.scatter(
        context.lick_positions,
        context.lick_trials,
        s=lick_marker_area,
        color=_COLOR_LICK,
        alpha=0.55,
        edgecolor="none",
        zorder=2,
    )
    for boundary in context.session_boundaries[1:-1]:
        ax_main.axhline(boundary, color="gray", linestyle="--", linewidth=0.6, zorder=1)

    ax_main.set_xlim(0, context.track_length_cm)
    # Inverts the y-axis so the smallest trial index (most negative when anchored, otherwise zero)
    # sits at the top of the plot and the largest sits at the bottom. Pre-anchoring this collapses
    # to ``set_ylim(total_trials, 0)`` because ``session_boundaries[0] == 0``.
    ax_main.set_ylim(context.session_boundaries[-1], context.session_boundaries[0])
    # Hides the top spine and top tick marks so the cue rectangles drawn at axes-fraction y=1.0
    # sit on a clean edge rather than on top of a black spine line and stray ticks.
    ax_main.spines["top"].set_visible(False)
    ax_main.tick_params(top=False)
    ax_main.set_xlabel("Position (cm)", fontsize=axis_label_fontsize)
    session_count = max(len(context.session_boundaries) - 1, 0)
    if trial_origin_session is not None:
        # Surfaces the origin in the y-axis label rather than the title so the negative-vs-positive
        # trial axis explains itself; the title stays free of per-figure parameters and the secondary
        # y-axis ``Session X (Day Y)`` ticks already document the per-session identities.
        y_axis_label = f"Trial number (since session {int(trial_origin_session)} onset)"
    elif session_count > 1:
        y_axis_label = "Trial number (chronological across all sessions)"
    else:
        y_axis_label = "Trial number"
    ax_main.set_ylabel(y_axis_label, fontsize=axis_label_fontsize)
    ax_main.tick_params(axis="x", labelsize=tick_fontsize)

    # Re-labels the trial-axis ticks at every session boundary so the displayed numbers reflect the
    # *uncapped* cumulative trial count even when ``max_trials_per_session`` clipped the rendered
    # band. The tick positions live on the compact display axis (``session_boundaries``) but the
    # labels come from ``true_session_boundaries``, so the figure stays free of cap-induced
    # white-space stripes while still telling the reader where each session sits in true count
    # space (e.g., session 17's first trial labeled with the full count of session 16, not the
    # 30-trial clip).
    _draw_true_count_yticks(ax_main=ax_main, context=context, fontsize=tick_fontsize)

    # Renders the per-session ``Session X (Day Y)`` labels on a secondary y-axis pinned to each
    # session's trial-midpoint. ``constrained_layout`` accounts for the secondary axis's tick-label
    # width; the main legend below is anchored further right so it does not collide.
    _draw_session_side_labels(ax_main=ax_main, context=context, fontsize=tick_fontsize)

    has_absent_track = any(
        block.trial_length_cm < context.track_length_cm
        for block in context.trial_blocks
    )
    legend_handles: list = [
        plt.Line2D(
            [0], [0],
            marker="o", color="none", markerfacecolor=_COLOR_LICK,
            markersize=legend_marker_size, label="Lick events",
        ),
        Patch(facecolor=_COLOR_REWARDED, edgecolor="black", linewidth=0.6, label="Rewarded location"),
        Patch(facecolor=_COLOR_NON_REWARDED, edgecolor="black", linewidth=0.6, label="Non-rewarded location"),
    ]
    # Only advertises the absent-track patch when at least one block actually triggers the hatched
    # mask in `_draw_trial_blocks`; otherwise the legend documents an artifact the figure does not show.
    if has_absent_track:
        legend_handles.append(
            Patch(
                facecolor=_COLOR_ABSENT_TRACK, edgecolor="#888888", linewidth=0.6, hatch="///",
                label="Absent track section",
            )
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.7, label="Session separator")
    )
    # Anchors the legend to the right outer edge of the plot, just above the topmost secondary-axis
    # tick label. The first retained session occupies ``[boundaries[0], boundaries[1]]`` in
    # trial-axis data coords; its midpoint is where the topmost ``Session X (Day Y)`` label hangs.
    # Converting that midpoint to axes-fraction (the ylim is inverted, so the smallest data-y maps
    # to ``axes_fraction_y == 1``) gives the y-coordinate the legend's bottom edge should sit just
    # above; ``loc="lower left"`` plus ``bbox_to_anchor=(1.0, ...)`` makes the legend hug the right
    # outer spine and grow upward into the figure margin without overlapping the data area.
    boundaries = context.session_boundaries
    if session_count > 0 and boundaries[-1] != boundaries[0]:
        topmost_label_data_y = 0.5 * (boundaries[0] + boundaries[1])
        topmost_label_axes_y = (
            (boundaries[-1] - topmost_label_data_y) / (boundaries[-1] - boundaries[0])
        )
        legend_anchor_y = min(topmost_label_axes_y + 0.04, 1.0)
    else:
        legend_anchor_y = 1.0

    ax_main.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(1.0, legend_anchor_y),
        borderaxespad=0.0,
        frameon=False,
        fontsize=legend_fontsize,
    )
    session_label = "session" if session_count == 1 else "sessions"
    # Uses ``ax_main.set_title`` rather than ``figure.suptitle`` so the title centers above the
    # plot rectangle alone; ``suptitle`` centers across the full figure width, which includes the
    # legend column anchored outside the axes and shifts the text off-center over the data.
    # ``set_title``'s default ``pad`` is measured from the axis spine, which would land the title
    # inside the cue strip drawn at ``y_axes > 1.0``; an explicit ``y`` just above the strip's top
    # gives constrained_layout enough headroom to place the title at the same figure-coord
    # position ``figure.suptitle`` used (window ≈ 0.97-0.99) without leaving extra whitespace
    # between the strip and the title.
    n_rows = max(len(rendered_trial_types), 1)
    title_y_axes = 1.0 + n_rows * 0.025 + 0.01
    # The origin reference now lives in the y-axis label (``Trial number (since session X
    # onset)``) so the title stays clean and consistent across origin / no-origin variants.
    title_description = f"lick events across {session_label}"
    ax_main.set_title(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=title_description,
        ),
        fontsize=title_fontsize,
        y=title_y_axes,
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
    cue_textures: dict[int, NDArray] | None = None,
) -> None:
    """Renders the cue reference strip directly above the main scatter axis.

    Notes:
        Each row reads as a horizontal sequence of rectangles spanning the full track length, one
        per `CueSpan` in the trial type's layout. The strip is drawn on the main axis using a
        blended transform (``x`` in data coordinates, ``y`` in axes-fraction coordinates) with
        ``clip_on=False``, so the rectangles render just above the axis's top spine with no
        inter-axis gap. The bottom-most row sits at axes-fraction ``y = 1.0`` (flush with the data
        area's top); additional rows stack upward.

        When ``cue_textures`` provides an entry for a span's ``code``, the matching image is
        rendered (stretched to the rectangle extent via ``imshow`` with ``aspect='auto'``) as the
        rectangle fill, so the strip mirrors the wall textures the animal actually saw. Codes
        without a texture entry fall back to the per-code solid color. A black border is drawn on
        top in either branch. ``constrained_layout`` reads the patches' tight bounding box (which
        extends above ``y_axes = 1.0``), so it auto-reserves room between the suptitle and the
        axis to fit the strip.

    Args:
        axis: The main scatter axis the strip overlays.
        context: The per-animal lick aggregate whose cue layouts and track length drive the strip.
        rendered_trial_types: Trial types to allocate rows for, in display order.
        cue_textures: Optional mapping from cue code to image array. When provided, codes present
            in the mapping render as textured rectangles; missing codes render as colored fills.
    """
    n_rows = max(len(rendered_trial_types), 1)
    row_height_axes = 0.025
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    for row_index, trial_type_name in enumerate(rendered_trial_types):
        layout = context.cue_layouts.get(trial_type_name, ())
        # Bottom-most row (last chronologically) sits at ``y_axes = 1.0`` so its bottom edge lines
        # up with the axis's top spine; additional trial-type rows stack upward.
        rect_y = 1.0 + (n_rows - 1 - row_index) * row_height_axes
        rect_height = row_height_axes
        for span in layout:
            width = span.end_cm - span.start_cm
            if width <= 0:
                continue
            texture = cue_textures.get(span.code) if cue_textures is not None else None
            if texture is not None:
                # ``imshow`` honours ``transform`` so ``extent`` is interpreted in the blended
                # (data x, axes y) frame just like the patches around it.
                axis.imshow(
                    texture,
                    extent=(span.start_cm, span.end_cm, rect_y, rect_y + rect_height),
                    transform=transform,
                    aspect="auto",
                    interpolation="nearest",
                    clip_on=False,
                    zorder=2,
                )
                fill = "none"
            else:
                fill = _cue_color(code=span.code)
            axis.add_patch(
                Rectangle(
                    (span.start_cm, rect_y),
                    width,
                    rect_height,
                    facecolor=fill,
                    edgecolor="black",
                    linewidth=0.5,
                    transform=transform,
                    # Cue rectangles span the full track edge-to-edge and sit above the axis box,
                    # so the default clip box would trim them. ``clip_on=False`` keeps the full
                    # rectangle (and its border on every side) visible.
                    clip_on=False,
                    zorder=3,
                )
            )


def load_cue_textures(
    *,
    experiment_config_path: Path,
    textures_dir: Path,
) -> dict[int, NDArray]:
    """Builds the ``cue_textures`` mapping consumed by `plot_lick_scatter` from a session's
    experiment configuration and the Unity textures directory.

    Notes:
        Reads the ``cues:`` block of the experiment configuration YAML produced by the upstream
        Mesoscope-VR pipeline (one file per session, written under ``<session>/raw_data/`` before
        forging strips raw data). Each cue carries a uint8 ``code`` and a ``texture`` filename; the
        loader resolves each filename against ``textures_dir`` (typically the
        ``Assets/InfiniteCorridorTask/Textures`` directory of ``sollertia-unity-tasks``) and
        ``matplotlib.image.imread``s the PNG into a numpy array. Cues whose ``texture`` is empty
        (legacy templates predating the field) or whose file is missing are skipped silently so
        the caller can pass partial mappings to `plot_lick_scatter`.

        Cue codes are template-level, so one mapping covers every session in a context whose
        sessions share the same experiment configuration. When animals span configurations, build
        one mapping per group and pass the matching one to each ``plot_lick_scatter`` call.

    Args:
        experiment_config_path: Path to a session's ``experiment_configuration.yaml``.
        textures_dir: Directory containing the Unity texture PNGs referenced by the configuration.

    Returns:
        Mapping from cue uint8 code to an image array as returned by ``matplotlib.image.imread``
        (HxW for grayscale, HxWx3 for RGB, HxWx4 for RGBA).
    """
    with experiment_config_path.open("r") as handle:
        configuration = yaml.safe_load(handle)
    textures: dict[int, NDArray] = {}
    for cue_entry in configuration.get("cues", ()):
        code = int(cue_entry["code"])
        filename = cue_entry.get("texture", "")
        if not filename:
            continue
        texture_path = textures_dir / filename
        if not texture_path.exists():
            continue
        textures[code] = imread(str(texture_path))
    return textures


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


def _cap_lick_context_trials(
    context: LickContext,
    max_trials_per_session: int,
) -> LickContext:
    """Returns a new `LickContext` where each session is downsampled into at most
    ``max_trials_per_session`` compact display rows, with every source trial mapped into one of
    those rows so its lick events still render.

    Notes:
        Two boundary tuples diverge here. ``session_boundaries`` is renumbered into a contiguous
        ``0..N`` display range so the rendered figure stays compact (no white-space stripes between
        the kept band and the next session's start). ``true_session_boundaries`` passes through
        unchanged so the rendering layer can still annotate the y-axis with each session's true
        cumulative trial count — i.e., the next session's tick label reflects the actual trial
        count of the previous session even though the cap collapses its band visually.

        Mapping is binning, not subsampling: a session of ``N`` source trials is uniformly mapped
        into ``kept_count = min(N, max_trials_per_session)`` compact rows by ``floor(i * kept /
        N)`` for each source trial ``i`` (0-based within the session). Every source trial's lick
        events are routed into one of the kept rows, so no data is dropped and no row is empty
        purely because the cap "skipped over" it. Sessions shorter than ``max_trials_per_session``
        degenerate to a one-source-trial-per-row identity mapping (no compression).

        Trial blocks remain contiguous in the compact display axis because the binning preserves
        chronological order; each block's clipped extent is the min/max of its source trials' new
        compact bin indices. ``session_indices`` and ``day_offsets`` pass through verbatim because
        the session set is unchanged.

    Args:
        context: The lick context to cap.
        max_trials_per_session: Maximum number of compact rows per session. Values <= 0 collapse
            the context to an empty result; the caller is responsible for filtering ``None`` out
            before delegating to this helper.

    Returns:
        A LickContext with the per-session cap applied: ``session_boundaries`` renumbered into a
        compact axis, ``true_session_boundaries`` left at the uncapped cumulative counts.
    """
    if max_trials_per_session <= 0:
        return _empty_filtered_lick_context(context=context)

    boundaries = context.session_boundaries
    n_sessions = max(len(boundaries) - 1, 0)
    if n_sessions == 0:
        return context

    total_old = boundaries[-1] if boundaries else 0
    new_boundaries: list[int] = [0]
    # Maps each old trial index to its compact bin in the new contiguous display axis. Unlike a
    # subsampling cap (which leaves most entries at -1), every source trial gets a valid bin so
    # its lick events render rather than being dropped purely because the cap "skipped over" them.
    # noinspection PyTypeChecker
    old_to_new_trial: NDArray[np.int64] = np.full(total_old, -1, dtype=np.int64)
    for old_idx in range(n_sessions):
        old_start = boundaries[old_idx]
        old_end = boundaries[old_idx + 1]
        original_count = old_end - old_start
        kept_count = min(original_count, max_trials_per_session)
        new_offset = new_boundaries[-1]
        new_boundaries.append(new_offset + kept_count)
        if kept_count > 0:
            # Bin every source trial uniformly into ``kept_count`` slots. ``within_session_idx`` is
            # the trial's 0-based offset within the session; integer division with truncation maps
            # consecutive source trials into the same bin until the bin advances. The clip guards
            # against the edge case ``i == original_count`` for tiny sessions where rounding could
            # otherwise overshoot ``kept_count - 1``.
            # noinspection PyTypeChecker
            within_session_idx: NDArray[np.int64] = np.arange(original_count, dtype=np.int64)
            # noinspection PyTypeChecker
            bin_idx: NDArray[np.int64] = (within_session_idx * kept_count) // original_count
            # noinspection PyTypeChecker
            bin_idx = np.clip(bin_idx, 0, kept_count - 1)
            old_to_new_trial[old_start:old_end] = bin_idx + new_offset

    new_blocks: list[TrialBlock] = []
    for block in context.trial_blocks:
        # noinspection PyTypeChecker
        retained: NDArray[np.int64] = old_to_new_trial[block.cum_trial_start:block.cum_trial_end]
        # noinspection PyTypeChecker
        valid: NDArray[np.int64] = retained[retained >= 0]
        if valid.size == 0:
            continue
        new_blocks.append(
            TrialBlock(
                cum_trial_start=int(valid[0]),
                cum_trial_end=int(valid[-1]) + 1,
                session_index=block.session_index,
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

    return LickContext(
        lick_positions=new_lick_positions,
        lick_trials=new_lick_trials,
        trial_blocks=tuple(new_blocks),
        session_boundaries=tuple(new_boundaries),
        true_session_boundaries=context.true_session_boundaries,
        track_length_cm=context.track_length_cm,
        cue_layouts=context.cue_layouts,
        session_indices=context.session_indices,
        day_offsets=context.day_offsets,
    )


def _anchor_lick_context_trials(
    context: LickContext,
    trial_origin_session: int,
    *,
    true_origin_position: int | None = None,
) -> LickContext:
    """Returns a new `LickContext` shifted so the first trial of ``trial_origin_session`` is at
    trial-axis index ``0``.

    Notes:
        ``trial_origin_session`` is 1-indexed against the original animal-level chronological
        ordering — the same convention as ``LickContext.session_indices`` and ``display_sessions``.
        Two axes are anchored independently:

        * **Display axis** (``session_boundaries``): shifted so the first trial of origin (or the
          boundary that stands in for origin when it was filtered out) sits at compact y == 0.
          Trials of earlier retained sessions render with negative compact y; later retained
          sessions render with non-negative compact y.
        * **True axis** (``true_session_boundaries``): shifted by ``-true_origin_position`` so the
          tick label "0" lands at the start of origin in the animal's full cumulative trial space.
          When the caller supplies ``true_origin_position`` (typically resolved from the pre-filter
          context via `_resolve_origin_true_position`), the labels respect inter-session gaps even
          when origin itself isn't displayed — a critical case when the user displays sessions
          (5, 10, 15) referenced to session 1 and expects the tick labels to grow with the four
          unrendered sessions between every retained pair.

        The fallback path (``true_origin_position is None``) recovers the old retained-only
        behavior and is retained for symmetry with the display computation; it is exercised when
        the caller doesn't have access to the full pre-filter context and accepts a relative
        labeling that hides skipped sessions.

    Args:
        context: The lick context to anchor. Typically already filtered via
            `_filter_context_to_sessions`.
        trial_origin_session: 1-indexed session number whose first trial becomes index ``0``.
        true_origin_position: Origin's TRUE cumulative trial count resolved against the
            pre-filter (full-animal) context, or ``None`` to fall back to retained-only
            cumulation. Pass the result of `_resolve_origin_true_position` to honor inter-session
            gaps.

    Returns:
        A new LickContext with shifted trial indices. Returns the input unchanged when both shifts
        evaluate to zero.
    """
    boundaries = context.session_boundaries
    true_boundaries = context.true_session_boundaries
    if not boundaries:
        return context

    # Walk retained sessions in chronological order; ``display_cum_before`` tracks the running
    # cumulative-trial index up to (but not including) the first retained session whose original
    # session number is >= origin. The display anchor uses retained cumulative because the display
    # axis only spans rendered sessions; the cap-induced compact axis would inflate by skipped
    # sessions otherwise.
    display_cum_before = 0
    for i, session_number in enumerate(context.session_indices):
        if session_number < trial_origin_session:
            display_cum_before = boundaries[i + 1]
        else:
            break

    display_shift = -display_cum_before
    if true_origin_position is not None:
        true_shift = -int(true_origin_position)
    else:
        # Fallback: retained-only cumulation. Equivalent to the pre-bugfix behavior; emits labels
        # that hide the inter-session gap when origin sits outside the retained set.
        retained_true_cum_before = 0
        for i, session_number in enumerate(context.session_indices):
            if session_number < trial_origin_session:
                if true_boundaries:
                    retained_true_cum_before = true_boundaries[i + 1]
            else:
                break
        true_shift = -retained_true_cum_before

    if display_shift == 0 and true_shift == 0:
        return context

    new_boundaries = tuple(b + display_shift for b in boundaries)
    new_true_boundaries = (
        tuple(b + true_shift for b in true_boundaries) if true_boundaries else true_boundaries
    )
    new_blocks = tuple(
        TrialBlock(
            cum_trial_start=block.cum_trial_start + display_shift,
            cum_trial_end=block.cum_trial_end + display_shift,
            session_index=block.session_index,
            trial_type=block.trial_type,
            trial_length_cm=block.trial_length_cm,
            reward_lo=block.reward_lo,
            reward_hi=block.reward_hi,
        )
        for block in context.trial_blocks
    )
    if context.lick_trials.size > 0:
        # noinspection PyTypeChecker
        new_lick_trials: NDArray[np.int64] = context.lick_trials + display_shift
    else:
        new_lick_trials = context.lick_trials

    return LickContext(
        lick_positions=context.lick_positions,
        lick_trials=new_lick_trials,
        trial_blocks=new_blocks,
        session_boundaries=new_boundaries,
        true_session_boundaries=new_true_boundaries,
        track_length_cm=context.track_length_cm,
        cue_layouts=context.cue_layouts,
        session_indices=context.session_indices,
        day_offsets=context.day_offsets,
    )


def _resolve_origin_true_position(
    context: LickContext, trial_origin_session: int,
) -> int | None:
    """Returns the true cumulative trial count at the start of ``trial_origin_session``.

    Notes:
        Resolved against the **full** pre-filter context so the anchor can honor inter-session
        gaps even when the user filters out the origin session itself. The full animal context
        produced by `..lick_protocol.aggregate_lick_events` always lists every session in
        ``session_indices``, so the lookup is exact when the caller supplies the unfiltered
        context here. Returns ``None`` when ``true_session_boundaries`` is empty, signaling the
        anchor helper to fall back to its retained-only cumulation path. When origin sits outside
        the context's session range, the helper picks the conceptual position (start of next
        session, or end of last session) so the label still reads coherently.

    Args:
        context: The lick context whose ``true_session_boundaries`` are queried. Must be the
            pre-filter (full-animal) context for accurate gap accounting.
        trial_origin_session: 1-indexed session number whose true start is being resolved.

    Returns:
        The true cumulative trial count at origin's start, or ``None`` when the context lacks
        ``true_session_boundaries``.
    """
    true_boundaries = context.true_session_boundaries
    if not true_boundaries:
        return None
    if trial_origin_session in context.session_indices:
        idx = context.session_indices.index(trial_origin_session)
        return int(true_boundaries[idx])
    # Origin not enumerated in this context (unusual; aggregate_lick_events normally covers every
    # animal session). Fall back to the conceptual position: the start of the first retained
    # session whose number exceeds origin, or the end of the last retained session if origin lies
    # past the entire context.
    for i, session_number in enumerate(context.session_indices):
        if session_number > trial_origin_session:
            return int(true_boundaries[i])
    return int(true_boundaries[-1])


def _empty_filtered_lick_context(context: LickContext) -> LickContext:
    """Returns an empty `LickContext` preserving ``context.track_length_cm``."""
    # noinspection PyTypeChecker
    empty_positions: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
    # noinspection PyTypeChecker
    empty_trials: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
    # noinspection PyTypeChecker
    empty_days: NDArray[np.int32] = np.zeros(0, dtype=np.int32)
    return LickContext(
        lick_positions=empty_positions,
        lick_trials=empty_trials,
        trial_blocks=(),
        session_boundaries=(0,),
        true_session_boundaries=(0,),
        track_length_cm=context.track_length_cm,
        cue_layouts={},
        session_indices=(),
        day_offsets=empty_days,
    )


def _draw_true_count_yticks(
    ax_main: plt.Axes, context: LickContext, *, fontsize: float = 9.0,
) -> None:
    """Re-bins the trial-axis ticks to label every session boundary with its uncapped cumulative
    trial count.

    Notes:
        The cap collapses each session's empty tail into a compact display axis to keep the figure
        free of white-space stripes; ``true_session_boundaries`` carries the uncapped cumulative
        counts so we can still surface them as the trial-number labels. Ticks are placed at
        ``session_boundaries[i]`` (display positions) while labels read ``true_session_boundaries[i]``,
        so the y-axis reads as "trial number" with discontinuities at session boundaries (a session
        labeled e.g. 100 → 130 visually but 0 → 100 in true cumulative terms when capped at 30).

        Falls back to default matplotlib ticking when ``true_session_boundaries`` is empty (legacy
        contexts built before this field existed) so the figure still renders without crashing.

    Args:
        ax_main: The trial-axis axes whose ticks are being relabeled.
        context: The lick context whose dual boundary tuples drive the tick positions and labels.
        fontsize: Tick-label size in points. Pinned by the caller to ``font_scale`` so labels
            stay legible across figure heights — Jupyter rescales taller figures down to fit the
            cell width, so a height-proportional fontsize keeps the rendered text size constant
            relative to the data area.
    """
    boundaries = context.session_boundaries
    true_boundaries = context.true_session_boundaries
    if not boundaries or len(boundaries) != len(true_boundaries):
        return
    ax_main.set_yticks(list(boundaries))
    ax_main.set_yticklabels([str(int(value)) for value in true_boundaries], fontsize=fontsize)


def _draw_session_side_labels(
    ax_main: plt.Axes, context: LickContext, *, fontsize: float = 9.0,
) -> None:
    """Draws ``Session X (Day Y)`` labels on a secondary y-axis pinned to each session's midpoint.

    Notes:
        The secondary axis (``twinx``) shares the data y-range with ``ax_main`` so the labels track
        the trial axis even when an origin anchor or a per-session cap shifts the boundaries. Tick
        marks and spines on the secondary axis are hidden so only the text floats next to each
        session's trial band. The day suffix is omitted for sessions whose timestamp could not be
        parsed at aggregation time (``day_offsets`` falls back to chronological index in that case;
        the suffix still renders, just with the index value, which is the same convention used by
        `..outcome_protocol.aggregate_trial_outcomes`).

    Args:
        ax_main: The trial-axis axes whose secondary y-axis carries the labels.
        context: The lick context whose ``session_boundaries`` drive the midpoint positions and
            ``session_indices`` / ``day_offsets`` drive the per-session label text.
        fontsize: Side-label size in points. Pinned by the caller to ``font_scale`` so labels
            stay legible across figure heights — Jupyter rescales taller figures down to fit the
            cell width, so a height-proportional fontsize keeps the rendered text size constant
            relative to the data area.
    """
    boundaries = context.session_boundaries
    session_count = max(len(boundaries) - 1, 0)
    if session_count == 0:
        return
    midpoints: list[float] = [
        0.5 * (boundaries[i] + boundaries[i + 1]) for i in range(session_count)
    ]
    has_day_offsets = (
        context.day_offsets.size == session_count and context.day_offsets.size > 0
    )
    labels: list[str] = []
    for i in range(session_count):
        session_number = (
            int(context.session_indices[i]) if i < len(context.session_indices)
            else (i + 1)
        )
        if has_day_offsets:
            labels.append(f"Session {session_number} (Day {int(context.day_offsets[i])})")
        else:
            labels.append(f"Session {session_number}")

    ax_right = ax_main.twinx()
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks(midpoints)
    ax_right.set_yticklabels(labels, fontsize=fontsize)
    # Hide every spine and the tick marks so only the labels float next to each session block.
    for spine_name in ("top", "right", "bottom", "left"):
        ax_right.spines[spine_name].set_visible(False)
    ax_right.tick_params(
        axis="y", which="both", left=False, right=False, length=0, pad=4,
    )
    # Match the main axis's top-tick suppression so a stray tick from the shared x-axis doesn't
    # poke above the cue strip.
    ax_right.tick_params(top=False)


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
        can keep its session-boundary logic unchanged. ``session_indices`` and ``day_offsets`` are
        sliced in lockstep so each retained session keeps its original 1-based number and absolute
        day offset rather than being renumbered against the filter's first entry.

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
        return _empty_filtered_lick_context(context=context)

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

    if context.day_offsets.size == len(context.session_indices):
        # noinspection PyTypeChecker
        new_day_offsets: NDArray[np.int32] = context.day_offsets[
            np.asarray(resolved_old_indices, dtype=np.int64)
        ]
    else:
        # noinspection PyTypeChecker
        new_day_offsets = np.zeros(0, dtype=np.int32)

    # Preserves the absolute true cumulative positions of every retained session so the post-filter
    # tick labels can show the gap to non-retained sessions faithfully. ``new_true_boundaries[i]``
    # for ``i in [0, n_retained)`` carries the true cumulative trial count at the *start* of the
    # i-th retained session in the animal's full chronological sequence; the trailing entry holds
    # the true end of the last retained session. With contiguous retained sessions this collapses
    # to the same shape as ``session_boundaries`` (each pair touches); with non-contiguous retained
    # sessions, neighbouring entries differ by the sum of skipped sessions' trial counts plus the
    # rendered session's own count, so the rendered tick labels reveal the inter-session gap.
    if context.true_session_boundaries and len(context.true_session_boundaries) - 1 == len(context.session_indices):
        new_true_boundaries: list[int] = [
            int(context.true_session_boundaries[old_idx]) for old_idx in resolved_old_indices
        ]
        new_true_boundaries.append(
            int(context.true_session_boundaries[resolved_old_indices[-1] + 1]),
        )
        new_true_boundaries_tuple: tuple[int, ...] = tuple(new_true_boundaries)
    else:
        new_true_boundaries_tuple = ()

    return LickContext(
        lick_positions=new_lick_positions,
        lick_trials=new_lick_trials,
        trial_blocks=tuple(new_blocks),
        session_boundaries=tuple(new_boundaries),
        true_session_boundaries=new_true_boundaries_tuple,
        track_length_cm=context.track_length_cm,
        cue_layouts=new_cue_layouts,
        session_indices=tuple(context.session_indices[i] for i in resolved_old_indices),
        day_offsets=new_day_offsets,
    )


def plot_trial_outcomes(
    context: TrialOutcomeContext,
    *,
    display_sessions: tuple[int, ...] | None = None,
    animal_id: str | None = None,
    figure_dpi: int = 150,
    figure_preset: str = "print",
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
        figure_dpi: Output figure DPI. Threaded through to ``matplotlib.figure.Figure`` and reused
            for the placeholder figure rendered when the context contains no sessions.
        figure_preset: ``"print"`` (default) or ``"presentation"``. Resolves the per-element font
            sizes through `..shared_utilities.resolve_figure_style` so this figure stays
            consistent with every other plotter in the analysis package.

    Returns:
        A matplotlib Figure showing the stacked success / failure / guided counts.
    """
    style = resolve_figure_style(preset=figure_preset)
    if display_sessions is not None:
        context = _filter_outcome_context_to_sessions(
            context=context, display_sessions=display_sessions,
        )
    if context.n_sessions == 0:
        figure, axis = plt.subplots(
            1, 1, figsize=(7, 4), facecolor="white", dpi=figure_dpi, layout="constrained",
        )
        axis.text(0.5, 0.5, "no sessions", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.suptitle(_title_with_animal_prefix(animal_id=animal_id, description="trial outcomes"))
        return figure

    width_scale = resolve_figure_width_scale(style=style)
    figure, ax_counts = plt.subplots(
        1, 1,
        figsize=(max(7.0, 0.5 * context.n_sessions + 4.0) * width_scale, 4.0),
        facecolor="white", dpi=figure_dpi, layout="constrained",
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

    ax_counts.set_xlabel("Days since first session", fontsize=style.axis_label_fontsize)
    ax_counts.set_ylabel("Trial count", fontsize=style.axis_label_fontsize)
    ax_counts.set_xticks(x_positions)
    ax_counts.set_xticklabels(
        [str(int(d)) for d in context.day_offsets], fontsize=style.tick_label_fontsize,
    )
    ax_counts.tick_params(axis="y", labelsize=style.tick_label_fontsize)

    legend_handles = [
        Patch(facecolor=_COLOR_SUCCESS, edgecolor="black", linewidth=0.4, label="Success"),
        Patch(facecolor=_COLOR_FAILURE, edgecolor="black", linewidth=0.4, label="Failure"),
        Patch(facecolor=_COLOR_GUIDED, edgecolor="black", linewidth=0.4, label="Guided"),
    ]
    ax_counts.legend(
        handles=legend_handles, loc="upper left", frameon=False, fontsize=style.legend_fontsize,
    )

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
        fontsize=style.suptitle_fontsize,
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
