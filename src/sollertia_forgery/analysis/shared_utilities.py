"""Cross-package utilities shared by the bleaching, SCE, and tuning analysis pipelines.

Currently exposes the acquisition-warmup trimming helper (every pipeline drops the same leading window), the
session-day display-unit resolver (cross-session aggregates label x-axes the same way regardless of which
modality they aggregate), the (animal, session) selection resolver consumed by the per-modality
orchestrators, and the cross-package figure-style preset (print vs presentation) every plotting helper
honors so a notebook can set legibility once and have every panel render at consistent text sizes.
Pipeline-specific helpers live in their per-package ``utilities`` modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import console

from ..shared_assets import DatasetColumn

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

    from ..shared_assets import DatasetData, DatasetSession


@dataclass(frozen=True, slots=True)
class FigureStyle:
    """Per-element font sizes shared across every analysis-package plotter.

    Notes:
        The fields enumerate the canonical roles every plotter needs (figure title / panel title /
        axis labels / tick labels / legend / in-figure annotations). Plotters resolve this via
        `resolve_figure_style` based on a string preset name and apply the fields to their
        ``set_title`` / ``set_xlabel`` / ``set_ylabel`` / ``tick_params`` / ``legend`` / inline
        ``axes.text`` calls. The lick scatter additionally multiplies these baseline sizes by its
        height-proportional ``font_scale`` so taller figures remain legible after Jupyter rescales
        them; other figures use the values verbatim.

        The print preset matches the historical sizes (12 / 10 / 9 / 9 / 9 / 8) used by
        `..tuning.plotting.plot_sorted_heatmap` and the lick scatter. The presentation preset is
        sized so the figure stays legible from the back of a typical conference room
        (>=18 pt floor); a doubling of the print baseline is a reasonable default.
    """

    suptitle_fontsize: float
    """Figure-level title size (``figure.suptitle`` / ``axes.set_title`` when used as the only
    text above the plot)."""
    panel_title_fontsize: float
    """Per-panel title size for multi-panel figures (each subplot's ``axes.set_title``)."""
    axis_label_fontsize: float
    """X- and y-axis label size (``axes.set_xlabel`` / ``axes.set_ylabel``)."""
    tick_label_fontsize: float
    """Default tick-label size for both axes (``tick_params(labelsize=)``,
    ``set_xticklabels`` / ``set_yticklabels``)."""
    legend_fontsize: float
    """Legend entry size (``axes.legend(fontsize=)``)."""
    annotation_fontsize: float
    """Size for in-figure annotations and inline labels (rotated bin labels, ``axes.text``,
    bracket / ``ns`` glyphs)."""


_PRINT_FIGURE_STYLE: FigureStyle = FigureStyle(
    suptitle_fontsize=12.0,
    panel_title_fontsize=10.0,
    axis_label_fontsize=9.0,
    tick_label_fontsize=9.0,
    legend_fontsize=9.0,
    annotation_fontsize=8.0,
)
"""Default print preset. Matches the sizes used by ``plot_sorted_heatmap`` and the historical
behavior of every other analysis plotter; safe for figures embedded in publications."""

_PRESENTATION_FIGURE_STYLE: FigureStyle = FigureStyle(
    suptitle_fontsize=18.0,
    panel_title_fontsize=16.0,
    axis_label_fontsize=16.0,
    tick_label_fontsize=14.0,
    legend_fontsize=14.0,
    annotation_fontsize=12.0,
)
"""Conference-talk preset, sized to the practical floor for slide-projected figures: the smallest
in-figure text sits at 12 pt, body decorations (ticks / legend) at 14 pt, axis labels and panel
titles at 16 pt, and the figure title at 18 pt. This is the minimum legible-from-the-back set
rather than a comfortable one — bump the values via a custom `FigureStyle` if a venue is larger
than typical or if the audience density is high."""


def resolve_figure_style(preset: str = "print") -> FigureStyle:
    """Returns the `FigureStyle` for the requested preset.

    Notes:
        Two presets are supported out of the box: ``"print"`` (12 / 10 / 9 / 9 / 9 / 8 pt) and
        ``"presentation"`` (18 / 16 / 16 / 14 / 14 / 12 pt). Plotters in the analysis package
        accept ``figure_preset: str`` and call this resolver to pick consistent sizes; users who
        need custom values can construct a `FigureStyle` directly and bypass the resolver via the
        plotter's lower-level fontsize arguments.

    Args:
        preset: ``"print"`` (default) or ``"presentation"``. Other values raise ``ValueError``
            so a typo surfaces immediately rather than silently falling back.

    Returns:
        The `FigureStyle` instance carrying the per-role font sizes for the requested preset.

    Raises:
        ValueError: When ``preset`` is not one of the recognized names.
    """
    if preset == "presentation":
        return _PRESENTATION_FIGURE_STYLE
    if preset == "print":
        return _PRINT_FIGURE_STYLE
    message = (
        f"Unknown figure preset {preset!r}. Expected 'print' or 'presentation'; pass a custom "
        f"FigureStyle via the plotter's lower-level fontsize arguments if other sizes are needed."
    )
    console.error(message=message, error=ValueError)
    # Unreachable; ``console.error`` is NoReturn but ruff cannot trace it through method calls.
    # noinspection PyUnreachableCode
    raise ValueError(message)  # pragma: no cover


def resolve_figure_width_scale(style: FigureStyle) -> float:
    """Returns the figure-width multiplier matching the requested style's tick-label size.

    Notes:
        Bigger fonts in the presentation preset push every text-heavy margin (y-axis ticks,
        secondary axis labels, legend column, x-axis tick labels) wider, and ``constrained_layout``
        compensates by shrinking the data axes. Multiplying the print-baseline figure width by
        this scale at construction time gives ``constrained_layout`` enough headroom to keep the
        data axes at roughly its print-preset width regardless of how big the labels became.

        The factor interpolates halfway between 1.0 and the tick-label ratio (``tick_size /
        print_tick_size``) so a presentation tick of 14 pt vs a print tick of 9 pt yields
        ~1.28× width — bumped enough that the data area recovers, but not so much that the
        figure overflows a slide. ``1.0`` is returned exactly for the print preset so existing
        figure dimensions are unchanged.

    Args:
        style: The resolved `FigureStyle` whose tick-label size drives the scaling.

    Returns:
        Width multiplier in ``[1.0, ...)``; multiply your ``figsize`` width by this value when
        constructing the figure to keep the data axes from compressing under preset-bumped labels.
    """
    print_baseline_tick = _PRINT_FIGURE_STYLE.tick_label_fontsize
    return 1.0 + 0.5 * (style.tick_label_fontsize / print_baseline_tick - 1.0)


_ACQUISITION_WARMUP_SECONDS: float = 60.0
"""Number of leading seconds discarded from every loaded session trace before any analysis runs. Sollertia
experiments include a multi-minute pre-imaging baseline period during which the PMT gain, resonant scanner
phase, shutter, and laser power have not yet stabilized; the resulting initial fluorescence valley would
otherwise contaminate downstream estimates (per-cell baselines, within-session bleaching, SCE statistics,
place-field tuning). Trimming at load time guarantees every analyzer operates on stabilized samples without
needing to know the artifact exists."""


def realign_trial_starts_to_first_cue(cue: NDArray[np.uint8]) -> NDArray[np.int32]:
    """Re-anchors a per-sample trial index so each trial starts at the canonical first-cue transition.

    Notes:
        The runtime begins recording each trial when the animal is already mid-first-cue (offset by
        ``cue_offset_cm`` into the canonical cue sequence), so runtime trial boundaries do not align with
        cue boundaries. Analyses that need canonical cue-aligned trial boundaries — so the rate-map x-axis
        at position 0 corresponds to the canonical first-cue start — re-bin samples here.

        ``first_cue`` is the cue id present at the first sample of the supplied array (the first run-state
        sample of the trial type). Each subsequent sample where ``cue == first_cue`` AND the previous
        sample's ``cue != first_cue`` increments the trial index. The first realigned trial usually starts
        mid-cycle (the run-state subset begins partway through the first cycle) and the last realigned
        trial usually ends mid-cycle, so callers that need only complete cycles should drop incomplete
        trials downstream — ``compute_within_trial_position``'s completeness mask handles this naturally.

    Args:
        cue: The per-sample cue id with length sample_count. Already filtered to a single trial type and
            run-state samples.

    Returns:
        Per-sample re-anchored trial id with length sample_count, dtype np.int32.
    """
    if cue.size == 0:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.int32)
    first_cue = cue[0]
    # noinspection PyTypeChecker
    is_first_cue: NDArray[np.bool_] = cue == first_cue
    # noinspection PyTypeChecker
    prev_not_first_cue: NDArray[np.bool_] = np.empty(cue.size, dtype=np.bool_)
    prev_not_first_cue[0] = True
    prev_not_first_cue[1:] = ~is_first_cue[:-1]
    # noinspection PyTypeChecker
    new_trial_start: NDArray[np.bool_] = is_first_cue & prev_not_first_cue
    # noinspection PyTypeChecker
    trial_id: NDArray[np.int32] = np.cumsum(new_trial_start.astype(np.int32)).astype(np.int32) - np.int32(1)
    return trial_id


def trim_acquisition_warmup(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Drops the leading ``_ACQUISITION_WARMUP_SECONDS`` of samples from a session dataframe via ``time_us``.

    Notes:
        Operates on the polars dataframe directly (rather than the post-explode numpy arrays) so the warmup
        window never enters any subsequent column-level reshape. Sessions whose entire trace falls within the
        warmup window collapse to an empty dataframe; downstream loaders' existing length guards then produce
        NaN sentinels for such degenerate sessions.

    Args:
        dataframe: Session dataframe loaded from ``DatasetFiles.DATA``. Must include ``DatasetColumn.TIME_US``
            among the selected columns; all other columns are passed through untouched.

    Returns:
        The input dataframe sliced to drop every row whose ``time_us`` value precedes the warmup cutoff.
    """
    if dataframe.height == 0:
        return dataframe
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = dataframe[DatasetColumn.TIME_US.value].to_numpy()
    warmup_us = int(
        convert_time(
            time=_ACQUISITION_WARMUP_SECONDS,
            from_units=TimeUnits.SECOND,
            to_units=TimeUnits.MICROSECOND,
            as_float=True,
        )
    )
    cutoff_us = int(time_us[0]) + warmup_us
    warmup_index = int(np.searchsorted(a=time_us, v=cutoff_us, side="left"))
    if warmup_index <= 0:
        return dataframe
    return dataframe.slice(offset=warmup_index)


def resolve_display_units(days_since_first: NDArray[np.float32]) -> tuple[str, NDArray[np.int64]]:
    """Resolves the integer display unit and per-session tick array used by dataset-level summaries and plots.

    Notes:
        Returns ``("day", round(days_since_first))`` when every session's day-rounded offset is unique.
        Otherwise, falls back to ``("hour", round(days_since_first * 24))``. Storage and any cross-session fits
        continue to operate on float days; the integer ticks returned here are display-only.

    Args:
        days_since_first: Per-session day offsets relative to the first session.

    Returns:
        A tuple of unit label (``"day"`` or ``"hour"``) and an int64 tick array aligned with ``days_since_first``.

    Raises:
        ValueError: When sessions cannot be assigned unique day or hour ticks. Sollertia acquisition protocols
            mandate at least one hour between consecutive sessions, so the hour-rounded values are by
            construction distinct; a collision indicates a violated input invariant.
    """
    # noinspection PyTypeChecker
    rounded_days: NDArray[np.int64] = np.round(days_since_first).astype(np.int64, copy=False)
    if int(np.unique(rounded_days).size) == int(rounded_days.size):
        return "day", rounded_days

    # Promotes through float64 first so the *24 multiplication does not lose precision near the float32 boundary.
    # noinspection PyTypeChecker
    rounded_hours: NDArray[np.int64] = np.round(days_since_first.astype(np.float64) * 24.0).astype(np.int64, copy=False)
    if int(np.unique(rounded_hours).size) == int(rounded_hours.size):
        return "hour", rounded_hours

    message = (
        "Unable to assign unique integer day or hour labels to the supplied sessions. Sollertia acquisition "
        "protocols require at least one hour of separation between consecutive sessions, but at least two "
        "sessions in this set rounded to the same hour-since-first value, which violates that invariant."
    )
    console.error(message=message, error=ValueError)
    # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
    # noinspection PyUnreachableCode
    raise ValueError(message)  # pragma: no cover


def resolve_session_selection(
    dataset: DatasetData,
    *,
    animal: str | tuple[str, ...] | None,
    session: str | tuple[str, ...] | None,
) -> tuple[DatasetSession, ...]:
    """Resolves the (animal, session) filter pair into a chronologically ordered tuple of DatasetSession instances.

    Notes:
        ``animal`` and ``session`` accept None, a single string, or a tuple of strings; None disables that
        filter. The two filters compose as a logical AND: each returned session must (i) belong to one of the
        requested animals (or to any animal when ``animal`` is None), and (ii) match one of the requested
        session names (or any session name when ``session`` is None). The result is sorted by the canonical
        timestamp-based session name so chronological per-animal aggregations downstream do not need to re-sort.

    Args:
        dataset: The DatasetData whose sessions are filtered.
        animal: Animal identifier filter; None matches every animal.
        session: Session identifier filter; None matches every session.

    Returns:
        A tuple of DatasetSession instances matching the resolved filters, sorted by ``(animal, session)``.

    Raises:
        ValueError: When the animal or session filter references identifiers absent from the dataset, or when
            the resolved selection is empty.
    """
    requested_animals = _normalize_filter(value=animal)
    requested_sessions = _normalize_filter(value=session)

    available_animals = {dataset_animal.animal for dataset_animal in dataset.animals}
    if requested_animals is not None:
        unknown_animals = requested_animals - available_animals
        if unknown_animals:
            available = ", ".join(sorted(available_animals)) if available_animals else "<none>"
            message = (
                f"Unable to resolve session selection on dataset {dataset.name!r}. The animal filter "
                f"references unknown animals: {sorted(unknown_animals)}. Available animals: {available}."
            )
            console.error(message=message, error=ValueError)

    matched: list[DatasetSession] = []
    for dataset_session in dataset.sessions:
        if requested_animals is not None and dataset_session.animal not in requested_animals:
            continue
        if requested_sessions is not None and dataset_session.session not in requested_sessions:
            continue
        matched.append(dataset_session)

    if requested_sessions is not None:
        matched_session_names = {dataset_session.session for dataset_session in matched}
        missing_sessions = requested_sessions - matched_session_names
        if missing_sessions:
            scope = (
                f"animals {sorted(requested_animals)}" if requested_animals is not None else f"dataset {dataset.name!r}"
            )
            message = (
                f"Unable to resolve session selection. The session filter references session names not present "
                f"under {scope}: {sorted(missing_sessions)}."
            )
            console.error(message=message, error=ValueError)

    if not matched:
        message = (
            f"Unable to resolve session selection on dataset {dataset.name!r}. The (animal, session) filter pair "
            f"yielded zero matching sessions."
        )
        console.error(message=message, error=ValueError)

    matched.sort(key=lambda dataset_session: (dataset_session.animal, dataset_session.session))
    return tuple(matched)


def _normalize_filter(value: str | tuple[str, ...] | None) -> set[str] | None:
    """Normalizes a None / single-string / tuple-of-strings filter into either ``None`` or a set of strings.

    Args:
        value: The raw filter value as accepted by `resolve_session_selection`.

    Returns:
        ``None`` when ``value`` is ``None`` (filter disabled); otherwise a set holding every requested
        identifier.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)
