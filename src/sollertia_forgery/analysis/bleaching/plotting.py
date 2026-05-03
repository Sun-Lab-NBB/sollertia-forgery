"""Per-animal bleaching plots.

Module-level functions consume the cross-session ``BleachingReport`` produced by ``bleaching_analysis``.
Mirrors the plotting layout of ``..sce.plotting`` and ``..tuning.plotting`` so each analysis package keeps
a single file responsible for matplotlib output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from ..shared_utilities import resolve_display_units
from .bleaching_analysis import BleachingColumn, BleachingReport

if TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_baseline_trend(report: BleachingReport, *, animal_id: str | None = None) -> plt.Figure:
    """Plots the across-session baseline fluorescence trend with per-cell distributions.

    Renders per-session per-cell baseline distributions as boxplots and overlays the population-median
    trend that drives the bleaching protocol's exponential fit. The x-axis uses whole-day or whole-hour
    ticks, picked to match the spacing convention used by every other bleaching figure for the report.

    Args:
        report: The cross-session bleaching report whose table drives the plot.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.

    Returns:
        A matplotlib Figure showing the across-session baseline fluorescence trend.
    """
    table = report.table

    # Pulls the per-session arrays the figure needs: a chronological day index, the population-median
    # baseline trace (one scalar per session), and the per-cell baseline distribution per session.
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = (
        table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    population_baseline: NDArray[np.float32] = (
        table[BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value]
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    cell_baseline_distributions: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.CELL_BASELINE_FLUORESCENCE.value].to_list()
    ]
    cell_count: int = (
        len(cell_baseline_distributions[0]) if cell_baseline_distributions else 0
    )

    # Resolves display units (whole days vs. whole hours) so the x-axis ticks line up with the same
    # convention used by every other bleaching figure for this report.
    unit, ticks = resolve_display_units(days_since_first=days)

    # Scales the box width with the smallest tick spacing so densely-spaced sessions don't overlap and
    # widely-spaced ones don't render as thin slivers.
    minimum_tick_step: float = float(np.diff(ticks).min()) if len(ticks) > 1 else 1.0
    box_width: float = 0.4 * minimum_tick_step

    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    # Renders per-session per-cell distributions as boxplots; fliers are hidden because the long upper
    # tail otherwise dominates the y-axis and crushes the body of the distribution.
    axes.boxplot(
        cell_baseline_distributions,
        positions=ticks,
        widths=box_width,
        showfliers=False,
    )

    # Overlays the population-median trend on top of the boxplots; this is the trace the bleaching
    # protocol's exponential fit was computed against.
    axes.plot(
        ticks,
        population_baseline,
        marker="o",
        color="tab:blue",
        linewidth=1.5,
        label="Population median",
    )

    axes.set_xlabel(f"{unit.capitalize()}s since first session")
    axes.set_ylabel("Baseline fluorescence (a.u.)")
    axes.set_title(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=f"baseline fluorescence trend across {cell_count} registered cells",
        ),
        fontsize=10,
    )
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def plot_within_session(report: BleachingReport, *, animal_id: str | None = None) -> plt.Figure:
    """Plots the within-session FOV-mean baseline trace for each session as overlaid curves.

    Sessions are colored chronologically with viridis so the cool->warm gradient reads as time
    progressing through the report. Per-session legend labels carry the session day plus the
    fractional start-to-end drop so high-drop sessions are easy to spot.

    Args:
        report: The cross-session bleaching report whose per-session within-session traces drive the
            plot.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.

    Returns:
        A matplotlib Figure showing within-session bleaching.
    """
    table = report.table

    # Pulls the per-session inputs: a chronological day index for legend labels, the per-session
    # within-session time grid (seconds from session start) and FOV-mean baseline trace, and the
    # per-session fractional drop from start-to-end so the legend can call out high-drop sessions.
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = (
        table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    )
    time_seconds_list: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value].to_list()
    ]
    baseline_list: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_BASELINE.value].to_list()
    ]
    # noinspection PyTypeChecker
    drops: NDArray[np.float32] = (
        table[BleachingColumn.WITHIN_SESSION_FRACTIONAL_DROP.value]
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    session_count: int = len(days)

    # Resolves display units (days or hours) so the legend label per session matches the
    # across-session plots; the within-session x-axis itself stays in minutes.
    unit, ticks = resolve_display_units(days_since_first=days)
    unit_capitalized: str = unit.capitalize()

    # Wider canvas reserves room for the per-session legend that is anchored outside the right of the
    # axes so it does not occlude the traces.
    figure, axes = plt.subplots(1, 1, figsize=(9, 4), facecolor="white", dpi=150)

    # Colors sessions with viridis so the chronological ordering reads as a cool->warm gradient. The
    # max(..., 1) divisor guards the single-session case from a zero-division error.
    colormap = plt.get_cmap("viridis")
    for index in range(session_count):
        color = colormap(index / max(session_count - 1, 1))
        label = f"{unit_capitalized} {int(ticks[index])} (drop={drops[index]:.1%})"
        # Converts seconds to minutes so the x-axis is readable.
        axes.plot(
            time_seconds_list[index] / 60.0,
            baseline_list[index],
            color=color,
            linewidth=1.0,
            label=label,
        )

    axes.set_xlabel("Time within session (minutes)")
    axes.set_ylabel("FOV-mean baseline (a.u.)")
    axes.set_title(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=f"within-session bleaching across {session_count} sessions",
        ),
        fontsize=10,
    )

    # Anchors the legend outside the right edge so trace inspection is not obstructed when many
    # sessions accumulate.
    axes.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        frameon=False,
    )
    figure.tight_layout()
    return figure


def plot_within_session_average(
    report: BleachingReport,
    *,
    animal_id: str | None = None,
    sessions_to_highlight: tuple[int, ...] | None = None,
) -> plt.Figure:
    """Plots the across-session mean of the within-session FOV-mean baseline trace, or selected sessions.

    Notes:
        The gray translucent bundle of per-session traces is always drawn first. When
        ``sessions_to_highlight`` is ``None``, a bold across-session mean trace is overlaid on top
        (averaged across NaN-padded per-session arrays so each column drops sessions that ended
        earlier). When ``sessions_to_highlight`` carries a tuple of 0-based session indices, the
        function instead overlays only those sessions in viridis colors and skips the mean trace —
        useful for inspecting specific suspect sessions against the bundle. Out-of-range indices
        are silently dropped so the same selection can be applied across animals with different
        session counts.

    Args:
        report: The cross-session bleaching report whose per-session within-session traces drive the
            plot.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.
        sessions_to_highlight: Optional tuple of 0-based session indices that select which sessions
            to highlight in color over the gray bundle. When ``None``, the across-session mean trace
            is overlaid instead.

    Returns:
        A matplotlib Figure showing the within-session bleaching bundle plus the selected overlay.
    """
    table = report.table

    # Pulls the per-session within-session time grid + baseline trace; per-session bin counts can
    # vary, so we keep them as a Python list of arrays rather than stacking up front. The day index
    # is also pulled so highlighted sessions can be labelled with their session day.
    time_seconds_list: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value].to_list()
    ]
    baseline_list: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_BASELINE.value].to_list()
    ]
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = (
        table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    )
    unit, ticks = resolve_display_units(days_since_first=days)
    unit_capitalized: str = unit.capitalize()

    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    # Draws each per-session trace as a translucent gray curve first so any bold overlay sits on top
    # of the bundle.
    for time_seconds, baseline in zip(time_seconds_list, baseline_list, strict=True):
        axes.plot(
            time_seconds / 60.0,
            baseline,
            color="grey",
            alpha=0.3,
            linewidth=0.8,
        )

    if sessions_to_highlight is None:
        # Drops sessions whose within-session compute produced an empty bin set (degenerate or fully
        # trimmed by the warmup cutoff); they would bias the mean toward zero-length contributors.
        usable_baselines: list[NDArray[np.float32]] = [
            baseline for baseline in baseline_list if baseline.size > 0
        ]
        if usable_baselines:
            # NaN-pads each session out to the longest session's length so np.nanmean reduces along
            # the session axis without truncating the rightmost bins. Each column's mean drops
            # sessions that ended earlier, which is honest about the shrinking sample size at the
            # right edge.
            max_length: int = max(baseline.size for baseline in usable_baselines)
            # noinspection PyTypeChecker
            baseline_matrix: NDArray[np.float32] = np.full(
                (len(usable_baselines), max_length),
                np.nan,
                dtype=np.float32,
            )
            for index, baseline in enumerate(usable_baselines):
                baseline_matrix[index, : baseline.size] = baseline
            # noinspection PyTypeChecker
            mean_baseline: NDArray[np.float32] = (
                np.nanmean(baseline_matrix, axis=0).astype(np.float32, copy=False)
            )
            # Uses the longest session's time grid as the x-axis for the mean line, clipped to the
            # matrix width.
            longest_time = max(
                time_seconds_list, key=lambda candidate: candidate.size,
            )[:max_length]
            axes.plot(
                longest_time / 60.0,
                mean_baseline,
                color="black",
                linewidth=2.5,
                label="Across-session mean",
            )
            axes.legend(loc="upper right", fontsize=8, frameon=False)
        usable_count: int = len(usable_baselines)
        title_description: str = (
            f"within-session bleaching averaged across {usable_count} sessions"
            if usable_count
            else "within-session bleaching with no usable sessions"
        )
    else:
        # Deduplicates the requested indices, sorts them chronologically, and clips to the in-range
        # subset so the legend reflects what actually rendered.
        ordered_indices: tuple[int, ...] = tuple(sorted(set(sessions_to_highlight)))
        valid_indices: tuple[int, ...] = tuple(
            session_index
            for session_index in ordered_indices
            if 0 <= session_index < len(baseline_list)
        )
        # Highlights selected sessions with viridis so the chronological ordering reads as a
        # cool->warm gradient; the max(..., 1) divisor guards the single-selection case.
        colormap = plt.get_cmap("viridis")
        for ordinal, session_index in enumerate(valid_indices):
            color = colormap(ordinal / max(len(valid_indices) - 1, 1))
            label = f"{unit_capitalized} {int(ticks[session_index])} (idx {session_index})"
            axes.plot(
                time_seconds_list[session_index] / 60.0,
                baseline_list[session_index],
                color=color,
                linewidth=2.0,
                label=label,
            )
        if valid_indices:
            axes.legend(loc="upper right", fontsize=8, frameon=False)
        title_description = (
            f"within-session bleaching highlighting sessions {', '.join(str(i) for i in valid_indices)}"
            if valid_indices
            else "within-session bleaching with no in-range sessions selected"
        )

    axes.set_xlabel("Time within session (minutes)")
    axes.set_ylabel("FOV-mean baseline (a.u.)")
    axes.set_title(
        _title_with_animal_prefix(animal_id=animal_id, description=title_description),
        fontsize=10,
    )
    figure.tight_layout()
    return figure


def plot_snr_distributions(report: BleachingReport, *, animal_id: str | None = None) -> plt.Figure:
    """Plots per-session per-cell SNR distributions as violins with the population-median trend overlaid.

    Args:
        report: The cross-session bleaching report whose per-cell SNR arrays drive the plot.
        animal_id: Optional animal id embedded in the figure title; omitted when ``None``.

    Returns:
        A matplotlib Figure showing the SNR-vs-session distribution.
    """
    table = report.table

    # Pulls the per-session inputs: the chronological day index, the per-session per-cell SNR
    # distribution, and the population-median SNR trace overlaid on top of the violins.
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = (
        table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    )
    snr_data: list[NDArray[np.float32]] = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.CELL_SNR.value].to_list()
    ]
    # noinspection PyTypeChecker
    population_snr: NDArray[np.float32] = (
        table[BleachingColumn.POPULATION_SNR.value].to_numpy().astype(np.float32, copy=False)
    )
    session_count: int = len(days)

    # Resolves display ticks so the SNR violins line up on the same x-axis as the baseline boxplots.
    unit, ticks = resolve_display_units(days_since_first=days)

    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    # Renders per-session per-cell SNR violins with median bars; preferred over boxplots here because
    # the bimodal-ish SNR distribution is easier to read as a density.
    axes.violinplot(snr_data, positions=ticks, showmedians=True)

    # Overlays the population-median SNR trend on top of the violins so the chronic SNR drift reads
    # next to the per-session spread.
    axes.plot(
        ticks,
        population_snr,
        marker="o",
        color="tab:blue",
        linewidth=1.5,
        label="Population median",
    )

    axes.set_xlabel(f"{unit.capitalize()}s since first session")
    axes.set_ylabel("Per-cell SNR")
    axes.set_title(
        _title_with_animal_prefix(
            animal_id=animal_id,
            description=f"per-cell SNR distributions across {session_count} sessions",
        ),
        fontsize=10,
    )
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def _title_with_animal_prefix(*, animal_id: str | None, description: str) -> str:
    """Builds a one-sentence figure title.

    When ``animal_id`` is supplied, prepends ``"Animal {id} "`` so the result reads as a single
    sentence. When ``animal_id`` is ``None``, capitalises the first character of ``description`` so
    the standalone form still reads naturally.
    """
    if animal_id is not None:
        return f"Animal {animal_id} {description}"
    return description[0].upper() + description[1:] if description else ""
