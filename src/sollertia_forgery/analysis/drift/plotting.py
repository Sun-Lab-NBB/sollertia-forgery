"""Per-animal plots for the cross-session tuning-drift pipeline.

Module-level functions consume the per-animal `DriftReport` produced by `.drift_report` and emit
matplotlib figures. Mirrors the plotting layout of `..bleaching.plotting`, `..sce.plotting`, and
`..tuning.plotting` so each analysis package keeps a single file responsible for matplotlib output.
Methodological references for the drift pipeline live on `.drift_report.compute_drift_report`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..tuning import TuningColumn
from .drift_report import DriftReport, DriftCellColumn, DriftPairColumn

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


_MINIMUM_SESSIONS_FOR_RECURRENCE_HEATMAP: int = 2
"""Minimum number of sessions required to fill the recurrence heatmap. With fewer than two sessions there is
no off-diagonal entry to render, so the heatmap collapses to a placeholder."""


def _animal_title_prefix(animal_id: str | None) -> str:
    """Returns ``f"Animal {animal_id} — "`` when ``animal_id`` is provided, else an empty string.

    Centralizes the title prefix so every drift plot stays consistent when consumers need a per-animal
    title without each helper duplicating the conditional.
    """
    return f"Animal {animal_id} — " if animal_id else ""


def plot_population_vector_correlation_vs_lag(
    report: DriftReport, *, animal_id: str | None = None, figure_dpi: int = 150,
) -> plt.Figure:
    """Plots per-pair population-vector Pearson r against calendar-day lag and overlays the decay fit.

    Notes:
        Each scatter point is one ordered ``(session_a, session_b)`` pair from ``report.pairs``. The fitted
        ``amplitude * exp(-lag / tau_days) + offset`` model from the summary is overlaid as a smooth curve
        when the fit converged. Mirrors the Climer et al. (2025) cross-session drift summary figure: a slow
        tau and a high asymptotic offset indicate stable representations; a fast tau and a low offset
        indicate drifting representations.

    Args:
        report: The drift report whose ``pairs`` and ``summary.decay_*`` parameters drive the plot.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes.
    """
    pairs = report.pairs
    summary = report.summary

    figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
    if pairs.height == 0:
        axes.text(0.5, 0.5, "No session pairs", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    lag_days: NDArray[np.float32] = (
        pairs[DriftPairColumn.LAG_DAYS.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    pv_correlation: NDArray[np.float32] = (
        pairs[DriftPairColumn.POPULATION_VECTOR_CORRELATION.value].to_numpy().astype(np.float32, copy=False)
    )
    finite_mask = np.isfinite(lag_days) & np.isfinite(pv_correlation)
    axes.scatter(
        lag_days[finite_mask],
        pv_correlation[finite_mask],
        alpha=0.4,
        s=18,
        color="#1f77b4",
        label="session pairs",
    )

    if summary.decay_fit_succeeded:
        x_max = float(np.max(lag_days[finite_mask])) if int(np.sum(finite_mask)) > 0 else 1.0
        # noinspection PyTypeChecker
        x_grid: NDArray[np.float32] = np.linspace(0.0, x_max, num=200, dtype=np.float32)
        y_fit = (
            summary.decay_amplitude * np.exp(-x_grid / summary.decay_tau_days) + summary.decay_offset
        )
        axes.plot(
            x_grid,
            y_fit,
            color="#d62728",
            linewidth=2,
            label=(
                f"fit: A={summary.decay_amplitude:.2f}, tau={summary.decay_tau_days:.2f}d, "
                f"offset={summary.decay_offset:.2f}"
            ),
        )

    axes.set_xlabel("Session lag (days)")
    axes.set_ylabel("Population-vector correlation r")
    axes.set_title(f"{_animal_title_prefix(animal_id)}PV correlation versus calendar-day lag")
    axes.set_ylim(-0.2, 1.0)
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def plot_drift_vs_bleaching(
    report: DriftReport, *, animal_id: str | None = None, figure_dpi: int = 150,
) -> plt.Figure:
    """Plots per-cell mean rate-map correlation against per-cell baseline-fluorescence slope.

    Notes:
        Verifies that cells flagged as "drifting" by the rate-map correlation criterion are not predominantly
        cells whose baseline fluorescence dropped sharply over the chronic recording. A negative correlation
        between the rate-map r axis and the baseline-slope axis would indicate that the apparent drift is
        actually a bleaching artifact; an absent correlation supports the drift interpretation. When the
        animal's bleaching cross-correlation is unavailable the panel renders an explanatory placeholder.

    Args:
        report: The drift report whose ``cells`` drive the plot.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes.
    """
    cells = report.cells
    summary = report.summary
    figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
    if not summary.bleaching_available or cells.height == 0:
        axes.text(
            0.5,
            0.5,
            "Bleaching cross-correlation unavailable",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    correlation: NDArray[np.float32] = (
        cells[DriftCellColumn.MEAN_RATE_MAP_CORRELATION.value]
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    baseline_slope: NDArray[np.float32] = (
        cells[DriftCellColumn.CELL_BASELINE_SLOPE.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    is_high_drift: NDArray[np.bool_] = cells[DriftCellColumn.IS_HIGH_BLEACHING_DRIFT.value].to_numpy()
    finite_mask = np.isfinite(correlation) & np.isfinite(baseline_slope)
    if int(np.sum(finite_mask)) == 0:
        axes.text(0.5, 0.5, "No finite rows", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    axes.scatter(
        baseline_slope[finite_mask & ~is_high_drift],
        correlation[finite_mask & ~is_high_drift],
        s=12,
        alpha=0.5,
        color="#1f77b4",
        label="bleaching-clean",
    )
    if int(np.sum(finite_mask & is_high_drift)) > 0:
        axes.scatter(
            baseline_slope[finite_mask & is_high_drift],
            correlation[finite_mask & is_high_drift],
            s=18,
            alpha=0.7,
            color="#d62728",
            label="high-bleaching-drift",
        )
    axes.set_xlabel("Cell baseline slope (units / day)")
    axes.set_ylabel("Mean rate-map correlation r")
    axes.set_title(f"{_animal_title_prefix(animal_id)}Drift versus baseline-fluorescence slope")
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.axvline(0.0, color="black", linewidth=0.5)
    axes.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    return figure


def plot_recurrence_heatmap(
    report: DriftReport, *, animal_id: str | None = None, figure_dpi: int = 150,
) -> plt.Figure:
    """Plots a session-by-session heatmap of population recurrence rates for place / reward classifications.

    Notes:
        The heatmap is symmetric across its diagonal; only the upper triangle is filled because the per-pair
        feather is upper-triangular. Color encodes the fraction of cells that maintain the classification
        between sessions A and B, computed as ``recurrence_count / max(n_A, n_B)`` where ``n_X`` is the count
        of cells classified in session ``X``. Diagonal entries are 1.0 by construction.

    Args:
        report: The drift report whose ``pairs`` and ``cells`` drive the plot.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with two axes (place / reward).
    """
    pairs = report.pairs
    cells = report.cells
    summary = report.summary
    session_count = summary.session_count

    figure, axes_array = plt.subplots(1, 2, figsize=(11, 5), facecolor="white", dpi=figure_dpi)
    if pairs.height == 0 or session_count < _MINIMUM_SESSIONS_FOR_RECURRENCE_HEATMAP:
        for axes in axes_array:
            axes.text(0.5, 0.5, "Not enough sessions", ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
        return figure

    # Per-session classification counts come straight from the persisted trajectory list columns; this avoids
    # rebuilding the per-session-per-cell matrix from the pairs feather.
    place_trajectory = np.asarray(
        cells[DriftCellColumn.PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_
    )
    reward_trajectory = np.asarray(
        cells[DriftCellColumn.REWARD_TRAJECTORY.value].to_list(), dtype=np.bool_
    )
    # noinspection PyTypeChecker
    place_counts: NDArray[np.int32] = np.sum(place_trajectory, axis=0).astype(np.int32, copy=False)
    # noinspection PyTypeChecker
    reward_counts: NDArray[np.int32] = np.sum(reward_trajectory, axis=0).astype(np.int32, copy=False)

    a_indices = pairs[DriftPairColumn.SESSION_A_INDEX.value].to_numpy()
    b_indices = pairs[DriftPairColumn.SESSION_B_INDEX.value].to_numpy()
    place_recurrence = pairs[DriftPairColumn.PLACE_RECURRENCE_COUNT.value].to_numpy()
    reward_recurrence = pairs[DriftPairColumn.REWARD_RECURRENCE_COUNT.value].to_numpy()

    # noinspection PyTypeChecker
    place_matrix: NDArray[np.float32] = np.full((session_count, session_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    reward_matrix: NDArray[np.float32] = np.full((session_count, session_count), np.nan, dtype=np.float32)
    for index in range(a_indices.shape[0]):
        a_index = int(a_indices[index])
        b_index = int(b_indices[index])
        place_denom = max(int(place_counts[a_index]), int(place_counts[b_index]), 1)
        reward_denom = max(int(reward_counts[a_index]), int(reward_counts[b_index]), 1)
        # Recurrence is symmetric in (A, B); the per-pair feather only stores ``a < b`` entries, so the lower
        # triangle is filled here by mirroring rather than left empty.
        place_value = float(place_recurrence[index]) / place_denom
        reward_value = float(reward_recurrence[index]) / reward_denom
        place_matrix[a_index, b_index] = place_value
        place_matrix[b_index, a_index] = place_value
        reward_matrix[a_index, b_index] = reward_value
        reward_matrix[b_index, a_index] = reward_value
    diag_mask = np.arange(session_count)
    place_matrix[diag_mask, diag_mask] = 1.0
    reward_matrix[diag_mask, diag_mask] = 1.0

    for axes, matrix, title in zip(
        axes_array,
        (place_matrix, reward_matrix),
        ("Place recurrence", "Reward recurrence"),
        strict=True,
    ):
        image = axes.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        axes.set_title(title)
        axes.set_xlabel("Session B (chronological)")
        axes.set_ylabel("Session A (chronological)")
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="recurrence rate")
    if animal_id:
        figure.suptitle(f"Animal {animal_id} — recurrence rates", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        figure.tight_layout()
    return figure


def plot_sessions_active_distribution(
    report: DriftReport,
    *,
    classifier: str = "place",
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots a Ziv-style sessions-active distribution histogram with a per-session activity-rate inset.

    Notes:
        Main panel: bar chart of cell counts grouped by the number of sessions in which each cell was
        classified by the requested classifier (default ``IS_PLACE``). Bars are colored by sessions-active
        count using the matplotlib ``viridis`` ramp so the reader can match groups to the inset legend.
        Inset: percent of cells classified per session as a function of days since the first session.
        Mirrors Ziv et al. (2013) Fig. 2b for the cross-day "sessions active" view of the registered
        population.

    Args:
        report: The drift report whose ``cells`` and ``summary`` drive the plot.
        classifier: Which classification to count (``"place"``, ``"reward"``, or ``"strict_place"``).
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    cells = report.cells
    summary = report.summary
    cell_count = cells.height
    session_count = summary.session_count

    figure, axes = plt.subplots(figsize=(6.5, 4.5), facecolor="white", dpi=figure_dpi)
    if cell_count == 0 or session_count == 0:
        axes.text(0.5, 0.5, "No drift data available", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    trajectory_column = {
        "place": DriftCellColumn.PLACE_TRAJECTORY.value,
        "reward": DriftCellColumn.REWARD_TRAJECTORY.value,
        "strict_place": DriftCellColumn.STRICT_PLACE_TRAJECTORY.value,
    }.get(classifier, DriftCellColumn.PLACE_TRAJECTORY.value)
    label = {"place": "place", "reward": "reward", "strict_place": "strict-place"}.get(classifier, classifier)

    # noinspection PyTypeChecker
    trajectory: NDArray[np.bool_] = np.asarray(cells[trajectory_column].to_list(), dtype=np.bool_)
    # noinspection PyTypeChecker
    sessions_active: NDArray[np.int64] = trajectory.sum(axis=1).astype(np.int64, copy=False)

    bin_edges = np.arange(1, session_count + 2)
    counts, _ = np.histogram(sessions_active, bins=bin_edges)
    bar_centers = bin_edges[:-1]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(session_count - 1, 1)) for i in range(session_count)]
    axes.bar(bar_centers, counts, color=colors, edgecolor="black", linewidth=0.4)
    axes.set_xticks(bar_centers)
    axes.set_xlabel(f"Sessions active ({label})")
    axes.set_ylabel("Cells")
    axes.set_title(f"{_animal_title_prefix(animal_id)}Sessions-active distribution ({label})")
    axes.set_xlim(0.5, session_count + 0.5)
    # Add headroom so the top-left inset clears the tallest bar in its horizontal span.
    counts_max = float(counts.max()) if counts.size > 0 else 0.0
    axes.set_ylim(0.0, counts_max * 1.4 if counts_max > 0 else 1.0)

    days_since_first = np.asarray(summary.days_since_first, dtype=np.float32)
    # noinspection PyTypeChecker
    activity_rate: NDArray[np.float32] = (
        100.0 * trajectory.sum(axis=0).astype(np.float32, copy=False) / max(cell_count, 1)
    )
    inset = axes.inset_axes((0.10, 0.76, 0.38, 0.22))
    inset.plot(days_since_first, activity_rate, color="#1f77b4", marker="o", markersize=2.5, linewidth=1.0)
    inset.set_xlabel("Time (d)", fontsize=8)
    inset.set_ylabel("Cells active (%)", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    inset.set_ylim(0, max(float(activity_rate.max()) * 1.2 if activity_rate.size > 0 else 1.0, 1.0))
    figure.tight_layout()
    return figure


def plot_recurrence_probability_vs_lag(
    report: DriftReport,
    *,
    peak_match_cm: float = 20.0,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots active-cell and place-field recurrence probability versus calendar-day lag.

    Notes:
        For each chronological session pair, computes two recurrence probabilities normalized by
        ``max(n_A, n_B)`` (the size of the larger active population) so the rate is symmetric in pair
        order:
        * "Active cells": ``count(IS_PLACE in both A and B) / max(n_A, n_B)``.
        * "Place fields": same numerator further restricted to cells whose absolute peak shift between A
          and B is at most ``peak_match_cm``.
        Pairs are then binned by ``lag_days`` (rounded to whole days) and the per-bin mean ± SEM is drawn.
        Mirrors Ziv et al. (2013) Fig. 2c, with the place-field stability gate matching the paper's
        spatially-stable subset definition.

    Args:
        report: The drift report whose ``pairs`` and ``cells`` drive the plot.
        peak_match_cm: Maximum absolute peak shift (cm) for a pair to be counted as a stable place
            field. Defaults to 20 cm, matching the reward-zone width and the convention used in the
            Ziv-2013 / Sheintuch-2017 lineage.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    pairs = report.pairs
    cells = report.cells
    summary = report.summary

    figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
    if pairs.height == 0 or cells.height == 0 or summary.session_count < 2:
        axes.text(0.5, 0.5, "Not enough sessions", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    place_trajectory: NDArray[np.bool_] = np.asarray(
        cells[DriftCellColumn.PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_
    )
    # noinspection PyTypeChecker
    place_counts: NDArray[np.int32] = place_trajectory.sum(axis=0).astype(np.int32, copy=False)

    a_indices = pairs[DriftPairColumn.SESSION_A_INDEX.value].to_numpy()
    b_indices = pairs[DriftPairColumn.SESSION_B_INDEX.value].to_numpy()
    lag_days = pairs[DriftPairColumn.LAG_DAYS.value].to_numpy().astype(np.float32, copy=False)
    place_recurrence = pairs[DriftPairColumn.PLACE_RECURRENCE_COUNT.value].to_numpy()
    peak_shift_lists = pairs[DriftPairColumn.PEAK_SHIFT_CM_PER_CELL.value].to_list()

    pair_count = a_indices.shape[0]
    # noinspection PyTypeChecker
    active_recurrence: NDArray[np.float32] = np.full(pair_count, np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    field_recurrence: NDArray[np.float32] = np.full(pair_count, np.nan, dtype=np.float32)
    for index in range(pair_count):
        a_index = int(a_indices[index])
        b_index = int(b_indices[index])
        denom = max(int(place_counts[a_index]), int(place_counts[b_index]), 1)
        active_recurrence[index] = float(place_recurrence[index]) / denom
        intersection = place_trajectory[:, a_index] & place_trajectory[:, b_index]
        # noinspection PyTypeChecker
        peak_shift_pair: NDArray[np.float32] = np.asarray(peak_shift_lists[index], dtype=np.float32)
        field_count = int(np.sum(intersection & np.isfinite(peak_shift_pair) & (peak_shift_pair <= peak_match_cm)))
        field_recurrence[index] = field_count / denom

    # Bin by integer lag days; pairs share a bin when the rounded lag matches.
    finite_mask = np.isfinite(lag_days) & np.isfinite(active_recurrence) & np.isfinite(field_recurrence)
    if int(np.sum(finite_mask)) == 0:
        axes.text(0.5, 0.5, "No finite lags", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    rounded_lags = np.round(lag_days[finite_mask]).astype(np.int64)
    unique_lags = np.unique(rounded_lags)
    active_means = np.zeros(unique_lags.size, dtype=np.float32)
    active_sems = np.zeros(unique_lags.size, dtype=np.float32)
    field_means = np.zeros(unique_lags.size, dtype=np.float32)
    field_sems = np.zeros(unique_lags.size, dtype=np.float32)
    finite_active = active_recurrence[finite_mask]
    finite_field = field_recurrence[finite_mask]
    for bin_index, lag_bin in enumerate(unique_lags):
        mask = rounded_lags == lag_bin
        n = int(np.sum(mask))
        active_values = finite_active[mask]
        field_values = finite_field[mask]
        active_means[bin_index] = float(np.mean(active_values))
        field_means[bin_index] = float(np.mean(field_values))
        active_sems[bin_index] = float(np.std(active_values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        field_sems[bin_index] = float(np.std(field_values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    axes.errorbar(
        unique_lags, active_means, yerr=active_sems,
        marker="o", markersize=4, color="#1f77b4", linewidth=1.4, capsize=2, label="Active cells",
    )
    axes.errorbar(
        unique_lags, field_means, yerr=field_sems,
        marker="o", markersize=4, color="#d62728", linewidth=1.4, capsize=2,
        label=f"Place fields (|peak shift| <= {peak_match_cm:.0f} cm)",
    )
    axes.set_xlabel("Elapsed time (days)")
    axes.set_ylabel("Recurrence probability")
    axes.set_title(f"{_animal_title_prefix(animal_id)}Recurrence probability versus elapsed time")
    axes.set_ylim(0.0, 1.05)
    axes.set_xlim(left=0)
    axes.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def plot_reference_day_sorted_rate_maps(
    report: DriftReport,
    sessions: tuple[DatasetSession, ...],
    *,
    display_sessions: tuple[int, ...] | None = None,
    reference_sessions: tuple[int, ...] | None = None,
    classifier: str = "place",
    cmap: str = "jet",
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots reference-session-sorted rate-map heatmaps across selected sessions in the report.

    Notes:
        For each requested reference session, filters to cells classified by the requested classifier
        (default ``IS_PLACE``) on that day, sorts those cells by their rate-map peak position on the
        reference session, and renders the row-normalized rate maps for that ordered cell set at each
        displayed session. Each reference session yields one row of the output figure with one sub-axis
        per displayed session; cell counts are reported per row in the row label. Mirrors Ziv et al.
        (2013) Fig. 2 e/f/g.

        Sessions are addressed by 1-indexed chronological session number (``1`` is the first session in
        the report, ``2`` is the second, etc.). Out-of-range entries are silently skipped, and duplicates
        that resolve to the same session are deduplicated.

        Per-session rate maps come from each session's persisted ``tuning_cells.feather`` filtered by the
        trial type the drift report evaluated for that session. Sessions whose tuning artifact is missing
        on disk surface as "no data" placeholders.

    Args:
        report: The drift report whose ``cells`` and ``summary`` drive the cell selection and ordering.
        sessions: Chronologically ordered DatasetSession entries aligned with ``report.summary.session_names``.
        display_sessions: 1-indexed session numbers to render as columns. Defaults to five evenly-spaced
            sessions across the report (or every session when fewer than five are present).
        reference_sessions: 1-indexed session numbers to use as reference rows. Defaults to the first /
            middle / last entries of the resolved displayed sessions, which for the default five-session
            view yields the 1st / 3rd / 5th displayed sessions.
        classifier: Which classification to use for cell selection (``"place"``, ``"reward"``,
            ``"strict_place"``).
        cmap: Matplotlib colormap name for the rate-map intensities.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    summary = report.summary
    cells = report.cells
    session_count = summary.session_count

    def _resolve_indices(requested: tuple[int, ...]) -> tuple[int, ...]:
        """Maps each 1-indexed session number to a 0-indexed position; out-of-range entries are skipped."""
        resolved: list[int] = []
        for target in requested:
            idx = int(target) - 1
            if 0 <= idx < session_count and idx not in resolved:
                resolved.append(idx)
        return tuple(resolved)

    if display_sessions is None:
        if session_count >= 5:
            # Five evenly-spaced 0-indexed sessions across the full window. Brackets any mid-window
            # protocol shift cleanly and keeps the figure's column count manageable.
            spaced = (int(round(value)) for value in np.linspace(0, session_count - 1, num=5))
            deduped: list[int] = []
            for idx in spaced:
                if idx not in deduped:
                    deduped.append(idx)
            display_session_indices: tuple[int, ...] = tuple(deduped)
        else:
            display_session_indices = tuple(range(session_count))
    else:
        display_session_indices = _resolve_indices(display_sessions)

    if reference_sessions is None:
        n_disp = len(display_session_indices)
        if n_disp >= 3:
            reference_session_indices: tuple[int, ...] = (
                display_session_indices[0],
                display_session_indices[n_disp // 2],
                display_session_indices[-1],
            )
        else:
            reference_session_indices = display_session_indices
    else:
        reference_session_indices = _resolve_indices(reference_sessions)

    n_refs = len(reference_session_indices)
    n_columns = len(display_session_indices)

    figure, axes_array = plt.subplots(
        max(n_refs, 1), max(n_columns, 1),
        figsize=(2.0 * max(n_columns, 1) + 1.2, 3.5 * max(n_refs, 1)),
        facecolor="white", dpi=figure_dpi, squeeze=False,
        gridspec_kw={"wspace": 0.08, "hspace": 0.25},
    )
    if cells.height == 0 or session_count == 0 or n_refs == 0 or n_columns == 0:
        axes_array[0, 0].text(0.5, 0.5, "No drift data available", ha="center", va="center",
                              transform=axes_array[0, 0].transAxes)
        for axes in axes_array.flat:
            axes.set_axis_off()
        return figure

    if len(sessions) != session_count:
        for axes in axes_array.flat:
            axes.set_axis_off()
        axes_array[0, 0].text(
            0.5, 0.5,
            f"sessions tuple length ({len(sessions)}) does not match report ({session_count})",
            ha="center", va="center", transform=axes_array[0, 0].transAxes,
        )
        return figure

    trajectory_column = {
        "place": DriftCellColumn.PLACE_TRAJECTORY.value,
        "reward": DriftCellColumn.REWARD_TRAJECTORY.value,
        "strict_place": DriftCellColumn.STRICT_PLACE_TRAJECTORY.value,
    }.get(classifier, DriftCellColumn.PLACE_TRAJECTORY.value)
    label = {"place": "place", "reward": "reward", "strict_place": "strict-place"}.get(classifier, classifier)
    # noinspection PyTypeChecker
    trajectory: NDArray[np.bool_] = np.asarray(cells[trajectory_column].to_list(), dtype=np.bool_)

    # Pre-load per-session rate maps for every session that needs to be drawn or that anchors a sort
    # order. Filtering by the drift-report trial type keeps the row vectors aligned with cell_id.
    needed_session_indices = sorted(set(display_session_indices) | set(reference_session_indices))
    rate_maps_per_session: dict[int, NDArray[np.float32] | None] = {}
    bin_count_reference: int | None = None
    track_length_reference: float | None = None
    for sess_idx in needed_session_indices:
        path = sessions[sess_idx].tuning_cells_path
        if not path.exists():
            rate_maps_per_session[sess_idx] = None
            continue
        trial_type = summary.trial_types[sess_idx] if sess_idx < len(summary.trial_types) else None
        frame = pl.read_ipc(source=path, memory_map=True)
        if trial_type is not None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == trial_type)
        frame = frame.sort(TuningColumn.CELL_ID.value)
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.asarray(
            frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
        )
        rate_maps_per_session[sess_idx] = rate_maps
        if bin_count_reference is None and rate_maps.shape[0] > 0:
            bin_count_reference = int(rate_maps.shape[1])
            geometry_path = sessions[sess_idx].geometry_path
            if geometry_path.exists() and trial_type is not None:
                # Lazy import to avoid pulling shared_assets into the plotting module's import chain.
                from ...shared_assets import TrialGeometry  # noqa: PLC0415
                geometry = TrialGeometry.from_yaml(file_path=geometry_path)
                entry = geometry.entries.get(trial_type)
                if entry is not None:
                    track_length_reference = float(entry.trial_length_cm)

    bin_count = bin_count_reference if bin_count_reference is not None else 0
    track_length_cm = track_length_reference if track_length_reference is not None else float(bin_count)
    bin_size_cm = track_length_cm / bin_count if bin_count > 0 else 1.0

    for row_index, ref_idx in enumerate(reference_session_indices):
        ref_rate_maps = rate_maps_per_session.get(ref_idx)
        active_mask = trajectory[:, ref_idx]
        if ref_rate_maps is None or not active_mask.any():
            for column_position in range(n_columns):
                axes = axes_array[row_index, column_position]
                axes.text(0.5, 0.5, "no active cells", ha="center", va="center", transform=axes.transAxes)
                axes.set_axis_off()
            continue
        active_indices = np.where(active_mask)[0]
        ref_maps_for_active = ref_rate_maps[active_indices]
        finite_for_argmax = np.where(np.isfinite(ref_maps_for_active), ref_maps_for_active, -np.inf)
        peak_bins = np.argmax(finite_for_argmax, axis=1)
        order = np.argsort(peak_bins)
        sorted_cell_ids = active_indices[order]
        n_total = sorted_cell_ids.size

        for column_position, sess_idx in enumerate(display_session_indices):
            axes = axes_array[row_index, column_position]
            session_rate_maps = rate_maps_per_session.get(sess_idx)
            if session_rate_maps is None:
                axes.text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.transAxes)
                axes.set_xticks([])
                axes.set_yticks([])
                continue
            selected = session_rate_maps[sorted_cell_ids]
            row_max = np.nanmax(selected, axis=1, keepdims=True)
            row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
            normalized = np.clip(selected / row_max, 0.0, 1.0)
            normalized = np.where(np.isfinite(normalized), normalized, 0.0)
            axes.imshow(
                normalized, aspect="auto", origin="upper", cmap=cmap,
                extent=[0, bin_count * bin_size_cm, n_total, 0], vmin=0.0, vmax=1.0,
                interpolation="nearest",
            )
            axes.set_xlim(0, bin_count * bin_size_cm)
            if row_index == 0:
                axes.set_title(f"Session {sess_idx + 1}", fontsize=10)
            if row_index == n_refs - 1:
                axes.set_xlabel("Position (cm)", fontsize=9)
            else:
                axes.set_xticklabels([])
            if column_position == 0:
                axes.set_ylabel(
                    f"Cell ID\n(ordered for session {ref_idx + 1})\nn={n_total}",
                    fontsize=9,
                )
            else:
                axes.set_yticklabels([])

    figure.suptitle(
        f"{_animal_title_prefix(animal_id)}Reference-session-sorted rate maps ({label} cells)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


_DRIFT_PROFILE_CATEGORIES: tuple[str, ...] = (
    "block-stable",
    "drift-out",
    "drift-in",
    "cycling",
    "mostly-unstable",
)
"""Per-cell drift-profile category labels. The order is the legend order; ``classify_drift_profile``
emits indices into this tuple."""

_DRIFT_PROFILE_COLORS: tuple[str, ...] = (
    "#1b7837",  # block-stable: green
    "#f1a340",  # drift-out: orange
    "#998ec3",  # drift-in: purple
    "#d73027",  # cycling: red
    "#999999",  # mostly-unstable: gray
)
"""Per-category colors for the bar chart and the per-cell stability raster's right-edge category strip."""

_MINIMUM_ACTIVE_SESSIONS_FOR_PROFILE: int = 3
"""Cells must be classified ``IS_PLACE`` in at least this many sessions to receive a drift-profile
category. Below that, run-length statistics carry no information so the cell is excluded from the
analysis (and from the raster)."""


def classify_drift_profile(
    report: DriftReport,
    sessions: tuple[DatasetSession, ...],
    *,
    peak_match_cm: float = 20.0,
) -> tuple[NDArray[np.int8], NDArray[np.bool_], NDArray[np.bool_]]:
    """Classifies each cell's drift profile from its per-session stability trace.

    Notes:
        For every cell that is classified ``IS_PLACE`` in at least
        ``_MINIMUM_ACTIVE_SESSIONS_FOR_PROFILE`` sessions, the helper:

        1. Reads the cell's per-session rate map peak (argmax of the smoothed rate map) from each
           session's ``tuning_cells.feather``, filtered by the trial type the drift report evaluated.
        2. Defines the cell's "anchor peak" as the median peak across active sessions and marks each
           active session as *stable* when the peak falls within ±``peak_match_cm`` of that anchor.
        3. Run-length-encodes the resulting binary stability trace and assigns one of five categories
           from the run-length statistics.

        Cells with fewer than the active-session minimum, with all-NaN rate maps, or whose anchor peak
        is undefined are excluded (the corresponding entries in the returned mask are False) and their
        category is left at the "block-stable" default; consumers should always intersect the returned
        category with the inclusion mask before grouping.

    Args:
        report: The drift report whose ``cells.PLACE_TRAJECTORY`` and ``summary`` drive the analysis.
        sessions: Chronologically ordered DatasetSession entries aligned with
            ``report.summary.session_names``.
        peak_match_cm: Maximum absolute peak distance from the cell's anchor for a session to be
            counted as stable. Defaults to 20 cm to mirror ``plot_recurrence_probability_vs_lag``.

    Returns:
        A tuple ``(category_index, included_mask, stability_trace)``:
        * ``category_index``: int8 array of length ``cell_count`` carrying indices into
          ``_DRIFT_PROFILE_CATEGORIES``. Cells excluded by ``included_mask`` carry an undefined value.
        * ``included_mask``: bool array of length ``cell_count`` flagging cells that received a
          well-defined category.
        * ``stability_trace``: bool array of shape ``(cell_count, session_count)`` where True marks
          "active and within ±peak_match_cm of anchor" and False marks "inactive or unstable". The
          per-session active flag is recoverable as ``stability_trace | active_unstable_trace`` if the
          caller needs the three-state form.
    """
    summary = report.summary
    cells = report.cells
    cell_count = cells.height
    session_count = summary.session_count

    if cell_count == 0 or session_count == 0 or len(sessions) != session_count:
        # noinspection PyTypeChecker
        empty_categories: NDArray[np.int8] = np.zeros(cell_count, dtype=np.int8)
        # noinspection PyTypeChecker
        empty_mask: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)
        # noinspection PyTypeChecker
        empty_trace: NDArray[np.bool_] = np.zeros((cell_count, session_count), dtype=np.bool_)
        return empty_categories, empty_mask, empty_trace

    # noinspection PyTypeChecker
    place_trajectory: NDArray[np.bool_] = np.asarray(
        cells[DriftCellColumn.PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_
    )

    # Resolve per-session peak position per cell from each session's rate map. The rate map is filtered
    # by the drift-report trial type so the per-session row order aligns with ``cell_id``.
    # noinspection PyTypeChecker
    peak_cm_per_session: NDArray[np.float32] = np.full(
        (cell_count, session_count), np.nan, dtype=np.float32,
    )
    bin_size_cm_default = 5.0  # Same default the place-field detector uses.
    for sess_idx in range(session_count):
        path = sessions[sess_idx].tuning_cells_path
        if not path.exists():
            continue
        trial_type = summary.trial_types[sess_idx] if sess_idx < len(summary.trial_types) else None
        frame = pl.read_ipc(source=path, memory_map=True)
        if trial_type is not None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == trial_type)
        frame = frame.sort(TuningColumn.CELL_ID.value)
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.asarray(
            frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
        )
        if rate_maps.shape[0] == 0:
            continue
        bin_count = rate_maps.shape[1]
        bin_size_cm = bin_size_cm_default
        geometry_path = sessions[sess_idx].geometry_path
        if geometry_path.exists() and trial_type is not None:
            from ...shared_assets import TrialGeometry  # noqa: PLC0415
            geometry = TrialGeometry.from_yaml(file_path=geometry_path)
            entry = geometry.entries.get(trial_type)
            if entry is not None and bin_count > 0:
                bin_size_cm = float(entry.trial_length_cm) / bin_count
        finite_for_argmax = np.where(np.isfinite(rate_maps), rate_maps, -np.inf)
        peak_bins = np.argmax(finite_for_argmax, axis=1)
        peaks_cm = (peak_bins.astype(np.float32) + np.float32(0.5)) * np.float32(bin_size_cm)
        # Cells whose rate map is all-NaN get NaN peaks rather than a meaningless bin-0 argmax.
        all_nan = ~np.any(np.isfinite(rate_maps), axis=1)
        peaks_cm[all_nan] = np.float32(np.nan)
        peak_cm_per_session[: rate_maps.shape[0], sess_idx] = peaks_cm

    # Build the per-cell binary stability trace plus the inclusion mask.
    # noinspection PyTypeChecker
    stability_trace: NDArray[np.bool_] = np.zeros((cell_count, session_count), dtype=np.bool_)
    # noinspection PyTypeChecker
    included_mask: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)
    # noinspection PyTypeChecker
    category_index: NDArray[np.int8] = np.zeros(cell_count, dtype=np.int8)

    for cell in range(cell_count):
        active_sessions = np.where(place_trajectory[cell])[0]
        if active_sessions.size < _MINIMUM_ACTIVE_SESSIONS_FOR_PROFILE:
            continue
        active_peaks = peak_cm_per_session[cell, active_sessions]
        finite_active = active_peaks[np.isfinite(active_peaks)]
        if finite_active.size == 0:
            continue
        anchor = float(np.median(finite_active))
        # Mark stable in t = active in t AND |peak(t) - anchor| <= peak_match_cm.
        for sess_idx in active_sessions:
            peak_value = float(peak_cm_per_session[cell, sess_idx])
            if np.isfinite(peak_value) and abs(peak_value - anchor) <= peak_match_cm:
                stability_trace[cell, sess_idx] = True

        active_count = int(active_sessions.size)
        active_set = active_sessions
        is_stable_active = stability_trace[cell, active_set]
        stable_count = int(is_stable_active.sum())
        if active_count == 0:
            continue
        stability_fraction = stable_count / active_count

        # Run-length encode the active-only stability trace; an "unstable" gap inside the active span
        # ends a stable run.
        runs: list[tuple[int, int]] = []  # (start_active_index, end_active_index_exclusive)
        in_run = False
        run_start = 0
        for active_pos, flag in enumerate(is_stable_active):
            if flag and not in_run:
                run_start = active_pos
                in_run = True
            elif not flag and in_run:
                runs.append((run_start, active_pos))
                in_run = False
        if in_run:
            runs.append((run_start, active_count))

        included_mask[cell] = True
        if stability_fraction < 0.3:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("mostly-unstable")
            continue
        if not runs:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("mostly-unstable")
            continue
        n_runs = len(runs)
        longest_run = max(end - start for start, end in runs)
        if n_runs >= 2:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("cycling")
            continue
        if longest_run >= int(round(0.7 * active_count)) and runs[0] == (0, active_count):
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("block-stable")
            continue
        first_stable, last_stable = runs[0]
        if first_stable == 0 and last_stable < active_count:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("drift-out")
        elif first_stable > 0 and last_stable == active_count:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("drift-in")
        elif longest_run >= int(round(0.7 * active_count)):
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("block-stable")
        else:
            category_index[cell] = _DRIFT_PROFILE_CATEGORIES.index("cycling")

    return category_index, included_mask, stability_trace


def plot_drift_profile_categories(
    report: DriftReport,
    sessions: tuple[DatasetSession, ...],
    *,
    peak_match_cm: float = 20.0,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots a category-count bar chart for the per-cell drift-profile classification.

    Notes:
        Bars are colored by category and labeled with the absolute count plus the percentage of all
        included cells (cells with at least ``_MINIMUM_ACTIVE_SESSIONS_FOR_PROFILE`` active sessions
        and at least one finite peak). The category counts directly answer the question "do unstable
        cells cycle versus maintain-then-drift": a tall ``cycling`` bar means the population flickers
        between stable and unstable; a tall ``drift-out`` bar means cells maintain stability for a
        stretch and then leave it monotonically.

    Args:
        report: The drift report whose ``cells`` and ``summary`` drive the classification.
        sessions: Chronologically ordered DatasetSession entries aligned with the report.
        peak_match_cm: Forwarded to ``classify_drift_profile``; default 20 cm.
        animal_id: Optional animal label injected into the title.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    category_index, included_mask, _ = classify_drift_profile(
        report=report, sessions=sessions, peak_match_cm=peak_match_cm,
    )
    figure, axes = plt.subplots(figsize=(7, 4.5), facecolor="white", dpi=figure_dpi)
    n_total = int(included_mask.sum())
    if n_total == 0:
        axes.text(0.5, 0.5, "No cells with sufficient activity for profile classification",
                  ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    counts = np.zeros(len(_DRIFT_PROFILE_CATEGORIES), dtype=np.int64)
    for category_position in range(len(_DRIFT_PROFILE_CATEGORIES)):
        counts[category_position] = int(np.sum((category_index == category_position) & included_mask))

    x_positions = np.arange(len(_DRIFT_PROFILE_CATEGORIES))
    bars = axes.bar(x_positions, counts, color=_DRIFT_PROFILE_COLORS, edgecolor="black", linewidth=0.5)
    counts_max = int(counts.max()) if counts.size > 0 else 0
    for bar, count in zip(bars, counts.tolist(), strict=True):
        percent = 100.0 * count / n_total if n_total > 0 else 0.0
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts_max, 1) * 0.01,
            f"{count}\n({percent:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    axes.set_xticks(x_positions)
    axes.set_xticklabels(_DRIFT_PROFILE_CATEGORIES, rotation=15, ha="right")
    axes.set_ylabel("Cells")
    # Add headroom so the two-line ``count\n(percent%)`` annotations on top of the tallest bar do not
    # collide with the figure title.
    axes.set_ylim(0.0, counts_max * 1.18 if counts_max > 0 else 1.0)
    axes.set_title(
        f"{_animal_title_prefix(animal_id)}Drift profile categories "
        f"(n={n_total} cells active in >= {_MINIMUM_ACTIVE_SESSIONS_FOR_PROFILE} sessions, "
        f"|peak shift| <= {peak_match_cm:.0f} cm)"
    )
    figure.tight_layout()
    return figure
