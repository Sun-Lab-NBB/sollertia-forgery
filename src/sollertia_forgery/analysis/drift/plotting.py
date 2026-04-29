"""Per-animal plots for the cross-session tuning-drift pipeline.

Module-level functions consume the per-animal `DriftReport` produced by `.drift_report` and emit
matplotlib figures. Mirrors the plotting layout of `..bleaching.plotting`, `..sce.plotting`, and
`..tuning.plotting` so each analysis package keeps a single file responsible for matplotlib output.
Methodological references for the drift pipeline live on `.drift_report.compute_drift_report`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .drift_report import DriftReport, DriftCellColumn, DriftPairColumn

if TYPE_CHECKING:
    from numpy.typing import NDArray


_CLASSIFICATION_RASTER_COLORS: tuple[str, str, str, str] = ("#f0f0f0", "#377eb8", "#e41a1c", "#984ea3")
"""(none, place-only, reward-only, place AND reward) palette for the per-cell classification raster.
Colorblind-friendly choices from the ColorBrewer Set1 family with a light-gray "not classified" base so the
raster reads as presence-of-tuning over time."""

_MINIMUM_SESSIONS_FOR_RECURRENCE_HEATMAP: int = 2
"""Minimum number of sessions required to fill the recurrence heatmap. With fewer than two sessions there is
no off-diagonal entry to render, so the heatmap collapses to a placeholder."""


def plot_classification_raster(
    report: DriftReport,
    *,
    classifier: str = "place",
    sort_by_persistence: bool = True,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots a per-cell-by-per-session classification raster colored by tuning identity.

    Notes:
        Each row is one multi-day-registered cell, each column is one session in chronological order. The
        colormap encodes (none, place-only, reward-only, both) so the raster shows simultaneously when a cell
        is classified and which classification fired. Cells are optionally sorted by total session-fraction
        of the requested classifier so persistent cells cluster at the top of the figure. The classifier
        selection is ``"place"``, ``"reward"``, or ``"strict_place"`` and only changes the sort order; the
        cell-by-session colors always reflect the joint place / reward classification.

    Args:
        report: The drift report whose ``cells`` and ``summary`` drive the plot.
        classifier: Which classification persistence column to sort by (``"place"``, ``"reward"``,
            ``"strict_place"``). Defaults to ``"place"``.
        sort_by_persistence: When True (default), sort cells by descending fraction of sessions classified
            as the requested classifier; when False, preserve the canonical ``cell_id`` order.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes carrying the raster, a top legend, and chronological session ticks.
    """
    cells = report.cells
    cell_count = cells.height
    session_count = report.summary.session_count

    if cell_count == 0 or session_count == 0:
        figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
        axes.text(0.5, 0.5, "No drift data available", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    place_trajectory = np.asarray(cells[DriftCellColumn.PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_)
    reward_trajectory = np.asarray(
        cells[DriftCellColumn.REWARD_TRAJECTORY.value].to_list(), dtype=np.bool_
    )
    # noinspection PyTypeChecker
    classification_codes: NDArray[np.int8] = np.zeros((cell_count, session_count), dtype=np.int8)
    classification_codes[place_trajectory & ~reward_trajectory] = 1
    classification_codes[~place_trajectory & reward_trajectory] = 2
    classification_codes[place_trajectory & reward_trajectory] = 3

    sort_column = {
        "place": DriftCellColumn.PLACE_SESSION_FRACTION.value,
        "reward": DriftCellColumn.REWARD_SESSION_FRACTION.value,
        "strict_place": DriftCellColumn.STRICT_PLACE_SESSION_FRACTION.value,
    }.get(classifier, DriftCellColumn.PLACE_SESSION_FRACTION.value)

    if sort_by_persistence:
        # noinspection PyTypeChecker
        sort_keys: NDArray[np.float32] = (
            cells[sort_column].to_numpy().astype(np.float32, copy=False)
        )
        finite_keys = np.where(np.isfinite(sort_keys), sort_keys, -1.0)
        order = np.argsort(-finite_keys, kind="stable")
        classification_codes = classification_codes[order]

    figure, axes = plt.subplots(figsize=(8, max(3.0, 0.04 * cell_count)), facecolor="white", dpi=figure_dpi)
    cmap = ListedColormap(_CLASSIFICATION_RASTER_COLORS)
    axes.imshow(
        classification_codes,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
    )
    axes.set_xlabel("Session index (chronological)")
    axes.set_ylabel("Cell" + (f" (sorted by {classifier} session fraction)" if sort_by_persistence else ""))
    axes.set_title("Cross-session classification raster")
    legend_handles = [
        Patch(facecolor=_CLASSIFICATION_RASTER_COLORS[0], edgecolor="black", label="not classified"),
        Patch(facecolor=_CLASSIFICATION_RASTER_COLORS[1], edgecolor="black", label="place"),
        Patch(facecolor=_CLASSIFICATION_RASTER_COLORS[2], edgecolor="black", label="reward"),
        Patch(facecolor=_CLASSIFICATION_RASTER_COLORS[3], edgecolor="black", label="place AND reward"),
    ]
    axes.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)
    figure.tight_layout()
    return figure


def plot_population_vector_correlation_vs_lag(report: DriftReport, *, figure_dpi: int = 150) -> plt.Figure:
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
    axes.set_title("PV correlation versus calendar-day lag")
    axes.set_ylim(-0.2, 1.0)
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def plot_peak_shift_distribution(report: DriftReport, *, figure_dpi: int = 150) -> plt.Figure:
    """Plots the per-cell mean peak-shift distribution split by classification persistence.

    Notes:
        Two histograms overlay on the same axes — persistent place cells versus the rest of the population
        — so the reader can immediately see whether the cells the pipeline calls "stable" actually have
        smaller peak shifts than the unstable cells. Cells with NaN ``mean_peak_shift_cm`` (no surviving
        pair contributed a finite shift) are excluded from both histograms.

    Args:
        report: The drift report whose ``cells`` drive the plot.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes.
    """
    cells = report.cells
    figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
    if cells.height == 0:
        axes.text(0.5, 0.5, "No cells", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    peak_shifts: NDArray[np.float32] = (
        cells[DriftCellColumn.MEAN_PEAK_SHIFT_CM.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    persistent_place: NDArray[np.bool_] = cells[DriftCellColumn.IS_PERSISTENT_PLACE.value].to_numpy()

    finite_mask = np.isfinite(peak_shifts)
    persistent_values = peak_shifts[finite_mask & persistent_place]
    other_values = peak_shifts[finite_mask & ~persistent_place]
    if persistent_values.size + other_values.size == 0:
        axes.text(0.5, 0.5, "No finite peak shifts", ha="center", va="center", transform=axes.transAxes)
        axes.set_axis_off()
        return figure

    bin_edges_array = np.linspace(
        0.0,
        float(max(persistent_values.max() if persistent_values.size > 0 else 0.0,
                  other_values.max() if other_values.size > 0 else 0.0)) + 1e-3,
        num=30,
    )
    bin_edges = [float(value) for value in bin_edges_array.tolist()]
    if persistent_values.size > 0:
        axes.hist(
            persistent_values,
            bins=bin_edges,
            alpha=0.55,
            label=f"persistent place (n={persistent_values.size})",
            color="#1b7837",
        )
    if other_values.size > 0:
        axes.hist(
            other_values,
            bins=bin_edges,
            alpha=0.55,
            label=f"other cells (n={other_values.size})",
            color="#762a83",
        )

    axes.set_xlabel("Mean |peak shift| across pairs (cm)")
    axes.set_ylabel("Cell count")
    axes.set_title("Peak shift split by place persistence")
    axes.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def plot_drift_vs_bleaching(report: DriftReport, *, figure_dpi: int = 150) -> plt.Figure:
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
    axes.set_title("Drift versus baseline-fluorescence slope")
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.axvline(0.0, color="black", linewidth=0.5)
    axes.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    return figure


def plot_recurrence_heatmap(report: DriftReport, *, figure_dpi: int = 150) -> plt.Figure:
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
    figure.tight_layout()
    return figure
