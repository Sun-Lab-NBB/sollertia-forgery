"""Per-animal and dataset-level bleaching plots.

Module-level functions consume the cross-session `BleachingReport` produced by `.bleaching_analysis`
or, for the dataset-wide trace, a `DatasetData` whose animals each carry a saved bleaching report. Mirrors
the plotting layout of `..sce.plotting` and `..tuning.plotting` so each analysis package keeps a single
file responsible for matplotlib output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
import matplotlib.pyplot as plt

from ..shared_utilities import resolve_display_units
from .bleaching_analysis import BleachingColumn, BleachingReport

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...shared_assets import DatasetData


def plot_baseline_trend(report: BleachingReport) -> plt.Figure:
    """Plots the population-median per-session baseline fluorescence trend, the exponential fit, and per-cell
    baseline fluorescence distributions.

    Args:
        report: The cross-session bleaching report whose table and decay fit drive the plot.

    Returns:
        A matplotlib Figure showing the across-session baseline fluorescence trend.
    """
    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    table = report.table
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    population_baseline: NDArray[np.float32] = (
        table[BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value].to_numpy().astype(np.float32, copy=False)
    )
    cell_baseline_distributions = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.CELL_BASELINE_FLUORESCENCE.value].to_list()
    ]
    cell_count = len(cell_baseline_distributions[0]) if cell_baseline_distributions else 0

    # Plots in display units (integer day or hour ticks).
    unit, ticks = resolve_display_units(days_since_first=days)

    # Computes a box width that scales with the smallest tick step. Integer ticks guarantee step >= 1, so the
    # prior float-step floor is no longer needed.
    minimum_tick_step = float(np.diff(ticks).min()) if len(ticks) > 1 else 1.0
    box_width = 0.4 * minimum_tick_step

    # Draws the per-cell distributions as boxplots so the population spread is visible alongside the median trend.
    axes.boxplot(cell_baseline_distributions, positions=ticks, widths=box_width, showfliers=False)

    # Overlays the population-median trend used for the exponential fit.
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
        f"Across-session baseline fluorescence trend (n={cell_count} registered cells)",
        fontsize=10,
    )
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def plot_within_session(report: BleachingReport) -> plt.Figure:
    """Plots the within-session FOV-mean baseline trace for each session as overlaid curves.

    Args:
        report: The cross-session bleaching report whose per-session within-session traces drive the plot.

    Returns:
        A matplotlib Figure showing within-session bleaching.
    """
    # Wider canvas reserves room for the per-session legend that is anchored outside the right of the axes
    # so it does not occlude the traces; the legend column scales linearly with session count.
    figure, axes = plt.subplots(1, 1, figsize=(9, 4), facecolor="white", dpi=150)

    table = report.table
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    time_seconds_list = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value].to_list()
    ]
    baseline_list = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_BASELINE.value].to_list()
    ]
    # noinspection PyTypeChecker
    drops: NDArray[np.float32] = (
        table[BleachingColumn.WITHIN_SESSION_FRACTIONAL_DROP.value].to_numpy().astype(np.float32, copy=False)
    )
    session_count = len(days)

    # Resolves the integer display unit so per-session legend labels match the across-session plots and summary
    # rather than displaying floats. The x-axis here is within-session minutes, so only the legend changes.
    unit, ticks = resolve_display_units(days_since_first=days)
    unit_capitalized = unit.capitalize()

    colormap = plt.get_cmap("viridis")
    for index in range(session_count):
        # Guards against division by zero when the report contains a single session.
        color = colormap(index / max(session_count - 1, 1))
        label = f"{unit_capitalized} {int(ticks[index])} (drop={drops[index]:.1%})"
        axes.plot(
            time_seconds_list[index] / 60.0,
            baseline_list[index],
            color=color,
            linewidth=1.0,
            label=label,
        )

    axes.set_xlabel("Time within session (minutes)")
    axes.set_ylabel("FOV-mean baseline (a.u.)")
    axes.set_title("Within-session bleaching", fontsize=10)
    # Anchors the legend to the right of the axes so trace inspection is not obstructed when many sessions
    # accumulate. ``tight_layout`` accounts for the externally placed legend in current matplotlib.
    axes.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    figure.tight_layout()
    return figure


def plot_within_session_average(report: BleachingReport) -> plt.Figure:
    """Plots the across-session mean of the within-session FOV-mean baseline trace, with each per-session trace
    overlaid as a translucent gray curve for context.

    Notes:
        All sessions share the same bin-center time grid (10 s, 30 s, 50 s, ... by default — the bin spacing
        equals ``session_baseline_window_seconds`` regardless of per-session sampling rate). Per-session
        baselines are NaN-padded to the longest session's length and the mean is taken over each bin via
        ``np.nanmean`` so the bold trace extends to the rightmost gray trace; bins beyond a given session's end
        simply do not contribute to that point. Sessions whose within-session computation produced an empty
        bin set (degenerate or fully trimmed by the warmup cutoff) are skipped to avoid biasing the mean
        toward zero-length contributors.

    Args:
        report: The cross-session bleaching report whose per-session within-session traces drive the plot.

    Returns:
        A matplotlib Figure showing the average within-session bleaching trend.
    """
    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    table = report.table
    time_seconds_list = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_TIME_SECONDS.value].to_list()
    ]
    baseline_list = [
        np.asarray(values, dtype=np.float32)
        for values in table[BleachingColumn.WITHIN_SESSION_BASELINE.value].to_list()
    ]

    # Draws each session as a translucent gray trace first so the bold mean line draws on top of the bundle.
    for time_seconds, baseline in zip(time_seconds_list, baseline_list, strict=True):
        axes.plot(time_seconds / 60.0, baseline, color="grey", alpha=0.3, linewidth=0.8)

    # Builds a NaN-padded (n_sessions, max_bins) matrix and takes ``np.nanmean`` along the session axis so the
    # mean trace extends to the longest session's last bin. Each column drops sessions that ended earlier from
    # its mean, which is honest about the shrinking sample size at the right edge without truncating the line.
    usable_baselines = [baseline for baseline in baseline_list if baseline.size > 0]
    if usable_baselines:
        max_length = max(baseline.size for baseline in usable_baselines)
        # noinspection PyTypeChecker
        baseline_matrix: NDArray[np.float32] = np.full((len(usable_baselines), max_length), np.nan, dtype=np.float32)
        for index, baseline in enumerate(usable_baselines):
            baseline_matrix[index, : baseline.size] = baseline
        # noinspection PyTypeChecker
        mean_baseline: NDArray[np.float32] = np.nanmean(baseline_matrix, axis=0).astype(np.float32, copy=False)
        longest_time = max(time_seconds_list, key=lambda candidate: candidate.size)[:max_length]
        axes.plot(
            longest_time / 60.0,
            mean_baseline,
            color="black",
            linewidth=2.5,
            label="Across-session mean",
        )
        axes.legend(loc="upper right", fontsize=8, frameon=False)

    axes.set_xlabel("Time within session (minutes)")
    axes.set_ylabel("FOV-mean baseline (a.u.)")
    axes.set_title("Average within-session bleaching", fontsize=10)
    figure.tight_layout()
    return figure


def plot_snr_distributions(report: BleachingReport) -> plt.Figure:
    """Plots per-session per-cell SNR distributions as violins with the population-median trend overlaid.

    Args:
        report: The cross-session bleaching report whose per-cell SNR arrays drive the plot.

    Returns:
        A matplotlib Figure showing the SNR-vs-session distribution.
    """
    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    table = report.table
    # noinspection PyTypeChecker
    days: NDArray[np.float32] = table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
    snr_data = [np.asarray(values, dtype=np.float32) for values in table[BleachingColumn.CELL_SNR.value].to_list()]
    # noinspection PyTypeChecker
    population_snr: NDArray[np.float32] = (
        table[BleachingColumn.POPULATION_SNR.value].to_numpy().astype(np.float32, copy=False)
    )

    # Plots in display units so the SNR violins line up with the baseline-trend boxplots on the same x-axis.
    unit, ticks = resolve_display_units(days_since_first=days)

    axes.violinplot(snr_data, positions=ticks, showmedians=True)
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
    axes.set_title("Per-cell SNR across sessions", fontsize=10)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure


def plot_dataset_baseline_trend(dataset: DatasetData) -> plt.Figure:
    """Plots per-animal population-median baseline fluorescence trends overlaid for every animal in the dataset,
    with the across-animal mean rendered as a thick black line on top.

    Notes:
        Loads the saved ``BleachingReport`` for each animal via ``BleachingReport.load``; animals without a
        persisted report are skipped silently so this can be called on partially-evaluated datasets. Per-animal
        traces are drawn as translucent gray lines using rounded integer days as x-coordinates so the cross-animal
        x-axis is consistent regardless of any per-animal hour-resolution display unit. The across-animal mean is
        computed on the integer-day union grid by inserting each animal's per-day F0 at its day index and taking
        nanmean across animals; days when no animal contributes a value are excluded from the mean line. Y-axis
        is raw fluorescence (a.u.) so absolute baseline differences across animals stay visible alongside the
        trend; absolute level differences are themselves diagnostic information.

    Args:
        dataset: The DatasetData instance whose animals contribute to the aggregate plot.

    Returns:
        A matplotlib Figure showing the across-animal baseline fluorescence trend.
    """
    figure, axes = plt.subplots(1, 1, figsize=(7, 4), facecolor="white", dpi=150)

    animal_traces: list[tuple[NDArray[np.int64], NDArray[np.float32]]] = []
    for dataset_animal in dataset.animals:
        try:
            report = BleachingReport.load(animal=dataset_animal)
        except FileNotFoundError:
            continue
        # noinspection PyTypeChecker
        days_float: NDArray[np.float32] = (
            report.table[BleachingColumn.DAYS_SINCE_FIRST.value].to_numpy().astype(np.float32, copy=False)
        )
        # noinspection PyTypeChecker
        baselines: NDArray[np.float32] = (
            report.table[BleachingColumn.POPULATION_BASELINE_FLUORESCENCE.value]
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        if days_float.size == 0:
            continue
        # noinspection PyTypeChecker
        days_int: NDArray[np.int64] = np.round(days_float).astype(np.int64, copy=False)
        animal_traces.append((days_int, baselines))

    if not animal_traces:
        axes.set_xlabel("Days since first session")
        axes.set_ylabel("Baseline fluorescence (a.u.)")
        axes.set_title("Across-animal baseline fluorescence trend (no reports found)", fontsize=10)
        figure.tight_layout()
        return figure

    for days_int, baselines in animal_traces:
        axes.plot(days_int, baselines, color="grey", alpha=0.5, linewidth=1.0, marker="o", markersize=3)

    # Builds the (n_animals, n_days) value matrix used by the median / IQR aggregates.
    max_day = int(max(days_int.max() for days_int, _ in animal_traces))
    # noinspection PyTypeChecker
    matrix: NDArray[np.float32] = np.full((len(animal_traces), max_day + 1), np.nan, dtype=np.float32)
    for index, (days_int, baselines) in enumerate(animal_traces):
        # Per-animal day collisions (rare under the protocol's >=1h spacing rule) overwrite earlier writes, which
        # is acceptable because the dataset-level plot only needs one value per (animal, day) cell.
        matrix[index, days_int] = baselines

    # Per-day median and interquartile range as outlier-robust replacements for mean +/- std. A single high- or
    # low-baseline animal can pull mean +/- std arbitrarily; median and IQR cap the influence of any single
    # animal at one rank position. ``np.nanmedian`` and ``np.nanpercentile`` emit a RuntimeWarning for any
    # all-NaN column, suppressed because the resulting NaNs are filtered out via ``valid_mask`` before plotting.
    # noinspection PyTypeChecker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # noinspection PyTypeChecker
        median_trace: NDArray[np.float32] = np.nanmedian(matrix, axis=0).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        lower_quartile: NDArray[np.float32] = np.nanpercentile(matrix, 25, axis=0).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        upper_quartile: NDArray[np.float32] = np.nanpercentile(matrix, 75, axis=0).astype(np.float32, copy=False)

    valid_mask = np.isfinite(median_trace)
    grid = np.arange(max_day + 1, dtype=np.int64)
    axes.fill_between(
        grid[valid_mask],
        lower_quartile[valid_mask],
        upper_quartile[valid_mask],
        color="black",
        alpha=0.15,
        linewidth=0,
        label="IQR (25-75%)",
    )
    axes.plot(grid[valid_mask], median_trace[valid_mask], color="black", linewidth=2.5, label="Across-animal median")

    axes.set_xlabel("Days since first session")
    axes.set_ylabel("Baseline fluorescence (a.u.)")
    axes.set_title(
        f"Across-animal baseline fluorescence trend (n={len(animal_traces)} animals)",
        fontsize=10,
    )
    axes.legend(loc="upper right", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure
