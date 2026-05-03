"""Per-session plotting helpers for the tuning pipeline.

Module-level functions consume a `TuningReport` plus the trial type the figure should depict, plus —
where noted — the session's ``data.feather`` opened memory-mapped at plot time. Persistence and summarization
stay on the report. Every function takes a ``trial_type`` argument that selects the long-format slice of the
report's cells feather and the matching `TuningTrialSummary` entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from ...forging import FluorescenceColumn
from .utilities import assemble_run_session_data
from .tuning_report import TuningColumn, TuningReport
from ...shared_assets import DatasetColumn, TrialGeometry
from ..shared_utilities import trim_acquisition_warmup

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


_PLOT_TICK_INTERVAL_CM: float = 25.0
"""Spacing in centimeters between x-axis ticks on track-position plots."""
_PLACE_STRIP_WIDTH_RATIO: float = 0.04
"""Per-strip width ratio (relative to the main heatmap) used by the per-cell significance strips."""
_PLACE_PVALUE_DISPLAY_FLOOR: float = 1e-4
"""P-value floor used when computing the ``-log10(p)`` color scale; keeps the dynamic range bounded."""


def plot_place_cell_heatmap(
    report: TuningReport,
    *,
    trial_type: str,
    title: str | None = None,
    sort_by_position: bool = True,
    show_only_place_cells: bool = True,
    require_place: bool = True,
    require_stable: bool = True,
    require_peak_significant: bool = True,
    mutually_exclusive: bool = True,
    show_significance_strip: bool = True,
    figure_dpi: int = 150,
    minimum_percentile: float = 0.5,
    maximum_percentile: float = 0.9,
    cmap: str = "gray_r",
    show_color_bar: bool = True,
) -> plt.Figure:
    """Plots the position-ordered binned-fluorescence heatmap for ``trial_type`` from the persisted rate maps.

    Args:
        report: TuningReport whose persisted long-format rate-map column drives the heatmap.
        trial_type: Trial type to extract; must match an entry in ``report.summary.trial_type_summaries``.
        title: Optional title displayed at the top of the figure.
        sort_by_position: Order cells by their place-field center along the track before plotting.
        show_only_place_cells: Display only cells that pass every requested criterion (the AND of the three
            ``require_*`` flags below).
        require_place: Require ``IS_PLACE`` (place-field morphology + lap coverage).
        require_stable: Require ``IS_STABLE`` (split-half stability shuffle).
        require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (per-cell peak shuffle).
        mutually_exclusive: When True (default), cells also flagged as ``IS_REWARD_CELL`` are removed from the
            place population so the panel shows only place cells that are not reward cells.
        show_significance_strip: When True, renders one thin per-cell ``-log10(p)`` strip to the left of the
            main heatmap for each active p-value-bearing criterion (Stable / Peak).
        figure_dpi: Figure resolution in dots per inch.
        minimum_percentile: Percentile used as the lower bound of the color scale.
        maximum_percentile: Percentile used as the upper bound of the color scale.
        cmap: Matplotlib colormap name for the rate-map panel.
        show_color_bar: Render a color bar alongside the heatmap.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    bin_size_cm = trial_summary.bin_size_cm
    bin_count = trial_summary.bin_count

    rate_maps = _stack_list_column(table=cells, column=TuningColumn.RATE_MAP, target_length=bin_count)
    cell_population, _ = report.resolve_population_masks(
        trial_type=trial_type,
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )
    order = _resolve_place_cell_order(table=cells)

    if not sort_by_position:
        # noinspection PyTypeChecker
        order = np.arange(rate_maps.shape[0], dtype=np.int64)

    if show_only_place_cells:
        order = order[np.isin(order, np.flatnonzero(cell_population))]

    sorted_data = rate_maps[order, :]
    if sorted_data.size == 0:
        sorted_data = rate_maps[:0, :]

    minimum_value = float(np.nanquantile(sorted_data, minimum_percentile)) if sorted_data.size > 0 else 0.0
    maximum_value = float(np.nanquantile(sorted_data, maximum_percentile)) if sorted_data.size > 0 else 1.0

    strip_columns = _active_significance_columns(
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        table=cells,
    )
    strip_count = len(strip_columns) if show_significance_strip else 0
    figure = _make_heatmap_figure(strip_count=strip_count, figure_dpi=figure_dpi)
    strip_axes, axes, colorbar_axes = _layout_heatmap_axes(
        figure=figure,
        strip_count=strip_count,
        include_colorbar=show_color_bar,
    )

    population_label = _compose_population_label(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
    )
    if title is not None:
        axes.set_title(f"{title} — {population_label} (n={order.size})", fontsize=8)
    elif show_only_place_cells:
        axes.set_title(f"{population_label} (n={order.size})", fontsize=8)

    extent: tuple[float, float, float, float] = (
        0.0,
        float(bin_size_cm * bin_count),
        float(sorted_data.shape[0]),
        0.0,
    )
    image = axes.imshow(
        sorted_data,
        cmap=cmap,
        extent=extent,
        interpolation="none",
        vmin=minimum_value,
        vmax=maximum_value,
        origin="upper",
    )
    axes.set_aspect("auto")
    axes.set_xlabel("Position (cm)")
    axes.set_ylabel("Cell number")
    if show_significance_strip and strip_axes:
        _render_significance_strips(
            figure=figure,
            strip_axes=strip_axes,
            strip_columns=strip_columns,
            table=cells,
            ordered_indices=order,
        )

    track_length_cm = trial_summary.track_length_cm
    # noinspection PyTypeChecker
    x_ticks: NDArray[np.float64] = np.arange(0, track_length_cm + 1, _PLOT_TICK_INTERVAL_CM)
    axes.set_xticks(x_ticks)

    if show_color_bar and colorbar_axes is not None:
        color_bar = figure.colorbar(image, cax=colorbar_axes)
        color_bar.set_label("ΔF/F₀")
        cbar_min = np.floor(minimum_value / 0.5) * 0.5
        cbar_max = np.ceil(maximum_value / 0.5) * 0.5
        # noinspection PyTypeChecker
        cbar_ticks: NDArray[np.float64] = np.arange(cbar_min, cbar_max, 0.5)
        color_bar.set_ticks(cbar_ticks.tolist())

    return figure


def plot_reward_com_histogram(
    report: TuningReport,
    *,
    trial_type: str,
    bin_count: int = 20,
    title: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots the spatially significant COM histogram with the fitted uniform + Gaussian mixture overlay for
    ``trial_type``.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )
    valid_centers = centers_of_mass[is_significant & (centers_of_mass >= 0.0)]

    figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
    # noinspection PyTypeChecker
    hist_bins: NDArray[np.float64] = np.linspace(0, trial_summary.track_length_cm, bin_count + 1)
    axes.hist(valid_centers, bins=hist_bins.tolist(), color="0.7", edgecolor="0.5", density=True, label="Observed COMs")

    # noinspection PyTypeChecker
    positions: NDArray[np.float64] = np.linspace(0, trial_summary.track_length_cm, 200)
    # noinspection PyTypeChecker
    uniform_density: NDArray[np.float64] = np.full_like(positions, 1.0 / trial_summary.track_length_cm)
    gaussian_std = max(trial_summary.gaussian_std_cm, 1.0)
    gaussian_density_reward = np.exp(-0.5 * ((positions - trial_summary.gaussian_mean_cm) / gaussian_std) ** 2) / (
        gaussian_std * np.sqrt(2.0 * np.pi)
    )
    track_start_std = max(trial_summary.track_start_std_cm, 1.0)
    gaussian_density_start = np.exp(-0.5 * ((positions - 0.0) / track_start_std) ** 2) / (
        track_start_std * np.sqrt(2.0 * np.pi)
    )
    track_end_std = max(trial_summary.track_end_std_cm, 1.0)
    gaussian_density_end = np.exp(-0.5 * ((positions - trial_summary.track_length_cm) / track_end_std) ** 2) / (
        track_end_std * np.sqrt(2.0 * np.pi)
    )
    uniform_weight = max(
        1.0 - trial_summary.mixture_weight - trial_summary.track_start_weight - trial_summary.track_end_weight,
        0.0,
    )

    uniform_band = uniform_weight * uniform_density
    landmark_band = uniform_band + (
        trial_summary.track_start_weight * gaussian_density_start
        + trial_summary.track_end_weight * gaussian_density_end
    )
    mixture_density = landmark_band + trial_summary.mixture_weight * gaussian_density_reward

    axes.fill_between(positions, 0, uniform_band, alpha=0.3, color="lightblue", label="Uniform (place cells)")
    axes.fill_between(positions, uniform_band, landmark_band, alpha=0.3, color="khaki", label="Track-end Gaussians")
    axes.fill_between(
        positions, landmark_band, mixture_density, alpha=0.4, color="mediumpurple", label="Reward Gaussian"
    )
    axes.plot(positions, mixture_density, color="black", linewidth=1.5, label="Mixture fit")
    axes.axvline(
        x=trial_summary.reward_position_cm,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Reward location",
    )

    axes.set_xlabel("Track Position (cm)")
    axes.set_ylabel("Density")
    axes.legend(fontsize=7, loc="upper left")
    annotation_text = (
        f"Significant: {trial_summary.spatially_significant_count}/{report.summary.cell_count} cells\n"
        f"Mixture weight: {trial_summary.mixture_weight:.1%} reward\n"
        f"Gaussian center: {trial_summary.gaussian_mean_cm:.0f} cm "
        f"(SD {trial_summary.gaussian_std_cm:.0f} cm)"
    )
    axes.text(
        0.98,
        0.95,
        annotation_text,
        transform=axes.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )
    if title:
        axes.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def plot_rate_map_heatmap(
    report: TuningReport,
    *,
    trial_type: str,
    title: str | None = None,
    require_place: bool = True,
    require_stable: bool = True,
    require_peak_significant: bool = True,
    mutually_exclusive: bool = True,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots row-normalized rate maps for reward cells and place cells side by side, sorted by COM, restricted
    to ``trial_type``.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    rate_maps = _stack_list_column(table=cells, column=TuningColumn.RATE_MAP, target_length=trial_summary.bin_count)
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )

    place_mask, reward_mask = report.resolve_population_masks(
        trial_type=trial_type,
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )

    reward_zone_half = report.summary.reward_configuration.reward_zone_width / 2.0
    reward_left = trial_summary.reward_position_cm - reward_zone_half
    reward_right = trial_summary.reward_position_cm + reward_zone_half

    figure, (axes_reward, axes_place) = plt.subplots(
        1, 2, figsize=(12, 6), facecolor="white", dpi=figure_dpi, sharey=False
    )

    place_label = "Place Cells — " + _compose_population_label(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
    )
    for axes, mask, panel_title in [
        (axes_reward, reward_mask, "Reward Cells"),
        (axes_place, place_mask, place_label),
    ]:
        maps = rate_maps[mask]
        coms = centers_of_mass[mask]
        # noinspection PyTypeChecker
        sort_order: NDArray[np.int64] = np.argsort(coms)
        sorted_maps = maps[sort_order]

        row_maxima = sorted_maps.max(axis=1, keepdims=True)
        row_maxima[row_maxima == 0] = 1.0
        normalized_maps = sorted_maps / row_maxima

        extent = [0, trial_summary.bin_size_cm * sorted_maps.shape[1], normalized_maps.shape[0], 0]
        axes.imshow(
            normalized_maps,
            cmap="gray_r",
            extent=extent,
            interpolation="none",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            aspect="auto",
        )
        axes.axvline(x=reward_left, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes.axvline(x=reward_right, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes.set_xlabel("Track Position (cm)")
        axes.set_xticks(np.arange(0, trial_summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
        axes.set_title(f"{panel_title} (n={int(np.sum(mask))})", fontsize=9)

    axes_reward.set_ylabel("Neuron (sorted by COM)")
    if title:
        figure.suptitle(title, fontsize=9)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        figure.tight_layout()
    return figure


def plot_population_activity_by_position(
    report: TuningReport,
    *,
    trial_type: str,
    title: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots mean population fluorescence vs track position for ``trial_type``, contrasting reward-predictive
    against all spatially modulated cells.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    rate_maps = _stack_list_column(table=cells, column=TuningColumn.RATE_MAP, target_length=trial_summary.bin_count)
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = cells[TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value].to_numpy()

    bin_centers = (np.arange(trial_summary.bin_count) + 0.5) * trial_summary.bin_size_cm
    all_significant = is_significant
    reward_predictive_mask = all_significant & is_reward_proximal & is_position_glm_significant

    figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
    if int(np.sum(all_significant)) > 0:
        mean_all = np.mean(rate_maps[all_significant], axis=0)
        axes.fill_between(bin_centers, 0, mean_all, color="0.8", alpha=0.6)
        axes.plot(
            bin_centers,
            mean_all,
            color="0.5",
            linewidth=1.5,
            label=f"All spatially modulated (n={int(np.sum(all_significant))})",
        )

    if int(np.sum(reward_predictive_mask)) > 0:
        mean_predictive = np.mean(rate_maps[reward_predictive_mask], axis=0)
        axes.fill_between(bin_centers, 0, mean_predictive, color="mediumpurple", alpha=0.3)
        axes.plot(
            bin_centers,
            mean_predictive,
            color="darkviolet",
            linewidth=2.5,
            label=f"Reward-predictive (GLM, n={int(np.sum(reward_predictive_mask))})",
        )

    axes.axvline(
        x=trial_summary.reward_position_cm,
        color="red",
        linestyle="-",
        linewidth=2.0,
        alpha=0.8,
        label="Reward location",
    )
    axes.set_xlabel("Track Position (cm)")
    axes.set_ylabel("Average Fluorescence (dF/F)")
    axes.legend(fontsize=7, loc="upper left")
    axes.set_xticks(np.arange(0, trial_summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
    if title:
        axes.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def plot_speed_and_activity_by_position(
    report: TuningReport,
    *,
    trial_type: str,
    session: DatasetSession,
    title: str | None = None,
    figure_dpi: int = 150,
    position_sigma_cm: float = 5.0,
) -> plt.Figure:
    """Plots binned-speed alongside reward-predictive cell activity for ``trial_type``. Reads ``data.feather``
    to bin speed by position; reward-predictive activity comes from the persisted rate maps.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    rate_maps = _stack_list_column(table=cells, column=TuningColumn.RATE_MAP, target_length=trial_summary.bin_count)
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = cells[TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value].to_numpy()
    reward_predictive_mask = is_significant & is_reward_proximal & is_position_glm_significant

    if int(np.sum(reward_predictive_mask)) == 0:
        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes.text(
            0.5,
            0.5,
            "No reward-predictive cells found",
            transform=axes.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        return figure

    binned_speed = _bin_speed_by_position(
        session=session,
        trial_type=trial_type,
        track_length_cm=trial_summary.track_length_cm,
        bin_size_cm=trial_summary.bin_size_cm,
        bin_count=trial_summary.bin_count,
    )
    sigma_bins = position_sigma_cm / trial_summary.bin_size_cm
    # noinspection PyTypeChecker
    smoothed_speed: NDArray[np.float32] = gaussian_filter1d(input=binned_speed, sigma=sigma_bins, mode="wrap")

    bin_centers = (np.arange(trial_summary.bin_count) + 0.5) * trial_summary.bin_size_cm
    mean_activity = np.mean(rate_maps[reward_predictive_mask], axis=0)

    figure, axes_speed = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
    axes_activity = axes_speed.twinx()
    axes_speed.plot(bin_centers, smoothed_speed, color="0.4", linewidth=1.5, label="Mean speed")
    axes_speed.set_xlabel("Track Position (cm)")
    axes_speed.set_ylabel("Speed (cm/s)", color="0.4")
    axes_speed.tick_params(axis="y", labelcolor="0.4")

    predictive_count = int(np.sum(reward_predictive_mask))
    axes_activity.plot(
        bin_centers,
        mean_activity,
        color="darkviolet",
        linewidth=2.0,
        label=f"Reward-predictive (n={predictive_count})",
    )
    axes_activity.set_ylabel("Mean Fluorescence (dF/F)", color="darkviolet")
    axes_activity.tick_params(axis="y", labelcolor="darkviolet")

    reward_zone_half = report.summary.reward_configuration.reward_zone_width / 2.0
    axes_speed.axvspan(
        trial_summary.reward_position_cm - reward_zone_half,
        trial_summary.reward_position_cm + reward_zone_half,
        alpha=0.1,
        color="red",
        label="Reward zone",
    )
    axes_speed.set_xticks(np.arange(0, trial_summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
    lines_speed, labels_speed = axes_speed.get_legend_handles_labels()
    lines_activity, labels_activity = axes_activity.get_legend_handles_labels()
    axes_speed.legend(lines_speed + lines_activity, labels_speed + labels_activity, fontsize=7, loc="upper left")
    if title:
        axes_speed.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def plot_per_trial_activity(
    report: TuningReport,
    *,
    trial_type: str,
    session: DatasetSession,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    title: str | None = None,
    require_place: bool = True,
    require_stable: bool = True,
    require_peak_significant: bool = True,
    mutually_exclusive: bool = True,
    figure_dpi: int = 150,
    position_bin_size_cm: float = 2.0,
    position_sigma_cm: float = 3.0,
    slowing_threshold_cm_s: float = 10.0,
) -> plt.Figure:
    """Plots per-trial activity heatmaps for an example reward-predictive cell and an example place cell
    drawn from ``trial_type``, with slowing-onset markers overlaid. Reads ``data.feather`` for the raw
    fluorescence and per-trial speed time series.
    """
    trial_summary = report.trial_summary(trial_type=trial_type)
    cells = report.trial_cells(trial_type=trial_type)
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = cells[TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    cv_partial_r2: NDArray[np.float32] = (
        cells[TuningColumn.CV_POSITION_PARTIAL_R2.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )

    place_mask, _ = report.resolve_population_masks(
        trial_type=trial_type,
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )
    predictive_mask = is_significant & is_reward_proximal & is_position_glm_significant
    # noinspection PyTypeChecker
    predictive_indices: NDArray[np.int64] = np.argwhere(predictive_mask).flatten()
    # noinspection PyTypeChecker
    place_indices: NDArray[np.int64] = np.argwhere(place_mask).flatten()

    if predictive_indices.size == 0 or place_indices.size == 0:
        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes.text(
            0.5,
            0.5,
            "Insufficient cells for comparison",
            transform=axes.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        return figure

    best_predictive = int(predictive_indices[np.argmax(cv_partial_r2[predictive_indices])])
    track_midpoint = trial_summary.track_length_cm / 2.0
    place_distances = np.abs(centers_of_mass[place_indices] - track_midpoint)
    best_place = int(place_indices[np.argmin(place_distances)])

    run_session = assemble_run_session_data(
        session_path=session.session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
    )

    # noinspection PyTypeChecker
    unique_trials: NDArray[np.int64] = np.unique(run_session.trial_ids)
    trial_count = int(unique_trials.size)
    # noinspection PyTypeChecker
    bin_edges: NDArray[np.float32] = np.arange(
        0, trial_summary.track_length_cm + position_bin_size_cm, position_bin_size_cm, dtype=np.float32
    )
    bin_count = len(bin_edges) - 1
    sigma_bins = position_sigma_cm / position_bin_size_cm

    reward_zone_half = report.summary.reward_configuration.reward_zone_width / 2.0
    reward_left = trial_summary.reward_position_cm - reward_zone_half
    reward_right = trial_summary.reward_position_cm + reward_zone_half
    pre_reward_start = trial_summary.reward_position_cm - report.summary.reward_configuration.pre_reward_window

    figure, (axes_predictive, axes_place) = plt.subplots(1, 2, figsize=(14, 8), facecolor="white", dpi=figure_dpi)

    cell_panels = [
        (axes_predictive, best_predictive, "Reward-predictive", "Purples"),
        (axes_place, best_place, "Place cell", "Blues"),
    ]
    for axes, cell_index, label, colormap in cell_panels:
        # noinspection PyTypeChecker
        activity_image: NDArray[np.float32] = np.zeros((trial_count, bin_count), dtype=np.float32)
        # noinspection PyTypeChecker
        slowing_onsets: NDArray[np.float32] = np.full(trial_count, np.nan, dtype=np.float32)

        for trial_index, trial_id in enumerate(unique_trials):
            # noinspection PyTypeChecker
            trial_mask: NDArray[np.bool_] = run_session.trial_ids == trial_id
            trial_positions = run_session.position[trial_mask]
            trial_speeds = run_session.speed[trial_mask]
            trial_fluorescence = run_session.fluorescence[cell_index, trial_mask]

            # noinspection PyTypeChecker
            trial_bin_indices: NDArray[np.int64] = np.clip(
                np.searchsorted(bin_edges, trial_positions, side="right") - 1, 0, bin_count - 1
            )
            # noinspection PyTypeChecker
            activity_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
            # noinspection PyTypeChecker
            activity_counts: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)
            np.add.at(activity_sums, trial_bin_indices, trial_fluorescence)
            np.add.at(activity_counts, trial_bin_indices, 1)
            # noinspection PyTypeChecker
            valid_bins: NDArray[np.bool_] = activity_counts > 0
            activity_image[trial_index, valid_bins] = activity_sums[valid_bins] / activity_counts[valid_bins]

            # noinspection PyTypeChecker
            pre_reward: NDArray[np.bool_] = (trial_positions >= pre_reward_start) & (
                trial_positions < trial_summary.reward_position_cm
            )
            # noinspection PyTypeChecker
            below_threshold: NDArray[np.bool_] = pre_reward & (trial_speeds < slowing_threshold_cm_s)
            if np.any(below_threshold):
                slowing_onsets[trial_index] = trial_positions[below_threshold][0]

        activity_image = gaussian_filter1d(input=activity_image, sigma=sigma_bins, axis=1, mode="wrap")
        row_maxima = activity_image.max(axis=1, keepdims=True)
        row_maxima[row_maxima == 0] = 1.0
        normalized_activity = activity_image / row_maxima

        axes.imshow(
            normalized_activity,
            cmap=colormap,
            extent=[0, trial_summary.track_length_cm, trial_count, 0],
            interpolation="none",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            aspect="auto",
            alpha=0.85,
        )

        # noinspection PyTypeChecker
        onset_trials: NDArray[np.int64] = np.argwhere(~np.isnan(slowing_onsets)).flatten()
        for marker_index, trial_index in enumerate(onset_trials):
            onset_label = f"Slowing onset (<{slowing_threshold_cm_s:.0f} cm/s)" if marker_index == 0 else None
            axes.plot(
                slowing_onsets[trial_index],
                trial_index + 0.5,
                marker="|",
                color="black",
                markersize=6,
                markeredgewidth=1.5,
                label=onset_label,
            )

        axes.axvline(x=reward_left, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes.axvline(x=reward_right, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes.legend(fontsize=6, loc="upper left")
        axes.set_xlabel("Track Position (cm)")
        axes.set_xticks(np.arange(0, trial_summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
        cell_com = centers_of_mass[cell_index]
        cell_partial_r2 = float(cv_partial_r2[cell_index]) if not np.isnan(cv_partial_r2[cell_index]) else 0.0
        axes.set_title(f"{label} (cell {cell_index}, COM={cell_com:.0f} cm, ΔR²={cell_partial_r2:.2f})", fontsize=9)

    axes_predictive.set_ylabel("Trial")
    if title:
        figure.suptitle(title, fontsize=9)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        figure.tight_layout()
    return figure


def plot_per_day_sorted_rate_maps(
    sessions: tuple[DatasetSession, ...],
    *,
    trial_type: str | None = None,
    display_sessions: tuple[int, ...] | None = None,
    classifier: str = "place",
    cmap: str = "magma",
    show_cue_boundaries: bool = True,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots row-normalized rate maps for each displayed session sorted independently by that session's peak.

    Notes:
        Each panel filters the session's persisted ``tuning_cells.feather`` to cells classified by the
        requested classifier (``IS_PLACE`` / ``IS_REWARD_CELL`` / ``IS_STRICT_PLACE``) on that day,
        sorts them by their rate-map peak position on the same day, and renders the row-normalized
        rate maps. There is no across-session correspondence: each panel uses the within-session
        classification only, so the figure shows the population's day-level tuning band rather than
        per-cell drift. Cell counts are reported per panel because the active-cell set differs across
        days. Pair with ``..drift.plotting.plot_reference_day_sorted_rate_maps`` for the per-cell
        drift view.

        Sessions are addressed by 1-indexed chronological session number (``1`` is the first session
        in ``sessions``, ``2`` is the second, etc.). Out-of-range entries are silently skipped, and
        duplicates that resolve to the same session are deduplicated.

        ``trial_type`` is resolved once from the first available session's tuning feather when not
        supplied by the caller, then applied uniformly to every panel.

        Sessions render in a single chronological row at a tall-rectangular panel aspect ratio. A
        shared horizontal colorbar at the bottom of the figure documents the 0..1 row-normalized
        intensity scale that every panel shares.

    Args:
        sessions: Chronologically ordered DatasetSession entries to render.
        trial_type: Trial type to evaluate; when ``None``, defaults to the first trial type present in
            the first session's tuning feather. Applied uniformly across panels.
        display_sessions: 1-indexed session numbers to render as columns. Defaults to every session
            in the supplied tuple — pass an explicit selection to render a subset.
        classifier: Within-session classification used to pick cells (``"place"``, ``"reward"``,
            ``"strict_place"``).
        cmap: Matplotlib colormap name for the rate-map intensities. Default ``"magma"`` is a
            perceptually-uniform colormap with strong contrast on dark backgrounds.
        show_cue_boundaries: When True, draw cyan dotted verticals on every panel at the start and
            end of the cue zone that contains the trigger zone for that session. Cue layout is
            derived once from the first available session's ``data.feather`` (using the ``cue`` and
            ``distance_cm`` columns) and reused for every panel; the per-session trigger zone center
            then selects which cue interval to highlight.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    session_count: int = len(sessions)

    def _resolve_indices(requested: tuple[int, ...]) -> tuple[int, ...]:
        """Maps each 1-indexed session number to a 0-indexed position; out-of-range entries are skipped."""
        resolved: list[int] = []
        for target in requested:
            idx = int(target) - 1
            if 0 <= idx < session_count and idx not in resolved:
                resolved.append(idx)
        return tuple(resolved)

    if display_sessions is None:
        # Renders every session by default; subsetting is opt-in through ``display_sessions``.
        display_session_indices: tuple[int, ...] = tuple(range(session_count))
    else:
        display_session_indices = _resolve_indices(display_sessions)

    n_panels: int = len(display_session_indices)
    n_cols: int = max(n_panels, 1)

    figure, axes_array = plt.subplots(
        1, n_cols,
        figsize=(2.0 * n_cols + 1.2, 4.5),
        facecolor="white", dpi=figure_dpi, squeeze=False,
        layout="constrained",
    )
    if session_count == 0 or n_panels == 0:
        axes_array[0, 0].text(0.5, 0.5, "No sessions supplied", ha="center", va="center",
                              transform=axes_array[0, 0].transAxes)
        for axes in axes_array.flat:
            axes.set_axis_off()
        return figure

    classifier_column = {
        "place": TuningColumn.IS_PLACE.value,
        "reward": TuningColumn.IS_REWARD_CELL.value,
        "strict_place": TuningColumn.IS_STRICT_PLACE.value,
    }.get(classifier, TuningColumn.IS_PLACE.value)
    label = {"place": "place", "reward": "reward", "strict_place": "strict-place"}.get(classifier, classifier)

    # Pre-loads each session's filtered tuning frame once so the trial-type / track-length resolution
    # logic stays out of the per-panel render loop. The per-session trigger zone is also captured here
    # so the render loop can overlay zone bounds without re-reading geometry on every panel. The
    # per-session ``cue_offset_cm`` lets the cue overlay shift data-derived runtime-trial-rel boundaries
    # back into canonical coordinates so rate maps and overlays share the same coord frame post-realign.
    frames_per_session: dict[int, pl.DataFrame | None] = {}
    trigger_zones: dict[int, tuple[float, float] | None] = {}
    cue_offset_per_session: dict[int, float] = {}
    bin_count_reference: int | None = None
    track_length_reference: float | None = None
    resolved_trial_type: str | None = trial_type
    for sess_idx in display_session_indices:
        path = sessions[sess_idx].tuning_cells_path
        if not path.exists():
            frames_per_session[sess_idx] = None
            trigger_zones[sess_idx] = None
            continue
        frame = pl.read_ipc(source=path, memory_map=True)
        # Resolves the trial type from the first available frame when the caller did not specify one.
        if resolved_trial_type is None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            unique_trial_types = frame[TuningColumn.TRIAL_TYPE.value].unique().to_list()
            if unique_trial_types:
                resolved_trial_type = str(unique_trial_types[0])
        if resolved_trial_type is not None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type)
        frame = frame.sort(TuningColumn.CELL_ID.value)
        frames_per_session[sess_idx] = frame
        if bin_count_reference is None and frame.height > 0:
            # noinspection PyTypeChecker
            first_rate_map = np.asarray(
                frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
            )
            if first_rate_map.size > 0:
                bin_count_reference = int(first_rate_map.shape[1])
        # Reads the per-session trigger zone (and the track-length reference, when not yet set) from
        # the trial geometry. This runs every iteration because trigger zones can shift across
        # sessions in protocols like the void reward shift.
        geometry_path = sessions[sess_idx].geometry_path
        if geometry_path.exists() and resolved_trial_type is not None:
            geometry = TrialGeometry.from_yaml(file_path=geometry_path)
            entry = geometry.entries.get(resolved_trial_type)
            if entry is not None:
                trigger_zones[sess_idx] = (
                    float(entry.stimulus_trigger_zone_start_cm),
                    float(entry.stimulus_trigger_zone_end_cm),
                )
                cue_offset_per_session[sess_idx] = float(entry.cue_offset_cm)
                if track_length_reference is None:
                    track_length_reference = float(entry.trial_length_cm)
            else:
                trigger_zones[sess_idx] = None
                cue_offset_per_session[sess_idx] = 0.0
        else:
            trigger_zones[sess_idx] = None
            cue_offset_per_session[sess_idx] = 0.0

    bin_count: int = bin_count_reference if bin_count_reference is not None else 0
    track_length_cm: float = (
        track_length_reference if track_length_reference is not None else float(bin_count)
    )
    bin_size_cm: float = track_length_cm / bin_count if bin_count > 0 else 1.0

    # Derives cue-zone boundaries once per call from the first available session's ``data.feather``.
    # The cue layout is constant per trial type (each trial type owns a fixed Segment.cue_sequence
    # upstream), so a single canonical-trial walk supplies the boundary positions for every panel.
    cue_boundaries_cm: tuple[float, ...] = ()
    if show_cue_boundaries and resolved_trial_type is not None and track_length_cm > 0:
        for sess_idx in display_session_indices:
            data_path = sessions[sess_idx].data_path
            if not data_path.exists():
                continue
            cue_boundaries_cm = _derive_cue_boundaries_cm(
                data_path=data_path,
                trial_type=resolved_trial_type,
                track_length_cm=track_length_cm,
            )
            if cue_boundaries_cm:
                break

    last_image = None
    for column_position, sess_idx in enumerate(display_session_indices):
        axes = axes_array[0, column_position]
        frame = frames_per_session.get(sess_idx)
        if frame is None or frame.height == 0 or classifier_column not in frame.columns:
            axes.text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.transAxes)
            axes.set_xticks([])
            axes.set_yticks([])
            continue
        # noinspection PyTypeChecker
        active_mask: NDArray[np.bool_] = frame[classifier_column].to_numpy().astype(np.bool_, copy=False)
        if not active_mask.any():
            axes.text(0.5, 0.5, "no active cells", ha="center", va="center", transform=axes.transAxes)
            axes.set_xticks([])
            axes.set_yticks([])
            continue
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.asarray(
            frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
        )
        active_rate_maps = rate_maps[active_mask]
        finite_for_argmax = np.where(np.isfinite(active_rate_maps), active_rate_maps, -np.inf)
        peak_bins = np.argmax(finite_for_argmax, axis=1)
        order = np.argsort(peak_bins)
        sorted_maps = active_rate_maps[order]
        row_max = np.nanmax(sorted_maps, axis=1, keepdims=True)
        row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
        normalized = np.clip(sorted_maps / row_max, 0.0, 1.0)
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        n_total: int = sorted_maps.shape[0]
        last_image = axes.imshow(
            normalized, aspect="auto", origin="upper", cmap=cmap,
            extent=[0, bin_count * bin_size_cm, n_total, 0], vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        # Overlays the start / end of the cue zone holding the trigger zone as cyan dotted verticals.
        # The cue layout is constant per trial type, so the same data-derived boundary set is reused
        # for every panel; per-session ``cue_offset_cm`` shifts the boundaries from runtime-trial-rel
        # into canonical coordinates so the overlay matches realigned rate maps. The trigger zone
        # center selects which canonical cue interval to bracket. Drawn before the trigger zone so
        # the red dashed lines sit on top.
        zone = trigger_zones.get(sess_idx)
        if cue_boundaries_cm and zone is not None:
            session_cue_boundaries = _shift_cue_boundaries_to_canonical(
                cue_boundaries=cue_boundaries_cm,
                cue_offset_cm=cue_offset_per_session.get(sess_idx, 0.0),
                track_length_cm=track_length_cm,
            )
            cue_left, cue_right = _cue_zone_around_trigger(
                cue_boundaries=session_cue_boundaries,
                trigger_center=0.5 * (zone[0] + zone[1]),
                track_length_cm=track_length_cm,
            )
            axes.axvline(cue_left, color="cyan", linestyle=":", linewidth=1.4, alpha=0.9)
            axes.axvline(cue_right, color="cyan", linestyle=":", linewidth=1.4, alpha=0.9)
        # Overlays the per-session trigger zone as red dashed verticals so the band's relationship to
        # the reward landmark is visible. Drawn last so the lines sit on top of everything.
        if zone is not None:
            zone_start, zone_end = zone
            axes.axvline(zone_start, color="red", linestyle="--", linewidth=1.0, alpha=0.85)
            axes.axvline(zone_end, color="red", linestyle="--", linewidth=1.0, alpha=0.85)
        axes.set_xlim(0, bin_count * bin_size_cm)
        axes.set_title(f"Session {sess_idx + 1} (n={n_total})", fontsize=10)
        axes.set_xlabel("Position (cm)", fontsize=9)
        # Drops concrete y-tick numbers everywhere; absolute cell counts vary across sessions and the
        # per-panel ``(n=X)`` already documents that. The shared y-axis label still annotates the sort
        # convention so the band shape stays interpretable.
        axes.set_yticks([])
        if column_position == 0:
            axes.set_ylabel(f"{label.capitalize()} cell (per-day peak sort)", fontsize=9)

    if last_image is not None:
        # Shared horizontal colorbar at the bottom of the figure documents the row-normalized scale.
        color_bar = figure.colorbar(
            last_image,
            ax=axes_array.ravel().tolist(),
            orientation="horizontal",
            shrink=0.5,
            aspect=40,
            pad=0.04,
        )
        color_bar.set_label("Row-normalized rate (peak = 1)", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)

    animal_prefix: str = f"Animal {animal_id} " if animal_id is not None else ""
    trial_label: str = f" — trial type {resolved_trial_type!r}" if resolved_trial_type is not None else ""
    figure.suptitle(
        f"{animal_prefix}per-day-sorted rate maps ({label} cells){trial_label}",
        fontsize=12,
    )
    return figure


def _derive_cue_boundaries_cm(
    data_path: "Path",
    trial_type: str,
    track_length_cm: float,
) -> tuple[float, ...]:
    """Returns trial-relative cm positions where the cue identity changes within a representative trial.

    Notes:
        Reads the session's ``data.feather`` and walks the first run-state trial of ``trial_type``,
        treating each step where the ``cue`` column changes as a cue-zone boundary. The cue layout
        is constant per trial type (each trial type owns a fixed cue sequence upstream), so a single
        canonical-trial walk supplies boundary positions for every panel that shares that trial type.
        Boundary positions are clipped to the open interval ``(0, track_length_cm)`` so the start /
        end of the track are not redundantly drawn over the figure edges.

    Args:
        data_path: Path to the session's ``data.feather``.
        trial_type: Trial type whose canonical cue layout to extract.
        track_length_cm: Trial length used to clip boundaries to in-range positions.

    Returns:
        A tuple of trial-relative cm positions of cue-identity transitions.
    """
    df = pl.read_ipc(
        source=data_path,
        columns=[
            DatasetColumn.SYSTEM_STATE.value,
            DatasetColumn.TRIAL.value,
            DatasetColumn.TRIAL_TYPE.value,
            DatasetColumn.DISTANCE_CM.value,
            DatasetColumn.CUE.value,
        ],
        memory_map=True,
    )
    run = df.filter(
        (pl.col(DatasetColumn.SYSTEM_STATE.value) == "run")
        & (pl.col(DatasetColumn.TRIAL.value) < 255)
        & (pl.col(DatasetColumn.TRIAL_TYPE.value) == trial_type)
    )
    if run.height == 0:
        return ()
    first_trial_id = int(run[DatasetColumn.TRIAL.value][0])
    first_trial = run.filter(pl.col(DatasetColumn.TRIAL.value) == first_trial_id)
    if first_trial.height == 0:
        return ()
    # noinspection PyTypeChecker
    distance: NDArray[np.float32] = (
        first_trial[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    cues: NDArray[np.int32] = (
        first_trial[DatasetColumn.CUE.value].to_numpy().astype(np.int32, copy=False)
    )
    if distance.size == 0:
        return ()
    relative = distance - distance[0]
    # noinspection PyTypeChecker
    transitions: NDArray[np.int64] = np.where(np.diff(cues) != 0)[0] + 1
    if transitions.size == 0:
        return ()
    boundaries = relative[transitions]
    # Drops the trivial 0 / track_length boundaries; only interior cue transitions are useful overlays.
    return tuple(
        float(position)
        for position in boundaries
        if 0.0 < float(position) < track_length_cm
    )


def _shift_cue_boundaries_to_canonical(
    cue_boundaries: tuple[float, ...],
    cue_offset_cm: float,
    track_length_cm: float,
) -> tuple[float, ...]:
    """Shifts runtime-trial-relative cue boundaries by ``+cue_offset_cm`` into canonical coordinates.

    Notes:
        The runtime starts each trial mid-first-cue (offset by ``cue_offset_cm`` into the canonical cue
        sequence), so cue transitions sampled in ``data.feather`` lie at runtime-trial-rel positions.
        Adding the offset shifts each transition into the canonical frame; boundaries that wrap past
        the track end are folded back via modulo. Returns the input unchanged when ``cue_offset_cm``
        is ``0`` (the canonical and runtime frames coincide).
    """
    if cue_offset_cm == 0.0 or not cue_boundaries:
        return cue_boundaries
    if track_length_cm <= 0.0:
        return cue_boundaries
    # Folds boundaries that cross the track wrap so every position stays in ``[0, track_length_cm)``,
    # then sorts the result so consumers can walk them linearly.
    shifted = sorted(
        float((boundary + cue_offset_cm) % track_length_cm) for boundary in cue_boundaries
    )
    return tuple(shifted)


def _cue_zone_around_trigger(
    cue_boundaries: tuple[float, ...],
    trigger_center: float,
    track_length_cm: float,
) -> tuple[float, float]:
    """Returns the (left, right) cm bounds of the cue zone containing ``trigger_center``.

    Notes:
        ``cue_boundaries`` carries the interior cue-transition positions only; this helper extends
        them with the implicit endpoints ``0`` and ``track_length_cm`` so every position on the track
        falls into exactly one cue zone. When ``trigger_center`` lies on a boundary, the zone to its
        right is selected (consistent with ``boundaries[i] <= center < boundaries[i + 1]``).
    """
    full_boundaries: tuple[float, ...] = (0.0,) + cue_boundaries + (float(track_length_cm),)
    for index in range(len(full_boundaries) - 1):
        left = full_boundaries[index]
        right = full_boundaries[index + 1]
        if left <= trigger_center < right:
            return left, right
    return 0.0, float(track_length_cm)


def _stack_list_column(table: pl.DataFrame, column: TuningColumn, target_length: int) -> NDArray[np.float32]:
    """Materializes a List(Float32) column into a (cell_count, target_length) numpy array, padding with NaN
    rows for nulls.
    """
    cell_count = table.height
    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.full((cell_count, target_length), np.nan, dtype=np.float32)
    values = table[column.value].to_list()
    for cell_index, vector in enumerate(values):
        if vector is None:
            continue
        as_array = np.asarray(vector, dtype=np.float32)
        clipped_length = min(target_length, as_array.size)
        output[cell_index, :clipped_length] = as_array[:clipped_length]
    return output


def _resolve_place_cell_order(table: pl.DataFrame) -> NDArray[np.int64]:
    """Returns a length-cell_count permutation that sorts cells by their place-field center along the track,
    placing cells without place fields after the sorted block.
    """
    cell_count = table.height
    # noinspection PyTypeChecker
    sort_keys: NDArray[np.float32] = np.full(cell_count, np.inf, dtype=np.float32)

    pf_centers = table[TuningColumn.PF_CENTER_CM.value].to_list()
    pf_intensities = table[TuningColumn.PF_MEAN_INTENSITY.value].to_list()
    for cell_index in range(cell_count):
        centers = pf_centers[cell_index]
        intensities = pf_intensities[cell_index]
        if not centers:
            continue
        intensity_array = np.asarray(intensities, dtype=np.float32)
        sort_keys[cell_index] = float(centers[int(np.argmax(intensity_array))])
    # noinspection PyTypeChecker
    return np.argsort(sort_keys, kind="stable").astype(np.int64)


def _bin_speed_by_position(
    session: DatasetSession,
    trial_type: str,
    track_length_cm: float,
    bin_size_cm: float,
    bin_count: int,
) -> NDArray[np.float32]:
    """Returns the per-bin mean running speed for the session, computed off ``data.feather``.

    Notes:
        Adds the trial type's ``cue_offset_cm`` to the cumulative distance before the modulo so the
        x-axis matches the canonical cue layout used by the rate-map binning. With ``cue_offset_cm == 0``
        the shift is a no-op and the result is identical to a plain ``distance % track_length_cm``.
    """
    geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
    cue_offset_cm = float(geometry.entries[trial_type].cue_offset_cm)

    df = pl.read_ipc(
        source=session.data_path,
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.DISTANCE_CM.value, DatasetColumn.SPEED_CM_S.value],
        memory_map=True,
    )
    df = trim_acquisition_warmup(dataframe=df)

    # noinspection PyTypeChecker
    distance: NDArray[np.float32] = df[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    speed: NDArray[np.float32] = df[DatasetColumn.SPEED_CM_S.value].to_numpy().astype(np.float32, copy=False)

    # noinspection PyTypeChecker
    position: NDArray[np.float32] = (
        (distance + np.float32(cue_offset_cm)) % np.float32(track_length_cm)
    ).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    bin_edges: NDArray[np.float32] = np.arange(0.0, track_length_cm + bin_size_cm, bin_size_cm, dtype=np.float32)
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int64] = np.clip(np.searchsorted(bin_edges, position, side="right") - 1, 0, bin_count - 1)
    # noinspection PyTypeChecker
    speed_sums: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
    # noinspection PyTypeChecker
    sample_counts: NDArray[np.int32] = np.zeros(bin_count, dtype=np.int32)
    np.add.at(speed_sums, bin_indices, speed)
    np.add.at(sample_counts, bin_indices, 1)
    # noinspection PyTypeChecker
    mean_speed: NDArray[np.float32] = np.zeros(bin_count, dtype=np.float32)
    # noinspection PyTypeChecker
    valid: NDArray[np.bool_] = sample_counts > 0
    mean_speed[valid] = speed_sums[valid] / sample_counts[valid]
    return mean_speed


def _active_significance_columns(
    *,
    require_stable: bool,
    require_peak_significant: bool,
    table: pl.DataFrame,
) -> list[tuple[str, TuningColumn]]:
    """Returns the active p-value-bearing criteria as (display label, p-value column) pairs in canonical
    order.
    """
    candidates: list[tuple[bool, str, TuningColumn]] = [
        (require_stable, "Stable", TuningColumn.STABILITY_P_VALUE),
        (require_peak_significant, "Peak", TuningColumn.PEAK_P_VALUE),
    ]
    return [(label, column) for active, label, column in candidates if active and column.value in table.columns]


def _make_heatmap_figure(strip_count: int, figure_dpi: int) -> plt.Figure:
    """Allocates a figure sized to leave room for the requested number of significance strips."""
    base_width = 8.0
    extra_width = 0.45 * strip_count
    return plt.figure(figsize=(base_width + extra_width, 4), facecolor="white", dpi=figure_dpi)


def _layout_heatmap_axes(
    figure: plt.Figure,
    strip_count: int,
    *,
    include_colorbar: bool,
) -> tuple[list[plt.Axes], plt.Axes, plt.Axes | None]:
    """Builds the gridspec layout for a heatmap with optional left-side significance strips and a right-side
    colorbar.
    """
    width_ratios: list[float] = [_PLACE_STRIP_WIDTH_RATIO] * strip_count + [1.0]
    if include_colorbar:
        width_ratios.append(0.05)
    grid = figure.add_gridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.08)
    strip_axes = [figure.add_subplot(grid[0, i]) for i in range(strip_count)]
    main_axes = figure.add_subplot(grid[0, strip_count])
    colorbar_axes = figure.add_subplot(grid[0, strip_count + 1]) if include_colorbar else None
    return strip_axes, main_axes, colorbar_axes


def _render_significance_strips(
    figure: plt.Figure,
    strip_axes: list[plt.Axes],
    strip_columns: list[tuple[str, TuningColumn]],
    table: pl.DataFrame,
    ordered_indices: NDArray[np.int64],
) -> None:
    """Renders one ``-log10(p)`` strip per (label, column) entry in ``strip_columns`` alongside the main
    heatmap.
    """
    if not strip_axes or not strip_columns:
        return
    floor = _PLACE_PVALUE_DISPLAY_FLOOR
    vmax = float(-np.log10(floor))
    cell_count = int(ordered_indices.size)
    image = None
    for axis, (label, column) in zip(strip_axes, strip_columns, strict=True):
        # noinspection PyTypeChecker
        p_values: NDArray[np.float32] = table[column.value].to_numpy().astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        ordered_p: NDArray[np.float32] = np.empty(0, dtype=np.float32) if cell_count == 0 else p_values[ordered_indices]
        # noinspection PyTypeChecker
        clipped: NDArray[np.float32] = np.clip(ordered_p, floor, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # noinspection PyTypeChecker
            neg_log_p: NDArray[np.float32] = (-np.log10(clipped)).astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        cleaned: NDArray[np.float32] = np.where(np.isnan(ordered_p), 0.0, neg_log_p).astype(np.float32, copy=False)
        column_data = cleaned.reshape(-1, 1) if cleaned.size > 0 else np.zeros((1, 1), dtype=np.float32)
        image = axis.imshow(
            column_data,
            cmap="Reds",
            extent=(0.0, 1.0, float(max(cell_count, 1)), 0.0),
            vmin=0.0,
            vmax=vmax,
            interpolation="none",
            origin="upper",
            aspect="auto",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel(label, fontsize=7, rotation=0, labelpad=2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("0.5")
    strip_axes[0].set_ylabel("-log₁₀(p)", fontsize=7)
    if image is not None:
        anchor = strip_axes[0].get_position()
        bar_height = 0.02
        bar_axes = figure.add_axes(
            (anchor.x0, anchor.y0 - bar_height - 0.04, anchor.width * len(strip_axes), bar_height)
        )
        color_bar = figure.colorbar(image, cax=bar_axes, orientation="horizontal")
        landmark_p_values = [1.0, 0.05, 0.01, 0.001]
        # noinspection PyTypeChecker
        landmark_ticks: list[float] = [float(-np.log10(max(p, floor))) for p in landmark_p_values if p >= floor]
        color_bar.set_ticks(landmark_ticks)
        color_bar.set_ticklabels(
            [f"{p:g}" for p in landmark_p_values if p >= floor],
            fontsize=6,
        )
        color_bar.ax.tick_params(length=2, pad=1)
        color_bar.set_label("p", fontsize=6, labelpad=2)


def _compose_population_label(
    *,
    require_place: bool,
    require_stable: bool,
    require_peak_significant: bool,
) -> str:
    """Returns ``"Place ∩ Stable ∩ Peak"``-style labels from active criterion bools; ``"All cells"`` when all
    False.
    """
    parts: list[str] = []
    if require_place:
        parts.append("Place")
    if require_stable:
        parts.append("Stable")
    if require_peak_significant:
        parts.append("Peak")
    return " ∩ ".join(parts) if parts else "All cells"
