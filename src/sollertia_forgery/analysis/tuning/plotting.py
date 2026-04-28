"""Per-session plotting helpers for the tuning pipeline.

Module-level functions consume a :class:`TuningReport` plus, where noted, the session's ``data.feather`` opened
memory-mapped at plot time. Persistence and summarization stay on the report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from ...forging import FluorescenceColumn
from ..shared_utilities import trim_acquisition_warmup
from .utilities import assemble_run_session_data
from ...shared_assets import DatasetColumn
from .tuning_report import TuningReport, TuningColumn

if TYPE_CHECKING:
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
    """Plots the position-ordered binned-fluorescence heatmap from the persisted per-cell rate maps.

    Args:
        report: TuningReport whose persisted rate-map column drives the heatmap.
        title: Optional title displayed at the top of the figure.
        sort_by_position: Order cells by their place-field center along the track before plotting.
        show_only_place_cells: Display only cells that pass every requested criterion (the AND of the three
            ``require_*`` flags below).
        require_place: Require ``IS_PLACE`` (Dombeck morphology + lap coverage).
        require_stable: Require ``IS_STABLE`` (Climer & Dombeck 2021 Stability shuffle).
        require_peak_significant: Require ``IS_PEAK_SIGNIFICANT`` (Climer & Dombeck 2021 Peak shuffle).
        mutually_exclusive: When True (default), cells also flagged as ``IS_REWARD_CELL`` are removed from the
            place population so the panel shows only place cells that are not reward cells. Set False to display
            the unfiltered place population.
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
    bin_size_cm = report.summary.bin_size_cm
    bin_count = report.summary.bin_count

    rate_maps = _stack_list_column(table=report.cells, column=TuningColumn.RATE_MAP, target_length=bin_count)
    cell_population, _ = report.resolve_population_masks(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )
    order = _resolve_place_cell_order(table=report.cells)

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
        table=report.cells,
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
            table=report.cells,
            ordered_indices=order,
        )

    track_length_cm = report.summary.track_length_cm
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
    bin_count: int = 20,
    title: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots the spatially significant COM histogram with the fitted uniform + Gaussian mixture overlay."""
    summary = report.summary
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = report.cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        report.cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )
    valid_centers = centers_of_mass[is_significant & (centers_of_mass >= 0.0)]

    figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
    # noinspection PyTypeChecker
    hist_bins: NDArray[np.float64] = np.linspace(0, summary.track_length_cm, bin_count + 1)
    axes.hist(
        valid_centers, bins=hist_bins.tolist(), color="0.7", edgecolor="0.5", density=True, label="Observed COMs"
    )

    # noinspection PyTypeChecker
    positions: NDArray[np.float64] = np.linspace(0, summary.track_length_cm, 200)
    # noinspection PyTypeChecker
    uniform_density: NDArray[np.float64] = np.full_like(positions, 1.0 / summary.track_length_cm)
    gaussian_std = max(summary.gaussian_std_cm, 1.0)
    gaussian_density_reward = np.exp(-0.5 * ((positions - summary.gaussian_mean_cm) / gaussian_std) ** 2) / (
        gaussian_std * np.sqrt(2.0 * np.pi)
    )
    track_start_std = max(summary.track_start_std_cm, 1.0)
    gaussian_density_start = np.exp(-0.5 * ((positions - 0.0) / track_start_std) ** 2) / (
        track_start_std * np.sqrt(2.0 * np.pi)
    )
    track_end_std = max(summary.track_end_std_cm, 1.0)
    gaussian_density_end = np.exp(-0.5 * ((positions - summary.track_length_cm) / track_end_std) ** 2) / (
        track_end_std * np.sqrt(2.0 * np.pi)
    )
    uniform_weight = max(1.0 - summary.mixture_weight - summary.track_start_weight - summary.track_end_weight, 0.0)

    uniform_band = uniform_weight * uniform_density
    landmark_band = uniform_band + (
        summary.track_start_weight * gaussian_density_start + summary.track_end_weight * gaussian_density_end
    )
    mixture_density = landmark_band + summary.mixture_weight * gaussian_density_reward

    axes.fill_between(
        positions,
        0,
        uniform_band,
        alpha=0.3,
        color="lightblue",
        label="Uniform (place cells)",
    )
    axes.fill_between(
        positions,
        uniform_band,
        landmark_band,
        alpha=0.3,
        color="khaki",
        label="Track-end Gaussians",
    )
    axes.fill_between(
        positions,
        landmark_band,
        mixture_density,
        alpha=0.4,
        color="mediumpurple",
        label="Reward Gaussian",
    )
    axes.plot(positions, mixture_density, color="black", linewidth=1.5, label="Mixture fit")
    axes.axvline(
        x=summary.reward_position_cm,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Reward location",
    )

    axes.set_xlabel("Track Position (cm)")
    axes.set_ylabel("Density")
    axes.legend(fontsize=7, loc="upper left")
    annotation_text = (
        f"Significant: {summary.spatially_significant_count}/{summary.cell_count} cells\n"
        f"Mixture weight: {summary.mixture_weight:.1%} reward\n"
        f"Gaussian center: {summary.gaussian_mean_cm:.0f} cm (SD {summary.gaussian_std_cm:.0f} cm)"
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
    title: str | None = None,
    require_place: bool = True,
    require_stable: bool = True,
    require_peak_significant: bool = True,
    mutually_exclusive: bool = True,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots row-normalized rate maps for reward cells and place cells side by side, sorted by COM."""
    summary = report.summary
    rate_maps = _stack_list_column(
        table=report.cells, column=TuningColumn.RATE_MAP, target_length=summary.bin_count
    )
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        report.cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )

    place_mask, reward_mask = report.resolve_population_masks(
        require_place=require_place,
        require_stable=require_stable,
        require_peak_significant=require_peak_significant,
        mutually_exclusive=mutually_exclusive,
    )

    reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
    reward_left = summary.reward_position_cm - reward_zone_half
    reward_right = summary.reward_position_cm + reward_zone_half

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

        extent = [0, summary.bin_size_cm * sorted_maps.shape[1], normalized_maps.shape[0], 0]
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
        axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
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
    title: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots mean population fluorescence vs track position, contrasting reward-predictive against all
    spatially modulated cells.
    """
    summary = report.summary
    rate_maps = _stack_list_column(
        table=report.cells, column=TuningColumn.RATE_MAP, target_length=summary.bin_count
    )
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = report.cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = report.cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = report.cells[
        TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value
    ].to_numpy()

    bin_centers = (np.arange(summary.bin_count) + 0.5) * summary.bin_size_cm
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
        x=summary.reward_position_cm,
        color="red",
        linestyle="-",
        linewidth=2.0,
        alpha=0.8,
        label="Reward location",
    )
    axes.set_xlabel("Track Position (cm)")
    axes.set_ylabel("Average Fluorescence (dF/F)")
    axes.legend(fontsize=7, loc="upper left")
    axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
    if title:
        axes.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def plot_speed_and_activity_by_position(
    report: TuningReport,
    *,
    session: DatasetSession,
    title: str | None = None,
    figure_dpi: int = 150,
    position_sigma_cm: float = 5.0,
) -> plt.Figure:
    """Plots binned-speed alongside reward-predictive cell activity. Reads ``data.feather`` to bin speed by
    position; reward-predictive activity comes from the persisted rate maps.
    """
    summary = report.summary
    rate_maps = _stack_list_column(
        table=report.cells, column=TuningColumn.RATE_MAP, target_length=summary.bin_count
    )
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = report.cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = report.cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = report.cells[
        TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value
    ].to_numpy()
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
        track_length_cm=summary.track_length_cm,
        bin_size_cm=summary.bin_size_cm,
        bin_count=summary.bin_count,
    )
    sigma_bins = position_sigma_cm / summary.bin_size_cm
    # noinspection PyTypeChecker
    smoothed_speed: NDArray[np.float32] = gaussian_filter1d(input=binned_speed, sigma=sigma_bins, mode="wrap")

    bin_centers = (np.arange(summary.bin_count) + 0.5) * summary.bin_size_cm
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

    reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
    axes_speed.axvspan(
        summary.reward_position_cm - reward_zone_half,
        summary.reward_position_cm + reward_zone_half,
        alpha=0.1,
        color="red",
        label="Reward zone",
    )
    axes_speed.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
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
    session: DatasetSession,
    trial_type: str = "ABC",
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
    """Plots per-trial activity heatmaps for an example reward-predictive cell and an example place cell, with
    slowing-onset markers overlaid. Reads ``data.feather`` for the raw fluorescence and per-trial speed time
    series.
    """
    summary = report.summary
    # noinspection PyTypeChecker
    is_significant: NDArray[np.bool_] = report.cells[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
    # noinspection PyTypeChecker
    is_reward_proximal: NDArray[np.bool_] = report.cells[TuningColumn.IS_REWARD_PROXIMAL.value].to_numpy()
    # noinspection PyTypeChecker
    is_position_glm_significant: NDArray[np.bool_] = report.cells[
        TuningColumn.IS_POSITION_GLM_SIGNIFICANT.value
    ].to_numpy()
    # noinspection PyTypeChecker
    cv_partial_r2: NDArray[np.float32] = (
        report.cells[TuningColumn.CV_POSITION_PARTIAL_R2.value].to_numpy().astype(np.float32, copy=False)
    )
    # noinspection PyTypeChecker
    centers_of_mass: NDArray[np.float32] = (
        report.cells[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
    )

    place_mask, _ = report.resolve_population_masks(
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
    track_midpoint = summary.track_length_cm / 2.0
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
        0, summary.track_length_cm + position_bin_size_cm, position_bin_size_cm, dtype=np.float32
    )
    bin_count = len(bin_edges) - 1
    sigma_bins = position_sigma_cm / position_bin_size_cm

    reward_zone_half = summary.reward_configuration.reward_zone_width / 2.0
    reward_left = summary.reward_position_cm - reward_zone_half
    reward_right = summary.reward_position_cm + reward_zone_half
    pre_reward_start = summary.reward_position_cm - summary.reward_configuration.pre_reward_window

    figure, (axes_predictive, axes_place) = plt.subplots(1, 2, figsize=(14, 8), facecolor="white", dpi=figure_dpi)

    cells = [
        (axes_predictive, best_predictive, "Reward-predictive", "Purples"),
        (axes_place, best_place, "Place cell", "Blues"),
    ]
    for axes, cell_index, label, colormap in cells:
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
                trial_positions < summary.reward_position_cm
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
            extent=[0, summary.track_length_cm, trial_count, 0],
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
        axes.set_xticks(np.arange(0, summary.track_length_cm + 1, _PLOT_TICK_INTERVAL_CM))
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


# ===== Private helpers ==========================================================================================


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
    track_length_cm: float,
    bin_size_cm: float,
    bin_count: int,
) -> NDArray[np.float32]:
    """Returns the per-bin mean running speed for the session, computed off ``data.feather``."""
    df = pl.read_ipc(
        source=session.data_path,
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.DISTANCE_CM.value, DatasetColumn.SPEED_CM_S.value],
        memory_map=True,
    )
    df = trim_acquisition_warmup(df)

    # noinspection PyTypeChecker
    distance: NDArray[np.float32] = df[DatasetColumn.DISTANCE_CM.value].to_numpy().astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    speed: NDArray[np.float32] = df[DatasetColumn.SPEED_CM_S.value].to_numpy().astype(np.float32, copy=False)

    # noinspection PyTypeChecker
    position: NDArray[np.float32] = (distance % np.float32(track_length_cm)).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    bin_edges: NDArray[np.float32] = np.arange(0.0, track_length_cm + bin_size_cm, bin_size_cm, dtype=np.float32)
    # noinspection PyTypeChecker
    bin_indices: NDArray[np.int64] = np.clip(
        np.searchsorted(bin_edges, position, side="right") - 1, 0, bin_count - 1
    )
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
    """Returns the active p-value-bearing criteria as (display label, p-value column) pairs in canonical order."""
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
        ordered_p: NDArray[np.float32] = (
            np.empty(0, dtype=np.float32) if cell_count == 0 else p_values[ordered_indices]
        )
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
