"""Per-session plotting helpers for the tuning pipeline.

Module-level functions consume a `TuningReport` plus the trial type the figure should depict, plus —
where noted — the session's ``data.feather`` opened memory-mapped at plot time. Persistence and summarization
stay on the report. Every function takes a ``trial_type`` argument that selects the long-format slice of the
report's cells feather and the matching `TuningTrialSummary` entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Mapping

import numpy as np
import polars as pl
from scipy.stats import binomtest
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp
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
_SESSION_TIMESTAMP_FORMAT: str = "%Y-%m-%d-%H-%M-%S-%f"
"""``strptime`` format string for the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` session-directory name."""


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


def _resolve_display_session_indices(
    session_count: int,
    display_sessions: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Maps 1-indexed display-session numbers to 0-indexed positions, dropping out-of-range entries.

    Notes:
        Used by every per-day plotting helper that exposes a ``display_sessions`` parameter so the
        1-indexed → 0-indexed mapping, dedup, and out-of-range filtering match across plots.

    Args:
        session_count: Total number of sessions available to the caller.
        display_sessions: Caller-supplied 1-indexed session numbers, or ``None`` to render every session.

    Returns:
        A tuple of 0-indexed positions in caller order, with duplicates removed and out-of-range
        entries silently dropped.
    """
    if display_sessions is None:
        return tuple(range(session_count))
    resolved: list[int] = []
    for target in display_sessions:
        idx = int(target) - 1
        if 0 <= idx < session_count and idx not in resolved:
            resolved.append(idx)
    return tuple(resolved)


def _compute_day_offsets_per_session(
    sessions: tuple[DatasetSession, ...],
    display_session_indices: tuple[int, ...],
) -> dict[int, int | None]:
    """Returns per-session integer day offsets relative to ``sessions[0]``.

    Notes:
        Anchored at ``sessions[0]`` so the day numbering matches the chronological ordering used to
        build ``sessions`` regardless of which subset the caller picked through ``display_sessions``.
        Sessions whose name fails to parse as the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` format
        receive ``None`` so callers can render them without a ``Day Y`` suffix.

    Args:
        sessions: Chronologically ordered DatasetSession entries; element 0 anchors the day axis.
        display_session_indices: 0-indexed positions for which to resolve day offsets.

    Returns:
        Mapping from each entry of ``display_session_indices`` to its rounded day offset, or ``None``
        when the session name cannot be parsed.
    """
    day_per_session: dict[int, int | None] = {}
    if not sessions:
        return day_per_session
    try:
        first_us = int(parse_timestamp(
            date_string=sessions[0].session,
            format_string=_SESSION_TIMESTAMP_FORMAT,
            output_format=TimestampFormats.INTEGER,
        ))
    except ValueError:
        first_us = None
    for sess_idx in display_session_indices:
        if first_us is None:
            day_per_session[sess_idx] = None
            continue
        try:
            session_us = int(parse_timestamp(
                date_string=sessions[sess_idx].session,
                format_string=_SESSION_TIMESTAMP_FORMAT,
                output_format=TimestampFormats.INTEGER,
            ))
        except ValueError:
            day_per_session[sess_idx] = None
            continue
        day_per_session[sess_idx] = round(float(convert_time(
            time=session_us - first_us,
            from_units=TimeUnits.MICROSECOND,
            to_units=TimeUnits.DAY,
            as_float=True,
        )))
    return day_per_session


def _resolve_panel_cue_boundaries_cm(
    sessions: tuple[DatasetSession, ...],
    display_session_indices: tuple[int, ...],
    resolved_trial_type: str | None,
    track_length_cm: float,
    show_cue_boundaries: bool,
) -> tuple[float, ...]:
    """Resolves canonical cue-zone boundaries for the per-day plotting helpers.

    Notes:
        The cue layout is constant per trial type, so the first session whose realignment yields a
        non-empty boundary set is reused for every panel. Routes through
        `_resolve_canonical_cue_boundaries`, which itself uses `assemble_run_session_data` — the
        same realignment path the rate maps consume — so canonical coordinates are guaranteed to
        match between the cue overlay and the heatmap x-axis.

    Args:
        sessions: Chronologically ordered DatasetSession entries.
        display_session_indices: 0-indexed positions to consult when sourcing the canonical walk.
        resolved_trial_type: Trial type whose cue layout is being requested, or ``None`` to skip.
        track_length_cm: Trial length used to clip boundaries to the open interval
            ``(0, track_length_cm)``.
        show_cue_boundaries: When False, skips the work and returns an empty tuple.

    Returns:
        A tuple of canonical cm positions of cue-identity transitions, or an empty tuple when the
        overlay is disabled or no session yields usable data.
    """
    if not show_cue_boundaries or resolved_trial_type is None or track_length_cm <= 0:
        return ()
    for sess_idx in display_session_indices:
        session = sessions[sess_idx]
        if not session.data_path.exists():
            continue
        boundaries = _resolve_canonical_cue_boundaries(
            session_path=session.session_path,
            trial_type=resolved_trial_type,
            track_length_cm=track_length_cm,
        )
        if boundaries:
            return boundaries
    return ()


def plot_sorted_heatmap(
    sessions: tuple[DatasetSession, ...],
    *,
    trial_type: str | None = None,
    display_sessions: tuple[int, ...] | None = None,
    cmap: str = "magma",
    show_cue_boundaries: bool = True,
    minimum_percentile: float = 0.0,
    maximum_percentile: float = 0.95,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots raw ΔF/F₀ rate maps for every registered cell, sorted independently by each session's peak.

    Notes:
        Each panel renders every registered cell as one row and sorts the rows by the cell's
        rate-map peak position on that session. There is no across-session correspondence (each panel
        sorts independently), so the figure shows the population's day-level tuning band rather than
        per-cell drift. Per-panel titles read ``Session X (Day Y)`` where ``Y`` is the integer day
        offset from the first chronologically ordered session in ``sessions``. Pair with
        `plot_classified_heatmap` for the row-normalized active-cells-only band, or with
        ``..drift.plotting.plot_reference_day_sorted_rate_maps`` for the per-cell drift view.

        Rate maps render as raw ΔF/F₀ values (no row normalization). A single shared color scale runs
        from the ``minimum_percentile`` quantile to the ``maximum_percentile`` quantile of the pooled
        rate-map values across every panel, so colors are directly comparable between sessions and
        the single horizontal colorbar at the bottom of the figure annotates the absolute scale.

        Sessions are addressed by 1-indexed chronological session number (``1`` is the first session
        in ``sessions``, ``2`` is the second, etc.). Out-of-range entries are silently skipped, and
        duplicates that resolve to the same session are deduplicated.

        ``trial_type`` is resolved once from the first available session's tuning feather when not
        supplied by the caller, then applied uniformly to every panel.

        Sessions render in a single chronological row at a tall-rectangular panel aspect ratio with a
        shared y-axis (the registered cell count is constant across multi-day-registered sessions).

    Args:
        sessions: Chronologically ordered DatasetSession entries to render.
        trial_type: Trial type to evaluate; when ``None``, defaults to the first trial type present in
            the first session's tuning feather. Applied uniformly across panels.
        display_sessions: 1-indexed session numbers to render as columns. Defaults to every session
            in the supplied tuple — pass an explicit selection to render a subset.
        cmap: Matplotlib colormap name for the rate-map intensities. Default ``"magma"`` is a
            perceptually-uniform colormap with strong contrast on dark backgrounds.
        show_cue_boundaries: When True, draw cyan dotted verticals on every panel at the start and
            end of the cue zone that contains the trigger zone for that session.
        minimum_percentile: Quantile (in ``[0, 1]``) of the pooled rate-map values used as the shared
            lower bound of the color scale. Clamped at zero so negative ΔF/F₀ noise renders at the
            colormap floor and the value 0 reads as "no signal" on the colorbar.
        maximum_percentile: Quantile (in ``[0, 1]``) of the pooled rate-map values used as the shared
            upper bound of the color scale. Default ``0.95`` clips the top 5% of pixels so
            outlier-bright bumps don't saturate the panel.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    session_count: int = len(sessions)
    display_session_indices = _resolve_display_session_indices(
        session_count=session_count, display_sessions=display_sessions,
    )
    n_panels: int = len(display_session_indices)
    n_cols: int = max(n_panels, 1)

    figure, axes_array = plt.subplots(
        1, n_cols,
        figsize=(2.0 * n_cols + 1.2, 4.5),
        facecolor="white", dpi=figure_dpi, squeeze=False,
        layout="constrained",
        sharey=True,
    )
    if session_count == 0 or n_panels == 0:
        axes_array[0, 0].text(0.5, 0.5, "No sessions supplied", ha="center", va="center",
                              transform=axes_array[0, 0].transAxes)
        for axes in axes_array.flat:
            axes.set_axis_off()
        return figure

    # Pre-loads each session's filtered tuning frame once and caches the per-cell rate maps the render
    # loop will consume. The per-session trigger zone is captured here too so the cyan cue-zone overlay
    # can locate which canonical interval to bracket without re-reading geometry on every panel; the
    # cue boundaries themselves come from the realignment path used by `assemble_run_session_data` and
    # are computed once for the whole figure below.
    rate_maps_per_session: dict[int, NDArray[np.float32] | None] = {}
    trigger_zones: dict[int, tuple[float, float] | None] = {}
    bin_count_reference: int | None = None
    track_length_reference: float | None = None
    resolved_trial_type: str | None = trial_type
    pooled_chunks: list[NDArray[np.float32]] = []
    for sess_idx in display_session_indices:
        path = sessions[sess_idx].tuning_cells_path
        if not path.exists():
            rate_maps_per_session[sess_idx] = None
            trigger_zones[sess_idx] = None
            continue
        frame = pl.read_ipc(source=path, memory_map=True)
        if resolved_trial_type is None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            unique_trial_types = frame[TuningColumn.TRIAL_TYPE.value].unique().to_list()
            if unique_trial_types:
                resolved_trial_type = str(unique_trial_types[0])
        if resolved_trial_type is not None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type)
        frame = frame.sort(TuningColumn.CELL_ID.value)
        if frame.height == 0:
            rate_maps_per_session[sess_idx] = None
        else:
            # noinspection PyTypeChecker
            session_rate_maps: NDArray[np.float32] = np.asarray(
                frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
            )
            rate_maps_per_session[sess_idx] = session_rate_maps
            if bin_count_reference is None and session_rate_maps.size > 0:
                bin_count_reference = int(session_rate_maps.shape[1])
            if session_rate_maps.size > 0:
                pooled_chunks.append(session_rate_maps.ravel())
        geometry_path = sessions[sess_idx].geometry_path
        if geometry_path.exists() and resolved_trial_type is not None:
            geometry = TrialGeometry.from_yaml(file_path=geometry_path)
            entry = geometry.entries.get(resolved_trial_type)
            if entry is not None:
                trigger_zones[sess_idx] = (
                    float(entry.stimulus_trigger_zone_start_cm),
                    float(entry.stimulus_trigger_zone_end_cm),
                )
                if track_length_reference is None:
                    track_length_reference = float(entry.trial_length_cm)
            else:
                trigger_zones[sess_idx] = None
        else:
            trigger_zones[sess_idx] = None

    bin_count: int = bin_count_reference if bin_count_reference is not None else 0
    track_length_cm: float = (
        track_length_reference if track_length_reference is not None else float(bin_count)
    )
    bin_size_cm: float = track_length_cm / bin_count if bin_count > 0 else 1.0

    # Resolves a single shared color scale from every cell pooled across panels so colors are directly
    # comparable between sessions. ``np.nanquantile`` skips NaN bins (occupancy gaps in the per-bin
    # mean). The lower bound is clamped to zero so negative ΔF/F₀ noise still renders at the floor.
    if pooled_chunks:
        # noinspection PyTypeChecker
        pooled_values: NDArray[np.float32] = np.concatenate(pooled_chunks)
        if pooled_values.size > 0 and np.any(np.isfinite(pooled_values)):
            color_min: float = max(0.0, float(np.nanquantile(pooled_values, minimum_percentile)))
            color_max: float = float(np.nanquantile(pooled_values, maximum_percentile))
        else:
            color_min, color_max = 0.0, 1.0
    else:
        color_min, color_max = 0.0, 1.0
    if color_max <= color_min:
        color_max = color_min + 1e-3

    day_per_session = _compute_day_offsets_per_session(
        sessions=sessions, display_session_indices=display_session_indices,
    )
    cue_boundaries_cm = _resolve_panel_cue_boundaries_cm(
        sessions=sessions,
        display_session_indices=display_session_indices,
        resolved_trial_type=resolved_trial_type,
        track_length_cm=track_length_cm,
        show_cue_boundaries=show_cue_boundaries,
    )

    last_image = None
    for column_position, sess_idx in enumerate(display_session_indices):
        axes = axes_array[0, column_position]
        rate_maps = rate_maps_per_session.get(sess_idx)
        if rate_maps is None or rate_maps.shape[0] == 0:
            axes.text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.transAxes)
            # Uses ``set_axis_off`` rather than clearing tick lists so the shared y-axis on the
            # neighboring rendered panels keeps its tick locations.
            axes.set_axis_off()
            continue
        n_cells: int = int(rate_maps.shape[0])

        # Sorts every cell by its rate-map peak position so the panel forms a position-sorted tuning
        # band. Peak positions ignore NaN bins (occupancy gaps) by replacing them with -inf before the
        # ``argmax``.
        finite_for_argmax = np.where(np.isfinite(rate_maps), rate_maps, -np.inf)
        # noinspection PyTypeChecker
        peak_bins: NDArray[np.int64] = np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
        # noinspection PyTypeChecker
        order: NDArray[np.int64] = np.argsort(peak_bins, kind="stable")
        sorted_maps = rate_maps[order]

        last_image = axes.imshow(
            sorted_maps, aspect="auto", origin="upper", cmap=cmap,
            extent=[0, bin_count * bin_size_cm, n_cells, 0],
            vmin=color_min, vmax=color_max,
            interpolation="nearest",
        )
        _draw_cue_zone_overlay(
            axes=axes,
            cue_boundaries_cm=cue_boundaries_cm,
            trigger_zone=trigger_zones.get(sess_idx),
            track_length_cm=track_length_cm,
        )
        axes.set_xlim(0, bin_count * bin_size_cm)
        day_offset = day_per_session.get(sess_idx)
        panel_title = (
            f"Session {sess_idx + 1} (Day {day_offset})" if day_offset is not None
            else f"Session {sess_idx + 1}"
        )
        axes.set_title(panel_title, fontsize=10)
        axes.set_xlabel("Position (cm)", fontsize=9)
        if column_position == 0:
            axes.set_ylabel("Cell #", fontsize=9)

    if last_image is not None:
        # Shared horizontal colorbar at the bottom of the figure documents the absolute ΔF/F₀ scale.
        color_bar = figure.colorbar(
            last_image,
            ax=axes_array.ravel().tolist(),
            orientation="horizontal",
            shrink=0.5,
            aspect=40,
            pad=0.04,
        )
        color_bar.set_label("ΔF/F₀", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)

    animal_prefix: str = f"Animal {animal_id} " if animal_id is not None else ""
    suptitle_subject: str = "tunings sorted by track position"
    figure.suptitle(
        f"{animal_prefix}{suptitle_subject}".capitalize() if animal_id is None
        else f"{animal_prefix}{suptitle_subject}",
        fontsize=12,
    )
    return figure


def plot_classified_heatmap(
    sessions: tuple[DatasetSession, ...],
    *,
    classifier: str = "place",
    trial_type: str | None = None,
    display_sessions: tuple[int, ...] | None = None,
    cmap: str = "magma",
    show_cue_boundaries: bool = True,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots row-normalized rate maps of classified cells per session, sorted by each session's peak.

    Notes:
        Each panel filters its session's persisted ``tuning_cells.feather`` to cells classified by the
        requested ``classifier`` on that session, sorts the surviving cells by their rate-map peak
        position, and renders the row-normalized rate maps. There is no across-session correspondence:
        each panel uses the within-session classification only, so the figure shows the day-level
        active-cell tuning band rather than per-cell drift. Cell counts vary across panels (the
        classifier-active set differs by day), which is reflected by the per-panel ``Session X
        (Day Y, n=N)`` titles and a tickless y-axis. Pair with `plot_sorted_heatmap` for the
        all-cells raw ΔF/F₀ view.

        ``classifier="place"`` always selects ``IS_STRICT_PLACE`` (place-field morphology + lap
        coverage + split-half stability + per-cell peak shuffle). ``classifier="reward"`` selects
        ``IS_REWARD_CELL``. Each row in the heatmap is divided by its own per-cell peak so the color
        scale is fixed at ``[0, 1]`` regardless of the cell's absolute ΔF/F₀ amplitude; this matches
        the convention used before the all-cells variant switched to raw fluorescence.

        Sessions are addressed by 1-indexed chronological session number (``1`` is the first session
        in ``sessions``, ``2`` is the second, etc.). Out-of-range entries are silently skipped, and
        duplicates that resolve to the same session are deduplicated. ``trial_type`` is resolved once
        from the first available session's tuning feather when not supplied by the caller, then
        applied uniformly to every panel.

    Args:
        sessions: Chronologically ordered DatasetSession entries to render.
        classifier: ``"place"`` (default) maps to ``IS_STRICT_PLACE``; ``"reward"`` maps to
            ``IS_REWARD_CELL``.
        trial_type: Trial type to evaluate; when ``None``, defaults to the first trial type present in
            the first session's tuning feather. Applied uniformly across panels.
        display_sessions: 1-indexed session numbers to render as columns. Defaults to every session
            in the supplied tuple — pass an explicit selection to render a subset.
        cmap: Matplotlib colormap name for the rate-map intensities. Default ``"magma"`` is a
            perceptually-uniform colormap with strong contrast on dark backgrounds.
        show_cue_boundaries: When True, draw cyan dotted verticals on every panel at the start and
            end of the cue zone that contains the trigger zone for that session.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    if classifier not in ("place", "reward"):
        message = (
            f"Unable to plot per-day classified rate maps. ``classifier`` must be 'place' or 'reward', "
            f"but got {classifier!r}."
        )
        raise ValueError(message)

    classifier_column = (
        TuningColumn.IS_STRICT_PLACE.value if classifier == "place"
        else TuningColumn.IS_REWARD_CELL.value
    )
    classifier_label = "strict-place" if classifier == "place" else "reward"

    session_count: int = len(sessions)
    display_session_indices = _resolve_display_session_indices(
        session_count=session_count, display_sessions=display_sessions,
    )
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

    # Pre-loads each session's classified rate maps, the per-session trigger zone (used to anchor the
    # cyan cue-zone overlay on the panel that contains the trigger), and ``cue_offset_cm`` (so the cue
    # overlay matches realigned rate maps). Only cells that pass the classifier survive into
    # ``rate_maps_per_session``; non-classified cells never enter the render loop.
    rate_maps_per_session: dict[int, NDArray[np.float32] | None] = {}
    trigger_zones: dict[int, tuple[float, float] | None] = {}
    bin_count_reference: int | None = None
    track_length_reference: float | None = None
    resolved_trial_type: str | None = trial_type
    for sess_idx in display_session_indices:
        path = sessions[sess_idx].tuning_cells_path
        if not path.exists():
            rate_maps_per_session[sess_idx] = None
            trigger_zones[sess_idx] = None
            continue
        frame = pl.read_ipc(source=path, memory_map=True)
        if resolved_trial_type is None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            unique_trial_types = frame[TuningColumn.TRIAL_TYPE.value].unique().to_list()
            if unique_trial_types:
                resolved_trial_type = str(unique_trial_types[0])
        if resolved_trial_type is not None and TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type)
        frame = frame.sort(TuningColumn.CELL_ID.value)
        if frame.height == 0 or classifier_column not in frame.columns:
            rate_maps_per_session[sess_idx] = None
        else:
            # noinspection PyTypeChecker
            full_rate_maps: NDArray[np.float32] = np.asarray(
                frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
            )
            # noinspection PyTypeChecker
            classifier_mask: NDArray[np.bool_] = (
                frame[classifier_column].to_numpy().astype(np.bool_, copy=False)
            )
            active_rate_maps = full_rate_maps[classifier_mask] if classifier_mask.any() else (
                full_rate_maps[:0]
            )
            rate_maps_per_session[sess_idx] = active_rate_maps
            if bin_count_reference is None and full_rate_maps.size > 0:
                bin_count_reference = int(full_rate_maps.shape[1])
        geometry_path = sessions[sess_idx].geometry_path
        if geometry_path.exists() and resolved_trial_type is not None:
            geometry = TrialGeometry.from_yaml(file_path=geometry_path)
            entry = geometry.entries.get(resolved_trial_type)
            if entry is not None:
                trigger_zones[sess_idx] = (
                    float(entry.stimulus_trigger_zone_start_cm),
                    float(entry.stimulus_trigger_zone_end_cm),
                )
                if track_length_reference is None:
                    track_length_reference = float(entry.trial_length_cm)
            else:
                trigger_zones[sess_idx] = None
        else:
            trigger_zones[sess_idx] = None

    bin_count: int = bin_count_reference if bin_count_reference is not None else 0
    track_length_cm: float = (
        track_length_reference if track_length_reference is not None else float(bin_count)
    )
    bin_size_cm: float = track_length_cm / bin_count if bin_count > 0 else 1.0

    day_per_session = _compute_day_offsets_per_session(
        sessions=sessions, display_session_indices=display_session_indices,
    )
    cue_boundaries_cm = _resolve_panel_cue_boundaries_cm(
        sessions=sessions,
        display_session_indices=display_session_indices,
        resolved_trial_type=resolved_trial_type,
        track_length_cm=track_length_cm,
        show_cue_boundaries=show_cue_boundaries,
    )

    last_image = None
    for column_position, sess_idx in enumerate(display_session_indices):
        axes = axes_array[0, column_position]
        active_rate_maps = rate_maps_per_session.get(sess_idx)
        if active_rate_maps is None or active_rate_maps.shape[0] == 0:
            axes.text(0.5, 0.5, "no active cells", ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            continue
        n_active: int = int(active_rate_maps.shape[0])

        finite_for_argmax = np.where(np.isfinite(active_rate_maps), active_rate_maps, -np.inf)
        # noinspection PyTypeChecker
        peak_bins: NDArray[np.int64] = np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
        # noinspection PyTypeChecker
        order: NDArray[np.int64] = np.argsort(peak_bins, kind="stable")
        sorted_maps = active_rate_maps[order]
        # Row-normalizes each cell to its own peak so the color scale is fixed at [0, 1] regardless of
        # absolute ΔF/F₀ amplitude. Rows whose peak is non-positive or non-finite stay at zero.
        # noinspection PyTypeChecker
        row_max: NDArray[np.float32] = np.nanmax(sorted_maps, axis=1, keepdims=True)
        # noinspection PyTypeChecker
        safe_row_max: NDArray[np.float32] = np.where(
            np.isfinite(row_max) & (row_max > 0), row_max, np.float32(1.0),
        )
        normalized = np.clip(sorted_maps / safe_row_max, 0.0, 1.0)
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)

        last_image = axes.imshow(
            normalized, aspect="auto", origin="upper", cmap=cmap,
            extent=[0, bin_count * bin_size_cm, n_active, 0],
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        _draw_cue_zone_overlay(
            axes=axes,
            cue_boundaries_cm=cue_boundaries_cm,
            trigger_zone=trigger_zones.get(sess_idx),
            track_length_cm=track_length_cm,
        )
        axes.set_xlim(0, bin_count * bin_size_cm)
        day_offset = day_per_session.get(sess_idx)
        if day_offset is not None:
            panel_title = f"Session {sess_idx + 1} (Day {day_offset}, n={n_active})"
        else:
            panel_title = f"Session {sess_idx + 1} (n={n_active})"
        axes.set_title(panel_title, fontsize=10)
        axes.set_xlabel("Position (cm)", fontsize=9)
        # Drops y-axis ticks because cell counts differ across panels — absolute row indices carry no
        # cross-panel meaning, and the per-panel ``n=N`` already documents the count.
        axes.set_yticks([])
        if column_position == 0:
            axes.set_ylabel(f"{classifier_label.capitalize()} cell (per-day peak sort)", fontsize=9)

    if last_image is not None:
        color_bar = figure.colorbar(
            last_image,
            ax=axes_array.ravel().tolist(),
            orientation="horizontal",
            shrink=0.5,
            aspect=40,
            pad=0.04,
        )
        color_bar.set_label("Row-normalized ΔF/F₀ (peak = 1)", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)

    animal_prefix: str = f"Animal {animal_id} " if animal_id is not None else ""
    suptitle_subject: str = f"{classifier_label} tunings sorted by track position"
    figure.suptitle(
        f"{animal_prefix}{suptitle_subject}".capitalize() if animal_id is None
        else f"{animal_prefix}{suptitle_subject}",
        fontsize=12,
    )
    return figure


def plot_cue_pair_place_counts(
    sessions: tuple[DatasetSession, ...],
    *,
    trial_type: str | None = None,
    display_sessions: tuple[int, ...] | None = None,
    animal_id: str | None = None,
    n_cue_pairs: int = 4,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots strict-place cell counts per cue-gray pair, one panel per session.

    Notes:
        Assumes uniform-length cue-gray pairs: the trial geometry's track length is divided into
        ``n_cue_pairs`` equal segments and each segment becomes one bar. There is no
        ``data.feather`` walk — only the per-session tuning feather and the trial geometry data file
        are read, which keeps the per-panel cost in the milliseconds. Bars are labelled
        ``<pair_index>-0`` (e.g., ``1-0`` for cue 1 paired with the gray that follows it) and the
        bar height is the count of strict-place cells whose rate-map peak position lands inside the
        pair.

    Args:
        sessions: Chronologically ordered DatasetSession entries to render.
        trial_type: Trial type to evaluate; when ``None``, defaults to the first trial type present
            in the first session's tuning feather. Applied uniformly across panels.
        display_sessions: 1-indexed session numbers to render as panels. Defaults to every session
            in the supplied tuple.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        n_cue_pairs: Number of uniform-length cue-gray pairs to partition the track into. Defaults
            to ``4`` (matching ``cyclic_4_cue``); pass an explicit value when the trial type uses a
            different cue count.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure.
    """
    session_count: int = len(sessions)
    display_session_indices = _resolve_display_session_indices(
        session_count=session_count, display_sessions=display_sessions,
    )
    n_panels: int = len(display_session_indices)
    n_cols: int = max(n_panels, 1)

    figure, axes_array = plt.subplots(
        1, n_cols,
        figsize=(2.6 * n_cols + 1.0, 4.0),
        facecolor="white", dpi=figure_dpi, squeeze=False,
        layout="constrained",
        sharey=True,
    )
    if session_count == 0 or n_panels == 0:
        axes_array[0, 0].text(0.5, 0.5, "No sessions supplied", ha="center", va="center",
                              transform=axes_array[0, 0].transAxes)
        for axes in axes_array.flat:
            axes.set_axis_off()
        return figure

    day_per_session = _compute_day_offsets_per_session(
        sessions=sessions, display_session_indices=display_session_indices,
    )

    # Per-panel render state: bars first, brackets in a second pass once the global y-headroom is
    # known. Sharing the y-axis means the bracket stack on the busiest panel decides the figure's
    # top, so the y-limit can only be set after every panel has reported its own bracket count.
    panel_state: list[tuple[plt.Axes, NDArray[np.int64], dict[tuple[int, int], float]]] = []
    max_bracket_count: int = 0
    max_bar_height: int = 0

    for column_position, sess_idx in enumerate(display_session_indices):
        axes = axes_array[0, column_position]
        session = sessions[sess_idx]
        day_offset = day_per_session.get(sess_idx)
        panel_label = (
            f"Session {sess_idx + 1} (Day {day_offset})" if day_offset is not None
            else f"Session {sess_idx + 1}"
        )

        result = _resolve_session_cue_pair_counts(
            session=session, trial_type=trial_type, n_cue_pairs=n_cue_pairs,
        )
        if result is None:
            axes.text(0.5, 0.5, "no cue pairs", ha="center", va="center", transform=axes.transAxes)
            axes.set_title(panel_label, fontsize=10)
            axes.set_axis_off()
            continue

        counts, pair_labels = result
        x_positions = np.arange(counts.size, dtype=np.int64)
        bars = axes.bar(
            x_positions, counts,
            color="#88abc1", edgecolor="black", linewidth=0.4,
        )
        for bar, count in zip(bars, counts.tolist(), strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(int(count)), ha="center", va="bottom", fontsize=8,
            )

        axes.set_title(panel_label, fontsize=10)
        axes.set_xlabel("Cue-gray pair", fontsize=9)
        axes.set_xticks(x_positions)
        axes.set_xticklabels(pair_labels, fontsize=8)
        if column_position == 0:
            axes.set_ylabel("Strict-place cells", fontsize=9)

        p_values = _compute_pairwise_pair_pvalues(counts=counts)
        bracket_count = int(sum(1 for p in p_values.values() if p >= 0.05))
        max_bracket_count = max(max_bracket_count, bracket_count)
        max_bar_height = max(max_bar_height, int(counts.max()) if counts.size > 0 else 0)
        panel_state.append((axes, counts, p_values))

    if max_bar_height > 0:
        bracket_spacing = max(max_bar_height * 0.10, 0.5)
        # Headroom: one spacing for the gap above the tallest bar plus one slot per stacked bracket
        # (with an extra half-slot of breathing room at the top so the ``ns`` glyph is not clipped).
        y_top = max_bar_height + bracket_spacing * (max_bracket_count + 1.5)
        axes_array[0, 0].set_ylim(top=y_top)
        for axes, counts, p_values in panel_state:
            _draw_nonsignificance_brackets(
                axes=axes, counts=counts, p_values=p_values,
                bar_max=max_bar_height, bracket_spacing=bracket_spacing,
            )

    animal_prefix: str = f"Animal {animal_id} " if animal_id is not None else ""
    suptitle_subject: str = "strict-place cells per cue-gray pair"
    figure.suptitle(
        f"{animal_prefix}{suptitle_subject}".capitalize() if animal_id is None
        else f"{animal_prefix}{suptitle_subject}",
        fontsize=12,
    )
    return figure


def _compute_pairwise_pair_pvalues(counts: NDArray[np.int64]) -> dict[tuple[int, int], float]:
    """Returns the two-sided binomial p-value for every unordered pair of bar indices.

    Notes:
        Conditions on a strict-place cell having landed in pair ``i`` or pair ``j`` and tests the
        null "the cell is equally likely to be in either" — ``counts[i] ~ Binomial(counts[i] +
        counts[j], 0.5)``. Pairs with a zero combined count get ``p = 1.0`` since the test is
        undefined there. No multiple-comparison correction is applied; the caller can apply
        Bonferroni / Holm if required (with ``n = n_cue_pairs * (n_cue_pairs - 1) / 2``).
    """
    n_pairs = int(counts.size)
    p_values: dict[tuple[int, int], float] = {}
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            k_i = int(counts[i])
            k_j = int(counts[j])
            n_total = k_i + k_j
            if n_total == 0:
                p_values[(i, j)] = 1.0
                continue
            p_values[(i, j)] = float(
                binomtest(k=k_i, n=n_total, p=0.5, alternative="two-sided").pvalue,
            )
    return p_values


def _draw_nonsignificance_brackets(
    axes: plt.Axes,
    counts: NDArray[np.int64],
    p_values: dict[tuple[int, int], float],
    bar_max: int,
    bracket_spacing: float,
) -> None:
    """Draws thin black ``ns`` brackets for every pairwise comparison that fails to reject p < 0.05.

    Notes:
        Inverts the usual significance overlay because in this dataset almost every pairwise
        comparison clears p < 0.05 — flagging the *equivalent* pairs is the informative subset.
        Brackets stack vertically above the tallest bar in the figure (``bar_max``) with constant
        ``bracket_spacing`` between rows, sorted by span length so the shortest comparisons sit at
        the bottom of the stack. ``clip_on=False`` keeps the brackets visible even when they
        exceed the panel's data range.
    """
    nonsignificant: list[tuple[int, int]] = sorted(
        (pair for pair, p in p_values.items() if p >= 0.05),
        key=lambda item: (item[1] - item[0], item[0]),
    )
    if not nonsignificant:
        return

    base_y = float(bar_max) + bracket_spacing
    tick_height = bracket_spacing * 0.35
    for level, (left, right) in enumerate(nonsignificant):
        y = base_y + level * bracket_spacing
        axes.plot([left, right], [y, y], color="black", linewidth=0.7, clip_on=False)
        axes.plot([left, left], [y, y - tick_height], color="black", linewidth=0.7, clip_on=False)
        axes.plot([right, right], [y, y - tick_height], color="black", linewidth=0.7, clip_on=False)
        axes.text(
            (left + right) / 2.0, y + bracket_spacing * 0.05, "ns",
            ha="center", va="bottom", fontsize=8, color="black", clip_on=False,
        )


def _resolve_session_cue_pair_counts(
    session: DatasetSession,
    trial_type: str | None,
    n_cue_pairs: int,
) -> tuple[NDArray[np.int64], list[str]] | None:
    """Returns ``(per_pair_counts, pair_labels)`` for one session, or ``None``.

    Notes:
        Partitions the trial-geometry track length into ``n_cue_pairs`` equal-length pairs (no
        ``data.feather`` walk) and bins each strict-place cell's rate-map peak position into the
        pair containing it. Filtering the tuning frame to strict-place cells before materialising
        the rate-map list column keeps the heavy ``to_list()`` deserialisation off the much larger
        non-strict subset.
    """
    if not session.tuning_cells_path.exists() or not session.geometry_path.exists():
        return None
    if n_cue_pairs <= 0:
        return None

    frame = pl.read_ipc(source=session.tuning_cells_path, memory_map=True)
    resolved_trial_type = trial_type
    if resolved_trial_type is None and TuningColumn.TRIAL_TYPE.value in frame.columns:
        unique_trial_types = frame[TuningColumn.TRIAL_TYPE.value].unique().to_list()
        if not unique_trial_types:
            return None
        resolved_trial_type = str(unique_trial_types[0])
    if resolved_trial_type is None:
        return None
    if TuningColumn.TRIAL_TYPE.value in frame.columns:
        frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type)
    if frame.height == 0 or TuningColumn.IS_STRICT_PLACE.value not in frame.columns:
        return None

    geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
    geometry_entry = geometry.entries.get(resolved_trial_type)
    if geometry_entry is None:
        return None
    track_length_cm = float(geometry_entry.trial_length_cm)
    if track_length_cm <= 0:
        return None

    pair_length_cm = track_length_cm / n_cue_pairs
    pair_bounds: list[tuple[float, float]] = [
        (i * pair_length_cm, (i + 1) * pair_length_cm) for i in range(n_cue_pairs)
    ]
    pair_labels: list[str] = [f"{i + 1}-0" for i in range(n_cue_pairs)]

    # Filter to strict-place cells before deserialising the rate-map list column. For a typical
    # session ~75% of cells are dropped here, so the heavy ``to_list()`` cost shrinks proportionally.
    strict_frame = frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value)).sort(TuningColumn.CELL_ID.value)
    # noinspection PyTypeChecker
    counts: NDArray[np.int64] = np.zeros(n_cue_pairs, dtype=np.int64)
    if strict_frame.height == 0:
        return counts, pair_labels

    # noinspection PyTypeChecker
    strict_rate_maps: NDArray[np.float32] = np.asarray(
        strict_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    if strict_rate_maps.size == 0:
        return counts, pair_labels
    bin_count = int(strict_rate_maps.shape[1])
    bin_size_cm = track_length_cm / bin_count if bin_count > 0 else 1.0

    # Cells whose entire rate-map row is non-finite get the ``-1`` sentinel so the pair tests skip
    # them rather than placing them at bin 0.
    finite_for_argmax = np.where(np.isfinite(strict_rate_maps), strict_rate_maps, -np.inf)
    # noinspection PyTypeChecker
    peak_bins: NDArray[np.int64] = np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
    # noinspection PyTypeChecker
    peak_cm: NDArray[np.float32] = (
        (peak_bins.astype(np.float32) + np.float32(0.5)) * np.float32(bin_size_cm)
    )
    # noinspection PyTypeChecker
    has_finite: NDArray[np.bool_] = np.any(np.isfinite(strict_rate_maps), axis=1)
    peak_cm[~has_finite] = np.float32(-1.0)

    for pair_index, (start_cm, end_cm) in enumerate(pair_bounds):
        # noinspection PyTypeChecker
        in_pair: NDArray[np.bool_] = (peak_cm >= start_cm) & (peak_cm < end_cm)
        counts[pair_index] = int(np.sum(in_pair))
    return counts, pair_labels


def _draw_cue_zone_overlay(
    axes: plt.Axes,
    cue_boundaries_cm: tuple[float, ...],
    trigger_zone: tuple[float, float] | None,
    track_length_cm: float,
) -> None:
    """Draws cyan dotted verticals at the start and end of the cue zone holding the trigger zone.

    Notes:
        Shared by `plot_sorted_heatmap` and `plot_classified_heatmap`. ``cue_boundaries_cm`` arrives
        already in canonical coordinates (produced by `_resolve_canonical_cue_boundaries`), so no
        per-session shift is needed; the trigger zone center selects which canonical cue interval to
        bracket. No-ops when either input is missing.

    Args:
        axes: The matplotlib axes to overlay.
        cue_boundaries_cm: Canonical cue-identity transition positions for the trial type.
        trigger_zone: ``(start_cm, end_cm)`` for the per-session trigger zone, or ``None``.
        track_length_cm: Trial length used to extend the canonical boundary set with implicit
            ``0`` and ``track_length_cm`` endpoints.
    """
    if not cue_boundaries_cm or trigger_zone is None:
        return
    cue_left, cue_right = _cue_zone_around_trigger(
        cue_boundaries=cue_boundaries_cm,
        trigger_center=0.5 * (trigger_zone[0] + trigger_zone[1]),
        track_length_cm=track_length_cm,
    )
    axes.axvline(cue_left, color="cyan", linestyle=":", linewidth=1.4, alpha=0.9)
    axes.axvline(cue_right, color="cyan", linestyle=":", linewidth=1.4, alpha=0.9)


def _resolve_canonical_cue_boundaries(
    session_path: Path,
    trial_type: str,
    track_length_cm: float,
) -> tuple[float, ...]:
    """Returns canonical cue-identity transition positions for the trial type.

    Notes:
        Routes through `assemble_run_session_data` so cue boundaries reuse the same realignment path
        that produces the rate maps: trials get re-anchored to canonical zero via
        `realign_trial_starts_to_first_cue`, the partial leading trial is dropped by the completeness
        threshold in `compute_within_trial_position`, and the surviving samples already live in
        canonical position coordinates. Walking ``np.diff`` on the post-realignment cue array then
        captures every interior cue-identity transition in canonical coords directly — no
        ``cue_offset_cm`` shift, no missing trial-start boundary, no track-wrap collision with the
        implicit ``0`` / ``track_length`` endpoints.

    Args:
        session_path: Path to the forged session directory.
        trial_type: Trial type to evaluate.
        track_length_cm: Trial length used to clip boundaries to the open interval
            ``(0, track_length_cm)`` so the implicit track endpoints are not duplicated.

    Returns:
        Sorted-unique tuple of canonical cm positions where cue identity changes inside the trial.
        Empty when the session lacks data or the assembly fails.
    """
    try:
        run_session = assemble_run_session_data(session_path=session_path, trial_type=trial_type)
    except (KeyError, ValueError):
        return ()
    if run_session.position.size < 2:
        return ()

    cue = run_session.cue
    position = run_session.position
    # noinspection PyTypeChecker
    transitions: NDArray[np.int64] = np.where(np.diff(cue) != 0)[0] + 1
    if transitions.size == 0:
        return ()
    boundary_set: set[float] = set()
    for sample_index in transitions:
        value = float(position[sample_index])
        if 0.0 < value < track_length_cm:
            boundary_set.add(value)
    return tuple(sorted(boundary_set))


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


_PEAK_DISTRIBUTION_COLOR_BEFORE: str = "#2ca02c"
"""Line color for the 'before' group in the peak-distribution plot, matching the green Day-0 trace
in Sun et al. 2022 Fig. 1h."""
_PEAK_DISTRIBUTION_COLOR_AFTER: str = "#000000"
"""Line color for the 'after' group, matching the black Day-1 trace in Sun et al. 2022 Fig. 1h."""


def plot_place_cell_peak_distribution_around_shift(
    sessions: tuple[DatasetSession, ...],
    *,
    before_sessions: tuple[int, ...],
    after_sessions: tuple[int, ...],
    trial_type: str | None = None,
    animal_id: str | None = None,
    classifier: str = "place",
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots the cue-aligned distribution of place-cell peak positions before vs after a reward shift.

    Notes:
        Replicates the layout of Sun et al. 2022 (Nature) Fig. 1h: the track is partitioned into
        cue-aligned bins (one cue wide, no two adjacent cues sharing a bin), the fraction of strict-
        place cells whose rate-map peak falls inside each bin is computed per session, and the
        per-bin fractions are averaged across the supplied "before" and "after" session groups. The
        two averages render as line traces with SEM error bars over a shared x-axis. Vertical dashed
        lines mark the trigger-zone center from each group's first session so a reward-zone shift
        between groups reads off the figure directly.

        Cue-aligned bins are derived via the same canonical-realignment path used by
        ``plot_sorted_heatmap``: the first 'before' session contributes the cue boundary set, which
        is extended with ``0`` and ``track_length_cm`` to close the bin sequence. For trial types
        whose cue catalog is uniform (e.g., ``cyclic_4_cue``: A-Gray-B-Gray-C-Gray-D-Gray, 30 cm
        each), every bin is exactly one cue wide and the reward-trailing gray remains separate from
        the preceding named cue.

    References:
        Sun, C., Yang, W., Martin, J. & Tonegawa, S. Hippocampal neurons represent events as
        transferable units of experience. Nature 612, 478-486 (2022). Figure 1h.

    Args:
        sessions: Chronologically ordered DatasetSession entries for the animal.
        before_sessions: 1-indexed session numbers averaged together as the 'before-shift' group.
        after_sessions: 1-indexed session numbers averaged together as the 'after-shift' group.
        trial_type: Trial type whose tuning frames drive the distribution. ``None`` resolves to the
            first trial type in the first 'before' session's tuning feather.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        classifier: ``"place"`` (default) selects ``IS_STRICT_PLACE`` cells; ``"reward"`` selects
            ``IS_REWARD_CELL`` cells.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes carrying the two line traces and the reward-shift markers.
    """
    figure, axes = plt.subplots(
        1, 1, figsize=(7.0, 4.0), facecolor="white", dpi=figure_dpi, layout="constrained",
    )
    classifier_column = (
        TuningColumn.IS_STRICT_PLACE.value if classifier == "place"
        else TuningColumn.IS_REWARD_CELL.value
    )
    classifier_label = "place cells" if classifier == "place" else "reward cells"

    before_indices = _resolve_one_indexed_sessions(
        session_count=len(sessions), one_indexed=before_sessions,
    )
    after_indices = _resolve_one_indexed_sessions(
        session_count=len(sessions), one_indexed=after_sessions,
    )
    if not before_indices or not after_indices:
        axes.text(
            0.5, 0.5, "no sessions resolved for one or both groups",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    resolved_trial_type, track_length_cm, bin_edges_cm = _resolve_peak_distribution_layout(
        sessions=sessions, reference_session_index=before_indices[0], trial_type=trial_type,
    )
    if resolved_trial_type is None or bin_edges_cm.size < 2:
        axes.text(
            0.5, 0.5, "no cue-aligned bin layout available",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    before_fractions = _stack_group_fractions(
        sessions=sessions, session_indices=before_indices,
        trial_type=resolved_trial_type, classifier_column=classifier_column,
        bin_edges_cm=bin_edges_cm,
    )
    after_fractions = _stack_group_fractions(
        sessions=sessions, session_indices=after_indices,
        trial_type=resolved_trial_type, classifier_column=classifier_column,
        bin_edges_cm=bin_edges_cm,
    )

    bin_centers_cm: NDArray[np.float64] = 0.5 * (bin_edges_cm[:-1] + bin_edges_cm[1:])
    bin_count = bin_centers_cm.size

    before_label_indices = ", ".join(str(i + 1) for i in before_indices)
    after_label_indices = ", ".join(str(i + 1) for i in after_indices)
    _draw_peak_distribution_line(
        axes=axes, bin_centers_cm=bin_centers_cm, fractions=before_fractions,
        color=_PEAK_DISTRIBUTION_COLOR_BEFORE,
        label=f"Before shift (sessions {before_label_indices})",
    )
    _draw_peak_distribution_line(
        axes=axes, bin_centers_cm=bin_centers_cm, fractions=after_fractions,
        color=_PEAK_DISTRIBUTION_COLOR_AFTER,
        label=f"After shift (sessions {after_label_indices})",
    )

    former_reward_cm = _resolve_trigger_zone_center(
        session=sessions[before_indices[0]], trial_type=resolved_trial_type,
    )
    current_reward_cm = _resolve_trigger_zone_center(
        session=sessions[after_indices[0]], trial_type=resolved_trial_type,
    )
    _annotate_reward_marker(
        axes=axes, position_cm=former_reward_cm,
        color=_PEAK_DISTRIBUTION_COLOR_BEFORE, label="former reward",
        track_length_cm=float(track_length_cm),
    )
    if current_reward_cm is not None and current_reward_cm != former_reward_cm:
        _annotate_reward_marker(
            axes=axes, position_cm=current_reward_cm,
            color=_PEAK_DISTRIBUTION_COLOR_AFTER, label="current reward",
            track_length_cm=float(track_length_cm),
        )

    axes.set_xlim(0, float(track_length_cm))
    axes.set_xticks(bin_centers_cm.tolist())
    axes.set_xticklabels(
        [
            f"{int(round(bin_edges_cm[i]))}-{int(round(bin_edges_cm[i + 1]))}"
            for i in range(bin_count)
        ],
        fontsize=8, rotation=45, ha="right",
    )
    axes.set_xlabel("PF peak location (cm)", fontsize=10)
    axes.set_ylabel(f"Fraction of {classifier_label} (%)", fontsize=10)
    axes.tick_params(axis="y", labelsize=8)
    axes.set_ylim(bottom=0)
    axes.legend(loc="upper left", frameon=False, fontsize=9)

    title_subject = f"{classifier_label} peak distribution before vs after reward shift"
    figure.suptitle(
        f"Animal {animal_id} {title_subject}" if animal_id is not None
        else title_subject[:1].upper() + title_subject[1:],
        fontsize=11,
    )
    return figure


def plot_place_cell_peak_distribution_per_session(
    sessions: tuple[DatasetSession, ...],
    *,
    display_sessions: tuple[int, ...] | None = None,
    trial_type: str | None = None,
    animal_id: str | None = None,
    classifier: str = "place",
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots the cue-aligned place-cell peak distribution as one trace per session, colored by day.

    Notes:
        Same cue-aligned binning as ``plot_place_cell_peak_distribution_around_shift`` but draws one
        line per session instead of two SEM-banded group averages, so the operator can read the
        per-day trajectory of the peak distribution directly. Sessions are colored along a viridis
        gradient in chronological order (cool → warm) and labeled by their 1-indexed session number
        in the legend. ``display_sessions`` selects which sessions to render and reuses the same
        1-indexed convention as the rest of the per-session plots in the notebook; ``None`` renders
        every session in ``sessions``.

        Cue-aligned bin edges come from the first selected session via the same canonical-
        realignment path used by ``plot_place_cell_peak_distribution_around_shift``, so the x-axis
        ticks line up across both figures.

    Args:
        sessions: Chronologically ordered DatasetSession entries for the animal.
        display_sessions: 1-indexed session numbers to render. Out-of-range entries are silently
            skipped, duplicates are deduplicated, order is normalized to chronological. ``None``
            renders every session in ``sessions``.
        trial_type: Trial type whose tuning frames drive the distribution. ``None`` resolves to the
            first trial type in the first selected session's tuning feather.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        classifier: ``"place"`` (default) selects ``IS_STRICT_PLACE`` cells; ``"reward"`` selects
            ``IS_REWARD_CELL`` cells.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one axes carrying one line per selected session.
    """
    figure, axes = plt.subplots(
        1, 1, figsize=(8.0, 4.0), facecolor="white", dpi=figure_dpi, layout="constrained",
    )
    classifier_column = (
        TuningColumn.IS_STRICT_PLACE.value if classifier == "place"
        else TuningColumn.IS_REWARD_CELL.value
    )
    classifier_label = "place cells" if classifier == "place" else "reward cells"

    if display_sessions is None:
        resolved_indices = tuple(range(len(sessions)))
    else:
        resolved_indices = _resolve_one_indexed_sessions(
            session_count=len(sessions), one_indexed=display_sessions,
        )
    if not resolved_indices:
        axes.text(
            0.5, 0.5, "no sessions resolved",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    resolved_trial_type, track_length_cm, bin_edges_cm = _resolve_peak_distribution_layout(
        sessions=sessions, reference_session_index=resolved_indices[0], trial_type=trial_type,
    )
    if resolved_trial_type is None or bin_edges_cm.size < 2:
        axes.text(
            0.5, 0.5, "no cue-aligned bin layout available",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    fractions = _stack_group_fractions(
        sessions=sessions, session_indices=resolved_indices,
        trial_type=resolved_trial_type, classifier_column=classifier_column,
        bin_edges_cm=bin_edges_cm,
    )

    bin_centers_cm: NDArray[np.float64] = 0.5 * (bin_edges_cm[:-1] + bin_edges_cm[1:])
    bin_count = bin_centers_cm.size

    # Maps session ordinal to a viridis sample so the chronological order reads as cool → warm;
    # the ``max(..., 1)`` divisor guards the single-session case.
    colormap = plt.get_cmap("viridis")
    n_sessions = len(resolved_indices)
    for ordinal, sess_idx in enumerate(resolved_indices):
        color = colormap(ordinal / max(n_sessions - 1, 1))
        axes.plot(
            bin_centers_cm, fractions[ordinal],
            color=color, marker="o", markersize=4, linewidth=1.4,
            label=f"Session {sess_idx + 1}",
        )

    axes.set_xlim(0, float(track_length_cm))
    axes.set_xticks(bin_centers_cm.tolist())
    axes.set_xticklabels(
        [
            f"{int(round(bin_edges_cm[i]))}-{int(round(bin_edges_cm[i + 1]))}"
            for i in range(bin_count)
        ],
        fontsize=8, rotation=45, ha="right",
    )
    axes.set_xlabel("PF peak location (cm)", fontsize=10)
    axes.set_ylabel(f"Fraction of {classifier_label} (%)", fontsize=10)
    axes.tick_params(axis="y", labelsize=8)
    axes.set_ylim(bottom=0)
    axes.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=8,
    )

    session_label_indices = ", ".join(str(i + 1) for i in resolved_indices)
    title_subject = (
        f"{classifier_label} peak distribution over sessions {session_label_indices}"
    )
    figure.suptitle(
        f"Animal {animal_id} {title_subject}" if animal_id is not None
        else title_subject[:1].upper() + title_subject[1:],
        fontsize=11,
    )
    return figure


def plot_place_cell_peak_distribution_across_animals(
    sessions_by_animal: Mapping[str, tuple[DatasetSession, ...]],
    *,
    display_sessions: tuple[int, ...],
    trial_type: str | None = None,
    classifier: str = "place",
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots one cue-aligned place-cell peak distribution per animal, averaged over the supplied sessions.

    Notes:
        For each animal in ``sessions_by_animal``, gathers one per-bin fraction row per session in
        ``display_sessions`` (resolved against that animal's own chronological list), then draws a
        single viridis-colored mean ± SEM line — so each line summarizes one animal's behavior over
        the requested session range. Animals whose chronological list is shorter than the highest
        requested index contribute only the rows that resolve, with the SEM aggregator counting the
        actual number of contributing sessions.

        Cue-aligned bin edges come from the first animal's first resolved session via the same
        canonical-realignment path used by the other peak-distribution plots, so x-axis ticks line
        up across this figure and the per-session / around-shift variants. The function assumes
        every animal shares the same task / cue layout; mismatched cue catalogs are not detected.

    Args:
        sessions_by_animal: Mapping from animal id to that animal's chronologically-ordered
            DatasetSession tuple. Iteration order of the mapping drives the viridis ordinal and
            the legend order.
        display_sessions: 1-indexed session numbers to average for each animal. Out-of-range
            entries (relative to a given animal's chronological list) are dropped silently for
            that animal; duplicates are deduplicated; order is normalized to chronological.
        trial_type: Trial type whose tuning frames drive the distribution. ``None`` resolves to
            the first trial type in the reference session's tuning feather.
        classifier: ``"place"`` (default) selects ``IS_STRICT_PLACE`` cells; ``"reward"`` selects
            ``IS_REWARD_CELL`` cells.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with one mean ± SEM line per animal in ``sessions_by_animal``.
    """
    figure, axes = plt.subplots(
        1, 1, figsize=(8.0, 4.0), facecolor="white", dpi=figure_dpi, layout="constrained",
    )
    classifier_column = (
        TuningColumn.IS_STRICT_PLACE.value if classifier == "place"
        else TuningColumn.IS_REWARD_CELL.value
    )
    classifier_label = "place cells" if classifier == "place" else "reward cells"

    if not sessions_by_animal:
        axes.text(
            0.5, 0.5, "no animals provided",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    animal_ids = tuple(sessions_by_animal.keys())
    reference_sessions = sessions_by_animal[animal_ids[0]]
    reference_indices = _resolve_one_indexed_sessions(
        session_count=len(reference_sessions), one_indexed=display_sessions,
    )
    if not reference_indices:
        axes.text(
            0.5, 0.5, "no sessions resolved against the reference animal",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    resolved_trial_type, track_length_cm, bin_edges_cm = _resolve_peak_distribution_layout(
        sessions=reference_sessions, reference_session_index=reference_indices[0],
        trial_type=trial_type,
    )
    if resolved_trial_type is None or bin_edges_cm.size < 2:
        axes.text(
            0.5, 0.5, "no cue-aligned bin layout available",
            ha="center", va="center", transform=axes.transAxes,
        )
        axes.set_axis_off()
        return figure

    bin_centers_cm: NDArray[np.float64] = 0.5 * (bin_edges_cm[:-1] + bin_edges_cm[1:])
    bin_count = bin_centers_cm.size

    # Maps animal ordinal to a viridis sample so the cohort reads as cool → warm; the ``max(...,
    # 1)`` divisor guards the single-animal case.
    colormap = plt.get_cmap("viridis")
    n_animals = len(animal_ids)
    for ordinal, animal_id in enumerate(animal_ids):
        animal_sessions = sessions_by_animal[animal_id]
        animal_indices = _resolve_one_indexed_sessions(
            session_count=len(animal_sessions), one_indexed=display_sessions,
        )
        if not animal_indices:
            continue
        animal_fractions = _stack_group_fractions(
            sessions=animal_sessions,
            session_indices=animal_indices,
            trial_type=resolved_trial_type,
            classifier_column=classifier_column,
            bin_edges_cm=bin_edges_cm,
        )
        if animal_fractions.shape[0] == 0:
            continue
        color = colormap(ordinal / max(n_animals - 1, 1))
        _draw_peak_distribution_line(
            axes=axes, bin_centers_cm=bin_centers_cm, fractions=animal_fractions,
            color=color,
            label=f"Animal {animal_id} (n={animal_fractions.shape[0]} sessions)",
        )

    axes.set_xlim(0, float(track_length_cm))
    axes.set_xticks(bin_centers_cm.tolist())
    axes.set_xticklabels(
        [
            f"{int(round(bin_edges_cm[i]))}-{int(round(bin_edges_cm[i + 1]))}"
            for i in range(bin_count)
        ],
        fontsize=8, rotation=45, ha="right",
    )
    axes.set_xlabel("PF peak location (cm)", fontsize=10)
    axes.set_ylabel(f"Fraction of {classifier_label} (%)", fontsize=10)
    axes.tick_params(axis="y", labelsize=8)
    axes.set_ylim(bottom=0)
    axes.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False, fontsize=8,
    )

    session_label_indices = ", ".join(str(i + 1) for i in reference_indices)
    title_subject = (
        f"{classifier_label} peak distribution per animal averaged over sessions "
        f"{session_label_indices}"
    )
    figure.suptitle(
        title_subject[:1].upper() + title_subject[1:],
        fontsize=11,
    )
    return figure


def plot_post_shift_reward_zone_cells_in_session(
    sessions: tuple[DatasetSession, ...],
    *,
    pre_shift_session: int,
    post_shift_session: int,
    trial_type: str | None = None,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots pre- and post-shift rate-map heatmaps with new-reward-zone cells' peaks in red.

    Notes:
        Renders two panels side by side, each showing one specific session — the pre-shift
        ``pre_shift_session`` on the left and the post-shift ``post_shift_session`` on the right.
        Cells included are the strict-place cells from the post-shift session that also have a
        rate map in the pre-shift session, ordered by their post-shift peak so cells that move
        into the new reward band cluster contiguously and align horizontally between panels.

        Both panels are row-normalised (each row scaled to its own peak) so faint cells stay
        legible alongside strong ones. On rows whose post-shift peak falls inside the new-reward-
        zone window — the single cue (typically the gray cue at the reward) that contains the
        post-shift trigger-zone center — only the major-peak band of the rate map is recoloured
        in ``Reds``; the rest of the row, and every other row, stays in ``gray_r``. The pre-shift
        panel therefore shows the red marks at each migrating cell's pre-shift peak (where they
        were tuned before the shift), and the post-shift panel shows them inside the new reward
        zone.

        Each panel marks its own trigger zone with red-dashed verticals and the other panel's
        trigger zone with gray-dotted verticals. Cell ids must be within-animal-stable across the
        two sessions; multi-day cindra registration provides this guarantee inside an animal.

    Args:
        sessions: Chronologically ordered DatasetSession entries for the animal.
        pre_shift_session: 1-indexed session whose rate maps render in the left panel.
        post_shift_session: 1-indexed session whose rate maps render in the right panel and whose
            per-cell peak positions drive the red/gray classification.
        trial_type: Trial type whose tuning frames drive the analysis. ``None`` resolves to the
            first trial type in the post-shift session's tuning feather.
        animal_id: Optional animal id embedded in the figure suptitle; omitted when ``None``.
        figure_dpi: Output figure DPI.

    Returns:
        A matplotlib Figure with two heatmap axes (pre-shift on the left, post-shift on the right).
    """
    figure, (axes_pre, axes_post) = plt.subplots(
        1, 2, figsize=(11.0, 6.0), facecolor="white", dpi=figure_dpi, layout="constrained",
        sharey=True,
    )

    pre_shift_index = int(pre_shift_session) - 1
    if pre_shift_index < 0 or pre_shift_index >= len(sessions):
        axes_pre.text(
            0.5, 0.5, f"pre-shift session {pre_shift_session} out of range",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    post_shift_index = int(post_shift_session) - 1
    if post_shift_index < 0 or post_shift_index >= len(sessions):
        axes_pre.text(
            0.5, 0.5, f"post-shift session {post_shift_session} out of range",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    # Anchors the bin layout and trigger zone on the post-shift session so the new-reward-zone
    # window reflects the geometry of the session whose rate maps drive the red/gray decision.
    resolved_trial_type, track_length_cm, bin_edges_cm = _resolve_peak_distribution_layout(
        sessions=sessions, reference_session_index=post_shift_index, trial_type=trial_type,
    )
    if resolved_trial_type is None or bin_edges_cm.size < 2:
        axes_pre.text(
            0.5, 0.5, "no cue-aligned bin layout available",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    post_trigger_cm = _resolve_trigger_zone_center(
        session=sessions[post_shift_index], trial_type=resolved_trial_type,
    )
    if post_trigger_cm is None:
        axes_pre.text(
            0.5, 0.5, "post-shift trigger-zone center unavailable",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    # ``np.searchsorted`` returns the insertion index, so subtracting 1 yields the bin whose
    # ``[start, end)`` interval contains the post-shift trigger-zone center. The window is
    # restricted to that single bin (the gray cue covering the reward delivery), not the preceding
    # cue, so the red highlight only marks cells whose peak truly sits in the reward cue.
    reward_bin_idx = int(np.searchsorted(bin_edges_cm, post_trigger_cm, side="right")) - 1
    reward_bin_idx = max(min(reward_bin_idx, bin_edges_cm.size - 2), 0)
    window_start_cm = float(bin_edges_cm[reward_bin_idx])
    window_end_cm = float(bin_edges_cm[reward_bin_idx + 1])

    pre_zone_bounds = _resolve_trigger_zone_bounds(
        session=sessions[pre_shift_index], trial_type=resolved_trial_type,
    )
    post_zone_bounds = _resolve_trigger_zone_bounds(
        session=sessions[post_shift_index], trial_type=resolved_trial_type,
    )

    # Pulls strict-place cells from the post-shift session — the right panel renders that
    # session's rate maps verbatim, and the same session's per-cell peak position drives the
    # red/gray decision.
    post_session = sessions[post_shift_index]
    if not post_session.tuning_cells_path.exists():
        axes_pre.text(
            0.5, 0.5, f"post-shift session {post_shift_session} has no tuning_cells.feather",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    post_frame = pl.read_ipc(source=post_session.tuning_cells_path, memory_map=True)
    if TuningColumn.TRIAL_TYPE.value in post_frame.columns:
        post_frame = post_frame.filter(
            pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type
        )
    if TuningColumn.IS_STRICT_PLACE.value in post_frame.columns:
        post_frame = post_frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value))
    post_frame = post_frame.sort(TuningColumn.CELL_ID.value)
    if post_frame.height == 0:
        axes_pre.text(
            0.5, 0.5, f"no strict-place cells in session {post_shift_session}",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    # noinspection PyTypeChecker
    post_rate_maps: NDArray[np.float32] = np.asarray(
        post_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    if post_rate_maps.size == 0:
        axes_pre.text(
            0.5, 0.5, f"session {post_shift_session} cells have empty rate maps",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    rate_map_bin_count = int(post_rate_maps.shape[1])
    bin_size_cm = track_length_cm / rate_map_bin_count if rate_map_bin_count > 0 else 1.0
    finite_for_argmax = np.where(np.isfinite(post_rate_maps), post_rate_maps, -np.inf)
    # noinspection PyTypeChecker
    post_peak_bins: NDArray[np.int64] = (
        np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
    )
    # noinspection PyTypeChecker
    post_peak_cm: NDArray[np.float64] = (
        (post_peak_bins.astype(np.float64) + 0.5) * bin_size_cm
    )
    # noinspection PyTypeChecker
    post_cell_ids: NDArray[np.int64] = (
        post_frame[TuningColumn.CELL_ID.value].to_numpy().astype(np.int64, copy=False)
    )
    cell_id_to_post_peak: dict[int, float] = {
        int(cid): float(p) for cid, p in zip(post_cell_ids, post_peak_cm)
    }
    cell_id_to_post_rate_map: dict[int, NDArray[np.float32]] = {
        int(cid): rm for cid, rm in zip(post_cell_ids, post_rate_maps)
    }

    pre_path = sessions[pre_shift_index].tuning_cells_path
    if not pre_path.exists():
        axes_pre.text(
            0.5, 0.5, f"pre-shift session {pre_shift_session} has no tuning_cells.feather",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    pre_frame = pl.read_ipc(source=pre_path, memory_map=True)
    if TuningColumn.TRIAL_TYPE.value in pre_frame.columns:
        pre_frame = pre_frame.filter(
            pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type
        )
    if TuningColumn.IS_STRICT_PLACE.value in pre_frame.columns:
        pre_frame = pre_frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value))
    pre_frame = pre_frame.filter(
        pl.col(TuningColumn.CELL_ID.value).is_in(sorted(cell_id_to_post_peak.keys()))
    )
    pre_frame = pre_frame.sort(TuningColumn.CELL_ID.value)
    if pre_frame.height == 0:
        axes_pre.text(
            0.5, 0.5,
            f"no strict-place cells overlap sessions {pre_shift_session} and {post_shift_session}",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    pre_rate_maps: NDArray[np.float32] = np.asarray(
        pre_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    # noinspection PyTypeChecker
    pre_cell_ids: NDArray[np.int64] = (
        pre_frame[TuningColumn.CELL_ID.value].to_numpy().astype(np.int64, copy=False)
    )
    # noinspection PyTypeChecker
    cohort_post_peaks: NDArray[np.float64] = np.array(
        [cell_id_to_post_peak[int(cid)] for cid in pre_cell_ids], dtype=np.float64,
    )
    # Builds the post-shift rate-map matrix in the same row order as the pre-shift matrix so the
    # two panels render the same cells at the same y-positions after sorting.
    # noinspection PyTypeChecker
    cohort_post_rate_maps: NDArray[np.float32] = np.stack(
        [cell_id_to_post_rate_map[int(cid)] for cid in pre_cell_ids], axis=0,
    )

    # Sorts both matrices by post-shift peak so cells that move into the new reward band cluster
    # contiguously; ties broken stably keep cell-id order within a peak.
    # noinspection PyTypeChecker
    order: NDArray[np.int64] = np.argsort(cohort_post_peaks, kind="stable")
    sorted_pre_maps = pre_rate_maps[order]
    sorted_post_maps = cohort_post_rate_maps[order]
    sorted_post_peaks = cohort_post_peaks[order]
    in_window = (sorted_post_peaks >= window_start_cm) & (sorted_post_peaks < window_end_cm)

    n_cells = int(sorted_pre_maps.shape[0])
    extent = (0.0, float(track_length_cm), float(n_cells), 0.0)

    _draw_red_gray_heatmap(
        axes=axes_pre, rate_maps=sorted_pre_maps, in_window=in_window, extent=extent,
        active_zone=pre_zone_bounds, inactive_zone=post_zone_bounds,
    )
    _draw_red_gray_heatmap(
        axes=axes_post, rate_maps=sorted_post_maps, in_window=in_window, extent=extent,
        active_zone=post_zone_bounds, inactive_zone=pre_zone_bounds,
    )

    axes_pre.set_xlim(0.0, float(track_length_cm))
    axes_post.set_xlim(0.0, float(track_length_cm))
    axes_pre.set_xlabel("Track Position (cm)", fontsize=10)
    axes_post.set_xlabel("Track Position (cm)", fontsize=10)
    axes_pre.set_ylabel(
        "Strict-place cell (sorted by post-shift peak)", fontsize=10,
    )
    axes_pre.set_yticks([])
    axes_post.set_yticks([])
    axes_pre.set_title(f"Session {pre_shift_session} (pre-shift)", fontsize=10)
    axes_post.set_title(f"Session {post_shift_session} (post-shift)", fontsize=10)

    n_red = int(in_window.sum())
    title_subject = (
        f"strict-place cells — {n_red}/{n_cells} tuned to the new reward zone "
        f"(red rows; window {int(round(window_start_cm))}-{int(round(window_end_cm))} cm)"
    )
    figure.suptitle(
        f"Animal {animal_id} {title_subject}" if animal_id is not None
        else title_subject[:1].upper() + title_subject[1:],
        fontsize=11,
    )
    return figure


def plot_pre_shift_reward_zone_cells_in_session(
    sessions: tuple[DatasetSession, ...],
    *,
    pre_shift_session: int,
    post_shift_session: int,
    trial_type: str | None = None,
    animal_id: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots pre- and post-shift rate-map heatmaps with old-reward-zone cells' peaks in red.

    Notes:
        Inverse of `plot_post_shift_reward_zone_cells_in_session`. Cells included are the strict-
        place cells from the PRE-shift session that also have a rate map in the post-shift session,
        ordered by their pre-shift peak. Rows whose pre-shift peak falls inside the OLD reward-zone
        window (the single cue containing the pre-shift trigger-zone center) are recoloured in
        ``Reds`` at their major-peak band, while every other row stays in ``gray_r``. The pre-shift
        panel therefore shows the red marks anchored on the old reward zone, and the post-shift
        panel shows where those same cells ended up after the shift, answering "where did the cells
        that used to track the old reward zone migrate to?"
    """
    figure, (axes_pre, axes_post) = plt.subplots(
        1, 2, figsize=(11.0, 6.0), facecolor="white", dpi=figure_dpi, layout="constrained",
        sharey=True,
    )

    pre_shift_index = int(pre_shift_session) - 1
    if pre_shift_index < 0 or pre_shift_index >= len(sessions):
        axes_pre.text(
            0.5, 0.5, f"pre-shift session {pre_shift_session} out of range",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    post_shift_index = int(post_shift_session) - 1
    if post_shift_index < 0 or post_shift_index >= len(sessions):
        axes_pre.text(
            0.5, 0.5, f"post-shift session {post_shift_session} out of range",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    # Anchors the bin layout and trigger zone on the PRE-shift session so the old-reward-zone
    # window reflects the geometry of the session whose rate maps drive the red/gray decision.
    resolved_trial_type, track_length_cm, bin_edges_cm = _resolve_peak_distribution_layout(
        sessions=sessions, reference_session_index=pre_shift_index, trial_type=trial_type,
    )
    if resolved_trial_type is None or bin_edges_cm.size < 2:
        axes_pre.text(
            0.5, 0.5, "no cue-aligned bin layout available",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    pre_trigger_cm = _resolve_trigger_zone_center(
        session=sessions[pre_shift_index], trial_type=resolved_trial_type,
    )
    if pre_trigger_cm is None:
        axes_pre.text(
            0.5, 0.5, "pre-shift trigger-zone center unavailable",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    reward_bin_idx = int(np.searchsorted(bin_edges_cm, pre_trigger_cm, side="right")) - 1
    reward_bin_idx = max(min(reward_bin_idx, bin_edges_cm.size - 2), 0)
    window_start_cm = float(bin_edges_cm[reward_bin_idx])
    window_end_cm = float(bin_edges_cm[reward_bin_idx + 1])

    pre_zone_bounds = _resolve_trigger_zone_bounds(
        session=sessions[pre_shift_index], trial_type=resolved_trial_type,
    )
    post_zone_bounds = _resolve_trigger_zone_bounds(
        session=sessions[post_shift_index], trial_type=resolved_trial_type,
    )

    # Pulls strict-place cells from the pre-shift session — the left panel renders that session's
    # rate maps verbatim, and the same session's per-cell peak position drives the red/gray
    # decision (peak inside the OLD reward cue → red).
    pre_session = sessions[pre_shift_index]
    if not pre_session.tuning_cells_path.exists():
        axes_pre.text(
            0.5, 0.5, f"pre-shift session {pre_shift_session} has no tuning_cells.feather",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    pre_frame = pl.read_ipc(source=pre_session.tuning_cells_path, memory_map=True)
    if TuningColumn.TRIAL_TYPE.value in pre_frame.columns:
        pre_frame = pre_frame.filter(
            pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type
        )
    if TuningColumn.IS_STRICT_PLACE.value in pre_frame.columns:
        pre_frame = pre_frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value))
    pre_frame = pre_frame.sort(TuningColumn.CELL_ID.value)
    if pre_frame.height == 0:
        axes_pre.text(
            0.5, 0.5, f"no strict-place cells in session {pre_shift_session}",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    # noinspection PyTypeChecker
    pre_rate_maps: NDArray[np.float32] = np.asarray(
        pre_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    if pre_rate_maps.size == 0:
        axes_pre.text(
            0.5, 0.5, f"session {pre_shift_session} cells have empty rate maps",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    rate_map_bin_count = int(pre_rate_maps.shape[1])
    bin_size_cm = track_length_cm / rate_map_bin_count if rate_map_bin_count > 0 else 1.0
    finite_for_argmax = np.where(np.isfinite(pre_rate_maps), pre_rate_maps, -np.inf)
    # noinspection PyTypeChecker
    pre_peak_bins: NDArray[np.int64] = (
        np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
    )
    # noinspection PyTypeChecker
    pre_peak_cm: NDArray[np.float64] = (
        (pre_peak_bins.astype(np.float64) + 0.5) * bin_size_cm
    )
    # noinspection PyTypeChecker
    pre_cell_ids: NDArray[np.int64] = (
        pre_frame[TuningColumn.CELL_ID.value].to_numpy().astype(np.int64, copy=False)
    )
    cell_id_to_pre_peak: dict[int, float] = {
        int(cid): float(p) for cid, p in zip(pre_cell_ids, pre_peak_cm)
    }
    cell_id_to_pre_rate_map: dict[int, NDArray[np.float32]] = {
        int(cid): rm for cid, rm in zip(pre_cell_ids, pre_rate_maps)
    }

    post_path = sessions[post_shift_index].tuning_cells_path
    if not post_path.exists():
        axes_pre.text(
            0.5, 0.5, f"post-shift session {post_shift_session} has no tuning_cells.feather",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure
    post_frame = pl.read_ipc(source=post_path, memory_map=True)
    if TuningColumn.TRIAL_TYPE.value in post_frame.columns:
        post_frame = post_frame.filter(
            pl.col(TuningColumn.TRIAL_TYPE.value) == resolved_trial_type
        )
    if TuningColumn.IS_STRICT_PLACE.value in post_frame.columns:
        post_frame = post_frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value))
    post_frame = post_frame.filter(
        pl.col(TuningColumn.CELL_ID.value).is_in(sorted(cell_id_to_pre_peak.keys()))
    )
    post_frame = post_frame.sort(TuningColumn.CELL_ID.value)
    if post_frame.height == 0:
        axes_pre.text(
            0.5, 0.5,
            f"no strict-place cells overlap sessions {pre_shift_session} and {post_shift_session}",
            ha="center", va="center", transform=axes_pre.transAxes,
        )
        for axes in (axes_pre, axes_post):
            axes.set_axis_off()
        return figure

    # noinspection PyTypeChecker
    post_rate_maps: NDArray[np.float32] = np.asarray(
        post_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    # noinspection PyTypeChecker
    post_cell_ids: NDArray[np.int64] = (
        post_frame[TuningColumn.CELL_ID.value].to_numpy().astype(np.int64, copy=False)
    )
    # noinspection PyTypeChecker
    cohort_pre_peaks: NDArray[np.float64] = np.array(
        [cell_id_to_pre_peak[int(cid)] for cid in post_cell_ids], dtype=np.float64,
    )
    # noinspection PyTypeChecker
    cohort_pre_rate_maps: NDArray[np.float32] = np.stack(
        [cell_id_to_pre_rate_map[int(cid)] for cid in post_cell_ids], axis=0,
    )

    # Sorts both matrices by pre-shift peak so cells that anchor on the old reward zone cluster
    # contiguously; ties broken stably keep cell-id order within a peak.
    # noinspection PyTypeChecker
    order: NDArray[np.int64] = np.argsort(cohort_pre_peaks, kind="stable")
    sorted_pre_maps = cohort_pre_rate_maps[order]
    sorted_post_maps = post_rate_maps[order]
    sorted_pre_peaks = cohort_pre_peaks[order]
    in_window = (sorted_pre_peaks >= window_start_cm) & (sorted_pre_peaks < window_end_cm)

    n_cells = int(sorted_pre_maps.shape[0])
    extent = (0.0, float(track_length_cm), float(n_cells), 0.0)

    _draw_red_gray_heatmap(
        axes=axes_pre, rate_maps=sorted_pre_maps, in_window=in_window, extent=extent,
        active_zone=pre_zone_bounds, inactive_zone=post_zone_bounds,
    )
    _draw_red_gray_heatmap(
        axes=axes_post, rate_maps=sorted_post_maps, in_window=in_window, extent=extent,
        active_zone=post_zone_bounds, inactive_zone=pre_zone_bounds,
    )

    axes_pre.set_xlim(0.0, float(track_length_cm))
    axes_post.set_xlim(0.0, float(track_length_cm))
    axes_pre.set_xlabel("Track Position (cm)", fontsize=10)
    axes_post.set_xlabel("Track Position (cm)", fontsize=10)
    axes_pre.set_ylabel(
        "Strict-place cell (sorted by pre-shift peak)", fontsize=10,
    )
    axes_pre.set_yticks([])
    axes_post.set_yticks([])
    axes_pre.set_title(f"Session {pre_shift_session} (pre-shift)", fontsize=10)
    axes_post.set_title(f"Session {post_shift_session} (post-shift)", fontsize=10)

    n_red = int(in_window.sum())
    title_subject = (
        f"strict-place cells — {n_red}/{n_cells} tuned to the old reward zone "
        f"(red rows; window {int(round(window_start_cm))}-{int(round(window_end_cm))} cm)"
    )
    figure.suptitle(
        f"Animal {animal_id} {title_subject}" if animal_id is not None
        else title_subject[:1].upper() + title_subject[1:],
        fontsize=11,
    )
    return figure


_PEAK_HIGHLIGHT_THRESHOLD: float = 0.5
"""Row-normalized fluorescence value above which a bin is treated as part of a cell's major peak.
A cell with a clean, smoothed place field has a roughly half-max-width band of bins above this
cutoff, so it captures the peak region without bleeding into the tails."""


def _draw_red_gray_heatmap(
    axes: plt.Axes,
    rate_maps: NDArray[np.float32],
    in_window: NDArray[np.bool_],
    extent: tuple[float, float, float, float],
    active_zone: tuple[float, float] | None,
    inactive_zone: tuple[float, float] | None,
) -> None:
    """Renders one row-normalized rate-map heatmap with peak bands colored red on flagged rows.

    Notes:
        Row-normalises each cell's rate map so peaks are uniformly bright across cells, then
        renders every row in ``gray_r`` for the bulk of its tuning curve. On rows where
        ``in_window`` is True, the bins above ``_PEAK_HIGHLIGHT_THRESHOLD`` (the half-max-width
        band of the peak) are overwritten with the ``Reds`` sample at the same intensity, so the
        red ink lands only on the cell's major peak rather than tinting the whole row. The full
        rate map remains visible underneath in grayscale, including any secondary peaks. Active
        and inactive trigger-zone bounds are drawn as red-dashed and gray-dotted vertical pairs,
        omitted when ``None`` or when the two bounds coincide.
    """
    # noinspection PyTypeChecker
    row_max: NDArray[np.float32] = np.nanmax(rate_maps, axis=1, keepdims=True)
    safe_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
    normalized = np.clip(rate_maps / safe_max, 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)

    gray_cmap = plt.get_cmap("gray_r")
    red_cmap = plt.get_cmap("Reds")
    rgba = gray_cmap(normalized)
    if in_window.any():
        red_rgba = red_cmap(normalized)
        # Broadcast ``in_window`` (one entry per row) against the bin axis to limit the red
        # overwrite to flagged rows, and gate by ``normalized >= threshold`` so only the major
        # peak survives. The combined boolean mask indexes the (n_cells, n_bins) RGBA array
        # directly, swapping in red samples where both conditions hold.
        peak_mask = normalized >= _PEAK_HIGHLIGHT_THRESHOLD
        replace_mask = in_window[:, None] & peak_mask
        rgba[replace_mask] = red_rgba[replace_mask]

    axes.imshow(rgba, aspect="auto", origin="upper", extent=extent, interpolation="nearest")

    if active_zone is not None:
        axes.axvline(active_zone[0], color="red", linestyle="--", linewidth=1.0, zorder=2)
        axes.axvline(active_zone[1], color="red", linestyle="--", linewidth=1.0, zorder=2)
    if inactive_zone is not None and inactive_zone != active_zone:
        axes.axvline(inactive_zone[0], color="gray", linestyle=":", linewidth=1.0, zorder=2)
        axes.axvline(inactive_zone[1], color="gray", linestyle=":", linewidth=1.0, zorder=2)


def _resolve_trigger_zone_bounds(
    session: DatasetSession, trial_type: str,
) -> tuple[float, float] | None:
    """Returns the trigger-zone ``(start_cm, end_cm)`` for a session, or ``None`` when unavailable."""
    if not session.geometry_path.exists():
        return None
    geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
    entry = geometry.entries.get(trial_type)
    if entry is None:
        return None
    return (
        float(entry.stimulus_trigger_zone_start_cm),
        float(entry.stimulus_trigger_zone_end_cm),
    )


def _collect_strict_place_peaks(
    session: DatasetSession,
    trial_type: str,
    track_length_cm: float,
) -> dict[int, float]:
    """Returns ``{cell_id: peak_cm}`` for every strict-place cell in the session's tuning frame.

    Notes:
        Reads the session's ``tuning_cells.feather``, filters to ``IS_STRICT_PLACE`` rows for the
        given trial type, and converts each surviving cell's ``RATE_MAP`` argmax to centimeters
        via the rate-map bin spacing. Returns an empty mapping when the feather is missing,
        empty, or has no strict-place cells.
    """
    if not session.tuning_cells_path.exists():
        return {}
    frame = pl.read_ipc(source=session.tuning_cells_path, memory_map=True)
    if TuningColumn.TRIAL_TYPE.value in frame.columns:
        frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == trial_type)
    if TuningColumn.IS_STRICT_PLACE.value in frame.columns:
        frame = frame.filter(pl.col(TuningColumn.IS_STRICT_PLACE.value))
    if frame.height == 0:
        return {}
    # noinspection PyTypeChecker
    rate_maps: NDArray[np.float32] = np.asarray(
        frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
    )
    if rate_maps.size == 0:
        return {}
    rate_map_bin_count = int(rate_maps.shape[1])
    bin_size_cm = track_length_cm / rate_map_bin_count if rate_map_bin_count > 0 else 1.0
    finite_for_argmax = np.where(np.isfinite(rate_maps), rate_maps, -np.inf)
    # noinspection PyTypeChecker
    peak_bins: NDArray[np.int64] = np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
    # noinspection PyTypeChecker
    peak_cm: NDArray[np.float64] = (peak_bins.astype(np.float64) + 0.5) * bin_size_cm
    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int64] = (
        frame[TuningColumn.CELL_ID.value].to_numpy().astype(np.int64, copy=False)
    )
    return {int(cid): float(p) for cid, p in zip(cell_ids, peak_cm)}


def _resolve_one_indexed_sessions(
    session_count: int, one_indexed: tuple[int, ...],
) -> tuple[int, ...]:
    """Maps a tuple of 1-indexed session numbers to deduplicated 0-indexed integers in chronological order."""
    resolved: list[int] = []
    for value in one_indexed:
        idx = int(value) - 1
        if 0 <= idx < session_count and idx not in resolved:
            resolved.append(idx)
    resolved.sort()
    return tuple(resolved)


def _resolve_peak_distribution_layout(
    sessions: tuple[DatasetSession, ...],
    reference_session_index: int,
    trial_type: str | None,
) -> tuple[str | None, float, NDArray[np.float64]]:
    """Returns ``(trial_type, track_length_cm, bin_edges_cm)`` for the peak-distribution figure.

    Notes:
        Resolves the trial type from the reference session's tuning frame when not supplied. Walks
        a single full canonical trial (post-realignment, skipping the leading partial trial whose
        transitions are offset by ``cue_offset_cm``) to collect clean cue-aligned bin edges, then
        prepends ``0`` and appends ``track_length_cm`` so the bin sequence closes the track.
        Returns an empty edges array when the reference session is missing data.
    """
    reference_session = sessions[reference_session_index]
    resolved_trial_type = trial_type
    if resolved_trial_type is None and reference_session.tuning_cells_path.exists():
        frame = pl.read_ipc(source=reference_session.tuning_cells_path, memory_map=True)
        if TuningColumn.TRIAL_TYPE.value in frame.columns:
            unique_trial_types = frame[TuningColumn.TRIAL_TYPE.value].unique().to_list()
            if unique_trial_types:
                resolved_trial_type = str(unique_trial_types[0])
    if resolved_trial_type is None:
        # noinspection PyTypeChecker
        empty: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        return None, 0.0, empty

    if not reference_session.geometry_path.exists():
        # noinspection PyTypeChecker
        empty = np.zeros(0, dtype=np.float64)
        return resolved_trial_type, 0.0, empty
    geometry = TrialGeometry.from_yaml(file_path=reference_session.geometry_path)
    geometry_entry = geometry.entries.get(resolved_trial_type)
    if geometry_entry is None:
        # noinspection PyTypeChecker
        empty = np.zeros(0, dtype=np.float64)
        return resolved_trial_type, 0.0, empty
    track_length_cm = float(geometry_entry.trial_length_cm)
    if track_length_cm <= 0:
        # noinspection PyTypeChecker
        empty = np.zeros(0, dtype=np.float64)
        return resolved_trial_type, 0.0, empty

    bin_edges_cm = _resolve_full_trial_bin_edges_cm(
        session=reference_session,
        trial_type=resolved_trial_type,
        track_length_cm=track_length_cm,
    )
    return resolved_trial_type, track_length_cm, bin_edges_cm


def _resolve_full_trial_bin_edges_cm(
    session: DatasetSession,
    trial_type: str,
    track_length_cm: float,
) -> NDArray[np.float64]:
    """Walks one full canonical trial of the session to derive clean cue-aligned bin edges.

    Notes:
        ``assemble_run_session_data`` re-anchors trials to the canonical first-cue start when
        ``cue_offset_cm > 0`` and drops trials that fall below the 90% completeness threshold. The
        leading post-realignment trial typically begins mid-first-cue (the run-state subset
        starts partway through a cycle), so its cue transitions are offset by ``cue_offset_cm``
        relative to the canonical layout. Picking the first trial whose measured length is closest
        to ``track_length_cm`` skips that partial leader and produces canonical 30-cm-aligned
        edges for ``cyclic_4_cue``-style trials.
    """
    try:
        run_session = assemble_run_session_data(session_path=session.session_path, trial_type=trial_type)
    except (KeyError, ValueError):
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float64)
    if run_session.position.size < 2:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float64)

    trial_ids = run_session.trial_ids
    position = run_session.position
    cue = run_session.cue
    unique_trials = np.unique(trial_ids)
    if unique_trials.size == 0:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float64)

    # Picks the trial whose measured length is closest to the canonical track length. The first
    # realigned trial is typically a leading partial whose measured length is smaller than the
    # canonical cycle, so this selection naturally drops it.
    best_trial_id = int(unique_trials[0])
    best_length = -np.inf
    for candidate in unique_trials.tolist():
        # noinspection PyTypeChecker
        mask: NDArray[np.bool_] = trial_ids == int(candidate)
        if not mask.any():
            continue
        candidate_position = position[mask]
        candidate_length = float(candidate_position[-1] - candidate_position[0])
        if abs(candidate_length - track_length_cm) < abs(best_length - track_length_cm):
            best_length = candidate_length
            best_trial_id = int(candidate)

    # noinspection PyTypeChecker
    selected_mask: NDArray[np.bool_] = trial_ids == best_trial_id
    target_cue = cue[selected_mask]
    target_position = position[selected_mask]
    if target_cue.size < 2:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.float64)

    # noinspection PyTypeChecker
    transitions: NDArray[np.int64] = np.flatnonzero(np.diff(target_cue.astype(np.int64))) + 1
    boundary_values: list[float] = []
    for sample_index in transitions.tolist():
        value = float(target_position[sample_index])
        if 0.0 < value < track_length_cm:
            boundary_values.append(value)
    boundary_values = sorted(set(boundary_values))
    # noinspection PyTypeChecker
    return np.array((0.0, *boundary_values, track_length_cm), dtype=np.float64)


def _stack_group_fractions(
    sessions: tuple[DatasetSession, ...],
    session_indices: tuple[int, ...],
    trial_type: str,
    classifier_column: str,
    bin_edges_cm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the (n_sessions, n_bins) per-session fractions of classified cells in each cue bin.

    Notes:
        Sessions whose tuning feather is missing or carries zero classified cells contribute a row
        of zeros so the SEM aggregator still sees the full group size. Peak positions are computed
        from each cell's ``RATE_MAP`` argmax converted to centimeters via the rate map's bin
        spacing; cells whose entire rate-map row is non-finite are dropped before binning.
    """
    bin_count = max(int(bin_edges_cm.size) - 1, 1)
    rows: list[NDArray[np.float64]] = []
    for sess_idx in session_indices:
        session = sessions[sess_idx]
        if not session.tuning_cells_path.exists():
            # noinspection PyTypeChecker
            rows.append(np.zeros(bin_count, dtype=np.float64))
            continue
        frame = pl.read_ipc(source=session.tuning_cells_path, memory_map=True)
        if TuningColumn.TRIAL_TYPE.value in frame.columns:
            frame = frame.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == trial_type)
        if frame.height == 0 or classifier_column not in frame.columns:
            # noinspection PyTypeChecker
            rows.append(np.zeros(bin_count, dtype=np.float64))
            continue
        classified_frame = frame.filter(pl.col(classifier_column))
        total_classified = int(classified_frame.height)
        if total_classified == 0:
            # noinspection PyTypeChecker
            rows.append(np.zeros(bin_count, dtype=np.float64))
            continue
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.asarray(
            classified_frame[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
        )
        if rate_maps.size == 0:
            # noinspection PyTypeChecker
            rows.append(np.zeros(bin_count, dtype=np.float64))
            continue
        rate_map_bin_count = int(rate_maps.shape[1])
        bin_size_cm = float(bin_edges_cm[-1]) / rate_map_bin_count if rate_map_bin_count > 0 else 1.0
        finite_for_argmax = np.where(np.isfinite(rate_maps), rate_maps, -np.inf)
        # noinspection PyTypeChecker
        peak_bins: NDArray[np.int64] = np.argmax(finite_for_argmax, axis=1).astype(np.int64, copy=False)
        # noinspection PyTypeChecker
        peak_cm: NDArray[np.float64] = (
            (peak_bins.astype(np.float64) + 0.5) * bin_size_cm
        )
        # noinspection PyTypeChecker
        has_finite: NDArray[np.bool_] = np.any(np.isfinite(rate_maps), axis=1)
        peak_cm = peak_cm[has_finite]
        # noinspection PyTypeChecker
        counts, _ = np.histogram(peak_cm, bins=bin_edges_cm)
        # noinspection PyTypeChecker
        fraction: NDArray[np.float64] = (
            counts.astype(np.float64) / float(total_classified) * 100.0
        )
        rows.append(fraction)
    if not rows:
        # noinspection PyTypeChecker
        return np.zeros((0, bin_count), dtype=np.float64)
    return np.stack(rows, axis=0)


def _draw_peak_distribution_line(
    axes: plt.Axes,
    bin_centers_cm: NDArray[np.float64],
    fractions: NDArray[np.float64],
    color: str,
    label: str,
) -> None:
    """Draws one mean ± SEM line trace from a (n_sessions, n_bins) fraction stack."""
    if fractions.shape[0] == 0:
        return
    # noinspection PyTypeChecker
    mean_per_bin: NDArray[np.float64] = fractions.mean(axis=0)
    if fractions.shape[0] > 1:
        # noinspection PyTypeChecker
        sem_per_bin: NDArray[np.float64] = fractions.std(axis=0, ddof=1) / np.sqrt(fractions.shape[0])
    else:
        # noinspection PyTypeChecker
        sem_per_bin = np.zeros_like(mean_per_bin)
    axes.errorbar(
        bin_centers_cm, mean_per_bin, yerr=sem_per_bin,
        color=color, marker="o", markersize=4, linewidth=1.4, capsize=3,
        label=label,
    )


def _resolve_trigger_zone_center(
    session: DatasetSession, trial_type: str,
) -> float | None:
    """Returns the trigger-zone center in centimeters for a session, or ``None`` when unavailable."""
    if not session.geometry_path.exists():
        return None
    geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
    entry = geometry.entries.get(trial_type)
    if entry is None:
        return None
    return 0.5 * (
        float(entry.stimulus_trigger_zone_start_cm) + float(entry.stimulus_trigger_zone_end_cm)
    )


def _annotate_reward_marker(
    axes: plt.Axes, position_cm: float | None, color: str, label: str, track_length_cm: float,
) -> None:
    """Draws one dashed vertical line at ``position_cm`` and labels it at the top of the axes.

    Notes:
        Anchors the label to the left of the marker when the marker sits in the right third of the
        track so the text fits inside the axes bounds, otherwise anchors to the right of the
        marker.
    """
    if position_cm is None:
        return
    axes.axvline(position_cm, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
    on_right_edge = position_cm > 0.66 * track_length_cm
    axes.annotate(
        label, xy=(position_cm, 1.0), xycoords=("data", "axes fraction"),
        xytext=(-2 if on_right_edge else 2, -2), textcoords="offset points",
        ha="right" if on_right_edge else "left", va="top", fontsize=8, color=color,
    )


