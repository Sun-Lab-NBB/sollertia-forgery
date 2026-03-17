"""Analyzes how spatial state representations depend on the previous trial type.

For each frame, labels its trial with a condition defined by the current and previous
trial type (e.g. ABC→ABDC). Computes condition-averaged tuning curves, trial-by-trial
variability, and remapping metrics restricted to the shared spatial segment where cue
sequences are identical across trial types.

Conditions: For two trial types A and B, the four history conditions are AA, BA, AB,
BB. Trials with no prior trial receive a "start" condition. Metrics are computed within
the shared spatial region, identified via get_shared_bins().

Analyses:
    1. Mean tuning curves per history condition — (n_shared_bins, n_cells) maps.
    2. Per-bin variance — trial-by-trial variability at each shared spatial bin.
    3. Per-cell spatial correlation — Pearson r between condition pairs, restricted
       to the shared segment. Computed as both session-mean and trial-by-trial.
    4. Population vector (PV) correlation at each shared bin — population-level
       remapping profile across conditions.
    5. Peak location shift — per-cell change in field center between conditions.
    6. Amplitude change — per-cell ratio of peak firing rate between conditions.
    7. Parametric statistics comparing condition pairs; optional shuffle validation.

Multi-session wrapper aggregates per-session metrics and plots time series by date.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
from ataraxis_base_utilities import console

from df_processing import get_bin_size
from place_field_detection import detect_place_fields
from cross_correlation_1 import (
    per_cell_spatial_correlation,
    population_vector_correlation,
    get_shared_bins,
    get_mean_tuning_curves,
)
import plot_utils as pfmt


# Minimum running speed used by detect_place_fields() and all tuning curve computations.
_MIN_SPEED_CM_S: float = 2.0

# Label assigned to the previous-trial-type slot when no prior trial exists.
_START_LABEL: str = "start"


# HELPERS


def _label_trial_conditions(df: pl.DataFrame) -> dict[int, tuple[str, str]]:
    """Maps each trial ID to its (current_type, previous_type) condition pair.

    Trials are ordered by their integer trial index. The first trial receives
    _START_LABEL as its previous type. All other trials receive the trial type
    of the immediately preceding trial.

    Args:
        df: Frame-level DataFrame with 'trial' and 'trial_type' columns.

    Returns:
        Dictionary mapping each trial integer ID to a (current_type, previous_type)
        tuple. previous_type is _START_LABEL for the first trial in the session.
    """
    trial_summary = (
        df.select(["trial", "trial_type"])
        .unique(subset=["trial"])
        .sort("trial")
    )
    trial_ids = trial_summary["trial"].to_list()
    trial_types = trial_summary["trial_type"].to_list()

    condition_by_trial: dict[int, tuple[str, str]] = {}
    for index, (trial_id, current_type) in enumerate(zip(trial_ids, trial_types)):
        previous_type = trial_types[index - 1] if index > 0 else _START_LABEL
        condition_by_trial[trial_id] = (current_type, previous_type)

    return condition_by_trial


def _compute_per_trial_tuning_curves(
    df: pl.DataFrame,
    condition_by_trial: dict[int, tuple[str, str]],
    n_shared_bins: int,
    signal_col: str,
) -> dict[str, NDArray[np.float64]]:
    """Computes per-trial tuning curves for each history condition.

    For each trial, accumulates signals into spatial bins via scatter-add and
    computes the bin mean. Bins beyond n_shared_bins are discarded. Bins with
    no frames in a trial are set to NaN.

    Args:
        df: Speed-filtered frame-level DataFrame with 'trial', 'distance_bin',
            and signal_col columns.
        condition_by_trial: Mapping from trial ID to (current_type, previous_type).
        n_shared_bins: Number of bins in the shared spatial segment.
        signal_col: Column name containing per-frame neural signals (list per row).

    Returns:
        Dictionary mapping condition label strings (e.g. "ABC_ABDC") to arrays
        of shape (n_trials_in_condition, n_shared_bins, n_cells).
    """
    all_signals = np.vstack(df[signal_col].to_list())
    n_cells = all_signals.shape[1]
    trial_ids = df["trial"].to_numpy()
    bin_indices = df["distance_bin"].to_numpy().clip(0, n_shared_bins - 1)

    per_condition: dict[str, list[NDArray[np.float64]]] = {}

    for trial_id, (current_type, previous_type) in condition_by_trial.items():
        condition = f"{current_type}_{previous_type}"
        trial_mask = trial_ids == trial_id
        if trial_mask.sum() == 0:
            continue

        trial_signals = all_signals[trial_mask]
        trial_bins = bin_indices[trial_mask]

        bin_sums = np.zeros((n_shared_bins, n_cells), dtype=np.float64)
        bin_counts = np.zeros(n_shared_bins, dtype=np.float64)
        np.add.at(bin_sums, trial_bins, trial_signals)
        np.add.at(bin_counts, trial_bins, 1)

        with np.errstate(invalid="ignore"):
            tuning_curve = bin_sums / bin_counts[:, np.newaxis]

        # Mark unvisited bins as NaN rather than zero.
        tuning_curve[bin_counts == 0] = np.nan

        if condition not in per_condition:
            per_condition[condition] = []
        per_condition[condition].append(tuning_curve)

    return {
        condition: np.stack(curves, axis=0)
        for condition, curves in per_condition.items()
    }


def _get_peak_bins(
    mean_curves: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the bin index of peak activity for each cell.

    Args:
        mean_curves: Session-averaged tuning curves, shape (n_shared_bins, n_cells).

    Returns:
        Peak bin index per cell, shape (n_cells,). NaN for cells with all-NaN bins.
    """
    n_cells = mean_curves.shape[1]
    peak_bins = np.full(n_cells, np.nan, dtype=np.float64)
    for cell_index in range(n_cells):
        column = mean_curves[:, cell_index]
        if not np.all(np.isnan(column)):
            peak_bins[cell_index] = float(np.nanargmax(column))
    return peak_bins


def _build_comparison_pairs(
    trial_types: list[str],
    comparison_type: str,
) -> list[tuple[str, str, str]]:
    """Builds the list of condition pairs to compare.

    Effect-of-prior comparisons hold current trial type fixed and vary the
    previous trial type. Effect-of-current comparisons hold previous trial type
    fixed and vary the current trial type.

    Args:
        trial_types: The two trial type strings present in the session (e.g.
            ["ABC", "ABDC"]).
        comparison_type: One of "prior" (effect of previous trial only), "current"
            (effect of current trial only), or "both".

    Returns:
        List of (condition_a, condition_b, display_label) tuples. Each entry
        defines one pairwise comparison by its two condition key strings and a
        human-readable label for plot titles and legend entries.
    """
    if len(trial_types) != 2:
        console.error(
            message=(
                f"Unable to build comparison pairs. Exactly two trial types are required, "
                f"but got {len(trial_types)}: {trial_types}."
            ),
            error=ValueError,
        )

    type_a, type_b = trial_types[0], trial_types[1]
    pairs: list[tuple[str, str, str]] = []

    if comparison_type in ("prior", "both"):
        # Same current track, different prior — pure history effect.
        pairs.append((
            f"{type_a}_{type_a}",
            f"{type_a}_{type_b}",
            f"History effect on {type_a} (prev {type_a} vs prev {type_b})",
        ))
        pairs.append((
            f"{type_b}_{type_b}",
            f"{type_b}_{type_a}",
            f"History effect on {type_b} (prev {type_b} vs prev {type_a})",
        ))

    if comparison_type in ("current", "both"):
        # Same prior, different current track — remapping conditioned on history.
        pairs.append((
            f"{type_a}_{type_a}",
            f"{type_b}_{type_a}",
            f"Remapping after {type_a} (current {type_a} vs current {type_b})",
        ))
        pairs.append((
            f"{type_a}_{type_b}",
            f"{type_b}_{type_b}",
            f"Remapping after {type_b} (current {type_a} vs current {type_b})",
        ))

    return pairs


def _compute_trial_by_trial_correlations(
    per_trial_curves: NDArray[np.float64],
    reference_curves: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Correlates each trial's tuning curve against a fixed reference.

    Args:
        per_trial_curves: Shape (n_trials, n_shared_bins, n_cells).
        reference_curves: Shape (n_shared_bins, n_cells). Used as the reference
            for all trials.

    Returns:
        Per-trial per-cell Pearson r, shape (n_trials, n_cells). NaN where
        correlation is undefined (zero-variance or insufficient valid bins).
    """
    n_trials = per_trial_curves.shape[0]
    n_cells = per_trial_curves.shape[2]
    correlations = np.full((n_trials, n_cells), np.nan, dtype=np.float64)

    for trial_index in range(n_trials):
        correlations[trial_index] = per_cell_spatial_correlation(
            tuning_a=per_trial_curves[trial_index],
            tuning_b=reference_curves,
        )

    return correlations


def _leave_one_out_correlations(
    per_trial_curves: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Correlates each trial against the mean of all other trials in the same condition.

    Provides a within-condition reliability baseline. Each trial is compared against
    the leave-one-out mean to avoid inflating the correlation by including the trial
    itself in the reference.

    Args:
        per_trial_curves: Shape (n_trials, n_shared_bins, n_cells).

    Returns:
        Per-trial per-cell Pearson r, shape (n_trials, n_cells).
    """
    n_trials = per_trial_curves.shape[0]
    correlations = np.full((n_trials, per_trial_curves.shape[2]), np.nan, dtype=np.float64)

    for trial_index in range(n_trials):
        other_mask = np.ones(n_trials, dtype=bool)
        other_mask[trial_index] = False
        leave_one_out_mean = np.nanmean(per_trial_curves[other_mask], axis=0)
        correlations[trial_index] = per_cell_spatial_correlation(
            tuning_a=per_trial_curves[trial_index],
            tuning_b=leave_one_out_mean,
        )

    return correlations


# CORE COMPUTATION


def compute_history_dependent_remapping(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = "multi_day_spikes",
    place_cells_only: bool = False,
    comparison_type: str = "both",
    run_shuffle: bool = False,
    n_shuffles: int = 500,
) -> dict:
    """Computes all history-dependent remapping metrics for a single session.

    Labels each trial with its (current_type, previous_type) condition, restricts
    analysis to the shared spatial segment, and computes per-condition tuning curves
    and a battery of remapping metrics for each pairwise comparison.

    Notes:
        Only sessions with exactly two trial types are currently supported. If
        place_cells_only is True, place field detection runs on the full session
        (not per-condition) and the resulting cell mask is applied uniformly.

        The shuffle test (run_shuffle=True) permutes the previous-trial-type labels
        across trials within each current-trial-type group, preserving the number of
        trials per current type but randomising history. Per-cell spatial correlations
        are recomputed on each shuffle to generate a null distribution.

    Args:
        df: Frame-level DataFrame from process_session().
        config: Experiment configuration dictionary.
        metadata: Session metadata dictionary from df_processing (must contain
            bin_size_cm).
        signal_col: Column name for neural signals used in tuning curve computation.
        place_cells_only: Determines whether to restrict analysis to place cells
            detected by detect_place_fields() on the full session.
        comparison_type: Which condition pairs to compute metrics for. "prior"
            computes effect-of-prior pairs, "current" computes effect-of-current
            pairs, and "both" computes all four pairs.
        run_shuffle: Determines whether to run a permutation test to assess
            statistical significance of per-cell spatial correlation differences.
        n_shuffles: Number of shuffle iterations used when run_shuffle is True.

    Returns:
        Dictionary containing the following keys. condition_means and
        condition_variances map condition label strings (e.g. "ABC_ABDC") to
        (n_shared_bins, n_cells) arrays. condition_per_trial maps condition labels
        to (n_trials, n_shared_bins, n_cells) arrays. pv_correlations,
        spatial_correlations, within_correlations, cross_correlations, peak_shifts,
        and amplitude_changes map comparison label strings to their respective
        arrays. comparisons is the list of (cond_a, cond_b, label) tuples used.
        divergence_cm, n_shared_bins, bin_size_cm, trial_types, conditions, and
        n_cells provide session-level metadata. cell_mask is a boolean array of
        shape (n_cells,) indicating which cells were included. parametric_tests
        maps comparison labels to scipy t-test results. shuffle_p_values maps
        comparison labels to per-cell permutation p-values if run_shuffle is True.
    """
    bin_size_cm = get_bin_size(metadata=metadata)

    # Speed filter matches the threshold used by detect_place_fields().
    if "speed_cm_s" in df.columns:
        df = df.filter(pl.col("speed_cm_s") >= _MIN_SPEED_CM_S)

    trial_types = sorted(df["trial_type"].unique().to_list())
    if len(trial_types) != 2:
        console.error(
            message=(
                f"Unable to compute history-dependent remapping. Exactly two trial types "
                f"are required, but found {len(trial_types)}: {trial_types}."
            ),
            error=ValueError,
        )

    type_a, type_b = trial_types[0], trial_types[1]
    divergence_cm, n_shared_bins = get_shared_bins(
        config=config,
        type_a=type_a,
        type_b=type_b,
        metadata=metadata,
    )

    n_cells = len(np.array(df[signal_col][0]))

    # Optionally restrict to cells that are place cells in any trial type.
    cell_mask = np.ones(n_cells, dtype=bool)
    if place_cells_only:
        place_field_result = detect_place_fields(
            df=df,
            config=config,
            signal_col=signal_col,
            bin_size_cm=bin_size_cm,
        )
        cell_mask = place_field_result.is_place_cell_any

    condition_by_trial = _label_trial_conditions(df=df)
    per_trial_all = _compute_per_trial_tuning_curves(
        df=df,
        condition_by_trial=condition_by_trial,
        n_shared_bins=n_shared_bins,
        signal_col=signal_col,
    )

    # Apply cell mask to all per-trial arrays.
    per_trial_masked: dict[str, NDArray[np.float64]] = {
        condition: curves[:, :, cell_mask]
        for condition, curves in per_trial_all.items()
    }

    # Compute mean tuning curves per condition.
    condition_means: dict[str, NDArray[np.float64]] = {}
    for condition, curves in per_trial_masked.items():
        condition_means[condition] = np.nanmean(curves, axis=0)

    # Normalize per-trial curves by each cell's peak across all conditions before computing
    # variance. This makes variance dimensionless (units of peak²) and comparable across
    # sessions and signal types regardless of raw dFF scale.
    n_cells_included = int(cell_mask.sum())
    global_peak = np.zeros(n_cells_included, dtype=np.float64)
    for mean_curve in condition_means.values():
        per_cell_peak = np.nanmax(mean_curve, axis=0)
        global_peak = np.maximum(global_peak, per_cell_peak)
    safe_peak = np.where(global_peak > 0, global_peak, 1.0)

    condition_variances: dict[str, NDArray[np.float64]] = {}
    for condition, curves in per_trial_masked.items():
        normalized_curves = curves / safe_peak[np.newaxis, np.newaxis, :]
        # ddof=1 requires at least 2 trials; single-trial conditions yield all-NaN variance.
        if curves.shape[0] > 1:
            condition_variances[condition] = np.nanvar(normalized_curves, axis=0, ddof=1)
        else:
            condition_variances[condition] = np.full(curves.shape[1:], np.nan, dtype=np.float64)

    conditions = sorted(condition_means.keys())
    comparisons = _build_comparison_pairs(
        trial_types=trial_types,
        comparison_type=comparison_type,
    )

    # Compute comparison metrics for each pair.
    pv_correlations: dict[str, NDArray[np.float64]] = {}
    spatial_correlations: dict[str, NDArray[np.float64]] = {}
    within_correlations: dict[str, NDArray[np.float64]] = {}
    cross_correlations: dict[str, NDArray[np.float64]] = {}
    peak_shifts: dict[str, NDArray[np.float64]] = {}
    amplitude_changes: dict[str, NDArray[np.float64]] = {}
    parametric_tests: dict[str, object] = {}

    for condition_a, condition_b, label in comparisons:
        if condition_a not in condition_means or condition_b not in condition_means:
            console.echo(
                message=f"Skipping comparison '{label}': one or both conditions have no trials.",
                level="WARNING",
            )
            continue

        mean_a = condition_means[condition_a]
        mean_b = condition_means[condition_b]

        pv_correlations[label] = population_vector_correlation(
            tuning_a=mean_a,
            tuning_b=mean_b,
        )

        spatial_correlations[label] = per_cell_spatial_correlation(
            tuning_a=mean_a,
            tuning_b=mean_b,
        )

        peak_bins_a = _get_peak_bins(mean_curves=mean_a)
        peak_bins_b = _get_peak_bins(mean_curves=mean_b)
        peak_shifts[label] = (peak_bins_b - peak_bins_a) * bin_size_cm

        with np.errstate(invalid="ignore"):
            amplitude_changes[label] = np.nanmax(mean_b, axis=0) / np.nanmax(mean_a, axis=0)

        # Within-condition leave-one-out reliability for each condition in the pair.
        if condition_a not in within_correlations:
            within_correlations[condition_a] = _leave_one_out_correlations(
                per_trial_curves=per_trial_masked[condition_a],
            )
        if condition_b not in within_correlations:
            within_correlations[condition_b] = _leave_one_out_correlations(
                per_trial_curves=per_trial_masked[condition_b],
            )

        # Cross-condition: each trial in A correlated against the mean of B, and vice versa.
        cross_correlations[f"{condition_a}_vs_mean_{condition_b}"] = _compute_trial_by_trial_correlations(
            per_trial_curves=per_trial_masked[condition_a],
            reference_curves=mean_b,
        )
        cross_correlations[f"{condition_b}_vs_mean_{condition_a}"] = _compute_trial_by_trial_correlations(
            per_trial_curves=per_trial_masked[condition_b],
            reference_curves=mean_a,
        )

        # Parametric t-test on per-cell spatial correlations against a correlation of 1
        # (testing whether the two conditions are represented similarly).
        valid_corrs = spatial_correlations[label][~np.isnan(spatial_correlations[label])]
        parametric_tests[label] = stats.ttest_1samp(valid_corrs, popmean=1.0)

    # Optional shuffle test: permute previous-trial-type labels within each current type.
    shuffle_p_values: dict[str, NDArray[np.float64]] = {}
    if run_shuffle:
        shuffle_p_values = _run_shuffle_test(
            df=df,
            condition_by_trial=condition_by_trial,
            n_shared_bins=n_shared_bins,
            signal_col=signal_col,
            cell_mask=cell_mask,
            comparisons=comparisons,
            observed_correlations=spatial_correlations,
            n_shuffles=n_shuffles,
        )

    # Session-averaged tuning curves per trial type (all trials, history-blind), restricted
    # to the shared segment and masked to included cells. Used as sort-order references.
    full_averages = get_mean_tuning_curves(
        df=df,
        config=config,
        metadata=metadata,
        signal_col=signal_col,
    )
    trial_type_averages: dict[str, NDArray[np.float64]] = {
        trial_type: curves[:n_shared_bins, cell_mask]
        for trial_type, curves in full_averages.items()
    }

    return {
        "condition_means": condition_means,
        "condition_variances": condition_variances,
        "condition_per_trial": per_trial_masked,
        "trial_type_averages": trial_type_averages,
        "pv_correlations": pv_correlations,
        "spatial_correlations": spatial_correlations,
        "within_correlations": within_correlations,
        "cross_correlations": cross_correlations,
        "peak_shifts": peak_shifts,
        "amplitude_changes": amplitude_changes,
        "parametric_tests": parametric_tests,
        "shuffle_p_values": shuffle_p_values,
        "comparisons": comparisons,
        "divergence_cm": divergence_cm,
        "n_shared_bins": n_shared_bins,
        "bin_size_cm": bin_size_cm,
        "trial_types": trial_types,
        "conditions": conditions,
        "n_cells": int(cell_mask.sum()),
        "cell_mask": cell_mask,
    }


def _run_shuffle_test(
    df: pl.DataFrame,
    condition_by_trial: dict[int, tuple[str, str]],
    n_shared_bins: int,
    signal_col: str,
    cell_mask: NDArray[np.bool_],
    comparisons: list[tuple[str, str, str]],
    observed_correlations: dict[str, NDArray[np.float64]],
    n_shuffles: int,
) -> dict[str, NDArray[np.float64]]:
    """Runs a permutation test by shuffling previous-trial-type labels.

    Within each current trial type, the previous-trial-type labels are permuted
    across trials on each shuffle, preserving the marginal distribution of current
    trial types. Per-cell spatial correlations are recomputed on each shuffle.

    Args:
        df: Speed-filtered frame-level DataFrame.
        condition_by_trial: Original trial-to-condition mapping.
        n_shared_bins: Number of bins in the shared segment.
        signal_col: Neural signal column name.
        cell_mask: Boolean array selecting cells to include.
        comparisons: List of (cond_a, cond_b, label) tuples.
        observed_correlations: Observed per-cell spatial correlations per comparison.
        n_shuffles: Number of permutations.

    Returns:
        Dictionary mapping comparison labels to per-cell p-values (fraction of
        shuffles with spatial correlation as extreme as or more extreme than observed).
    """
    rng = np.random.default_rng()
    n_cells_included = int(cell_mask.sum())

    # Group trial IDs by current trial type so shuffles are stratified.
    trials_by_current: dict[str, list[int]] = {}
    for trial_id, (current_type, _) in condition_by_trial.items():
        if current_type not in trials_by_current:
            trials_by_current[current_type] = []
        trials_by_current[current_type].append(trial_id)

    shuffle_counts: dict[str, NDArray[np.float64]] = {
        label: np.zeros(n_cells_included, dtype=np.float64)
        for _, _, label in comparisons
    }

    for _ in range(n_shuffles):
        shuffled_conditions: dict[int, tuple[str, str]] = {}
        for current_type, trial_list in trials_by_current.items():
            previous_types = [condition_by_trial[t][1] for t in trial_list]
            rng.shuffle(previous_types)
            for trial_id, previous_type in zip(trial_list, previous_types):
                shuffled_conditions[trial_id] = (current_type, previous_type)

        shuffled_per_trial = _compute_per_trial_tuning_curves(
            df=df,
            condition_by_trial=shuffled_conditions,
            n_shared_bins=n_shared_bins,
            signal_col=signal_col,
        )
        shuffled_masked = {
            condition: curves[:, :, cell_mask]
            for condition, curves in shuffled_per_trial.items()
        }
        shuffled_means = {
            condition: np.nanmean(curves, axis=0)
            for condition, curves in shuffled_masked.items()
        }

        for condition_a, condition_b, label in comparisons:
            if condition_a not in shuffled_means or condition_b not in shuffled_means:
                continue
            shuffled_corr = per_cell_spatial_correlation(
                tuning_a=shuffled_means[condition_a],
                tuning_b=shuffled_means[condition_b],
            )
            observed = observed_correlations.get(label, np.full(n_cells_included, np.nan))
            # Count shuffles where the shuffled correlation is <= observed (lower = more remapping).
            shuffle_counts[label] += shuffled_corr <= observed

    return {
        label: counts / n_shuffles
        for label, counts in shuffle_counts.items()
    }


# PLOTTING


def plot_condition_heatmaps(
    axes: list[Axes],
    condition_means: dict[str, NDArray[np.float64]],
    conditions_to_plot: list[str],
    bin_size_cm: float,
    sort_order: NDArray[np.int64] | None = None,
) -> None:
    """Plots session-averaged tuning curve heatmaps for a list of conditions.

    Args:
        axes: List of Axes, one per condition in conditions_to_plot.
        condition_means: Dictionary mapping condition labels to (n_shared_bins, n_cells)
            mean tuning curve arrays.
        conditions_to_plot: Ordered list of condition label strings to render.
        bin_size_cm: Spatial bin size in cm, used to label the x-axis in cm.
        sort_order: Cell sort order (indices). If None, sorts by peak position in the
            first condition that has data.
    """
    # Determine cell sort order from the first available condition if not provided.
    if sort_order is None:
        for condition in conditions_to_plot:
            if condition in condition_means:
                peak_bins = _get_peak_bins(mean_curves=condition_means[condition])
                sort_order = np.argsort(np.nan_to_num(peak_bins, nan=1e9))
                break

    for axis, condition in zip(axes, conditions_to_plot):
        if condition not in condition_means:
            axis.set_visible(False)
            continue

        mean_curves = condition_means[condition]  # (n_shared_bins, n_cells)
        n_shared_bins, n_cells = mean_curves.shape

        sorted_map = mean_curves[:, sort_order].T  # (n_cells, n_shared_bins)
        row_max = np.nanmax(sorted_map, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            normalized_map = sorted_map / np.where(row_max > 0, row_max, 1.0)

        x_extent = n_shared_bins * bin_size_cm
        axis.imshow(
            normalized_map,
            aspect="auto",
            origin="lower",
            extent=[0, x_extent, 0, n_cells],
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Session averages are keyed by bare trial type; history conditions by "type_prev".
        parts = condition.split("_", maxsplit=1)
        if len(parts) == 1:
            display_label = f"{parts[0]} (session avg)"
        else:
            display_label = f"{parts[0]} | prev: {parts[1]}"
        axis.set_title(display_label, fontsize=10, fontweight="bold")
        axis.set_xlabel("Position (cm)", fontsize=9)
        axis.set_ylabel("Cells", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def plot_pv_correlation_panels(
    axes: list[Axes],
    pv_correlations: dict[str, NDArray[np.float64]],
    comparisons: list[tuple[str, str, str]],
    bin_size_cm: float,
    divergence_cm: float,
    config: dict | None = None,
) -> None:
    """Plots PV correlation vs spatial position for each comparison.

    Args:
        axes: List of Axes, one per comparison in comparisons.
        pv_correlations: Dictionary mapping comparison labels to (n_shared_bins,) arrays.
        comparisons: List of (cond_a, cond_b, label) tuples.
        bin_size_cm: Spatial bin size in cm.
        divergence_cm: Track position (cm) where trial types diverge, drawn as a
            reference line.
        config: Experiment configuration dictionary. If provided, draws a cue bar
            below each panel using the trial type of condition_a.
    """
    for axis, (condition_a, _, label) in zip(axes, comparisons):
        if label not in pv_correlations:
            axis.set_visible(False)
            continue

        pv_corr = pv_correlations[label]
        positions_cm = np.arange(len(pv_corr)) * bin_size_cm + bin_size_cm / 2

        axis.plot(positions_cm, pv_corr, color="#2E86AB", linewidth=1.5)
        axis.axhline(y=0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        axis.axvline(
            x=divergence_cm,
            color="black",
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            label=f"divergence ({divergence_cm:.0f} cm)",
        )
        axis.set_ylim(-1.0, 1.0)
        axis.set_xlabel("Position (cm)", fontsize=9)
        axis.set_ylabel("PV correlation (r)", fontsize=9)
        axis.set_title(label, fontsize=9, fontweight="bold")
        axis.tick_params(labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if config is not None:
            trial_type = condition_a.split("_")[0]
            pfmt.add_cue_bar(ax=axis, config=config, trial_type=trial_type)


def plot_spatial_correlation_panels(
    axes: list[Axes],
    spatial_correlations: dict[str, NDArray[np.float64]],
    comparisons: list[tuple[str, str, str]],
) -> None:
    """Plots per-cell spatial correlation distributions for each comparison.

    Args:
        axes: List of Axes, one per comparison.
        spatial_correlations: Dictionary mapping comparison labels to (n_cells,) arrays.
        comparisons: List of (cond_a, cond_b, label) tuples.
    """
    for axis, (_, _, label) in zip(axes, comparisons):
        if label not in spatial_correlations:
            axis.set_visible(False)
            continue

        valid_corrs = spatial_correlations[label]
        valid_corrs = valid_corrs[~np.isnan(valid_corrs)]

        axis.hist(
            valid_corrs,
            bins=np.linspace(-1.0, 1.0, 41),
            color="#E84855",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        median_value = float(np.median(valid_corrs)) if len(valid_corrs) > 0 else np.nan
        axis.axvline(
            x=median_value,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"median = {median_value:.3f}",
        )
        axis.axvline(x=0.0, color="gray", linewidth=0.8, alpha=0.5)
        axis.set_xlabel("Spatial correlation (r)", fontsize=9)
        axis.set_ylabel("Cells", fontsize=9)
        axis.set_title(label, fontsize=9, fontweight="bold")
        axis.legend(frameon=False, fontsize=8)
        axis.text(
            0.02, 0.95,
            f"n = {len(valid_corrs)} cells",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def plot_variance_panels(
    axes: list[Axes],
    condition_variances: dict[str, NDArray[np.float64]],
    conditions_to_plot: list[str],
    bin_size_cm: float,
    config: dict | None = None,
) -> None:
    """Plots mean per-bin variance profiles for a list of conditions.

    Args:
        axes: List of Axes, one per condition.
        condition_variances: Dictionary mapping condition labels to (n_shared_bins, n_cells).
        conditions_to_plot: Conditions to render.
        bin_size_cm: Spatial bin size in cm.
        config: Experiment configuration dictionary. If provided, draws a cue bar
            below each panel using the trial type of that condition.
    """
    for axis, condition in zip(axes, conditions_to_plot):
        if condition not in condition_variances:
            axis.set_visible(False)
            continue

        variance_map = condition_variances[condition]  # (n_shared_bins, n_cells)
        mean_variance = np.nanmean(variance_map, axis=1)
        positions_cm = np.arange(len(mean_variance)) * bin_size_cm + bin_size_cm / 2

        parts = condition.split("_", maxsplit=1)
        display_label = f"{parts[0]} | prev: {parts[1]}" if len(parts) == 2 else condition

        axis.plot(positions_cm, mean_variance, linewidth=1.5, color="#6A0572")
        axis.fill_between(positions_cm, mean_variance, alpha=0.3, color="#6A0572")
        axis.set_xlabel("Position (cm)", fontsize=9)
        axis.set_ylabel("Mean variance", fontsize=9)
        axis.set_title(display_label, fontsize=9, fontweight="bold")
        axis.tick_params(labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        if config is not None:
            trial_type = parts[0]
            pfmt.add_cue_bar(ax=axis, config=config, trial_type=trial_type)


def plot_history_dependent_remapping(
    results: dict,
    config: dict,
    animal_id: str | None = None,
    date: str | None = None,
    save_path_maps: Path | None = None,
    save_path_metrics: Path | None = None,
    show: bool = True,
) -> tuple[Figure, Figure, Figure]:
    """Plots a two-figure summary of history-dependent remapping for one session.

    Figure 1 (maps) shows session-averaged tuning curve heatmaps. Row 0 has the ABC
    session average and its two history conditions; row 1 has the same for ABDC. Cells
    within each row are sorted by the history-blind session average for that trial type.

    Figure 2 (variance) shows mean per-bin variance profiles. Row 0 has the two ABC
    history conditions; row 1 has the two ABDC history conditions.

    Figure 3 (metrics) shows PV correlation vs spatial position in row 0 and per-cell
    spatial correlation distributions in row 1.

    Args:
        results: Output dictionary from compute_history_dependent_remapping().
        config: Experiment configuration dictionary (reserved for future cue-shading).
        animal_id: Animal identifier string for figure titles.
        date: Session date string for figure titles.
        save_path_maps: If provided, saves the maps figure to this path.
        save_path_metrics: If provided, saves the metrics figure to this path.
        show: Determines whether to call plt.show() after rendering each figure.

    Returns:
        Tuple of (maps_figure, variance_figure, metrics_figure).
    """
    conditions = results["conditions"]
    core_conditions = [c for c in conditions if _START_LABEL not in c]
    n_core = len(core_conditions)
    comparisons = results["comparisons"]
    n_comparisons = len(comparisons)
    trial_types = results["trial_types"]
    trial_type_averages = results["trial_type_averages"]
    condition_means = results["condition_means"]
    bin_size_cm = results["bin_size_cm"]

    title_base = pfmt.build_title(
        "History-Dependent Remapping",
        animal_id=animal_id,
        date=date,
    )

    # Per-trial-type sort orders from the history-blind session average.
    sort_orders: dict[str, NDArray[np.int64]] = {}
    for trial_type in trial_types:
        if trial_type in trial_type_averages:
            peak_bins = _get_peak_bins(mean_curves=trial_type_averages[trial_type])
            sort_orders[trial_type] = np.argsort(np.nan_to_num(peak_bins, nan=1e9))

    # Per-trial-type ordered condition lists: same-prior-and-current first.
    conditions_by_type: dict[str, list[str]] = {}
    for trial_type in trial_types:
        type_conditions = sorted(
            [c for c in core_conditions if c.startswith(f"{trial_type}_")],
            key=lambda c: (0 if c == f"{trial_type}_{trial_type}" else 1),
        )
        conditions_by_type[trial_type] = type_conditions

    # Merge session averages into a combined dict for heatmap plotting.
    all_means_with_avg = dict(condition_means)
    for trial_type, avg_curve in trial_type_averages.items():
        all_means_with_avg[trial_type] = avg_curve

    # Figure 1: heatmaps only. 2 rows × 3 cols.
    # Each row: [session avg | hist cond 1 | hist cond 2] for one trial type.
    fig_maps, axes_maps = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))

    fig_maps.suptitle(f"{title_base} — heatmaps", fontsize=13, fontweight="bold")

    for row_index, trial_type in enumerate(trial_types):
        sort_order = sort_orders.get(trial_type)
        panels = [trial_type] + conditions_by_type.get(trial_type, [])
        plot_condition_heatmaps(
            axes=list(axes_maps[row_index, :len(panels)]),
            condition_means=all_means_with_avg,
            conditions_to_plot=panels,
            bin_size_cm=bin_size_cm,
            sort_order=sort_order,
        )
        for extra_column in range(len(panels), 3):
            axes_maps[row_index, extra_column].set_visible(False)

    fig_maps.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path_maps is not None:
        fig_maps.savefig(save_path_maps, dpi=150, bbox_inches="tight")
        console.echo(message=f"Saved maps figure to {save_path_maps}.")

    if show:
        plt.show()

    # Figure 2: variance profiles. 2 rows × 2 cols.
    # Row 0: ABC history conditions. Row 1: ABDC history conditions.
    fig_variance, axes_variance = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))

    fig_variance.suptitle(f"{title_base} — Variance", fontsize=13, fontweight="bold")

    for row_index, trial_type in enumerate(trial_types):
        type_conditions = conditions_by_type.get(trial_type, [])
        plot_variance_panels(
            axes=list(axes_variance[row_index, :len(type_conditions)]),
            condition_variances=results["condition_variances"],
            conditions_to_plot=type_conditions,
            bin_size_cm=bin_size_cm,
            config=config,
        )
        for extra_column in range(len(type_conditions), 2):
            axes_variance[row_index, extra_column].set_visible(False)

    fig_variance.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()

    # Figure 3: PV correlation (row 0) and spatial correlation distributions (row 1).
    n_metric_cols = max(n_comparisons, 1)
    fig_metrics, axes_metrics = plt.subplots(
        nrows=2,
        ncols=n_metric_cols,
        figsize=(4.5 * n_metric_cols, 8),
    )
    if n_metric_cols == 1:
        axes_metrics = axes_metrics.reshape(2, 1)

    fig_metrics.suptitle(f"{title_base} — Metrics", fontsize=13, fontweight="bold")

    plot_pv_correlation_panels(
        axes=list(axes_metrics[0, :n_comparisons]),
        pv_correlations=results["pv_correlations"],
        comparisons=comparisons,
        bin_size_cm=bin_size_cm,
        divergence_cm=results["divergence_cm"],
        config=config,
    )
    for extra_column in range(n_comparisons, n_metric_cols):
        axes_metrics[0, extra_column].set_visible(False)

    plot_spatial_correlation_panels(
        axes=list(axes_metrics[1, :n_comparisons]),
        spatial_correlations=results["spatial_correlations"],
        comparisons=comparisons,
    )
    for extra_column in range(n_comparisons, n_metric_cols):
        axes_metrics[1, extra_column].set_visible(False)

    fig_metrics.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path_metrics is not None:
        fig_metrics.savefig(save_path_metrics, dpi=150, bbox_inches="tight")
        console.echo(message=f"Saved metrics figure to {save_path_metrics}.")

    if show:
        plt.show()

    return fig_maps, fig_variance, fig_metrics


# SESSION WRAPPERS


def analyze_session(
    df: pl.DataFrame,
    config: dict,
    metadata: dict,
    signal_col: str = "multi_day_spikes",
    place_cells_only: bool = False,
    comparison_type: str = "both",
    run_shuffle: bool = False,
    n_shuffles: int = 500,
    animal_id: str | None = None,
    date: str | None = None,
    output_directory: Path | None = None,
) -> tuple[dict, tuple[Figure, Figure]]:
    """Computes and plots history-dependent remapping for a single session.

    Convenience wrapper that calls compute_history_dependent_remapping() followed
    by plot_history_dependent_remapping(). If output_directory is provided, saves
    both figures using the standard naming convention with _maps and _metrics suffixes.

    Args:
        df: Frame-level DataFrame from process_session().
        config: Experiment configuration dictionary.
        metadata: Session metadata dictionary.
        signal_col: Neural signal column name for tuning curve computation.
        place_cells_only: Determines whether to restrict analysis to detected
            place cells.
        comparison_type: Which comparison pairs to compute. One of "prior",
            "current", or "both".
        run_shuffle: Determines whether to run the permutation test.
        n_shuffles: Number of shuffle iterations when run_shuffle is True.
        animal_id: Animal identifier used in the figure title and filename.
        date: Session date used in the figure title and filename.
        output_directory: If provided, saves all three figures to this directory.

    Returns:
        Tuple of (results_dict, (maps_figure, variance_figure, metrics_figure)).
        results_dict contains all computed metrics as returned by
        compute_history_dependent_remapping().
    """
    results = compute_history_dependent_remapping(
        df=df,
        config=config,
        metadata=metadata,
        signal_col=signal_col,
        place_cells_only=place_cells_only,
        comparison_type=comparison_type,
        run_shuffle=run_shuffle,
        n_shuffles=n_shuffles,
    )

    save_path_maps: Path | None = None
    save_path_metrics: Path | None = None
    if output_directory is not None and animal_id is not None and date is not None:
        prefix = Path(output_directory) / f"{animal_id}_{date}_history_dependent_remapping"
        save_path_maps = Path(f"{prefix}_maps.pdf")
        save_path_metrics = Path(f"{prefix}_metrics.pdf")

    figures = plot_history_dependent_remapping(
        results=results,
        config=config,
        animal_id=animal_id,
        date=date,
        save_path_maps=save_path_maps,
        save_path_metrics=save_path_metrics,
        show=True,
    )

    return results, figures


def plot_multiday_metrics(
    session_results: dict[str, dict],
    metric_key: str = "spatial_correlations",
    reduction: str = "median",
    save_path: Path | None = None,
    show: bool = True,
) -> Figure:
    """Plots a time series of a remapping metric across sessions.

    Displays individual session values as scatter points and the cross-session
    mean ± SEM as an overlaid line with shaded error band. One line per comparison.

    Args:
        session_results: Dictionary mapping date strings to results dicts as
            returned by compute_history_dependent_remapping(). Keys must be
            sortable date strings (e.g. "2025-09-15").
        metric_key: Which metric to plot. "spatial_correlations" plots the median
            per-cell spatial correlation per comparison. "pv_correlations" plots
            the mean PV correlation across shared bins. Only these two keys are
            currently supported.
        reduction: Statistic used to reduce per-cell or per-bin values to a scalar
            for each session. "median" or "mean".
        save_path: If provided, saves the figure to this path.
        show: Determines whether to call plt.show() after rendering.

    Returns:
        The rendered Matplotlib Figure.
    """
    if not session_results:
        console.error(
            message="Unable to plot multiday metrics. session_results is empty.",
            error=ValueError,
        )

    sorted_dates = sorted(session_results.keys())
    date_indices = np.arange(len(sorted_dates))

    # Collect comparison labels from the first session that has them.
    comparison_labels: list[str] = []
    for date in sorted_dates:
        metric = session_results[date].get(metric_key, {})
        if metric:
            comparison_labels = list(metric.keys())
            break

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(comparison_labels), 1)))

    fig, axis = plt.subplots(figsize=(max(8, len(sorted_dates) * 0.8), 5))

    reduce_fn = np.nanmedian if reduction == "median" else np.nanmean

    for color, label in zip(colors, comparison_labels):
        session_scalars: list[float] = []
        for date in sorted_dates:
            metric = session_results[date].get(metric_key, {})
            if label not in metric:
                session_scalars.append(np.nan)
                continue
            values = metric[label]
            session_scalars.append(float(reduce_fn(values)))

        scalar_array = np.array(session_scalars, dtype=np.float64)
        valid = ~np.isnan(scalar_array)

        # Scatter individual session values.
        axis.scatter(
            date_indices[valid],
            scalar_array[valid],
            color=color,
            alpha=0.5,
            s=30,
            zorder=3,
        )
        # Connect with a line.
        axis.plot(
            date_indices[valid],
            scalar_array[valid],
            color=color,
            linewidth=1.5,
            label=label,
            zorder=2,
        )

    axis.set_xticks(date_indices)
    axis.set_xticklabels(sorted_dates, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(f"{reduction.capitalize()} {metric_key.replace('_', ' ')}", fontsize=10)
    axis.set_xlabel("Session date", fontsize=10)
    axis.set_title("History-Dependent Remapping Across Sessions", fontsize=12, fontweight="bold")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        console.echo(message=f"Saved figure to {save_path}.")

    if show:
        plt.show()

    return fig


def analyze_multiday(
    sessions: dict[str, dict],
    signal_col: str = "multi_day_spikes",
    place_cells_only: bool = False,
    comparison_type: str = "both",
    run_shuffle: bool = False,
    n_shuffles: int = 500,
    output_directory: Path | None = None,
) -> tuple[dict[str, dict], Figure | None]:
    """Runs history-dependent remapping analysis on each session and plots a time series.

    Iterates over all sessions in chronological order, calling analyze_session() on
    each. After all sessions are processed, calls plot_multiday_metrics() to produce
    a cross-session summary figure showing spatial and PV correlation metrics over time.

    Args:
        sessions: Dictionary as returned by load_multiday_sessions(), mapping date
            strings to session dictionaries with 'data', 'config', 'session_data',
            and 'metadata' keys.
        signal_col: Neural signal column name for tuning curve computation.
        place_cells_only: Determines whether to restrict analysis to detected place
            cells in each session.
        comparison_type: Which comparison pairs to compute. One of "prior",
            "current", or "both".
        run_shuffle: Determines whether to run the permutation test per session.
        n_shuffles: Number of shuffle iterations per session when run_shuffle is True.
        output_directory: If provided, saves each per-session figure and the
            multi-session summary figure to this directory.

    Returns:
        Tuple of (all_results, summary_figure). all_results maps date strings to the
        results dict from compute_history_dependent_remapping() for each session.
        summary_figure is the rendered multi-session time series Figure.
    """
    all_results: dict[str, dict] = {}

    for date in sorted(sessions.keys()):
        session = sessions[date]
        data_frame = session["data"]
        config = session["config"]
        metadata = session["metadata"]
        animal_id = session.get("session_data", {}).get("animal_id")

        console.echo(message=f"Processing session {date}...")

        session_save_dir: Path | None = None
        if output_directory is not None:
            session_save_dir = Path(output_directory)

        try:
            session_results, _ = analyze_session(
                df=data_frame,
                config=config,
                metadata=metadata,
                signal_col=signal_col,
                place_cells_only=place_cells_only,
                comparison_type=comparison_type,
                run_shuffle=run_shuffle,
                n_shuffles=n_shuffles,
                animal_id=animal_id,
                date=date,
                output_directory=session_save_dir,
            )
            all_results[date] = session_results
            console.echo(message=f"  Completed {date}.", level="SUCCESS")

        except Exception as error:  # noqa: BLE001
            console.echo(
                message=f"  WARNING: Failed to process {date}: {error}",
                level="WARNING",
            )

    if not all_results:
        console.echo(
            message="No sessions processed successfully. Returning empty results.",
            level="WARNING",
        )
        return all_results, None

    multiday_save_path: Path | None = None
    if output_directory is not None:
        first_date = sorted(all_results.keys())[0]
        first_session = sessions.get(first_date, {})
        animal_id = first_session.get("session_data", {}).get("animal_id", "unknown")
        multiday_save_path = (
            Path(output_directory) / f"{animal_id}_history_dependent_remapping_multiday.pdf"
        )

    summary_figure = plot_multiday_metrics(
        session_results=all_results,
        metric_key="spatial_correlations",
        reduction="median",
        save_path=multiday_save_path,
        show=True,
    )

    return all_results, summary_figure



if __name__ == '__main__':
    from df_processing import (
        find_session_dir, load_session_context, get_session_paths,
        load_processed_session,
    )
    from place_field_detection import detect_place_fields, DetectionParams

    mouse_id = '26'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)
    date = '2025-09-10'

    # Load session
    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])

    bin_size_cm = meta['bin_size_cm']
    signal_col = 'multi_day_dff'  # was 'multi_day_spikes'

    analyze_session(data, exp_config, metadata=meta, signal_col=signal_col)
