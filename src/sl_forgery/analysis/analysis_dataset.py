"""Generates a unified per-cell analysis DataFrame from all cell analysis pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console

from sl_forgery.analysis.sce import SCEDetector, SCEDetectionConfiguration
from sl_forgery.analysis.utilities import compute_track_length, compute_reward_position
from sl_forgery.analysis.place_cell_analysis import PlaceFieldDetector, PlaceFieldDetectionConfiguration
from sl_forgery.analysis.reward_cell_analysis import RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sl_forgery.analysis.sce import SCEResult
    from sl_forgery.analysis.place_cell_analysis import PlaceFields
    from sl_forgery.analysis.reward_cell_analysis import RewardCellResults


def _aggregate_place_field_columns(cell_count: int, place_fields: PlaceFields) -> dict:
    """Extracts per-cell place field metrics as column arrays.

    Args:
        cell_count: Total number of cells.
        place_fields: Detected place fields from the place field detection pipeline.

    Returns:
        A dictionary mapping column names to numpy arrays or lists of per-cell values.
    """
    cell_ids = place_fields.cell_id
    centers = place_fields.centers
    max_intensities = place_fields.max_intensity
    bin_size = place_fields.bin_size

    field_count = np.bincount(cell_ids, minlength=cell_count).astype(np.int32)
    field_center = np.full(cell_count, np.nan, dtype=np.float32)
    field_peak = np.full(cell_count, np.nan, dtype=np.float32)
    all_centers: list[list[float]] = [[] for _ in range(cell_count)]

    # Collects all field centers per cell and selects the strongest (highest max intensity).
    for cell in np.where(field_count > 0)[0]:
        cell_mask = cell_ids == cell
        cell_field_indices = np.where(cell_mask)[0]
        for field_index in cell_field_indices:
            all_centers[cell].append(float(centers[field_index, 1] * bin_size))
        strongest = cell_field_indices[np.argmax(max_intensities[cell_mask])]
        field_center[cell] = centers[strongest, 1] * bin_size
        field_peak[cell] = max_intensities[strongest]

    return {
        "has_place_field": place_fields.has_place_field,
        "place_field_count": field_count,
        "place_field_center_cm": field_center,
        "place_field_peak_intensity": field_peak,
        "place_field_centers_cm": all_centers,
    }


def _aggregate_reward_cell_columns(reward_results: RewardCellResults) -> dict[str, NDArray]:
    """Extracts per-cell reward cell metrics as flat column arrays.

    Args:
        reward_results: Results from the reward cell detection pipeline.

    Returns:
        A dictionary mapping column names to numpy arrays of per-cell values.
    """
    spatial = reward_results.spatial_results
    is_significant = spatial.is_significant
    is_proximal = reward_results.is_reward_proximal
    is_slowing = reward_results.is_slowing_correlated

    return {
        "spatial_information": spatial.spatial_information,
        "spatial_p_value": spatial.p_values,
        "is_spatially_significant": is_significant,
        "center_of_mass_cm": spatial.centers_of_mass,
        "is_reward_proximal": is_proximal,
        "speed_activity_correlation": reward_results.speed_activity_correlations,
        "is_slowing_correlated": is_slowing,
        "is_reward_cell": is_significant & is_proximal,
        "is_reward_predictive": is_significant & is_proximal & is_slowing,
    }


def _aggregate_sce_columns(cell_count: int, sce_results: list[SCEResult]) -> dict:
    """Computes per-cell SCE participation and timing metrics across all detected periods.

    Args:
        cell_count: Total number of cells.
        sce_results: List of SCE detection results across all rest and run periods.

    Returns:
        A dictionary mapping column names to numpy arrays or lists of per-cell values.
    """
    # Uses PeriodType as row index (REST=0, RUN=1) and cell as column index.
    participation = np.zeros((2, cell_count), dtype=np.int32)
    rank_sum = np.zeros((2, cell_count), dtype=np.float32)
    rank_count = np.zeros((2, cell_count), dtype=np.int32)
    total_sces = np.zeros(2, dtype=np.int32)
    period_counter = np.zeros(2, dtype=np.int32)
    sce_events: list[list[list[tuple[int, int]]]] = [[[] for _ in range(cell_count)] for _ in range(2)]

    for result in sce_results:
        period = int(result.period_type)
        period_index = int(period_counter[period])
        period_counter[period] += 1
        sce_count = int(np.max(result.sce_labels))
        total_sces[period] += sce_count

        for sce_label in range(1, sce_count + 1):
            sce_frames = np.where(result.sce_labels == sce_label)[0]
            onset_window = result.onset_matrix[:, sce_frames]
            participating_indices = np.where(np.any(onset_window, axis=1))[0]
            participant_count = len(participating_indices)
            participation[period, participating_indices] += 1

            # Records the (period_index, sce_label) tuple for each participating cell.
            for cell in participating_indices:
                sce_events[period][cell].append((period_index, sce_label))

            if participant_count > 1:
                # Finds the first onset frame for each participating cell using argmax on the boolean rows.
                first_onset = np.argmax(onset_window[participating_indices], axis=1)
                normalized_ranks = np.argsort(np.argsort(first_onset)).astype(np.float32) / (participant_count - 1)
                rank_sum[period, participating_indices] += normalized_ranks
                rank_count[period, participating_indices] += 1

            elif participant_count == 1:
                rank_sum[period, participating_indices] += 0.5
                rank_count[period, participating_indices] += 1

    # Computes participation rates and mean onset ranks per period type.
    rate = np.full((2, cell_count), np.nan, dtype=np.float32)
    mean_rank = np.full((2, cell_count), np.nan, dtype=np.float32)
    for period in range(2):
        if total_sces[period] > 0:
            rate[period] = (participation[period] / total_sces[period]).astype(np.float32)
        has_ranks = rank_count[period] > 0
        mean_rank[period, has_ranks] = rank_sum[period, has_ranks] / rank_count[period, has_ranks]

    return {
        "sce_participation_count_rest": participation[0],
        "sce_participation_count_run": participation[1],
        "sce_participation_rate_rest": rate[0],
        "sce_participation_rate_run": rate[1],
        "sce_mean_onset_rank_rest": mean_rank[0],
        "sce_mean_onset_rank_run": mean_rank[1],
        "sce_events_rest": sce_events[0],
        "sce_events_run": sce_events[1],
    }


def generate_analysis_dataframe(
    session_path: Path,
    track_length: float | None = None,
    output_directory: Path | None = None,
    fluorescence_column: str = "single_day_dff",
    trial_type: str = "ABC",
    place_configuration: PlaceFieldDetectionConfiguration | None = None,
    reward_configuration: RewardCellConfiguration | None = None,
    sce_configuration: SCEDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs all three analysis pipelines and assembles a unified per-cell DataFrame.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters. Computed automatically from the session file if None.
        output_directory: Directory to save the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: Name of the fluorescence column to read from the feather file.
        trial_type: Trial type to analyze.
        place_configuration: Place field detection parameters. Uses defaults if None.
        reward_configuration: Reward cell detection parameters. Uses defaults if None.
        sce_configuration: SCE detection parameters. Uses defaults if None.

    Returns:
        A polars DataFrame with one row per cell and columns for place field, reward cell, and SCE metrics.
    """
    if track_length is None:
        track_length = compute_track_length(session_path=session_path, trial_type=trial_type)
        console.echo(message=f"Computed track length: {track_length} cm.", level=LogLevel.INFO)

    # Runs the place field detection.
    console.echo(message="Running place field detection...", level=LogLevel.INFO)
    place_detector = PlaceFieldDetector(
        session_path=session_path,
        track_length=track_length,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        configuration=place_configuration,
    )
    place_fields = place_detector.detect(run_shuffle=False)
    cell_count = place_fields.binned_fluorescence.shape[0]
    place_cell_count = int(place_fields.has_place_field.sum())
    console.echo(
        message=f"Place field detection complete: {place_cell_count}/{cell_count} place cells.",
        level=LogLevel.SUCCESS,
    )

    # Runs the reward cell detection.
    console.echo(message="Running reward cell detection...", level=LogLevel.INFO)
    reward_position = compute_reward_position(
        session_path=session_path,
        track_length=track_length,
        trial_type=trial_type,
    )
    reward_detector = RewardCellDetector(
        session_path=session_path,
        track_length=track_length,
        reward_position=reward_position,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        configuration=reward_configuration,
    )
    reward_results = reward_detector.detect()
    console.echo(
        message=(
            f"Reward cell detection complete: {reward_results.reward_cell_count} reward cells, "
            f"{len(reward_results.reward_predictive_indices)} reward-predictive."
        ),
        level=LogLevel.SUCCESS,
    )

    # Runs the SCE detection, passing pre-computed place fields for run-period masking.
    console.echo(message="Running SCE detection...", level=LogLevel.INFO)
    sce_detector = SCEDetector(
        session_path=session_path,
        track_length=track_length,
        fluorescence_column=fluorescence_column,
        place_fields=place_fields,
        configuration=sce_configuration,
    )

    sce_results = sce_detector.detect()
    rest_count = len(sce_detector.rest_results)
    run_count = len(sce_detector.run_results)
    total_rest_sces = sum(int(np.max(r.sce_labels)) for r in sce_detector.rest_results)
    total_run_sces = sum(int(np.max(r.sce_labels)) for r in sce_detector.run_results)
    console.echo(
        message=(
            f"SCE detection complete: {rest_count} rest periods ({total_rest_sces} SCEs), "
            f"{run_count} run periods ({total_run_sces} SCEs)."
        ),
        level=LogLevel.SUCCESS,
    )

    # Assembles all per-cell metrics into a single dataframe.
    console.echo(message="Assembling cell summary dataFrame...", level=LogLevel.INFO)
    columns: dict = {"cell_id": np.arange(cell_count, dtype=np.int32)}
    columns.update(_aggregate_place_field_columns(cell_count=cell_count, place_fields=place_fields))
    columns.update(_aggregate_reward_cell_columns(reward_results=reward_results))
    columns.update(_aggregate_sce_columns(cell_count=cell_count, sce_results=sce_results))

    dataframe = pl.DataFrame(columns)
    console.echo(
        message=f"Cell analysis dataframe assembled: {len(dataframe)} cells, {len(dataframe.columns)} columns.",
        level=LogLevel.SUCCESS,
    )

    # Saves the dataframe as an uncompressed feather file.
    save_directory = output_directory if output_directory is not None else session_path.parent
    output_path = save_directory / f"{session_path.stem}_analysis.feather"
    dataframe.write_ipc(file=output_path)
    
    console.echo(message=f"Cell analysis dataframe saved to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe
