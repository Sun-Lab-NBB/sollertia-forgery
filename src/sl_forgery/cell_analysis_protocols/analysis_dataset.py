"""Generates a unified per-place-field analysis DataFrame from all cell analysis pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
from pathlib import Path

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console

from .utilities import compute_track_length, compute_reward_position
from .sce_analysis import SCEDetector, SCEDetectionConfiguration
from .place_cell_analysis import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_cell_analysis import RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .sce_analysis import SCEResult
    from .reward_cell_analysis import RewardCellResults


class _PlaceFieldRows(NamedTuple):
    """Stores pre-allocated per-place-field row arrays and the row-level cell IDs for column expansion."""

    row_cell_ids: NDArray[np.int32]
    """Maps each row to its parent cell index."""
    pf_start_cm: NDArray[np.float32]
    """Starting position of each place field in centimeters."""
    pf_end_cm: NDArray[np.float32]
    """Ending position of each place field in centimeters."""
    pf_center_cm: NDArray[np.float32]
    """Intensity-weighted centroid of each place field in centimeters."""
    pf_mean_intensity: NDArray[np.float32]
    """Mean fluorescence intensity within each place field region."""
    pf_max_intensity: NDArray[np.float32]
    """Peak fluorescence intensity within each place field region."""
    pf_width_cm: NDArray[np.float32]
    """Spatial width of each place field in centimeters."""
    binned_fluorescence: list[list[float]]
    """Full spatial tuning curve for each row's parent cell."""


def _build_place_field_rows(cell_count: int, place_fields: PlaceFields) -> _PlaceFieldRows:
    """Builds per-place-field row arrays from detected place fields.

    Each detected place field produces one row. Cells without any place field produce a single row with NaN for all
    field-specific columns. The binned fluorescence for a cell is duplicated across all rows belonging to that cell.

    Args:
        cell_count: Total number of cells.
        place_fields: Detected place fields from the place field detection pipeline.

    Returns:
        A named tuple containing pre-allocated arrays for all place field columns and the row-level cell IDs needed
        to expand per-cell columns to per-row.
    """
    label_image = place_fields.label_image
    bin_size = place_fields.bin_size
    region_count = int(np.max(label_image)) if label_image.size > 0 else 0

    cell_ids = place_fields.cell_id
    centers = place_fields.centers
    mean_intensities = place_fields.mean_intensity
    max_intensities = place_fields.max_intensity
    fluorescence = place_fields.binned_fluorescence

    # Identifies cells with and without place fields using numpy set operations.
    unique_field_cells = np.unique(cell_ids[:region_count])
    cells_without_fields = np.setdiff1d(np.arange(cell_count, dtype=np.int32), unique_field_cells)
    total_rows = region_count + len(cells_without_fields)

    # Pre-allocates all column arrays. Field-specific columns default to NaN so that cells without place fields
    # automatically receive NaN values without additional assignment.
    row_cell_ids = np.empty(total_rows, dtype=np.int32)
    pf_start_cm = np.full(total_rows, np.nan, dtype=np.float32)
    pf_end_cm = np.full(total_rows, np.nan, dtype=np.float32)
    pf_center_cm = np.full(total_rows, np.nan, dtype=np.float32)
    pf_mean_intensity = np.full(total_rows, np.nan, dtype=np.float32)
    pf_max_intensity = np.full(total_rows, np.nan, dtype=np.float32)
    pf_width_cm = np.full(total_rows, np.nan, dtype=np.float32)

    # Collects the fluorescence row indices to batch-convert at the end instead of calling .tolist() per row.
    fluorescence_row_indices = np.empty(total_rows, dtype=np.int32)

    # Extracts labeled region boundaries for each place field, handling wrapped circular fields via gap detection.
    for field_index in range(region_count):
        label = field_index + 1
        cell_index = int(cell_ids[field_index])
        bins = np.where(label_image[cell_index, :] == label)[0]

        # Identifies wrapped fields by a gap in the sorted bin indices.
        gap_indices = np.where(np.diff(bins) > 1)[0]
        if len(gap_indices) > 0:
            gap = gap_indices[0]
            start_bin = bins[gap + 1]
            end_bin = bins[gap]
        else:
            start_bin = bins[0]
            end_bin = bins[-1]

        row_cell_ids[field_index] = cell_index
        pf_start_cm[field_index] = start_bin * bin_size
        pf_end_cm[field_index] = end_bin * bin_size
        pf_center_cm[field_index] = centers[field_index, 1]
        pf_mean_intensity[field_index] = mean_intensities[field_index]
        pf_max_intensity[field_index] = max_intensities[field_index]
        pf_width_cm[field_index] = len(bins) * bin_size
        fluorescence_row_indices[field_index] = cell_index

    # Assigns NaN rows for cells without place fields using the pre-computed array.
    no_field_start = region_count
    no_field_end = no_field_start + len(cells_without_fields)
    row_cell_ids[no_field_start:no_field_end] = cells_without_fields
    fluorescence_row_indices[no_field_start:no_field_end] = cells_without_fields

    # Batch-converts fluorescence rows to lists using fancy indexing instead of per-row .tolist() calls.
    row_fluorescence = fluorescence[fluorescence_row_indices].tolist()

    return _PlaceFieldRows(
        row_cell_ids=row_cell_ids,
        pf_start_cm=pf_start_cm,
        pf_end_cm=pf_end_cm,
        pf_center_cm=pf_center_cm,
        pf_mean_intensity=pf_mean_intensity,
        pf_max_intensity=pf_max_intensity,
        pf_width_cm=pf_width_cm,
        binned_fluorescence=row_fluorescence,
    )


def _expand_cell_array_to_rows(
    cell_array: NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_],
    row_cell_ids: NDArray[np.int32],
) -> NDArray[np.float32] | NDArray[np.int32] | NDArray[np.bool_]:
    """Expands a per-cell array to per-row by indexing with row-level cell IDs.

    Args:
        cell_array: Per-cell array with one value per cell.
        row_cell_ids: Array mapping each row to its parent cell index.

    Returns:
        An array with the same dtype as the input, expanded to match the row count.
    """
    return cell_array[row_cell_ids]


def _expand_cell_list_to_rows(
    cell_list: list[list[tuple[int, int]]],
    row_cell_ids: NDArray[np.int32],
) -> list[list[tuple[int, int]]]:
    """Expands a per-cell list to per-row by indexing with row-level cell IDs.

    Args:
        cell_list: Per-cell list with one entry per cell.
        row_cell_ids: Array mapping each row to its parent cell index.

    Returns:
        A list expanded to match the row count.
    """
    return [cell_list[cell_id] for cell_id in row_cell_ids]


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

        if sce_count == 0:
            continue

        # Builds a (frame_count, sce_count) binary matrix mapping frames to their SCE label, then computes a
        # (cell_count, sce_count) participation matrix via matrix multiplication with the onset matrix.
        sce_frame_indices = np.where(result.sce_labels > 0)[0]
        frame_to_sce = np.zeros((result.onset_matrix.shape[1], sce_count), dtype=np.float32)
        frame_to_sce[sce_frame_indices, result.sce_labels[sce_frame_indices] - 1] = 1.0
        cell_sce_participation = (result.onset_matrix.astype(np.float32) @ frame_to_sce) > 0

        # Accumulates participation counts per cell across all SCEs in this period.
        participation[period] += cell_sce_participation.sum(axis=1).astype(np.int32)

        # Records per-cell (period_index, sce_label) tuples and computes onset ranks for each SCE.
        for sce_label in range(1, sce_count + 1):
            participating_indices = np.where(cell_sce_participation[:, sce_label - 1])[0]
            participant_count = len(participating_indices)

            for cell in participating_indices:
                sce_events[period][cell].append((period_index, sce_label))

            if participant_count > 1:
                # Computes normalized onset ranks from the first onset frame within this SCE.
                sce_frames = np.where(result.sce_labels == sce_label)[0]
                onset_window = result.onset_matrix[participating_indices][:, sce_frames]
                first_onset = np.argmax(onset_window, axis=1)
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


def _reconstruct_place_fields(analysis_path: Path, track_length: float) -> PlaceFields:
    """Reconstructs a PlaceFields object from a previously saved analysis feather file.

    Rebuilds the label_image, binned_fluorescence, centers, and bin_size from the per-place-field rows stored in the
    feather file. The reconstructed object is suitable for SCE run-period masking.

    Args:
        analysis_path: Path to the analysis feather file containing place field columns.
        track_length: Length of the track in centimeters, used to derive bin_size from the binned fluorescence width.

    Returns:
        A PlaceFields object with label_image, binned_fluorescence, centers, and bin_size reconstructed from the
        feather data.
    """
    dataframe = pl.read_ipc(source=analysis_path, memory_map=True)

    cell_ids = dataframe["cell_id"].to_numpy()
    cell_count = int(cell_ids.max()) + 1

    # Extracts per-cell binned fluorescence by taking the first occurrence of each cell (all rows for the same cell
    # share identical binned fluorescence). Batch-converts the list of lists to a single numpy array.
    unique_cells = dataframe.group_by("cell_id", maintain_order=True).first()
    unique_cell_ids = unique_cells["cell_id"].to_numpy()
    all_fluorescence = np.array(unique_cells["binned_fluorescence"].to_list(), dtype=np.float32)
    bin_count = all_fluorescence.shape[1]
    bin_size = track_length / bin_count
    binned_fluorescence = np.zeros((cell_count, bin_count), dtype=np.float32)
    binned_fluorescence[unique_cell_ids] = all_fluorescence

    # Reconstructs label_image and centers from the non-NaN place field rows.
    label_image = np.zeros((cell_count, bin_count), dtype=np.int32)
    field_rows = dataframe.filter(pl.col("pf_start_cm").is_not_null())
    field_count = len(field_rows)

    centers_list: list[list[float]] = []
    for field_index in range(field_count):
        cell_index = int(field_rows["cell_id"][field_index])
        start_bin = int(field_rows["pf_start_cm"][field_index] / bin_size)
        end_bin = int(field_rows["pf_end_cm"][field_index] / bin_size)
        label = field_index + 1

        # Handles both contiguous and wrapped place fields.
        if start_bin <= end_bin:
            label_image[cell_index, start_bin : end_bin + 1] = label
        else:
            # Wrapped field: bins from start_bin to end of track, then from 0 to end_bin.
            label_image[cell_index, start_bin:] = label
            label_image[cell_index, : end_bin + 1] = label

        centers_list.append([float(cell_index), float(field_rows["pf_center_cm"][field_index])])

    centers = np.array(centers_list, dtype=np.float32) if centers_list else np.array([], dtype=np.float32).reshape(0, 2)

    return PlaceFields(
        label_image=label_image,
        binned_fluorescence=binned_fluorescence,
        centers=centers,
        bin_size=bin_size,
    )


def _resolve_output_path(session_path: Path, output_directory: Path | None) -> Path:
    """Resolves the output feather file path for a given session.

    Args:
        session_path: Path to the session feather file.
        output_directory: Directory to save the analysis feather file. Defaults to the session file's directory.

    Returns:
        The resolved output path for the analysis feather file.
    """
    save_directory = output_directory if output_directory is not None else session_path.parent
    return save_directory / f"{session_path.stem}_analysis.feather"


def generate_place_field_dataframe(
    session_path: Path,
    track_length: float | None = None,
    output_directory: Path | None = None,
    fluorescence_column: str = "single_day_dff",
    trial_type: str = "ABC",
    place_configuration: PlaceFieldDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the place field detection pipeline and writes a per-place-field analysis feather file.

    Each row in the output represents a single place field. Cells with multiple place fields produce multiple rows,
    and cells without any place field produce a single row with NaN for all field-specific columns.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters. Computed automatically from the session file if None.
        output_directory: Directory to save the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: Name of the fluorescence column to read from the feather file.
        trial_type: Trial type to analyze.
        place_configuration: Place field detection parameters. Uses defaults if None.

    Returns:
        A polars DataFrame with one row per place field and columns for place field boundaries, intensities, and
        binned fluorescence.
    """
    if track_length is None:
        track_length = compute_track_length(session_path=session_path, trial_type=trial_type)
        console.echo(message=f"Computed track length: {track_length} cm.", level=LogLevel.INFO)

    # Runs the place field detection pipeline.
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

    # Builds per-place-field rows from the detection results.
    console.echo(message="Assembling per-place-field analysis DataFrame...", level=LogLevel.INFO)
    place_field_rows = _build_place_field_rows(cell_count=cell_count, place_fields=place_fields)

    dataframe = pl.DataFrame(
        {
            "cell_id": place_field_rows.row_cell_ids,
            "pf_start_cm": place_field_rows.pf_start_cm,
            "pf_end_cm": place_field_rows.pf_end_cm,
            "pf_center_cm": place_field_rows.pf_center_cm,
            "binned_fluorescence": pl.Series(
                name="binned_fluorescence",
                values=place_field_rows.binned_fluorescence,
                dtype=pl.List(pl.Float32),
            ),
            "pf_mean_intensity": place_field_rows.pf_mean_intensity,
            "pf_max_intensity": place_field_rows.pf_max_intensity,
            "pf_width_cm": place_field_rows.pf_width_cm,
        },
    ).sort("cell_id")

    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    place_field_row_count = dataframe.filter(pl.col("pf_start_cm").is_not_null()).shape[0]
    console.echo(
        message=(
            f"Place field DataFrame assembled: {row_count} rows, {column_count} columns. "
            f"{cell_count} total cells, {place_cell_count} with place fields, "
            f"{place_field_row_count} place field entries."
        ),
        level=LogLevel.SUCCESS,
    )

    # Saves the DataFrame as an uncompressed feather file.
    output_path = _resolve_output_path(session_path=session_path, output_directory=output_directory)
    dataframe.write_ipc(file=output_path)
    console.echo(message=f"Place field DataFrame saved to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe


def append_reward_cell_columns(
    session_path: Path,
    track_length: float | None = None,
    output_directory: Path | None = None,
    fluorescence_column: str = "single_day_dff",
    trial_type: str = "ABC",
    reward_configuration: RewardCellConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the reward cell detection pipeline and appends reward cell columns to an existing analysis feather file.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters. Computed automatically from the session file if None.
        output_directory: Directory containing the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: Name of the fluorescence column to read from the feather file.
        trial_type: Trial type to analyze.
        reward_configuration: Reward cell detection parameters. Uses defaults if None.

    Returns:
        The updated polars DataFrame with reward cell columns appended.
    """
    output_path = _resolve_output_path(session_path=session_path, output_directory=output_directory)
    dataframe = pl.read_ipc(source=output_path, memory_map=True)
    row_cell_ids = dataframe["cell_id"].to_numpy()

    if track_length is None:
        track_length = compute_track_length(session_path=session_path, trial_type=trial_type)
        console.echo(message=f"Computed track length: {track_length} cm.", level=LogLevel.INFO)

    # Runs the reward cell detection pipeline.
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

    # Expands per-cell reward columns to match the per-place-field row structure and appends to the DataFrame.
    reward_columns = _aggregate_reward_cell_columns(reward_results=reward_results)
    reward_series = [
        pl.Series(name=name, values=_expand_cell_array_to_rows(cell_array=array, row_cell_ids=row_cell_ids))
        for name, array in reward_columns.items()
    ]
    dataframe = dataframe.hstack(reward_series)

    # Writes the updated DataFrame back to the same feather file.
    dataframe.write_ipc(file=output_path)
    console.echo(message=f"Reward cell columns appended to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe


def append_sce_columns(
    session_path: Path,
    track_length: float | None = None,
    output_directory: Path | None = None,
    fluorescence_column: str = "single_day_dff",
    trial_type: str = "ABC",
    sce_configuration: SCEDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the SCE detection pipeline and appends SCE columns to an existing analysis feather file.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters. Computed automatically from the session file if None.
        output_directory: Directory containing the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: Name of the fluorescence column to read from the feather file.
        trial_type: Trial type to analyze.
        sce_configuration: SCE detection parameters. Uses defaults if None.

    Returns:
        The updated polars DataFrame with SCE columns appended.
    """
    output_path = _resolve_output_path(session_path=session_path, output_directory=output_directory)
    dataframe = pl.read_ipc(source=output_path, memory_map=True)
    row_cell_ids = dataframe["cell_id"].to_numpy()
    cell_count = int(row_cell_ids.max()) + 1

    if track_length is None:
        track_length = compute_track_length(session_path=session_path, trial_type=trial_type)
        console.echo(message=f"Computed track length: {track_length} cm.", level=LogLevel.INFO)

    # Reconstructs place fields from the feather data for run-period masking during SCE detection.
    console.echo(message="Reconstructing place fields from feather...", level=LogLevel.INFO)
    place_fields = _reconstruct_place_fields(analysis_path=output_path, track_length=track_length)

    # Runs the SCE detection pipeline.
    console.echo(message="Running SCE detection...", level=LogLevel.INFO)
    sce_detector = SCEDetector(
        session_path=session_path,
        track_length=track_length,
        fluorescence_column=fluorescence_column,
        place_fields=place_fields,
        configuration=sce_configuration,
    )

    sce_results = sce_detector.detect_events()
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

    # Expands per-cell SCE columns to match the per-place-field row structure and appends to the DataFrame.
    sce_columns = _aggregate_sce_columns(cell_count=cell_count, sce_results=sce_results)

    # Separates numpy array columns from list columns for proper expansion.
    array_column_names = [
        "sce_participation_count_rest",
        "sce_participation_count_run",
        "sce_participation_rate_rest",
        "sce_participation_rate_run",
        "sce_mean_onset_rank_rest",
        "sce_mean_onset_rank_run",
    ]
    list_column_names = ["sce_events_rest", "sce_events_run"]

    sce_series: list[pl.Series] = [
        pl.Series(name=name, values=_expand_cell_array_to_rows(cell_array=sce_columns[name], row_cell_ids=row_cell_ids))
        for name in array_column_names
    ]
    for name in list_column_names:
        sce_series.append(
            pl.Series(
                name=name, values=_expand_cell_list_to_rows(cell_list=sce_columns[name], row_cell_ids=row_cell_ids)
            )
        )

    dataframe = dataframe.hstack(sce_series)

    # Writes the updated DataFrame back to the same feather file.
    dataframe.write_ipc(file=output_path)
    console.echo(message=f"SCE columns appended to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe


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
    """Runs all three analysis pipelines sequentially and assembles a unified per-place-field DataFrame.

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
        A polars DataFrame with one row per place field and columns for place field, reward cell, and SCE metrics.
    """
    if track_length is None:
        track_length = compute_track_length(session_path=session_path, trial_type=trial_type)
        console.echo(message=f"Computed track length: {track_length} cm.", level=LogLevel.INFO)

    generate_place_field_dataframe(
        session_path=session_path,
        track_length=track_length,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        place_configuration=place_configuration,
    )

    append_reward_cell_columns(
        session_path=session_path,
        track_length=track_length,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        reward_configuration=reward_configuration,
    )

    dataframe = append_sce_columns(
        session_path=session_path,
        track_length=track_length,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        sce_configuration=sce_configuration,
    )

    return dataframe
