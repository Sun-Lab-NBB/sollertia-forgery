"""Generates a unified per-place-field analysis DataFrame from all cell analysis pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console

from ..forging import TRIAL_GEOMETRY_FILENAME, TrialGeometry, FluorescenceColumn
from .sce_analysis import PeriodType, SCEDetector, SCEDetectionConfiguration
from .place_cell_analysis import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_cell_analysis import RewardCellDetector, RewardCellConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .sce_analysis import SCEResult
    from .reward_cell_analysis import RewardCellResults


class _PerCellRows(NamedTuple):
    """Stores per-cell arrays and per-cell per-field lists for assembling the wide-format DataFrame."""

    cell_ids: NDArray[np.int32]
    """Contiguous cell identifiers with length cell_count."""
    is_place: NDArray[np.bool_]
    """Boolean mask marking cells with at least one detected place field."""
    pf_start_cm: list[list[float]]
    """Per-cell list of place-field start positions in centimeters."""
    pf_end_cm: list[list[float]]
    """Per-cell list of place-field end positions in centimeters."""
    pf_center_cm: list[list[float]]
    """Per-cell list of intensity-weighted place-field centroids in centimeters."""
    pf_mean_intensity: list[list[float]]
    """Per-cell list of mean fluorescence intensities within each place field."""
    pf_max_intensity: list[list[float]]
    """Per-cell list of peak fluorescence intensities within each place field."""
    pf_width_cm: list[list[float]]
    """Per-cell list of spatial widths of each place field in centimeters."""
    binned_fluorescence: list[list[list[float]] | None]
    """Per-cell trial x bin fluorescence matrix. None for cells without any detected place field."""


def _build_per_cell_rows(cell_count: int, place_fields: PlaceFields) -> _PerCellRows:
    """Builds per-cell arrays and per-field lists from detected place fields.

    Args:
        cell_count: Total number of cells.
        place_fields: Detected place fields from the place field detection pipeline.

    Returns:
        A named tuple of per-cell arrays and per-cell per-field lists.
    """
    label_image = place_fields.label_image
    bin_size = place_fields.bin_size
    region_count = int(np.max(label_image)) if label_image.size > 0 else 0

    field_cell_ids = place_fields.cell_id
    centers = place_fields.centers
    mean_intensities = place_fields.mean_intensity
    max_intensities = place_fields.max_intensity
    binned_per_trial = place_fields.binned_fluorescence_per_trial

    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)
    # noinspection PyTypeChecker
    is_place: NDArray[np.bool_] = np.zeros(cell_count, dtype=np.bool_)

    pf_start_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_end_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_center_cm: list[list[float]] = [[] for _ in range(cell_count)]
    pf_mean_intensity: list[list[float]] = [[] for _ in range(cell_count)]
    pf_max_intensity: list[list[float]] = [[] for _ in range(cell_count)]
    pf_width_cm: list[list[float]] = [[] for _ in range(cell_count)]

    # Extracts labeled region boundaries for each field and appends to its parent cell's per-field lists.
    for field_index in range(region_count):
        label = field_index + 1
        cell_index = int(field_cell_ids[field_index])
        # noinspection PyTypeChecker
        bins: NDArray[np.int64] = np.where(label_image[cell_index, :] == label)[0]

        # Identifies wrapped fields by a gap in the sorted bin indices.
        # noinspection PyTypeChecker
        gap_indices: NDArray[np.int64] = np.where(np.diff(bins) > 1)[0]
        if len(gap_indices) > 0:
            gap = gap_indices[0]
            start_bin = bins[gap + 1]
            end_bin = bins[gap]
        else:
            start_bin = bins[0]
            end_bin = bins[-1]

        is_place[cell_index] = True
        pf_start_cm[cell_index].append(float(start_bin * bin_size))
        pf_end_cm[cell_index].append(float(end_bin * bin_size))
        pf_center_cm[cell_index].append(float(centers[field_index, 1]))
        pf_mean_intensity[cell_index].append(float(mean_intensities[field_index]))
        pf_max_intensity[cell_index].append(float(max_intensities[field_index]))
        pf_width_cm[cell_index].append(float(len(bins) * bin_size))

    # Populates the trial x bin matrix only for cells with a detected place field; stores None for all other cells
    # to keep the feather file compact.
    binned_fluorescence: list[list[list[float]] | None] = [
        binned_per_trial[cell_index].tolist() if is_place[cell_index] else None for cell_index in range(cell_count)
    ]

    return _PerCellRows(
        cell_ids=cell_ids,
        is_place=is_place,
        pf_start_cm=pf_start_cm,
        pf_end_cm=pf_end_cm,
        pf_center_cm=pf_center_cm,
        pf_mean_intensity=pf_mean_intensity,
        pf_max_intensity=pf_max_intensity,
        pf_width_cm=pf_width_cm,
        binned_fluorescence=binned_fluorescence,
    )


def _aggregate_reward_cell_columns(reward_results: RewardCellResults) -> dict[str, NDArray]:
    """Extracts per-cell reward cell metrics as flat column arrays.

    Args:
        reward_results: Results from the reward cell detection pipeline.

    Returns:
        A dictionary mapping column names to numpy arrays of per-cell values.
    """
    spatial = reward_results.spatial_results

    return {
        "center_of_mass_cm": spatial.centers_of_mass,
        "speed_activity_correlation": reward_results.speed_activity_correlations,
        "is_slowing_correlated": reward_results.is_slowing_correlated,
        "is_reward_cell": spatial.is_significant & reward_results.is_reward_proximal,
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
    # noinspection PyTypeChecker
    participation: NDArray[np.int32] = np.zeros((2, cell_count), dtype=np.int32)
    # noinspection PyTypeChecker
    rank_sum: NDArray[np.float32] = np.zeros((2, cell_count), dtype=np.float32)
    # noinspection PyTypeChecker
    rank_count: NDArray[np.int32] = np.zeros((2, cell_count), dtype=np.int32)
    # noinspection PyTypeChecker
    total_sces: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    # noinspection PyTypeChecker
    period_counter: NDArray[np.int32] = np.zeros(2, dtype=np.int32)
    sce_events: list[list[list[tuple[int, int]]]] = [[[] for _ in range(cell_count)] for _ in range(2)]

    for result in sce_results:
        # Maps the PeriodType string enum to the numeric row index (REST=0, RUN=1) used throughout the aggregation.
        period = 0 if result.period_type == PeriodType.REST else 1
        period_index = int(period_counter[period])
        period_counter[period] += 1
        sce_count = int(np.max(result.sce_labels))
        total_sces[period] += sce_count

        if sce_count == 0:
            continue

        # Builds a (sample_count, sce_count) binary matrix mapping samples to their SCE label, then computes a
        # (cell_count, sce_count) participation matrix via matrix multiplication with the onset matrix.
        # noinspection PyTypeChecker
        sce_sample_indices: NDArray[np.int64] = np.where(result.sce_labels > 0)[0]
        # noinspection PyTypeChecker
        sample_to_sce: NDArray[np.float32] = np.zeros((result.onset_matrix.shape[1], sce_count), dtype=np.float32)
        sample_to_sce[sce_sample_indices, result.sce_labels[sce_sample_indices] - 1] = 1.0
        # noinspection PyTypeChecker
        cell_sce_participation: NDArray[np.bool_] = (result.onset_matrix.astype(np.float32) @ sample_to_sce) > 0

        # Accumulates participation counts per cell across all SCEs in this period.
        participation[period] += cell_sce_participation.sum(axis=1).astype(np.int32)

        # Records per-cell (period_index, sce_label) tuples and computes onset ranks for each SCE.
        for sce_label in range(1, sce_count + 1):
            # noinspection PyTypeChecker
            participating_indices: NDArray[np.int64] = np.where(cell_sce_participation[:, sce_label - 1])[0]
            participant_count = len(participating_indices)

            for cell in participating_indices:
                sce_events[period][cell].append((period_index, sce_label))

            if participant_count > 1:
                # Computes normalized onset ranks from the first onset sample within this SCE.
                # noinspection PyTypeChecker
                sce_samples: NDArray[np.int64] = np.where(result.sce_labels == sce_label)[0]
                onset_window = result.onset_matrix[participating_indices][:, sce_samples]
                first_onset = np.argmax(onset_window, axis=1)
                # noinspection PyTypeChecker
                normalized_ranks: NDArray[np.float32] = (
                    np.argsort(np.argsort(first_onset)).astype(np.float32) / (participant_count - 1)
                ).astype(np.float32)
                rank_sum[period, participating_indices] += normalized_ranks
                rank_count[period, participating_indices] += 1
            elif participant_count == 1:
                rank_sum[period, participating_indices] += 0.5
                rank_count[period, participating_indices] += 1

    # Computes participation rates and mean onset ranks per period type.
    # noinspection PyTypeChecker
    rate: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    # noinspection PyTypeChecker
    mean_rank: NDArray[np.float32] = np.full((2, cell_count), np.nan, dtype=np.float32)
    for period in range(2):
        if total_sces[period] > 0:
            rate[period] = (participation[period] / total_sces[period]).astype(np.float32)
        # noinspection PyTypeChecker
        has_ranks: NDArray[np.bool_] = rank_count[period] > 0
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

    Notes:
        The saved binned_fluorescence column now holds a per-lap (trial x bin) matrix and is null for cells without
        place fields, so this function does not reconstruct the pooled binned_fluorescence. A zero-filled placeholder
        is returned in its slot, which satisfies the dataclass contract without requiring per-lap data to be loaded
        or averaged.

    Args:
        analysis_path: Path to the analysis feather file containing place field columns.
        track_length: Length of the track in centimeters, used to derive bin_size from the per-lap fluorescence width.

    Returns:
        A PlaceFields object with label_image, centers, and bin_size reconstructed from the feather data, and a
        zero-filled binned_fluorescence placeholder.
    """
    dataframe = pl.read_ipc(source=analysis_path, memory_map=True)
    cell_count = len(dataframe)

    # Derives the bin count from the first non-null trial x bin matrix; falls back to a single-bin placeholder.
    populated_binned = dataframe.filter(pl.col("binned_fluorescence").is_not_null())["binned_fluorescence"]
    if len(populated_binned) > 0:
        first_binned = populated_binned[0].to_list()
        bin_count = len(first_binned[0]) if first_binned and len(first_binned[0]) > 0 else 1
    else:
        bin_count = 1
    bin_size = track_length / bin_count

    # noinspection PyTypeChecker
    label_image: NDArray[np.int32] = np.zeros((cell_count, bin_count), dtype=np.int32)
    centers_list: list[list[float]] = []

    cell_id_column = dataframe["cell_id"].to_list()
    pf_start_column = dataframe["pf_start_cm"].to_list()
    pf_end_column = dataframe["pf_end_cm"].to_list()
    pf_center_column = dataframe["pf_center_cm"].to_list()

    # Walks through each cell's per-field lists and reconstructs the labeled image one field at a time.
    next_label = 1
    for row_index in range(cell_count):
        starts = pf_start_column[row_index]
        if not starts:
            continue

        cell_index = int(cell_id_column[row_index])
        ends = pf_end_column[row_index]
        centers = pf_center_column[row_index]

        for field_index in range(len(starts)):
            start_bin = int(starts[field_index] / bin_size)
            end_bin = int(ends[field_index] / bin_size)

            # Handles both contiguous and wrapped place fields.
            if start_bin <= end_bin:
                label_image[cell_index, start_bin : end_bin + 1] = next_label
            else:
                label_image[cell_index, start_bin:] = next_label
                label_image[cell_index, : end_bin + 1] = next_label

            centers_list.append([float(cell_index), float(centers[field_index])])
            next_label += 1

    # noinspection PyTypeChecker
    centers_array: NDArray[np.float32] = (
        np.array(centers_list, dtype=np.float32) if centers_list else np.array([], dtype=np.float32).reshape(0, 2)
    )

    return PlaceFields(
        label_image=label_image,
        binned_fluorescence=np.zeros((cell_count, bin_count), dtype=np.float32),
        centers=centers_array,
        bin_size=bin_size,
    )


def _get_track_length(session_path: Path, trial_type: str) -> float:
    """Returns the canonical track length for the given trial type read from the session's trial_geometry.yaml file.

    Args:
        session_path: Path to the session's dataset directory containing the trial geometry data file.
        trial_type: Trial type name to look up in the trial geometry data file.

    Returns:
        Canonical track length in centimeters.
    """
    geometry = TrialGeometry.from_yaml(file_path=session_path.joinpath(TRIAL_GEOMETRY_FILENAME))
    return geometry.entries[trial_type].trial_length_cm


def _resolve_output_path(session_path: Path, output_directory: Path | None) -> Path:
    """Resolves the output feather file path for a given session.

    Args:
        session_path: Path to the session's dataset directory.
        output_directory: Directory to save the analysis feather file. Defaults to the session's dataset directory.

    Returns:
        The resolved output path for the analysis feather file.
    """
    save_directory = output_directory if output_directory is not None else session_path
    return save_directory / f"{session_path.name}_analysis.feather"


def generate_place_field_dataframe(
    session_path: Path,
    output_directory: Path | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
    trial_type: str = "ABC",
    place_configuration: PlaceFieldDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the place field detection pipeline and writes a per-place-field analysis feather file.

    Args:
        session_path: Path to the session's dataset directory.
        output_directory: Directory to save the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the analysis
            input.
        trial_type: Trial type to analyze.
        place_configuration: Place field detection parameters. Uses defaults if None.

    Returns:
        A polars DataFrame with one row per place field and columns for place field boundaries, intensities, and
        binned fluorescence.
    """
    # Runs the place field detection pipeline.
    console.echo(message="Running place field detection...", level=LogLevel.INFO)
    place_detector = PlaceFieldDetector(
        session_path=session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
        configuration=place_configuration,
    )
    place_fields = place_detector.detect(run_shuffle=False)
    cell_count = place_fields.binned_fluorescence.shape[0]
    place_cell_count = int(place_fields.has_place_field.sum())
    console.echo(
        message=f"Place field detection complete: {place_cell_count}/{cell_count} place cells.",
        level=LogLevel.SUCCESS,
    )

    # Builds per-cell rows from the detection results.
    console.echo(message="Assembling per-cell analysis DataFrame...", level=LogLevel.INFO)
    per_cell_rows = _build_per_cell_rows(cell_count=cell_count, place_fields=place_fields)

    # Builds the wide-format DataFrame where each row represents one cell and all per-field metrics are list columns
    # indexable by field number.
    dataframe = pl.DataFrame(
        {
            "cell_id": per_cell_rows.cell_ids,
            "is_place": pl.Series(values=per_cell_rows.is_place, dtype=pl.Boolean),
            "pf_start_cm": pl.Series(values=per_cell_rows.pf_start_cm, dtype=pl.List(pl.Float32)),
            "pf_end_cm": pl.Series(values=per_cell_rows.pf_end_cm, dtype=pl.List(pl.Float32)),
            "pf_center_cm": pl.Series(values=per_cell_rows.pf_center_cm, dtype=pl.List(pl.Float32)),
            "binned_fluorescence": pl.Series(
                name="binned_fluorescence",
                values=per_cell_rows.binned_fluorescence,
                dtype=pl.List(pl.List(pl.Float32)),
            ),
            "pf_mean_intensity": pl.Series(values=per_cell_rows.pf_mean_intensity, dtype=pl.List(pl.Float32)),
            "pf_max_intensity": pl.Series(values=per_cell_rows.pf_max_intensity, dtype=pl.List(pl.Float32)),
            "pf_width_cm": pl.Series(values=per_cell_rows.pf_width_cm, dtype=pl.List(pl.Float32)),
        },
    ).sort("cell_id")

    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    console.echo(
        message=(
            f"Per-cell DataFrame assembled: {row_count} rows, {column_count} columns. "
            f"{place_cell_count} cells with place fields."
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
    output_directory: Path | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
    trial_type: str = "ABC",
    reward_configuration: RewardCellConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the reward cell detection pipeline and appends reward cell columns to an existing analysis feather file.

    Args:
        session_path: Path to the session's dataset directory.
        output_directory: Directory containing the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the analysis
            input.
        trial_type: Trial type to analyze.
        reward_configuration: Reward cell detection parameters. Uses defaults if None.

    Returns:
        The updated polars DataFrame with reward cell columns appended.
    """
    output_path = _resolve_output_path(session_path=session_path, output_directory=output_directory)

    # Reads without memory-mapping so Windows allows writing back to the same path after appending columns.
    dataframe = pl.read_ipc(source=output_path, memory_map=False)
    # noinspection PyTypeChecker
    is_place_row: NDArray[np.bool_] = dataframe["is_place"].to_numpy()

    # Runs the reward cell detection pipeline. The detector resolves the canonical track length and the reward zone
    # position from the session's trial geometry data file.
    console.echo(message="Running reward cell detection...", level=LogLevel.INFO)
    reward_detector = RewardCellDetector(
        session_path=session_path,
        trial_type=trial_type,
        fluorescence_column=fluorescence_column,
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

    # Nulls reward columns for place cells to preserve mutual exclusion.
    reward_columns = _aggregate_reward_cell_columns(reward_results=reward_results)
    # noinspection PyTypeChecker
    place_indices: NDArray[np.int64] = np.where(is_place_row)[0]
    reward_series: list[pl.Series] = []
    for name, cell_array in reward_columns.items():
        series = pl.Series(name=name, values=cell_array)
        if len(place_indices) > 0:
            series = series.scatter(indices=place_indices, values=None)
        reward_series.append(series)

    dataframe = dataframe.hstack(reward_series)

    # Writes the updated DataFrame back to the same feather file.
    dataframe.write_ipc(file=output_path)
    console.echo(message=f"Reward cell columns appended to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe


def append_sce_columns(
    session_path: Path,
    output_directory: Path | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
    trial_type: str = "ABC",
    sce_configuration: SCEDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs the SCE detection pipeline and appends SCE columns to an existing analysis feather file.

    Args:
        session_path: Path to the session's dataset directory.
        output_directory: Directory containing the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the analysis
            input.
        trial_type: Trial type to analyze.
        sce_configuration: SCE detection parameters. Uses defaults if None.

    Returns:
        The updated polars DataFrame with SCE columns appended.
    """
    output_path = _resolve_output_path(session_path=session_path, output_directory=output_directory)

    # Reads without memory-mapping so Windows allows writing back to the same path after appending columns.
    dataframe = pl.read_ipc(source=output_path, memory_map=False)
    cell_count = len(dataframe)

    # Resolves the canonical track length from the trial geometry data file for SCE detection and place field
    # reconstruction. SCEDetector still receives track_length explicitly because it does not share the
    # assemble_run_session_data pipeline with the place- and reward-cell detectors.
    track_length = _get_track_length(session_path=session_path, trial_type=trial_type)

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

    # Appends per-cell SCE columns directly to the wide-format DataFrame.
    sce_columns = _aggregate_sce_columns(cell_count=cell_count, sce_results=sce_results)
    sce_series: list[pl.Series] = [pl.Series(name=name, values=values) for name, values in sce_columns.items()]

    dataframe = dataframe.hstack(sce_series)

    # Writes the updated DataFrame back to the same feather file.
    dataframe.write_ipc(file=output_path)
    console.echo(message=f"SCE columns appended to {output_path}.", level=LogLevel.SUCCESS)

    return dataframe


def generate_analysis_dataframe(
    session_path: Path,
    output_directory: Path | None = None,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.SINGLE_DAY_SUBTRACTED,
    trial_type: str = "ABC",
    place_configuration: PlaceFieldDetectionConfiguration | None = None,
    reward_configuration: RewardCellConfiguration | None = None,
    sce_configuration: SCEDetectionConfiguration | None = None,
) -> pl.DataFrame:
    """Runs all three analysis pipelines sequentially and assembles a unified per-place-field DataFrame.

    Args:
        session_path: Path to the session's dataset directory.
        output_directory: Directory to save the analysis feather file. Defaults to the session file's directory.
        fluorescence_column: The neuropil-subtracted, baseline-corrected fluorescence column to use as the analysis
            input.
        trial_type: Trial type to analyze.
        place_configuration: Place field detection parameters. Uses defaults if None.
        reward_configuration: Reward cell detection parameters. Uses defaults if None.
        sce_configuration: SCE detection parameters. Uses defaults if None.

    Returns:
        A polars DataFrame with one row per place field and columns for place field, reward cell, and SCE metrics.
    """
    generate_place_field_dataframe(
        session_path=session_path,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        place_configuration=place_configuration,
    )

    append_reward_cell_columns(
        session_path=session_path,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        reward_configuration=reward_configuration,
    )

    return append_sce_columns(
        session_path=session_path,
        output_directory=output_directory,
        fluorescence_column=fluorescence_column,
        trial_type=trial_type,
        sce_configuration=sce_configuration,
    )
