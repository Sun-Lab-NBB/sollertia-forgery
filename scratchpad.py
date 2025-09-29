import json
from pathlib import Path
import warnings

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel, console

warnings.filterwarnings("ignore")

root_path = Path("/mnt/data/data_checks/Data/2025-08-21-16-35-48-370149")

console.enable()


@njit(parallel=True, cache=True)
def _find_duplicate_cell_candidates(
    x_positions: NDArray[np.uint16],
    y_positions: NDArray[np.uint16],
    cell_sizes: NDArray[np.uint16],
    cell_radii: NDArray[np.float32],
    stripe_borders: NDArray[np.uint16],
    size_threshold: float = 0.8,
    radius_threshold: float = 0.8,
    height_threshold: int = 20,
) -> NDArray[np.uint16]:
    """Finds cell ROIs that might be duplicates based on their morphological feature similarities.

    Notes:
        This function only considers cell pairs separated by a stripe border, since it is specifically tuned to detect
        duplication due to the Mesoscope voltage divider malfunction.

    Args:
        x_positions: The x-position of each cell's ROI centroid.
        y_positions: The y-position of each cell's ROI centroid.
        cell_sizes: The size of each cell's ROI, in pixels.
        cell_radii: The radius of each cell's ROI, in pixels.
        stripe_borders: The x-positions of the borders between the Mesoscope stripes (imaging ROIs).
        size_threshold: The maximum allowed relative size difference between a pair of cells for the cells to be
            considered duplicates.
        radius_threshold: The maximum allowed relative radius difference between a pair of cells for the cells to be
            considered duplicates.
        height_threshold: The maximum difference in the height (y-position) between the centroids of a pair of cells
            for the cells to be considered duplicates.

    Returns:
        A NumPy array that contains the data for each duplicate candidate cell pair. For each candidate, the array
        stores the index of the first cell and the index of the second cell.
    """
    n_cells = len(x_positions)
    n_borders = len(stripe_borders)

    # Pre-allocates the intermediate result arrays
    max_pairs = n_cells * (n_cells - 1) // 2
    cell_1_array = np.empty(max_pairs, dtype=np.uint16)
    cell_2_array = np.empty(max_pairs, dtype=np.uint16)

    # Processes each cell in parallel
    for cell_1 in prange(n_cells):
        # Compares each processed cell with every other cell. This accounts for the inverse cell pairing to avoid
        # processing the same pair of cells more than once.
        for cell_2 in range(cell_1 + 1, n_cells):
            # Calculate unique index for this pair
            # For pair (i,j), index = i*(n_cells - (i+1)/2) + (j-i-1)
            pair_indices = cell_1 * n_cells - (cell_1 * (cell_1 + 1)) // 2 + (cell_2 - cell_1 - 1)

            # Extracts the x-positions of the evaluated cells.
            cell_1_x_position = x_positions[cell_1]
            cell_2_x_position = x_positions[cell_2]

            # Determines which stripe (Mesoscope ROI) each of the processes cells belongs to.
            stripe_1 = -1
            stripe_2 = -1
            for stripe in range(n_borders - 1):
                if stripe_borders[stripe] <= cell_1_x_position < stripe_borders[stripe + 1]:
                    stripe_1 = stripe
                if stripe_borders[stripe] <= cell_2_x_position < stripe_borders[stripe + 1]:
                    stripe_2 = stripe

            # Handles cells located in the last stripe (beyond the second border)
            if cell_1_x_position >= stripe_borders[-1]:
                stripe_1 = n_borders - 1
            if cell_2_x_position >= stripe_borders[-1]:
                stripe_2 = n_borders - 1

            # Checks if cells are in adjacent stripes (must differ by exactly 1).
            if abs(stripe_1 - stripe_2) != 1:
                continue

            # Verifies that the cells are located on the opposite sides of the same border.
            straddled_border = stripe_borders[max(stripe_1, stripe_2)]
            if not (
                (cell_1_x_position < straddled_border < cell_2_x_position)
                or (cell_2_x_position < straddled_border < cell_1_x_position)
            ):
                continue

            # Ensures that the cell centroids are separated by at most height thresholds of pixels. Since cell
            # duplication forces the same cell to appear in different stripes, the y-position of the duplicates must
            # be mostly identical. Since non-rigid registration may modify this position, a certain degree of y-axis
            # mismatch is tolerated.
            if abs(y_positions[cell_1] - y_positions[cell_2]) >= height_threshold:
                continue

            # Checks whether the two cells have an approximately similar pixel size and radius. If true, the two cells
            # are considered morphologically similar and, by extensions, a good candidate for being a duplicate.
            # noinspection PyTypeChecker
            npix_similarity = min(cell_sizes[cell_1], cell_sizes[cell_2]) / max(cell_sizes[cell_1], cell_sizes[cell_2])
            if npix_similarity < size_threshold:
                continue

            # noinspection PyTypeChecker
            radius_similarity = min(cell_radii[cell_1], cell_radii[cell_2]) / max(
                cell_radii[cell_1], cell_radii[cell_2]
            )
            if radius_similarity < radius_threshold:
                continue

            # If all checks are passed, stores the discovered pair of cells in the output arrays.
            cell_1_array[pair_indices] = cell_1
            cell_2_array[pair_indices] = cell_2

    # Filters out sentinel values
    valid_mask = cell_1_array >= 0  # All valid indices are >= 0
    valid_indices = np.where(valid_mask)[0]

    valid_cell_1 = cell_1_array[valid_indices]
    valid_cell_2 = cell_2_array[valid_indices]

    cell_pairs: NDArray[np.uint16] = np.empty((len(valid_cell_1), 2), dtype=np.uint16)
    cell_pairs[:, 0] = valid_cell_1[valid_indices]
    cell_pairs[:, 1] = valid_cell_2[valid_indices]

    return cell_pairs


@njit(parallel=True, cache=True)
def compute_correlations_parallel(
    cell_fluorescence: NDArray[np.float32], pair_candidates: NDArray[np.uint16], correlation_threshold: float
) -> tuple[NDArray[np.uint16], NDArray[np.float32]]:
    """
    Compute correlations for all pairs in parallel.
    """

    # Pre-allocates the output array
    n_pairs = len(pair_candidates)
    correlations = np.zeros(n_pairs, dtype=np.float32)

    # Processes each pair of cells in parallel
    for pair_id in prange(n_pairs):
        cell_1 = pair_candidates[pair_id][0]
        cell_2 = pair_candidates[pair_id][1]

        cell_1_signal = cell_fluorescence[cell_1]
        cell_2_signal = cell_fluorescence[cell_2]

        # Calculates the Pearson correlation coefficient for the pair's fluorescence.
        mean_cell_1_signal = np.mean(cell_1_signal)
        mean_cell_2_signal = np.mean(cell_2_signal)
        cell_1_signal_std = np.std(cell_1_signal)
        cell_2_signal_std = np.std(cell_2_signal)

        if cell_1_signal_std < 1e-10 or cell_2_signal_std < 1e-10:
            # Signals with no variation are likely non-cell artifacts that are automatically excluded from analysis.
            correlations[pair_id] = 0.0
        else:
            # Otherwise, calculates the correlation coefficient for the processed cells.
            signal1_norm = (cell_1_signal - mean_cell_1_signal) / cell_1_signal_std
            signal2_norm = (cell_2_signal - mean_cell_2_signal) / cell_2_signal_std
            correlations[pair_id] = np.mean(signal1_norm * signal2_norm)

    # Filters out the pairs of cells that do not satisfy the correlation threshold.
    valid_mask = correlations >= correlation_threshold
    n_valid = np.sum(valid_mask)
    filtered_candidates = np.empty((n_valid, 2), dtype=pair_candidates.dtype)
    filtered_correlations = np.empty(n_valid, dtype=np.float32)
    valid_idx = 0
    for idx in range(n_pairs):
        if valid_mask[idx]:
            filtered_candidates[valid_idx] = pair_candidates[idx]
            filtered_correlations[valid_idx] = correlations[idx]
            valid_idx += 1

    return filtered_candidates, filtered_correlations


@njit(cache=True)
def get_cell_stripe_index(x_positions: NDArray[np.uint16], stripe_borders: NDArray[np.uint16]) -> np.ndarray:
    """
    Vectorized stripe index calculation for all cells.
    """
    n_cells = len(x_positions)
    stripe_indices = np.zeros(n_cells, dtype=np.int32)

    for i in range(n_cells):
        x_pos = x_positions[i]
        stripe_idx = 0

        for j in range(len(stripe_borders) - 1):
            if stripe_borders[j] <= x_pos < stripe_borders[j + 1]:
                stripe_idx = j
                break
        else:
            if x_pos >= stripe_borders[-1]:
                stripe_idx = len(stripe_borders) - 1

        stripe_indices[i] = stripe_idx

    return stripe_indices


@njit(parallel=True, cache=True)
def select_best_duplicate_pairs(
    pair_candidates: NDArray[np.uint16],
    correlations: NDArray[np.float32],
    cell_sizes: NDArray[np.uint16],
    cell_radii: NDArray[np.float32],
    stripe_indices: NDArray[np.uint16]
) -> np.ndarray:
    """
    Select the best duplicate pairs ensuring each cell appears at most once.

    Returns
    -------
    np.ndarray
        Array of cell indices to remove (cells from right stripes)
    """
    n_pairs = len(pair_candidates)
    n_cells = len(cell_sizes)

    # Pre-compute all similarity scores
    similarity_scores = np.zeros(n_pairs, dtype=np.float32)

    for idx in prange(n_pairs):
        cell_1 = int(pair_candidates[idx, 0])
        cell_2 = int(pair_candidates[idx, 1])

        npix_ratio = abs(cell_sizes[cell_1] - cell_sizes[cell_2]) / max(cell_sizes[cell_1], cell_sizes[cell_2])
        radius_ratio = abs(cell_radii[cell_1] - cell_radii[cell_2]) / max(cell_radii[cell_1], cell_radii[cell_2])

        # Combined score: 50% correlation, 25% size similarity, 25% radius similarity
        similarity_scores[idx] = (
                0.5 * correlations[idx] +
                0.25 * (1 - npix_ratio) +
                0.25 * (1 - radius_ratio)
        )

    # Sort by similarity score (descending)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Track which cells have been used (numba doesn't support sets)
    used_cells = np.zeros(n_cells, dtype=np.bool_)
    cells_to_remove_list = np.zeros(n_pairs, dtype=np.int32)
    remove_count = 0

    for idx in sorted_indices:
        cell_1 = int(pair_candidates[idx, 0])
        cell_2 = int(pair_candidates[idx, 1])

        if not used_cells[cell_1] and not used_cells[cell_2]:
            # Remove cell from right stripe (higher stripe index)
            if stripe_indices[cell_1] < stripe_indices[cell_2]:
                cells_to_remove_list[remove_count] = cell_2
            else:
                cells_to_remove_list[remove_count] = cell_1
            remove_count += 1

            used_cells[cell_1] = True
            used_cells[cell_2] = True

    # Return only the valid portion of the array
    return cells_to_remove_list[:remove_count]


def discover_duplicate_cells(
    session_data_path: Path,
    size_threshold: float = 0.5,
    radius_threshold: float = 0.5,
    height_threshold: int = 200,
    correlation_threshold: float = 0.5,
) -> np.ndarray:
    """
    Discover duplicate cells with parallelized fluorescence verification.
    Returns array of original cell indices to keep.
    """
    # Resolves the paths to the necessary session data files
    mask_path = session_data_path.joinpath("processed_data", "mesoscope_data", "suite2p", "combined", "stat.npy")
    cell_classification_path = session_data_path.joinpath(
        "processed_data", "mesoscope_data", "suite2p", "combined", "iscell.npy"
    )
    line_path = session_data_path.joinpath("source_data", "mesoscope_data", "ops.json")
    fluorescence_path = session_data_path.joinpath("processed_data", "mesoscope_data", "suite2p", "combined", "F.npy")
    session_name = root_path.stem  # Extracts the session name to use in console printouts

    # Loads stripe border data to benefit from the heuristic that duplication does not occur in the same stripe.
    with open(line_path) as ops_file:
        ops_data = json.load(ops_file)
        stripe_borders = ops_data["dx"]

    # Loads cell ROI (mask), fluorescence, and classification data.
    masks = np.load(mask_path, allow_pickle=True)
    classification = np.load(cell_classification_path)
    fluorescence = np.load(fluorescence_path)

    # Filters out non-cells using the binary cell classification data.
    original_cell_indices = np.where(classification[:, 0] == 1)[0]
    cell_masks = masks[original_cell_indices]
    cell_fluorescence = fluorescence[original_cell_indices].astype(np.float32)
    n_cells = len(original_cell_indices)

    # Extracts ROI properties used by the morphological duplicate cell discovery algorithm.
    x_positions = np.array([float(mask["med"][1]) for mask in cell_masks], dtype=np.uint16)
    y_positions = np.array([float(mask["med"][0]) for mask in cell_masks], dtype=np.uint16)
    cell_sizes = np.array([int(mask["npix"]) for mask in cell_masks], dtype=np.uint16)
    cell_radii = np.array([float(mask["radius"]) for mask in cell_masks], dtype=np.float32)
    stripe_border_indices = np.array(stripe_borders, dtype=np.uint16)
    console.echo(
        message=(
            f"Processing {n_cells} cells with {len(stripe_borders) - 1} stripe borders for session {session_name}..."
        ),
        level=LogLevel.INFO,
    )

    # Finds duplicate cell pair candidates using ROI morphology.
    pair_candidates = _find_duplicate_cell_candidates(
        x_positions=x_positions,
        y_positions=y_positions,
        cell_sizes=cell_sizes,
        cell_radii=cell_radii,
        stripe_borders=stripe_border_indices,
        size_threshold=size_threshold,
        radius_threshold=radius_threshold,
        height_threshold=height_threshold,
    )

    # If no candidates are found based on morphology, aborts processing early.
    if len(pair_candidates) == 0:
        console.echo(
            message=f"No duplicate cells found based on ROI morphology.",
            level=LogLevel.SUCCESS,
        )
        return original_cell_indices

    # Notifies the user about the discovered candidates.
    console.echo(
        message=(
            f"Found {len(pair_candidates)} duplicate candidates based on ROI morphology. Advancing to fluorescence "
            f"correlation analysis..."
        ),
        level=LogLevel.INFO,
    )

    # Computes the fluorescence correlation across the entire session's movie for each candidate cell pair. Discards
    # cell pairs that do not satisfy the fluorescence correlation threshold.
    pre_filter_count = len(pair_candidates)
    pair_candidates, correlations = compute_correlations_parallel(
        cell_fluorescence=cell_fluorescence,
        pair_candidates=pair_candidates,
        correlation_threshold=correlation_threshold,
    )
    post_filter_count = len(pair_candidates)

    unique_cell_1 = np.unique(pair_candidates[:, 0].astype(int))
    print(f"Number of unique cell_1 indices: {len(unique_cell_1)}")
    print(f"Unique cell_1 values: {unique_cell_1}")

    if len(pair_candidates) == 0:
        console.echo(
            message=(
                f"No duplicate cells found after evaluating the fluorescence correlation between morphological pairs."
            ),
            level=LogLevel.SUCCESS,
        )
        return original_cell_indices

    console.echo(
        message=(
            f"Discarded {pre_filter_count} after evaluating the fluorescence correlation between morphological pairs. "
            f"Advancing the remaining {post_filter_count} candidates to similarity scoring..."
        ),
        level=LogLevel.INFO,
    )

    # Gets the stripe (Mesoscope ROI) index for each cell.
    stripe_indices = get_cell_stripe_index(x_positions, stripe_border_indices)

    # Gets the stripe (Mesoscope ROI) index for each cell.
    stripe_indices = get_cell_stripe_index(x_positions, stripe_border_indices)

    # Select best duplicate pairs (returns numpy array)
    cells_to_remove_array = select_best_duplicate_pairs(
        pair_candidates,
        correlations,
        cell_sizes,
        cell_radii,
        stripe_indices
    )

    # Convert to set if needed for compatibility with existing code
    filtered_indices_to_remove = set(cells_to_remove_array)

    console.echo(
        message=(
            f"Removing the {len(filtered_indices_to_remove)} duplicate cells after resolving the best duplicate "
            f"candidates..."
        ),
        level=LogLevel.INFO,
    )

    # Create a boolean mask for cells to keep
    keep_mask = np.ones(n_cells, dtype=bool)
    for index in filtered_indices_to_remove:
        keep_mask[index] = False

    # Return original indices of cells to keep
    return original_cell_indices[keep_mask]


# Usage
cells_to_keep = discover_duplicate_cells(root_path, correlation_threshold=0.5, height_threshold=200, size_threshold=0.5, radius_threshold=0.5)
console.echo(f"Final cell count: {len(cells_to_keep)}.", level=LogLevel.SUCCESS)
