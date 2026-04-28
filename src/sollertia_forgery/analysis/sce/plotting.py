"""Per-session plotting helpers for the SCE pipeline.

Module-level functions consume an :class:`SCEReport` plus, where noted, the session's ``data.feather`` opened
memory-mapped at plot time. Persistence and summarization stay on the report; assembly detection runs locally so
the report itself does not need to carry a live detector.

References:
    - Lopes-dos-Santos, Ribeiro & Tort (2013). Detecting cell assemblies in large neuronal populations.
      J Neurosci Methods. https://doi.org/10.1016/j.jneumeth.2013.04.010 -- the ICA-CS algorithm.
    - Mölter, Avitan & Goodhill (2018). Detecting neural assemblies in calcium imaging data. BMC Biol.
      https://doi.org/10.1186/s12915-018-0606-4 -- comparative benchmark recommending ICA-CS over hierarchical
      clustering.
    - Hyvärinen (1999). Fast and robust fixed-point algorithms for ICA. IEEE Trans Neural Netw.
      https://doi.org/10.1109/72.761722 -- deflation FastICA fixed-point iteration used here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from threadpoolctl import threadpool_limits
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
import matplotlib.pyplot as plt
from ataraxis_base_utilities import resolve_worker_count

from ...forging import FluorescenceColumn
from ..shared_utilities import trim_acquisition_warmup
from .sce_report import SCEReport, SCECellColumn, SCEPeriodColumn
from .sce_protocol import SCEDetectionConfiguration
from ...shared_assets import DatasetColumn

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...shared_assets import DatasetSession


_MINIMUM_SCE_COUNT_FOR_ASSEMBLY: int = 2
"""Minimum number of SCEs required in a period to attempt cell-assembly detection."""
_MAX_DEFAULT_SCE_TRACE_CELLS: int = 5
"""Default cap on the number of cells drawn by ``plot_sce_cells_across_periods`` when the caller does not
supply explicit cell indices."""
_MINIMUM_OBSERVATIONS_FOR_VARIANCE: int = 2
"""Minimum number of samples or cells required for a variance-, PCA-, or rank-correlation-based step to produce
a defined output. Below this threshold the corresponding helper short-circuits to NaN."""
_ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE: int = 10
"""Preferred BLAS thread count per ICA-CS / reactivation shuffle worker. Mirrors the bleaching analyzer's
``_PREFERRED_WORKERS_PER_SESSION = 10`` constant: the Lanczos matvec and the per-period reactivation GEMMs
are BLAS-bound and stop scaling cleanly past ten threads, so the shuffle-level allocator
(:func:`_resolve_ica_shuffle_allocation`) targets this width and uses the remaining budget to spawn more
parallel workers."""
_ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE: int = 5
"""Floor on per-worker BLAS threads. Falling below this floor reduces parallel-shuffle count one worker at a
time rather than spawning under-resourced workers whose matvec performance would collapse."""
_ICA_BLAS_THREAD_MULTIPLE: int = 5
"""Per-worker BLAS thread counts are rounded down to this multiple for clean allocation."""


def plot_sce_cells_across_periods(
    report: SCEReport,
    *,
    session: DatasetSession,
    fluorescence_column: FluorescenceColumn = FluorescenceColumn.MULTI_DAY_SUBTRACTED,
    cell_indices: NDArray[np.int32] | list[int] | None = None,
    cell_count: int = _MAX_DEFAULT_SCE_TRACE_CELLS,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Plots smoothed fluorescence for selected cells across every contiguous experiment period in the
    session, with each period labelled by its ``DatasetColumn.SYSTEM_STATE`` value.

    Notes:
        Reads ``data.feather`` and re-applies Savitzky-Golay smoothing once over the entire trimmed trace
        using the persisted ``sce_configuration``. Period boundaries come directly from the system-state
        column so the plot adapts to whatever experiment-state palette the session was recorded with.
        When ``cell_indices`` is omitted, the top SCE-recruited cells by smoothed-trace variance are used;
        if no cell is flagged ``IS_SCE_CELL``, the top cells by variance across the whole session are used
        instead.

    Args:
        report: The SCE report whose persisted ``IS_SCE_CELL`` flags drive default cell selection.
        session: The DatasetSession whose ``data.feather`` is read.
        fluorescence_column: Fluorescence column to use as the analysis input.
        cell_indices: Optional explicit cell indices to plot. Default selects up to ``cell_count`` cells by
            SCE recruitment (``IS_SCE_CELL``) and smoothed-trace variance.
        cell_count: Maximum number of cells in the default selection. Ignored when ``cell_indices`` is
            supplied.
        figure_dpi: Figure DPI.
    """
    timestamps_minutes, smoothed, period_spans = _walk_session_periods(
        session=session,
        fluorescence_column=fluorescence_column,
        sce_configuration=report.summary.sce_configuration,
    )
    if not period_spans:
        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes.text(0.5, 0.5, "No experiment periods detected", transform=axes.transAxes, ha="center", va="center")
        axes.axis("off")
        return figure

    if cell_indices is None:
        cell_indices = _select_sce_cells_by_variance(
            cells=report.cells,
            smoothed=smoothed,
            cell_count=cell_count,
        )
    else:
        cell_indices = np.asarray(cell_indices, dtype=np.int32)

    if cell_indices.size == 0:
        figure, axes = plt.subplots(1, 1, figsize=(10, 4), facecolor="white", dpi=figure_dpi)
        axes.text(0.5, 0.5, "No cells available to plot", transform=axes.transAxes, ha="center", va="center")
        axes.axis("off")
        return figure

    unique_states = sorted({state for _, _, state in period_spans})
    colormap = plt.get_cmap("tab20", max(len(unique_states), 1))
    state_colors = {state: colormap(state_index) for state_index, state in enumerate(unique_states)}

    height_ratios = [0.4] + [1.0] * len(cell_indices)
    figure, all_axes = plt.subplots(
        nrows=1 + len(cell_indices),
        ncols=1,
        figsize=(14, 1.5 * len(cell_indices) + 1),
        facecolor="white",
        dpi=figure_dpi,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    label_axis = all_axes[0]
    state_run_index: dict[str, int] = {}
    for start_sample, end_sample, state in period_spans:
        start_time = float(timestamps_minutes[start_sample])
        end_time = float(timestamps_minutes[end_sample - 1])
        state_run_index[state] = state_run_index.get(state, 0) + 1
        display_label = f"{state.title()} {state_run_index[state]}"
        face_color = state_colors[state]
        label_axis.axvspan(xmin=start_time, xmax=end_time, color=face_color, alpha=0.6)
        label_axis.text(
            x=(start_time + end_time) / 2,
            y=0.5,
            s=display_label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    label_axis.set_xlim(float(timestamps_minutes[0]), float(timestamps_minutes[-1]))
    label_axis.set_ylim(0, 1)
    label_axis.set_yticks([])
    for spine_name in ("top", "right", "left", "bottom"):
        label_axis.spines[spine_name].set_visible(False)
    label_axis.set_title("SCE cell traces across experiment periods", fontsize=11)

    trace_axes = all_axes[1:]
    for axis_index, cell_index in enumerate(cell_indices):
        axis = trace_axes[axis_index]
        for start_sample, end_sample, state in period_spans:
            period_time = timestamps_minutes[start_sample:end_sample]
            axis.axvspan(
                xmin=float(period_time[0]),
                xmax=float(period_time[-1]),
                alpha=0.15,
                color=state_colors[state],
            )

        fluorescence_trace = smoothed[cell_index]
        trace_mean = float(np.mean(fluorescence_trace))
        trace_std = float(np.std(fluorescence_trace))
        normalized_trace = (
            (fluorescence_trace - trace_mean) / trace_std if trace_std > 0 else fluorescence_trace - trace_mean
        )
        axis.plot(timestamps_minutes, normalized_trace, color="black", linewidth=0.5, alpha=0.8)
        axis.set_ylabel(f"Cell {int(cell_index)}", fontsize=9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    trace_axes[-1].set_xlabel("Time (minutes)")
    figure.tight_layout()
    return figure


def plot_sce_assemblies(
    report: SCEReport,
    *,
    period_index: int = 0,
    top_n: int = 5,
    shuffle_count: int = 200,
    eigenvalue_significance_percentile: float = 99.0,
    membership_z_threshold: float = 2.0,
    minimum_assembly_size: int = 3,
    title: str | None = None,
    figure_dpi: int = 150,
) -> plt.Figure:
    """Detects and plots the most prominent SCE cell assemblies as raster panels using ICA-CS
    (Lopes-dos-Santos 2013) on the persisted per-period SCE participation matrix. Recomputes assemblies on
    every call so no live detector is required.
    """
    if period_index >= report.periods.height:
        figure, axes = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
        axes.text(
            0.5,
            0.5,
            f"Stationary period {period_index + 1} not found",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes.axis("off")
        return figure

    row = report.periods.row(period_index, named=True)
    sce_labels = np.asarray(row[SCEPeriodColumn.SCE_LABELS.value], dtype=np.int32)
    onset_matrix = _reconstruct_onset_matrix(
        cell_count=int(row[SCEPeriodColumn.CELL_COUNT.value]),
        sample_count=int(row[SCEPeriodColumn.SAMPLE_COUNT.value]),
        onset_cell_indices=row[SCEPeriodColumn.ONSET_CELL_INDICES.value],
        onset_sample_indices=row[SCEPeriodColumn.ONSET_SAMPLE_INDICES.value],
    )
    total_sce_count = int(np.max(sce_labels)) if sce_labels.size > 0 else 0

    # Build the per-SCE participation matrix (cells x sce_count) for ICA-CS, mirroring the upstream
    # cell-feather aggregation but kept local to the plot so callers can rerun with different thresholds.
    if total_sce_count < _MINIMUM_SCE_COUNT_FOR_ASSEMBLY:
        participation_matrix: NDArray[np.float32] = np.empty((onset_matrix.shape[0], 0), dtype=np.float32)
    else:
        sample_count = onset_matrix.shape[1]
        # noinspection PyTypeChecker
        sce_sample_indices: NDArray[np.int64] = np.where(sce_labels > 0)[0]
        # noinspection PyTypeChecker
        sample_to_sce: NDArray[np.float32] = np.zeros((sample_count, total_sce_count), dtype=np.float32)
        sample_to_sce[sce_sample_indices, sce_labels[sce_sample_indices] - 1] = 1.0
        # noinspection PyTypeChecker
        participation_matrix = (onset_matrix.astype(np.float32) @ sample_to_sce > 0).astype(np.float32)

    _, assemblies = _detect_assemblies_ica_cs(
        activity_matrix=participation_matrix,
        shuffle_count=shuffle_count,
        eigenvalue_significance_percentile=eigenvalue_significance_percentile,
        membership_z_threshold=membership_z_threshold,
        minimum_assembly_size=minimum_assembly_size,
    )
    if not assemblies:
        figure, axis = plt.subplots(figsize=(6, 4), facecolor="white", dpi=figure_dpi)
        axis.text(0.5, 0.5, "No assemblies detected", ha="center", va="center", fontsize=12)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        return figure

    display_count = min(top_n, len(assemblies))
    displayed = assemblies[:display_count]
    total_cells = onset_matrix.shape[0]

    column_width = 1.8
    figure_width = column_width * display_count + 1.5
    figure_height = max(5, min(12, total_cells * 0.003 + 2))
    figure, axes_array = plt.subplots(
        nrows=1, ncols=display_count, figsize=(figure_width, figure_height), facecolor="white", dpi=figure_dpi
    )
    axes_list = [axes_array] if display_count == 1 else list(axes_array)

    assembly_colors = ["black", "red", "blue", "green", "magenta"]
    last_index = 0
    for assembly_index, member_cells in enumerate(displayed):
        axis = axes_list[assembly_index]
        color = assembly_colors[assembly_index % len(assembly_colors)]
        sorted_members = np.sort(member_cells)
        axis.scatter(
            x=np.zeros(len(sorted_members)),
            y=sorted_members,
            color=color,
            s=13,
            marker=".",
            linewidths=0,
        )
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(total_cells - 0.5, -0.5)
        axis.set_xticks([])
        yticks = list(range(0, total_cells, 500))
        axis.set_yticks(yticks)
        axis.set_yticklabels([str(t) for t in yticks], fontsize=6)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color(color)
        axis.set_xlabel(f"Assembly {assembly_index + 1}", fontsize=7, color=color)
        last_index = assembly_index

    if last_index > 0:
        axes_list[last_index].tick_params(axis="y", labelleft=False)
    axes_list[0].set_ylabel("Cell number")

    if title is None:
        title = (
            f"SCE Assemblies — Stationary Period {period_index + 1}  "
            f"({total_sce_count} total SCEs, showing top {display_count} assemblies)"
        )
    figure.suptitle(title, fontsize=11)
    figure.subplots_adjust(wspace=0.1, left=0.06, right=0.98, top=0.94, bottom=0.06)
    return figure


# ===== Private helpers ==========================================================================================


def _walk_session_periods(
    *,
    session: DatasetSession,
    fluorescence_column: FluorescenceColumn,
    sce_configuration: SCEDetectionConfiguration,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[tuple[int, int, str]]]:
    """Reads the session's ``data.feather`` and returns the elapsed-minutes timestamps, smoothed fluorescence,
    and contiguous-state period spans suitable for the across-period trace plot.

    Each entry in the returned ``period_spans`` list is ``(start_sample, end_sample_exclusive, state_name)``;
    ``state_name`` is taken verbatim from ``DatasetColumn.SYSTEM_STATE`` so the caller can label periods with
    whatever state palette the session was recorded with.
    """
    df = pl.read_ipc(
        source=session.data_path,
        columns=[DatasetColumn.TIME_US.value, DatasetColumn.SYSTEM_STATE.value, fluorescence_column.value],
        memory_map=True,
    )
    df = trim_acquisition_warmup(df)
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy()
    if time_us.size < _MINIMUM_OBSERVATIONS_FOR_VARIANCE:
        # noinspection PyTypeChecker
        empty_timestamps: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # noinspection PyTypeChecker
        empty_smoothed: NDArray[np.float32] = np.zeros((0, 0), dtype=np.float32)
        return empty_timestamps, empty_smoothed, []

    elapsed_minutes = (time_us - time_us[0]).astype(np.float32) / np.float32(60_000_000.0)
    sampling_rate_hz = 1_000_000.0 / float(np.median(np.diff(time_us)))

    # noinspection PyTypeChecker
    fluorescence: NDArray[np.float32] = np.array(df[fluorescence_column.value].to_list(), dtype=np.float32).T

    smoothing_window_samples = int(sce_configuration.smoothing_window_seconds * sampling_rate_hz)
    if smoothing_window_samples % 2 == 0:
        smoothing_window_samples += 1
    smoothing_window_samples = max(smoothing_window_samples, sce_configuration.smoothing_order + 2)
    smoothing_window_samples = min(smoothing_window_samples, fluorescence.shape[1])
    if smoothing_window_samples <= sce_configuration.smoothing_order:
        # noinspection PyTypeChecker
        smoothed: NDArray[np.float32] = fluorescence.astype(np.float32, copy=False)
    else:
        # noinspection PyTypeChecker
        smoothed = savgol_filter(
            x=fluorescence,
            window_length=smoothing_window_samples,
            polyorder=sce_configuration.smoothing_order,
            axis=1,
        ).astype(np.float32, copy=False)

    states = df[DatasetColumn.SYSTEM_STATE.value].to_list()
    period_spans: list[tuple[int, int, str]] = []
    if states:
        run_start = 0
        current_state = states[0]
        for sample_index in range(1, len(states)):
            if states[sample_index] != current_state:
                period_spans.append((run_start, sample_index, str(current_state)))
                current_state = states[sample_index]
                run_start = sample_index
        period_spans.append((run_start, len(states), str(current_state)))

    return elapsed_minutes, smoothed, period_spans


def _select_sce_cells_by_variance(
    *,
    cells: pl.DataFrame,
    smoothed: NDArray[np.float32],
    cell_count: int,
) -> NDArray[np.int32]:
    """Returns up to ``cell_count`` cell indices to plot, preferring SCE-recruited cells ranked by smoothed-trace
    variance. Falls back to top-variance cells across the full session when no cell is flagged ``IS_SCE_CELL``.
    """
    if smoothed.size == 0:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.int32)

    per_cell_variance = np.std(smoothed, axis=1)
    if SCECellColumn.IS_SCE_CELL.value in cells.columns:
        # noinspection PyTypeChecker
        sce_cell_mask: NDArray[np.bool_] = cells[SCECellColumn.IS_SCE_CELL.value].to_numpy()
    else:
        # noinspection PyTypeChecker
        sce_cell_mask = np.zeros(per_cell_variance.size, dtype=np.bool_)

    # noinspection PyTypeChecker
    candidate_mask: NDArray[np.bool_] = sce_cell_mask if sce_cell_mask.any() else np.ones_like(sce_cell_mask)
    # noinspection PyTypeChecker
    candidate_indices: NDArray[np.int64] = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        # noinspection PyTypeChecker
        return np.zeros(0, dtype=np.int32)

    candidate_variance = per_cell_variance[candidate_indices]
    # noinspection PyTypeChecker
    sorted_descending: NDArray[np.int64] = np.argsort(-candidate_variance, kind="stable")
    selected = candidate_indices[sorted_descending[:cell_count]]
    # noinspection PyTypeChecker
    return np.sort(selected).astype(np.int32)


def _reconstruct_onset_matrix(
    cell_count: int,
    sample_count: int,
    onset_cell_indices: list[int] | None,
    onset_sample_indices: list[int] | None,
) -> NDArray[np.bool_]:
    """Materializes the dense (cell_count, sample_count) bool onset matrix from the persisted sparse encoding."""
    # noinspection PyTypeChecker
    matrix: NDArray[np.bool_] = np.zeros((cell_count, sample_count), dtype=np.bool_)
    if onset_cell_indices is None or onset_sample_indices is None:
        return matrix
    cells = np.asarray(onset_cell_indices, dtype=np.int64)
    samples = np.asarray(onset_sample_indices, dtype=np.int64)
    if cells.size != samples.size or cells.size == 0:
        return matrix
    matrix[cells, samples] = True
    return matrix


def _z_score_along_samples(activity: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Z-scores each row of ``activity`` along the sample axis and returns the (z, has_variance) pair.

    Cells with zero sample-axis variance get a zeroed row in the returned z-matrix and a False entry in the
    mask, so downstream consumers can either drop them or fall back to the all-zero contribution.
    """
    cell_mean = activity.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    cell_std = activity.std(axis=1, keepdims=True).astype(np.float32, copy=False)
    # noinspection PyTypeChecker
    has_variance: NDArray[np.bool_] = cell_std[:, 0] > 0
    safe_std = np.where(cell_std > 0.0, cell_std, np.float32(1.0))
    # noinspection PyTypeChecker
    z: NDArray[np.float32] = ((activity - cell_mean) / safe_std).astype(np.float32, copy=False)
    z[~has_variance, :] = np.float32(0.0)
    return z, has_variance


def _resolve_ica_shuffle_allocation(budget: int, shuffle_count: int) -> tuple[int, int]:
    """Splits a CPU budget between per-shuffle BLAS threads and concurrent shuffle workers.

    Notes:
        Mirrors the saturating allocator used by bleaching / cindra: each worker is filled to
        ``_ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE`` BLAS threads before a new parallel shuffle is spawned, the
        per-worker thread count is rounded down to a multiple of ``_ICA_BLAS_THREAD_MULTIPLE`` for clean
        allocation, and parallelism is reduced one worker at a time whenever the per-worker share would fall
        below ``_ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE``.

    Args:
        budget: Total CPU cores available after the system reservation, as returned by ``resolve_worker_count``.
        shuffle_count: Number of shuffle iterations to run.

    Returns:
        A tuple of ``(blas_threads_per_shuffle, parallel_shuffles)`` whose product never exceeds the budget.
    """
    if shuffle_count <= 1:
        return max(1, budget), 1
    max_at_preferred = max(1, budget // _ICA_PREFERRED_BLAS_THREADS_PER_SHUFFLE)
    parallel_shuffles = min(shuffle_count, max_at_preferred)
    raw_threads = budget // parallel_shuffles
    blas_threads = max(1, (raw_threads // _ICA_BLAS_THREAD_MULTIPLE) * _ICA_BLAS_THREAD_MULTIPLE)

    while blas_threads < _ICA_MINIMUM_BLAS_THREADS_PER_SHUFFLE and parallel_shuffles > 1:
        parallel_shuffles -= 1
        raw_threads = budget // parallel_shuffles
        blas_threads = max(1, (raw_threads // _ICA_BLAS_THREAD_MULTIPLE) * _ICA_BLAS_THREAD_MULTIPLE)

    return blas_threads, parallel_shuffles


def _shuffle_max_eigenvalue(
    z: NDArray[np.float32],
    *,
    shuffle_count: int,
    rng: np.random.Generator,
    minimum_shift_samples: int,
    progress_description: str = "Assembly null shuffle",
    requested_workers: int = 0,
) -> NDArray[np.float32]:
    """Returns the shuffled-distribution maximum eigenvalue per shuffle for the cell-by-cell correlation matrix
    obtained after independent circular shifts of each cell's z-scored activity (Lopes-dos-Santos 2013 ICA-CS
    null).

    Notes:
        Each iteration draws independent per-cell shifts and gathers the shifted z-matrix. The largest
        eigenvalue of ``(shuffled @ shuffled.T) / N`` is recovered via ARPACK Lanczos with ``k=1`` on a
        ``LinearOperator`` whose ``matvec`` is ``shuffled @ (shuffled.T @ v) / N``; the (cell_count,
        cell_count) correlation matrix is never materialised. The shuffles run concurrently on a
        ``ThreadPoolExecutor`` whose budget is split by :func:`_resolve_ica_shuffle_allocation`.
    """
    cell_count, sample_count = z.shape
    # noinspection PyTypeChecker
    output: NDArray[np.float32] = np.zeros(shuffle_count, dtype=np.float32)
    if cell_count == 0 or sample_count < _MINIMUM_OBSERVATIONS_FOR_VARIANCE or shuffle_count == 0:
        return output
    floor = max(1, int(minimum_shift_samples))
    ceil = max(floor + 1, sample_count - floor)
    # noinspection PyTypeChecker
    sample_index_arange: NDArray[np.int64] = np.arange(sample_count, dtype=np.int64)
    # noinspection PyTypeChecker
    cell_arange: NDArray[np.int64] = np.arange(cell_count, dtype=np.int64)[:, np.newaxis]
    inverse_sample_count = np.float32(1.0 / float(sample_count))

    # noinspection PyTypeChecker
    all_shifts: NDArray[np.int64] = rng.integers(low=floor, high=ceil, size=(shuffle_count, cell_count)).astype(
        np.int64
    )

    total_budget = resolve_worker_count(requested_workers=requested_workers)
    blas_threads_per_shuffle, parallel_shuffles = _resolve_ica_shuffle_allocation(
        budget=total_budget,
        shuffle_count=shuffle_count,
    )

    def _one_shuffle(shuffle_index: int) -> tuple[int, float]:
        # noinspection PyTypeChecker
        gather: NDArray[np.int64] = (
            sample_index_arange[np.newaxis, :] - all_shifts[shuffle_index][:, np.newaxis]
        ) % sample_count
        shuffled = z[cell_arange, gather]
        return shuffle_index, _largest_zzt_eigenvalue(z=shuffled, inverse_sample_count=inverse_sample_count)

    if parallel_shuffles <= 1:
        with threadpool_limits(limits=blas_threads_per_shuffle):
            for shuffle_index in tqdm(
                range(shuffle_count),
                desc=progress_description,
                unit="iter",
                leave=False,
            ):
                _, eigval = _one_shuffle(shuffle_index)
                output[shuffle_index] = eigval
        return output

    def _init_worker() -> None:
        threadpool_limits(limits=blas_threads_per_shuffle)

    with ThreadPoolExecutor(max_workers=parallel_shuffles, initializer=_init_worker) as executor:
        futures = [executor.submit(_one_shuffle, index) for index in range(shuffle_count)]
        for future in tqdm(
            as_completed(futures),
            total=shuffle_count,
            desc=progress_description,
            unit="iter",
            leave=False,
        ):
            shuffle_index, eigval = future.result()
            output[shuffle_index] = eigval
    return output


def _largest_zzt_eigenvalue(
    z: NDArray[np.float32],
    inverse_sample_count: float | np.float32,
) -> float:
    """Returns the largest eigenvalue of ``(z @ z.T) / sample_count`` without materialising the correlation
    matrix. Falls back to a full ``np.linalg.eigvalsh`` call on the rare ``ArpackNoConvergence`` so callers
    always receive a defined value.
    """
    cell_count, sample_count = z.shape
    if cell_count == 0 or sample_count == 0:
        return 0.0

    def matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        # noinspection PyTypeChecker
        projected: NDArray[np.float32] = z.T @ vector.astype(np.float32, copy=False)
        # noinspection PyTypeChecker
        result: NDArray[np.float32] = z @ projected
        return (result.astype(np.float64, copy=False)) * float(inverse_sample_count)

    operator = LinearOperator(shape=(cell_count, cell_count), matvec=matvec, dtype=np.float64)
    try:
        eigvals = eigsh(operator, k=1, which="LA", tol=1e-3, return_eigenvectors=False)
        return float(eigvals[0])
    except ArpackNoConvergence as failure:
        if failure.eigenvalues.size > 0:
            return float(np.max(failure.eigenvalues.real))
        correlation = (z @ z.T) * float(inverse_sample_count)
        return float(np.linalg.eigvalsh(correlation)[-1])


def _fast_ica_deflation(
    whitened: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    maximum_iterations: int = 200,
    tolerance: float = 1e-4,
) -> NDArray[np.float64]:
    """Runs deflation FastICA with the ``tanh`` non-linearity on a whitened ``(n_components, n_samples)``
    matrix and returns the unmixing matrix ``W`` with shape ``(n_components, n_components)``.
    """
    n_components, n_samples = whitened.shape
    # noinspection PyTypeChecker
    unmixing: NDArray[np.float64] = np.zeros((n_components, n_components), dtype=np.float64)
    for component_index in range(n_components):
        # noinspection PyTypeChecker
        candidate: NDArray[np.float64] = rng.standard_normal(n_components).astype(np.float64, copy=False)
        candidate /= np.linalg.norm(candidate) + 1e-12
        for previous in range(component_index):
            candidate -= float(candidate @ unmixing[previous]) * unmixing[previous]
        candidate /= np.linalg.norm(candidate) + 1e-12

        for _ in range(maximum_iterations):
            projection = candidate @ whitened
            g_value = np.tanh(projection)
            g_derivative = np.float64(1.0) - g_value * g_value
            updated = (whitened @ g_value) / float(n_samples) - g_derivative.mean() * candidate
            for previous in range(component_index):
                updated -= float(updated @ unmixing[previous]) * unmixing[previous]
            updated /= np.linalg.norm(updated) + 1e-12

            cos_similarity = float(np.abs(updated @ candidate))
            candidate = updated
            if abs(cos_similarity - 1.0) < tolerance:
                break

        unmixing[component_index] = candidate
    return unmixing


def _detect_assemblies_ica_cs(
    activity_matrix: NDArray[np.bool_] | NDArray[np.float32],
    *,
    shuffle_count: int = 200,
    eigenvalue_significance_percentile: float = 99.0,
    membership_z_threshold: float = 2.0,
    minimum_assembly_size: int = 3,
    minimum_shift_samples: int = 5,
    rng_seed: int = 0,
) -> tuple[NDArray[np.float32], list[NDArray[np.int32]]]:
    """Detects neural assemblies via the Lopes-dos-Santos 2013 ICA-CS pipeline.

    Notes:
        Z-scores activity along the sample axis, computes the cell-by-cell correlation matrix, retains
        principal components whose eigenvalues exceed the configured percentile of a circular-shift null
        distribution, whitens the data via these PCs, runs deflation FastICA, sign-corrects each independent
        component so its peak weight is positive, and reports cells whose weight magnitudes exceed the
        configured z-threshold as assembly members.
    """
    cell_count = int(activity_matrix.shape[0])
    rng = np.random.default_rng(seed=rng_seed)

    # noinspection PyTypeChecker
    activity: NDArray[np.float32] = np.asarray(activity_matrix, dtype=np.float32)
    if activity.shape[1] < _MINIMUM_OBSERVATIONS_FOR_VARIANCE:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    z, has_variance = _z_score_along_samples(activity=activity)
    active_cell_indices = np.where(has_variance)[0].astype(np.int32)
    if active_cell_indices.size < minimum_assembly_size:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    z_active = z[active_cell_indices, :]
    sample_count = z_active.shape[1]

    correlation = (z_active @ z_active.T) / float(sample_count)
    eigvals_all, eigvecs_all = np.linalg.eigh(correlation)

    shuffled_max = _shuffle_max_eigenvalue(
        z=z_active,
        shuffle_count=shuffle_count,
        rng=rng,
        minimum_shift_samples=minimum_shift_samples,
        progress_description="ICA-CS template shuffle",
    )
    threshold = float(np.percentile(shuffled_max, eigenvalue_significance_percentile))
    # noinspection PyTypeChecker
    significant_mask: NDArray[np.bool_] = eigvals_all > threshold
    if not significant_mask.any():
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []

    # noinspection PyTypeChecker
    significant_eigvals: NDArray[np.float64] = eigvals_all[significant_mask].astype(np.float64, copy=False)
    significant_eigvecs = eigvecs_all[:, significant_mask].astype(np.float64, copy=False)

    whitener = significant_eigvecs / np.sqrt(significant_eigvals)[np.newaxis, :]
    projected = whitener.T @ z_active.astype(np.float64, copy=False)

    unmixing = _fast_ica_deflation(whitened=projected, rng=rng)

    templates_active = (whitener @ unmixing.T).astype(np.float32, copy=False)

    for component_index in range(templates_active.shape[1]):
        peak_index = int(np.argmax(np.abs(templates_active[:, component_index])))
        if templates_active[peak_index, component_index] < 0.0:
            templates_active[:, component_index] *= np.float32(-1.0)

    # noinspection PyTypeChecker
    templates: NDArray[np.float32] = np.zeros((templates_active.shape[1], cell_count), dtype=np.float32)
    templates[:, active_cell_indices] = templates_active.T

    member_lists: list[NDArray[np.int32]] = []
    surviving: list[NDArray[np.float32]] = []
    for component_index in range(templates.shape[0]):
        active_weights = templates[component_index, active_cell_indices]
        weight_std = float(np.std(active_weights))
        if weight_std == 0.0:
            continue
        # noinspection PyTypeChecker
        member_mask: NDArray[np.bool_] = templates[component_index] > membership_z_threshold * weight_std
        # noinspection PyTypeChecker
        member_indices: NDArray[np.int32] = np.where(member_mask)[0].astype(np.int32)
        if member_indices.size < minimum_assembly_size:
            continue
        member_lists.append(member_indices)
        surviving.append(templates[component_index])

    if not surviving:
        # noinspection PyTypeChecker
        return np.empty((0, cell_count), dtype=np.float32), []
    # noinspection PyTypeChecker
    surviving_templates: NDArray[np.float32] = np.stack(surviving, axis=0).astype(np.float32, copy=False)
    return surviving_templates, member_lists
