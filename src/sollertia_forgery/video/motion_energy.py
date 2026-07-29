"""Provides the system-agnostic per-camera motion-energy analysis run by the video-processing pipeline."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count

from ..shared_assets import pinned_worker_threads

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

MOTION_ENERGY_SUFFIX: str = "_energy.feather"
"""The filename suffix appended to a camera's manifest name to form its motion-energy feather (for example, the
``left_camera`` source produces ``left_camera_energy.feather``)."""

_VIDEO_SUFFIX: str = ".mp4"
"""The container suffix of the camera recordings this analysis reads. Every VideoSystem writes its recordings into the
session's raw camera-data directory under this suffix."""

_SPATIAL_BIN_SIZE: int = 3
"""The edge length, in pixels, of the square block each frame is mean-binned over before differencing. A small block
averages out the single-pixel sensor and codec noise that the later absolute difference would otherwise rectify into a
positive bias, while staying small enough to leave the movement the measure captures intact. An odd edge keeps the box
filter's anchor on the pixel at each block's center, which the strided sampling that reads the block means relies on."""

_MINIMUM_CHUNK_FRAMES: int = 4000
"""The smallest frame count a parallel decode chunk is allowed to cover. Seeking into a chunk decodes from the
preceding keyframe, so each chunk discards up to one group of frames worth of decoded frames. At roughly sixteen
times the typical keyframe interval, that waste stays under seven percent of the chunk."""

_SINGLE_PLANE_DIMENSIONS: int = 2
"""The dimension count that identifies a decoded frame as a single grayscale plane. A frame matching it is used as-is,
and a frame with more dimensions has one of its planes extracted instead."""

_MONOCHROME_PLANE_INDEX: int = 1
"""The channel taken from a multi-plane frame, the single plane the motion-energy analysis runs on. A monochrome
source's channels are identical, so any one carries the image."""


class MotionEnergyColumn(StrEnum):
    """Every column written into a camera's motion-energy feather by the video-processing pipeline.

    Notes:
        Values are raw, un-normalized gray levels, so motion energy is a within-session signal. The feather is a
        positional table, one row per decoded frame in acquisition order, matching the timestamp feather files.
    """

    MOTION_ENERGY = "motion_energy"
    """Mean absolute intensity difference in gray levels between consecutive frames, over the whole 3x3-binned frame.
    The behavioral movement magnitude, high during running, low during idleness. NaN in the first row, which has no
    predecessor, and nowhere else."""
    FRAME_LUMINANCE = "frame_luminance"
    """Mean intensity of the same binned frame, in gray levels. It separates a scene-illumination change from a behavior
    change, since a whole-field brightness shift inflates ``motion_energy`` without anything moving. Use its slow
    session-scale drift to detrend energy across a recording. Defined at every frame."""


def resolve_camera_video(camera_data_directory: Path, session_name: str, camera_name: str) -> Path | None:
    """Resolves the recording a camera produced within the session's raw camera-data directory.

    Every VideoSystem names its recording ``{session_name}_{camera_name}.mp4``, so the recording is resolved by
    reconstructing that exact name rather than by pattern-matching the camera name against the directory. Matching on
    a suffix would be ambiguous: a camera named ``camera`` would match a ``left_camera`` recording, since that name
    also ends in ``_camera``.

    Args:
        camera_data_directory: The session's raw camera-data directory (``session.raw_data.camera_data_path``), which
            holds the recordings themselves.
        session_name: The name of the session the recording belongs to, which prefixes every recording filename.
        camera_name: The colloquial camera name recorded in the acquisition-time camera manifest.

    Returns:
        The path to the camera's recording, or None if the directory or the recording is absent, which means there is
        nothing for this camera to analyze.
    """
    if not camera_data_directory.is_dir():
        return None

    video_path = camera_data_directory.joinpath(f"{session_name}_{camera_name}{_VIDEO_SUFFIX}")
    return video_path if video_path.is_file() else None


def compute_camera_motion_energy(
    video_path: Path,
    output_path: Path,
    *,
    workers: int = -1,
    executor: ProcessPoolExecutor | None = None,
    display_progress: bool = False,
) -> None:
    """Computes the per-frame motion energy of a camera recording and writes it as a feather.

    Decodes the recording, mean-bins each frame over 3x3 pixel blocks, and reduces every consecutive pair to the mean
    absolute intensity difference between them. The recording is split into contiguous frame chunks decoded in
    parallel. Each chunk beyond the first decodes one extra priming frame so that the difference spanning its leading
    boundary is computed rather than lost, which makes the chunked result bit-identical to a sequential pass over the
    frames that decode. The chunk count is a pure performance knob for an intact recording. On a recording whose
    container over-reports its frame count by more than the final chunk's size, a higher chunk count turns a warning
    into a truncation error.

    Notes:
        "Motion energy" here means frame-differencing motion energy, the mean absolute inter-frame intensity change.
        Nothing filters for direction or speed, so the measure is undirected and unsigned. Binning before differencing
        is load-bearing: the absolute difference rectifies per-pixel sensor and codec noise into a positive bias, so
        binning afterward would not suppress it.

        The bin-then-difference extraction is modeled on Facemap (Syeda et al., 2024, Nature Neuroscience, 27(1),
        187-195) with numerous local efficiency enhancements.

    Args:
        video_path: The path to the camera recording to analyze.
        output_path: The path of the motion-energy feather to write.
        workers: The number of worker processes to decode with. Set to -1 to use all available CPU cores (minus
            reserved cores).
        executor: An optional process pool to submit the decode chunks into, shared across cameras so the cost of
            spawning worker processes is paid once. When None, a pool is created and torn down for this recording,
            unless the recording plans a single decode chunk, which runs in-process with no pool.
        display_progress: Determines whether per-chunk completion is reported as the analysis runs.

    Raises:
        ValueError: If the recording cannot be opened, reports no frames, cannot decode the priming frame preceding a
            chunk, or ends early at a chunk other than the last. Such an early end means the file is truncated and
            every later frame index would be wrong.
    """
    frame_count = _read_frame_count(video_path=video_path)
    resolved_workers = resolve_worker_count(requested_workers=workers)
    chunks = _plan_chunks(frame_count=frame_count, workers=resolved_workers)

    # A pool is only worth its startup cost with more than one chunk to decode. A single chunk runs in-process.
    if len(chunks) == 1:
        results = [_energy_chunk(video_path=str(video_path), start_frame=0, frame_count=frame_count)]
    elif executor is not None:
        results = _submit_chunks(executor=executor, video_path=video_path, chunks=chunks, display=display_progress)
    else:
        # Caps the worker threading layers before the pool starts its children, so each of them costs the single
        # core it was budgeted. A shared pool is instead capped by whoever created it, since its children may
        # already exist by the time this runs.
        with (
            pinned_worker_threads(),
            ProcessPoolExecutor(max_workers=min(resolved_workers, len(chunks))) as own_executor,
        ):
            results = _submit_chunks(
                executor=own_executor, video_path=video_path, chunks=chunks, display=display_progress
            )

    energy, luminance = _join_chunks(results=results, chunks=chunks, video_path=video_path)

    # A positional table: one row per decoded frame in acquisition order, row position serving as the frame index, so
    # no explicit index column is stored. This is the same index-free positional convention the camera's timestamp and
    # other per-frame feathers follow.
    pl.DataFrame(
        {
            MotionEnergyColumn.MOTION_ENERGY: energy,
            MotionEnergyColumn.FRAME_LUMINANCE: luminance,
        }
    ).write_ipc(file=output_path, compression="uncompressed")

    console.echo(
        message=f"Wrote motion energy for {energy.size} frame(s) to '{output_path.name}'.",
        level=LogLevel.SUCCESS,
    )


def _read_frame_count(video_path: Path) -> int:
    """Reads the frame count a recording reports in its container metadata.

    Args:
        video_path: The path to the camera recording.

    Returns:
        The reported frame count.

    Raises:
        ValueError: If the recording cannot be opened or reports no frames.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            message = (
                f"Unable to compute motion energy for '{video_path.name}'. The recording could not be opened for "
                f"decoding, which usually means the file is truncated or its codec is unavailable."
            )
            console.error(message=message, error=ValueError)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if frame_count <= 0:
        message = (
            f"Unable to compute motion energy for '{video_path.name}'. The recording reports {frame_count} frames."
        )
        console.error(message=message, error=ValueError)

    return frame_count


def _plan_chunks(frame_count: int, workers: int) -> list[tuple[int, int]]:
    """Splits a recording into contiguous decode chunks that tile it exactly.

    Args:
        frame_count: The number of frames in the recording.
        workers: The resolved number of worker processes available to decode with.

    Returns:
        A list of ``(start_frame, frame_count)`` pairs covering the recording with no gap and no overlap.
    """
    chunk_count = max(1, min(workers, frame_count // _MINIMUM_CHUNK_FRAMES))
    base, remainder = divmod(frame_count, chunk_count)

    chunks: list[tuple[int, int]] = []
    start = 0
    for index in range(chunk_count):
        # Spreads the remainder across the leading chunks so every chunk differs in size by at most one frame.
        size = base + 1 if index < remainder else base
        chunks.append((start, size))
        start += size
    return chunks


def _submit_chunks(
    executor: ProcessPoolExecutor,
    video_path: Path,
    chunks: list[tuple[int, int]],
    *,
    display: bool,
) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """Submits every decode chunk into a process pool and collects the results in chunk order.

    Args:
        executor: The process pool to submit the chunks into.
        video_path: The path to the camera recording.
        chunks: The planned ``(start_frame, frame_count)`` chunks.
        display: Determines whether per-chunk completion is reported.

    Returns:
        The per-chunk ``(energy, luminance)`` arrays, ordered to match the input chunks.
    """
    futures = [
        executor.submit(_energy_chunk, video_path=str(video_path), start_frame=start_frame, frame_count=frame_count)
        for start_frame, frame_count in chunks
    ]

    progress_context = (
        console.progress(total=len(futures), description="Decoding motion-energy chunks", unit="chunk")
        if display
        else nullcontext()
    )

    results: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    with progress_context as progress_bar:
        for future in futures:
            results.append(future.result())
            if progress_bar is not None:
                progress_bar.update(1)
    return results


def _join_chunks(
    results: list[tuple[NDArray[np.float32], NDArray[np.float32]]],
    chunks: list[tuple[int, int]],
    video_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Concatenates the per-chunk results, verifying that only the final chunk may have ended early.

    A container's reported frame count is an estimate, so the last chunk legitimately runs out early. Any earlier
    chunk running out means the recording is truncated partway, which would shift every later frame index.

    Args:
        results: The per-chunk ``(energy, luminance)`` arrays, in chunk order.
        chunks: The planned ``(start_frame, frame_count)`` chunks, in the same order.
        video_path: The path to the camera recording, used only for error messages.

    Returns:
        The concatenated ``(energy, luminance)`` arrays covering the recording.

    Raises:
        ValueError: If a chunk other than the last ended early.
    """
    for index, ((energy, _), (_, planned)) in enumerate(zip(results, chunks, strict=True)):
        if energy.size != planned and index != len(chunks) - 1:
            message = (
                f"Unable to compute motion energy for '{video_path.name}'. Decode chunk {index + 1} of "
                f"{len(chunks)} ended after {energy.size} of its {planned} frames, which means the recording is "
                f"truncated and every later frame index would be misaligned."
            )
            console.error(message=message, error=ValueError)

    energy = np.concatenate([chunk_energy for chunk_energy, _ in results])
    luminance = np.concatenate([chunk_luminance for _, chunk_luminance in results])

    # A short final chunk is benign but must be announced: a consumer joining on frame index needs to know the feather
    # covers fewer frames than the recording claimed.
    planned_total = sum(planned for _, planned in chunks)
    if energy.size != planned_total:
        console.echo(
            message=(
                f"The recording '{video_path.name}' yielded {energy.size} frames against the {planned_total} its "
                f"container reported. Motion energy covers the frames that decoded."
            ),
            level=LogLevel.WARNING,
        )

    return energy, luminance


def _energy_chunk(
    video_path: str, start_frame: int, frame_count: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the motion energy and luminance of one contiguous chunk of a recording.

    Runs inside a pool worker when more than one chunk is planned, and in-process otherwise, so it takes only
    picklable primitives. Chunks beyond the first seek one frame early and decode a priming frame, whose own energy is
    discarded, so that the difference spanning the chunk's leading boundary is computed exactly as a sequential pass
    would compute it.

    Args:
        video_path: The path to the camera recording.
        start_frame: The zero-based index of the chunk's first frame.
        frame_count: The number of frames the chunk covers.

    Returns:
        A ``(energy, luminance)`` tuple of per-frame arrays. Both are truncated to the frames that actually decoded,
        which the caller checks against the chunk's plan.

    Raises:
        ValueError: If the recording cannot be opened, or if the priming frame preceding a chunk cannot be decoded.
    """
    # Pins every layer of threading to one thread per worker. The decoder honors this environment variable when the
    # capture is constructed rather than at import, and it is the knob that reaches the decoder without rebuilding the
    # capture. The alternative cv2.CAP_PROP_N_THREADS also does, but only through the VideoCapture constructor's params
    # argument, and the OpenCV thread count governs its own kernels instead. Left unpinned, each of the many workers
    # spawns its own decode threads and the pool oversubscribes the machine several times over.
    os.environ["OPENCV_FFMPEG_THREADS"] = "1"
    cv2.setNumThreads(1)
    # The recordings are encoded as yuv420p, which the decoder does not recognize and reports as an unsupported
    # picture format on every single frame before handing back its luma plane. Silenced here rather than per frame,
    # since a chunk decodes many thousands.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

    energy = np.full(frame_count, fill_value=np.nan, dtype=np.float32)
    luminance = np.full(frame_count, fill_value=np.nan, dtype=np.float32)

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            message = f"Unable to open '{video_path}' to decode motion-energy frames {start_frame} onward."
            console.error(message=message, error=ValueError)

        # The decoder hands back its planes untouched, which for this monochrome source is the single plane the
        # measurement needs, at no conversion cost.
        capture.set(propId=cv2.CAP_PROP_CONVERT_RGB, value=0)

        previous = None
        if start_frame > 0:
            capture.set(propId=cv2.CAP_PROP_POS_FRAMES, value=start_frame - 1)
            decoded, frame = capture.read()
            if not decoded:
                message = (
                    f"Unable to decode the frame preceding motion-energy chunk starting at frame {start_frame} of "
                    f"'{video_path}'."
                )
                console.error(message=message, error=ValueError)
            previous = _bin_frame(frame=frame)  # type: ignore[arg-type]

        for index in range(frame_count):
            decoded, frame = capture.read()
            if not decoded:
                # Ends the chunk where the recording ended. The caller decides whether that is benign.
                return energy[:index], luminance[:index]

            binned = _bin_frame(frame=frame)  # type: ignore[arg-type]
            luminance[index] = cv2.mean(src=binned)[0]
            if previous is not None:
                # OpenCV forms the differences in single precision and accumulates them in double. For the similar
                # consecutive frames this loop sees, the subtraction is exact, so this reproduces a float64 numpy
                # mean of the absolute difference while running through the vectorized kernel.
                energy[index] = cv2.norm(src1=binned, src2=previous, normType=cv2.NORM_L1) / binned.size
            previous = binned
    finally:
        capture.release()

    return energy, luminance


def _bin_frame(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Reduces a decoded frame to its 3x3 block means.

    Uses a normalized box filter sampled on a stride, which is an exact block mean for any frame size.

    Args:
        frame: The decoded frame, either a single luma plane or a BGR expansion of one.

    Returns:
        The block-mean frame, as single-precision gray levels.
    """
    # The decoder yields one plane for monochrome sources. When it falls back to a BGR expansion, the three channels
    # carry the same content and the second is taken. Rows and columns past the last whole block are dropped, since a
    # partial block would average fewer pixels and carry different noise statistics than every other block.
    gray = frame if frame.ndim == _SINGLE_PLANE_DIMENSIONS else frame[:, :, _MONOCHROME_PLANE_INDEX]
    height, width = gray.shape
    bin_height = height // _SPATIAL_BIN_SIZE * _SPATIAL_BIN_SIZE
    bin_width = width // _SPATIAL_BIN_SIZE * _SPATIAL_BIN_SIZE

    # The filter's anchor is the block center, so sampling from index 1 on a stride of 3 reads exactly the block
    # means, and every sampled neighborhood is fully interior so border handling never applies. The depth is fixed
    # to single precision by the CV_32F argument, which the OpenCV stubs do not express in their return type.
    binned: NDArray[np.float32] = cv2.boxFilter(  # type: ignore[assignment]
        src=gray, ddepth=cv2.CV_32F, ksize=(_SPATIAL_BIN_SIZE, _SPATIAL_BIN_SIZE), normalize=True
    )[1:bin_height:_SPATIAL_BIN_SIZE, 1:bin_width:_SPATIAL_BIN_SIZE]
    return binned
