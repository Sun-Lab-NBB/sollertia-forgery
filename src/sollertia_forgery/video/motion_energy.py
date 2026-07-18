"""Provides the system-agnostic per-camera motion-energy analysis run by the video-processing pipeline.

The analysis reduces each camera recording to a per-frame scalar measuring how much the image changed since the
previous frame, which indexes how much the animal moved. It is a pure function of pixels and carries no
acquisition-system-specific content, so it lives here rather than being donated per system: the camera manifest
already names every camera, and the same computation applies to all of them.

Notes:
    "Motion energy" here means frame-differencing motion energy -- the mean absolute inter-frame intensity change --
    following Stringer et al. 2019, Musall et al. 2019, and the Facemap reference implementation. It is NOT
    Adelson-Bergen spatiotemporal energy, an unrelated oriented-filter model of motion perception that shares the
    name. Nothing here filters for direction or speed; the measure is undirected and unsigned.

    Each frame is mean-binned over 3x3 pixel blocks before differencing. Binning first is load-bearing: the absolute
    difference rectifies per-pixel sensor and codec noise into a positive bias, so binning afterwards would not
    suppress it.

    The recordings are lossy, constant-quantization video, which bounds what the quiet end of the range can resolve.
    The encoder's deadzone codes sub-threshold motion as no change at all, and its zero-residual regions are
    block-structured at a scale far larger than the 3x3 bin, so binning averages pixels that were coded zero together
    rather than decorrelating them. Idle-period energy therefore floors and compresses non-linearly. A periodic
    component at the encoder's keyframe interval is also possible, since an intra-coded frame does not share its
    predecessor's quantization error; check the autocorrelation at that lag before trusting slow structure.

    A second video-derived analysis should be added as a sibling module in this package following this module's
    shape. The chunked-decode scaffolding here is deliberately not shared: a common decode engine belongs in its own
    module only once a third analysis needs one, at which point three working implementations will show what
    actually varies between them.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

MOTION_ENERGY_SUFFIX: str = "_energy.feather"
"""The filename suffix appended to a camera's manifest name to form its motion-energy feather (for example, the
``face_camera`` source produces ``face_camera_energy.feather``). Declared once here so the pipeline that writes the
artifact and any consumer that later reads it build the filename from a single definition."""

VIDEO_SUFFIX: str = ".mp4"
"""The container suffix of the camera recordings this analysis reads. Every VideoSystem writes its recordings into the
session's raw camera-data directory under this suffix."""

SPATIAL_BIN_SIZE: int = 3
"""The edge length, in pixels, of the square block each frame is mean-binned over before differencing. Matches the
Facemap ``sbin`` default. Binning before differencing suppresses per-pixel sensor and codec noise; binning after would
not, because the absolute difference rectifies that noise into a positive bias first."""

MINIMUM_CHUNK_FRAMES: int = 4000
"""The smallest frame count a parallel decode chunk is allowed to cover. Seeking into a chunk decodes from the
preceding keyframe, so each chunk discards up to one group of pictures worth of decoded frames; at roughly sixteen
times the typical keyframe interval, that waste stays under seven percent of the chunk."""

_SINGLE_PLANE_DIMENSIONS: int = 2
"""The dimension count of a decoded frame the decoder handed back as a single image plane, which needs no plane
selection."""

_MONOCHROME_PLANE_INDEX: int = 1
"""The index of the plane read from a frame the decoder handed back as several planes. Monochrome sources are stored
across identical planes, so any one of them carries the image; this one is read for all frames so the choice never
varies within a recording."""


class MotionEnergyColumn(StrEnum):
    """Defines every column written into a camera's motion-energy feather by the video-processing pipeline.

    Notes:
        Values are raw gray levels and are deliberately left un-normalized. Their scale depends on sensor gain,
        exposure, illuminator intensity, focus, and the encoder's quantization parameter, so motion energy is a
        within-session signal that is not comparable across sessions or rigs without normalization. Choosing that
        normalization (z-score for regression, a percentile or absolute cut for frame rejection, raw for quality
        control) belongs to whoever owns the comparison, exactly as the analysis package consumes pre-normalized
        fluorescence rather than re-baselining it. The signal is also left unsmoothed, because its value is precisely
        that it resolves movement faster than pupil diameter does.

        Values are computed over the whole frame, with no region of interest. For rejecting frames that contain
        movement this is the sensitive choice rather than a compromise: any movement anywhere in the field registers.
        The same property makes the measure unsuitable as a graded speed estimate on a camera that images the running
        wheel, where it is partly an optical tachometer and saturates once the frame decorrelates.

        The feather is strictly a frame index and the metrics keyed to it. It carries no timestamps: aligning frames
        to the acquisition clock belongs to dataset assembly, which owns every other stream's alignment too. Three
        obligations follow for any consumer that joins this to a time base. It must verify this feather's row count
        equals the camera's timestamp feather's before joining, since nothing upstream enforces that. It must
        subtract one from ``frame`` to reach the timestamp feather's zero-based positional rows. And because motion
        energy is a per-interval rather than a per-second quantity, it must either divide each sample by its actual
        inter-frame interval or mask samples whose interval departs from the modal one -- a difference taken across a
        dropped frame spans more real time than its neighbours and reads as spuriously high motion.
    """

    FRAME = "frame"
    """One-based camera frame index, numbered to match the frame identifiers in the forged session dataset and the
    ``frame`` column of this camera's tracking feathers. The feather's only key: every other column is a per-frame
    metric. Note that this camera's timestamp feather is zero-based positional instead, so joining the two requires
    subtracting one."""
    MOTION_ENERGY = "motion_energy"
    """Mean absolute intensity difference, in gray levels, between this frame and the one before it, taken over the
    whole frame after 3x3 mean binning. The behavioral movement magnitude and the fast complement to the pupil's slow
    arousal signal: high during running and stereotyped behavior, low during idleness. NaN at frame 1, which has no
    predecessor, and nowhere else. See the class notes on why it is neither normalized nor interval-corrected here."""
    FRAME_LUMINANCE = "frame_luminance"
    """Mean intensity of this frame in gray levels, taken over the same binned frame. Not a behavioral signal: it is
    the illumination-artifact diagnostic for ``motion_energy``. A whole-field brightness step -- a task display
    changing, an infrared illuminator switching, room lights -- lands on every pixel at once and produces a large
    spurious energy transient, and because such steps are typically time-locked to trial and cue events the artifact
    correlates with the experimental design rather than averaging out. The absolute difference of this column between
    consecutive frames flags exactly those frames, and can be regressed out. Defined at every frame, including
    frame 1."""


def resolve_camera_video(camera_data_directory: Path, session_name: str, camera_name: str) -> Path | None:
    """Resolves the recording a camera produced within the session's raw camera-data directory.

    Every VideoSystem names its recording ``{session_name}_{camera_name}.mp4``, so the recording is resolved by
    reconstructing that exact name rather than by pattern-matching the camera name against the directory. Matching on
    a suffix would be ambiguous: a camera named ``camera`` would match a ``body_camera`` recording, since that name
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

    video_path = camera_data_directory.joinpath(f"{session_name}_{camera_name}{VIDEO_SUFFIX}")
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
    parallel; each chunk beyond the first decodes one extra priming frame so that the difference spanning its leading
    boundary is computed rather than lost, which makes the chunked result bit-identical to a sequential pass and the
    chunk count a pure performance knob.

    Args:
        video_path: The path to the camera recording to analyze.
        output_path: The path of the motion-energy feather to write.
        workers: The number of worker processes to decode with. Set to -1 to use all available CPU cores (minus
            reserved cores). Resolved here, so an unresolved count may be passed in.
        executor: An optional process pool to submit the decode chunks into, shared across cameras so the cost of
            spawning worker processes is paid once. When None, a pool is created and torn down for this recording.
        display_progress: Determines whether per-chunk completion is reported as the analysis runs.

    Raises:
        ValueError: If the recording cannot be opened, reports no frames, or ends early at a chunk other than the
            last, which means the file is truncated and every later frame index would be wrong.
    """
    frame_count = _read_frame_count(video_path=video_path)
    resolved_workers = resolve_worker_count(requested_workers=workers)
    chunks = _plan_chunks(frame_count=frame_count, workers=resolved_workers)

    console.echo(
        message=(
            f"Computing motion energy for {frame_count} frame(s) of '{video_path.name}' across {len(chunks)} "
            f"decode chunk(s)..."
        ),
        level=LogLevel.INFO,
    )

    # A pool is only worth its startup cost with more than one chunk to decode; a single chunk runs in-process.
    if len(chunks) == 1:
        results = [_energy_chunk(video_path=str(video_path), start_frame=0, frame_count=frame_count)]
    elif executor is not None:
        results = _submit_chunks(executor=executor, video_path=video_path, chunks=chunks, display=display_progress)
    else:
        with ProcessPoolExecutor(max_workers=min(resolved_workers, len(chunks))) as own_executor:
            results = _submit_chunks(
                executor=own_executor, video_path=video_path, chunks=chunks, display=display_progress
            )

    energy, luminance = _join_chunks(results=results, chunks=chunks, video_path=video_path)

    # Numbers frames from 1 to match the one-based frame identifiers the forging pipeline assigns in 'data.feather',
    # so the two can be joined without an off-by-one correction.
    frame = pl.Series(name=MotionEnergyColumn.FRAME, values=np.arange(1, energy.size + 1, dtype=np.uint32))
    pl.DataFrame(
        {
            MotionEnergyColumn.FRAME: frame,
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
    chunk_count = max(1, min(workers, frame_count // MINIMUM_CHUNK_FRAMES))
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
        executor.submit(_energy_chunk, str(video_path), start_frame, frame_count) for start_frame, frame_count in chunks
    ]

    results: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    for index, future in enumerate(futures):
        results.append(future.result())
        if display:
            console.echo(message=f"Decoded motion-energy chunk {index + 1} of {len(futures)}.")
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

    # The container's frame count is an estimate, so a short final chunk is benign but must be announced: the feather
    # then covers fewer frames than the recording claimed, and a consumer joining on frame index needs to know.
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

    Runs inside a pool worker, so it takes only picklable primitives and imports nothing from this package. Chunks
    beyond the first seek one frame early and decode a priming frame, whose own energy is discarded, so that the
    difference spanning the chunk's leading boundary is computed exactly as a sequential pass would compute it.

    Args:
        video_path: The path to the camera recording, as a string.
        start_frame: The zero-based index of the chunk's first frame.
        frame_count: The number of frames the chunk covers.

    Returns:
        A ``(energy, luminance)`` tuple of per-frame arrays. Both are truncated to the frames that actually decoded,
        which the caller checks against the chunk's plan.

    Raises:
        ValueError: If the recording cannot be opened, or if the priming frame preceding a chunk cannot be decoded.
    """
    # Pins every layer of threading to one thread per worker. The decoder honors this environment variable when the
    # capture is constructed rather than at import, and it is the only knob that reaches the decoder: the OpenCV
    # thread count governs its own kernels instead. Left unpinned, each of the many workers spawns its own decode
    # threads and the pool oversubscribes the machine several times over.
    os.environ["OPENCV_FFMPEG_THREADS"] = "1"
    cv2.setNumThreads(1)
    # The recordings store monochrome content across three planes, which the decoder reports as an unsupported
    # format on every single frame. Silenced here rather than per frame, since a chunk decodes many thousands.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

    energy = np.full(frame_count, np.nan, dtype=np.float32)
    luminance = np.full(frame_count, np.nan, dtype=np.float32)

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            message = f"Unable to open '{video_path}' to decode motion-energy frames {start_frame} onward."
            raise ValueError(message)

        # Hands back the decoded planes untouched instead of interleaving them into a color image. For this
        # monochrome source that yields the single plane the measurement needs, at no conversion cost.
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        previous = None
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
            decoded, frame = capture.read()
            if not decoded:
                message = (
                    f"Unable to decode the frame preceding motion-energy chunk starting at frame {start_frame} of "
                    f"'{video_path}'."
                )
                raise ValueError(message)
            previous = _bin_frame(frame=frame)  # type: ignore[arg-type]

        for index in range(frame_count):
            decoded, frame = capture.read()
            if not decoded:
                # Ends the chunk where the recording ended. The caller decides whether that is benign.
                return energy[:index], luminance[:index]

            binned = _bin_frame(frame=frame)  # type: ignore[arg-type]
            luminance[index] = cv2.mean(binned)[0]
            if previous is not None:
                # OpenCV accumulates the absolute differences in double internally, so this matches a numpy mean of
                # the absolute difference exactly while running through its vectorized kernel.
                energy[index] = cv2.norm(binned, previous, cv2.NORM_L1) / binned.size
            previous = binned
    finally:
        capture.release()

    return energy, luminance


def _bin_frame(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Reduces a decoded frame to its 3x3 block means.

    Uses a normalized box filter sampled on a stride, which is an exact block mean for any frame size. Resizing with
    pixel-area interpolation is not a safe substitute: it only coincides with a block mean when both dimensions
    divide evenly by the block size, and silently blends across block boundaries when they do not.

    Args:
        frame: The decoded frame, either a single plane or several identical ones.

    Returns:
        The block-mean frame, as single-precision gray levels.
    """
    # The decoder yields one plane for monochrome sources; when it falls back to reporting three, they carry the same
    # content and the second is taken. Rows and columns past the last whole block are dropped, since a partial block
    # would average fewer pixels and carry different noise statistics than every other block.
    gray = frame if frame.ndim == _SINGLE_PLANE_DIMENSIONS else frame[:, :, _MONOCHROME_PLANE_INDEX]
    height, width = gray.shape
    bin_height = height // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE
    bin_width = width // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE

    # The filter's anchor is the block center, so sampling from index 1 on a stride of 3 reads exactly the block
    # means, and every sampled neighbourhood is fully interior so border handling never applies. The depth is fixed
    # to single precision by the CV_32F argument, which the OpenCV stubs do not express in their return type.
    binned: NDArray[np.float32] = cv2.boxFilter(  # type: ignore[assignment]
        gray, cv2.CV_32F, (SPATIAL_BIN_SIZE, SPATIAL_BIN_SIZE), normalize=True
    )[1:bin_height:SPATIAL_BIN_SIZE, 1:bin_width:SPATIAL_BIN_SIZE]
    return binned
