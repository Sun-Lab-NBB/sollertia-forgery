"""Provides the system-agnostic per-camera motion-energy analysis run by the video-processing pipeline.

The analysis reduces each camera recording to a per-frame scalar indexing how much the animal moved, and as a pure
function of pixels it applies to every camera the manifest names rather than being donated per acquisition system.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING
from contextlib import nullcontext, contextmanager
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

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
``sbin`` value Facemap's internal SVD and ROI helpers are written around. Facemap's own user-facing defaults are 1 in
``process.run`` and 7 in its GUI."""

MINIMUM_CHUNK_FRAMES: int = 4000
"""The smallest frame count a parallel decode chunk is allowed to cover. Seeking into a chunk decodes from the
preceding keyframe, so each chunk discards up to one group of pictures worth of decoded frames. At roughly sixteen
times the typical keyframe interval, that waste stays under seven percent of the chunk."""

WORKER_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
)
"""The environment variables that cap each threading layer a worker process can start a pool for. Most of these
libraries size their pool to the machine's core count on first import, so a worker that inherits the defaults reserves
far more of the machine than the one core it was budgeted. ``OPENCV_FFMPEG_THREADS`` is the exception: the decoder
reads it when a capture is constructed rather than at import, so a worker re-sets it for itself."""

_SINGLE_PLANE_DIMENSIONS: int = 2
"""The dimension count of a decoded frame the decoder handed back as a single image plane, which needs no plane
selection."""

_MONOCHROME_PLANE_INDEX: int = 1
"""The index of the plane read from a frame the decoder handed back as several planes. When the decoder cannot hand
back the raw luma plane it falls back to a BGR expansion whose three channels are identical for a monochrome source, so
any one of them carries the image. This one is read for all frames, so the choice never varies within a recording."""


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

        Whole-frame coverage also means that a camera imaging the eye folds pupil motion and blinks into its movement
        signal, and those are not facial movement. In Talluri et al., units that looked movement-modulated during free
        viewing (67 percent) mostly stopped looking so once the animal held fixation and the retinal input was
        stabilized (5 percent). Blinks co-occur with volitional whisking in about 40 percent of awake blink events
        (Turner et al.), so the leak is not redundant with the body-movement signal. This column remains the right
        primary product for movement-frame rejection, where an eye event is a frame worth rejecting anyway. But an
        analysis that claims to measure FACIAL movement from an eye-bearing camera must exclude the eye region and say
        so, and must not treat this column as independent of the pupil feather extracted from the same recording.

        The absolute difference is a full-wave rectifier, so a symmetric movement that oscillates appears in this
        signal at TWICE its true frequency: an eight-hertz rhythm reads as a sixteen-hertz peak. An asymmetric rhythm,
        such as one whose protraction and retraction velocities differ, retains power at its own frequency as well. A
        frequency-domain analysis of this column must therefore check for a harmonic pair before naming a behavioral
        rhythm. Amplitude-domain uses (thresholding, frame rejection, regression against a slow signal) are
        unaffected.

        The recordings are lossy, constant-quantization video, which bounds what the quiet end of the range can
        resolve. The encoder's deadzone codes sub-threshold motion as no change at all, and its zero-residual regions
        are block-structured at a scale far larger than the 3x3 bin, so binning averages pixels that were coded zero
        together rather than decorrelating them. Idle-period energy therefore floors and compresses non-linearly. A
        periodic component at the encoder's keyframe interval is also possible, since an intra-coded frame does not
        share its predecessor's quantization error. Check the autocorrelation at that lag before trusting slow
        structure.

        The feather is strictly a frame index and the metrics keyed to it. It carries no timestamps: aligning frames
        to the acquisition clock belongs to dataset assembly, which owns every other stream's alignment too. Three
        obligations follow for any consumer that joins this to a time base. It must verify this feather's row count
        equals the camera's timestamp feather's before joining, since nothing upstream enforces that. It must
        subtract one from ``frame`` to reach the timestamp feather's zero-based positional rows. And because motion
        energy is a per-interval rather than a per-second quantity, it must either divide each sample by its actual
        inter-frame interval or mask samples whose interval departs from the modal one. A difference taken across a
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
    """Mean intensity of this frame in gray levels, taken over the same binned frame. Its purpose is to separate a
    change in scene illumination from a change in the animal's posture, since a whole-field brightness shift lands on
    every pixel at once and inflates ``motion_energy`` without anything having moved. Defined at every frame,
    including frame 1.

    Read it at the session timescale, not per frame. On a head-fixed face recording its frame-to-frame changes track
    movement rather than illumination. They correlate with ``motion_energy`` at about 0.7, and that correlation
    survives almost undiminished when the largest brightness jumps are excluded, because a moving animal displaces
    bright structure within the frame and drags the mean along with it. Large single-frame excursions were checked
    for and behave like movement rather than like switching: they are embedded in sustained activity rather than
    isolated, and most revert instead of holding a new level. Treating a per-frame jump in this column as evidence of
    an illumination artifact will therefore reject real movement.

    What does survive as illumination is slow. After the movement-explained part is regressed out, a drift of a few
    gray levels remains across a session (around an order of magnitude larger than the energy floor) and it is
    not attributable to the animal. That drift is what this column is for. Use it to detrend, or to include a slow
    luminance regressor, when comparing energy across the length of a session. It also confirms that a change in
    energy level between the start and end of a recording is behavioral rather than a lamp warming up.

    These figures come from one face recording on one rig. The balance between the two regimes depends on how much
    of the frame the animal fills and on what else in the scene emits or reflects light. Re-measure both the
    correlation and the residual drift before relying on either on a different camera."""


@contextmanager
def pinned_worker_threads() -> Iterator[None]:
    """Caps every threading layer to a single thread for the duration of the block, then restores the environment.

    Worker processes are started with the environment they inherit at spawn time. Each scientific library sizes its
    thread pool when it is first imported, which, because a worker re-imports rather than inheriting the parent's
    modules, happens before any code in the worker runs. Setting the caps inside the worker is therefore too late
    for those libraries. They have to be in place in the parent before the pool starts its children. Wrapping only
    the pool's lifetime, rather than setting the caps at import, keeps the restriction off the rest of the library:
    the analysis package deliberately runs multi-threaded numba kernels, and a process-wide cap set here would
    silently serialize them.

    Yields:
        None. The caps are in effect for the duration of the block.
    """
    previous = {variable: os.environ.get(variable) for variable in WORKER_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(WORKER_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, default=None)
            else:
                os.environ[variable] = value


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
    parallel. Each chunk beyond the first decodes one extra priming frame so that the difference spanning its leading
    boundary is computed rather than lost, which makes the chunked result bit-identical to a sequential pass over the
    frames that decode. The chunk count is a pure performance knob for an intact recording. On a recording whose
    container over-reports its frame count by more than the final chunk's size, a higher chunk count turns a warning
    into a truncation error.

    Notes:
        "Motion energy" here means frame-differencing motion energy, the mean absolute inter-frame intensity change,
        not the Adelson-Bergen spatiotemporal-energy model that shares the name. Nothing filters for direction or
        speed, so the measure is undirected and unsigned. Binning before differencing is load-bearing: the absolute
        difference rectifies per-pixel sensor and codec noise into a positive bias, so binning afterwards would not
        suppress it.

    References:
        The measure and its use as a behavioral-state regressor:
            Stringer, C., et al. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. Science,
                364(6437), eaav7893.
            Musall, S., et al. (2019). Single-trial neural dynamics are dominated by richly varied movements. Nature
                Neuroscience, 22(10), 1677-1686.
            Steinmetz, N. A., et al. (2019). Distributed coding of choice, action and engagement across the mouse
                brain. Nature, 576(7786), 266-273.
        The reference implementation this module's spatial binning follows:
            Syeda, A., et al. (2024). Facemap: a framework for modeling neural activity based on orofacial tracking.
                Nature Neuroscience, 27(1), 187-195.
        Why an eye-bearing camera must exclude the eye before its energy is called facial movement:
            Talluri, B. C., et al. (2023). Activity in primate visual cortex is minimally driven by spontaneous
                movements. Nature Neuroscience, 26(11), 1953-1959.
            Turner, K. L., Gheres, K. W., & Drew, P. J. (2023). Relating pupil diameter and blinking to cortical
                activity and hemodynamics across arousal states. Journal of Neuroscience, 43(6), 949-964.
        The unrelated oriented-filter model that shares the name, disclaimed in the notes above:
            Adelson, E. H., & Bergen, J. R. (1985). Spatiotemporal energy models for the perception of motion.
                Journal of the Optical Society of America A, 2(2), 284-299.

    Args:
        video_path: The path to the camera recording to analyze.
        output_path: The path of the motion-energy feather to write.
        workers: The number of worker processes to decode with. Set to -1 to use all available CPU cores (minus
            reserved cores). Resolved here, so an unresolved count may be passed in.
        executor: An optional process pool to submit the decode chunks into, shared across cameras so the cost of
            spawning worker processes is paid once. When None, a pool is created and torn down for this recording,
            unless the recording plans to a single decode chunk, which runs in-process with no pool.
        display_progress: Determines whether per-chunk completion is reported as the analysis runs.

    Raises:
        ValueError: If the recording cannot be opened, reports no frames, cannot decode the priming frame preceding a
            chunk, or ends early at a chunk other than the last, which means the file is truncated and every later
            frame index would be wrong.
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
    # capture: cv2.CAP_PROP_N_THREADS also does, but only through the VideoCapture constructor's params argument, and
    # the OpenCV thread count governs its own kernels instead. Left unpinned, each of the many workers spawns its own
    # decode threads and the pool oversubscribes the machine several times over.
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

        # Hands back the decoded planes untouched instead of interleaving them into a color image. For this
        # monochrome source that yields the single plane the measurement needs, at no conversion cost.
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

    Uses a normalized box filter sampled on a stride, which is an exact block mean for any frame size. Resizing with
    pixel-area interpolation is not a safe substitute: it only coincides with a block mean when both dimensions
    divide evenly by the block size, and silently blends across block boundaries when they do not.

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
    bin_height = height // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE
    bin_width = width // SPATIAL_BIN_SIZE * SPATIAL_BIN_SIZE

    # The filter's anchor is the block center, so sampling from index 1 on a stride of 3 reads exactly the block
    # means, and every sampled neighbourhood is fully interior so border handling never applies. The depth is fixed
    # to single precision by the CV_32F argument, which the OpenCV stubs do not express in their return type.
    binned: NDArray[np.float32] = cv2.boxFilter(  # type: ignore[assignment]
        src=gray, ddepth=cv2.CV_32F, ksize=(SPATIAL_BIN_SIZE, SPATIAL_BIN_SIZE), normalize=True
    )[1:bin_height:SPATIAL_BIN_SIZE, 1:bin_width:SPATIAL_BIN_SIZE]
    return binned
