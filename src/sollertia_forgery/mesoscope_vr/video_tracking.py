"""Provides the Mesoscope-VR video-tracking function donated to the system-agnostic video-processing pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_EYE_TRACKING_PROJECT_NAME: str = "eye_tracking"
"""The DeepLabCut project name baked into the prediction filename, used to select this pipeline's ``.h5`` from
the predictions written beside the face-camera video in the session's raw camera_data directory."""

PUPIL_CAMERA_NAME: str = "face_camera"
"""The colloquial camera name whose recordings carry the eye, used to name this function's output feather
(``{camera}_{target}.feather``)."""

_PUPIL_TARGET: str = "pupil"
"""The tracking-target label used to name this function's output feather (``{camera}_{target}.feather``)."""

_REFLECTION_POINT: str = "reflection"
"""The canonical bodypart for the corneal (infrared) reflection, used as a motion-robust positional reference."""

_PUPIL_POINTS: tuple[str, ...] = (
    "pupil_right",
    "pupil_bottom_right",
    "pupil_bottom",
    "pupil_bottom_left",
    "pupil_left",
    "pupil_top_left",
    "pupil_top",
    "pupil_top_right",
)
"""The eight canonical pupil-perimeter bodyparts, in ring order starting from the right and advancing toward the
bottom (image coordinates, y increasing downward). The order is load-bearing: it assigns each point its parametric
angle on the pupil ellipse, evenly spaced around the ring."""

_EYE_POINTS: tuple[str, ...] = ("eye_right", "eye_bottom", "eye_left", "eye_top")
"""The four canonical eye-perimeter bodyparts, in the same ring order as the pupil's."""

_CANONICAL_POINTS: tuple[str, ...] = (_REFLECTION_POINT, *_PUPIL_POINTS, *_EYE_POINTS)
"""All thirteen canonical bodyparts the DLC ``.h5`` must provide for this function to parse it."""

_LIKELIHOOD_THRESHOLD: float = 0.9
"""The minimum DLC likelihood for a point to be trusted in a frame, set at the strict end of the field's 0.6-0.9
``pcutoff`` band. The gate is strict because the labeled-angle fit trusts each surviving point's ring identity, so a
mislabeled low-confidence point biases the ellipse directly rather than averaging out. It applies uniformly to the
pupil ring, the eye ring, and the corneal reflection."""

_MINIMUM_PERIMETER_POINTS: int = 3
"""The number of a feature's ring points that must clear the likelihood threshold for its ellipse to be determined.
Each point's ring position fixes its parametric angle, so three points supply six equations for the fit's six
unknowns regardless of where on the ring they sit. Two can never suffice."""

_MAXIMUM_FIT_CONDITION: float = 12.0
"""The largest condition number a feature's ellipse fit may have and still be trusted, above which the frame is
rejected as underdetermined. The condition number measures how far the surviving arc must reach to pin the rest
of the ellipse. This cap is the worst value a minimally-determined three-point fit produces, so it admits every
frame with enough confident points to fit."""

_BLINK_FRACTION: float = 0.5
"""The fraction of the session-median eye openness below which a frame is flagged as a blink."""

_COORDINATES: tuple[str, str, str] = ("x", "y", "likelihood")
"""The three DLC per-bodypart coordinate channels, in the order each bodypart's columns are read into its per-frame
``(x, y, likelihood)`` array."""

type _MetricArray = NDArray[np.float64] | NDArray[np.bool_]
"""The per-frame array types the output columns take: floating-point geometry and the boolean state flags."""


class PupilColumn(StrEnum):
    """Defines every column written into the Mesoscope-VR pupil-tracking feather by the donated video-tracking worker.

    Notes:
        Positions and lengths are expressed in the face camera's pixel coordinate frame (``_px``), matching the frame
        DeepLabCut reports its predictions in, and are never converted to physical units: the camera is not calibrated
        against a physical scale. Dimensionless ratios and flags carry no unit suffix.

        Every geometric column is NaN on frames whose feature kept too few confident points to fit, so consumers see
        NaN rather than a confidently-wrong ellipse. For the pupil columns those frames are exactly the ones
        ``blinking_state`` and ``dilation_state`` flag. The eye columns answer only to their own fit, so they stay
        populated on a blink flagged by low openness or by a lost corneal reflection.
    """

    PUPIL_CENTER_X_PX = "pupil_center_x_px"
    """Horizontal position of the fitted pupil ellipse center in pixels at each frame."""
    PUPIL_CENTER_Y_PX = "pupil_center_y_px"
    """Vertical position of the fitted pupil ellipse center in pixels at each frame."""
    PUPIL_DIAMETER_PX = "pupil_diameter_px"
    """Mean of the fitted pupil ellipse's two axis diameters in pixels at each frame. The pipeline's primary arousal
    proxy."""
    PUPIL_AREA_PX2 = "pupil_area_px2"
    """Area enclosed by the fitted pupil ellipse in square pixels at each frame."""
    PUPIL_FIT_CONDITION = "pupil_fit_condition"
    """Condition number of the fitted pupil ellipse at each frame, a fit-precision proxy on which higher values
    indicate a less reliable fit. NaN wherever the pupil is unmeasured."""
    PUPIL_FIT_RESIDUAL_PX = "pupil_fit_residual_px"
    """Root-mean-square distance in pixels between the confident pupil-perimeter points and their positions on the
    fitted ellipse at each frame, a fit-quality measure. NaN on unmeasured frames and on exactly-determined
    three-point fits. Only overdetermined fits report a residual."""
    EYE_CENTER_X_PX = "eye_center_x_px"
    """Horizontal position of the fitted eye ellipse center in pixels at each frame."""
    EYE_CENTER_Y_PX = "eye_center_y_px"
    """Vertical position of the fitted eye ellipse center in pixels at each frame."""
    EYE_WIDTH_PX = "eye_width_px"
    """Length of the fitted eye ellipse's left-right chord in pixels at each frame."""
    EYE_HEIGHT_PX = "eye_height_px"
    """Length of the fitted eye ellipse's top-bottom chord in pixels at each frame."""
    EYE_OPENNESS = "eye_openness"
    """Ratio of the fitted eye ellipse's height to its width at each frame, a distance-invariant measure of how open
    the eye is. NaN wherever the eye ring cannot be fit."""
    BLINKING_STATE = "blinking_state"
    """Boolean flag marking frames where the eye is closed or covered, leaving no measurable pupil. True when the eye
    ring cannot be fit, the corneal reflection is lost, or the eye opens less than half its session-median amount."""
    DILATION_STATE = "dilation_state"
    """Boolean flag marking frames where the pupil dilated past the eye's aperture and was clipped too far to fit a
    diameter. True only on non-blink frames whose pupil is otherwise unmeasurable. A fully visible dilated pupil is
    measured normally and flagged False."""
    REFLECTION_X_PX = "reflection_x_px"
    """Horizontal position of the corneal reflection in pixels at each frame."""
    REFLECTION_Y_PX = "reflection_y_px"
    """Vertical position of the corneal reflection in pixels at each frame."""
    PUPIL_REFLECTION_OFFSET_X_PX = "pupil_reflection_offset_x_px"
    """Horizontal offset in pixels of the pupil center from the corneal reflection at each frame. Referencing the
    reflection cancels the eye's common-mode motion relative to the camera, making this a motion-robust horizontal
    eye-position signal."""
    PUPIL_REFLECTION_OFFSET_Y_PX = "pupil_reflection_offset_y_px"
    """Vertical offset in pixels of the pupil center from the corneal reflection at each frame. Referencing the
    reflection cancels the eye's common-mode motion relative to the camera, making this a motion-robust vertical
    eye-position signal."""
    PUPIL_IN_EYE_X = "pupil_in_eye_x"
    """Horizontal offset of the pupil center from the eye center at each frame, normalized to the eye ellipse's
    horizontal semi-axis. Dimensionless and therefore comparable across animals."""
    PUPIL_IN_EYE_Y = "pupil_in_eye_y"
    """Vertical offset of the pupil center from the eye center at each frame, normalized to the eye ellipse's vertical
    semi-axis. Dimensionless and therefore comparable across animals."""


def process_mesoscope_video_tracking(session: SessionData, output_directory: Path) -> None:
    """Post-processes the Mesoscope-VR face-camera DLC predictions into per-frame pupil and eye metrics.

    Locates the pupil project's externally-produced DLC ``.h5`` beside the face-camera video in the session's raw
    camera_data directory, where the acquisition rig writes it during preprocessing. If none is present, returns
    without doing anything: the stage is optional and gated on detecting the prediction file.

    Otherwise, reads the thirteen canonical bodyparts and fits an ellipse to the pupil and to the eye for each frame.
    Flags occluded frames as blinks and derives motion-robust eye-position signals from the pupil relative to the eye
    and the corneal reflection. Writes the results into a ``{camera}_pupil.feather`` in the processed video-data
    directory.

    Notes:
        The predictions are produced upstream rather than here because DeepLabCut pins ``numpy<2`` while
        sollertia-forgery runs ``numpy>=2`` on Python 3.14, so DeepLabCut cannot be imported in-process. This worker
        only reads DeepLabCut's ``.h5`` output and never depends on the ``deeplabcut`` library.

    Args:
        session: The loaded session whose pupil DLC predictions are post-processed. Its raw camera_data directory
            supplies the DLC ``.h5``.
        output_directory: The processed video-data directory (``session.processed_data.video_data_path``) the pupil
            feather is written into.

    Raises:
        ValueError: If a DLC ``.h5`` is present but is missing a canonical bodypart or has an unrecognized layout.
    """
    # DeepLabCut runs upstream on the acquisition rig, which writes its predictions beside the face-camera video in
    # the session's raw camera_data directory during preprocessing. This pipeline's file is identified by the
    # DeepLabCut project name baked into its filename. When several match, the natural-sort-first one is used.
    matches = natsorted(session.raw_data.camera_data_path.glob(f"*{_EYE_TRACKING_PROJECT_NAME}*.h5"))
    if not matches:
        console.echo(
            message=(
                f"No DeepLabCut '{_EYE_TRACKING_PROJECT_NAME}' '.h5' prediction file was found beside the face-camera "
                f"video in the raw camera_data directory of session '{session.session_name}'. Skipping pupil tracking."
            ),
            level=LogLevel.INFO,
        )
        return
    h5_path = matches[0]

    points = _read_points_from_h5(h5_path=h5_path, bodyparts=_CANONICAL_POINTS)
    frame_count = next(iter(points.values())).shape[0]

    pupil_metrics = _compute_pupil_metrics(points=points)

    # The feather is a positional table: one row per frame in acquisition order, storing only the per-frame metrics.
    # Row position supplies the frame index, so none is stored. Timestamps are left to dataset assembly, which owns
    # every stream's alignment to the acquisition clock. The float64 geometry is cast to single precision because it
    # derives from pixel coordinates far coarser than float32 resolves, so float64 would only double the feather size.
    pupil_frame = pl.DataFrame(pupil_metrics).with_columns(pl.col(pl.Float64).cast(pl.Float32))

    output_path = output_directory.joinpath(f"{PUPIL_CAMERA_NAME}_{_PUPIL_TARGET}.feather")
    pupil_frame.write_ipc(file=output_path, compression="uncompressed")

    console.echo(
        message=f"Wrote pupil tracking for {frame_count} frame(s) to '{output_path.name}'.",
        level=LogLevel.SUCCESS,
    )


def _read_points_from_h5(h5_path: Path, bodyparts: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    """Reads the requested bodyparts from a DeepLabCut prediction ``.h5`` into per-bodypart coordinate arrays.

    DeepLabCut serializes its predictions with ``pandas.DataFrame.to_hdf`` as a PyTables ``table``-format HDF5, so they
    are read back with ``pandas``. The columns are a ``(scorer, bodypart, coordinate)`` MultiIndex, so this returns one
    ``(frame_count, 3)`` array of ``(x, y, likelihood)`` per requested bodypart, with rows in ascending frame order.

    Args:
        h5_path: The path to the DLC prediction ``.h5`` file.
        bodyparts: The canonical bodyparts to extract.

    Returns:
        A mapping from each requested bodypart to its ``(frame_count, 3)`` array of per-frame ``(x, y, likelihood)``.

    Raises:
        ValueError: If the file does not hold a DeepLabCut prediction frame, or a requested bodypart is missing.
    """
    predictions = pd.read_hdf(h5_path)
    if not isinstance(predictions, pd.DataFrame):
        message = (
            f"Unable to read pupil tracking from '{h5_path.name}'. The file does not contain a DeepLabCut prediction "
            f"frame."
        )
        console.error(message=message, error=ValueError)
    predictions = predictions.sort_index()

    # DeepLabCut labels columns with a (scorer, bodypart, coordinate) MultiIndex. Keys each column by the trailing
    # (bodypart, coordinate) pair, so the single scorer level need not be named, then reads the whole matrix once.
    # A file carrying more than one scorer would collapse onto the last, which single-scorer DLC output never emits.
    bodypart_labels = predictions.columns.get_level_values(-2)
    coordinate_labels = predictions.columns.get_level_values(-1)
    column_position: dict[tuple[str, str], int] = {
        (bodypart, coordinate): position
        for position, (bodypart, coordinate) in enumerate(zip(bodypart_labels, coordinate_labels, strict=True))
    }
    matrix = predictions.to_numpy(dtype=np.float64)

    result: dict[str, NDArray[np.float64]] = {}
    for bodypart in bodyparts:
        channels = []
        for coordinate in _COORDINATES:
            position = column_position.get((bodypart, coordinate))
            if position is None:
                message = (
                    f"Unable to read pupil tracking from '{h5_path.name}'. The DeepLabCut prediction file does not "
                    f"contain the required '{bodypart}' '{coordinate}' column."
                )
                console.error(message=message, error=ValueError)
            channels.append(matrix[:, position])
        result[bodypart] = np.stack(channels, axis=1)
    return result


def _compute_pupil_metrics(points: dict[str, NDArray[np.float64]]) -> dict[str, _MetricArray]:
    """Computes per-frame pupil and eye metrics from the thirteen canonical points.

    Fits an ellipse to the pupil and to the eye each frame from whichever ring points survive the likelihood gate.
    Flags occluded frames as blinks, derives motion-robust eye-position signals from the pupil relative to the eye and
    to the corneal reflection, and reports the pupil fit's per-frame quality (condition number and, where the ring is
    overdetermined, RMS residual). Frames whose feature is too occluded to fit yield NaN geometry.

    Args:
        points: A mapping from each canonical bodypart to its ``(frame_count, 3)`` ``(x, y, likelihood)`` array.

    Returns:
        A mapping from each output column name to its per-frame array, ready to assemble into the output feather.
    """
    # A feature is measurable exactly when its fit is determined and well enough conditioned to trust. The pupil
    # columns additionally answer to the blink. The eye columns answer only to their own fit.
    pupil_center, pupil_semi_a, pupil_semi_b, pupil_condition, pupil_residual, pupil_valid = _fit_ring_ellipse(
        points=points, names=_PUPIL_POINTS
    )
    # The ellipse area is pi times the cross-product magnitude of its two conjugate semi-diameters.
    pupil_area = np.pi * np.abs(pupil_semi_a[:, 0] * pupil_semi_b[:, 1] - pupil_semi_a[:, 1] * pupil_semi_b[:, 0])
    # The pupil diameter is the mean of the ellipse's two full-axis diameters, (2|semi_a| + 2|semi_b|) / 2, i.e. the
    # sum of the two semi-diameter lengths. There is no pupil width or height output column, so neither is materialized.
    pupil_diameter = _norm(pupil_semi_a) + _norm(pupil_semi_b)

    eye_center, eye_semi_a, eye_semi_b, _, _, eye_valid = _fit_ring_ellipse(points=points, names=_EYE_POINTS)
    eye_semi_width, eye_semi_height = _norm(eye_semi_a), _norm(eye_semi_b)
    eye_width, eye_height = 2.0 * eye_semi_width, 2.0 * eye_semi_height
    # Eye openness is the eye's vertical-to-horizontal aspect ratio, which is invariant to camera distance. A
    # zero-width eye is a degenerate fit rather than a closed eye, so it yields NaN instead of dividing.
    eye_openness = np.divide(eye_height, eye_width, out=np.full_like(eye_height, np.nan), where=eye_width > 0.0)

    # With no confident, non-degenerate eye fit anywhere in the session there is no openness baseline to compare
    # against. Resolves to NaN and the openness term drops out of the flag below, leaving the eye's visibility
    # to carry it.
    confident_openness = eye_openness[eye_valid]
    baseline = float(np.nanmedian(confident_openness)) if np.isfinite(confident_openness).any() else np.nan

    reflection = points[_REFLECTION_POINT][:, :2]
    reflection_valid = points[_REFLECTION_POINT][:, 2] >= _LIKELIHOOD_THRESHOLD

    # A blink is read from the eye and its cornea alone, never from the pupil. Something covering the eye takes the
    # eye ring, the corneal reflection, and the opening down together, and the cause does not change the consequence:
    # a lid and a paw read the same. The pupil is deliberately excluded because a pupil that vanishes under an OPEN
    # eye has outgrown the aperture rather than been hidden by a lid. Folding that in here would delete the most
    # dilated pupils from the arousal signal exactly when arousal is highest.
    is_blink = (
        ~(eye_valid & reflection_valid) | ~np.isfinite(eye_openness) | (eye_openness < _BLINK_FRACTION * baseline)
    )
    not_blink = ~is_blink

    # Behind a shut lid there is no pupil to measure, so a pupil fit that happens to converge on a blink frame is
    # reporting on points the eye was covering. The pupil columns answer to the blink as well as to their own fit.
    # The eye columns answer only to theirs, since their openness is what detects the blink in the first place.
    pupil_measured = pupil_valid & not_blink

    # The remaining way to lose the pupil is for it to outgrow the palpebral opening, which then clips it past what
    # the surviving arc can reconstruct. The eye is plainly open, so this is dilation rather than a blink, and the two
    # flags partition every unmeasured frame between them: a frame is measured, dilated, or blinked, never two.
    is_dilated = not_blink & ~pupil_measured

    pupil_reflection_offset = pupil_center - reflection
    pupil_reflection_offset = np.where((pupil_measured & reflection_valid)[:, None], pupil_reflection_offset, np.nan)
    # Normalizes the pupil's offset to the eye's semi-axes. A degenerate zero-extent eye divides to NaN on the
    # collapsed axis rather than to an infinity, matching how eye openness handles the same fit.
    eye_semi_axes = np.stack([eye_semi_width, eye_semi_height], axis=1)
    pupil_in_eye = np.divide(
        pupil_center - eye_center,
        eye_semi_axes,
        out=np.full_like(eye_semi_axes, np.nan),
        where=eye_semi_axes > 0.0,
    )
    pupil_in_eye = np.where((pupil_measured & eye_valid)[:, None], pupil_in_eye, np.nan)

    # Masks geometry of frames whose source points were gated out so downstream consumers see NaN, not a bad fit.
    pupil_metrics = _mask_invalid(
        metrics={
            PupilColumn.PUPIL_CENTER_X_PX: pupil_center[:, 0],
            PupilColumn.PUPIL_CENTER_Y_PX: pupil_center[:, 1],
            PupilColumn.PUPIL_DIAMETER_PX: pupil_diameter,
            PupilColumn.PUPIL_AREA_PX2: pupil_area,
            PupilColumn.PUPIL_FIT_CONDITION: pupil_condition,
            PupilColumn.PUPIL_FIT_RESIDUAL_PX: pupil_residual,
        },
        valid=pupil_measured,
    )
    eye_metrics = _mask_invalid(
        metrics={
            PupilColumn.EYE_CENTER_X_PX: eye_center[:, 0],
            PupilColumn.EYE_CENTER_Y_PX: eye_center[:, 1],
            PupilColumn.EYE_WIDTH_PX: eye_width,
            PupilColumn.EYE_HEIGHT_PX: eye_height,
            PupilColumn.EYE_OPENNESS: eye_openness,
        },
        valid=eye_valid,
    )

    return {
        **pupil_metrics,
        **eye_metrics,
        PupilColumn.BLINKING_STATE: is_blink,
        PupilColumn.DILATION_STATE: is_dilated,
        PupilColumn.REFLECTION_X_PX: np.where(reflection_valid, reflection[:, 0], np.nan),
        PupilColumn.REFLECTION_Y_PX: np.where(reflection_valid, reflection[:, 1], np.nan),
        PupilColumn.PUPIL_REFLECTION_OFFSET_X_PX: pupil_reflection_offset[:, 0],
        PupilColumn.PUPIL_REFLECTION_OFFSET_Y_PX: pupil_reflection_offset[:, 1],
        PupilColumn.PUPIL_IN_EYE_X: pupil_in_eye[:, 0],
        PupilColumn.PUPIL_IN_EYE_Y: pupil_in_eye[:, 1],
    }


def _fit_ring_ellipse(
    points: dict[str, NDArray[np.float64]], names: tuple[str, ...]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    """Fits a centrally symmetric ellipse per frame from whichever of a feature's ring points are confident.

    A point's position in the ring fixes its parametric angle, so a point labeled ``t`` lies at
    ``P(t) = center + cos(t) * semi_a + sin(t) * semi_b``, where ``semi_a`` and ``semi_b`` are the ellipse's conjugate
    semi-diameters. That is linear in the unknowns and separates by axis, so both coordinates share one
    ``[1, cos(t), sin(t)]`` design matrix and three confident points already determine the fit, wherever on the ring
    they sit. Points are used as they come: no point is ever reconstructed or imputed.

    Determined is not the same as trustworthy. The design matrix depends only on which points survived, so its
    condition number measures how far the surviving arc has to reach to pin the rest of the ellipse. Frames whose arc
    is too narrow to carry a measurement are rejected rather than fitted.

    Args:
        points: A mapping from each canonical bodypart to its ``(frame_count, 3)`` ``(x, y, likelihood)`` array.
        names: The feature's ring bodyparts, in ring order.

    Returns:
        A ``(center, semi_a, semi_b, condition, residual, valid)`` tuple. ``center``, ``semi_a`` and ``semi_b`` are
        ``(frame_count, 2)`` arrays, NaN wherever the fit was rejected. ``condition`` is the per-frame fit condition
        number (NaN where rejected), ``residual`` is the per-frame RMS point-to-ellipse distance in pixels, NaN where
        rejected or where the fit was exactly determined (too few points to over-constrain it). ``valid`` is the
        per-frame mask of frames that were fitted.
    """
    frame_count = points[names[0]].shape[0]
    # The evenly spaced parametric angle of each ring point, in radians and starting at zero.
    angles = 2.0 * np.pi * np.arange(len(names), dtype=np.float64) / len(names)
    confident = np.stack([points[name][:, 2] >= _LIKELIHOOD_THRESHOLD for name in names], axis=1)
    coordinates = np.stack([points[name][:, :2] for name in names], axis=1)

    center = np.full((frame_count, 2), np.nan)
    semi_a = np.full((frame_count, 2), np.nan)
    semi_b = np.full((frame_count, 2), np.nan)
    condition = np.full(frame_count, np.nan)
    residual = np.full(frame_count, np.nan)
    valid = np.zeros(frame_count, dtype=np.bool_)

    # Frames that lost the same points share a design matrix, and therefore a conditioning verdict, so each distinct
    # occlusion pattern is solved once for every frame that carries it. Packing each frame's confidence mask into one
    # integer code groups the patterns with a fast 1-D unique, avoiding the void-row lexsort np.unique(..., axis=0)
    # would run over the whole (frame_count, point_count) mask.
    codes = confident.astype(np.int64) @ (1 << np.arange(len(names), dtype=np.int64))
    _, representatives, inverse = np.unique(codes, return_index=True, return_inverse=True)
    inverse = np.ravel(inverse)
    for index in range(representatives.size):
        kept = np.flatnonzero(confident[representatives[index]])
        if kept.size < _MINIMUM_PERIMETER_POINTS:
            continue
        design = np.stack([np.ones(kept.size), np.cos(angles[kept]), np.sin(angles[kept])], axis=1)
        condition_number = float(np.linalg.cond(design))
        if not condition_number <= _MAXIMUM_FIT_CONDITION:
            continue
        rows = np.flatnonzero(inverse == index)
        observed = coordinates[np.ix_(rows, kept)].transpose(1, 0, 2).reshape(kept.size, -1)
        solution = np.linalg.lstsq(design, observed, rcond=None)[0]
        center[rows], semi_a[rows], semi_b[rows] = solution.reshape(3, rows.size, 2)
        condition[rows] = condition_number

        # The residual only carries information when the ring is overdetermined: as many points as the design has
        # unknowns (three) determine the ellipse exactly, so their residual is structurally zero and says nothing about
        # fit quality. Those frames keep their NaN residual and are judged on the condition number alone.
        if kept.size > design.shape[1]:
            deviations = (design @ solution - observed).reshape(kept.size, rows.size, 2)
            residual[rows] = np.sqrt(np.mean(deviations[:, :, 0] ** 2 + deviations[:, :, 1] ** 2, axis=0))
        valid[rows] = True
    return center, semi_a, semi_b, condition, residual, valid


def _norm(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the per-frame length of a ``(frame_count, 2)`` array of vectors.

    Args:
        vectors: The ``(frame_count, 2)`` array of per-frame vectors.

    Returns:
        The per-frame vector lengths.
    """
    return np.hypot(vectors[:, 0], vectors[:, 1])


def _mask_invalid(metrics: dict[str, NDArray[np.float64]], valid: NDArray[np.bool_]) -> dict[str, NDArray[np.float64]]:
    """Replaces metric values with NaN wherever the per-frame validity mask is False.

    Args:
        metrics: A mapping from each metric column name to its per-frame array.
        valid: The per-frame validity mask.

    Returns:
        A new mapping with the same keys whose values are NaN where ``valid`` is False.
    """
    return {name: np.where(valid, values, np.nan) for name, values in metrics.items()}
