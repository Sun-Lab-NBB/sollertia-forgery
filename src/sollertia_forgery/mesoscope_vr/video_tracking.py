"""Provides the Mesoscope-VR video-tracking function donated to the system-agnostic video-processing pipeline.

The video pipeline resolves this function from the registry hub and runs it once per session, expecting it to do all
the work (mirroring how a system donates its forging assembler). It post-processes externally-produced DeepLabCut
(DLC) pose predictions for the Mesoscope-VR face camera into per-frame pupil and eye metrics.

Notes:
    sollertia-forgery never runs DeepLabCut: DLC pins ``numpy<2`` while slf is ``numpy>=2`` / Python 3.14, so DLC
    cannot be imported in-process. DLC instead runs on the acquisition rig during preprocessing, so its ``.h5``
    predictions travel with the raw data and are read from the session's raw ``camera_data`` directory, beside the
    face-camera video they were produced from. This module only READS that output -- via ``h5py`` and ``numpy``
    (never ``pandas`` and never the ``deeplabcut`` library), reading only the columns it needs without materializing
    the full table -- and only when the file is present. With no ``.h5`` the function is a no-op, so the pipeline can
    run it unconditionally.

    The model that produced the predictions is irrelevant here: the pupil project's prediction file is identified by
    its DeepLabCut project name, and this function only requires that the ``.h5`` carries the thirteen canonical
    bodyparts below. They are spelled in a DLC-friendly format (``eye_left`` etc.): ``reflection`` (the corneal
    reflection), the eight pupil-perimeter points (the four cardinals ``pupil_top``/``pupil_bottom``/``pupil_left``/
    ``pupil_right`` plus the four diagonals ``pupil_top_left``/``pupil_top_right``/``pupil_bottom_left``/
    ``pupil_bottom_right``), and the four eye-perimeter points (``eye_top``/``eye_bottom``/``eye_left``/
    ``eye_right``).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

import h5py
import numpy as np
import polars as pl
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

EYE_TRACKING_PROJECT_NAME: str = "eye_tracking"
"""The DeepLabCut project (task) name baked into the prediction filename, used to identify this pipeline's ``.h5``
among any predictions written beside the face-camera video in the session's raw camera_data directory. The donor is a
pure reader: the ``.h5`` is produced upstream on the acquisition rig, and this name selects which model's predictions
it consumes."""

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
"""The minimum DLC likelihood for a point to be trusted in a frame. Set at the strict end of the field's 0.6-0.9
``pcutoff`` band (matching keypoint-pupillometry pipelines such as Pupil-DLC) because the labeled-angle fit trusts
each surviving point's ring identity: a displaced or mislabeled low-confidence point biases the ellipse directly,
rather than being averaged out as it would be in an identity-agnostic conic fit. The fit needs only three points, so a
conservative gate trades a handful of borderline frames for lower per-fit bias -- a trade the three-point tolerance
easily affords. Applied uniformly to the pupil ring, the eye ring, and the corneal reflection: a low-confidence eyelid
point or glint is treated as not seen, feeding the same blink logic that reads occlusion from the eye. A feature whose
surviving points cannot support a fit yields NaN metrics for that frame and marks it as a blink."""

_MINIMUM_PERIMETER_POINTS: int = 3
"""The number of a feature's ring points that must clear the likelihood threshold for its ellipse to be determined.
Each point's ring position fixes its parametric angle, so three points supply six equations for the fit's six
unknowns regardless of where on the ring they sit. Two can never suffice."""

_MAXIMUM_FIT_CONDITION: float = 12.0
"""The largest tolerated condition number of a feature's fit, which measures how far the surviving arc must reach to
pin the rest of the ellipse. Points spread around the ring condition at ~1.4; three bunched into a single quarter
condition at ~12, the worst any determinable frame can be, so this cap admits every frame that clears
``_MINIMUM_PERIMETER_POINTS`` and rejects only what is underdetermined.

Notes:
    The cap is deliberately permissive because badly conditioned frames are not random. An arc collapses precisely
    when the pupil has dilated past the aperture, so the worst-conditioned frames are the largest pupils, and fit
    error tracks label noise rather than pupil size -- which makes the same absolute error a far smaller fraction of
    a big pupil. At the size where a quarter-arc actually occurs that is ~3% of the diameter and essentially
    unbiased, and the scatter averages out. Rejecting those frames would not buy precision, it would carve a hole out
    of the top of the arousal range. A small pupil keeps its whole ring and never reaches this cap.

    The value depends only on which points survived, never on their coordinates, so every frame sharing an occlusion
    pattern shares its verdict.
"""

_BLINK_FRACTION: float = 0.5
"""The fraction of the session-median eye openness below which a frame is flagged as a blink."""

_COORDINATES: tuple[str, str, str] = ("x", "y", "likelihood")
"""The three DLC per-bodypart coordinate columns. Recognizes the coordinate level of the column MultiIndex, and fixes
the order each bodypart's channels are read in."""


def _ring_angles(point_count: int) -> NDArray[np.float64]:
    """Computes the parametric angle of each point on an evenly spaced perimeter ring.

    Args:
        point_count: The number of points in the ring.

    Returns:
        The per-point parametric angles in radians, starting at zero and advancing evenly around the ring.
    """
    return 2.0 * np.pi * np.arange(point_count, dtype=np.float64) / point_count


class PupilColumn(StrEnum):
    """Defines every column written into the Mesoscope-VR pupil-tracking feather by the donated video-tracking worker.

    Notes:
        Positions and lengths are expressed in the face camera's pixel coordinate frame (``_px``), matching the frame
        DeepLabCut reports its predictions in, and are never converted to physical units: the camera is not calibrated
        against a physical scale. Dimensionless ratios and flags carry no unit suffix.

        Every geometric column is NaN on frames whose feature kept too few confident points to fit, so consumers see
        NaN rather than a confidently-wrong ellipse. Those frames are exactly the ones ``blinking_state`` and
        ``dilation_state`` flag.
    """

    FRAME = "frame"
    """One-based face-camera frame index, numbered to match the frame identifiers in the forged session dataset. The
    feather's only key: every other column is a per-frame metric, and aligning frames to the acquisition clock is left
    to dataset assembly."""
    PUPIL_CENTER_X_PX = "pupil_center_x_px"
    """Horizontal position of the fitted pupil ellipse center in pixels at each frame."""
    PUPIL_CENTER_Y_PX = "pupil_center_y_px"
    """Vertical position of the fitted pupil ellipse center in pixels at each frame."""
    PUPIL_DIAMETER_PX = "pupil_diameter_px"
    """Mean of the fitted pupil ellipse's two chords in pixels at each frame. The primary arousal proxy."""
    PUPIL_AREA_PX2 = "pupil_area_px2"
    """Area enclosed by the fitted pupil ellipse in square pixels at each frame."""
    PUPIL_FIT_CONDITION = "pupil_fit_condition"
    """Condition number of the pupil ellipse fit at each frame -- a precision proxy present on every measured frame and
    NaN wherever the pupil is unmeasured (blink or dilation). It measures how far the surviving arc had to reach to pin
    the rest of the ellipse: a full ring conditions near 1.4, while three points bunched into a single quarter approach
    the ~12 cap that admits a fit. Higher is less precise. Unlike ``pupil_fit_residual_px`` it is defined even for an
    exactly-determined three-point fit, so it is the quality signal to weight or threshold frames on when the residual
    is NaN. It depends only on which points survived, so every frame sharing an occlusion pattern shares its value."""
    PUPIL_FIT_RESIDUAL_PX = "pupil_fit_residual_px"
    """Root-mean-square distance in pixels between the confident pupil-perimeter points and where the fitted ellipse
    places them, at each frame. Catches a confident-but-displaced point (an eyelash or specular artifact the fit was
    dragged toward) that the condition number cannot see, since conditioning measures the surviving arc's geometry, not
    whether the points agree with an ellipse. NaN on any unmeasured frame AND on an exactly-determined three-point fit,
    whose residual is structurally zero and so carries no information -- read those frames' quality from
    ``pupil_fit_condition`` instead. Only overdetermined (four-or-more-point) fits report a residual."""
    EYE_CENTER_X_PX = "eye_center_x_px"
    """Horizontal position of the fitted eye ellipse center in pixels at each frame."""
    EYE_CENTER_Y_PX = "eye_center_y_px"
    """Vertical position of the fitted eye ellipse center in pixels at each frame."""
    EYE_WIDTH_PX = "eye_width_px"
    """Length of the fitted eye ellipse's left-right chord in pixels at each frame."""
    EYE_HEIGHT_PX = "eye_height_px"
    """Length of the fitted eye ellipse's top-bottom chord in pixels at each frame."""
    EYE_OPENNESS = "eye_openness"
    """Ratio of the fitted eye ellipse's height to its width at each frame -- the palpebral aperture aspect ratio, a
    distance-invariant squint index and the quantity the blink threshold is applied to. Read graded squint (drowsiness,
    grimace) from this; it captures state variance the pupil diameter misses.

    Caveat: on a deep closure the eye ring fails to fit, so this goes NaN rather than to zero. Read full closure from
    ``blinking_state``, never from a low openness value, which only spans the shallow-squint band where the fit still
    converges. The threshold is session-relative (below half the session-median), so it indexes squint within a
    session, not an absolute openness across sessions."""
    BLINKING_STATE = "blinking_state"
    """Boolean flag: the eye is closed or covered at this frame. Set when the eye ring cannot be fit, when the corneal
    reflection is lost, when the fit is too degenerate to define an openness, or when eye openness falls below half of
    the session-median openness. A lid and a paw are not distinguished; both hide the eye and leave no usable pupil.
    Read from the eye alone, never the pupil -- see ``dilation_state``. A flagged frame's geometry is NaN."""
    DILATION_STATE = "dilation_state"
    """Boolean flag: the pupil dilated past the eye's aperture at this frame, so the aperture clipped it beyond
    recovery. Set when the frame is not a blink yet too little of the pupil perimeter survives to fit a diameter. This
    is a right-censored large-pupil marker, not a general 'is the pupil dilated' signal: a fully visible dilated pupil
    is measured and carries ``dilation_state`` False, with its size in ``pupil_diameter_px``. ``dilation_state`` True
    says only that the true diameter ran off the top of the measurable range. With ``blinking_state`` it partitions
    every unmeasured frame -- a frame is measured, dilated, or blinking, never two at once."""
    REFLECTION_X_PX = "reflection_x_px"
    """Horizontal position of the corneal reflection in pixels at each frame."""
    REFLECTION_Y_PX = "reflection_y_px"
    """Vertical position of the corneal reflection in pixels at each frame."""
    PUPIL_REFLECTION_OFFSET_X_PX = "pupil_reflection_offset_x_px"
    """Horizontal offset of the pupil center from the corneal reflection in pixels at each frame. Cancels common-mode
    translation of the eye relative to the camera, making this the motion-robust horizontal eye-position signal.

    Caveat: NaN whenever the pupil fit OR the corneal reflection failed its gate (their union), so it is blank on more
    frames than the pupil alone -- never filter arousal frames on this column's NaNs. It is uncalibrated pixels
    referenced to a glint that itself drifts as the pupil dilates, so it is a within-session signal, not a gaze angle
    comparable across sessions."""
    PUPIL_REFLECTION_OFFSET_Y_PX = "pupil_reflection_offset_y_px"
    """Vertical offset of the pupil center from the corneal reflection in pixels at each frame. Cancels common-mode
    translation of the eye relative to the camera, making this the motion-robust vertical eye-position signal. Shares
    the union-mask and within-session caveats of ``pupil_reflection_offset_x_px``."""
    PUPIL_IN_EYE_X = "pupil_in_eye_x"
    """Horizontal offset of the pupil center from the eye center at each frame, normalized to the eye ellipse's
    horizontal semi-axis. Dimensionless, and therefore comparable across animals; also a glint-independent backup for
    ``pupil_reflection_offset_x_px`` when the corneal reflection is lost.

    Caveat: this conflates dilation with gaze. As the pupil dilates its apparent center shifts, and because this
    references the eyelid frame it does not cancel that shift the way the corneal-reflection offset does -- so it is
    not a clean gaze nuisance regressor for ``pupil_diameter_px``. Prefer the reflection offset as the primary gaze
    signal."""
    PUPIL_IN_EYE_Y = "pupil_in_eye_y"
    """Vertical offset of the pupil center from the eye center at each frame, normalized to the eye ellipse's vertical
    semi-axis. Dimensionless, and therefore comparable across animals. The weakest gaze component: eyelid squint moves
    both the eye center and the eye height, folding lid motion into its reference and its scale. Shares the
    dilation-gaze confound of ``pupil_in_eye_x``."""


def process_mesoscope_video_tracking(session: SessionData, output_directory: Path) -> None:
    """Post-processes the Mesoscope-VR face-camera DLC predictions into per-frame pupil and eye metrics.

    Locates the pupil project's externally-produced DLC ``.h5`` beside the face-camera video in the session's raw
    camera_data directory, where the acquisition rig writes it during preprocessing so that it travels with the raw
    data. If none is present, returns without doing anything: the stage is optional and gated on detecting the
    prediction file.

    Otherwise reads the thirteen canonical bodyparts and fits an ellipse to the pupil and to the eye each frame. Flags
    occluded frames as blinks and derives motion-robust eye-position signals from the pupil relative to the eye and
    the corneal reflection. Writes the results into a ``{camera}_pupil.feather`` in the processed video-data
    directory.

    The feather is strictly a frame index and the metrics keyed to it. It carries no timestamps: aligning frames to
    the acquisition clock belongs to dataset assembly, which owns every other stream's alignment too.

    Args:
        session: The loaded session whose pupil DLC predictions are post-processed. Its raw camera_data directory
            supplies the DLC ``.h5``.
        output_directory: The processed video-data directory (``session.processed_data.video_data_path``) the pupil
            feather is written into.

    Raises:
        ValueError: If a DLC ``.h5`` is present but is missing a canonical bodypart or has an unrecognized layout.
    """
    h5_path = _locate_pupil_h5(camera_data_directory=session.raw_data.camera_data_path)
    if h5_path is None:
        console.echo(
            message=(
                f"No DeepLabCut '{EYE_TRACKING_PROJECT_NAME}' '.h5' prediction file was found beside the face-camera "
                f"video in the raw camera_data directory of session '{session.session_name}'. Skipping pupil tracking."
            ),
            level=LogLevel.INFO,
        )
        return

    console.echo(
        message=f"Post-processing pupil tracking from '{h5_path.name}' for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    points = _read_points_from_h5(h5_path=h5_path, bodyparts=_CANONICAL_POINTS)
    frame_count = next(iter(points.values())).shape[0]

    pupil_metrics = _compute_pupil_metrics(points=points)

    # Numbers frames from 1 to match the one-based frame identifiers the forging pipeline assigns in 'data.feather',
    # so the two can be joined without an off-by-one correction. The feather carries nothing but the frame index and
    # the metrics keyed to it: aligning frames to the acquisition clock is the dataset assembler's job.
    columns: dict[str, NDArray[np.generic]] = {PupilColumn.FRAME: np.arange(1, frame_count + 1, dtype=np.uint32)}
    columns.update(pupil_metrics)

    # Narrows the metrics to single precision for storage: they are derived from pixel coordinates whose precision is
    # far coarser than float32 resolves, so double precision would only double the feather's size.
    pupil_frame = pl.DataFrame(columns).with_columns(pl.col(pl.Float64).cast(pl.Float32))

    output_path = output_directory.joinpath(f"{PUPIL_CAMERA_NAME}_{_PUPIL_TARGET}.feather")
    pupil_frame.write_ipc(file=output_path, compression="uncompressed")

    console.echo(
        message=f"Wrote pupil tracking for {frame_count} frame(s) to '{output_path.name}'.",
        level=LogLevel.SUCCESS,
    )


def _locate_pupil_h5(camera_data_directory: Path) -> Path | None:
    """Resolves the pupil project's externally-produced DLC ``.h5`` within the session's raw camera_data directory.

    DeepLabCut runs upstream on the acquisition rig, which writes its predictions beside the face-camera video in the
    session's raw camera_data directory during preprocessing, so the predictions travel with the raw data and this
    donor only reads them. The pupil pipeline's file is identified by the DeepLabCut project name baked into its
    filename.

    Args:
        camera_data_directory: The session's raw camera_data directory (``session.raw_data.camera_data_path``) where
            the DLC predictions are expected, beside the face-camera video.

    Returns:
        The path to the discovered ``.h5`` file, or None if no matching prediction file is present (meaning there is
        nothing to process).
    """
    matches = natsorted(camera_data_directory.glob(f"*{EYE_TRACKING_PROJECT_NAME}*.h5"))
    return matches[0] if matches else None


def _read_points_from_h5(h5_path: Path, bodyparts: tuple[str, ...]) -> dict[str, NDArray[np.float64]]:
    """Reads the requested bodyparts from a DLC ``.h5`` using ``h5py`` and ``numpy`` (no ``pandas``).

    Reads only the columns for the requested bodyparts -- via lazy ``h5py`` slicing, so the full prediction table is
    never materialized -- and returns one ``(frame_count, 3)`` array of ``(x, y, likelihood)`` per bodypart.

    Args:
        h5_path: The path to the DLC prediction ``.h5`` file.
        bodyparts: The canonical bodyparts to extract.

    Returns:
        A mapping from each requested bodypart to its ``(frame_count, 3)`` array of per-frame ``(x, y, likelihood)``.

    Raises:
        ValueError: If the file layout is unrecognized, or if a requested bodypart is missing from the file.
    """
    with h5py.File(name=str(h5_path), mode="r") as h5_file:
        group = _resolve_predictions_group(h5_file=h5_file, h5_path=h5_path)
        per_column_bodypart, per_column_coordinate = _read_column_labels(group=group, h5_path=h5_path)
        values, frames_first = _resolve_value_matrix(
            group=group, column_count=len(per_column_bodypart), h5_path=h5_path
        )

        # Builds, for every requested bodypart, the column index of each of its three coordinate channels.
        column_index: dict[tuple[str, str], int] = {
            (bodypart, coordinate): index
            for index, (bodypart, coordinate) in enumerate(zip(per_column_bodypart, per_column_coordinate, strict=True))
        }

        result: dict[str, NDArray[np.float64]] = {}
        for bodypart in bodyparts:
            channels = []
            for coordinate in _COORDINATES:
                index = column_index.get((bodypart, coordinate))
                if index is None:
                    message = (
                        f"Unable to read pupil tracking from '{h5_path.name}'. The DeepLabCut prediction file does "
                        f"not contain the required '{bodypart}' '{coordinate}' column."
                    )
                    console.error(message=message, error=ValueError)
                # Reads a single column lazily: the matrix is oriented either (frames, columns) or (columns, frames).
                channel = values[:, index] if frames_first else values[index, :]
                channels.append(np.asarray(channel, dtype=np.float64))
            result[bodypart] = np.stack(channels, axis=1)
    return result


def _resolve_predictions_group(h5_file: h5py.File, h5_path: Path) -> h5py.Group:
    """Resolves the HDF5 group holding the serialized DLC predictions frame.

    Args:
        h5_file: The opened DLC ``.h5`` file.
        h5_path: The path to the file, used only for error messages.

    Returns:
        The predictions group (``df_with_missing`` when present, otherwise the single top-level group).

    Raises:
        ValueError: If no group can be resolved.
    """
    if "df_with_missing" in h5_file:
        node = h5_file["df_with_missing"]
        if isinstance(node, h5py.Group):
            return node
    groups = [key for key in h5_file if isinstance(h5_file[key], h5py.Group)]
    if len(groups) == 1:
        node = h5_file[groups[0]]
        if isinstance(node, h5py.Group):
            return node
    message = (
        f"Unable to read pupil tracking from '{h5_path.name}'. The file does not contain a recognizable DeepLabCut "
        f"predictions group ('df_with_missing')."
    )
    console.error(message=message, error=ValueError)
    raise ValueError(message)  # pragma: no cover - console.error is NoReturn; satisfies RET503/return typing.


def _read_column_labels(group: h5py.Group, h5_path: Path) -> tuple[list[str], list[str]]:
    """Reconstructs the per-column ``(bodypart, coordinate)`` labels from a pandas fixed-format column MultiIndex.

    pandas stores a MultiIndex axis as paired ``*_levelN`` (unique values per level) and ``*_labelN`` (per-column
    integer codes) datasets, all of which ``h5py`` can read directly. This recovers, for each data column, which
    bodypart and which coordinate (``x``/``y``/``likelihood``) it holds -- without ``pandas``.

    Args:
        group: The DLC predictions group.
        h5_path: The path to the file, used only for error messages.

    Returns:
        A ``(per_column_bodypart, per_column_coordinate)`` tuple of equal-length lists, one entry per data column.

    Raises:
        ValueError: If the column-label datasets are absent (e.g. a PyTables ``table``-format file) or cannot be
            interpreted, which means the reader needs extending or the file re-exported (fixed-format ``.h5`` or CSV).
    """
    levels, labels = _collect_multiindex_datasets(group=group)
    shared = sorted(set(levels) & set(labels))
    if not shared:
        message = (
            f"Unable to read pupil tracking from '{h5_path.name}'. The DeepLabCut prediction file does not expose a "
            f"fixed-format column index ('axis0_level*'/'axis0_label*'); re-export it as a fixed-format '.h5' or CSV."
        )
        console.error(message=message, error=ValueError)

    coordinate_suffix = next((suffix for suffix in shared if set(_COORDINATES).issubset(set(levels[suffix]))), None)
    bodypart_suffix = next((suffix for suffix in shared if suffix != coordinate_suffix), None)
    if coordinate_suffix is None or bodypart_suffix is None:
        message = (
            f"Unable to read pupil tracking from '{h5_path.name}'. Could not identify the bodypart and coordinate "
            f"levels of the DeepLabCut column index."
        )
        console.error(message=message, error=ValueError)

    bodypart_levels, bodypart_codes = levels[bodypart_suffix], labels[bodypart_suffix]
    coordinate_levels, coordinate_codes = levels[coordinate_suffix], labels[coordinate_suffix]
    per_column_bodypart = [bodypart_levels[code] for code in bodypart_codes]
    per_column_coordinate = [coordinate_levels[code] for code in coordinate_codes]
    return per_column_bodypart, per_column_coordinate


def _collect_multiindex_datasets(group: h5py.Group) -> tuple[dict[str, list[str]], dict[str, NDArray[np.intp]]]:
    """Collects the ``*_levelN`` (decoded strings) and ``*_labelN`` (integer codes) column-index datasets from a group.

    Prefers the ``axis0`` datasets (the column axis) and falls back to any ``*_level*``/``*_label*`` datasets, keying
    each by its trailing integer so a level and its codes share a key.

    Args:
        group: The DLC predictions group.

    Returns:
        A ``(levels, labels)`` tuple: ``levels`` maps each level suffix to its decoded unique values, ``labels`` maps
        each level suffix to its per-column integer codes.
    """
    for prefix in ("axis0", ""):
        levels: dict[str, list[str]] = {}
        labels: dict[str, NDArray[np.intp]] = {}
        for key in group:
            node = group[key]
            if not isinstance(node, h5py.Dataset) or (prefix and not key.startswith(prefix)):
                continue
            # Reads only after the name identifies the dataset as an index: the fallback pass has no prefix to filter
            # on, so an unconditional read here would pull the whole prediction matrix into memory just to discard it.
            match = re.search(r"_level(\d+)$", key)
            if match is not None:
                levels[match.group(1)] = _decode_strings(node[()])
                continue
            match = re.search(r"_label(\d+)$", key)
            if match is not None:
                labels[match.group(1)] = np.asarray(node[()], dtype=np.intp)
        if levels and labels:
            return levels, labels
    return {}, {}


def _resolve_value_matrix(group: h5py.Group, column_count: int, h5_path: Path) -> tuple[h5py.Dataset, bool]:
    """Resolves the float value matrix dataset and its orientation for lazy, column-wise reads.

    Args:
        group: The DLC predictions group.
        column_count: The number of data columns reconstructed from the column index, used to orient the matrix.
        h5_path: The path to the file, used only for error messages.

    Returns:
        A ``(dataset, frames_first)`` tuple. ``frames_first`` is True when the matrix is stored as
        ``(frame_count, column_count)`` (column ``j`` is ``dataset[:, j]``) and False when stored transposed as
        ``(column_count, frame_count)`` (column ``j`` is ``dataset[j, :]``).

    Raises:
        ValueError: If no value matrix dataset is present.
    """
    values = group.get("block0_values")
    if isinstance(values, h5py.Dataset):
        # pandas can store a block transposed; orient by matching whichever dimension equals the column count.
        frames_first = values.shape[1] == column_count
        return values, frames_first

    message = (
        f"Unable to read pupil tracking from '{h5_path.name}'. The DeepLabCut prediction file does not expose a "
        f"fixed-format value matrix ('block0_values'); re-export it as a fixed-format '.h5' or CSV."
    )
    console.error(message=message, error=ValueError)
    raise ValueError(message)  # pragma: no cover - console.error is NoReturn; satisfies the return typing.


def _decode_strings(raw: NDArray[np.generic]) -> list[str]:
    """Decodes an ``h5py`` string dataset (bytes or fixed-width bytes) into a list of Python strings.

    Args:
        raw: The raw array read from an ``h5py`` string dataset.

    Returns:
        The decoded strings.
    """
    return [value.decode() if isinstance(value, bytes) else str(value) for value in raw.tolist()]


def _compute_pupil_metrics(points: dict[str, NDArray[np.float64]]) -> dict[str, NDArray[np.generic]]:
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
    # A feature is measurable exactly when its fit is determined and well enough conditioned to trust, so 'valid' and
    # 'not occluded' are one condition: the geometry is computed for precisely the frames that are not blinks.
    pupil_center, pupil_semi_a, pupil_semi_b, pupil_condition, pupil_residual, pupil_valid = _fit_ring_ellipse(
        points=points, names=_PUPIL_POINTS
    )
    pupil_width, pupil_height = 2.0 * _norm(pupil_semi_a), 2.0 * _norm(pupil_semi_b)
    pupil_area = np.pi * _cross(pupil_semi_a, pupil_semi_b)
    pupil_diameter = (pupil_width + pupil_height) / 2.0

    eye_center, eye_semi_a, eye_semi_b, _, _, eye_valid = _fit_ring_ellipse(points=points, names=_EYE_POINTS)
    eye_width, eye_height = 2.0 * _norm(eye_semi_a), 2.0 * _norm(eye_semi_b)
    # Eye openness is the eye's vertical-to-horizontal aspect ratio, which is invariant to camera distance. A
    # zero-width eye is a degenerate fit rather than a closed eye, so it yields NaN instead of dividing.
    eye_openness = np.divide(eye_height, eye_width, out=np.full_like(eye_height, np.nan), where=eye_width > 0.0)

    # With no confident, non-degenerate eye fit anywhere in the session there is no openness baseline to compare
    # against, so it resolves to NaN and the openness term drops out of the flag below, leaving the eye's visibility
    # to carry it.
    confident_openness = eye_openness[eye_valid]
    baseline = float(np.nanmedian(confident_openness)) if np.isfinite(confident_openness).any() else np.nan

    reflection = points[_REFLECTION_POINT][:, :2]
    reflection_valid = points[_REFLECTION_POINT][:, 2] >= _LIKELIHOOD_THRESHOLD

    # A blink is read from the eye and its cornea alone, never from the pupil. Something covering the eye takes the
    # eye ring, the corneal reflection, and the opening down together, and the cause does not change the consequence:
    # a lid and a paw read the same. The pupil is deliberately excluded because a pupil that vanishes under an OPEN
    # eye has outgrown the aperture rather than been hidden by a lid, and folding that in here would delete the most
    # dilated pupils from the arousal signal exactly when arousal is highest.
    is_blink = (
        ~(eye_valid & reflection_valid) | ~np.isfinite(eye_openness) | (eye_openness < _BLINK_FRACTION * baseline)
    )

    # Behind a shut lid there is no pupil to measure, so a pupil fit that happens to converge on a blink frame is
    # reporting on points the eye was covering. The pupil columns answer to the blink as well as to their own fit;
    # the eye columns answer only to theirs, since their openness is what detects the blink in the first place.
    pupil_measured = pupil_valid & ~is_blink

    # The remaining way to lose the pupil is for it to outgrow the palpebral opening, which then clips it past what
    # the surviving arc can reconstruct. The eye is plainly open, so this is dilation rather than a blink, and the two
    # flags partition every unmeasured frame between them: a frame is measured, dilated, or blinked, never two.
    is_dilated = ~is_blink & ~pupil_measured

    pupil_reflection_offset = pupil_center - reflection
    pupil_reflection_offset = np.where((pupil_measured & reflection_valid)[:, None], pupil_reflection_offset, np.nan)
    # Normalizes the pupil's offset to the eye's semi-axes. A degenerate zero-extent eye divides to NaN on the
    # collapsed axis rather than to an infinity, matching how eye openness handles the same fit.
    eye_semi_axes = np.stack([eye_width / 2.0, eye_height / 2.0], axis=1)
    pupil_in_eye = np.divide(
        pupil_center - eye_center,
        eye_semi_axes,
        out=np.full_like(eye_semi_axes, np.nan),
        where=eye_semi_axes > 0.0,
    )
    pupil_in_eye = np.where((pupil_measured & eye_valid)[:, None], pupil_in_eye, np.nan)

    # Masks geometry of frames whose source points were gated out so downstream consumers see NaN, not a bad fit.
    pupil_metrics = _mask_invalid(
        {
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
        {
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
    condition number measures how far the surviving arc has to reach to pin the rest of the ellipse, and frames whose
    arc is too narrow to carry a measurement are rejected rather than fitted.

    Args:
        points: A mapping from each canonical bodypart to its ``(frame_count, 3)`` ``(x, y, likelihood)`` array.
        names: The feature's ring bodyparts, in ring order.

    Returns:
        A ``(center, semi_a, semi_b, condition, residual, valid)`` tuple. ``center``, ``semi_a`` and ``semi_b`` are
        ``(frame_count, 2)`` arrays, NaN wherever the fit was rejected. ``condition`` is the per-frame fit condition
        number (NaN where rejected); ``residual`` is the per-frame RMS point-to-ellipse distance in pixels, NaN where
        rejected or where the fit was exactly determined (too few points to over-constrain it). ``valid`` is the
        per-frame mask of frames that were fitted.
    """
    frame_count = points[names[0]].shape[0]
    angles = _ring_angles(point_count=len(names))
    confident = np.stack([points[name][:, 2] >= _LIKELIHOOD_THRESHOLD for name in names], axis=1)
    coordinates = np.stack([points[name][:, :2] for name in names], axis=1)

    center = np.full((frame_count, 2), np.nan)
    semi_a = np.full((frame_count, 2), np.nan)
    semi_b = np.full((frame_count, 2), np.nan)
    condition = np.full(frame_count, np.nan)
    residual = np.full(frame_count, np.nan)
    valid = np.zeros(frame_count, dtype=np.bool_)

    # Frames that lost the same points share a design matrix, and therefore a conditioning verdict, so each distinct
    # occlusion pattern is solved once for every frame that carries it.
    patterns, inverse = np.unique(confident, axis=0, return_inverse=True)
    for index, pattern in enumerate(patterns):
        kept = np.flatnonzero(pattern)
        if kept.size < _MINIMUM_PERIMETER_POINTS:
            continue
        design = np.stack([np.ones(kept.size), np.cos(angles[kept]), np.sin(angles[kept])], axis=1)
        condition_number = float(np.linalg.cond(design))
        if not condition_number <= _MAXIMUM_FIT_CONDITION:
            continue
        rows = np.flatnonzero(np.ravel(inverse) == index)
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


def _cross(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
    """Computes the per-frame magnitude of the 2-D cross product of two ``(frame_count, 2)`` arrays of vectors.

    Args:
        first: The ``(frame_count, 2)`` array of the first vectors.
        second: The ``(frame_count, 2)`` array of the second vectors.

    Returns:
        The per-frame absolute cross-product magnitudes, which are the areas of the parallelograms the pairs span.
    """
    return np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def _mask_invalid(metrics: dict[str, NDArray[np.float64]], valid: NDArray[np.bool_]) -> dict[str, NDArray[np.float64]]:
    """Replaces metric values with NaN wherever the per-frame validity mask is False.

    Args:
        metrics: A mapping from each metric column name to its per-frame array.
        valid: The per-frame validity mask.

    Returns:
        A new mapping with the same keys whose values are NaN where ``valid`` is False.
    """
    return {name: np.where(valid, values, np.nan) for name, values in metrics.items()}
