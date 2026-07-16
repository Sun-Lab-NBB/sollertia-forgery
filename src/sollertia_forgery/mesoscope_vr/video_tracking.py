"""Provides the Mesoscope-VR video-tracking function donated to the system-agnostic video-processing pipeline.

The video pipeline resolves this function from the registry hub and runs it once per session, expecting it to do all
the work (mirroring how a system donates its forging assembler). It post-processes externally-produced DeepLabCut
(DLC) pose predictions for the Mesoscope-VR face camera into per-frame pupil and eye metrics.

Notes:
    sollertia-forgery never runs DeepLabCut: DLC pins ``numpy<2`` while slf is ``numpy>=2`` / Python 3.14, so DLC
    cannot be imported in-process. DLC runs out-of-band (its own conda environment) and writes its ``.h5`` predictions
    straight into the session's processed video-data directory -- the same well-defined location the timestamp stage
    writes to, consistent with how every other pipeline keeps its intermediates and finals in one ``processed_data``
    directory. This module only READS that output -- via ``h5py`` and ``numpy`` (never ``pandas`` and never the
    ``deeplabcut`` library), reading only the columns it needs without materializing the full table -- and only when
    the file is present. With no ``.h5`` the function is a no-op, so the pipeline can run it unconditionally.

    The model that produced the predictions is irrelevant here: the pupil project's prediction file is identified by
    its DeepLabCut project name, and this function only requires that the ``.h5`` carries the nine canonical bodyparts
    below. They are spelled in a DLC-friendly format (``eye_left`` etc.):
    ``reflection`` (the corneal reflection), the four pupil-perimeter points (``pupil_top``/``pupil_bottom``/
    ``pupil_left``/``pupil_right``), and the four eye-perimeter points (``eye_top``/``eye_bottom``/``eye_left``/
    ``eye_right``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import h5py
import numpy as np
import polars as pl
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

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
"""The colloquial camera name whose recordings carry the eye. Used to locate the camera's per-frame timestamp feather
(for attaching ``time_us``) and to name this function's output feather (``{camera}_{target}.feather``)."""

_PUPIL_TARGET: str = "pupil"
"""The tracking-target label used to name this function's output feather (``{camera}_{target}.feather``)."""

_REFLECTION_POINT: str = "reflection"
"""The canonical bodypart for the corneal (infrared) reflection, used as a motion-robust positional reference."""

_PUPIL_POINTS: tuple[str, str, str, str] = ("pupil_top", "pupil_bottom", "pupil_left", "pupil_right")
"""The four canonical pupil-perimeter bodyparts, ordered ``(top, bottom, left, right)``."""

_EYE_POINTS: tuple[str, str, str, str] = ("eye_top", "eye_bottom", "eye_left", "eye_right")
"""The four canonical eye-perimeter bodyparts, ordered ``(top, bottom, left, right)``."""

_CANONICAL_POINTS: tuple[str, ...] = (_REFLECTION_POINT, *_PUPIL_POINTS, *_EYE_POINTS)
"""All nine canonical bodyparts the DLC ``.h5`` must provide for this function to parse it."""

_LIKELIHOOD_THRESHOLD: float = 0.6
"""The minimum DLC likelihood for a point to be trusted in a frame. An ellipse that needs a below-threshold point
yields NaN metrics for that frame and flags it as low-confidence (which, for the pupil, also reads as a blink)."""

_BLINK_FRACTION: float = 0.5
"""The fraction of the session-median eye openness below which a frame is flagged as a blink."""

_CAMERA_TIMESTAMP_SUFFIX: str = "_timestamps.feather"
"""The suffix the video pipeline appends to a camera's manifest name to form its canonical per-frame timestamp feather
(e.g. ``face_camera_timestamps.feather``). Duplicated here so this donor never imports the video worker package."""

_COORDINATES: tuple[str, str, str] = ("x", "y", "likelihood")
"""The three DLC per-bodypart coordinate columns, used to recognize the coordinate level of the column MultiIndex."""


@dataclass
class _PupilTrackingProvenance(YamlConfig):
    """Captures the provenance of one pupil-tracking output for reproducibility, written beside the output feather."""

    h5_file: str = ""
    """The filename of the DLC ``.h5`` the metrics were parsed from."""
    bodyparts: list[str] = field(default_factory=list)
    """The canonical bodyparts that were read from the ``.h5``."""
    frame_count: int = 0
    """The number of frames (rows) parsed from the ``.h5``."""
    likelihood_threshold: float = 0.0
    """The likelihood threshold applied when gating points."""
    blink_fraction: float = 0.0
    """The eye-openness fraction used as the blink threshold."""


def process_mesoscope_video_tracking(session: SessionData, output_directory: Path) -> None:
    """Post-processes the Mesoscope-VR face-camera DLC predictions into per-frame pupil and eye metrics.

    Locates the pupil project's externally-produced DLC ``.h5`` beside the face-camera video in the session's raw
    camera_data directory (where the acquisition rig writes it during preprocessing, so it travels with the raw data);
    if none is present, returns without doing anything (the stage is optional and gated on detecting the prediction
    file). Otherwise reads the nine canonical bodyparts, fits an ellipse to the pupil and to the eye each frame,
    derives a blink flag from the variation in eye shape, derives motion-robust eye-position signals from the pupil
    relative to the eye and the corneal reflection, and writes the results into a ``{camera}_pupil.feather`` (plus a
    provenance sidecar) in the processed video-data directory.

    Args:
        session: The loaded session whose pupil DLC predictions are post-processed. Its raw camera_data directory
            supplies the DLC ``.h5``.
        output_directory: The processed video-data directory (``session.processed_data.video_data_path``) where the
            pupil feather and its provenance sidecar are written, and where the camera-timestamp feather (if already
            produced by the timestamp stage) is read to attach per-frame ``time_us``.

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
    frame_times = _load_frame_times(output_directory=output_directory, frame_count=frame_count)

    columns: dict[str, NDArray[np.generic]] = {"frame": np.arange(frame_count, dtype=np.uint64)}
    if frame_times is not None:
        columns["time_us"] = frame_times
    columns.update(pupil_metrics)

    output_path = output_directory.joinpath(f"{PUPIL_CAMERA_NAME}_{_PUPIL_TARGET}.feather")
    pl.DataFrame(columns).write_ipc(file=output_path)

    _PupilTrackingProvenance(
        h5_file=h5_path.name,
        bodyparts=list(_CANONICAL_POINTS),
        frame_count=frame_count,
        likelihood_threshold=_LIKELIHOOD_THRESHOLD,
        blink_fraction=_BLINK_FRACTION,
    ).to_yaml(file_path=output_directory.joinpath(f"{PUPIL_CAMERA_NAME}_{_PUPIL_TARGET}_provenance.yaml"))

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
    with h5py.File(str(h5_path), "r") as h5_file:
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

    coordinate_suffix = next(
        (suffix for suffix in shared if {"x", "y", "likelihood"}.issubset(set(levels[suffix]))), None
    )
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
            data = node[()]
            match = re.search(r"_level(\d+)$", key)
            if match is not None:
                levels[match.group(1)] = _decode_strings(data)
                continue
            match = re.search(r"_label(\d+)$", key)
            if match is not None:
                labels[match.group(1)] = np.asarray(data, dtype=np.intp)
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
    """Computes per-frame pupil and eye metrics from the nine canonical points.

    Fits an axis-conjugate ellipse to the pupil and to the eye, gates each frame on point likelihood, flags blinks
    from the variation in eye openness, and derives motion-robust eye-position signals (pupil relative to the eye and
    to the corneal reflection).

    Args:
        points: A mapping from each canonical bodypart to its ``(frame_count, 3)`` ``(x, y, likelihood)`` array.

    Returns:
        A mapping from each output column name to its per-frame array, ready to assemble into the output feather.
    """
    pupil_valid = _points_valid(points=points, names=_PUPIL_POINTS)
    pupil_center, pupil_width, pupil_height, pupil_angle, pupil_area = _ellipse_from_cardinal(
        top=points["pupil_top"][:, :2],
        bottom=points["pupil_bottom"][:, :2],
        left=points["pupil_left"][:, :2],
        right=points["pupil_right"][:, :2],
    )
    pupil_diameter = (pupil_width + pupil_height) / 2.0

    eye_valid = _points_valid(points=points, names=_EYE_POINTS)
    eye_center, eye_width, eye_height, _eye_angle, eye_area = _ellipse_from_cardinal(
        top=points["eye_top"][:, :2],
        bottom=points["eye_bottom"][:, :2],
        left=points["eye_left"][:, :2],
        right=points["eye_right"][:, :2],
    )
    # Eye openness is the eye's vertical-to-horizontal aspect ratio, which is invariant to camera distance.
    eye_openness = np.where(eye_width > 0.0, eye_height / eye_width, np.nan)

    # A blink shrinks the eye opening relative to its session-typical value; loss of the pupil points (occlusion) is
    # an additional cue. Frames where the eye itself cannot be seen are treated as blinks.
    confident_openness = eye_openness[eye_valid]
    baseline = float(np.nanmedian(confident_openness)) if confident_openness.size else np.nan
    pupil_min_likelihood = np.min(np.stack([points[name][:, 2] for name in _PUPIL_POINTS], axis=1), axis=1)
    is_blink = (eye_openness < _BLINK_FRACTION * baseline) | (pupil_min_likelihood < _LIKELIHOOD_THRESHOLD) | ~eye_valid

    reflection = points[_REFLECTION_POINT][:, :2]
    reflection_valid = points[_REFLECTION_POINT][:, 2] >= _LIKELIHOOD_THRESHOLD
    pupil_cr = pupil_center - reflection
    pupil_cr = np.where((pupil_valid & reflection_valid)[:, None], pupil_cr, np.nan)
    pupil_in_eye = (pupil_center - eye_center) / np.stack([eye_width / 2.0, eye_height / 2.0], axis=1)
    pupil_in_eye = np.where((pupil_valid & eye_valid)[:, None], pupil_in_eye, np.nan)

    min_likelihood = np.min(np.stack([points[name][:, 2] for name in _CANONICAL_POINTS], axis=1), axis=1)

    # Masks geometry of frames whose source points were gated out so downstream consumers see NaN, not a bad fit.
    pupil_metrics = _mask_invalid(
        {
            "pupil_center_x": pupil_center[:, 0],
            "pupil_center_y": pupil_center[:, 1],
            "pupil_diameter": pupil_diameter,
            "pupil_area": pupil_area,
            "pupil_angle": pupil_angle,
        },
        valid=pupil_valid,
    )
    eye_metrics = _mask_invalid(
        {
            "eye_center_x": eye_center[:, 0],
            "eye_center_y": eye_center[:, 1],
            "eye_width": eye_width,
            "eye_height": eye_height,
            "eye_area": eye_area,
            "eye_openness": eye_openness,
        },
        valid=eye_valid,
    )

    return {
        **pupil_metrics,
        **eye_metrics,
        "is_blink": is_blink,
        "reflection_x": np.where(reflection_valid, reflection[:, 0], np.nan),
        "reflection_y": np.where(reflection_valid, reflection[:, 1], np.nan),
        "pupil_cr_x": pupil_cr[:, 0],
        "pupil_cr_y": pupil_cr[:, 1],
        "pupil_in_eye_x": pupil_in_eye[:, 0],
        "pupil_in_eye_y": pupil_in_eye[:, 1],
        "min_likelihood": min_likelihood,
        "low_confidence": min_likelihood < _LIKELIHOOD_THRESHOLD,
    }


def _points_valid(points: dict[str, NDArray[np.float64]], names: tuple[str, ...]) -> NDArray[np.bool_]:
    """Computes the per-frame mask that is True only where every named point clears the likelihood threshold.

    Args:
        points: A mapping from each canonical bodypart to its ``(frame_count, 3)`` ``(x, y, likelihood)`` array.
        names: The bodyparts that must all be confident for the frame to be valid.

    Returns:
        A per-frame boolean array.
    """
    valid = np.ones(points[names[0]].shape[0], dtype=np.bool_)
    for name in names:
        valid &= points[name][:, 2] >= _LIKELIHOOD_THRESHOLD
    return valid


def _ellipse_from_cardinal(
    top: NDArray[np.float64],
    bottom: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fits a per-frame ellipse to four cardinal perimeter points via their two conjugate semi-diameters.

    The four cardinal points determine a (possibly rotated) ellipse: its center is their centroid, its horizontal
    extent and orientation come from the left-to-right chord, and its vertical extent from the top-to-bottom chord. A
    general conic least-squares fit would need five or more perimeter points and is a future-model option.

    Args:
        top: The ``(frame_count, 2)`` array of the top point's ``(x, y)`` per frame.
        bottom: The ``(frame_count, 2)`` array of the bottom point's ``(x, y)`` per frame.
        left: The ``(frame_count, 2)`` array of the left point's ``(x, y)`` per frame.
        right: The ``(frame_count, 2)`` array of the right point's ``(x, y)`` per frame.

    Returns:
        A ``(center, width, height, angle, area)`` tuple: ``center`` is ``(frame_count, 2)``; the rest are per-frame
        1-D arrays. ``width`` is the left-right chord length, ``height`` the top-bottom chord length, ``angle`` the
        left-right chord orientation in radians, and ``area`` the enclosed ellipse area.
    """
    center = (top + bottom + left + right) / 4.0
    horizontal = right - left
    vertical = bottom - top
    width = np.hypot(horizontal[:, 0], horizontal[:, 1])
    height = np.hypot(vertical[:, 0], vertical[:, 1])
    angle = np.arctan2(horizontal[:, 1], horizontal[:, 0])
    area = np.pi * (width / 2.0) * (height / 2.0)
    return center, width, height, angle, area


def _mask_invalid(metrics: dict[str, NDArray[np.float64]], valid: NDArray[np.bool_]) -> dict[str, NDArray[np.float64]]:
    """Replaces metric values with NaN wherever the per-frame validity mask is False.

    Args:
        metrics: A mapping from each metric column name to its per-frame array.
        valid: The per-frame validity mask.

    Returns:
        A new mapping with the same keys whose values are NaN where ``valid`` is False.
    """
    return {name: np.where(valid, values, np.nan) for name, values in metrics.items()}


def _load_frame_times(output_directory: Path, frame_count: int) -> NDArray[np.generic] | None:
    """Loads the face camera's per-frame ``time_us`` from its timestamp feather, if the timestamp stage has run.

    Args:
        output_directory: The processed video-data directory holding the camera-timestamp feathers.
        frame_count: The number of pose frames, used to confirm the timestamps align one-to-one with the predictions.

    Returns:
        The per-frame timestamp array when a matching, equal-length timestamp feather exists, otherwise None (so the
        output carries only the frame index).
    """
    timestamp_path = output_directory.joinpath(f"{PUPIL_CAMERA_NAME}{_CAMERA_TIMESTAMP_SUFFIX}")
    if not timestamp_path.is_file():
        return None

    frame = pl.read_ipc(source=timestamp_path, memory_map=True)
    if frame.height != frame_count:
        console.echo(
            message=(
                f"The '{PUPIL_CAMERA_NAME}' timestamp feather has {frame.height} row(s) but the DeepLabCut "
                f"predictions have {frame_count} frame(s); writing pupil tracking without 'time_us'."
            ),
            level=LogLevel.WARNING,
        )
        return None

    column = "time_us" if "time_us" in frame.columns else frame.columns[0]
    return frame.get_column(column).to_numpy()
