"""Contains tests for the Mesoscope-VR pupil and eye tracking worker donated to the video-processing pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sollertia_forgery.mesoscope_vr.video_tracking import (
    _EYE_POINTS,
    _PUPIL_POINTS,
    PUPIL_CAMERA_NAME,
    _CANONICAL_POINTS,
    _REFLECTION_POINT,
    PupilColumn,
    _fit_ring_ellipse,
    _read_points_from_h5,
    _compute_pupil_metrics,
    process_mesoscope_video_tracking,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_PUPIL_CENTER: tuple[float, float] = (100.0, 80.0)
"""The center around which every synthetic pupil ellipse is generated, in face-camera pixels."""

_PUPIL_SEMI_A: tuple[float, float] = (10.0, 0.0)
"""The horizontal conjugate semi-diameter of the synthetic pupil, making the pupil a circle of radius ten."""

_PUPIL_SEMI_B: tuple[float, float] = (0.0, 10.0)
"""The vertical conjugate semi-diameter of the synthetic pupil."""

_EYE_CENTER: tuple[float, float] = (104.0, 82.0)
"""The center around which every synthetic eye ellipse is generated, offset from the pupil so the in-eye columns are
non-zero."""

_EYE_SEMI_A: tuple[float, float] = (40.0, 0.0)
"""The horizontal conjugate semi-diameter of the synthetic eye, making the eye eighty pixels wide."""

_EYE_SEMI_B: tuple[float, float] = (0.0, 10.0)
"""The vertical conjugate semi-diameter of the synthetic eye, making the eye twenty pixels tall."""

_REFLECTION_POSITION: tuple[float, float] = (97.0, 77.0)
"""The corneal reflection position every synthetic frame records."""

_CONFIDENT: float = 0.99
"""The DeepLabCut likelihood generated for a point that clears the tracking gate."""

_UNCONFIDENT: float = 0.10
"""The DeepLabCut likelihood generated for a point that fails the tracking gate."""

_OUTPUT_FILENAME: str = f"{PUPIL_CAMERA_NAME}_pupil.feather"
"""The name of the feather the tracking worker writes into the processed video-data directory."""


def _ring_positions(
    center: tuple[float, float], semi_a: tuple[float, float], semi_b: tuple[float, float], count: int
) -> NDArray[np.float64]:
    """Generates one feature's ring point positions on the ellipse the given conjugate semi-diameters describe.

    Args:
        center: The ellipse center in pixels.
        semi_a: The first conjugate semi-diameter.
        semi_b: The second conjugate semi-diameter.
        count: The number of evenly spaced ring points to generate.

    Returns:
        The ``(count, 2)`` array of ring point positions, in the ring order in which the worker assigns angles.
    """
    angles = 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    return (
        np.asarray(center, dtype=np.float64)
        + np.cos(angles)[:, None] * np.asarray(semi_a, dtype=np.float64)
        + np.sin(angles)[:, None] * np.asarray(semi_b, dtype=np.float64)
    )


def _frame_specification(**overrides: object) -> dict[str, object]:
    """Builds one synthetic frame specification, starting from a fully confident, fully visible eye.

    Args:
        **overrides: The specification entries to replace, named after the defaults this builder supplies.

    Returns:
        The frame specification the point builder consumes.
    """
    specification: dict[str, object] = {
        "pupil_center": _PUPIL_CENTER,
        "pupil_semi_a": _PUPIL_SEMI_A,
        "pupil_semi_b": _PUPIL_SEMI_B,
        "eye_center": _EYE_CENTER,
        "eye_semi_a": _EYE_SEMI_A,
        "eye_semi_b": _EYE_SEMI_B,
        "reflection": _REFLECTION_POSITION,
        "pupil_confident": tuple(range(len(_PUPIL_POINTS))),
        "eye_confident": tuple(range(len(_EYE_POINTS))),
        "reflection_confident": True,
    }
    specification.update(overrides)
    return specification


def _build_points(specifications: Sequence[dict[str, object]]) -> dict[str, NDArray[np.float64]]:
    """Builds the thirteen canonical point arrays from a sequence of per-frame specifications.

    Args:
        specifications: One specification per frame, in acquisition order.

    Returns:
        A mapping from each canonical bodypart to its ``(frame_count, 3)`` array of horizontal position, vertical
        position, and likelihood.
    """
    points: dict[str, list[list[float]]] = {name: [] for name in _CANONICAL_POINTS}
    for specification in specifications:
        pupil = _ring_positions(
            center=specification["pupil_center"],  # type: ignore[arg-type]
            semi_a=specification["pupil_semi_a"],  # type: ignore[arg-type]
            semi_b=specification["pupil_semi_b"],  # type: ignore[arg-type]
            count=len(_PUPIL_POINTS),
        )
        eye = _ring_positions(
            center=specification["eye_center"],  # type: ignore[arg-type]
            semi_a=specification["eye_semi_a"],  # type: ignore[arg-type]
            semi_b=specification["eye_semi_b"],  # type: ignore[arg-type]
            count=len(_EYE_POINTS),
        )
        for index, name in enumerate(_PUPIL_POINTS):
            confident = index in specification["pupil_confident"]  # type: ignore[operator]
            points[name].append([pupil[index, 0], pupil[index, 1], _CONFIDENT if confident else _UNCONFIDENT])
        for index, name in enumerate(_EYE_POINTS):
            confident = index in specification["eye_confident"]  # type: ignore[operator]
            points[name].append([eye[index, 0], eye[index, 1], _CONFIDENT if confident else _UNCONFIDENT])
        reflection = specification["reflection"]
        points[_REFLECTION_POINT].append(
            [
                reflection[0],  # type: ignore[index]
                reflection[1],  # type: ignore[index]
                _CONFIDENT if specification["reflection_confident"] else _UNCONFIDENT,
            ]
        )
    return {name: np.asarray(rows, dtype=np.float64) for name, rows in points.items()}


@pytest.fixture
def tracking_session(experiment_session: SessionData) -> SessionData:
    """Prepares the session's raw camera and processed video directories that the tracking worker reads and writes.

    Args:
        experiment_session: The acquired Mesoscope-VR experiment session.

    Returns:
        The same session, with both directories present on disk.
    """
    experiment_session.raw_data.camera_data_path.mkdir(parents=True, exist_ok=True)
    experiment_session.processed_data.video_data_path.mkdir(parents=True, exist_ok=True)
    return experiment_session


def test_process_mesoscope_video_tracking_skips_when_no_prediction_file_is_present(
    tracking_session: SessionData,
) -> None:
    process_mesoscope_video_tracking(
        session=tracking_session, output_directory=tracking_session.processed_data.video_data_path
    )

    assert list(tracking_session.processed_data.video_data_path.iterdir()) == []


def test_process_mesoscope_video_tracking_writes_every_pupil_column(
    tracking_session: SessionData, write_dlc_predictions: Callable[..., Path]
) -> None:
    points = _build_points(specifications=[_frame_specification(), _frame_specification()])
    write_dlc_predictions(
        tracking_session.raw_data.camera_data_path.joinpath("face_eye_tracking_predictions.h5"), points
    )

    process_mesoscope_video_tracking(
        session=tracking_session, output_directory=tracking_session.processed_data.video_data_path
    )

    written = pl.read_ipc(tracking_session.processed_data.video_data_path.joinpath(_OUTPUT_FILENAME))
    assert written.columns == [column.value for column in PupilColumn]
    assert written.height == 2
    assert written.schema[PupilColumn.PUPIL_DIAMETER_PX] == pl.Float32
    assert written.schema[PupilColumn.BLINKING_STATE] == pl.Boolean
    assert written[PupilColumn.PUPIL_DIAMETER_PX].to_list() == pytest.approx([20.0, 20.0], abs=1e-3)
    assert written[PupilColumn.PUPIL_AREA_PX2].to_list() == pytest.approx([np.pi * 100.0] * 2, rel=1e-5)
    # Each center column carries its own coordinate, so the horizontal and vertical components never swap places.
    assert written[PupilColumn.PUPIL_CENTER_X_PX].to_list() == pytest.approx([100.0, 100.0], abs=1e-3)
    assert written[PupilColumn.PUPIL_CENTER_Y_PX].to_list() == pytest.approx([80.0, 80.0], abs=1e-3)
    assert written[PupilColumn.EYE_CENTER_X_PX].to_list() == pytest.approx([104.0, 104.0], abs=1e-3)
    assert written[PupilColumn.EYE_CENTER_Y_PX].to_list() == pytest.approx([82.0, 82.0], abs=1e-3)
    assert written[PupilColumn.EYE_WIDTH_PX].to_list() == pytest.approx([80.0, 80.0], abs=1e-3)
    assert written[PupilColumn.EYE_HEIGHT_PX].to_list() == pytest.approx([20.0, 20.0], abs=1e-3)
    assert written[PupilColumn.EYE_OPENNESS].to_list() == pytest.approx([0.25, 0.25], abs=1e-5)
    assert written[PupilColumn.BLINKING_STATE].to_list() == [False, False]
    assert written[PupilColumn.DILATION_STATE].to_list() == [False, False]
    assert written[PupilColumn.REFLECTION_X_PX].to_list() == pytest.approx([97.0, 97.0], abs=1e-3)
    assert written[PupilColumn.REFLECTION_Y_PX].to_list() == pytest.approx([77.0, 77.0], abs=1e-3)
    assert written[PupilColumn.PUPIL_REFLECTION_OFFSET_X_PX].to_list() == pytest.approx([3.0, 3.0], abs=1e-3)
    assert written[PupilColumn.PUPIL_REFLECTION_OFFSET_Y_PX].to_list() == pytest.approx([3.0, 3.0], abs=1e-3)
    # The pupil center sits four pixels left of and two pixels above the eye center, whose semi-axes are forty by ten.
    assert written[PupilColumn.PUPIL_IN_EYE_X].to_list() == pytest.approx([-0.1, -0.1], abs=1e-4)
    assert written[PupilColumn.PUPIL_IN_EYE_Y].to_list() == pytest.approx([-0.2, -0.2], abs=1e-4)


def test_process_mesoscope_video_tracking_reads_the_natural_sort_first_prediction_file(
    tracking_session: SessionData, write_dlc_predictions: Callable[..., Path]
) -> None:
    camera_data = tracking_session.raw_data.camera_data_path
    write_dlc_predictions(
        camera_data.joinpath("face_eye_tracking_2.h5"), _build_points(specifications=[_frame_specification()] * 3)
    )
    write_dlc_predictions(
        camera_data.joinpath("face_eye_tracking_10.h5"), _build_points(specifications=[_frame_specification()] * 7)
    )

    process_mesoscope_video_tracking(
        session=tracking_session, output_directory=tracking_session.processed_data.video_data_path
    )

    # Natural sort orders the '_2' suffix before the '_10' suffix, so the three-row file is the one read.
    assert pl.read_ipc(tracking_session.processed_data.video_data_path.joinpath(_OUTPUT_FILENAME)).height == 3


def test_read_points_from_h5_rejects_a_file_holding_no_prediction_frame(tmp_path: Path) -> None:
    h5_path = tmp_path.joinpath("face_eye_tracking.h5")
    pd.Series([1.0, 2.0]).to_hdf(path_or_buf=h5_path, key="series", format="table")

    with pytest.raises(ValueError, match=re.escape("does not contain a DeepLabCut prediction")):
        _read_points_from_h5(h5_path=h5_path, bodyparts=_CANONICAL_POINTS)


def test_read_points_from_h5_rejects_a_file_holding_a_flat_column_frame(tmp_path: Path) -> None:
    h5_path = tmp_path.joinpath("face_eye_tracking.h5")
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_hdf(path_or_buf=h5_path, key="df", format="table")

    with pytest.raises(ValueError, match=re.escape("does not contain a DeepLabCut prediction")):
        _read_points_from_h5(h5_path=h5_path, bodyparts=_CANONICAL_POINTS)


def test_read_points_from_h5_rejects_a_file_missing_a_canonical_bodypart(
    tmp_path: Path, write_dlc_predictions: Callable[..., Path]
) -> None:
    points = _build_points(specifications=[_frame_specification()])
    del points["pupil_top"]
    h5_path = write_dlc_predictions(tmp_path.joinpath("face_eye_tracking.h5"), points)

    with pytest.raises(ValueError, match=re.escape("does not contain the required")):
        _read_points_from_h5(h5_path=h5_path, bodyparts=_CANONICAL_POINTS)


def test_read_points_from_h5_returns_rows_in_ascending_frame_order(
    tmp_path: Path, write_dlc_predictions: Callable[..., Path]
) -> None:
    h5_path = write_dlc_predictions(
        tmp_path.joinpath("face_eye_tracking.h5"),
        {_REFLECTION_POINT: np.array([[3.0, 30.0, 0.9], [1.0, 10.0, 0.9], [2.0, 20.0, 0.9]])},
    )
    # Rewrites the same frame under a shuffled index, so the reader has to restore acquisition order itself.
    stored = pd.read_hdf(path_or_buf=h5_path)
    stored.index = pd.Index([2, 0, 1])
    stored.to_hdf(path_or_buf=h5_path, key="df_with_missing", format="table")

    points = _read_points_from_h5(h5_path=h5_path, bodyparts=(_REFLECTION_POINT,))

    assert points[_REFLECTION_POINT][:, 0].tolist() == [1.0, 2.0, 3.0]


def test_compute_pupil_metrics_measures_a_tilted_pupil_by_its_semi_diameter_cross_product() -> None:
    # The two semi-diameters are perpendicular, eight and twelve pixels long, and tilted off both image axes, so the
    # ellipse area is pi times the magnitude of their cross product, pi * 96. No sum of products can stand in for that
    # area the way it can for the axis-aligned circle every other synthetic frame carries.
    tilted = _frame_specification(pupil_semi_a=(-4.8, 6.4), pupil_semi_b=(9.6, 7.2))

    metrics = _compute_pupil_metrics(points=_build_points(specifications=[tilted]))

    assert metrics[PupilColumn.PUPIL_AREA_PX2][0] == pytest.approx(np.pi * 96.0, rel=1e-5)
    assert metrics[PupilColumn.PUPIL_DIAMETER_PX][0] == pytest.approx(20.0, abs=1e-3)


def test_compute_pupil_metrics_keeps_a_lost_eye_ring_open_while_the_pupil_resolves() -> None:
    points = _build_points(specifications=[_frame_specification(), _frame_specification(eye_confident=(0, 1))])

    metrics = _compute_pupil_metrics(points=points)

    # A covered eye presents no pupil ring to fit, so a pupil that resolves is positive evidence of an open eye and
    # overrides an eye ring that is merely absent.
    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, False]
    assert metrics[PupilColumn.DILATION_STATE].tolist() == [False, False]
    assert metrics[PupilColumn.PUPIL_DIAMETER_PX][1] == pytest.approx(metrics[PupilColumn.PUPIL_DIAMETER_PX][0])
    # The eye columns answer only to their own fit, which stays lost.
    assert np.isnan(metrics[PupilColumn.EYE_WIDTH_PX][1])


def test_compute_pupil_metrics_flags_a_blink_when_the_eye_ring_and_the_pupil_are_both_lost() -> None:
    points = _build_points(
        specifications=[_frame_specification(), _frame_specification(eye_confident=(0, 1), pupil_confident=(0, 1))]
    )

    metrics = _compute_pupil_metrics(points=points)

    # With no pupil to vouch for an open eye, a lost eye ring carries the flag on its own.
    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, True]
    assert np.isnan(metrics[PupilColumn.PUPIL_DIAMETER_PX][1])
    assert np.isnan(metrics[PupilColumn.EYE_WIDTH_PX][1])


def test_compute_pupil_metrics_keeps_a_lost_reflection_open_while_the_pupil_resolves() -> None:
    points = _build_points(specifications=[_frame_specification(), _frame_specification(reflection_confident=False)])

    metrics = _compute_pupil_metrics(points=points)

    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, False]
    assert np.isnan(metrics[PupilColumn.REFLECTION_X_PX][1])
    assert np.isnan(metrics[PupilColumn.REFLECTION_Y_PX][1])
    # The offset needs both of its endpoints, so it drops with the reflection even on a frame that is not a blink.
    assert np.isnan(metrics[PupilColumn.PUPIL_REFLECTION_OFFSET_X_PX][1])
    # The eye ring is untouched by a lost reflection, so the eye columns stay populated.
    assert metrics[PupilColumn.EYE_WIDTH_PX][1] == pytest.approx(80.0)


def test_compute_pupil_metrics_flags_a_blink_when_the_eye_closes_below_half_its_median_openness() -> None:
    open_frames = [_frame_specification() for _ in range(4)]
    # A one-pixel vertical semi-axis leaves the eye at a tenth of its open aspect ratio, well under the half-median cut.
    closed = _frame_specification(eye_semi_b=(0.0, 1.0))

    metrics = _compute_pupil_metrics(points=_build_points(specifications=[*open_frames, closed]))

    # The closed frame keeps a fully confident pupil ring, so this also pins the openness term as the one term a
    # resolving pupil does not override. A fitted eye measured to be closing is an observation rather than a gap.
    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, False, False, False, True]
    assert metrics[PupilColumn.EYE_OPENNESS][4] == pytest.approx(0.025)
    assert np.isnan(metrics[PupilColumn.PUPIL_CENTER_X_PX][4])
    # Every pupil column answers to the blink, so the motion-robust offset and the normalized in-eye position drop
    # with the geometry rather than reporting a pupil read through a closing lid.
    assert np.isnan(metrics[PupilColumn.PUPIL_REFLECTION_OFFSET_X_PX][4])
    assert np.isnan(metrics[PupilColumn.PUPIL_REFLECTION_OFFSET_Y_PX][4])
    assert np.isnan(metrics[PupilColumn.PUPIL_IN_EYE_X][4])
    assert np.isnan(metrics[PupilColumn.PUPIL_IN_EYE_Y][4])


def test_compute_pupil_metrics_flags_dilation_when_the_pupil_is_lost_under_an_open_eye() -> None:
    points = _build_points(specifications=[_frame_specification(), _frame_specification(pupil_confident=(0, 1))])

    metrics = _compute_pupil_metrics(points=points)

    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, False]
    assert metrics[PupilColumn.DILATION_STATE].tolist() == [False, True]
    assert np.isnan(metrics[PupilColumn.PUPIL_DIAMETER_PX][1])
    assert np.isnan(metrics[PupilColumn.PUPIL_IN_EYE_X][1])
    # The eye is plainly open on a dilation frame, so its own columns survive.
    assert metrics[PupilColumn.EYE_HEIGHT_PX][1] == pytest.approx(20.0)


def test_compute_pupil_metrics_keeps_every_frame_open_when_no_eye_fit_survives_but_the_pupil_does() -> None:
    points = _build_points(
        specifications=[_frame_specification(eye_confident=(0,)), _frame_specification(eye_confident=(1,))]
    )

    metrics = _compute_pupil_metrics(points=points)

    # With no openness baseline anywhere in the session the openness term drops out, leaving the pupil to vouch for
    # every frame the lost eye would otherwise have flagged.
    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False, False]
    assert np.isnan(metrics[PupilColumn.EYE_OPENNESS]).all()
    assert not np.isnan(metrics[PupilColumn.PUPIL_AREA_PX2]).any()


def test_compute_pupil_metrics_returns_not_a_number_for_a_zero_extent_eye_fit() -> None:
    collapsed = _frame_specification(eye_semi_a=(0.0, 0.0), eye_semi_b=(0.0, 0.0))

    metrics = _compute_pupil_metrics(points=_build_points(specifications=[collapsed]))

    assert metrics[PupilColumn.EYE_WIDTH_PX][0] == pytest.approx(0.0)
    assert np.isnan(metrics[PupilColumn.EYE_OPENNESS][0])
    # A collapsed eye divides to not-a-number rather than to an infinity on both normalized in-eye axes.
    assert np.isnan(metrics[PupilColumn.PUPIL_IN_EYE_X][0])
    assert np.isnan(metrics[PupilColumn.PUPIL_IN_EYE_Y][0])
    # A collapsed ring is a degenerate fit rather than a shut lid, and the pupil resolving through it says as much.
    assert metrics[PupilColumn.BLINKING_STATE].tolist() == [False]


def test_compute_pupil_metrics_reports_a_residual_only_for_an_overdetermined_pupil_ring() -> None:
    determined = _frame_specification(pupil_confident=(0, 2, 4))
    points = _build_points(specifications=[_frame_specification(), determined])
    # Displaces one confident perimeter point off the true ellipse, so the overdetermined fit carries a real residual.
    points["pupil_top"][0, 1] += 4.0

    metrics = _compute_pupil_metrics(points=points)

    # Over eight evenly spaced ring points the [1, cos, sin] design columns are orthogonal, so every point carries a
    # leverage of 1/8 + 1/4, and a single four-pixel displacement leaves a root-mean-square deviation of sqrt(1.25).
    assert metrics[PupilColumn.PUPIL_FIT_RESIDUAL_PX][0] == pytest.approx(np.sqrt(1.25), abs=1e-5)
    assert np.isnan(metrics[PupilColumn.PUPIL_FIT_RESIDUAL_PX][1])
    assert metrics[PupilColumn.PUPIL_FIT_CONDITION][0] == pytest.approx(np.sqrt(2.0))
    assert metrics[PupilColumn.PUPIL_FIT_CONDITION][1] == pytest.approx(np.sqrt(2.0) + 1.0)


def test_fit_ring_ellipse_rejects_an_arc_that_is_too_short_to_pin_the_ellipse() -> None:
    # A sixteen-point ring samples the ellipse finely enough that three adjacent points leave the fit ill conditioned,
    # while three points spread a quarter turn apart pin it exactly as the eight-point pupil ring does.
    names = tuple(f"dense_{index}" for index in range(16))
    positions = _ring_positions(center=_PUPIL_CENTER, semi_a=_PUPIL_SEMI_A, semi_b=_PUPIL_SEMI_B, count=len(names))
    adjacent, spread = {0, 1, 2}, {0, 4, 8}
    points = {
        name: np.array(
            [
                [positions[index, 0], positions[index, 1], _CONFIDENT if index in adjacent else _UNCONFIDENT],
                [positions[index, 0], positions[index, 1], _CONFIDENT if index in spread else _UNCONFIDENT],
            ],
            dtype=np.float64,
        )
        for index, name in enumerate(names)
    }

    fit = _fit_ring_ellipse(points=points, names=names)

    assert fit.valid.tolist() == [False, True]
    assert np.isnan(fit.condition[0])
    assert fit.condition[1] == pytest.approx(np.sqrt(2.0) + 1.0)
    assert fit.center[1].tolist() == pytest.approx(list(_PUPIL_CENTER))
