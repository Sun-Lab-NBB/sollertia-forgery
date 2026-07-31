from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sollertia_shared_assets import SessionData as SessionData

_EYE_TRACKING_PROJECT_NAME: str
PUPIL_CAMERA_NAME: str
_PUPIL_TARGET: str
_REFLECTION_POINT: str
_PUPIL_POINTS: tuple[str, ...]
_EYE_POINTS: tuple[str, ...]
_CANONICAL_POINTS: tuple[str, ...]
_LIKELIHOOD_THRESHOLD: float
_MINIMUM_PERIMETER_POINTS: int
_MAXIMUM_FIT_CONDITION: float
_BLINK_FRACTION: float
_COORDINATES: tuple[str, str, str]
type _MetricArray = NDArray[np.float64] | NDArray[np.bool_]

class PupilColumn(StrEnum):
    PUPIL_CENTER_X_PX = "pupil_center_x_px"
    PUPIL_CENTER_Y_PX = "pupil_center_y_px"
    PUPIL_DIAMETER_PX = "pupil_diameter_px"
    PUPIL_AREA_PX2 = "pupil_area_px2"
    PUPIL_FIT_CONDITION = "pupil_fit_condition"
    PUPIL_FIT_RESIDUAL_PX = "pupil_fit_residual_px"
    EYE_CENTER_X_PX = "eye_center_x_px"
    EYE_CENTER_Y_PX = "eye_center_y_px"
    EYE_WIDTH_PX = "eye_width_px"
    EYE_HEIGHT_PX = "eye_height_px"
    EYE_OPENNESS = "eye_openness"
    BLINKING_STATE = "blinking_state"
    DILATION_STATE = "dilation_state"
    REFLECTION_X_PX = "reflection_x_px"
    REFLECTION_Y_PX = "reflection_y_px"
    PUPIL_REFLECTION_OFFSET_X_PX = "pupil_reflection_offset_x_px"
    PUPIL_REFLECTION_OFFSET_Y_PX = "pupil_reflection_offset_y_px"
    PUPIL_IN_EYE_X = "pupil_in_eye_x"
    PUPIL_IN_EYE_Y = "pupil_in_eye_y"

def process_mesoscope_video_tracking(session: SessionData, output_directory: Path) -> None: ...
def _read_points_from_h5(h5_path: Path, bodyparts: tuple[str, ...]) -> dict[str, NDArray[np.float64]]: ...
def _compute_pupil_metrics(points: dict[str, NDArray[np.float64]]) -> dict[str, _MetricArray]: ...

@dataclass(frozen=True, slots=True)
class _RingFit:
    center: NDArray[np.float64]
    semi_a: NDArray[np.float64]
    semi_b: NDArray[np.float64]
    condition: NDArray[np.float64]
    residual: NDArray[np.float64]
    valid: NDArray[np.bool_]

def _fit_ring_ellipse(points: dict[str, NDArray[np.float64]], names: tuple[str, ...]) -> _RingFit: ...
def _norm(vectors: NDArray[np.float64]) -> NDArray[np.float64]: ...
def _mask_invalid(
    metrics: dict[str, NDArray[np.float64]], valid: NDArray[np.bool_]
) -> dict[str, NDArray[np.float64]]: ...
