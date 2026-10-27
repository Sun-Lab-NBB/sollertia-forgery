from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray

from .metadata import BehaviorDataFiles as BehaviorDataFiles
from .frame_windows import (
    flag_window_onsets as flag_window_onsets,
    resolve_window_signed_peaks as resolve_window_signed_peaks,
)

_MICROSECONDS_PER_SECOND: int
_MICROSECONDS_PER_MINUTE: int
_RUNNING_SPEED_WINDOW_US: int
_TIME_COLUMNS: frozenset[str]
_ENCODER_STATES: tuple[str, ...]

def assemble_behavior_dataset(
    microcontroller_data_path: Path,
    runtime_data_path: Path,
    raw_data_path: Path,
    reference_time: NDArray[np.uint64],
    *,
    drop_time_columns: bool = False,
) -> pl.DataFrame: ...
def _calculate_running_speed(
    sample_time: NDArray[np.uint64], distance: NDArray[np.float64], window_size_us: int = ...
) -> NDArray[np.float32]: ...
