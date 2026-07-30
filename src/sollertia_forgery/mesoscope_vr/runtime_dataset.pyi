from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import MesoscopeExperimentConfiguration as MesoscopeExperimentConfiguration

from .metadata import BehaviorDataFiles as BehaviorDataFiles

_CUE_UNDEFINED: int
_TRIAL_UNDEFINED: int
_SYSTEM_STATE_IDLE: int
_RUNTIME_STATE_IDLE: int

def assemble_runtime_dataset(
    microcontroller_data_path: Path,
    runtime_data_path: Path,
    experiment_configuration: MesoscopeExperimentConfiguration,
    reference_time: NDArray[np.uint64],
) -> pl.DataFrame: ...
def mask_non_run_experiment_data(experiment_data: pl.DataFrame) -> pl.DataFrame: ...
def clip_to_session_bounds(assembled_data: pl.DataFrame, runtime_data_path: Path) -> pl.DataFrame: ...
def _check_trigger_zones(
    traversed_distance: NDArray[np.float64],
    trigger_zone_starts: NDArray[np.float64],
    trigger_zone_ends: NDArray[np.float64],
) -> NDArray[np.uint8]: ...
