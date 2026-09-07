from pathlib import Path
from collections.abc import Iterable

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import (
    SessionData as SessionData,
    TaskTemplate,
    TrialStructure as TrialStructure,
    MesoscopeExperimentConfiguration,
)

from .metadata import BehaviorDataFiles as BehaviorDataFiles

RUNTIME_SOURCE_ID: str
_CUE_SEQUENCE_MIN_LENGTH: int
_SYSTEM_STATE_CODE: int
_RUNTIME_STATE_CODE: int
_REINFORCING_GUIDANCE_STATE_CODE: int
_AVERSIVE_GUIDANCE_STATE_CODE: int
_DISTANCE_SNAPSHOT_CODE: int
_ERROR_CONTEXT_CUE_COUNT: int

def parse_runtime(decoded_messages: pl.DataFrame, output_directory: Path, session: SessionData) -> None: ...
def _export_runtime_data(
    messages: Iterable[tuple[np.uint64, NDArray[np.uint8]]],
    output_directory: Path,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
    task_template: TaskTemplate | None,
) -> None: ...
def _resolve_experiment_configuration(session: SessionData) -> MesoscopeExperimentConfiguration | None: ...
def _resolve_task_template(
    session: SessionData, experiment_configuration: MesoscopeExperimentConfiguration | None
) -> TaskTemplate | None: ...
def _resolve_trial_geometries(task_template: TaskTemplate, trial_names: list[str]) -> list[TrialStructure]: ...
def _decompose_multiple_cue_sequences_into_trials(
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
    cue_sequences: list[NDArray[np.uint8]],
    distance_breakpoints: list[np.float64],
) -> tuple[NDArray[np.int32], NDArray[np.float64]]: ...
def _prepare_motif_data(
    trial_motifs: list[NDArray[np.uint8]], trial_distances: list[float]
) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]: ...
def _decompose_cue_sequence_into_trials(
    cue_sequence: NDArray[np.uint8],
    motifs_flat: NDArray[np.uint8],
    motif_starts: NDArray[np.int32],
    motif_lengths: NDArray[np.int32],
    motif_indices: NDArray[np.int32],
    maximum_trials: int,
) -> tuple[NDArray[np.int32], int, int]: ...
def _process_trial_sequence(
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
    trial_types: NDArray[np.int32],
    trial_distances: NDArray[np.float64],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
