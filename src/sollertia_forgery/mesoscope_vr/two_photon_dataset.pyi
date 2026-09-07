from pathlib import Path

import numpy as np
from cindra import RecordingArrays
import polars as pl
from numpy.typing import NDArray as NDArray

from .metadata import (
    DatasetColumn as DatasetColumn,
    BehaviorDataFiles as BehaviorDataFiles,
)

_MICROSECONDS_PER_MILLISECOND: int
_MICROSECONDS_PER_SECOND: float
_MICROSECONDS_PER_MINUTE: int
_SCAN_PULSE_TOLERANCE_MS: int
_FRAME_VARIANT_METADATA_FILENAME: str
_SCANIMAGE_FRAME_NUMBER_KEY: str
_SCANIMAGE_FRAME_TIMESTAMP_KEY: str
_SCANIMAGE_MATCH_TOLERANCE_US: int
_SCANIMAGE_ANCHOR_SEARCH_LIMIT: int
_SCANIMAGE_ACQUISITION_NUMBER_KEY: str
_PULSE_RUN_GAP_FACTOR: float
_MINIMUM_SPLITTABLE_PULSE_COUNT: int
_UNMATCHED_COST: int

def assemble_cindra_dataset(
    cindra_data_path: Path, microcontroller_data_path: Path, multiday_data_path: Path, raw_data_path: Path
) -> pl.DataFrame: ...
def _discard_unacquired_pulse_runs(frame_aligned_data: pl.DataFrame, raw_data_path: Path) -> pl.DataFrame: ...
def _resolve_acquisition_sizes(raw_data_path: Path) -> list[int]: ...
def _match_runs_to_acquisitions(
    run_lengths: list[int], acquisition_sizes: list[int]
) -> list[tuple[int, int]] | None: ...
def _search_run_spans(
    run_lengths: list[int],
    acquisition_sizes: list[int],
    order: tuple[int, ...],
    run_index: int,
    position: int,
    claimed: list[tuple[int, int]],
    cost: int,
    best: tuple[int, list[tuple[int, int]]],
) -> tuple[int, list[tuple[int, int]]]: ...
def _align_pulses_to_scanimage(
    paired_pulses: pl.DataFrame, raw_data_path: Path, expected_frame_count: int
) -> pl.DataFrame: ...
def _count_matches_within_tolerance(
    pulse_microseconds: NDArray[np.int64], scanimage_microseconds: NDArray[np.int64]
) -> int: ...
def _nearest_target_index(values: NDArray[np.int64], sorted_targets: NDArray[np.int64]) -> NDArray[np.intp]: ...
def _load_cindra_fluorescence(
    data_path: Path, array: RecordingArrays, column_name: str, *, cell_mask: NDArray[np.bool_] | None = None
) -> pl.Series: ...
