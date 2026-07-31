from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray as NDArray

_MODULE_FEATHER_PATTERN: str

def find_module_feathers(data_directory: Path) -> list[Path]: ...
def parse_module_feather_name(feather_path: Path) -> tuple[int, int, int]: ...
def partition_events(module_dataframe: pl.DataFrame) -> dict[int, pl.DataFrame]: ...
def get_event_timestamps(partition: dict[int, pl.DataFrame], event_code: int) -> NDArray[np.uint64]: ...
def get_event_data[ScalarT: np.generic](
    partition: dict[int, pl.DataFrame], event_code: int, values_dtype: type[ScalarT]
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]: ...
def merge_event_streams[ScalarT: np.generic](
    timestamps_a: NDArray[np.uint64],
    values_a: NDArray[ScalarT],
    timestamps_b: NDArray[np.uint64],
    values_b: NDArray[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]: ...
