"""Provides primitives for discovering and parsing the microcontroller module feather files produced by
ataraxis-communication-interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from numpy.typing import NDArray

_MODULE_FEATHER_PATTERN: str = "controller_*_module_*.feather"
"""The glob pattern used to discover microcontroller module feather files produced by ataraxis-communication-interface.
"""


def find_module_feathers(data_directory: Path) -> list[Path]:
    """Discovers microcontroller module feather files under the data directory.

    Searches ``data_directory`` non-recursively for feather files matching the ``controller_*_module_*.feather``
    naming convention used by ataraxis-communication-interface. The directory is expected to be the session's
    canonical ``processed_data/microcontroller_data`` location exposed by
    ``SessionData.processed_data.microcontroller_data_path``.

    Args:
        data_directory: The path to the session's microcontroller data directory.

    Returns:
        The discovered module feather paths, sorted, or an empty list when the directory does not exist or holds no
        matching files.
    """
    if not data_directory.is_dir():
        return []
    return sorted(data_directory.glob(_MODULE_FEATHER_PATTERN))


def parse_module_feather_name(feather_path: Path) -> tuple[int, int, int]:
    """Extracts the controller ID, module type, and module ID from a module feather filename.

    Args:
        feather_path: The path to the module feather file. The filename must follow the
            ``controller_{controller_id}_module_{module_type}_{module_id}.feather`` naming convention, where the
            leading field is the ataraxis-communication-interface source id (the controller id for microcontroller
            archives).

    Returns:
        The (controller_id, module_type, module_id) parsed from the filename.

    Raises:
        ValueError: If the filename does not follow the expected naming convention.
    """
    stem = feather_path.stem  # e.g., "controller_101_module_3_1"
    parts = stem.split("_")

    _expected_part_count = 5
    if len(parts) != _expected_part_count or parts[0] != "controller" or parts[2] != "module":
        message = (
            f"Unable to parse module feather filename '{feather_path.name}'. The filename does not follow the "
            f"expected 'controller_{{id}}_module_{{type}}_{{id}}.feather' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1]), int(parts[3]), int(parts[4])


def partition_events(module_dataframe: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Partitions a module DataFrame into per-event sub-DataFrames in a single pass.

    Notes:
        Groups the rows by event code in a single ``partition_by`` traversal of the DataFrame, returning the groups
        keyed by event code so subsequent per-code lookups are O(1).

    Args:
        module_dataframe: The Polars DataFrame read from an ataraxis-communication-interface module feather file with
            the standard 5-column schema (timestamp_us: UInt64, command: UInt8, event: UInt8, dtype: String,
            data: Binary).

    Returns:
        A dictionary mapping integer event codes to their corresponding sub-DataFrames.
    """
    # polars 1.x partition_by(as_dict=True) returns single-element tuples as keys, even when partitioning on
    # a single column, so the event code is always at index 0.
    raw_partition = module_dataframe.partition_by("event", as_dict=True)
    return {int(key[0]): value for key, value in raw_partition.items()}


def get_event_timestamps(partition: dict[int, pl.DataFrame], event_code: int) -> NDArray[np.uint64]:
    """Returns the timestamp array for a given event code from a partitioned DataFrame.

    Notes:
        Designed for state-only events that do not carry data payloads.

    Args:
        partition: The event-code-keyed partition dictionary produced by partition_events().
        event_code: The event code to look up.

    Returns:
        The timestamps for the requested event code, or an empty array when the code is not present.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64)
    return event_dataframe["timestamp_us"].to_numpy().astype(np.uint64)


def get_event_data[ScalarT: np.generic](
    partition: dict[int, pl.DataFrame],
    event_code: int,
    values_dtype: type[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Returns timestamps and vectorized-reconstructed data values for a given event code.

    Notes:
        Relies on the ataraxis-communication-interface protocol guarantee that all messages sharing an event code also
        share a payload dtype, so binary payloads can be concatenated and decoded with a single np.frombuffer() call.
        The reconstructed values are then cast to the requested output dtype for uniform downstream handling.

    Args:
        partition: The event-code-keyed partition dictionary produced by partition_events().
        event_code: The event code to look up.
        values_dtype: The NumPy scalar type to cast the reconstructed values to.

    Returns:
        The timestamps and the reconstructed values cast to values_dtype, both empty when the event code is not
        present in the partition.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=values_dtype)

    timestamps: NDArray[np.uint64] = event_dataframe["timestamp_us"].to_numpy().astype(np.uint64)

    data_list = event_dataframe["data"].to_list()
    dtype_list = event_dataframe["dtype"].to_list()
    payload_dtype = dtype_list[0]
    values: NDArray[ScalarT] = np.frombuffer(b"".join(data_list), dtype=payload_dtype).astype(values_dtype)

    return timestamps, values


def merge_event_streams[ScalarT: np.generic](
    timestamps_a: NDArray[np.uint64],
    values_a: NDArray[ScalarT],
    timestamps_b: NDArray[np.uint64],
    values_b: NDArray[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Merges two chronologically-sorted event streams into a single timestamp-sorted stream.

    Notes:
        Uses NumPy's stable sort (``kind="stable"``), which NumPy maps to a linear-time radix sort for the uint64
        timestamp keys.

    Args:
        timestamps_a: The uint64 timestamp array for the first event stream.
        values_a: The value array for the first event stream.
        timestamps_b: The uint64 timestamp array for the second event stream.
        values_b: The value array for the second event stream.

    Returns:
        A tuple of (merged timestamps, reordered values) sorted chronologically.
    """
    timestamps = np.concatenate([timestamps_a, timestamps_b])
    values = np.concatenate([values_a, values_b])
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order]
