"""Provides the system-agnostic substrate shared across every acquisition system: the forged dataset data hierarchy,
microcontroller feather primitives, and terminal utilities.
"""

from .utilities import delay_timer, delay_terminal
from .dataset_data import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)
from .microcontroller import (
    get_event_data,
    partition_events,
    merge_event_streams,
    find_module_feathers,
    get_event_timestamps,
    parse_module_feather_name,
)

__all__ = [
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "delay_terminal",
    "delay_timer",
    "find_module_feathers",
    "get_event_data",
    "get_event_timestamps",
    "merge_event_streams",
    "parse_module_feather_name",
    "partition_events",
]
