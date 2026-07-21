"""Provides the system-agnostic substrate shared across acquisition systems."""

from .tracking import tracked_job, prepare_tracker
from .utilities import (
    DELAY_TIMER,
    LOG_ARCHIVE_SUFFIX,
    delay_terminal,
    multi_recording_dataset_directory,
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
    "DELAY_TIMER",
    "LOG_ARCHIVE_SUFFIX",
    "delay_terminal",
    "find_module_feathers",
    "get_event_data",
    "get_event_timestamps",
    "merge_event_streams",
    "multi_recording_dataset_directory",
    "parse_module_feather_name",
    "partition_events",
    "prepare_tracker",
    "tracked_job",
]
