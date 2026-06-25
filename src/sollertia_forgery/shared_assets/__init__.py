"""Provides the system-agnostic substrate shared across every acquisition system: the processing-tracker
job-registry alignment helper and tracked-job execution envelope, the DataLogger log-archive naming convention,
the microcontroller feather discovery and parsing primitives, and terminal utilities.
"""

from .tracking import tracked_job, prepare_tracker
from .utilities import delay_timer, delay_terminal
from .datalogger import LOG_ARCHIVE_SUFFIX
from .microcontroller import (
    get_event_data,
    partition_events,
    merge_event_streams,
    find_module_feathers,
    get_event_timestamps,
    parse_module_feather_name,
)

__all__ = [
    "LOG_ARCHIVE_SUFFIX",
    "delay_terminal",
    "delay_timer",
    "find_module_feathers",
    "get_event_data",
    "get_event_timestamps",
    "merge_event_streams",
    "parse_module_feather_name",
    "partition_events",
    "prepare_tracker",
    "tracked_job",
]
