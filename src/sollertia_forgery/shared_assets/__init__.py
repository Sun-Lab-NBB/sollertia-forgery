"""Provides the system-agnostic substrate shared across every acquisition system: the forged dataset data hierarchy,
batch-orchestration primitives, microcontroller feather primitives, and terminal utilities.
"""

from .utilities import delay_timer, delay_terminal
from .dataset_data import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)
from .orchestration import (
    RESERVED_CORES,
    ActiveJob,
    PendingJob,
    GenericPendingJob,
    JobExecutionState,
    prepare_tracker,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    group_jobs_by_tracker,
    job_execution_manager,
    clean_output_subdirectory,
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
    "RESERVED_CORES",
    "ActiveJob",
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "GenericPendingJob",
    "JobExecutionState",
    "PendingJob",
    "analyze_feather_file",
    "clean_output_subdirectory",
    "delay_terminal",
    "delay_timer",
    "derive_tracker_status",
    "find_module_feathers",
    "get_event_data",
    "get_event_timestamps",
    "group_jobs_by_tracker",
    "job_execution_manager",
    "merge_event_streams",
    "parse_module_feather_name",
    "partition_events",
    "prepare_tracker",
    "read_tracker_status",
]
