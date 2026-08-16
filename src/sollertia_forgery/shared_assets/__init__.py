"""Provides the system-agnostic substrate shared across acquisition systems, including the pipeline-identity
enumeration, the per-session tracker locations, the microcontroller event-stream primitive, and the shared utilities.
"""

from .pipelines import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    resolve_session_tracker_path,
)
from .utilities import (
    DELAY_TIMER,
    delay_terminal,
    multi_recording_dataset_directory,
)
from .microcontroller import merge_event_streams

__all__ = [
    "DELAY_TIMER",
    "SESSION_PIPELINES",
    "ProcessingPipelines",
    "delay_terminal",
    "merge_event_streams",
    "multi_recording_dataset_directory",
    "resolve_session_tracker_path",
]
