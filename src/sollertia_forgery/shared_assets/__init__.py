"""Provides the system-agnostic substrate shared across acquisition systems, including the pipeline-identity
enumeration, the per-session tracker locations, the microcontroller event-stream primitive, the OpenMP runtime
discovery, and the shared utilities.
"""

from .openmp import (
    OpenMPStatus,
    OpenMPSummary,
    verify_openmp_runtime,
    resolve_openmp_runtime,
)
from .pipelines import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    resolve_session_tracker_path,
)
from .utilities import (
    DELAY_TIMER,
    natural_sort,
    delay_terminal,
    multi_recording_dataset_name,
)
from .microcontroller import merge_event_streams

__all__ = [
    "DELAY_TIMER",
    "SESSION_PIPELINES",
    "OpenMPStatus",
    "OpenMPSummary",
    "ProcessingPipelines",
    "delay_terminal",
    "merge_event_streams",
    "multi_recording_dataset_name",
    "natural_sort",
    "resolve_openmp_runtime",
    "resolve_session_tracker_path",
    "verify_openmp_runtime",
]
