"""Provides the camera-data processing pipeline."""

from .pipeline import (
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    TIMESTAMP_JOB_NAME,
    run_video_processing_pipeline,
)
from .configuration import (
    VideoTrackingConfiguration,
    get_video_tracking_configuration,
    get_video_tracking_configuration_path,
    create_video_tracking_configuration_file,
)

__all__ = [
    "RENAME_JOB_NAME",
    "TIMESTAMP_JOB_NAME",
    "TRACKING_JOB_NAME",
    "VideoTrackingConfiguration",
    "create_video_tracking_configuration_file",
    "get_video_tracking_configuration",
    "get_video_tracking_configuration_path",
    "run_video_processing_pipeline",
]
