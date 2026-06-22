"""Provides the system-agnostic raw-data processing pipelines shared across acquisition systems."""

from .video import VIDEO_JOB_NAME, run_video_processing_pipeline

__all__ = [
    "VIDEO_JOB_NAME",
    "run_video_processing_pipeline",
]
