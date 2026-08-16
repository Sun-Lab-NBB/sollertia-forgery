"""Provides the camera video-processing pipeline."""

from ataraxis_video_system import CAMERA_EXTRACTION_JOB_NAME

from .pipeline import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    discover_video_jobs,
    video_job_prerequisites,
    run_video_processing_pipeline,
)
from .motion_energy import MotionEnergyColumn

__all__ = [
    "CAMERA_EXTRACTION_JOB_NAME",
    "ENERGY_JOB_NAME",
    "RENAME_JOB_NAME",
    "TRACKING_JOB_NAME",
    "MotionEnergyColumn",
    "discover_video_jobs",
    "run_video_processing_pipeline",
    "video_job_prerequisites",
]
