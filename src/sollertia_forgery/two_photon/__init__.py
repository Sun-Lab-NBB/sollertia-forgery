"""Provides the single-recording two-photon (calcium-imaging) processing pipeline based on cindra."""

from cindra import SingleRecordingJobNames

from .pipeline import (
    STAGE_DEFAULT_WORKERS,
    CINDRA_CONFIGURATION_FILENAME,
    discover_two_photon_jobs,
    prime_two_photon_recording,
    two_photon_job_prerequisites,
    run_two_photon_processing_pipeline,
)

__all__ = [
    "CINDRA_CONFIGURATION_FILENAME",
    "STAGE_DEFAULT_WORKERS",
    "SingleRecordingJobNames",
    "discover_two_photon_jobs",
    "prime_two_photon_recording",
    "run_two_photon_processing_pipeline",
    "two_photon_job_prerequisites",
]
