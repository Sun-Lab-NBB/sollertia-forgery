"""Provides the single-recording two-photon (calcium-imaging) processing pipeline based on cindra."""

from cindra import SingleRecordingJobNames

from .pipeline import run_two_photon_processing_pipeline

__all__ = [
    "SingleRecordingJobNames",
    "run_two_photon_processing_pipeline",
]
