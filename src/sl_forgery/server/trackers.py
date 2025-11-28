"""This module provides the tracker filename enumerations used by various data management, processing, and analysis
pipelines available from this library.
"""

from enum import StrEnum


class ManagingTrackers(StrEnum):
    """Defines the filenames for tracker files used by data managing pipelines."""

    CHECKSUM = "checksum.yaml"
    """The tracker file used by the checksum resolution pipeline."""
    MANIFEST = "manifest.yaml"
    """The tracker file used by the project manifest generation pipeline."""


class ProcessingTrackers(StrEnum):
    """Defines the filenames for tracker files used by data processing pipelines."""

    SUITE2P = "suite2p.yaml"
    """The tracker file used by the suite2p processing pipeline."""
    BEHAVIOR = "behavior.yaml"
    """The tracker file used by the behavior extraction pipeline."""
    VIDEO = "video.yaml"
    """The tracker file used by the video (DeepLabCut) processing pipeline."""


class DatasetTrackers(StrEnum):
    """Defines the filenames for tracker files used by dataset forging and multi-day analysis pipelines."""

    FORGING = "forging.yaml"
    """The tracker file used by the dataset forging pipeline."""
    MULTIDAY = "multiday.yaml"
    """The tracker file used by the multi-day suite2p registration pipeline."""
