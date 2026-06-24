"""Provides the ProcessingPipelines enumeration that identifies the data processing and management pipelines
orchestrated against the remote compute server.
"""

from __future__ import annotations

from enum import StrEnum


class ProcessingPipelines(StrEnum):
    """Enumerates the data processing and management pipelines orchestrated against the remote compute server.

    Notes:
        The member names mirror the corresponding members of sollertia-shared-assets' ``ProcessingTrackers`` enum
        (one tracker per pipeline), but the values are short pipeline identifiers rather than tracker filenames.
        Only the subset of pipelines that sollertia-forgery dispatches to the remote server is enumerated here.
    """

    MANIFEST = "manifest"
    """The project manifest generation pipeline."""
    CHECKSUM = "checksum"
    """The raw data integrity (checksum) verification pipeline."""
    BEHAVIOR = "behavior"
    """The behavior and camera data processing pipeline."""
    CINDRA_SINGLE_RECORDING = "cindra_single_recording"
    """The single-day cindra (calcium imaging) processing pipeline."""
    CINDRA_MULTI_RECORDING = "cindra_multi_recording"
    """The multi-day cindra (across-session cell tracking) processing pipeline."""
    FORGING = "forging"
    """The dataset assembly (forging) pipeline."""
