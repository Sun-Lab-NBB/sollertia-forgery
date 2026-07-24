"""Provides the ProcessingPipelines enumeration that identifies the data processing and management pipelines an agent
orchestrates over a project's sessions.
"""

from __future__ import annotations

from enum import StrEnum


class ProcessingPipelines(StrEnum):
    """Enumerates the data processing and management pipelines an agent orchestrates over a project's sessions.

    Notes:
        The member names mirror the corresponding members of sollertia-shared-assets' ``ProcessingTrackers`` enum
        (one tracker per pipeline), but the values are short pipeline identifiers rather than tracker filenames.
    """

    MANIFEST = "manifest"
    """The project manifest generation pipeline."""
    CHECKSUM = "checksum"
    """The raw data integrity (checksum) verification pipeline."""
    FORGING = "forging"
    """The dataset assembly (forging) pipeline."""
