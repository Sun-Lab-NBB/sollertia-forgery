"""Provides the pipeline-identity enumeration and resolves the processing tracker each per-session pipeline records its
jobs on.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData


class ProcessingPipelines(StrEnum):
    """Enumerates the data processing and management pipelines an agent orchestrates over a project, its sessions, and
    its datasets.

    Notes:
        The member names mirror the corresponding members of sollertia-shared-assets' ``ProcessingTrackers`` enum. The
        values here are short pipeline identifiers, so a caller resolves a tracker filename through that enum. Cindra's
        multi-recording stage has no member in either enum, since its tracker is written once per dataset inside a
        dataset-named directory that no fixed per-session path addresses. This library resolves that stage's paths
        through cindra's own ``resolve_dataset_path`` instead.
    """

    MANIFEST = "manifest"
    """The project manifest generation pipeline."""
    CHECKSUM = "checksum"
    """The raw data integrity (checksum) verification pipeline."""
    RUNTIME = "runtime"
    """The acquisition-runtime log processing pipeline."""
    MICROCONTROLLER = "microcontroller"
    """The microcontroller log processing pipeline."""
    VIDEO = "video"
    """The camera video-processing pipeline."""
    TWO_PHOTON = "two_photon"
    """The single-recording two-photon (calcium-imaging) processing pipeline."""
    FORGING = "forging"
    """The dataset assembly (forging) pipeline."""


_SESSION_TRACKER_LOCATIONS: dict[ProcessingPipelines, Callable[[SessionData], Path]] = {
    ProcessingPipelines.CHECKSUM: lambda session: session.raw_data.checksum_tracker_path,
    ProcessingPipelines.RUNTIME: lambda session: session.processed_data.runtime_tracker_path,
    ProcessingPipelines.MICROCONTROLLER: lambda session: session.processed_data.microcontroller_tracker_path,
    ProcessingPipelines.VIDEO: lambda session: session.processed_data.video_tracker_path,
    ProcessingPipelines.TWO_PHOTON: lambda session: session.processed_data.two_photon_tracker_path,
}
"""Maps each per-session pipeline to the accessor that resolves its processing tracker from a loaded session.

Notes:
    The checksum tracker sits under the acquired data, since that pipeline verifies the acquired data in place. Every
    other pipeline records beside the output it produces.
"""


SESSION_PIPELINES: tuple[ProcessingPipelines, ...] = tuple(_SESSION_TRACKER_LOCATIONS.keys())
"""The pipelines that process a single session, in the order a per-pipeline report presents them.

Notes:
    Derived from the tracker mapping, so the two can never disagree about which pipelines a session carries. The
    manifest and forging pipelines are absent, since one operates on a project and the other on a dataset.
"""


def resolve_session_tracker_path(session: SessionData, pipeline: ProcessingPipelines) -> Path:
    """Resolves the processing tracker one per-session pipeline records its jobs on.

    Args:
        session: The loaded session whose tracker location to resolve.
        pipeline: The pipeline whose tracker to resolve. Must be one of ``SESSION_PIPELINES``.

    Returns:
        The path to that pipeline's processing tracker for this session.

    Raises:
        ValueError: If the pipeline operates on a unit other than a single session.
    """
    if pipeline not in _SESSION_TRACKER_LOCATIONS:
        message = (
            f"Unable to resolve the processing tracker path of pipeline '{pipeline}' for session "
            f"'{session.session_name}'. Only a pipeline that processes a single session records a per-session "
            f"tracker, so the pipeline must be one of {[member.value for member in SESSION_PIPELINES]}."
        )
        console.error(message=message, error=ValueError)
    return _SESSION_TRACKER_LOCATIONS[pipeline](session)
