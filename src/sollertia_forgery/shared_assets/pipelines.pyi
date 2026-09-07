from enum import StrEnum
from pathlib import Path
from collections.abc import Callable as Callable

from sollertia_shared_assets import SessionData as SessionData

class ProcessingPipelines(StrEnum):
    MANIFEST = "manifest"
    CHECKSUM = "checksum"
    RUNTIME = "runtime"
    MICROCONTROLLER = "microcontroller"
    VIDEO = "video"
    TWO_PHOTON = "two_photon"
    FORGING = "forging"

_SESSION_TRACKER_LOCATIONS: dict[ProcessingPipelines, Callable[[SessionData], Path]]
SESSION_PIPELINES: tuple[ProcessingPipelines, ...]

def resolve_session_tracker_path(session: SessionData, pipeline: ProcessingPipelines) -> Path: ...
