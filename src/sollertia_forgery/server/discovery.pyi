from pathlib import Path
from dataclasses import dataclass

from sollertia_shared_assets import DatasetSession

from .server import Server as Server
from ..shared_assets import delay_terminal as delay_terminal
from .server_configuration import get_server_configuration as get_server_configuration

_DATASET_MARKER_DEPTH: int
_SESSION_MARKER_DEPTH: int

@dataclass(frozen=True, slots=True)
class _ProjectMarkers:
    datasets: tuple[Path, ...]
    sessions: tuple[DatasetSession, ...]

def discover_project_markers(
    project_path: Path, server: Server, *, include_sessions: bool = True
) -> _ProjectMarkers: ...
def discover_project_data(project: str) -> tuple[DatasetSession, ...]: ...
def _discover_project_sessions(project: str, server: Server) -> tuple[DatasetSession, ...]: ...
