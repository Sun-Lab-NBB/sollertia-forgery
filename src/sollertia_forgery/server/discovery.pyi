from sollertia_shared_assets import DatasetSession

from .server import Server as Server
from ..shared_assets import delay_terminal as delay_terminal
from .server_configuration import get_server_configuration as get_server_configuration

def discover_project_sessions(project: str, server: Server) -> tuple[DatasetSession, ...]: ...
def discover_project_data(project: str) -> tuple[DatasetSession, ...]: ...
