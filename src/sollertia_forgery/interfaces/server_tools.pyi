from typing import Any

from ..server import (
    ServerConfiguration as ServerConfiguration,
    get_server_configuration as get_server_configuration,
    get_server_configuration_path as get_server_configuration_path,
)
from .responses import (
    ok_response as ok_response,
    error_response as error_response,
)
from .mcp_instance import mcp as mcp

def read_server_configuration_tool() -> dict[str, Any]: ...
def write_server_configuration_tool(
    configuration_payload: dict[str, Any], *, overwrite: bool = False
) -> dict[str, Any]: ...
def _render_configuration(instance: ServerConfiguration) -> dict[str, Any]: ...
