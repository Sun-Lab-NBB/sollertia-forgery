"""Provides the shared Model Context Protocol (MCP) server entry points for agentic interaction with the library."""

from __future__ import annotations

from typing import Literal

from .mcp_instance import mcp
from ..cross_system import (
    mcp_tools as _cross_system_mcp_tools,  # noqa: F401
    server_mcp_tools as _server_mcp_tools,  # noqa: F401
)
from ..mesoscope_vr import (
    forging_mcp_tools as _forging_mcp_tools,  # noqa: F401
    processing_mcp_tools as _processing_mcp_tools,  # noqa: F401
)


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the shared MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication and 'streamable-http' for HTTP-based communication.
    """
    # Delegates to the FastMCP run loop, which blocks until the transport connection is closed. For 'stdio',
    # the server runs until the parent process closes stdin. For 'streamable-http', runs an HTTP server that
    # accepts connections until explicitly terminated.
    mcp.run(transport=transport)


def run_mcp_server() -> None:
    """Starts the shared MCP server with stdio transport.

    Serves as a CLI entry point, launching the MCP server using the stdio transport protocol recommended for
    Claude Desktop integration.
    """
    run_server(transport="stdio")
