"""Provides the shared Model Context Protocol (MCP) server entry points for agentic interaction with the library."""

from __future__ import annotations

from typing import Literal
from pathlib import Path
import importlib

from .mcp_instance import mcp

__all__ = ["run_server"]


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the shared MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication, 'sse' for server-sent-event streaming, and 'streamable-http' for HTTP-based communication.
    """
    # Blocks until the transport connection is closed, so the caller owns the process for the server's lifetime.
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        mcp.run(transport=transport, json_response=True)
        return

    mcp.run(transport=transport)


def _register_tool_modules() -> None:
    """Imports every ``*_tools`` module in this package so its ``@mcp.tool()`` decorators register on import."""
    package_name = __name__.rpartition(".")[0]
    for module_path in sorted(Path(__file__).parent.glob("*_tools.py")):
        importlib.import_module(f"{package_name}.{module_path.stem}")


_register_tool_modules()
