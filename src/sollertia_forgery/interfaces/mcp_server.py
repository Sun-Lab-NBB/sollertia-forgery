"""Provides the shared Model Context Protocol (MCP) server entry points for agentic interaction with the library."""

from __future__ import annotations

from typing import Literal
from pathlib import Path
import importlib

from .mcp_instance import mcp

__all__ = ["run_mcp_server", "run_server"]


def run_mcp_server() -> None:
    """Starts the shared MCP server with stdio transport.

    Serves as a CLI entry point, launching the MCP server using the stdio transport protocol recommended for
    Claude Desktop integration.
    """
    run_server(transport="stdio")


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the shared MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication, 'sse' for server-sent-event streaming, and 'streamable-http' for HTTP-based communication.
    """
    # Blocks until the transport connection is closed, so the caller owns the process for the server's lifetime.
    mcp.run(transport=transport)


def _register_tool_modules() -> None:
    """Imports every ``*_tools`` module in this package so its ``@mcp.tool()`` decorators register on import.

    Tool modules register their MCP tools purely as an import side effect. Discovering them by the ``_tools`` filename
    suffix means each tool module registers automatically, so adding a new tool module requires no edit here.
    """
    package_name = __name__.rpartition(".")[0]
    for module_path in sorted(Path(__file__).parent.glob("*_tools.py")):
        importlib.import_module(f"{package_name}.{module_path.stem}")


_register_tool_modules()
