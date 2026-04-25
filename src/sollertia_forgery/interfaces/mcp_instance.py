"""Provides the shared FastMCP instance used by every package's ``mcp_tools`` submodule."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP(name="sollertia-forgery", json_response=True)
"""Stores the MCP server instance used to expose tools to AI agents across every package that ships a
``mcp_tools`` submodule."""
