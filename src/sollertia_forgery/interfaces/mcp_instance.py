"""Provides the shared FastMCP instance that the interface tool modules register their tools on."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP(name="sollertia-forgery", json_response=True)
"""Stores the MCP server instance that exposes tools to AI agents. Every interface tool module registers its tools
on this shared instance."""
