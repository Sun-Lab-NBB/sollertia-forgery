"""Provides the shared MCP server instance that the interface tool modules register their tools on."""

from __future__ import annotations

from mcp.server import MCPServer

mcp: MCPServer = MCPServer(name="sollertia-forgery")
"""Stores the MCP server instance that exposes tools to AI agents."""
