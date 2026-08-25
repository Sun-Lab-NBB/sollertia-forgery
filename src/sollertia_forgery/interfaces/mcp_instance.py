"""Provides the shared MCP server instance on which the interface tool modules register their tools."""

from __future__ import annotations

from mcp.server import MCPServer

mcp: MCPServer = MCPServer(name="sollertia-forgery")
"""Stores the MCP server instance that exposes tools to AI agents."""
