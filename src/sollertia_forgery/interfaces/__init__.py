"""Provides CLIs and the Model Context Protocol (MCP) server exposed by installing this library into a Python
environment.
"""

from .entry_points import slf_cli
from .mcp_instance import mcp

__all__ = [
    "mcp",
    "slf_cli",
]
