"""Provides Model Context Protocol (MCP) tools for authoring and reading the remote compute server configuration."""

from __future__ import annotations

import uuid
import contextlib
from typing import Any
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from ..interfaces import mcp
from .server_configuration import (
    ServerConfiguration,
    get_server_configuration,
    get_server_configuration_path,
)


def _ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def _error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}


def _serialize(instance: ServerConfiguration) -> dict[str, Any]:
    """Converts a ServerConfiguration instance into a JSON-friendly dict."""
    return {"username": instance.username, "password": instance.password, "host": instance.host}


@mcp.tool()
def read_server_configuration_tool() -> dict[str, Any]:
    """Loads the ServerConfiguration from the working directory with the password masked.

    Returns:
        A response dict with ``data`` containing the server configuration payload. The password field is
        replaced with the literal string ``"<masked>"`` for security.
    """
    try:
        instance = get_server_configuration()
    except (FileNotFoundError, OSError, ValueError) as exception:
        return _error_response(message=str(exception))
    serialized = _serialize(instance=instance)
    serialized["password"] = "<masked>"  # noqa: S105 - literal masking placeholder, not a real password.
    return _ok_response(data=serialized)


@mcp.tool()
def write_server_configuration_tool(
    configuration_payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Creates or replaces the ServerConfiguration YAML in the working directory.

    Args:
        configuration_payload: The complete ServerConfiguration payload (must include ``username``, ``password``,
            and ``host``).
        overwrite: Determines whether to overwrite an existing server configuration file.

    Returns:
        A response dict with ``file_path`` and ``data`` containing the validated payload with the password masked.
    """
    try:
        file_path = get_server_configuration_path()
    except FileNotFoundError as exception:
        return _error_response(message=str(exception))

    if file_path.exists() and not overwrite:
        return _error_response(message=f"File already exists: {file_path}. Pass overwrite=True to replace.")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Writes the payload to a temporary sibling file and validates by round-tripping through ServerConfiguration.
    # Keeps the temp file ending in .yaml because YamlConfig.from_yaml rejects non-.yaml paths.
    temp_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        temp_path.write_text(yaml.safe_dump(data=configuration_payload, sort_keys=False))
        instance = ServerConfiguration.from_yaml(file_path=temp_path)
    except Exception as exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        return _error_response(message=f"Validation failed for ServerConfiguration: {exception}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()

    try:
        instance.to_yaml(file_path=file_path)
    except Exception as exception:
        return _error_response(message=f"Failed to persist ServerConfiguration to {file_path}: {exception}")

    serialized = _serialize(instance=instance)
    serialized["password"] = "<masked>"  # noqa: S105 - literal masking placeholder, not a real password.
    return _ok_response(file_path=str(file_path), data=serialized)
