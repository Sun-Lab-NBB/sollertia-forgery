"""Provides the Model Context Protocol (MCP) tools for authoring and reading the remote compute server configuration."""

from __future__ import annotations

import uuid
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..server import (
    ServerConfiguration,
    get_server_configuration,
    get_server_configuration_path,
)
from .responses import ok_response, error_response
from .mcp_instance import mcp


@mcp.tool()
def read_server_configuration_tool() -> dict[str, Any]:
    """Loads the ServerConfiguration from the working directory with the password masked.

    Returns:
        A response dict with ``data`` containing the server configuration payload. The password field is
        replaced with the literal string ``"<masked>"`` for security.
    """
    try:
        instance = get_server_configuration()
    except (OSError, ValueError) as exception:
        return error_response(message=f"Unable to read the server configuration. {exception}")
    serialized = _render_configuration(instance=instance)
    serialized["password"] = "<masked>"  # noqa: S105 - literal masking placeholder, not a real password.
    return ok_response(data=serialized)


@mcp.tool()
def write_server_configuration_tool(
    configuration_payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Creates or replaces the ServerConfiguration YAML in the working directory.

    Args:
        configuration_payload: The complete ServerConfiguration payload. Supply ``username``, ``password``, ``host``,
            ``root``, and ``environment``, since an omitted field is persisted as an empty string rather than rejected.
        overwrite: Determines whether to overwrite an existing server configuration file.

    Returns:
        A response dict with ``file_path`` and ``data`` containing the validated payload with the password masked.
    """
    try:
        file_path = get_server_configuration_path()
    except FileNotFoundError as exception:
        return error_response(message=f"Unable to resolve the server configuration path. {exception}")

    if file_path.exists() and not overwrite:
        return error_response(
            message=(
                f"Unable to write the server configuration. A file already exists at '{file_path}'. Pass "
                f"overwrite=True to replace it."
            )
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Writes the payload to a temporary sibling file and validates by round-tripping through ServerConfiguration.
    # Keeps the temp file ending in .yaml because YamlConfig.from_yaml rejects non-.yaml paths.
    temp_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        temp_path.write_text(yaml.safe_dump(data=configuration_payload, sort_keys=False))
        instance = ServerConfiguration.from_yaml(file_path=temp_path)
    except Exception as exception:
        return error_response(message=f"Unable to validate the supplied server configuration payload. {exception}")
    finally:
        temp_path.unlink(missing_ok=True)

    try:
        instance.to_yaml(file_path=file_path)
    except Exception as exception:
        return error_response(message=f"Unable to write the server configuration to '{file_path}'. {exception}")

    serialized = _render_configuration(instance=instance)
    serialized["password"] = "<masked>"  # noqa: S105 - literal masking placeholder, not a real password.
    return ok_response(file_path=str(file_path), data=serialized)


def _render_configuration(instance: ServerConfiguration) -> dict[str, Any]:
    """Converts a ServerConfiguration instance into a JSON-friendly dict.

    Args:
        instance: The configuration to render.

    Returns:
        A dictionary carrying the username, password, host, root, and environment the configuration holds.
    """
    return {
        "username": instance.username,
        "password": instance.password,
        "host": instance.host,
        "root": instance.root,
        "environment": instance.environment,
    }
