"""Provides the single ``slf`` console-script root command group for the sollertia-forgery library."""

from __future__ import annotations

from typing import Literal

import click
from ataraxis_base_utilities import LogLevel, console

from .plan import plan_cli
from .forge import forge_command
from .manage import (
    manifest_cli,
    clean_command,
    reset_command,
    checksum_command,
    dataset_state_command,
)
from .server import server_cli
from .process import process_cli

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.group("slf", context_settings=_CONTEXT_SETTINGS)
def slf_cli() -> None:
    """Processes and manages data acquired with the Sollertia data acquisition platform.

    Exposes system-agnostic management commands ('manifest', 'checksum', 'dataset-state', 'server', 'reset',
    'clean'), the agentic MCP server ('mcp'), and the generic processing, forging, and planning commands ('process',
    'forge', 'plan'). The acquisition system is inferred from the data, so no command takes a system selector.
    """


@slf_cli.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="The transport protocol the MCP server uses to communicate with the connected client.",
)
def run_mcp_server_command(transport: Literal["stdio", "sse", "streamable-http"]) -> None:
    """Starts the agentic Model Context Protocol server using the requested transport.

    The 'stdio' transport exchanges messages over the standard input and output streams of this process, which is the
    transport local agent clients are expected to use. The 'sse' and 'streamable-http' transports instead serve the
    same tools over the network, which is how remote agent clients reach a server running on the processing host.
    """
    from .mcp_server import run_server  # noqa: PLC0415

    # The stdio transport sends the JSON-RPC messages over stdout, which is also the stream the console writes all
    # messages up to the WARNING level to. Since the MCP tools drive the processing pipelines, which echo status
    # updates as they work, the console has to be silenced: any echoed line lands inside a JSON-RPC message and makes
    # it unparsable for the connected client. The network transports leave stdout unused, so the console stays on and
    # reports that the server has started.
    if transport == "stdio":
        console.disable()
    else:
        console.echo(
            message=f"Starting the sollertia-forgery MCP server with the {transport} transport.",
            level=LogLevel.INFO,
        )

    run_server(transport=transport)


def _register_subcommands() -> None:
    """Registers every subcommand and subcommand group on the top-level ``slf`` Click group."""
    slf_cli.add_command(cmd=manifest_cli)
    slf_cli.add_command(cmd=checksum_command)
    slf_cli.add_command(cmd=dataset_state_command)
    slf_cli.add_command(cmd=reset_command)
    slf_cli.add_command(cmd=clean_command)
    slf_cli.add_command(cmd=server_cli)
    slf_cli.add_command(cmd=process_cli)
    slf_cli.add_command(cmd=forge_command)
    slf_cli.add_command(cmd=plan_cli)


_register_subcommands()
