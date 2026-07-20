"""Provides the single ``slf`` console-script root command group for the sollertia-forgery library.

Notes:
    One root command exposes the system-agnostic management commands (manifest, checksum, server), the agentic Model
    Context Protocol server, and the generic processing and forging commands. Subcommand groups are imported and
    registered lazily at module load so that resolving ``slf --help`` stays inexpensive, and the heavy
    acquisition-library bindings are pulled in only inside the individual command callbacks.
"""

import click

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.group("slf", context_settings=CONTEXT_SETTINGS)
def slf_cli() -> None:
    """Processes and manages data acquired with the Sollertia data acquisition platform.

    Exposes system-agnostic management commands ('manifest', 'checksum', 'server'), the agentic MCP server ('mcp'),
    and the generic processing and forging commands ('process', 'forge'). The acquisition system is inferred from
    the data, so no command takes a system selector.
    """


@slf_cli.command("mcp")
def run_mcp_server_command() -> None:
    """Starts the agentic Model Context Protocol server using the stdio transport."""
    from .mcp_server import run_mcp_server  # noqa: PLC0415

    run_mcp_server()


def _register_subcommands() -> None:
    """Imports and registers every subcommand group on the top-level ``slf`` Click group."""
    from .forge import forge_command  # noqa: PLC0415
    from .manage import manifest_cli, checksum_command  # noqa: PLC0415
    from .server import server_cli  # noqa: PLC0415
    from .process import process_cli  # noqa: PLC0415

    slf_cli.add_command(cmd=manifest_cli)
    slf_cli.add_command(cmd=checksum_command)
    slf_cli.add_command(cmd=server_cli)
    slf_cli.add_command(cmd=process_cli)
    slf_cli.add_command(cmd=forge_command)


_register_subcommands()
