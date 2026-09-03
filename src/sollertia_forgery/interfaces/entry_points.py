"""Provides the single ``slf`` console-script root command group for the sollertia-forgery library."""

from __future__ import annotations

from typing import Literal
from pathlib import Path

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
from ..shared_assets import OpenMPStatus, resolve_openmp_runtime

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the sollertia platform standard."""


@click.group("slf", context_settings=_CONTEXT_SETTINGS)
def slf_cli() -> None:
    """Processes and manages data acquired with the Sollertia data acquisition platform.

    Exposes system-agnostic management commands ('manifest', 'checksum', 'dataset-state', 'server', 'reset',
    'clean', 'omp'), the agentic MCP server ('mcp'), and the generic processing, forging, and planning commands
    ('process', 'forge', 'plan'). The acquisition system is inferred from the data, so no command takes a system
    selector.
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
    transport that local agent clients are expected to use. The 'sse' and 'streamable-http' transports instead serve the
    same tools over the network, which is how remote agent clients reach a server running on the processing host.
    """
    # Importing at module level runs '_register_tool_modules', which imports every MCP tool module and the
    # pipelines they reach, so every 'slf' subcommand would pay that cost at startup.
    from .mcp_server import run_server  # noqa: PLC0415

    # The stdio transport sends the JSON-RPC messages over stdout, which is also where the console writes every message
    # up to the WARNING level. Since the MCP tools drive the processing pipelines, which echo status updates as they
    # work, the console has to be silenced: any echoed line lands inside a JSON-RPC message and makes it unparsable for
    # the connected client. The network transports leave stdout unused, so the console stays on and reports that the
    # server has started.
    if transport == "stdio":
        console.disable()
    else:
        console.echo(
            message=f"Starting the sollertia-forgery MCP server with the {transport} transport.",
            level=LogLevel.INFO,
        )

    run_server(transport=transport)


@slf_cli.command("omp")
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help=(
        "The path to the OpenMP runtime to link. Omit to search the macOS package manager directories, the active "
        "conda environment, and the installed Python distributions for one."
    ),
)
@click.option(
    "-t",
    "--target",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path receiving the link. Omit to derive it from the directory the dynamic loader searches by default.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Determines whether to link a runtime on a host whose OpenMP runtime already loads.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help=(
        "Determines whether to create the resolved link. Without this flag the command reports what it would do and "
        "changes nothing."
    ),
)
def omp_command(source: Path | None, target: Path | None, *, force: bool, yes: bool) -> None:
    """Links the OpenMP runtime that the Numba threading layer loads on macOS into a directory the loader searches.

    The Numba macOS wheel names its OpenMP dependency through an rpath that carries no entries, so the runtime
    resolves from the dynamic loader's default search path alone. This command finds an installed runtime and links it
    into that path. Writing the link usually requires running the command through sudo. Running the command on any
    other platform errors, because those platforms run the TBB threading layer instead.
    """
    summary = resolve_openmp_runtime(runtime_path=source, link_path=target, execute=yes, force=force)

    if summary.searched_paths:
        console.echo(message=f"searched: {', '.join(str(path) for path in summary.searched_paths)}", raw=True)
    if summary.runtime_path is not None:
        console.echo(message=f"runtime:  {summary.runtime_path}", raw=True)
        console.echo(message=f"link:     {summary.link_path}", raw=True)
    console.echo(message=summary.describe())
    if summary.status == OpenMPStatus.UNRESOLVED:
        raise SystemExit(1)


def _register_subcommands() -> None:
    """Registers every subcommand and subcommand group this module does not define itself on the ``slf`` group."""
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
