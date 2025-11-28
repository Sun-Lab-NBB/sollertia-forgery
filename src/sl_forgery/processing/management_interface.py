"""This module provides the Command-Line Interfaces (CLIs) for managing Sun lab sessions and projects. Most of these
CLIs are intended to run on the remote compute server and should not be used by end-users directly.
"""

from typing import Any
from pathlib import Path

import click
from ataraxis_base_utilities import LogLevel, console

from .project_management import (
    prepare_session,
    resolve_checksum,
    generate_project_manifest,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = dict(max_content_width=120)  # or any width you want


@click.group("manage", context_settings=CONTEXT_SETTINGS)
def manage() -> None:
    """This Command-Line Interface (CLI) allows managing session and project data acquired in the Sun lab.

    This CLI is intended to run on the Sun lab remote compute server(s) and should not be called by the end-user
    directly. Instead, commands from this CLI are designed to be accessed through the bindings in the sl-experiment and
    sl-forgery libraries.
    """


# Session data management commands
@manage.group("session")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "The absolute path to the root session directory to process. This directory must contain the 'raw_data' "
        "subdirectory."
    ),
)
@click.option(
    "-pdr",
    "--processed-data-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help=(
        "The absolute path to the directory that stores the processed data from all Sun lab projects, if it is "
        "different from the root directory included in the 'session-path' argument value."
    ),
)
@click.option(
    "-id",
    "--manager-id",
    type=int,
    required=True,
    default=0,
    show_default=True,
    help="The unique identifier of the process that manages this runtime.",
)
@click.option(
    "-r",
    "--reset-tracker",
    is_flag=True,
    required=False,
    help=(
        "Determines whether to forcibly reset the tracker file for the target session management pipeline before "
        "processing runtime. This flag should only be used in exceptional cases to recover from improper runtime "
        "terminations."
    ),
)
@click.pass_context
def manage_session(
    ctx: Any, session_path: Path, processed_data_root: Path | None, manager_id: int, reset_tracker: bool
) -> None:
    """This group provides commands for managing the data of a Sun lab data acquisition session.

    Commands from this group are used to support data processing and dataset-formation (forging) on remote compute
    servers.
    """
    ctx.ensure_object(dict)
    ctx.obj["session_path"] = session_path
    ctx.obj["processed_data_root"] = processed_data_root
    ctx.obj["manager_id"] = manager_id
    ctx.obj["reset_tracker"] = reset_tracker


# noinspection PyUnresolvedReferences
@manage_session.command("checksum")
@click.pass_context
@click.option(
    "-rc",
    "--recalculate-checksum",
    is_flag=True,
    help=(
        "Determines whether to recalculate and overwrite the cached checksum value for the processed session. When "
        "the command is called with this flag, it effectively re-checksums the data instead of verifying its integrity."
    ),
)
def resolve_session_checksum(ctx: Any, recalculate_checksum: bool) -> None:
    """Resolves the data integrity checksum for the target session's 'raw_data' directory.

    This command can be used to verify the integrity of the session's 'raw_data' directory using an existing
    checksum or to re-generate the checksum to reflect the current state of the directory. It only works with the
    'raw_data' session directory and ignores all other directories. Primarily, this command is used to verify the
    integrity of the session's data as it is transferred from data acquisition systems to long-term storage
    destinations.
    """
    # Extracts shared parameters from context
    session_path = ctx.obj["session_path"]
    processed_data_root = ctx.obj["processed_data_root"]
    manager_id = ctx.obj["manager_id"]
    reset_tracker = ctx.obj["reset_tracker"]

    resolve_checksum(
        session_path=session_path,
        manager_id=manager_id,
        processed_data_root=processed_data_root,
        regenerate_checksum=recalculate_checksum,
        reset_tracker=reset_tracker,
    )


# noinspection PyUnresolvedReferences
@manage_session.command("prepare")
@click.pass_context
def prepare_session_for_processing(
    ctx: Any,
) -> None:
    """Prepares the target session for data processing by moving all session data to the working volume.

    This command is intended to run on remote compute servers that use slow HDD volumes to maximize data integrity and
    fast NVME volumes to maximize data processing speed. For such systems, moving the data to the fast volume before
    processing results in a measurable processing time decrease.
    """
    # Extracts shared parameters from context
    session_path = ctx.obj["session_path"]
    processed_data_root = ctx.obj["processed_data_root"]
    manager_id = ctx.obj["manager_id"]
    reset_tracker = ctx.obj["reset_tracker"]

    prepare_session(
        session_path=session_path,
        manager_id=manager_id,
        processed_data_root=processed_data_root,
        reset_tracker=reset_tracker,
    )


@manage.group("project")
@click.pass_context
@click.option(
    "-pp",
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the project-specific directory where raw session data is stored.",
)
@click.option(
    "-pdr",
    "--processed-data-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help=(
        "The absolute path to the directory that stores the processed data from all Sun lab projects, if it is "
        "different from the root directory included in the 'session-path' argument value."
    ),
)
def manage_project(ctx: Any, project_path: Path, processed_data_root: Path | None) -> None:
    """This group provides commands for managing the data of a Sun lab project.

    Commands from this group are used to support all interactions with the data stored on the Sun lab remote compute
    server(s).
    """
    ctx.ensure_object(dict)
    ctx.obj["project_path"] = project_path
    ctx.obj["processed_data_root"] = processed_data_root


# noinspection PyUnresolvedReferences
@manage_project.command("manifest")
@click.pass_context
def generate_project_manifest_file(ctx: Any) -> None:
    """Generates the manifest .feather file that captures the current state of the target project's data.

    The manifest file contains the comprehensive snapshot of the available project's data. It includes the information
    about the management and processing pipelines that have been applied to each session's data, as well as the
    descriptive information about each session. The manifest file is used as an entry-point for all interactions with
    the Sun lab data stored on the remote compute server(s).
    """
    # Extracts shared parameters from context
    project_path = ctx.obj["project_path"]
    processed_data_root = ctx.obj["processed_data_root"]

    generate_project_manifest(
        raw_project_directory=project_path,
        processed_data_root=processed_data_root,
        manager_id=1,
    )

    console.echo(message=f"Project {Path(project_path).stem} data manifest file: generated.", level=LogLevel.SUCCESS)
