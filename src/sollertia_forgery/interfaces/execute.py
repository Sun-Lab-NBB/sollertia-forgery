"""Provides CLIs for executing management, processing, and analysis workflows on adopted project sessions."""

import click
from ataraxis_base_utilities import console
from sollertia_shared_assets import filter_sessions, get_working_directory

from ..server import (
    Server,
    forge_dataset,
    manage_project_data,
    process_project_data,
    get_server_configuration,
    resolve_project_manifest,
)
from ..forging import DatasetSession
from ..shared_assets import ProjectManifest

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("execute", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project whose sessions to work with.",
)
@click.option(
    "-sd",
    "--start-date",
    type=str,
    required=False,
    help=(
        "The start date for selecting the sessions to work with (format: YYYY-MM-DD). Sessions recorded on or after "
        "this date are included."
    ),
)
@click.option(
    "-ed",
    "--end-date",
    type=str,
    required=False,
    help=(
        "The end date for selecting the sessions to work with (format: YYYY-MM-DD). Sessions recorded on or before "
        "this date are included."
    ),
)
@click.option(
    "-is",
    "--include-session",
    type=str,
    multiple=True,
    help=(
        "The session(s) to work with. Can be specified multiple times. These sessions are included even if they "
        "fall outside the date range specified with the --start-date and --end-date arguments, unless explicitly "
        "excluded by the --exclude-session argument."
    ),
)
@click.option(
    "-es",
    "--exclude-session",
    type=str,
    multiple=True,
    help=(
        "The session(s) to exclude from processing. Can be specified multiple times. Exclusion takes precedence over "
        "all other inclusion criteria."
    ),
)
@click.option(
    "-ia",
    "--include-animal",
    type=str,
    multiple=True,
    help=(
        "The animal(s) whose sessions to work with. Can be specified multiple times. If not specified, sessions from "
        "all animals participating in the target project are considered for inclusion."
    ),
)
@click.option(
    "-ea",
    "--exclude-animal",
    type=str,
    multiple=True,
    help=(
        "The animal(s) whose sessions to exclude from processing. Can be specified multiple times. Exclusion takes "
        "precedence over inclusion."
    ),
)
@click.option(
    "-k",
    "--keep-job-logs",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to keep completed job logs on the server or (default) remove them after each pipeline "
        "completes successfully. If the pipeline fails, the job logs are kept regardless of this argument's value."
    ),
)
def execute_cli(
    ctx: click.Context,
    *,
    project: str,
    start_date: str | None,
    end_date: str | None,
    include_session: tuple[str, ...],
    exclude_session: tuple[str, ...],
    include_animal: tuple[str, ...],
    exclude_animal: tuple[str, ...],
    keep_job_logs: bool,
) -> None:
    """Executes management, processing, and analysis data workflows on the adopted project's sessions.

    This CLI group functions as the entry-point for all data processing pipelines supported by the Sollertia data
    workflows. See the documentation for each of the workflow subgroups ('managing', 'processing', 'forging', or
    'analysis') for the available processing pipelines.
    """
    ctx.ensure_object(dict)

    # Resolves the path to the manifest file.
    manifest_path = get_working_directory().joinpath(project, "manifest.feather")

    # If the manifest file does not exist on the local machine, ensures it is fetched from the remote server.
    if not manifest_path.exists():
        configuration = get_server_configuration()
        server = Server(configuration=configuration)
        resolve_project_manifest(project=project, server=server, generate=False)

    # Loads the manifest file data into memory.
    manifest = ProjectManifest(manifest_file=manifest_path)

    # Builds the set of all available sessions from the manifest.
    all_sessions: set[DatasetSession] = set()
    for animal_id in manifest.animals:
        for session_name in manifest.get_sessions(animal=animal_id, exclude_incomplete=False):
            all_sessions.add(DatasetSession(session=session_name, animal=str(animal_id)))

    # Applies filtering based on the provided options. sollertia-shared-assets' filter_sessions operates
    # on plain (session_name, animal) tuples; the map rehydrates the filtered keys back to DatasetSession.
    session_map = {(session.session, session.animal): session for session in all_sessions}
    filtered_sessions = {
        session_map[key]
        for key in filter_sessions(
            sessions=set(session_map.keys()),
            start_date=start_date,
            end_date=end_date,
            include_sessions=set(include_session) if include_session else None,
            exclude_sessions=set(exclude_session) if exclude_session else None,
            include_animals=set(include_animal) if include_animal else None,
            exclude_animals=set(exclude_animal) if exclude_animal else None,
        )
    }

    # If no sessions match the filter criteria, raises an error.
    if not filtered_sessions:
        message = "No sessions match the specified filtering criteria. Adjust the filtering criteria and try again."
        console.error(message=message, error=ValueError)

    # Stores resolved data in the context for subcommands.
    ctx.obj["project"] = project
    ctx.obj["manifest_path"] = manifest_path
    ctx.obj["sessions"] = tuple(filtered_sessions)
    ctx.obj["keep_job_logs"] = keep_job_logs


@execute_cli.group("manage")
def manage_cli() -> None:
    """Manages adopted project sessions on the remote compute server.

    This command group provides the pipelines for managing the adopted session's data integrity and lifecycle.
    """


# noinspection PyUnresolvedReferences
@manage_cli.command("checksum")
@click.option(
    "-r",
    "--recompute",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to recompute (regenerate) the data integrity checksum instead of verifying it. "
        "This overwrites the existing checksum stored in the ax_checksum.txt file for each session."
    ),
)
@click.pass_context
def checksum_command(ctx: click.Context, *, recompute: bool) -> None:
    """Verifies or recomputes the data integrity checksum for the selected sessions.

    By default, this command verifies that the session data matches the stored checksum. Use the --recompute flag
    to regenerate the checksum instead (useful after intentionally modifying the session's data).
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]
    manifest_path = ctx.obj["manifest_path"]
    sessions = ctx.obj["sessions"]
    keep_job_logs = ctx.obj["keep_job_logs"]

    # Executes the checksum operation.
    manage_project_data(
        manifest_path=manifest_path,
        project=project,
        sessions=sessions,
        verify_checksum=not recompute,
        recompute_checksum=recompute,
        delete_sessions=False,
        keep_job_logs=keep_job_logs,
    )


# noinspection PyUnresolvedReferences
@manage_cli.command("delete")
@click.pass_context
def delete_command(ctx: click.Context) -> None:
    """Deletes the selected sessions from the user's working directory on the remote server."""
    # Retrieves shared context data.
    project = ctx.obj["project"]
    manifest_path = ctx.obj["manifest_path"]
    sessions = ctx.obj["sessions"]
    keep_job_logs = ctx.obj["keep_job_logs"]

    # Executes the deletion operation.
    manage_project_data(
        manifest_path=manifest_path,
        project=project,
        sessions=sessions,
        verify_checksum=False,
        recompute_checksum=False,
        delete_sessions=True,
        keep_job_logs=keep_job_logs,
    )


# noinspection PyUnresolvedReferences
@execute_cli.command("process")
@click.option(
    "-b",
    "--behavior",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to execute the behavior data processing pipeline.",
)
@click.option(
    "-s",
    "--cindra",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to execute the single-day cindra data processing pipeline.",
)
@click.option(
    "-r",
    "--reprocess",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to reprocess sessions that have already been processed.",
)
@click.option(
    "-sc",
    "--cindra-config",
    type=str,
    default="GCaMP6f_CA1_SD.yaml",
    show_default=True,
    help="The name of the configuration file for the single-day cindra processing pipeline.",
)
@click.option(
    "-pc",
    "--plane-count",
    type=int,
    default=3,
    show_default=True,
    help="The number of planes in the session's data to be processed with the single-day cindra pipeline.",
)
@click.option(
    "-bs",
    "--batch-size",
    type=int,
    default=4,
    show_default=True,
    help="The number of processing pipelines that can be submitted to the remote compute server at a time.",
)
@click.pass_context
def process_command(
    ctx: click.Context,
    *,
    behavior: bool,
    cindra: bool,
    reprocess: bool,
    cindra_config: str,
    plane_count: int,
    batch_size: int,
) -> None:
    """Executes data processing pipelines for the selected sessions.

    This command runs the behavior and/or single-day cindra processing pipelines on the selected sessions.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]
    manifest_path = ctx.obj["manifest_path"]
    sessions = ctx.obj["sessions"]
    keep_job_logs = ctx.obj["keep_job_logs"]

    # Executes the processing operation.
    process_project_data(
        manifest_path=manifest_path,
        project=project,
        sessions=sessions,
        process_behavior=behavior,
        process_cindra=cindra,
        reprocess=reprocess,
        keep_job_logs=keep_job_logs,
        cindra_configuration_file=cindra_config,
        plane_count=plane_count,
        processing_batch_size=batch_size,
    )


# noinspection PyUnresolvedReferences
@execute_cli.command("forge")
@click.option(
    "-dn",
    "--dataset-name",
    type=str,
    required=True,
    help="The unique name for the dataset to create.",
)
@click.option(
    "-m",
    "--multiday",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to execute the multi-day cindra processing pipeline.",
)
@click.option(
    "-a",
    "--assemble",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to execute the data assembly pipeline.",
)
@click.option(
    "-r",
    "--reprocess",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to reprocess sessions that have already been processed.",
)
@click.option(
    "-sc",
    "--cindra-config",
    type=str,
    default="GCaMP6f_CA1_MD.yaml",
    show_default=True,
    help="The name of the configuration file for the multi-day cindra processing pipeline.",
)
@click.option(
    "-bs",
    "--batch-size",
    type=int,
    default=4,
    show_default=True,
    help="The number of processing pipelines that can be submitted to the remote compute server at a time.",
)
@click.pass_context
def forge_command(
    ctx: click.Context,
    *,
    dataset_name: str,
    multiday: bool,
    assemble: bool,
    reprocess: bool,
    cindra_config: str,
    batch_size: int,
) -> None:
    """Forges an analysis dataset from the selected sessions.

    This command creates a new dataset from the selected sessions, optionally runs the multi-day processing
    pipeline, and/or assembles the processed data into unified data.feather files. The session type and acquisition
    system are derived automatically from the first session's metadata.
    """
    # Retrieves shared context data.
    project = ctx.obj["project"]
    manifest_path = ctx.obj["manifest_path"]
    sessions = ctx.obj["sessions"]
    keep_job_logs = ctx.obj["keep_job_logs"]

    # Executes the forging operation.
    forge_dataset(
        manifest_path=manifest_path,
        project=project,
        sessions=sessions,
        dataset_name=dataset_name,
        process_multiday=multiday,
        assemble_data=assemble,
        reprocess=reprocess,
        keep_job_logs=keep_job_logs,
        cindra_configuration_file=cindra_config,
        processing_batch_size=batch_size,
    )
