from pathlib import Path
from dataclasses import dataclass

import click
from _typeshed import Incomplete

from ..forging import generate_dataset_state as generate_dataset_state
from ..managing import (
    ProjectManifest as ProjectManifest,
    project_manifest_path as project_manifest_path,
    generate_project_manifest as generate_project_manifest,
    run_checksum_processing_pipeline as run_checksum_processing_pipeline,
)
from ..orchestration import (
    BATCH_PIPELINES as BATCH_PIPELINES,
    reset_tracked_jobs as reset_tracked_jobs,
    clean_pipeline_output as clean_pipeline_output,
)

_CONTEXT_SETTINGS: dict[str, int]

@dataclass(frozen=True, slots=True)
class _SharedManifestParameters:
    project_path: Path | None
    def require_project_path(self) -> Path: ...

_pass_shared_parameters: Incomplete

@click.pass_context
def manifest_cli(context: click.Context, project_path: Path | None) -> None: ...
@_pass_shared_parameters
def create_manifest(shared: _SharedManifestParameters, *, no_progress: bool) -> None: ...
@_pass_shared_parameters
def print_project_manifest_data(
    shared: _SharedManifestParameters, *, animal: str | None, notes: bool, summary: bool
) -> None: ...
def checksum_command(session_path: Path, workers: int, *, regenerate_checksum: bool, no_progress: bool) -> None: ...
def dataset_state_command(dataset_path: tuple[Path, ...]) -> None: ...
def reset_command(pipeline: str, unit_path: tuple[Path, ...], job_id: tuple[str, ...]) -> None: ...
def clean_command(pipeline: str, unit_path: tuple[Path, ...]) -> None: ...
