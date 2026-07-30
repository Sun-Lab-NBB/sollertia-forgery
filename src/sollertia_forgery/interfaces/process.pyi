from pathlib import Path
from dataclasses import dataclass

import click
from _typeshed import Incomplete

from ..video import run_video_processing_pipeline as run_video_processing_pipeline
from ..runtime import run_runtime_processing_pipeline as run_runtime_processing_pipeline
from ..two_photon import run_two_photon_processing_pipeline as run_two_photon_processing_pipeline
from ..microcontrollers import run_microcontroller_processing_pipeline as run_microcontroller_processing_pipeline

_CONTEXT_SETTINGS: dict[str, int]

@dataclass(frozen=True, slots=True)
class _SharedProcessingParameters:
    session_path: Path | None
    job_id: str | None
    workers: int
    display_progress: bool
    def require_session_path(self) -> Path: ...

_pass_shared_parameters: Incomplete

@click.pass_context
def process_cli(
    context: click.Context, session_path: Path | None, job_id: str | None, workers: int, *, no_progress: bool
) -> None: ...
@_pass_shared_parameters
def video_command(
    shared: _SharedProcessingParameters, target_camera: int, *, timestamp: bool, track: bool, energy: bool
) -> None: ...
@_pass_shared_parameters
def microcontroller_command(shared: _SharedProcessingParameters) -> None: ...
@_pass_shared_parameters
def runtime_command(shared: _SharedProcessingParameters) -> None: ...
@_pass_shared_parameters
def two_photon_command(
    shared: _SharedProcessingParameters,
    target_plane: int,
    *,
    binarize: bool,
    register: bool,
    process: bool,
    combine: bool,
) -> None: ...
