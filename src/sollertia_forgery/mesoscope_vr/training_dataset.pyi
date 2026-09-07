from pathlib import Path

from sollertia_shared_assets import SessionData

from .video_dataset import (
    assemble_video_dataset as assemble_video_dataset,
    resolve_slowest_camera_clock as resolve_slowest_camera_clock,
    resolve_reference_clock_samples as resolve_reference_clock_samples,
)
from ..shared_assets import AssemblyGeometry as AssemblyGeometry
from .runtime_dataset import clip_to_session_bounds as clip_to_session_bounds
from .assembly_sources import resolve_mesoscope_assembly_sources as resolve_mesoscope_assembly_sources
from .behavior_dataset import assemble_behavior_dataset as assemble_behavior_dataset

def assemble_training_dataset(source_session_path: Path, output_path: Path) -> None: ...
def resolve_mesoscope_assembly_geometry(session: SessionData) -> AssemblyGeometry: ...
