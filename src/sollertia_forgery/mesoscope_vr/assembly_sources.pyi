from pathlib import Path

from sollertia_shared_assets import (
    SessionData as SessionData,
    SessionTypes,
)

from .metadata import BehaviorDataFiles as BehaviorDataFiles
from .video_dataset import count_camera_source_samples as count_camera_source_samples
from ..shared_assets import count_feather_rows as count_feather_rows

_TRAINING_MICROCONTROLLER_SOURCES: tuple[BehaviorDataFiles, ...]
_TRAINING_RUNTIME_SOURCES: tuple[BehaviorDataFiles, ...]
_EXPERIMENT_MICROCONTROLLER_SOURCES: tuple[BehaviorDataFiles, ...]
_EXPERIMENT_RUNTIME_SOURCES: tuple[BehaviorDataFiles, ...]
_SOURCE_FILES: dict[SessionTypes, tuple[tuple[BehaviorDataFiles, ...], tuple[BehaviorDataFiles, ...]]]

def resolve_mesoscope_assembly_sources(session: SessionData) -> tuple[int, ...]: ...
def _source_samples(directory: Path, files: tuple[BehaviorDataFiles, ...]) -> tuple[int, ...]: ...
