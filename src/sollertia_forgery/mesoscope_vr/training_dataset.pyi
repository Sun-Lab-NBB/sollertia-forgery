from pathlib import Path

from .video_dataset import (
    assemble_video_dataset as assemble_video_dataset,
    resolve_slowest_camera_clock as resolve_slowest_camera_clock,
)
from .runtime_dataset import clip_to_session_bounds as clip_to_session_bounds
from .behavior_dataset import assemble_behavior_dataset as assemble_behavior_dataset

def assemble_training_dataset(source_session_path: Path, output_path: Path) -> None: ...
