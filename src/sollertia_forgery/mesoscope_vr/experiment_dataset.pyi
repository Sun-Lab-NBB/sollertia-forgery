from pathlib import Path

from .video_dataset import assemble_video_dataset as assemble_video_dataset
from ..shared_assets import multi_recording_dataset_directory as multi_recording_dataset_directory
from .runtime_dataset import (
    clip_to_session_bounds as clip_to_session_bounds,
    assemble_runtime_dataset as assemble_runtime_dataset,
    mask_non_run_experiment_data as mask_non_run_experiment_data,
)
from .behavior_dataset import assemble_behavior_dataset as assemble_behavior_dataset
from .two_photon_dataset import assemble_cindra_dataset as assemble_cindra_dataset

def assemble_experiment_dataset(source_session_path: Path, output_path: Path, dataset_name: str) -> None: ...
