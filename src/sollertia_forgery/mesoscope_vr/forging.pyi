from pathlib import Path

from sollertia_shared_assets import SessionTypes

from ..shared_assets import ProcessingPipelines as ProcessingPipelines
from .training_dataset import assemble_training_dataset as assemble_training_dataset
from .experiment_dataset import assemble_experiment_dataset as assemble_experiment_dataset

MESOSCOPE_ADMISSION_PIPELINES: dict[SessionTypes, frozenset[ProcessingPipelines]]
_TRAINING_SESSION_TYPES: frozenset[SessionTypes]

def assemble_mesoscope_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None: ...
