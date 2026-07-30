from pathlib import Path

from sollertia_shared_assets import (
    DatasetData,
    SessionTypes as SessionTypes,
)

from .admission import verify_session_admissibility as verify_session_admissibility
from ..registries import resolve_forging_column_descriptions as resolve_forging_column_descriptions

DATASET_MARKER_FILENAME: str

def resolve_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
    force_recreate: bool = False,
    recreate_animals: tuple[str, ...] = (),
) -> DatasetData: ...
def discover_project_datasets(project_root: Path) -> list[DatasetData]: ...
def _create_dataset(
    name: str, sessions: tuple[str, ...], project_root: Path, *, required_session_type: SessionTypes | None = None
) -> DatasetData: ...
def _update_dataset(
    dataset: DatasetData, session_names: tuple[str, ...], project_root: Path, recreate_animals: tuple[str, ...]
) -> None: ...
def _resolve_session_paths(sessions: tuple[str, ...], project_root: Path) -> list[Path]: ...
def _verify_session_compatibility(dataset: DatasetData, session_paths: list[Path]) -> None: ...
def _copy_animal_surgery_files(dataset_name: str, dataset: DatasetData, source_session_paths: list[Path]) -> None: ...
