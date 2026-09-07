from pathlib import Path
from dataclasses import dataclass
from collections.abc import Collection

from cindra import MultiRecordingJobNames
from sollertia_shared_assets import (
    DatasetData,
    DatasetSession as DatasetSession,
)
from ataraxis_data_structures import ProcessingTracker

from .dataset import resolve_dataset as resolve_dataset
from ..registries import (
    ForgingAssembler as ForgingAssembler,
    resolve_forging_assembly_worker as resolve_forging_assembly_worker,
    resolve_multi_recording_session_types as resolve_multi_recording_session_types,
    resolve_multi_recording_configuration_resolver as resolve_multi_recording_configuration_resolver,
)
from ..shared_assets import (
    verify_openmp_runtime as verify_openmp_runtime,
    multi_recording_dataset_name as multi_recording_dataset_name,
)

MULTIDAY_DISCOVERY_JOB_NAME: str
MULTIDAY_EXTRACTION_JOB_NAME: str
FORGING_JOB_NAME: str
FORGING_JOB_CONCURRENCY_LIMITS: dict[str, int]
_MULTIDAY_JOB_NAMES: dict[MultiRecordingJobNames, str]

@dataclass(frozen=True, slots=True)
class _MultidayStage:
    configuration_path: Path
    job_name: MultiRecordingJobNames
    specifier: str
    prime: bool

def define_forging_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    display_progress: bool = False,
    force_recreate: bool = False,
    recreate_animals: tuple[str, ...] = (),
) -> DatasetData: ...
def run_forging_pipeline(
    name: str, project_root: Path, job_id: str | None = None, *, workers: int = -1, display_progress: bool = False
) -> None: ...
def forging_tracker_path(dataset: DatasetData) -> Path: ...
def discover_forging_jobs(dataset_path: Path) -> tuple[DatasetData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def forging_cross_recording_paths(dataset: DatasetData) -> tuple[Path, ...]: ...
def forging_job_prerequisites(
    dataset: DatasetData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _materialize_multiday_plan(
    dataset: DatasetData, project_root: Path, *, display_progress: bool, animals: Collection[str] | None = None
) -> dict[str, tuple[Path, list[str]]]: ...
def _load_multiday_plan(dataset: DatasetData) -> dict[str, tuple[Path, list[str]]]: ...
def _build_forging_universe(
    dataset: DatasetData, multiday_plan: dict[str, tuple[Path, list[str]]]
) -> list[tuple[str, str]]: ...
def _forging_job(animal: str, job: tuple[str, str]) -> tuple[str, str]: ...
def _resolve_multiday_stages(
    multiday_plan: dict[str, tuple[Path, list[str]]],
) -> dict[tuple[str, str], _MultidayStage]: ...
def _resolve_runnable_jobs(tracker: ProcessingTracker, universe: list[tuple[str, str]]) -> list[tuple[str, str]]: ...
def _reset_animal_jobs(tracker: ProcessingTracker, dataset: DatasetData, animals: tuple[str, ...]) -> None: ...
def _run_multiday_job(
    stage: _MultidayStage, job: tuple[str, str], tracker: ProcessingTracker, job_id: str, workers: int
) -> None: ...
def _execute_remote_forging_job(
    job_id: str,
    universe: list[tuple[str, str]],
    dataset: DatasetData,
    session_lookup: dict[str, DatasetSession],
    multiday_stages: dict[tuple[str, str], _MultidayStage],
    project_root: Path,
    tracker: ProcessingTracker,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    workers: int,
) -> None: ...
def _execute_jobs_sequential(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    *,
    display_progress: bool,
) -> None: ...
def _execute_jobs_parallel(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssembler,
    described_columns: frozenset[str],
    workers: int,
    *,
    display_progress: bool,
) -> None: ...
def _execute_job(
    session_name: str,
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_id: str,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
) -> None: ...
def _forge_session(
    source_session_path: Path,
    output_path: Path,
    dataset_name: str,
    worker: ForgingAssembler,
    described_columns: frozenset[str],
) -> None: ...
