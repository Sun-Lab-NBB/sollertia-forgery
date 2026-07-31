from pathlib import Path

from sollertia_shared_assets import (
    DatasetData,
    DatasetSession as DatasetSession,
)
from ataraxis_data_structures import ProcessingTracker

from .dataset import resolve_dataset as resolve_dataset
from ..registries import (
    ForgingAssembler as ForgingAssembler,
    resolve_forging_assembly_worker as resolve_forging_assembly_worker,
    resolve_multi_recording_configuration_resolver as resolve_multi_recording_configuration_resolver,
)
from ..shared_assets import (
    tracked_job as tracked_job,
    pinned_worker_threads as pinned_worker_threads,
    multi_recording_dataset_directory as multi_recording_dataset_directory,
)

MULTIDAY_DISCOVERY_JOB_NAME: str
MULTIDAY_EXTRACTION_JOB_NAME: str
FORGING_JOB_NAME: str
FORGING_JOB_CONCURRENCY_LIMITS: dict[str, int]
_MULTI_RECORDING_CONFIGURATION_FILENAME: str

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
def resolve_multiday_plan(dataset: DatasetData, project_root: Path) -> dict[str, tuple[Path, list[str]]]: ...
def materialize_multiday_plan(
    dataset: DatasetData, project_root: Path, *, display_progress: bool
) -> dict[str, tuple[Path, list[str]]]: ...
def load_multiday_plan(dataset: DatasetData) -> dict[str, tuple[Path, list[str]]]: ...
def build_forging_universe(
    dataset: DatasetData, multiday_plan: dict[str, tuple[Path, list[str]]]
) -> list[tuple[str, str]]: ...
def forging_job_prerequisites(
    dataset: DatasetData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _resolve_runnable_jobs(tracker: ProcessingTracker, universe: list[tuple[str, str]]) -> list[tuple[str, str]]: ...
def _reset_animal_jobs(tracker: ProcessingTracker, dataset: DatasetData, animals: tuple[str, ...]) -> None: ...
def _run_discovery_job(
    configuration_path: Path, animal: str, tracker: ProcessingTracker, job_id: str, workers: int
) -> None: ...
def _run_extraction_job(
    configuration_path: Path, session: str, tracker: ProcessingTracker, job_id: str, workers: int
) -> None: ...
def _execute_remote_forging_job(
    job_id: str,
    universe: list[tuple[str, str]],
    dataset: DatasetData,
    session_lookup: dict[str, DatasetSession],
    session_to_configuration: dict[str, Path],
    multiday_plan: dict[str, tuple[Path, list[str]]],
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
