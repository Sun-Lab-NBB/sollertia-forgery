from typing import Any, NoReturn
from pathlib import Path
from dataclasses import field, dataclass

import polars as pl
from sollertia_shared_assets import (
    DatasetData as DatasetData,
    SessionData as SessionData,
)
from ataraxis_data_structures import YamlConfig, ProcessingStatus

from ..forging import discover_project_datasets as discover_project_datasets
from .dispatch import (
    DATASET_UNIT as DATASET_UNIT,
    SESSION_UNIT as SESSION_UNIT,
    PipelineDispatch as PipelineDispatch,
    resolve_dispatch as resolve_dispatch,
    resolve_job_cores as resolve_job_cores,
    resolve_unit_dispatches as resolve_unit_dispatches,
)
from .footprints import (
    JobFootprint as JobFootprint,
    resolve_model_version as resolve_model_version,
)
from ..shared_assets import (
    SESSION_PIPELINES as SESSION_PIPELINES,
    natural_sort as natural_sort,
)

_PLAN_FILENAME: str
_WHOLE_PIPELINE_LABEL: str
_RETAINED_STATUSES: frozenset[ProcessingStatus]
_LOCK_TIMEOUT_SECONDS: float
PROJECT_PLAN_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType]

@dataclass(slots=True)
class _JobPlanEntry:
    pipeline: str = ...
    job_name: str = ...
    specifier: str = ...
    cores: int = ...
    memory_mb: int = ...
    resident_mb: int = ...
    memory_modeled: bool = ...
    prerequisite_ids: list[str] = field(default_factory=list)
    @property
    def key(self) -> tuple[str, str, str]: ...
    @property
    def job_id(self) -> str: ...

@dataclass
class _JobPlan(YamlConfig):
    unit_name: str = ...
    unit_kind: str = ...
    model_version: str = ...
    entries: list[_JobPlanEntry] = field(default_factory=list)
    unsized_jobs: dict[str, str] = field(default_factory=dict)
    def entry_map(self) -> dict[tuple[str, str, str], _JobPlanEntry]: ...

def project_plan_path(project_directory: Path) -> Path: ...
def resolve_session_plan(
    session_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> _JobPlan: ...
def resolve_dataset_plan(
    dataset_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> _JobPlan: ...
def generate_project_plan(project_directory: Path, *, display_progress: bool = False) -> Path: ...
def _session_plan_path(session: SessionData) -> Path: ...
def _dataset_plan_path(dataset: DatasetData) -> Path: ...
def _resolve_unit_plan(
    dispatches: list[PipelineDispatch[Any]],
    unit_path: Path,
    unit_kind: str,
    *,
    regenerate_plan: bool,
    display_progress: bool,
) -> _JobPlan: ...
def _discover_unit(
    dispatch: PipelineDispatch[Any], unit_path: Path, skipped: dict[str, str]
) -> tuple[Any, list[tuple[str, str]], list[tuple[str, str]]] | None: ...
def _size_unit(
    dispatch: PipelineDispatch[Any],
    unit: Any,
    jobs: list[tuple[str, str]],
    declared: dict[str, int],
    unsized: dict[str, str],
) -> dict[tuple[str, str], JobFootprint]: ...
def _size_jobs_separately(
    dispatch: PipelineDispatch[Any],
    unit: Any,
    jobs: list[tuple[str, str]],
    declared: dict[str, int],
    unsized: dict[str, str],
) -> dict[tuple[str, str], JobFootprint]: ...
def _refusal_key(pipeline: str, job_name: str, specifier: str) -> str: ...
def _align_tracker(
    tracker_path: Path, jobs: list[tuple[str, str]], universe: list[tuple[str, str]], refused: list[tuple[str, str]]
) -> None: ...
def _reject_unit(unit_path: Path, unit_kind: str, skipped: dict[str, str], unsized: dict[str, str]) -> NoReturn: ...
def _load_plan(plan_path: Path) -> _JobPlan | None: ...
def _save_plan(plan: _JobPlan, plan_path: Path) -> None: ...
def _projection_row(
    entry: _JobPlanEntry, animal: str | None, session: str | None, dataset: str | None
) -> dict[str, Any]: ...
