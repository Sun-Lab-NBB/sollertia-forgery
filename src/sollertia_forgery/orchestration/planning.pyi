from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

import polars as pl
from sollertia_shared_assets import (
    DatasetData as DatasetData,
    SessionData as SessionData,
)
from ataraxis_data_structures import YamlConfig

from ..forging import discover_project_datasets as discover_project_datasets
from .dispatch import (
    PipelineDispatch as PipelineDispatch,
    resolve_dispatch as resolve_dispatch,
    resolve_job_cores as resolve_job_cores,
)
from ..shared_assets import (
    SESSION_PIPELINES as SESSION_PIPELINES,
    ProcessingPipelines as ProcessingPipelines,
)

_PLAN_FILENAME: str
_LOCK_TIMEOUT_SECONDS: float
SESSION_UNIT: str
DATASET_UNIT: str
PROJECT_PLAN_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType]

@dataclass
class JobPlanEntry:
    pipeline: str = ...
    job_name: str = ...
    specifier: str = ...
    cores: int = ...
    memory_mb: int = ...
    memory_modeled: bool = ...
    prerequisite_ids: list[str] = field(default_factory=list)
    @property
    def key(self) -> tuple[str, str, str]: ...
    @property
    def job_id(self) -> str: ...

@dataclass
class JobPlan(YamlConfig):
    unit_name: str = ...
    unit_kind: str = ...
    entries: list[JobPlanEntry] = field(default_factory=list)
    def entry_map(self) -> dict[tuple[str, str, str], JobPlanEntry]: ...

def session_plan_path(session: SessionData) -> Path: ...
def dataset_plan_path(dataset: DatasetData) -> Path: ...
def project_plan_path(project_directory: Path) -> Path: ...
def resolve_session_plan(
    session_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> JobPlan: ...
def resolve_dataset_plan(
    dataset_path: Path, *, regenerate_plan: bool = False, display_progress: bool = False
) -> JobPlan: ...
def generate_project_plan(project_directory: Path, *, display_progress: bool = False) -> Path: ...
def _resolve_unit_plan(
    dispatches: list[PipelineDispatch[Any]],
    unit_path: Path,
    unit_kind: str,
    *,
    regenerate_plan: bool,
    display_progress: bool,
) -> JobPlan: ...
def _discover_unit(
    dispatch: PipelineDispatch[Any], unit_path: Path, skipped: dict[str, str]
) -> tuple[Any, list[tuple[str, str]], list[tuple[str, str]]] | None: ...
def _load_plan(plan_path: Path) -> JobPlan | None: ...
def _save_plan(plan: JobPlan, plan_path: Path) -> None: ...
def _projection_row(
    entry: JobPlanEntry, animal: str | None, session: str | None, dataset: str | None
) -> dict[str, Any]: ...
