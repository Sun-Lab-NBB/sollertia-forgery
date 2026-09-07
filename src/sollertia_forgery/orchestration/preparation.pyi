from typing import Any
from pathlib import Path
from collections.abc import Sequence

from .graph import (
    BatchDocument as BatchDocument,
    build_batch_document as build_batch_document,
)
from .hosts import (
    ExecutionHost as ExecutionHost,
    plan_artifact_path as plan_artifact_path,
    state_artifact_paths as state_artifact_paths,
)
from .dispatch import (
    DATASET_UNIT as DATASET_UNIT,
    SESSION_UNIT as SESSION_UNIT,
    resolve_dispatch as resolve_dispatch,
)

_UNIT_DEPTHS: dict[str, int]

def prepare_batch(
    host: ExecutionHost,
    pipeline: str,
    unit_paths: Sequence[str],
    options: dict[str, Any] | None = None,
    *,
    replan: bool = False,
) -> BatchDocument: ...
def resolve_project_root(unit_paths: Sequence[Path], unit_kind: str) -> Path: ...
