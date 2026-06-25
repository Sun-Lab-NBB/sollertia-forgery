"""Provides the system-agnostic dataset forging pipeline: the forged dataset data hierarchy, the ``resolve_dataset``
entry point that resolves and creates that hierarchy from processed sessions, running the optional cindra multi-day
stage, dispatching the registered per-session assembly worker, and re-exporting the shared assets into the unified
per-session dataset.

Notes:
    The forging batch/MCP adapters (prepare/verify/clean/overview, the per-session worker, and the concurrency
    descriptor) live in ``orchestration.forging_batch`` alongside the generic batch engine they plug into, and are
    imported on demand by the batch-registry wiring rather than re-exported here.
"""

from .dataset import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
    resolve_dataset,
)
from .pipeline import FORGING_JOB_NAME, run_forging_pipeline

__all__ = [
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "FORGING_JOB_NAME",
    "resolve_dataset",
    "run_forging_pipeline",
]
