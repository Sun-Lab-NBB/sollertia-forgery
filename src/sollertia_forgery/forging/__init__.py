"""Provides the system-agnostic dataset forging pipeline: dataset definition and per-session assembly dispatch."""

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
