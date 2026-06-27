"""Provides the system-agnostic dataset forging pipeline: dataset resolution and per-session assembly dispatch.

The dataset hierarchy classes (``DatasetData``, ``DatasetFiles``, ``DatasetAnimal``, ``DatasetSession``) are owned by
sollertia-shared-assets and re-exported here for convenience; this package owns the forging resolution policy and the
assembly pipeline.
"""

from sollertia_shared_assets import (
    DatasetData,
    DatasetFiles,
    DatasetAnimal,
    DatasetSession,
)

from .dataset import resolve_dataset
from .pipeline import FORGING_JOB_NAME, run_forging_pipeline

__all__ = [
    "FORGING_JOB_NAME",
    "DatasetAnimal",
    "DatasetData",
    "DatasetFiles",
    "DatasetSession",
    "resolve_dataset",
    "run_forging_pipeline",
]
