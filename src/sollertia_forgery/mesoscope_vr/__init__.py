"""Provides the Mesoscope-VR system-specific assets: the behavior/dataset metadata schema, the end-to-end
processing pipelines (behavior and cell-activity extraction), and the dataset forging pipeline.

Notes:
    This package depends on the ``shared_assets``, ``server``, and ``managing`` layers for the shared data
    hierarchy, management, and orchestration assets. Heavy acquisition-library bindings (ataraxis-video-system,
    ataraxis-communication-interface, cindra) are
    imported lazily inside the pipeline functions so importing this package stays cheap.
"""

from .batch import (
    FORGING_CONCURRENCY,
    BEHAVIOR_CONCURRENCY,
    run_forging_job,
    run_behavior_job,
    clean_forging_unit,
    clean_behavior_unit,
    verify_forging_unit,
    prepare_forging_unit,
    verify_behavior_unit,
    prepare_behavior_unit,
    iterate_forging_overview,
    iterate_behavior_overview,
)
from .forging import (
    FORGING_JOB_NAME,
    run_forging_pipeline,
)
from .activity import run_multidataset_processing_pipeline
from .metadata import (
    StimulusMode,
    DatasetColumn,
    TrialGeometry,
    BehaviorDataFiles,
    TrialGeometryEntry,
)
from .processing import (
    discover_behavior_jobs,
    run_behavior_processing_pipeline,
)
from .fluorescence import FluorescenceColumn
from .server_forging import forge_dataset
from .server_processing import process_project_data

__all__ = [
    "BEHAVIOR_CONCURRENCY",
    "FORGING_CONCURRENCY",
    "FORGING_JOB_NAME",
    "BehaviorDataFiles",
    "DatasetColumn",
    "FluorescenceColumn",
    "StimulusMode",
    "TrialGeometry",
    "TrialGeometryEntry",
    "clean_behavior_unit",
    "clean_forging_unit",
    "discover_behavior_jobs",
    "forge_dataset",
    "iterate_behavior_overview",
    "iterate_forging_overview",
    "prepare_behavior_unit",
    "prepare_forging_unit",
    "process_project_data",
    "run_behavior_job",
    "run_behavior_processing_pipeline",
    "run_forging_job",
    "run_forging_pipeline",
    "run_multidataset_processing_pipeline",
    "verify_behavior_unit",
    "verify_forging_unit",
]
