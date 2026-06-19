"""Provides the Mesoscope-VR system-specific assets: the behavior/dataset metadata schema, the end-to-end
processing pipelines (behavior and cell-activity extraction), and the dataset forging pipeline.

Notes:
    This package depends on ``cross_system`` for the shared data hierarchy, management, and orchestration assets.
    Heavy acquisition-library bindings (ataraxis-video-system, ataraxis-communication-interface, cindra) are
    imported lazily inside the pipeline functions so importing this package stays cheap.
"""

from .forging import (
    FORGING_JOB_NAME,
    resolve_dataset,
    run_forging_pipeline,
)
from .activity import (
    run_activity_processing_pipeline,
    run_multidataset_processing_pipeline,
)
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
    "FORGING_JOB_NAME",
    "BehaviorDataFiles",
    "DatasetColumn",
    "FluorescenceColumn",
    "StimulusMode",
    "TrialGeometry",
    "TrialGeometryEntry",
    "discover_behavior_jobs",
    "forge_dataset",
    "process_project_data",
    "resolve_dataset",
    "run_activity_processing_pipeline",
    "run_behavior_processing_pipeline",
    "run_forging_pipeline",
    "run_multidataset_processing_pipeline",
]
