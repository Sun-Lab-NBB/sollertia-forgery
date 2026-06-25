"""Provides the Mesoscope-VR system-specific assets: the behavior/dataset metadata schema, the system-specific
processing assets (behavior processing and cell-activity extraction), and the per-session forging data-assembly worker
donated to the agnostic forging pipeline.

Notes:
    This package depends on the ``shared_assets``, ``server``, and ``managing`` layers for the shared data hierarchy,
    management, and orchestration assets. Heavy acquisition-library bindings (ataraxis-video-system,
    ataraxis-communication-interface, cindra) are imported lazily inside the pipeline functions so importing this
    package stays cheap. The package donates system-specific assets into the central registries but never imports the
    agnostic processors that consume them; the dependency is strictly one-way.
"""

from .batch import (
    BEHAVIOR_CONCURRENCY,
    run_behavior_job,
    clean_behavior_unit,
    verify_behavior_unit,
    prepare_behavior_unit,
    iterate_behavior_overview,
)
from .forging import assemble_mesoscope_session
from .activity import run_multidataset_processing_pipeline
from .metadata import (
    DatasetColumn,
    BehaviorDataFiles,
    SessionDataFormat,
)
from .processing import (
    discover_behavior_jobs,
    run_behavior_processing_pipeline,
)
from .fluorescence import FluorescenceColumn
from .server_processing import process_project_data

__all__ = [
    "BEHAVIOR_CONCURRENCY",
    "BehaviorDataFiles",
    "DatasetColumn",
    "FluorescenceColumn",
    "SessionDataFormat",
    "assemble_mesoscope_session",
    "clean_behavior_unit",
    "discover_behavior_jobs",
    "iterate_behavior_overview",
    "prepare_behavior_unit",
    "process_project_data",
    "run_behavior_job",
    "run_behavior_processing_pipeline",
    "run_multidataset_processing_pipeline",
    "verify_behavior_unit",
]
