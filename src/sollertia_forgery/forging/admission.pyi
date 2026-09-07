from sollertia_shared_assets import SessionData as SessionData

from ..registries import resolve_forging_admission_pipelines as resolve_forging_admission_pipelines
from ..shared_assets import (
    ProcessingPipelines as ProcessingPipelines,
    resolve_session_tracker_path as resolve_session_tracker_path,
)

_PARTIAL_STATE_TEMPLATE: str

def verify_session_admissibility(session: SessionData) -> None: ...
def _resolve_pipeline_state(session: SessionData, pipeline: ProcessingPipelines) -> str: ...
