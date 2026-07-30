from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
from sollertia_shared_assets import SessionData as SessionData

class _CalciumIndicator(StrEnum):
    GCAMP6F = "GCaMP6f"
    JGCAMP8S = "jGCaMP8s"

@dataclass(frozen=True, slots=True)
class _IndicatorParameters:
    tau: float
    neuropil_coefficient: float
    probability_threshold: float

_INDICATOR_PARAMETERS: dict[_CalciumIndicator, _IndicatorParameters]
_GENOTYPE_INDICATOR_REGISTRY: dict[str, _CalciumIndicator]

def locate_two_photon_data(session: SessionData) -> Path: ...
def resolve_single_recording_configuration(session: SessionData) -> SingleRecordingConfiguration: ...
def resolve_multi_recording_configuration(session: SessionData) -> MultiRecordingConfiguration | None: ...
def _resolve_calcium_indicator(genotype: str) -> _CalciumIndicator: ...
def _read_session_genotype(session: SessionData) -> str: ...
def _build_single_recording_configuration(genotype: str) -> SingleRecordingConfiguration: ...
def _build_multi_recording_configuration(genotype: str) -> MultiRecordingConfiguration: ...
def _assert_indicator_coverage() -> None: ...
