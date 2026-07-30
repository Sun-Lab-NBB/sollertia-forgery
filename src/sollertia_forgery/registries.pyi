from typing import Protocol
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
import polars as pl
from sollertia_shared_assets import SessionData, SessionTypes, AcquisitionSystems

from .shared_assets import ProcessingPipelines

__all__ = [
    "ForgingAssembler",
    "MicrocontrollerParser",
    "RuntimeParser",
    "TwoPhotonDataLocator",
    "VideoTracker",
    "resolve_eligible_microcontroller_modules",
    "resolve_forging_admission_pipelines",
    "resolve_forging_assembly_worker",
    "resolve_forging_column_descriptions",
    "resolve_microcontroller_event_codes",
    "resolve_microcontroller_parsers",
    "resolve_multi_recording_configuration_resolver",
    "resolve_runtime_binding",
    "resolve_single_recording_configuration_resolver",
    "resolve_two_photon_data_locator",
    "resolve_video_tracking",
]

class ForgingAssembler(Protocol):
    def __call__(self, source_session_path: Path, output_path: Path, dataset_name: str) -> None: ...

class MicrocontrollerParser(Protocol):
    def __call__(
        self, event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData
    ) -> None: ...

class RuntimeParser(Protocol):
    def __call__(self, decoded_messages: pl.DataFrame, output_directory: Path, session: SessionData) -> None: ...

class TwoPhotonDataLocator(Protocol):
    def __call__(self, session: SessionData) -> Path: ...

class VideoTracker(Protocol):
    def __call__(self, session: SessionData, output_directory: Path) -> None: ...

@dataclass(frozen=True, slots=True)
class _ForgingAssemblyAsset:
    assembler: ForgingAssembler
    column_descriptions: dict[str, str]

@dataclass(frozen=True, slots=True)
class _CindraConfigurationAsset:
    resolve_single_recording: Callable[[SessionData], SingleRecordingConfiguration]
    resolve_multi_recording: Callable[[SessionData], MultiRecordingConfiguration | None]

def resolve_forging_assembly_worker(system: str | AcquisitionSystems) -> ForgingAssembler: ...
def resolve_forging_admission_pipelines(
    system: str | AcquisitionSystems,
) -> dict[SessionTypes, frozenset[ProcessingPipelines]]: ...
def resolve_forging_column_descriptions(system: str | AcquisitionSystems) -> dict[str, str]: ...
def resolve_single_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], SingleRecordingConfiguration]: ...
def resolve_multi_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], MultiRecordingConfiguration | None]: ...
def resolve_microcontroller_event_codes(system: str | AcquisitionSystems) -> dict[tuple[int, int], tuple[int, ...]]: ...
def resolve_eligible_microcontroller_modules(
    system: str | AcquisitionSystems, session: SessionData
) -> set[tuple[int, int]]: ...
def resolve_microcontroller_parsers(
    system: str | AcquisitionSystems,
) -> dict[tuple[int, int], MicrocontrollerParser]: ...
def resolve_runtime_binding(system: str | AcquisitionSystems) -> tuple[str, RuntimeParser]: ...
def resolve_two_photon_data_locator(system: str | AcquisitionSystems) -> TwoPhotonDataLocator: ...
def resolve_video_tracking(system: str | AcquisitionSystems) -> VideoTracker: ...
