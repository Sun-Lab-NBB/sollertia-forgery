from pathlib import Path
from dataclasses import dataclass

import polars as pl
from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import (
    SessionData as SessionData,
    MesoscopeHardwareState,
)

from .metadata import BehaviorDataFiles as BehaviorDataFiles
from ..shared_assets import merge_event_streams as merge_event_streams

_PRIMARY_EVENT_CODE: int
_SECONDARY_EVENT_CODE: int
_TONE_ON_EVENT_CODE: int
_TONE_OFF_EVENT_CODE: int
_ENCODER_MODULE: tuple[int, int]
_MESOSCOPE_FRAME_MODULE: tuple[int, int]
_BRAKE_MODULE: tuple[int, int]
_VALVE_MODULE: tuple[int, int]
_GAS_PUFF_MODULE: tuple[int, int]
_LICK_MODULE: tuple[int, int]
_TORQUE_MODULE: tuple[int, int]
_SCREEN_MODULE: tuple[int, int]

@dataclass(frozen=True, slots=True)
class _ModuleSpecification:
    required_fields: tuple[str, ...]
    usage_flags: tuple[str, ...]
    event_codes: tuple[int, ...]
    def check_eligibility(self, hardware_state: MesoscopeHardwareState) -> bool: ...

_MODULE_REGISTRY: dict[tuple[int, int], _ModuleSpecification]

def parse_encoder(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_mesoscope_frame(
    event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData
) -> None: ...
def parse_brake(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_valve(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_gas_puff(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_lick(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_torque(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def parse_screen(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None: ...
def get_eligible_modules(session: SessionData) -> set[tuple[int, int]]: ...
def get_module_event_codes() -> dict[tuple[int, int], tuple[int, ...]]: ...
def _is_module_eligible(module_key: tuple[int, int], hardware_state: MesoscopeHardwareState) -> bool: ...
def _resolve_hardware_state(session: SessionData) -> MesoscopeHardwareState: ...
def _parse_encoder_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_ttl_data(event_partition: dict[int, pl.DataFrame], output_file: Path, session: SessionData) -> None: ...
def _parse_brake_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_valve_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_gas_puff_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_lick_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_torque_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
def _parse_screen_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None: ...
