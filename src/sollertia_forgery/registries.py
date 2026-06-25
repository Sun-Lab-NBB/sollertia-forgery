"""Collects every sollertia-forgery dispatch registry in one place and runs the import-time checks that guard them.

This module binds each acquisition system's donated assets (microcontroller module parsers, the runtime log parser,
and the per-session forging data-assembly worker) into the dispatch registries and exposes the ``resolve_*`` helpers
that consumers use to look them up. The registries are keyed by acquisition system (from sollertia-shared-assets) and,
for microcontroller parsers, by hardware ``(module type, module id)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems

from .mesoscope_vr import assemble_mesoscope_session
from .mesoscope_vr.runtime import RUNTIME_SOURCE_ID, parse_runtime
from .mesoscope_vr.microcontrollers import (
    parse_lick,
    parse_brake,
    parse_valve,
    parse_screen,
    parse_torque,
    parse_encoder,
    parse_gas_puff,
    parse_mesoscope_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FORGING_ASSEMBLY_REGISTRY",
    "MICROCONTROLLER_PARSER_REGISTRY",
    "RUNTIME_PARSER_REGISTRY",
    "resolve_forging_assembly_worker",
    "resolve_microcontroller_parsers",
    "resolve_runtime_binding",
]


MICROCONTROLLER_PARSER_REGISTRY: dict[tuple[AcquisitionSystems, int, int], Callable[..., None]] = {
    (AcquisitionSystems.MESOSCOPE_VR, 1, 1): parse_mesoscope_frame,
    (AcquisitionSystems.MESOSCOPE_VR, 2, 1): parse_encoder,
    (AcquisitionSystems.MESOSCOPE_VR, 3, 1): parse_brake,
    (AcquisitionSystems.MESOSCOPE_VR, 4, 1): parse_lick,
    (AcquisitionSystems.MESOSCOPE_VR, 5, 1): parse_valve,
    (AcquisitionSystems.MESOSCOPE_VR, 5, 2): parse_gas_puff,
    (AcquisitionSystems.MESOSCOPE_VR, 6, 1): parse_torque,
    (AcquisitionSystems.MESOSCOPE_VR, 7, 1): parse_screen,
}
"""The single, fully-visible registry of microcontroller module parsers, keyed by ``(acquisition system, module type,
module id)``. Each value is a plain module-level ``parse(event_partition, output_directory, session)`` function that an
acquisition-system package implements for one hardware module; a module is parseable for a system exactly when it
appears here."""

FORGING_ASSEMBLY_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: assemble_mesoscope_session,
}
"""The single, fully-visible registry of per-session forging data-assembly workers, keyed by acquisition system. Each
value is a plain module-level ``assemble(source_session_path, output_path, dataset_name)`` function that an
acquisition-system package implements to assemble one session's ``data.feather`` and its system-specific data-format
descriptor. This worker is the only forging asset a system donates; dataset definition, the cindra multi-day stage,
job/tracker orchestration, and shared-asset re-export are owned by the agnostic ``forging`` package."""

RUNTIME_PARSER_REGISTRY: dict[AcquisitionSystems, tuple[str, Callable[..., None]]] = {
    AcquisitionSystems.MESOSCOPE_VR: (RUNTIME_SOURCE_ID, parse_runtime),
}
"""The single, fully-visible registry of runtime log parsers, keyed by acquisition system. Each value pairs the
system's runtime DataLogger source id (which locates the ``{source_id}_log.npz`` archive) with a plain module-level
``parse(decoded_messages, output_directory, session)`` function that interprets the decoded runtime payloads into the
system's behavior feathers."""


def resolve_forging_assembly_worker(system: str | AcquisitionSystems) -> Callable[..., None]:
    """Resolves the per-session forging data-assembly worker registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, as an AcquisitionSystems member or its
            string value (for example, the value carried by ``DatasetData.acquisition_system``).

    Returns:
        The registered assembly worker callable for the acquisition system. The agnostic forging pipeline invokes it
        once per session to write that session's ``data.feather`` and data-format descriptor.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return FORGING_ASSEMBLY_REGISTRY[_resolve_system(system)]


def resolve_microcontroller_parsers(system: str | AcquisitionSystems) -> dict[tuple[int, int], Callable[..., None]]:
    """Resolves the microcontroller module parsers registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        A mapping from each ``(module_type, module_id)`` pair the system parses to its parser function. The agnostic
        microcontroller pipeline treats this mapping as the set of parseable modules for the session.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    resolved_system = _resolve_system(system)
    return {
        (module_type, module_id): parser
        for (registered_system, module_type, module_id), parser in MICROCONTROLLER_PARSER_REGISTRY.items()
        if registered_system == resolved_system
    }


def resolve_runtime_binding(system: str | AcquisitionSystems) -> tuple[str, Callable[..., None]]:
    """Resolves the runtime source id and parser registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        A ``(source_id, parser)`` tuple. The source id locates the system's runtime DataLogger archive, and the parser
        interprets the decoded runtime messages into the system's behavior feathers. The agnostic runtime pipeline uses
        the source id to find the archive and dispatches the parser once the archive is decoded.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return RUNTIME_PARSER_REGISTRY[_resolve_system(system)]


def _resolve_system(system: str | AcquisitionSystems) -> AcquisitionSystems:
    """Validates and normalizes an acquisition-system identifier to an AcquisitionSystems member.

    Args:
        system: An AcquisitionSystems member or its string value (e.g., ``"mesoscope"``).

    Returns:
        The corresponding AcquisitionSystems member.

    Raises:
        ValueError: If the identifier is not a valid AcquisitionSystems member.
    """
    if system not in AcquisitionSystems:
        valid = ", ".join(member.value for member in AcquisitionSystems)
        message = (
            f"Unable to resolve the acquisition system. The system must be one of the supported AcquisitionSystems "
            f"members ({valid}), but got '{system}'."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover

    return AcquisitionSystems(system)


def _assert_registry_coverage() -> None:
    """Verifies at import time that every acquisition system has registered every donated asset.

    Confirms that every ``AcquisitionSystems`` member has an entry in the forging-assembly registry and the
    runtime-parser registry, and registers at least one microcontroller module parser.

    Raises:
        RuntimeError: If any acquisition system is missing from a donor registry. The error names the offending
            members so extenders can immediately locate the unwired touch point.
    """
    systems = frozenset(AcquisitionSystems)
    microcontroller_systems = frozenset(system for system, _, _ in MICROCONTROLLER_PARSER_REGISTRY)

    for registry_name, registered_systems in (
        ("FORGING_ASSEMBLY_REGISTRY", frozenset(FORGING_ASSEMBLY_REGISTRY)),
        ("RUNTIME_PARSER_REGISTRY", frozenset(RUNTIME_PARSER_REGISTRY)),
        ("MICROCONTROLLER_PARSER_REGISTRY", microcontroller_systems),
    ):
        missing = systems - registered_systems
        if missing:
            missing_names = ", ".join(sorted(member.name for member in missing))
            message = (
                f"Unable to validate donor-registry coverage for {registry_name}. Every acquisition system must "
                f"register its donated processing and forging assets in this module ('registries.py'), but entries "
                f"are missing for {missing_names}."
            )
            console.error(message=message, error=RuntimeError)


_assert_registry_coverage()
