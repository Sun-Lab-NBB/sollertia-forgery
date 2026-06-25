"""Collects every sollertia-forgery dispatch registry in one place and runs the import-time checks that guard them.

This module is the single canonical surface for wiring an acquisition system's donated processing assets into the
library. It is the only module that imports the per-system subpackages (``mesoscope_vr`` and its future siblings) and
binds their donated assets (microcontroller module parsers, the runtime log parser, and the per-session forging
data-assembly worker) into the dispatch registries, so the system-agnostic worker packages and the interface layer
never import a system subpackage directly; they resolve a system's assets through the ``resolve_*`` helpers below. The
keying enumerations live in the leaf ``pipelines`` module and in sollertia-shared-assets, which keeps this module
importable by every registry consumer without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems

from .pipelines import ProcessingPipelines
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
    "AGNOSTIC_PIPELINES",
    "FORGING_ASSEMBLY_REGISTRY",
    "MICROCONTROLLER_PARSER_REGISTRY",
    "RUNTIME_PARSER_REGISTRY",
    "SYSTEM_PIPELINES",
    "resolve_forging_assembly_worker",
    "resolve_microcontroller_parsers",
    "resolve_runtime_binding",
]


AGNOSTIC_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.MANIFEST,
        ProcessingPipelines.CHECKSUM,
        ProcessingPipelines.FORGING,
    }
)
"""The pipelines that are platform-wide rather than acquisition-system-specific. Their entry points live in the
system-agnostic processing layer (the ``managing`` layer for manifest and checksum, the ``forging`` package for
forging) and are invoked directly by the interface. The forging pipeline still resolves a system-specific data-assembly
worker, but it does so through its own ``FORGING_ASSEMBLY_REGISTRY`` keyed by acquisition system."""

SYSTEM_PIPELINES: dict[AcquisitionSystems, frozenset[ProcessingPipelines]] = {
    AcquisitionSystems.MESOSCOPE_VR: frozenset(),
}
"""Maps each acquisition system to the set of system-specific pipelines it dispatches through the remote compute
server. Every per-asset processing pipeline (microcontroller, runtime, video, two-photon) is now owned by a
system-agnostic worker package that resolves the system's donated assets through the registries below, so no
acquisition system declares a system-specific dispatched pipeline; the set is empty. The structure is retained so the
import-time checks verify the pipeline partition and so a future system can declare a genuinely system-specific
pipeline here."""

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
acquisition-system package implements for one hardware module. The agnostic microcontroller pipeline infers the system
from the processed session and dispatches the matching function for every extracted module, so a module is parseable
for a system exactly when it appears here. Per-session hardware eligibility (whether the module's conversion parameters
were configured for the session) is handled inside each function, which skips silently when its hardware was not
configured rather than being gated by a separate predicate."""

FORGING_ASSEMBLY_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: assemble_mesoscope_session,
}
"""The single, fully-visible registry of per-session forging data-assembly workers, keyed by acquisition system. Each
value is a plain module-level ``assemble(source_session_path, output_path, dataset_name)`` function that an
acquisition-system package implements to assemble one session's ``data.feather`` and its system-specific data-format
descriptor. The agnostic forging pipeline infers the system from the resolved dataset and dispatches the matching
worker for every session, so the pipeline itself stays system-agnostic and never names a system-specific type. This is
the only forging asset a system donates: dataset definition, the optional cindra multi-day stage, all job/tracker
orchestration, and the re-export of the shared assets (the VR configuration and the session descriptor) are owned by
the agnostic ``forging`` package."""

RUNTIME_PARSER_REGISTRY: dict[AcquisitionSystems, tuple[str, Callable[..., None]]] = {
    AcquisitionSystems.MESOSCOPE_VR: (RUNTIME_SOURCE_ID, parse_runtime),
}
"""The single, fully-visible registry of runtime log parsers, keyed by acquisition system. Each value pairs the
system's runtime DataLogger source id (which locates the ``{source_id}_log.npz`` archive) with a plain module-level
``parse(decoded_messages, output_directory, session)`` function that interprets the decoded runtime payloads into the
system's behavior feathers. The agnostic runtime pipeline infers the system from the processed session, decodes the
archive into a raw ``(time_us, payload)`` table, and dispatches the matching parser, so the pipeline itself stays
system-agnostic and never names a system-specific type. Per-session eligibility (such as experiment-only data) is
handled inside the parser, which resolves its own configuration from the session."""


def _resolve_system(system: str | AcquisitionSystems) -> AcquisitionSystems:
    """Validates and normalizes an acquisition-system identifier to an AcquisitionSystems member.

    Args:
        system: An AcquisitionSystems member or its string value (e.g., ``"mesoscope-vr"``).

    Returns:
        The corresponding AcquisitionSystems member.

    Raises:
        ValueError: If the identifier is not a valid AcquisitionSystems member.
    """
    if system not in AcquisitionSystems:
        valid = ", ".join(member.value for member in AcquisitionSystems)
        message = (
            f"Unable to resolve the acquisition system '{system}'. Expected one of the supported AcquisitionSystems "
            f"members: {valid}."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover

    return AcquisitionSystems(system)


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


def _assert_registry_coverage() -> None:
    """Verifies at import time that every acquisition system is fully wired into the dispatch registries.

    Confirms that every ``AcquisitionSystems`` member declares its pipelines in ``SYSTEM_PIPELINES`` and has an entry
    in the forging-assembly registry and the runtime-parser registry, and that ``SYSTEM_PIPELINES`` together with
    ``AGNOSTIC_PIPELINES`` covers exactly the ``ProcessingPipelines`` enum. With a single acquisition system these
    checks are structural scaffolding that enforces full wiring; they begin catching cross-system gaps once a second
    system is added.

    Raises:
        RuntimeError: If any acquisition system is missing from a registry or if the pipeline partition does not cover
            the ``ProcessingPipelines`` enum. The error names the offending members so extenders can immediately locate
            the unwired touch point.
    """
    systems = frozenset(AcquisitionSystems)

    for registry_name, registry in (
        ("SYSTEM_PIPELINES", SYSTEM_PIPELINES),
        ("FORGING_ASSEMBLY_REGISTRY", FORGING_ASSEMBLY_REGISTRY),
        ("RUNTIME_PARSER_REGISTRY", RUNTIME_PARSER_REGISTRY),
    ):
        missing = systems - frozenset(registry)
        if missing:
            missing_names = ", ".join(sorted(member.name for member in missing))
            message = (
                f"{registry_name} is missing entries for {missing_names}. Every acquisition system must register its "
                f"donated processing and forging assets. See the README's 'Adding New Acquisition Systems' section."
            )
            console.error(message=message, error=RuntimeError)

    claimed = frozenset().union(*SYSTEM_PIPELINES.values()) if SYSTEM_PIPELINES else frozenset()
    covered = claimed | AGNOSTIC_PIPELINES
    uncovered = frozenset(ProcessingPipelines) - covered
    if uncovered:
        uncovered_names = ", ".join(sorted(member.name for member in uncovered))
        message = (
            f"The pipeline partition does not cover {uncovered_names}. Every ProcessingPipelines member must be "
            f"either declared in SYSTEM_PIPELINES for at least one acquisition system or listed in AGNOSTIC_PIPELINES."
        )
        console.error(message=message, error=RuntimeError)


_assert_registry_coverage()
