"""Collects every sollertia-forgery dispatch registry in one place and runs the import-time checks that guard them.

Each acquisition system donates a set of assets to this module. The donated assets are the microcontroller module
parsers and the event codes they read, the runtime log parser, the per-session forging data-assembly worker, the raw
two-photon imaging directory locator, and the cindra configuration resolvers. This module binds those assets into the
dispatch registries and exposes the ``resolve_*`` helpers that consumers use to look them up. The registries are keyed
by acquisition system (from sollertia-shared-assets) and, for microcontroller parsers, by hardware ``(module type,
module id)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems

from .mesoscope_vr import (
    RUNTIME_SOURCE_ID,
    MESOSCOPE_COLUMN_DESCRIPTIONS,
    parse_lick,
    parse_brake,
    parse_valve,
    parse_screen,
    parse_torque,
    parse_encoder,
    parse_runtime,
    parse_gas_puff,
    parse_mesoscope_frame,
    get_module_event_codes,
    locate_two_photon_data,
    assemble_mesoscope_session,
    process_mesoscope_video_tracking,
    resolve_multi_recording_configuration,
    resolve_single_recording_configuration,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
    from sollertia_shared_assets import SessionData

__all__ = [
    "CINDRA_CONFIGURATION_REGISTRY",
    "FORGING_ASSEMBLY_REGISTRY",
    "MICROCONTROLLER_EVENT_CODE_REGISTRY",
    "MICROCONTROLLER_PARSER_REGISTRY",
    "RUNTIME_PARSER_REGISTRY",
    "TWO_PHOTON_DATA_REGISTRY",
    "VIDEO_TRACKING_REGISTRY",
    "CindraConfigurationAsset",
    "ForgingAssemblyAsset",
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


@dataclass(frozen=True, slots=True)
class ForgingAssemblyAsset:
    """Bundles an acquisition system's donated forging assets: the per-session assembly worker and the dataset-level
    column descriptions.

    Notes:
        A system donates both as a single unit because the descriptions document exactly the columns the assembler
        emits into ``data.feather``. The agnostic forging pipeline invokes ``assembler`` once per session and writes
        ``column_descriptions`` once per dataset into the dataset's ``data_descriptions.feather``.
    """

    assembler: Callable[[Path, Path, str], None]
    """The picklable, module-level ``assemble(source_session_path, output_path, dataset_name)`` worker that assembles
    one session's ``data.feather``."""
    column_descriptions: dict[str, str]
    """The mapping from each column name the assembler can emit into ``data.feather`` to its human-readable
    description, baked into the forged dataset's per-dataset ``data_descriptions.feather``."""


@dataclass(frozen=True, slots=True)
class CindraConfigurationAsset:
    """Bundles an acquisition system's donated cindra configuration resolvers: the single-recording and
    multi-recording resolvers.

    Notes:
        Each resolver receives a loaded session and returns the runnable cindra configuration for it, deciding for
        itself how the configuration is derived. The agnostic two-photon pipeline calls ``resolve_single_recording``
        for the session it processes, and the agnostic forging pipeline calls ``resolve_multi_recording`` for each
        animal it tracks across recordings.
    """

    resolve_single_recording: Callable[[SessionData], SingleRecordingConfiguration]
    """The ``resolve(session)`` resolver that returns the system's single-recording cindra configuration for the
    session, or raises when it cannot resolve one."""
    resolve_multi_recording: Callable[[SessionData], MultiRecordingConfiguration | None]
    """The ``resolve(session)`` resolver that returns the system's multi-recording cindra configuration for the
    session, None when the system performs no cross-recording tracking for it, or raises when it cannot resolve one."""


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
acquisition-system package implements for one hardware module. A module is parseable for a system exactly when it
appears here."""

MICROCONTROLLER_EVENT_CODE_REGISTRY: dict[AcquisitionSystems, Callable[[], dict[tuple[int, int], tuple[int, ...]]]] = {
    AcquisitionSystems.MESOSCOPE_VR: get_module_event_codes,
}
"""The single, fully-visible registry of microcontroller event-code accessors, keyed by acquisition system. Each value
is a plain module-level ``get_module_event_codes()`` function returning the ``(module type, module id) -> event codes``
mapping for every module the system parses. The agnostic microcontroller pipeline derives each controller's extraction
filter from this mapping, so a system's event codes live next to the parsers that read them rather than in an
acquisition-time configuration file."""

FORGING_ASSEMBLY_REGISTRY: dict[AcquisitionSystems, ForgingAssemblyAsset] = {
    AcquisitionSystems.MESOSCOPE_VR: ForgingAssemblyAsset(
        assembler=assemble_mesoscope_session,
        column_descriptions=MESOSCOPE_COLUMN_DESCRIPTIONS,
    ),
}
"""The single, fully-visible registry of per-session forging assets, keyed by acquisition system. Each value is a
``ForgingAssemblyAsset`` bundling the system's per-session ``assemble(source_session_path, output_path,
dataset_name)`` worker with its column-description mapping. These are the only forging assets a system donates.
Dataset definition, the cindra multi-day stage, job/tracker orchestration, the per-dataset column-description
binding, and shared-asset re-export are owned by the agnostic ``forging`` package."""

CINDRA_CONFIGURATION_REGISTRY: dict[AcquisitionSystems, CindraConfigurationAsset] = {
    AcquisitionSystems.MESOSCOPE_VR: CindraConfigurationAsset(
        resolve_single_recording=resolve_single_recording_configuration,
        resolve_multi_recording=resolve_multi_recording_configuration,
    ),
}
"""The single, fully-visible registry of cindra configuration resolvers, keyed by acquisition system. Each value is a
``CindraConfigurationAsset`` bundling the system's single- and multi-recording ``resolve(session)`` resolvers. The
agnostic two-photon and forging pipelines obtain a runnable cindra configuration through these resolvers. Each system
therefore decides for itself how its configuration is derived, keeping its system-specific logic next to its parsers."""

RUNTIME_PARSER_REGISTRY: dict[AcquisitionSystems, tuple[str, Callable[..., None]]] = {
    AcquisitionSystems.MESOSCOPE_VR: (RUNTIME_SOURCE_ID, parse_runtime),
}
"""The single, fully-visible registry of runtime log parsers, keyed by acquisition system. Each value pairs the
system's runtime DataLogger source id (which locates the ``{source_id}_log.npz`` archive) with a plain module-level
``parse(decoded_messages, output_directory, session)`` function that interprets the decoded runtime payloads into the
system's behavior feathers."""

TWO_PHOTON_DATA_REGISTRY: dict[AcquisitionSystems, Callable[..., Path]] = {
    AcquisitionSystems.MESOSCOPE_VR: locate_two_photon_data,
}
"""The single, fully-visible registry of raw two-photon imaging directory locators, keyed by acquisition system. Each
value is a plain module-level ``locate(session)`` function that an acquisition-system package implements to resolve
the loaded session's raw two-photon (calcium-imaging) directory, which the agnostic two-photon worker hands to the
cindra single-recording pipeline as its input. A system donates a locator exactly when it produces two-photon data."""

VIDEO_TRACKING_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: process_mesoscope_video_tracking,
}
"""The single, fully-visible registry of video-tracking functions, keyed by acquisition system. Each value is a plain
module-level ``process(session, output_directory)`` function that an acquisition-system package implements to do all
of that system's video tracking: it locates its own externally-produced DeepLabCut ``.h5`` predictions, hardcodes the
bodyparts it wants, parses them, and writes its outputs into the session's processed video-data directory. The
agnostic video pipeline simply runs it (no-op when no predictions are present), like the per-session forging
assembler. A system donates a no-op function when it performs no video tracking."""


def resolve_forging_assembly_worker(system: str | AcquisitionSystems) -> Callable[..., None]:
    """Resolves the per-session forging data-assembly worker registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, as an AcquisitionSystems member or its
            string value (for example, the value carried by ``DatasetData.acquisition_system``).

    Returns:
        The registered assembly worker callable for the acquisition system. The agnostic forging pipeline invokes it
        once per session to write that session's ``data.feather``.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return FORGING_ASSEMBLY_REGISTRY[_resolve_system(system)].assembler


def resolve_forging_column_descriptions(system: str | AcquisitionSystems) -> dict[str, str]:
    """Resolves the dataset column-description mapping registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, as an AcquisitionSystems member or its
            string value (for example, the value carried by ``DatasetData.acquisition_system``).

    Returns:
        The mapping from each column name the system's assembly worker can emit into ``data.feather`` to its
        human-readable description. The agnostic forging pipeline bakes it into the dataset's
        ``data_descriptions.feather`` at dataset-definition time.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return FORGING_ASSEMBLY_REGISTRY[_resolve_system(system)].column_descriptions


def resolve_single_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], SingleRecordingConfiguration]:
    """Resolves the single-recording cindra configuration resolver registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        The registered ``resolve(session)`` resolver for the acquisition system. The agnostic two-photon pipeline
        calls it with the session it processes to obtain a runnable single-recording configuration.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return CINDRA_CONFIGURATION_REGISTRY[_resolve_system(system)].resolve_single_recording


def resolve_multi_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], MultiRecordingConfiguration | None]:
    """Resolves the multi-recording cindra configuration resolver registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, as an AcquisitionSystems member or its
            string value (for example, the value carried by ``DatasetData.acquisition_system``).

    Returns:
        The registered ``resolve(session)`` resolver for the acquisition system. The agnostic forging pipeline calls
        it for each animal to obtain a runnable multi-recording configuration, or None when the system performs no
        cross-recording tracking for that session.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return CINDRA_CONFIGURATION_REGISTRY[_resolve_system(system)].resolve_multi_recording


def resolve_microcontroller_event_codes(system: str | AcquisitionSystems) -> dict[tuple[int, int], tuple[int, ...]]:
    """Resolves the microcontroller module event codes registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        A mapping from each ``(module_type, module_id)`` pair the system parses to the tuple of event codes its parser
        reads. The agnostic microcontroller pipeline builds every controller's extraction filter from this mapping.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return MICROCONTROLLER_EVENT_CODE_REGISTRY[_resolve_system(system)]()


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


def resolve_two_photon_data_locator(system: str | AcquisitionSystems) -> Callable[..., Path]:
    """Resolves the raw two-photon imaging directory locator registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        The registered locator callable for the acquisition system. The agnostic two-photon pipeline invokes it with
        the loaded session to obtain that session's raw two-photon imaging directory (cindra's input).

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return TWO_PHOTON_DATA_REGISTRY[_resolve_system(system)]


def resolve_video_tracking(system: str | AcquisitionSystems) -> Callable[..., None]:
    """Resolves the video-tracking function registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, as an AcquisitionSystems member or
            its string value (for example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        The registered ``process(session, output_directory)`` function for the acquisition system. The agnostic video
        pipeline runs it once per session, expecting it to perform all of that system's video tracking and to no-op
        when no DeepLabCut predictions are present.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return VIDEO_TRACKING_REGISTRY[_resolve_system(system)]


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

    Confirms that every ``AcquisitionSystems`` member has an entry in the forging-assembly registry, the
    runtime-parser registry, the two-photon-data registry, the microcontroller event-code registry, and the cindra
    configuration registry, and registers at least one microcontroller module parser. Additionally confirms that every
    parseable microcontroller module declares the event codes its parser reads.

    Raises:
        RuntimeError: If any acquisition system is missing from a donor registry, or if a parseable microcontroller
            module does not declare its event codes. The error names the offending members so extenders can
            immediately locate the unwired touch point.
    """
    systems = frozenset(AcquisitionSystems)
    microcontroller_systems = frozenset(system for system, _, _ in MICROCONTROLLER_PARSER_REGISTRY)

    for registry_name, registered_systems in (
        ("FORGING_ASSEMBLY_REGISTRY", frozenset(FORGING_ASSEMBLY_REGISTRY)),
        ("RUNTIME_PARSER_REGISTRY", frozenset(RUNTIME_PARSER_REGISTRY)),
        ("TWO_PHOTON_DATA_REGISTRY", frozenset(TWO_PHOTON_DATA_REGISTRY)),
        ("VIDEO_TRACKING_REGISTRY", frozenset(VIDEO_TRACKING_REGISTRY)),
        ("MICROCONTROLLER_EVENT_CODE_REGISTRY", frozenset(MICROCONTROLLER_EVENT_CODE_REGISTRY)),
        ("CINDRA_CONFIGURATION_REGISTRY", frozenset(CINDRA_CONFIGURATION_REGISTRY)),
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

    # Every parseable module must also declare its event codes, because the extraction stage filters each module by
    # the codes resolved from MICROCONTROLLER_EVENT_CODE_REGISTRY. A parseable module missing from that registry
    # would be dropped from its controller's extraction configuration, and its parse job would never be discovered.
    for registered_system in sorted(microcontroller_systems, key=lambda member: member.name):
        parseable = {
            (module_type, module_id)
            for (system, module_type, module_id) in MICROCONTROLLER_PARSER_REGISTRY
            if system == registered_system
        }
        uncoded = sorted(parseable - set(MICROCONTROLLER_EVENT_CODE_REGISTRY[registered_system]()))
        if uncoded:
            module_names = ", ".join(f"({module_type}, {module_id})" for module_type, module_id in uncoded)
            message = (
                f"Unable to validate donor-registry coverage for MICROCONTROLLER_EVENT_CODE_REGISTRY. Every module "
                f"registered in MICROCONTROLLER_PARSER_REGISTRY must also declare the event codes its parser reads, "
                f"but {registered_system.name} does not declare codes for the following modules: {module_names}."
            )
            console.error(message=message, error=RuntimeError)


_assert_registry_coverage()
