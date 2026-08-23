"""Collects every acquisition-system-keyed dispatch registry in one place and runs the import-time checks that guard
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import SYSTEM_SESSION_TYPES, SessionTypes, AcquisitionSystems

from .mesoscope_vr import (
    RUNTIME_SOURCE_ID,
    MESOSCOPE_ADMISSION_PIPELINES,
    MESOSCOPE_COLUMN_DESCRIPTIONS,
    MESOSCOPE_MULTI_RECORDING_SESSION_TYPES,
    parse_lick,
    parse_brake,
    parse_valve,
    parse_screen,
    parse_torque,
    parse_encoder,
    parse_runtime,
    parse_gas_puff,
    get_eligible_modules,
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
    import polars as pl
    from sollertia_shared_assets import SessionData

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
    "resolve_multi_recording_session_types",
    "resolve_runtime_binding",
    "resolve_single_recording_configuration_resolver",
    "resolve_two_photon_data_locator",
    "resolve_video_tracking",
]


class ForgingAssembler(Protocol):
    """Defines the call signature of the per-session forging data-assembly worker an acquisition system donates."""

    def __call__(self, source_session_path: Path, output_path: Path, dataset_name: str) -> None:
        """Assembles the source session's data into the target dataset's ``data.feather``."""


class MicrocontrollerParser(Protocol):
    """Defines the call signature of the microcontroller module parsers an acquisition system donates."""

    def __call__(self, event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
        """Parses one hardware module's extracted events into the session's behavior feathers."""


class RuntimeParser(Protocol):
    """Defines the call signature of the runtime log parser an acquisition system donates."""

    def __call__(self, decoded_messages: pl.DataFrame, output_directory: Path, session: SessionData) -> None:
        """Interprets the decoded runtime payloads into the session's behavior feathers."""


class TwoPhotonDataLocator(Protocol):
    """Defines the call signature of the raw two-photon imaging directory locator an acquisition system donates."""

    def __call__(self, session: SessionData) -> Path:
        """Resolves the session's raw two-photon imaging directory."""


class VideoTracker(Protocol):
    """Defines the call signature of the video-tracking function an acquisition system donates."""

    def __call__(self, session: SessionData, output_directory: Path) -> None:
        """Performs all the acquisition system's video tracking for the session."""


@dataclass(frozen=True, slots=True)
class _ForgingAssemblyAsset:
    """Bundles an acquisition system's donated forging assets."""

    assembler: ForgingAssembler
    """The picklable, module-level worker that assembles one session's ``data.feather``. The agnostic forging
    pipeline invokes it once per session.
    """
    column_descriptions: dict[str, str]
    """The mapping from each column name the assembler can emit into ``data.feather`` to its human-readable
    description. The agnostic forging pipeline bakes it into the dataset's ``data_descriptions.feather`` once, at
    dataset-definition time.
    """


@dataclass(frozen=True, slots=True)
class _CindraConfigurationAsset:
    """Bundles an acquisition system's donated cindra configuration resolvers."""

    resolve_single_recording: Callable[[SessionData], SingleRecordingConfiguration]
    """The resolver that returns the system's single-recording cindra configuration for the session, or raises when
    it cannot resolve one. The agnostic two-photon pipeline calls it for the session it processes.
    """
    resolve_multi_recording: Callable[[SessionData], MultiRecordingConfiguration | None]
    """The resolver that returns the system's multi-recording cindra configuration for the session, or None when the
    system performs no cross-recording tracking for it. The agnostic forging pipeline calls it once per animal in the
    dataset and skips the animals for which it returns None.
    """


_MICROCONTROLLER_PARSER_REGISTRY: dict[tuple[AcquisitionSystems, int, int], MicrocontrollerParser] = {
    (AcquisitionSystems.MESOSCOPE_VR, 1, 1): parse_mesoscope_frame,
    (AcquisitionSystems.MESOSCOPE_VR, 2, 1): parse_encoder,
    (AcquisitionSystems.MESOSCOPE_VR, 3, 1): parse_brake,
    (AcquisitionSystems.MESOSCOPE_VR, 4, 1): parse_lick,
    (AcquisitionSystems.MESOSCOPE_VR, 5, 1): parse_valve,
    (AcquisitionSystems.MESOSCOPE_VR, 5, 2): parse_gas_puff,
    (AcquisitionSystems.MESOSCOPE_VR, 6, 1): parse_torque,
    (AcquisitionSystems.MESOSCOPE_VR, 7, 1): parse_screen,
}
"""Maps each ``(acquisition system, module_type, module_id)`` triplet to the module-level parser an
acquisition-system package implements for that hardware module. A module is parseable for a system exactly when it
appears here.
"""

_MICROCONTROLLER_EVENT_CODE_REGISTRY: dict[AcquisitionSystems, Callable[[], dict[tuple[int, int], tuple[int, ...]]]] = {
    AcquisitionSystems.MESOSCOPE_VR: get_module_event_codes,
}
"""Maps each acquisition system to the module-level accessor returning its ``(module_type, module_id) -> event
codes`` mapping for every module the system parses. The agnostic microcontroller pipeline derives each controller's
extraction filter from this mapping, so a system's event codes live next to the parsers that read them.
"""

_MICROCONTROLLER_ELIGIBILITY_REGISTRY: dict[AcquisitionSystems, Callable[[SessionData], set[tuple[int, int]]]] = {
    AcquisitionSystems.MESOSCOPE_VR: get_eligible_modules,
}
"""Maps each acquisition system to the module-level accessor returning the hardware modules a given session
configured for use. The agnostic microcontroller pipeline narrows each controller's extraction filter to these
modules, so a system's eligibility rules live next to the parsers that apply them.
"""

_FORGING_ASSEMBLY_REGISTRY: dict[AcquisitionSystems, _ForgingAssemblyAsset] = {
    AcquisitionSystems.MESOSCOPE_VR: _ForgingAssemblyAsset(
        assembler=assemble_mesoscope_session,
        column_descriptions=MESOSCOPE_COLUMN_DESCRIPTIONS,
    ),
}
"""Maps each acquisition system to the ``_ForgingAssemblyAsset`` bundling its per-session assembly worker with its
column-description mapping. Dataset definition, the cindra multi-recording stages, in-pipeline job and tracker
preparation, the per-dataset column-description binding, and shared-asset re-export are owned by the agnostic
``forging`` package.
"""

_FORGING_ADMISSION_REGISTRY: dict[AcquisitionSystems, dict[SessionTypes, frozenset[ProcessingPipelines]]] = {
    AcquisitionSystems.MESOSCOPE_VR: MESOSCOPE_ADMISSION_PIPELINES,
}
"""Maps each acquisition system to the pipelines each of its session types must have completed before a session may
join a forged dataset. Every pipeline resolves its own job universe from the acquisition manifests, so a completed
tracker already accounts for every source a session recorded, which is why a system declares pipelines rather than
source counts. A session type a system does not list joins no dataset.
"""

_CINDRA_CONFIGURATION_REGISTRY: dict[AcquisitionSystems, _CindraConfigurationAsset] = {
    AcquisitionSystems.MESOSCOPE_VR: _CindraConfigurationAsset(
        resolve_single_recording=resolve_single_recording_configuration,
        resolve_multi_recording=resolve_multi_recording_configuration,
    ),
}
"""Maps each acquisition system to the ``_CindraConfigurationAsset`` bundling its single- and multi-recording
configuration resolvers. The agnostic two-photon and forging pipelines obtain a runnable cindra configuration through
these resolvers, so each system decides for itself how its configuration is derived.
"""

_MULTI_RECORDING_SESSION_TYPE_REGISTRY: dict[AcquisitionSystems, frozenset[SessionTypes]] = {
    AcquisitionSystems.MESOSCOPE_VR: MESOSCOPE_MULTI_RECORDING_SESSION_TYPES,
}
"""Maps each acquisition system to the session types whose animals it tracks across recordings.

Notes:
    The multi-recording resolver decides the same question per session, but answering it needs a loaded session and
    therefore the source data. Declaring the session types separately lets the forging pipeline read the answer from a
    dataset's own recorded type, which is what keeps a dataset growing while part of its source data lives elsewhere.
    A system that tracks nothing across recordings declares an empty set.
"""

_RUNTIME_PARSER_REGISTRY: dict[AcquisitionSystems, tuple[str, RuntimeParser]] = {
    AcquisitionSystems.MESOSCOPE_VR: (RUNTIME_SOURCE_ID, parse_runtime),
}
"""Maps each acquisition system to its runtime DataLogger source id, which locates the ``{source_id}_log.npz``
archive, paired with the module-level parser that interprets the decoded runtime payloads into the system's behavior
feathers.
"""

_TWO_PHOTON_DATA_REGISTRY: dict[AcquisitionSystems, TwoPhotonDataLocator] = {
    AcquisitionSystems.MESOSCOPE_VR: locate_two_photon_data,
}
"""Maps each acquisition system to the module-level locator that resolves the loaded session's raw two-photon
(calcium-imaging) directory, which the agnostic two-photon worker hands to the cindra single-recording pipeline as
its input. Every system donates a locator, and a system that produces no two-photon data donates one returning the
path it would use.
"""

_VIDEO_TRACKING_REGISTRY: dict[AcquisitionSystems, VideoTracker] = {
    AcquisitionSystems.MESOSCOPE_VR: process_mesoscope_video_tracking,
}
"""Maps each acquisition system to the module-level function that performs all of that system's video tracking. The
function locates its own externally-produced DeepLabCut ``.h5`` predictions, parses the bodyparts it targets, and
writes its outputs into the session's processed video-data directory. The agnostic video pipeline runs it once per
session, and a system donates a no-op function when it performs no video tracking.
"""


def resolve_forging_assembly_worker(system: str | AcquisitionSystems) -> ForgingAssembler:
    """Resolves the per-session forging data-assembly worker registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, for example the value carried by
            ``DatasetData.acquisition_system``.

    Returns:
        The registered assembly worker. The agnostic forging pipeline invokes it once per session to write that
        session's ``data.feather``.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _FORGING_ASSEMBLY_REGISTRY[_resolve_system(system=system)].assembler


def resolve_forging_admission_pipelines(
    system: str | AcquisitionSystems,
) -> dict[SessionTypes, frozenset[ProcessingPipelines]]:
    """Resolves the per-session-type pipeline requirements a session must satisfy to join the system's datasets.

    Args:
        system: The acquisition system that recorded the session, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        The mapping from session type to the pipelines that must report every job as succeeded. A session type absent
        from the mapping joins no dataset for this system.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _FORGING_ADMISSION_REGISTRY[_resolve_system(system=system)]


def resolve_forging_column_descriptions(system: str | AcquisitionSystems) -> dict[str, str]:
    """Resolves the dataset column-description mapping registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, for example the value carried by
            ``DatasetData.acquisition_system``.

    Returns:
        The mapping from each column name the system's assembly worker can emit into ``data.feather`` to its
        human-readable description. The agnostic forging pipeline bakes it into the dataset's
        ``data_descriptions.feather`` at dataset-definition time.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _FORGING_ASSEMBLY_REGISTRY[_resolve_system(system=system)].column_descriptions


def resolve_single_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], SingleRecordingConfiguration]:
    """Resolves the single-recording cindra configuration resolver registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        The registered resolver. The agnostic two-photon pipeline calls it with the session it processes to obtain a
        runnable single-recording configuration.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _CINDRA_CONFIGURATION_REGISTRY[_resolve_system(system=system)].resolve_single_recording


def resolve_multi_recording_configuration_resolver(
    system: str | AcquisitionSystems,
) -> Callable[[SessionData], MultiRecordingConfiguration | None]:
    """Resolves the multi-recording cindra configuration resolver registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the dataset being forged, for example the value carried by
            ``DatasetData.acquisition_system``.

    Returns:
        The registered resolver. The agnostic forging pipeline calls it once per animal in the dataset to obtain a
        runnable multi-recording configuration, and skips the animals for which it returns None.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _CINDRA_CONFIGURATION_REGISTRY[_resolve_system(system=system)].resolve_multi_recording


def resolve_multi_recording_session_types(system: str | AcquisitionSystems) -> frozenset[SessionTypes]:
    """Resolves the session types the target acquisition system tracks across recordings.

    Notes:
        Answers whether a session type carries cross-recording tracking without loading a session, which the
        multi-recording configuration resolver needs one for. A caller holding a dataset therefore reads the answer
        from the dataset's own recorded session type rather than from its animals' source data.

    Args:
        system: The acquisition system that recorded the dataset being forged, for example the value carried by
            ``DatasetData.acquisition_system``.

    Returns:
        The session types whose animals the system registers against each other, which is empty for a system that
        performs no cross-recording tracking.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _MULTI_RECORDING_SESSION_TYPE_REGISTRY[_resolve_system(system=system)]


def resolve_microcontroller_event_codes(system: str | AcquisitionSystems) -> dict[tuple[int, int], tuple[int, ...]]:
    """Resolves the microcontroller module event codes registered for the target acquisition system.

    Notes:
        This helper invokes the registered accessor, so any system-specific exception the accessor raises propagates
        unchanged.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        A mapping from each ``(module_type, module_id)`` pair the system parses to the tuple of event codes its parser
        reads. The agnostic microcontroller pipeline builds every controller's extraction filter from this mapping.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _MICROCONTROLLER_EVENT_CODE_REGISTRY[_resolve_system(system=system)]()


def resolve_eligible_microcontroller_modules(
    system: str | AcquisitionSystems,
    session: SessionData,
) -> set[tuple[int, int]]:
    """Resolves the microcontroller modules the target session configured for use.

    Notes:
        This helper invokes the registered accessor, so any system-specific exception the accessor raises propagates
        unchanged.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.
        session: The loaded session whose hardware state determines module eligibility.

    Returns:
        The ``(module_type, module_id)`` pairs the session configured for use. The agnostic microcontroller pipeline
        narrows every controller's extraction filter to these modules.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _MICROCONTROLLER_ELIGIBILITY_REGISTRY[_resolve_system(system=system)](session)


def resolve_microcontroller_parsers(system: str | AcquisitionSystems) -> dict[tuple[int, int], MicrocontrollerParser]:
    """Resolves the microcontroller module parsers registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        A mapping from each ``(module_type, module_id)`` pair the system parses to its parser. The agnostic
        microcontroller pipeline treats this mapping as the set of parseable modules for the session.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    resolved_system = _resolve_system(system=system)
    return {
        (module_type, module_id): parser
        for (registered_system, module_type, module_id), parser in _MICROCONTROLLER_PARSER_REGISTRY.items()
        if registered_system == resolved_system
    }


def resolve_runtime_binding(system: str | AcquisitionSystems) -> tuple[str, RuntimeParser]:
    """Resolves the runtime source id and parser registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        A ``(source_id, parser)`` tuple. The source id locates the system's runtime DataLogger archive, and the parser
        interprets the decoded runtime messages into the system's behavior feathers. The agnostic runtime pipeline uses
        the source id to find the archive and dispatches the parser once the archive is decoded.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _RUNTIME_PARSER_REGISTRY[_resolve_system(system=system)]


def resolve_two_photon_data_locator(system: str | AcquisitionSystems) -> TwoPhotonDataLocator:
    """Resolves the raw two-photon imaging directory locator registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        The registered locator. The agnostic two-photon pipeline invokes it with the loaded session to obtain that
        session's raw two-photon imaging directory (cindra's input).

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _TWO_PHOTON_DATA_REGISTRY[_resolve_system(system=system)]


def resolve_video_tracking(system: str | AcquisitionSystems) -> VideoTracker:
    """Resolves the video-tracking function registered for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session being processed, for example the value carried by
            ``SessionData.acquisition_system``.

    Returns:
        The registered tracking function. The agnostic video pipeline runs it once per session, expecting it to
        perform all of that system's video tracking and to no-op when no DeepLabCut predictions are present.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return _VIDEO_TRACKING_REGISTRY[_resolve_system(system=system)]


def _resolve_system(system: str | AcquisitionSystems) -> AcquisitionSystems:
    """Validates and normalizes an acquisition-system identifier to an ``AcquisitionSystems`` member.

    Args:
        system: The acquisition-system identifier to validate and normalize.

    Returns:
        The corresponding ``AcquisitionSystems`` member.

    Raises:
        ValueError: If the identifier is not a valid ``AcquisitionSystems`` member.
    """
    if system not in AcquisitionSystems:
        valid_system_values = ", ".join(member.value for member in AcquisitionSystems)
        message = (
            f"Unable to resolve the acquisition system. The system must be one of the supported AcquisitionSystems "
            f"members ({valid_system_values}), but got '{system}'."
        )
        console.error(message=message, error=ValueError)

    return AcquisitionSystems(system)


def _assert_registry_coverage() -> None:
    """Verifies at import time that every acquisition system has registered every donated asset.

    Confirms that every ``AcquisitionSystems`` member has an entry in the forging-assembly, runtime-parser,
    two-photon-data, video-tracking, microcontroller event-code, microcontroller eligibility, cindra configuration,
    and forging-admission registries. Confirms that every member registers at least one microcontroller module parser,
    that every parseable microcontroller module declares the event codes its parser reads, and that every session type
    a system admits into a dataset is a session type that system records.

    Raises:
        RuntimeError: If any acquisition system is missing from a donor registry, if a parseable microcontroller
            module does not declare its event codes, or if a forging-admission entry names a session type its system
            does not record. The error names the offending members so extenders can immediately locate the unwired
            touch point.
    """
    systems = frozenset(AcquisitionSystems)
    microcontroller_systems = frozenset(system for system, _, _ in _MICROCONTROLLER_PARSER_REGISTRY)

    for registry_name, registered_systems in (
        ("_FORGING_ASSEMBLY_REGISTRY", frozenset(_FORGING_ASSEMBLY_REGISTRY)),
        ("_RUNTIME_PARSER_REGISTRY", frozenset(_RUNTIME_PARSER_REGISTRY)),
        ("_TWO_PHOTON_DATA_REGISTRY", frozenset(_TWO_PHOTON_DATA_REGISTRY)),
        ("_VIDEO_TRACKING_REGISTRY", frozenset(_VIDEO_TRACKING_REGISTRY)),
        ("_MICROCONTROLLER_EVENT_CODE_REGISTRY", frozenset(_MICROCONTROLLER_EVENT_CODE_REGISTRY)),
        ("_MICROCONTROLLER_ELIGIBILITY_REGISTRY", frozenset(_MICROCONTROLLER_ELIGIBILITY_REGISTRY)),
        ("_CINDRA_CONFIGURATION_REGISTRY", frozenset(_CINDRA_CONFIGURATION_REGISTRY)),
        ("_MULTI_RECORDING_SESSION_TYPE_REGISTRY", frozenset(_MULTI_RECORDING_SESSION_TYPE_REGISTRY)),
        ("_FORGING_ADMISSION_REGISTRY", frozenset(_FORGING_ADMISSION_REGISTRY)),
        ("_MICROCONTROLLER_PARSER_REGISTRY", microcontroller_systems),
    ):
        missing_systems = systems - registered_systems
        if missing_systems:
            missing_names = ", ".join(sorted(member.name for member in missing_systems))
            message = (
                f"Unable to validate donor-registry coverage for {registry_name}. Every acquisition system must "
                f"register its donated processing and forging assets in this module ('registries.py'), but entries "
                f"are missing for {missing_names}."
            )
            console.error(message=message, error=RuntimeError)

    # Requires every parseable module to also declare its event codes, because the extraction stage filters each
    # module by the codes resolved from _MICROCONTROLLER_EVENT_CODE_REGISTRY. Dropping a parseable module from that
    # registry would remove it from its controller's extraction configuration, leaving its parse job undiscovered.
    for target_system in sorted(microcontroller_systems, key=lambda member: member.name):
        parseable_modules = {
            (module_type, module_id)
            for (registered_system, module_type, module_id) in _MICROCONTROLLER_PARSER_REGISTRY
            if registered_system == target_system
        }
        uncoded_modules = sorted(parseable_modules - set(_MICROCONTROLLER_EVENT_CODE_REGISTRY[target_system]()))
        if uncoded_modules:
            module_names = ", ".join(f"({module_type}, {module_id})" for module_type, module_id in uncoded_modules)
            message = (
                f"Unable to validate donor-registry coverage for _MICROCONTROLLER_EVENT_CODE_REGISTRY. Every module "
                f"registered in _MICROCONTROLLER_PARSER_REGISTRY must also declare the event codes its parser reads, "
                f"but {target_system.name} does not declare codes for the following modules: {module_names}."
            )
            console.error(message=message, error=RuntimeError)

    # A system tracks across recordings only the session types it records, so a declaration naming a type outside the
    # shared assets library's own is a typo or a stale entry rather than a type this library knows more about.
    for target_system, tracked_types in sorted(
        _MULTI_RECORDING_SESSION_TYPE_REGISTRY.items(), key=lambda item: item[0].name
    ):
        untracked_types = sorted(tracked_types - SYSTEM_SESSION_TYPES[target_system])
        if untracked_types:
            type_names = ", ".join(session_type.value for session_type in untracked_types)
            message = (
                f"Unable to validate donor-registry coverage for _MULTI_RECORDING_SESSION_TYPE_REGISTRY. Every "
                f"session type a system tracks across recordings must be a session type that system records, but "
                f"{target_system.name} declares the following unrecorded type(s): {type_names}."
            )
            console.error(message=message, error=RuntimeError)

    # The session types a system records are the shared assets library's to declare, so an admission entry naming a
    # type outside that declaration is a typo or a stale entry rather than a system this library knows more about. A
    # type the system records and this registry omits is not an error, since a session type may deliberately join no
    # dataset.
    for target_system, requirements in sorted(_FORGING_ADMISSION_REGISTRY.items(), key=lambda item: item[0].name):
        unrecorded_types = sorted(set(requirements) - SYSTEM_SESSION_TYPES[target_system])
        if unrecorded_types:
            type_names = ", ".join(session_type.value for session_type in unrecorded_types)
            message = (
                f"Unable to validate donor-registry coverage for _FORGING_ADMISSION_REGISTRY. Every session type a "
                f"system admits into a forged dataset must be a session type that system records, but "
                f"{target_system.name} declares admission requirements for the following unrecorded type(s): "
                f"{type_names}."
            )
            console.error(message=message, error=RuntimeError)


_assert_registry_coverage()
