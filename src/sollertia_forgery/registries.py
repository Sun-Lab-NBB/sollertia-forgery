"""Collects every sollertia-forgery dispatch registry in one place and runs the import-time checks that guard them.

This module is the single canonical surface for wiring an acquisition system's processing and forging entry points
into the library. It is the only module that imports the per-system subpackages (``mesoscope_vr`` and its future
siblings) and binds their entry points into the dispatch registries, so the system-agnostic interface layer never
imports a system subpackage. The keying enumerations live in the leaf ``pipelines`` module and in
sollertia-shared-assets, which keeps this module importable by every registry consumer without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems

from .pipelines import ProcessingPipelines
from .mesoscope_vr import (
    BEHAVIOR_CONCURRENCY,
    run_behavior_job,
    clean_behavior_unit,
    process_project_data,
    verify_behavior_unit,
    prepare_behavior_unit,
    iterate_behavior_overview,
    assemble_mesoscope_session,
    run_behavior_processing_pipeline,
    run_multidataset_processing_pipeline,
)
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
    from typing import Any
    from collections.abc import Callable, Iterator

    from .orchestration import GenericPendingJob, ConcurrencyDescriptor

__all__ = [
    "AGNOSTIC_PIPELINES",
    "CLEAN_REGISTRY",
    "CONCURRENCY_REGISTRY",
    "FORGING_ASSEMBLY_REGISTRY",
    "LOCAL_PIPELINE_REGISTRY",
    "MCP_BATCH_PIPELINES",
    "MICROCONTROLLER_PARSER_REGISTRY",
    "OVERVIEW_REGISTRY",
    "PREPARE_REGISTRY",
    "REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY",
    "RUNTIME_PARSER_REGISTRY",
    "SYSTEM_PIPELINES",
    "VERIFY_REGISTRY",
    "WORKER_REGISTRY",
    "resolve_clean",
    "resolve_concurrency",
    "resolve_forging_assembly_worker",
    "resolve_local_pipeline",
    "resolve_microcontroller_parsers",
    "resolve_overview",
    "resolve_prepare",
    "resolve_remote_processing_orchestrator",
    "resolve_runtime_binding",
    "resolve_verify",
    "resolve_worker",
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
forging) and are invoked directly by the interface, so they are not dispatched through the per-system local-pipeline
registry. The forging pipeline still resolves a system-specific data-assembly worker, but it does so through its own
``FORGING_ASSEMBLY_REGISTRY`` rather than the ``(system, pipeline)`` local-pipeline registry."""

SYSTEM_PIPELINES: dict[AcquisitionSystems, frozenset[ProcessingPipelines]] = {
    AcquisitionSystems.MESOSCOPE_VR: frozenset(
        {
            ProcessingPipelines.BEHAVIOR,
            ProcessingPipelines.CINDRA_MULTI_RECORDING,
        }
    ),
}
"""Maps each acquisition system to the set of system-specific pipelines it runs. The interface consults this map to
decide which system-specific subcommands a system exposes, and the import-time checks use it to verify that every
declared pipeline has a registered local entry point."""

LOCAL_PIPELINE_REGISTRY: dict[AcquisitionSystems, dict[ProcessingPipelines, Callable[..., None]]] = {
    AcquisitionSystems.MESOSCOPE_VR: {
        ProcessingPipelines.BEHAVIOR: run_behavior_processing_pipeline,
        ProcessingPipelines.CINDRA_MULTI_RECORDING: run_multidataset_processing_pipeline,
    },
}
"""Maps each acquisition system and system-specific pipeline to the in-process entry point that runs that pipeline on
a single session or dataset. The generic ``slf process`` command and the remote SLURM jobs both dispatch through this
registry after inferring the session's acquisition system. Each pipeline defines its own call convention shared across
every system that implements it. The forging pipeline is agnostic and is invoked directly by the interface, so it is
not registered here; it resolves its system-specific data-assembly worker through ``FORGING_ASSEMBLY_REGISTRY``."""

REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: process_project_data,
}
"""Maps each acquisition system to the orchestrator that resolves and submits its remote data-processing pipelines to
the compute server. The generic ``slf execute process`` command dispatches through this registry after inferring the
project's acquisition system from the manifest."""

MCP_BATCH_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.BEHAVIOR,
    }
)
"""The pipelines exposed by the system-agnostic batch MCP tools in ``interfaces/processing_tools.py``. Only the
local, per-session/per-dataset batch pipelines belong here: the cindra pipelines run via the remote orchestrators
(``REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY``) rather than these in-process batch tools, and the agnostic pipelines
(``AGNOSTIC_PIPELINES``) have their own dedicated MCP tools (manifest, checksum). The generic processing tools
validate every ``pipeline`` argument against this set and dispatch through the batch registries below, which are
keyed by ``ProcessingPipelines`` rather than ``(system, pipeline)`` because reset
and overview operate on bare tracker/root paths and cannot infer the acquisition system before dispatch; the adapter
loads the session or dataset internally where it needs system specifics."""

PREPARE_REGISTRY: dict[ProcessingPipelines, Callable[..., dict[str, Any]]] = {
    ProcessingPipelines.BEHAVIOR: prepare_behavior_unit,
}
"""Maps each batch pipeline to its discover-and-prepare adapter. The behavior adapter discovers jobs for a single
session and initializes its tracker."""

VERIFY_REGISTRY: dict[ProcessingPipelines, Callable[..., dict[str, Any]]] = {
    ProcessingPipelines.BEHAVIOR: verify_behavior_unit,
}
"""Maps each batch pipeline to its output-verification adapter. Each adapter owns the full verification result,
including the ``verified`` boolean and the tracker block, which the generic tool merges into its response envelope."""

CLEAN_REGISTRY: dict[ProcessingPipelines, tuple[Callable[..., dict[str, Any]], bool]] = {
    ProcessingPipelines.BEHAVIOR: (clean_behavior_unit, False),
}
"""Maps each batch pipeline to its ``(clean_fn, guard_on_active)`` pair. ``guard_on_active`` is True for pipelines
whose cleanup must refuse to run while an execution session is still writing to the targeted files (behavior removes
a per-session subdirectory, so it does not guard)."""

OVERVIEW_REGISTRY: dict[ProcessingPipelines, Callable[[str], Iterator[dict[str, Any]]]] = {
    ProcessingPipelines.BEHAVIOR: iterate_behavior_overview,
}
"""Maps each batch pipeline to its overview iterator: a callable accepting a root directory and yielding per-unit
status descriptors (per-session for behavior)."""

CONCURRENCY_REGISTRY: dict[ProcessingPipelines, ConcurrencyDescriptor] = {
    ProcessingPipelines.BEHAVIOR: BEHAVIOR_CONCURRENCY,
}
"""Maps each batch pipeline to its concurrency descriptor (``cores_per_job`` and ``default_max_parallel``) used by
the generic ``execute_jobs_tool`` to floor the parallel-job cap by the resolved worker budget."""

WORKER_REGISTRY: dict[ProcessingPipelines, Callable[[GenericPendingJob], None]] = {
    ProcessingPipelines.BEHAVIOR: run_behavior_job,
}
"""Maps each batch pipeline to its picklable module-level worker, which maps the generic ``GenericPendingJob`` fields
onto the pipeline's call convention and runs the single job in the worker subprocess."""

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


def resolve_local_pipeline(system: str | AcquisitionSystems, pipeline: ProcessingPipelines) -> Callable[..., None]:
    """Resolves the in-process entry point that runs the target pipeline for the target acquisition system.

    Args:
        system: The acquisition system that recorded the session or dataset being processed.
        pipeline: The system-specific pipeline to run.

    Returns:
        The registered entry-point callable for the (system, pipeline) pair.

    Raises:
        ValueError: If the acquisition system is unknown or does not run the requested pipeline.
    """
    resolved_system = _resolve_system(system)
    pipelines = LOCAL_PIPELINE_REGISTRY[resolved_system]
    if pipeline not in pipelines:
        valid = ", ".join(member.value for member in pipelines)
        message = (
            f"Unable to resolve the '{pipeline}' pipeline for the '{resolved_system.value}' acquisition system. The "
            f"system runs the following local pipelines: {valid}."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover
    return pipelines[pipeline]


def resolve_remote_processing_orchestrator(system: str | AcquisitionSystems) -> Callable[..., None]:
    """Resolves the remote data-processing orchestrator for the target acquisition system.

    Args:
        system: The acquisition system that recorded the project's sessions.

    Returns:
        The registered orchestrator callable for the acquisition system.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY[_resolve_system(system)]


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


def _resolve_batch_pipeline(pipeline: ProcessingPipelines) -> ProcessingPipelines:
    """Validates that the requested pipeline is exposed by the system-agnostic batch MCP tools.

    Args:
        pipeline: The pipeline to validate against ``MCP_BATCH_PIPELINES``.

    Returns:
        The validated pipeline.

    Raises:
        ValueError: If the pipeline is not a member of ``MCP_BATCH_PIPELINES``.
    """
    if pipeline not in MCP_BATCH_PIPELINES:
        valid = ", ".join(member.value for member in MCP_BATCH_PIPELINES)
        message = (
            f"Unable to resolve the batch adapter for the '{pipeline}' pipeline. The system-agnostic batch MCP "
            f"tools expose only the following pipelines: {valid}."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover
    return pipeline


def resolve_prepare(pipeline: ProcessingPipelines) -> Callable[..., dict[str, Any]]:
    """Resolves the discover-and-prepare adapter for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose prepare adapter to resolve.

    Returns:
        The registered prepare adapter callable.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return PREPARE_REGISTRY[_resolve_batch_pipeline(pipeline)]


def resolve_verify(pipeline: ProcessingPipelines) -> Callable[..., dict[str, Any]]:
    """Resolves the output-verification adapter for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose verify adapter to resolve.

    Returns:
        The registered verify adapter callable.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return VERIFY_REGISTRY[_resolve_batch_pipeline(pipeline)]


def resolve_clean(pipeline: ProcessingPipelines) -> tuple[Callable[..., dict[str, Any]], bool]:
    """Resolves the cleanup adapter and its active-guard policy for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose clean adapter to resolve.

    Returns:
        A ``(clean_fn, guard_on_active)`` tuple.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return CLEAN_REGISTRY[_resolve_batch_pipeline(pipeline)]


def resolve_overview(pipeline: ProcessingPipelines) -> Callable[[str], Iterator[dict[str, Any]]]:
    """Resolves the status-overview iterator for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose overview iterator to resolve.

    Returns:
        The registered overview iterator callable.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return OVERVIEW_REGISTRY[_resolve_batch_pipeline(pipeline)]


def resolve_concurrency(pipeline: ProcessingPipelines) -> ConcurrencyDescriptor:
    """Resolves the concurrency descriptor for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose concurrency descriptor to resolve.

    Returns:
        The registered concurrency descriptor.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return CONCURRENCY_REGISTRY[_resolve_batch_pipeline(pipeline)]


def resolve_worker(pipeline: ProcessingPipelines) -> Callable[[GenericPendingJob], None]:
    """Resolves the picklable per-job worker for the target batch pipeline.

    Args:
        pipeline: The batch pipeline whose worker to resolve.

    Returns:
        The registered worker callable.

    Raises:
        ValueError: If the pipeline is not exposed by the batch MCP tools.
    """
    return WORKER_REGISTRY[_resolve_batch_pipeline(pipeline)]


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
    in the remote-processing orchestrator registry, the forging-assembly registry, and the runtime-parser registry;
    that every pipeline a system declares has a registered local entry point (and no extra entries); and that
    ``SYSTEM_PIPELINES`` together with
    ``AGNOSTIC_PIPELINES`` covers exactly the ``ProcessingPipelines`` enum. With a single acquisition system these
    checks are structural scaffolding that enforces full wiring; they begin catching cross-system gaps once a second
    system is added.

    Raises:
        RuntimeError: If any acquisition system is missing from a registry, declares a pipeline without a local entry
            point, or if the pipeline partition does not cover the ``ProcessingPipelines`` enum. The error names the
            offending members so extenders can immediately locate the unwired touch point.
    """
    systems = frozenset(AcquisitionSystems)

    for registry_name, registry in (
        ("SYSTEM_PIPELINES", SYSTEM_PIPELINES),
        ("LOCAL_PIPELINE_REGISTRY", LOCAL_PIPELINE_REGISTRY),
        ("REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY", REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY),
        ("FORGING_ASSEMBLY_REGISTRY", FORGING_ASSEMBLY_REGISTRY),
        ("RUNTIME_PARSER_REGISTRY", RUNTIME_PARSER_REGISTRY),
    ):
        missing = systems - frozenset(registry)
        if missing:
            missing_names = ", ".join(sorted(member.name for member in missing))
            message = (
                f"{registry_name} is missing entries for {missing_names}. Every acquisition system must register its "
                f"processing and forging entry points. See the README's 'Adding New Acquisition Systems' section."
            )
            console.error(message=message, error=RuntimeError)

    for system, pipelines in SYSTEM_PIPELINES.items():
        registered = frozenset(LOCAL_PIPELINE_REGISTRY.get(system, {}))
        unwired = pipelines - registered
        if unwired:
            unwired_names = ", ".join(sorted(member.name for member in unwired))
            message = (
                f"LOCAL_PIPELINE_REGISTRY is missing entries for {system.name} pipelines {unwired_names}. Every "
                f"pipeline declared in SYSTEM_PIPELINES must have a registered local entry point."
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

    # Verifies that every pipeline exposed by the system-agnostic batch MCP tools is wired into each batch registry.
    # The generic processing tools resolve their adapters through these registries at call time, so a missing entry
    # would only surface as a KeyError mid-batch rather than at import; this check moves the failure to import time.
    for batch_registry_name, batch_registry in (
        ("PREPARE_REGISTRY", PREPARE_REGISTRY),
        ("VERIFY_REGISTRY", VERIFY_REGISTRY),
        ("CLEAN_REGISTRY", CLEAN_REGISTRY),
        ("OVERVIEW_REGISTRY", OVERVIEW_REGISTRY),
        ("CONCURRENCY_REGISTRY", CONCURRENCY_REGISTRY),
        ("WORKER_REGISTRY", WORKER_REGISTRY),
    ):
        batch_missing = MCP_BATCH_PIPELINES - frozenset(batch_registry)
        if batch_missing:
            batch_missing_names = ", ".join(sorted(member.name for member in batch_missing))
            message = (
                f"{batch_registry_name} is missing entries for {batch_missing_names}. Every pipeline listed in "
                f"MCP_BATCH_PIPELINES must register its batch adapter in every batch registry so the system-agnostic "
                f"processing tools can resolve it. See the README's 'Adding New Acquisition Systems' section."
            )
            console.error(message=message, error=RuntimeError)


_assert_registry_coverage()
