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
    FORGING_CONCURRENCY,
    BEHAVIOR_CONCURRENCY,
    forge_dataset,
    run_forging_job,
    run_behavior_job,
    clean_forging_unit,
    clean_behavior_unit,
    verify_forging_unit,
    prepare_forging_unit,
    process_project_data,
    run_forging_pipeline,
    verify_behavior_unit,
    prepare_behavior_unit,
    iterate_forging_overview,
    iterate_behavior_overview,
    run_activity_processing_pipeline,
    run_behavior_processing_pipeline,
    run_multidataset_processing_pipeline,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable, Iterator

    from .orchestration import GenericPendingJob
    from .mesoscope_vr.batch import ConcurrencyDescriptor

__all__ = [
    "AGNOSTIC_PIPELINES",
    "CLEAN_REGISTRY",
    "CONCURRENCY_REGISTRY",
    "LOCAL_PIPELINE_REGISTRY",
    "MCP_BATCH_PIPELINES",
    "OVERVIEW_REGISTRY",
    "PREPARE_REGISTRY",
    "REMOTE_FORGING_ORCHESTRATOR_REGISTRY",
    "REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY",
    "SYSTEM_PIPELINES",
    "VERIFY_REGISTRY",
    "WORKER_REGISTRY",
    "resolve_clean",
    "resolve_concurrency",
    "resolve_local_pipeline",
    "resolve_overview",
    "resolve_prepare",
    "resolve_remote_forging_orchestrator",
    "resolve_remote_processing_orchestrator",
    "resolve_verify",
    "resolve_worker",
]


AGNOSTIC_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.MANIFEST,
        ProcessingPipelines.CHECKSUM,
    }
)
"""The pipelines that are platform-wide rather than acquisition-system-specific. Their entry points live in the
system-agnostic ``managing`` layer and are invoked directly by the interface, so they are not dispatched through the
per-system registries."""

SYSTEM_PIPELINES: dict[AcquisitionSystems, frozenset[ProcessingPipelines]] = {
    AcquisitionSystems.MESOSCOPE_VR: frozenset(
        {
            ProcessingPipelines.BEHAVIOR,
            ProcessingPipelines.CINDRA_SINGLE_RECORDING,
            ProcessingPipelines.CINDRA_MULTI_RECORDING,
            ProcessingPipelines.FORGING,
        }
    ),
}
"""Maps each acquisition system to the set of system-specific pipelines it runs. The interface consults this map to
decide which system-specific subcommands a system exposes, and the import-time checks use it to verify that every
declared pipeline has a registered local entry point."""

LOCAL_PIPELINE_REGISTRY: dict[AcquisitionSystems, dict[ProcessingPipelines, Callable[..., None]]] = {
    AcquisitionSystems.MESOSCOPE_VR: {
        ProcessingPipelines.BEHAVIOR: run_behavior_processing_pipeline,
        ProcessingPipelines.CINDRA_SINGLE_RECORDING: run_activity_processing_pipeline,
        ProcessingPipelines.CINDRA_MULTI_RECORDING: run_multidataset_processing_pipeline,
        ProcessingPipelines.FORGING: run_forging_pipeline,
    },
}
"""Maps each acquisition system and system-specific pipeline to the in-process entry point that runs that pipeline on
a single session or dataset. The generic ``slf process``/``slf forge`` commands and the remote SLURM jobs both
dispatch through this registry after inferring the session's acquisition system. Each pipeline defines its own call
convention shared across every system that implements it."""

REMOTE_PROCESSING_ORCHESTRATOR_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: process_project_data,
}
"""Maps each acquisition system to the orchestrator that resolves and submits its remote data-processing pipelines to
the compute server. The generic ``slf execute process`` command dispatches through this registry after inferring the
project's acquisition system from the manifest."""

REMOTE_FORGING_ORCHESTRATOR_REGISTRY: dict[AcquisitionSystems, Callable[..., None]] = {
    AcquisitionSystems.MESOSCOPE_VR: forge_dataset,
}
"""Maps each acquisition system to the orchestrator that resolves and submits its remote dataset-forging pipelines to
the compute server. The generic ``slf execute forge`` command dispatches through this registry after inferring the
project's acquisition system from the manifest."""

MCP_BATCH_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.BEHAVIOR,
        ProcessingPipelines.FORGING,
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
    ProcessingPipelines.FORGING: prepare_forging_unit,
}
"""Maps each batch pipeline to its discover-and-prepare adapter. The behavior adapter discovers jobs for a single
session and initializes its tracker; the forging adapter resolves a dataset hierarchy and initializes its tracker."""

VERIFY_REGISTRY: dict[ProcessingPipelines, Callable[..., dict[str, Any]]] = {
    ProcessingPipelines.BEHAVIOR: verify_behavior_unit,
    ProcessingPipelines.FORGING: verify_forging_unit,
}
"""Maps each batch pipeline to its output-verification adapter. Each adapter owns the full verification result,
including the ``verified`` boolean and the tracker block, which the generic tool merges into its response envelope."""

CLEAN_REGISTRY: dict[ProcessingPipelines, tuple[Callable[..., dict[str, Any]], bool]] = {
    ProcessingPipelines.BEHAVIOR: (clean_behavior_unit, False),
    ProcessingPipelines.FORGING: (clean_forging_unit, True),
}
"""Maps each batch pipeline to its ``(clean_fn, guard_on_active)`` pair. ``guard_on_active`` is True for pipelines
whose cleanup must refuse to run while an execution session is still writing to the targeted files (forging deletes
the entire dataset tree, so it guards; behavior only removes a per-session subdirectory)."""

OVERVIEW_REGISTRY: dict[ProcessingPipelines, Callable[[str], Iterator[dict[str, Any]]]] = {
    ProcessingPipelines.BEHAVIOR: iterate_behavior_overview,
    ProcessingPipelines.FORGING: iterate_forging_overview,
}
"""Maps each batch pipeline to its overview iterator: a callable accepting a root directory and yielding per-unit
status descriptors (per-session for behavior, per-dataset for forging)."""

CONCURRENCY_REGISTRY: dict[ProcessingPipelines, ConcurrencyDescriptor] = {
    ProcessingPipelines.BEHAVIOR: BEHAVIOR_CONCURRENCY,
    ProcessingPipelines.FORGING: FORGING_CONCURRENCY,
}
"""Maps each batch pipeline to its concurrency descriptor (``cores_per_job`` and ``default_max_parallel``) used by
the generic ``execute_jobs_tool`` to floor the parallel-job cap by the resolved worker budget."""

WORKER_REGISTRY: dict[ProcessingPipelines, Callable[[GenericPendingJob], None]] = {
    ProcessingPipelines.BEHAVIOR: run_behavior_job,
    ProcessingPipelines.FORGING: run_forging_job,
}
"""Maps each batch pipeline to its picklable module-level worker, which maps the generic ``GenericPendingJob`` fields
onto the pipeline's call convention and runs the single job in the worker subprocess."""


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


def resolve_remote_forging_orchestrator(system: str | AcquisitionSystems) -> Callable[..., None]:
    """Resolves the remote dataset-forging orchestrator for the target acquisition system.

    Args:
        system: The acquisition system that recorded the project's sessions.

    Returns:
        The registered orchestrator callable for the acquisition system.

    Raises:
        ValueError: If the acquisition system is unknown.
    """
    return REMOTE_FORGING_ORCHESTRATOR_REGISTRY[_resolve_system(system)]


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


def _assert_registry_coverage() -> None:
    """Verifies at import time that every acquisition system is fully wired into the dispatch registries.

    Confirms that every ``AcquisitionSystems`` member declares its pipelines in ``SYSTEM_PIPELINES`` and has an entry
    in both remote-orchestrator registries; that every pipeline a system declares has a registered local entry point
    (and no extra entries); and that ``SYSTEM_PIPELINES`` together with ``AGNOSTIC_PIPELINES`` covers exactly the
    ``ProcessingPipelines`` enum. With a single acquisition system these checks are structural scaffolding that
    enforces full wiring; they begin catching cross-system gaps once a second system is added.

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
        ("REMOTE_FORGING_ORCHESTRATOR_REGISTRY", REMOTE_FORGING_ORCHESTRATOR_REGISTRY),
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
