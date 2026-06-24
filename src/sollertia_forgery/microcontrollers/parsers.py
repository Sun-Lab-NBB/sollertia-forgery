"""Provides the parser-injection layer for the microcontroller processing pipeline: the per-module parser
descriptor, the per-system parser provider protocol, and the unified registry that maps each acquisition system to
its provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping, Callable

    import polars as pl
    from sollertia_shared_assets import SessionData


@dataclass(frozen=True, slots=True)
class ModuleParser:
    """Binds a single hardware module's interpretation logic to the destination of its parsed output.

    Notes:
        Instances are produced by a MicrocontrollerParserProvider for one eligible module. The pipeline reads the
        module's extracted feather, partitions it by event code, and hands the partition plus ``output_path`` to
        ``parse``. The provider closes any acquisition-system configuration (such as a hardware state) over
        ``parse`` so the pipeline never sees system-specific parameters.
    """

    parse: Callable[[dict[int, pl.DataFrame], Path], None]
    """Transforms one module's event-code-keyed partition into a domain-specific feather written to the output
    path. Must be picklable (a module-level function or a partial of one, not a lambda or closure) so it can be
    dispatched to worker processes during parallel runs."""
    output_path: Path
    """The absolute path of the domain-specific feather file this parser writes."""


class MicrocontrollerParserProvider(Protocol):
    """Supplies the per-system module parsers consumed by the microcontroller processing pipeline.

    Notes:
        An acquisition system implements ``resolve`` to map each module it understands to a ModuleParser, then
        registers the provider against its acquisition system via ``register_parsers``. The pipeline stays
        system-agnostic: it extracts and dispatches uniformly and learns how to interpret a module only through the
        provider it looks up for the session's acquisition system.
    """

    def resolve(self, session: SessionData) -> Mapping[tuple[int, int], ModuleParser]:
        """Resolves the eligible module parsers for the target session.

        Args:
            session: The loaded session whose microcontroller data is being processed. Implementations load any
                system-specific configuration (such as a hardware state) from the session, decide which modules are
                eligible for processing, bind that configuration into each parser, and choose each output path.

        Returns:
            A mapping from each eligible ``(module_type, module_id)`` pair to its ModuleParser. Modules absent from
            the mapping are treated as ineligible and are skipped by the pipeline.
        """
        ...


_PARSER_REGISTRY: dict[AcquisitionSystems, MicrocontrollerParserProvider] = {}
"""The unified registry mapping each acquisition system to its microcontroller parser provider. Acquisition-system
packages populate this registry via ``register_parsers``; the agnostic pipeline reads it and selects the provider
matching the processed session's acquisition system."""


def register_parsers(system: AcquisitionSystems, provider: MicrocontrollerParserProvider) -> None:
    """Registers an acquisition system's parser provider in the unified registry.

    Notes:
        Acquisition-system packages call this (typically at import time) to make their module parsers available to
        the agnostic pipeline. Registering the same system again replaces the previous provider, so a system can
        override its registration.

    Args:
        system: The acquisition system the provider interprets microcontroller data for.
        provider: The provider that resolves the system's per-module parsers for a session.
    """
    _PARSER_REGISTRY[system] = provider


def resolve_parsers(system: str | AcquisitionSystems) -> MicrocontrollerParserProvider:
    """Resolves the registered parser provider for the target acquisition system.

    Args:
        system: The acquisition system to resolve, as an AcquisitionSystems member or its string value (for
            example, the value carried by ``SessionData.acquisition_system``).

    Returns:
        The registered MicrocontrollerParserProvider for the acquisition system.

    Raises:
        ValueError: If the identifier is not a valid acquisition system, or if no provider has been registered for
            it. The error names the registered systems so the caller can locate the missing registration.
    """
    if system not in AcquisitionSystems:
        valid = ", ".join(member.value for member in AcquisitionSystems)
        message = (
            f"Unable to resolve microcontroller parsers for the acquisition system '{system}'. Expected one of the "
            f"supported AcquisitionSystems members: {valid}."
        )
        console.error(message=message, error=ValueError)

    resolved_system = AcquisitionSystems(system)
    if resolved_system not in _PARSER_REGISTRY:
        registered = ", ".join(member.value for member in _PARSER_REGISTRY) or "<none>"
        message = (
            f"Unable to resolve microcontroller parsers for the '{resolved_system.value}' acquisition system. No "
            f"parser provider has been registered for it. Registered systems: {registered}. Acquisition-system "
            f"packages register their providers via register_parsers()."
        )
        console.error(message=message, error=ValueError)

    return _PARSER_REGISTRY[resolved_system]
