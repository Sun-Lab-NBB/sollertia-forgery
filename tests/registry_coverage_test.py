"""Tests the import-time donor-registry coverage check and the resolver guarantees it underwrites.

The coverage check is what lets every resolver in ``registries.py`` document ``ValueError`` as its only failure mode.
A registry left out of the check would let a valid acquisition system reach a bare ``KeyError`` at dispatch time, so
every donor registry's guard is pinned here alongside the resolver behavior it protects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sollertia_shared_assets import AcquisitionSystems

from sollertia_forgery.registries import (
    _MICROCONTROLLER_ELIGIBILITY_REGISTRY,
    resolve_video_tracking,
    resolve_runtime_binding,
    _assert_registry_coverage,
    resolve_forging_assembly_worker,
    resolve_microcontroller_parsers,
    resolve_two_photon_data_locator,
    resolve_forging_admission_pipelines,
    resolve_forging_column_descriptions,
    resolve_microcontroller_event_codes,
    resolve_eligible_microcontroller_modules,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from sollertia_forgery.mesoscope_vr import (
    RUNTIME_SOURCE_ID,
    MESOSCOPE_ADMISSION_PIPELINES,
    MESOSCOPE_COLUMN_DESCRIPTIONS,
    parse_runtime,
    get_module_event_codes,
    locate_two_photon_data,
    assemble_mesoscope_session,
    process_mesoscope_video_tracking,
    resolve_multi_recording_configuration,
    resolve_single_recording_configuration,
)

if TYPE_CHECKING:
    from sollertia_shared_assets import SessionData

DONOR_REGISTRY_NAMES: tuple[str, ...] = (
    "_FORGING_ASSEMBLY_REGISTRY",
    "_RUNTIME_PARSER_REGISTRY",
    "_TWO_PHOTON_DATA_REGISTRY",
    "_VIDEO_TRACKING_REGISTRY",
    "_MICROCONTROLLER_EVENT_CODE_REGISTRY",
    "_MICROCONTROLLER_ELIGIBILITY_REGISTRY",
    "_CINDRA_CONFIGURATION_REGISTRY",
    "_FORGING_ADMISSION_REGISTRY",
    "_MICROCONTROLLER_PARSER_REGISTRY",
)
"""Every registry an acquisition system donates to, each of which the import-time check must cover."""


def test_an_unknown_acquisition_system_is_rejected_with_a_value_error():
    """The eligibility resolver reports an unsupported identifier as the ValueError its docstring documents."""
    session = SimpleNamespace(session_name="2026-01-02-03-04-05-000006")
    with pytest.raises(ValueError, match="Unable to resolve the acquisition system"):
        resolve_eligible_microcontroller_modules(system="not-a-system", session=session)


def test_every_acquisition_system_donates_an_eligibility_accessor():
    """A registered system resolves through the eligibility registry instead of reaching a bare KeyError."""
    assert set(_MICROCONTROLLER_ELIGIBILITY_REGISTRY) == set(AcquisitionSystems)


@pytest.mark.parametrize("registry_name", DONOR_REGISTRY_NAMES)
def test_the_coverage_check_guards_every_donor_registry(monkeypatch, registry_name):
    """Emptying any one donor registry fails the import-time check, naming the registry that lost its entries."""
    monkeypatch.setattr(f"sollertia_forgery.registries.{registry_name}", {})
    with pytest.raises(RuntimeError, match=registry_name):
        _assert_registry_coverage()


def test_a_parseable_module_that_declares_no_event_codes_fails_the_check(monkeypatch):
    """Dropping a parseable module from the event-code registry would leave its parse job undiscovered."""
    monkeypatch.setattr(
        "sollertia_forgery.registries._MICROCONTROLLER_EVENT_CODE_REGISTRY",
        dict.fromkeys(AcquisitionSystems, dict),
    )

    uncoded = r"MESOSCOPE_VR does not declare\s+codes for the following modules: \(1, 1\)"
    with pytest.raises(RuntimeError, match=uncoded):
        _assert_registry_coverage()


@pytest.mark.parametrize("system", [AcquisitionSystems.MESOSCOPE_VR, AcquisitionSystems.MESOSCOPE_VR.value])
def test_a_system_resolves_the_same_assets_whether_named_by_member_or_by_value(system):
    """A caller reads the identifier off a marker as a plain string, so both spellings must reach one registration."""
    assert resolve_forging_assembly_worker(system=system) is assemble_mesoscope_session
    assert resolve_forging_column_descriptions(system=system) is MESOSCOPE_COLUMN_DESCRIPTIONS
    assert resolve_forging_admission_pipelines(system=system) is MESOSCOPE_ADMISSION_PIPELINES
    assert resolve_two_photon_data_locator(system=system) is locate_two_photon_data
    assert resolve_video_tracking(system=system) is process_mesoscope_video_tracking
    assert resolve_runtime_binding(system=system) == (RUNTIME_SOURCE_ID, parse_runtime)


def test_the_cindra_configuration_resolvers_are_donated_as_a_pair():
    """The two-photon and forging pipelines each reach one half of the bundle, so both must resolve separately."""
    system = AcquisitionSystems.MESOSCOPE_VR

    assert resolve_single_recording_configuration_resolver(system=system) is resolve_single_recording_configuration
    assert resolve_multi_recording_configuration_resolver(system=system) is resolve_multi_recording_configuration


def test_every_parseable_module_declares_the_event_codes_its_parser_reads():
    """The extraction stage filters each module by these codes, so the two mappings have to name the same modules."""
    parsers = resolve_microcontroller_parsers(system=AcquisitionSystems.MESOSCOPE_VR)

    assert set(parsers) == set(resolve_microcontroller_event_codes(system=AcquisitionSystems.MESOSCOPE_VR))
    assert set(parsers) == set(get_module_event_codes())


def test_a_session_resolves_the_modules_its_hardware_state_configured(experiment_session: SessionData):
    """The extraction filter is narrowed to these modules, so the resolver has to reach the system's own accessor."""
    eligible = resolve_eligible_microcontroller_modules(
        system=experiment_session.acquisition_system, session=experiment_session
    )

    assert eligible == set(resolve_microcontroller_parsers(system=experiment_session.acquisition_system))
