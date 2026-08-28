"""Contains tests for the import-time donor-registry coverage check and the resolver guarantees that it underwrites."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sollertia_shared_assets import SYSTEM_SESSION_TYPES, SessionTypes, AcquisitionSystems

from sollertia_forgery import registries as registries_module
from sollertia_forgery.registries import (
    _MICROCONTROLLER_ELIGIBILITY_REGISTRY,
    resolve_video_tracking,
    resolve_runtime_binding,
    _assert_registry_coverage,
    resolve_forging_assembly_worker,
    resolve_microcontroller_parsers,
    resolve_two_photon_data_locator,
    resolve_assembly_geometry_resolver,
    resolve_forging_admission_pipelines,
    resolve_forging_column_descriptions,
    resolve_microcontroller_event_codes,
    resolve_multi_recording_session_types,
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
    resolve_mesoscope_assembly_geometry,
    resolve_multi_recording_configuration,
    resolve_single_recording_configuration,
)

if TYPE_CHECKING:
    from sollertia_shared_assets import SessionData

_DONOR_REGISTRY_NAMES: tuple[str, ...] = (
    "_FORGING_ASSEMBLY_REGISTRY",
    "_ASSEMBLY_GEOMETRY_REGISTRY",
    "_ASSEMBLY_SOURCE_REGISTRY",
    "_RUNTIME_PARSER_REGISTRY",
    "_TWO_PHOTON_DATA_REGISTRY",
    "_VIDEO_TRACKING_REGISTRY",
    "_POSE_PREDICTION_REGISTRY",
    "_MICROCONTROLLER_EVENT_CODE_REGISTRY",
    "_MICROCONTROLLER_ELIGIBILITY_REGISTRY",
    "_CINDRA_CONFIGURATION_REGISTRY",
    "_MULTI_RECORDING_SESSION_TYPE_REGISTRY",
    "_FORGING_ADMISSION_REGISTRY",
    "_MICROCONTROLLER_PARSER_REGISTRY",
)
"""Every registry to which an acquisition system donates, each of which the import-time check must cover."""


def test_an_unknown_acquisition_system_is_rejected_with_a_value_error():
    """Verifies that the eligibility resolver reports an unsupported identifier as its documented ValueError."""
    session = SimpleNamespace(session_name="2026-01-02-03-04-05-000006")
    with pytest.raises(ValueError, match="Unable to resolve the acquisition system"):
        resolve_eligible_microcontroller_modules(system="not-a-system", session=session)


def test_every_acquisition_system_donates_an_eligibility_accessor():
    """Verifies that a registered system resolves through the eligibility registry instead of reaching KeyError."""
    assert set(_MICROCONTROLLER_ELIGIBILITY_REGISTRY) == set(AcquisitionSystems)


def test_the_guarded_names_are_every_registry_the_module_declares():
    # The tuple above is what parametrizes the guard test below, so a registry added to the module and left out of it
    # is a registry nothing checks. Deriving the expectation from the module itself is what closes that gap: the
    # pose-prediction registry sat unguarded here for exactly as long as nothing tied the two together.
    declared = {name for name in vars(registries_module) if name.endswith("_REGISTRY")}

    assert declared == set(_DONOR_REGISTRY_NAMES)


@pytest.mark.parametrize("registry_name", _DONOR_REGISTRY_NAMES)
def test_the_coverage_check_guards_every_donor_registry(monkeypatch, registry_name):
    """Verifies that emptying any one donor registry fails the import-time check, naming the registry that lost its
    entries.
    """
    monkeypatch.setattr(f"sollertia_forgery.registries.{registry_name}", {})
    with pytest.raises(RuntimeError, match=registry_name):
        _assert_registry_coverage()


def test_a_parseable_module_that_declares_no_event_codes_fails_the_check(monkeypatch):
    """Verifies that dropping a parseable module from the event-code registry would leave its parse job undiscovered."""
    monkeypatch.setattr(
        "sollertia_forgery.registries._MICROCONTROLLER_EVENT_CODE_REGISTRY",
        dict.fromkeys(AcquisitionSystems, dict),
    )

    uncoded = r"MESOSCOPE_VR does not declare\s+codes for the following modules: \(1, 1\)"
    with pytest.raises(RuntimeError, match=uncoded):
        _assert_registry_coverage()


def test_admitting_a_session_type_the_system_does_not_record_fails_the_check(monkeypatch):
    """Verifies that the shared assets library declares which session types a system records, so an admission entry
    naming a type outside that declaration is a stale entry rather than a system about which this library knows more.
    """
    monkeypatch.setattr(
        "sollertia_forgery.registries._FORGING_ADMISSION_REGISTRY",
        {AcquisitionSystems.MESOSCOPE_VR: {SessionTypes.WINDOW_CHECKING: frozenset()}},
    )
    monkeypatch.setattr(
        "sollertia_forgery.registries.SYSTEM_SESSION_TYPES",
        {AcquisitionSystems.MESOSCOPE_VR: frozenset({SessionTypes.MESOSCOPE_EXPERIMENT})},
    )

    unrecorded = (
        r"MESOSCOPE_VR declares admission requirements for the\s+following unrecorded type\(s\): window checking"
    )
    with pytest.raises(RuntimeError, match=unrecorded):
        _assert_registry_coverage()


def test_tracking_a_session_type_the_system_does_not_record_fails_the_check(monkeypatch):
    """Verifies that the cross-recording declaration answers whether a dataset needs multi-day plans without loading a
    session, so a type outside the shared assets library's own declaration would quietly change a dataset's plan.
    """
    monkeypatch.setattr(
        "sollertia_forgery.registries._MULTI_RECORDING_SESSION_TYPE_REGISTRY",
        {AcquisitionSystems.MESOSCOPE_VR: frozenset({SessionTypes.WINDOW_CHECKING})},
    )
    monkeypatch.setattr(
        "sollertia_forgery.registries.SYSTEM_SESSION_TYPES",
        {AcquisitionSystems.MESOSCOPE_VR: frozenset({SessionTypes.MESOSCOPE_EXPERIMENT})},
    )

    unrecorded = r"MESOSCOPE_VR declares the following unrecorded\s+type\(s\): window checking"
    with pytest.raises(RuntimeError, match=unrecorded):
        _assert_registry_coverage()


def test_every_cross_recording_session_type_is_one_the_system_records():
    """Verifies that a dataset's recorded session type is matched against this set, so a type that the system never
    records would make the answer unreachable rather than merely wrong.
    """
    tracked = resolve_multi_recording_session_types(system=AcquisitionSystems.MESOSCOPE_VR)

    assert tracked <= SYSTEM_SESSION_TYPES[AcquisitionSystems.MESOSCOPE_VR]
    # Cross-recording tracking needs calcium imaging, which only an experiment session records.
    assert tracked == frozenset({SessionTypes.MESOSCOPE_EXPERIMENT})


def test_every_admitted_session_type_is_one_the_system_records():
    """Verifies that the admission registry is keyed by session type, so a typo there would silently hold every session
    of the mistyped type out of every dataset.
    """
    admitted = set(resolve_forging_admission_pipelines(system=AcquisitionSystems.MESOSCOPE_VR))

    assert admitted <= SYSTEM_SESSION_TYPES[AcquisitionSystems.MESOSCOPE_VR]


@pytest.mark.parametrize("system", [AcquisitionSystems.MESOSCOPE_VR, AcquisitionSystems.MESOSCOPE_VR.value])
def test_a_system_resolves_the_same_assets_whether_named_by_member_or_by_value(system):
    """Verifies that a caller reads the identifier off a marker as a plain string, so both spellings must reach one
    registration.
    """
    assert resolve_forging_assembly_worker(system=system) is assemble_mesoscope_session
    assert resolve_forging_column_descriptions(system=system) is MESOSCOPE_COLUMN_DESCRIPTIONS
    assert resolve_forging_admission_pipelines(system=system) is MESOSCOPE_ADMISSION_PIPELINES
    assert resolve_two_photon_data_locator(system=system) is locate_two_photon_data
    assert resolve_assembly_geometry_resolver(system=system) is resolve_mesoscope_assembly_geometry
    assert resolve_video_tracking(system=system) is process_mesoscope_video_tracking
    assert resolve_runtime_binding(system=system) == (RUNTIME_SOURCE_ID, parse_runtime)


def test_the_cindra_configuration_resolvers_are_donated_as_a_pair():
    """Verifies that the two-photon and forging pipelines each reach one half of the bundle, so both must resolve
    separately.
    """
    system = AcquisitionSystems.MESOSCOPE_VR

    assert resolve_single_recording_configuration_resolver(system=system) is resolve_single_recording_configuration
    assert resolve_multi_recording_configuration_resolver(system=system) is resolve_multi_recording_configuration


def test_every_parseable_module_declares_the_event_codes_its_parser_reads():
    """Verifies that the extraction stage filters each module by these codes, so the two mappings have to name the same
    modules.
    """
    parsers = resolve_microcontroller_parsers(system=AcquisitionSystems.MESOSCOPE_VR)

    assert set(parsers) == set(resolve_microcontroller_event_codes(system=AcquisitionSystems.MESOSCOPE_VR))
    assert set(parsers) == set(get_module_event_codes())


def test_a_session_resolves_the_modules_its_hardware_state_configured(experiment_session: SessionData):
    """Verifies that the extraction filter is narrowed to these modules, so the resolver has to reach the system's own
    accessor.
    """
    eligible = resolve_eligible_microcontroller_modules(
        system=experiment_session.acquisition_system, session=experiment_session
    )

    assert eligible == set(resolve_microcontroller_parsers(system=experiment_session.acquisition_system))
