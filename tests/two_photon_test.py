"""Contains tests for the Mesoscope-VR genotype-driven cindra configuration resolvers, the two-photon input locator,
and their registry wiring.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path
import dataclasses

from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
import pytest
from sollertia_shared_assets import SessionTypes, AcquisitionSystems, MesoscopeDirectories

from sollertia_forgery.registries import (
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
import sollertia_forgery.mesoscope_vr.two_photon as two_photon_module
from sollertia_forgery.mesoscope_vr.two_photon import (
    _INDICATOR_PARAMETERS,
    _CalciumIndicator,
    locate_two_photon_data,
    _assert_indicator_coverage,
    _resolve_calcium_indicator,
    _build_multi_recording_configuration,
    _build_single_recording_configuration,
    resolve_multi_recording_configuration,
    resolve_single_recording_configuration,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_FIXTURES_DIRECTORY: Path = Path(__file__).parent / "fixtures" / "cindra"
"""The directory holding the reference mesoscope-vr cindra configuration YAMLs the golden regression pins against."""


def _neutralize_single(configuration: dict[str, Any]) -> dict[str, Any]:
    """Clears the deploy-time single-recording fields the two-photon pipeline overrides, so the golden comparison
    isolates the indicator-tuned base.
    """
    configuration["runtime"] = None
    configuration["file_io"]["data_path"] = None
    configuration["file_io"]["output_path"] = None
    return configuration


def _neutralize_multi(configuration: dict[str, Any]) -> dict[str, Any]:
    """Clears the deploy-time multi-recording fields the forging pipeline overrides, so the golden comparison isolates
    the indicator-tuned base.
    """
    configuration["runtime"] = None
    configuration["recording_io"]["recording_directories"] = None
    configuration["recording_io"]["dataset_name"] = None
    return configuration


def _stub_surgery_loader(monkeypatch: pytest.MonkeyPatch, genotype: str) -> None:
    """Replaces the surgery loader so the resolvers read the given genotype without a real surgery YAML."""
    monkeypatch.setattr(
        two_photon_module,
        "SurgeryData",
        SimpleNamespace(
            from_yaml=lambda file_path: SimpleNamespace(subject=SimpleNamespace(genotype=genotype))  # noqa: ARG005
        ),
    )


def _stub_session(tmp_path: Path, session_type: SessionTypes, *, surgery_present: bool = True) -> SimpleNamespace:
    """Returns a stand-in session exposing the surgery metadata path and session type the resolvers read."""
    surgery_metadata_path = tmp_path.joinpath("surgery_metadata.yaml")
    if surgery_present:
        surgery_metadata_path.write_text(data="stub")
    return SimpleNamespace(
        raw_data=SimpleNamespace(surgery_metadata_path=surgery_metadata_path),
        session_type=session_type,
        session_name="test-session",
    )


@pytest.fixture
def stubbed_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., SimpleNamespace]:
    """Returns a builder that stubs the surgery loader and produces a matching stand-in session."""

    def _build(genotype: str, session_type: SessionTypes, *, surgery_present: bool = True) -> SimpleNamespace:
        _stub_surgery_loader(monkeypatch=monkeypatch, genotype=genotype)
        return _stub_session(tmp_path=tmp_path, session_type=session_type, surgery_present=surgery_present)

    return _build


@pytest.mark.parametrize(
    ("genotype", "expected"),
    [
        ("GP5.17", _CalciumIndicator.GCAMP6F),
        ("GP5.17 (hemi)", _CalciumIndicator.GCAMP6F),
        ("  GP5.17  (hemi) ", _CalciumIndicator.GCAMP6F),
        ("gp5.17", _CalciumIndicator.GCAMP6F),
        ("GCaMP8s x CamKIICre", _CalciumIndicator.JGCAMP8S),
        ("gcamp8s x camkiicre", _CalciumIndicator.JGCAMP8S),
    ],
)
def test_resolve_calcium_indicator_recognized(genotype: str, expected: _CalciumIndicator) -> None:
    """Verifies recognized genotypes resolve to their calcium indicator across casing, whitespace, and the qualifier."""
    assert _resolve_calcium_indicator(genotype) == expected


@pytest.mark.parametrize("genotype", ["GCaMP8f x CamKIICre", "GCaMP8m", "GCaMP6f", "wildtype", "", "GP5.18"])
def test_resolve_calcium_indicator_rejects_unknown(genotype: str) -> None:
    """Verifies an unrecognized genotype raises rather than defaulting, so a jGCaMP8f or jGCaMP8m line never maps to
    jGCaMP8s.
    """
    with pytest.raises(ValueError, match="Unable to resolve the calcium indicator"):
        _resolve_calcium_indicator(genotype)


def test_single_recording_genotype_delta() -> None:
    """Verifies the single-recording tau and neuropil coefficient differ between the two indicators."""
    gcamp6f = _build_single_recording_configuration(genotype="GP5.17")
    jgcamp8s = _build_single_recording_configuration(genotype="GCaMP8s x CamKIICre")
    assert gcamp6f.main.tau == pytest.approx(0.4)
    assert jgcamp8s.main.tau == pytest.approx(0.7)
    assert gcamp6f.spike_deconvolution.neuropil_coefficient == pytest.approx(0.7)
    assert jgcamp8s.spike_deconvolution.neuropil_coefficient == pytest.approx(0.8)


def test_multi_recording_genotype_delta() -> None:
    """Verifies the multi-recording probability threshold and neuropil coefficient differ between the two indicators,
    and that the base enables overlapping ROIs for both.
    """
    gcamp6f = _build_multi_recording_configuration(genotype="GP5.17")
    jgcamp8s = _build_multi_recording_configuration(genotype="GCaMP8s x CamKIICre")
    assert gcamp6f.roi_selection.probability_threshold == pytest.approx(0.85)
    assert jgcamp8s.roi_selection.probability_threshold == pytest.approx(0.80)
    assert gcamp6f.spike_deconvolution.neuropil_coefficient == pytest.approx(0.7)
    assert jgcamp8s.spike_deconvolution.neuropil_coefficient == pytest.approx(0.8)
    assert gcamp6f.signal_extraction.allow_overlap
    assert jgcamp8s.signal_extraction.allow_overlap


@pytest.mark.parametrize(
    ("genotype", "fixture_name"),
    [("GP5.17", "CA1_GCaMP6f_SD.yaml"), ("GCaMP8s x CamKIICre", "CA1_GCaMP8s_SD.yaml")],
)
def test_single_recording_matches_reference_yaml(genotype: str, fixture_name: str) -> None:
    """Verifies each built single-recording configuration equals the reference mesoscope-vr YAML field-for-field."""
    built = _neutralize_single(dataclasses.asdict(_build_single_recording_configuration(genotype=genotype)))
    reference = _neutralize_single(
        dataclasses.asdict(SingleRecordingConfiguration.from_yaml(file_path=_FIXTURES_DIRECTORY / fixture_name))
    )
    assert built == reference


@pytest.mark.parametrize(
    ("genotype", "fixture_name"),
    [("GP5.17", "CA1_GCaMP6f_MD.yaml"), ("GCaMP8s x CamKIICre", "CA1_GCaMP8s_MD.yaml")],
)
def test_multi_recording_matches_reference_yaml(genotype: str, fixture_name: str) -> None:
    """Verifies each built multi-recording configuration equals the reference mesoscope-vr YAML field-for-field."""
    built = _neutralize_multi(dataclasses.asdict(_build_multi_recording_configuration(genotype=genotype)))
    reference = _neutralize_multi(
        dataclasses.asdict(MultiRecordingConfiguration.from_yaml(file_path=_FIXTURES_DIRECTORY / fixture_name))
    )
    assert built == reference


def test_resolve_single_recording_configuration_reads_genotype(
    stubbed_session: Callable[..., SimpleNamespace],
) -> None:
    """Verifies the single-recording resolver selects the configuration from the session's genotype."""
    configuration = resolve_single_recording_configuration(
        stubbed_session(genotype="GCaMP8s x CamKIICre", session_type=SessionTypes.MESOSCOPE_EXPERIMENT)
    )
    assert configuration.main.tau == pytest.approx(0.7)


def test_resolve_single_recording_configuration_missing_surgery_errors(
    stubbed_session: Callable[..., SimpleNamespace],
) -> None:
    """Verifies the single-recording resolver raises when the session has no surgery metadata to read the genotype."""
    session = stubbed_session(genotype="GP5.17", session_type=SessionTypes.MESOSCOPE_EXPERIMENT, surgery_present=False)
    with pytest.raises(FileNotFoundError, match="Unable to resolve the cindra configuration"):
        resolve_single_recording_configuration(session)


def test_resolve_multi_recording_configuration_experiment_reads_genotype(
    stubbed_session: Callable[..., SimpleNamespace],
) -> None:
    """Verifies the multi-recording resolver returns a genotype-tuned configuration for an experiment session."""
    configuration = resolve_multi_recording_configuration(
        stubbed_session(genotype="GP5.17", session_type=SessionTypes.MESOSCOPE_EXPERIMENT)
    )
    assert configuration is not None
    assert configuration.roi_selection.probability_threshold == pytest.approx(0.85)


@pytest.mark.parametrize("session_type", [SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING])
def test_resolve_multi_recording_configuration_training_returns_none(
    stubbed_session: Callable[..., SimpleNamespace], session_type: SessionTypes
) -> None:
    """Verifies the multi-recording resolver returns None for a training session, since Mesoscope-VR tracks no cells
    for it.
    """
    assert resolve_multi_recording_configuration(stubbed_session(genotype="GP5.17", session_type=session_type)) is None


def test_locate_two_photon_data_resolves_the_raw_imaging_directory(experiment_session: SessionData) -> None:
    """Verifies the two-photon input locator points at the session's raw mesoscope_data directory."""
    located = locate_two_photon_data(experiment_session)
    assert located.name == "mesoscope_data"
    assert located.name == MesoscopeDirectories.MESOSCOPE_DATA
    assert located.parent == experiment_session.raw_data_path


def test_assert_indicator_coverage_names_uncovered_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the import-time coverage check names every calcium indicator that declares no tuned parameters."""
    monkeypatch.setattr(
        two_photon_module,
        "_INDICATOR_PARAMETERS",
        {_CalciumIndicator.GCAMP6F: _INDICATOR_PARAMETERS[_CalciumIndicator.GCAMP6F]},
    )
    with pytest.raises(RuntimeError, match=r"(?s)members\s+do\s+not:\s+JGCAMP8S"):
        _assert_indicator_coverage()


def test_registry_dispatches_by_member_and_by_value() -> None:
    """Verifies the cindra configuration registry dispatches to the Mesoscope-VR resolvers by member and by value."""
    assert resolve_single_recording_configuration_resolver(AcquisitionSystems.MESOSCOPE_VR) is (
        resolve_single_recording_configuration
    )
    assert resolve_multi_recording_configuration_resolver("mesoscope") is resolve_multi_recording_configuration


def test_registry_rejects_unknown_system() -> None:
    """Verifies resolving a resolver for an unknown acquisition system raises."""
    with pytest.raises(ValueError, match="supported AcquisitionSystems"):
        resolve_single_recording_configuration_resolver("nonexistent-system")
    with pytest.raises(ValueError, match="supported AcquisitionSystems"):
        resolve_multi_recording_configuration_resolver("nonexistent-system")
