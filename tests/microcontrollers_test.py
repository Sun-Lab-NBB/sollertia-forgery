"""Tests for the standalone microcontroller log processing pipeline.

The pipeline is exercised through stub parser functions injected in place of the central
``resolve_microcontroller_parsers`` lookup. A real microcontroller manifest fixture lets ``_resolve_controllers``
derive the extraction configurations for real, and a faked Stage 1 extraction removes the need for a real ``.npz``
archive. Stage 2 parsing, the unified tracker, job discovery, and local/remote dispatch all run as production code.
"""

from __future__ import annotations

from types import SimpleNamespace
import pickle
from pathlib import Path

import polars as pl
import pytest
from sollertia_shared_assets import AcquisitionSystems, ProcessingTrackers, MesoscopeHardwareState
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker
from ataraxis_communication_interface.microcontroller import (
    EXTRACTION_JOB_NAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ModuleSourceData,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
)

from sollertia_forgery.registries import (
    _MICROCONTROLLER_PARSER_REGISTRY,
    resolve_microcontroller_parsers,
    resolve_two_photon_data_locator,
    resolve_microcontroller_event_codes,
)
from sollertia_forgery.microcontrollers import (
    PARSE_JOB_NAME,
    pipeline as pipeline_module,
    run_microcontroller_processing_pipeline,
)
from sollertia_forgery.mesoscope_vr.two_photon import locate_two_photon_data
from sollertia_forgery.mesoscope_vr.microcontrollers import _is_module_eligible

# Module-level stub parsers so the parallel parse path can pickle them by reference. Each writes a trivial domain
# feather, named for its module, recording which event codes the partition carried. Their signature matches the
# registered parsers: parse(event_partition, output_directory, session) -> None.


def _stub_parse_2_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:  # noqa: ARG001
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_2_1.feather", compression="uncompressed"
    )


def _stub_parse_4_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:  # noqa: ARG001
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_4_1.feather", compression="uncompressed"
    )


def _stub_parse_6_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:  # noqa: ARG001
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_6_1.feather", compression="uncompressed"
    )


_STUB_PARSERS: dict[tuple[int, int], object] = {
    (2, 1): _stub_parse_2_1,
    (4, 1): _stub_parse_4_1,
    (6, 1): _stub_parse_6_1,
}


def _patch_parsers(monkeypatch: pytest.MonkeyPatch, eligible: set[tuple[int, int]]) -> None:
    """Replaces the central parser lookup with one returning only the requested stub parsers.

    The pipeline binds ``resolve_microcontroller_parsers`` into its own namespace via a top-level import, so the
    stub set is injected by patching the name on the pipeline module rather than on the registries hub.
    """
    parsers = {key: _STUB_PARSERS[key] for key in eligible}
    monkeypatch.setattr(pipeline_module, "resolve_microcontroller_parsers", lambda system: dict(parsers))  # noqa: ARG005


def _make_session(
    tmp_path: Path, *, acquisition_system: str = AcquisitionSystems.MESOSCOPE_VR.value
) -> SimpleNamespace:
    """Builds a lightweight stand-in for SessionData exposing only the attributes the pipeline reads."""
    raw_behavior = tmp_path / "raw_data" / "behavior_data"
    raw_behavior.mkdir(parents=True)

    # Job discovery narrows the extraction filter to the modules the session configured, so the stand-in carries a
    # hardware state that marks every module the tests exercise as used.
    hardware_state_path = tmp_path / "raw_data" / "hardware_state.yaml"
    _make_hardware_state().to_yaml(file_path=hardware_state_path)

    return SimpleNamespace(
        session_name="test_session",
        acquisition_system=acquisition_system,
        raw_data=SimpleNamespace(behavior_data_path=raw_behavior, hardware_state_path=hardware_state_path),
        processed_data=SimpleNamespace(
            microcontroller_data_path=tmp_path / "processed_data" / "microcontroller_data",
        ),
    )


def _make_hardware_state() -> MesoscopeHardwareState:
    """Builds a hardware state marking every module the microcontroller tests exercise as configured and used."""
    return MesoscopeHardwareState(
        cm_per_pulse=0.0057652,
        maximum_brake_strength=11.30234233,
        minimum_brake_strength=0.42383811,
        lick_threshold=600,
        valve_scale_coefficient=2.0e-07,
        valve_nonlinearity_exponent=1.61941797,
        torque_per_adc_unit=0.00506377,
        screens_initially_on=False,
        recorded_mesoscope_ttl=True,
        delivered_gas_puffs=True,
    )


def _write_inputs(
    session: SimpleNamespace, *, modules: tuple[tuple[int, int], ...] = ((2, 1), (4, 1)), stage_archive: bool = True
) -> None:
    """Writes the acquisition-time microcontroller manifest and (optionally) a log archive placeholder."""
    behavior = session.raw_data.behavior_data_path

    manifest = MicroControllerManifest(
        controllers=[
            MicroControllerSourceData(
                id=101,
                name="actor",
                modules=tuple(
                    ModuleSourceData(
                        module_type=module_type, module_id=module_id, name=f"module_{module_type}_{module_id}"
                    )
                    for module_type, module_id in modules
                ),
            )
        ]
    )
    manifest.save(file_path=behavior / MICROCONTROLLER_MANIFEST_FILENAME)

    if stage_archive:
        (behavior / "101_log.npz").touch()


def _make_raw_module_dataframe() -> pl.DataFrame:
    """Builds a raw per-module feather in the five-column schema produced by the acquisition library."""
    return pl.DataFrame(
        {
            "timestamp_us": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "command": pl.Series([1, 1, 1], dtype=pl.UInt8),
            "event": pl.Series([51, 52, 51], dtype=pl.UInt8),
            "dtype": pl.Series([None, None, None], dtype=pl.String),
            "data": pl.Series([None, None, None], dtype=pl.Binary),
        }
    )


def _fake_extract_factory(skip: set[tuple[int, int]] | None = None):
    """Returns a stand-in for ``_extract_controller`` that writes raw module feathers and drives the tracker."""
    skipped = set(skip or set())

    def fake_extract(
        archive_path,  # noqa: ANN001, ARG001
        output_directory,  # noqa: ANN001
        controller_id,  # noqa: ANN001
        controller_config,  # noqa: ANN001
        job_id,  # noqa: ANN001
        tracker,  # noqa: ANN001
        *,
        workers,  # noqa: ANN001, ARG001
        display_progress,  # noqa: ANN001, ARG001
        executor=None,  # noqa: ANN001, ARG001
    ) -> None:
        tracker.start_job(job_id=job_id)
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        for module in controller_config.modules:
            if (module.module_type, module.module_id) in skipped:
                continue
            feather = (
                Path(output_directory)
                / f"controller_{controller_id}_module_{module.module_type}_{module.module_id}.feather"
            )
            _make_raw_module_dataframe().write_ipc(file=feather, compression="uncompressed")
        tracker.complete_job(job_id=job_id)

    return fake_extract


def _fail_if_called(*args, **kwargs) -> None:  # noqa: ANN002, ANN003, ARG001
    raise AssertionError("_extract_controller must not be called for a remote parse job.")


def _status(tracker_path: Path, job_name: str, specifier: str) -> ProcessingStatus:
    # ProcessingTracker loads its persisted state lazily inside its public methods, so query through them.
    tracker = ProcessingTracker(file_path=tracker_path)
    return tracker.get_job_status(job_id=ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier))


def _count_by_status(tracker_path: Path) -> dict[ProcessingStatus, int]:
    tracker = ProcessingTracker(file_path=tracker_path)
    return {status: len(tracker.get_jobs_by_status(status)) for status in ProcessingStatus}


# ----------------------------------------------------------------------------------------------------------------------
# Central registry and helpers
# ----------------------------------------------------------------------------------------------------------------------


def test_resolve_microcontroller_parsers_returns_mesoscope_callables() -> None:
    parsers = resolve_microcontroller_parsers(AcquisitionSystems.MESOSCOPE_VR)
    assert parsers  # Mesoscope-VR registers at least one module parser.
    assert all(callable(parser) for parser in parsers.values())
    # The session stores the acquisition system as a string. Resolution must accept that form too.
    assert resolve_microcontroller_parsers(AcquisitionSystems.MESOSCOPE_VR.value).keys() == parsers.keys()


def test_resolve_microcontroller_parsers_invalid_system_raises() -> None:
    with pytest.raises(ValueError):
        resolve_microcontroller_parsers("not_a_real_system")


def test_resolve_two_photon_data_locator_returns_mesoscope_callable() -> None:
    locator = resolve_two_photon_data_locator(AcquisitionSystems.MESOSCOPE_VR)
    assert callable(locator)
    # The session stores the acquisition system as a string. Resolution must accept that form too.
    assert resolve_two_photon_data_locator(AcquisitionSystems.MESOSCOPE_VR.value) is locator


def test_resolve_two_photon_data_locator_invalid_system_raises() -> None:
    with pytest.raises(ValueError):
        resolve_two_photon_data_locator("not_a_real_system")


def test_locate_two_photon_data_resolves_mesoscope_data_directory(tmp_path: Path) -> None:
    # The Mesoscope-VR locator places the raw two-photon imaging data in the 'mesoscope_data' directory under the
    # session's raw-data root.
    session = SimpleNamespace(raw_data_path=tmp_path)
    assert locate_two_photon_data(session) == tmp_path / "mesoscope_data"


def test_registered_parsers_are_picklable() -> None:
    # Every registered parser is dispatched to worker processes, so each must pickle by reference.
    for key, parser in _MICROCONTROLLER_PARSER_REGISTRY.items():
        assert pickle.loads(pickle.dumps(parser)) is parser, key


def test_event_code_registry_covers_every_parseable_module() -> None:
    # The extraction stage filters each module by its registered codes, so a parseable module that declares none would
    # be dropped from its controller's extraction configuration and its parse job would silently never be discovered.
    for system, module_type, module_id in _MICROCONTROLLER_PARSER_REGISTRY:
        assert (module_type, module_id) in resolve_microcontroller_event_codes(system=system)


def test_screen_module_is_eligible_when_initially_off() -> None:
    # screens_initially_on records the screens' initial state, not whether the module was used, so a False value must
    # not suppress screen parsing. Only None, the MesoscopeHardwareState unused-module marker, does.
    assert _is_module_eligible(module_type=7, module_id=1, hardware_state=SimpleNamespace(screens_initially_on=False))
    assert not _is_module_eligible(
        module_type=7, module_id=1, hardware_state=SimpleNamespace(screens_initially_on=None)
    )


def test_usage_flag_modules_are_skipped_when_flag_is_unset() -> None:
    # delivered_gas_puffs and recorded_mesoscope_ttl are genuine usage flags, so False does mark the module unused.
    assert not _is_module_eligible(
        module_type=5, module_id=2, hardware_state=SimpleNamespace(delivered_gas_puffs=False)
    )
    assert _is_module_eligible(module_type=5, module_id=2, hardware_state=SimpleNamespace(delivered_gas_puffs=True))
    assert not _is_module_eligible(
        module_type=1, module_id=1, hardware_state=SimpleNamespace(recorded_mesoscope_ttl=False)
    )


def test_resolve_controllers_derives_config_from_manifest(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    # (9, 9) is not registered for any system, so it must be excluded from the derived configuration.
    _write_inputs(session, modules=((2, 1), (9, 9)), stage_archive=False)

    controllers = pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52), (4, 1): (51,)})

    assert set(controllers) == {"101"}
    config = controllers["101"]
    assert config.kernel is None
    assert config.modules == (ModuleExtractionConfig(module_type=2, module_id=1, event_codes=(51, 52)),)


def test_resolve_controllers_rejects_manifest_with_no_extractable_module(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session, modules=((9, 9),), stage_archive=False)

    with pytest.raises(ValueError, match="declares a module"):
        pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52)})


def test_resolve_controllers_requires_manifest(tmp_path: Path) -> None:
    session = _make_session(tmp_path)

    with pytest.raises(FileNotFoundError, match="microcontroller manifest"):
        pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52)})


def test_discover_jobs_filters_by_eligibility_and_presence(tmp_path: Path) -> None:
    (tmp_path / "101_log.npz").touch()  # controller 101 archive present, controller 102 absent
    controllers = {
        "101": ControllerExtractionConfig(
            controller_id=101,
            modules=(
                ModuleExtractionConfig(module_type=2, module_id=1, event_codes=(51,)),
                ModuleExtractionConfig(module_type=4, module_id=1, event_codes=(51,)),
            ),
            kernel=None,
        ),
        "102": ControllerExtractionConfig(
            controller_id=102,
            modules=(ModuleExtractionConfig(module_type=6, module_id=1, event_codes=(51,)),),
            kernel=None,
        ),
    }
    # Module (4, 1) is not registered for the system, so it is never parseable.
    parsers = {(2, 1): _stub_parse_2_1, (6, 1): _stub_parse_6_1}

    universe, requested, archives, parse_specifiers = pipeline_module._discover_jobs(
        controllers=controllers, parsers=parsers, log_directory=tmp_path, extraction_job_name=EXTRACTION_JOB_NAME
    )

    # Module (4, 1) has no registered parser, so it never appears. Controller 102 is parseable but has no archive.
    assert set(universe) == {
        (EXTRACTION_JOB_NAME, "101"),
        (PARSE_JOB_NAME, "101-2-1"),
        (EXTRACTION_JOB_NAME, "102"),
        (PARSE_JOB_NAME, "102-6-1"),
    }
    assert set(requested) == {(EXTRACTION_JOB_NAME, "101"), (PARSE_JOB_NAME, "101-2-1")}
    assert set(archives) == {"101"}
    assert parse_specifiers == {"101-2-1": ("101", 2, 1)}


# ----------------------------------------------------------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------------------------------------------------------


def test_local_pipeline_runs_both_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    output_directory = session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch, {(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    microcontroller_data = session.processed_data.microcontroller_data_path
    # Stage 1 wrote the raw per-module feathers into microcontroller_data.
    assert (microcontroller_data / "controller_101_module_2_1.feather").is_file()
    assert (microcontroller_data / "controller_101_module_4_1.feather").is_file()
    # Stage 2 wrote one domain feather per parseable module into microcontroller_data.
    assert (output_directory / "module_2_1.feather").is_file()
    assert (output_directory / "module_4_1.feather").is_file()
    # The unified tracker lives in microcontroller_data alongside the extracted and parsed feathers.
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert tracker_path.is_file()
    counts = _count_by_status(tracker_path)
    assert sum(counts.values()) == 3  # one extraction job + two parse jobs
    assert counts[ProcessingStatus.SUCCEEDED] == 3


def test_unregistered_module_produces_no_parse_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    output_directory = session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch, {(2, 1)})  # only (2, 1) is registered for this system
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    assert not (output_directory / "module_4_1.feather").exists()
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert sum(counts.values()) == 2  # extraction + the single parseable parse job
    assert counts[ProcessingStatus.SUCCEEDED] == 2


def test_missing_feather_completes_parse_job_without_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    output_directory = session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch, {(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    # Extraction produces no feather for (4, 1), mimicking a configured module that logged no messages.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory(skip={(4, 1)}))

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    assert not (output_directory / "module_4_1.feather").exists()
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    # The parse job for the data-less module is still resolved to SUCCEEDED with no output.
    assert sum(counts.values()) == 3
    assert counts[ProcessingStatus.SUCCEEDED] == 3


def test_no_parseable_controllers_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    _patch_parsers(monkeypatch, set())  # nothing registered
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(ValueError):
        run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)


def test_remote_extraction_runs_single_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    output_directory = session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch, {(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    extraction_job_id = ProcessingTracker.generate_job_id(job_name=EXTRACTION_JOB_NAME, specifier="101")
    run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=extraction_job_id, workers=1)

    # The extraction ran (raw feathers present) but no parse job did (no domain feathers).
    assert (session.processed_data.microcontroller_data_path / "controller_101_module_2_1.feather").is_file()
    assert not (output_directory / "module_2_1.feather").exists()
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path, EXTRACTION_JOB_NAME, "101") == ProcessingStatus.SUCCEEDED
    assert _status(tracker_path, PARSE_JOB_NAME, "101-2-1") == ProcessingStatus.SCHEDULED


def test_remote_parse_runs_single_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    output_directory = session.processed_data.microcontroller_data_path
    microcontroller_data = session.processed_data.microcontroller_data_path
    microcontroller_data.mkdir(parents=True)
    # A prior extraction run already produced the raw module feather.
    _make_raw_module_dataframe().write_ipc(
        file=microcontroller_data / "controller_101_module_2_1.feather", compression="uncompressed"
    )
    _patch_parsers(monkeypatch, {(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    # A remote parse job must not trigger extraction.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fail_if_called)

    parse_job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier="101-2-1")
    run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=parse_job_id, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path, PARSE_JOB_NAME, "101-2-1") == ProcessingStatus.SUCCEEDED


def test_invalid_job_id_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _make_session(tmp_path)
    _write_inputs(session)
    _patch_parsers(monkeypatch, {(2, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(ValueError):
        run_microcontroller_processing_pipeline(session_path=tmp_path, job_id="deadbeef", workers=1)
