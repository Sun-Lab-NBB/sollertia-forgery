"""Tests the standalone microcontroller log processing pipeline."""

from __future__ import annotations

from types import SimpleNamespace
import pickle
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ataraxis_base_utilities import console
from sollertia_shared_assets import AcquisitionSystems, ProcessingTrackers, MesoscopeHardwareState
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker
from ataraxis_communication_interface import (
    CONTROLLER_EXTRACTION_JOB_NAME,
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    ModuleSourceData,
    ModuleExtractionConfig,
    MicroControllerManifest,
    MicroControllerSourceData,
    ControllerExtractionConfig,
)
from ataraxis_communication_interface.communication import SerialProtocols, SerialPrototypes

from sollertia_forgery.registries import (
    _MICROCONTROLLER_PARSER_REGISTRY,
    resolve_microcontroller_parsers,
    resolve_two_photon_data_locator,
    resolve_microcontroller_event_codes,
)
from sollertia_forgery.microcontrollers import (
    PARSE_JOB_NAME,
    pipeline as pipeline_module,
    discover_microcontroller_jobs,
    microcontroller_job_prerequisites,
    run_microcontroller_processing_pipeline,
)
from sollertia_forgery.mesoscope_vr.two_photon import locate_two_photon_data
from sollertia_forgery.mesoscope_vr.microcontrollers import _is_module_eligible

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import ProcessPoolExecutor

    from sollertia_forgery.registries import MicrocontrollerParser

_MODULE_STATE_PROTOCOL: int = SerialProtocols.MODULE_STATE
"""The axci serial protocol code marking a state-only hardware module message."""

_MODULE_DATA_PROTOCOL: int = SerialProtocols.MODULE_DATA
"""The axci serial protocol code marking a data-carrying hardware module message."""

_ONE_UINT16_PROTOTYPE: int = SerialPrototypes.ONE_UINT16
"""The axci payload prototype code of a message carrying a single unsigned 16-bit value."""

_ONE_UINT32_PROTOTYPE: int = SerialPrototypes.ONE_UINT32
"""The axci payload prototype code of a message carrying a single unsigned 32-bit value."""


def _stub_parse_2_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:
    """Writes the module 2_1 domain feather recording the event codes the partition carried."""
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_2_1.feather", compression="uncompressed"
    )


def _stub_parse_4_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:
    """Writes the module 4_1 domain feather recording the event codes the partition carried."""
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_4_1.feather", compression="uncompressed"
    )


def _stub_parse_6_1(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:
    """Writes the module 6_1 domain feather recording the event codes the partition carried."""
    pl.DataFrame({"event_code": sorted(event_partition.keys())}).write_ipc(
        file=output_directory / "module_6_1.feather", compression="uncompressed"
    )


def _stub_parse_failing(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: object) -> None:
    """Raises inside the parse call, standing in for a module parser that fails on its partition."""
    message = "Synthetic parse failure raised inside a worker process."
    raise RuntimeError(message)


_STUB_PARSERS: dict[tuple[int, int], MicrocontrollerParser] = {
    (2, 1): _stub_parse_2_1,
    (4, 1): _stub_parse_4_1,
    (6, 1): _stub_parse_6_1,
}
"""Maps each stubbed module to its parser. Declared below the stubs it references, which the parallel parse path
pickles by reference."""


def _patch_parsers(monkeypatch: pytest.MonkeyPatch, eligible: set[tuple[int, int]]) -> None:
    """Replaces the central parser lookup with one returning only the requested stub parsers.

    The pipeline binds ``resolve_microcontroller_parsers`` into its own namespace via a top-level import, so the
    stub set is injected by patching the name on the pipeline module rather than on the registries hub.
    """
    _patch_parser_map(monkeypatch=monkeypatch, parsers={key: _STUB_PARSERS[key] for key in eligible})


def _patch_parser_map(monkeypatch: pytest.MonkeyPatch, parsers: dict[tuple[int, int], MicrocontrollerParser]) -> None:
    """Replaces the central parser lookup with one returning exactly the supplied parser mapping.

    Args:
        monkeypatch: The patching fixture the replacement is registered on.
        parsers: The module parsers the pipeline resolves for the session's acquisition system.
    """
    monkeypatch.setattr(pipeline_module, "resolve_microcontroller_parsers", lambda system: dict(parsers))  # noqa: ARG005


def _make_session(tmp_path: Path) -> SimpleNamespace:
    """Builds a lightweight stand-in for SessionData exposing only the attributes the pipeline reads."""
    raw_behavior = tmp_path / "raw_data" / "behavior_data"
    raw_behavior.mkdir(parents=True)

    # Job discovery narrows the extraction filter to the modules the session configured, so the stand-in carries a
    # hardware state that marks every module the tests exercise as used.
    hardware_state_path = tmp_path / "raw_data" / "hardware_state.yaml"
    _make_hardware_state().to_yaml(file_path=hardware_state_path)

    return SimpleNamespace(
        session_name="test_session",
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR.value,
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
    manifest.to_yaml(file_path=behavior / MICROCONTROLLER_MANIFEST_FILENAME)

    if stage_archive:
        (behavior / "101_log.npz").touch()


def _write_manifest(session: SimpleNamespace, controllers: dict[int, tuple[tuple[int, int], ...]]) -> None:
    """Writes an acquisition-time microcontroller manifest declaring several controllers.

    Args:
        session: The session stand-in whose raw behavior directory receives the manifest.
        controllers: The declared ``(module_type, module_id)`` pairs, keyed by controller identifier.
    """
    manifest = MicroControllerManifest(
        controllers=[
            MicroControllerSourceData(
                id=controller_id,
                name=f"controller_{controller_id}",
                modules=tuple(
                    ModuleSourceData(
                        module_type=module_type, module_id=module_id, name=f"module_{module_type}_{module_id}"
                    )
                    for module_type, module_id in modules
                ),
            )
            for controller_id, modules in controllers.items()
        ]
    )
    manifest.to_yaml(file_path=session.raw_data.behavior_data_path / MICROCONTROLLER_MANIFEST_FILENAME)


def _module_state_payload(module_type: int, module_id: int, event_code: int) -> bytes:
    """Builds the serial payload of a state-only hardware module message.

    Args:
        module_type: The type code of the module that emitted the message.
        module_id: The instance identifier of the module that emitted the message.
        event_code: The event code the message reports.

    Returns:
        The serialized message payload the archive reader decodes.
    """
    return bytes([_MODULE_STATE_PROTOCOL, module_type, module_id, 1, event_code])


def _module_data_payload(module_type: int, module_id: int, event_code: int, prototype: int, value: np.generic) -> bytes:
    """Builds the serial payload of a data-carrying hardware module message.

    Args:
        module_type: The type code of the module that emitted the message.
        module_id: The instance identifier of the module that emitted the message.
        event_code: The event code the message reports.
        prototype: The axci payload prototype code describing the serialized value.
        value: The numpy scalar the message carries.

    Returns:
        The serialized message payload the archive reader decodes.
    """
    return bytes([_MODULE_DATA_PROTOCOL, module_type, module_id, 1, event_code, prototype]) + value.tobytes()


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


def _read_controller_config(config_path: Path, controller_id: str) -> ControllerExtractionConfig:
    """Reads one controller's entry out of the extraction configuration the pipeline materialized.

    Args:
        config_path: The path to the materialized extraction configuration file.
        controller_id: The identifier of the controller whose entry to read.

    Returns:
        The extraction configuration the file declares for the requested controller.
    """
    configuration = ExtractionConfig.from_yaml(file_path=config_path)
    return next(entry for entry in configuration.controllers if str(entry.controller_id) == controller_id)


def _fake_extract_factory(skip: set[tuple[int, int]] | None = None) -> Callable[..., None]:
    """Returns a stand-in for ``_extract_controller`` that writes raw module feathers and drives the tracker.

    The stand-in reads its controller's modules out of the materialized extraction configuration, exactly as the
    acquisition binding it replaces does, so a test exercises the file the pipeline writes rather than an in-memory
    object the pipeline no longer passes.
    """
    skipped = set(skip or set())

    def fake_extract(
        archive_path: Path,
        output_directory: Path,
        controller_id: str,
        config_path: Path,
        job_id: str,
        tracker: ProcessingTracker,
        *,
        workers: int,
        display_progress: bool,
        executor: ProcessPoolExecutor | None = None,
    ) -> None:
        """Writes a raw module feather for each configured module and records the job on the tracker."""
        tracker.start_job(job_id=job_id)
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        controller_config = _read_controller_config(config_path=config_path, controller_id=controller_id)
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


def _fail_if_called(*args: object, **kwargs: object) -> None:
    """Fails the calling test, standing in for the extraction stage a remote parse job leaves untouched."""
    message = "_extract_controller must not be called for a remote parse job."
    raise AssertionError(message)


def _status(tracker_path: Path, job_name: str, specifier: str) -> ProcessingStatus:
    """Returns the tracker status recorded for the target job."""
    # ProcessingTracker loads its persisted state lazily inside its public methods, so query through them.
    tracker = ProcessingTracker(file_path=tracker_path)
    return tracker.get_job_status(job_id=ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier))


def _count_by_status(tracker_path: Path) -> dict[ProcessingStatus, int]:
    """Returns the number of tracker jobs recorded under each processing status."""
    tracker = ProcessingTracker(file_path=tracker_path)
    return {status: len(tracker.get_jobs_by_status(status=status)) for status in ProcessingStatus}


@pytest.fixture
def disabled_console() -> Iterator[None]:
    """Disables the shared console for the duration of one test and restores it afterwards.

    Yields:
        None. The console reports as disabled for the duration of the block.
    """
    console.disable()
    yield
    console.enable()


@pytest.fixture
def staged_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Builds a session with a staged manifest and log archive, and binds the stubbed loader onto the pipeline."""
    session = _make_session(tmp_path)
    _write_inputs(session)
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    return session


# Central registry and helpers.


def test_resolve_microcontroller_parsers_returns_mesoscope_callables() -> None:
    """Verifies that resolve_microcontroller_parsers returns callable parsers for the Mesoscope-VR system."""
    parsers = resolve_microcontroller_parsers(AcquisitionSystems.MESOSCOPE_VR)
    assert parsers  # Mesoscope-VR registers at least one module parser.
    assert all(callable(parser) for parser in parsers.values())
    # The session stores the acquisition system as a string. Resolution must accept that form too.
    assert resolve_microcontroller_parsers(AcquisitionSystems.MESOSCOPE_VR.value).keys() == parsers.keys()


def test_resolve_microcontroller_parsers_invalid_system_raises() -> None:
    """Verifies that resolve_microcontroller_parsers rejects an unknown acquisition system."""
    with pytest.raises(ValueError, match="Unable to resolve the acquisition system"):
        resolve_microcontroller_parsers("not_a_real_system")


def test_resolve_two_photon_data_locator_returns_mesoscope_callable() -> None:
    """Verifies that resolve_two_photon_data_locator returns the Mesoscope-VR locator."""
    locator = resolve_two_photon_data_locator(AcquisitionSystems.MESOSCOPE_VR)
    assert callable(locator)
    # The session stores the acquisition system as a string. Resolution must accept that form too.
    assert resolve_two_photon_data_locator(AcquisitionSystems.MESOSCOPE_VR.value) is locator


def test_resolve_two_photon_data_locator_invalid_system_raises() -> None:
    """Verifies that resolve_two_photon_data_locator rejects an unknown acquisition system."""
    with pytest.raises(ValueError, match="Unable to resolve the acquisition system"):
        resolve_two_photon_data_locator("not_a_real_system")


def test_locate_two_photon_data_resolves_mesoscope_data_directory(tmp_path: Path) -> None:
    """Verifies that locate_two_photon_data resolves the session's mesoscope_data directory."""
    # The Mesoscope-VR locator places the raw two-photon imaging data in the 'mesoscope_data' directory under the
    # session's raw-data root.
    session = SimpleNamespace(raw_data_path=tmp_path)
    assert locate_two_photon_data(session) == tmp_path / "mesoscope_data"


def test_registered_parsers_are_picklable() -> None:
    """Verifies that every registered microcontroller parser pickles by reference."""
    # Every registered parser is dispatched to worker processes, so each must pickle by reference.
    assert _MICROCONTROLLER_PARSER_REGISTRY
    for key, parser in _MICROCONTROLLER_PARSER_REGISTRY.items():
        assert pickle.loads(pickle.dumps(parser)) is parser, key  # noqa: S301 - the payload is a local parser reference.


def test_event_code_registry_covers_every_parseable_module() -> None:
    """Verifies that every registered parser's module also declares its extraction event codes."""
    # The extraction stage filters each module by its registered codes, so a parseable module that declares none would
    # be dropped from its controller's extraction configuration and its parse job would silently never be discovered.
    assert _MICROCONTROLLER_PARSER_REGISTRY
    for system, module_type, module_id in _MICROCONTROLLER_PARSER_REGISTRY:
        assert (module_type, module_id) in resolve_microcontroller_event_codes(system=system)


def test_screen_module_is_eligible_when_initially_off() -> None:
    """Verifies that a screen module stays eligible when the screens start off."""
    # screens_initially_on records the screens' initial state, not whether the module was used, so a False value must
    # not suppress screen parsing. Only None, the MesoscopeHardwareState unused-module marker, does.
    assert _is_module_eligible(module_key=(7, 1), hardware_state=SimpleNamespace(screens_initially_on=False))
    assert not _is_module_eligible(module_key=(7, 1), hardware_state=SimpleNamespace(screens_initially_on=None))


def test_usage_flag_modules_are_skipped_when_flag_is_unset() -> None:
    """Verifies that a module gated by a usage flag is skipped when the flag is unset."""
    # delivered_gas_puffs and recorded_mesoscope_ttl are genuine usage flags, so False does mark the module unused.
    assert not _is_module_eligible(module_key=(5, 2), hardware_state=SimpleNamespace(delivered_gas_puffs=False))
    assert _is_module_eligible(module_key=(5, 2), hardware_state=SimpleNamespace(delivered_gas_puffs=True))
    assert not _is_module_eligible(module_key=(1, 1), hardware_state=SimpleNamespace(recorded_mesoscope_ttl=False))


def test_resolve_controllers_derives_config_from_manifest(tmp_path: Path) -> None:
    """Verifies that _resolve_controllers derives extraction configurations from the acquisition manifest."""
    session = _make_session(tmp_path)
    # (9, 9) is not registered for any system, so it must be excluded from the derived configuration.
    _write_inputs(session, modules=((2, 1), (9, 9)), stage_archive=False)

    controllers = pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52), (4, 1): (51,)})

    assert set(controllers) == {"101"}
    config = controllers["101"]
    assert config.kernel is None
    assert config.modules == (ModuleExtractionConfig(module_type=2, module_id=1, event_codes=(51, 52)),)


def test_resolve_controllers_rejects_manifest_with_no_extractable_module(tmp_path: Path) -> None:
    """Verifies that _resolve_controllers rejects a manifest whose modules are all unparseable."""
    session = _make_session(tmp_path)
    _write_inputs(session, modules=((9, 9),), stage_archive=False)

    with pytest.raises(ValueError, match="declares a module"):
        pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52)})


def test_resolve_controllers_requires_manifest(tmp_path: Path) -> None:
    """Verifies that _resolve_controllers requires the acquisition-time microcontroller manifest."""
    session = _make_session(tmp_path)

    with pytest.raises(FileNotFoundError, match="microcontroller manifest"):
        pipeline_module._resolve_controllers(session=session, event_codes={(2, 1): (51, 52)})


def test_discover_jobs_filters_by_eligibility_and_presence(tmp_path: Path) -> None:
    """Verifies that _discover_jobs requests only parseable modules whose controller archive is present."""
    (tmp_path / "101_log.npz").touch()  # Controller 101's archive is present and controller 102's is absent.
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
        controllers=controllers, parsers=parsers, log_directory=tmp_path
    )

    # Module (4, 1) has no registered parser, so it never appears. Controller 102 is parseable but has no archive.
    assert set(universe) == {
        (CONTROLLER_EXTRACTION_JOB_NAME, "101"),
        (PARSE_JOB_NAME, "101-2-1"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "102"),
        (PARSE_JOB_NAME, "102-6-1"),
    }
    assert set(requested) == {(CONTROLLER_EXTRACTION_JOB_NAME, "101"), (PARSE_JOB_NAME, "101-2-1")}
    assert set(archives) == {"101"}
    assert parse_specifiers == {"101-2-1": ("101", 2, 1)}


# End-to-end pipeline.


def test_local_pipeline_runs_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a local run performs extraction and parsing and records every job on the unified tracker."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    # The extraction configuration reaches disk before any job runs, since the acquisition binding reads each
    # controller's targets from the file rather than from an in-memory object.
    controller_config = _read_controller_config(
        config_path=output_directory / EXTRACTION_CONFIGURATION_FILENAME, controller_id="101"
    )
    assert [(module.module_type, module.module_id) for module in controller_config.modules] == [(2, 1), (4, 1)]
    # Stage 1 wrote the raw per-module feathers into the microcontroller data directory.
    assert (output_directory / "controller_101_module_2_1.feather").is_file()
    assert (output_directory / "controller_101_module_4_1.feather").is_file()
    # Stage 2 wrote one domain feather per parseable module into the same directory.
    assert (output_directory / "module_2_1.feather").is_file()
    assert (output_directory / "module_4_1.feather").is_file()
    # The unified tracker lives alongside the extracted and parsed feathers.
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert tracker_path.is_file()
    counts = _count_by_status(tracker_path)
    assert sum(counts.values()) == 3  # One extraction job and two parse jobs.
    assert counts[ProcessingStatus.SUCCEEDED] == 3


def test_unregistered_module_produces_no_parse_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a module with no registered parser contributes no parse job."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1)})  # Only module (2, 1) is registered for this system.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    assert not (output_directory / "module_4_1.feather").exists()
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert sum(counts.values()) == 2  # One extraction job and the single parseable parse job.
    assert counts[ProcessingStatus.SUCCEEDED] == 2


def test_missing_feather_completes_parse_job_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a parse job whose module produced no feather completes without writing output."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    # Extraction produces no feather for (4, 1), mimicking a configured module that logged no messages.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory(skip={(4, 1)}))

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    assert not (output_directory / "module_4_1.feather").exists()
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    # The parse job for the data-less module is still resolved to SUCCEEDED with no output.
    assert sum(counts.values()) == 3
    assert counts[ProcessingStatus.SUCCEEDED] == 3


@pytest.mark.usefixtures("staged_session")
def test_no_parseable_controllers_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the pipeline rejects a session whose controllers expose no parseable module."""
    _patch_parsers(monkeypatch=monkeypatch, eligible=set())  # No parser is registered for this system.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(ValueError, match=r"No configured controller with both a present log\s+archive"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)


def test_remote_extraction_runs_single_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a remote extraction job runs its controller and leaves the parse jobs scheduled."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    extraction_job_id = ProcessingTracker.generate_job_id(job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier="101")
    run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=extraction_job_id, workers=1)

    # The extraction ran (raw feathers present) but no parse job did (no domain feathers).
    assert (output_directory / "controller_101_module_2_1.feather").is_file()
    assert not (output_directory / "module_2_1.feather").exists()
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path=tracker_path, job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier="101") == (
        ProcessingStatus.SUCCEEDED
    )
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == (
        ProcessingStatus.SCHEDULED
    )


def test_remote_parse_runs_single_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a remote parse job parses its module without re-running extraction."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    output_directory.mkdir(parents=True)
    # A prior extraction run already produced the raw module feather.
    _make_raw_module_dataframe().write_ipc(
        file=output_directory / "controller_101_module_2_1.feather", compression="uncompressed"
    )
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    # A remote parse job must not trigger extraction.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fail_if_called)

    parse_job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier="101-2-1")
    run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=parse_job_id, workers=1)

    assert (output_directory / "module_2_1.feather").is_file()
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == (
        ProcessingStatus.SUCCEEDED
    )


@pytest.mark.usefixtures("staged_session")
def test_invalid_job_id_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the pipeline rejects an unrecognized job identifier."""
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(ValueError, match=r"must name a job the pipeline could\s+produce"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, job_id="deadbeef", workers=1)


# Job discovery contract.


def test_discover_microcontroller_jobs_reports_universe_and_possible_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that discovery reports every declared job and requests only the ones a staged archive supports."""
    session = _make_session(tmp_path)
    # Controller 101 staged its archive, controller 102 did not, so only 101 contributes possible jobs.
    _write_manifest(session=session, controllers={101: ((2, 1), (4, 1)), 102: ((6, 1),)})
    (session.raw_data.behavior_data_path / "101_log.npz").touch()
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1), (6, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005

    loaded, universe, requested = discover_microcontroller_jobs(session_path=tmp_path)

    assert loaded is session
    assert set(universe) == {
        (CONTROLLER_EXTRACTION_JOB_NAME, "101"),
        (PARSE_JOB_NAME, "101-2-1"),
        (PARSE_JOB_NAME, "101-4-1"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "102"),
        (PARSE_JOB_NAME, "102-6-1"),
    }
    assert set(requested) == {
        (CONTROLLER_EXTRACTION_JOB_NAME, "101"),
        (PARSE_JOB_NAME, "101-2-1"),
        (PARSE_JOB_NAME, "101-4-1"),
    }
    # Discovery leaves the output side of the session untouched.
    assert not session.processed_data.microcontroller_data_path.exists()


def test_microcontroller_job_prerequisites_orders_parses_after_their_extraction() -> None:
    """Verifies that each parse job declares its controller's extraction job as its only prerequisite."""
    universe = [
        (CONTROLLER_EXTRACTION_JOB_NAME, "101"),
        (PARSE_JOB_NAME, "101-2-1"),
        (CONTROLLER_EXTRACTION_JOB_NAME, "102"),
        (PARSE_JOB_NAME, "102-6-1"),
    ]

    prerequisites = microcontroller_job_prerequisites(session=SimpleNamespace(), universe=universe)

    assert prerequisites == {
        (CONTROLLER_EXTRACTION_JOB_NAME, "101"): (),
        (PARSE_JOB_NAME, "101-2-1"): ((CONTROLLER_EXTRACTION_JOB_NAME, "101"),),
        (CONTROLLER_EXTRACTION_JOB_NAME, "102"): (),
        (PARSE_JOB_NAME, "102-6-1"): ((CONTROLLER_EXTRACTION_JOB_NAME, "102"),),
    }


def test_controller_archive_discovery_is_empty_for_an_absent_log_directory(tmp_path: Path) -> None:
    """Verifies that the archive lookup reports no archives when the log directory does not exist."""
    assert pipeline_module._discover_controller_archives(log_directory=tmp_path / "never_acquired") == {}


def test_controller_archive_discovery_reads_the_log_directory_alone(tmp_path: Path) -> None:
    """Verifies that the archive lookup resolves each source from the log directory's own entries.

    One DataLogger writes every archive of a session side by side, so an archive nested under the log directory
    belongs to a different logger and must not be mistaken for this session's.
    """
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "102_log.npz").touch()
    (tmp_path / "101_log.npz").touch()

    assert pipeline_module._discover_controller_archives(log_directory=tmp_path) == {"101": tmp_path / "101_log.npz"}


# Stage helpers with nothing to run.


def test_extraction_stage_writes_nothing_without_archives(tmp_path: Path) -> None:
    """Verifies that the extraction stage writes nothing when no controller staged an archive."""
    output_directory = tmp_path / "microcontroller_data"
    output_directory.mkdir()
    tracker = ProcessingTracker(file_path=output_directory / ProcessingTrackers.MICROCONTROLLER)

    pipeline_module._run_extraction_stage(
        extraction_archives={},
        extraction_output=output_directory,
        config_path=output_directory / EXTRACTION_CONFIGURATION_FILENAME,
        tracker=tracker,
        workers=1,
        executor=None,
        display_progress=False,
    )

    assert list(output_directory.iterdir()) == []


def test_parse_stage_writes_nothing_without_specifiers(tmp_path: Path) -> None:
    """Verifies that the parse stage writes nothing when no parse specifier is dispatched."""
    output_directory = tmp_path / "microcontroller_data"
    output_directory.mkdir()
    tracker = ProcessingTracker(file_path=output_directory / ProcessingTrackers.MICROCONTROLLER)

    pipeline_module._run_parse_stage(
        parse_specifiers={},
        parsers={},
        session=SimpleNamespace(),
        extraction_output=output_directory,
        parse_output=output_directory,
        tracker=tracker,
        executor=None,
        display_progress=False,
    )

    assert list(output_directory.iterdir()) == []


def test_parse_stage_stops_when_no_module_produced_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that the parse stage completes every job when extraction produced no module feather."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    # The controller logged no messages at all, so the extraction writes no raw feather for either module.
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory(skip={(2, 1), (4, 1)}))

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    assert list(output_directory.glob("*.feather")) == []
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert counts[ProcessingStatus.SUCCEEDED] == 3


# Real extraction, progress reporting, and the disabled-console path.


def test_local_pipeline_extracts_a_real_log_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that a local run extracts a real log archive and filters each module by its registered codes."""
    session = _make_session(tmp_path)
    _write_inputs(session, stage_archive=False)
    write_log_archive(
        path=session.raw_data.behavior_data_path / "101_log.npz",
        source_id=101,
        messages=[
            (10, _module_data_payload(2, 1, 51, _ONE_UINT32_PROTOTYPE, np.uint32(7))),
            (20, _module_state_payload(2, 1, 52)),
            (30, _module_data_payload(4, 1, 51, _ONE_UINT16_PROTOTYPE, np.uint16(700))),
            # Event code 60 is outside the modules' registered filters, so extraction drops it.
            (40, _module_state_payload(4, 1, 60)),
        ],
    )
    output_directory = session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    encoder_feather = pl.read_ipc(source=output_directory / "controller_101_module_2_1.feather", memory_map=False)
    assert encoder_feather["event"].to_list() == [51, 52]
    assert encoder_feather["dtype"].to_list() == ["uint32", None]
    assert np.frombuffer(encoder_feather["data"][0], dtype="uint32").tolist() == [7]

    lick_feather = pl.read_ipc(source=output_directory / "controller_101_module_4_1.feather", memory_map=False)
    # The lick module registers event 51 alone, so the out-of-filter event 60 never reaches the raw feather.
    assert lick_feather["event"].to_list() == [51]

    # The stub parsers see exactly the extracted event codes.
    assert pl.read_ipc(source=output_directory / "module_2_1.feather")["event_code"].to_list() == [51, 52]
    assert pl.read_ipc(source=output_directory / "module_4_1.feather")["event_code"].to_list() == [51]
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert counts[ProcessingStatus.SUCCEEDED] == 3


def test_local_pipeline_reports_progress_for_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a local run drives both stages with progress reporting enabled."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1, display_progress=True)

    assert (output_directory / "module_2_1.feather").is_file()
    assert (output_directory / "module_4_1.feather").is_file()
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert counts[ProcessingStatus.SUCCEEDED] == 3


@pytest.mark.usefixtures("disabled_console")
def test_local_pipeline_leaves_a_disabled_console_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a local run restores the disabled console it started from."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=1)

    # The extraction stage silences the console around each controller and restores only the state it found.
    assert not console.enabled
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert counts[ProcessingStatus.SUCCEEDED] == 3


# Parallel parse stage.


def test_parallel_parse_stage_runs_every_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that the parallel parse stage runs every dispatched module parser."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    run_microcontroller_processing_pipeline(session_path=tmp_path, workers=2, display_progress=True)

    assert pl.read_ipc(source=output_directory / "module_2_1.feather")["event_code"].to_list() == [51, 52]
    assert pl.read_ipc(source=output_directory / "module_4_1.feather")["event_code"].to_list() == [51, 52]
    counts = _count_by_status(output_directory / ProcessingTrackers.MICROCONTROLLER)
    assert counts[ProcessingStatus.SUCCEEDED] == 3


def test_parallel_parse_stage_records_a_failing_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that the parallel parse stage records the failing module and completes the healthy one."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parser_map(monkeypatch=monkeypatch, parsers={(2, 1): _stub_parse_2_1, (4, 1): _stub_parse_failing})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(RuntimeError, match="Synthetic parse failure"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, workers=2)

    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    # The healthy module still completed, so the tracker stays accurate for every dispatched job.
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == (
        ProcessingStatus.SUCCEEDED
    )
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-4-1") == ProcessingStatus.FAILED
    assert (output_directory / "module_2_1.feather").is_file()
    assert not (output_directory / "module_4_1.feather").exists()


# Remote dispatch.


def test_remote_extraction_requires_the_controller_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a remote extraction job rejects a controller whose log archive is absent."""
    session = _make_session(tmp_path)
    # Controller 102 is registered in the manifest but never staged its archive.
    _write_manifest(session=session, controllers={101: ((2, 1),), 102: ((4, 1),)})
    (session.raw_data.behavior_data_path / "101_log.npz").touch()
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "SessionData", SimpleNamespace(load=lambda session_path: session))  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fail_if_called)

    job_id = ProcessingTracker.generate_job_id(job_name=CONTROLLER_EXTRACTION_JOB_NAME, specifier="102")
    with pytest.raises(FileNotFoundError, match=r"No log archive\s+'102_log.npz' was found"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=job_id, workers=1)


def test_remote_parse_completes_without_extracted_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a remote parse job succeeds when its module produced no extracted feather."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parsers(monkeypatch=monkeypatch, eligible={(2, 1), (4, 1)})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fail_if_called)

    job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier="101-2-1")
    run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=job_id, workers=1)

    assert not (output_directory / "module_2_1.feather").exists()
    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == (
        ProcessingStatus.SUCCEEDED
    )


def test_remote_parse_records_a_failing_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that a remote parse job records its module as failed when the parser raises."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    output_directory.mkdir(parents=True)
    _make_raw_module_dataframe().write_ipc(
        file=output_directory / "controller_101_module_2_1.feather", compression="uncompressed"
    )
    _patch_parser_map(monkeypatch=monkeypatch, parsers={(2, 1): _stub_parse_failing, (4, 1): _stub_parse_4_1})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fail_if_called)

    job_id = ProcessingTracker.generate_job_id(job_name=PARSE_JOB_NAME, specifier="101-2-1")
    with pytest.raises(RuntimeError, match="Synthetic parse failure"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, job_id=job_id, workers=1)

    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == ProcessingStatus.FAILED


def test_parallel_parse_stage_keeps_the_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, staged_session: SimpleNamespace
) -> None:
    """Verifies that the parallel parse stage records every failure before it re-raises the first one."""
    output_directory = staged_session.processed_data.microcontroller_data_path
    _patch_parser_map(monkeypatch=monkeypatch, parsers={(2, 1): _stub_parse_failing, (4, 1): _stub_parse_failing})
    monkeypatch.setattr(pipeline_module, "_extract_controller", _fake_extract_factory())

    with pytest.raises(RuntimeError, match="Synthetic parse failure"):
        run_microcontroller_processing_pipeline(session_path=tmp_path, workers=2)

    tracker_path = output_directory / ProcessingTrackers.MICROCONTROLLER
    # Every dispatched future is allowed to resolve, so both failures are recorded before the first one is re-raised.
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-2-1") == ProcessingStatus.FAILED
    assert _status(tracker_path=tracker_path, job_name=PARSE_JOB_NAME, specifier="101-4-1") == ProcessingStatus.FAILED
    assert _count_by_status(tracker_path)[ProcessingStatus.FAILED] == 2
