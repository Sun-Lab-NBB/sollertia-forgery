"""Contains tests for the system-agnostic two-photon pipeline, covering priming, job resolution, and both dispatch
runtimes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from cindra import (
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    SingleRecordingJobNames,
    SingleRecordingConfiguration,
)
import pytest
from sollertia_shared_assets import (
    SubjectData,
    SurgeryData,
    ProcedureData,
    AcquisitionSystems,
    MesoscopeDirectories,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.two_photon import (
    discover_two_photon_jobs,
    prime_two_photon_recording,
    two_photon_job_prerequisites,
    run_two_photon_processing_pipeline,
)
from sollertia_forgery.shared_assets import SESSION_PIPELINES, ProcessingPipelines
import sollertia_forgery.two_photon.pipeline as two_photon_pipeline
from sollertia_forgery.orchestration.dispatch import resolve_dispatch

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_PLANE_COUNT: int = 2
"""The number of physical imaging planes the synthetic acquisition parameters declare, which is also the number of
virtual planes cindra resolves for this single-ROI recording."""

_GENOTYPE: str = "GP5.17"
"""The genotype the synthetic surgery metadata records, which the Mesoscope-VR resolver maps to a GCaMP6f
configuration."""

_CONFIGURATION_FILENAME: str = SINGLE_RECORDING_CONFIGURATION_FILENAME
"""The name the session's shared cindra configuration is materialized under, inside its cindra directory. cindra owns
the name, and the pipeline writes the file where cindra's own priming step would."""

_STUB_SESSION_NAME: str = "2024_11_04"
"""The directory name every stand-in session is built under, which is also the name the pipeline reads from it."""


def _session_path(session: SessionData) -> Path:
    """Resolves the root session directory every pipeline entry point takes as its argument."""
    return session.raw_data_path.parent


def _write_surgery_metadata(session: SessionData) -> None:
    """Writes the surgery metadata the Mesoscope-VR configuration resolver reads the animal's genotype from."""
    SurgeryData(
        subject=SubjectData(
            id=int(session.animal_id),
            ear_punch="left",
            sex="M",
            genotype=_GENOTYPE,
            date_of_birth_us=1_600_000_000_000_000,
            weight_g=25.0,
            cage=1,
            location_housed="vivarium",
            status="alive",
        ),
        procedure=ProcedureData(
            surgery_start_us=1_600_000_000_000_000,
            surgery_end_us=1_600_000_003_600_000,
            surgeon="tester",
            protocol="synthetic",
            surgery_notes="A synthetic surgery.",
            post_op_notes="A synthetic recovery.",
        ),
        drugs=[],
        implants=[],
        injections=[],
    ).to_yaml(file_path=session.raw_data.surgery_metadata_path)


def _write_acquisition_parameters(session: SessionData, *, plane_number: int = _PLANE_COUNT) -> Path:
    """Writes the raw cindra acquisition parameters file that declares the recording's virtual plane count, and
    returns the raw two-photon imaging directory holding it.
    """
    imaging_directory = session.raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA)
    imaging_directory.mkdir(parents=True, exist_ok=True)
    imaging_directory.joinpath("cindra_parameters.json").write_text(
        data=json.dumps({"frame_rate": 10.0, "plane_number": plane_number, "channel_number": 1}), encoding="utf-8"
    )
    return imaging_directory


@pytest.fixture
def imaging_session(experiment_session: SessionData) -> SessionData:
    """Builds an acquired experiment session carrying the surgery metadata and the raw imaging data cindra resolves a
    recording from.
    """
    _write_surgery_metadata(session=experiment_session)
    _write_acquisition_parameters(session=experiment_session)
    return experiment_session


@pytest.fixture
def primed_session(imaging_session: SessionData) -> SessionData:
    """Primes the imaging session, which is the single-threaded bootstrap step every dispatched job reads."""
    prime_two_photon_recording(session_path=_session_path(imaging_session))
    return imaging_session


@pytest.fixture
def dispatched_jobs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces cindra's single-recording job entry point with a recorder that collects every dispatch it receives, in
    dispatch order.
    """
    recorded: list[dict[str, Any]] = []

    def _record(
        configuration_path: Path,
        job_name: SingleRecordingJobNames,
        specifier: str,
        job_id: str,
        tracker: ProcessingTracker,
        *,
        workers: int | None = None,
    ) -> None:
        recorded.append(
            {
                "configuration_path": configuration_path,
                "job_name": job_name,
                "specifier": specifier,
                "job_id": job_id,
                "tracker_path": tracker.file_path,
                "workers": workers,
            }
        )

    monkeypatch.setattr(two_photon_pipeline, "execute_single_recording_job", _record)
    return recorded


def _dispatched_pairs(dispatched: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Renders the recorded dispatches as the ``(job_name, specifier)`` pairs they ran, in dispatch order."""
    return [(str(record["job_name"]), record["specifier"]) for record in dispatched]


def _stub_session(session_path: Path) -> SimpleNamespace:
    """Builds a stand-in session exposing the attributes the pipeline reads, under the given session directory."""
    cindra_directory = session_path.joinpath("processed_data", "cindra")
    cindra_directory.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        session_name=session_path.name,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        raw_data_path=session_path.joinpath("raw_data"),
        processed_data_path=session_path.joinpath("processed_data"),
        processed_data=SimpleNamespace(cindra_data_path=cindra_directory),
    )


def _patch_session_loader(monkeypatch: pytest.MonkeyPatch, session: SimpleNamespace) -> None:
    """Replaces the session loader so the pipeline resolves the stand-in session."""
    monkeypatch.setattr(
        two_photon_pipeline.SessionData,
        "load",
        classmethod(lambda _cls, session_path: session),  # noqa: ARG005
    )


def _forbid_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test if anything attempts to materialize the configuration or the bootstrap."""

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        """Raises so that any attempt to materialize the bootstrap fails the test."""
        message = "The call materialized the bootstrap, which the priming split forbids."
        raise AssertionError(message)

    monkeypatch.setattr(two_photon_pipeline, "_resolve_configuration", _refuse)
    monkeypatch.setattr(two_photon_pipeline, "prime_recording", _refuse)
    monkeypatch.setattr(two_photon_pipeline, "resolve_single_recording_contexts", _refuse)


def _expected_universe(plane_count: int = _PLANE_COUNT) -> list[tuple[str, str]]:
    """Builds the ordered ``(job_name, specifier)`` universe cindra resolves for the given virtual plane count."""
    return [
        (str(SingleRecordingJobNames.BINARIZE), ""),
        *[(str(SingleRecordingJobNames.REGISTER), f"plane_{plane}") for plane in range(plane_count)],
        *[(str(SingleRecordingJobNames.PROCESS), f"plane_{plane}") for plane in range(plane_count)],
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]


@pytest.fixture
def stubbed_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., SimpleNamespace]:
    """Returns a builder that installs a stand-in session reporting the requested primed plane count."""

    def _build(plane_count: int | None) -> SimpleNamespace:
        session = _stub_session(session_path=tmp_path.joinpath(_STUB_SESSION_NAME))
        _patch_session_loader(monkeypatch=monkeypatch, session=session)
        monkeypatch.setattr(
            two_photon_pipeline,
            "_resolve_data_path",
            lambda session: session.raw_data_path,
        )
        # Discovery reads the recording's shape through cindra's own universe resolver, so the stand-in reports the
        # requested plane count there. A None plane count stands for a recording carrying no acquisition parameters,
        # which the resolver reports through its resolved flag rather than by raising.
        monkeypatch.setattr(
            two_photon_pipeline,
            "resolve_single_recording_job_universe",
            lambda output_root, data_path: SimpleNamespace(  # noqa: ARG005
                resolved=plane_count is not None,
                universe=tuple(_expected_universe(plane_count=plane_count or 0)),
            ),
        )
        # Priming still asks whether the bootstrap this session carries is complete, which is the guard that keeps a
        # second preparation pass from rewriting it.
        monkeypatch.setattr(
            two_photon_pipeline,
            "_resolve_primed_plane_count",
            lambda session: plane_count,  # noqa: ARG005
        )
        return session

    return _build


# Priming and job resolution


def test_priming_materializes_the_configuration_and_every_plane_bootstrap(imaging_session: SessionData) -> None:
    """Verifies priming writes the shared configuration and one runtime file per virtual plane, in a single thread."""
    prime_two_photon_recording(session_path=_session_path(imaging_session))

    cindra_directory = imaging_session.processed_data.cindra_data_path
    configuration_path = cindra_directory.joinpath(_CONFIGURATION_FILENAME)
    configuration = SingleRecordingConfiguration.from_yaml(file_path=configuration_path)

    # The resolver's genotype-tuned parameters stand, while the pipeline overrides only the session-bound locations.
    assert configuration.main.tau == pytest.approx(0.4)
    assert configuration.file_io.data_path == imaging_session.raw_data_path.joinpath(
        MesoscopeDirectories.MESOSCOPE_DATA
    )
    assert configuration.file_io.output_path == imaging_session.processed_data_path
    assert not configuration.runtime.display_progress_bars
    primed_planes = [
        plane
        for plane in range(_PLANE_COUNT)
        if cindra_directory.joinpath(f"plane_{plane}", "runtime_data.yaml").is_file()
    ]
    assert primed_planes == list(range(_PLANE_COUNT))


def test_priming_an_already_primed_recording_leaves_the_bootstrap_untouched(primed_session: SessionData) -> None:
    """Verifies priming is idempotent, so repeated preparation of one session rewrites none of its bootstrap files."""
    bootstrap = sorted(primed_session.processed_data.cindra_data_path.rglob("*.yaml"))
    before = {path: path.stat().st_mtime_ns for path in bootstrap}
    assert len(before) > _PLANE_COUNT

    prime_two_photon_recording(session_path=_session_path(primed_session))

    assert {path: path.stat().st_mtime_ns for path in bootstrap} == before


def test_discovering_jobs_reports_the_cindra_universe(primed_session: SessionData) -> None:
    """Verifies the universe holds one binarization job, a registration and a processing job per plane, and one
    combination job.
    """
    session, universe, possible = discover_two_photon_jobs(session_path=_session_path(primed_session))

    assert session.session_name == primed_session.session_name
    assert universe == _expected_universe()
    # Every stage is possible once the bootstrap exists, so the subset covers the whole universe.
    assert possible == universe


def test_discovering_jobs_without_a_bootstrap_reads_the_raw_acquisition_parameters(
    imaging_session: SessionData,
) -> None:
    """Verifies a session that has never been primed still resolves its universe, read from the raw parameters.

    Resolution follows the recording's acquisition parameters rather than the bootstrap, so a session can be planned
    before any preparation pass has primed it.
    """
    _session, universe, possible = discover_two_photon_jobs(session_path=_session_path(imaging_session))

    assert universe == _expected_universe()
    assert possible == universe


def test_an_incomplete_bootstrap_still_resolves_the_universe(primed_session: SessionData) -> None:
    """Verifies a missing per-plane runtime file leaves resolution intact, since it reads the parameters instead.

    An interrupted preparation pass is repaired by priming again rather than by failing every later resolution.
    """
    primed_session.processed_data.cindra_data_path.joinpath("plane_1", "runtime_data.yaml").unlink()

    _session, universe, possible = discover_two_photon_jobs(session_path=_session_path(primed_session))

    assert universe == _expected_universe()
    assert possible == universe


def test_priming_rewrites_an_incomplete_bootstrap(primed_session: SessionData) -> None:
    """Verifies the half-written bootstrap of an interrupted preparation is completed by priming again."""
    runtime_path = primed_session.processed_data.cindra_data_path.joinpath("plane_1", "runtime_data.yaml")
    runtime_path.unlink()

    prime_two_photon_recording(session_path=_session_path(primed_session))

    assert runtime_path.is_file()
    _session, universe, _possible = discover_two_photon_jobs(session_path=_session_path(primed_session))
    assert universe == _expected_universe()


def test_prerequisites_follow_the_cindra_stage_chain(primed_session: SessionData) -> None:
    """Verifies registration waits on binarization, processing on its own plane, and combination on every plane."""
    session, universe, _possible = discover_two_photon_jobs(session_path=_session_path(primed_session))

    prerequisites = two_photon_job_prerequisites(session=session, universe=universe)

    binarize = (str(SingleRecordingJobNames.BINARIZE), "")
    assert prerequisites[binarize] == ()
    assert prerequisites[(str(SingleRecordingJobNames.REGISTER), "plane_1")] == (binarize,)
    assert prerequisites[(str(SingleRecordingJobNames.PROCESS), "plane_1")] == (
        (str(SingleRecordingJobNames.REGISTER), "plane_1"),
    )
    assert set(prerequisites[(str(SingleRecordingJobNames.COMBINE), "")]) == {
        (str(SingleRecordingJobNames.PROCESS), f"plane_{plane}") for plane in range(_PLANE_COUNT)
    }


def test_only_the_two_photon_pipeline_declares_a_priming_step() -> None:
    """Verifies priming is declared only for the pipeline whose job model lives in state a dependency writes."""
    priming = {
        pipeline: resolve_dispatch(pipeline=pipeline).prime is not None  # type: ignore[union-attr]
        for pipeline in (*SESSION_PIPELINES, ProcessingPipelines.FORGING)
    }
    assert priming[ProcessingPipelines.TWO_PHOTON]
    assert not any(declares for pipeline, declares in priming.items() if pipeline is not ProcessingPipelines.TWO_PHOTON)


def test_resolving_jobs_reads_the_bootstrap_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, stubbed_recording: Callable[..., SimpleNamespace]
) -> None:
    """Verifies resolution shapes the universe from the plane count cindra's own resolver reports, without
    materializing the bootstrap.
    """
    session = stubbed_recording(plane_count=3)
    _forbid_writes(monkeypatch=monkeypatch)

    _resolved, universe, possible = discover_two_photon_jobs(session_path=_session_path(session))

    assert universe == _expected_universe(plane_count=3)
    assert possible == universe


def test_resolving_jobs_reports_a_recording_carrying_no_parameters(
    stubbed_recording: Callable[..., SimpleNamespace],
) -> None:
    """Verifies a recording whose acquisition parameters resolve nowhere is refused as holding no imaging data."""
    session = stubbed_recording(plane_count=None)

    with pytest.raises(FileNotFoundError, match="carries the acquisition parameters"):
        discover_two_photon_jobs(session_path=_session_path(session))


def test_priming_a_primed_recording_materializes_nothing(
    monkeypatch: pytest.MonkeyPatch, stubbed_recording: Callable[..., SimpleNamespace]
) -> None:
    """Verifies priming is idempotent, so repeated preparation of one session costs a read rather than a rewrite."""
    session = stubbed_recording(plane_count=3)
    _forbid_writes(monkeypatch=monkeypatch)

    prime_two_photon_recording(session_path=_session_path(session))

    # A primed recording is left exactly as it was found, so its cindra directory gains nothing.
    assert list(session.processed_data.cindra_data_path.iterdir()) == []


def test_priming_an_unprimed_recording_persists_both_halves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed_recording: Callable[..., SimpleNamespace]
) -> None:
    """Verifies an absent bootstrap is written once, the single-threaded step cindra requires before any job runs."""
    session = stubbed_recording(plane_count=None)
    configuration_path = tmp_path.joinpath(_CONFIGURATION_FILENAME)

    persisted: list[bool] = []
    primed: list[Path] = []

    def _record_configuration(
        session: Any,
        data_path: Path,
        *,
        display_progress: bool,
        persist: bool,
    ) -> tuple[None, Path]:
        """Records the persist flag the pipeline passed when building the configuration."""
        persisted.append(persist)
        return None, configuration_path

    def _record_priming(configuration_path: Path) -> SimpleNamespace:
        """Records the configuration the pipeline handed cindra's own single-threaded priming step."""
        primed.append(configuration_path)
        return SimpleNamespace(plane_count=_PLANE_COUNT)

    monkeypatch.setattr(two_photon_pipeline, "_resolve_configuration", _record_configuration)
    monkeypatch.setattr(two_photon_pipeline, "prime_recording", _record_priming)

    prime_two_photon_recording(session_path=_session_path(session))

    # Both halves of the bootstrap are written, which is what an unprimed recording needs before its jobs dispatch.
    # The configuration is persisted first, then handed to cindra, which writes each plane's runtime data from it.
    assert persisted == [True]
    assert primed == [configuration_path]


# Local dispatch


def test_naming_no_stage_runs_every_stage_in_order(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies an invocation naming no stage runs every stage, in binarize, register, process, combine order."""
    run_two_photon_processing_pipeline(session_path=_session_path(primed_session))

    assert _dispatched_pairs(dispatched_jobs) == _expected_universe()
    # Every dispatch reads the one configuration the session materialized and shares the session's tracker.
    configuration_path = primed_session.processed_data.cindra_data_path.joinpath(_CONFIGURATION_FILENAME)
    assert {record["configuration_path"] for record in dispatched_jobs} == {configuration_path}
    assert {record["tracker_path"] for record in dispatched_jobs} == {
        primed_session.processed_data.two_photon_tracker_path
    }
    # A worker count of -1 leaves the allocation to cindra's measured default for each stage.
    assert {record["workers"] for record in dispatched_jobs} == {None}


def test_a_single_stage_registers_only_the_jobs_it_runs(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies naming one stage dispatches that stage alone and leaves the sibling jobs out of the tracker."""
    run_two_photon_processing_pipeline(session_path=_session_path(primed_session), binarize=True, workers=8)

    assert _dispatched_pairs(dispatched_jobs) == [(str(SingleRecordingJobNames.BINARIZE), "")]
    # A named worker count overrides cindra's measured default for the stage this invocation runs.
    assert dispatched_jobs[0]["workers"] == 8

    snapshot = ProcessingTracker(file_path=primed_session.processed_data.two_photon_tracker_path).snapshot()
    assert [state.job_name for state in snapshot.values()] == [str(SingleRecordingJobNames.BINARIZE)]
    assert snapshot[dispatched_jobs[0]["job_id"]].status == ProcessingStatus.SCHEDULED


@pytest.mark.parametrize(
    ("stage_flags", "expected"),
    [
        (
            {"register": True},
            [
                (str(SingleRecordingJobNames.REGISTER), "plane_0"),
                (str(SingleRecordingJobNames.REGISTER), "plane_1"),
            ],
        ),
        (
            {"process": True},
            [
                (str(SingleRecordingJobNames.PROCESS), "plane_0"),
                (str(SingleRecordingJobNames.PROCESS), "plane_1"),
            ],
        ),
        ({"combine": True}, [(str(SingleRecordingJobNames.COMBINE), "")]),
    ],
)
def test_naming_any_one_stage_dispatches_that_stage_alone(
    primed_session: SessionData,
    dispatched_jobs: list[dict[str, Any]],
    stage_flags: dict[str, bool],
    expected: list[tuple[str, str]],
) -> None:
    """Verifies naming any single stage runs that stage alone, rather than falling into the run-everything default.

    An invocation naming no stage runs all four, so a stage the request does not recognize would silently re-run the
    whole recording and overwrite the outputs the operator asked to leave alone.
    """
    run_two_photon_processing_pipeline(session_path=_session_path(primed_session), **stage_flags)

    assert _dispatched_pairs(dispatched_jobs) == expected


def test_a_target_plane_narrows_the_per_plane_stages(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies naming a plane runs its per-plane stages alone, leaving the whole-recording stages unaffected."""
    run_two_photon_processing_pipeline(
        session_path=_session_path(primed_session), register=True, process=True, combine=True, target_plane=1
    )

    assert _dispatched_pairs(dispatched_jobs) == [
        (str(SingleRecordingJobNames.REGISTER), "plane_1"),
        (str(SingleRecordingJobNames.PROCESS), "plane_1"),
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]


def test_a_per_plane_stage_on_a_recording_holding_no_plane_dispatches_nothing(
    monkeypatch: pytest.MonkeyPatch,
    primed_session: SessionData,
    dispatched_jobs: list[dict[str, Any]],
) -> None:
    """Verifies a per-plane stage requested for a recording that holds no plane resolves no job and aligns nothing.

    The tracker refuses an empty alignment request, so the run has to skip the alignment rather than offer it one.
    """
    monkeypatch.setattr(
        two_photon_pipeline,
        "prime_recording",
        lambda configuration_path: SimpleNamespace(plane_count=0),  # noqa: ARG005
    )

    run_two_photon_processing_pipeline(session_path=_session_path(primed_session), register=True, process=True)

    assert dispatched_jobs == []
    # Nothing was registered, so the tracker keeps whatever the earlier priming left it holding.
    tracker = ProcessingTracker(file_path=primed_session.processed_data.two_photon_tracker_path)
    assert tracker.snapshot() == {}


def test_a_local_run_keeps_the_recorded_state_of_the_jobs_it_skips(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies alignment against the full universe lets a partial invocation preserve its siblings' recorded state."""
    run_two_photon_processing_pipeline(session_path=_session_path(primed_session))
    tracker = ProcessingTracker(file_path=primed_session.processed_data.two_photon_tracker_path)
    combine_id = ProcessingTracker.generate_job_id(job_name=str(SingleRecordingJobNames.COMBINE), specifier="")
    tracker.start_job(job_id=combine_id)
    tracker.complete_job(job_id=combine_id)
    dispatched_jobs.clear()

    run_two_photon_processing_pipeline(session_path=_session_path(primed_session), binarize=True)

    snapshot = tracker.snapshot()
    assert len(snapshot) == len(_expected_universe())
    assert snapshot[combine_id].status == ProcessingStatus.SUCCEEDED


# Remote dispatch


def test_a_job_identifier_runs_that_job_alone(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies remote mode selects the single job by identifier, leaving the stage flags and target plane unread."""
    job_id = ProcessingTracker.generate_job_id(job_name=str(SingleRecordingJobNames.PROCESS), specifier="plane_1")

    run_two_photon_processing_pipeline(
        session_path=_session_path(primed_session), job_id=job_id, binarize=True, target_plane=0, workers=4
    )

    assert _dispatched_pairs(dispatched_jobs) == [(str(SingleRecordingJobNames.PROCESS), "plane_1")]
    assert dispatched_jobs[0]["job_id"] == job_id
    assert dispatched_jobs[0]["workers"] == 4

    snapshot = ProcessingTracker(file_path=primed_session.processed_data.two_photon_tracker_path).snapshot()
    assert list(snapshot) == [job_id]


def test_a_remote_job_keeps_the_recorded_state_of_its_sibling_jobs(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies a scheduler-dispatched job leaves the state its sibling jobs recorded in the shared tracker intact.

    The scheduler dispatches each job of the universe separately against one tracker, so a job that treated its
    siblings as foreign entries would erase their completion and have every finished stage dispatched again.
    """
    run_two_photon_processing_pipeline(session_path=_session_path(primed_session))
    tracker = ProcessingTracker(file_path=primed_session.processed_data.two_photon_tracker_path)
    combine_id = ProcessingTracker.generate_job_id(job_name=str(SingleRecordingJobNames.COMBINE), specifier="")
    tracker.start_job(job_id=combine_id)
    tracker.complete_job(job_id=combine_id)
    dispatched_jobs.clear()

    run_two_photon_processing_pipeline(
        session_path=_session_path(primed_session),
        job_id=ProcessingTracker.generate_job_id(job_name=str(SingleRecordingJobNames.BINARIZE), specifier=""),
    )

    assert _dispatched_pairs(dispatched_jobs) == [(str(SingleRecordingJobNames.BINARIZE), "")]
    snapshot = tracker.snapshot()
    assert len(snapshot) == len(_expected_universe())
    assert snapshot[combine_id].status == ProcessingStatus.SUCCEEDED


def test_an_unknown_job_identifier_lists_the_available_jobs(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies a job identifier outside the session's universe is refused before anything is dispatched."""
    with pytest.raises(ValueError, match="must name a job the pipeline could produce"):
        run_two_photon_processing_pipeline(session_path=_session_path(primed_session), job_id="0123456789abcdef")

    assert dispatched_jobs == []


def test_a_remote_job_requires_the_materialized_configuration(
    primed_session: SessionData, dispatched_jobs: list[dict[str, Any]]
) -> None:
    """Verifies a scheduler-dispatched job reads the configuration its preparation step wrote."""
    primed_session.processed_data.cindra_data_path.joinpath(_CONFIGURATION_FILENAME).unlink()
    job_id = ProcessingTracker.generate_job_id(job_name=str(SingleRecordingJobNames.BINARIZE), specifier="")

    with pytest.raises(FileNotFoundError, match="No cindra configuration was found"):
        run_two_photon_processing_pipeline(session_path=_session_path(primed_session), job_id=job_id)

    assert dispatched_jobs == []


# Input screening


def test_a_session_without_raw_imaging_data_is_screened_out(experiment_session: SessionData) -> None:
    """Verifies a session that acquired no two-photon data is refused for holding no raw imaging directory."""
    _write_surgery_metadata(session=experiment_session)

    with pytest.raises(FileNotFoundError, match="raw two-photon imaging"):
        run_two_photon_processing_pipeline(session_path=_session_path(experiment_session))


def test_raw_imaging_data_without_acquisition_parameters_is_refused(experiment_session: SessionData) -> None:
    """Verifies raw imaging data without its acquisition parameters file is refused, since that file is the canonical
    source of the recording's metadata.
    """
    _write_surgery_metadata(session=experiment_session)
    experiment_session.raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No cindra acquisition"):
        run_two_photon_processing_pipeline(session_path=_session_path(experiment_session))


def test_a_directory_carrying_the_parameters_name_does_not_satisfy_the_screen(
    experiment_session: SessionData,
) -> None:
    """Verifies the screen answers on files alone, so a directory carrying the parameters name refuses the session.

    The screen exists to guarantee the recording's acquisition metadata can be read, which a directory sharing the
    filename cannot supply.
    """
    _write_surgery_metadata(session=experiment_session)
    imaging_directory = experiment_session.raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA)
    imaging_directory.joinpath("cindra_parameters.json").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No cindra acquisition"):
        run_two_photon_processing_pipeline(session_path=_session_path(experiment_session))


def test_a_session_without_surgery_metadata_cannot_resolve_a_configuration(
    session_factory: Callable[..., SessionData],
) -> None:
    """Verifies a session with no surgery metadata is refused, since the resolver reads the genotype from it."""
    session = session_factory()
    _write_acquisition_parameters(session=session)

    with pytest.raises(FileNotFoundError, match="No surgery metadata"):
        run_two_photon_processing_pipeline(session_path=_session_path(session))
