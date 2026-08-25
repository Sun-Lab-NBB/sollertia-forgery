"""Contains tests for the system-agnostic forging pipeline against a real project hierarchy, from dataset definition
through the cross-recording stages and the per-session assembly.
"""

from __future__ import annotations

from types import SimpleNamespace
import shutil
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from cindra import MultiRecordingJobNames, MultiRecordingConfiguration
import polars as pl
import pytest
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SubjectData,
    SurgeryData,
    RawDataFiles,
    SessionTypes,
    ProcedureData,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    forging_tracker_path,
    run_forging_pipeline,
    discover_forging_jobs,
    define_forging_dataset,
)
from sollertia_forgery.shared_assets import multi_recording_dataset_name
import sollertia_forgery.forging.pipeline as pipeline_module
from sollertia_forgery.forging.pipeline import (
    _load_multiday_plan,
    _build_forging_universe,
    _materialize_multiday_plan,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

_DATASET_NAME: str = "TestDataset"
"""The name every forged dataset in this module is defined under, carrying uppercase so the lowercasing the cindra
output directory applies stays observable."""

_MULTIDAY_CONFIGURATION_FILENAME: str = "multi_recording_configuration.yaml"
"""The filename each tracked animal's materialized multi-recording configuration is written under."""

_ARGUMENT_RECORD_FILENAME: str = "assembly_arguments.txt"
"""The filename the stand-in assembly worker records its source path and dataset name into, which lets a job
dispatched into a worker process report what it was called with."""

_DESCRIBED_COLUMN: str = "time_us"
"""A column name the Mesoscope-VR description binding declares, so a feather holding it passes the dataset's
description check."""

_UNDESCRIBED_COLUMN: str = "unregistered_measurement"
"""A column name no acquisition system describes, which the assembly stage rejects."""

_GENOTYPE: str = "GP5.17"
"""The genotype every synthetic animal records, which the Mesoscope-VR resolver maps to a calcium indicator."""

_PRIMING_STEP: str = "priming"
"""The label the shared dispatch recorder marks a bootstrap priming call with, which collides with no cindra job
name."""


def assemble_described_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Writes one session's assembled feather holding a single described column.

    Args:
        source_session_path: The path to the source session's root directory.
        output_path: The path to the session's ``data.feather`` inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name the pipeline forwarded.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.joinpath(_ARGUMENT_RECORD_FILENAME).write_text(f"{source_session_path}\n{dataset_name}\n")
    pl.DataFrame({_DESCRIBED_COLUMN: [0, 1, 2]}).write_ipc(file=output_path, compression="uncompressed")


def assemble_undescribed_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Writes one session's assembled feather holding a column no dataset describes.

    Args:
        source_session_path: The path to the source session's root directory.
        output_path: The path to the session's ``data.feather`` inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name the pipeline forwarded.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({_DESCRIBED_COLUMN: [0], _UNDESCRIBED_COLUMN: [1]}).write_ipc(
        file=output_path, compression="uncompressed"
    )


def assemble_failing_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Fails the assembly of every session it is handed.

    Args:
        source_session_path: The path to the source session's root directory.
        output_path: The path to the session's ``data.feather`` inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name the pipeline forwarded.

    Raises:
        RuntimeError: Always, standing in for an assembly worker that cannot complete.
    """
    message = f"The assembly worker cannot process '{source_session_path.name}'."
    raise RuntimeError(message)


@dataclass(frozen=True)
class MultidayCall:
    """Records one cross-recording invocation the pipeline dispatched.

    Attributes:
        configuration_path: The configuration the invocation was pointed at.
        job_name: The cindra job the invocation requested.
        specifier: The recording identifier the invocation named.
        job_id: The forging tracker identifier the invocation recorded against.
        workers: The worker count the invocation was given.
    """

    configuration_path: Path
    job_name: MultiRecordingJobNames
    specifier: str
    job_id: str
    workers: int


@dataclass(frozen=True)
class ForgingProject:
    """Bundles a source project laid out for forging with the sessions it holds.

    Attributes:
        project_root: The path to the project's root directory.
        sessions: The loaded sessions, in the order they were created.
    """

    project_root: Path
    sessions: tuple[SessionData, ...]

    def names(self) -> tuple[str, ...]:
        """Returns the names of every session the project holds, in creation order."""
        return tuple(session.session_name for session in self.sessions)


def write_surgery_metadata(session: SessionData) -> None:
    """Writes the surgery metadata snapshot the cindra configuration resolver reads the animal's genotype from.

    Args:
        session: The loaded session whose raw data receives the snapshot.
    """
    SurgeryData(
        subject=SubjectData(
            id=int(session.animal_id),
            ear_punch="left",
            sex="F",
            genotype=_GENOTYPE,
            date_of_birth_us=1_600_000_000_000_000,
            weight_g=25.0,
            cage=1,
            location_housed="vivarium",
            status="alive",
        ),
        procedure=ProcedureData(
            surgery_start_us=1_650_000_000_000_000,
            surgery_end_us=1_650_000_003_600_000,
            surgeon="tester",
            protocol="TEST-1",
            surgery_notes="A synthetic procedure.",
            post_op_notes="A synthetic recovery.",
        ),
        drugs=[],
        implants=[],
        injections=[],
    ).to_yaml(file_path=session.raw_data.surgery_metadata_path)


@pytest.fixture
def recorded_dispatch_order() -> list[tuple[str, str]]:
    """Returns the one timeline both cross-recording recorders append to, in the order the pipeline reached them.

    Priming and dispatch are separate calls, so recording them into two lists makes only their contents assertable.
    A single interleaved timeline is what makes the order between them observable, which is the guarantee that
    matters: every stage reads the bootstrap its animal's priming wrote, and every extraction reads the ROI set its
    animal's discovery wrote.

    Returns:
        The list every priming and every dispatched stage is appended to, as a ``(step, subject)`` pair naming the
        animal a priming or a whole-animal stage belongs to and the session a per-recording stage names.
    """
    return []


@pytest.fixture
def recorded_primings(monkeypatch: pytest.MonkeyPatch, recorded_dispatch_order: list[tuple[str, str]]) -> list[Path]:
    """Replaces the cindra dataset priming call with a recorder that writes no bootstrap.

    Every cross-recording stage reads a shared bootstrap the priming call writes, and writing it needs processed
    imaging output these tests never produce, so the call is recorded rather than performed.

    Args:
        monkeypatch: The fixture used to replace the priming call the pipeline module holds.
        recorded_dispatch_order: The shared timeline each priming is appended to alongside the dispatched stages.

    Returns:
        The list every primed configuration path is appended to, in call order.
    """
    primed: list[Path] = []

    def _prime(configuration_path: Path) -> None:
        """Records one priming call without writing the bootstrap it would otherwise materialize."""
        primed.append(configuration_path)
        recorded_dispatch_order.append((_PRIMING_STEP, configuration_path.parent.name))

    monkeypatch.setattr(pipeline_module, "prime_dataset", _prime)
    return primed


@pytest.fixture
def recorded_multiday_jobs(
    monkeypatch: pytest.MonkeyPatch, recorded_primings: list[Path], recorded_dispatch_order: list[tuple[str, str]]
) -> list[MultidayCall]:
    """Replaces the cindra cross-recording entry point with a recorder that succeeds on the forging tracker.

    cindra records each stage's state directly on the tracker it is handed, so the recorder drives the same transitions
    the real entry point would. The priming recorder is requested alongside it, since the discovery stage primes the
    shared bootstrap before it dispatches.

    Args: monkeypatch: The fixture used to replace the entry point the pipeline module holds. recorded_primings: The
    recorder standing in for the dataset priming the discovery stage performs first. recorded_dispatch_order: The shared
    timeline each dispatch is appended to alongside the primings.

    Returns: The list every dispatched invocation is appended to, in dispatch order.
    """
    calls: list[MultidayCall] = []

    def _execute(
        configuration_path: Path,
        job_name: MultiRecordingJobNames,
        specifier: str,
        job_id: str,
        tracker: ProcessingTracker,
        *,
        workers: int | None = None,
    ) -> None:
        """Records one invocation and marks its job as succeeded on the forging tracker."""
        calls.append(
            MultidayCall(
                configuration_path=configuration_path,
                job_name=job_name,
                specifier=specifier,
                job_id=job_id,
                workers=workers,
            )
        )
        # A stage spanning the whole animal carries no specifier, so it is named by the animal its configuration
        # belongs to, which is the same subject the priming of that animal is recorded under.
        recorded_dispatch_order.append((job_name.value, specifier or configuration_path.parent.name))
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    monkeypatch.setattr(pipeline_module, "execute_multi_recording_job", _execute)
    return calls


@pytest.fixture
def install_assembly_worker(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], None]:
    """Returns an installer that binds one picklable assembly worker for the dataset's acquisition system.

    The registered Mesoscope-VR worker assembles a session out of its whole processed-data tree, which the forging
    pipeline itself never inspects. Binding a stand-in keeps these tests on the pipeline's own contract, and the
    worker stays module-level so the parallel stage can hand it to a worker process.

    Args:
        monkeypatch: The fixture used to replace the registry accessor the pipeline module holds.

    Returns:
        A callable taking the worker every acquisition system resolves to.
    """

    def _install(worker: Any) -> None:
        monkeypatch.setattr(pipeline_module, "resolve_forging_assembly_worker", lambda system: worker)  # noqa: ARG005

    return _install


@pytest.fixture
def experiment_project(
    project_root: Path,
    session_factory: Callable[..., SessionData],
    mark_session_processed: Callable[[SessionData], None],
) -> ForgingProject:
    """Builds a project holding two experiment sessions for one animal and one for a second animal.

    Every session is fully processed and carries its surgery metadata, which is the state forging admission and the
    cross-recording configuration resolver both require.

    Args:
        project_root: The created project the sessions are placed under.
        session_factory: The builder that creates and reloads each session.
        mark_session_processed: The helper that records every per-session pipeline as succeeded.

    Returns:
        The project and the three loaded sessions, in creation order.
    """
    sessions = [
        session_factory(animal_id=animal_id, experiment_name="test_experiment") for animal_id in ("305", "305", "321")
    ]
    for session in sessions:
        mark_session_processed(session)
        write_surgery_metadata(session=session)
    return ForgingProject(project_root=project_root, sessions=tuple(sessions))


@pytest.fixture
def training_project(
    project_root: Path,
    session_factory: Callable[..., SessionData],
    mark_session_processed: Callable[[SessionData], None],
) -> ForgingProject:
    """Builds a project holding two run-training sessions of one animal, which carry no cross-recording tracking.

    Args:
        project_root: The created project the sessions are placed under.
        session_factory: The builder that creates and reloads each session.
        mark_session_processed: The helper that records every per-session pipeline as succeeded.

    Returns:
        The project and the two loaded sessions, in creation order.
    """
    sessions = [session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING) for _ in range(2)]
    for session in sessions:
        mark_session_processed(session)
        write_surgery_metadata(session=session)
    return ForgingProject(project_root=project_root, sessions=tuple(sessions))


def session_entry(dataset: DatasetData, name: str) -> Any:
    """Returns the dataset's own entry for one session name.

    Args:
        dataset: The resolved dataset holding the session.
        name: The name of the session whose entry to return.

    Returns:
        The dataset session entry, which carries the output paths the forged hierarchy uses.
    """
    return next(entry for entry in dataset.sessions if entry.session == name)


def tracker_states(dataset: DatasetData) -> dict[tuple[str, str], ProcessingStatus]:
    """Returns the recorded status of every job the dataset's forging tracker holds.

    Args:
        dataset: The resolved dataset whose tracker to read.

    Returns:
        A mapping of each ``(job_name, specifier)`` pair to the status the tracker records for it.
    """
    snapshot = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset)).snapshot()
    return {(state.job_name, state.specifier): state.status for state in snapshot.values()}


# Tests for dataset definition and the multi-day plan


def test_define_forging_dataset_materializes_a_configuration_for_each_tracked_animal(
    experiment_project: ForgingProject,
) -> None:
    """Verifies that defining a dataset writes each animal's cross-recording configuration pointing at its cindra
    outputs, under the qualified, unfolded dataset name the pipeline records for it.
    """
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )

    plan = _load_multiday_plan(dataset=dataset)
    assert set(plan) == {"305", "321"}
    assert plan["305"][1] == list(experiment_project.names()[:2])

    written = MultiRecordingConfiguration.from_yaml(file_path=plan["305"][0])
    assert written.recording_io.dataset_name == multi_recording_dataset_name(
        animal_id="305", dataset_name=_DATASET_NAME
    )
    assert written.recording_io.recording_directories == tuple(
        session.processed_data.cindra_data_path for session in experiment_project.sessions[:2]
    )
    assert not written.runtime.display_progress_bars


def test_define_forging_dataset_names_a_repeated_session_once(experiment_project: ForgingProject) -> None:
    """Verifies that the session list names the sessions the dataset must contain rather than the sessions to append,
    and the shared hierarchy rejects a request naming the same session twice, so a repeat resolves to one directory.
    """
    names = experiment_project.names()

    dataset = define_forging_dataset(
        name=_DATASET_NAME,
        session_names=(names[0], names[1], names[0]),
        project_root=experiment_project.project_root,
    )

    assert [entry.session for entry in dataset.sessions] == [names[0], names[1]]


def test_define_forging_dataset_skips_an_animal_without_cross_recording_tracking(
    training_project: ForgingProject,
) -> None:
    """Verifies that an animal whose sessions resolve no multi-recording configuration carries no plan entry."""
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=training_project.names(), project_root=training_project.project_root
    )

    replanned = _materialize_multiday_plan(
        dataset=dataset, project_root=training_project.project_root, display_progress=True
    )
    assert replanned == {}
    assert _load_multiday_plan(dataset=dataset) == {}
    assert _build_forging_universe(dataset=dataset, multiday_plan={}) == [
        (FORGING_JOB_NAME, name) for name in training_project.names()
    ]


def test_define_forging_dataset_records_the_progress_preference(experiment_project: ForgingProject) -> None:
    """Verifies that the progress-bar flag reaches every materialized cross-recording configuration."""
    dataset = define_forging_dataset(
        name=_DATASET_NAME,
        session_names=experiment_project.names(),
        project_root=experiment_project.project_root,
        display_progress=True,
    )

    plan = _load_multiday_plan(dataset=dataset)
    assert MultiRecordingConfiguration.from_yaml(file_path=plan["321"][0]).runtime.display_progress_bars


def test_define_forging_dataset_resets_a_rebuilt_animals_recorded_jobs(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that rebuilding an animal returns its recorded stages to the scheduled state and rewrites its cross-
    recording configuration over its new session set, so the next run redoes those stages against the recordings the
    dataset still holds for it.
    """
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    define_forging_dataset(name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root)
    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    rebuilt = define_forging_dataset(
        name=_DATASET_NAME,
        session_names=(names[0],),
        project_root=experiment_project.project_root,
        recreate_animals=("305",),
    )

    states = tracker_states(dataset=rebuilt)
    assert states[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == ProcessingStatus.SCHEDULED
    assert states[FORGING_JOB_NAME, names[0]] == ProcessingStatus.SCHEDULED
    assert states[FORGING_JOB_NAME, names[2]] == ProcessingStatus.SUCCEEDED
    assert states[MULTIDAY_DISCOVERY_JOB_NAME, "321"] == ProcessingStatus.SUCCEEDED

    written = MultiRecordingConfiguration.from_yaml(file_path=_load_multiday_plan(dataset=rebuilt)["305"][0])
    assert written.recording_io.recording_directories == (
        experiment_project.sessions[0].processed_data.cindra_data_path,
    )


def test_load_multiday_plan_skips_an_animal_holding_no_sessions(tmp_path: Path) -> None:
    """Verifies that an animal the dataset lists without any session contributes no plan entry."""
    animal_path = tmp_path.joinpath("305")
    animal_path.mkdir()
    animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).touch()
    dataset = SimpleNamespace(
        animals=(SimpleNamespace(animal="305", animal_path=animal_path),),
        get_sessions_for_animal=lambda animal: (),  # noqa: ARG005
    )

    assert _load_multiday_plan(dataset=dataset) == {}


def test_load_multiday_plan_skips_an_animal_without_a_written_configuration(
    experiment_project: ForgingProject,
) -> None:
    """Verifies that reading the plan back reports only the animals whose configuration is on disk."""
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )
    dataset.get_animal(animal="321").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).unlink()

    assert set(_load_multiday_plan(dataset=dataset)) == {"305"}


def test_define_forging_dataset_rewrites_a_configuration_that_was_never_written(
    experiment_project: ForgingProject,
) -> None:
    """Verifies that repeating a definition materializes the configuration of an animal the dataset already holds
    without one.

    The dataset marker is committed before the configurations are written, so a definition that died in between leaves
    an animal the marker names and the disk does not. Reading the plan back passes such an animal over in silence, so
    its discovery and extraction stages vanish from the job universe until a repeated definition repairs it.
    """
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )
    dataset.get_animal(animal="321").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).unlink()

    redefined = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )

    plan = _load_multiday_plan(dataset=redefined)
    assert set(plan) == {"305", "321"}
    written = MultiRecordingConfiguration.from_yaml(file_path=plan["321"][0])
    assert written.recording_io.recording_directories == (
        experiment_project.sessions[2].processed_data.cindra_data_path,
    )


def test_define_forging_dataset_passes_over_an_unconfigured_animal_whose_sources_moved_away(
    experiment_project: ForgingProject,
) -> None:
    """Verifies that an animal the dataset holds without a configuration is left as it stands once its source data has
    moved off this machine, so a large project is still forged in passes.

    Materializing a configuration loads every one of the animal's sessions from the project root, so an animal whose
    directory is gone cannot be repaired and attempting it would fail the whole definition.
    """
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names[:2], project_root=experiment_project.project_root
    )
    dataset.get_animal(animal="305").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).unlink()
    shutil.rmtree(experiment_project.project_root.joinpath("305"))

    extended = define_forging_dataset(
        name=_DATASET_NAME, session_names=(names[2],), project_root=experiment_project.project_root
    )

    assert set(_load_multiday_plan(dataset=extended)) == {"321"}
    assert not extended.get_animal(animal="305").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).is_file()


def test_define_forging_dataset_extends_a_dataset_whose_forged_animal_lost_its_sources(
    experiment_project: ForgingProject,
) -> None:
    """Verifies that adding an animal succeeds while an animal the dataset already holds has no source data left."""
    names = experiment_project.names()
    define_forging_dataset(name=_DATASET_NAME, session_names=names[:2], project_root=experiment_project.project_root)

    # The forged animal's sessions move to long-term storage, leaving the dataset holding it with no source data.
    shutil.rmtree(experiment_project.project_root.joinpath("305"))

    extended = define_forging_dataset(
        name=_DATASET_NAME, session_names=(names[2],), project_root=experiment_project.project_root
    )

    assert {dataset_animal.animal for dataset_animal in extended.animals} == {"305", "321"}
    assert len(extended.sessions) == 3
    # The frozen animal keeps the configuration written when it was added, and the added animal receives its own.
    assert extended.get_animal(animal="305").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).is_file()
    assert extended.get_animal(animal="321").animal_path.joinpath(_MULTIDAY_CONFIGURATION_FILENAME).is_file()


def test_discover_forging_jobs_reads_no_source_data(experiment_project: ForgingProject) -> None:
    """Verifies that job discovery resolves the whole universe after every source session has moved away."""
    dataset_path = _define_whole_project(experiment_project=experiment_project).dataset_data_path.parent
    for animal_id in ("305", "321"):
        shutil.rmtree(experiment_project.project_root.joinpath(animal_id))

    _, universe, possible = discover_forging_jobs(dataset_path=dataset_path)

    assert universe == possible
    assert (MULTIDAY_DISCOVERY_JOB_NAME, "305") in universe
    assert (FORGING_JOB_NAME, experiment_project.names()[0]) in universe


def test_discover_forging_jobs_reports_the_whole_universe_as_possible(experiment_project: ForgingProject) -> None:
    """Verifies that batch discovery names every stage the hierarchy holds and treats each one as runnable."""
    dataset_path = _define_whole_project(experiment_project=experiment_project).dataset_data_path.parent

    dataset, universe, possible = discover_forging_jobs(dataset_path=dataset_path)

    names = experiment_project.names()
    assert dataset.name == _DATASET_NAME
    assert universe == possible
    assert universe == [
        (MULTIDAY_DISCOVERY_JOB_NAME, "305"),
        (MULTIDAY_EXTRACTION_JOB_NAME, names[0]),
        (MULTIDAY_EXTRACTION_JOB_NAME, names[1]),
        (MULTIDAY_DISCOVERY_JOB_NAME, "321"),
        (MULTIDAY_EXTRACTION_JOB_NAME, names[2]),
        *((FORGING_JOB_NAME, name) for name in names),
    ]


def _define_whole_project(experiment_project: ForgingProject) -> DatasetData:
    """Defines the module's dataset over every session the project holds.

    Args:
        experiment_project: The source project whose sessions the dataset is defined from.

    Returns:
        The resolved dataset.
    """
    return define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )


# Tests for the local pipeline run


@pytest.mark.xdist_group(name="worker_pool")
def test_run_forging_pipeline_completes_every_stage_across_a_worker_pool(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    recorded_dispatch_order: list[tuple[str, str]],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a full local run primes each animal's bootstrap, then runs that animal's discovery and every one of
    its extractions before moving to the next animal, records every stage as succeeded, and re-exports each session's
    own shared assets beside its assembled feather.

    The cross-recording steps are compared as one ordered timeline rather than counted per kind, because the order is
    the guarantee: an extraction reads the ROI set its animal's discovery wrote, and every stage reads the bootstrap the
    priming wrote ahead of that animal's first stage.
    """
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )

    run_forging_pipeline(
        name=_DATASET_NAME, project_root=experiment_project.project_root, workers=4, display_progress=True
    )

    assert recorded_dispatch_order == [
        (_PRIMING_STEP, "305"),
        (MultiRecordingJobNames.DISCOVER.value, "305"),
        (MultiRecordingJobNames.EXTRACT.value, names[0]),
        (MultiRecordingJobNames.EXTRACT.value, names[1]),
        (_PRIMING_STEP, "321"),
        (MultiRecordingJobNames.DISCOVER.value, "321"),
        (MultiRecordingJobNames.EXTRACT.value, names[2]),
    ]

    states = tracker_states(dataset=DatasetData.load(dataset_path=dataset.dataset_data_path.parent))
    assert set(states.values()) == {ProcessingStatus.SUCCEEDED}
    assert len(states) == 8
    # Each re-exported asset is compared against its own source, since a forged directory holding the right filenames
    # over the wrong documents passes every existence check while every downstream reader parses the wrong type.
    for source in experiment_project.sessions:
        entry = session_entry(dataset=dataset, name=source.session_name)
        assert entry.data_path.is_file()
        output_directory = entry.data_path.parent
        assert (
            output_directory.joinpath(RawDataFiles.SESSION_DESCRIPTOR).read_bytes()
            == source.raw_data.session_descriptor_path.read_bytes()
        )
        assert (
            output_directory.joinpath(RawDataFiles.VR_CONFIGURATION).read_bytes()
            == source.raw_data.vr_configuration_path.read_bytes()
        )
        assert (
            output_directory.joinpath(RawDataFiles.EXPERIMENT_CONFIGURATION).read_bytes()
            == source.raw_data.experiment_configuration_path.read_bytes()
        )


def test_run_forging_pipeline_assembles_sequentially_with_one_worker(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a single-worker run assembles in the parent process and forwards the dataset name to the worker."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )

    run_forging_pipeline(
        name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1, display_progress=True
    )

    first = session_entry(dataset=dataset, name=names[0])
    recorded = first.data_path.parent.joinpath(_ARGUMENT_RECORD_FILENAME).read_text().splitlines()
    assert recorded == [str(experiment_project.project_root.joinpath("305", names[0])), _DATASET_NAME]
    assert set(tracker_states(dataset=dataset).values()) == {ProcessingStatus.SUCCEEDED}


@pytest.mark.xdist_group(name="worker_pool")
def test_run_forging_pipeline_assembles_at_the_default_worker_count(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that a run naming no worker count assembles every session across the parallel pool.

    The documented default asks for every available core by spelling the request as a negative number, so the request
    itself is never a pool width. The host's resolved core count is pinned here so the parallel branch is the one taken
    wherever this suite runs.
    """
    monkeypatch.setattr(pipeline_module, "resolve_worker_count", lambda requested_workers: 16)  # noqa: ARG005
    install_assembly_worker(assemble_described_session)
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root)

    for entry in dataset.sessions:
        assert entry.data_path.is_file()
    assert set(tracker_states(dataset=dataset).values()) == {ProcessingStatus.SUCCEEDED}


def test_run_forging_pipeline_registers_the_only_job_of_a_single_session_dataset(
    training_project: ForgingProject,
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a dataset whose whole universe is one outstanding job registers that job before running it.

    A training session carries no cross-recording stages, so a dataset holding one of them has a universe of exactly one
    assembly job. A tracker refuses a job it was never told to track, so a run that skipped the registration would die
    on the first session it assembled rather than forge it.
    """
    install_assembly_worker(assemble_described_session)
    names = training_project.names()[:1]
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=training_project.project_root
    )

    run_forging_pipeline(name=_DATASET_NAME, project_root=training_project.project_root, workers=1)

    assert session_entry(dataset=dataset, name=names[0]).data_path.is_file()
    assert tracker_states(dataset=dataset) == {(FORGING_JOB_NAME, names[0]): ProcessingStatus.SUCCEEDED}


def test_run_forging_pipeline_resolves_the_assembly_worker_under_the_acquisition_system(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that the per-session assembly worker is looked up under the dataset's acquisition system.

    The registry is keyed by acquisition system alone, so a lookup spelled in any other vocabulary resolves no worker
    and the run dies in the registry before a single session is assembled.
    """
    requested: list[str] = []

    def _resolve(system: str) -> Any:
        """Records the key the pipeline resolved the per-session assembly worker under."""
        requested.append(system)
        return assemble_described_session

    monkeypatch.setattr(pipeline_module, "resolve_forging_assembly_worker", _resolve)
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    assert requested == [dataset.acquisition_system]


def test_run_forging_pipeline_reexports_only_the_assets_a_training_session_carries(
    training_project: ForgingProject,
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a dataset of training sessions runs assembly alone and re-exports the descriptor by itself."""
    install_assembly_worker(assemble_described_session)
    names = training_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=training_project.project_root
    )

    run_forging_pipeline(name=_DATASET_NAME, project_root=training_project.project_root, workers=1)

    states = tracker_states(dataset=dataset)
    assert set(states) == {(FORGING_JOB_NAME, name) for name in names}
    output_directory = session_entry(dataset=dataset, name=names[0]).data_path.parent
    assert output_directory.joinpath(RawDataFiles.SESSION_DESCRIPTOR).is_file()
    assert not output_directory.joinpath(RawDataFiles.VR_CONFIGURATION).exists()
    assert not output_directory.joinpath(RawDataFiles.EXPERIMENT_CONFIGURATION).exists()


@pytest.mark.xdist_group(name="worker_pool")
def test_run_forging_pipeline_skips_the_stages_already_recorded_as_succeeded(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a run dispatches only the outstanding stages, leaving every succeeded one untouched."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names[:2], project_root=experiment_project.project_root
    )
    universe = _build_forging_universe(dataset=dataset, multiday_plan=_load_multiday_plan(dataset=dataset))
    tracker = ProcessingTracker(file_path=forging_tracker_path(dataset=dataset))
    tracker.align_jobs(jobs=universe, universe=universe)
    for job in (
        (MULTIDAY_DISCOVERY_JOB_NAME, "305"),
        (MULTIDAY_EXTRACTION_JOB_NAME, names[0]),
        (FORGING_JOB_NAME, names[0]),
    ):
        job_id = ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=4)

    assert [call.specifier for call in recorded_multiday_jobs] == [names[1]]
    assert not session_entry(dataset=dataset, name=names[0]).data_path.exists()
    assert session_entry(dataset=dataset, name=names[1]).data_path.is_file()
    # The run declares the whole universe while requesting only the outstanding jobs, so the record of every stage it
    # skipped survives. Aligning against the outstanding subset alone would delete the skipped stages as foreign.
    assert set(tracker_states(dataset=dataset)) == set(universe)
    assert set(tracker_states(dataset=dataset).values()) == {ProcessingStatus.SUCCEEDED}


def test_run_forging_pipeline_reports_a_fully_assembled_dataset(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that a second run over a finished dataset dispatches nothing and says so."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    define_forging_dataset(name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root)
    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    messages: list[str] = []
    monkeypatch.setattr(pipeline_module.console, "echo", lambda message, **_keywords: messages.append(message))
    install_assembly_worker(assemble_failing_session)
    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    assert "Every session in the dataset is already assembled." in messages
    assert "Prepared 0 outstanding forging job(s) out of 8 total." in messages


# Tests for the assembly stage's own guarantees


def test_run_forging_pipeline_rejects_a_session_missing_a_required_asset(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a session whose required VR snapshot is absent fails before the worker is invoked."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=(names[0],), project_root=experiment_project.project_root
    )
    experiment_project.sessions[0].raw_data.vr_configuration_path.unlink()

    with pytest.raises(FileNotFoundError, match="does not"):
        run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    assert not session_entry(dataset=dataset, name=names[0]).data_path.exists()
    assert tracker_states(dataset=dataset)[FORGING_JOB_NAME, names[0]] == ProcessingStatus.FAILED


def test_run_forging_pipeline_rejects_an_undescribed_column(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a feather carrying a column the dataset does not describe fails its own session's assembly."""
    install_assembly_worker(assemble_undescribed_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=(names[0],), project_root=experiment_project.project_root
    )

    with pytest.raises(ValueError, match="columns are undescribed"):
        run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=1)

    assert tracker_states(dataset=dataset)[FORGING_JOB_NAME, names[0]] == ProcessingStatus.FAILED


@pytest.mark.xdist_group(name="worker_pool")
def test_run_forging_pipeline_records_every_parallel_failure_before_re_raising(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],  # Requested so the cross-recording stages succeed.
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a failing parallel assembly is recorded against every dispatched job and then re-raised."""
    install_assembly_worker(assemble_failing_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )

    with pytest.raises(RuntimeError, match="The assembly worker cannot process"):
        run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, workers=4)

    states = tracker_states(dataset=dataset)
    assert {states[FORGING_JOB_NAME, name] for name in names} == {ProcessingStatus.FAILED}


# Tests for the single-job remote mode


def test_run_forging_pipeline_rejects_an_unknown_job_id(
    experiment_project: ForgingProject,
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a job identifier the dataset does not carry is refused rather than silently skipped."""
    install_assembly_worker(assemble_described_session)
    define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )

    with pytest.raises(ValueError, match="does not match any forging job"):
        run_forging_pipeline(
            name=_DATASET_NAME, project_root=experiment_project.project_root, job_id="0123456789abcdef"
        )


def test_run_forging_pipeline_runs_the_named_discovery_job_alone(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that dispatching a discovery identifier runs that animal's discovery and nothing else."""
    install_assembly_worker(assemble_described_session)
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )
    job_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_DISCOVERY_JOB_NAME, specifier="305")

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, job_id=job_id, workers=6)

    assert len(recorded_multiday_jobs) == 1
    assert recorded_multiday_jobs[0].job_name is MultiRecordingJobNames.DISCOVER
    assert recorded_multiday_jobs[0].workers == 6
    assert recorded_multiday_jobs[0].configuration_path == _load_multiday_plan(dataset=dataset)["305"][0]
    assert tracker_states(dataset=dataset)[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == ProcessingStatus.SUCCEEDED


def test_run_forging_pipeline_normalizes_a_non_positive_worker_request(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that a worker request below one reaches the cross-recording stage as -1.

    Every worker count below one means every available core throughout this library, but cindra spells that request as
    -1 alone and rejects every other non-positive value, so an un-normalized zero would be refused by the stage.
    """
    install_assembly_worker(assemble_described_session)
    define_forging_dataset(
        name=_DATASET_NAME, session_names=experiment_project.names(), project_root=experiment_project.project_root
    )
    job_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_DISCOVERY_JOB_NAME, specifier="305")

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, job_id=job_id, workers=0)

    assert [call.workers for call in recorded_multiday_jobs] == [-1]


def test_run_forging_pipeline_runs_the_named_extraction_job_alone(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that dispatching an extraction identifier runs that session against its own animal's configuration."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )
    job_id = ProcessingTracker.generate_job_id(job_name=MULTIDAY_EXTRACTION_JOB_NAME, specifier=names[2])

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, job_id=job_id)

    assert len(recorded_multiday_jobs) == 1
    assert recorded_multiday_jobs[0].job_name is MultiRecordingJobNames.EXTRACT
    assert recorded_multiday_jobs[0].specifier == names[2]
    assert recorded_multiday_jobs[0].configuration_path == _load_multiday_plan(dataset=dataset)["321"][0]


def test_run_forging_pipeline_runs_the_named_assembly_job_alone(
    experiment_project: ForgingProject,
    recorded_multiday_jobs: list[MultidayCall],
    install_assembly_worker: Callable[[Any], None],
) -> None:
    """Verifies that dispatching an assembly identifier assembles that one session and leaves the rest outstanding."""
    install_assembly_worker(assemble_described_session)
    names = experiment_project.names()
    dataset = define_forging_dataset(
        name=_DATASET_NAME, session_names=names, project_root=experiment_project.project_root
    )
    job_id = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=names[1])

    run_forging_pipeline(name=_DATASET_NAME, project_root=experiment_project.project_root, job_id=job_id)

    assert recorded_multiday_jobs == []
    assert session_entry(dataset=dataset, name=names[1]).data_path.is_file()
    assert not session_entry(dataset=dataset, name=names[0]).data_path.exists()
    states = tracker_states(dataset=dataset)
    assert states[FORGING_JOB_NAME, names[1]] == ProcessingStatus.SUCCEEDED
    assert states[FORGING_JOB_NAME, names[0]] == ProcessingStatus.SCHEDULED
