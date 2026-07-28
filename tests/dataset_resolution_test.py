"""Tests for the forging dataset resolution policy and the forging pipeline's tracker-state helpers.

The resolution policy is exercised against a source project laid out on disk, with session discovery, session loading,
and the column-description registry stubbed so the tests turn on the policy rather than on the shared hierarchy's
marker formats. The dataset hierarchy itself is the real one, so appending and rebuilding go through the shared
mutators the policy composes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pytest
from sollertia_shared_assets import DatasetData, SessionTypes, AcquisitionSystems
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    resolve_dataset,
    run_forging_pipeline,
    build_forging_universe,
    define_forging_dataset,
    forging_job_prerequisites,
)
import sollertia_forgery.forging.dataset as dataset_module
import sollertia_forgery.forging.pipeline as pipeline_module
from sollertia_forgery.forging.pipeline import _reset_animal_jobs, _resolve_runnable_jobs

_COLUMN_DESCRIPTIONS: dict[str, str] = {"time_us": "Microsecond-precision sample timestamps."}
"""A minimal column-description binding, standing in for what the acquisition system's registry entry returns."""

_SURGERY_FILENAME: str = "surgery_metadata.yaml"
"""The per-animal metadata filename the resolution policy copies into each animal's dataset directory."""


def _install_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, list[str]],
    session_types: dict[str, SessionTypes] | None = None,
    *,
    write_surgery: bool = True,
) -> Path:
    """Creates a source project with the given animal-to-session layout and stubs the resolution dependencies.

    Args:
        tmp_path: The temporary directory the project root is created under.
        monkeypatch: The fixture used to replace the module-level discovery, loading, and registry dependencies.
        sessions: The session names to create for each animal.
        session_types: The session type to report for individual sessions, keyed by session name. Sessions absent
            from the mapping report the Mesoscope experiment type.
        write_surgery: Determines whether each created session carries a surgery metadata snapshot.

    Returns:
        The path to the created project root.
    """
    resolved_types = session_types or {}
    project_root = tmp_path.joinpath("test_project")
    session_paths: list[Path] = []
    for animal, session_names in sessions.items():
        for session_name in session_names:
            session_path = project_root.joinpath(animal, session_name)
            session_path.mkdir(parents=True)
            if write_surgery:
                session_path.joinpath(_SURGERY_FILENAME).write_text(f"animal: {animal}")
            session_paths.append(session_path)

    def _discover_sessions(root_path: Path) -> list[Path]:  # noqa: ARG001
        """Returns the sessions the project was seeded with, standing in for marker-based discovery."""
        return list(session_paths)

    def _load(session_path: Path) -> SimpleNamespace:
        """Returns a stand-in session carrying the fields the resolution policy reads."""
        return SimpleNamespace(
            session_type=resolved_types.get(session_path.name, SessionTypes.MESOSCOPE_EXPERIMENT),
            acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
            raw_data=SimpleNamespace(surgery_metadata_path=session_path.joinpath(_SURGERY_FILENAME)),
        )

    monkeypatch.setattr(dataset_module, "discover_sessions", _discover_sessions)
    monkeypatch.setattr(dataset_module, "SessionData", SimpleNamespace(load=_load))
    monkeypatch.setattr(
        dataset_module,
        "resolve_forging_column_descriptions",
        lambda system: _COLUMN_DESCRIPTIONS,  # noqa: ARG005
    )
    return project_root


def _membership(dataset: DatasetData) -> dict[str, set[str]]:
    """Returns the dataset's session names grouped by owning animal."""
    membership: dict[str, set[str]] = {}
    for entry in dataset.sessions:
        membership.setdefault(entry.animal, set()).add(entry.session)
    return membership


# Tests for dataset creation and extension


def test_resolve_dataset_creates_hierarchy_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dataset that does not exist is created from the provided session list."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})

    dataset = resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    assert _membership(dataset) == {"animal_a": {"session_1"}}
    assert project_root.joinpath("test_dataset", "animal_a", _SURGERY_FILENAME).is_file()


def test_resolve_dataset_forges_without_optional_surgery_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an animal whose sessions carry no surgery metadata is still forged into the dataset."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]}, write_surgery=False)

    dataset = resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    assert _membership(dataset) == {"animal_a": {"session_1"}}
    assert not project_root.joinpath("test_dataset", "animal_a", _SURGERY_FILENAME).exists()


def test_resolve_dataset_appends_sessions_of_a_new_animal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that sessions of an animal the dataset does not hold are appended to it."""
    project_root = _install_project(
        tmp_path, monkeypatch, {"animal_a": ["session_1"], "animal_b": ["session_2", "session_3"]}
    )
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    extended = resolve_dataset(name="test_dataset", session_names=("session_2", "session_3"), project_root=project_root)

    assert _membership(extended) == {"animal_a": {"session_1"}, "animal_b": {"session_2", "session_3"}}
    assert project_root.joinpath("test_dataset", "animal_b", _SURGERY_FILENAME).is_file()

    # The marker is rewritten, so a fresh load agrees with the returned instance.
    reloaded = DatasetData.load(dataset_path=project_root.joinpath("test_dataset"))
    assert _membership(reloaded) == _membership(extended)


def test_resolve_dataset_leaves_membership_alone_when_sessions_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that providing sessions the dataset already holds changes nothing."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1", "session_2"]})
    resolve_dataset(name="test_dataset", session_names=("session_1", "session_2"), project_root=project_root)

    # A strict subset of the dataset's own sessions is a no-op rather than a redefinition.
    resolved = resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    assert _membership(resolved) == {"animal_a": {"session_1", "session_2"}}


def test_resolve_dataset_rejects_widening_a_frozen_animal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that adding a session to an animal already in the dataset is rejected."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1", "session_2"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="rebuilt as a whole"):
        resolve_dataset(name="test_dataset", session_names=("session_2",), project_root=project_root)

    reloaded = DatasetData.load(dataset_path=project_root.joinpath("test_dataset"))
    assert _membership(reloaded) == {"animal_a": {"session_1"}}


def test_resolve_dataset_rejects_a_session_of_a_differing_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an added session whose type differs from the dataset's is rejected."""
    project_root = _install_project(
        tmp_path,
        monkeypatch,
        {"animal_a": ["session_1"], "animal_b": ["session_2"]},
        session_types={"session_2": SessionTypes.RUN_TRAINING},
    )
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="must share the same session"):
        resolve_dataset(name="test_dataset", session_names=("session_2",), project_root=project_root)


def test_resolve_dataset_errors_when_absent_without_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dataset that does not exist cannot be resolved without a session list."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})

    with pytest.raises(ValueError, match="no sessions were provided"):
        resolve_dataset(name="test_dataset", session_names=(), project_root=project_root)


# Tests for the whole-dataset and per-animal rebuild paths


def test_resolve_dataset_force_recreate_rebuilds_an_unchanged_session_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that force_recreate rebuilds the dataset even when the provided list matches the existing one."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    # A stray artifact inside the hierarchy disappears only if the dataset is genuinely deleted and rebuilt.
    stray = project_root.joinpath("test_dataset", "animal_a", "session_1", "data.feather")
    stray.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name="test_dataset", session_names=("session_1",), project_root=project_root, force_recreate=True
    )

    assert _membership(rebuilt) == {"animal_a": {"session_1"}}
    assert not stray.exists()


def test_resolve_dataset_force_recreate_requires_a_session_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that force_recreate without a session list is rejected instead of deleting the dataset."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="requires a session list"):
        resolve_dataset(name="test_dataset", session_names=(), project_root=project_root, force_recreate=True)

    assert project_root.joinpath("test_dataset", "dataset.yaml").is_file()


def test_resolve_dataset_rebuilds_a_named_animal_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding one animal replaces its session set while every other animal is untouched."""
    project_root = _install_project(
        tmp_path,
        monkeypatch,
        {"animal_a": ["session_1", "session_2", "session_3"], "animal_b": ["session_4"]},
    )
    resolve_dataset(
        name="test_dataset", session_names=("session_1", "session_2", "session_4"), project_root=project_root
    )

    # Marks the untouched animal's data so the rebuild can be shown to leave it in place.
    untouched = project_root.joinpath("test_dataset", "animal_b", "session_4", "data.feather")
    untouched.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name="test_dataset",
        session_names=("session_1", "session_3"),
        project_root=project_root,
        recreate_animals=("animal_a",),
    )

    assert _membership(rebuilt) == {"animal_a": {"session_1", "session_3"}, "animal_b": {"session_4"}}
    assert untouched.read_bytes() == b"payload"
    assert not project_root.joinpath("test_dataset", "animal_a", "session_2").exists()
    assert project_root.joinpath("test_dataset", "animal_a", _SURGERY_FILENAME).is_file()


def test_resolve_dataset_rebuilds_a_named_animal_with_an_unchanged_session_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding an animal is allowed when its provided session set matches the existing one."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)
    stray = project_root.joinpath("test_dataset", "animal_a", "session_1", "data.feather")
    stray.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name="test_dataset",
        session_names=("session_1",),
        project_root=project_root,
        recreate_animals=("animal_a",),
    )

    assert _membership(rebuilt) == {"animal_a": {"session_1"}}
    assert not stray.exists()


def test_resolve_dataset_rejects_rebuilding_an_animal_absent_from_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that naming an animal the dataset does not hold for rebuilding is rejected."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"], "animal_b": ["session_2"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="Every animal named for rebuilding"):
        resolve_dataset(
            name="test_dataset",
            session_names=("session_2",),
            project_root=project_root,
            recreate_animals=("animal_b",),
        )


def test_resolve_dataset_rejects_rebuilding_an_animal_without_its_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding an animal requires the provided list to hold at least one of its sessions."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"], "animal_b": ["session_2"]})
    resolve_dataset(name="test_dataset", session_names=("session_1", "session_2"), project_root=project_root)

    with pytest.raises(ValueError, match="at least one session for it"):
        resolve_dataset(
            name="test_dataset",
            session_names=("session_2",),
            project_root=project_root,
            recreate_animals=("animal_a",),
        )


def test_resolve_dataset_rejects_combining_the_two_rebuild_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that force_recreate and recreate_animals cannot be combined."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_dataset(
            name="test_dataset",
            session_names=("session_1",),
            project_root=project_root,
            force_recreate=True,
            recreate_animals=("animal_a",),
        )


def test_resolve_dataset_rejects_rebuilding_an_animal_without_a_session_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding an animal without any provided sessions is rejected."""
    project_root = _install_project(tmp_path, monkeypatch, {"animal_a": ["session_1"]})
    resolve_dataset(name="test_dataset", session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="the session list must define that set"):
        resolve_dataset(
            name="test_dataset", session_names=(), project_root=project_root, recreate_animals=("animal_a",)
        )


# Tests for the forging pipeline's tracker-state helpers


def _make_tracker(tmp_path: Path, universe: list[tuple[str, str]]) -> ProcessingTracker:
    """Returns a forging tracker aligned against the provided job universe."""
    tracker = ProcessingTracker(file_path=tmp_path.joinpath("forging_tracker.yaml"))
    tracker.align_jobs(jobs=universe, universe=universe)
    return tracker


def test_resolve_runnable_jobs_reports_the_whole_universe_for_an_unwritten_tracker(tmp_path: Path) -> None:
    """Verifies that a tracker that has never been written reports every job as outstanding."""
    universe = [(MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"), (FORGING_JOB_NAME, "session_1")]
    tracker = ProcessingTracker(file_path=tmp_path.joinpath("forging_tracker.yaml"))

    assert _resolve_runnable_jobs(tracker=tracker, universe=universe) == universe
    assert not tracker.file_path.exists()


def test_resolve_runnable_jobs_excludes_succeeded_jobs(tmp_path: Path) -> None:
    """Verifies that a job the tracker records as succeeded is left out of the outstanding subset."""
    universe = [(FORGING_JOB_NAME, "session_1"), (FORGING_JOB_NAME, "session_2")]
    tracker = _make_tracker(tmp_path, universe)
    succeeded = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier="session_1")
    tracker.start_job(job_id=succeeded)
    tracker.complete_job(job_id=succeeded)

    assert _resolve_runnable_jobs(tracker=tracker, universe=universe) == [(FORGING_JOB_NAME, "session_2")]


def test_resolve_runnable_jobs_includes_failed_jobs(tmp_path: Path) -> None:
    """Verifies that a job the tracker records as failed stays outstanding so it is retried."""
    universe = [(FORGING_JOB_NAME, "session_1")]
    tracker = _make_tracker(tmp_path, universe)
    failed = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier="session_1")
    tracker.start_job(job_id=failed)
    tracker.fail_job(job_id=failed, error_message="assembly failed")

    assert _resolve_runnable_jobs(tracker=tracker, universe=universe) == universe


def test_reset_animal_jobs_targets_only_the_named_animals(tmp_path: Path) -> None:
    """Verifies that a rebuilt animal's stages are reset while every other animal keeps its recorded state."""
    universe = [
        (MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_1"),
        (FORGING_JOB_NAME, "session_1"),
        (MULTIDAY_DISCOVERY_JOB_NAME, "animal_b"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_2"),
        (FORGING_JOB_NAME, "session_2"),
    ]
    tracker = _make_tracker(tmp_path, universe)
    for job_name, specifier in universe:
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    dataset = SimpleNamespace(
        get_sessions_for_animal=lambda animal: (
            (SimpleNamespace(session="session_1"),) if animal == "animal_a" else (SimpleNamespace(session="session_2"),)
        )
    )
    _reset_animal_jobs(tracker=tracker, dataset=dataset, animals=("animal_a",))

    snapshot = tracker.snapshot()
    reset_jobs = {
        (state.job_name, state.specifier) for state in snapshot.values() if state.status == ProcessingStatus.SCHEDULED
    }
    assert reset_jobs == {
        (MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_1"),
        (FORGING_JOB_NAME, "session_1"),
    }


def test_reset_animal_jobs_leaves_an_unwritten_tracker_alone(tmp_path: Path) -> None:
    """Verifies that resetting against a tracker that has never been written creates no tracker file."""
    tracker = ProcessingTracker(file_path=tmp_path.joinpath("forging_tracker.yaml"))
    dataset = SimpleNamespace(get_sessions_for_animal=lambda animal: ())  # noqa: ARG005

    _reset_animal_jobs(tracker=tracker, dataset=dataset, animals=("animal_a",))

    assert not tracker.file_path.exists()


# Tests for the dataset-definition entry point


def _record_resolution(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replaces the definition entry point's dataset resolver with a recorder that halts once it captures arguments.

    Args:
        monkeypatch: The fixture used to replace the resolver the pipeline module calls.

    Returns:
        The mapping the recorder fills with the keyword arguments the entry point passed to the resolver.
    """
    recorded: dict[str, Any] = {}

    def _resolve(**arguments: Any) -> None:
        recorded.update(arguments)
        message = "halted after dataset resolution"
        raise RuntimeError(message)

    monkeypatch.setattr(pipeline_module, "resolve_dataset", _resolve)
    return recorded


def test_define_forging_dataset_applies_every_definition_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the definition entry point forwards the session list and the rebuild arguments."""
    recorded = _record_resolution(monkeypatch)

    with pytest.raises(RuntimeError, match="halted"):
        define_forging_dataset(
            name="test_dataset",
            session_names=("session_1", "session_2"),
            project_root=tmp_path,
            force_recreate=True,
        )

    assert recorded["session_names"] == ("session_1", "session_2")
    assert recorded["force_recreate"] is True


def test_run_forging_pipeline_never_redefines_the_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that running the pipeline loads the dataset without changing its session set.

    The hierarchy is built by the definition entry point before any job is prepared, so a run that could widen it
    would let a dispatched job mutate the universe its siblings were planned against.
    """
    recorded = _record_resolution(monkeypatch)

    with pytest.raises(RuntimeError, match="halted"):
        run_forging_pipeline(name="test_dataset", project_root=tmp_path)

    assert recorded["session_names"] == ()


# Tests for the forging job universe and its ordering


_MULTIDAY_PLAN: dict[str, tuple[Path, list[str]]] = {
    "animal_a": (Path("animal_a/multi_recording_configuration.yaml"), ["session_1", "session_2"]),
    "animal_b": (Path("animal_b/multi_recording_configuration.yaml"), ["session_3"]),
}
"""A two-animal multi-day plan standing in for what a defined dataset leaves on disk."""

_SESSION_ANIMALS: dict[str, str] = {"session_1": "animal_a", "session_2": "animal_a", "session_3": "animal_b"}
"""The animal owning each session in the plan, which the ordering resolves an extraction's discovery through."""


def _dataset_stub(*sessions: str) -> Any:
    """Returns a dataset stub carrying the session and animal pairs the universe and ordering read."""
    return SimpleNamespace(
        sessions=tuple(
            SimpleNamespace(session=session, animal=_SESSION_ANIMALS.get(session, "animal_a")) for session in sessions
        )
    )


def test_build_forging_universe_covers_every_stage() -> None:
    """Verifies that the universe holds each animal's discovery alongside every extraction and assembly."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")

    assert build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN) == [
        (MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_1"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_2"),
        (MULTIDAY_DISCOVERY_JOB_NAME, "animal_b"),
        (MULTIDAY_EXTRACTION_JOB_NAME, "session_3"),
        (FORGING_JOB_NAME, "session_1"),
        (FORGING_JOB_NAME, "session_2"),
        (FORGING_JOB_NAME, "session_3"),
    ]


def test_build_forging_universe_omits_multiday_jobs_without_a_plan() -> None:
    """Verifies that a dataset needing no multi-day processing carries assembly jobs alone."""
    dataset = _dataset_stub("session_1")

    assert build_forging_universe(dataset=dataset, multiday_plan={}) == [(FORGING_JOB_NAME, "session_1")]


def test_forging_job_prerequisites_chains_each_session_through_its_own_animal() -> None:
    """Verifies that an extraction waits on its own animal's discovery and an assembly waits on its extraction."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)

    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    assert ordering[MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"] == ()
    assert ordering[MULTIDAY_EXTRACTION_JOB_NAME, "session_2"] == ((MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"),)
    assert ordering[MULTIDAY_EXTRACTION_JOB_NAME, "session_3"] == ((MULTIDAY_DISCOVERY_JOB_NAME, "animal_b"),)
    assert ordering[FORGING_JOB_NAME, "session_3"] == ((MULTIDAY_EXTRACTION_JOB_NAME, "session_3"),)


def test_forging_job_prerequisites_leaves_a_training_assembly_unordered() -> None:
    """Verifies that a session with no multi-day stage carries no upstream job."""
    dataset = _dataset_stub("session_1", "session_2")
    universe = build_forging_universe(dataset=dataset, multiday_plan={})

    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    assert ordering[FORGING_JOB_NAME, "session_1"] == ()
    assert ordering[FORGING_JOB_NAME, "session_2"] == ()


def test_forging_job_prerequisites_covers_the_whole_universe() -> None:
    """Verifies that every job in the universe receives an ordering entry."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)

    assert set(forging_job_prerequisites(dataset=dataset, universe=universe)) == set(universe)


def test_every_multiday_session_reaches_its_own_animals_discovery() -> None:
    """Verifies that each assembly's upstream chain terminates at the discovery of the animal owning its session."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)
    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    for session, animal in _SESSION_ANIMALS.items():
        chain = [(FORGING_JOB_NAME, session)]
        while ordering[chain[-1]]:
            chain.append(ordering[chain[-1]][0])
        assert chain[-1] == (MULTIDAY_DISCOVERY_JOB_NAME, animal), f"{session} does not reach its animal's discovery"
