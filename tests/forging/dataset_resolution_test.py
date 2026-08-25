"""Contains tests for the forging dataset resolution policy, the forging job universe and its ordering, the admission
gate, and the forging pipeline's tracker-state helpers.
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
    run_forging_pipeline,
    define_forging_dataset,
    discover_project_datasets,
    forging_job_prerequisites,
)
from sollertia_forgery.shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    resolve_session_tracker_path,
)
import sollertia_forgery.forging.dataset as dataset_module
from sollertia_forgery.forging.dataset import resolve_dataset, _verify_session_compatibility
import sollertia_forgery.forging.pipeline as pipeline_module
from sollertia_forgery.forging.pipeline import _reset_animal_jobs, _resolve_runnable_jobs, _build_forging_universe
from sollertia_forgery.forging.admission import verify_session_admissibility

_COLUMN_DESCRIPTIONS: dict[str, str] = {"time_us": "Microsecond-precision sample timestamps."}
"""A minimal column-description binding, standing in for what the acquisition system's registry entry returns."""

_SURGERY_FILENAME: str = "surgery_metadata.yaml"
"""The per-animal metadata filename the resolution policy copies into each animal's dataset directory."""

_DATASET_NAME: str = "test_dataset"
"""The dataset name every resolution test defines and re-resolves."""

_FORGING_TRACKER_FILENAME: str = "forging_tracker.yaml"
"""The tracker filename the pipeline helpers read and write under the test's temporary directory."""

_ADMISSION_TRACKERS: dict[ProcessingPipelines, tuple[str, str]] = {
    ProcessingPipelines.CHECKSUM: ("raw_data", "checksum_tracker.yaml"),
    ProcessingPipelines.RUNTIME: ("processed_data", "runtime_tracker.yaml"),
    ProcessingPipelines.MICROCONTROLLER: ("processed_data", "microcontroller_tracker.yaml"),
    ProcessingPipelines.VIDEO: ("processed_data", "video_tracker.yaml"),
    ProcessingPipelines.TWO_PHOTON: ("processed_data", "two_photon_tracker.yaml"),
}
"""The tracker location the stand-in session reports for each pipeline forging admission can require. The stand-in
replaces the shared hierarchy's own accessors, so these names need only agree between the writer and the reader here."""

_MULTIDAY_PLAN: dict[str, tuple[Path, list[str]]] = {
    "animal_a": (Path("animal_a/multi_recording_configuration.yaml"), ["session_1", "session_2"]),
    "animal_b": (Path("animal_b/multi_recording_configuration.yaml"), ["session_3"]),
}
"""A two-animal multi-day plan standing in for what a defined dataset leaves on disk."""

_SESSION_ANIMALS: dict[str, str] = {"session_1": "animal_a", "session_2": "animal_a", "session_3": "animal_b"}
"""The animal owning each session in the plan, which the ordering resolves an extraction's discovery through."""


def _tracker_path(session_path: Path, pipeline: ProcessingPipelines) -> Path:
    """Returns the stand-in tracker path one pipeline records against for a session."""
    directory, filename = _ADMISSION_TRACKERS[pipeline]
    return session_path.joinpath(directory, filename)


def _mark_processed(session_path: Path) -> None:
    """Writes a completed tracker for every pipeline forging admission can require.

    Admission holds a session out of a dataset until its required pipelines report every job as succeeded, so a
    session standing in for a processed one has to carry those trackers.
    """
    for pipeline in _ADMISSION_TRACKERS:
        tracker_path = _tracker_path(session_path=session_path, pipeline=pipeline)
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker = ProcessingTracker(file_path=tracker_path)
        jobs = [(f"{pipeline.value}_stage", "")]
        tracker.align_jobs(jobs=jobs, universe=jobs)
        job_id = ProcessingTracker.generate_job_id(job_name=f"{pipeline.value}_stage", specifier="")
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)


def _install_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, list[str]],
    session_types: dict[str, SessionTypes] | None = None,
    unprocessed: frozenset[str] = frozenset(),
    acquisition_systems: dict[str, str] | None = None,
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
        unprocessed: The session names to leave without completed processing trackers, which forging admission
            rejects. Every other session is marked as fully processed.
        acquisition_systems: The acquisition system to report for individual sessions, keyed by session name.
            Sessions absent from the mapping report the Mesoscope-VR system.
        write_surgery: Determines whether each created session carries a surgery metadata snapshot.

    Returns:
        The path to the created project root.
    """
    resolved_types = session_types or {}
    resolved_systems = acquisition_systems or {}
    project_root = tmp_path.joinpath("test_project")
    session_paths: list[Path] = []
    for animal, session_names in sessions.items():
        for session_name in session_names:
            session_path = project_root.joinpath(animal, session_name)
            session_path.mkdir(parents=True)
            if write_surgery:
                session_path.joinpath(_SURGERY_FILENAME).write_text(f"animal: {animal}")
            if session_name not in unprocessed:
                _mark_processed(session_path=session_path)
            session_paths.append(session_path)

    def _discover_sessions(root_path: Path) -> list[Path]:
        """Returns the sessions the project was seeded with, standing in for marker-based discovery."""
        return list(session_paths)

    def _load(session_path: Path) -> SimpleNamespace:
        """Returns a stand-in session carrying the fields the resolution policy reads."""
        return SimpleNamespace(
            session_name=session_path.name,
            session_type=resolved_types.get(session_path.name, SessionTypes.MESOSCOPE_EXPERIMENT),
            acquisition_system=resolved_systems.get(session_path.name, AcquisitionSystems.MESOSCOPE_VR),
            raw_data=SimpleNamespace(
                surgery_metadata_path=session_path.joinpath(_SURGERY_FILENAME),
                checksum_tracker_path=_tracker_path(session_path=session_path, pipeline=ProcessingPipelines.CHECKSUM),
            ),
            processed_data=SimpleNamespace(
                runtime_tracker_path=_tracker_path(session_path=session_path, pipeline=ProcessingPipelines.RUNTIME),
                microcontroller_tracker_path=_tracker_path(
                    session_path=session_path, pipeline=ProcessingPipelines.MICROCONTROLLER
                ),
                video_tracker_path=_tracker_path(session_path=session_path, pipeline=ProcessingPipelines.VIDEO),
                two_photon_tracker_path=_tracker_path(
                    session_path=session_path, pipeline=ProcessingPipelines.TWO_PHOTON
                ),
            ),
        )

    monkeypatch.setattr(target=dataset_module, name="discover_sessions", value=_discover_sessions)
    monkeypatch.setattr(target=dataset_module, name="SessionData", value=SimpleNamespace(load=_load))
    monkeypatch.setattr(
        target=dataset_module,
        name="resolve_forging_column_descriptions",
        value=lambda system: _COLUMN_DESCRIPTIONS,  # noqa: ARG005
    )
    return project_root


def _group_sessions_by_animal(dataset: DatasetData) -> dict[str, set[str]]:
    """Returns the dataset's session names grouped by owning animal."""
    membership: dict[str, set[str]] = {}
    for entry in dataset.sessions:
        membership.setdefault(entry.animal, set()).add(entry.session)
    return membership


@pytest.fixture
def single_session_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Creates a one-animal, one-session source project with the dataset already defined."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)
    return project_root


# Tests for dataset creation and extension


def test_resolve_dataset_creates_hierarchy_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dataset that does not exist is created from the provided session list."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    dataset = resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    assert _group_sessions_by_animal(dataset) == {"animal_a": {"session_1"}}
    assert project_root.joinpath(_DATASET_NAME, "animal_a", _SURGERY_FILENAME).is_file()


def test_resolve_dataset_forges_without_optional_surgery_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an animal whose sessions carry no surgery metadata is still forged into the dataset."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]}, write_surgery=False
    )

    dataset = resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    assert _group_sessions_by_animal(dataset) == {"animal_a": {"session_1"}}
    assert not project_root.joinpath(_DATASET_NAME, "animal_a", _SURGERY_FILENAME).exists()


def test_resolve_dataset_copies_the_surgery_snapshot_of_every_covered_animal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a dataset created over several animals receives each animal's own surgery metadata snapshot.

    The snapshot is per-animal provenance, and an animal whose source file is missing is only reported as a warning,
    so an animal never visited at all would leave the dataset silently.
    """
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]}
    )

    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    assert project_root.joinpath(_DATASET_NAME, "animal_a", _SURGERY_FILENAME).read_text() == "animal: animal_a"
    assert project_root.joinpath(_DATASET_NAME, "animal_b", _SURGERY_FILENAME).read_text() == "animal: animal_b"


def test_resolve_dataset_copies_the_surgery_snapshot_of_an_animals_latest_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an animal's snapshot is taken from its most recent source session rather than its earliest.

    Session names are timestamped, so the last of an animal's sessions in natural order is the most recent one and
    carries the surgery record closest to the sessions the dataset is forged over.
    """
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1", "session_2"]}
    )
    for session_name in ("session_1", "session_2"):
        project_root.joinpath("animal_a", session_name, _SURGERY_FILENAME).write_text(f"session: {session_name}")

    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    assert project_root.joinpath(_DATASET_NAME, "animal_a", _SURGERY_FILENAME).read_text() == "session: session_2"


def test_resolve_dataset_appends_sessions_of_a_new_animal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that sessions of an animal the dataset does not hold are appended to it."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2", "session_3"]},
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    extended = resolve_dataset(name=_DATASET_NAME, session_names=("session_2", "session_3"), project_root=project_root)

    assert _group_sessions_by_animal(extended) == {"animal_a": {"session_1"}, "animal_b": {"session_2", "session_3"}}
    assert project_root.joinpath(_DATASET_NAME, "animal_b", _SURGERY_FILENAME).is_file()

    # The marker is rewritten, so a fresh load agrees with the returned instance.
    reloaded = DatasetData.load(dataset_path=project_root.joinpath(_DATASET_NAME))
    assert _group_sessions_by_animal(reloaded) == _group_sessions_by_animal(extended)


def test_resolve_dataset_leaves_membership_alone_when_sessions_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that providing sessions the dataset already holds changes nothing."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1", "session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    # A strict subset of the dataset's own sessions is a no-op.
    resolved = resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    assert _group_sessions_by_animal(resolved) == {"animal_a": {"session_1", "session_2"}}


def test_resolve_dataset_rejects_widening_a_frozen_animal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that adding a session to an animal already in the dataset is rejected."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1", "session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="rebuilt as a whole"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_2",), project_root=project_root)

    reloaded = DatasetData.load(dataset_path=project_root.joinpath(_DATASET_NAME))
    assert _group_sessions_by_animal(reloaded) == {"animal_a": {"session_1"}}


def test_resolve_dataset_rejects_an_unprocessed_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a session carrying no completed trackers never enters a dataset."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"]},
        unprocessed=frozenset({"session_1"}),
    )

    with pytest.raises(ValueError, match="Unable to admit session"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    # Rejection precedes hierarchy creation, so a refused definition leaves nothing behind.
    assert not project_root.joinpath(_DATASET_NAME).exists()


def test_resolve_dataset_rejects_an_unprocessed_session_the_list_does_not_begin_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that every session a dataset is created from is screened, not only the one its metadata comes from.

    The admission gate is the only thing standing between an unfinished pipeline and a forged dataset, so a session
    named after the first has to hold the whole definition out just as the first one does.
    """
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]},
        unprocessed=frozenset({"session_2"}),
    )

    with pytest.raises(ValueError, match="Unable to admit session"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    # Rejection precedes hierarchy creation, so a refused definition leaves nothing behind.
    assert not project_root.joinpath(_DATASET_NAME).exists()


def test_resolve_dataset_rejects_a_session_whose_pipeline_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that one failed job in one required pipeline is enough to hold a session out."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})
    tracker_path = _tracker_path(
        session_path=project_root.joinpath("animal_a", "session_1"), pipeline=ProcessingPipelines.VIDEO
    )
    tracker = ProcessingTracker(file_path=tracker_path)
    failing_id = next(iter(tracker.snapshot()))
    tracker.fail_job(job_id=failing_id, error_message="motion energy failed")

    with pytest.raises(ValueError, match="Unable to admit session"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)


def test_resolve_dataset_admits_a_training_session_without_imaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a training session joins a dataset with no two-photon tracker, since it records no imaging."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"]},
        session_types={"session_1": SessionTypes.RUN_TRAINING},
    )
    _tracker_path(
        session_path=project_root.joinpath("animal_a", "session_1"), pipeline=ProcessingPipelines.TWO_PHOTON
    ).unlink()

    dataset = resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    assert _group_sessions_by_animal(dataset) == {"animal_a": {"session_1"}}


def test_resolve_dataset_rejects_a_session_of_a_differing_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an added session whose type differs from the dataset's is rejected."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]},
        session_types={"session_2": SessionTypes.RUN_TRAINING},
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="must share the same session"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_2",), project_root=project_root)


@pytest.mark.parametrize(
    ("installation", "message"),
    [
        ({"session_types": {"session_3": SessionTypes.RUN_TRAINING}}, r"must\s+share\s+the\s+same\s+session\s+type"),
        ({"unprocessed": frozenset({"session_3"})}, r"Unable\s+to\s+admit\s+session"),
    ],
)
def test_resolve_dataset_screens_every_added_session_rather_than_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installation: dict[str, Any], message: str
) -> None:
    """Verifies that an extension naming several sessions screens each of them against the dataset it joins.

    The dataset marker keeps claiming one session type and one acquisition system, so a second added session the
    check never reaches leaves the membership mixed while the marker still reads as uniform.
    """
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2", "session_3"]},
        **installation,
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match=message):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_2", "session_3"), project_root=project_root)

    reloaded = DatasetData.load(dataset_path=project_root.joinpath(_DATASET_NAME))
    assert _group_sessions_by_animal(reloaded) == {"animal_a": {"session_1"}}


def test_resolve_dataset_errors_when_absent_without_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a dataset that does not exist cannot be resolved without a session list."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    # The console wraps the rendered message at a width that depends on the temporary path length, so the pattern
    # tolerates a line break between any two words of the phrase.
    with pytest.raises(ValueError, match=r"no\s+sessions\s+were\s+provided"):
        resolve_dataset(name=_DATASET_NAME, session_names=(), project_root=project_root)


def test_resolve_dataset_loads_an_existing_dataset_without_a_session_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an existing dataset resolved with no session list is returned with its membership untouched."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1", "session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    resolved = resolve_dataset(name=_DATASET_NAME, session_names=(), project_root=project_root)

    assert _group_sessions_by_animal(resolved) == {"animal_a": {"session_1", "session_2"}}
    assert resolved.name == _DATASET_NAME


def test_resolve_dataset_rejects_rebuilding_an_animal_of_an_absent_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding an animal of a dataset that was never created is rejected before anything is built."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    with pytest.raises(ValueError, match="Rebuilding an animal requires an existing"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_1",),
            project_root=project_root,
            recreate_animals=("animal_a",),
        )

    assert not project_root.joinpath(_DATASET_NAME).exists()


def test_resolve_dataset_accepts_the_required_session_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that restricting creation to one session type admits a dataset whose first session carries it."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"]},
        session_types={"session_1": SessionTypes.RUN_TRAINING},
    )

    dataset = resolve_dataset(
        name=_DATASET_NAME,
        session_names=("session_1",),
        project_root=project_root,
        required_session_type=SessionTypes.RUN_TRAINING,
    )

    assert dataset.session_type == SessionTypes.RUN_TRAINING


def test_resolve_dataset_rejects_a_first_session_of_the_wrong_required_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that restricting creation to one session type rejects a first session carrying another."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    with pytest.raises(ValueError, match="acquisition system is supported only"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_1",),
            project_root=project_root,
            required_session_type=SessionTypes.RUN_TRAINING,
        )

    assert not project_root.joinpath(_DATASET_NAME).exists()


def test_resolve_dataset_rejects_a_created_dataset_mixing_session_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that creating a dataset from sessions of differing types is rejected on the divergent session."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]},
        session_types={"session_2": SessionTypes.RUN_TRAINING},
    )

    with pytest.raises(ValueError, match="has type 'run training' while the first session has type"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    assert not project_root.joinpath(_DATASET_NAME).exists()


def test_resolve_dataset_reports_a_session_name_matching_no_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an unknown session name is reported as missing rather than silently dropped."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    with pytest.raises(FileNotFoundError, match="but found 0"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_9",), project_root=project_root)


def test_resolve_dataset_reports_a_session_name_matching_two_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a session name held by two animals is ambiguous, since the policy resolves names project-wide."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_1"]}
    )

    with pytest.raises(RuntimeError, match="but found 2"):
        resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)


def test_create_dataset_rejects_a_session_of_a_differing_acquisition_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that creating a dataset from sessions of two acquisition systems is rejected.

    Only one acquisition system is registered, so admission rejects a divergent value before the creator compares
    it. The creator is therefore driven directly with admission held open, which is what a second registered system
    would reach through the public policy.
    """
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]},
        acquisition_systems={"session_2": "other_system"},
    )
    monkeypatch.setattr(
        target=dataset_module,
        name="verify_session_admissibility",
        value=lambda session: None,  # noqa: ARG005
    )

    with pytest.raises(ValueError, match="session 'session_2' was acquired by 'other_system'"):
        dataset_module._create_dataset(
            name=_DATASET_NAME, sessions=("session_1", "session_2"), project_root=project_root
        )


def test_verify_session_compatibility_rejects_a_differing_acquisition_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a session acquired by another system never joins an existing dataset.

    The dataset's own recorded system is moved off the session's, which is the state a dataset forged under a second
    acquisition system would present to the compatibility check.
    """
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]}
    )
    dataset = resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)
    dataset.acquisition_system = "other_system"

    with pytest.raises(ValueError, match="the dataset was acquired by 'other_system'"):
        _verify_session_compatibility(dataset=dataset, session_paths=[project_root.joinpath("animal_b", "session_2")])


def test_discover_project_datasets_loads_every_marked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that discovery returns the datasets a project holds, ordered by name, and skips its animal trees."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]}
    )
    resolve_dataset(name="second_dataset", session_names=("session_2",), project_root=project_root)
    resolve_dataset(name="first_dataset", session_names=("session_1",), project_root=project_root)

    discovered = discover_project_datasets(project_root=project_root)

    assert [dataset.name for dataset in discovered] == ["first_dataset", "second_dataset"]
    assert _group_sessions_by_animal(dataset=discovered[0]) == {"animal_a": {"session_1"}}


def test_discover_project_datasets_reports_nothing_for_a_project_without_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a project holding animal directories alone reports no datasets."""
    project_root = _install_project(tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"]})

    assert discover_project_datasets(project_root=project_root) == []


# Tests for the whole-dataset and per-animal rebuild paths


def test_resolve_dataset_force_recreate_rebuilds_an_unchanged_session_list(single_session_project: Path) -> None:
    """Verifies that force_recreate rebuilds the dataset even when the provided list matches the existing one."""
    project_root = single_session_project

    # A stray artifact inside the hierarchy disappears only if the dataset is genuinely deleted and rebuilt.
    stray = project_root.joinpath(_DATASET_NAME, "animal_a", "session_1", "data.feather")
    stray.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name=_DATASET_NAME, session_names=("session_1",), project_root=project_root, force_recreate=True
    )

    assert _group_sessions_by_animal(rebuilt) == {"animal_a": {"session_1"}}
    assert not stray.exists()


def test_resolve_dataset_force_recreate_requires_a_session_list(single_session_project: Path) -> None:
    """Verifies that force_recreate without a session list is rejected, leaving the dataset on disk."""
    project_root = single_session_project

    with pytest.raises(ValueError, match="requires a session list"):
        resolve_dataset(name=_DATASET_NAME, session_names=(), project_root=project_root, force_recreate=True)

    assert project_root.joinpath(_DATASET_NAME, "dataset.yaml").is_file()


def test_resolve_dataset_rebuilds_a_named_animal_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding one animal replaces its session set while every other animal is untouched."""
    project_root = _install_project(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        sessions={"animal_a": ["session_1", "session_2", "session_3"], "animal_b": ["session_4"]},
    )
    resolve_dataset(
        name=_DATASET_NAME, session_names=("session_1", "session_2", "session_4"), project_root=project_root
    )

    # Marks the untouched animal's data so the rebuild can be shown to leave it in place.
    untouched = project_root.joinpath(_DATASET_NAME, "animal_b", "session_4", "data.feather")
    untouched.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name=_DATASET_NAME,
        session_names=("session_1", "session_3"),
        project_root=project_root,
        recreate_animals=("animal_a",),
    )

    assert _group_sessions_by_animal(rebuilt) == {"animal_a": {"session_1", "session_3"}, "animal_b": {"session_4"}}
    assert untouched.read_bytes() == b"payload"
    assert not project_root.joinpath(_DATASET_NAME, "animal_a", "session_2").exists()
    assert project_root.joinpath(_DATASET_NAME, "animal_a", _SURGERY_FILENAME).is_file()


def test_resolve_dataset_rebuilds_a_named_animal_with_an_unchanged_session_set(
    single_session_project: Path,
) -> None:
    """Verifies that rebuilding an animal is allowed when its provided session set matches the existing one."""
    project_root = single_session_project
    stray = project_root.joinpath(_DATASET_NAME, "animal_a", "session_1", "data.feather")
    stray.write_bytes(b"payload")

    rebuilt = resolve_dataset(
        name=_DATASET_NAME,
        session_names=("session_1",),
        project_root=project_root,
        recreate_animals=("animal_a",),
    )

    assert _group_sessions_by_animal(rebuilt) == {"animal_a": {"session_1"}}
    assert not stray.exists()


def test_resolve_dataset_rejects_rebuilding_an_animal_absent_from_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that naming an animal the dataset does not hold for rebuilding is rejected."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1",), project_root=project_root)

    with pytest.raises(ValueError, match="Every animal named for rebuilding"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_2",),
            project_root=project_root,
            recreate_animals=("animal_b",),
        )


def test_resolve_dataset_rejects_naming_one_animal_for_rebuilding_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a repeated rebuild entry is refused before the hierarchy is touched.

    A rebuild removes the animal before anything is added, so a second removal of the same animal would raise after
    the first had already deleted its forged outputs and dropped it from the dataset.
    """
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1", "session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    with pytest.raises(ValueError, match="named for rebuilding at most"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_1",),
            project_root=project_root,
            recreate_animals=("animal_a", "animal_a"),
        )

    # The refusal leaves the dataset exactly as it stood, holding both of the animal's forged sessions.
    dataset = DatasetData.load(dataset_path=project_root.joinpath(_DATASET_NAME))
    assert sorted(entry.session for entry in dataset.sessions) == ["session_1", "session_2"]


def test_resolve_dataset_rejects_rebuilding_an_animal_without_its_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that rebuilding an animal requires the provided list to hold at least one of its sessions."""
    project_root = _install_project(
        tmp_path=tmp_path, monkeypatch=monkeypatch, sessions={"animal_a": ["session_1"], "animal_b": ["session_2"]}
    )
    resolve_dataset(name=_DATASET_NAME, session_names=("session_1", "session_2"), project_root=project_root)

    with pytest.raises(ValueError, match="at least one session for it"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_2",),
            project_root=project_root,
            recreate_animals=("animal_a",),
        )


def test_resolve_dataset_rejects_combining_the_two_rebuild_arguments(single_session_project: Path) -> None:
    """Verifies that force_recreate and recreate_animals cannot be combined."""
    project_root = single_session_project

    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_dataset(
            name=_DATASET_NAME,
            session_names=("session_1",),
            project_root=project_root,
            force_recreate=True,
            recreate_animals=("animal_a",),
        )


def test_resolve_dataset_rejects_rebuilding_an_animal_without_a_session_list(single_session_project: Path) -> None:
    """Verifies that rebuilding an animal without any provided sessions is rejected."""
    project_root = single_session_project

    with pytest.raises(ValueError, match="the session list must define that set"):
        resolve_dataset(name=_DATASET_NAME, session_names=(), project_root=project_root, recreate_animals=("animal_a",))


# Tests for the forging pipeline's tracker-state helpers


def _make_tracker(tmp_path: Path, universe: list[tuple[str, str]]) -> ProcessingTracker:
    """Returns a forging tracker aligned against the provided job universe."""
    tracker = ProcessingTracker(file_path=tmp_path.joinpath(_FORGING_TRACKER_FILENAME))
    tracker.align_jobs(jobs=universe, universe=universe)
    return tracker


def test_resolve_runnable_jobs_reports_the_whole_universe_for_an_unwritten_tracker(tmp_path: Path) -> None:
    """Verifies that a tracker that has never been written reports every job as outstanding."""
    universe = [(MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"), (FORGING_JOB_NAME, "session_1")]
    tracker = ProcessingTracker(file_path=tmp_path.joinpath(_FORGING_TRACKER_FILENAME))

    assert _resolve_runnable_jobs(tracker=tracker, universe=universe) == universe
    assert not tracker.file_path.exists()


def test_resolve_runnable_jobs_excludes_succeeded_jobs(tmp_path: Path) -> None:
    """Verifies that a job the tracker records as succeeded is left out of the outstanding subset."""
    universe = [(FORGING_JOB_NAME, "session_1"), (FORGING_JOB_NAME, "session_2")]
    tracker = _make_tracker(tmp_path=tmp_path, universe=universe)
    succeeded = ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier="session_1")
    tracker.start_job(job_id=succeeded)
    tracker.complete_job(job_id=succeeded)

    assert _resolve_runnable_jobs(tracker=tracker, universe=universe) == [(FORGING_JOB_NAME, "session_2")]


def test_resolve_runnable_jobs_includes_failed_jobs(tmp_path: Path) -> None:
    """Verifies that a job the tracker records as failed stays outstanding so it is retried."""
    universe = [(FORGING_JOB_NAME, "session_1")]
    tracker = _make_tracker(tmp_path=tmp_path, universe=universe)
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
    tracker = _make_tracker(tmp_path=tmp_path, universe=universe)
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


def test_reset_animal_jobs_leaves_an_untracked_animal_alone(tmp_path: Path) -> None:
    """Verifies that resetting an animal the tracker holds no job for leaves every recorded state in place."""
    universe = [(MULTIDAY_DISCOVERY_JOB_NAME, "animal_b"), (FORGING_JOB_NAME, "session_2")]
    tracker = _make_tracker(tmp_path=tmp_path, universe=universe)
    for job_name, specifier in universe:
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)
    dataset = SimpleNamespace(get_sessions_for_animal=lambda animal: ())  # noqa: ARG005

    _reset_animal_jobs(tracker=tracker, dataset=dataset, animals=("animal_a",))

    statuses = {state.status for state in tracker.snapshot().values()}
    assert statuses == {ProcessingStatus.SUCCEEDED}


def test_reset_animal_jobs_leaves_an_unwritten_tracker_alone(tmp_path: Path) -> None:
    """Verifies that resetting against a tracker that has never been written creates no tracker file."""
    tracker = ProcessingTracker(file_path=tmp_path.joinpath(_FORGING_TRACKER_FILENAME))
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

    def _resolve(**arguments: object) -> None:
        recorded.update(arguments)
        message = "halted after dataset resolution"
        raise RuntimeError(message)

    monkeypatch.setattr(target=pipeline_module, name="resolve_dataset", value=_resolve)
    return recorded


def test_define_forging_dataset_applies_every_definition_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the definition entry point forwards the session list and the force-recreate flag."""
    recorded = _record_resolution(monkeypatch)

    with pytest.raises(RuntimeError, match="halted"):
        define_forging_dataset(
            name=_DATASET_NAME,
            session_names=("session_1", "session_2"),
            project_root=tmp_path,
            force_recreate=True,
        )

    assert recorded["session_names"] == ("session_1", "session_2")
    assert recorded["force_recreate"]


def test_run_forging_pipeline_never_redefines_the_hierarchy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that running the pipeline loads the dataset without changing its session set.

    The hierarchy is built by the definition entry point before any job is prepared, so a run that could widen it
    would let a dispatched job mutate the universe its siblings were planned against.
    """
    recorded = _record_resolution(monkeypatch)

    with pytest.raises(RuntimeError, match="halted"):
        run_forging_pipeline(name=_DATASET_NAME, project_root=tmp_path)

    assert recorded["session_names"] == ()


# Tests for the forging job universe and its ordering


def _dataset_stub(*sessions: str) -> SimpleNamespace:
    """Returns a dataset stub carrying the session and animal pairs the universe and ordering read."""
    return SimpleNamespace(
        sessions=tuple(
            SimpleNamespace(session=session, animal=_SESSION_ANIMALS.get(session, "animal_a")) for session in sessions
        )
    )


def test_build_forging_universe_covers_every_stage() -> None:
    """Verifies that the universe holds each animal's discovery alongside every extraction and assembly."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")

    assert _build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN) == [
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

    assert _build_forging_universe(dataset=dataset, multiday_plan={}) == [(FORGING_JOB_NAME, "session_1")]


def test_forging_job_prerequisites_chains_each_session_through_its_own_animal() -> None:
    """Verifies that an extraction waits on its own animal's discovery and an assembly waits on its extraction."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = _build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)

    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    assert ordering[MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"] == ()
    assert ordering[MULTIDAY_EXTRACTION_JOB_NAME, "session_2"] == ((MULTIDAY_DISCOVERY_JOB_NAME, "animal_a"),)
    assert ordering[MULTIDAY_EXTRACTION_JOB_NAME, "session_3"] == ((MULTIDAY_DISCOVERY_JOB_NAME, "animal_b"),)
    assert ordering[FORGING_JOB_NAME, "session_3"] == ((MULTIDAY_EXTRACTION_JOB_NAME, "session_3"),)


def test_forging_job_prerequisites_leaves_a_training_assembly_unordered() -> None:
    """Verifies that a session with no multi-day stage carries no upstream job."""
    dataset = _dataset_stub("session_1", "session_2")
    universe = _build_forging_universe(dataset=dataset, multiday_plan={})

    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    assert ordering[FORGING_JOB_NAME, "session_1"] == ()
    assert ordering[FORGING_JOB_NAME, "session_2"] == ()


def test_forging_job_prerequisites_covers_the_whole_universe() -> None:
    """Verifies that every job in the universe receives an ordering entry."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = _build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)

    assert set(forging_job_prerequisites(dataset=dataset, universe=universe)) == set(universe)


def test_every_multiday_session_reaches_its_own_animals_discovery() -> None:
    """Verifies that each assembly's upstream chain terminates at the discovery of the animal owning its session."""
    dataset = _dataset_stub("session_1", "session_2", "session_3")
    universe = _build_forging_universe(dataset=dataset, multiday_plan=_MULTIDAY_PLAN)
    ordering = forging_job_prerequisites(dataset=dataset, universe=universe)

    for session, animal in _SESSION_ANIMALS.items():
        chain = [(FORGING_JOB_NAME, session)]
        while ordering[chain[-1]]:
            chain.append(ordering[chain[-1]][0])
        assert chain[-1] == (MULTIDAY_DISCOVERY_JOB_NAME, animal), f"{session} does not reach its animal's discovery"


# Tests for the forging admission gate


def test_verify_session_admissibility_admits_a_fully_processed_session(
    training_session: Any, mark_session_processed: Any
) -> None:
    """Verifies that a session whose every required pipeline succeeded is admitted without complaint."""
    mark_session_processed(training_session)

    assert verify_session_admissibility(session=training_session) is None


def test_verify_session_admissibility_rejects_a_session_type_that_joins_no_dataset(
    session_factory: Any,
) -> None:
    """Verifies that a window-checking session is held out, since its type joins no Mesoscope-VR dataset."""
    session = session_factory(animal_id="305", session_type=SessionTypes.WINDOW_CHECKING)

    with pytest.raises(ValueError, match="dataset for the 'mesoscope' acquisition system, which admits"):
        verify_session_admissibility(session=session)


def test_verify_session_admissibility_reports_a_pipeline_that_never_ran(training_session: Any) -> None:
    """Verifies that a session carrying no tracker at all names every required pipeline as not started."""
    with pytest.raises(ValueError, match="not_started"):
        verify_session_admissibility(session=training_session)


def test_verify_session_admissibility_reports_an_empty_tracker_as_not_started(
    training_session: Any, mark_session_processed: Any, write_tracker: Any
) -> None:
    """Verifies that a tracker holding no job counts as not run, since a dispatched pipeline records its jobs first."""
    mark_session_processed(training_session)
    write_tracker(resolve_session_tracker_path(session=training_session, pipeline=ProcessingPipelines.VIDEO), [])

    with pytest.raises(ValueError, match="'video': 'not_started'"):
        verify_session_admissibility(session=training_session)


def test_verify_session_admissibility_reports_the_unfinished_job_count(
    training_session: Any, mark_session_processed: Any, write_tracker: Any
) -> None:
    """Verifies that a partially finished pipeline is reported with how many of its jobs are outstanding."""
    mark_session_processed(training_session)
    jobs = [("motion_energy", "face"), ("motion_energy", "body")]
    write_tracker(
        resolve_session_tracker_path(session=training_session, pipeline=ProcessingPipelines.VIDEO),
        jobs,
        succeeded=[jobs[0]],
    )

    with pytest.raises(ValueError, match="1 of 2 job\\(s\\) not"):
        verify_session_admissibility(session=training_session)


def test_verify_session_admissibility_ignores_a_pipeline_the_session_type_omits(
    training_session: Any, mark_session_processed: Any
) -> None:
    """Verifies that a training session is admitted with no two-photon tracker, since it records no imaging."""
    mark_session_processed(training_session)
    resolve_session_tracker_path(session=training_session, pipeline=ProcessingPipelines.TWO_PHOTON).unlink()

    assert ProcessingPipelines.TWO_PHOTON in SESSION_PIPELINES
    assert verify_session_admissibility(session=training_session) is None
