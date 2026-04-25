"""Contains tests for the DatasetData and DatasetSession dataclasses relocated from sollertia-shared-assets."""

from pathlib import Path

import pytest
from sollertia_shared_assets import SessionTypes, AcquisitionSystems

from sollertia_forgery.forging import DatasetData, DatasetSession


# Tests for DatasetSession dataclass


def test_dataset_session_default_initialization() -> None:
    """Verifies default initialization of DatasetSession.

    This test ensures session_path defaults to an empty Path() when not provided.
    """
    dataset_session = DatasetSession(session="2024-01-15-12-30-45-123456", animal="test_animal")

    assert dataset_session.session == "2024-01-15-12-30-45-123456"
    assert dataset_session.animal == "test_animal"
    assert dataset_session.session_path == Path()


def test_dataset_session_is_frozen() -> None:
    """Verifies that DatasetSession instances are immutable.

    This test ensures attempting to modify a DatasetSession field raises an error.
    """
    dataset_session = DatasetSession(
        session="2024-01-15-12-30-45-123456",
        animal="test_animal",
        session_path=Path("/tmp/test"),
    )

    with pytest.raises(AttributeError):
        dataset_session.session = "new_session"  # type: ignore[misc]


def test_dataset_session_data_and_descriptor_paths() -> None:
    """Verifies that data_path and descriptor_path resolve relative to session_path."""
    session_path = Path("/tmp/test_dataset/animal_a/2024-01-15-12-30-45-123456")
    dataset_session = DatasetSession(
        session="2024-01-15-12-30-45-123456",
        animal="animal_a",
        session_path=session_path,
    )

    assert dataset_session.data_path == session_path / "data.feather"
    assert dataset_session.descriptor_path == session_path / "session_descriptor.yaml"


# Tests for DatasetData dataclass


def test_dataset_data_direct_initialization() -> None:
    """Verifies that DatasetData can be constructed directly for in-memory use (load path)."""
    dataset_data = DatasetData(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
    )

    assert dataset_data.name == "test_dataset"
    assert dataset_data.project == "test_project"
    assert dataset_data.session_type == SessionTypes.LICK_TRAINING
    assert dataset_data.acquisition_system == AcquisitionSystems.MESOSCOPE_VR
    assert dataset_data.sessions == ()


def test_dataset_data_create_initializes_directory_structure(tmp_path: Path) -> None:
    """Verifies that DatasetData.create materializes the dataset hierarchy on disk."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-123456", animal="animal_a"),
        DatasetSession(session="2024-01-16-09-15-22-654321", animal="animal_b"),
    )

    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    dataset_root = tmp_path / "test_dataset"
    assert dataset_root.is_dir()
    assert (dataset_root / "dataset.yaml").is_file()
    assert (dataset_root / "animal_a" / "2024-01-15-12-30-45-123456").is_dir()
    assert (dataset_root / "animal_b" / "2024-01-16-09-15-22-654321").is_dir()
    assert dataset_data.dataset_data_path == dataset_root / "dataset.yaml"


def test_dataset_data_create_resolves_session_paths(tmp_path: Path) -> None:
    """Verifies that create() rebuilds each input DatasetSession with its resolved session_path."""
    inputs = (DatasetSession(session="2024-01-15-12-30-45-123456", animal="animal_a", session_path=Path("/ignored")),)

    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=inputs,
        datasets_root=tmp_path,
    )

    resolved = dataset_data.sessions[0]
    assert resolved.session_path == tmp_path / "test_dataset" / "animal_a" / "2024-01-15-12-30-45-123456"


def test_dataset_data_create_accepts_set_of_sessions(tmp_path: Path) -> None:
    """Verifies that create() accepts a set of DatasetSession instances and converts them to a tuple."""
    sessions = {
        DatasetSession(session="2024-01-15-12-30-45-123456", animal="animal_a"),
        DatasetSession(session="2024-01-16-09-15-22-654321", animal="animal_b"),
    }

    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    assert isinstance(dataset_data.sessions, tuple)
    assert len(dataset_data.sessions) == 2


def test_dataset_data_create_raises_on_empty_sessions(tmp_path: Path) -> None:
    """Verifies that create() rejects an empty sessions collection."""
    with pytest.raises(ValueError, match="at least one"):
        DatasetData.create(
            name="empty_dataset",
            project="test_project",
            session_type=SessionTypes.LICK_TRAINING,
            acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
            sessions=(),
            datasets_root=tmp_path,
        )


def test_dataset_data_create_rejects_existing_directory(tmp_path: Path) -> None:
    """Verifies that create() refuses to overwrite an existing dataset directory."""
    sessions = (DatasetSession(session="2024-01-15-12-30-45-123456", animal="animal_a"),)
    (tmp_path / "existing").mkdir()

    with pytest.raises(FileExistsError):
        DatasetData.create(
            name="existing",
            project="test_project",
            session_type=SessionTypes.LICK_TRAINING,
            acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
            sessions=sessions,
            datasets_root=tmp_path,
        )


def test_dataset_data_load_roundtrips_through_yaml(tmp_path: Path) -> None:
    """Verifies that load() reconstructs a DatasetData instance from a previously saved dataset.yaml file."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-123456", animal="animal_a"),
        DatasetSession(session="2024-01-16-09-15-22-654321", animal="animal_b"),
    )
    created = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    loaded = DatasetData.load(dataset_path=tmp_path / "test_dataset")

    assert loaded.name == created.name
    assert loaded.project == created.project
    assert loaded.session_type == SessionTypes.LICK_TRAINING
    assert loaded.acquisition_system == AcquisitionSystems.MESOSCOPE_VR
    assert len(loaded.sessions) == 2


def test_dataset_data_load_errors_when_no_marker(tmp_path: Path) -> None:
    """Verifies that load() raises FileNotFoundError when dataset.yaml cannot be located."""
    (tmp_path / "empty_dataset").mkdir()

    with pytest.raises(FileNotFoundError):
        DatasetData.load(dataset_path=tmp_path / "empty_dataset")


def test_dataset_data_surgery_paths_maps_each_animal(tmp_path: Path) -> None:
    """Verifies that surgery_paths yields a per-animal mapping anchored on the dataset root."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-000001", animal="animal_a"),
        DatasetSession(session="2024-01-15-12-30-45-000002", animal="animal_b"),
        DatasetSession(session="2024-01-15-12-30-45-000003", animal="animal_a"),
    )
    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    surgery_paths = dataset_data.surgery_paths
    dataset_root = tmp_path / "test_dataset"

    assert set(surgery_paths.keys()) == {"animal_a", "animal_b"}
    assert surgery_paths["animal_a"] == dataset_root / "animal_a" / "surgery_metadata.yaml"
    assert surgery_paths["animal_b"] == dataset_root / "animal_b" / "surgery_metadata.yaml"


def test_dataset_data_animals_returns_unique_sorted_ids(tmp_path: Path) -> None:
    """Verifies that the animals property exposes a sorted tuple of unique animal identifiers."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-000001", animal="animal_b"),
        DatasetSession(session="2024-01-15-12-30-45-000002", animal="animal_a"),
        DatasetSession(session="2024-01-15-12-30-45-000003", animal="animal_b"),
    )
    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    assert dataset_data.animals == ("animal_a", "animal_b")


def test_dataset_data_get_sessions_for_animal(tmp_path: Path) -> None:
    """Verifies that get_sessions_for_animal returns only sessions belonging to the requested animal."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-000001", animal="animal_a"),
        DatasetSession(session="2024-01-15-12-30-45-000002", animal="animal_b"),
        DatasetSession(session="2024-01-15-12-30-45-000003", animal="animal_a"),
    )
    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    animal_a_sessions = dataset_data.get_sessions_for_animal(animal="animal_a")

    assert len(animal_a_sessions) == 2
    assert all(session.animal == "animal_a" for session in animal_a_sessions)


def test_dataset_data_get_session_found(tmp_path: Path) -> None:
    """Verifies that get_session() returns the DatasetSession matching the specified animal and session."""
    sessions = (
        DatasetSession(session="2024-01-15-12-30-45-000001", animal="animal_a"),
        DatasetSession(session="2024-01-15-12-30-45-000002", animal="animal_b"),
    )
    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    found = dataset_data.get_session(animal="animal_a", session="2024-01-15-12-30-45-000001")

    assert found.animal == "animal_a"
    assert found.session == "2024-01-15-12-30-45-000001"


def test_dataset_data_get_session_not_found(tmp_path: Path) -> None:
    """Verifies that get_session() raises ValueError when the animal/session pair is not in the dataset."""
    sessions = (DatasetSession(session="2024-01-15-12-30-45-000001", animal="animal_a"),)
    dataset_data = DatasetData.create(
        name="test_dataset",
        project="test_project",
        session_type=SessionTypes.LICK_TRAINING,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=sessions,
        datasets_root=tmp_path,
    )

    with pytest.raises(ValueError, match="must exist in the 'test_dataset' dataset"):
        dataset_data.get_session(animal="animal_z", session="2024-01-15-12-30-45-999999")
