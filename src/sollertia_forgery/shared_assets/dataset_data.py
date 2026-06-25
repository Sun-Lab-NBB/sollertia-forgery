"""Provides the forged dataset data hierarchy: the container classes and the canonical filenames written by the
forging pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import RawDataFiles, SessionTypes, AcquisitionSystems
from ataraxis_data_structures import YamlConfig


class DatasetFiles(StrEnum):
    """Enumerates the canonical, system-agnostic filenames written into a forged dataset hierarchy at session
    granularity.

    Notes:
        Centralizes the forging-pipeline filenames so new artifacts can be added in one place and referenced
        symbolically from path-resolution properties on DatasetSession and DatasetAnimal. Only the universal output
        contract lives here: every system's forged session writes a ``data.feather``. System-specific per-session
        artifacts (for example a data-format/schema descriptor) are named by the donating acquisition-system package,
        not here, while shared raw-data assets re-exported alongside the data (the VR configuration, the session
        descriptor) keep their canonical ``RawDataFiles`` names.
    """

    DATA = "data.feather"
    """The assembled per-session data feather written by the forging pipeline."""


@dataclass(frozen=True, slots=True)
class DatasetSession:
    """Defines a single session included in a forged dataset.

    Combines the session identity metadata with the resolved path to the session's directory within the dataset
    hierarchy.
    """

    session: str
    """The unique identifier of the session. Session names follow the format 'YYYY-MM-DD-HH-MM-SS-microseconds' and
    encode the session's acquisition timestamp.
    """
    animal: str
    """The unique identifier of the animal that participated in the session."""
    session_path: Path = Path()
    """The path to the session's directory within the dataset hierarchy (dataset/animal/session)."""

    @property
    def data_path(self) -> Path:
        """Returns the path to the session's assembled ``data.feather`` file within the dataset hierarchy."""
        return self.session_path.joinpath(DatasetFiles.DATA)

    @property
    def descriptor_path(self) -> Path:
        """Returns the path to the session's ``session_descriptor.yaml`` file within the dataset hierarchy."""
        return self.session_path.joinpath(RawDataFiles.SESSION_DESCRIPTOR)

    @property
    def vr_configuration_path(self) -> Path:
        """Returns the path to the session's ``vr_configuration.yaml`` file within the dataset hierarchy.

        The forging pipeline re-exports the session's shared VR configuration (the task template carrying the cue
        sequence, VR environment, and per-trial geometry) alongside ``data.feather`` so downstream consumers can
        reconstruct canonical per-trial position without reaching back into the raw session.
        """
        return self.session_path.joinpath(RawDataFiles.VR_CONFIGURATION)


@dataclass(frozen=True, slots=True)
class DatasetAnimal:
    """Defines a single animal included in a forged dataset.

    Combines the animal identity metadata with the resolved path to the animal's directory within the dataset
    hierarchy. Per-animal artifacts (surgery metadata) are co-located in this directory and exposed as derived
    properties.
    """

    animal: str
    """The unique identifier of the animal."""
    animal_path: Path = Path()
    """The path to the animal's directory within the dataset hierarchy (dataset/animal)."""

    @property
    def surgery_path(self) -> Path:
        """Returns the path to the animal's ``surgery_metadata.yaml`` file within the dataset hierarchy."""
        return self.animal_path.joinpath(RawDataFiles.SURGERY_METADATA)


@dataclass
class DatasetData(YamlConfig):
    """Defines the structure and the metadata of a forged dataset.

    A forged dataset aggregates multiple data acquisition sessions of the same type, recorded across different
    animals by the same acquisition system. This class encapsulates the information necessary to access the dataset's
    assembled (forged) data stored on disk and functions as the entry point for all interactions with the dataset.

    Notes:
        Do not initialize this class directly. Instead, use the create() method when creating new datasets or the
        load() method when accessing data for an existing dataset.

        Datasets are created using a pre-filtered set of session + animal pairs, typically obtained through the
        session filtering functionality in sollertia-forgery. The dataset stores only the assembled data, not raw or
        processed data.
    """

    name: str
    """The unique name of the dataset."""
    project: str
    """The name of the project from which the dataset's sessions originate."""
    session_type: str | SessionTypes
    """The type of data acquisition sessions included in the dataset. All sessions in a dataset must be of the
    same type.
    """
    acquisition_system: str | AcquisitionSystems
    """The name of the data acquisition system used to acquire all sessions in the dataset."""
    sessions: tuple[DatasetSession, ...] = field(default_factory=tuple)
    """The DatasetSession instances that identify and locate each session included in the dataset."""
    dataset_data_path: Path = Path()
    """The path to the dataset.yaml file cached to disk."""

    def __post_init__(self) -> None:
        """Ensures that all fields used to define the dataset are properly initialized."""
        # Converts string values loaded from YAML to proper enum types.
        if isinstance(self.session_type, str):
            self.session_type = SessionTypes(self.session_type)
        if isinstance(self.acquisition_system, str):
            self.acquisition_system = AcquisitionSystems(self.acquisition_system)

    @classmethod
    def create(
        cls,
        name: str,
        project: str,
        session_type: str | SessionTypes,
        acquisition_system: str | AcquisitionSystems,
        sessions: tuple[DatasetSession, ...] | set[DatasetSession],
        datasets_root: Path,
    ) -> DatasetData:
        """Creates a new forged dataset and initializes its data structure on disk.

        Notes:
            To access the data of an already existing dataset, use the load() method.

        Args:
            name: The unique name for the dataset.
            project: The name of the project from which the dataset's sessions originate.
            session_type: The type of data acquisition sessions included in the dataset.
            acquisition_system: The name of the data acquisition system used to acquire all sessions included in the
                dataset.
            sessions: The set of DatasetSession instances that identify the sessions whose data should be included in
                the dataset. The session_path attribute of each input instance is ignored and replaced with the
                resolved path inside the dataset hierarchy.
            datasets_root: The path to the root directory where to create the dataset's hierarchy.

        Returns:
            An initialized DatasetData instance that stores the structure and the metadata of the created dataset.

        Raises:
            ValueError: If no sessions are provided.
            FileExistsError: If a dataset with the same name already exists.
        """
        # Converts sessions to tuple if provided as set.
        if isinstance(sessions, set):
            sessions = tuple(sessions)

        if not sessions:
            message = (
                f"Unable to create the '{name}' forged dataset. The 'sessions' argument must contain at least one "
                f"DatasetSession instance, but got an empty collection."
            )
            console.error(message=message, error=ValueError)

        # Constructs the dataset root directory path.
        dataset_path = datasets_root.joinpath(name)

        # Prevents overwriting existing datasets.
        if dataset_path.exists():
            message = (
                f"Unable to create the '{name}' forged dataset. The destination directory must not exist, but a "
                f"dataset already exists at {dataset_path}."
            )
            console.error(message=message, error=FileExistsError)

        # Creates the dataset root directory. Downstream consumers populate it with their own files.
        ensure_directory_exists(path=dataset_path)

        # Creates animal/session subdirectories and rebuilds each session with its resolved path.
        resolved_sessions: list[DatasetSession] = []
        for session in sessions:
            session_path = dataset_path.joinpath(session.animal, session.session)
            ensure_directory_exists(path=session_path)
            resolved_sessions.append(
                DatasetSession(session=session.session, animal=session.animal, session_path=session_path)
            )

        # Generates the DatasetData instance.
        instance = cls(
            name=name,
            project=project,
            session_type=session_type,
            acquisition_system=acquisition_system,
            sessions=tuple(resolved_sessions),
            dataset_data_path=dataset_path.joinpath("dataset.yaml"),
        )

        # Saves the configured instance data to disk.
        instance.save()

        return instance

    @classmethod
    def load(cls, dataset_path: Path) -> DatasetData:
        """Loads the target dataset's data from the specified dataset.yaml file.

        Notes:
            To create a new dataset, use the create() method.

        Args:
            dataset_path: The path to the directory where to search for the dataset.yaml file. Typically, this
                is the path to the root dataset directory.

        Returns:
            An initialized DatasetData instance that stores the loaded dataset's data.

        Raises:
            FileNotFoundError: If multiple or no 'dataset.yaml' file instances are found under the input directory.
        """
        # Locates the dataset.yaml file.
        dataset_data_files = list(dataset_path.rglob("dataset.yaml"))
        if len(dataset_data_files) != 1:
            message = (
                f"Unable to load the target dataset's data. Expected a single dataset.yaml file to be located "
                f"under the directory tree specified by the input path: {dataset_path}. Instead, encountered "
                f"{len(dataset_data_files)} candidate files. This indicates that the input path does not point to a "
                f"valid dataset data hierarchy."
            )
            console.error(message=message, error=FileNotFoundError)

        # Loads the dataset's data from the .yaml file.
        dataset_data_path = dataset_data_files.pop()
        instance: DatasetData = cls.from_yaml(file_path=dataset_data_path)

        # Re-resolves the dataset_data_path and each session's session_path against the YAML file's filesystem
        # location so the dataset remains portable across processing machines.
        local_root = dataset_data_path.parent
        instance.dataset_data_path = dataset_data_path
        instance.sessions = tuple(
            DatasetSession(
                session=session.session,
                animal=session.animal,
                session_path=local_root.joinpath(session.animal, session.session),
            )
            for session in instance.sessions
        )

        return instance

    def save(self) -> None:
        """Caches the instance's data to the dataset's root directory as a 'dataset.yaml' file."""
        self.to_yaml(file_path=self.dataset_data_path)

    @property
    def animals(self) -> tuple[DatasetAnimal, ...]:
        """Returns a tuple of DatasetAnimal instances, one per unique animal in the dataset.

        Each instance carries the animal identifier and the resolved path to the animal's directory under
        the dataset root, anchored on the ``dataset.yaml`` file's filesystem location so the result remains
        portable across processing machines.
        """
        dataset_root = self.dataset_data_path.parent
        unique_animals = sorted({session.animal for session in self.sessions})
        return tuple(
            DatasetAnimal(animal=animal, animal_path=dataset_root.joinpath(animal)) for animal in unique_animals
        )

    def get_animal(self, animal: str) -> DatasetAnimal:
        """Returns the DatasetAnimal instance for the specified animal identifier.

        Args:
            animal: The unique identifier of the animal to look up.

        Returns:
            The DatasetAnimal instance carrying the animal identity metadata and the path to the animal's
            directory within the dataset hierarchy.

        Raises:
            ValueError: If the specified animal is not found in the dataset.
        """
        for candidate in self.animals:
            if candidate.animal == animal:
                return candidate

        message = (
            f"Unable to look up the animal '{animal}'. The animal must exist in the '{self.name}' dataset, "
            f"but no matching DatasetAnimal was found."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        # noinspection PyUnreachableCode
        raise ValueError(message)  # pragma: no cover

    def get_sessions_for_animal(self, animal: str) -> tuple[DatasetSession, ...]:
        """Returns the DatasetSession instances for all sessions performed by the specified animal.

        Args:
            animal: The unique identifier of the animal for which to retrieve the session data.

        Returns:
            A tuple of DatasetSession instances for the specified animal.
        """
        return tuple(session for session in self.sessions if session.animal == animal)

    def get_session(self, animal: str, session: str) -> DatasetSession:
        """Returns the DatasetSession instance for the specified animal and session pair.

        Args:
            animal: The unique identifier of the animal that participated in the session.
            session: The unique identifier of the session to look up.

        Returns:
            The DatasetSession instance containing the session identity metadata and the path to the session's
            directory within the dataset hierarchy.

        Raises:
            ValueError: If the specified animal and session combination is not found in the dataset.
        """
        for candidate in self.sessions:
            if candidate.animal == animal and candidate.session == session:
                return candidate

        message = (
            f"Unable to look up the session '{session}' performed by the animal '{animal}'. The animal and "
            f"session combination must exist in the '{self.name}' dataset, but no matching DatasetSession was found."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        # noinspection PyUnreachableCode
        raise ValueError(message)  # pragma: no cover
