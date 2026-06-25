"""Provides the forged dataset data hierarchy and the operations that resolve and create it from acquisition
sessions.

Defines the dataset container classes (``DatasetData``, ``DatasetSession``, ``DatasetAnimal``) and the canonical
filenames (``DatasetFiles``) written into a forged dataset, together with the ``resolve_dataset`` entry point that
loads, creates, or recreates the dataset hierarchy from a processed-session set.
"""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import (
    SessionData,
    RawDataFiles,
    SessionTypes,
    discover_sessions,
    AcquisitionSystems,
)
from ataraxis_data_structures import YamlConfig, delete_directory


class DatasetFiles(StrEnum):
    """Enumerates the canonical, system-agnostic filenames written into a forged dataset hierarchy at session
    granularity.

    Notes:
        Centralizes the forging-pipeline filenames so new artifacts can be added in one place and referenced
        symbolically from path-resolution properties on DatasetSession and DatasetAnimal. Only the universal output
        contract lives here: every system's forged session writes a ``data.feather``. System-specific per-session
        artifacts (for example a data-format/schema descriptor) are named by the donating acquisition-system package,
        not here, while shared raw-data assets re-exported alongside the data keep their canonical ``RawDataFiles``
        names: the VR configuration and session descriptor at session granularity, and the per-animal surgery
        metadata at animal granularity.
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
    """The resolved path to this dataset's ``dataset.yaml`` file. Re-derived from the YAML's on-disk location on
    load so the dataset remains portable across machines."""

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


def resolve_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
    force_recreate: bool = False,
) -> DatasetData:
    """Creates, loads, or recreates a dataset hierarchy based on the provided session set.

    Acts as the dataset-definition entry point for both the forging pipeline and the MCP ``prepare`` tool so that
    tracker setup and batch dispatch can layer above this helper rather than duplicating the resolution logic. When
    the dataset already exists, loads it and, when a session list is provided, verifies that the provided set matches
    the existing definition. A mismatch surfaces as an error unless ``force_recreate`` is True, which unlocks
    deletion and fresh recreation from the provided session names. When the dataset does not exist, creates it from
    the provided session names; a non-empty session list is required in that case.

    Args:
        name: The unique name of the dataset.
        session_names: The session names to include in the dataset. Pass an empty tuple to work with an
            already-defined dataset without triggering the session-set verification step.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        required_session_type: The session type the dataset's sessions must have, supplied by the calling
            acquisition system to restrict dataset creation to its forgeable session type. When None, any session
            type is accepted (the cross-session consistency check still applies).
        force_recreate: Determines whether to allow deletion of the existing dataset hierarchy when the provided
            session list does not match the existing definition.

    Returns:
        The resolved DatasetData instance (either loaded from disk or freshly created).

    Raises:
        ValueError: If required_session_type is provided and the first session's type differs from it, if the
            dataset does not exist and no sessions were provided to create it, or if the provided session list does
            not match the existing dataset and force_recreate is False.
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
    """
    dataset_directory = project_root.joinpath(name)
    if dataset_directory.exists():
        dataset = DatasetData.load(dataset_path=dataset_directory)

        if session_names:
            provided_sessions = set(session_names)
            existing_sessions = {entry.session for entry in dataset.sessions}
            if provided_sessions != existing_sessions:
                if not force_recreate:
                    message = (
                        f"Unable to use the existing '{name}' dataset. The provided session list does not match "
                        f"the dataset's existing session set. Call with force_recreate=True to delete and "
                        f"recreate the dataset, or provide a matching session list."
                    )
                    console.error(message=message, error=ValueError)
                delete_directory(directory_path=dataset_directory)
                console.echo(
                    message=f"Dataset '{name}': Removed existing hierarchy for recreation.",
                    level=LogLevel.INFO,
                )
                dataset = _create_dataset(
                    name=name,
                    sessions=session_names,
                    project_root=project_root,
                    required_session_type=required_session_type,
                )
        return dataset

    if not session_names:
        message = (
            f"Unable to define dataset '{name}'. The dataset does not exist under '{project_root}' and no "
            f"sessions were provided to create it."
        )
        console.error(message=message, error=ValueError)
    return _create_dataset(
        name=name,
        sessions=session_names,
        project_root=project_root,
        required_session_type=required_session_type,
    )


def _create_dataset(
    name: str,
    sessions: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
) -> DatasetData:
    """Creates a fresh dataset hierarchy by resolving the provided session names under the project root.

    Each session name is resolved by probing the discovered ``<project_root>/<animal>/<session_name>`` layout and
    requiring the session's ``session_data.yaml`` marker to be present. Every included session must share the first
    session's session type and acquisition system, and when ``required_session_type`` is provided, the first
    session's type must also match it.

    Args:
        name: The unique name for the dataset.
        sessions: The non-empty tuple of session names to include in the dataset.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        required_session_type: The session type the first session must have. When None, any session type is
            accepted as the dataset's reference type.

    Returns:
        The newly created DatasetData instance.

    Raises:
        FileNotFoundError: If a session name does not resolve to any animal directory under the project root.
        RuntimeError: If a session name resolves to more than one animal directory under the project root.
        ValueError: If required_session_type is provided and the first session's type differs from it, or if any
            subsequent session's session type or acquisition system differs from the first session's.
    """
    # Builds a session-name -> session-root index via shared-assets discovery so the project layout is not
    # assumed here. A session name colliding across animals surfaces as a RuntimeError during lookup rather
    # than silently selecting the first match.
    discovered: dict[str, list[Path]] = {}
    for session_root in discover_sessions(root_path=project_root):
        discovered.setdefault(session_root.name, []).append(session_root)

    session_paths: list[Path] = []
    for session_name in sessions:
        matches = discovered.get(session_name, [])
        if len(matches) != 1:
            message = (
                f"Unable to resolve the directory for session '{session_name}' under '{project_root}'. "
                f"Expected exactly one session named '{session_name}' to be discoverable via "
                f"'{RawDataFiles.SESSION_DATA}' markers, but found {len(matches)}."
            )
            console.error(message=message, error=FileNotFoundError if not matches else RuntimeError)
        session_paths.append(matches[0])

    first_session_data = SessionData.load(session_path=session_paths[0])
    if required_session_type is not None and first_session_data.session_type != required_session_type:
        message = (
            f"Unable to define dataset '{name}'. Dataset creation for this acquisition system is supported only "
            f"for '{required_session_type}' sessions, but the first session's type resolved to "
            f"'{first_session_data.session_type}'."
        )
        console.error(message=message, error=ValueError)

    # Verifies that every remaining session shares the first session's type and acquisition system. A dataset
    # must contain only sessions of the same type acquired by the same acquisition system; the assembly logic
    # downstream assumes this invariant when deriving the dataset-level metadata from the first session.
    for session_path in session_paths[1:]:
        session_data = SessionData.load(session_path=session_path)
        if session_data.session_type != first_session_data.session_type:
            message = (
                f"Unable to define dataset '{name}'. All sessions in a dataset must share the same session "
                f"type, but session '{session_path.name}' has type '{session_data.session_type}' while the "
                f"first session has type '{first_session_data.session_type}'."
            )
            console.error(message=message, error=ValueError)
        if session_data.acquisition_system != first_session_data.acquisition_system:
            message = (
                f"Unable to define dataset '{name}'. All sessions in a dataset must be acquired by the same "
                f"acquisition system, but session '{session_path.name}' was acquired by "
                f"'{session_data.acquisition_system}' while the first session was acquired by "
                f"'{first_session_data.acquisition_system}'."
            )
            console.error(message=message, error=ValueError)

    dataset_sessions = tuple(DatasetSession(animal=path.parent.name, session=path.name) for path in session_paths)
    dataset = DatasetData.create(
        name=name,
        project=project_root.name,
        session_type=first_session_data.session_type,
        acquisition_system=first_session_data.acquisition_system,
        sessions=dataset_sessions,
        datasets_root=project_root,
    )

    _copy_animal_surgery_files(dataset_name=name, dataset=dataset, source_session_paths=session_paths)

    console.echo(
        message=(
            f"Dataset '{name}' data hierarchy: Defined with {len(sessions)} sessions from "
            f"{len(dataset.animals)} animals."
        ),
        level=LogLevel.SUCCESS,
    )
    return dataset


def _copy_animal_surgery_files(
    dataset_name: str,
    dataset: DatasetData,
    source_session_paths: list[Path],
) -> None:
    """Copies the surgery metadata YAML for each animal into the dataset's animal directory.

    For each animal in the dataset, selects that animal's most recent source session and copies its
    ``surgery_metadata.yaml`` from the session's raw data directory to the dataset's animal directory root.
    Surgery metadata is per-animal rather than per-session, so a single copy is materialized for each animal.

    Args:
        dataset_name: The name of the dataset, used for error messages.
        dataset: The freshly created DatasetData instance, used to locate per-animal directories.
        source_session_paths: The resolved source session directory paths used to define the dataset, in the
            order provided to ``_create_dataset``. Grouped by animal to pick each animal's latest session.

    Raises:
        FileNotFoundError: If the latest session for any animal does not contain a ``surgery_metadata.yaml`` file.
    """
    # Groups source session paths by owning animal. The animal name is the parent directory name in the source
    # project layout.
    sessions_by_animal: dict[str, list[Path]] = {}
    for source_path in source_session_paths:
        sessions_by_animal.setdefault(source_path.parent.name, []).append(source_path)

    # The dataset hierarchy stores each animal at ``<dataset_root>/<animal>/``. DatasetAnimal.surgery_path
    # resolves the per-animal destination relative to that directory.
    for dataset_animal in dataset.animals:
        animal_sessions = sessions_by_animal[dataset_animal.animal]

        # Picks the most recent session for the animal via natural sort over the timestamped session names.
        latest_session_name = natsorted([path.name for path in animal_sessions])[-1]
        latest_session_path = next(path for path in animal_sessions if path.name == latest_session_name)
        session_data = SessionData.load(session_path=latest_session_path)

        source_surgery_path = session_data.raw_data.surgery_metadata_path
        if not source_surgery_path.is_file():
            message = (
                f"Unable to define dataset '{dataset_name}'. The latest session '{latest_session_name}' for "
                f"animal '{dataset_animal.animal}' does not contain a '{RawDataFiles.SURGERY_METADATA}' file at "
                f"'{source_surgery_path}'. Surgery metadata is required for every animal in a forged dataset."
            )
            console.error(message=message, error=FileNotFoundError)

        shutil.copy2(src=source_surgery_path, dst=dataset_animal.surgery_path)
