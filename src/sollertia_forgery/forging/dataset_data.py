"""Provides assets for maintaining the Sollertia platform analysis dataset data hierarchy across all processing
machines.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import RawDataFiles, SessionTypes, AcquisitionSystems
from ataraxis_data_structures import YamlConfig

from .trial_geometry import TRIAL_GEOMETRY_FILENAME

DATA_FILENAME: str = "data.feather"
"""The filename of the assembled session data inside each session directory of a forged dataset."""


class DatasetColumn(StrEnum):
    """Defines every column that can appear in the assembled session data feather produced by the forging pipeline.

    Notes:
        Members covering optional columns (`REINFORCING_GUIDED`, `AVERSIVE_GUIDED`) are present in the feather only
        when the corresponding upstream events were recorded. All other members are guaranteed to exist in every
        forged session.
    """

    # Behavior alignment columns (from forging.behavior).
    TIME_US = "time_us"
    """Microsecond-precision sample timestamps from the acquisition reference clock."""
    ELAPSED_MINUTES = "elapsed_minutes"
    """Elapsed session time in minutes since the first sample."""
    BRAKE = "brake"
    """Wheel brake engagement at each sample."""
    SCREENS = "screens"
    """Display panel state at each sample."""
    TORQUE_N_CM = "torque_N_cm"
    """Wheel torque in N·cm at each sample. Forced to zero during 'run' periods upstream."""
    DISTANCE_CM = "distance_cm"
    """Cumulative distance traveled by the animal in centimeters at each sample."""
    SPEED_CM_S = "speed_cm_s"
    """Animal running speed in cm/s at each sample."""
    LICK = "lick"
    """Lick sensor state at each sample."""
    WATER_UL = "water_uL"
    """Per-sample water reward delivery in microliters."""
    REWARD = "reward"
    """Reward event flag at each sample."""
    SYSTEM_STATE = "system_state"
    """Acquisition system state at each sample (idle, rest, run)."""

    # Runtime/experiment columns (from forging.runtime).
    TRIAL = "trial"
    """One-based trial identifier at each sample. 255 marks samples outside any trial."""
    TRIAL_TYPE = "trial_type"
    """Trial type label at each sample (e.g. 'ABC', 'ABCD'). 'undefined' marks non-run samples."""
    CUE = "cue"
    """Active virtual reality cue identifier at each sample."""
    IN_TRIGGER_ZONE = "in_trigger_zone"
    """Boolean flag indicating whether the animal is inside a stimulus trigger zone at each sample."""
    RUNTIME_STATE = "runtime_state"
    """Experiment runtime state label at each sample."""
    REINFORCING_GUIDED = "reinforcing_guided"
    """Optional. Reinforcing guidance state at each sample. Present only when reinforcing guidance was recorded."""
    AVERSIVE_GUIDED = "aversive_guided"
    """Optional. Aversive guidance state at each sample. Present only when aversive guidance was recorded."""

    # Cindra fluorescence columns (from forging.cindra).
    SINGLE_DAY_CELL_FLUORESCENCE = "single_day_cell_fluorescence"
    """Single-recording raw cell fluorescence trace per ROI."""
    SINGLE_DAY_NEUROPIL_FLUORESCENCE = "single_day_neuropil_fluorescence"
    """Single-recording raw neuropil fluorescence trace per ROI."""
    SINGLE_DAY_SUBTRACTED_FLUORESCENCE = "single_day_subtracted_fluorescence"
    """Single-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence."""
    SINGLE_DAY_SPIKES = "single_day_spikes"
    """Single-recording OASIS-deconvolved spike rates per ROI."""
    MULTI_DAY_CELL_FLUORESCENCE = "multi_day_cell_fluorescence"
    """Multi-recording raw cell fluorescence trace per ROI."""
    MULTI_DAY_NEUROPIL_FLUORESCENCE = "multi_day_neuropil_fluorescence"
    """Multi-recording raw neuropil fluorescence trace per ROI."""
    MULTI_DAY_SUBTRACTED_FLUORESCENCE = "multi_day_subtracted_fluorescence"
    """Multi-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence aligned across recording days."""
    MULTI_DAY_SPIKES = "multi_day_spikes"
    """Multi-recording OASIS-deconvolved spike rates per ROI aligned across recording days."""


@dataclass(frozen=True, slots=True)
class DatasetSession:
    """Defines a single session included in an analysis dataset.

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
        return self.session_path.joinpath(DATA_FILENAME)

    @property
    def descriptor_path(self) -> Path:
        """Returns the path to the session's ``session_descriptor.yaml`` file within the dataset hierarchy."""
        return self.session_path.joinpath(RawDataFiles.SESSION_DESCRIPTOR)

    @property
    def geometry_path(self) -> Path:
        """Returns the path to the session's ``trial_geometry.yaml`` data file within the dataset hierarchy."""
        return self.session_path.joinpath(TRIAL_GEOMETRY_FILENAME)


@dataclass
class DatasetData(YamlConfig):
    """Defines the structure and the metadata of an analysis dataset.

    An analysis dataset aggregates multiple data acquisition sessions of the same type, recorded across different
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
        """Creates a new analysis dataset and initializes its data structure on disk.

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
                f"Unable to create the '{name}' analysis dataset. The 'sessions' argument must contain at least one "
                f"DatasetSession instance, but got an empty collection."
            )
            console.error(message=message, error=ValueError)

        # Constructs the dataset root directory path.
        dataset_path = datasets_root.joinpath(name)

        # Prevents overwriting existing datasets.
        if dataset_path.exists():
            message = (
                f"Unable to create the '{name}' analysis dataset. The destination directory must not exist, but a "
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
    def animals(self) -> tuple[str, ...]:
        """Returns a tuple of unique animal identifiers included in the dataset."""
        return tuple(sorted({session.animal for session in self.sessions}))

    @property
    def surgery_paths(self) -> dict[str, Path]:
        """Returns a mapping of each animal identifier to the path of its surgery metadata YAML file.

        The returned paths point to ``surgery_metadata.yaml`` files stored at the root of each animal
        directory within the forged dataset hierarchy.
        """
        dataset_root = self.dataset_data_path.parent
        return {animal: dataset_root.joinpath(animal, RawDataFiles.SURGERY_METADATA) for animal in self.animals}

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
