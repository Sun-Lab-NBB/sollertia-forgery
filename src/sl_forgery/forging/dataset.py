import re
import copy
from enum import IntEnum
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import field, dataclass

import polars as pl
from dateutil import parser
from sl_shared_assets import SessionTypes, AcquisitionSystems
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig

from ..shared_assets import ProjectManifest

# Stores the types of sessions that currently support dataset integration.
_supported_sessions = (SessionTypes.MESOSCOPE_EXPERIMENT, SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING)

# Stores the acquisition systems that currently support dataset integration
_supported_acquisition_systems = (AcquisitionSystems.MESOSCOPE_VR,)


class DatasetTypes(IntEnum):
    """Stores the types of datasets currently supported by the Sun lab's data processing workflow."""

    MESOSCOPE_VR_LICK_TRAINING = 1
    """Mesoscope-VR acquisition system + Lick training session type."""
    MESOSCOPE_VR_RUN_TRAINING = 2
    """Mesoscope-VR acquisition system + Run training session type."""
    MESOSCOPE_VR_EXPERIMENT = 3
    """Mesoscope-VR acquisition system + Mesoscope Experiment session type."""


@dataclass()
class AnimalDataset:
    """Specifies the filtering parameters used to extract a subset of all data acquisition sessions performed by the
    target animal for further analysis.

    This class is used when building analysis datasets to determine which data to include in the dataset from that
    specific animal. Multiple instances of this class are used as part of the DatasetManifest class.

    Notes:
        All sessions in the Sun lab use their timestamps stored as microseconds elapsed since UTC epoch onset as IDs.
        The sessions are also identifiable based on the EDT / ETC timestamp for when the session was acquired, which
        exactly matches the session ID, but is translated from UTC to EDT/ETC time zone.

        Filtering hierarchy:
            1. Sessions must be of the type specified in the DatasetManifest class instance that uses this class.
            2. Sessions must belong to the target animal.
            3. Sessions must not be in the `exclude` list.
            4. Sessions can either be in the `include` list or fall within the `start_date` / `end_date` range.

    """

    animal: int = 11
    """The ID of the animal for which to generate the dataset."""
    start_date: str = "2025-07-01"
    """The data slice start date. All sessions recorded on or after this date are included in the dataset."""
    end_date: str = "2025-08-01"
    """The data slice end date. All sessions recorded on or before this date are included in the dataset."""
    include: list[str] = field(default_factory=list)
    """The sessions to include in the dataset even if they fall outside of the `start_date` / `end_date` range. This 
    field must use the full session ID (name), rather than a shortened session date."""
    exclude: list[str] = field(default_factory=list)
    """The sessions to exclude from the dataset even if they fall within the `start_date` / `end_date` range. This 
    field takes precedence over the `include` field if a session is included in both fields. This field must use the 
    full session ID (name), rather than a shortened session date.
    """


@dataclass()
class DatasetManifest(YamlConfig):
    """Specifies the filtering parameters used to generate an analysis dataset for the target project.

    This class is used to build analysis datasets using the raw and processed data of the target project. Instances
    of this class are used by the ProjectData class during the dataset assembly process.
    """

    name: str
    """The name of the dataset."""
    project: str
    """The name of the project for which the dataset is generated."""
    session_type: str | SessionTypes
    """The type of data acquisition sessions making up the dataset. At this time, datasets can only be created using 
    sessions of the same type."""
    acquisition_system: str | AcquisitionSystems
    """The acquisition system that acquired the sessions making up the dataset. At this time, datasets can only be 
    created using sessions acquired by the same acquisition system."""
    animals: list[AnimalDataset]
    """The list of AnimalDataset instances that specify the session selection criteria for each animal to be included 
    into the dataset."""

    def __post_init__(self):
        # Ensures that enumeration-mapped arguments are stored as proper enumeration types.
        self.session_type = SessionTypes(self.session_type)
        self.acquisition_system = AcquisitionSystems(self.acquisition_system)

        # Prevents initializing the class to construct a dataset from an unsupported type of sessions.
        if self.session_type not in _supported_sessions:
            message = (
                f"Unable to construct the dataset using the requested type of sessions {self.session_type} as it "
                f"is not supported. Use one of the supported session types: {_supported_sessions}."
            )
            console.error(message=message, error=ValueError)

        # Prevents initializing the class to construct a dataset from sessions acquired by an unsupported acquisition
        # system.
        if self.acquisition_system not in _supported_acquisition_systems:
            message = (
                f"Unable to construct the dataset using the sessions acquired by the requested acquisition system "
                f"{self.acquisition_system} as the system is not supported. Use sessions acquired by one of the "
                f"supported acquisition systems: {_supported_acquisition_systems}."
            )
            console.error(message=message, error=ValueError)

    def save(self, file_path: Path) -> None:
        """Saves instance data to the specified .yaml file."""
        original = copy.deepcopy(self)
        original.session_type = str(original.session_type)  # Converts session_type to string before saving.
        # Converts acquisition_system to string before saving.
        original.acquisition_system = str(original.acquisition_system)
        original.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> DatasetManifest:
        """Loads the data from the specified .yaml file and uses it to initialize and return the class instance."""
        return cls.from_yaml(file_path=file_path)


@dataclass
class ProcessedSessionData:
    """Stores the processed data of a single data acquisition session."""

    name: str
    """Stores the name of the session."""
    directory_path: Path
    """Stores the path to the session's data directory under the broader dataset structure."""
    data: pl.DataFrame = field(init=False)
    """Stores the memory-mapped contents of the session's data file as a Polars dataframe."""

    def __post_init__(self):
        """Loads the session's data by memory-mapping its data.feather file."""
        self.data = pl.read_ipc(
            source=self.directory_path.joinpath("data.feather"), use_pyarrow=True, memory_map=True, rechunk=True
        )


@dataclass
class AnimalData:
    """Stores the processed data for multiple sessions performed by a single animal."""

    name: int
    """Stores the unique identifier (name) of the animal."""
    directory_path: Path
    """Stores the path to the animal's data directory under the broader dataset structure."""
    sessions: dict[str, ProcessedSessionData]
    """Stores the processed data for each of the data acquisition sessions performed by the animal. The data for each 
    session is queryable by session name."""

    def get_session_data(self, name: str) -> ProcessedSessionData:
        """Returns the processed data for the specified session."""
        if name in self.sessions:
            return self.sessions[name]
        message = (
            f"Unable to retrieve the processed session data for the session {name}. The animal's data does not "
            f"contain a session with this name."
        )
        console.error(message, error=ValueError)
        # Fallback to appease mypy, should not be reachable
        raise ValueError(message)  # pragma: no cover

    @property
    def session_names(self) -> tuple[str, ...]:
        """Returns a tuple of all session names present in the animal's data."""
        return tuple(self.sessions.keys())


@dataclass
class ProjectData:
    def __init__(self, dataset_path: Path):
        # Resolves the path to the dataset's root directory
        # The root dataset directory is resolved through the presence of the dataset.manifest file. It is expected
        # that a single copy of the file is stored under the root directory of the dataset hierarchy.
        manifest_candidates = [candidate for candidate in dataset_path.rglob("manifest.yaml")]
        if len(manifest_candidates) != 1:
            message = (
                f"Unable to construct a ProjectData instance for the dataset stored under the path '{dataset_path}'. "
                f"Expected a single manifest.yaml file found under the input path, but found a total of "
                f"{len(manifest_candidates)} candidates."
            )
            console.error(message, error=ValueError)
            # Fallback to appease mypy, should not be reachable
            raise ValueError(message)  # pragma: no cover

        # Extracts the (only) manifest candidate path
        manifest_path = manifest_candidates.pop()

        # The parent of the manifest file path is the root dataset directory
        self._dataset_path: Path = manifest_path.parent

        # Loads the dataset's manifest data as a DatasetManifest instance
        self._manifest: DatasetManifest = DatasetManifest.load(file_path=manifest_path)

        # If the dataset contains a metadata file, loads the metadata as a Polars DataFrame. Otherwise, initializes
        # the metadata attribute to None.
        metadata_path = dataset_path.joinpath("metadata.feather")
        self._metadata: pl.DataFrame | None = None
        if metadata_path.exists():
            self._metadata = pl.read_ipc(source=metadata_path, use_pyarrow=True, memory_map=True, rechunk=True)

        self._animals: dict[int, AnimalData] = {}
        for animal in [candidate for candidate in self._dataset_path.glob("") if candidate.is_dir()]:
            session_data: dict[str, ProcessedSessionData] = {}
            for session in animal.rglob("data.feather"):
                session_data[session.parent.stem] = ProcessedSessionData(
                    name=session.parent.stem, directory_path=session
                )
            self._animals[int(animal.stem)] = AnimalData(
                name=int(animal.stem), directory_path=animal, sessions=session_data
            )

    @staticmethod
    def create(output_directory: Path, dataset_name: str, project: str, dataset_type: int | DatasetTypes) -> None:
        """Creates the requested project dataset hierarchy and a precursor manifest.feather file used to select the
        data for the dataset integration.

        This function sets up the root dataset directory at the specified path and partially configures the manifest
        file for the dataset. It functions as the initial access point for creating all analysis datasets in the Sun
        lab.

        Notes:
            It is expected that the user finishes the dataset creation process by editing the manifest file and calling
            the 'forge' method of the initialized dataset's ProjectData instance.

        Args:
            output_directory: The directory where to create the dataset hierarchy.
            dataset_name: The name of the dataset.
            project: The name of the project for which the dataset is created.
            dataset_type: A DatasetTypes enumeration member that specifies the type of the dataset.
        """
        # Ensures that the dataset type is one of the supported types.
        dataset_type = DatasetTypes(dataset_type)

        # Resolves and creates the dataset directory
        dataset_path = output_directory / dataset_name
        ensure_directory_exists(dataset_path)

        # Depending on the requested dataset type, creates a precursor dataset manifest file.
        if dataset_type == DatasetTypes.MESOSCOPE_VR_LICK_TRAINING:
            precursor_manifest = DatasetManifest(
                name=dataset_name,
                project=project,
                session_type=SessionTypes.LICK_TRAINING,
                acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
                animals=[AnimalDataset()],
            )
        elif dataset_type == DatasetTypes.MESOSCOPE_VR_RUN_TRAINING:
            precursor_manifest = DatasetManifest(
                name=dataset_name,
                project=project,
                session_type=SessionTypes.RUN_TRAINING,
                acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
                animals=[AnimalDataset()],
            )
        elif dataset_type == DatasetTypes.MESOSCOPE_VR_EXPERIMENT:
            precursor_manifest = DatasetManifest(
                name=dataset_name,
                project=project,
                session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
                acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
                animals=[AnimalDataset()],
            )
        else:
            message = (
                f"Unable to create the dataset '{dataset_name}' for the project {project}. Unsupported dataset "
                f"type code {dataset_type} encountered when resolving the precursor dataset manifest file. Use one "
                f"of the supported members of the DatasetTypes enumeration."
            )
            console.error(message, error=ValueError)
            raise ValueError(message)  # Fallback to appease mypy, should not be reachable

        manifest_path = dataset_path / "manifest.yaml"
        precursor_manifest.save(file_path=manifest_path)

    @staticmethod
    def _parse_date_boundary(date_string: str, is_end_date: bool = False) -> datetime:
        """Parses the input date and time string preserving any time information provided.

        Args:
            date_string: A Date and Time string in various formats (YYYY-MM-DD or with time).
            is_end_date: If True and only the date data is provided in the string, sets the time component to the
                end of the day.

        Returns:
            The Timezone-aware datetime object in America/New_York timezone constructed from the input string's data.
        """
        parsed = parser.parse(date_string)

        # Checks if only the date was provided (parser defaults to midnight)
        date_only = "T" not in date_string and " " not in date_string and ":" not in date_string

        if date_only and is_end_date:
            # Makes end dates inclusive of the entire day
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Ensures timezone awareness
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            parsed = parsed.astimezone(ZoneInfo("America/New_York"))

        return parsed

    def forge(self, project_manifest: ProjectManifest) -> None:
        """Filters the project manifest's session data according to rules defined in a YAML filter file.

        This function reads the filtering criteria from the specified filter file and applies them to the
        manifest's internal dataframe in place. Filtering rules can limit sessions by animal ID, date range,
        explicit inclusion or exclusion lists, training type flags, and dataset readiness. Any sessions
        excluded because they are not yet ready for integration (`dataset == 0`) will trigger a warning message.

        Args:
            project_manifest: The initialized ProjectManifest instance for the dataset's project.

        Notes:
            This method modifies the `manifest._data` attribute directly and is intended for use when curating
            a subset of sessions for analysis or processing. Although currently implemented as a static helper
            method, it could be refactored into the `ProjectManifest` class itself to avoid passing the manifest
            object explicitly.
        """
        # Initializes the result dictionary
        result = {}

        # Extracts the project's Polars DataFrame from the manifest instance
        df = project_manifest.data

        # Filters the dataframe by session type, acquisition system, and session data completeness status
        df_filtered = df.filter(
            (pl.col("type") == str(self._manifest.session_type))
            & (pl.col("system") == str(self._manifest.acquisition_system))
            & (pl.col("complete") == 1)
        )

        # Filters the dataframe for each animal specified in the dataset manifest file
        for animal_dataset in self._manifest.animals:
            animal_id = animal_dataset.animal

            # Filters for the target animal ID
            animal_df = df_filtered.filter(pl.col("animal") == animal_id)

            # If no sessions are found for the target animal, skips processing the animal
            if animal_df.is_empty():
                console.echo(
                    message=(
                        f"No complete sessions with type {self._manifest.session_type} and "
                        f"acquisition system {self._manifest.acquisition_system} found for animal {animal_id}. "
                        f"Excluding the animal from dataset integration..."
                    ),
                    level=LogLevel.WARNING,
                )
                result[animal_id] = []
                continue

            # Parses the session date and time range. Converts to EDT/EST timezone for comparison with session dates.
            start_date = self._parse_date_boundary(date_string=animal_dataset.start_date, is_end_date=False)
            end_date = self._parse_date_boundary(date_string=animal_dataset.end_date, is_end_date=True)

            # Applies the date and time filter OR include list
            # Sessions are included if they fall within the date range OR are in the include list
            date_filter = (pl.col("date") >= start_date) & (pl.col("date") <= end_date)

            # Include list override
            if animal_dataset.include:
                include_filter = pl.col("session").is_in(animal_dataset.include)
                combined_filter = date_filter | include_filter
            else:
                combined_filter = date_filter

            # Filters the animal dataset to only include the requested sessions
            animal_df = animal_df.filter(combined_filter)

            # Applies the exclusion list (takes precedence over everything)
            if animal_dataset.exclude:
                animal_df = animal_df.filter(~pl.col("session").is_in(animal_dataset.exclude))

            # Additional filtering: excludes sessions not ready for dataset integration:

            # These processing tasks must be carried out for all session types
            readiness_conditions = [pl.col("integrity") == 1, pl.col("behavior") == 1]

            # Mesoscope experiment also requires the 'suite2p' processing
            if self._manifest.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
                readiness_conditions.append(pl.col("suite2p") == 1)

            # Combines all readiness conditions (all must be true)
            readiness_filter = pl.all_horizontal(readiness_conditions)

            # Finds sessions that are NOT ready (inverts the filter)
            excluded_sessions = animal_df.filter(~readiness_filter)

            # If the session range contains sessions not ready for dataset integration, excludes them from processing.
            if not excluded_sessions.is_empty():
                # For each excluded session, determines the exclusion criteria to display them as a warning message.
                for row in excluded_sessions.iter_rows(named=True):
                    session_name = row["session"]
                    missing_steps = []

                    if row["integrity"] == 0:
                        missing_steps.append("integrity")
                    if row["behavior"] == 0:
                        missing_steps.append("behavior")
                    if self._manifest.session_type == SessionTypes.MESOSCOPE_EXPERIMENT and row["suite2p"] == 0:
                        missing_steps.append("suite2p")

                    console.echo(
                        message=(
                            f"The session {session_name} for animal {animal_id} is missing processing steps:"
                            f" {', '.join(missing_steps)}. Excluding the session from dataset integration..."
                        ),
                        level=LogLevel.WARNING,
                    )
            # Filters out the sessions that are not ready for the dataset integration
            animal_df = animal_df.filter(readiness_filter)

            # Extracts the session names that passed the filtering and appends them to the output list for the
            # processed animal
            session_names = animal_df["session"].sort().to_list()
            result[animal_id] = session_names

            # Notifies the user about the filtering outcome for each animal.
            console.echo(
                message=(
                    f"Animal {animal_id}: Processed. Selected {len(session_names)} sessions "
                    f"(date range: {animal_dataset.start_date} to {animal_dataset.end_date})."
                ),
                level=LogLevel.SUCCESS,
            )

    @staticmethod
    def parse_session(session_name):
        """If the session matches the form YYYY-MM-DD-HH-MM-SS-microseconds,
        return only 'MM-DD'. Otherwise, return the session unchanged.
        """
        pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+$"

        if isinstance(session_name, str) and re.match(pattern, session_name):
            try:
                # Only use the date part before the first dash after YYYY-MM-DD
                date_part = "-".join(session_name.split("-")[:3])
                dt = datetime.strptime(date_part, "%Y-%m-%d")
                return dt.strftime("%m-%d")
            except ValueError:
                pass  # If parsing fails, return the original

        return session_name
