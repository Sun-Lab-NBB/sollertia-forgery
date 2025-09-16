import re
import copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import field, dataclass
import polars as pl

import yaml
from dateutil import parser
from ..utils import ProjectManifest
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig
from sl_shared_assets import SessionTypes, AcquisitionSystems

# Stores the types of sessions that currently support dataset integration.
_supported_sessions = (SessionTypes.MESOSCOPE_EXPERIMENT, SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING)

# Stores the acquisition systems that currently support dataset integration
_supported_acquisition_systems = (AcquisitionSystems.MESOSCOPE_VR,)


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
    include: list[str] = field(default_factory=lambda: ["2025-07-14-13-49-04-018601"])
    """The sessions to include in the dataset even if they fall outside of the `start_date` / `end_date` range. This 
    field must use the full session ID (name), rather than a shortened session date."""
    exclude: list[str] = field(default_factory=lambda: ["2025-07-21-11-50-11-637172", "2025-07-22-12-54-42-553484"])
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
        self.to_yaml(file_path=file_path)

    @classmethod
    def load(cls, file_path: Path) -> "DatasetManifest":
        """Loads the data from the specified .yaml file and uses it to initialize and return the class instance."""
        return cls.from_yaml(file_path=file_path)


@dataclass
class ProcessedSessionData:
    name: str
    """Stores the name of the session."""
    directory_path: Path
    """Stores the path to the session's directory under the broader dataset structure."""
    metadata: pl.DataFrame = field(init=False)
    """Stores the memory-mapped contents of the session's data file as a Polars dataframe."""
    data: pl.DataFrame = field(init=False)
    """Stores the memory-mapped contents of the session's data file as a Polars dataframe."""

    def __post_init__(self):
        """Loads the session's data and metadata by memory-mapping their respective .feather files."""
        # memory-maps the session's data
        self.data = pl.read_ipc(source=self.directory_path.joinpath("data"), use_pyarrow=True, memory_map=True, rechunk=True)
        self.metadata = pl.read_ipc(source=self.directory_path.joinpath("data"), use_pyarrow=True, memory_map=True,
                                rechunk=True)


@dataclass
class AnimalData:
    name: int
    sessions: list[ProcessedSessionData]
    root_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        for session in self.sessions:
            session.resolve_paths(root_directory / session.name)

    def make_directories(self):
        ensure_directory_exists(self.root_path)
        for session in self.sessions:
            session.make_directories()

    def get_session(self, name: str):
        for session in self.sessions:
            if session.name == name:
                return session
        console.error(f"Session {name} is not present.", error=ValueError)


@dataclass
class ProjectData(YamlConfig):
    name: str
    animals: list[AnimalData]
    manifest: ProjectManifest
    root_path: Path = Path()

    # TODO Ivan I think this should really be part of the ProjectManifest class as opposed to a helper function here
    # It is easier to leave it here for now because then I don't have to update sl_shared_assets but this function could
    # very easily be moved, you would just need to replace manifest with self
    @staticmethod
    def filter_manifest(manifest: ProjectManifest, filter_path: Path) -> None:
        """Filters the project manifest's session data according to rules defined in a YAML filter file.

        This function reads the filtering criteria from the specified filter file and applies them to the
        manifest's internal dataframe in place. Filtering rules can limit sessions by animal ID, date range,
        explicit inclusion or exclusion lists, training type flags, and dataset readiness. Any sessions
        excluded because they are not yet ready for integration (`dataset == 0`) will trigger a warning message.

        Args:
            manifest: The project manifest object whose `_data` attribute (a Polars DataFrame) will be updated.
            filter_path: Path to a YAML file specifying filtering rules. The file must include:
                - `animals`: list of allowed animal IDs.
                - `sessions.start_date` / `sessions.end_date`: date range for allowed sessions.
                - `sessions.include`: list of explicitly included session IDs.
                - `sessions.exclude`: list of explicitly excluded session IDs.
                - `include_lick_training` (bool): whether to keep "lick training" sessions.
                - `include_run_training` (bool): whether to keep "run training" sessions.

        Notes:
            This method modifies the `manifest._data` attribute directly and is intended for use when curating
            a subset of sessions for analysis or processing. Although currently implemented as a static helper
            method, it could be refactored into the `ProjectManifest` class itself to avoid passing the manifest
            object explicitly.
        """
        with filter_path.open() as f:
            filter = yaml.safe_load(f)

        df = manifest._data

        if "animals" in filter:
            df = df.filter(pl.col("animal").is_in(filter["animals"]))

        if "sessions" in filter:
            include_lst = [] if "include" not in filter else filter["sessions"]["include"]
            if "start" in filter["sessions"]:
                start = parser.parse(filter["sessions"]["start_date"]).astimezone(ZoneInfo("America/New_York"))
                df = df.filter(pl.col("date") >= start | pl.col("session").is_in(include_lst))
            if "end" in filter["sessions"]:
                end = parser.parse(filter["sessions"]["end_date"]).astimezone(ZoneInfo("America/New_York"))
                df = df.filter(pl.col("date") <= end | pl.col("session").is_in(include_lst))
            if "exclude" in filter["sessions"]:
                df = df.filter(~pl.col("session").is_in(filter["sessions"]["exclude"]))

        if filter.get("exclude_lick_training"):
            df = df.filter(pl.col("type") != "lick training")

        if filter.get("exclude_run_training"):
            df = df.filter(pl.col("type") != "run training")

        for session_name in df.filter(pl.col("dataset") == 0)["session"]:
            console.echo(
                f"Excluded session {session_name}, which has data that is not ready to be integrated into the dataset.",
                level=LogLevel.WARNING)
        df = df.filter(pl.col("dataset") != 0)

        manifest._data = df

    @classmethod
    def create(cls, project_name: str, working_directory: Path, manifest_path: Path,
               filter_path: Path) -> "ProjectData":

        manifest = ProjectManifest(manifest_path)
        ProjectData.filter_manifest(manifest, filter_path)

        project_path = working_directory / project_name
        animals = list(AnimalData(name=animal_id, sessions=list(
            ProcessedSessionData(name=session_name) for session_name in manifest.get_sessions(animal_id))) for animal_id
                       in manifest.animals)
        for animal in animals:
            animal.resolve_paths(root_directory=project_path / str(animal.name))
            animal.make_directories()

        instance = ProjectData(
            name=project_name,
            animals=animals,
            manifest=manifest
        )

        instance.root_path = project_path

        instance._save()
        return instance

    @classmethod
    def load(cls, working_directory: Path, yml_path: Path):
        instance: ProjectData = cls.from_yaml(yml_path)

        instance.root_path = working_directory / instance.name
        for animal in instance.animals:
            animal.resolve_paths(root_directory=instance.root_path / str(animal.name))
            animal.make_directories()

        return instance

    def _save(self) -> None:
        origin = copy.deepcopy(self)

        origin.root_path = None
        for animal in origin.animals:
            animal.root_path = None
            for session in animal.sessions:
                session.root_path = None
                session.behavior_data = None
                session.single_day_data = None
                session.multi_day_data = None

        origin.to_yaml(file_path=self.root_path / "project_data.yaml")

    @staticmethod
    def parse_session(session_name):
        """
        If the session matches the form YYYY-MM-DD-HH-MM-SS-microseconds,
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

    def get_animal(self, name: int | str):
        for mouse in self.animals:
            if str(mouse.name) == str(name):
                return mouse
        console.error(f"Animal {name} is not present.", error=ValueError)

    def get_session(self, name: str):
        return self.get_animal(self.manifest.get_session_info(name)['animal'].item()).get_session(name)