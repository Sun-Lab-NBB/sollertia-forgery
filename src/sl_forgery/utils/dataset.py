import re
import copy
from enum import Enum
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import field, dataclass

import yaml
import numpy as np
import polars as pl
from dateutil import parser
from . import ProjectManifest
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import YamlConfig


class TargetGroup(str, Enum):
    SINGLE_DAY = "single_day"
    MULTI_DAY = "multi_day"


@dataclass
class BehaviorData:
    root_path: Path = Path()
    encoder_data_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path: Path = root_directory
        self.encoder_data_path = root_directory / "encoder_data.feather"

    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)


@dataclass
class SingleDayData:
    root_path: Path = Path()
    fluorescence_path: Path = Path()
    neuropil_fluorescence_path: Path = Path()
    subtracted_fluorescence_path: Path = Path()
    cell_classification_path: Path = Path()
    spikes_path: Path = Path()
    umap_embedding_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        self.fluorescence_path = root_directory / "F.npy"
        self.neuropil_fluorescence_path = root_directory / "Fneu.npy"
        self.subtracted_fluorescence_path = root_directory / "Fsub.npy"
        self.cell_classification_path = root_directory / "iscell.npy"
        self.spikes_path = root_directory / "spks.npy"
        self.umap_embedding_path = root_directory / "umap_embedding.npy"

    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)


@dataclass
class MultiDayData:
    selection_name: str = ""
    root_path: Path = Path()
    fluorescence_path: Path = Path()
    neuropil_fluorescence_path: Path = Path()
    subtracted_fluorescence_path: Path = Path()
    cell_classification_path: Path = Path()
    spikes_path: Path = Path()
    umap_embedding_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        self.fluorescence_path = root_directory / "F.npy"
        self.neuropil_fluorescence_path = root_directory / "Fneu.npy"
        self.subtracted_fluorescence_path = root_directory / "Fsub.npy"
        self.cell_classification_path = root_directory / "iscell.npy"
        self.spikes_path = root_directory / "spks.npy"
        self.umap_embedding_path = root_directory / "umap_embedding.npy"

    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)


@dataclass
class ProcessedSessionData:
    name: str
    root_path: Path
    behavior_data: BehaviorData
    single_day_data: SingleDayData
    multi_day_data: list[MultiDayData]

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        if self.behavior_data is None:
            self.behavior_data = BehaviorData()
        self.behavior_data.resolve_paths(root_directory / "behavior")
        if self.single_day_data is None:
            self.single_day_data = SingleDayData()
        self.single_day_data.resolve_paths(root_directory / "single_day")
        if self.multi_day_data is None:
            self.multi_day_data = [MultiDayData()]
        self.multi_day_data.resolve_paths(root_directory / "multi_day")

    def make_directories(self):
        ensure_directory_exists(self.root_path)
        self.behavior_data.make_directories()
        self.single_day_data.make_directories()
        self.multi_day_data.make_directories()


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

            Filtering hierarchy:
                1. Sessions must match the allowed `animals` list.
                2. Sessions in `sessions.include` are always kept, regardless of date range.
                3. Sessions must be within the `sessions.start_date` / `sessions.end_date` range unless
                   explicitly included in `sessions.include`.
                4. Sessions in `sessions.exclude` are always removed, even if explicitly included.
                5. If `include_lick_training` is false, remove all "lick training" sessions.
                6. If `include_run_training` is false, remove all "run training" sessions.
                7. Remove sessions where `dataset == 0` (not ready for integration).

            The filter file must be in YAML format and contain the following keys. Below is an example:

            ```yaml
            animals:
              - 11
              - 15
              - 16

            sessions:
              start_date: 2025-7-1
              end_date: 2025-8-1
              include:
                - 2025-07-14-13-49-04-018601
              exclude:
                - 2025-07-21-11-50-11-637172
                - 2025-07-22-12-54-42-553484

            exclude_lick_training: true
            exclude_run_training: true
            ```
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
        If session matches the form YYYY-MM-DD-HH-MM-SS-microseconds,
        return only 'MM-DD'. Otherwise return the session unchanged.
        """
        pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+$"

        if isinstance(session_name, str) and re.match(pattern, session_name):
            try:
                # Only use the date part before the first dash after YYYY-MM-DD
                date_part = "-".join(session_name.split("-")[:3])
                dt = datetime.strptime(date_part, "%Y-%m-%d")
                return dt.strftime("%m-%d")
            except ValueError:
                pass  # If parsing fails, return original

        return session_name

    def get_mouse(self, name: int | str):
        for mouse in self.animals:
            if str(mouse.name) == str(name):
                return mouse
        console.error(f"Mouse {name} is not present.", error=ValueError)

    def get_session(self, name: str):
        return self.get_mouse(self.manifest.get_session_info(name)['animal'].item()).get_session(name)