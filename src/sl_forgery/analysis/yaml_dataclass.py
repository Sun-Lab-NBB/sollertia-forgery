from dataclasses import dataclass, field
from pathlib import Path

from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import YamlConfig

@dataclass
class BehaviorData:
    root_path: Path = Path()
    behavior_path: Path = Path()
    desktop_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path: Path = root_directory
        self.behavior_path = root_directory / "behavior_at_frame.feather"
        self.desktop_path = root_directory / "desktop.ini" # TODO: Decide if this should be kept


    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)

@dataclass
class SingleDayData():
    root_path: Path = Path()
    desktop_path: Path = Path()
    F_path: Path = Path()
    Fneu_path: Path = Path()
    iscell_path: Path = Path()
    ops_path: Path = Path()
    umap_embedding_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        self.desktop_path = root_directory / "desktop.ini" # TODO: Decide if this should be kept
        self.F_path = root_directory / "F.npy"
        self.Fneu_path = root_directory / "Fneu.npy"
        self.iscell_path = root_directory / "iscell.npy"
        self.ops_path = root_directory / "ops.npy"
        self.umap_embedding_path = root_directory / "umap_embedding.npy"

    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)

@dataclass
class MultiDayData():
    root_path: Path = Path()
    backwards_deformed_cell_masks_path: Path = Path()
    desktop_path: Path = Path()
    F_path: Path = Path()
    Fneu_path: Path = Path()
    ops_path: Path = Path()
    original_images_path: Path = Path()
    registered_masks_path: Path = Path()
    session_multiday_masks_path: Path = Path()
    shared_multiday_masks_path: Path = Path()
    single_day_s2p_configuration_path: Path = Path()
    spks_path: Path = Path()
    template_cell_masks_path: Path = Path()
    transformed_images_path: Path = Path()
    unregistered_masks_path: Path = Path()
    umap_embedding_path: Path = Path()

    def resolve_paths(self, root_directory: Path) -> None:
        self.root_path = root_directory
        self.backwards_deformed_cell_masks_path = root_directory / "backwards_deformed_cell_masks.npy"
        self.desktop_path = root_directory / "desktop.ini" # TODO: Decide if this should be kept
        self.F_path = root_directory / "F.npy"
        self.Fneu_path = root_directory / "Fneu.npy"
        self.ops_path = root_directory / "ops.npy"
        self.original_images_path = root_directory / "original_images.npy"
        self.registered_masks_path = root_directory / "registered_masks.npy"
        self.session_multiday_masks_path = root_directory / "session_multiday_masks.npy"
        self.shared_multiday_masks_path = root_directory / "shared_multiday_masks.npy"
        self.single_day_s2p_configuration_path = root_directory / "single_day_s2p_configuration.yaml"
        self.spks_path = root_directory / "spks.npy"
        self.template_cell_masks_path = root_directory / "template_cell_masks.npy"
        self.transformed_images_path = root_directory / "transformed_images.npy"
        self.unregistered_masks_path = root_directory / "unregistered_masks.npy"
        self.umap_embedding_path = root_directory / "umap_embedding.npy"

    def make_directories(self) -> None:
        ensure_directory_exists(self.root_path)

@dataclass
class ProcessedSessionData():
    behavior_data: BehaviorData = field(default_factory=lambda: BehaviorData)
    single_day_data: SingleDayData = field(default_factory=lambda: SingleDayData)
    multi_day_data: MultiDayData = field(default_factory=lambda: MultiDayData)

    @classmethod
    def create(cls) -> "ProcessedSessionData":
        pass

    @classmethod
    def load(cls, session_path: Path):
        pass
    




@dataclass 
class MouseData():
    pass

@dataclass
class ProjectData(YamlConfig):
    pass
