from pathlib import Path
import shutil as sh


def collect_single_day_data(source_root: Path, destination_root: Path) -> None:
    combined_path = source_root / "combined"

    files = {
        combined_path / "F.npy",
        combined_path / "Fneu.npy",
        combined_path / "Fsub.npy",
        combined_path / "iscell.npy",
        combined_path / "ops.npy",
        combined_path / "spks.npy",
        combined_path / "stat.npy",
        source_root / "single_day_ss2p_configuration.yaml"
    }

    for file in files:
        sh.copy2(src=file, dst=destination_root.joinpath(file.name))


def collect_multi_day_data(source_root: Path, destination_root: Path) -> None:
    files = {
        source_root / "F.npy",
        source_root / "Fneu.npy",
        source_root / "Fsub.npy",
        source_root / "ops.npy",
        source_root / "spks.npy",
        source_root / "original_images.npy",
        source_root / "transformed_images.npy",
        source_root / "unregistered_masks.npy",
        source_root / "registered_masks.npy",
        source_root / "shared_multiday_masks.npy",
        source_root / "session_multiday_masks.npy",
        source_root / "single_day_ss2p_configuration.yaml",
        source_root.parent / "multi_day_ss2p_configuration.yaml"
    }

    for file in files:
        sh.copy2(src=file, dst=destination_root.joinpath(file.name))