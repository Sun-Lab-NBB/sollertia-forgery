import re
import json
import shutil
from typing import Any, Optional
from pathlib import Path

from tqdm import tqdm
import zarr
import numpy as np
import pandas as pd
from natsort_rs import natsorted
import numcodecs


def _add_imaging_data(
    zarr_file: zarr.Group,
    folder_path: Path,
    group: str,
    counter: int,
    selected_cells: list[bool] | None = None,
) -> None:
    """Load, process, and add imaging data to the zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        folder_path (Path): Path to the data folder.
        group (str): Group name for data organization ('single_session' or 'multi_session').
        counter (int): Current session index.
        selected_cells (list[bool], optional): Boolean mask for selecting cells. Defaults to None.
    """
    # Load fluorescence data
    if selected_cells is not None:
        F = np.load(folder_path / "F.npy")[selected_cells, :]
        Fneu = np.load(folder_path / "Fneu.npy")[selected_cells, :]
    else:
        F = np.load(folder_path / "F.npy")
        Fneu = np.load(folder_path / "Fneu.npy")

    # Store raw fluorescence data
    zarr_file.require_group(f"{group}/F").create_dataset(str(counter), data=F, chunks=(1000, 10000))

    zarr_file.require_group(f"{group}/Fneu").create_dataset(str(counter), data=Fneu, chunks=(1000, 10000))

    for signal_type in ["Fns", "Fdemix", "spks"]:
        signal_path = folder_path / f"{signal_type}.npy"
        if (signal_path).exists():
            if selected_cells is not None:
                signal_data = np.load(signal_path)[selected_cells, :]
            else:
                signal_data = np.load(signal_path)
            zarr_file.require_group(f"{group}/{signal_type}").create_dataset(
                str(counter), data=signal_data, chunks=(1000, 10000)
            )


def _process_behavior_data(zarr_file: zarr.Group, data_info: dict[str, Any], data_path: str, counter: int) -> None:
    """Process VR data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        data_info (dict): Information about data locations.
        data_path (str): Path to the current session's data.
        counter (int): Current session index.

    Raises:
        NameError: If the Gimbl log file cannot be found or multiple log files are detected.
    """
    feather_files = {
        "brake": "brake_data.feather",
        "frame": "frame_data.feather",
        "valve": "valve_data.feather",
        "cue": "cue_data.feather",
        "lick": "lick_data.feather",
        "vr": "vr_data.feather",
        "encoder": "encoder_data.feather",
        "experiment": "experiment_data.feather",
        "torque": "torque_data.feather",
    }

    for key, feather_name in feather_files.items():
        file_path = Path(data_path) / feather_name
        if file_path.exists():
            df = pd.read_feather(file_path)
            df_array = df.to_records(index=False)
            zarr_file.require_group(f"behavior/{key}").create_dataset(str(counter), data=df_array)


def _process_single_session(zarr_file: zarr.Group, session_path: Path, settings: dict[str, Any], counter: int) -> None:
    """Process single session data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        session_path (Path): Path to the session data.
        settings (dict): Processing settings.
        counter (int): Current session index.
    """
    # Read cell info
    stat = np.load(session_path / "stat.npy", allow_pickle=True)
    iscell = np.load(session_path / "iscell.npy", allow_pickle=True)

    # Select valid cells
    selected_cells = [
        (iscell[icell, 1] > settings["cell_detection"]["prob_threshold"])
        and (mask["npix"] < settings["cell_detection"]["max_size"])
        for icell, mask in enumerate(stat)
    ]

    # Store cell info
    zarr_file.require_group("cells/single_session").create_dataset(
        str(counter), data=stat[selected_cells], dtype=object, object_codec=numcodecs.Pickle()
    )

    # Process and store imaging data
    _add_imaging_data(
        zarr_file,
        session_path,
        "single_session",
        counter,
        selected_cells,
    )


def _process_multi_session(
    zarr_file: zarr.Group,
    session_path: str,
    backwards_deformed_cm: np.ndarray,
    trans_images: np.ndarray,
    original_images: np.ndarray,
    counter: int,
    settings: dict[str, Any],
) -> None:
    """Process multi-session data and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        data_path (str): Path to the current session's data.
        backwards_deformed_cm (np.ndarray): Backwards deformed cell masks.
        trans_images (np.ndarray): Transformed images.
        original_images (np.ndarray): Original images.
        counter (int): Current session index.
        settings (dict): Processing settings.
    """
    stat = backwards_deformed_cm

    # Store fluorescence data
    _add_imaging_data(zarr_file, session_path, "multi_session", counter, None)

    # Store cell masks and images
    zarr_file.require_group("cells/multi_session/original").create_dataset(
        str(counter), data=stat, dtype=object, object_codec=numcodecs.Pickle()
    )

    zarr_file.require_group("images/registered").create_dataset(
        str(counter), data=trans_images, dtype=object, object_codec=numcodecs.Pickle()
    )

    zarr_file.require_group("images/original").create_dataset(
        str(counter), data=original_images, dtype=object, object_codec=numcodecs.Pickle()
    )


def _process_individual_session(
    zarr_file: zarr.Group, session_path: Path, cell_templates: np.ndarray, counter: int
) -> None:
    """Process individual session data (non-aligned) and store it in a zarr file.

    Args:
        zarr_file (zarr.Group): Open zarr file for writing.
        session_path (Path): Path to the session data.
        cell_templates (np.ndarray): Cell templates for reference.
        counter (int): Current session index.
    """
    # Read ops
    ops = np.load(session_path / "ops.npy", allow_pickle=True).item()

    # Create empty stat structure
    empty_stat = {}
    for key in ops["stat"][0].keys() if "stat" in ops else {"xpix", "ypix", "lam", "npix"}:
        empty_stat[key] = None

    zarr_file.require_group("cells/multi_session/original").create_dataset(
        str(counter), data=[empty_stat], dtype=object, object_codec=numcodecs.Pickle()
    )

    # Create empty fluorescence data
    vr_group = zarr_file["gimbl/vr"]
    vr_info = vr_group[str(counter)][()]

    num_frames = vr_info.position.frame.reset_index()["frame"].max() if hasattr(vr_info, "position") else 0
    num_cells = len(cell_templates)

    empty_array = np.empty((num_cells, num_frames))
    empty_array[:] = np.nan

    for field in ["F", "Fneu", "Fns", "Fdemix", "spks"]:
        zarr_file.require_group(f"multi_session/{field}").create_dataset(
            str(counter), data=empty_array, chunks=(1000, 10000)
        )

    # Store images
    imgs = {"mean_img": ops["meanImg"], "enhanced_img": ops["meanImgE"], "max_img": ops["max_proj"]}

    zarr_file.require_group("images/original").create_dataset(
        str(counter), data=imgs, dtype=object, object_codec=numcodecs.Pickle()
    )

    # Create empty registered images
    empty_img = np.zeros(ops["meanImg"].shape)
    empty_imgs = {key: empty_img for key in ["mean_img", "enhanced_img", "max_img"]}

    zarr_file.require_group("images/registered").create_dataset(
        str(counter), data=empty_imgs, dtype=object, object_codec=numcodecs.Pickle()
    )


def process_session_data(data_info: dict[str, Any], settings: dict[str, Any], sessions_data) -> None:
    """Process and aggregate multi-day experimental data.

    This function takes data information and settings, processes all sessions, and
    stores the results in a standardized zarr format.

    Args:
        data_info (dict): Information about data locations and parameters.
        settings (dict): Processing settings generated by parse_settings.

    Raises:
        NameError: If session paths are invalid or duplicated.
        FileNotFoundError: If required files cannot be found.

    Notes:
        Zarr organization structure:

        Signal data:
            - single_session/(F, Fdemix, Fneu, Fns, spks)/0   # Single session data
            - multi_session/(F, Fdemix, Fneu, Fns, spks)/0    # Multi-session aligned data

        Cell masks:
            - cells/single_session/0             # Cell data of original sessions
            - cells/multi_session/original/0     # Multi-session cell masks in original coordinates
            - cells/multi_session/registered/0   # Multi-session cell masks in registered coords

        Images:
            - images/original/0                  # Original session images
            - images/registered/0                # Registered session images

        VR Info:
            - gimbl/log/0                        # Raw pandas VR log
            - gimbl/vr/0                         # Processed VR data
    """
    # Create output directory
    save_folder = Path(data_info["data"]["save_folder"])
    save_folder.mkdir(parents=True, exist_ok=True)

    # Get imaging information (assumes all sessions have the consistent ops file)
    first_multiday_folder = (
        sessions_data[0].processed_data.mesoscope_data_path
        / data_info["data"]["suite2p_folder"]
        / data_info["data"]["multiday_output_folder"]
    )
    ops_file = first_multiday_folder / "ops.npy"
    ops = np.load(ops_file, allow_pickle=True).item()

    # Assumes all sessions were ran through the same multi-day registration pipeline
    cell_templates = np.load(first_multiday_folder / "template_cell_masks.npy", allow_pickle=True)

    # Update settings with imaging info
    settings["imaging"] = {"frame_rate": ops["fs"], "num_planes": ops["nplanes"]}

    # Copy necessary settings from ops
    for field in ["fs", "Lx", "Ly"]:
        settings["demix"][field] = ops[field]

    settings["animal"] = data_info["animal"]

    # Initialize zarr storage
    zarr_folder = save_folder / "vr2p.zarr"
    if zarr_folder.is_dir():
        print(f"Removing previous data at {zarr_folder}")
        shutil.rmtree(zarr_folder, ignore_errors=True)

    # Process data paths
    sessions_data_ord = natsorted(sessions_data, key=lambda sd: sd.session_name)
    nsessions = len(sessions_data_ord)

    # Create and populate zarr store
    with zarr.open(zarr_folder.as_posix(), mode="w") as f:
        # Store settings and cell templates
        f.create_dataset("meta", data=settings, dtype=object, object_codec=numcodecs.Pickle())
        f.create_dataset("sessions_data", data=sessions_data_ord, dtype=object, object_codec=numcodecs.Pickle())
        f.require_group("cells/multi_session").create_dataset(
            "registered", data=cell_templates, dtype=object, object_codec=numcodecs.Pickle()
        )

        # Process each session
        for counter, sess_data in tqdm(enumerate(sessions_data_ord), desc="Processing session", total=nsessions):
            multiday_folder = (
                sess_data.processed_data.mesoscope_data_path
                / data_info["data"]["suite2p_folder"]
                / data_info["data"]["multiday_output_folder"]
            )
            # check if multiday_folder exists
            multiday_exists = multiday_folder.is_dir()

            # Process VR data
            _process_behavior_data(f, data_info, sess_data.processed_data.behavior_data_path, counter)

            # Process single session data
            singleday_path = (
                sess_data.processed_data.mesoscope_data_path / data_info["data"]["suite2p_folder"] / "combined"
            )
            _process_single_session(f, singleday_path, settings, counter)

            # Process multi-session data if available
            if multiday_exists:
                # Load registration data
                backwards_deformed_masks = np.load(
                    multiday_folder / "backwards_deformed_cell_masks.npy", allow_pickle=True
                )
                trans_images = np.load(multiday_folder / "transformed_images.npy", allow_pickle=True).item()
                original_images = np.load(multiday_folder / "original_images.npy", allow_pickle=True).item()
                _process_multi_session(
                    f, multiday_folder, backwards_deformed_masks, trans_images, original_images, counter, settings
                )
            else:
                _process_individual_session(f, singleday_path, cell_templates, counter)
