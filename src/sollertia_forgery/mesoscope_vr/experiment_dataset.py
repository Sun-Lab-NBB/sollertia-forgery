"""Provides the Mesoscope-VR experiment-session data-assembly worker donated to the system-agnostic forging pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import reduce, partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from cindra import resolve_dataset_path
import polars as pl
from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import (
    SessionData,
    RawDataFiles,
    ProcessingTrackers,
    MesoscopeExperimentConfiguration,
)

from .video_dataset import assemble_video_dataset
from ..shared_assets import multi_recording_dataset_name
from .runtime_dataset import clip_to_session_bounds, assemble_runtime_dataset, mask_non_run_experiment_data
from .behavior_dataset import assemble_behavior_dataset
from .two_photon_dataset import assemble_cindra_dataset

if TYPE_CHECKING:
    from pathlib import Path


def assemble_experiment_dataset(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Assembles a single Mesoscope-VR experiment session's unified data feather.

    Combines the session's fluorescence, behavior, runtime, and video sub-datasets into a single time-aligned Polars
    DataFrame, written as an uncompressed ``data.feather`` at ``output_path``. The fluorescence sub-dataset is
    assembled first because its ``time_us`` column is the reference clock the other sub-datasets align to. The video
    sub-dataset is optional and contributes columns only when the session carries processed camera feathers. The
    meaning of each emitted column is documented by ``DatasetColumn`` and donated to the dataset's
    ``data_descriptions.feather`` via ``MESOSCOPE_COLUMN_DESCRIPTIONS``.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, combined with the animal identifier to resolve the cindra
            multi-recording output directory.

    Raises:
        FileNotFoundError: If the session's processed microcontroller-data, runtime-data, or single-recording cindra
            output directory is missing. Also raised when the multi-recording cindra output (cell_fluorescence.npy
            and its companions under the resolved multi-recording directory), the session's experiment
            configuration, or its hardware state file is absent.
        ValueError: If a sub-dataset cannot be assembled (for example, the ScanImage fallback alignment cannot
            recover the expected frame count, or a required hardware-state field is missing).
    """
    session = SessionData.load(session_path=source_session_path)

    microcontroller_data_path = session.processed_data.microcontroller_data_path
    runtime_data_path = session.processed_data.runtime_data_path
    cindra_data_path = session.processed_data.cindra_data_path
    video_data_path = session.processed_data.video_data_path
    raw_data_path = session.raw_data_path

    # Validates that the processed microcontroller, runtime, and single-recording cindra outputs exist before any
    # expensive work. The parsed behavior feathers are split across the per-worker ``microcontroller_data`` (module
    # parsing) and ``runtime_data`` (runtime decode) directories.
    if not microcontroller_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the processed "
            f"microcontroller data directory '{microcontroller_data_path}' to exist and contain "
            f"'{ProcessingTrackers.MICROCONTROLLER}'."
        )
        console.error(message=message, error=FileNotFoundError)
    if not runtime_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the processed runtime "
            f"data directory '{runtime_data_path}' to exist and contain '{ProcessingTrackers.RUNTIME}'."
        )
        console.error(message=message, error=FileNotFoundError)
    if not cindra_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the single-recording "
            f"cindra output directory '{cindra_data_path}' to exist and contain "
            f"'{ProcessingTrackers.TWO_PHOTON}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # The forging pipeline qualifies the dataset name with the animal identifier, so an animal's multi-recording
    # output stays separate when a dataset spans several animals. cindra owns the directory that name resolves to, so
    # its own resolver locates it here rather than this module respelling the layout.
    multiday_data_path = resolve_dataset_path(
        output_root=session.processed_data_path,
        dataset_name=multi_recording_dataset_name(animal_id=str(session.animal_id), dataset_name=dataset_name),
    )

    ensure_directory_exists(path=output_path, is_file=True)

    # Loads the experiment configuration once so the runtime assembly resolves its state and trial mappings without
    # re-reading the same YAML.
    experiment_configuration = MesoscopeExperimentConfiguration.from_yaml(
        file_path=raw_data_path.joinpath(RawDataFiles.EXPERIMENT_CONFIGURATION)
    )

    # Assembles the fluorescence sub-dataset first. Its ``time_us`` column is the reference clock for the other two.
    fluorescence_data = assemble_cindra_dataset(
        cindra_data_path=cindra_data_path,
        microcontroller_data_path=microcontroller_data_path,
        multiday_data_path=multiday_data_path,
        raw_data_path=raw_data_path,
    )
    reference_time = fluorescence_data["time_us"].to_numpy()

    # Assembles the behavior, runtime, and video sub-datasets in parallel. All three align to the fluorescence
    # reference clock. The video sub-dataset is empty when the session carries no processed camera feathers.
    tasks = {
        "behavior": partial(
            assemble_behavior_dataset,
            microcontroller_data_path=microcontroller_data_path,
            runtime_data_path=runtime_data_path,
            raw_data_path=raw_data_path,
            reference_time=reference_time,
            drop_time_columns=True,
        ),
        "runtime": partial(
            assemble_runtime_dataset,
            microcontroller_data_path=microcontroller_data_path,
            runtime_data_path=runtime_data_path,
            experiment_configuration=experiment_configuration,
            reference_time=reference_time,
        ),
        "video": partial(
            assemble_video_dataset,
            video_data_path=video_data_path,
            reference_time=reference_time,
        ),
    }
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_name = {executor.submit(task): name for name, task in tasks.items()}
        results: dict[str, pl.DataFrame] = {
            future_to_name[future]: future.result() for future in as_completed(future_to_name)
        }

    # Stacks the sub-datasets into the unified feather, masks non-run experiment columns, and writes it uncompressed so
    # downstream consumers can memory-map it. Stacking requires every sub-dataset to carry the reference clock's
    # height, so one that drifts off that clock raises rather than being padded. The video sub-dataset joins only when
    # it produced columns, so a session processed without camera data still forges.
    sub_datasets = [fluorescence_data, results["behavior"], results["runtime"]]
    if results["video"].width > 0:
        sub_datasets.append(results["video"])
    result = reduce(pl.DataFrame.hstack, sub_datasets)
    result = mask_non_run_experiment_data(experiment_data=result)
    result = clip_to_session_bounds(assembled_data=result, runtime_data_path=runtime_data_path)
    result.write_ipc(file=output_path)
