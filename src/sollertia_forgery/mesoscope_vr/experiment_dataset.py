"""Provides the Mesoscope-VR experiment-session data-assembly worker donated to the system-agnostic forging pipeline.
The worker combines a mesoscope experiment session's fluorescence, behavior, runtime, and video sub-datasets on the
fluorescence reference clock into the session's unified ``data.feather``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import (
    SessionData,
    RawDataFiles,
    ProcessingTrackers,
    MesoscopeExperimentConfiguration,
)

from .metadata import BehaviorDataFiles
from .video_dataset import assemble_video_dataset
from ..shared_assets import multi_recording_dataset_directory
from .runtime_dataset import assemble_runtime_dataset, mask_non_run_experiment_data
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

    Notes:
        Requires a fully processed mesoscope experiment session: the experiment configuration and the single- and
        multi-recording cindra outputs must be present on disk.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, combined with the animal identifier to resolve the cindra
            multi-recording output directory.

    Raises:
        FileNotFoundError: If the session's processed microcontroller-data, runtime-data, or single-recording cindra
            output directory is missing, or if the multi-recording cindra output (cell_fluorescence.npy and its
            companions under the resolved multi-recording directory) is absent.
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

    # cindra writes the multi-recording dataset directory as ``{animal_id}_{dataset_name}`` (lowercased) to avoid
    # collisions when batching multiple animals under a single forged dataset name. The same shared helper the forging
    # pipeline uses to name the directory resolves it here.
    multiday_data_path = session.processed_data.cindra_multi_recording_path.joinpath(
        multi_recording_dataset_directory(animal_id=str(session.animal_id), dataset_name=dataset_name)
    )

    # Ensures the output directory exists before any sub-dataset assembly runs.
    ensure_directory_exists(path=output_path)

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
    results: dict[str, pl.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {executor.submit(task): name for name, task in tasks.items()}
        for future in as_completed(future_to_name):
            results[future_to_name[future]] = future.result()

    # Concatenates the sub-datasets into the unified feather, masks non-run experiment columns, and writes it
    # uncompressed so downstream consumers can memory-map it. The video sub-dataset joins only when it produced
    # columns, so a session processed without camera data still forges.
    sub_datasets = [fluorescence_data, results["behavior"], results["runtime"]]
    if results["video"].width > 0:
        sub_datasets.append(results["video"])
    result = pl.concat(items=sub_datasets, how="horizontal")
    result = mask_non_run_experiment_data(experiment_data=result)
    result = _clip_to_runtime_end(experiment_data=result, runtime_data_path=runtime_data_path)
    result.write_ipc(file=output_path)


def _clip_to_runtime_end(experiment_data: pl.DataFrame, runtime_data_path: Path) -> pl.DataFrame:
    """Discards the assembled samples acquired after the session's runtime ended.

    Notes:
        Session teardown stops the acquisition assets in sequence, so each asset contributes data for a different
        span past the end of the runtime. The cameras stop about a second after the runtime, the mesoscope continues
        for several more seconds, and the microcontrollers log for several more minutes. Every sub-dataset aligns to
        the fluorescence clock, so the samples in that trailing span carry the last camera value held constant rather
        than acquired data. Clipping the fully assembled dataset at the final runtime-state entry removes that span
        from every column at once, which keeps the sub-dataset assemblers free of teardown-specific handling.

    Args:
        experiment_data: The fully assembled and masked experiment DataFrame, ordered by the fluorescence clock.
        runtime_data_path: The path to the session's processed runtime-data directory.

    Returns:
        The experiment DataFrame containing only the samples acquired at or before the end of the runtime.
    """
    runtime_state_data = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.RUNTIME_STATE), memory_map=True
    )
    runtime_end_time = runtime_state_data["time_us"][-1]
    return experiment_data.filter(pl.col("time_us") <= runtime_end_time)
