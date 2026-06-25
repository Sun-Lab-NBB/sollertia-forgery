"""Provides the Mesoscope-VR data-assembly worker donated to the system-agnostic forging pipeline.

Notes:
    This module's sole public entry point, ``assemble_mesoscope_session``, is the Mesoscope-VR "data assembly" asset
    contributed to the central ``FORGING_ASSEMBLY_REGISTRY``; the agnostic forging pipeline resolves it by acquisition
    system and invokes it once per session to produce that session's ``data.feather``. The pipeline owns dataset
    definition, the cindra multi-day stage, tracker orchestration, the per-dataset column-description binding, and
    shared-asset re-export; this worker owns only the assembly of the Mesoscope-VR data.
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

from .fluorescence import assemble_cindra_dataset
from .runtime_dataset import assemble_runtime_dataset, _mask_non_run_experiment_data
from .behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path


def assemble_mesoscope_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Assembles a single Mesoscope-VR session's unified data feather.

    Combines the session's fluorescence, behavior, and runtime sub-datasets into a single time-aligned Polars
    DataFrame, written as an uncompressed ``data.feather`` at ``output_path``. The fluorescence sub-dataset is
    assembled first because its ``time_us`` column is the reference clock the behavior and runtime sub-datasets align
    to. The meaning of each emitted column is documented by ``DatasetColumn`` and donated to the dataset's
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
        FileNotFoundError: If the session's processed behavior data directory or single-recording cindra output
            directory is missing.
        ValueError: If a sub-dataset cannot be assembled (for example, the ScanImage fallback alignment cannot
            recover the expected frame count, or a required hardware-state field is missing).
    """
    session = SessionData.load(session_path=source_session_path)

    behavior_data_path = session.processed_data.behavior_data_path
    cindra_data_path = session.processed_data.cindra_data_path
    raw_data_path = session.raw_data_path

    # Validates that the canonical behavior and single-recording cindra outputs exist before any expensive work.
    if not behavior_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the processed behavior "
            f"data directory '{behavior_data_path}' to exist and contain '{ProcessingTrackers.BEHAVIOR}'."
        )
        console.error(message=message, error=FileNotFoundError)
    if not cindra_data_path.is_dir():
        message = (
            f"Unable to assemble the data for session '{source_session_path.name}'. Expected the single-recording "
            f"cindra output directory '{cindra_data_path}' to exist and contain "
            f"'{ProcessingTrackers.CINDRA_SINGLE_RECORDING}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # Cindra writes the multi-recording dataset directory as ``{animal_id}_{dataset_name}`` to avoid collisions when
    # batching multiple animals under a single forged dataset name, so the animal identifier is prepended here.
    multiday_data_path = session.processed_data.cindra_multi_recording_path.joinpath(
        f"{session.animal_id}_{dataset_name}"
    )

    # Ensures the output directory exists before any sub-dataset assembly runs.
    ensure_directory_exists(path=output_path)

    # Loads the experiment configuration once so the runtime assembly resolves its state and trial mappings without
    # re-reading the same YAML.
    experiment_configuration = MesoscopeExperimentConfiguration.from_yaml(
        file_path=raw_data_path.joinpath(RawDataFiles.EXPERIMENT_CONFIGURATION)
    )

    # Assembles the fluorescence sub-dataset first; its ``time_us`` column is the reference clock for the other two.
    fluorescence_data = assemble_cindra_dataset(
        cindra_data_path=cindra_data_path,
        behavior_data_path=behavior_data_path,
        multiday_data_path=multiday_data_path,
        raw_data_path=raw_data_path,
    )
    reference_time = fluorescence_data["time_us"].to_numpy()

    # Assembles the behavior and runtime sub-datasets in parallel; both align to the fluorescence reference clock.
    tasks = {
        "behavior": partial(
            assemble_behavior_dataset,
            behavior_data_path=behavior_data_path,
            raw_data_path=raw_data_path,
            reference_time=reference_time,
            drop_time_columns=True,
        ),
        "runtime": partial(
            assemble_runtime_dataset,
            behavior_data_path=behavior_data_path,
            experiment_configuration=experiment_configuration,
            reference_time=reference_time,
        ),
    }
    results: dict[str, pl.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_name = {executor.submit(task): name for name, task in tasks.items()}
        for future in as_completed(future_to_name):
            results[future_to_name[future]] = future.result()

    # Concatenates the three sub-datasets into the unified feather, masks non-run experiment columns, and writes it
    # uncompressed so downstream consumers can memory-map it.
    result = pl.concat([fluorescence_data, results["behavior"], results["runtime"]], how="horizontal")
    result = _mask_non_run_experiment_data(experiment_data=result)
    result.write_ipc(file=output_path)
