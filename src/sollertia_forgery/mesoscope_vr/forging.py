"""Provides the Mesoscope-VR data-assembly worker donated to the system-agnostic forging pipeline.

Notes:
    This module's sole public entry point, ``assemble_mesoscope_session``, is the system-specific "data assembly"
    asset the Mesoscope-VR package contributes to the central ``FORGING_ASSEMBLY_REGISTRY``. The agnostic forging
    pipeline (``sollertia_forgery.forging``) resolves this worker by the session's acquisition system and invokes it
    once per session inside its worker pool to produce that session's ``data.feather`` together with the
    system-specific ``data_format.yaml`` schema descriptor. The pipeline owns dataset definition, the optional cindra
    multi-day stage, all job/tracker orchestration, and the re-export of shared assets (the VR configuration and the
    session descriptor); this worker owns only the assembly of the Mesoscope-VR data. It therefore imports nothing
    from the agnostic ``forging`` package: the dependency is strictly one-way, from the pipeline to the donated
    worker, so a system package never reaches back into an agnostic processor.
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

from .metadata import DATA_FORMAT_FILE, SessionDataFormat
from .fluorescence import assemble_cindra_dataset
from .runtime_dataset import assemble_runtime_dataset, _mask_non_run_experiment_data
from .behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path


def assemble_mesoscope_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Assembles a single Mesoscope-VR session's unified data feather and writes its data-format descriptor.

    Extracts, post-processes, and combines the session's fluorescence, behavior, and runtime sub-datasets into a
    single, time-aligned Polars DataFrame and saves it to an uncompressed ``data.feather`` at ``output_path``. The
    fluorescence sub-dataset is assembled first because it produces the reference time vector that the behavior and
    runtime sub-datasets align to; behavior and runtime are then assembled in parallel and concatenated horizontally
    with the fluorescence data. Cue, trial, and trial-type columns are masked for non-run experiment states. After
    writing the feather, the system-specific ``data_format.yaml`` schema descriptor is written alongside it.

    Notes:
        This is the atomic unit of work the agnostic forging pipeline dispatches to its worker pool, so it is a
        module-level function accepting only picklable arguments. It performs pure computation and writes its
        outputs; the calling pipeline owns the processing-tracker state transitions and the re-export of shared
        assets (the VR configuration and the session descriptor). The worker validates and assembles only the data
        the Mesoscope-VR system produces, so it implicitly requires a fully processed mesoscope experiment session
        (the experiment configuration and the single- and multi-recording cindra outputs must be present on disk).

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy. The
            data-format descriptor is written into the same directory.
        dataset_name: The unqualified dataset name, combined with the animal identifier to resolve the cindra
            multi-recording output directory.

    Raises:
        FileNotFoundError: If the session's processed behavior data directory or single-recording cindra output
            directory is missing.
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

    # Writes the system-specific data-format descriptor describing the assembled feather's columns.
    _write_data_format(result=result, output_directory=output_path.parent)


def _write_data_format(result: pl.DataFrame, output_directory: Path) -> None:
    """Writes the per-session ``data_format.yaml`` schema descriptor for the assembled feather.

    Args:
        result: The assembled session DataFrame whose schema is recorded in the descriptor.
        output_directory: The directory the descriptor is written into (the directory holding ``data.feather``).
    """
    data_format = SessionDataFormat(columns={name: str(dtype) for name, dtype in result.schema.items()})
    data_format.to_yaml(file_path=output_directory.joinpath(DATA_FORMAT_FILE))
