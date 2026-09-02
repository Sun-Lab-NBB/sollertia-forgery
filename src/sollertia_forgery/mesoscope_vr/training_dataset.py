"""Provides the Mesoscope-VR training-session data-assembly worker donated to the system-agnostic forging pipeline,
and the assembly-geometry resolver that reports the shape one such assembly takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import reduce

import polars as pl
from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import SessionData, ProcessingTrackers

from .video_dataset import (
    assemble_video_dataset,
    resolve_slowest_camera_clock,
    resolve_reference_clock_samples,
)
from ..shared_assets import AssemblyGeometry
from .runtime_dataset import clip_to_session_bounds
from .assembly_sources import resolve_mesoscope_assembly_sources
from .behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path


def assemble_training_dataset(source_session_path: Path, output_path: Path) -> None:
    """Assembles a single Mesoscope-VR training session's unified data feather.

    Resolves the reference clock from the slowest camera, then combines the session's behavior and video sub-datasets
    onto that clock into a single Polars DataFrame, written as an uncompressed ``data.feather`` at ``output_path``. The
    behavior sub-dataset supplies the ``time_us`` and ``elapsed_minutes`` columns, and the video sub-dataset contributes
    columns only when the session carries processed camera feathers.

    Notes:
        The assembled feather is clipped to the session bounds, so it begins when the system first leaves the idle
        state and ends at the final runtime-state entry. That drops the setup span the cameras record before the
        session and the teardown span they record after it.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy.

    Raises:
        FileNotFoundError: If the session's processed microcontroller-data or runtime-data directory is missing, if the
            session's hardware state file is absent, or if no camera clock is available to serve as the reference
            clock.
        ValueError: If a sub-dataset cannot be assembled (for example, a required hardware-state field is missing).
    """
    session = SessionData.load(session_path=source_session_path)

    microcontroller_data_path = session.processed_data.microcontroller_data_path
    runtime_data_path = session.processed_data.runtime_data_path
    video_data_path = session.processed_data.video_data_path
    raw_data_path = session.raw_data_path

    # Validates that the processed microcontroller and runtime outputs exist before any expensive work. A training
    # session has no cindra output, so only the behavior sources are required. The parsed behavior feathers are split
    # across the per-worker ``microcontroller_data`` (module parsing) and ``runtime_data`` (runtime decode) directories.
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

    # Resolves the reference clock from the slowest camera, since a training session has no fluorescence clock. Every
    # other data source is interpolated onto this clock. Resolving before creating the output directory avoids leaving
    # an empty directory behind when no camera clock is available.
    reference_time = resolve_slowest_camera_clock(video_data_path=video_data_path)

    ensure_directory_exists(path=output_path, is_file=True)

    # Assembles the behavior sub-dataset with its own time columns, since it supplies the unified feather's time axis.
    # The video sub-dataset aligns to the same reference clock and is empty when the session carries no camera feathers.
    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=reference_time,
        drop_time_columns=False,
    )
    video_data = assemble_video_dataset(video_data_path=video_data_path, reference_time=reference_time)

    # Stacks the behavior and video sub-datasets into the unified feather and writes it uncompressed so downstream
    # consumers can memory-map it. Stacking requires both sub-datasets to carry the reference clock's height, so one
    # that drifts off that clock raises rather than being padded. The video sub-dataset joins only when it produced
    # columns.
    sub_datasets = [behavior_data]
    if video_data.width:
        sub_datasets.append(video_data)
    result = reduce(pl.DataFrame.hstack, sub_datasets)
    result = clip_to_session_bounds(assembled_data=result, runtime_data_path=runtime_data_path)
    result.write_ipc(file=output_path)


def resolve_mesoscope_assembly_geometry(session: SessionData) -> AssemblyGeometry:
    """Reports the shape the assembly of a Mesoscope-VR session that records no imaging takes.

    States the two heights ``assemble_training_dataset`` works at: the reference clock its columns are placed on, and
    the clock each source it reads was sampled on. The two are different heights and neither bounds the other, since
    the reference clock is the slowest camera's while the sources include every faster camera the session recorded.

    Notes:
        The reference clock is resolved through the same selection the assembler settles on, so the height reported
        here is the height of the frame that assembler builds rather than that of any other clock the session holds.

        The sources are reported through the system's own source resolver rather than enumerated again here, so the
        two heights this record carries stay drawn from one statement of what the assemblers read. That resolver
        routes the source set by session type, and this record is built for the session types whose assembly settles
        its reference clock on a camera.

        Reads the IPC metadata of each source feather and, for the camera clocks alone, two timestamps of each. No
        column is materialized, so measuring a session costs the same whatever its length.

    Args:
        session: The loaded session whose assembly geometry is measured.

    Returns:
        The samples the reference clock holds and the samples each source the assembly reads holds.

    Raises:
        FileNotFoundError: If no camera timestamp feather with at least two frames spanning a positive duration is
            present, in which case the assembler settles on no reference clock either and the job cannot run.
        ValueError: If the session's type is not one the Mesoscope-VR assemblers cover.
    """
    video_data_path = session.processed_data.video_data_path
    reference_samples = resolve_reference_clock_samples(video_data_path=video_data_path)
    if reference_samples is None:
        message = (
            f"Unable to resolve the assembly geometry of session '{session.session_name}'. No camera timestamp "
            f"feather with at least two frames spanning a positive duration was found under '{video_data_path}', so "
            f"no camera clock can serve as the assembly reference clock and the job assembling the session could not "
            f"run either."
        )
        console.error(message=message, error=FileNotFoundError)

    return AssemblyGeometry(
        reference_samples=reference_samples,
        source_samples=resolve_mesoscope_assembly_sources(session=session),
    )
