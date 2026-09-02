"""Provides the assembly-source resolver donated to the system-agnostic forging pipeline, which reports the height at
which one Mesoscope-VR session's assembly holds each source it reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionTypes

from .metadata import BehaviorDataFiles
from .video_dataset import count_camera_source_samples
from ..shared_assets import count_feather_rows

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

_TRAINING_MICROCONTROLLER_SOURCES: tuple[BehaviorDataFiles, ...] = (
    BehaviorDataFiles.VALVE,
    BehaviorDataFiles.LICK,
    BehaviorDataFiles.ENCODER,
    BehaviorDataFiles.SCREEN,
    BehaviorDataFiles.BRAKE,
    BehaviorDataFiles.TORQUE,
)
"""The module-parsed feathers a training assembly reads out of the session's processed microcontroller-data
directory. The valve and lick feathers are required and the rest are read whenever the session type produced them,
which is why the resolver counts the ones that are present rather than requiring the whole set."""

_TRAINING_RUNTIME_SOURCES: tuple[BehaviorDataFiles, ...] = (
    BehaviorDataFiles.SYSTEM_STATE,
    BehaviorDataFiles.RUNTIME_STATE,
)
"""The runtime-parsed feathers a training assembly reads out of the session's processed runtime-data directory. The
system-state feather supplies the behavior sub-dataset's state column and both feathers are read again to clip the
assembled frame to the session bounds."""

_EXPERIMENT_MICROCONTROLLER_SOURCES: tuple[BehaviorDataFiles, ...] = (
    *_TRAINING_MICROCONTROLLER_SOURCES,
    BehaviorDataFiles.MESOSCOPE_FRAME,
)
"""The module-parsed feathers an experiment assembly reads out of the same directory. It reads every feather the
training assembly reads, because it builds the same behavior sub-dataset, and the mesoscope-frame feather besides.
That feather carries one row per logged TTL edge on the microcontroller's own clock, and the fluorescence assembly
sorts it, derives two transition columns beside it, and splits it into a rising-edge and a falling-edge table that
are then joined, so it stands at its own height several times over."""

_EXPERIMENT_RUNTIME_SOURCES: tuple[BehaviorDataFiles, ...] = (
    *_TRAINING_RUNTIME_SOURCES,
    BehaviorDataFiles.VR_TRIGGER_ZONE,
    BehaviorDataFiles.VR_CUE,
    BehaviorDataFiles.TRIAL,
    BehaviorDataFiles.REINFORCING_GUIDANCE,
    BehaviorDataFiles.AVERSIVE_GUIDANCE,
)
"""The runtime-parsed feathers an experiment assembly reads out of the session's processed runtime-data directory. An
experiment session carries a runtime sub-dataset that a training session does not, and that sub-dataset reads the VR
trigger-zone, wall-cue and trial feathers unconditionally and the two guidance feathers whenever the session recorded
guidance events. None of them is read by the training assembler, so counting a session of either type against the
training set alone would leave an experiment assembly charged for a fraction of the sources it holds."""

_SOURCE_FILES: dict[SessionTypes, tuple[tuple[BehaviorDataFiles, ...], tuple[BehaviorDataFiles, ...]]] = {
    SessionTypes.MESOSCOPE_EXPERIMENT: (_EXPERIMENT_MICROCONTROLLER_SOURCES, _EXPERIMENT_RUNTIME_SOURCES),
    SessionTypes.RUN_TRAINING: (_TRAINING_MICROCONTROLLER_SOURCES, _TRAINING_RUNTIME_SOURCES),
    SessionTypes.LICK_TRAINING: (_TRAINING_MICROCONTROLLER_SOURCES, _TRAINING_RUNTIME_SOURCES),
}
"""Maps each session type the Mesoscope-VR assemblers cover to the microcontroller-parsed and runtime-parsed feathers
the assembler routed that type reads. A type this mapping omits is one ``assemble_mesoscope_session`` refuses, so no
assembly of it exists to report the sources of."""


def resolve_mesoscope_assembly_sources(session: SessionData) -> tuple[int, ...]:
    """Reports the samples each source the assembly of one Mesoscope-VR session reads holds on that source's own clock.

    Notes:
        A source arrives at the clock it was sampled on rather than at the clock the assembled frame is placed on. A
        camera's timestamp, motion-energy and pupil feathers are read at that camera's frame count, and a behavior
        feather is read at the count of samples its own parser logged, both before anything is interpolated onto the
        reference clock. A camera faster than the reference one therefore contributes arrays taller than the frame the
        job builds, and neither family bounds the other.

        The source set is routed by session type, because the two assemblers read different sets. An experiment
        assembly builds a runtime sub-dataset and a fluorescence sub-dataset that a training assembly does not, and
        those read the VR trigger-zone, wall-cue, trial and guidance feathers and the mesoscope-frame feather on top
        of everything the training assembly reads. Reporting one set for both types would leave an experiment
        assembly charged for a fraction of the arrays it holds, which is an under-estimate rather than a coarse one.

        A source is reported once per clock rather than once per file. The three feathers of one camera carry one row
        per acquired frame, so that camera's frame count describes every array it contributes, while each behavior
        feather carries its own clock and is reported on its own. A feather the session did not produce is left out,
        because the assembler reads whichever of the optional sources its session type wrote and no others.

        Reads the IPC metadata of each source feather alone. No column is materialized, so measuring a session costs
        the same whatever its length.

    Args:
        session: The loaded session whose assembly sources are measured.

    Returns:
        The samples each source the assembly reads holds, one entry per source, cameras first. Empty when the session
        carries none of the sources its assembler reads.

    Raises:
        ValueError: If the session's type is not one the Mesoscope-VR assemblers cover, in which case no assembly of
            it exists whose sources could be reported.
    """
    session_type = SessionTypes(session.session_type)

    # Routed on the same mapping the dispatcher routes the assembler itself on, so a type the dispatcher refuses is
    # refused here rather than answered from whichever set happens to be the default.
    source_files = _SOURCE_FILES.get(session_type)
    if source_files is None:
        supported = ", ".join(sorted(str(covered) for covered in _SOURCE_FILES))
        message = (
            f"Unable to resolve the assembly sources of session '{session.session_name}'. Its session type "
            f"'{session_type}' is not a supported forging session type, so no assembler reads sources for it. The "
            f"supported session types are: {supported}."
        )
        console.error(message=message, error=ValueError)

    microcontroller_files, runtime_files = source_files
    return (
        *count_camera_source_samples(video_data_path=session.processed_data.video_data_path),
        *_source_samples(directory=session.processed_data.microcontroller_data_path, files=microcontroller_files),
        *_source_samples(directory=session.processed_data.runtime_data_path, files=runtime_files),
    )


def _source_samples(directory: Path, files: tuple[BehaviorDataFiles, ...]) -> tuple[int, ...]:
    """Counts the rows of each named behavior feather the session produced.

    Args:
        directory: The processed-data directory holding the feathers.
        files: The canonical filenames of the feathers the assembly reads out of that directory.

    Returns:
        The rows each present feather holds, one entry per feather, in the order the names were given. Empty when
        the directory holds none of them.
    """
    if not directory.is_dir():
        return ()

    return tuple(
        count_feather_rows(feather_path=feather_path)
        for source_file in files
        if (feather_path := directory.joinpath(source_file)).is_file()
    )
