"""Provides the Mesoscope-VR data-assembly dispatcher donated to the system-agnostic forging pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, SessionTypes

from ..shared_assets import ProcessingPipelines
from .training_dataset import assemble_training_dataset
from .experiment_dataset import assemble_experiment_dataset

if TYPE_CHECKING:
    from pathlib import Path


MESOSCOPE_ADMISSION_PIPELINES: dict[SessionTypes, frozenset[ProcessingPipelines]] = {
    SessionTypes.MESOSCOPE_EXPERIMENT: frozenset(
        {
            ProcessingPipelines.CHECKSUM,
            ProcessingPipelines.RUNTIME,
            ProcessingPipelines.MICROCONTROLLER,
            ProcessingPipelines.VIDEO,
            ProcessingPipelines.TWO_PHOTON,
        }
    ),
    SessionTypes.RUN_TRAINING: frozenset(
        {
            ProcessingPipelines.CHECKSUM,
            ProcessingPipelines.RUNTIME,
            ProcessingPipelines.MICROCONTROLLER,
            ProcessingPipelines.VIDEO,
        }
    ),
    SessionTypes.LICK_TRAINING: frozenset(
        {
            ProcessingPipelines.CHECKSUM,
            ProcessingPipelines.RUNTIME,
            ProcessingPipelines.MICROCONTROLLER,
            ProcessingPipelines.VIDEO,
        }
    ),
}
"""Maps each Mesoscope-VR session type to the pipelines that must have completed before the session may join a forged
dataset.

Notes:
    Every pipeline resolves its own job universe from the acquisition manifests, so a completed tracker already means
    every source recorded by the session was processed. Admission therefore checks which pipelines completed rather than
    counting sources.

    A training session records no imaging, so the two-photon pipeline is absent from its requirement. A session type
    absent from this mapping joins no dataset, which is the case for window checking.
"""


_TRAINING_SESSION_TYPES: frozenset[SessionTypes] = frozenset({SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING})
"""The session types routed to the training-session assembler."""


def assemble_mesoscope_session(source_session_path: Path, output_path: Path, dataset_name: str) -> None:
    """Assembles a single Mesoscope-VR session's unified data feather, routing by session type.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the ``data.feather`` file to write inside the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, forwarded to the experiment assembler to resolve the cindra
            multi-recording output directory.

    Raises:
        FileNotFoundError: If a required processed-data directory or reference clock is missing (propagated from the
            resolved assembler).
        ValueError: If the session type is not a supported forging session type, or if a sub-dataset cannot be
            assembled (propagated from the resolved assembler).
    """
    session = SessionData.load(session_path=source_session_path)
    session_type = session.session_type

    if session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
        assemble_experiment_dataset(
            source_session_path=source_session_path, output_path=output_path, dataset_name=dataset_name
        )
        return
    if session_type in _TRAINING_SESSION_TYPES:
        assemble_training_dataset(source_session_path=source_session_path, output_path=output_path)
        return

    supported = ", ".join(
        sorted(member.value for member in (SessionTypes.MESOSCOPE_EXPERIMENT, *_TRAINING_SESSION_TYPES))
    )
    message = (
        f"Unable to assemble the data for session '{source_session_path.name}'. Its session type '{session_type}' is "
        f"not a supported forging session type. The supported session types are: {supported}."
    )
    console.error(message=message, error=ValueError)
