"""Provides the admission gate that holds a session out of a forged dataset until its processing has completed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, SessionTypes
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..registries import resolve_forging_admission_pipelines
from ..shared_assets import resolve_session_tracker_path

if TYPE_CHECKING:
    from ..shared_assets import ProcessingPipelines

_COMPLETED_STATE: str = "completed"
"""The state label of a pipeline whose every tracked job succeeded, which is the only state that admits a session."""

_NOT_STARTED_STATE: str = "not_started"
"""The state label of a pipeline that holds no tracker or no tracked jobs for the session."""

_PARTIAL_STATE_TEMPLATE: str = "{unfinished} of {total} job(s) not succeeded"
"""The state label template of a pipeline that has tracked jobs which have not all succeeded."""


def verify_session_admissibility(session: SessionData) -> None:
    """Verifies that one session has completed every pipeline its acquisition system requires for forging.

    Notes:
        Every pipeline resolves its own job universe from the manifests written at acquisition, so a tracker reporting
        every job as succeeded already accounts for every camera and controller the session recorded. Admission
        therefore checks which pipelines completed rather than counting sources.

        A pipeline whose tracker holds no jobs is treated as not run, since a pipeline that resolved jobs records
        them before dispatching any.

    Args:
        session: The loaded source session being admitted.

    Raises:
        ValueError: If the session's acquisition system is unknown, if the session's type joins no dataset for that
            system, or if any required pipeline has not completed. The error names the pipelines that are outstanding
            and the state each is in.
    """
    requirements = resolve_forging_admission_pipelines(system=session.acquisition_system)

    required = requirements.get(SessionTypes(session.session_type))
    if required is None:
        admissible = ", ".join(sorted(str(session_type) for session_type in requirements))
        message = (
            f"Unable to admit session '{session.session_name}' into a forged dataset. Its session type "
            f"'{session.session_type}' joins no dataset for the '{session.acquisition_system}' acquisition system, "
            f"which admits the session type(s): {admissible}."
        )
        console.error(message=message, error=ValueError)

    outstanding: dict[str, str] = {}
    for pipeline in sorted(required):
        state = _resolve_pipeline_state(session=session, pipeline=pipeline)
        if state != _COMPLETED_STATE:
            outstanding[pipeline.value] = state

    if outstanding:
        message = (
            f"Unable to admit session '{session.session_name}' into a forged dataset. A session joins a dataset only "
            f"once every pipeline its acquisition system requires has completed, but the following are outstanding: "
            f"{outstanding}. Process the session through them, then define the dataset again."
        )
        console.error(message=message, error=ValueError)


def _resolve_pipeline_state(session: SessionData, pipeline: ProcessingPipelines) -> str:
    """Reads how far one pipeline has progressed for a session.

    Args:
        session: The loaded session whose tracker to read.
        pipeline: The pipeline whose progress to resolve.

    Returns:
        ``completed`` when every tracked job succeeded, ``not_started`` when the pipeline has no tracker or no tracked
        jobs, and otherwise a label naming how many of its jobs are not yet succeeded.
    """
    tracker_path = resolve_session_tracker_path(session=session, pipeline=pipeline)
    if not tracker_path.is_file():
        return _NOT_STARTED_STATE

    jobs = ProcessingTracker(file_path=tracker_path).snapshot()
    if not jobs:
        return _NOT_STARTED_STATE

    unfinished = sum(1 for state in jobs.values() if state.status is not ProcessingStatus.SUCCEEDED)
    if unfinished == 0:
        return _COMPLETED_STATE
    return _PARTIAL_STATE_TEMPLATE.format(unfinished=unfinished, total=len(jobs))
