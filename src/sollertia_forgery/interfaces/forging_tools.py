"""Provides the Model Context Protocol (MCP) tools for defining the dataset hierarchy a forging batch runs against."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from ..forging import forging_tracker_path, define_forging_dataset
from .mcp_instance import mcp


@mcp.tool()
def define_forging_dataset_tool(
    project_path: str,
    dataset_name: str,
    session_names: list[str],
    recreate_animals: list[str] | None = None,
    *,
    force_recreate: bool = False,
) -> dict[str, Any]:
    """Creates or extends a forged dataset hierarchy and materializes the per-animal configurations it resolves.

    Every forging job runs against a hierarchy this tool established, so a dataset is defined here before its jobs
    are prepared, and preparing a dataset this tool has not built reports an error. A per-animal configuration is
    written only for the animals whose acquisition system resolves a multi-recording configuration, which is the
    case for sessions carrying two-photon imaging data.

    Provided sessions the dataset does not hold are appended, so a dataset grows by naming the sessions to add. An
    animal already in the dataset is frozen, because widening its session set invalidates the outputs already forged
    for the sessions it keeps. Name that animal in ``recreate_animals`` to rebuild it from the provided sessions
    while every other animal keeps its data, which also returns that animal's tracked jobs to the scheduled state.

    Args:
        project_path: The path to the project's root directory holding the animal and session data directories. The
            dataset hierarchy is created under this directory.
        dataset_name: The unique name of the dataset to create or extend.
        session_names: The session names the dataset must contain.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions provided
            for them. Omit to leave every existing animal frozen.
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list. Mutually exclusive with ``recreate_animals``.

    Returns:
        A response dict with the ``dataset_name``, the ``dataset_path`` the hierarchy was built at, the
        ``tracker_path`` its jobs record on, the ``session_count`` and ``animal_count`` the dataset now holds, and
        the ``animals`` it covers. Returns an error when the resolution policy rejects the request.
    """
    try:
        dataset = define_forging_dataset(
            name=dataset_name,
            session_names=tuple(session_names),
            project_root=Path(project_path),
            force_recreate=force_recreate,
            recreate_animals=tuple(recreate_animals or ()),
        )
    except Exception as exception:
        return {"success": False, "error": str(exception)}

    dataset_path = dataset.dataset_data_path.parent
    return {
        "success": True,
        "dataset_name": dataset.name,
        "dataset_path": str(dataset_path),
        "tracker_path": str(forging_tracker_path(dataset=dataset)),
        "session_count": len(dataset.sessions),
        "animal_count": len(dataset.animals),
        "animals": [animal.animal for animal in dataset.animals],
    }
