"""Provides the one preparation path that resolves every batch, whichever host holds the data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from ataraxis_base_utilities import console

from .graph import build_batch_document
from .hosts import plan_artifact_path, state_artifact_paths
from .dispatch import DATASET_UNIT, SESSION_UNIT, resolve_dispatch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .graph import BatchDocument
    from .hosts import ExecutionHost

_UNIT_DEPTHS: dict[str, int] = {SESSION_UNIT: 2, DATASET_UNIT: 1}
"""How many directory levels separate a unit root from its project root, keyed by unit kind. A dataset sits directly
under its project root while a session sits under its animal."""


def prepare_batch(
    host: ExecutionHost,
    pipeline: str,
    unit_paths: Sequence[str],
    options: dict[str, Any] | None = None,
    *,
    replan: bool = False,
) -> BatchDocument:
    """Resolves a pipeline's dispatchable jobs for the named units out of the project's own artifacts.

    Notes:
        Materialization runs on the host that holds the data, since estimating a job's cost and reading a tracker both
        need the data itself. The resulting artifacts are then read onto this machine and the batch is built here, so
        the graph that a run dispatches is resolved in one place whichever host prepared it.

        Planning re-estimates nothing a unit's cache already holds unless a caller asks for it, because a submission's
        sizing figures must not change underneath it. Recorded status is always refreshed, since that is the part that
        changes between runs.

        A job that the unit cannot run never reaches its processing tracker, so its absence from the state artifact
        is what rules it out. A job whose upstream stage this run can neither dispatch nor find already succeeded is
        reported as blocked rather than dispatched.

    Args:
        host: The host holding the units, which materializes the artifacts and delivers them here.
        pipeline: The batch pipeline to prepare.
        unit_paths: The processing unit directories on that host whose jobs to prepare. Every unit must belong to one
            project, since the artifacts from which a batch is resolved are written per project.
        options: The pipeline-specific parameters given to the prepared jobs.
        replan: Determines whether to re-estimate the resource figures the units' caches already hold.

    Returns:
        The prepared batch document.

    Raises:
        ValueError: If the named pipeline is not a supported batch pipeline, if no unit is named, if the named units
            span more than one project, or if a named unit sits too few directory levels below its project for that
            project to be resolved from it.
        FileNotFoundError: If the host holds no plan table for the units' project.
        RuntimeError: If a step fails on the host.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        message = f"Unable to prepare a batch for pipeline '{pipeline}', which is not a supported batch pipeline."
        console.error(message=message, error=ValueError)

    units = [Path(unit_path) for unit_path in unit_paths]
    unit_kind = dispatch.unit_kind
    project_root = resolve_project_root(unit_paths=units, unit_kind=unit_kind)

    host.materialize(project_root=project_root, unit_paths=units, unit_kind=unit_kind, replan=replan)

    plan_rows = host.read_rows(path=plan_artifact_path(project_root=project_root))
    if not plan_rows:
        message = (
            f"Unable to prepare the '{dispatch.pipeline.value}' batch. The {host.label} host holds no plan table for "
            f"project '{project_root.name}', so the planning step wrote nothing for it."
        )
        console.error(message=message, error=FileNotFoundError)

    state_rows = [
        row
        for artifact in state_artifact_paths(project_root=project_root, unit_paths=units, unit_kind=unit_kind)
        for row in host.read_rows(path=artifact)
    ]

    return build_batch_document(
        pipeline=dispatch.pipeline.value,
        host=host.label,
        unit_column=unit_kind,
        plan_rows=plan_rows,
        state_rows=state_rows,
        unit_paths=units,
        options=dict(options or {}),
        tracker_paths=host.resolve_tracker_paths(pipeline=dispatch.pipeline.value, unit_paths=units),
    )


def resolve_project_root(unit_paths: Sequence[Path], unit_kind: str) -> Path:
    """Resolves the project that owns the named units.

    Args:
        unit_paths: The processing unit directories whose project is resolved.
        unit_kind: Whether the units are sessions or datasets, which sets how far above a unit its project sits.

    Returns:
        The path to the project root.

    Raises:
        ValueError: If no unit is named, if a named unit's path holds fewer parent directories than its kind sits
            below its project, or if the named units span more than one project.
    """
    if not unit_paths:
        message = "Unable to resolve the project of a batch. No processing unit was named."
        console.error(message=message, error=ValueError)

    depth = _UNIT_DEPTHS[unit_kind]

    # Indexing the parents of a path shallower than this depth raises an IndexError, and every caller documents a
    # ValueError for this failure, so a shallow path is refused here under the documented error.
    shallow = sorted(str(unit_path) for unit_path in unit_paths if len(unit_path.parents) < depth)
    if shallow:
        message = (
            f"Unable to resolve the project of a batch from the {unit_kind} path(s) {shallow}. A {unit_kind} sits "
            f"{depth} directory level(s) below its project, so each path must name at least that many parents."
        )
        console.error(message=message, error=ValueError)

    roots = {unit_path.parents[depth - 1] for unit_path in unit_paths}
    if len(roots) > 1:
        message = (
            f"Unable to resolve a batch spanning the projects {sorted(str(root) for root in roots)}. The plan and "
            f"state artifacts that resolve a batch are written per project, so every unit of one batch must "
            f"belong to the same project."
        )
        console.error(message=message, error=ValueError)
    return roots.pop()
