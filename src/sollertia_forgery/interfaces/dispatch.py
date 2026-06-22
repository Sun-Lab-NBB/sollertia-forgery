"""Provides the acquisition-system inference helpers the generic interface uses to dispatch system-specific
processing through the registries.

The interface never names a system package. Instead, it infers the acquisition system from the data it is pointed at
(a session or a project of sessions) and resolves the registered entry point for that system from ``registries``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, AcquisitionSystems, discover_sessions

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["infer_system_from_project", "infer_system_from_session"]


def infer_system_from_session(session_path: Path) -> AcquisitionSystems:
    """Infers the acquisition system that recorded the target session.

    Args:
        session_path: The path to the session root directory containing the session data hierarchy.

    Returns:
        The AcquisitionSystems member parsed from the session's metadata.
    """
    session = SessionData.load(session_path=session_path)
    return AcquisitionSystems(session.acquisition_system)


def infer_system_from_project(project_root: Path) -> AcquisitionSystems:
    """Infers the acquisition system of a project from the first session discovered under its root.

    All sessions in a project share a single acquisition system, so the system parsed from the first discovered
    session keys the registry dispatch for every session in the project.

    Args:
        project_root: The path to the project's root data directory storing the animal and session directories.

    Returns:
        The AcquisitionSystems member parsed from the first discovered session's metadata.

    Raises:
        ValueError: If no sessions are discovered under the project root.
    """
    session_roots = discover_sessions(root_path=project_root)
    if not session_roots:
        message = (
            f"Unable to infer the acquisition system for the project at '{project_root}'. No sessions were "
            f"discovered under the project root."
        )
        console.error(message=message, error=ValueError)
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover
    return infer_system_from_session(session_path=session_roots[0])
