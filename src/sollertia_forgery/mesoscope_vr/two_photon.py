"""Provides the Mesoscope-VR raw two-photon imaging directory locator donated to the system-agnostic two-photon
worker package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sollertia_shared_assets import MesoscopeDirectories

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData


def locate_two_photon_data(session: SessionData) -> Path:
    """Resolves the Mesoscope-VR session's raw two-photon imaging directory, which is the input to the cindra pipeline.

    Args:
        session: The loaded session whose raw two-photon imaging directory is resolved.

    Returns:
        The path to the session's ``mesoscope_data`` directory under its raw-data root. This directory stores the
        compressed 2-Photon Random Access Mesoscope (2P-RAM) acquisition output and accompanying metadata that the
        cindra single-recording pipeline consumes.
    """
    return session.raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA)
