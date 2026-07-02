"""Provides the local filesystem location of the project mirror and the path-rebasing primitive used to translate
mirror-relative asset paths onto a real data root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sollertia_shared_assets import get_working_directory

if TYPE_CHECKING:
    from pathlib import Path

MIRROR_DIRECTORY_NAME: str = "mirror"
"""The name of the mirror root directory created under the Sollertia platform working directory. The mirror lives
alongside the platform's configuration and credentials directories and is deliberately named distinctly from the
data root to avoid conflating the shallow state copy with the real data hierarchy."""


def get_mirror_directory() -> Path:
    """Resolves and creates the local mirror root under the Sollertia platform working directory.

    The mirror root is the parent directory under which per-project shallow copies are materialized. It persists
    across runs so that job monitoring can read the last synchronized state without re-contacting the data source.

    Returns:
        The path to the mirror root directory, created if it did not already exist.

    Raises:
        FileNotFoundError: If the Sollertia platform working directory has not been configured for the host machine.
    """
    mirror_directory = get_working_directory().joinpath(MIRROR_DIRECTORY_NAME)
    mirror_directory.mkdir(parents=True, exist_ok=True)
    return mirror_directory


def rebase_path(path: Path, source_root: Path, destination_root: Path) -> Path:
    """Re-anchors a path built under one hierarchy root onto a different hierarchy root.

    Both roots must describe the same project hierarchy so that the input path's location relative to the source root
    is preserved under the destination root. This is the mechanism that lets a caller build an asset path under the
    mirror and then translate it onto the local or remote data root for transfer or execution.

    Args:
        path: The path located under the source root to re-anchor.
        source_root: The hierarchy root the input path is currently resolved against.
        destination_root: The hierarchy root to resolve the returned path against.

    Returns:
        The input path resolved under the destination root.

    Raises:
        ValueError: If the input path is not located under the source root.
    """
    return destination_root.joinpath(path.relative_to(source_root))
