"""This package provides the assets for managing the session and project data acquired in the Sun lab."""

from .interface import resolve_project_manifest
from .processing import (
    resolve_checksum,
    transfer_session,
    generate_project_manifest,
    _construct_adoption_pipeline,
    _construct_checksum_resolution_pipeline,
)
from ..shared_assets import ProjectManifest

__all__ = [
    "ProjectManifest",
    "_construct_adoption_pipeline",
    "_construct_checksum_resolution_pipeline",
    "generate_project_manifest",
    "resolve_checksum",
    "resolve_project_manifest",
    "transfer_session",
]
