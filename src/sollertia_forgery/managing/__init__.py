"""Provides project data-management assets: raw-data checksum verification and project manifest generation and viewing.
"""

from .checksum import CHECKSUM_JOB_NAME, resolve_checksum
from .manifest import MANIFEST_JOB_NAME, ProjectManifest, generate_project_manifest
from .remote_management import (
    manage_project_data,
    discover_project_data,
    resolve_project_manifest,
    discover_project_sessions,
)

__all__ = [
    "CHECKSUM_JOB_NAME",
    "MANIFEST_JOB_NAME",
    "ProjectManifest",
    "discover_project_data",
    "discover_project_sessions",
    "generate_project_manifest",
    "manage_project_data",
    "resolve_checksum",
    "resolve_project_manifest",
]
