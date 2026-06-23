"""Provides system-agnostic management pipelines: raw-data checksum verification and project manifest generation."""

from .checksum import CHECKSUM_JOB_NAME, resolve_checksum
from .manifest import MANIFEST_JOB_NAME, ProjectManifest, generate_project_manifest

__all__ = [
    "CHECKSUM_JOB_NAME",
    "MANIFEST_JOB_NAME",
    "ProjectManifest",
    "generate_project_manifest",
    "resolve_checksum",
]
