"""Provides the remote (SLURM) orchestration layer: the compute-server pipeline engine and the system-agnostic
management orchestrators that dispatch project manifest and checksum pipelines to the server.
"""

from .managing import manage_project_data, resolve_project_manifest
from .pipeline import ProcessingPipeline, execute_pipelines, check_session_eligibility

__all__ = [
    "ProcessingPipeline",
    "check_session_eligibility",
    "execute_pipelines",
    "manage_project_data",
    "resolve_project_manifest",
]
