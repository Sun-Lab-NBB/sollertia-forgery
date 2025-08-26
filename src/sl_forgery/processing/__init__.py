"""This module provides tools to query the current state of any Sun lab project and process raw project data into an
intermediate (processed) state. The processed data can then be integrated into an analysis dataset using tools from
the 'dataset' package from this library."""

from .data_processing import process_project_data
from .project_management import fetch_remote_project_manifest, generate_remote_project_manifest

__all__ = [
    "fetch_remote_project_manifest",
    "generate_remote_project_manifest",
    "process_project_data",
]
