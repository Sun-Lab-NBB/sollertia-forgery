"""This module provides tools to query the current state of any Sun lab project and process raw project data into an
intermediate (processed) state. The processed data can then be integrated into an analysis dataset using tools from
the 'forging' package.
"""

from .interface import process_project_data

__all__ = [
    "process_project_data",
]
