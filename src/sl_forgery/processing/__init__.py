"""Provides tools for querying project state and processing raw data into intermediate (processed) state.

Notes:
    Processed data can be integrated into an analysis dataset using tools from the 'forging' package.
"""

from .interface import process_project_data

__all__ = [
    "process_project_data",
]
