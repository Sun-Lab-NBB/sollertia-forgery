"""Provides tools for querying project state and processing raw data into intermediate (processed) state.

Notes:
    Processed data can be integrated into an analysis dataset using tools from the 'forging' package.
"""

from .pipeline import run_behavior_processing_pipeline

__all__ = [
    "run_behavior_processing_pipeline",
]
