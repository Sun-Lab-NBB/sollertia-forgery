"""Provides the DataLogger log-archive naming convention shared across the library's log-processing pipelines."""

from __future__ import annotations

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The filename suffix of the raw log archives written by the ataraxis DataLogger. Every archive is named
``{source_id}_log.npz`` after the DataLogger source id that produced it. Declared once here so the library's
system-agnostic log-processing workers build their discovery globs and archive paths from a single definition."""
