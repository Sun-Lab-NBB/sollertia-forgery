"""Provides the DataLogger log-archive naming convention shared across the library's log-processing pipelines."""

from __future__ import annotations

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The filename suffix of the raw log archives written by the ataraxis DataLogger. Every archive is named
``{source_id}_log.npz`` after the DataLogger source id that produced it, a convention shared across the Sollertia
stack (ataraxis-video-system and ataraxis-communication-interface both expose the same value). The library's
log-processing workers (video, runtime, and microcontrollers) build their discovery globs and archive paths from
this single definition, so the convention is declared exactly once for the system-agnostic layer."""
