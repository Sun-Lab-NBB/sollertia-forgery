"""Provides the standalone, system-agnostic microcontroller log processing pipeline: a two-stage worker pipeline
that extracts raw per-module data from controller log archives and parses it into domain-specific feathers using
per-system parsers looked up from a unified registry.
"""

from .parsers import (
    ModuleParser,
    MicrocontrollerParserProvider,
    resolve_parsers,
    register_parsers,
)
from .pipeline import PARSE_JOB_NAME, run_microcontroller_processing_pipeline
from .extraction import extract_controller, resolve_controllers

__all__ = [
    "PARSE_JOB_NAME",
    "MicrocontrollerParserProvider",
    "ModuleParser",
    "extract_controller",
    "register_parsers",
    "resolve_controllers",
    "resolve_parsers",
    "run_microcontroller_processing_pipeline",
]
