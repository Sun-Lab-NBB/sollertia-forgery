"""Provides the Mesoscope-VR system-specific assets donated to the system-agnostic worker packages: the
behavior/dataset metadata schema (conventions), the microcontroller module parsers, the runtime log parser, and the
per-session forging data-assembly worker.

Notes:
    This package contributes system-specific assets into the central registries (``registries.py``) that the agnostic
    worker packages (``microcontrollers``, ``runtime``, ``forging``) resolve and invoke; it never imports those
    agnostic processors, so the dependency is strictly one-way. It owns no pipeline orchestration of its own. Heavy
    acquisition-library bindings are imported lazily inside the donated functions so importing this package stays
    cheap. The microcontroller parsers (``mesoscope_vr.microcontrollers``) and the runtime parser
    (``mesoscope_vr.runtime``) are imported directly from their submodules by the registries hub.
"""

from .forging import assemble_mesoscope_session
from .metadata import (
    DatasetColumn,
    BehaviorDataFiles,
    SessionDataFormat,
)
from .fluorescence import FluorescenceColumn

__all__ = [
    "BehaviorDataFiles",
    "DatasetColumn",
    "FluorescenceColumn",
    "SessionDataFormat",
    "assemble_mesoscope_session",
]
