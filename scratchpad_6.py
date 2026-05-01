"""Scratch companion to ``notebook_1.ipynb`` — animal 11 reward-shift / anchor session.

Catches keepers as they emerge during the interactive exploration: distilled figure builders,
cached intermediate computations, derived datasets, and any ad-hoc analyses worth re-running
offline. Mirrors the structure of ``scratchpad_4.py`` / ``scratchpad_5.py`` so the file can be
executed end-to-end with ``python scratchpad_6.py`` once enough has accumulated to warrant it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sollertia_forgery.shared_assets import DatasetData, TrialGeometry
from sollertia_forgery.analysis import (
    DriftCellColumn,
    DriftDetectionConfiguration,
    TuningColumn,
)
from sollertia_forgery.analysis.drift import compute_drift_report


# ---- Configuration ----------------------------------------------------------------------------------------

DATASET_PATH = Path("/mnt/data/MaalstroomicFlow/void")
OUTPUT_DIR = Path("/mnt/data/MaalstroomicFlow_animal11_session")
ANIMAL: str = "11"

# MaalstroomicFlow void reward-shift constants (cyclic_4_cue trial type, 240-cm track).
TRACK_LENGTH_CM: float = 240.0
OLD_REWARD_LO, OLD_REWARD_HI = 218.0, 232.0
NEW_REWARD_LO, NEW_REWARD_HI = 98.0, 109.0
SHIFT_AT_SESSION: int = 15
PRE_SHIFT_SESSIONS: tuple[int, ...] = tuple(range(0, SHIFT_AT_SESSION))
ANCHOR_WINDOW_SESSIONS: int = 15

DRIFT_CONFIG = DriftDetectionConfiguration(peak_shift_shuffle_count=200)


# ---- Helpers populated as the notebook session produces keepers ------------------------------------------


def main() -> None:
    """Entry point — populate as the notebook session produces keepers worth saving."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = DatasetData.load(dataset_path=DATASET_PATH)
    animal = dataset.get_animal(animal=ANIMAL)
    sessions = tuple(
        sorted(dataset.get_sessions_for_animal(animal=ANIMAL), key=lambda s: s.session)
    )
    print(f"Loaded animal {ANIMAL}: {len(sessions)} sessions, dataset {dataset.name!r}")
    # TODO: paste keepers from the notebook here as they crystallize.


if __name__ == "__main__":
    main()
