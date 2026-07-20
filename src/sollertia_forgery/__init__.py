"""Provides tools for processing and managing the data acquired using the Sollertia data acquisition platform."""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports. macOS uses OpenMP because tbb4py publishes no
# Apple Silicon wheel. All other platforms use TBB for lower overhead on flat prange loops.
import sys

from numba import config

config.THREADING_LAYER = "omp" if sys.platform == "darwin" else "tbb"  # type: ignore[attr-defined]

from ataraxis_base_utilities import console  # noqa: E402 - imported after numba config per ordering requirement.

# Ensures console and progress bars are enabled when this library is used.
if not console.enabled:
    console.enable()
if not console.progress_enabled:
    console.enable_progress()
