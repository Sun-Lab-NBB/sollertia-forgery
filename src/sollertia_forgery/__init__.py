"""Provides tools for processing and managing the data acquired using the Sollertia data acquisition platform."""

import sys
import logging
import multiprocessing

from numba import config

# numexpr, pulled in transitively through pandas and PyTables, logs three INFO lines reporting its thread count the
# first time it is imported. Raising its logger above INFO here, before any import triggers it, keeps that report out
# of the pipeline's progress output.
logging.getLogger("numexpr").setLevel(logging.WARNING)

# Python 3.14 defaults the multiprocessing start method to 'forkserver' on Linux, whose server process creates each
# worker with os.fork(). In a process that carries library threads (numba, numexpr, BLAS) that fork emits a
# DeprecationWarning into the pipeline output. 'spawn' launches workers with posix_spawn rather than os.fork(), so the
# warning cannot arise. Every pool in this library uses the default context, so setting the default here applies the
# choice everywhere.
multiprocessing.set_start_method("spawn", force=True)

# Configures the numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled. macOS uses OpenMP because tbb4py publishes no Apple Silicon wheel. All other platforms use
# TBB for lower overhead on flat prange loops.
config.THREADING_LAYER = "omp" if sys.platform == "darwin" else "tbb"  # type: ignore[attr-defined]

from ataraxis_base_utilities import console  # noqa: E402 - imported after the process-wide configuration above.

# Ensures console and progress bars are enabled when this library is used.
if not console.enabled:
    console.enable()
if not console.progress_enabled:
    console.enable_progress()
