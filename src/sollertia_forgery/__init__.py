"""Provides tools for processing and managing the data acquired using the Sollertia data acquisition platform.

See the `API documentation <https://sollertia-forgery-api-docs.netlify.app/>`_ for the description of available
assets. See the `source code repository <https://github.com/Sun-Lab-NBB/sollertia-forgery>`_ for more details.

Authors: Ivan Kondratyev (Inkaros), Natalie Yeung, Kushaan Gupta
"""

import sys
import logging
import multiprocessing

from numba import config

# numexpr, required by PyTables and imported by pandas whenever it is installed, logs three INFO lines reporting its
# thread count the first time it is imported. Raising its logger above INFO here, before any import triggers it, keeps
# that report out of the pipeline's progress output.
logging.getLogger("numexpr").setLevel(logging.WARNING)

# Python 3.14 defaults the multiprocessing start method to 'forkserver' on Linux, whose server process creates each
# worker with os.fork(). In a process that carries library threads (numba, numexpr, BLAS) that fork emits a
# DeprecationWarning into the pipeline output. 'spawn' launches each worker through _posixsubprocess.fork_exec, which
# immediately execs a fresh interpreter, so a worker starts from a thread-free process and the warning cannot arise.
# Every pool in this library uses the default context, so setting the default here applies the choice everywhere.
multiprocessing.set_start_method("spawn", force=True)

# Configures the numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled. macOS uses OpenMP because tbb4py publishes no Apple Silicon wheel. All other platforms use
# TBB for lower overhead on flat prange loops.
#
# Whether the selected layer has a runtime to load is checked by 'verify_openmp_runtime', which every processing
# pipeline calls before it dispatches a job. Checking there rather than here keeps importing this library to read the
# dataset types it publishes free of the macOS OpenMP setup a host needs only to process data.
config.THREADING_LAYER = "omp" if sys.platform == "darwin" else "tbb"  # type: ignore[attr-defined]

from ataraxis_base_utilities import console  # noqa: E402 - imported after the process-wide configuration above.

# Ensures console and progress bars are enabled when this library is used.
if not console.enabled:
    console.enable()
if not console.progress_enabled:
    console.enable_progress()

# The distribution's public surface is the 'slf' command-line interface and the MCP server it launches, so the
# top-level package re-exports no library symbol.
__all__: list[str] = []
