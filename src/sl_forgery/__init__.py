"""Provides assets for forging analysis datasets from processed experimental data."""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports.
from numba import config  # type: ignore[import-untyped]

config.THREADING_LAYER = "tbb"

from ataraxis_base_utilities import console  # noqa: E402 - imported after numba config per ordering requirement.

# Ensures console is enabled when this library is used
if not console.enabled:
    console.enable()
