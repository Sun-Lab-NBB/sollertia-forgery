"""Provides the verification that the threading layer this library compiles its parallel kernels against is loadable."""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

from ataraxis_base_utilities import console

_OPENMP_PLATFORM: str = "darwin"
"""The platform whose threading layer resolves a separately installed runtime.

Every other platform runs TBB, which this library pins as a dependency, so its threading layer is installed with the
distribution and needs no verification.
"""

_OPENMP_LIBRARY_NAME: str = "libomp.dylib"
"""The file name of the OpenMP runtime the threading layer loads."""

_RUNTIME_DIRECTORIES: tuple[str, ...] = (
    "/opt/homebrew/opt/libomp/lib",
    "/usr/local/opt/libomp/lib",
    "/opt/local/lib/libomp",
)
"""The directories the macOS package managers install the OpenMP runtime into.

The first two are the Homebrew prefixes for Apple Silicon and Intel hosts, and the third is the MacPorts prefix. A
host that carries the runtime in one of these is reported the directory holding it, since making an installed runtime
loadable is a different remedy from installing one.
"""


def verify_parallel_runtime() -> None:
    """Refuses to run when the threading layer this library selected has no runtime to load.

    Notes:
        This library selects the OpenMP threading layer on macOS, where the TBB runtime it depends on for every other
        platform publishes no wheel. OpenMP is installed separately from the Python distribution, so a macOS host can
        carry every dependency and still be unable to run a parallel kernel.

        The verification runs while the package is imported, before any parallel kernel is compiled. The threading
        layer otherwise fails at the first parallelized call, deep inside whichever stage reached it first, and names
        no remedy. Failing here instead reports the remedy while the runtime has read no data and written no output.

        A host carrying the runtime in a package manager prefix is told to make it loadable rather than to install it,
        because the loader resolves the file name against its own search path and no package manager prefix is on that
        path by default.

    Raises:
        RuntimeError: If the platform runs the OpenMP threading layer and no OpenMP runtime is loadable.
    """
    if sys.platform != _OPENMP_PLATFORM:
        return

    try:
        ctypes.CDLL(_OPENMP_LIBRARY_NAME)
    except OSError:
        console.error(message=_resolve_refusal_message(), error=RuntimeError)


def _resolve_refusal_message() -> str:
    """Builds the message reporting that no OpenMP runtime is loadable, carrying the remedy this host calls for.

    Returns:
        The assembled message.
    """
    installed = [directory for directory in _RUNTIME_DIRECTORIES if Path(directory, _OPENMP_LIBRARY_NAME).is_file()]
    remedy = (
        f"Add '{installed[0]}' to the DYLD_LIBRARY_PATH environment variable, so the loader finds the runtime "
        f"already installed there."
        if installed
        else "Install the runtime with 'brew install libomp', or with 'mamba install llvm-openmp' into the "
        "environment this library runs from."
    )
    return (
        f"Unable to load the OpenMP runtime ('{_OPENMP_LIBRARY_NAME}') that this library's threading layer requires "
        f"on macOS. Every parallel processing stage fails without it. {remedy}"
    )
