"""Provides miscellaneous utility assets used across multiple library modules."""

import os
from typing import TYPE_CHECKING
from contextlib import contextmanager

from ataraxis_time import PrecisionTimer, TimerPrecisions

if TYPE_CHECKING:
    from collections.abc import Iterator

DELAY_TIMER: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared timer used across the library to delay the runtime's execution."""

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The filename suffix of the raw log archives written by the ataraxis DataLogger. Every archive is named
``{source_id}_log.npz`` after the DataLogger source id that produced it."""

_WORKER_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
)
"""The environment variables that cap each threading layer a worker process can start a pool for. Most of these
libraries size their pool to the machine's core count on first import, so a worker that inherits the defaults reserves
far more of the machine than the one core it was budgeted. ``OPENCV_FFMPEG_THREADS`` is the exception: the decoder
reads it when a capture is constructed rather than at import, so a worker re-sets it for itself."""


def delay_terminal() -> None:
    """Uses the shared DELAY_TIMER instance to delay the runtime execution for 100 milliseconds to ensure proper
    visual separation of terminal printouts.
    """
    DELAY_TIMER.delay(delay=100, allow_sleep=True, block=False)


def multi_recording_dataset_directory(animal_id: str, dataset_name: str) -> str:
    """Returns the on-disk cindra multi-recording dataset directory name for one animal within a forged dataset.

    Notes:
        The forging pipeline prepends the animal identifier to the forged dataset name so an animal's multi-recording
        outputs stay separate when a dataset spans several animals. cindra lowercases the configured dataset name when
        it builds the output directory, so this helper lowercases the qualified name too, keeping the pipeline that
        writes the cindra configuration and the assembler that reads the output directory in agreement.

    Args:
        animal_id: The identifier of the animal whose recordings are tracked together.
        dataset_name: The unqualified forged dataset name.

    Returns:
        The lowercased ``{animal_id}_{dataset_name}`` directory name cindra writes the multi-recording output under.
    """
    return f"{animal_id}_{dataset_name}".lower()


@contextmanager
def pinned_worker_threads() -> Iterator[None]:
    """Caps every threading layer to a single thread for the duration of the block, then restores the environment.

    Notes:
        Wrap the creation of a worker pool in this. Each scientific library sizes its thread pool when it is first
        imported, and a spawned worker re-imports rather than inheriting the parent's modules, so that sizing happens
        before any code in the worker runs. The caps therefore have to be in place in the parent before the pool
        starts its children. A worker that inherits the defaults reserves a thread per core of the whole machine
        rather than the single core its job budgeted for it.

        Scoping the caps to a pool's lifetime, rather than setting them once in a long-lived process, is what keeps
        them correct. ``NUMBA_NUM_THREADS`` is the reason: numba reads it at import and compares the variable against
        that latched value on every compilation, refusing a disagreement once its threads have started. Setting it
        for the life of a process that already imported numba would fail every job that compiles a numba function,
        while setting it only around a spawn leaves the parent's own state untouched by the time it compiles
        anything.

    Yields:
        None. The caps are in effect for the duration of the block.
    """
    previous = {variable: os.environ.get(variable) for variable in _WORKER_THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(_WORKER_THREAD_VARIABLES, "1"))
    try:
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, default=None)
            else:
                os.environ[variable] = value
