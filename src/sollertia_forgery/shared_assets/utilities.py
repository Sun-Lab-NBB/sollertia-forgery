"""Provides miscellaneous utility assets used across multiple library modules."""

from ataraxis_time import PrecisionTimer, TimerPrecisions

DELAY_TIMER: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared timer used across the library to delay the runtime's execution."""

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The filename suffix of the raw log archives written by the ataraxis DataLogger. Every archive is named
``{source_id}_log.npz`` after the DataLogger source id that produced it. Declared once here so the library's
system-agnostic log-processing workers build their discovery globs and archive paths from a single definition."""


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
