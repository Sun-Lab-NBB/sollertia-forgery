"""Provides miscellaneous utility assets used across multiple library modules."""

from ataraxis_time import PrecisionTimer, TimerPrecisions

DELAY_TIMER: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared timer ``delay_terminal`` uses to delay the runtime's execution."""


def delay_terminal() -> None:
    """Delays the runtime execution for 100 milliseconds using the shared ``DELAY_TIMER``, so consecutive terminal
    printouts stay visually separated.
    """
    DELAY_TIMER.delay(delay=100, allow_sleep=True, block=False)


def multi_recording_dataset_name(animal_id: str, dataset_name: str) -> str:
    """Returns the cindra multi-recording dataset name one animal's recordings are tracked under within a forged
    dataset.

    Notes:
        The forging pipeline prepends the animal identifier to the forged dataset name so an animal's multi-recording
        outputs stay separate when a dataset spans several animals. That qualification is this library's, while the
        directory the name resolves to is cindra's, so a caller that needs the directory passes this name to cindra's
        own ``resolve_dataset_path`` rather than building the path here.

    Args:
        animal_id: The identifier of the animal whose recordings are tracked together.
        dataset_name: The unqualified forged dataset name.

    Returns:
        The ``{animal_id}_{dataset_name}`` name cindra records the animal's multi-recording output under.
    """
    return f"{animal_id}_{dataset_name}"
