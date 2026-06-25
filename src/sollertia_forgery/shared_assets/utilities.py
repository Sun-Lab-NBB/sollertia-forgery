"""Provides miscellaneous utility assets used across multiple library modules."""

from ataraxis_time import PrecisionTimer, TimerPrecisions

DELAY_TIMER: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared PrecisionTimer instance used across the library to delay the runtime's execution."""


def delay_terminal() -> None:
    """Uses the shared DELAY_TIMER instance to delay the runtime execution for 100 milliseconds to ensure proper
    visual separation of terminal printouts.
    """
    DELAY_TIMER.delay(delay=100, allow_sleep=True, block=False)
