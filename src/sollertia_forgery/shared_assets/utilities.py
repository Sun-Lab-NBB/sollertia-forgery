"""Provides miscellaneous utility assets used across multiple library modules."""

from ataraxis_time import PrecisionTimer, TimerPrecisions

delay_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared PrecisionTimer instance used across the library to delay the runtime's execution."""


def delay_terminal() -> None:
    """Uses the shared delay_timer instance to delay the runtime execution for 100 milliseconds to ensure proper
    visual separation of terminal printouts.
    """
    delay_timer.delay(delay=100, allow_sleep=True, block=False)
