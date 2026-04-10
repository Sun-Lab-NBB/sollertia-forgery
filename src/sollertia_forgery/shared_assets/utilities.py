"""Provides miscellaneous utility assets used across multiple library modules."""

from ataraxis_time import PrecisionTimer, TimerPrecisions

delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
"""The shared PrecisionTimer instance used across the library to delay the runtime's execution."""


def delay_terminal() -> None:
    """Uses the shared delay_timer instance to delay the runtime execution for one second to ensure proper visual
    separation of terminal printouts.
    """
    delay_timer.delay(delay=1, allow_sleep=True, block=False)
