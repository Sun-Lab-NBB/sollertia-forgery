"""Provides the resamplers that summarize what a source recorded inside each reference frame's window, for sources whose
brief events fall between two reference timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
import numpy as np
from ataraxis_base_utilities import console
from ataraxis_data_structures import interpolate_data

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flag_window_onsets(
    event_time: NDArray[np.uint64], states: NDArray[np.uint8], reference_time: NDArray[np.uint64]
) -> NDArray[np.uint8]:
    """Flags every reference frame whose window holds the onset of an engaged state.

    Notes:
        Frame i covers the window (t[i - 1], t[i]], so a reading taken at a frame's own timestamp belongs to that
        frame. The first window opens one frame interval before t[0], or one microsecond before it for a single-frame
        clock. An onset is an engaged reading that follows a disengaged one, or an engaged first reading. An onset
        outside every window is dropped.

    Args:
        event_time: The timestamp of every state reading, in microseconds, in ascending order.
        states: The state reported by every reading, nonzero while engaged.
        reference_time: The reference frame timestamps, in microseconds, in ascending order.

    Returns:
        One flag per reference frame, 1 when the frame's window holds at least one onset.
    """
    flags = np.zeros(reference_time.size, dtype=np.uint8)
    if not event_time.size or not reference_time.size:
        return flags

    engaged = states > 0
    previously_engaged = np.concatenate((np.zeros(1, dtype=np.bool_), engaged[:-1]))
    onset_time = event_time[engaged & ~previously_engaged].astype(np.int64)

    # Locates, for each onset, the first frame at or past it. That frame's window holds the onset unless the onset
    # falls at or before the window's exclusive start.
    frames = np.searchsorted(a=reference_time.astype(np.int64), v=onset_time, side="left")
    inside = frames < reference_time.size
    frames = frames[inside]
    inside = onset_time[inside] > _resolve_window_starts(reference_time=reference_time)[frames]
    flags[frames[inside]] = 1
    return flags


def resolve_window_signed_peaks(
    event_time: NDArray[np.uint64], values: NDArray[np.float64], reference_time: NDArray[np.uint64]
) -> NDArray[np.float64]:
    """Resolves the value of largest magnitude held by a stepwise source inside every reference frame's window, with
    its sign kept.

    Notes:
        A window's candidates are the value held at its start and every value reported inside it, because a stepwise
        source holds each reported value until the next report. A window that opens before the first report takes the
        first reported value. Frame windows follow the convention of ``flag_window_onsets``.

    Args:
        event_time: The timestamp of every report, in microseconds, in ascending order.
        values: The value carried by every report.
        reference_time: The reference frame timestamps, in microseconds, in ascending order.

    Returns:
        The signed peak of every reference frame's window.

    Raises:
        ValueError: If the source holds no report.
    """
    if not reference_time.size:
        return np.zeros(0, dtype=np.float64)
    if not event_time.size:
        message = (
            f"Unable to resolve the windowed peaks of a stepwise source. The source must hold at least one report, but "
            f"got {event_time.size} reports."
        )
        console.error(message=message, error=ValueError)

    report_time = event_time.astype(np.int64)
    first = np.searchsorted(a=report_time, v=_resolve_window_starts(reference_time=reference_time), side="right") - 1
    np.clip(a=first, a_min=0, a_max=None, out=first)
    last = np.searchsorted(a=report_time, v=reference_time.astype(np.int64), side="right")
    np.clip(a=last, a_min=first + 1, a_max=None, out=last)
    return _select_signed_peaks(values=values.astype(np.float64), first=first, last=last)


def resolve_window_maxima(
    frame_time: NDArray[np.uint64], values: NDArray[np.float32], reference_time: NDArray[np.uint64]
) -> NDArray[np.float32]:
    """Resolves the largest per-frame source value acquired inside every reference frame's window.

    Notes:
        Not-a-number values are skipped, and a window whose every value is not-a-number reports not-a-number. A window
        that holds no source frame takes the value linearly interpolated at its frame's timestamp, or the nearest edge
        value outside the source's span. A source slower than the reference clock therefore still yields a value at
        every frame. Frame windows follow the convention of ``flag_window_onsets``.

    Args:
        frame_time: The acquisition timestamp of every source frame, in microseconds, in ascending order.
        values: The value carried by every source frame.
        reference_time: The reference frame timestamps, in microseconds, in ascending order.

    Returns:
        The maximum of every reference frame's window, carried at single precision.
    """
    if not reference_time.size:
        return np.zeros(0, dtype=np.float32)

    source_time = frame_time.astype(np.int64)
    first = np.searchsorted(a=source_time, v=_resolve_window_starts(reference_time=reference_time), side="right")
    last = np.searchsorted(a=source_time, v=reference_time.astype(np.int64), side="right")
    interpolated = interpolate_data(
        source_coordinates=frame_time,
        source_values=values,
        target_coordinates=reference_time,
        is_discrete=False,
    ).astype(np.float32)
    return _select_maxima(values=values.astype(np.float32), first=first, last=last, fallback=interpolated)


def _resolve_window_starts(reference_time: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Resolves the exclusive start of every reference frame's window as the timestamp of the preceding frame.

    Notes:
        The first window opens one frame interval before the first timestamp.

    Args:
        reference_time: The reference frame timestamps, in microseconds, in ascending order.

    Returns:
        The window starts in microseconds, signed so that the first start can precede zero.
    """
    timestamps = reference_time.astype(np.int64)
    starts = np.empty_like(timestamps)
    starts[1:] = timestamps[:-1]
    # The window of a single-frame clock holds the frame's own timestamp alone, because such a clock defines no
    # interval.
    first_interval = timestamps[1] - timestamps[0] if timestamps.size > 1 else 1
    starts[0] = timestamps[0] - first_interval
    return starts


@njit(cache=True)
def _select_signed_peaks(
    values: NDArray[np.float64], first: NDArray[np.int64], last: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Selects the value of largest magnitude in each index range, keeping its sign and the earliest value of a tie.

    Args:
        values: The values to select from.
        first: The inclusive start index of every range.
        last: The exclusive end index of every range, past its start.

    Returns:
        The selected value of every range.
    """
    peaks = np.empty(first.size, dtype=np.float64)
    for frame in range(first.size):
        peak = values[first[frame]]
        for index in range(first[frame] + 1, last[frame]):
            if abs(values[index]) > abs(peak):
                peak = values[index]
        peaks[frame] = peak
    return peaks


@njit(cache=True)
def _select_maxima(
    values: NDArray[np.float32], first: NDArray[np.int64], last: NDArray[np.int64], fallback: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Selects the largest non-NaN value in each index range.

    Args:
        values: The values to select from.
        first: The inclusive start index of every range.
        last: The exclusive end index of every range.
        fallback: The value reported for every empty range.

    Returns:
        The largest non-NaN value of every range, not-a-number for a range holding only not-a-number values, and the
        fallback value for an empty range.
    """
    maxima = fallback.copy()
    for frame in range(first.size):
        if first[frame] >= last[frame]:
            continue
        maximum = np.float32(np.nan)
        for index in range(first[frame], last[frame]):
            value = values[index]
            if not np.isnan(value) and (np.isnan(maximum) or value > maximum):
                maximum = value
        maxima[frame] = maximum
    return maxima
