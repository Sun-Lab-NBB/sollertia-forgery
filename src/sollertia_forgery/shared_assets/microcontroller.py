"""Provides the event-stream primitives shared across the acquisition systems' microcontroller parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def merge_event_streams[ScalarT: np.generic](
    timestamps_a: NDArray[np.uint64],
    values_a: NDArray[ScalarT],
    timestamps_b: NDArray[np.uint64],
    values_b: NDArray[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Merges two chronologically-sorted event streams into a single timestamp-sorted stream.

    Notes:
        Uses NumPy's stable sort (``kind="stable"``), which NumPy maps to a linear-time radix sort for the uint64
        timestamp keys.

    Args:
        timestamps_a: The uint64 timestamp array for the first event stream.
        values_a: The value array for the first event stream.
        timestamps_b: The uint64 timestamp array for the second event stream.
        values_b: The value array for the second event stream.

    Returns:
        The chronologically sorted timestamps and the values reordered to match them.
    """
    timestamps = np.concatenate([timestamps_a, timestamps_b])
    values = np.concatenate([values_a, values_b])
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order]
