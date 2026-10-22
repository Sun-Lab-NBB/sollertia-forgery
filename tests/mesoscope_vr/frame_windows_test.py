"""Contains tests for the Mesoscope-VR resamplers that summarize each reference frame's window."""

from __future__ import annotations

import numpy as np
import pytest

from sollertia_forgery.mesoscope_vr.frame_windows import (
    _select_maxima,
    flag_window_onsets,
    _select_signed_peaks,
    resolve_window_maxima,
    _resolve_window_starts,
    resolve_window_signed_peaks,
)

_REFERENCE_TIME = np.array([1_000, 2_000, 3_000, 4_000], dtype=np.uint64)
"""The reference frame timestamps shared by most tests, one frame per millisecond."""


def test_resolve_window_starts_opens_every_window_at_the_preceding_frame() -> None:
    """Verifies that each window opens at the preceding frame, and that the first window opens one frame interval
    before the first frame.
    """
    assert _resolve_window_starts(reference_time=_REFERENCE_TIME).tolist() == [0, 1_000, 2_000, 3_000]


def test_resolve_window_starts_holds_a_single_frame_clock_to_its_own_timestamp() -> None:
    """Verifies that a single-frame clock opens its window one microsecond before its timestamp."""
    assert _resolve_window_starts(reference_time=np.array([500], dtype=np.uint64)).tolist() == [499]


def test_flag_window_onsets_flags_a_contact_that_falls_between_two_frames() -> None:
    """Verifies that a contact starting and ending between two frame timestamps flags the later frame."""
    flags = flag_window_onsets(
        event_time=np.array([0, 2_200, 2_260], dtype=np.uint64),
        states=np.array([0, 1, 0], dtype=np.uint8),
        reference_time=_REFERENCE_TIME,
    )

    assert flags.tolist() == [0, 0, 1, 0]
    assert flags.dtype == np.uint8


def test_flag_window_onsets_assigns_an_onset_at_a_frame_timestamp_to_that_frame() -> None:
    """Verifies that an onset read exactly at a frame timestamp belongs to that frame, and one microsecond later to the
    next frame.
    """
    flags = flag_window_onsets(
        event_time=np.array([2_000, 2_050, 3_001, 3_050], dtype=np.uint64),
        states=np.array([1, 0, 1, 0], dtype=np.uint8),
        reference_time=_REFERENCE_TIME,
    )

    assert flags.tolist() == [0, 1, 0, 1]


def test_flag_window_onsets_flags_one_frame_for_a_contact_held_across_several_readings() -> None:
    """Verifies that repeated engaged readings and a second onset in the same window flag a single frame."""
    flags = flag_window_onsets(
        event_time=np.array([1_100, 1_300, 1_500, 1_700, 2_500], dtype=np.uint64),
        states=np.array([1, 1, 0, 1, 1], dtype=np.uint8),
        reference_time=_REFERENCE_TIME,
    )

    assert flags.tolist() == [0, 1, 0, 0]


def test_flag_window_onsets_drops_onsets_outside_every_window() -> None:
    """Verifies that onsets before the first window and after the last frame are dropped, and that the first window
    spans one frame interval before the first frame.
    """
    flags = flag_window_onsets(
        event_time=np.array([0, 50, 100, 500, 600, 4_500, 4_600], dtype=np.uint64),
        states=np.array([1, 0, 0, 1, 0, 1, 0], dtype=np.uint8),
        reference_time=_REFERENCE_TIME,
    )

    assert flags.tolist() == [1, 0, 0, 0]


def test_flag_window_onsets_returns_no_flag_without_a_reading() -> None:
    """Verifies that an empty state stream flags no frame, and that an empty clock yields no flag at all."""
    empty_time = np.zeros(0, dtype=np.uint64)
    empty_states = np.zeros(0, dtype=np.uint8)

    assert flag_window_onsets(event_time=empty_time, states=empty_states, reference_time=_REFERENCE_TIME).tolist() == [
        0,
        0,
        0,
        0,
    ]
    assert (
        flag_window_onsets(
            event_time=np.array([10], dtype=np.uint64),
            states=np.array([1], dtype=np.uint8),
            reference_time=np.zeros(0, dtype=np.uint64),
        ).size
        == 0
    )


def test_resolve_window_signed_peaks_keeps_the_sign_of_a_brief_push() -> None:
    """Verifies that each window keeps the held or reported value of largest magnitude, with its sign."""
    peaks = resolve_window_signed_peaks(
        event_time=np.array([1_500, 2_200, 2_300, 3_500], dtype=np.uint64),
        values=np.array([1.0, -3.0, 0.5, 2.0], dtype=np.float64),
        reference_time=_REFERENCE_TIME,
    )

    # The first window closes before the first report and takes its value, the third weighs the held 1.0 against the
    # reported -3.0 and 0.5, and the fourth weighs the held 0.5 against the reported 2.0.
    assert peaks.tolist() == pytest.approx([1.0, 1.0, -3.0, 2.0])


def test_resolve_window_signed_peaks_returns_no_value_for_an_empty_clock() -> None:
    """Verifies that an empty reference clock yields no peak."""
    peaks = resolve_window_signed_peaks(
        event_time=np.array([10], dtype=np.uint64),
        values=np.array([1.0], dtype=np.float64),
        reference_time=np.zeros(0, dtype=np.uint64),
    )

    assert peaks.size == 0


def test_resolve_window_signed_peaks_rejects_a_source_without_a_report() -> None:
    """Verifies that a stepwise source holding no report aborts the resampling."""
    with pytest.raises(ValueError, match="must hold at least one report"):
        resolve_window_signed_peaks(
            event_time=np.zeros(0, dtype=np.uint64),
            values=np.zeros(0, dtype=np.float64),
            reference_time=_REFERENCE_TIME,
        )


def test_select_signed_peaks_keeps_the_earliest_value_of_a_tie() -> None:
    """Verifies that the kernel keeps the earliest of two values sharing the largest magnitude, and takes a later value
    of larger magnitude with its sign.
    """
    peaks = _select_signed_peaks.py_func(
        values=np.array([2.0, -2.0, 1.0, -3.0], dtype=np.float64),
        first=np.array([0, 2], dtype=np.int64),
        last=np.array([2, 4], dtype=np.int64),
    )

    assert peaks.tolist() == pytest.approx([2.0, -3.0])


def test_resolve_window_maxima_keeps_the_largest_frame_and_skips_unmeasured_frames() -> None:
    """Verifies that each window keeps its largest non-NaN value, reports not-a-number when every value is
    not-a-number, and falls back to linear interpolation when it holds no source frame.
    """
    maxima = resolve_window_maxima(
        frame_time=np.array([900, 1_400, 1_800, 2_600, 2_900], dtype=np.uint64),
        values=np.array([1.0, np.nan, 4.0, np.nan, np.nan], dtype=np.float32),
        reference_time=np.array([1_000, 2_000, 3_000, 3_500], dtype=np.uint64),
    )

    assert maxima.dtype == np.float32
    assert maxima[:2].tolist() == pytest.approx([1.0, 4.0])
    assert np.isnan(maxima[2])
    # The last window holds no source frame, and the interpolation past the final frame clamps to its value.
    assert np.isnan(maxima[3])


def test_resolve_window_maxima_interpolates_a_window_without_a_source_frame() -> None:
    """Verifies that a window holding no source frame takes the value interpolated at its frame's timestamp."""
    maxima = resolve_window_maxima(
        frame_time=np.array([0, 3_000], dtype=np.uint64),
        values=np.array([0.0, 30.0], dtype=np.float32),
        reference_time=_REFERENCE_TIME,
    )

    # The source frame at 0 sits on the first window's exclusive start, so only the third window holds a frame.
    assert maxima.tolist() == pytest.approx([10.0, 20.0, 30.0, 30.0])


def test_resolve_window_maxima_returns_no_value_for_an_empty_clock() -> None:
    """Verifies that an empty reference clock yields no maximum."""
    maxima = resolve_window_maxima(
        frame_time=np.array([10], dtype=np.uint64),
        values=np.array([1.0], dtype=np.float32),
        reference_time=np.zeros(0, dtype=np.uint64),
    )

    assert maxima.size == 0


def test_select_maxima_reports_each_range_through_the_interpreted_kernel() -> None:
    """Verifies the kernel's maximum, not-a-number, and fallback branches outside the compiled path."""
    maxima = _select_maxima.py_func(
        values=np.array([1.0, 5.0, np.nan], dtype=np.float32),
        first=np.array([0, 2, 3], dtype=np.int64),
        last=np.array([2, 3, 3], dtype=np.int64),
        fallback=np.array([-1.0, -1.0, 7.0], dtype=np.float32),
    )

    assert maxima[0] == pytest.approx(5.0)
    assert np.isnan(maxima[1])
    assert maxima[2] == pytest.approx(7.0)
