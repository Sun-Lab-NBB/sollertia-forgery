"""Provides the response helpers and the paging machinery every Model Context Protocol read tool shares."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

from ataraxis_time import TimeUnits, convert_time

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import polars as pl
    from ataraxis_time import PrecisionTimer

_DEFAULT_ITEM_LIMIT: int = 200
"""The items a semi-detail page carries when the caller names no limit."""

_DEFAULT_DETAILED_LIMIT: int = 50
"""The items a detailed page carries when the caller names no limit. Detail is meant for reading a few items closely,
so its page is deliberately shorter."""

_BREAKDOWN_AXIS_LIMIT: int = 50
"""The distinct values one breakdown axis lists."""


@dataclass(frozen=True, slots=True)
class _PageWindow:
    """Describes the slice of a matched item set one response carries."""

    start: int
    """The index at which the page begins, counted from the first matching item."""
    length: int | None
    """The items the page carries, or None when the page runs to the end of the matches."""
    next_start_row: int | None
    """The ``start_row`` that retrieves the following page, or None when this page ends the matches. A caller walks a
    matched set by following this until it is None, which is a stronger signal than comparing counts."""

    @property
    def stop(self) -> int | None:
        """Returns the index before which the page ends, or None when it runs to the end of the matches."""
        return None if self.length is None else self.start + self.length


def resolve_page(total: int, limit: int, start_row: int) -> _PageWindow:
    """Resolves which slice of a matched item set a response carries.

    Notes:
        The caller slices its own frame or list from the returned window, so one paging rule serves the tools backed by
        a stored table and the tools backed by an in-memory job set alike.

        A limit at or below zero lifts the cap and returns every match from the requested start. That escape exists so
        a caller reading under a tight filter can take the whole result in one response, and so the useful page size
        can grow as an agent's context does. It is never the default.

    Args:
        total: The items matching the caller's filters, before any cap.
        limit: The items to carry, or a value at or below zero to carry every match.
        start_row: The index at which to begin, counted from the first matching item. A negative value starts at the
            beginning.

    Returns:
        The window describing the page, carrying the start index, its length, and the start row of the next page.
    """
    start = max(0, start_row)
    if start >= total:
        return _PageWindow(start=start, length=0, next_start_row=None)
    if limit <= 0:
        return _PageWindow(start=start, length=None, next_start_row=None)

    remaining = total - start
    length = min(limit, remaining)
    return _PageWindow(start=start, length=length, next_start_row=start + length if length < remaining else None)


def page_fields(window: _PageWindow, total: int, listed: int) -> dict[str, Any]:
    """Renders the paging fields a response reports alongside its items.

    Args:
        window: The resolved page window.
        total: The items matching the caller's filters, before any cap.
        listed: The items the response actually carries.

    Returns:
        A dictionary carrying the listed count, the matched total, the start row, and the next start row.
    """
    return {
        "rows": listed,
        "matched_rows": total,
        "start_row": window.start,
        "next_start_row": window.next_start_row,
    }


def count_values(values: Iterable[Any]) -> dict[str, int]:
    """Counts how often each value occurs, which is one axis of a breakdown.

    Notes:
        Values are keyed by their string form, so an enumeration member and its value count as one. A null counts under
        ``none``, since an absent subject is itself a category on which a caller filters.

    Args:
        values: The column of values to count.

    Returns:
        A dictionary mapping each value to its count, ordered by value.
    """
    counts: dict[str, int] = {}
    for value in values:
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def bounded_counts(values: Iterable[Any]) -> dict[str, Any]:
    """Counts how often each value occurs, reporting the size of an axis that holds too many distinct values to list.

    Notes:
        An axis carrying one value per unit grows with the project, so listing it would eventually cost more than the
        summary that carries it. Past the limit the axis reports how many distinct values it holds, and a caller
        reaches the values themselves by filtering on that axis.

    Args:
        values: The column of values to count.

    Returns:
        A dictionary mapping each value to its count, or one carrying ``distinct_values`` and an ``elided`` note
        stating why the counts are left out.
    """
    counts = count_values(values=values)
    if len(counts) <= _BREAKDOWN_AXIS_LIMIT:
        return counts
    return {
        "distinct_values": len(counts),
        "elided": (
            f"This axis holds more than {_BREAKDOWN_AXIS_LIMIT} distinct values, so its counts are left out. Filter "
            f"on this axis to read the items carrying one of its values."
        ),
    }


def frame_breakdown(frame: pl.DataFrame, axes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Counts how many rows of a stored table carry each value of every filterable axis.

    Notes:
        This is what a bare call reports in place of a listing. It names the values on which a caller can filter and how
        much each would match, so an agent orients itself on one response rather than paging a whole artifact.

    Args:
        frame: The whole stored table.
        axes: The columns to count, which are the columns by which a caller may filter.

    Returns:
        A dictionary mapping each present axis to its value counts, or to the size of an axis that holds too many
        distinct values to list.
    """
    return {axis: bounded_counts(values=frame[axis].to_list()) for axis in axes if axis in frame.columns}


def resolve_elapsed_seconds(timer: PrecisionTimer) -> float:
    """Resolves how long an operation ran as the seconds in which a response reports it.

    Args:
        timer: The millisecond-precision timer instantiated when the operation began.

    Returns:
        The elapsed seconds, rounded to the millisecond.
    """
    seconds = convert_time(
        time=timer.elapsed, from_units=TimeUnits.MILLISECOND, to_units=TimeUnits.SECOND, as_float=True
    )
    return round(seconds, 3)


def resolve_detail_limit(limit: int | None, *, detailed: bool) -> int:
    """Resolves the page size to use when the caller named none, from the detail the response carries.

    Notes:
        The default follows the weight of one item rather than one figure for every response, because a detailed item
        carries several times what a semi-detail one does. Every read tool holds its items to one row of a stored table
        or one tracked job, so two tiers cover them all.

    Args:
        limit: The limit the caller named, or None to take the default.
        detailed: Determines whether the response carries full per-item fields.

    Returns:
        The page size to apply.
    """
    if limit is not None:
        return limit
    if not detailed:
        return _DEFAULT_ITEM_LIMIT
    return _DEFAULT_DETAILED_LIMIT


def project_item(item: dict[str, Any], fields: Sequence[str], *, drop_empty: bool = True) -> dict[str, Any]:
    """Narrows one item to the named fields, leaving out the ones carrying nothing.

    Notes:
        An absent key reads as empty, so omitting a field that holds nothing costs a reader no information and keeps
        the common case small. Most jobs carry no options, no prerequisites, and no error, so those keys would
        otherwise be dead weight on every row.

    Args:
        item: The item to narrow.
        fields: The fields to keep, in the order they should appear.
        drop_empty: Determines whether to leave out the fields whose value is None, an empty string, or an empty list
            or dictionary.

    Returns:
        The narrowed item.
    """
    narrowed: dict[str, Any] = {}
    for field_name in fields:
        if field_name not in item:
            continue
        value = item[field_name]
        if drop_empty and (value is None or (isinstance(value, list | dict | str) and not value)):
            continue
        narrowed[field_name] = value
    return narrowed


def ok_response(**payload: Any) -> dict[str, Any]:
    """Constructs a successful response dict with a ``success`` flag set to True.

    Args:
        payload: The response fields to carry alongside the success flag.

    Returns:
        The response dictionary.
    """
    return {"success": True, **payload}


def error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message.

    Args:
        message: The error text to report.

    Returns:
        The response dictionary.
    """
    return {"success": False, "error": message}


def reject_unknown(frame: pl.DataFrame, column: str, values: list[str], subject: str) -> dict[str, Any] | None:
    """Builds the error response for a filter naming a value the stored table does not hold.

    Notes:
        Reports what is available rather than returning an empty page, because an empty page and a mistyped filter look
        identical to a caller otherwise.

    Args:
        frame: The whole stored table.
        column: The column being filtered.
        values: The values the caller named.
        subject: The noun naming what one row of the table describes.

    Returns:
        The error response, or None when every named value is present.
    """
    if column not in frame.columns:
        return error_response(message=f"Unknown column '{column}'. Available: {sorted(frame.columns)}.")
    available = sorted({str(entry) for entry in frame[column].to_list() if entry is not None})
    unknown = sorted({value for value in values if value not in available})
    if unknown:
        return error_response(message=f"No {subject} has '{column}' in {unknown}. Available: {available}.")
    return None
