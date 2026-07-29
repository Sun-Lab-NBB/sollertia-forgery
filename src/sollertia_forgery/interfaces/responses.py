"""Provides the response helpers and the paging machinery every Model Context Protocol read tool shares."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_ITEM_LIMIT: int = 200
"""The items a semi-detail page carries when the caller names no limit. A semi-detail row runs to roughly two hundred
bytes, so a full page stays near forty kilobytes."""

DEFAULT_DETAILED_LIMIT: int = 50
"""The items a detailed page carries when the caller names no limit. A detailed row runs to roughly five hundred
bytes, so this holds a full page near twenty-five kilobytes. Detail is meant for reading a few items closely, so its
page is deliberately far shorter than a semi-detail one."""


@dataclass(frozen=True, slots=True)
class PageWindow:
    """Describes the slice of a matched item set one response carries."""

    start: int
    """The index the page begins at, counted from the first matching item."""
    length: int | None
    """The items the page carries, or None when the page runs to the end of the matches."""
    next_start_row: int | None
    """The ``start_row`` that retrieves the following page, or None when this page ends the matches. A caller walks a
    matched set by following this until it is None, which is a stronger signal than comparing counts."""

    @property
    def stop(self) -> int | None:
        """Returns the index the page ends before, or None when it runs to the end of the matches."""
        return None if self.length is None else self.start + self.length


def resolve_page(total: int, limit: int, start_row: int) -> PageWindow:
    """Resolves which slice of a matched item set a response carries.

    Notes:
        Container-agnostic by design. The caller slices its own frame or list from the returned window, so one paging
        rule serves the tools backed by a stored table and the tools backed by an in-memory job set alike.

        A limit at or below zero lifts the cap and returns every match from the requested start. That escape exists so
        a caller reading under a tight filter can take the whole result in one response, and so the useful page size
        can grow as an agent's context does. It is never the default.

    Args:
        total: The items matching the caller's filters, before any cap.
        limit: The items to carry, or a value at or below zero to carry every match.
        start_row: The index to begin at, counted from the first matching item. A negative value starts at the
            beginning.

    Returns:
        The window describing the page, carrying the start index, its length, and the start row of the next page.
    """
    start = max(0, start_row)
    if start >= total:
        return PageWindow(start=start, length=0, next_start_row=None)
    if limit <= 0:
        return PageWindow(start=start, length=None, next_start_row=None)

    remaining = total - start
    length = min(limit, remaining)
    return PageWindow(start=start, length=length, next_start_row=start + length if length < remaining else None)


def page_fields(window: PageWindow, total: int, listed: int) -> dict[str, Any]:
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
        Values are keyed by their string form, so an enumeration member and its value count as one. A null counts
        under ``none``, since an absent subject is itself a category a caller filters on.

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


def resolve_detail_limit(limit: int | None, *, detailed: bool) -> int:
    """Resolves the page size to use when the caller named none, from the detail the response carries.

    Notes:
        The default follows the weight of one item rather than one figure for every response, because a detailed item
        carries several times what a semi-detail one does. Every read tool holds its items to one row of a stored table
        or one tracked job, so two tiers cover them all.

    Args:
        limit: The limit the caller named, or None to take the default.
        detailed: Whether the response carries full per-item fields.

    Returns:
        The page size to apply.
    """
    if limit is not None:
        return limit
    if not detailed:
        return DEFAULT_ITEM_LIMIT
    return DEFAULT_DETAILED_LIMIT


def project_item(item: dict[str, Any], fields: Sequence[str], *, drop_empty: bool = True) -> dict[str, Any]:
    """Narrows one item to the named fields, leaving out the ones carrying nothing.

    Notes:
        An absent key reads as empty, so omitting a field that holds nothing costs a reader no information and keeps
        the common case small. Most jobs carry no options, no prerequisites, and no error, so those keys would
        otherwise be dead weight on every row.

    Args:
        item: The item to narrow.
        fields: The fields to keep, in the order they should appear.
        drop_empty: Determines whether to leave out the fields whose value is None or an empty collection.

    Returns:
        The narrowed item.
    """
    narrowed: dict[str, Any] = {}
    for field_name in fields:
        if field_name not in item:
            continue
        value = item[field_name]
        if drop_empty and (value is None or (isinstance(value, list | dict | str) and len(value) == 0)):
            continue
        narrowed[field_name] = value
    return narrowed


def ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}
