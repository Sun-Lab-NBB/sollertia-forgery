"""Provides miscellaneous utility assets used across multiple library modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions

if TYPE_CHECKING:
    from collections.abc import Sequence

_NATURAL_RANK_PREFIX: str = "__natural_rank_"
"""The prefix of the temporary rank columns a natural sort adds and drops. Prefixed so a column of the sorted frame
cannot collide with one."""

DELAY_TIMER: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The shared timer ``delay_terminal`` uses to delay the runtime's execution."""


def delay_terminal() -> None:
    """Delays the runtime execution for 100 milliseconds using the shared ``DELAY_TIMER``, so consecutive terminal
    printouts stay visually separated.
    """
    DELAY_TIMER.delay(delay=100, allow_sleep=True, block=False)


def natural_sort(frame: pl.DataFrame, by: Sequence[str], *, nulls_last: bool = False) -> pl.DataFrame:
    """Orders a frame on the named string columns the way a reader reads them, so 2 precedes 10.

    Notes:
        Many of the identifiers this library sorts on embed a number in text, which covers animal identifiers, camera
        source identifiers, plane specifiers, and operator-chosen dataset names. Ordering those as plain text puts 10
        ahead of 2, so a listing disagrees with the order the same identifiers are read and written in everywhere
        else.

        Each column is ranked over its distinct values alone and the frame is then ordered on the resulting integers.
        A column holds far fewer distinct identifiers than rows, so the text comparison runs over the small set and the
        row ordering stays a native integer sort.

    Args:
        frame: The frame to order.
        by: The names of the columns to order on, in precedence order.
        nulls_last: Determines whether rows holding no value for a column sort after the rows that do.

    Returns:
        The ordered frame, carrying the same columns it was given.
    """
    ranked = frame
    rank_columns: list[str] = []
    for column in by:
        ranks = {value: rank for rank, value in enumerate(natsorted(frame[column].unique().drop_nulls().to_list()))}
        rank_column = f"{_NATURAL_RANK_PREFIX}{column}"
        ranked = ranked.with_columns(
            pl.col(column).replace_strict(ranks, default=None, return_dtype=pl.UInt32).alias(rank_column)
        )
        rank_columns.append(rank_column)

    return ranked.sort(by=rank_columns, nulls_last=nulls_last).drop(rank_columns)


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
