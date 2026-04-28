"""Cross-package utilities shared by the bleaching, SCE, and tuning analysis pipelines.

Currently exposes the acquisition-warmup trimming helper (every pipeline drops the same leading window) and the
session-day display-unit resolver (cross-session aggregates label x-axes the same way regardless of which
modality they aggregate). Pipeline-specific helpers live in their per-package ``utilities`` modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import console

from ..shared_assets import DatasetColumn

if TYPE_CHECKING:
    from numpy.typing import NDArray


_ACQUISITION_WARMUP_SECONDS: float = 60.0
"""Number of leading seconds discarded from every loaded session trace before any analysis runs. Sollertia
experiments include a multi-minute pre-imaging baseline period during which the PMT gain, resonant scanner
phase, shutter, and laser power have not yet stabilized; the resulting initial fluorescence valley would
otherwise contaminate downstream estimates (per-cell baselines, within-session bleaching, SCE statistics,
place-field tuning). Trimming at load time guarantees every analyzer operates on stabilized samples without
needing to know the artifact exists."""


def trim_acquisition_warmup(df: pl.DataFrame) -> pl.DataFrame:
    """Drops the leading ``_ACQUISITION_WARMUP_SECONDS`` of samples from a session dataframe based on the
    ``time_us`` column.

    Notes:
        Operates on the polars dataframe directly (rather than the post-explode numpy arrays) so the warmup
        window never enters any subsequent column-level reshape. Sessions whose entire trace falls within the
        warmup window collapse to an empty dataframe; downstream loaders' existing length guards then produce
        NaN sentinels for such degenerate sessions.

    Args:
        df: Session dataframe loaded from ``DatasetFiles.DATA``. Must include ``DatasetColumn.TIME_US`` among
            the selected columns; all other columns are passed through untouched.

    Returns:
        The input dataframe sliced to drop every row whose ``time_us`` value precedes the warmup cutoff.
    """
    if df.height == 0:
        return df
    # noinspection PyTypeChecker
    time_us: NDArray[np.int64] = df[DatasetColumn.TIME_US.value].to_numpy()
    warmup_us = int(
        convert_time(
            time=_ACQUISITION_WARMUP_SECONDS,
            from_units=TimeUnits.SECOND,
            to_units=TimeUnits.MICROSECOND,
            as_float=True,
        )
    )
    cutoff_us = int(time_us[0]) + warmup_us
    warmup_index = int(np.searchsorted(a=time_us, v=cutoff_us, side="left"))
    if warmup_index <= 0:
        return df
    return df.slice(offset=warmup_index)


def resolve_display_units(days_since_first: NDArray[np.float32]) -> tuple[str, NDArray[np.int64]]:
    """Resolves the integer display unit and per-session tick array used by dataset-level summaries and plots.

    Notes:
        Returns ``("day", round(days_since_first))`` when every session's day-rounded offset is unique.
        Otherwise, falls back to ``("hour", round(days_since_first * 24))``. Storage and any cross-session fits
        continue to operate on float days; the integer ticks returned here are display-only.

    Args:
        days_since_first: Per-session day offsets relative to the first session.

    Returns:
        A tuple of unit label (``"day"`` or ``"hour"``) and an int64 tick array aligned with ``days_since_first``.

    Raises:
        ValueError: When sessions cannot be assigned unique day or hour ticks. Sollertia acquisition protocols
            mandate at least one hour between consecutive sessions, so the hour-rounded values are by
            construction distinct; a collision indicates a violated input invariant.
    """
    # noinspection PyTypeChecker
    rounded_days: NDArray[np.int64] = np.round(days_since_first).astype(np.int64, copy=False)
    if int(np.unique(rounded_days).size) == int(rounded_days.size):
        return "day", rounded_days

    # Promotes through float64 first so the *24 multiplication does not lose precision near the float32 boundary.
    # noinspection PyTypeChecker
    rounded_hours: NDArray[np.int64] = np.round(days_since_first.astype(np.float64) * 24.0).astype(
        np.int64, copy=False
    )
    if int(np.unique(rounded_hours).size) == int(rounded_hours.size):
        return "hour", rounded_hours

    message = (
        "Unable to assign unique integer day or hour labels to the supplied sessions. Sollertia acquisition "
        "protocols require at least one hour of separation between consecutive sessions, but at least two "
        "sessions in this set rounded to the same hour-since-first value, which violates that invariant."
    )
    console.error(message=message, error=ValueError)
    # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
    # noinspection PyUnreachableCode
    raise ValueError(message)  # pragma: no cover
