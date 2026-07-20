#!/usr/bin/env python
"""Inspect a Sollertia pipeline feather table for correctness.

Edit the TARGET block below, then run this file straight from your IDE (Run button) or with
``python inspect_feather.py``. It loads the polars-written ``.feather`` (Arrow IPC) file (or every
feather in a directory) and prints a per-column summary (dtype, valid / null / NaN counts, min, max,
mean, boolean true-fraction) followed by a head preview.

Null and NaN are reported separately because the pipeline writes NaN, not null, into the frames it
masks out, so a healthy pupil table reads as mostly NaN on its blink frames rather than null.

Set SPARK to a column name for a zero-dependency ASCII sparkline, or PNG to a file path to save a
matplotlib line plot of every numeric column against row position.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ======================================================================================================================
# EDIT THIS, then run the file. TARGET may be a single ``.feather`` or a directory of them.
# ======================================================================================================================
TARGET = "/home/data/Data/MaalstroomicFlow/305/2026-07-10-16-38-46-452689/processed_data/video_data/body_camera_timestamps.feather"
ROWS = 8  # Head preview row count. Set to 0 to skip the preview.
SPARK = None  # Column name to draw an ASCII sparkline for, or None.
PNG = None  # Output path to save a line plot of every numeric column, or None.
RECURSIVE = False  # Walk subdirectories when TARGET is a directory.
# ======================================================================================================================

_FLOATS = (pl.Float32, pl.Float64)
_INTS = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_BLOCKS = "▁▂▃▄▅▆▇█"


def discover_feathers(path: Path, *, recursive: bool) -> list[Path]:
    """Resolves a path to the feather files it stands for.

    A file resolves to itself. A directory resolves to the naturally sorted ``.feather`` files it
    holds, walking subdirectories when recursive is requested.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        globber = path.rglob if recursive else path.glob
        return sorted(globber("*.feather"))
    return []


def _fmt(value: object) -> str:
    """Formats a summary statistic compactly, blanking a missing value."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.4g}"
    return str(value)


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    """Builds a one-row-per-column summary table of the frame's contents.

    Float columns report their NaN count separately from their null count and compute min, max, and
    mean over the values that are neither. Integer columns report min, max, and mean. Boolean columns
    report the count and fraction of True. Every other dtype reports its number of unique values.
    """
    total = frame.height
    records: list[dict[str, object]] = []
    for name, dtype in frame.schema.items():
        series = frame.get_column(name)
        nulls = series.null_count()
        present = series.drop_nulls()
        nans = 0
        low = high = mean = None
        notes = ""
        if dtype in _FLOATS:
            valid = present.drop_nans()
            nans = present.len() - valid.len()
            if valid.len():
                low, high, mean = valid.min(), valid.max(), valid.mean()
        elif dtype in _INTS:
            if present.len():
                low, high, mean = present.min(), present.max(), present.mean()
        elif dtype == pl.Boolean:
            true_count = int(present.sum()) if present.len() else 0
            fraction = true_count / present.len() if present.len() else float("nan")
            notes = f"true={true_count} ({fraction:.1%})"
        else:
            notes = f"unique={present.n_unique()}"
        records.append(
            {
                "column": name,
                "dtype": str(dtype),
                "valid": total - nulls - nans,
                "nulls": nulls,
                "nans": nans,
                "min": _fmt(low),
                "max": _fmt(high),
                "mean": _fmt(mean),
                "notes": notes,
            }
        )
    return pl.DataFrame(records)


def _column_values(frame: pl.DataFrame, column: str) -> list[float]:
    """Returns a column's finite numeric values in row order, dropping null and NaN entries."""
    series = frame.get_column(column)
    if series.dtype in _FLOATS:
        series = series.drop_nans()
    return series.drop_nulls().cast(pl.Float64).to_list()


def sparkline(values: list[float], width: int = 100) -> str:
    """Renders finite values as a single-line block sparkline, averaging into at most width buckets."""
    if not values:
        return "(no finite values to plot)"
    if len(values) > width:
        step = len(values) / width
        buckets = [
            sum(values[int(i * step) : int((i + 1) * step)]) / max(1, int((i + 1) * step) - int(i * step))
            for i in range(width)
        ]
    else:
        buckets = values
    low, high = min(buckets), max(buckets)
    span = (high - low) or 1.0
    line = "".join(_BLOCKS[int((value - low) / span * (len(_BLOCKS) - 1))] for value in buckets)
    return f"{low:.4g} {line} {high:.4g}"


def save_png(frame: pl.DataFrame, out_path: Path) -> int:
    """Saves a stacked line plot of every numeric column against row position, returning column count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric = [name for name, dtype in frame.schema.items() if dtype in _FLOATS + _INTS]
    if not numeric:
        return 0
    figure, axes = plt.subplots(len(numeric), 1, figsize=(12, 1.6 * len(numeric)), sharex=True, squeeze=False)
    row_index = range(frame.height)
    for axis, name in zip(axes[:, 0], numeric, strict=True):
        axis.plot(row_index, frame.get_column(name).cast(pl.Float64).to_list(), linewidth=0.7)
        axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        axis.margins(x=0)
    axes[-1, 0].set_xlabel("row position")
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return len(numeric)


def inspect_one(path: Path, *, rows: int, spark: str | None, png: Path | None) -> None:
    """Prints the summary and head preview for one feather, plus any requested sparkline or plot."""
    frame = pl.read_ipc(path)
    size_mb = path.stat().st_size / 1e6
    print(f"\n{'=' * 100}")
    print(f"{path}")
    print(f"{frame.height:,} rows x {frame.width} cols   {size_mb:.2f} MB on disk")
    print(f"{'-' * 100}")

    with pl.Config(tbl_rows=frame.width + 1, tbl_cols=-1, tbl_hide_dataframe_shape=True, fmt_str_lengths=60):
        print(summarize(frame))

    if spark is not None:
        if spark not in frame.columns:
            print(f"\n[spark] column '{spark}' not in table; available: {frame.columns}")
        else:
            print(f"\n{spark}:\n{sparkline(_column_values(frame, spark))}")

    if png is not None:
        drawn = save_png(frame, png)
        print(f"\n[png] wrote {drawn} numeric column(s) to {png}" if drawn else "\n[png] no numeric columns to plot")

    if rows > 0:
        print(f"\nhead({min(rows, frame.height)}):")
        with pl.Config(tbl_rows=rows, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
            print(frame.head(rows))


if __name__ == "__main__":
    feathers = discover_feathers(Path(TARGET), recursive=RECURSIVE)
    if not feathers:
        print(f"No '.feather' files found under: {TARGET}")
    for feather in feathers:
        inspect_one(feather, rows=ROWS, spark=SPARK, png=Path(PNG) if PNG else None)
