"""
cell_change_classification.py
=============================
Cell-level analysis of how place-field representations change across
conditions and across days. Operates on the long-form Polars `field_records`
table from `field_records.py` — the data layer.

Refactored from `cell_change_classification_4.py` to consume
field_records DataFrames instead of FieldUnit lists. The match/classify
logic is unchanged; the inputs and identifier types are.

Four analyses share two matching primitives:
    1. Across-days, single trial type (e.g. ABC over pre-extension days):
       baseline place-cell stability / day-to-day drift rate. Track is
       physically identical across days, so position matching suffices.
    2. Across-days, single trial type, post-extension days: same machinery
       as (1), just called separately on a different day range.
    3. Pre-vs-post extension on the base track (last pre-day vs first
       post-day, or pooled): does the introduction of ABDC perturb ABC?
       Mechanically identical to (1); the *interpretation* is different —
       change here is *attributable to* the extension. Drift baseline from
       (1) is the control; extension-induced change = excess over drift.
    4. Within-session, between tracks (ABC vs ABDC, same day): how does the
       brain build the extension on top of the base track? Uses DUAL
       matching:
         - position matcher: cm-based, Hungarian + tolerance. Works for the
           shared pre-bifurcation segment (A, 0a, B, 0b).
         - cue-identity matcher: same primary_zone (A↔A, B↔B, C↔C', etc.).
           Captures the case where a cell fires at C in ABC and at C' in
           ABDC — cue identity preserved even though absolute position has
           shifted because D was inserted before C.
       Per-field tags reveal which reference frame each field is locked to.

Field matching algorithm
------------------------
Per cell, build the pairwise position-distance matrix between fields in set
A and fields in set B, then run Hungarian assignment. Hungarian over greedy
nearest-neighbor matters when cells have multiple fields and a naïve greedy
match would steal a real shared-region pairing for a far-away field. After
assignment, two thresholds gate the result:
    - stable_threshold_cm (default 15): shift ≤ → 'stable', else 'shifted'.
    - match_max_cm (default 30 ≈ typical CA1 PF width): pairs above this
      are unmatched; the fields are treated as separate (one in A, one in B).
Two tolerance modes:
    - 'fixed_cm' (default): thresholds are absolute cm.
    - 'half_field_width': thresholds scale with the matched fields'
      detected widths.

There is no community-standard threshold for stable vs shifted; recommended
workflow:
    - Always emit the continuous shift values regardless of threshold.
    - Run sensitivity analysis across {10, 15, 20, 25} cm.
    - Report the full distribution of nearest-neighbor shifts in figures.

Cell-level categories (mutually exclusive, per cell)
----------------------------------------------------
    'non_field':   no fields in either A or B.
    'recruited':   fields only in B (newly appeared).
    'lost':        fields only in A (disappeared).
    'stable':      same n_fields in A and B; all matches within stable_threshold.
    'shifted':     same n_fields; ≥1 match outside stable_threshold (still
                   within match_max_cm).
    'field_added': all of A's fields matched stably; B has extras.
    'field_lost':  all of B's fields matched stably; A has extras.
    'complex':     anything else — typically 'kept B-field stable, shifted
                   another, added a third'.

Presence axis (orthogonal): 'a_only', 'b_only', 'both', 'neither'.

Per-field tags (analysis 4 only)
--------------------------------
    'position_and_cue_stable':     pos match (stable) AND cue match.
    'cue_stable_position_shifted': pos match (shifted) AND cue match.
    'cue_stable_only':             cue match exists, position match did not.
                                   The C↔C' case.
    'position_stable_only':        pos match (stable), cue zones differ.
                                   Cell firing at same cm under different cue
                                   contexts — position- (or possibly reward-
                                   distance-) anchored rather than cue-anchored.
    'position_shifted_only':       pos match (shifted), cue zones differ.
    'track_specific':              neither matcher found a partner (e.g.
                                   D-only fields on ABDC).

Caveat: 'cue_stable_only' for C↔C' is consistent with cue-locking OR
reward-locking (since C is reward-adjacent on both tracks). Pure cue-locking
is cleaner to argue from non-terminal cues like B↔B.

Public API
----------
    match_records                     - position-based matcher (Hungarian)
    match_records_by_cue              - cue-identity matcher
    classify_across_days              - analysis 1 + 2
    classify_pre_vs_post_extension    - analysis 3
    classify_between_tracks           - analysis 4 (dual matching)
    cell_changes_to_df                - per-cell results -> Polars DataFrame
    field_outcomes_to_df              - per-field tags  -> Polars DataFrame
    crosstab_categories               - joint distribution across two analyses
    multi_crosstab                    - run several preset cross-tabs at once
    cell_change_summary               - text diagnostic
    plot_category_distribution        - stacked bar of cell categories
    plot_shift_distributions          - overlaid |shift_cm| histograms
    plot_field_tags_by_zone           - per-zone field-tag bars (analysis 4)
    plot_crosstab_heatmap             - heatmap from a multi_crosstab DataFrame
    run_full_change_analysis          - orchestrator: runs all four + plots

Identifier conventions
----------------------
- Field IDs are the global string `field_id` from field_records (format
  `{cell_idx}_{date}_{trial_type}_{i}`). Joinable directly to the records
  frame for any extra metadata.
- Cell IDs use `cell_idx` to match field_records.

Dependencies: numpy, polars, scipy.optimize, matplotlib, field_records.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment


# DATA STRUCTURES

@dataclass
class FieldMatch:
    """One paired (A-field, B-field) for a single cell.

    'A' and 'B' are abstract labels — could be two conditions in the same
    session, two days for the same trial type, or anything else.

    Attributes:
        cell_idx: Cell index (consistent across A and B; for across-day
            analyses this assumes multiday cell registration is already done).
        a_field_id: Global field_id of the A-side field (str, joinable to
            field_records).
        b_field_id: Global field_id of the B-side field.
        a_position_cm: Peak position of the A-field (cm).
        b_position_cm: Peak position of the B-field (cm).
        shift_cm: Signed shift (b_position_cm - a_position_cm). Positive means
            the B-field is further along its track than the A-field.
        status: 'stable' if |shift_cm| ≤ stable_threshold_cm at match time,
            else 'shifted'.
    """
    cell_idx: int
    a_field_id: str
    b_field_id: str
    a_position_cm: float
    b_position_cm: float
    shift_cm: float
    status: str  # 'stable' | 'shifted'


@dataclass
class CellChange:
    """Per-cell summary of how a cell's place fields changed between A and B.

    Attributes:
        cell_idx: Cell index.
        n_fields_a: Number of fields detected in A.
        n_fields_b: Number of fields detected in B.
        matches: All matched (A, B) field pairs for this cell. Length ≤
            min(n_fields_a, n_fields_b).
        unmatched_a_field_ids: field_ids in A with no B partner. 'lost' fields
            (or, if n_fields_b == 0, the cell is 'lost' overall).
        unmatched_b_field_ids: field_ids in B with no A partner. 'gained/added'.
        category: One of 'non_field', 'recruited', 'lost', 'stable', 'shifted',
            'field_added', 'field_lost', 'complex'.
        presence: 'a_only', 'b_only', 'both', 'neither'. Orthogonal to category.
        max_shift_cm: max(|shift_cm|) across matches. NaN if no matches.
        mean_shift_cm: mean(|shift_cm|) across matches. NaN if no matches.
    """
    cell_idx: int
    n_fields_a: int
    n_fields_b: int
    matches: list[FieldMatch]
    unmatched_a_field_ids: list[str]
    unmatched_b_field_ids: list[str]
    category: str
    presence: str
    max_shift_cm: float
    mean_shift_cm: float


@dataclass
class FieldOutcome:
    """Per-field tag from the dual-matcher (analysis 4).

    Attributes:
        cell_idx: Cell index.
        condition: Which condition this field belongs to (e.g. 'ABC').
        field_id: Global field_id (joinable to field_records).
        peak_position_cm: Peak position (cm).
        zone: Cue-zone label (primary_zone from field_records).
        tag: One of 'position_and_cue_stable', 'cue_stable_position_shifted',
            'cue_stable_only', 'position_stable_only', 'position_shifted_only',
            'track_specific'.
        partner_position_field_id: field_id of the position-matched partner
            in the other condition, or None.
        partner_cue_field_id: field_id of the cue-matched partner, or None.
        partner_position_shift_cm: signed shift if a position match exists,
            else NaN.
    """
    cell_idx: int
    condition: str
    field_id: str
    peak_position_cm: float
    zone: str
    tag: str
    partner_position_field_id: str | None
    partner_cue_field_id: str | None
    partner_position_shift_cm: float


# FIELD GROUPING

def _group_records_by_cell(
    records: pl.DataFrame,
) -> dict[int, list[tuple[str, float, float, str]]]:
    """
    Group a (filtered) field_records frame by cell_idx, into a dict of small
    per-cell lists.

    Memory note: extracts the four columns we need as numpy arrays once, then
    builds the dict by indexed slicing — avoids per-row Python overhead from
    iter_rows.

    Returns:
        Dict mapping cell_idx -> list of (field_id, phys_cm_center, width_cm,
        primary_zone), sorted by phys_cm_center within each cell.
    """
    if records.is_empty():
        return {}

    # Pull columns once. Note: phys_cm_center sort is done in numpy below.
    cells = records['cell_idx'].to_numpy()
    fids = records['field_id'].to_numpy()
    pos = records['phys_cm_center'].to_numpy()
    widths = records['width_cm'].to_numpy()
    zones = records['primary_zone'].to_numpy()

    # Sort all rows by (cell_idx, phys_cm_center) so per-cell lists are in
    # left-to-right track order, matching the FieldUnit field_id semantics.
    order = np.lexsort((pos, cells))

    by_cell: dict[int, list[tuple[str, float, float, str]]] = defaultdict(list)
    for k in order:
        by_cell[int(cells[k])].append(
            (str(fids[k]), float(pos[k]), float(widths[k]), str(zones[k]))
        )
    return dict(by_cell)


# PRIMITIVE 1: POSITION MATCHER (HUNGARIAN ASSIGNMENT)

def _match_one_cell_position(
    a_fields: list[tuple[str, float, float, str]],
    b_fields: list[tuple[str, float, float, str]],
    cell_idx: int,
    stable_threshold_cm: float,
    match_max_cm: float,
    tolerance_mode: str,
) -> tuple[list[FieldMatch], list[str], list[str]]:
    """
    Hungarian-assign one cell's A-fields to its B-fields.

    Cost matrix is absolute pairwise position distance in cm. Hungarian
    returns the minimum-total-cost one-to-one assignment over the smaller
    side. After assignment, each pair is gated by match_max_cm (above →
    unmatched) and labelled stable/shifted by stable_threshold_cm.

    Returns:
        (matches, unmatched_a_field_ids, unmatched_b_field_ids)
    """
    n_a = len(a_fields)
    n_b = len(b_fields)
    if n_a == 0 or n_b == 0:
        return [], [t[0] for t in a_fields], [t[0] for t in b_fields]

    # Cost matrix = absolute pairwise shift in cm.
    a_pos = np.array([t[1] for t in a_fields])
    b_pos = np.array([t[1] for t in b_fields])
    a_w = np.array([t[2] for t in a_fields])
    b_w = np.array([t[2] for t in b_fields])
    cost = np.abs(a_pos[:, None] - b_pos[None, :])

    # Hungarian: minimum-cost assignment over the smaller dimension.
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_a_local: set[int] = set()
    matched_b_local: set[int] = set()
    matches: list[FieldMatch] = []

    for i, j in zip(row_idx, col_idx):
        shift_abs = float(cost[i, j])
        shift_signed = float(b_pos[j] - a_pos[i])

        # Pick effective thresholds for THIS pair.
        if tolerance_mode == 'fixed_cm':
            stable_thr = stable_threshold_cm
            max_thr = match_max_cm
        elif tolerance_mode == 'half_field_width':
            mean_w = 0.5 * (float(a_w[i]) + float(b_w[j]))
            # Width 0 (no records column populated) collapses thresholds to 0
            # which would unmatch everything — fall back to fixed_cm.
            if mean_w <= 0:
                stable_thr = stable_threshold_cm
                max_thr = match_max_cm
            else:
                stable_thr = 0.5 * mean_w
                max_thr = mean_w
        else:
            raise ValueError(f"Unknown tolerance_mode: {tolerance_mode!r}")

        if shift_abs > max_thr:
            # Pair is too far apart — treat A and B fields as separate.
            continue

        status = 'stable' if shift_abs <= stable_thr else 'shifted'
        matches.append(FieldMatch(
            cell_idx=cell_idx,
            a_field_id=a_fields[i][0],
            b_field_id=b_fields[j][0],
            a_position_cm=float(a_pos[i]),
            b_position_cm=float(b_pos[j]),
            shift_cm=shift_signed,
            status=status,
        ))
        matched_a_local.add(i)
        matched_b_local.add(j)

    unmatched_a = [a_fields[i][0] for i in range(n_a) if i not in matched_a_local]
    unmatched_b = [b_fields[j][0] for j in range(n_b) if j not in matched_b_local]
    return matches, unmatched_a, unmatched_b


def match_records(
    records_a: pl.DataFrame,
    records_b: pl.DataFrame,
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    tolerance_mode: str = 'fixed_cm',
) -> dict[int, tuple[list[FieldMatch], list[str], list[str]]]:
    """
    Position-based field matcher for two field_records subsets.

    'A' and 'B' are arbitrary roles — this function does not interpret them.
    Cell IDs in `records_a` and `records_b` MUST share a coordinate system.
    For across-day uses that means the dataset is already cross-day cell-
    registered (e.g. via Suite2P multi-day matching).

    Per cell, runs Hungarian assignment over the pairwise position-distance
    matrix between the cell's A-fields and B-fields. Pairs above match_max_cm
    are unmatched. Surviving pairs are tagged 'stable' or 'shifted'.

    Args:
        records_a: Polars frame of records from context A. Filter to one
            (date, trial_type) upstream.
        records_b: Polars frame of records from context B.
        stable_threshold_cm: Pairs with |shift| ≤ this are 'stable',
            otherwise 'shifted'. Default 15 cm.
        match_max_cm: Pairs with |shift| > this are NOT matched (treated as
            separate fields). Default 30 cm ≈ typical CA1 PF width.
        tolerance_mode: 'fixed_cm' (use the cm thresholds verbatim) or
            'half_field_width' (thresholds scale with mean matched widths).

    Returns:
        Dict mapping cell_idx -> (matches, unmatched_a_field_ids,
        unmatched_b_field_ids). Includes every cell with ≥1 field in
        either A or B.
    """
    by_cell_a = _group_records_by_cell(records_a)
    by_cell_b = _group_records_by_cell(records_b)
    all_cells = set(by_cell_a) | set(by_cell_b)

    out: dict[int, tuple[list[FieldMatch], list[str], list[str]]] = {}
    for c in sorted(all_cells):
        a_fields = by_cell_a.get(c, [])
        b_fields = by_cell_b.get(c, [])
        out[c] = _match_one_cell_position(
            a_fields, b_fields, cell_idx=c,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm,
            tolerance_mode=tolerance_mode,
        )
    return out


# PRIMITIVE 2: CUE-IDENTITY MATCHER

def _match_one_cell_cue(
    a_fields: list[tuple[str, float, float, str]],
    b_fields: list[tuple[str, float, float, str]],
) -> list[tuple[str, str, str]]:
    """
    Per-cell cue-identity match.

    Within a shared zone, pair fields by position order (earliest A-in-zone
    ↔ earliest B-in-zone). Zones are short (≤ ~50 cm) so multiple fields in
    the same zone are uncommon, and position order within the zone is the
    only sensible tiebreak.

    Each input list element is (field_id, pos_cm, width_cm, zone_id).

    Returns:
        List of (a_field_id, b_field_id, zone_label) for paired fields.
        Unmatched fields are simply absent.
    """
    if not a_fields or not b_fields:
        return []

    # Bucket each side's fields by zone label.
    a_by_zone: dict[str, list[tuple[str, float]]] = defaultdict(list)
    b_by_zone: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for fid, p, _w, z in a_fields:
        a_by_zone[z].append((fid, p))
    for fid, p, _w, z in b_fields:
        b_by_zone[z].append((fid, p))

    pairs: list[tuple[str, str, str]] = []
    for z, a_in_zone in a_by_zone.items():
        if z not in b_by_zone:
            continue
        b_in_zone = b_by_zone[z]
        # Sort by position; pair zip-style.
        a_sorted = sorted(a_in_zone, key=lambda t: t[1])
        b_sorted = sorted(b_in_zone, key=lambda t: t[1])
        for (a_fid, _), (b_fid, _) in zip(a_sorted, b_sorted):
            pairs.append((a_fid, b_fid, z))
    return pairs


def match_records_by_cue(
    records_a: pl.DataFrame,
    records_b: pl.DataFrame,
) -> dict[int, list[tuple[str, str, str]]]:
    """
    Cue-identity field matcher.

    Pairs fields between two conditions if they fall in zones with the same
    label (`primary_zone` in records). Useful only when comparing across
    DIFFERENT track structures (e.g. ABC vs ABDC); within a single track,
    cue-identity is degenerate with position so the position matcher already
    covers it.

    Args:
        records_a: Polars frame from condition A.
        records_b: Polars frame from condition B.

    Returns:
        Dict mapping cell_idx -> list of (a_field_id, b_field_id, zone_label).
    """
    by_cell_a = _group_records_by_cell(records_a)
    by_cell_b = _group_records_by_cell(records_b)
    all_cells = set(by_cell_a) | set(by_cell_b)

    out: dict[int, list[tuple[str, str, str]]] = {}
    for c in sorted(all_cells):
        out[c] = _match_one_cell_cue(
            by_cell_a.get(c, []),
            by_cell_b.get(c, []),
        )
    return out


# CELL CATEGORIZATION

def _classify_one_cell(
    cell_idx: int,
    matches: list[FieldMatch],
    unmatched_a: list[str],
    unmatched_b: list[str],
) -> CellChange:
    """Build a CellChange from one cell's match results."""
    n_match = len(matches)
    n_a = n_match + len(unmatched_a)
    n_b = n_match + len(unmatched_b)
    n_shifted = sum(1 for m in matches if m.status == 'shifted')

    # Presence axis.
    if n_a == 0 and n_b == 0:
        presence = 'neither'
    elif n_a > 0 and n_b > 0:
        presence = 'both'
    elif n_a > 0:
        presence = 'a_only'
    else:
        presence = 'b_only'

    # Mutually-exclusive category.
    if n_a == 0 and n_b == 0:
        category = 'non_field'
    elif n_a == 0 and n_b > 0:
        category = 'recruited'
    elif n_a > 0 and n_b == 0:
        category = 'lost'
    elif n_a == n_b == n_match and n_shifted == 0:
        category = 'stable'
    elif n_a == n_b == n_match and n_shifted > 0:
        category = 'shifted'
    elif n_match == n_a and n_b > n_a and n_shifted == 0:
        category = 'field_added'
    elif n_match == n_b and n_a > n_b and n_shifted == 0:
        category = 'field_lost'
    else:
        # Mixed: some stable, some shifted, possibly extras.
        category = 'complex'

    if matches:
        abs_shifts = np.array([abs(m.shift_cm) for m in matches])
        max_shift = float(abs_shifts.max())
        mean_shift = float(abs_shifts.mean())
    else:
        max_shift = float('nan')
        mean_shift = float('nan')

    return CellChange(
        cell_idx=cell_idx,
        n_fields_a=n_a,
        n_fields_b=n_b,
        matches=matches,
        unmatched_a_field_ids=unmatched_a,
        unmatched_b_field_ids=unmatched_b,
        category=category,
        presence=presence,
        max_shift_cm=max_shift,
        mean_shift_cm=mean_shift,
    )


def _classify_all(
    match_result: dict[int, tuple[list[FieldMatch], list[str], list[str]]],
) -> list[CellChange]:
    """Build CellChange list from the dict returned by match_records()."""
    return [_classify_one_cell(c, m, ua, ub)
            for c, (m, ua, ub) in match_result.items()]


# RECORDS-FILTERING HELPERS

def _filter_records(
    records: pl.DataFrame,
    date: str | None = None,
    trial_type: str | None = None,
    cells: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Filter a field_records frame by date, trial_type, and/or cell subset.
    Centralizes the boilerplate so each analysis is one or two filter calls.
    """
    df = records
    if date is not None:
        df = df.filter(pl.col('date') == date)
    if trial_type is not None:
        df = df.filter(pl.col('trial_type') == trial_type)
    if cells is not None:
        cells = np.atleast_1d(cells)
        if cells.dtype == bool:
            cells = np.where(cells)[0]
        df = df.filter(pl.col('cell_idx').is_in(cells.astype(int).tolist()))
    return df


def _available_dates(records: pl.DataFrame, trial_type: str) -> list[str]:
    """Sorted list of dates for which `trial_type` has any records."""
    dates = (records.filter(pl.col('trial_type') == trial_type)
                    ['date'].unique().to_list())
    return sorted(dates)


# ANALYSIS 1+2: ACROSS DAYS, SAME TRIAL TYPE

def classify_across_days(
    records: pl.DataFrame,
    trial_type: str,
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    tolerance_mode: str = 'fixed_cm',
    dates: list[str] | None = None,
    cells: np.ndarray | None = None,
) -> dict[str, dict]:
    """
    Classify cell-level field changes across consecutive day pairs.

    Within each pair: A = day_a (EARLIER), B = day_b (LATER). Shifts are
    signed (b - a) so positive means the field moved forward on the later day.

    Args:
        records: field_records frame spanning the days/trial types of interest.
        trial_type: Trial type to track (e.g. 'ABC').
        stable_threshold_cm: See match_records().
        match_max_cm: See match_records().
        tolerance_mode: See match_records().
        dates: Restrict to these dates only (default: all dates that contain
            the trial type, in sorted order). Use to pass pre-only or post-
            only ranges.
        cells: Optional cell-index subset.

    Returns:
        Dict keyed by '<day_a>_vs_<day_b>'. Each value is itself a dict:
            'day_a':         str, the earlier day's date.
            'day_b':         str, the later day's date.
            'cell_changes':  list[CellChange]
            'match_result':  raw output of match_records (cell_idx -> tuple)
    """
    # Filter once up front; further per-day filters are cheap on the result.
    records = _filter_records(records, trial_type=trial_type, cells=cells)

    if dates is None:
        dates = _available_dates(records, trial_type)
    else:
        dates = sorted(dates)
    if len(dates) < 2:
        raise ValueError(
            f"Need ≥2 days with {trial_type}, got {len(dates)} after filtering."
        )

    out: dict[str, dict] = {}
    for i in range(len(dates) - 1):
        day_a, day_b = dates[i], dates[i + 1]
        rec_a = _filter_records(records, date=day_a)
        rec_b = _filter_records(records, date=day_b)

        match_result = match_records(
            rec_a, rec_b,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm,
            tolerance_mode=tolerance_mode,
        )
        cell_changes = _classify_all(match_result)

        out[f'{day_a}_vs_{day_b}'] = {
            'day_a': day_a,
            'day_b': day_b,
            'cell_changes': cell_changes,
            'match_result': match_result,
        }
    return out


# ANALYSIS 3: PRE-VS-POST EXTENSION

def classify_pre_vs_post_extension(
    records: pl.DataFrame,
    introduction_day: str,
    trial_type: str = 'ABC',
    pool: str = 'last_pre_vs_first_post',
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    tolerance_mode: str = 'fixed_cm',
    cells: np.ndarray | None = None,
) -> dict:
    """
    Compare base-track field structure pre- vs post-extension introduction.

    A = the LAST pre-extension day, B = the FIRST post-extension day. A
    positive shift_cm means the field moved forward AFTER the extension.

    Args:
        records: field_records frame spanning all days of interest.
        introduction_day: First session that includes the extension. Pre-
            extension days are strictly before this date; post-extension days
            are this date and after.
        trial_type: Base trial type (e.g. 'ABC').
        pool: 'last_pre_vs_first_post' (only mode currently supported).
        Others: see match_records().

    Returns:
        Dict with keys 'day_a', 'day_b', 'cell_changes', 'match_result'.
        'cell_changes' is None if no valid pre/post day pair exists.
    """
    records = _filter_records(records, trial_type=trial_type, cells=cells)
    valid_dates = _available_dates(records, trial_type)
    pre_dates = [d for d in valid_dates if d < introduction_day]
    post_dates = [d for d in valid_dates if d >= introduction_day]

    if not pre_dates or not post_dates:
        return {'day_a': None, 'day_b': None,
                'cell_changes': None, 'match_result': None}

    if pool == 'last_pre_vs_first_post':
        day_a = pre_dates[-1]
        day_b = post_dates[0]
    else:
        raise ValueError(f"Unsupported pool mode: {pool!r}")

    rec_a = _filter_records(records, date=day_a)
    rec_b = _filter_records(records, date=day_b)

    match_result = match_records(
        rec_a, rec_b,
        stable_threshold_cm=stable_threshold_cm,
        match_max_cm=match_max_cm,
        tolerance_mode=tolerance_mode,
    )
    cell_changes = _classify_all(match_result)

    return {'day_a': day_a, 'day_b': day_b,
            'cell_changes': cell_changes, 'match_result': match_result}


# ANALYSIS 4: BETWEEN TRACKS, WITHIN SESSION (DUAL MATCHING)

def _build_field_outcomes(
    records_a: pl.DataFrame,
    records_b: pl.DataFrame,
    pos_match: dict[int, tuple[list[FieldMatch], list[str], list[str]]],
    cue_match: dict[int, list[tuple[str, str, str]]],
    condition_a: str,
    condition_b: str,
) -> list[FieldOutcome]:
    """
    For each record (A and B side), derive its dual-matcher tag and partner info.
    """
    # Build fast partner lookups: (cell_idx, field_id) -> partner info under
    # each matcher.
    pos_a_to_b: dict[tuple[int, str], tuple[str, str, float]] = {}
    pos_b_to_a: dict[tuple[int, str], tuple[str, str, float]] = {}
    for c, (matches, _, _) in pos_match.items():
        for m in matches:
            pos_a_to_b[(c, m.a_field_id)] = (m.b_field_id, m.status, m.shift_cm)
            pos_b_to_a[(c, m.b_field_id)] = (m.a_field_id, m.status, -m.shift_cm)

    cue_a_to_b: dict[tuple[int, str], str] = {}
    cue_b_to_a: dict[tuple[int, str], str] = {}
    for c, pairs in cue_match.items():
        for a_fid, b_fid, _zone in pairs:
            cue_a_to_b[(c, a_fid)] = b_fid
            cue_b_to_a[(c, b_fid)] = a_fid

    def _tag(pos_partner, cue_partner) -> str:
        """Combine the two matchers' results into one of six tags."""
        if pos_partner is None and cue_partner is None:
            return 'track_specific'
        if pos_partner is None and cue_partner is not None:
            return 'cue_stable_only'
        if pos_partner is not None and cue_partner is None:
            _, status, _ = pos_partner
            return ('position_stable_only' if status == 'stable'
                    else 'position_shifted_only')
        # Both matchers found a partner.
        b_fid_pos = pos_partner[0]
        b_fid_cue = cue_partner
        status = pos_partner[1]
        if b_fid_pos == b_fid_cue:
            return ('position_and_cue_stable' if status == 'stable'
                    else 'cue_stable_position_shifted')
        # Disagree (rare): tag according to cue match.
        return 'cue_stable_only'

    outcomes: list[FieldOutcome] = []

    def _walk(records: pl.DataFrame, condition: str,
              pos_lookup, cue_lookup) -> None:
        # Pull columns once — avoids per-row Python overhead from iter_rows.
        cells = records['cell_idx'].to_numpy()
        fids = records['field_id'].to_numpy()
        pos = records['phys_cm_center'].to_numpy()
        zones = records['primary_zone'].to_numpy()

        for k in range(records.height):
            cell_idx = int(cells[k])
            fid = str(fids[k])
            pos_partner = pos_lookup.get((cell_idx, fid))
            cue_partner = cue_lookup.get((cell_idx, fid))
            outcomes.append(FieldOutcome(
                cell_idx=cell_idx,
                condition=condition,
                field_id=fid,
                peak_position_cm=float(pos[k]),
                zone=str(zones[k]),
                tag=_tag(pos_partner, cue_partner),
                partner_position_field_id=(pos_partner[0] if pos_partner else None),
                partner_cue_field_id=cue_partner,
                partner_position_shift_cm=(pos_partner[2] if pos_partner else float('nan')),
            ))

    _walk(records_a, condition_a, pos_a_to_b, cue_a_to_b)
    _walk(records_b, condition_b, pos_b_to_a, cue_b_to_a)
    return outcomes


def classify_between_tracks(
    records: pl.DataFrame,
    date: str,
    condition_a: str = 'ABC',
    condition_b: str = 'ABDC',
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    tolerance_mode: str = 'fixed_cm',
    cells: np.ndarray | None = None,
) -> dict:
    """
    Within-session between-tracks classification (analysis 4).

    A = condition_a (default 'ABC'), B = condition_b (default 'ABDC'). Both
    come from the same session (cell IDs are session-local; no cross-day
    registration needed because A and B are simultaneous).

    Runs both matchers (position + cue identity), tags each field, and
    derives per-cell categories from the position matcher.

    Note: per-cell category uses the POSITION matcher only, for consistency
    with analyses 1-3. A cell whose only "match" is C↔C' (cue-stable but
    position-shifted beyond match_max_cm) will be categorized as 'recruited'
    or 'lost' based on position alone — its FieldOutcomes will reveal the
    cue-stable relationship. Inspect both outputs together.

    Args:
        records: field_records frame spanning the session of interest.
        date: Session date to use (filters records to this date).
        condition_a, condition_b: Trial type names.
        Others: see match_records().

    Returns:
        Dict with keys 'condition_a', 'condition_b', 'cell_changes',
        'field_outcomes', 'match_position', 'match_cue'.
    """
    rec_day = _filter_records(records, date=date, cells=cells)
    rec_a = _filter_records(rec_day, trial_type=condition_a)
    rec_b = _filter_records(rec_day, trial_type=condition_b)

    pos_match = match_records(
        rec_a, rec_b,
        stable_threshold_cm=stable_threshold_cm,
        match_max_cm=match_max_cm,
        tolerance_mode=tolerance_mode,
    )
    cell_changes = _classify_all(pos_match)

    cue_match = match_records_by_cue(rec_a, rec_b)

    field_outcomes = _build_field_outcomes(
        rec_a, rec_b, pos_match, cue_match, condition_a, condition_b,
    )

    return {
        'condition_a': condition_a,
        'condition_b': condition_b,
        'cell_changes': cell_changes,
        'field_outcomes': field_outcomes,
        'match_position': pos_match,
        'match_cue': cue_match,
    }


# POLARS CONVERSION

def cell_changes_to_df(
    changes: list[CellChange],
    extra_cols: dict | None = None,
) -> pl.DataFrame:
    """
    One row per cell. Match details are summarized; full match objects are
    not embedded (use the raw list for that).

    Schema:
        cell_idx (Int64)
        n_fields_a (Int64), n_fields_b (Int64)
        n_matches (Int64), n_stable (Int64), n_shifted (Int64)
        n_unmatched_a (Int64), n_unmatched_b (Int64)
        category (Utf8), presence (Utf8)
        max_shift_cm (Float64), mean_shift_cm (Float64)
        + any (col_name, scalar) pairs in extra_cols.
    """
    rows = []
    for ch in changes:
        n_stable = sum(1 for m in ch.matches if m.status == 'stable')
        n_shifted = sum(1 for m in ch.matches if m.status == 'shifted')
        rows.append({
            'cell_idx': ch.cell_idx,
            'n_fields_a': ch.n_fields_a,
            'n_fields_b': ch.n_fields_b,
            'n_matches': len(ch.matches),
            'n_stable': n_stable,
            'n_shifted': n_shifted,
            'n_unmatched_a': len(ch.unmatched_a_field_ids),
            'n_unmatched_b': len(ch.unmatched_b_field_ids),
            'category': ch.category,
            'presence': ch.presence,
            'max_shift_cm': ch.max_shift_cm,
            'mean_shift_cm': ch.mean_shift_cm,
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        'cell_idx': pl.Int64, 'n_fields_a': pl.Int64, 'n_fields_b': pl.Int64,
        'n_matches': pl.Int64, 'n_stable': pl.Int64, 'n_shifted': pl.Int64,
        'n_unmatched_a': pl.Int64, 'n_unmatched_b': pl.Int64,
        'category': pl.String, 'presence': pl.String,
        'max_shift_cm': pl.Float64, 'mean_shift_cm': pl.Float64,
    })
    if extra_cols:
        df = df.with_columns([pl.lit(v).alias(k) for k, v in extra_cols.items()])
    return df


def field_outcomes_to_df(
    outcomes: list[FieldOutcome],
    extra_cols: dict | None = None,
) -> pl.DataFrame:
    """One row per FieldOutcome (one row per detected field across both
    conditions in analysis 4)."""
    rows = []
    for o in outcomes:
        rows.append({
            'cell_idx': o.cell_idx,
            'condition': o.condition,
            'field_id': o.field_id,
            'peak_position_cm': o.peak_position_cm,
            'zone': o.zone,
            'tag': o.tag,
            'partner_position_field_id': o.partner_position_field_id,
            'partner_cue_field_id': o.partner_cue_field_id,
            'partner_position_shift_cm': o.partner_position_shift_cm,
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        'cell_idx': pl.Int64, 'condition': pl.String, 'field_id': pl.String,
        'peak_position_cm': pl.Float64, 'zone': pl.String, 'tag': pl.String,
        'partner_position_field_id': pl.String,
        'partner_cue_field_id': pl.String,
        'partner_position_shift_cm': pl.Float64,
    })
    if extra_cols:
        df = df.with_columns([pl.lit(v).alias(k) for k, v in extra_cols.items()])
    return df


# CROSS-TABULATION

def crosstab_categories(
    changes_a: list[CellChange],
    changes_b: list[CellChange],
    label_a: str = 'A',
    label_b: str = 'B',
    use: str = 'category',
) -> pl.DataFrame:
    """
    Joint distribution of categories across two analyses, on shared cells.

    Cells in the intersection of the two cell ID sets are counted; cells in
    only one analysis are dropped.

    Args:
        changes_a: First analysis's CellChange list.
        changes_b: Second analysis's CellChange list.
        label_a, label_b: Column-name suffixes.
        use: 'category' or 'presence'.

    Returns:
        Long-format Polars DataFrame with columns
        (<use>_<label_a>, <use>_<label_b>, count).
    """
    if use not in ('category', 'presence'):
        raise ValueError(f"use must be 'category' or 'presence', got {use!r}")

    map_a = {ch.cell_idx: getattr(ch, use) for ch in changes_a}
    map_b = {ch.cell_idx: getattr(ch, use) for ch in changes_b}
    shared = set(map_a) & set(map_b)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in shared:
        counts[(map_a[c], map_b[c])] += 1

    col_a = f'{use}_{label_a}'
    col_b = f'{use}_{label_b}'
    rows = [{col_a: a, col_b: b, 'count': n} for (a, b), n in counts.items()]
    if not rows:
        return pl.DataFrame(schema={col_a: pl.String, col_b: pl.String,
                                    'count': pl.Int64})
    return (pl.DataFrame(rows)
            .sort(['count', col_a, col_b], descending=[True, False, False]))


def multi_crosstab(
    results: dict,
    base_trial_type: str = 'ABC',
    extension_trial_type: str = 'ABDC',
) -> dict[str, pl.DataFrame]:
    """
    Run a battery of preset cross-tabs from a run_full_change_analysis output.

    Built-in cross-tabs (see legacy docstring for full descriptions):
        'last_pre_drift_vs_first_post_drift'
        'last_pre_drift_vs_pre_post'
        'first_post_drift_vs_pre_post'
        'pre_post_vs_intro_between_tracks'
        'last_pre_drift_vs_intro_between_tracks'
    Cross-tabs that can't be computed (e.g. not enough pre-days) are absent.
    """
    out: dict[str, pl.DataFrame] = {}

    pre_drift = results.get('pre_drift', {}) or {}
    post_drift_base = results.get('post_drift_base', {}) or {}
    pre_vs_post = results.get('pre_vs_post', {}) or {}
    between_tracks = results.get('between_tracks', {}) or {}

    pre_pair_keys = sorted(pre_drift.keys())
    post_pair_keys = sorted(post_drift_base.keys())

    if pre_pair_keys and post_pair_keys:
        last_pre_key = pre_pair_keys[-1]
        first_post_key = post_pair_keys[0]
        out['last_pre_drift_vs_first_post_drift'] = crosstab_categories(
            pre_drift[last_pre_key]['cell_changes'],
            post_drift_base[first_post_key]['cell_changes'],
            label_a=f'lastPre[{last_pre_key}]',
            label_b=f'firstPost[{first_post_key}]',
        )

    if pre_pair_keys and pre_vs_post.get('cell_changes') is not None:
        last_pre_key = pre_pair_keys[-1]
        out['last_pre_drift_vs_pre_post'] = crosstab_categories(
            pre_drift[last_pre_key]['cell_changes'],
            pre_vs_post['cell_changes'],
            label_a=f'lastPre[{last_pre_key}]',
            label_b='preVsPost',
        )

    if post_pair_keys and pre_vs_post.get('cell_changes') is not None:
        first_post_key = post_pair_keys[0]
        out['first_post_drift_vs_pre_post'] = crosstab_categories(
            post_drift_base[first_post_key]['cell_changes'],
            pre_vs_post['cell_changes'],
            label_a=f'firstPost[{first_post_key}]',
            label_b='preVsPost',
        )

    intro_day = pre_vs_post.get('day_b')
    if (pre_vs_post.get('cell_changes') is not None
            and intro_day is not None
            and intro_day in between_tracks):
        out['pre_post_vs_intro_between_tracks'] = crosstab_categories(
            pre_vs_post['cell_changes'],
            between_tracks[intro_day]['cell_changes'],
            label_a='preVsPost',
            label_b=f'btTracks[{intro_day}]',
        )

    if (pre_pair_keys
            and intro_day is not None
            and intro_day in between_tracks):
        last_pre_key = pre_pair_keys[-1]
        out['last_pre_drift_vs_intro_between_tracks'] = crosstab_categories(
            pre_drift[last_pre_key]['cell_changes'],
            between_tracks[intro_day]['cell_changes'],
            label_a=f'lastPre[{last_pre_key}]',
            label_b=f'btTracks[{intro_day}]',
        )

    return out


# DIAGNOSTICS

def cell_change_summary(changes: list[CellChange], name: str = '') -> str:
    """Multi-line text summary of category and presence distributions."""
    if not changes:
        return f"[{name}] No CellChange entries."

    n = len(changes)
    lines = [f"[{name}] {n} cells classified."]

    cat_counts: dict[str, int] = defaultdict(int)
    for ch in changes:
        cat_counts[ch.category] += 1
    lines.append('  Categories:')
    for cat in ['non_field', 'recruited', 'lost', 'stable', 'shifted',
                'field_added', 'field_lost', 'complex']:
        c = cat_counts.get(cat, 0)
        if c:
            lines.append(f'    {cat:12s}: {c:5d}  ({100 * c / n:5.1f}%)')

    pres_counts: dict[str, int] = defaultdict(int)
    for ch in changes:
        pres_counts[ch.presence] += 1
    lines.append('  Presence:')
    for pres in ['neither', 'a_only', 'b_only', 'both']:
        c = pres_counts.get(pres, 0)
        if c:
            lines.append(f'    {pres:8s}: {c:5d}  ({100 * c / n:5.1f}%)')

    fields_a = [ch.n_fields_a for ch in changes if ch.n_fields_a > 0]
    fields_b = [ch.n_fields_b for ch in changes if ch.n_fields_b > 0]
    if fields_a:
        lines.append(
            f'  A-side: {sum(fields_a)} fields across {len(fields_a)} cells '
            f'(mean {np.mean(fields_a):.2f} fields/cell)'
        )
    if fields_b:
        lines.append(
            f'  B-side: {sum(fields_b)} fields across {len(fields_b)} cells '
            f'(mean {np.mean(fields_b):.2f} fields/cell)'
        )

    shifts = [abs(m.shift_cm) for ch in changes for m in ch.matches]
    if shifts:
        s = np.array(shifts)
        lines.append(
            f'  |shift_cm| over {len(s)} matches: '
            f'median={np.median(s):.1f}, '
            f'p90={np.percentile(s, 90):.1f}, '
            f'max={s.max():.1f}'
        )

    return '\n'.join(lines)


# PLOTTING

# Color palettes and ordering — consistent with
# multiday_place_field_comparison.plot_recruitment_categories.
_CATEGORY_ORDER: list[str] = [
    'stable', 'shifted', 'field_added', 'field_lost', 'complex',
    'recruited', 'lost', 'non_field',
]

_CATEGORY_COLORS: dict[str, str] = {
    'stable':      '#59A14F',  # green
    'shifted':     '#EDC948',  # gold
    'field_added': '#B07AA1',  # purple
    'field_lost':  '#FF9DA7',  # pink
    'complex':     '#9C755F',  # brown
    'recruited':   '#4E79A7',  # blue
    'lost':        '#E15759',  # red
    'non_field':   '#FFFFFF',  # white (invisible in stack)
}

_TAG_ORDER: list[str] = [
    'position_and_cue_stable',
    'cue_stable_position_shifted',
    'cue_stable_only',
    'position_stable_only',
    'position_shifted_only',
    'track_specific',
]

_TAG_COLORS: dict[str, str] = {
    'position_and_cue_stable':     '#59A14F',  # green   — fully stable
    'cue_stable_position_shifted': '#8CD17D',  # light green
    'cue_stable_only':             '#4E79A7',  # blue    — the C↔C' case
    'position_stable_only':        '#B07AA1',  # purple  — CSCG-violator
    'position_shifted_only':       '#EDC948',  # gold
    'track_specific':              '#E15759',  # red     — genuinely new
}


def _count_categories(changes: list[CellChange]) -> dict[str, int]:
    """Count per-category cell totals; missing categories get 0."""
    counts = {c: 0 for c in _CATEGORY_ORDER}
    for ch in changes:
        counts[ch.category] = counts.get(ch.category, 0) + 1
    return counts


def plot_category_distribution(
    changes_by_label: dict[str, list[CellChange]],
    normalize: bool = False,
    introduction_day: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 5),
    title: str = '',
    legend: bool = True,
    show: bool = True,
) -> Figure:
    """Stacked bar chart of cell-category distribution across analyses."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.figure

    labels = list(changes_by_label.keys())
    if not labels:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    counts_per_label = [_count_categories(changes_by_label[l]) for l in labels]
    totals = [sum(c.values()) or 1 for c in counts_per_label]

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for cat in _CATEGORY_ORDER:
        vals = np.array([c.get(cat, 0) for c in counts_per_label], dtype=float)
        if normalize:
            vals = vals / np.array(totals)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom,
               color=_CATEGORY_COLORS.get(cat, '#cccccc'),
               edgecolor='none', label=cat)
        bottom += vals

    # Format `date_a_vs_date_b` labels onto two lines for legibility.
    bar_labels = [l.replace('_vs_', '\nto\n') for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Fraction of cells' if normalize else 'Cell count',
                  fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Mark the bar that crosses the introduction boundary.
    if introduction_day is not None:
        # Bar i is `day_a_vs_day_b`; mark when day_b == introduction_day.
        for i, l in enumerate(labels):
            if '_vs_' in l and l.split('_vs_')[1] == introduction_day:
                ax.axvline(i - 0.5, color='red', linestyle='--', linewidth=1.5,
                           alpha=0.7, label='extension introduced')
                break

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='upper left',
                  bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_shift_distributions(
    changes_by_label: dict[str, list[CellChange]],
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    bins: int = 30,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    title: str = '',
    show: bool = True,
) -> Figure:
    """Overlaid |shift_cm| histograms for matched fields across analyses."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.figure

    series = {}
    for label, changes in changes_by_label.items():
        shifts = [abs(m.shift_cm) for ch in changes for m in ch.matches]
        if shifts:
            series[label] = np.array(shifts)

    if not series:
        ax.text(0.5, 0.5, 'No matched fields to histogram',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    all_vals = np.concatenate(list(series.values()))
    upper = float(min(max(all_vals.max() * 1.05, match_max_cm * 1.1),
                      max(match_max_cm * 2.0, all_vals.max() * 1.05)))
    edges = np.linspace(0, upper, bins + 1)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    for i, (label, vals) in enumerate(series.items()):
        ax.hist(vals, bins=edges, alpha=0.5,
                color=colors[i % len(colors)],
                label=f'{label}  (n={len(vals)}, median={np.median(vals):.1f}cm)',
                density=True)

    ax.axvline(stable_threshold_cm, color='black', linestyle='--',
               linewidth=1, alpha=0.7,
               label=f'stable_thr ({stable_threshold_cm:.0f}cm)')
    ax.axvline(match_max_cm, color='grey', linestyle=':',
               linewidth=1, alpha=0.7,
               label=f'match_max ({match_max_cm:.0f}cm)')

    ax.set_xlabel('|shift_cm| between matched fields', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_field_tags_by_zone(
    field_outcomes: list[FieldOutcome],
    config: dict,
    condition_a: str = 'ABC',
    condition_b: str = 'ABDC',
    normalize: bool = False,
    figsize: tuple[float, float] = (12, 5),
    title: str = '',
    show: bool = True,
) -> Figure:
    """Per-condition stacked bar of field-tag counts by cue zone (analysis 4)."""
    # Local import to avoid pulling field_records into module load if unused.
    from field_records import get_zone_extents

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150,
                             sharey=normalize)

    for ax, condition in zip(axes, [condition_a, condition_b]):
        # Zone order from config (left-to-right physical order).
        zone_extents = get_zone_extents(config, condition)
        zone_labels_in_order: list[str] = []
        for z in zone_extents['zone_id'].to_list():
            if z not in zone_labels_in_order:
                zone_labels_in_order.append(z)

        cond_outcomes = [o for o in field_outcomes if o.condition == condition]

        counts: dict[str, dict[str, int]] = {
            z: {t: 0 for t in _TAG_ORDER} for z in zone_labels_in_order
        }
        for o in cond_outcomes:
            if o.zone not in counts:
                # Unexpected zone label — append at right edge for visibility.
                counts[o.zone] = {t: 0 for t in _TAG_ORDER}
                if o.zone not in zone_labels_in_order:
                    zone_labels_in_order.append(o.zone)
            counts[o.zone][o.tag] = counts[o.zone].get(o.tag, 0) + 1

        x = np.arange(len(zone_labels_in_order))
        bottom = np.zeros(len(zone_labels_in_order))
        totals = np.array([sum(counts[z].values())
                           for z in zone_labels_in_order], dtype=float)
        totals_safe = np.where(totals == 0, 1, totals)

        for tag in _TAG_ORDER:
            vals = np.array([counts[z].get(tag, 0)
                             for z in zone_labels_in_order], dtype=float)
            if normalize:
                vals = vals / totals_safe
            if vals.sum() == 0:
                continue
            ax.bar(x, vals, bottom=bottom,
                   color=_TAG_COLORS.get(tag, '#cccccc'),
                   edgecolor='none', label=tag)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels_in_order, fontsize=9)
        ax.set_xlabel('Cue zone', fontsize=10)
        ax.set_ylabel('Fraction of fields' if normalize else 'Field count',
                      fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(condition, fontsize=12, fontweight='bold')

        y_max = bottom.max() if bottom.size else 1.0
        if y_max == 0:
            y_max = 1.0
        for i, n in enumerate(totals):
            ax.text(x[i], y_max * 1.02, f'n={int(n)}',
                    ha='center', va='bottom', fontsize=8, color='#444')
        ax.set_ylim(0, y_max * 1.12)

    handles, lbls = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc='center left',
                   bbox_to_anchor=(0.92, 0.5), fontsize=8, frameon=False,
                   title='Field tag')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(rect=(0, 0, 0.91, 1))
    if show:
        plt.show()
    return fig


def plot_crosstab_heatmap(
    ct_df: pl.DataFrame,
    title: str = '',
    figsize: tuple[float, float] = (6, 5),
    cmap: str = 'magma_r',
    annotate: bool = True,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> Figure:
    """Heatmap rendering of a crosstab DataFrame from crosstab_categories()."""
    cols = ct_df.columns
    if 'count' not in cols or len(cols) != 3:
        raise ValueError(
            "ct_df must have columns (axis_a, axis_b, count); got " f"{cols}"
        )
    axis_cols = [c for c in cols if c != 'count']
    col_a, col_b = axis_cols[0], axis_cols[1]

    def _sort_key(cats: list[str]) -> list[str]:
        ordered = [c for c in _CATEGORY_ORDER if c in cats]
        leftover = sorted(c for c in cats if c not in _CATEGORY_ORDER)
        return ordered + leftover

    rows_uniq = _sort_key(ct_df[col_a].unique().to_list())
    cols_uniq = _sort_key(ct_df[col_b].unique().to_list())

    mat = np.zeros((len(rows_uniq), len(cols_uniq)), dtype=int)
    row_to_i = {r: i for i, r in enumerate(rows_uniq)}
    col_to_j = {c: j for j, c in enumerate(cols_uniq)}
    for row in ct_df.iter_rows(named=True):
        mat[row_to_i[row[col_a]], col_to_j[row[col_b]]] = row['count']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.figure

    im = ax.imshow(mat, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(cols_uniq)))
    ax.set_xticklabels(cols_uniq, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(rows_uniq)))
    ax.set_yticklabels(rows_uniq, fontsize=8)
    ax.set_xlabel(col_b, fontsize=10)
    ax.set_ylabel(col_a, fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    if annotate:
        thr = 0.5 * mat.max() if mat.max() > 0 else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v == 0:
                    continue
                color = 'white' if v >= thr else 'black'
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Cell count', shrink=0.85)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ORCHESTRATOR

def run_full_change_analysis(
    records: pl.DataFrame,
    config: dict,
    introduction_day: str,
    base_trial_type: str = 'ABC',
    extension_trial_type: str = 'ABDC',
    stable_threshold_cm: float = 15.0,
    match_max_cm: float = 30.0,
    tolerance_mode: str = 'fixed_cm',
    cells: np.ndarray | None = None,
    verbose: bool = True,
    make_figures: bool = True,
    show: bool = False,
    save_dir: str | None = None,
) -> dict:
    """
    Run all four analyses + cross-tabs + figures, return one results dict.

    Now takes a `records` DataFrame as the data layer (build it with
    field_records.build_field_records on a MultidayPlaceFieldResult).

    See module docstring + legacy `run_full_change_analysis` docstring for
    the full results-dict shape. Keys: 'pre_drift', 'post_drift_base',
    'post_drift_ext', 'pre_vs_post', 'between_tracks', 'timeline_base',
    'timeline_ext', 'crosstabs', 'figures'.
    """
    out: dict = {}
    figs: dict[str, Figure] = {}

    # Filter once up front; per-analysis filters are downstream of this.
    records = _filter_records(records, cells=cells)

    base_dates = _available_dates(records, base_trial_type)
    ext_dates = _available_dates(records, extension_trial_type)

    # ANALYSIS 1: pre-extension drift on base track
    pre_dates = [d for d in base_dates if d < introduction_day]
    if len(pre_dates) >= 2:
        out['pre_drift'] = classify_across_days(
            records, base_trial_type, dates=pre_dates,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
        if verbose:
            print(f"\n=== Analysis 1: pre-extension drift "
                  f"({base_trial_type}, {len(pre_dates)} days, "
                  f"{len(out['pre_drift'])} pair(s)) ===")
            for k, v in out['pre_drift'].items():
                cats = _count_categories(v['cell_changes'])
                total = sum(cats.values())
                stable_frac = (cats.get('stable', 0) / total) if total else 0
                print(f"  {k}: n={total} cells, stable={100*stable_frac:.1f}%")
    else:
        out['pre_drift'] = {}
        if verbose:
            print(f"\n[skip] Analysis 1: only {len(pre_dates)} pre-extension day(s).")

    # ANALYSIS 2: post-extension drift, separately for base and extension.
    post_dates_base = [d for d in base_dates if d >= introduction_day]
    if len(post_dates_base) >= 2:
        out['post_drift_base'] = classify_across_days(
            records, base_trial_type, dates=post_dates_base,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
        if verbose:
            print(f"\n=== Analysis 2a: post-extension drift on base "
                  f"({base_trial_type}, {len(post_dates_base)} days) ===")
            for k, v in out['post_drift_base'].items():
                cats = _count_categories(v['cell_changes'])
                total = sum(cats.values())
                stable_frac = (cats.get('stable', 0) / total) if total else 0
                print(f"  {k}: n={total} cells, stable={100*stable_frac:.1f}%")
    else:
        out['post_drift_base'] = {}

    post_dates_ext = [d for d in ext_dates if d >= introduction_day]
    if len(post_dates_ext) >= 2:
        out['post_drift_ext'] = classify_across_days(
            records, extension_trial_type, dates=post_dates_ext,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
        if verbose:
            print(f"\n=== Analysis 2b: post-extension drift on extension "
                  f"({extension_trial_type}, {len(post_dates_ext)} days) ===")
            for k, v in out['post_drift_ext'].items():
                cats = _count_categories(v['cell_changes'])
                total = sum(cats.values())
                stable_frac = (cats.get('stable', 0) / total) if total else 0
                print(f"  {k}: n={total} cells, stable={100*stable_frac:.1f}%")
    else:
        out['post_drift_ext'] = {}

    # ANALYSIS 3: pre-vs-post on base track
    out['pre_vs_post'] = classify_pre_vs_post_extension(
        records, introduction_day=introduction_day,
        trial_type=base_trial_type,
        stable_threshold_cm=stable_threshold_cm,
        match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
    )
    if verbose and out['pre_vs_post']['cell_changes'] is not None:
        cats = _count_categories(out['pre_vs_post']['cell_changes'])
        total = sum(cats.values())
        print(f"\n=== Analysis 3: pre-vs-post on {base_trial_type} ===")
        print(f"  {out['pre_vs_post']['day_a']} -> "
              f"{out['pre_vs_post']['day_b']}: n={total} cells, "
              f"stable={100*cats.get('stable', 0)/max(total, 1):.1f}%, "
              f"shifted={100*cats.get('shifted', 0)/max(total, 1):.1f}%, "
              f"recruited={100*cats.get('recruited', 0)/max(total, 1):.1f}%, "
              f"lost={100*cats.get('lost', 0)/max(total, 1):.1f}%")

    # ANALYSIS 4: between tracks, per session that has both conditions
    out['between_tracks'] = {}
    # Only days with both conditions.
    both_dates = sorted(set(base_dates) & set(ext_dates))
    for date in both_dates:
        bt = classify_between_tracks(
            records, date=date,
            condition_a=base_trial_type, condition_b=extension_trial_type,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
        out['between_tracks'][date] = bt
        if verbose:
            cats = _count_categories(bt['cell_changes'])
            total = sum(cats.values())
            tag_counts: dict[str, int] = defaultdict(int)
            for fo in bt['field_outcomes']:
                tag_counts[fo.tag] += 1
            tag_total = sum(tag_counts.values())
            print(f"\n=== Analysis 4: {date} between {base_trial_type} "
                  f"vs {extension_trial_type} ===")
            print(f"  Cells: n={total}, "
                  f"recruited={100*cats.get('recruited',0)/max(total,1):.1f}%, "
                  f"lost={100*cats.get('lost',0)/max(total,1):.1f}%")
            print(f"  Fields: n={tag_total}, "
                  f"track_specific={100*tag_counts.get('track_specific',0)/max(tag_total,1):.1f}%, "
                  f"cue_stable_only={100*tag_counts.get('cue_stable_only',0)/max(tag_total,1):.1f}%")

    # FULL-EXPERIMENT TIMELINE
    if len(base_dates) >= 2:
        out['timeline_base'] = classify_across_days(
            records, base_trial_type, dates=base_dates,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
    else:
        out['timeline_base'] = {}

    if len(ext_dates) >= 2:
        out['timeline_ext'] = classify_across_days(
            records, extension_trial_type, dates=ext_dates,
            stable_threshold_cm=stable_threshold_cm,
            match_max_cm=match_max_cm, tolerance_mode=tolerance_mode,
        )
    else:
        out['timeline_ext'] = {}

    # CROSS-TABS
    out['crosstabs'] = multi_crosstab(
        out, base_trial_type=base_trial_type,
        extension_trial_type=extension_trial_type,
    )
    if verbose:
        print(f"\n=== Cross-tabs: {len(out['crosstabs'])} preset comparisons ===")
        for name in out['crosstabs']:
            print(f"  {name}")

    # FIGURES
    if make_figures:
        from pathlib import Path
        sd = Path(save_dir) if save_dir is not None else None
        if sd is not None:
            sd.mkdir(parents=True, exist_ok=True)

        def _finalize(name: str, fig: Figure) -> None:
            figs[name] = fig
            if sd is not None:
                path = sd / f'cell_change_{name}.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                if verbose:
                    print(f"  Saved figure: {path}")
            if show:
                if verbose:
                    print(f"  Showing: {name}  (close window to continue, Ctrl-C to abort)")
                plt.show()
                plt.close(fig)

        if out['timeline_base']:
            _finalize(
                f'recruitment_timeline_{base_trial_type}',
                plot_category_distribution(
                    {k: v['cell_changes'] for k, v in out['timeline_base'].items()},
                    normalize=False, introduction_day=introduction_day,
                    title=f'Cell recruitment across days — {base_trial_type}',
                    show=False,
                ),
            )
        if out['timeline_ext']:
            _finalize(
                f'recruitment_timeline_{extension_trial_type}',
                plot_category_distribution(
                    {k: v['cell_changes'] for k, v in out['timeline_ext'].items()},
                    normalize=False, introduction_day=introduction_day,
                    title=f'Cell recruitment across days — {extension_trial_type}',
                    show=False,
                ),
            )

        if out['pre_vs_post']['cell_changes'] is not None:
            comparison: dict[str, list[CellChange]] = {}
            pooled: list[CellChange] = []
            if out['pre_drift']:
                for v in out['pre_drift'].values():
                    pooled.extend(v['cell_changes'])
                comparison['pre_drift_pooled'] = pooled
            comparison['pre_vs_post'] = out['pre_vs_post']['cell_changes']
            _finalize(
                'pre_post_summary',
                plot_category_distribution(
                    comparison, normalize=True,
                    title=f'Pre-vs-post extension on {base_trial_type} (vs pre-drift baseline)',
                    show=False,
                ),
            )

            shift_lists: dict[str, list[CellChange]] = {}
            if pooled:
                shift_lists['pre_drift_pooled'] = pooled
            shift_lists['pre_vs_post'] = out['pre_vs_post']['cell_changes']
            _finalize(
                'shift_distribution',
                plot_shift_distributions(
                    shift_lists,
                    stable_threshold_cm=stable_threshold_cm,
                    match_max_cm=match_max_cm,
                    title='|shift_cm|: extension introduction vs baseline drift',
                    show=False,
                ),
            )

        if out['between_tracks']:
            _finalize(
                'between_tracks_categories',
                plot_category_distribution(
                    {date: v['cell_changes']
                     for date, v in sorted(out['between_tracks'].items())},
                    normalize=True,
                    title=f'Between-tracks ({base_trial_type} vs {extension_trial_type}) across days',
                    show=False,
                ),
            )
            intro_bt = (out['between_tracks'].get(introduction_day)
                        or next(iter(out['between_tracks'].values())))
            _finalize(
                'between_tracks_field_tags',
                plot_field_tags_by_zone(
                    intro_bt['field_outcomes'], config,
                    condition_a=base_trial_type,
                    condition_b=extension_trial_type,
                    normalize=False,
                    title='Field-tag distribution by zone (intro day)',
                    show=False,
                ),
            )

        for name, ct_df in out['crosstabs'].items():
            if ct_df.height == 0:
                continue
            _finalize(
                f'crosstab_{name}',
                plot_crosstab_heatmap(
                    ct_df, title=name, figsize=(7, 5.5), show=False,
                ),
            )

        out['figures'] = figs

    return out


if __name__ == '__main__':
    from pathlib import Path
    import polars as pl
    from df_processing import load_multiday_sessions, get_bin_size
    from place_field_detection import detect_place_fields, DetectionParams
    from field_records import build_field_records_single_session

    mouse_id = '26'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)
    save_dir = Path(f'/Users/cs963/Desktop/sun_lab_outputs/cell_change/{mouse_id}')

    # ---- Load multiday sessions ----
    sessions = load_multiday_sessions(
        mouse_dir, date_range=('2025-09-02', '2025-09-10'), auto_process=False,
    )

    # ---- Detect place fields per session, build records as we go ----
    # (Use the same DetectionParams across days for comparable results.)
    params = DetectionParams(smooth_sigma=0, signal_threshold=0.3)
    record_frames: list[pl.DataFrame] = []
    bin_size_cm: float | None = None
    config = None
    for date in sorted(sessions):
        s = sessions[date]
        bin_size_cm = get_bin_size(s['metadata'])
        config = s['config']  # assume the config is the same across days
        pf = detect_place_fields(
            s['data'], s['config'],
            signal_col='multi_day_dff',
            bin_size_cm=bin_size_cm, params=params,
        )
        record_frames.append(build_field_records_single_session(
            pf, config, bin_size_cm=bin_size_cm, date=date,
        ))
    records = pl.concat(record_frames)
    print(f'Built records frame: {records.height} fields across '
          f'{records["date"].n_unique()} dates, '
          f'{records["cell_idx"].n_unique()} cells.')

    # ---- Run all four analyses + cross-tabs + figures in one call ----
    # `show` and `save_dir` are independent toggles:
    #   show=True             -> pop up figures interactively (good for iterating)
    #   save_dir='/path/dir/' -> write each figure to PNG (good for batch runs)
    # Combine, do one, or do neither — figures are still in results['figures'].
    results = run_full_change_analysis(
        records, config,
        introduction_day='2025-09-08',  # first ABDC day
        base_trial_type='ABC',
        extension_trial_type='ABDC',
        stable_threshold_cm=15.0,
        match_max_cm=30.0,
        verbose=True,
        make_figures=True,
        show=True,                       # display interactively while iterating
        save_dir=None,                   # set to a path once figures look right
    )

    # ---- Inspect the standard cross-tab DataFrames (multi_crosstab) ----
    print('\n=== Cross-tab DataFrames (preset pairs) ===')
    for name, ct_df in results['crosstabs'].items():
        print(f'\n[{name}]')
        print(ct_df)

    # ---- Build Polars DataFrames for ad-hoc downstream stats ----
    # Pool every pre-drift pair into one tagged DataFrame.
    pre_dfs = []
    for pair_key, pair in results.get('pre_drift', {}).items():
        df = cell_changes_to_df(
            pair['cell_changes'],
            extra_cols={'pair': pair_key, 'analysis': 'pre_drift'},
        )
        pre_dfs.append(df)
    pre_drift_df = pl.concat(pre_dfs) if pre_dfs else pl.DataFrame()
    print('\n=== Pre-drift cell-change DataFrame (head) ===')
    print(pre_drift_df.head(10))

    # Between-tracks per-field outcomes for the introduction day, grouped by
    # zone and tag — directly answers "where do new fields appear, and which
    # fields are cue-anchored?".
    intro_day = '2025-09-08'
    if intro_day in results['between_tracks']:
        bt = results['between_tracks'][intro_day]
        fo_df = field_outcomes_to_df(
            bt['field_outcomes'],
            extra_cols={'date': intro_day},
        )
        print(f'\n=== Between-tracks field outcomes ({intro_day}), '
              f'by condition/zone/tag ===')
        print(fo_df.group_by(['condition', 'zone', 'tag']).len()
                   .sort(['condition', 'zone', 'tag']))

    # ---- Show the figure dict ----
    figs = results.get('figures', {})
    print(f'\nFigures generated: {len(figs)}')
    for name in figs:
        print(f'  {name}')