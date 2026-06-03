"""
Place Cell Remapping — Single Session ABC vs ABDC Analysis
==========================================================

Tests whether place cells are position-locked, cue-locked, or remapped between
the ABC and ABDC trial types in a session where both trial types occur.

Track design
------------
ABC  : [A, 0a, B, 0b, C, 0c]            180 cm (6 segments × 30 cm)
ABDC : [A, 0a, B, 0b, D, 0d, C, 0c]     240 cm (D inserted between B and C)

Because D is *inserted* (not substituted), the same absolute cm in ABC and
ABDC can hold different cues:

    abs cm    ABC cue   ABDC cue
    ---------------------------
     0–60     A, 0a     A, 0a       (shared)
    60–120    B, 0b     B, 0b       (shared)
    120–180   C, 0c     D, 0d       (cue changes at same cm)
    180–240   —         C, 0c       (C moved 60 cm later)

This dissociates absolute-position coding from cue-identity coding: a cell
that "follows C" must fire 60 cm later in ABDC than in ABC.

Two summary views
-----------------
1. Correlation view (`compute_remapping_scores` → `classify_cells`):
   per-cell Pearson r between ABC and ABDC tuning curves under TWO
   alignments — absolute-position and cue-identity. Collapses each cell to
   a single class.

2. Field view (`compute_field_summary`): detects every place field in each
   trial type and matches them by position and/or cue. Produces a per-field
   match_type (`kept`, `split`, `gained`, …) and a per-cell `field_outcome`.

Why both views
--------------
Correlation summarises but flattens multi-field cells: a cell that splits a
single ABC C field into a position copy (at 130 cm) AND a cue copy (at
190 cm) in ABDC has moderate-to-low values on both correlations and ends up
"ambiguous" — even though it cleanly demonstrates dual coding. Field
matching catches this as `split`, the most informative category for the
ABC-vs-ABDC question.

Classification (correlation view)
---------------------------------
    invariant        : both corrs high, similar values (A/B cells, trivial)
    position_locked  : corr_position >> corr_cue
                       e.g. corr_pos=0.7, corr_cue=0.1 — fires at the same
                       cm regardless of cue identity
    cue_locked       : corr_cue >> corr_position
                       e.g. corr_pos=0.1, corr_cue=0.7 — follows C/0c to
                       its new location 60 cm later
    remapped         : both corrs low — full reorganisation
    gained_in_ABDC   : place cell only in ABDC
    lost_in_ABDC    : place cell only in ABC
    ambiguous        : everything else (incl. dual-coding split cells)

Field match types (field view; see `compute_field_summary` for full detail)
-------------------------------------------------------------------------
    kept             : same cm AND same cue between trial types
    position_locked  : same cm only (cue changed)
    cue_locked       : same cue only (cm changed)
    split            : ONE ABC field → TWO ABDC fields (one at same cm,
                       one at same cue) — dual coding revealed
    lost / gained    : field present in only one trial type

Outputs
-------
- {mouse}_{date}_remapping_cells.csv  — per-cell DF (correlation +
  field-match counts + field_outcome)
- {mouse}_{date}_remapping_fields.csv  — per-field DF with field_id,
  match_type, partner_field_ids
- 7 figures (scatter, examples per class, correlation×field-outcome
  heatmap, gained-by-cue, ABC-fate-by-cue, top splits, strong splits).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from df_processing import (
    find_session_dir,
    get_bin_size,
    get_session_paths,
    load_processed_session,
    load_session_context,
)
from matplotlib.colors import LinearSegmentedColormap

from place_field_detection import (
    DetectionParams,
    PlaceFieldResult,
    detect_place_fields,
)


# ----------------------------------------------------------------------
# Style — dusty/muted-earth palette and helpers
# ----------------------------------------------------------------------

# Categorical palette (muted earth + complements). Chosen so that each
# category is distinguishable without any colour being harsh on the eye.
#   clay        warm tan-brown      A-region / position coding
#   sage        muted green         success / preservation / 'kept'
#   slate       grey-blue           cue coding / cool counterpoint to clay
#   dusty rose  desaturated red     loss / 'remapped'
#   mauve       desaturated violet  loss-of-cell / 'lost_in_ABDC'
#   taupe       warm grey           neutral / invariant
#   teal        muted blue-green    novelty / 'gained'
#   ochre       muted mustard       (reserved)
#   light-taupe pale warm grey      ambiguous

# Dusty versions of real colors — recognizably red / teal / blue / green
# at lower saturation. Target HSL ≈ (varied hue, S 35–50%, L 60–68%) —
# muted enough to read as a palette, saturated enough to not look pastel.
CLAY        = '#C9956A'   # dusty caramel
SAGE        = '#94B894'   # dusty sage green
SLATE       = '#809AB6'   # dusty cornflower blue
DUSTY_ROSE  = '#C97A7E'   # dusty rose-red
MAUVE       = '#A98AAE'   # dusty plum
TAUPE       = '#A89C8C'   # warm dusty grey (anchor / neutral lines)
TEAL        = '#7FB0AB'   # dusty teal
OCHRE       = '#D4B361'   # dusty mustard (used for `shifted`)
LIGHT_TAUPE = '#C9BFB1'   # pale dusty neutral (ambiguous category)
SOFT_CLAY   = '#DCB088'   # lighter caramel (split_position)
SOFT_SLATE  = '#A8BCD0'   # lighter cornflower (split_cue)
TERRACOTTA  = '#CC8669'   # dusty terracotta (ABDC trace; distinct from clay)

# Diverging dusty colormap, vlag-styled (slate → cream → rose).
# Used vmin=0 so count heatmaps render in the warm half only.
VLAG_DUSTY = LinearSegmentedColormap.from_list(
    'vlag_dusty',
    [SLATE, SOFT_SLATE, '#EFE6DA', SOFT_CLAY, DUSTY_ROSE],
)

# Neutral ink colours and grid colour used across paper-style plots.
_INK       = '#222222'
_INK_MUTED = '#555555'
_INK_SOFT  = '#888888'
_GRID      = '#E7E2D8'


def _paper_style():
    """rc_context for journal-quality plots (clean spines, muted ink).

    Use as ``with _paper_style(): fig, ax = plt.subplots(...)`` inside
    every plotting function.
    """
    return plt.rc_context({
        'font.family':         'sans-serif',
        'font.sans-serif':     ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':            9.5,
        'axes.titlesize':       11.5,
        'axes.titleweight':    'semibold',
        'axes.labelsize':       10,
        'axes.labelcolor':     _INK,
        'axes.edgecolor':      _INK_MUTED,
        'axes.linewidth':       0.8,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.titlecolor':     _INK,
        'xtick.color':         _INK_MUTED,
        'ytick.color':         _INK_MUTED,
        'xtick.labelsize':      9,
        'ytick.labelsize':      9,
        'xtick.major.size':     0,
        'ytick.major.size':     3,
        'legend.frameon':       False,
        'legend.fontsize':      8.5,
        'legend.handlelength':  1.2,
        'legend.handleheight':  1.1,
        'legend.handletextpad': 0.55,
        'legend.columnspacing': 1.4,
        'figure.facecolor':    'white',
        'axes.facecolor':      'white',
        'savefig.facecolor':   'white',
        'savefig.dpi':          200,
    })


def _add_subtitle(ax, subtitle: str | None) -> None:
    """Muted-grey subtitle just above the axes title (left-aligned)."""
    if not subtitle:
        return
    ax.text(
        0, 1.02, subtitle,
        transform=ax.transAxes,
        fontsize=8.5, color=_INK_SOFT, ha='left', va='bottom',
    )


def _build_subtitle(*parts: str | None) -> str | None:
    """Join non-empty parts with a middle-dot separator for plot subtitles."""
    parts_clean = [str(p) for p in parts if p is not None]
    return '   ·   '.join(parts_clean) if parts_clean else None


def _title(
    base: str,
    mouse_id: str | None = None,
    date: str | None = None,
) -> str:
    """Format a plot title with a 'Mouse {id} · {date}' prefix when given.

    Use in every plot function so multi-session callers can attach the
    correct animal/date by passing kwargs; single-session callers can also
    omit them and just pass `base`.

    Example
    -------
    >>> _title('ABC vs ABDC remapping', mouse_id='26', date='2025-09-16')
    'Mouse 26 · 2025-09-16 — ABC vs ABDC remapping'
    """
    parts = []
    if mouse_id is not None:
        parts.append(f'Mouse {mouse_id}')
    if date is not None:
        parts.append(str(date))
    prefix = ' · '.join(parts)
    return f'{prefix} — {base}' if prefix else base


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _find_trial_type(result: PlaceFieldResult, name: str):
    """Find the dict key in result.fields matching name, whether str or enum."""
    for tt in result.fields:
        if tt == name or str(tt) == name:
            return tt
        if hasattr(tt, 'value') and str(tt.value) == name:
            return tt
        if hasattr(tt, 'name') and tt.name == name:
            return tt
    raise KeyError(
        f"No trial type matching '{name}'. Available: {list(result.fields)}"
    )


def _bin_cue_ids(
    df: pl.DataFrame,
    trial_type,
    n_bins: int,
) -> list[str | None]:
    """Dominant frame-level cue_id per spatial bin.

    Uses the frame-level cue_id column (which has unique values per gray zone,
    e.g. '0a', '0b') rather than the config's cue_sequence (which may use a
    shared id like 'gray' for all gray zones).

    Returns a list of length n_bins; entries are None for bins with no frames.
    """
    # Trial type may be a string or enum — cast for robust comparison
    sub = df.filter(pl.col('trial_type').cast(pl.String) == str(trial_type))
    sub = sub.select(['distance_bin', 'cue_id']).drop_nulls()
    if sub.height == 0:
        return [None] * n_bins

    bin_modes = (
        sub.group_by('distance_bin')
        .agg(pl.col('cue_id').mode().first().alias('cue_id'))
        .sort('distance_bin')
    )
    bin_to_cue = dict(zip(
        bin_modes['distance_bin'].to_list(),
        bin_modes['cue_id'].to_list(),
    ))
    return [bin_to_cue.get(i) for i in range(n_bins)]


def _build_cue_aligned_indices(
    abc_bin_cues: list[str | None],
    abdc_bin_cues: list[str | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-index arrays for ABC and ABDC that align on shared frame-level cue_id.

    Bins whose cue_id appears in only one trial type (e.g., D or 0d in ABDC)
    are dropped. Bins are paired by order of appearance: the k-th occurrence
    of cue X in ABDC pairs with the k-th occurrence of cue X in ABC.
    """
    abc_by_cue: dict[str, list[int]] = {}
    for i, c in enumerate(abc_bin_cues):
        if c is not None:
            abc_by_cue.setdefault(c, []).append(i)

    abdc_by_cue: dict[str, list[int]] = {}
    for i, c in enumerate(abdc_bin_cues):
        if c is not None:
            abdc_by_cue.setdefault(c, []).append(i)

    # Iterate in ABDC bin order so the aligned vectors mirror traversal order
    shared = sorted(
        set(abc_by_cue) & set(abdc_by_cue),
        key=lambda c: abdc_by_cue[c][0],
    )

    abc_idx: list[int] = []
    abdc_idx: list[int] = []
    for c in shared:
        a, d = abc_by_cue[c], abdc_by_cue[c]
        n = min(len(a), len(d))
        abc_idx.extend(a[:n])
        abdc_idx.extend(d[:n])

    return np.array(abc_idx, dtype=int), np.array(abdc_idx, dtype=int)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN handling. Returns NaN if either input is flat."""
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x, y = x[mask], y[mask]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _peak_info(
    tc: np.ndarray,
    bin_cues: list[str | None],
    bin_size_cm: float,
) -> tuple[float, str | None]:
    """Peak position (cm, bin center) and the cue_id at that bin."""
    if np.all(np.isnan(tc)) or np.nanmax(tc) == np.nanmin(tc):
        return float('nan'), None
    peak_bin = int(np.nanargmax(tc))
    peak_cm = (peak_bin + 0.5) * bin_size_cm
    peak_cue = bin_cues[peak_bin] if peak_bin < len(bin_cues) else None
    return peak_cm, peak_cue


def _per_cell_field_count(pf) -> np.ndarray:
    """Number of detected fields per cell, shape (n_cells,)."""
    n_cells = pf.binF.shape[0]
    counts = np.zeros(n_cells, dtype=int)
    if pf.n_fields == 0:
        return counts
    ids, c = np.unique(pf.cell_id, return_counts=True)
    counts[ids] = c
    return counts


# ----------------------------------------------------------------------
# Core scoring
# ----------------------------------------------------------------------

def compute_remapping_scores(
    result: PlaceFieldResult,
    df: pl.DataFrame,
    bin_size_cm: float,
    abc_name: str = 'ABC',
    abdc_name: str = 'ABDC',
) -> pl.DataFrame:
    """Per-cell ABC↔ABDC tuning-curve correlations under two alignments.

    What
    ----
    For every cell that has at least one place field in either trial type
    (`is_place_cell_any`), compute two Pearson r values between its ABC and
    ABDC binned tuning curves:

        corr_position : ABC vs the FIRST n_bins_ABC bins of ABDC
                        (absolute-position alignment). A cell with
                        corr_position ≈ 1 fires at the same absolute cm in
                        both trial types.
        corr_cue      : ABC vs ABDC after dropping ABDC-only bins (the
                        D / 0d insertion) and pairing remaining bins by
                        frame-level cue_id. A cell with corr_cue ≈ 1 fires
                        at whichever absolute cm its preferred cue moved to.

    Why two alignments
    ------------------
    The ABDC track *inserts* D between B and C, so C-tuned cells now fire
    60 cm later than they did in ABC. Absolute-position alignment treats
    this as remapping; cue-identity alignment recovers the C-tuning. The
    two correlations together dissociate position-coding from cue-coding
    populations.

    Why frame-level cue_id (not config sequence)
    --------------------------------------------
    The config's `cue_sequence` may use a single token (e.g. 'gray') for
    every gray zone, which would conflate the four ABDC gray zones with the
    three ABC ones. The frame-level `cue_id` column resolves each gray zone
    uniquely ('0a', '0b', '0c', '0d'), so the alignment pairs the matching
    zones across trial types.

    Examples
    --------
    Tuning curve interpretation for a hypothetical cell:

    - Fires at 60 cm (B) in both trial types:
        corr_position ≈ high, corr_cue ≈ high  →  invariant
    - Fires at 130 cm in both trial types
      (which is cue C in ABC but cue D in ABDC):
        corr_position ≈ high, corr_cue ≈ low   →  position_locked
    - Fires at 130 cm in ABC (cue C) and at 190 cm in ABDC
      (cue C, now 60 cm later):
        corr_position ≈ low,  corr_cue ≈ high  →  cue_locked

    Parameters
    ----------
    result : PlaceFieldResult
        Output of `detect_place_fields`; supplies `binF` (n_cells × n_bins)
        per trial type and `is_place_cell_any`.
    df : pl.DataFrame
        Processed-session frame-level data. Must include `trial_type`,
        `distance_bin`, and `cue_id` columns.
    bin_size_cm : float
        Spatial bin width; used to convert peak bins → cm.
    abc_name, abdc_name : str
        Trial-type names in `result.fields` (str or enum-castable).

    Returns
    -------
    pl.DataFrame
        One row per place cell with columns:
            cell_id, n_fields_ABC, n_fields_ABDC,
            peak_pos_ABC, peak_pos_ABDC, peak_cue_ABC, peak_cue_ABDC,
            corr_position, corr_cue.
    """
    abc_tt = _find_trial_type(result, abc_name)
    abdc_tt = _find_trial_type(result, abdc_name)

    abc_pf = result.fields[abc_tt]
    abdc_pf = result.fields[abdc_tt]
    abc_tc = abc_pf.binF       # (n_cells, n_bins_abc)
    abdc_tc = abdc_pf.binF     # (n_cells, n_bins_abdc)
    n_bins_abc = abc_tc.shape[1]
    n_bins_abdc = abdc_tc.shape[1]

    abc_bin_cues = _bin_cue_ids(df, abc_tt, n_bins_abc)
    abdc_bin_cues = _bin_cue_ids(df, abdc_tt, n_bins_abdc)
    abc_align_idx, abdc_align_idx = _build_cue_aligned_indices(
        abc_bin_cues, abdc_bin_cues,
    )

    n_fields_abc = _per_cell_field_count(abc_pf)
    n_fields_abdc = _per_cell_field_count(abdc_pf)
    cell_idxs = np.where(result.is_place_cell_any)[0]

    rows = []
    for c in cell_idxs:
        tc_a = abc_tc[c]
        tc_d = abdc_tc[c]

        corr_pos = _safe_corr(tc_a, tc_d[:n_bins_abc])
        corr_cue = _safe_corr(tc_a[abc_align_idx], tc_d[abdc_align_idx])

        peak_pos_a, peak_cue_a = _peak_info(tc_a, abc_bin_cues, bin_size_cm)
        peak_pos_d, peak_cue_d = _peak_info(tc_d, abdc_bin_cues, bin_size_cm)

        rows.append({
            'cell_id': int(c),
            'n_fields_ABC': int(n_fields_abc[c]),
            'n_fields_ABDC': int(n_fields_abdc[c]),
            'peak_pos_ABC': peak_pos_a,
            'peak_pos_ABDC': peak_pos_d,
            'peak_cue_ABC': peak_cue_a,
            'peak_cue_ABDC': peak_cue_d,
            'corr_position': corr_pos,
            'corr_cue': corr_cue,
        })

    return pl.DataFrame(rows)


def classify_cells(
    df: pl.DataFrame,
    high_thresh: float = 0.5,
    diff_thresh: float = 0.2,
    low_thresh: float = 0.3,
) -> pl.DataFrame:
    """Assign each cell a single class from its (corr_position, corr_cue).

    What
    ----
    Adds a `classification` column. Rules are applied in order (first
    matching rule wins):

        gained_in_ABDC   : ABC has 0 fields, ABDC has ≥1
        lost_in_ABDC     : ABDC has 0 fields, ABC has ≥1
        ambiguous        : either correlation is NaN (flat TC, too few
                           valid bins, etc.)
        invariant        : corr_pos > high_thresh AND corr_cue > high_thresh
                           AND |corr_pos − corr_cue| < diff_thresh
                           Example: corr_pos=0.85, corr_cue=0.83
        position_locked  : corr_pos − corr_cue > diff_thresh
                           Example: corr_pos=0.70, corr_cue=0.10
        cue_locked       : corr_cue − corr_pos > diff_thresh
                           Example: corr_pos=0.10, corr_cue=0.70
        remapped         : corr_pos < low_thresh AND corr_cue < low_thresh
                           Example: corr_pos=0.05, corr_cue=0.10
        ambiguous        : anything else (moderate, undifferentiated)

    Why a single label
    ------------------
    The two correlations give a 2D summary, but downstream comparisons
    (counts per class, group stats, cohort plots) want one categorical
    label. The defaults give a coarse but interpretable split; tighten
    `high_thresh` and loosen `diff_thresh` to be stricter about calling
    cells locked.

    Limitation
    ----------
    This per-cell summary collapses multi-field cells. A cell whose two
    ABC fields are (kept, split) cannot be cleanly labelled here — see
    `compute_field_summary` for the field-level view.

    Parameters
    ----------
    df : pl.DataFrame
        Output of `compute_remapping_scores`.
    high_thresh : float, default 0.5
        Lower bound for calling a correlation "high".
    diff_thresh : float, default 0.2
        Minimum difference between corr_position and corr_cue to assign a
        locked class.
    low_thresh : float, default 0.3
        Upper bound for calling both correlations "low" (→ remapped).

    Returns
    -------
    pl.DataFrame
        Input with an added `classification` column (String).
    """
    def _row_class(row: dict) -> str:
        if row['n_fields_ABC'] == 0 and row['n_fields_ABDC'] > 0:
            return 'gained_in_ABDC'
        if row['n_fields_ABDC'] == 0 and row['n_fields_ABC'] > 0:
            return 'lost_in_ABDC'

        cp, cc = row['corr_position'], row['corr_cue']
        if cp is None or cc is None or np.isnan(cp) or np.isnan(cc):
            return 'ambiguous'
        if cp > high_thresh and cc > high_thresh and abs(cp - cc) < diff_thresh:
            return 'invariant'
        if cp - cc > diff_thresh:
            return 'position_locked'
        if cc - cp > diff_thresh:
            return 'cue_locked'
        if cp < low_thresh and cc < low_thresh:
            return 'remapped'
        return 'ambiguous'

    classes = [_row_class(r) for r in df.iter_rows(named=True)]
    return df.with_columns(pl.Series('classification', classes))


# ----------------------------------------------------------------------
# Field-level matching
# ----------------------------------------------------------------------

@dataclass
class FieldInfo:
    cell_id: int
    field_idx: int           # 0-based within cell, sorted by position
    center_cm: float
    center_bin: int
    cue_id: str | None
    trial_type: str          # 'ABC' or 'ABDC'


@dataclass
class FieldMatch:
    cell_id: int
    abc_field: FieldInfo | None
    abdc_partners: list[FieldInfo]   # 0, 1, or 2 (split case)
    match_type: str          # 'kept', 'position_locked', 'cue_locked',
                             # 'split', 'lost', 'gained'


def _extract_fields(
    pf,
    bin_cues: list[str | None],
    bin_size_cm: float,
    trial_type_name: str,
) -> dict[int, list[FieldInfo]]:
    """Per-cell list of FieldInfo, sorted by position within each cell."""
    fields_by_cell: dict[int, list[FieldInfo]] = {}
    if pf.n_fields == 0:
        return fields_by_cell

    # pf.centers is (n_fields, 2): [cell_idx, position_cm]
    for cell_idx, pos_cm in pf.centers:
        cell_idx = int(cell_idx)
        center_bin = int(pos_cm / bin_size_cm)
        center_bin = max(0, min(center_bin, len(bin_cues) - 1))
        cue_id = bin_cues[center_bin] if bin_cues else None
        fields_by_cell.setdefault(cell_idx, []).append(FieldInfo(
            cell_id=cell_idx,
            field_idx=0,
            center_cm=float(pos_cm),
            center_bin=center_bin,
            cue_id=cue_id,
            trial_type=trial_type_name,
        ))

    for fields in fields_by_cell.values():
        fields.sort(key=lambda f: f.center_cm)
        for i, f in enumerate(fields):
            f.field_idx = i

    return fields_by_cell


def _match_cell_fields(
    abc_fields: list[FieldInfo],
    abdc_fields: list[FieldInfo],
    position_tol_cm: float,
) -> list[FieldMatch]:
    """Match each ABC field to ABDC fields by position and/or cue.

    Algorithm
    ---------
    For each ABC field (iterated in ABC's position order):
      1. position-candidates : unconsumed ABDC fields within
         `position_tol_cm` of this ABC field's center.
         position-best = closest of those (None if no candidate).
      2. cue-candidates : unconsumed ABDC fields with the same cue_id
         (any distance).
         cue-best = the one whose center is closest in cm
         (None if no candidate).
      3. Resolve and consume:
           position-best and cue-best are the SAME field
               → 'kept'             (consume 1 ABDC field)
           position-best and cue-best are DIFFERENT fields
               → 'split'            (consume BOTH: the position-half AND
                                     the cue-half)
           only position-best exists
               → 'position_locked'  (consume 1)
           only cue-best exists
               → 'cue_locked'       (consume 1)
           neither exists
               → 'lost'             (consume 0)

    Unmatched ABDC fields become 'gained' matches at the end.

    Examples
    --------
    - ABC cell with single field at 15 cm (A region), ABDC cell with
      single field at 15 cm (A region):
          'kept'.
    - ABC cell with single field at 130 cm (C region), ABDC cell with
      single field at 130 cm (now D region) — position-match exists,
      cue-match does not (no C field in ABDC at all):
          'position_locked'.
    - ABC cell with single field at 130 cm (C region), ABDC cell with
      single field at 190 cm (C region, now 60 cm later) — cue-match
      exists at 190 cm, position-match exists if 190 cm is within
      position_tol_cm of 130 cm (it isn't, at tol=15), so only cue:
          'cue_locked'.
    - ABC cell with single field at 130 cm (C region), ABDC cell with
      TWO fields, one at 130 cm (D region) and one at 190 cm (C region) —
      position-best=130 cm field, cue-best=190 cm field, different:
          'split' (consumes both ABDC fields).

    Greedy, not globally optimal
    ----------------------------
    Matching is greedy by ABC's position order, so once an ABDC field is
    consumed a later ABC field cannot claim it. Pathological cases (two
    ABC fields competing for the same ABDC partner) are rare under
    `position_tol_cm`=15 cm given typical field spacing, but be aware.

    Why allow 'split' to consume TWO ABDC fields
    --------------------------------------------
    The biological question is whether a single ABC field spawns BOTH a
    position copy AND a cue copy in ABDC. Forcing 1-to-1 matching would
    misclassify this as either position_locked or cue_locked depending on
    which won — losing the most informative category for the experiment.

    Returns
    -------
    list[FieldMatch]
        One entry per ABC field, plus one entry per unmatched ABDC field
        (as 'gained').
    """
    matches: list[FieldMatch] = []
    consumed: set[int] = set()

    for abc_f in abc_fields:
        available = [i for i in range(len(abdc_fields)) if i not in consumed]

        pos_candidates = [
            i for i in available
            if abs(abdc_fields[i].center_cm - abc_f.center_cm) <= position_tol_cm
        ]
        pos_best = (
            min(pos_candidates,
                key=lambda i: abs(abdc_fields[i].center_cm - abc_f.center_cm))
            if pos_candidates else None
        )

        cue_candidates = [
            i for i in available
            if abc_f.cue_id is not None
            and abdc_fields[i].cue_id == abc_f.cue_id
        ]
        cue_best = (
            min(cue_candidates,
                key=lambda i: abs(abdc_fields[i].center_cm - abc_f.center_cm))
            if cue_candidates else None
        )

        if pos_best is not None and cue_best is not None:
            if pos_best == cue_best:
                match_type = 'kept'
                partners = [abdc_fields[pos_best]]
                consumed.add(pos_best)
            else:
                match_type = 'split'
                partners = [abdc_fields[pos_best], abdc_fields[cue_best]]
                consumed.update([pos_best, cue_best])
        elif pos_best is not None:
            match_type = 'position_locked'
            partners = [abdc_fields[pos_best]]
            consumed.add(pos_best)
        elif cue_best is not None:
            match_type = 'cue_locked'
            partners = [abdc_fields[cue_best]]
            consumed.add(cue_best)
        else:
            match_type = 'lost'
            partners = []

        matches.append(FieldMatch(
            cell_id=abc_f.cell_id,
            abc_field=abc_f,
            abdc_partners=partners,
            match_type=match_type,
        ))

    for i, d in enumerate(abdc_fields):
        if i not in consumed:
            matches.append(FieldMatch(
                cell_id=d.cell_id,
                abc_field=None,
                abdc_partners=[d],
                match_type='gained',
            ))

    return matches


def compute_field_summary(
    result: PlaceFieldResult,
    df: pl.DataFrame,
    bin_size_cm: float,
    position_tol_cm: float = 15.0,
    abc_name: str = 'ABC',
    abdc_name: str = 'ABDC',
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-cell and per-field summary of ABC↔ABDC field outcomes.

    What
    ----
    Detects every place field in each trial type, then for each ABC field
    finds its ABDC partner(s) by:
      1. position : closest ABDC field within `position_tol_cm`.
      2. cue      : closest ABDC field with the same cue_id (any distance).

    Returns two DataFrames:
      - `cell_df`  : one row per place cell with counts of each match_type
                     plus a single `field_outcome` label.
      - `field_df` : one row per detected field (BOTH ABC and ABDC sides)
                     with field_id, match_type, and partner_field_ids.

    Match types (field-level)
    -------------------------
    ABC-side `match_type` (one per ABC field):

        kept             : the ABC field has a single ABDC partner that is
                           BOTH the position match AND the cue match.
                           Example: ABC field at 15 cm (cue A) -> ABDC
                           field at 15 cm (cue A). Both criteria point to
                           the same ABDC field.
        position_locked  : matched by position only (cue identity differs).
                           Example: ABC field at 130 cm (cue C) -> ABDC
                           field at 130 cm (cue D). The cell tracks
                           absolute position; its cue has changed beneath
                           it.
        cue_locked       : matched by cue only (position differs).
                           Example: ABC field at 130 cm (cue C) -> ABDC
                           field at 190 cm (cue C, now 60 cm later). The
                           cell tracks cue identity; position has shifted.
        split            : matched by position AND cue, but to TWO
                           DIFFERENT ABDC fields.
                           Example: a cell with ONE ABC field at 130 cm
                           (cue C) has, in ABDC, TWO fields - one at
                           130 cm (now cue D, the "position copy") AND a
                           second at 190 cm (cue C, the "cue copy"). The
                           ABC field effectively spawned both a position
                           successor and a cue successor in ABDC. This is
                           the key category for testing dual coding of
                           position and cue identity in the hippocampus.
        lost             : no ABDC partner (no position match, no cue
                           match).

    ABDC-side `match_type` (one per ABDC field):

        kept / position_locked / cue_locked :
                           This ABDC field is the partner that completed
                           the corresponding ABC-side match. Same label as
                           the ABC side it pairs with.
        split_position   : POSITION-half of a `split` ABC field - same cm
                           as the ABC field, different cue (e.g. the
                           130 cm / D successor of an ABC 130 cm / C
                           field).
        split_cue        : CUE-half of a `split` ABC field - same cue as
                           the ABC field, different cm (e.g. the 190 cm /
                           C successor of an ABC 130 cm / C field).
        gained           : no ABC partner. Either a brand-new field, or
                           one outside `position_tol_cm` with a different
                           cue identity. The prototypical 'gained' is a
                           field in the new D / 0d region.

    Cell-level outcome (`field_outcome` in cell_df)
    -----------------------------------------------
        stable             : every ABC field matched, no gained fields.
        stable+gain        : every ABC field matched, >=1 gained.
        partial_loss       : some ABC fields matched, some lost, no
                             gained.
        partial_loss+gain  : matched, lost, AND gained all present.
        complete_loss      : every ABC field lost, no gained (cell
                             silenced in ABDC).
        swap               : every ABC field lost AND >=1 gained (cell
                             remapped to entirely new locations).
        gain_only          : no ABC fields, >=1 ABDC field (place cell
                             only in ABDC).
        loss_only          : >=1 ABC field, no ABDC fields (place cell
                             only in ABC).

    Why field-level matching in addition to correlation
    ---------------------------------------------------
    Correlation collapses the whole tuning curve to one scalar, hiding
    multi-field cells. The 'split' case in particular has moderate-to-low
    values on BOTH alignments (neither captures both peaks at once), so
    `classify_cells` will call it 'ambiguous'. Only the field-level view
    exposes it cleanly. Same for 'partial_loss' - invisible to a single
    correlation.

    Field id and partners
    ---------------------
    Each detected field gets a unique `field_id` of the form
    "{cell_id}_{trial_type}_{field_idx}", where `field_idx` is the
    0-based index of the field within that cell's trial-type fields,
    sorted by position.

    `partner_field_ids` lists the matched fields on the other side:
      kept / position_locked / cue_locked  : 1 partner
      split (ABC-side)                     : 2 partners
      split_position / split_cue           : 1 partner (the source ABC
                                             field's id)
      lost / gained                        : 0 partners

    Parameters
    ----------
    result : PlaceFieldResult
        Output of `detect_place_fields`.
    df : pl.DataFrame
        Processed session data (for frame-level cue_id -> bin mapping).
    bin_size_cm : float
        Spatial bin width.
    position_tol_cm : float, default 15.0
        Max distance (cm) for a position match. Tighter -> more
        'cue_locked' and 'lost'; looser -> more 'kept' and
        'position_locked'.
    abc_name, abdc_name : str
        Trial-type names in `result.fields`.

    Returns
    -------
    (cell_df, field_df) : tuple[pl.DataFrame, pl.DataFrame]
        cell_df  : one row per place cell with field counts +
                   `field_outcome`.
        field_df : one row per detected field with `field_id`,
                   `match_type`, `partner_field_ids` (list[str]),
                   `n_partners`, `trial_type`, `cue_id`, `center_cm`,
                   `center_bin`, `field_idx`.
    """
    abc_tt = _find_trial_type(result, abc_name)
    abdc_tt = _find_trial_type(result, abdc_name)
    abc_pf = result.fields[abc_tt]
    abdc_pf = result.fields[abdc_tt]

    n_bins_abc = abc_pf.binF.shape[1]
    n_bins_abdc = abdc_pf.binF.shape[1]
    abc_bin_cues = _bin_cue_ids(df, abc_tt, n_bins_abc)
    abdc_bin_cues = _bin_cue_ids(df, abdc_tt, n_bins_abdc)

    abc_fields_by_cell = _extract_fields(abc_pf, abc_bin_cues, bin_size_cm, abc_name)
    abdc_fields_by_cell = _extract_fields(abdc_pf, abdc_bin_cues, bin_size_cm, abdc_name)

    cell_idxs = np.where(result.is_place_cell_any)[0]

    def _fid(cell_id: int, tt_name: str, field_idx: int) -> str:
        return f"{cell_id}_{tt_name}_{field_idx}"

    cell_rows: list[dict] = []
    field_rows: list[dict] = []
    for cid in cell_idxs:
        cid = int(cid)
        abc_fields = abc_fields_by_cell.get(cid, [])
        abdc_fields = abdc_fields_by_cell.get(cid, [])
        matches = _match_cell_fields(abc_fields, abdc_fields, position_tol_cm)

        # Each ABDC field's role from its own perspective (set by ABC-side matches).
        # Defaults to 'gained'; overwritten when an ABC field matches it.
        abdc_role: dict[int, tuple[str, str | None]] = {
            d.field_idx: ('gained', None) for d in abdc_fields
        }

        # ABC-side field rows + populate abdc_role
        for m in matches:
            if m.abc_field is None:
                continue
            abc_fid = _fid(cid, abc_name, m.abc_field.field_idx)
            partner_fids = [
                _fid(cid, abdc_name, p.field_idx) for p in m.abdc_partners
            ]
            field_rows.append({
                'field_id': abc_fid,
                'cell_id': cid,
                'trial_type': abc_name,
                'field_idx': m.abc_field.field_idx,
                'center_cm': m.abc_field.center_cm,
                'center_bin': m.abc_field.center_bin,
                'cue_id': m.abc_field.cue_id,
                'match_type': m.match_type,
                'partner_field_ids': partner_fids,
                'n_partners': len(partner_fids),
            })

            if m.match_type == 'split':
                pos_p, cue_p = m.abdc_partners
                abdc_role[pos_p.field_idx] = ('split_position', abc_fid)
                abdc_role[cue_p.field_idx] = ('split_cue', abc_fid)
            elif m.match_type in ('kept', 'position_locked', 'cue_locked'):
                abdc_role[m.abdc_partners[0].field_idx] = (m.match_type, abc_fid)

        # ABDC-side field rows
        for d in abdc_fields:
            role, abc_partner = abdc_role[d.field_idx]
            partner_fids = [abc_partner] if abc_partner is not None else []
            field_rows.append({
                'field_id': _fid(cid, abdc_name, d.field_idx),
                'cell_id': cid,
                'trial_type': abdc_name,
                'field_idx': d.field_idx,
                'center_cm': d.center_cm,
                'center_bin': d.center_bin,
                'cue_id': d.cue_id,
                'match_type': role,
                'partner_field_ids': partner_fids,
                'n_partners': len(partner_fids),
            })

        # Cell-level counts (from ABC-perspective match types)
        counts = {
            'kept': 0, 'position_locked': 0, 'cue_locked': 0,
            'split': 0, 'lost': 0, 'gained': 0,
        }
        for m in matches:
            counts[m.match_type] += 1

        n_abc = len(abc_fields)
        n_abdc = len(abdc_fields)
        n_lost = counts['lost']
        n_gained = counts['gained']
        n_matched = (counts['kept'] + counts['position_locked']
                     + counts['cue_locked'] + counts['split'])

        if n_abc == 0 and n_abdc > 0:
            outcome = 'gain_only'
        elif n_abdc == 0 and n_abc > 0:
            outcome = 'loss_only'
        elif n_lost == 0 and n_gained == 0:
            outcome = 'stable'
        elif n_lost == 0 and n_gained > 0:
            outcome = 'stable+gain'
        elif n_lost > 0 and n_gained == 0 and n_matched > 0:
            outcome = 'partial_loss'
        elif n_lost > 0 and n_gained > 0 and n_matched > 0:
            outcome = 'partial_loss+gain'
        elif n_lost > 0 and n_gained == 0 and n_matched == 0:
            outcome = 'complete_loss'
        elif n_lost > 0 and n_gained > 0 and n_matched == 0:
            outcome = 'swap'
        else:
            outcome = 'other'

        cell_rows.append({
            'cell_id': cid,
            'n_abc_fields': n_abc,
            'n_abdc_fields': n_abdc,
            'n_kept': counts['kept'],
            'n_position_locked': counts['position_locked'],
            'n_cue_locked': counts['cue_locked'],
            'n_split': counts['split'],
            'n_lost': counts['lost'],
            'n_gained': counts['gained'],
            'field_outcome': outcome,
        })

    cell_df = pl.DataFrame(cell_rows)

    field_schema = {
        'field_id': pl.String,
        'cell_id': pl.Int64,
        'trial_type': pl.String,
        'field_idx': pl.Int64,
        'center_cm': pl.Float64,
        'center_bin': pl.Int64,
        'cue_id': pl.String,
        'match_type': pl.String,
        'partner_field_ids': pl.List(pl.String),
        'n_partners': pl.Int64,
    }
    field_df = (
        pl.DataFrame(field_rows, schema=field_schema)
        if field_rows
        else pl.DataFrame(schema=field_schema)
    )
    # Sort: by cell_id, then ABC before ABDC, then by position within cell
    field_df = field_df.sort(['cell_id', 'trial_type', 'center_cm'])

    return cell_df, field_df


def combine_summaries(
    corr_df: pl.DataFrame,
    cell_df: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join field-level cell_df onto correlation-based corr_df by cell_id."""
    drop = [c for c in ('n_abc_fields', 'n_abdc_fields') if c in cell_df.columns]
    return corr_df.join(cell_df.drop(drop), on='cell_id', how='left')


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

CLASS_COLORS = {
    'invariant':       TAUPE,
    'position_locked': CLAY,
    'cue_locked':      SLATE,
    'remapped':        DUSTY_ROSE,
    'gained_in_ABDC':  SAGE,
    'lost_in_ABDC':    MAUVE,
    'ambiguous':       LIGHT_TAUPE,
}


def plot_remapping_scatter(
    df: pl.DataFrame,
    title: str = 'ABC vs ABDC remapping',
    mouse_id: str | None = None,
    date: str | None = None,
) -> plt.Figure:
    """Scatter of corr_position vs corr_cue with marginal histograms.

    What
    ----
    Each point is one place cell, coloured by its `classification` from
    `classify_cells`. Marginal histograms (top, right) show the
    distributions of corr_position and corr_cue across all cells.

    How to read it
    --------------
    - Diagonal (corr_position == corr_cue): cells equally well-explained
      by both alignments. Top-right corner = invariant cells.
    - Below-diagonal, lower-right: position_locked (high corr_pos, low
      corr_cue). Cells anchored to absolute cm.
    - Above-diagonal, upper-left: cue_locked (low corr_pos, high
      corr_cue). Cells following C/0c to its new position in ABDC.
    - Lower-left corner: remapped (both correlations low). Total
      reorganisation - or split cells the correlation can't capture
      (check `compute_field_summary` for those).

    Parameters
    ----------
    df : pl.DataFrame
        Output of `classify_cells` (must have `corr_position`, `corr_cue`,
        `classification`).
    title : str
        Base title; mouse_id and date are prepended if provided.
    mouse_id, date : str or None
        Animal id and session date for the title prefix.
    """
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.08, wspace=0.08)
    ax_main = fig.add_subplot(gs[1:, :3])
    ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    for cls, color in CLASS_COLORS.items():
        sub = df.filter(pl.col('classification') == cls)
        if sub.height == 0:
            continue
        ax_main.scatter(
            sub['corr_position'].to_numpy(),
            sub['corr_cue'].to_numpy(),
            s=18, c=color, alpha=0.75, edgecolors='none',
            label=f'{cls} (n={sub.height})',
        )

    ax_main.plot([-1, 1], [-1, 1], color=TAUPE, ls='--', lw=0.7, alpha=0.6)
    ax_main.axhline(0, color=TAUPE, lw=0.4, alpha=0.5)
    ax_main.axvline(0, color=TAUPE, lw=0.4, alpha=0.5)
    ax_main.set_xlabel('corr_position (absolute alignment)')
    ax_main.set_ylabel('corr_cue (cue-identity alignment)')
    ax_main.set_xlim(-0.6, 1.05)
    ax_main.set_ylim(-0.6, 1.05)
    ax_main.legend(loc='lower right', fontsize=8, framealpha=0.9)

    cp = df['corr_position'].to_numpy()
    cc = df['corr_cue'].to_numpy()
    cp = cp[np.isfinite(cp)]
    cc = cc[np.isfinite(cc)]
    ax_top.hist(cp, bins=40, color=TAUPE, alpha=0.8)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel('count')
    ax_right.hist(cc, bins=40, orientation='horizontal', color=TAUPE, alpha=0.8)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel('count')

    fig.suptitle(_title(title, mouse_id, date), y=0.995)
    return fig


def plot_example_tuning_curves(
    result: PlaceFieldResult,
    df: pl.DataFrame,
    scores: pl.DataFrame,
    bin_size_cm: float,
    n_examples: int = 6,
    abc_name: str = 'ABC',
    abdc_name: str = 'ABDC',
    classes_to_show: list[str] | None = None,
    seed: int = 0,
    mouse_id: str | None = None,
    date: str | None = None,
) -> plt.Figure:
    """Example cells per classification, plotted in absolute position.

    What
    ----
    Small-multiples grid: rows = classification, columns = randomly chosen
    example cells from that class. Each panel overlays the cell's ABC
    tuning curve (slate) and ABDC tuning curve (terracotta). The ABDC-only
    region (the D / 0d insertion) is shaded so the cell's relation to the
    insertion is obvious at a glance.

    How to read it
    --------------
    - 'invariant' / 'position_locked' rows : ABC and ABDC traces should
      overlap closely (invariant) or have peaks at the same cm but
      different cues underneath (position_locked).
    - 'cue_locked' row : ABDC peak should be shifted right by ~60 cm
      relative to the ABC peak (cell follows C/0c to its new position).
    - 'remapped' / 'ambiguous' rows : traces will look unrelated. Inspect
      visually - some 'ambiguous' cells are real splits.
    - 'gained_in_ABDC' / 'lost_in_ABDC' : one trace will be flat.

    Parameters
    ----------
    result, df, bin_size_cm : as elsewhere.
    scores : pl.DataFrame
        Output of `classify_cells`.
    n_examples : int
        Number of example cells per class (columns in the grid).
    classes_to_show : list[str] or None
        Restrict the rows to a subset. Defaults to all non-ambiguous
        classes. Empty classes are skipped.
    seed : int
        RNG seed for reproducible sampling.
    mouse_id, date : str or None
        Animal id and session date for the title prefix.
    """
    if classes_to_show is None:
        classes_to_show = [
            'invariant', 'position_locked', 'cue_locked',
            'remapped', 'gained_in_ABDC', 'lost_in_ABDC',
        ]
    classes_to_show = [
        c for c in classes_to_show
        if scores.filter(pl.col('classification') == c).height > 0
    ]

    abc_tt = _find_trial_type(result, abc_name)
    abdc_tt = _find_trial_type(result, abdc_name)
    abc_tc = result.fields[abc_tt].binF
    abdc_tc = result.fields[abdc_tt].binF
    n_bins_abc = abc_tc.shape[1]
    n_bins_abdc = abdc_tc.shape[1]
    x_abc = (np.arange(n_bins_abc) + 0.5) * bin_size_cm
    x_abdc = (np.arange(n_bins_abdc) + 0.5) * bin_size_cm

    # Detect ABDC-only bins (the D / 0d insertion) from frame-level cue_ids
    abc_bin_cues = _bin_cue_ids(df, abc_tt, n_bins_abc)
    abdc_bin_cues = _bin_cue_ids(df, abdc_tt, n_bins_abdc)
    abc_cue_set = {c for c in abc_bin_cues if c is not None}
    insertion_bins = [
        i for i, c in enumerate(abdc_bin_cues)
        if c is not None and c not in abc_cue_set
    ]

    # Convert insertion bins to contiguous span(s) in cm for shading
    insertion_spans: list[tuple[float, float]] = []
    if insertion_bins:
        start = insertion_bins[0]
        prev = start
        for b in insertion_bins[1:]:
            if b == prev + 1:
                prev = b
            else:
                insertion_spans.append((start * bin_size_cm, (prev + 1) * bin_size_cm))
                start = b
                prev = b
        insertion_spans.append((start * bin_size_cm, (prev + 1) * bin_size_cm))

    n_classes = len(classes_to_show)
    fig, axes = plt.subplots(
        n_classes, n_examples,
        figsize=(2.3 * n_examples, 1.7 * n_classes),
        sharex=True, squeeze=False,
    )

    rng = np.random.default_rng(seed)
    for row, cls in enumerate(classes_to_show):
        sub = scores.filter(pl.col('classification') == cls)
        cell_ids = sub['cell_id'].to_numpy()
        chosen = rng.choice(
            cell_ids, size=min(n_examples, len(cell_ids)), replace=False,
        )
        for col in range(n_examples):
            ax = axes[row, col]
            if col >= len(chosen):
                ax.axis('off')
                continue
            cid = int(chosen[col])
            for s, e in insertion_spans:
                ax.axvspan(s, e, color=TAUPE, alpha=0.15, lw=0)
            ax.plot(x_abc, abc_tc[cid], color=SLATE, lw=1.4, label='ABC')
            ax.plot(x_abdc, abdc_tc[cid], color=TERRACOTTA, lw=1.4, label='ABDC')
            ax.tick_params(labelsize=7)
            ax.set_title(f'cell {cid}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{cls}\nactivity', fontsize=8)
            if row == 0 and col == n_examples - 1:
                ax.legend(fontsize=7, loc='upper right')

    for col in range(n_examples):
        axes[-1, col].set_xlabel('position (cm)', fontsize=8)
    fig.suptitle(
        _title(
            'Example cells per class - absolute position '
            '(shaded band = ABDC-only bins, i.e. D / 0d insertion)',
            mouse_id, date,
        ),
        y=1.0,
    )
    fig.tight_layout()
    return fig


def plot_outcome_comparison(
    combined: pl.DataFrame,
    mouse_id: str | None = None,
    date: str | None = None,
) -> plt.Figure:
    """Heatmap: correlation classification (rows) vs field outcome (cols).

    What
    ----
    Each cell of the heatmap is the count of place cells with that combo
    of correlation-based class and field-matching outcome. Colours use a
    dusty diverging map (slate -> cream -> rose) with vmin=0 so values
    render only in the warm half.

    How to read it
    --------------
    Diagonal-ish agreement: 'invariant' cells should mostly be 'stable',
    'position_locked' cells should be 'stable' or 'partial_loss', etc.

    The most diagnostic cells are the off-diagonal ones:
    - 'ambiguous' x 'stable+gain' or 'partial_loss+gain' : multi-field
      cells (often splits) that single correlations cannot summarise.
    - 'invariant' x 'stable+gain' : a cell whose existing fields are
      preserved AND that gained a new field (e.g. in the D region).
    - 'remapped' x 'swap' : agreement that the cell is doing something
      different in ABDC.

    Parameters
    ----------
    combined : pl.DataFrame
        Output of `combine_summaries` (must have `classification` and
        `field_outcome`).
    mouse_id, date : str or None
        Animal id and session date for the title prefix.
    """
    class_order = ['invariant', 'position_locked', 'cue_locked', 'remapped',
                   'gained_in_ABDC', 'lost_in_ABDC', 'ambiguous']
    outcome_order = ['stable', 'stable+gain', 'partial_loss', 'partial_loss+gain',
                     'complete_loss', 'swap', 'gain_only', 'loss_only', 'other']

    classes = [c for c in class_order if combined.filter(pl.col('classification') == c).height > 0]
    outcomes = [o for o in outcome_order if combined.filter(pl.col('field_outcome') == o).height > 0]

    mat = np.zeros((len(classes), len(outcomes)), dtype=int)
    for i, c in enumerate(classes):
        for j, o in enumerate(outcomes):
            mat[i, j] = combined.filter(
                (pl.col('classification') == c) & (pl.col('field_outcome') == o)
            ).height

    fig, ax = plt.subplots(figsize=(1.1 * len(outcomes) + 3, 0.6 * len(classes) + 2))
    im = ax.imshow(mat, cmap=VLAG_DUSTY, aspect='auto', vmin=0)
    ax.set_xticks(np.arange(len(outcomes)))
    ax.set_xticklabels(outcomes, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel('field_outcome (field matching)')
    ax.set_ylabel('classification (correlation)')
    ax.set_title(_title(
        'Correlation-based vs field-matching classification',
        mouse_id, date,
    ))

    threshold = mat.max() * 0.55 if mat.max() > 0 else 0
    for i in range(len(classes)):
        for j in range(len(outcomes)):
            v = mat[i, j]
            if v > 0:
                color = 'white' if v > threshold else '#3a3a3a'
                ax.text(j, i, str(v), ha='center', va='center',
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='n cells')
    fig.tight_layout()
    return fig


# Consistent dusty colours for field-level match types - used by every plot
# below. split_position / split_cue are faded variants of position_locked /
# cue_locked so the family relationship is visually obvious.
MATCH_COLOR_MAP = {
    'kept':            SAGE,
    'position_locked': CLAY,
    'cue_locked':      SLATE,
    'split':           MAUVE,
    'split_position':  SOFT_CLAY,
    'split_cue':       SOFT_SLATE,
    'lost':            DUSTY_ROSE,
    'gained':          TEAL,
}


def _abdc_cue_order(field_df: pl.DataFrame, trial_type: str) -> list[str]:
    """Cue ids for a trial type, ordered by mean position along the track."""
    return (
        field_df.filter(pl.col('trial_type') == trial_type)
                .group_by('cue_id')
                .agg(pl.col('center_cm').mean().alias('mean_cm'))
                .sort('mean_cm')['cue_id'].to_list()
    )


def plot_gained_fields_by_cue(
    field_df: pl.DataFrame,
    mouse_id: str | None = None,
    date: str | None = None,
) -> tuple[plt.Figure, pl.DataFrame]:
    """Bar chart: how many `gained` fields appear in each ABDC cue region.

    What
    ----
    Counts ABDC-side fields with `match_type == 'gained'` (no ABC partner),
    grouped by their cue_id, in track order. Bars are annotated with the
    percentage of all gained fields in that cue.

    How to read it
    --------------
    - Concentration in D / 0d : new fields are responses to the novel
      insertion - cells tuned to the new cue/region.
    - Spread across cues : broader remapping, with new fields appearing
      throughout the track.
    - Notable counts in C / 0c : worth a closer look - these might be
      cells that "moved with C" and were unmatched because the ABC C
      field was outside `position_tol_cm` from the new C position.

    Parameters
    ----------
    field_df : pl.DataFrame
        Output of `compute_field_summary` (the per-field DF).
    mouse_id, date : str or None
        Animal id and session date for the title prefix.

    Returns
    -------
    (fig, gained_by_cue) : tuple[plt.Figure, pl.DataFrame]
        Summary DataFrame has columns cue_id, n, pct.
    """
    cue_order = _abdc_cue_order(field_df, 'ABDC')
    gained_by_cue = (
        field_df.filter(pl.col('match_type') == 'gained')
                .group_by('cue_id').agg(pl.len().alias('n'))
                .with_columns(pl.col('cue_id').cast(pl.Enum(cue_order)))
                .sort('cue_id')
                .with_columns(
                    pl.col('cue_id').cast(pl.String),
                    (100 * pl.col('n') / pl.col('n').sum())
                        .round(1).alias('pct'),
                )
    )

    cue_ids = gained_by_cue['cue_id'].to_list()
    ns = gained_by_cue['n'].to_list()
    pcts = gained_by_cue['pct'].to_list()

    with _paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = np.arange(len(cue_ids), dtype=float)
        ax.bar(x, ns, width=0.6, color=MATCH_COLOR_MAP['gained'],
               edgecolor='white', linewidth=0.9, zorder=2)
        for xi, n, p in zip(x, ns, pcts):
            ax.text(xi, n, f'{p}%', ha='center', va='bottom',
                    fontsize=8.5, color=_INK)
        ax.set_xticks(x)
        ax.set_xticklabels(cue_ids, fontsize=10.5, fontweight='semibold',
                           color=_INK)
        ax.set_xlim(-0.6, len(cue_ids) - 0.4)
        ax.set_ylabel('Gained fields  (count)')
        ax.set_xlabel('ABDC cue')
        ax.yaxis.grid(True, color=_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
        ax.margins(y=0.12)
        ax.set_title('Where new fields appear in ABDC', loc='left', pad=22)
        _add_subtitle(ax, _build_subtitle(
            f'Mouse {mouse_id}' if mouse_id else None, date,
        ))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.15)
    return fig, gained_by_cue


def plot_abc_fate_by_cue(
    field_df: pl.DataFrame,
    mouse_id: str | None = None,
    date: str | None = None,
) -> plt.Figure:
    """Stacked bar: fate (match_type) of ABC fields, grouped by source cue.

    What
    ----
    For each ABC cue (A, 0a, B, 0b, C, 0c), stacks the count of ABC-side
    `match_type` values. Cues are ordered along the ABC track. Colours
    follow `MATCH_COLOR_MAP`.

    How to read it
    --------------
    - A and B cues : expect mostly 'kept' (their cues exist at the same
      cm in ABDC, so they should be preserved).
    - C and 0c cues : the most informative. In ABDC, the original C
      position now contains D, so C-region cells must remap (lost, split,
      cue_locked, position_locked). A C bar dominated by 'split' is direct
      evidence of dual coding (position + cue) for that cue.
    - High 'lost' anywhere : cells whose ABC field disappeared - either
      true silencing, or a field that moved more than `position_tol_cm`
      AND took on a different cue.

    Parameters
    ----------
    field_df : pl.DataFrame
        Output of `compute_field_summary` (per-field DF).
    mouse_id, date : str or None
        Animal id and session date for the title prefix.
    """
    cue_order = _abdc_cue_order(field_df, 'ABC')
    abc_fate = (
        field_df.filter(pl.col('trial_type') == 'ABC')
                .group_by(['cue_id', 'match_type']).agg(pl.len().alias('n'))
    )
    pivot = (
        abc_fate.pivot(values='n', index='cue_id', on='match_type')
                .fill_null(0)
                .with_columns(pl.col('cue_id').cast(pl.Enum(cue_order)))
                .sort('cue_id')
                .with_columns(pl.col('cue_id').cast(pl.String))
    )
    cue_ids = pivot['cue_id'].to_list()
    match_cols = [c for c in pivot.columns if c != 'cue_id']
    # Plot in the canonical bottom→top order (preserved → partial → lost).
    canonical = ['kept', 'position_locked', 'cue_locked',
                 'split', 'split_position', 'split_cue', 'lost', 'gained']
    plot_order = [m for m in canonical if m in match_cols] + [
        m for m in match_cols if m not in canonical
    ]

    with _paper_style():
        fig_w = max(6.5, 0.8 * len(cue_ids) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6))
        x = np.arange(len(cue_ids), dtype=float)
        bottom = np.zeros(len(cue_ids))
        for mt in plot_order:
            vals = np.array(pivot[mt].to_list(), dtype=float)
            ax.bar(x, vals, bottom=bottom, width=0.6,
                   color=MATCH_COLOR_MAP.get(mt, '#999'),
                   edgecolor='white', linewidth=0.9,
                   label=mt.replace('_', ' '), zorder=2)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(cue_ids, fontsize=10.5, fontweight='semibold',
                           color=_INK)
        ax.set_xlim(-0.6, len(cue_ids) - 0.4)
        ax.set_ylabel('ABC fields  (count)')
        ax.set_xlabel('Source cue (ABC)')
        ax.yaxis.grid(True, color=_GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
        ax.margins(y=0.08)

        ax.set_title('Fate of ABC fields by cue', loc='left', pad=22)
        _add_subtitle(ax, _build_subtitle(
            f'Mouse {mouse_id}' if mouse_id else None, date,
        ))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
                  ncol=min(len(plot_order), 7), borderaxespad=0)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.22)
    return fig


def find_strong_split_cells(
    combined: pl.DataFrame,
    field_df: pl.DataFrame,
    anchor_cue: str = 'A',
    anchor_match_type: str = 'kept',
) -> pl.DataFrame:
    """Cells with ≥1 split field AND a kept field at a non-reward anchor cue.

    What
    ----
    Returns the subset of `combined` (per-cell DF) where:
      - n_split > 0 (at least one ABC field was split into a position +
        cue successor in ABDC), AND
      - the cell has a field at `anchor_cue` whose `match_type` equals
        `anchor_match_type` (defaults: cue A, match_type 'kept').

    Why anchor at A
    ---------------
    Reward is delivered at cue C, so any cell with a C-region field is
    potentially modulated by reward expectation rather than place. By
    requiring a stable, well-matched field at A (the start of the track,
    far from reward), we screen for cells with genuine spatial selectivity
    - their C-region split is then most plausibly real remapping, not
    reward modulation.

    Loosening the anchor
    --------------------
    `anchor_match_type='kept'` is strict (identical position AND cue). Use
    `'position_locked'` or pass a different `anchor_cue` (e.g. 'B') if A
    has too few cells in a given session.

    Example
    -------
    >>> strong = find_strong_split_cells(combined, field_df,
    ...                                   anchor_cue='A',
    ...                                   anchor_match_type='kept')
    >>> # cells in `strong` have a stable A field AND ≥1 split

    Parameters
    ----------
    combined : pl.DataFrame
        Per-cell DF (output of `combine_summaries`).
    field_df : pl.DataFrame
        Per-field DF (output of `compute_field_summary`).
    anchor_cue : str, default 'A'
        Cue at which to require a kept field (far from reward at C).
    anchor_match_type : str, default 'kept'
        Required match type at the anchor cue.

    Returns
    -------
    pl.DataFrame
        Filtered slice of `combined`.
    """
    anchor_cells = (
        field_df.filter(
            (pl.col('match_type') == anchor_match_type)
            & (pl.col('cue_id') == anchor_cue)
        )['cell_id'].unique()
    )
    return combined.filter(
        pl.col('cell_id').is_in(anchor_cells) & (pl.col('n_split') > 0)
    )


def plot_split_cells(
    field_df: pl.DataFrame,
    result: PlaceFieldResult,
    bin_size_cm: float,
    cell_ids: list[int],
    title: str = 'Split cells',
    abc_name: str = 'ABC',
    abdc_name: str = 'ABDC',
    mouse_id: str | None = None,
    date: str | None = None,
) -> plt.Figure | None:
    """ABC + ABDC tuning curves for the given cells, with detected fields
    annotated as triangles on the x-axis.

    What
    ----
    One panel per cell. Each panel overlays:
      - ABC tuning curve (slate)
      - ABDC tuning curve (terracotta)
      - Triangles at the bottom of each panel marking detected field
        centers. Triangle direction encodes trial type:
            ▲ (up-pointing)   = ABC field
            ▼ (down-pointing) = ABDC field
        Triangle colour encodes `match_type` (from `MATCH_COLOR_MAP`).

    How to read it
    --------------
    A 'split' cell will show ONE ▲ (the ABC field) plus TWO ▼ fields
    (one at the same cm as the ▲ in soft clay = 'split_position', one
    at a different cm but the same cue in soft slate = 'split_cue').
    Compare peak positions and triangle colours across panels to interpret
    each cell's remapping pattern.

    Parameters
    ----------
    field_df : pl.DataFrame
        Output of `compute_field_summary` (per-field DF).
    result : PlaceFieldResult
        Output of `detect_place_fields`.
    bin_size_cm : float
        Spatial bin width.
    cell_ids : list[int]
        Cells to plot, one per subplot. Returns None if empty.
    title : str
        Base figure title; prefixed by mouse_id/date if given.
    abc_name, abdc_name : str
        Trial-type names in `result.fields`.
    mouse_id, date : str or None
        Animal id and session date for the title prefix.

    Returns
    -------
    plt.Figure or None
        None if `cell_ids` is empty.
    """
    if not cell_ids:
        return None

    abc_tt = _find_trial_type(result, abc_name)
    abdc_tt = _find_trial_type(result, abdc_name)
    abc_tc = result.fields[abc_tt].binF
    abdc_tc = result.fields[abdc_tt].binF
    x_abc = (np.arange(abc_tc.shape[1]) + 0.5) * bin_size_cm
    x_abdc = (np.arange(abdc_tc.shape[1]) + 0.5) * bin_size_cm

    n = len(cell_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.0), sharey=False)
    axes = np.atleast_1d(axes)

    seen_match_types: set[str] = set()
    for ax, cid in zip(axes, cell_ids):
        ax.plot(x_abc, abc_tc[cid], color=SLATE, lw=1.5)
        ax.plot(x_abdc, abdc_tc[cid], color=TERRACOTTA, lw=1.5)
        for row in field_df.filter(pl.col('cell_id') == cid).iter_rows(named=True):
            marker = '^' if row['trial_type'] == abc_name else 'v'
            mt = row['match_type']
            seen_match_types.add(mt)
            ax.plot(
                row['center_cm'], 0,
                marker=marker, color=MATCH_COLOR_MAP.get(mt, '#ccc'),
                markersize=9, clip_on=False,
            )
        ax.set_title(f'cell {cid}', fontsize=9)
        ax.set_xlabel('cm', fontsize=8)
        ax.tick_params(labelsize=7)

    legend_elems = [
        Line2D([0], [0], color=SLATE, lw=1.5, label='ABC trace'),
        Line2D([0], [0], color=TERRACOTTA, lw=1.5, label='ABDC trace'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=TAUPE,
               markersize=8, label='▲ ABC field'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=TAUPE,
               markersize=8, label='▼ ABDC field'),
    ] + [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=MATCH_COLOR_MAP[mt],
               markersize=8, label=mt)
        for mt in MATCH_COLOR_MAP
        if mt in seen_match_types
    ]
    axes[0].legend(handles=legend_elems, fontsize=6, loc='upper left',
                   ncol=2, framealpha=0.9)

    fig.suptitle(_title(title, mouse_id, date))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# __main__
# ----------------------------------------------------------------------

if __name__ == '__main__':
    mouse_id = '26'
    date = '2025-09-16'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths['parquet'])
    bin_size_cm = get_bin_size(meta)

    params = DetectionParams(smooth_sigma=1, signal_threshold=0.25)
    result = detect_place_fields(
        data, exp_config,
        signal_col='multi_day_dff',
        bin_size_cm=bin_size_cm,
        params=params,
    )
    print(result.summary())

    df = compute_remapping_scores(result, data, bin_size_cm)
    df = classify_cells(df)

    cell_df, field_df = compute_field_summary(
        result, data, bin_size_cm, position_tol_cm=15.0,
    )
    combined = combine_summaries(df, cell_df)

    total = combined.height
    print('\nCorrelation-based classification counts:')
    print(
        combined.group_by('classification')
                .agg(pl.len().alias('n'))
                .with_columns((100 * pl.col('n') / total).round(1).alias('pct'))
                .sort('n', descending=True)
    )
    print('\nField-matching outcome counts:')
    print(
        combined.group_by('field_outcome')
                .agg(pl.len().alias('n'))
                .with_columns((100 * pl.col('n') / total).round(1).alias('pct'))
                .sort('n', descending=True)
    )

    out_combined = session_dir / f'{mouse_id}_{date}_remapping_cells.csv'
    out_fields = session_dir / f'{mouse_id}_{date}_remapping_fields.csv'
    combined.write_csv(out_combined)
    # Stringify list column for CSV; the parquet version preserves typing.
    field_df.with_columns(
        pl.col('partner_field_ids').list.join(',')
    ).write_csv(out_fields)
    field_df.write_parquet(out_fields.with_suffix('.parquet'))
    print(f'\nSaved: {out_combined}')
    print(f'Saved: {out_fields}')
    print(f'Saved: {out_fields.with_suffix(".parquet")}')

    # Pass mouse_id and date to every plot so titles are self-documenting.
    plot_meta = dict(mouse_id=mouse_id, date=date)

    fig1 = plot_remapping_scatter(
        combined, title='ABC vs ABDC remapping', **plot_meta,
    )
    fig2 = plot_example_tuning_curves(
        result, data, combined, bin_size_cm, n_examples=6, **plot_meta,
    )
    fig3 = plot_outcome_comparison(combined, **plot_meta)

    # Where do new fields appear?
    fig4, gained_by_cue = plot_gained_fields_by_cue(field_df, **plot_meta)
    print('\nGained fields by cue:')
    print(gained_by_cue)

    # What happens to each ABC cue's fields?
    fig5 = plot_abc_fate_by_cue(field_df, **plot_meta)

    # Top splits - any cell with a split field
    all_split_ids = (
        combined.filter(pl.col('n_split') > 0)
                .sort('n_split', descending=True)['cell_id']
                .to_list()[:6]
    )
    fig6 = plot_split_cells(
        field_df, result, bin_size_cm, all_split_ids,
        title='Split cells (top 6 by n_split)',
        **plot_meta,
    )

    # Strong splits - split AND a kept field at A (rules out reward-only cells)
    strong_splits = find_strong_split_cells(
        combined, field_df, anchor_cue='A', anchor_match_type='kept',
    )
    print(f'\nStrong-split cells (split + kept A field): {strong_splits.height}')
    strong_split_ids = (
        strong_splits.sort('n_split', descending=True)['cell_id']
                     .to_list()[:6]
    )
    fig7 = plot_split_cells(
        field_df, result, bin_size_cm, strong_split_ids,
        title='Strong splits - n_split>0 AND kept field at A '
              '(supports real place tuning, not reward-only)',
        **plot_meta,
    )

    plt.show()
