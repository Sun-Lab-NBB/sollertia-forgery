"""
Remapping Analysis
==================
Quantifies how cells from a pre-extension trial type (e.g. ABC) are reused or replaced
in a post-extension trial type (e.g. ABDC) within a single session where both trial
types are interleaved.

Per cell: finds strongest place field in each trial type → assigns to a cue zone →
builds a cross-tab of ABDC zone × ABC origin.

Outputs:
    - Full cross-tab (counts, row-normalized %, column-normalized %)
    - Coarse origin breakdowns for D+0d and the entire extended portion
    - Position-vs-cue remapping for cells that had ABC fields at C/0c
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from skimage.measure import regionprops

from place_field_detection import PlaceFields1d, PlaceFieldResult

@dataclass
class RemappingResult:
    """Per-cell remapping analysis between pre- and post-extension trial types.

    Args:
        cell_assignments: Per-cell zone labels and strongest-field positions.
            Columns: cell_idx, abc_zone, abdc_zone, abc_pos_cm, abdc_pos_cm.
        cross_tab_counts: Wide DataFrame, rows = ABDC zone, cols = ABC zones.
            Values are cell counts.
        cross_tab_pct: Same shape as cross_tab_counts but row-normalized to %
            (each ABDC row sums to 100). Reads "of cells in this ABDC zone, what
            fraction came from each ABC origin."
        cross_tab_col_pct: Column-normalized to % (each ABC origin sums to 100).
            Reads "of cells with this ABC origin, what fraction end up in each
            ABDC zone." This is the recruitment view: pick an ABDC column and
            the bars/segments tell you composition.
        count_mat: Numpy view of counts. Shape (n_abdc_zones, n_abc_zones).
        pct_mat: Numpy view of row-normalized percentages.
        col_pct_mat: Numpy view of column-normalized percentages.
        coarse_d0d: Origin breakdown for cells coding D+0d (the new region).
        coarse_extended: Origin breakdown for cells coding the full extended portion
            (D, 0d, C', 0c').
        c_zone_remapping: Where ABC C/0c cells go in ABDC.
        abc_zone_order: Canonical column order (zones + 'none' last).
        abdc_zone_order: Canonical row order.
        n_cells_analyzed: Number of cells in the analysis (after `cells` filter).
    """
    cell_assignments: pl.DataFrame
    cross_tab_counts: pl.DataFrame
    cross_tab_pct: pl.DataFrame
    cross_tab_col_pct: pl.DataFrame
    count_mat: np.ndarray
    pct_mat: np.ndarray
    col_pct_mat: np.ndarray
    coarse_d0d: dict
    coarse_extended: dict
    c_zone_remapping: dict
    abc_zone_order: list[str]
    abdc_zone_order: list[str]
    n_cells_analyzed: int

    def summary(self) -> str:
        """Multi-line summary of the remapping analysis."""
        lines = [f"Remapping analysis: {self.n_cells_analyzed} cells"]

        lines.append("\n  Coarse — cells coding D + 0d (new region):")
        c = self.coarse_d0d
        if c['n_total'] == 0:
            lines.append("    (no cells)")
        else:
            shared_label = '/'.join(c.get('shared_zones', ['A', '0a', 'B', '0b']))
            c_label = '/'.join(c.get('c_origin_zones', ['C', '0c']))
            lines.append(f"    n = {c['n_total']}")
            lines.append(f"    From shared zones ({shared_label}): "
                         f"{c['n_shared']} ({c['shared_location_pct']:.1f}%)")
            lines.append(f"    From C-zone in ABC ({c_label}): "
                         f"{c['n_c_zone']} ({c['cue_shared_displaced_pct']:.1f}%)")
            lines.append(f"    Brand new (no ABC field): "
                         f"{c['n_brand_new']} ({c['brand_new_pct']:.1f}%)")

        lines.append("\n  Coarse — cells coding entire extended portion (D + 0d + C' + 0c'):")
        c = self.coarse_extended
        if c['n_total'] == 0:
            lines.append("    (no cells)")
        else:
            shared_label = '/'.join(c.get('shared_zones', ['A', '0a', 'B', '0b']))
            c_label = '/'.join(c.get('c_origin_zones', ['C', '0c']))
            lines.append(f"    n = {c['n_total']}")
            lines.append(f"    From shared zones ({shared_label}): "
                         f"{c['n_shared']} ({c['shared_location_pct']:.1f}%)")
            lines.append(f"    From C-zone in ABC ({c_label}): "
                         f"{c['n_c_zone']} ({c['cue_shared_displaced_pct']:.1f}%)")
            lines.append(f"    Brand new (no ABC field): "
                         f"{c['n_brand_new']} ({c['brand_new_pct']:.1f}%)")

        lines.append("\n  C-zone remapping (cells with field at C/0c in ABC):")
        c = self.c_zone_remapping
        if c['n_total'] == 0:
            lines.append("    (no cells)")
        else:
            lines.append(f"    n = {c['n_total']}")
            lines.append(f"    Position-following → D/0d:   {c['n_position']} ({c['position_following_pct']:.1f}%)")
            lines.append(f"    Cue-following → C'/0c':      {c['n_cue']} ({c['cue_following_pct']:.1f}%)")
            lines.append(f"    Elsewhere in ABDC:           {c['n_elsewhere']} ({c['elsewhere_pct']:.1f}%)")
            lines.append(f"    Dropped (no ABDC field):     {c['n_dropped']} ({c['dropped_pct']:.1f}%)")

        return '\n'.join(lines)


def _strongest_field_position(pf: PlaceFields1d, n_cells: int) -> np.ndarray:
    """For each cell, return its strongest field's centroid position in cm.

    "Strongest" = highest mean intensity, matching the cell-ordering logic in the
    heatmap (PlaceFields1d.order and plot_combined_heatmap).

    Args:
        pf: PlaceFields1d from a single trial type.
        n_cells: Total number of cells (some cells may have no field).

    Returns:
        Array of shape (n_cells,). NaN where the cell has no detected field.
    """
    best_pos = np.full(n_cells, np.nan)
    best_int = np.full(n_cells, -np.inf)

    if pf.label_im.max() == 0:
        return best_pos

    for prop in regionprops(pf.label_im, pf.binF, cache=False):
        cell_idx = prop['coords'][0, 0]
        mean_int = prop['mean_intensity']
        if mean_int > best_int[cell_idx]:
            best_int[cell_idx] = mean_int
            best_pos[cell_idx] = prop['weighted_centroid'][1] * pf.bin_size_cm

    return best_pos


def _build_zone_lookup(config: dict, trial_type: str, relabel: dict | None = None
                       ) -> list[tuple[float, float, str]]:
    """Build a position-sorted list of (start_cm, end_cm, label) zones for a trial type.

    Disambiguates gray zones by labeling them with the preceding cue's letter
    (e.g. '0a', '0b'). Uses config['trial_structures'][trial_type]['cue_sequence']
    and config['cue_map'] for widths, matching the convention used elsewhere in this
    module (see plot() at line ~340).

    Args:
        config: Experiment config.
        trial_type: e.g. 'ABC' or 'ABDC'.
        relabel: Optional renaming applied after disambiguation. Useful for
            distinguishing same-cue-different-position labels across trial types,
            e.g. {'C': "C'", '0c': "0c'"} for ABDC.

    Returns:
        List of (start_cm, end_cm, label) tuples in track order.
    """
    import plot_utils as pfmt

    GRAY_CUE_ID = 0  # convention used in plot() and plot_combined_heatmap
    ts = config.get('trial_structures', {}).get(trial_type, {})
    cue_widths = config.get('cue_map', {})
    cue_sequence = ts.get('cue_sequence', [])
    labels_map = pfmt.get_cue_labels(config)

    zones = []
    pos = 0.0
    last_letter = None

    for cue_id in cue_sequence:
        width = cue_widths[cue_id]
        if cue_id == GRAY_CUE_ID:
            full_label = f'0{last_letter.lower()}' if last_letter else '0'
        else:
            label = labels_map.get(cue_id, str(cue_id))
            full_label = label
            last_letter = label

        if relabel and full_label in relabel:
            full_label = relabel[full_label]

        zones.append((pos, pos + width, full_label))
        pos += width

    return zones


def _assign_zone(pos_cm: float, zones: list[tuple[float, float, str]]) -> str | None:
    """Assign a position (cm) to a zone label, or None if NaN/out of range."""
    if np.isnan(pos_cm):
        return None
    last_end = zones[-1][1]
    if pos_cm >= last_end:  # tolerate floating-point overflow at track end
        pos_cm = last_end - 1e-6
    for start, end, label in zones:
        if start <= pos_cm < end:
            return label
    return None


def quantify_remapping(
    result: PlaceFieldResult,
    config: dict,
    abc_trial_type: str = 'ABC',
    abdc_trial_type: str = 'ABDC',
    cells: np.ndarray | None = None,
    filter_mode: str = 'either_session',
    abdc_relabel: dict | None = None,
    new_zones: tuple = ('D', '0d'),
    extended_zones: tuple = ('D', '0d', "C'", "0c'"),
    shared_zones: tuple = ('A', '0a', 'B', '0b'),
    c_origin_zones: tuple = ('C', '0c'),
) -> RemappingResult:
    """Quantify how ABC place cells are reused or replaced in ABDC.

    Per cell, finds the strongest place field (highest mean intensity) in each trial
    type and assigns it to a cue zone (one of A, 0a, B, 0b, C, 0c in ABC; A, 0a, B,
    0b, D, 0d, C', 0c' in ABDC; or 'none' if the cell has no field in that trial
    type). This matches the cell-ordering logic in the heatmap.

    Computes:
        - Cross-tab of ABDC zone × ABC zone (counts, row-normalized %, and
          column-normalized %).
        - Coarse: cells coding the new D+0d region — % from shared / from C-zone /
          brand new.
        - Coarse: same breakdown for the entire extended portion (D, 0d, C', 0c').
        - Position-vs-cue remapping for ABC C/0c cells: do they follow position
          (→ D/0d, same x), follow the cue (→ C'/0c', same letter different x),
          go elsewhere, or drop their field?

    Within-session analysis: ABC and ABDC trials are interleaved in a single session
    (post-extension).

    Args:
        result: PlaceFieldResult containing both ABC and ABDC trial types.
        config: Experiment config (for cue regions and widths).
        abc_trial_type: Pre-extension trial type. Default 'ABC'.
        abdc_trial_type: Post-extension trial type. Default 'ABDC'.
        cells: Optional cell indices to restrict analysis to. If provided, this
            takes precedence over filter_mode.
        filter_mode: How to filter cells when `cells` is None.
            'either_session' (default): keep cells that are place cells in ABC OR
                ABDC in this session (uses result.is_place_cell). Recommended for
                within-session remapping — excludes silent cells without removing
                cells brand-new to ABDC.
            'both_session': keep cells that are place cells in BOTH trial types.
                Kills the "brand new" category by definition.
            'all': no filter — every cell, including silent ones. The (none, none)
                bucket will be large.
        abdc_relabel: Renaming for ABDC zone labels to disambiguate same-letter
            cues at different positions. Defaults to {'C': "C'", '0c': "0c'"}.
        new_zones: ABDC zones counted as the new region (D + post-D gray).
        extended_zones: ABDC zones counted as the full extended portion.
        shared_zones: ABC zones in the pre-extension shared section.
        c_origin_zones: ABC zones at the C cue (cue-shared but spatially displaced
            in ABDC).

    Returns:
        RemappingResult with per-cell assignments and summary statistics.
    """
    if abdc_relabel is None:
        abdc_relabel = {'C': "C'", '0c': "0c'"}

    n_cells = result.n_cells

    if abc_trial_type not in result.fields or abdc_trial_type not in result.fields:
        raise ValueError(
            f"Both '{abc_trial_type}' and '{abdc_trial_type}' must be present in "
            f"result.fields. Got: {list(result.fields.keys())}"
        )

    # Per-cell strongest-field position (matches heatmap sort)
    abc_pos = _strongest_field_position(result.fields[abc_trial_type], n_cells)
    abdc_pos = _strongest_field_position(result.fields[abdc_trial_type], n_cells)

    # Zone lookups
    abc_zones = _build_zone_lookup(config, abc_trial_type)
    abdc_zones = _build_zone_lookup(config, abdc_trial_type, relabel=abdc_relabel)

    abc_zone_order = [z[2] for z in abc_zones] + ['none']
    abdc_zone_order = [z[2] for z in abdc_zones] + ['none']

    abc_labels = [_assign_zone(p, abc_zones) or 'none' for p in abc_pos]
    abdc_labels = [_assign_zone(p, abdc_zones) or 'none' for p in abdc_pos]

    cell_df = pl.DataFrame({
        'cell_idx': np.arange(n_cells, dtype=np.int64),
        'abc_zone': abc_labels,
        'abdc_zone': abdc_labels,
        'abc_pos_cm': abc_pos,
        'abdc_pos_cm': abdc_pos,
    })

    if cells is not None:
        cells_arr = np.asarray(cells)
        if cells_arr.dtype == bool:
            cells_arr = np.where(cells_arr)[0]
        cell_df = cell_df.filter(pl.col('cell_idx').is_in(cells_arr.tolist()))
    else:
        # Apply filter_mode using result.is_place_cell
        if filter_mode == 'all':
            pass  # no filter
        elif filter_mode in ('either_session', 'both_session'):
            abc_pc = result.is_place_cell.get(abc_trial_type, np.zeros(n_cells, dtype=bool))
            abdc_pc = result.is_place_cell.get(abdc_trial_type, np.zeros(n_cells, dtype=bool))
            if filter_mode == 'either_session':
                keep = abc_pc | abdc_pc
            else:
                keep = abc_pc & abdc_pc
            keep_idx = np.where(keep)[0]
            cell_df = cell_df.filter(pl.col('cell_idx').is_in(keep_idx.tolist()))
        else:
            raise ValueError(
                f"filter_mode must be 'either_session', 'both_session', or 'all'. "
                f"Got: {filter_mode!r}"
            )

    n_analyzed = len(cell_df)

    # Cross-tab as count matrix indexed by canonical zone order
    abdc_idx = {z: i for i, z in enumerate(abdc_zone_order)}
    abc_idx = {z: i for i, z in enumerate(abc_zone_order)}
    count_mat = np.zeros((len(abdc_zone_order), len(abc_zone_order)), dtype=np.int64)

    if n_analyzed > 0:
        gb = cell_df.group_by(['abdc_zone', 'abc_zone']).len()
        for row in gb.iter_rows(named=True):
            i = abdc_idx.get(row['abdc_zone'])
            j = abc_idx.get(row['abc_zone'])
            if i is not None and j is not None:
                count_mat[i, j] = row['len']

    row_sums = count_mat.sum(axis=1, keepdims=True).astype(float)
    safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
    pct_mat = 100.0 * count_mat / safe_row_sums

    col_sums = count_mat.sum(axis=0, keepdims=True).astype(float)
    safe_col_sums = np.where(col_sums == 0, 1.0, col_sums)
    col_pct_mat = 100.0 * count_mat / safe_col_sums

    cross_tab_counts = pl.DataFrame(
        {'abdc_zone': abdc_zone_order,
         **{abc_zone_order[j]: count_mat[:, j] for j in range(len(abc_zone_order))}}
    )
    cross_tab_pct = pl.DataFrame(
        {'abdc_zone': abdc_zone_order,
         **{abc_zone_order[j]: pct_mat[:, j] for j in range(len(abc_zone_order))}}
    )
    cross_tab_col_pct = pl.DataFrame(
        {'abdc_zone': abdc_zone_order,
         **{abc_zone_order[j]: col_pct_mat[:, j] for j in range(len(abc_zone_order))}}
    )

    # Coarse origin splits
    def _origin_split(target_abdc: list[str]) -> dict:
        if n_analyzed == 0:
            return {'shared_location_pct': 0.0, 'cue_shared_displaced_pct': 0.0,
                    'brand_new_pct': 0.0, 'n_shared': 0, 'n_c_zone': 0,
                    'n_brand_new': 0, 'n_total': 0,
                    'shared_zones': list(shared_zones),
                    'c_origin_zones': list(c_origin_zones)}
        target_idx = [abdc_idx[z] for z in target_abdc if z in abdc_idx]
        sub = count_mat[target_idx, :].sum(axis=0)  # sum over the chosen ABDC rows
        n = int(sub.sum())
        if n == 0:
            return {'shared_location_pct': 0.0, 'cue_shared_displaced_pct': 0.0,
                    'brand_new_pct': 0.0, 'n_shared': 0, 'n_c_zone': 0,
                    'n_brand_new': 0, 'n_total': 0,
                    'shared_zones': list(shared_zones),
                    'c_origin_zones': list(c_origin_zones)}
        n_shared = int(sum(sub[abc_idx[z]] for z in shared_zones if z in abc_idx))
        n_c = int(sum(sub[abc_idx[z]] for z in c_origin_zones if z in abc_idx))
        n_new = int(sub[abc_idx['none']])
        return {
            'shared_location_pct': 100 * n_shared / n,
            'cue_shared_displaced_pct': 100 * n_c / n,
            'brand_new_pct': 100 * n_new / n,
            'n_shared': n_shared, 'n_c_zone': n_c,
            'n_brand_new': n_new, 'n_total': n,
            'shared_zones': list(shared_zones),
            'c_origin_zones': list(c_origin_zones),
        }

    coarse_d0d = _origin_split(list(new_zones))
    coarse_extended = _origin_split(list(extended_zones))

    # C-zone remapping: where do ABC C/0c cells go in ABDC?
    c_origin_idx = [abc_idx[z] for z in c_origin_zones if z in abc_idx]
    if not c_origin_idx or n_analyzed == 0:
        c_remap = {'position_following_pct': 0.0, 'cue_following_pct': 0.0,
                   'elsewhere_pct': 0.0, 'dropped_pct': 0.0,
                   'n_position': 0, 'n_cue': 0, 'n_elsewhere': 0,
                   'n_dropped': 0, 'n_total': 0}
    else:
        c_col = count_mat[:, c_origin_idx].sum(axis=1)  # sum over ABC C/0c columns
        n_c_total = int(c_col.sum())
        if n_c_total == 0:
            c_remap = {'position_following_pct': 0.0, 'cue_following_pct': 0.0,
                       'elsewhere_pct': 0.0, 'dropped_pct': 0.0,
                       'n_position': 0, 'n_cue': 0, 'n_elsewhere': 0,
                       'n_dropped': 0, 'n_total': 0}
        else:
            cue_following_zones = [abdc_relabel.get(z, z) for z in c_origin_zones]
            n_pos = int(sum(c_col[abdc_idx[z]] for z in new_zones if z in abdc_idx))
            n_cue = int(sum(c_col[abdc_idx[z]] for z in cue_following_zones if z in abdc_idx))
            n_drop = int(c_col[abdc_idx['none']])
            n_else = n_c_total - n_pos - n_cue - n_drop
            c_remap = {
                'position_following_pct': 100 * n_pos / n_c_total,
                'cue_following_pct': 100 * n_cue / n_c_total,
                'elsewhere_pct': 100 * n_else / n_c_total,
                'dropped_pct': 100 * n_drop / n_c_total,
                'n_position': n_pos, 'n_cue': n_cue,
                'n_elsewhere': n_else, 'n_dropped': n_drop,
                'n_total': n_c_total,
            }

    return RemappingResult(
        cell_assignments=cell_df,
        cross_tab_counts=cross_tab_counts,
        cross_tab_pct=cross_tab_pct,
        cross_tab_col_pct=cross_tab_col_pct,
        count_mat=count_mat,
        pct_mat=pct_mat,
        col_pct_mat=col_pct_mat,
        coarse_d0d=coarse_d0d,
        coarse_extended=coarse_extended,
        c_zone_remapping=c_remap,
        abc_zone_order=abc_zone_order,
        abdc_zone_order=abdc_zone_order,
        n_cells_analyzed=n_analyzed,
    )


def plot_remapping(
    rr: RemappingResult,
    mouse_id: str | None = None,
    date: str | None = None,
    figsize: tuple = (14, 11),
    cmap: str = 'magma',
    drop_empty_abdc: bool = False,
    drop_none_abdc: bool = False,
    annotate: bool = True,
    show: bool = True,
) -> plt.Figure:
    """Plot the remapping cross-tab and per-zone stacked bars.

    Both panels are column-normalized: each ABDC zone (x-axis) sums to 100% of the
    cells with their strongest ABDC field in that zone. Reading direction: pick an
    ABDC column → it tells you the ABC-origin composition of cells now coding that
    zone. The 'none' row in the heatmap = cells brand-new to ABDC (no ABC field).

    Top: column-normalized cross-tab heatmap (rows = ABC origin including 'none',
    cols = ABDC zone).
    Bottom: stacked bar chart per ABDC zone, same data.

    Args:
        rr: RemappingResult from quantify_remapping().
        mouse_id: Mouse ID for title.
        date: Session date for title.
        figsize: Figure size.
        cmap: Heatmap colormap.
        drop_empty_abdc: If True, omit ABDC columns with zero cells.
        drop_none_abdc: If True, omit the 'none' ABDC column (cells lost in ABDC).
            Default False — keeps the symmetric view of ABC cells that dropped out.
        annotate: If True, write percentage values inside heatmap cells.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    abdc_zones = list(rr.abdc_zone_order)
    abc_zones = list(rr.abc_zone_order)
    counts = rr.count_mat.copy()                     # (n_abdc, n_abc)
    col_pct = rr.col_pct_mat.copy()                  # column-normalized (cols = ABC)

    # Transpose so x = ABDC, y = ABC
    counts_T = counts.T                              # (n_abc, n_abdc)
    # We want each ABDC column (x) to sum to 100% — that's column-normalization
    # of the transposed matrix, which is row-normalization of the original.
    col_norm = rr.pct_mat.T                          # (n_abc, n_abdc), each column sums to 100

    # Drop columns (ABDC) as requested
    abdc_keep_mask = np.ones(len(abdc_zones), dtype=bool)
    if drop_empty_abdc:
        abdc_keep_mask &= counts.sum(axis=1) > 0
    if drop_none_abdc and 'none' in abdc_zones:
        abdc_keep_mask[abdc_zones.index('none')] = False

    abdc_zones_kept = [z for z, k in zip(abdc_zones, abdc_keep_mask) if k]
    counts_T_kept = counts_T[:, abdc_keep_mask]
    col_norm_kept = col_norm[:, abdc_keep_mask]

    # Recompute column-normalization on the kept subset (so each visible bar still
    # sums to 100 — a dropped 'none' column doesn't make remaining bars >100).
    sub_col_sums = counts_T_kept.sum(axis=0, keepdims=True).astype(float)
    safe = np.where(sub_col_sums == 0, 1.0, sub_col_sums)
    col_norm_kept = 100.0 * counts_T_kept / safe

    fig, (ax_hm, ax_bar) = plt.subplots(
        2, 1, figsize=figsize, dpi=150,
        gridspec_kw={'height_ratios': [1, 1.1], 'hspace': 0.35},
    )

    # Suptitle
    title_bits = []
    if mouse_id:
        title_bits.append(str(mouse_id))
    if date:
        title_bits.append(str(date))
    title_bits.append(f'n={rr.n_cells_analyzed}')
    fig.suptitle(' — '.join(title_bits) + '  ·  ABC → ABDC remapping',
                 fontsize=13, fontweight='bold', y=0.985)

    # Heatmap
    im = ax_hm.imshow(col_norm_kept, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    ax_hm.set_xticks(range(len(abdc_zones_kept)))
    ax_hm.set_xticklabels(abdc_zones_kept, fontsize=10)
    ax_hm.set_yticks(range(len(abc_zones)))
    ax_hm.set_yticklabels(abc_zones, fontsize=10)
    ax_hm.set_xlabel('ABDC zone (current, strongest field)', fontsize=11)
    ax_hm.set_ylabel('ABC zone (origin, strongest field)', fontsize=11)
    ax_hm.set_title('Column-normalized: each ABDC zone = 100%', fontsize=11)

    cb = fig.colorbar(im, ax=ax_hm, label='% of ABDC column', fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=9)

    if annotate:
        for i in range(col_norm_kept.shape[0]):
            for j in range(col_norm_kept.shape[1]):
                v = col_norm_kept[i, j]
                if v > 0:
                    color = 'white' if v < 50 else 'black'
                    ax_hm.text(j, i, f'{v:.0f}', ha='center', va='center',
                               fontsize=8, color=color)

    # Stacked bar — each x position = one ABDC zone, segments = ABC origins
    n_abc = len(abc_zones)
    # Distinct color for 'none' (gray); otherwise tab10
    base_cmap = plt.cm.tab10(np.linspace(0, 1, max(n_abc, 10)))[:n_abc]
    colors = []
    for j, z in enumerate(abc_zones):
        if z == 'none':
            colors.append((0.65, 0.65, 0.65, 1.0))
        else:
            colors.append(base_cmap[j])

    x = np.arange(len(abdc_zones_kept))
    bottoms = np.zeros(len(abdc_zones_kept))
    for i, abc_zone in enumerate(abc_zones):
        ax_bar.bar(x, col_norm_kept[i, :], bottom=bottoms, label=abc_zone,
                   color=colors[i], edgecolor='white', linewidth=0.5, width=0.78)
        bottoms += col_norm_kept[i, :]

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(abdc_zones_kept, fontsize=10)
    ax_bar.set_ylabel('% of cells (ABC origin composition)', fontsize=11)
    ax_bar.set_xlabel('ABDC zone', fontsize=11)
    ax_bar.set_ylim(0, 108)  # headroom for n labels
    ax_bar.legend(title='ABC origin', bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=9, title_fontsize=10, frameon=False)
    ax_bar.set_title('ABC origin composition per ABDC zone', fontsize=11)

    n_per_col = counts_T_kept.sum(axis=0)
    for xi, n in zip(x, n_per_col):
        ax_bar.text(xi, 102, f'n={int(n)}', ha='center', va='bottom', fontsize=8)

    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.86)
    if show:
        plt.show()
    return fig



if __name__ == "__main__":
    from df_processing import find_session_dir, get_session_paths, load_session_context, load_processed_session
    from place_field_detection import detect_place_fields, DetectionParams
    from experiment_place_cells import load_experiment_place_cells
    from pathlib import Path

    mouse_id = "26"
    mouse_dir = Path("/Users/cs963/Desktop/sun_lab_projects/datasets", mouse_id)
    date = "2025-09-16"

    session_dir = find_session_dir(mouse_dir, date)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    data, meta = load_processed_session(paths["parquet"])

    params = DetectionParams(smooth_sigma=0, signal_threshold=0.3)
    result = detect_place_fields(
        data, exp_config, signal_col="multi_day_dff",
        bin_size_cm=meta["bin_size_cm"], params=params,
    )

    # Use experiment-wide PCs (≥2 days, any trial type) as the cell filter
    exp_pcs = load_experiment_place_cells(mouse_dir)
    cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)

    rr = quantify_remapping(result, exp_config, cells=cells)
    print(rr.summary())
    plot_remapping(rr, mouse_id=mouse_id, date=date)
    plt.show()
