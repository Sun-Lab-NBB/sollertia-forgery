"""
Experiment-Wide Place Cells
===========================
Discovers all sessions for a mouse, runs single-session detection on each, and
aggregates results into a per-cell × per-day × per-trial-type presence matrix.

This is the experiment-wide answer to "is this cell a place cell?" — it does not
depend on which subset of dates you happened to load, and is cached to disk so
downstream analyses don't re-detect every run.

Usage:
    # Compute once (caches to mouse_dir / 'experiment_place_cells.npz')
    exp_pcs = compute_experiment_place_cells(mouse_dir)

    # In any later script:
    exp_pcs = load_experiment_place_cells(mouse_dir)
    cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)

Notes:
    - Aggregation only counts days where the trial type *existed*. A pre-extension
      day with no ABDC trials is not counted as a "no" for ABDC — it's masked out.
    - Cached file records the params + signal_col + dates. If those change, force=True
      rebuilds; otherwise load is fast.
"""
from __future__ import annotations

import json
import hashlib
import pickle
import yaml
from dataclasses import dataclass, asdict, field, field as dc_field
from pathlib import Path

import numpy as np
import polars as pl

from place_field_detection import (
    DetectionParams, PlaceFieldResult, detect_place_fields,
)
from df_processing import (
    find_session_dir, load_session_context, get_session_paths,
    load_processed_session,
)

CACHE_FILENAME = 'experiment_place_cells.npz'


# PER-SESSION CACHING (memory-bounded multiday detection building block)

def _params_hash(
    params: DetectionParams,
    signal_col: str,
    bin_size_cm: int,
) -> str:
    """8-char sha1 of detection params + signal_col + bin_size_cm.

    Used to disambiguate per-session cache files so different param sets coexist
    instead of silently overwriting each other.
    """
    payload = {**asdict(params), 'signal_col': signal_col, 'bin_size_cm': int(bin_size_cm)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def _get_pf_cache_path(
    session_dir: Path,
    animal_id: str,
    date: str,
    signal_col: str,
    params: DetectionParams,
    bin_size_cm: int,
) -> Path:
    """Standard path for a per-session cached PlaceFieldResult.

    Filename includes an 8-char params hash so caches built under different
    DetectionParams (or different bin_size_cm) coexist on disk and never
    silently shadow each other.
    """
    h = _params_hash(params, signal_col, bin_size_cm)
    return Path(session_dir) / f'{animal_id}_{date}_place_fields_{signal_col}_{h}.pkl'


def save_place_field_result(result: PlaceFieldResult, path: Path):
    """Pickle a PlaceFieldResult to disk."""
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def load_place_field_result(path: Path) -> PlaceFieldResult:
    """Load a pickled PlaceFieldResult from disk."""
    path = Path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_and_verify_cache(
    cache_path: Path,
    expected_params: DetectionParams,
) -> PlaceFieldResult | None:
    """Load a cached PlaceFieldResult and verify its params match.

    The pkl contents are the source of truth — even though the filename hash
    *should* guarantee a match, we still open the file and compare the stored
    DetectionParams field-by-field. This catches:
      - Hash collisions (vanishingly rare with sha1, but cheap to defend against).
      - Pickles from before this hashing scheme existed (params=None inside).
      - Manually moved/renamed cache files whose filename no longer reflects
        their contents.
      - Future param-schema changes where new fields are added: a freshly
        constructed DetectionParams will have the new defaults, but an old pkl
        won't, and equality will (correctly) fail → forces a recompute.

    Returns:
        The loaded PlaceFieldResult if its params equal `expected_params`,
        otherwise None (caller should treat as a cache miss and recompute).
    """
    result = load_place_field_result(cache_path)

    # Defensive: very old caches predate `params` being stored on the result.
    if result.params is None:
        print(f"  Cache {cache_path.name} has no params recorded — treating as stale.")
        return None

    # DetectionParams is a plain dataclass, so `==` compares all fields.
    if result.params != expected_params:
        print(f"  Cache {cache_path.name} params do not match current — recomputing.")
        return None

    return result


def detect_place_fields_for_session(
    session_dir: Path,
    signal_col: str = 'multi_day_dff',
    params: DetectionParams | None = None,
    force_recompute: bool = False,
) -> PlaceFieldResult:
    """Run detect_place_fields with on-disk auto-caching.

    Bin size is always read from the session's metadata yaml — there is no
    fallback default. If the metadata is missing or doesn't carry a
    ``bin_size_cm`` field, this function raises.

    On cache hit: returns the saved PlaceFieldResult without loading the
    session df.
    On cache miss: loads the session, runs detection, pickles the result, returns.

    The heavy frame-level df is local to this function and gets garbage collected
    when the function returns. This is the building block for memory-bounded
    multiday detection — never holds more than one session in memory at a time.

    Cache file: {session_dir}/{animal_id}_{date}_place_fields_{signal_col}_{hash}.pkl
    The 8-char {hash} encodes DetectionParams + signal_col + bin_size_cm.

    Args:
        session_dir: Path to the session directory.
        signal_col: Column for which to detect place fields.
        params: DetectionParams. Defaults if None.
        force_recompute: If True, ignore the cache and rerun detection.

    Returns:
        PlaceFieldResult.
    """
    if params is None:
        params = DetectionParams()

    session_dir = Path(session_dir)
    session_data, exp_config = load_session_context(session_dir)
    paths = get_session_paths(session_dir, session_data)
    animal_id = session_data['animal_id']
    date = session_data['session_name'][:10]

    # Read bin_size_cm from the processed-session metadata yaml (cheap — no parquet).
    meta_path = Path(paths['parquet']).with_suffix('.yaml')
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Processed metadata yaml not found at {meta_path}. "
            f"Re-run processing or supply bin_size_cm explicitly upstream."
        )
    with open(meta_path, 'r') as f:
        meta_yaml = yaml.safe_load(f) or {}
    if 'bin_size_cm' not in meta_yaml:
        raise ValueError(
            f"{meta_path.name} has no bin_size_cm — re-run processing to write it."
        )
    bin_size_cm = int(meta_yaml['bin_size_cm'])

    cache_path = _get_pf_cache_path(
        session_dir, animal_id, date, signal_col, params, bin_size_cm,
    )

    if cache_path.exists() and not force_recompute:
        print(f"  Loading cached place fields: {cache_path.name}")
        cached = _load_and_verify_cache(cache_path, params)
        if cached is not None:
            return cached

    print(f"  Running place field detection for {animal_id} {date} ({signal_col})...")
    data, metadata = load_processed_session(paths['parquet'])
    result = detect_place_fields(
        data, exp_config, signal_col=signal_col,
        bin_size_cm=bin_size_cm, params=params, metadata=metadata,
    )
    save_place_field_result(result, cache_path)
    print(f"  Saved cache: {cache_path.name}")
    return result


@dataclass
class ExperimentPlaceCells:
    """Experiment-wide place cell identities aggregated across all sessions.

    Args:
        dates: Sorted list of session dates (YYYY-MM-DD).
        trial_types: Union of all trial types seen across the experiment.
        n_cells: Number of multi-day-registered cells.
        presence: {trial_type: bool array (n_cells, n_days)}. True = cell was a
            place cell in that trial type on that day.
        centers: {trial_type: float array (n_cells, n_days)}. Strongest field
            position in cm per cell per day. NaN = no field that day OR trial
            type didn't run that day. Use trial_type_existed to disambiguate.
        trial_type_existed: {trial_type: bool array (n_days,)}. True = the trial
            type ran on that day. Used to exclude masked-out days from aggregation.
        signal_col: Signal column used for detection (e.g. 'multi_day_dff').
        params: DetectionParams used.
        params_hash: SHA1 of (params + signal_col) for cache invalidation.
    """
    dates: list[str]
    trial_types: list[str]
    n_cells: int
    presence: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    trial_type_existed: dict[str, np.ndarray]
    signal_col: str
    params: DetectionParams
    params_hash: str

    def days_present_for(self, trial_type: str) -> np.ndarray:
        """Per cell: number of days a place cell in this trial type.

        Only counts days where the trial type existed.

        Returns:
            int array of shape (n_cells,).
        """
        if trial_type not in self.presence:
            return np.zeros(self.n_cells, dtype=int)
        existed = self.trial_type_existed[trial_type]  # (n_days,)
        # presence is already False on days where trial type didn't exist (we set
        # it that way at build time), so summing across days is correct
        return self.presence[trial_type][:, existed].sum(axis=1)

    def place_cells_in(self, trial_type: str, min_days: int = 2) -> np.ndarray:
        """Cell indices that were place cells in this trial type on >= min_days days.

        Args:
            trial_type: Trial type to query.
            min_days: Minimum number of days. Default 2 — chance double-detection
                is much rarer than chance single-detection, defensible without
                shuffle validation.

        Returns:
            int array of cell indices.
        """
        return np.where(self.days_present_for(trial_type) >= min_days)[0]

    def place_cells_in_any_trial_type(self, min_days: int = 2) -> np.ndarray:
        """Cell indices that were a PC in ANY trial type on >= min_days days.

        Aggregates by taking the max days-present across trial types per cell.
        A cell with 1 day in ABC and 1 day in ABDC has max=1 — does not pass
        min_days=2. Use this if you want cells with persistent place coding.

        Args:
            min_days: Minimum days in any single trial type. Default 2.

        Returns:
            int array of cell indices.
        """
        max_days = np.zeros(self.n_cells, dtype=int)
        for tt in self.trial_types:
            max_days = np.maximum(max_days, self.days_present_for(tt))
        return np.where(max_days >= min_days)[0]

    def summary(self) -> str:
        lines = [
            f'ExperimentPlaceCells: {self.n_cells} cells, {len(self.dates)} sessions',
            f'  Dates: {self.dates[0]} → {self.dates[-1]}',
            f'  Signal: {self.signal_col}',
        ]
        for tt in self.trial_types:
            existed = int(self.trial_type_existed[tt].sum())
            n_pc_any = int((self.days_present_for(tt) >= 1).sum())
            n_pc_2 = int((self.days_present_for(tt) >= 2).sum())
            lines.append(
                f'  {tt}: ran on {existed}/{len(self.dates)} days · '
                f'{n_pc_any} cells PC ≥1 day, {n_pc_2} PC ≥2 days'
            )
        n_any_2 = len(self.place_cells_in_any_trial_type(min_days=2))
        lines.append(f'  Any trial type, ≥2 days: {n_any_2} cells')
        return '\n'.join(lines)


def _hash_params(params: DetectionParams, signal_col: str) -> str:
    """Stable hash of detection params + signal column."""
    payload = json.dumps(
        {**asdict(params), 'signal_col': signal_col},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _discover_session_dates(mouse_dir: Path) -> list[str]:
    """Find all session dates under mouse_dir.

    Looks for subdirectories whose names start with YYYY-MM-DD.
    """
    dates = set()
    for child in mouse_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        # YYYY-MM-DD prefix
        if len(name) >= 10 and name[4] == '-' and name[7] == '-':
            try:
                _ = int(name[:4]), int(name[5:7]), int(name[8:10])
                dates.add(name[:10])
            except ValueError:
                continue
    return sorted(dates)


def compute_experiment_place_cells(
    mouse_dir: Path,
    signal_col: str = 'multi_day_dff',
    params: DetectionParams | None = None,
    force: bool = False,
    save: bool = True,
) -> ExperimentPlaceCells:
    """Run detection on every session for this mouse and aggregate.

    Args:
        mouse_dir: Path to mouse data root.
        signal_col: Signal column for detection. Default 'multi_day_dff'.
        params: DetectionParams. If None, uses defaults.
        force: If True, rebuild even if cached file exists with matching hash.
        save: If True, write result to mouse_dir / CACHE_FILENAME.

    Returns:
        ExperimentPlaceCells.
    """
    if params is None:
        params = DetectionParams()

    cache_path = mouse_dir / CACHE_FILENAME
    params_hash = _hash_params(params, signal_col)

    # Reuse cache if valid
    if cache_path.exists() and not force:
        try:
            cached = load_experiment_place_cells(mouse_dir)
            if cached.params_hash == params_hash:
                print(f'Reusing cached experiment_place_cells '
                      f'(hash={params_hash}, {cached.n_cells} cells, '
                      f'{len(cached.dates)} sessions)')
                return cached
            else:
                print(f'Cache exists but params changed '
                      f'(cached={cached.params_hash}, new={params_hash}). Rebuilding.')
        except Exception as e:
            print(f'Cache load failed ({e}). Rebuilding.')

    dates = _discover_session_dates(mouse_dir)
    if not dates:
        raise FileNotFoundError(f'No session dates found under {mouse_dir}')
    print(f'Found {len(dates)} sessions for {mouse_dir.name}: {dates[0]} → {dates[-1]}')

    # Run detection per session — uses per-session pickle cache so reruns are
    # cheap and only one session's df is in memory at a time.
    per_day_results: dict[str, PlaceFieldResult] = {}
    for date in dates:
        print(f'\n── {date} ──')
        try:
            session_dir = find_session_dir(mouse_dir, date)
        except Exception as e:
            print(f'  Skipping {date}: {e}')
            continue

        try:
            result = detect_place_fields_for_session(
                session_dir,
                signal_col=signal_col,
                params=params,
                force_recompute=force,
            )
            per_day_results[date] = result
        except Exception as e:
            print(f'  Detection failed for {date}: {e}')
            continue

    if not per_day_results:
        raise RuntimeError('No sessions processed successfully.')

    valid_dates = sorted(per_day_results.keys())
    n_cells = per_day_results[valid_dates[0]].n_cells

    # Sanity: all sessions should agree on n_cells (multi-day-registered)
    for d, r in per_day_results.items():
        if r.n_cells != n_cells:
            print(f'  WARNING: {d} has {r.n_cells} cells, expected {n_cells}. '
                  f'Multi-day registration may be inconsistent.')

    # Union of trial types
    trial_types = sorted({tt for r in per_day_results.values() for tt in r.fields})

    # Build presence + centers + trial_type_existed
    # Lazy-import to avoid plotting deps; _strongest_field_position is the same logic
    # used in heatmap sort and remapping analysis.
    from remapping_analysis import _strongest_field_position

    n_days = len(valid_dates)
    presence: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {}
    existed: dict[str, np.ndarray] = {}

    for tt in trial_types:
        pres = np.zeros((n_cells, n_days), dtype=bool)
        ctrs = np.full((n_cells, n_days), np.nan, dtype=np.float32)
        ex = np.zeros(n_days, dtype=bool)
        for di, date in enumerate(valid_dates):
            r = per_day_results[date]
            if tt in r.fields:
                ex[di] = True
                pres[:, di] = r.is_place_cell.get(tt, np.zeros(n_cells, dtype=bool))
                ctrs[:, di] = _strongest_field_position(r.fields[tt], n_cells)
        presence[tt] = pres
        centers[tt] = ctrs
        existed[tt] = ex

    exp_pcs = ExperimentPlaceCells(
        dates=valid_dates,
        trial_types=trial_types,
        n_cells=n_cells,
        presence=presence,
        centers=centers,
        trial_type_existed=existed,
        signal_col=signal_col,
        params=params,
        params_hash=params_hash,
    )

    print('\n' + exp_pcs.summary())

    if save:
        save_experiment_place_cells(exp_pcs, cache_path)
    return exp_pcs


def save_experiment_place_cells(exp_pcs: ExperimentPlaceCells, path: Path):
    """Save to .npz. Inverse of load_experiment_place_cells."""
    path = Path(path)
    save_dict = {
        'dates': np.array(exp_pcs.dates),
        'trial_types': np.array(exp_pcs.trial_types),
        'n_cells': exp_pcs.n_cells,
        'signal_col': exp_pcs.signal_col,
        'params_hash': exp_pcs.params_hash,
        'params_json': json.dumps(asdict(exp_pcs.params)),
    }
    for tt in exp_pcs.trial_types:
        save_dict[f'presence__{tt}'] = exp_pcs.presence[tt]
        save_dict[f'centers__{tt}'] = exp_pcs.centers[tt]
        save_dict[f'existed__{tt}'] = exp_pcs.trial_type_existed[tt]
    np.savez(path, **save_dict)
    print(f'Saved: {path}')


def load_experiment_place_cells(mouse_dir: Path) -> ExperimentPlaceCells:
    """Load cached experiment-wide place cells from mouse_dir / CACHE_FILENAME."""
    path = Path(mouse_dir) / CACHE_FILENAME
    data = np.load(path, allow_pickle=False)

    trial_types = [str(tt) for tt in data['trial_types']]
    presence = {tt: data[f'presence__{tt}'] for tt in trial_types}
    centers = {tt: data[f'centers__{tt}'] for tt in trial_types}
    existed = {tt: data[f'existed__{tt}'] for tt in trial_types}

    params_dict = json.loads(str(data['params_json']))
    params = DetectionParams(**params_dict)

    return ExperimentPlaceCells(
        dates=[str(d) for d in data['dates']],
        trial_types=trial_types,
        n_cells=int(data['n_cells']),
        presence=presence,
        centers=centers,
        trial_type_existed=existed,
        signal_col=str(data['signal_col']),
        params=params,
        params_hash=str(data['params_hash']),
    )


# MULTIDAY DETECTION (legacy — prefer ExperimentPlaceCells for new code)

@dataclass
class MultidayPlaceFieldResult:
    """Results from place field detection across multiple sessions.

    Args:
        per_day: Per-date PlaceFieldResult.
        dates: Sorted list of session dates.
        trial_types: Union of all trial types across days.
        n_cells: Total number of registered cells.
        presence: Boolean presence matrix per trial type, shape (n_cells, n_days).
        centers: Field center position per trial type, shape (n_cells, n_days). NaN if no field.
        union_indices: Cell indices that are place cells on >= min_days.
        params: Detection parameters used.
    """
    per_day: dict[str, PlaceFieldResult] = field(default_factory=dict)
    dates: list[str] = field(default_factory=list)
    trial_types: list[str] = field(default_factory=list)
    n_cells: int = 0
    presence: dict[str, np.ndarray] = field(default_factory=dict)
    centers: dict[str, np.ndarray] = field(default_factory=dict)
    union_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    params: DetectionParams | None = None

    def stability_score(self, trial_type: str) -> np.ndarray:
        """Fraction of days each cell has a place field."""
        return self.presence[trial_type].mean(axis=1)

    def summary(self) -> str:
        lines = [f"Multiday place field detection: {self.n_cells} cells, {len(self.dates)} days"]
        for tt in self.trial_types:
            p = self.presence[tt]
            n_ever = np.any(p, axis=1).sum()
            n_all = np.all(p, axis=1).sum()
            mean_stab = self.stability_score(tt).mean()
            lines.append(
                f"  {tt}: {n_ever} cells with field on ≥1 day, "
                f"{n_all} on all days, mean stability={mean_stab:.2f}"
            )
            day_counts = [f"{d[5:]}: {int(p[:, i].sum())}" for i, d in enumerate(self.dates)]
            lines.append(f"    Per day: {', '.join(day_counts)}")
        lines.append(f"  Union (any type, any day): {len(self.union_indices)} cells")
        return '\n'.join(lines)


def _combine_per_day_results(
    per_day: dict[str, PlaceFieldResult],
    dates: list[str],
    params: DetectionParams,
    min_days: int,
) -> MultidayPlaceFieldResult:
    """Combine per-session detection results into a MultidayPlaceFieldResult.

    Builds presence/centers matrices and computes union_indices.
    """
    n_days = len(dates)
    n_cells = per_day[dates[0]].n_cells
    all_trial_types = sorted(set(
        tt for r in per_day.values() for tt in r.fields
    ))

    presence = {}
    centers_mat = {}

    for tt in all_trial_types:
        pres = np.zeros((n_cells, n_days), dtype=bool)
        ctrs = np.full((n_cells, n_days), np.nan)

        for di, date in enumerate(dates):
            result = per_day[date]
            if tt not in result.fields:
                continue

            pf = result.fields[tt]
            pres[:, di] = pf.has_place_field

            if pf.centers.size > 0:
                cell_ids = pf.cell_id
                intensities = pf.mean_intensity
                for icell in range(n_cells):
                    cell_mask = cell_ids == icell
                    if cell_mask.any():
                        best = np.argmax(intensities[cell_mask])
                        field_idx = np.where(cell_mask)[0][best]
                        ctrs[icell, di] = pf.centers[field_idx, 1]

        presence[tt] = pres
        centers_mat[tt] = ctrs

    any_type_days = np.zeros(n_cells, dtype=int)
    for tt in all_trial_types:
        any_type_days = np.maximum(any_type_days, presence[tt].sum(axis=1))
    union_idx = np.where(any_type_days >= min_days)[0]

    return MultidayPlaceFieldResult(
        per_day=per_day,
        dates=dates,
        trial_types=all_trial_types,
        n_cells=n_cells,
        presence=presence,
        centers=centers_mat,
        union_indices=union_idx,
        params=params,
    )


def detect_multiday_place_fields(
    sessions: dict[str, dict],
    signal_col: str = 'multi_day_dff',
    params: DetectionParams | None = None,
    min_days: int = 1,
) -> MultidayPlaceFieldResult:
    """Detect place fields independently per session, then combine across days.

    NOTE: takes sessions ALREADY LOADED in memory. For 8+ sessions this can OOM.
    Use detect_multiday_place_fields_cached() instead — it loads sessions one
    at a time and caches results to disk.

    Bin size is taken from each session's metadata; sessions without
    ``bin_size_cm`` in their metadata are skipped with a warning.

    Args:
        sessions: From load_multiday_sessions(). Each value has keys:
            'data', 'config', 'session_data', 'metadata'.
        signal_col: Column containing neural signals.
        params: Detection parameters. Uses defaults if None.
        min_days: Minimum number of days a cell must have a field to be
            included in union_indices. Default 1 (any day).
    """
    if params is None:
        params = DetectionParams()

    dates = sorted(sessions.keys())

    per_day = {}
    for date in dates:
        s = sessions[date]
        meta = s.get('metadata') or {}
        if 'bin_size_cm' not in meta:
            print(f"  WARNING: {date} metadata lacks bin_size_cm — skipping.")
            continue
        print(f"\n--- {date} ---")
        per_day[date] = detect_place_fields(
            s['data'], s['config'],
            signal_col=signal_col,
            bin_size_cm=int(meta['bin_size_cm']),
            params=params,
            metadata=meta,
        )

    multiday_result = _combine_per_day_results(per_day, dates, params, min_days)
    print(f"\n{multiday_result.summary()}")
    return multiday_result


def detect_multiday_place_fields_cached(
    mouse_dir: Path,
    dates: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    signal_col: str = 'multi_day_dff',
    params: DetectionParams | None = None,
    min_days: int = 1,
    force_recompute: bool = False,
) -> MultidayPlaceFieldResult:
    """Memory-bounded multiday place field detection with on-disk caching.

    Loops sessions one at a time, calling detect_place_fields_for_session() (which
    auto-caches per-session results). Each session's df is loaded only on cache
    miss and freed before the next session is processed. Bin size is always
    pulled from per-session metadata yamls.
    """
    mouse_dir = Path(mouse_dir)

    if dates is not None and date_range is not None:
        raise ValueError("Provide either dates or date_range, not both.")

    if date_range is not None:
        start, end = date_range
        dates = sorted(set(
            d.name[:10]
            for d in mouse_dir.iterdir()
            if d.is_dir() and len(d.name) >= 10 and start <= d.name[:10] <= end
        ))
        print(f"Found {len(dates)} sessions in range {start} to {end}: {dates}")

    if not dates:
        raise ValueError("No dates provided or discovered.")

    if params is None:
        params = DetectionParams()

    per_day = {}
    for date in sorted(dates):
        print(f"\n--- {date} ---")
        try:
            session_dir = find_session_dir(mouse_dir, date)
            per_day[date] = detect_place_fields_for_session(
                session_dir,
                signal_col=signal_col,
                params=params,
                force_recompute=force_recompute,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  WARNING: skipping {date}: {e}")

    if not per_day:
        raise ValueError("No sessions could be processed.")

    sorted_dates = sorted(per_day.keys())
    multiday_result = _combine_per_day_results(per_day, sorted_dates, params, min_days)
    print(f"\n{multiday_result.summary()}")
    return multiday_result


def save_multiday_result(result: MultidayPlaceFieldResult, path: Path):
    """Save multiday detection summary to .npz (presence + centers + union)."""
    path = Path(path)
    save_dict = {
        'dates': np.array(result.dates),
        'trial_types': np.array(result.trial_types),
        'n_cells': result.n_cells,
        'union_indices': result.union_indices,
    }
    for tt in result.trial_types:
        save_dict[f'{tt}_presence'] = result.presence[tt]
        save_dict[f'{tt}_centers'] = result.centers[tt]
    np.savez(path, **save_dict)
    print(f"Saved: {path}")


def load_multiday_result(path: Path) -> MultidayPlaceFieldResult:
    """Load multiday detection summary from .npz. per_day will be empty."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    trial_types = [str(tt) for tt in data['trial_types']]
    presence = {tt: data[f'{tt}_presence'] for tt in trial_types}
    centers = {tt: data[f'{tt}_centers'] for tt in trial_types}
    return MultidayPlaceFieldResult(
        dates=[str(d) for d in data['dates']],
        trial_types=trial_types,
        n_cells=int(data['n_cells']),
        presence=presence,
        centers=centers,
        union_indices=data['union_indices'],
    )


# FIELD TRACKING ACROSS DAYS
#
# Continuous per-day quantification at positions where a cell ever had a strict
# detection. Replaces the binary "is/isn't a place field on day X" framing with
# continuous metrics (amplitude, position, prominence) that downstream code can
# threshold per analysis. The strict per-session detector is used only to pick
# anchor positions; tracking measurements never apply a detection threshold.


@dataclass
class TrackedFields:
    """Continuous per-day response metrics for fields tracked across sessions.

    A 'tracked field' is one (cell × trial_type × anchor_position) tuple
    representing a location where the cell had at least one strict detection
    across the analyzed dates. For each day, we report the actual measured
    response in a ±window_cm/2 window around the anchor — never a binary
    'is/isn't a field' label.

    Per-row interpretation (one row = one tracked field):
        - anchor_positions_cm[i] is the canonical position (mean centroid of
          all strict-detected days that joined this tracked field).
        - positions_cm[i, d] is the actual peak position within the window on
          day d (the per-segment view). drifts_cm[i, d] = positions_cm - anchor.
        - amplitudes[i, d] and prominences[i, d] are continuous response
          metrics; strict_detected[i, d] flags whether the strict per-session
          detector also flagged a field for this cell on this day.

    NaN in amplitudes/positions/prominences means the trial type was absent on
    that day, or the cell index was out of range — not 'no field.'

    Args:
        trial_type: Trial type these tracked fields belong to.
        cell_ids: Original cell indices, shape (n_tracked,).
        anchor_positions_cm: Mean centroid per tracked field, shape (n_tracked,).
        dates: Sorted list of session dates, length n_days.
        amplitudes: Peak amplitude in window, shape (n_tracked, n_days).
        positions_cm: Per-day peak position, shape (n_tracked, n_days).
        prominences: Peak minus window minimum, shape (n_tracked, n_days).
        drifts_cm: positions_cm minus anchor, shape (n_tracked, n_days).
        strict_detected: Whether the strict detector flagged a field at this
            anchor on each day, shape (n_tracked, n_days), bool.
        window_cm: Total window width used for measurements.
    """
    trial_type: str
    cell_ids: np.ndarray
    anchor_positions_cm: np.ndarray
    dates: list[str]
    amplitudes: np.ndarray
    positions_cm: np.ndarray
    prominences: np.ndarray
    drifts_cm: np.ndarray
    strict_detected: np.ndarray
    window_cm: float


def track_fields_across_days(
    multiday: MultidayPlaceFieldResult,
    window_cm: float = 10.0,
    smooth_sigma: float = 1.5,
) -> dict[str, TrackedFields]:
    """Build continuous per-day response trajectories at every anchor position
    that strict detection ever flagged for each (cell, trial_type) pair.

    Per trial type, this clusters all per-day field centroids by cell × position
    into tracked fields, then on EVERY day measures the response inside a
    ±window_cm/2 window around the anchor — independent of whether that day
    passed strict detection. The output is therefore lossless with respect to
    'rate remapping': a strong field on day A and a weakened-but-present field
    on day B both show up in the same row, with comparable continuous metrics.

    The strict per-session detections are used solely to nominate the anchor
    positions worth tracking. They never gate the per-day measurements.

    Clustering rule: two detections in the same cell join the same tracked
    field if the second's position is within `window_cm` of the running mean
    of the cluster (greedy 1-D agglomeration after sorting by position). A
    cell with fields at e.g. 50 cm and 150 cm yields two distinct tracked
    fields.

    Args:
        multiday: Strict per-session detection results, one per date.
        window_cm: Total window width around the anchor for per-day
            measurement. The peak position and amplitude are taken within
            ±window_cm/2 of the anchor. 10 cm is a good default for 5 cm bins
            (roughly two bins each side, accommodates ~10 cm of drift).
        smooth_sigma: Gaussian smoothing in bins applied to each day's binF
            before measurement. Matches the detection-time smoothing so the
            measured curve is what the detector saw.

    Returns:
        Dict keyed by trial_type, each value a TrackedFields with the
        continuous trajectories.
    """
    from collections import defaultdict
    from scipy.ndimage import gaussian_filter1d as _gauss1d

    results: dict[str, TrackedFields] = {}
    n_days = len(multiday.dates)

    for tt in multiday.trial_types:
        # Step 1: collect every strict detection for this trial type as
        # (cell_idx -> [(date_idx, position_cm), ...]). pf.centers[:, 1] is the
        # weighted centroid in cm; pf.cell_id is the matching cell index.
        detections_per_cell: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for di, date in enumerate(multiday.dates):
            pfr = multiday.per_day.get(date)
            if pfr is None or tt not in pfr.fields:
                continue
            pf = pfr.fields[tt]
            if pf.centers.size == 0:
                continue
            cell_ids = pf.cell_id
            positions = pf.centers[:, 1]
            for c_id, pos in zip(cell_ids, positions):
                detections_per_cell[int(c_id)].append((di, float(pos)))

        # Step 2: cluster each cell's detections into tracked fields. Greedy
        # 1-D agglomeration: a sorted list of positions is walked left-to-right;
        # a new detection joins the current cluster if it's within window_cm of
        # the cluster's running mean, else it starts a new cluster.
        tracked_meta: list[dict] = []
        for cell_idx, dets in detections_per_cell.items():
            dets_sorted = sorted(dets, key=lambda x: x[1])
            clusters: list[list[tuple[int, float]]] = []
            for di, pos in dets_sorted:
                if clusters:
                    current_mean = float(np.mean([p for _, p in clusters[-1]]))
                    if abs(pos - current_mean) <= window_cm:
                        clusters[-1].append((di, pos))
                        continue
                clusters.append([(di, pos)])
            for cluster in clusters:
                tracked_meta.append({
                    'cell_idx': cell_idx,
                    'anchor': float(np.mean([p for _, p in cluster])),
                    'strict_days': {di for di, _ in cluster},
                })

        n_tracked = len(tracked_meta)
        amplitudes = np.full((n_tracked, n_days), np.nan)
        positions_cm = np.full((n_tracked, n_days), np.nan)
        prominences = np.full((n_tracked, n_days), np.nan)
        strict_detected = np.zeros((n_tracked, n_days), dtype=bool)

        # Step 3: per day, pull the cached binF, smooth, and measure each
        # tracked field inside its ±window_cm/2 window.
        for di, date in enumerate(multiday.dates):
            pfr = multiday.per_day.get(date)
            if pfr is None or tt not in pfr.fields:
                continue
            pf = pfr.fields[tt]
            binF = pf.binF
            if binF.size == 0:
                continue
            bin_size_cm = pf.bin_size_cm
            n_bins = binF.shape[1]
            smoothed = _gauss1d(binF, sigma=smooth_sigma, axis=1, mode='wrap')

            half_window_bins = int(np.ceil((window_cm / 2) / bin_size_cm))

            for ti, tf in enumerate(tracked_meta):
                cell_idx = tf['cell_idx']
                if cell_idx >= binF.shape[0]:
                    continue
                # Anchor cm -> bin (bin center at i*bin + bin/2, so subtract half).
                anchor_bin = int(round(tf['anchor'] / bin_size_cm - 0.5))
                left = max(0, anchor_bin - half_window_bins)
                right = min(n_bins, anchor_bin + half_window_bins + 1)
                if right <= left:
                    continue
                window_curve = smoothed[cell_idx, left:right]

                peak_local = int(np.argmax(window_curve))
                peak_bin = left + peak_local
                peak_amp = float(window_curve[peak_local])
                peak_pos_cm = peak_bin * bin_size_cm + (bin_size_cm / 2)
                # Local prominence: peak minus minimum within the same window.
                # This is intentionally window-local — global prominence would
                # leak information about distant peaks.
                window_min = float(np.min(window_curve))

                amplitudes[ti, di] = peak_amp
                positions_cm[ti, di] = peak_pos_cm
                prominences[ti, di] = peak_amp - window_min
                strict_detected[ti, di] = di in tf['strict_days']

        anchors = np.array([tf['anchor'] for tf in tracked_meta])
        drifts_cm = positions_cm - anchors[:, None] if n_tracked else np.empty((0, n_days))

        results[tt] = TrackedFields(
            trial_type=tt,
            cell_ids=np.array([tf['cell_idx'] for tf in tracked_meta], dtype=int),
            anchor_positions_cm=anchors,
            dates=list(multiday.dates),
            amplitudes=amplitudes,
            positions_cm=positions_cm,
            prominences=prominences,
            drifts_cm=drifts_cm,
            strict_detected=strict_detected,
            window_cm=window_cm,
        )

    return results


def zone_labels_per_day(
    exp_pcs: ExperimentPlaceCells,
    trial_type: str,
    config: dict,
    abdc_relabel: dict | None = None,
) -> np.ndarray:
    """Per-cell, per-day zone label for a given trial type.

    Uses the cached field centers and the trial type's cue layout to assign each
    (cell, day) to a zone label like 'A', '0a', 'D', etc. Days where the trial
    type didn't run are labeled 'no_run'. Cells with no field that day are 'none'.

    Args:
        exp_pcs: Loaded ExperimentPlaceCells.
        trial_type: Which trial type to label.
        config: Experiment config (for cue layout).
        abdc_relabel: Optional renaming, e.g. {'C': "C'", '0c': "0c'"} for ABDC.

    Returns:
        np.ndarray of shape (n_cells, n_days), dtype object (strings).
    """
    from remapping_analysis import _build_zone_lookup, _assign_zone

    if abdc_relabel is None and trial_type == 'ABDC':
        abdc_relabel = {'C': "C'", '0c': "0c'"}
    zones = _build_zone_lookup(config, trial_type, relabel=abdc_relabel)

    centers = exp_pcs.centers[trial_type]              # (n_cells, n_days)
    existed = exp_pcs.trial_type_existed[trial_type]   # (n_days,)
    n_cells, n_days = centers.shape

    labels = np.empty((n_cells, n_days), dtype=object)
    for di in range(n_days):
        if not existed[di]:
            labels[:, di] = 'no_run'
            continue
        for ci in range(n_cells):
            pos = centers[ci, di]
            labels[ci, di] = _assign_zone(pos, zones) or 'none'
    return labels


# DEFAULT ZONE COLOR PALETTE (consistent with plot_remapping)
_DEFAULT_ZONE_COLORS = {
    'A':    '#1f77b4',  # blue
    '0a':   '#ff7f0e',  # orange
    'B':    '#2ca02c',  # green
    '0b':   '#d62728',  # red
    'C':    '#9467bd',  # purple
    '0c':   '#8c564b',  # brown
    "C'":   '#9467bd',
    "0c'":  '#8c564b',
    'D':    '#e377c2',  # pink
    '0d':   '#bcbd22',  # olive
    'none': '#cccccc',  # light gray
    'no_run': '#000000',  # black for masked-out days
}


def plot_zone_trajectory(
    exp_pcs: ExperimentPlaceCells,
    config: dict,
    cells: np.ndarray | None = None,
    trial_type: str = 'ABDC',
    sort_by_day: int | str = -1,
    min_days_pc: int = 0,
    mouse_id: str | None = None,
    figsize: tuple = (12, 8),
    show: bool = True,
) -> 'plt.Figure':
    """Per-cell zone trajectory raster.

    Rows = cells, cols = days. Cell color = which zone the strongest field was
    in that day (categorical). Days where the trial type didn't run are masked
    out in black.

    Use this to inspect when cells become / stop being place cells, and how
    their zone identity changes across the experiment.

    Args:
        exp_pcs: Loaded ExperimentPlaceCells.
        config: Experiment config.
        cells: Cells to plot. Default: PCs in any trial type, ≥2 days.
        trial_type: Which trial type's labels to plot.
        sort_by_day: Day index or date string for sort. Default -1 (last day).
        min_days_pc: Filter out cells that were PC in this trial type on fewer
            than this many days. Useful for cutting noise. Default 0 (no filter).
        mouse_id: For title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    if cells is None:
        cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)

    # Filter by min_days_pc using presence matrix
    if min_days_pc > 0:
        existed = exp_pcs.trial_type_existed[trial_type]
        days_pc = exp_pcs.presence[trial_type][:, existed].sum(axis=1)
        keep_mask = days_pc[cells] >= min_days_pc
        cells = cells[keep_mask]

    labels = zone_labels_per_day(exp_pcs, trial_type, config)
    labels = labels[cells, :]

    # Resolve sort_by_day to an index
    if isinstance(sort_by_day, str):
        sort_idx = exp_pcs.dates.index(sort_by_day)
    else:
        sort_idx = sort_by_day if sort_by_day >= 0 else len(exp_pcs.dates) + sort_by_day

    # Categorical colormap based on zones present in this plot
    unique_zones = sorted(set(labels.flatten()))
    # Stable order: shared zones first, then new zones, then none/no_run
    zone_order_pref = ['A', '0a', 'B', '0b', 'C', '0c', "C'", "0c'", 'D', '0d',
                       'none', 'no_run']
    unique_zones = [z for z in zone_order_pref if z in unique_zones] + \
                   [z for z in unique_zones if z not in zone_order_pref]
    zone_to_int = {z: i for i, z in enumerate(unique_zones)}
    int_im = np.vectorize(zone_to_int.get)(labels).astype(int)

    colors = [_DEFAULT_ZONE_COLORS.get(z, '#666666') for z in unique_zones]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(unique_zones) + 1) - 0.5, len(unique_zones))

    # Sort cells by zone on sort_by_day, with no_run/none at bottom
    sort_priority = []
    for z in unique_zones:
        if z == 'no_run':
            sort_priority.append(2)
        elif z == 'none':
            sort_priority.append(1)
        else:
            sort_priority.append(0)
    primary_key = np.array([sort_priority[zone_to_int[labels[i, sort_idx]]]
                            for i in range(labels.shape[0])])
    secondary_key = int_im[:, sort_idx]
    order = np.lexsort((secondary_key, primary_key))
    int_im = int_im[order, :]

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    ax.imshow(int_im, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    ax.set_xticks(range(len(exp_pcs.dates)))
    ax.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=9)
    ax.set_xlabel('Date (mm-dd)')
    ax.set_ylabel(f'Cell # (n={len(cells)}, sorted by {exp_pcs.dates[sort_idx]})')

    title_bits = []
    if mouse_id:
        title_bits.append(str(mouse_id))
    title_bits.append(f'{trial_type} zone trajectory')
    title_bits.append(f'sorted by {exp_pcs.dates[sort_idx]}')
    if min_days_pc > 0:
        title_bits.append(f'≥{min_days_pc} days as PC')
    ax.set_title(' — '.join(title_bits), fontweight='bold')

    handles = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.3,
                     label=z) for i, z in enumerate(unique_zones)]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, title='Zone', frameon=False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# CELLS GAINING / LOSING PC STATUS

def plot_state_change_trajectories(
    exp_pcs: ExperimentPlaceCells,
    config: dict,
    trial_type: str = 'ABC',
    direction: str = 'dropout',
    reference_day: int | str = -1,
    cells: np.ndarray | None = None,
    min_days_pc: int = 2,
    mouse_id: str | None = None,
    figsize: tuple = (12, 8),
    show: bool = True,
) -> 'plt.Figure':
    """Trajectories of cells that change PC status by a reference day.

    Two modes:
        - 'dropout': cells that ARE 'none' on reference_day but were PCs on
          enough other days (min_days_pc). Sorted by their dominant zone
          across earlier days. Shows what these cells used to code.
        - 'recruitment': cells that are NOT 'none' on reference_day and were
          'none' on the first day. Sorted by their zone on reference_day.
          Shows what 'silent' cells became.

    Args:
        exp_pcs: Loaded ExperimentPlaceCells.
        config: Experiment config.
        trial_type: Which trial type to analyze. Default 'ABC'.
        direction: 'dropout' or 'recruitment'.
        reference_day: Day index or date string. The "end state" day. Default -1
            (last day).
        cells: Optional starting pool. Default: all cells (will be filtered
            internally).
        min_days_pc: For dropout, require cells to have been a PC ≥N days. Cuts
            cells that were never really PCs anyway. Default 2.
        mouse_id: For title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    if direction not in ('dropout', 'recruitment'):
        raise ValueError(f"direction must be 'dropout' or 'recruitment', got {direction!r}")

    # Resolve reference_day to an index
    if isinstance(reference_day, str):
        ref_idx = exp_pcs.dates.index(reference_day)
    else:
        ref_idx = reference_day if reference_day >= 0 else len(exp_pcs.dates) + reference_day

    labels = zone_labels_per_day(exp_pcs, trial_type, config)
    n_cells_total = labels.shape[0]

    if cells is None:
        cells = np.arange(n_cells_total)

    # Filter by direction
    existed = exp_pcs.trial_type_existed[trial_type]
    days_pc = exp_pcs.presence[trial_type][:, existed].sum(axis=1)

    if direction == 'dropout':
        # 'none' on ref day, but PC ≥min_days on other days
        is_none_ref = labels[:, ref_idx] == 'none'
        target_mask = is_none_ref & (days_pc >= min_days_pc)
    else:  # recruitment
        # 'none' on first day (where trial type ran), real zone on ref day
        first_existed_idx = int(np.argmax(existed))
        is_none_first = labels[:, first_existed_idx] == 'none'
        is_real_ref = (labels[:, ref_idx] != 'none') & (labels[:, ref_idx] != 'no_run')
        target_mask = is_none_first & is_real_ref

    # Intersect with cells pool
    cells = cells[np.isin(cells, np.where(target_mask)[0])]
    if len(cells) == 0:
        raise RuntimeError(f"No cells match direction={direction!r} criteria.")

    cell_labels = labels[cells, :]

    # Canonical zone color setup
    zone_order_pref = ['A', '0a', 'B', '0b', 'C', '0c', "C'", "0c'", 'D', '0d',
                       'none', 'no_run']
    unique_zones_present = sorted(set(cell_labels.flatten()))
    unique_zones = [z for z in zone_order_pref if z in unique_zones_present] + \
                   [z for z in unique_zones_present if z not in zone_order_pref]
    zone_to_int = {z: i for i, z in enumerate(unique_zones)}
    int_im = np.vectorize(zone_to_int.get)(cell_labels).astype(int)
    colors = [_DEFAULT_ZONE_COLORS.get(z, '#666666') for z in unique_zones]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(unique_zones) + 1) - 0.5, len(unique_zones))

    # Sort cells: by their dominant non-none zone on the relevant pre/post-ref days
    if direction == 'dropout':
        # Look at days BEFORE ref to find dominant zone
        pre_labels = cell_labels[:, :ref_idx]
        sort_zones = _dominant_zone_per_row(pre_labels, exclude={'none', 'no_run'})
    else:
        # Recruitment: zone on ref day directly
        sort_zones = cell_labels[:, ref_idx]

    primary = np.array([
        2 if z in ('no_run', 'none') else 0 for z in sort_zones
    ])
    secondary = np.array([zone_to_int.get(z, len(unique_zones)) for z in sort_zones])
    order = np.lexsort((secondary, primary))
    int_im = int_im[order, :]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    ax.imshow(int_im, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
    ax.set_xticks(range(len(exp_pcs.dates)))
    ax.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=9)
    ax.set_xlabel('Date (mm-dd)')

    sort_label = ('dominant zone before ref day' if direction == 'dropout'
                  else f'zone on {exp_pcs.dates[ref_idx]}')
    ax.set_ylabel(f'Cell # (n={len(cells)}, sorted by {sort_label})')

    # Vertical line at ref day
    ax.axvline(ref_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    title_bits = []
    if mouse_id:
        title_bits.append(str(mouse_id))
    title_bits.append(f'{trial_type} {direction} trajectories')
    title_bits.append(f'ref = {exp_pcs.dates[ref_idx]}')
    if direction == 'dropout':
        title_bits.append(f'PC ≥{min_days_pc} days')
    ax.set_title('  ·  '.join(title_bits), fontweight='bold')

    handles = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.3,
                     label=z) for i, z in enumerate(unique_zones)]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, title='Zone', frameon=False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _dominant_zone_per_row(labels_2d: np.ndarray, exclude: set) -> np.ndarray:
    """For each row, return the most common label (mode) excluding given set.

    Falls back to 'none' if all values in a row are in exclude.
    """
    n_rows = labels_2d.shape[0]
    out = np.empty(n_rows, dtype=object)
    for ri in range(n_rows):
        row = labels_2d[ri, :]
        kept = [x for x in row if x not in exclude]
        if not kept:
            out[ri] = 'none'
            continue
        # mode
        vals, counts = np.unique(kept, return_counts=True)
        out[ri] = vals[np.argmax(counts)]
    return out


def plot_dual_track_trajectory(
    exp_pcs: ExperimentPlaceCells,
    config: dict,
    cells: np.ndarray | None = None,
    abc_trial_type: str = 'ABC',
    abdc_trial_type: str = 'ABDC',
    sort_by_day: int | str = -1,
    sort_by_track: str = 'ABDC',
    mouse_id: str | None = None,
    figsize: tuple = (14, 10),
    show: bool = True,
) -> 'plt.Figure':
    """Three-panel view of identity dynamics across pre- and post-extension phases.

    - Panel 1 (top): ABC zone trajectory across all days.
    - Panel 2 (middle): ABDC zone trajectory, only on days where ABDC ran.
    - Panel 3 (bottom): per-day population match status — what fraction of cells
      have ABC == ABDC, ABC != ABDC, ABC-only, ABDC-only, or both none.

    Same cell-sort across panels 1 and 2 — read vertically to compare a cell's
    ABC and ABDC fields on the same day. The match-status panel summarizes that
    comparison for the population.

    Args:
        exp_pcs: Loaded ExperimentPlaceCells.
        config: Experiment config.
        cells: Cells to plot. Default: PCs in any trial type, ≥2 days.
        abc_trial_type: Pre-extension trial type. Default 'ABC'.
        abdc_trial_type: Post-extension trial type. Default 'ABDC'.
        sort_by_day: Day index or date string for sort. Default -1 (last day).
        sort_by_track: Which track's zone to sort by ('ABC' or 'ABDC'). Default
            'ABDC' so cells that ended up at D/0d/C' cluster together.
        mouse_id: For title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    if cells is None:
        cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)

    abc_labels_full = zone_labels_per_day(exp_pcs, abc_trial_type, config)[cells, :]
    abdc_labels_full = zone_labels_per_day(exp_pcs, abdc_trial_type, config)[cells, :]
    abdc_existed = exp_pcs.trial_type_existed[abdc_trial_type]

    # Resolve sort_by_day
    if isinstance(sort_by_day, str):
        sort_idx = exp_pcs.dates.index(sort_by_day)
    else:
        sort_idx = sort_by_day if sort_by_day >= 0 else len(exp_pcs.dates) + sort_by_day

    # Build canonical zone order across both tracks
    zone_order_pref = ['A', '0a', 'B', '0b', 'C', '0c', "C'", "0c'", 'D', '0d',
                       'none', 'no_run']
    all_zones = sorted(set(abc_labels_full.flatten()) | set(abdc_labels_full.flatten()))
    unique_zones = [z for z in zone_order_pref if z in all_zones] + \
                   [z for z in all_zones if z not in zone_order_pref]
    zone_to_int = {z: i for i, z in enumerate(unique_zones)}
    colors = [_DEFAULT_ZONE_COLORS.get(z, '#666666') for z in unique_zones]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(unique_zones) + 1) - 0.5, len(unique_zones))

    # Sort cells: primary by sort_by_track zone on sort_by_day, push none/no_run to bottom
    if sort_by_track.upper() == 'ABC':
        sort_labels = abc_labels_full[:, sort_idx]
    else:
        sort_labels = abdc_labels_full[:, sort_idx]
    priority = np.array([
        2 if z == 'no_run' else (1 if z == 'none' else 0)
        for z in sort_labels
    ])
    secondary = np.array([zone_to_int[z] for z in sort_labels])
    order = np.lexsort((secondary, priority))

    abc_int = np.vectorize(zone_to_int.get)(abc_labels_full).astype(int)[order, :]
    abdc_int = np.vectorize(zone_to_int.get)(abdc_labels_full).astype(int)[order, :]

    # Find first ABDC-existed day for cropping middle panel
    abdc_first = int(np.argmax(abdc_existed)) if abdc_existed.any() else len(exp_pcs.dates)
    abdc_int_cropped = abdc_int[:, abdc_first:]

    # Match-status per post-extension day
    overlap_days = np.where(abdc_existed)[0]
    n_overlap = len(overlap_days)
    match_cats = ['same', 'diff', 'abc_only', 'abdc_only', 'both_none']
    match_colors = {
        'same':       '#2ca02c',
        'diff':       '#d62728',
        'abc_only':   '#1f77b4',
        'abdc_only':  '#ff7f0e',
        'both_none':  '#cccccc',
    }
    match_props = np.zeros((len(match_cats), n_overlap))
    n_sel = len(cells)
    for j, di in enumerate(overlap_days):
        abc_col = abc_labels_full[:, di]
        abdc_col = abdc_labels_full[:, di]
        for ci, cat in enumerate(match_cats):
            if cat == 'same':
                m = (abc_col == abdc_col) & (abc_col != 'none')
            elif cat == 'diff':
                m = (abc_col != abdc_col) & (abc_col != 'none') & (abdc_col != 'none')
            elif cat == 'abc_only':
                m = (abc_col != 'none') & (abdc_col == 'none')
            elif cat == 'abdc_only':
                m = (abc_col == 'none') & (abdc_col != 'none')
            else:
                m = (abc_col == 'none') & (abdc_col == 'none')
            match_props[ci, j] = m.sum() / n_sel * 100

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, dpi=140,
        gridspec_kw={'height_ratios': [4, 4, 1.4], 'hspace': 0.3},
    )
    ax_abc, ax_abdc, ax_match = axes

    ax_abc.imshow(abc_int, cmap=cmap, norm=norm, aspect='auto', interpolation='none',
                  extent=[-0.5, len(exp_pcs.dates) - 0.5, len(cells), 0])
    ax_abc.set_title(f'{abc_trial_type} zone trajectory  ·  all days  ·  '
                     f'sorted by {sort_by_track} on {exp_pcs.dates[sort_idx]}',
                     fontsize=11, fontweight='bold')
    ax_abc.set_ylabel(f'Cell #  (n={len(cells)})')
    ax_abc.set_xticks(range(len(exp_pcs.dates)))
    ax_abc.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=8)
    ax_abc.set_xlim(-0.5, len(exp_pcs.dates) - 0.5)

    if n_overlap > 0:
        ax_abdc.imshow(abdc_int_cropped, cmap=cmap, norm=norm, aspect='auto',
                       interpolation='none',
                       extent=[abdc_first - 0.5, len(exp_pcs.dates) - 0.5,
                               len(cells), 0])
    ax_abdc.set_title(f'{abdc_trial_type} zone trajectory  ·  post-extension only',
                      fontsize=11, fontweight='bold')
    ax_abdc.set_ylabel(f'Cell #  (n={len(cells)})')
    ax_abdc.set_xticks(range(len(exp_pcs.dates)))
    ax_abdc.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=8)
    ax_abdc.set_xlim(-0.5, len(exp_pcs.dates) - 0.5)

    if abdc_first < len(exp_pcs.dates):
        for ax in (ax_abc, ax_abdc, ax_match):
            ax.axvline(abdc_first - 0.5, color='red', linestyle='--',
                       linewidth=1.2, alpha=0.7, zorder=5)

    if n_overlap > 0:
        bottoms = np.zeros(n_overlap)
        for ci, cat in enumerate(match_cats):
            ax_match.bar(overlap_days, match_props[ci, :], bottom=bottoms,
                         color=match_colors[cat],
                         label=cat.replace('_', ' '),
                         width=0.85, edgecolor='white', linewidth=0.3)
            bottoms += match_props[ci, :]
    ax_match.set_xticks(range(len(exp_pcs.dates)))
    ax_match.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=8)
    ax_match.set_xlabel('Date (mm-dd)')
    ax_match.set_ylabel('% of cells')
    ax_match.set_xlim(-0.5, len(exp_pcs.dates) - 0.5)
    ax_match.set_ylim(0, 100)
    ax_match.set_title(f'{abc_trial_type} vs {abdc_trial_type} match status per day',
                       fontsize=10, fontweight='bold')
    ax_match.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8,
                    frameon=False, title='Status')

    handles = [Patch(facecolor=colors[i], edgecolor='black', linewidth=0.3,
                     label=z) for i, z in enumerate(unique_zones)]
    ax_abc.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                  fontsize=8, title='Zone', frameon=False)

    title_bits = []
    if mouse_id:
        title_bits.append(str(mouse_id))
    title_bits.append('Dual-track identity dynamics')
    fig.suptitle(' — '.join(title_bits), fontsize=13, fontweight='bold', y=0.995)

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.86)
    if show:
        plt.show()
    return fig


def plot_zone_proportions(
    exp_pcs: ExperimentPlaceCells,
    config: dict,
    cells: np.ndarray | None = None,
    trial_type: str = 'ABDC',
    mouse_id: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
) -> 'plt.Figure':
    """Stream graph: zone proportions across days for a cell population.

    x = day, y = % of selected cells in each zone. Days where trial_type didn't
    run are blank.

    Use this to see population-level dynamics, e.g. when cells start coding D.

    Args:
        exp_pcs: Loaded ExperimentPlaceCells.
        config: Experiment config.
        cells: Cell indices. Default: all PCs in any trial type, ≥2 days.
        trial_type: Which trial type's labels to plot.
        mouse_id: For the title.
        figsize: Figure size.
        show: Call plt.show().

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if cells is None:
        cells = exp_pcs.place_cells_in_any_trial_type(min_days=2)

    labels = zone_labels_per_day(exp_pcs, trial_type, config)[cells, :]
    n_cells_sel, n_days = labels.shape
    existed = exp_pcs.trial_type_existed[trial_type]

    zone_order_pref = ['A', '0a', 'B', '0b', 'C', '0c', "C'", "0c'", 'D', '0d',
                       'none']
    zones_in_data = sorted({z for z in labels.flatten() if z != 'no_run'})
    zones = [z for z in zone_order_pref if z in zones_in_data]

    # Build proportion matrix: (n_zones, n_days)
    props = np.zeros((len(zones), n_days))
    for di in range(n_days):
        if not existed[di]:
            continue
        col = labels[:, di]
        for zi, z in enumerate(zones):
            props[zi, di] = (col == z).sum() / n_cells_sel * 100

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    x = np.arange(n_days)
    colors = [_DEFAULT_ZONE_COLORS.get(z, '#666666') for z in zones]
    ax.stackplot(x, props, labels=zones, colors=colors, alpha=0.95,
                 edgecolor='white', linewidth=0.3)

    # Mask out no-run days with hatched overlay
    for di in range(n_days):
        if not existed[di]:
            ax.axvspan(di - 0.5, di + 0.5, color='black', alpha=0.85, zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in exp_pcs.dates], rotation=45, fontsize=9)
    ax.set_xlabel('Date (mm-dd)')
    ax.set_ylabel(f'% of cells (n={len(cells)})')
    ax.set_xlim(-0.5, n_days - 0.5)
    ax.set_ylim(0, 100)

    title_bits = []
    if mouse_id:
        title_bits.append(str(mouse_id))
    title_bits.append(f'{trial_type} zone proportions over time')
    ax.set_title(' — '.join(title_bits), fontweight='bold')

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
              title='Zone', frameon=False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    mouse_id = '14'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    params = DetectionParams(smooth_sigma=0, signal_threshold=0.3)

    # Compute (or reuse cache if params haven't changed)
    exp_pcs = compute_experiment_place_cells(
        mouse_dir, signal_col='multi_day_dff', params=params,
    )

    print()
    print(exp_pcs.summary())

    cells_2d = exp_pcs.place_cells_in_any_trial_type(min_days=2)
    cells_abc_2d = exp_pcs.place_cells_in('ABC', min_days=2)
    cells_abdc_2d = exp_pcs.place_cells_in('ABDC', min_days=2)
    print(f'\nCell sets:')
    print(f'  any trial type, ≥2 days: {len(cells_2d)}')
    print(f'  ABC ≥2 days: {len(cells_abc_2d)}')
    print(f'  ABDC ≥2 days: {len(cells_abdc_2d)}')

    # Need exp_config for the plotting functions; load from any session
    from df_processing import find_session_dir, load_session_context
    session_dir = find_session_dir(mouse_dir, exp_pcs.dates[-1])
    _, exp_config = load_session_context(session_dir)

    import matplotlib.pyplot as plt
    plot_dual_track_trajectory(exp_pcs, exp_config, cells=cells_2d,
                                mouse_id=mouse_id)
    plt.show()

    plot_zone_proportions(exp_pcs, exp_config, cells=cells_2d,
                          trial_type='ABDC', mouse_id=mouse_id)
    plt.show()

    # Sweep min_days_pc on ABC-sorted-by-ABC, to see how stability looks
    # at different stringency thresholds. Start at 0 (no filter) and go up.
    for thresh in (0, 2, 4, 7):
        plot_zone_trajectory(
            exp_pcs, exp_config, cells=cells_2d, trial_type='ABC',
            sort_by_day=-1, min_days_pc=thresh, mouse_id=mouse_id,
        )
        plt.show()

    # Cells that were PCs at some point but became 'none' by the last day
    plot_state_change_trajectories(
        exp_pcs, exp_config, trial_type='ABC', direction='dropout',
        reference_day=-1, min_days_pc=5, mouse_id=mouse_id,
    )
    plt.show()

    # Cells that were 'none' on the first day but became PCs by the last day
    plot_state_change_trajectories(
        exp_pcs, exp_config, trial_type='ABC', direction='recruitment',
        reference_day=-1, mouse_id=mouse_id,
    )
    plt.show()
