"""Combined Lick + Anchor analysis scratchpad.

Generates a unified figure set under ``/home/data/Data/Lick_Anchor/`` for the configured datasets:

- Anchor-cell counts across animals (matched 15-session window) with column-shuffle chance-null overlay.
- Strict-place anchor rate-map heatmaps per animal (mean across 15 sessions, sorted by peak).
- Top-5 strict-place anchor x snapshot-session per-trial heatmap grids per animal.
- Lick-event scatter per animal across **every** available session, with per-trial reward-zone
  rectangles that handle both:
    * longitudinal trial-geometry shifts — same ``trial_type`` name with different reward zones across
      sessions (e.g., MaalstroomicFlow void: 228 cm → 112 cm reward shift mid-recording).
    * intra-session multi-trial-type recordings — multiple trial types within a single session, each
      with its own reward zone (e.g., StateSpaceOdyssey extension: ABC + ABDC).

Each trial's reward zone is resolved from its session's ``trial_geometry.yaml`` using its trial-type label,
so the rectangles in the lick scatter are always the *active* zones per row regardless of which dataset is
being processed.

The script is self-contained: data loading, trial-geometry resolution, anchor selection, statistical tests,
and figure rendering all live here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from itertools import groupby
from dataclasses import dataclass

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy.stats import fisher_exact

from sollertia_forgery.shared_assets import DatasetData, TrialGeometry
from sollertia_forgery.analysis import DriftCellColumn, TuningColumn, DriftDetectionConfiguration
from sollertia_forgery.analysis.drift import compute_drift_report


# ---- Configuration ----------------------------------------------------------------------------------------

OUTPUT_DIR = Path("/home/data/Data/Lick_Anchor")

# Dataset definitions: each dataset has a path and a per-animal label dict (used for figure labels and colors).
DATASETS: dict[str, dict] = {
    "MaalstroomicFlow_void": {
        "path": Path("/home/data/Data/MaalstroomicFlow/void"),
        "animals": ("11", "15", "16"),
        "labels": {"11": "11 (good)", "15": "15 (good)", "16": "16 (poor)"},
        "colors": {"11": "#1f77b4", "15": "#2ca02c", "16": "#d62728"},
    },
    "StateSpaceOdyssey_extension": {
        "path": Path("/home/data/Data/StateSpaceOdyssey/extension"),
        "animals": None,  # None = use every animal in the dataset
        "labels": None,
        "colors": None,
    },
}

ANCHOR_WINDOW_SESSIONS: int = 15
SNAPSHOT_SESSIONS: tuple[int, ...] = (0, 5, 10, 14)
TOP_N_FOR_GRID: int = 5
PEAK_SHUFFLE_COUNT: int = 200
COLUMN_SHUFFLE_ITER: int = 1000

PROXIMITY_DEFAULT_CM: float = 30.0  # for "around-zone" filters in downstream analyses
COLOR_LICK = "#c2185b"
COLOR_REWARDED = "#a6c8e5"
COLOR_NON_REWARDED = "#f5b8b8"


# ---- Data containers --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialBlock:
    """One block of contiguous trials sharing the same reward zone."""
    cum_trial_start: int
    cum_trial_end: int  # exclusive
    session_index: int
    trial_type: str
    trial_length_cm: float
    reward_lo: float
    reward_hi: float


@dataclass
class AnchorContext:
    """Per-(animal, trial_type) drift / anchor aggregate."""
    dataset_label: str
    animal_id: str
    trial_type: str
    sessions: tuple
    drift_report: object
    place_anchor: np.ndarray
    strict_anchor: np.ndarray
    tuned_place_anchor: np.ndarray
    tuned_strict_anchor: np.ndarray
    cell_count: int
    track_length_cm: float


@dataclass
class LickContext:
    """Per-animal lick aggregate that combines every trial type."""
    dataset_label: str
    animal_id: str
    sessions: tuple
    lick_positions: np.ndarray
    lick_trials: np.ndarray
    trial_blocks: list[TrialBlock]
    session_boundaries: list[int]
    total_trials: int
    licks_total: int
    track_length_cm: float


# ---- Utilities --------------------------------------------------------------------------------------------


def column_shuffle_null(trajectory: np.ndarray, *, n_iter: int = COLUMN_SHUFFLE_ITER, seed: int = 42) -> np.ndarray:
    """Independence null: shuffle each session's classification column to break per-cell consistency."""
    rng = np.random.default_rng(seed=seed)
    n_cells, n_sessions = trajectory.shape
    counts = np.zeros(n_iter, dtype=np.int64)
    for it in range(n_iter):
        shuffled = np.empty_like(trajectory)
        for session_idx in range(n_sessions):
            shuffled[:, session_idx] = trajectory[rng.permutation(n_cells), session_idx]
        counts[it] = int(shuffled.all(axis=1).sum())
    return counts


def resolve_per_trial_data(
    session_path: Path,
    geometry: TrialGeometry,
) -> tuple[list[dict], np.ndarray, np.ndarray, int]:
    """Resolves per-trial info, lick events, and trial count for one session.

    Returns:
        trials: list of {trial_id_in_session, trial_type, reward_lo, reward_hi, trial_length_cm}
        lick_positions: float32 array of within-trial lick positions for every rising-edge event
        lick_trial_indices: int array of within-session trial indices parallel to lick_positions
        n_trials: number of unique completed trials in run state
    """
    df = pl.read_ipc(
        source=session_path / "data.feather",
        columns=["system_state", "lick", "distance_cm", "trial", "trial_type"],
        memory_map=True,
    )
    run = df.filter((pl.col("system_state") == "run") & (pl.col("trial") < 255))
    if run.height == 0:
        return [], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), 0

    distance = run["distance_cm"].to_numpy()
    trials = run["trial"].to_numpy()
    licks = run["lick"].to_numpy().astype(np.int8)
    trial_types = run["trial_type"].to_list()

    position = np.zeros_like(distance, dtype=np.float64)
    trial_index_in_session = np.zeros_like(trials, dtype=np.int64)
    seen_trials: list[int] = []
    seen_trial_types: list[str] = []
    current_trial = trials[0]
    start_distance = distance[0]
    seen_trials.append(int(current_trial))
    seen_trial_types.append(trial_types[0])
    for i in range(distance.shape[0]):
        if trials[i] != current_trial:
            current_trial = int(trials[i])
            start_distance = distance[i]
            seen_trials.append(current_trial)
            seen_trial_types.append(trial_types[i])
        trial_index_in_session[i] = len(seen_trials) - 1
        position[i] = distance[i] - start_distance
    n_trials = len(seen_trials)

    # Per-trial geometry resolution from the session's trial_geometry.yaml.
    per_trial: list[dict] = []
    for trial_idx in range(n_trials):
        trial_type_name = seen_trial_types[trial_idx]
        if trial_type_name not in geometry.entries:
            warnings.warn(f"trial_type {trial_type_name!r} missing from geometry; skipping reward zone")
            per_trial.append({
                "trial_index_in_session": trial_idx,
                "trial_type": trial_type_name,
                "reward_lo": float("nan"),
                "reward_hi": float("nan"),
                "trial_length_cm": 240.0,
            })
            continue
        entry = geometry.entries[trial_type_name]
        per_trial.append({
            "trial_index_in_session": trial_idx,
            "trial_type": trial_type_name,
            "reward_lo": float(entry.stimulus_trigger_zone_start_cm),
            "reward_hi": float(entry.stimulus_trigger_zone_end_cm),
            "trial_length_cm": float(entry.trial_length_cm),
        })

    # Wrap by trial length so cyclic tracks render at within-trial position.
    trial_lengths = np.array([per_trial[idx]["trial_length_cm"] for idx in trial_index_in_session])
    position_wrapped = np.mod(position, trial_lengths)
    rising_edges = np.diff(np.concatenate([[0], licks])) == 1
    lick_positions = position_wrapped[rising_edges].astype(np.float32)
    lick_trial_indices = trial_index_in_session[rising_edges].astype(np.int64)
    return per_trial, lick_positions, lick_trial_indices, n_trials


def aggregate_lick_data_for_animal(sessions: tuple) -> tuple[np.ndarray, np.ndarray, list[TrialBlock], list[int], int, float]:
    """Walks every session for an animal, resolves per-trial reward zones, and returns concatenated lick events."""
    cum_trial_count = 0
    boundaries: list[int] = [0]
    blocks: list[TrialBlock] = []
    lick_positions_chunks: list[np.ndarray] = []
    lick_trial_chunks: list[np.ndarray] = []
    track_length_max = 0.0
    for session_index, session in enumerate(sessions):
        geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
        per_trial, lick_pos, lick_trial_in_sess, n_trials = resolve_per_trial_data(
            session_path=session.session_path, geometry=geometry,
        )
        if n_trials == 0:
            boundaries.append(cum_trial_count)
            continue
        # Group adjacent trials with the same reward zone into blocks for the rectangle overlay.
        zone_keys = [
            (entry["reward_lo"], entry["reward_hi"], entry["trial_type"], entry["trial_length_cm"])
            for entry in per_trial
        ]
        for key, group in groupby(enumerate(zone_keys), key=lambda kv: kv[1]):
            indices_in_block = [item[0] for item in group]
            start_in_session = indices_in_block[0]
            end_in_session = indices_in_block[-1] + 1
            blocks.append(TrialBlock(
                cum_trial_start=cum_trial_count + start_in_session,
                cum_trial_end=cum_trial_count + end_in_session,
                session_index=session_index,
                trial_type=key[2],
                trial_length_cm=key[3],
                reward_lo=key[0],
                reward_hi=key[1],
            ))
        track_length_max = max(track_length_max, max(entry["trial_length_cm"] for entry in per_trial))
        if lick_pos.size > 0:
            lick_positions_chunks.append(lick_pos)
            lick_trial_chunks.append(lick_trial_in_sess + cum_trial_count)
        cum_trial_count += n_trials
        boundaries.append(cum_trial_count)
    if lick_positions_chunks:
        all_positions = np.concatenate(lick_positions_chunks)
        all_trials = np.concatenate(lick_trial_chunks)
    else:
        all_positions = np.zeros(0, dtype=np.float32)
        all_trials = np.zeros(0, dtype=np.int64)
    return all_positions, all_trials, blocks, boundaries, cum_trial_count, track_length_max


def compute_anchor_masks(drift_report) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell anchor masks: classified across every session in the drift window."""
    cells = drift_report.cells
    place_traj = np.asarray(cells[DriftCellColumn.PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_)
    strict_traj = np.asarray(cells[DriftCellColumn.STRICT_PLACE_TRAJECTORY.value].to_list(), dtype=np.bool_)
    is_field_stable = cells[DriftCellColumn.IS_FIELD_STABLE.value].to_numpy()
    is_peak_stable = cells[DriftCellColumn.IS_PEAK_STABLE.value].to_numpy()
    is_clean = ~cells[DriftCellColumn.IS_HIGH_BLEACHING_DRIFT.value].to_numpy()
    place_anchor = place_traj.all(axis=1)
    strict_anchor = strict_traj.all(axis=1)
    tuned_place = place_anchor & is_field_stable & is_peak_stable & is_clean
    tuned_strict = strict_anchor & is_field_stable & is_peak_stable & is_clean
    return place_anchor, strict_anchor, tuned_place, tuned_strict


def select_primary_trial_type(session) -> str | None:
    """Selects the most-frequent run-state trial_type for a session (drift expects one per session)."""
    geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
    if not geometry.entries:
        return None
    df = pl.read_ipc(source=session.session_path / "data.feather",
                     columns=["system_state", "trial", "trial_type"], memory_map=True)
    run = df.filter((pl.col("system_state") == "run") & (pl.col("trial") < 255))
    if run.height == 0:
        return next(iter(geometry.entries))
    counts: dict[str, int] = {}
    for trial_type_name in run["trial_type"].to_list():
        counts[trial_type_name] = counts.get(trial_type_name, 0) + 1
    if not counts:
        return next(iter(geometry.entries))
    return max(counts.items(), key=lambda kv: kv[1])[0]


def discover_trial_types_for_animal(sessions: tuple) -> dict[str, list]:
    """Returns ``{trial_type: [sessions_containing_it_in_chronological_order]}`` across the animal's sessions.

    Each session's trial-geometry data file enumerates every trial type used by that session; the same
    trial type may have different ``trial_length_cm`` in different sessions, but the drift pipeline
    requires constant bin count, so callers must downstream-filter to sessions where the chosen trial
    type's length is constant.
    """
    out: dict[str, list] = {}
    for session in sessions:
        geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
        for trial_type in geometry.entries:
            out.setdefault(trial_type, []).append(session)
    return out


def filter_sessions_for_trial_type(sessions: tuple, trial_type: str) -> tuple:
    """Returns the longest leading run of ``sessions`` whose ``trial_type`` has constant ``trial_length_cm``.

    Notes:
        Mid-stream changes to a trial type's track length break the drift pipeline's bin-count assumption.
        Rather than rejecting the entire animal in that case, we truncate the analysis to the longest
        leading run with a constant length and let downstream callers see only that prefix.
    """
    qualifying: list = []
    expected_length: float | None = None
    for session in sessions:
        geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
        entry = geometry.entries.get(trial_type)
        if entry is None:
            continue
        length = float(entry.trial_length_cm)
        if expected_length is None:
            expected_length = length
        elif abs(length - expected_length) > 1e-6:
            break
        qualifying.append(session)
    return tuple(qualifying)


# ---- Per-animal processing --------------------------------------------------------------------------------


def process_anchor_context(
    dataset_label: str,
    dataset: DatasetData,
    animal_id: str,
    trial_type: str,
    sessions_for_type: tuple,
) -> AnchorContext | None:
    """Runs the drift pipeline on a single (animal, trial_type) pair.

    Notes:
        ``sessions_for_type`` must be a constant-trial-length leading run for the requested trial type;
        callers are expected to use `filter_sessions_for_trial_type` upstream.
    """
    animal = dataset.get_animal(animal=animal_id)
    anchor_sessions = sessions_for_type[:ANCHOR_WINDOW_SESSIONS]
    if len(anchor_sessions) < 2:
        print(f"    {animal_id}/{trial_type}: fewer than 2 qualifying sessions; skipping drift")
        return None
    overrides = {session.session: trial_type for session in anchor_sessions}
    config = DriftDetectionConfiguration(peak_shift_shuffle_count=PEAK_SHUFFLE_COUNT)
    try:
        drift_report = compute_drift_report(
            animal=animal, sessions=anchor_sessions,
            trial_type_overrides=overrides, use_bleaching=True, configuration=config,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"    {animal_id}/{trial_type}: drift compute failed "
              f"({exc.__class__.__name__}: {exc}); skipping")
        return None
    place_anchor, strict_anchor, tuned_place, tuned_strict = compute_anchor_masks(drift_report)
    cell_count = drift_report.summary.cell_count
    geometry = TrialGeometry.from_yaml(file_path=anchor_sessions[0].geometry_path)
    track_length_cm = float(geometry.entries[trial_type].trial_length_cm)
    print(f"    {animal_id}/{trial_type}: {cell_count} cells across {len(anchor_sessions)} sessions, "
          f"place={int(place_anchor.sum())}, strict={int(strict_anchor.sum())}, "
          f"tuned_place={int(tuned_place.sum())}, tuned_strict={int(tuned_strict.sum())}, "
          f"track_length={track_length_cm:.0f} cm")
    return AnchorContext(
        dataset_label=dataset_label,
        animal_id=animal_id,
        trial_type=trial_type,
        sessions=anchor_sessions,
        drift_report=drift_report,
        place_anchor=place_anchor,
        strict_anchor=strict_anchor,
        tuned_place_anchor=tuned_place,
        tuned_strict_anchor=tuned_strict,
        cell_count=cell_count,
        track_length_cm=track_length_cm,
    )


def process_lick_context(
    dataset_label: str,
    dataset: DatasetData,
    animal_id: str,
) -> LickContext | None:
    """Aggregates licks across **every** session for the animal regardless of trial type."""
    sessions_all = tuple(sorted(
        dataset.get_sessions_for_animal(animal=animal_id), key=lambda s: s.session,
    ))
    if not sessions_all:
        print(f"  {animal_id}: no sessions; skipping lick context")
        return None
    lick_positions, lick_trials, trial_blocks, boundaries, total_trials, track_length_max = (
        aggregate_lick_data_for_animal(sessions_all)
    )
    print(f"  {animal_id}: {total_trials} trials across {len(sessions_all)} sessions, "
          f"{lick_positions.size} discrete lick events, max track length {track_length_max:.0f} cm")
    return LickContext(
        dataset_label=dataset_label,
        animal_id=animal_id,
        sessions=sessions_all,
        lick_positions=lick_positions,
        lick_trials=lick_trials,
        trial_blocks=trial_blocks,
        session_boundaries=boundaries,
        total_trials=total_trials,
        licks_total=int(lick_positions.size),
        track_length_cm=track_length_max if track_length_max > 0 else 240.0,
    )


# ---- Figure 1: anchor cell counts (cross-animal) ----------------------------------------------------------


def render_anchor_counts(contexts: list[AnchorContext], dataset_label: str, trial_type: str,
                          colors: dict, labels: dict, output_path: Path) -> None:
    """4-panel bar chart with column-shuffle chance-null overlay."""
    figure, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="white", dpi=150)
    criteria = [
        ("place_anchor", "place_anchor", "Place anchor\n(classified all 15 sessions)"),
        ("strict_anchor", "strict_anchor", "Strict-place anchor"),
        ("tuned_place_anchor", None, "Tuned place anchor\n(+ field & peak stable, clean)"),
        ("tuned_strict_anchor", None, "Tuned strict-place anchor"),
    ]
    n_animals = len(contexts)
    for axis_index, (attr, traj_key, title) in enumerate(criteria):
        axis = axes[axis_index]
        counts = [int(getattr(c, attr).sum()) for c in contexts]
        cells = [c.cell_count for c in contexts]
        fractions = [100.0 * v / n if n > 0 else 0.0 for v, n in zip(counts, cells, strict=True)]
        bars = axis.bar(
            range(n_animals), counts,
            color=[colors.get(c.animal_id, "#888888") for c in contexts],
            edgecolor="black",
        )
        for bar, fraction, count in zip(bars, fractions, counts, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(counts + [1]) * 0.01,
                      f"{count}\n({fraction:.1f}%)", ha="center", va="bottom", fontsize=9)
        if traj_key is not None:
            null_means = []
            for index, context in enumerate(contexts):
                cells_table = context.drift_report.cells
                column_value = (
                    DriftCellColumn.PLACE_TRAJECTORY.value if traj_key == "place_anchor"
                    else DriftCellColumn.STRICT_PLACE_TRAJECTORY.value
                )
                trajectory = np.asarray(cells_table[column_value].to_list(), dtype=np.bool_)
                null_means.append(float(np.mean(column_shuffle_null(trajectory, n_iter=200, seed=10 + index))))
            axis.plot(range(n_animals), null_means, color="black", linestyle="--", marker="o",
                      label="chance null (mean)")
            axis.legend(loc="upper left", frameon=False)
        axis.set_xticks(range(n_animals))
        axis.set_xticklabels([labels.get(c.animal_id, c.animal_id) for c in contexts], rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Cell count")
    figure.suptitle(f"Anchor-cell counts — {dataset_label} ({trial_type}, first 15 sessions)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ---- Figure 2: strict-anchor mean rate maps (cross-animal) ------------------------------------------------


def render_strict_anchor_rate_maps(contexts: list[AnchorContext], dataset_label: str, trial_type: str,
                                    labels: dict, output_path: Path) -> None:
    """Per-animal heatmap of strict-anchor mean rate maps across the 15-session window, sorted by peak."""
    n_animals = len(contexts)
    figure, axes = plt.subplots(1, n_animals, figsize=(6 * n_animals, 7), facecolor="white", dpi=150)
    if n_animals == 1:
        axes = [axes]
    for axis, context in zip(axes, contexts, strict=True):
        anchor_mask = context.strict_anchor
        sessions = context.sessions
        if not anchor_mask.any() or not sessions:
            axis.text(0.5, 0.5, "no strict anchors", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        rate_maps_per_session = []
        zone_centers: list[float] = []
        for session in sessions:
            tcells = pl.read_ipc(source=session.tuning_cells_path, memory_map=True)
            tcells = tcells.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == context.trial_type)
            tcells = tcells.sort(TuningColumn.CELL_ID.value)
            rate_maps = np.asarray(tcells[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32)
            rate_maps_per_session.append(rate_maps)
            geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
            entry = geometry.entries.get(context.trial_type)
            if entry is not None:
                zone_centers.append(0.5 * (entry.stimulus_trigger_zone_start_cm + entry.stimulus_trigger_zone_end_cm))
        stack = np.stack(rate_maps_per_session, axis=0)
        anchor_maps = np.nanmean(stack[:, anchor_mask, :], axis=0)
        peak_bins = np.argmax(anchor_maps, axis=1)
        order = np.argsort(peak_bins)
        normalized = anchor_maps / np.maximum(anchor_maps.max(axis=1, keepdims=True), 1e-6)
        normalized = normalized[order]
        bin_count = stack.shape[2]
        bin_size = context.track_length_cm / bin_count
        axis.imshow(normalized, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, bin_count * bin_size, 0, normalized.shape[0]])
        if zone_centers:
            mean_zone_center = float(np.mean(zone_centers))
            axis.axvline(mean_zone_center, color="orange", linestyle="--", linewidth=1.5,
                         label="reward zone center")
            axis.legend(loc="upper left", frameon=False)
        axis.set_title(f"Animal {labels.get(context.animal_id, context.animal_id)}\n"
                       f"(n={int(anchor_mask.sum())} strict-place anchors)")
        axis.set_xlabel("Track position (cm)")
        axis.set_ylabel("Strict-place anchor cell (sorted by peak)")
    figure.suptitle(
        f"Strict-place anchor rate maps — {dataset_label} ({trial_type})\n"
        f"strict = IS_PLACE & IS_STABLE & IS_PEAK_SIGNIFICANT in every session",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ---- Figure 3: top-N strict-anchor session grid (per-animal) ----------------------------------------------


def render_strict_anchor_session_grid(context: AnchorContext, label: str, output_path: Path) -> None:
    """5x4 grid of top strict anchors (rows) at snapshot sessions (columns); per-trial heatmap per cell-session."""
    cells = context.drift_report.cells
    cell_ids = cells[DriftCellColumn.CELL_ID.value].to_numpy()
    mean_corr = cells[DriftCellColumn.MEAN_RATE_MAP_CORRELATION.value].to_numpy()
    candidate_ids = cell_ids[context.strict_anchor]
    candidate_corr = mean_corr[context.strict_anchor]
    finite = np.isfinite(candidate_corr)
    candidate_ids = candidate_ids[finite]
    candidate_corr = candidate_corr[finite]
    if candidate_ids.size == 0:
        print(f"    {context.animal_id}: no strict anchors with finite stability — skipping grid")
        return
    order = np.argsort(-candidate_corr)
    top_ids = candidate_ids[order][:TOP_N_FOR_GRID]
    top_corrs = candidate_corr[order][:TOP_N_FOR_GRID]
    n_rows = top_ids.size

    snapshot_indices = tuple(idx for idx in SNAPSHOT_SESSIONS if idx < len(context.sessions))
    if not snapshot_indices:
        print(f"    {context.animal_id}/{context.trial_type}: no snapshot sessions in window; skipping grid")
        return
    per_cell_per_session: dict[int, dict[int, np.ndarray | None]] = {int(cid): {} for cid in top_ids}
    for sess_idx in snapshot_indices:
        tcells = pl.read_ipc(source=context.sessions[sess_idx].tuning_cells_path, memory_map=True)
        tcells = tcells.filter(pl.col(TuningColumn.TRIAL_TYPE.value) == context.trial_type)
        tcells = tcells.sort(TuningColumn.CELL_ID.value)
        per_trial_lists = tcells[TuningColumn.BINNED_FLUORESCENCE_PER_TRIAL.value].to_list()
        for cid in top_ids:
            entry = per_trial_lists[int(cid)]
            per_cell_per_session[int(cid)][sess_idx] = (
                np.asarray(entry, dtype=np.float32) if entry else None
            )

    figure, axes = plt.subplots(
        n_rows, len(snapshot_indices),
        figsize=(4 * len(snapshot_indices), 2.4 * n_rows),
        facecolor="black", dpi=150, squeeze=False,
    )
    figure.patch.set_facecolor("black")
    figure.suptitle(
        f"Strict-place anchor tuning across sessions — animal {label} ({context.trial_type})\n"
        f"Top {TOP_N_FOR_GRID} anchors at sessions {snapshot_indices}",
        color="white", fontsize=14,
    )

    for row_index, cid in enumerate(top_ids):
        cid_int = int(cid)
        all_blocks = [
            per_cell_per_session[cid_int][s]
            for s in snapshot_indices if per_cell_per_session[cid_int][s] is not None
        ]
        if all_blocks:
            stack = np.concatenate(all_blocks, axis=0)
            finite_vals = stack[np.isfinite(stack)]
            if finite_vals.size > 0:
                vmin = float(np.percentile(finite_vals, 1))
                vmax = float(np.percentile(finite_vals, 99))
                if vmax <= vmin:
                    vmax = vmin + 1e-3
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        for col_index, sess_idx in enumerate(snapshot_indices):
            axis = axes[row_index, col_index]
            axis.set_facecolor("black")
            for spine in axis.spines.values():
                spine.set_color("white")
            axis.tick_params(colors="white", labelsize=9)
            block = per_cell_per_session[cid_int][sess_idx]
            if block is None or block.size == 0:
                axis.text(0.5, 0.5, "no data", ha="center", va="center",
                          transform=axis.transAxes, color="white", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(f"session {sess_idx}", color="white", fontsize=11)
                continue
            bin_count = block.shape[1]
            bin_size = context.track_length_cm / bin_count
            axis.imshow(block, aspect="auto", origin="upper", cmap="jet",
                        interpolation="nearest", vmin=vmin, vmax=vmax,
                        extent=[0, bin_count * bin_size, block.shape[0], 0])
            # Reward-zone shading from this snapshot session's matching trial type.
            session = context.sessions[sess_idx]
            geometry = TrialGeometry.from_yaml(file_path=session.geometry_path)
            entry = geometry.entries.get(context.trial_type)
            if entry is not None:
                axis.axvspan(entry.stimulus_trigger_zone_start_cm, entry.stimulus_trigger_zone_end_cm,
                             color="orange", alpha=0.2)
            if row_index == 0:
                axis.set_title(f"session {sess_idx}", color="white", fontsize=11)
            if col_index == 0:
                axis.set_ylabel(f"cell {cid_int}\nr={top_corrs[row_index]:.2f}", color="white", fontsize=10)
            if row_index == n_rows - 1:
                axis.set_xlabel("Position (cm)", color="white", fontsize=10)
            else:
                axis.set_xticklabels([])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, bbox_inches="tight", facecolor="black")
    plt.close(figure)


# ---- Figure 4: lick scatter across all sessions (per-animal) ----------------------------------------------


def render_lick_scatter(context: LickContext, label: str, output_path: Path) -> None:
    """Lick-event scatter aggregated across **every** session for the animal.

    Per-trial reward-zone rectangles are drawn from each trial's resolved geometry. Trial-type and
    geometry shifts (longitudinal MF or intra-session SSO) appear as different per-trial rectangles
    with the same color coding (rewarded vs non-rewarded relative to the trial's own reward zone).
    Top reference bar lists every unique reward zone present across the animal's sessions.
    """
    if context.total_trials == 0:
        print(f"    {context.animal_id}: zero trials; skipping lick scatter")
        return
    figure = plt.figure(figsize=(11, max(8.0, 0.025 * context.total_trials + 4.0)),
                        facecolor="white", dpi=150)
    grid = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1.6, 22], hspace=0.06)
    ax_zones = figure.add_subplot(grid[0, 0])
    ax_main = figure.add_subplot(grid[1, 0], sharex=ax_zones)

    # ---- Top reference bar: list every unique (trial_type, reward_zone) the animal saw.
    unique_zones: dict[tuple[str, float, float], int] = {}
    for block in context.trial_blocks:
        key = (block.trial_type, block.reward_lo, block.reward_hi)
        if not np.isfinite(block.reward_lo) or not np.isfinite(block.reward_hi):
            continue
        unique_zones.setdefault(key, 0)
        unique_zones[key] += block.cum_trial_end - block.cum_trial_start
    n_zones = max(len(unique_zones), 1)
    bar_height = 1.0 / n_zones
    ax_zones.set_facecolor("#f8f8f8")
    ax_zones.set_xlim(0, context.track_length_cm)
    ax_zones.set_ylim(0, 1)
    sorted_zone_keys = sorted(unique_zones.keys(), key=lambda k: (k[1], k[0]))
    for index, key in enumerate(sorted_zone_keys):
        trial_type_name, reward_lo, reward_hi = key
        y_lo = 1.0 - (index + 1) * bar_height
        y_hi = 1.0 - index * bar_height
        ax_zones.add_patch(Rectangle(
            (reward_lo, y_lo + 0.08 * bar_height),
            reward_hi - reward_lo, 0.84 * bar_height,
            facecolor=COLOR_REWARDED, edgecolor="black", linewidth=0.7,
        ))
        center = 0.5 * (reward_lo + reward_hi)
        ax_zones.text(center, 0.5 * (y_lo + y_hi), f"{trial_type_name}\n{reward_lo:.0f}-{reward_hi:.0f} cm",
                      ha="center", va="center", fontsize=8)
    ax_zones.set_yticks([])
    ax_zones.tick_params(labelbottom=False)
    for spine in ax_zones.spines.values():
        spine.set_visible(False)
    ax_zones.text(-2, 0.5, f"Reward zones\nused", ha="right", va="center", fontsize=9, fontweight="bold")

    # ---- Main scatter: per-trial reward-zone rectangles drawn behind the lick scatter, plus a
    # diagonal-hatched "absent track" mask for trial blocks whose trial_length_cm is shorter than the
    # max track length seen across the animal (e.g., 180-cm ABC trials in a 240-cm-max ABDC animal).
    for block in context.trial_blocks:
        height = block.cum_trial_end - block.cum_trial_start
        # Mask the section of the position axis that doesn't exist for this trial's track length.
        if block.trial_length_cm < context.track_length_cm:
            ax_main.add_patch(Rectangle(
                (block.trial_length_cm, block.cum_trial_start),
                context.track_length_cm - block.trial_length_cm, height,
                facecolor="#cccccc", edgecolor="#888888", alpha=0.55,
                hatch="///", linewidth=0.0, zorder=0,
            ))
        if not np.isfinite(block.reward_lo) or not np.isfinite(block.reward_hi):
            continue
        ax_main.add_patch(Rectangle(
            (block.reward_lo, block.cum_trial_start),
            block.reward_hi - block.reward_lo, height,
            facecolor=COLOR_REWARDED, edgecolor="none", alpha=0.45, zorder=0,
        ))
        # Mirror non-rewarded zones for *other* trial types this animal sees: every alternate trial
        # type's reward zone gets a non-rewarded rectangle within this block, but only when the
        # alternate zone falls within this block's actual track length.
        for other_key in unique_zones:
            if other_key == (block.trial_type, block.reward_lo, block.reward_hi):
                continue
            other_lo, other_hi = other_key[1], other_key[2]
            if not np.isfinite(other_lo) or not np.isfinite(other_hi):
                continue
            if other_hi > block.trial_length_cm:
                continue  # alternate zone is in the masked section for this block — skip
            ax_main.add_patch(Rectangle(
                (other_lo, block.cum_trial_start),
                other_hi - other_lo, height,
                facecolor=COLOR_NON_REWARDED, edgecolor="none", alpha=0.35, zorder=0,
            ))

    ax_main.scatter(context.lick_positions, context.lick_trials,
                    s=3, color=COLOR_LICK, alpha=0.55, edgecolor="none", zorder=2)
    for boundary in context.session_boundaries[1:-1]:
        ax_main.axhline(boundary, color="gray", linestyle="--", linewidth=0.6, zorder=1)
    ax_main.set_xlim(0, context.track_length_cm)
    ax_main.set_ylim(context.total_trials, 0)
    ax_main.set_xlabel("Position (cm)", fontsize=11)
    ax_main.set_ylabel("Trial number (chronological across all sessions)", fontsize=11)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_LICK,
                   markersize=7, label="Lick events"),
        Patch(facecolor=COLOR_REWARDED, edgecolor="black", linewidth=0.6, label="Rewarded location"),
        Patch(facecolor=COLOR_NON_REWARDED, edgecolor="black", linewidth=0.6,
              label="Non-rewarded location"),
        Patch(facecolor="#cccccc", edgecolor="#888888", linewidth=0.6, hatch="///",
              label="Absent track section"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.7, label="Session separator"),
    ]
    ax_main.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0, frameon=False, fontsize=9)
    figure.suptitle(
        f"Animal {label} — discrete lick events across every session\n"
        f"{context.total_trials} trials, {context.licks_total} rising-edge lick events; "
        f"{len(unique_zones)} unique reward zone(s) seen",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 0.85, 0.96))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ---- Statistics dump --------------------------------------------------------------------------------------


def report_statistics(
    anchor_contexts_by_trial_type: dict[str, list[AnchorContext]],
    lick_contexts: list[LickContext],
    dataset_label: str,
    output_path: Path,
) -> None:
    """Writes a plain-text statistics summary covering every (animal, trial_type) and the lick aggregates."""
    lines: list[str] = [f"Dataset: {dataset_label}", "=" * 70, ""]
    for trial_type, contexts in sorted(anchor_contexts_by_trial_type.items()):
        lines.append(f"Trial type {trial_type!r} — anchor-cell counts (first 15 sessions)")
        lines.append("-" * 70)
        for context in contexts:
            n = context.cell_count
            lines.append(
                f"  animal {context.animal_id}: cells={n}, "
                f"place_anchor={int(context.place_anchor.sum())} ({100 * context.place_anchor.sum() / max(n, 1):.2f}%), "
                f"strict_anchor={int(context.strict_anchor.sum())} ({100 * context.strict_anchor.sum() / max(n, 1):.2f}%), "
                f"tuned_place={int(context.tuned_place_anchor.sum())} ({100 * context.tuned_place_anchor.sum() / max(n, 1):.2f}%), "
                f"tuned_strict={int(context.tuned_strict_anchor.sum())} ({100 * context.tuned_strict_anchor.sum() / max(n, 1):.2f}%)"
            )
        lines.append("")
        lines.append(f"  Pairwise Fisher exact tests for {trial_type!r}:")
        for attr in ("place_anchor", "strict_anchor", "tuned_place_anchor", "tuned_strict_anchor"):
            lines.append(f"    {attr}:")
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    a, b = contexts[i], contexts[j]
                    ca, cb = int(getattr(a, attr).sum()), int(getattr(b, attr).sum())
                    if ca == 0 and cb == 0:
                        continue
                    try:
                        odds, p = fisher_exact(
                            [[ca, a.cell_count - ca], [cb, b.cell_count - cb]],
                            alternative="two-sided",
                        )
                        lines.append(f"      {a.animal_id} vs {b.animal_id}: "
                                     f"odds_ratio={odds:.2f}, p={p:.3e}")
                    except ValueError as exc:
                        lines.append(f"      {a.animal_id} vs {b.animal_id}: skipped ({exc})")
        lines.append("")

    lines.append("Lick aggregation (all sessions, all trial types combined)")
    lines.append("---------------------------------------------------------")
    for context in lick_contexts:
        unique_zones = {(b.trial_type, b.reward_lo, b.reward_hi) for b in context.trial_blocks
                        if np.isfinite(b.reward_lo) and np.isfinite(b.reward_hi)}
        lines.append(
            f"  animal {context.animal_id}: {context.total_trials} trials, "
            f"{context.licks_total} discrete lick events, "
            f"track length up to {context.track_length_cm:.0f} cm, "
            f"{len(unique_zones)} unique reward zone(s): "
            f"{sorted(unique_zones, key=lambda z: (z[1], z[0]))}"
        )
    output_path.write_text("\n".join(lines) + "\n")


# ---- Main -------------------------------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_label, settings in DATASETS.items():
        dataset_path: Path = settings["path"]
        if not dataset_path.exists():
            print(f"Skipping {dataset_label}: path does not exist ({dataset_path})")
            continue
        print(f"\n=== {dataset_label} ===")
        dataset = DatasetData.load(dataset_path=dataset_path)
        animals = settings.get("animals") or tuple(a.animal for a in dataset.animals)
        labels = settings.get("labels") or {a: a for a in animals}
        colors = settings.get("colors") or {
            a: c for a, c in zip(animals, plt.cm.tab10.colors, strict=False)
        }

        anchor_contexts_by_trial_type: dict[str, list[AnchorContext]] = {}
        lick_contexts: list[LickContext] = []
        for animal_id in animals:
            print(f"\nProcessing animal {animal_id}...")
            sessions_all = tuple(sorted(
                dataset.get_sessions_for_animal(animal=animal_id), key=lambda s: s.session,
            ))
            if not sessions_all:
                print(f"  {animal_id}: no sessions; skipping")
                continue

            # Discover every trial type the animal saw and run the drift pipeline once per trial type
            # whose qualifying-session prefix has at least 2 entries with constant trial length.
            discovered = discover_trial_types_for_animal(sessions_all)
            print(f"  {animal_id}: discovered trial types {sorted(discovered)}")
            for trial_type in sorted(discovered):
                qualifying = filter_sessions_for_trial_type(sessions_all, trial_type)
                if len(qualifying) < 2:
                    print(f"    {animal_id}/{trial_type}: only {len(qualifying)} qualifying session(s); skipping")
                    continue
                ctx = process_anchor_context(
                    dataset_label=dataset_label, dataset=dataset,
                    animal_id=animal_id, trial_type=trial_type,
                    sessions_for_type=qualifying,
                )
                if ctx is not None:
                    anchor_contexts_by_trial_type.setdefault(trial_type, []).append(ctx)

            # Lick aggregation across every session and trial type.
            lick_ctx = process_lick_context(
                dataset_label=dataset_label, dataset=dataset, animal_id=animal_id,
            )
            if lick_ctx is not None:
                lick_contexts.append(lick_ctx)

        if not anchor_contexts_by_trial_type and not lick_contexts:
            print(f"  no usable contexts in {dataset_label}; skipping figures")
            continue

        prefix = OUTPUT_DIR / dataset_label
        # Cross-animal anchor figures, one set per trial type.
        for trial_type, contexts in sorted(anchor_contexts_by_trial_type.items()):
            render_anchor_counts(
                contexts, dataset_label, trial_type, colors, labels,
                output_path=Path(f"{prefix}_anchor_counts_{trial_type}.png"),
            )
            render_strict_anchor_rate_maps(
                contexts, dataset_label, trial_type, labels,
                output_path=Path(f"{prefix}_strict_anchor_rate_maps_{trial_type}.png"),
            )
            for context in contexts:
                label = labels.get(context.animal_id, context.animal_id)
                render_strict_anchor_session_grid(
                    context, label,
                    output_path=Path(
                        f"{prefix}_strict_anchor_session_grid_{context.animal_id}_{trial_type}.png"
                    ),
                )

        # Per-animal lick figures (combine every trial type).
        for lick_context in lick_contexts:
            label = labels.get(lick_context.animal_id, lick_context.animal_id)
            render_lick_scatter(
                lick_context, label,
                output_path=Path(f"{prefix}_lick_scatter_{lick_context.animal_id}.png"),
            )

        report_statistics(
            anchor_contexts_by_trial_type, lick_contexts, dataset_label,
            output_path=Path(f"{prefix}_statistics.txt"),
        )
        print(f"\nFigures and statistics for {dataset_label} written under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
