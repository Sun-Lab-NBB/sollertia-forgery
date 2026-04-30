"""Pre-shift "future-reward-tuned" stable place cells for the MaalstroomicFlow void cohort.

Mirrors the post-shift heatmap in ``stable_new_zone_cells_animal_*.png`` but inverts the selection:
the drift report is computed over the pre-shift block, cells are filtered to those whose pre-shift mean
rate-map peak already sat at the future (post-shift) reward location, and rows are sorted by the pre-shift
peak. Output panels span sessions 14 through 20 (the last pre-shift session and every post-shift session
that exists for the animal) so the figure shows what those pre-tuned cells do after the reward shift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sollertia_forgery.shared_assets import DatasetData
from sollertia_forgery.analysis import DriftCellColumn, TuningColumn, DriftDetectionConfiguration
from sollertia_forgery.analysis.drift import compute_drift_report


DATASET_PATH = Path("/home/data/Data/MaalstroomicFlow/void")
OUTPUT_DIR = Path("/home/data/Data/MF_Lick")
ANIMALS: tuple[str, ...] = ("11", "15")

# Pre-shift covers indices 0..14 (zone 218-232), shift happens at index 15 (zone 98-108.675).
PRE_SHIFT_SESSIONS: tuple[int, ...] = tuple(range(0, 15))
SHIFT_AT_SESSION: int = 15
PANEL_SESSIONS: tuple[int, ...] = (14, 15, 16, 17, 18, 19)

TRACK_LENGTH_CM: float = 240.0
OLD_REWARD_LO, OLD_REWARD_HI = 218.0, 232.0
NEW_REWARD_LO, NEW_REWARD_HI = 98.0, 109.0
OLD_REWARD_CENTER = 0.5 * (OLD_REWARD_LO + OLD_REWARD_HI)
NEW_REWARD_CENTER = 0.5 * (NEW_REWARD_LO + NEW_REWARD_HI)
PROXIMITY_CM: float = 30.0


def render_bw(
    animal_id: str,
    rate_maps_per_panel: dict[int, np.ndarray],
    bin_count: int,
    sorted_ids: np.ndarray,
    title_suffix: str,
    output_name: str,
) -> None:
    """Renders the per-session heatmap grid for the supplied cell ordering."""
    bin_size = TRACK_LENGTH_CM / max(bin_count, 1)
    n_total = sorted_ids.size
    figure, axes = plt.subplots(
        1, len(PANEL_SESSIONS), figsize=(24, 8), facecolor="white", dpi=150,
        gridspec_kw={"wspace": 0.05},
    )
    for axis_index, sess_idx in enumerate(PANEL_SESSIONS):
        axis = axes[axis_index]
        if sess_idx not in rate_maps_per_panel:
            axis.text(0.5, 0.5, "session not present", ha="center", va="center",
                      transform=axis.transAxes)
            axis.set_axis_off()
            continue
        selected_maps = rate_maps_per_panel[sess_idx][sorted_ids]
        finite_max = np.nanmax(selected_maps, axis=1, keepdims=True)
        finite_max = np.where(np.isfinite(finite_max) & (finite_max > 0), finite_max, 1.0)
        intensities = np.clip(selected_maps / finite_max, 0.0, 1.0)
        intensities = np.where(np.isfinite(intensities), intensities, 0.0)

        post_shift = sess_idx >= SHIFT_AT_SESSION
        rz_lo, rz_hi = (NEW_REWARD_LO, NEW_REWARD_HI) if post_shift else (OLD_REWARD_LO, OLD_REWARD_HI)
        block_label = "POST-SHIFT" if post_shift else "PRE-SHIFT"
        axis.imshow(intensities, aspect="auto", origin="upper", cmap="gray_r",
                    extent=[0, bin_count * bin_size, n_total, 0], vmin=0, vmax=1)
        axis.axvline(rz_lo, color="red", linestyle="--", linewidth=1.5)
        axis.axvline(rz_hi, color="red", linestyle="--", linewidth=1.5)
        if post_shift:
            axis.axvline(OLD_REWARD_LO, color="gray", linestyle=":", linewidth=0.8)
            axis.axvline(OLD_REWARD_HI, color="gray", linestyle=":", linewidth=0.8)
        else:
            axis.axvline(NEW_REWARD_LO, color="gray", linestyle=":", linewidth=0.8)
            axis.axvline(NEW_REWARD_HI, color="gray", linestyle=":", linewidth=0.8)
        axis.set_title(f"session {sess_idx}  ({block_label})", fontsize=11)
        axis.set_xlabel("Track Position (cm)", fontsize=10)
        axis.set_xlim(0, TRACK_LENGTH_CM)
        if axis_index == 0:
            axis.set_ylabel("Stable place cell (sorted by pre-shift peak)", fontsize=10)

    figure.suptitle(
        f"Animal {animal_id} — {title_suffix}\n"
        f"red dashed = active reward zone; gray dotted = inactive reward zone; n={n_total}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUTPUT_DIR / output_name
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = DatasetData.load(dataset_path=DATASET_PATH)
    config = DriftDetectionConfiguration(peak_shift_shuffle_count=200)

    for animal_id in ANIMALS:
        print(f"animal {animal_id}: building pre-shift future-reward-tuned heatmap...")
        sessions = tuple(sorted(
            dataset.get_sessions_for_animal(animal=animal_id), key=lambda s: s.session,
        ))
        drift_animal = dataset.get_animal(animal=animal_id)

        pre_sessions = tuple(sessions[idx] for idx in PRE_SHIFT_SESSIONS if idx < len(sessions))
        if len(pre_sessions) < 2:
            print(f"  not enough pre-shift sessions ({len(pre_sessions)}); skipping")
            continue

        pre_drift_report = compute_drift_report(
            animal=drift_animal, sessions=pre_sessions,
            use_bleaching=True, configuration=config,
        )
        pre_cells = pre_drift_report.cells
        cell_ids = pre_cells[DriftCellColumn.CELL_ID.value].to_numpy()
        stable_place = pre_cells[DriftCellColumn.IS_STABLY_TUNED_PLACE.value].to_numpy()

        rate_maps_per_panel: dict[int, np.ndarray] = {}
        bin_count_reference: int | None = None
        for sess_idx in (*PRE_SHIFT_SESSIONS, *PANEL_SESSIONS):
            if sess_idx in rate_maps_per_panel or sess_idx >= len(sessions):
                continue
            sc = pl.read_ipc(source=sessions[sess_idx].tuning_cells_path, memory_map=True)
            sc = sc.sort(TuningColumn.CELL_ID.value)
            rate_maps_per_panel[sess_idx] = np.asarray(
                sc[TuningColumn.RATE_MAP.value].to_list(), dtype=np.float32,
            )
            if bin_count_reference is None:
                bin_count_reference = rate_maps_per_panel[sess_idx].shape[1]
        bin_count = bin_count_reference if bin_count_reference is not None else 0
        bin_size = TRACK_LENGTH_CM / max(bin_count, 1)

        # Mean pre-shift rate map and per-cell peak position from the pre-shift block only.
        pre_stack = np.stack(
            [rate_maps_per_panel[s] for s in PRE_SHIFT_SESSIONS if s in rate_maps_per_panel], axis=0,
        )
        mean_pre = np.nanmean(pre_stack, axis=0)
        finite_mean = np.where(np.isfinite(mean_pre), mean_pre, -np.inf)
        peak_bins = np.argmax(finite_mean, axis=1)
        peak_positions = (peak_bins + 0.5) * bin_size
        max_pre = np.nanmax(mean_pre, axis=1)
        has_activity = np.isfinite(max_pre) & (max_pre > 0)
        near_new = has_activity & (np.abs(peak_positions - NEW_REWARD_CENTER) <= PROXIMITY_CM)

        sel = stable_place & near_new
        ids_sel = cell_ids[sel]
        peaks_sel = peak_positions[sel]
        sorted_ids = ids_sel[np.argsort(peaks_sel)]
        print(f"  pre-shift stable place tuned to NEW zone: n={sorted_ids.size}")
        if sorted_ids.size == 0:
            continue

        render_bw(
            animal_id=animal_id,
            rate_maps_per_panel=rate_maps_per_panel,
            bin_count=bin_count,
            sorted_ids=sorted_ids,
            title_suffix=(
                f"PRE-shift stable PLACE cells already tuned to the NEW reward zone "
                f"(peak within ±{PROXIMITY_CM:.0f} cm of {NEW_REWARD_CENTER:.0f} cm)"
            ),
            output_name=f"pre_shift_new_zone_cells_animal_{animal_id}.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
