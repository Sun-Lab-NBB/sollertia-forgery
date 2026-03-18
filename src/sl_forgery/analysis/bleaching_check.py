"""
Bleaching check module (rewritten)

Assesses photobleaching in calcium imaging data across sessions and across days.
Uses raw fluorescence (single_day_f) — not ΔF/F — because ΔF/F normalizes away baseline decay.

Plots:
    1. Overall mean F per day — one point per session, detects cross-day signal loss
    2. Per-frame mean F within session — time-series per day, shows intra-session decay
    3. Per-cell exponential decay fits — distribution of decay rates across days
    4. Half-session comparison — first vs second half per day, paired scatter

Usage:
    from df_processing import load_multiday_sessions
    sessions = load_multiday_sessions(mouse_dir, dates=[...])
    fig = plot_bleaching_summary(sessions)
"""


from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.stats import linregress, wilcoxon



# EXTRACTION
def extract_session_signals(
    sessions: dict[str, dict],
    signal_col: str = 'single_day_f',
) -> dict[str, np.ndarray]:
    """Extract signal matrices from sessions dict.

    Args:
        sessions: from load_multiday_sessions(), keyed by date string.
        signal_col: column containing per-frame signal lists. Use raw F

    Returns:
        Signal matrices keyed by date, each shape (n_frames, n_cells).
    """
    signals = {}
    for date, session in sessions.items():
        data = session['data']
        signals[date] = np.vstack(data[signal_col].to_list())
    return signals


# METRICS

def compute_per_day_mean(signals: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute overall mean fluorescence per session.

    Args:
        signals: from extract_session_signals(), keyed by date.

    Returns:
        Mean F value per date.
    """
    return {date: float(matrix.mean()) for date, matrix in signals.items()}


def compute_per_frame_mean(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute mean fluorescence across all cells at each frame.

    Args:
        signals: from extract_session_signals(), keyed by date.

    Returns:
        1D array of per-frame means per date.
    """
    return {date: matrix.mean(axis=1) for date, matrix in signals.items()}


def compute_half_session_means(signals: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Split each session in half and compute mean F for each half.

    Args:
        signals: from extract_session_signals(), keyed by date.

    Returns:
        (first_half_mean, second_half_mean) per date.
    """
    results = {}
    for date, matrix in signals.items():
        midpoint = matrix.shape[0] // 2
        first_half = float(matrix[:midpoint].mean())
        second_half = float(matrix[midpoint:].mean())
        results[date] = (first_half, second_half)
    return results


def _exp_decay(x: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    """Exponential decay model: a * exp(-x / tau) + c."""
    return a * np.exp(-x / tau) + c


def compute_per_cell_decay(
    signals: dict[str, np.ndarray],
    max_cells: int | None = None,
) -> np.ndarray:
    """Fit exponential decay to each cell's mean F across days.

    Fits f(day) = a * exp(-day/tau) + c for each cell (exponential decay curve).
    Returns tau values (time constant in units of days).
    Positive tau = decay, negative = unlikely but means signal grew.

    Args:
        signals: from extract_session_signals(), keyed by date (sorted).
        max_cells: cap on number of cells to fit (for speed). None = all cells.

    Returns:
        Array of tau values, shape (n_cells_fit,). NaN for failed fits.
    """
    dates = sorted(signals.keys())
    n_days = len(dates)
    if n_days < 3:
        print("Need >= 3 days for exponential fits, skipping.")
        return np.array([])

    # Per-cell mean F per day: (n_days, n_cells)
    n_cells = signals[dates[0]].shape[1]
    cell_means = np.column_stack([signals[d].mean(axis=0) for d in dates])  # (n_cells, n_days)

    if max_cells is not None and n_cells > max_cells:
        rng = np.random.default_rng(42)
        subset = rng.choice(n_cells, max_cells, replace=False)
        cell_means = cell_means[subset]
        n_cells = max_cells

    x = np.arange(n_days, dtype=float)
    taus = np.full(n_cells, np.nan)

    for i in range(n_cells):
        y = cell_means[i]
        try:
            p0 = [y[0] - y[-1], max(n_days / 2, 1.0), y[-1]]
            lower = [-np.inf, 1e-3, -np.inf]
            upper = [np.inf, np.inf, np.inf]
            popt, _ = curve_fit(_exp_decay, x, y, p0=p0, maxfev=2000, bounds=(lower, upper))
            taus[i] = popt[1]
        except (RuntimeError, ValueError):
            continue

    n_fit = np.count_nonzero(~np.isnan(taus))
    print(f"Exponential fits: {n_fit}/{n_cells} cells converged.")
    return taus


def compute_per_cell_decay_multi(
    sessions: dict[str, dict],
    signal_col: str = 'multi_day_f',
    bin_minutes: float = 5.0,
    frame_rate: float = 10.0,
    max_cells: int | None = None,
) -> np.ndarray:
    """Different version. Requires multi_day_f so arrays are same size
    Fit exponential decay to each cell across all sessions, binned by time.

    Concatenates all sessions chronologically, bins into time chunks, and fits
    f(t) = a * exp(-t/tau) + c per cell. Returns tau in units of hours.

    Args:
        sessions: from load_multiday_sessions(), keyed by date string.
        signal_col: column containing per-frame signal lists.
        bin_minutes: time bin size for fitting.
        frame_rate: imaging frame rate in Hz (defaults to 10 hz)
        max_cells: cap on number of cells to fit (for speed). None = all cells.

    Returns:
        Array of tau values in hours, shape (n_cells_fit,). NaN for failed fits.
    """
    dates = sorted(sessions.keys())

    # Build binned means per cell across all sessions
    binned_means = []
    binned_times = []
    cumulative_hours = 0.0

    for i, date in enumerate(dates):
        data = sessions[date]['data']
        matrix = np.vstack(data[signal_col].to_list())
        elapsed = data['elapsed_minutes'].to_numpy()

        # Derive frame rate from this session's timestamps
        session_duration_sec = (elapsed[-1] - elapsed[0]) * 60.0
        n_frames = matrix.shape[0]
        frame_rate = n_frames / session_duration_sec
        frames_per_bin = int(bin_minutes * 60 * frame_rate)
        n_bins = n_frames // frames_per_bin

        if n_bins == 0:
            continue

        trimmed = matrix[:n_bins * frames_per_bin]
        chunk_means = trimmed.reshape(n_bins, frames_per_bin, -1).mean(axis=1)
        binned_means.append(chunk_means)

        bin_hours = cumulative_hours + np.arange(n_bins) * (bin_minutes / 60.0)
        binned_times.append(bin_hours)

        # Advance clock: session duration + gap to next day
        session_hours = session_duration_sec / 3600.0
        if i < len(dates) - 1:
            # Parse actual gap between session dates
            from datetime import date as dt_date
            current = dt_date.fromisoformat(date)
            next_d = dt_date.fromisoformat(dates[i + 1])
            gap_hours = (next_d - current).days * 24.0
            cumulative_hours += gap_hours
        else:
            cumulative_hours += session_hours

    if not binned_means:
        return np.array([])

    all_means = np.concatenate(binned_means, axis=0)
    all_times = np.concatenate(binned_times)
    n_cells = all_means.shape[1]

    print(f"  Decay fits: {len(all_times)} time bins across {len(dates)} days")

    if max_cells is not None and n_cells > max_cells:
        rng = np.random.default_rng(42)
        subset = rng.choice(n_cells, max_cells, replace=False)
        all_means = all_means[:, subset]
        n_cells = max_cells

    taus = np.full(n_cells, np.nan)
    for i in range(n_cells):
        y = all_means[:, i]
        try:
            p0 = [y[0] - y[-1], max(all_times[-1] / 2, 1.0), y[-1]]
            lower = [-np.inf, 1e-3, -np.inf]
            upper = [np.inf, np.inf, np.inf]
            popt, _ = curve_fit(_exp_decay, all_times, y, p0=p0, maxfev=2000, bounds=(lower, upper))
            taus[i] = popt[1]
        except (RuntimeError, ValueError):
            continue

    n_fit = np.count_nonzero(~np.isnan(taus))
    print(f"  Exponential fits: {n_fit}/{n_cells} cells converged.")

    return taus


def print_bleaching_stats(
    signals: dict[str, np.ndarray],
    per_day_mean: dict[str, float],
    per_frame_mean: dict[str, np.ndarray],
    half_means: dict[str, tuple[float, float]],
    taus: np.ndarray,
) -> dict:
    """Computes and prints quantitative bleaching metrics to accompany the 4-panel figure.

    Prints a formatted summary table covering cross-day signal loss, intra-session decay rates,
    exponential decay time constants, half-session comparisons, and per-day signal-to-noise ratios.
    Returns the computed metrics as a dictionary for programmatic use.

    Args:
        signals: Signal matrices keyed by date, each shape (n_frames, n_cells).
        per_day_mean: Mean fluorescence per date string.
        per_frame_mean: 1D array of per-frame means per date.
        half_means: (first_half_mean, second_half_mean) per date.
        taus: Array of tau values from compute_per_cell_decay or compute_per_cell_decay_multi.

    Returns:
        Dictionary containing all computed metrics organized by analysis category.
    """
    dates = sorted(per_day_mean.keys())
    n_days = len(dates)
    results: dict = {}

    # --- Cross-day signal loss (Panel 1) ---
    day_values = np.array([per_day_mean[d] for d in dates])
    day_indices = np.arange(n_days, dtype=float)

    total_percent_change = (day_values[-1] - day_values[0]) / day_values[0] * 100.0

    cross_day: dict = {"total_percent_change": total_percent_change}
    if n_days >= 3:
        regression = linregress(x=day_indices, y=day_values)
        cross_day["slope_per_day"] = regression.slope
        cross_day["r_squared"] = regression.rvalue ** 2
        cross_day["p_value"] = regression.pvalue
    results["cross_day"] = cross_day

    # --- Intra-session decay (Panel 2) ---
    intra_session: dict = {}
    for date in dates:
        trace = per_frame_mean[date]
        n_frames = len(trace)
        # Compares the mean of the first and last 10% of frames.
        edge_count = max(n_frames // 10, 1)
        first_mean = float(trace[:edge_count].mean())
        last_mean = float(trace[-edge_count:].mean())
        percent_drop = (last_mean - first_mean) / first_mean * 100.0

        frame_indices = np.arange(n_frames, dtype=float)
        regression = linregress(x=frame_indices, y=trace)

        intra_session[date] = {
            "slope_per_frame": regression.slope,
            "r_squared": regression.rvalue ** 2,
            "percent_drop": percent_drop,
        }
    results["intra_session"] = intra_session

    # --- Decay tau distribution (Panel 3) ---
    valid_taus = taus[~np.isnan(taus)]
    n_total_cells = len(taus)
    n_converged = len(valid_taus)

    tau_stats: dict = {
        "n_converged": n_converged,
        "n_total": n_total_cells,
        "percent_converged": n_converged / max(n_total_cells, 1) * 100.0,
    }
    if n_converged > 0:
        tau_stats["median"] = float(np.median(valid_taus))
        tau_stats["mean"] = float(np.mean(valid_taus))
        tau_stats["std"] = float(np.std(valid_taus))
        tau_stats["iqr_25"] = float(np.percentile(valid_taus, 25))
        tau_stats["iqr_75"] = float(np.percentile(valid_taus, 75))
        # Fraction of cells with a tau shorter than 3 days (rapid bleaching).
        tau_stats["percent_below_3"] = float(np.mean(valid_taus < 3.0) * 100.0)
    results["tau_distribution"] = tau_stats

    # --- Half-session comparison (Panel 4) ---
    first_halves = np.array([half_means[d][0] for d in dates])
    second_halves = np.array([half_means[d][1] for d in dates])
    per_day_drop = (second_halves - first_halves) / first_halves * 100.0

    half_stats: dict = {
        "per_day_percent_drop": {d: float(per_day_drop[i]) for i, d in enumerate(dates)},
        "mean_percent_drop": float(per_day_drop.mean()),
    }
    if n_days >= 6:
        # Wilcoxon signed-rank test requires at least 6 paired observations.
        stat, p_value = wilcoxon(x=first_halves, y=second_halves)
        half_stats["wilcoxon_statistic"] = float(stat)
        half_stats["wilcoxon_p_value"] = float(p_value)
    results["half_session"] = half_stats

    # --- Per-day SNR (cross-cutting) ---
    snr_per_day: dict = {}
    for date in dates:
        matrix = signals[date]
        cell_means = matrix.mean(axis=0)
        snr_per_day[date] = float(cell_means.mean() / max(cell_means.std(), 1e-12))
    results["snr_per_day"] = snr_per_day

    # Uses print() for the formatted table since console.echo() would disrupt alignment.
    _print_stats_table(
        results=results,
        dates=dates,
    )

    return results


def _print_stats_table(
    results: dict,
    dates: list[str],
) -> None:
    """Formats and prints the bleaching statistics as an aligned table.

    Args:
        results: Dictionary of computed metrics from print_bleaching_stats.
        dates: Sorted list of date strings.
    """
    separator = "=" * 80
    thin_separator = "-" * 80

    print(f"\n{separator}")
    print("BLEACHING QUANTITATIVE SUMMARY")
    print(separator)

    # Cross-day signal loss.
    cross_day = results["cross_day"]
    print(f"\n[1] CROSS-DAY SIGNAL LOSS (overall mean F per day)")
    print(thin_separator)
    print(f"  Total change (first → last):  {cross_day['total_percent_change']:+.1f}%")
    if "slope_per_day" in cross_day:
        print(f"  Linear slope:                 {cross_day['slope_per_day']:.2f} F/day")
        print(f"  R²:                           {cross_day['r_squared']:.4f}")
        print(f"  p-value:                      {cross_day['p_value']:.2e}")

    # Intra-session decay.
    intra = results["intra_session"]
    print(f"\n[2] INTRA-SESSION DECAY (per-frame mean F)")
    print(thin_separator)
    print(f"  {'Date':<12} {'Slope (F/frame)':<18} {'R²':<10} {'Drop (%)':<10}")
    print(f"  {'----':<12} {'---------------':<18} {'--':<10} {'--------':<10}")
    for date in dates:
        stats = intra[date]
        print(
            f"  {date[5:]:<12} {stats['slope_per_frame']:<18.6f} "
            f"{stats['r_squared']:<10.4f} {stats['percent_drop']:<+10.1f}"
        )

    # Tau distribution.
    tau_stats = results["tau_distribution"]
    print(f"\n[3] DECAY TIME CONSTANTS (exponential fit per cell)")
    print(thin_separator)
    print(f"  Cells converged:              {tau_stats['n_converged']}/{tau_stats['n_total']} "
          f"({tau_stats['percent_converged']:.0f}%)")
    if "median" in tau_stats:
        print(f"  Median τ:                     {tau_stats['median']:.1f}")
        print(f"  Mean ± SD:                    {tau_stats['mean']:.1f} ± {tau_stats['std']:.1f}")
        print(f"  IQR (25th–75th):              {tau_stats['iqr_25']:.1f} – {tau_stats['iqr_75']:.1f}")
        print(f"  Cells with τ < 3:             {tau_stats['percent_below_3']:.1f}%")

    # Half-session comparison.
    half = results["half_session"]
    print(f"\n[4] HALF-SESSION COMPARISON (1st half vs 2nd half)")
    print(thin_separator)
    print(f"  {'Date':<12} {'Drop (%)':<10}")
    print(f"  {'----':<12} {'--------':<10}")
    for date in dates:
        drop = half["per_day_percent_drop"][date]
        print(f"  {date[5:]:<12} {drop:<+10.1f}")
    print(f"  Mean drop across days:        {half['mean_percent_drop']:+.1f}%")
    if "wilcoxon_p_value" in half:
        print(f"  Wilcoxon signed-rank p-value: {half['wilcoxon_p_value']:.4f}")

    # SNR.
    snr = results["snr_per_day"]
    print(f"\n[5] SIGNAL-TO-NOISE RATIO (mean / std across cells per day)")
    print(thin_separator)
    print(f"  {'Date':<12} {'SNR':<10}")
    print(f"  {'----':<12} {'---':<10}")
    for date in dates:
        print(f"  {date[5:]:<12} {snr[date]:<10.2f}")

    # Interpretation guide.
    print(f"\n{separator}")
    print("INTERPRETATION GUIDE")
    print(separator)
    print("""
  [1] Cross-day signal loss
      - A negative total change means fluorescence is dropping across days.
      - R² near 1 with a significant p-value (< 0.05) means the decline is
        steady and linear — classic photobleaching or indicator degradation.
      - R² low but total change large: signal is dropping but erratically
        (could be FOV drift, expression changes, or inconsistent laser power).
      - Modest drop (< 15-20%) across many days is usually acceptable.

  [2] Intra-session decay
      - Negative slope = fluorescence declining within a single session.
      - Drop > 10% within one session suggests meaningful intra-session bleaching;
        this can bias ΔF/F especially for late-trial analyses.
      - Low R² means the decline is noisy (motion artifacts, running-related
        fluctuations), not a clean exponential bleach.
      - Consistent drops across all days point to a systematic issue (laser too high).

  [3] Decay time constants (τ)
      - τ is in the same units as the x-axis of the fit (days or hours depending
        on which decay function was used).
      - Larger τ = slower decay = less bleaching. τ of 20+ days is healthy.
      - τ < 3 days means that cell lost most of its signal within a few sessions —
        these cells may be unreliable for longitudinal tracking.
      - A wide IQR means bleaching is uneven across the FOV (could indicate
        uneven illumination or variable indicator expression).
      - Low convergence rate (< 70%) suggests the exponential model is a poor
        fit — signal may not be decaying exponentially (good news, possibly).

  [4] Half-session comparison
      - Negative drop = second half dimmer than first (expected with bleaching).
      - If drops are consistently < -5% across days, intra-session bleaching is
        likely not a major concern for most analyses.
      - A significant Wilcoxon p-value (< 0.05) means the first-vs-second-half
        difference is consistent across days (systematic bleaching, not noise).
      - Non-significant p-value: any drops are within session-to-session noise.

  [5] Signal-to-noise ratio
      - SNR = mean(cell means) / std(cell means) per day.
      - Declining SNR across days means dimmer cells are approaching the noise
        floor — even if bright cells look fine, dim cells may become unusable.
      - Stable or increasing SNR alongside a dropping mean F suggests all cells
        are dimming proportionally (uniform bleaching, correctable with ΔF/F).
""")
    print(separator)


def print_bleaching_stats_from_metrics(
    per_day_mean: dict[str, float],
    per_frame_mean: dict[str, np.ndarray],
    half_means: dict[str, tuple[float, float]],
    taus: np.ndarray,
    snr_per_day: dict[str, float],
) -> dict:
    """Computes and prints quantitative bleaching metrics from pre-computed summaries.

    Same output as print_bleaching_stats but does not require the full signal matrices —
    accepts pre-computed SNR values instead. This allows the caller to free large arrays
    before calling this function.

    Args:
        per_day_mean: Mean fluorescence per date string.
        per_frame_mean: 1D array of per-frame means per date.
        half_means: (first_half_mean, second_half_mean) per date.
        taus: Array of tau values from compute_per_cell_decay or compute_per_cell_decay_multi.
        snr_per_day: Pre-computed signal-to-noise ratio per date.

    Returns:
        Dictionary containing all computed metrics organized by analysis category.
    """
    dates = sorted(per_day_mean.keys())
    n_days = len(dates)
    results: dict = {}

    # --- Cross-day signal loss (Panel 1) ---
    day_values = np.array([per_day_mean[d] for d in dates])
    day_indices = np.arange(n_days, dtype=float)

    total_percent_change = (day_values[-1] - day_values[0]) / day_values[0] * 100.0

    cross_day: dict = {"total_percent_change": total_percent_change}
    if n_days >= 3:
        regression = linregress(x=day_indices, y=day_values)
        cross_day["slope_per_day"] = regression.slope
        cross_day["r_squared"] = regression.rvalue ** 2
        cross_day["p_value"] = regression.pvalue
    results["cross_day"] = cross_day

    # --- Intra-session decay (Panel 2) ---
    intra_session: dict = {}
    for date in dates:
        trace = per_frame_mean[date]
        n_frames = len(trace)
        edge_count = max(n_frames // 10, 1)
        first_mean = float(trace[:edge_count].mean())
        last_mean = float(trace[-edge_count:].mean())
        percent_drop = (last_mean - first_mean) / first_mean * 100.0

        frame_indices = np.arange(n_frames, dtype=float)
        regression = linregress(x=frame_indices, y=trace)

        intra_session[date] = {
            "slope_per_frame": regression.slope,
            "r_squared": regression.rvalue ** 2,
            "percent_drop": percent_drop,
        }
    results["intra_session"] = intra_session

    # --- Decay tau distribution (Panel 3) ---
    valid_taus = taus[~np.isnan(taus)]
    n_total_cells = len(taus)
    n_converged = len(valid_taus)

    tau_stats: dict = {
        "n_converged": n_converged,
        "n_total": n_total_cells,
        "percent_converged": n_converged / max(n_total_cells, 1) * 100.0,
    }
    if n_converged > 0:
        tau_stats["median"] = float(np.median(valid_taus))
        tau_stats["mean"] = float(np.mean(valid_taus))
        tau_stats["std"] = float(np.std(valid_taus))
        tau_stats["iqr_25"] = float(np.percentile(valid_taus, 25))
        tau_stats["iqr_75"] = float(np.percentile(valid_taus, 75))
        tau_stats["percent_below_3"] = float(np.mean(valid_taus < 3.0) * 100.0)
    results["tau_distribution"] = tau_stats

    # --- Half-session comparison (Panel 4) ---
    first_halves = np.array([half_means[d][0] for d in dates])
    second_halves = np.array([half_means[d][1] for d in dates])
    per_day_drop = (second_halves - first_halves) / first_halves * 100.0

    half_stats: dict = {
        "per_day_percent_drop": {d: float(per_day_drop[i]) for i, d in enumerate(dates)},
        "mean_percent_drop": float(per_day_drop.mean()),
    }
    if n_days >= 6:
        stat, p_value = wilcoxon(x=first_halves, y=second_halves)
        half_stats["wilcoxon_statistic"] = float(stat)
        half_stats["wilcoxon_p_value"] = float(p_value)
    results["half_session"] = half_stats

    # --- Per-day SNR (pre-computed) ---
    results["snr_per_day"] = snr_per_day

    _print_stats_table(results=results, dates=dates)

    return results


def _significance_label(p_value: float) -> str:
    """Converts a p-value to a significance star label.

    Args:
        p_value: The p-value from a statistical test.

    Returns:
        A string of stars indicating significance level, or 'n.s.' if not significant.
    """
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'n.s.'


# PLOTTING
def plot_daily_mean(
    ax: plt.Axes,
    per_day_mean: dict[str, float],
    stats: dict | None = None,
) -> None:
    """Overall mean F per day as scatter + line, with optional regression overlay.

    Args:
        ax: Matplotlib axes.
        per_day_mean: Mean F per date string.
        stats: Full stats dictionary from print_bleaching_stats. When provided, overlays
            the linear regression line and a text box with R², significance, and total % change.
    """
    dates = sorted(per_day_mean.keys())
    values = [per_day_mean[d] for d in dates]
    x_values = np.arange(len(dates), dtype=float)

    ax.plot(x_values, values, 'o-', color='#2E86AB', markersize=8, linewidth=2)

    # Overlays the regression fit line and annotation when stats are available.
    if stats is not None:
        cross_day = stats["cross_day"]
        pct = cross_day["total_percent_change"]
        if "slope_per_day" in cross_day:
            slope = cross_day["slope_per_day"]
            regression = linregress(x=x_values, y=np.array(values))
            fit_line = regression.intercept + regression.slope * x_values
            ax.plot(x_values, fit_line, '--', color='#E15759', linewidth=1.5, alpha=0.7)
            stars = _significance_label(p_value=cross_day["p_value"])
            label_text = f'R²={cross_day["r_squared"]:.3f} {stars}\nΔ={pct:+.1f}%\nslope={slope:.1f} F/day'
        else:
            label_text = f'Δ={pct:+.1f}%'
        ax.text(
            0.97, 0.97, label_text,
            transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
        )

    ax.set_xticks(list(range(len(dates))))
    ax.set_xticklabels([d[5:] for d in dates], fontsize=9)
    ax.set_xlabel('Session date')
    ax.set_ylabel('Mean raw F')
    ax.set_title('Overall mean F per day')
    ax.grid(True, alpha=0.3)


def plot_intra_session(
    ax: plt.Axes,
    per_frame_mean: dict[str, np.ndarray],
    downsample: int = 50,
    stats: dict | None = None,
) -> None:
    """Per-frame mean F within each session, with optional per-session drop annotations.

    Args:
        ax: Matplotlib axes.
        per_frame_mean: 1D array of frame-level means per date.
        downsample: Bin this many frames together to reduce plot density.
        stats: Full stats dictionary from print_bleaching_stats. When provided, annotates the
            mean intra-session percent drop across all sessions.
    """
    dates = sorted(per_frame_mean.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dates)))

    for i, date in enumerate(dates):
        trace = per_frame_mean[date]
        if downsample > 1:
            n_bins = len(trace) // downsample
            trace = trace[:n_bins * downsample].reshape(n_bins, downsample).mean(axis=1)
        x = np.arange(len(trace)) * downsample

        # Appends the per-session percent drop to the legend label when stats are available.
        label = date[5:]
        if stats is not None:
            drop = stats["intra_session"][date]["percent_drop"]
            label += f' ({drop:+.1f}%)'
        ax.plot(x, trace, color=colors[i], linewidth=1, alpha=0.8, label=label)

    if stats is not None:
        drops = [stats["intra_session"][d]["percent_drop"] for d in dates]
        mean_drop = np.mean(drops)
        ax.text(
            0.97, 0.03, f'mean drop: {mean_drop:+.1f}%',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
        )

    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean raw F (across cells)')
    ax.set_title('Intra-session signal')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_decay_distribution(
    ax: plt.Axes,
    taus: np.ndarray,
    stats: dict | None = None,
) -> None:
    """Distribution of tau across cells, with optional IQR shading and convergence annotation.

    Args:
        ax: Matplotlib axes.
        taus: Array of tau values from compute_per_cell_decay.
        stats: Full stats dictionary from print_bleaching_stats. When provided, shades the IQR
            region and annotates convergence rate and percent of rapidly-bleaching cells.
    """
    valid = taus[~np.isnan(taus)]
    if len(valid) == 0:
        ax.text(0.5, 0.5, 'Not enough days\nfor decay fits',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Per-cell decay τ')
        return

    # Clips extreme outliers for a readable histogram.
    lo, hi = np.percentile(valid, [2, 98])
    clipped = valid[(valid >= lo) & (valid <= hi)]

    ax.hist(clipped, bins=50, color='#59A14F', edgecolor='white', linewidth=0.5)
    median_tau = np.median(valid)
    ax.axvline(median_tau, color='#E15759', linewidth=2, linestyle='--',
               label=f'median τ = {median_tau:.1f}')

    # Shades the IQR region and adds a summary annotation when stats are available.
    if stats is not None and "median" in stats["tau_distribution"]:
        tau_stats = stats["tau_distribution"]
        iqr_25 = tau_stats["iqr_25"]
        iqr_75 = tau_stats["iqr_75"]
        ax.axvspan(iqr_25, iqr_75, alpha=0.15, color='#59A14F', label=f'IQR: {iqr_25:.1f}–{iqr_75:.1f}')

        label_text = (
            f'{tau_stats["percent_converged"]:.0f}% converged\n'
            f'{tau_stats["percent_below_3"]:.0f}% with τ<3'
        )
        ax.text(
            0.97, 0.97, label_text,
            transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
        )

    ax.set_xlabel('Decay τ (days)')  # Uses 'hours' if compute_per_cell_decay_multi was used.
    ax.set_ylabel('Cell count')
    ax.set_title('Per-cell decay time constants')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)


def plot_half_session(
    ax: plt.Axes,
    half_means: dict[str, tuple[float, float]],
    stats: dict | None = None,
) -> None:
    """First half vs second half mean F per day, with optional percent drop labels and significance.

    Args:
        ax: Matplotlib axes.
        half_means: (first_half_mean, second_half_mean) per date.
        stats: Full stats dictionary from print_bleaching_stats. When provided, labels each bar pair
            with its percent drop and shows the Wilcoxon signed-rank significance if available.
    """
    dates = sorted(half_means.keys())
    x = np.arange(len(dates))
    first = [half_means[d][0] for d in dates]
    second = [half_means[d][1] for d in dates]

    width = 0.3
    ax.bar(x - width / 2, first, width, color='#4E79A7', label='First half', edgecolor='white')
    ax.bar(x + width / 2, second, width, color='#F28E2B', label='Second half', edgecolor='white')

    # Connects pairs with lines.
    for i in range(len(dates)):
        ax.plot([x[i] - width / 2, x[i] + width / 2], [first[i], second[i]],
                'k-', linewidth=0.8, alpha=0.5)

    # Places per-day percent drop labels above each bar pair when stats are available.
    if stats is not None:
        half_stats = stats["half_session"]
        for i, date in enumerate(dates):
            drop = half_stats["per_day_percent_drop"][date]
            bar_top = max(first[i], second[i])
            ax.text(
                x[i], bar_top, f'{drop:+.1f}%',
                ha='center', va='bottom', fontsize=7, color='#333333',
            )

        # Annotates the Wilcoxon test result when enough sessions are present.
        if "wilcoxon_p_value" in half_stats:
            p_value = half_stats["wilcoxon_p_value"]
            stars = _significance_label(p_value=p_value)
            label_text = f'Wilcoxon: p={p_value:.3f} ({stars})'
            ax.text(
                0.97, 0.97, label_text,
                transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
            )

    # Scale y-axis to data range with headroom for labels, instead of starting at 0.
    all_values = first + second
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)

    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in dates], fontsize=9)
    ax.set_xlabel('Session date')
    ax.set_ylabel('Mean raw F')
    ax.set_title('First vs second half')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')


# SUMMARY

def plot_bleaching_summary(
    sessions: dict[str, dict],
    signal_col: str = 'single_day_f',
    downsample: int = 50,
    max_cells_fit: int | None = 1000,
    max_sessions: int | None = 8,
    frame_rate: float = 10.0,
    figsize: tuple = (16, 10),
    save_path: Path | None = None,
    show: bool = True,
) -> Figure:
    """Generate 4-panel bleaching summary figure.
    1. daily mean  2. within-session bleaching
    3. Tau decay (needs >3 sessions)  4. 1st vs 2nd half of sessions

    Processes each session individually to avoid holding all signal matrices in memory at once.

    Args:
        sessions: from load_multiday_sessions().
        signal_col: raw fluorescence column name.
        downsample: frame binning for intra-session plot.
        max_cells_fit: max cells for exponential fits (None = all, slow for >3k).
        max_sessions: max number of sessions.  Code seems to crash after 7 sessions, depending on cells. Temp fix
        frame_rate: frame rate for imaging; will extract from data but for now ~10 hz
        figsize: figure size.
        save_path: if provided, saves figure to this path.
        show: call plt.show().

    Returns:
        The matplotlib Figure.
    """
    dates = sorted(sessions.keys())
    if max_sessions is not None and len(dates) > max_sessions:
        step = len(dates) // max_sessions + 1
        dates = dates[::step]
        sessions = {d: sessions[d] for d in dates}
        print(f"  Subsampled to {len(dates)} sessions (step={step})")

    # Process each session one at a time to save memory. Extract only the small summary metrics
    # and discard the full signal matrix before moving to the next session.
    print("Extracting per-session metrics (one at a time to save memory)...")
    per_day: dict[str, float] = {}
    per_frame: dict[str, np.ndarray] = {}
    halves: dict[str, tuple[float, float]] = {}
    snr_per_day: dict[str, float] = {}
    total_frames = 0
    n_cells = 0

    for date in dates:
        matrix = np.vstack(sessions[date]['data'][signal_col].to_list())
        n_cells = matrix.shape[1]
        total_frames += matrix.shape[0]

        # Per-day mean
        per_day[date] = float(matrix.mean())

        # Per-frame mean
        per_frame[date] = matrix.mean(axis=1)

        # Half-session means
        midpoint = matrix.shape[0] // 2
        halves[date] = (float(matrix[:midpoint].mean()), float(matrix[midpoint:].mean()))

        # SNR (pre-compute here so we don't need to reload signals later)
        cell_means = matrix.mean(axis=0)
        snr_per_day[date] = float(cell_means.mean() / max(cell_means.std(), 1e-12))

        del matrix  # Free the large array immediately

    print(f"  {len(dates)} sessions, {n_cells} cells, {total_frames} total frames")

    print("Computing decay fits...")
    taus = compute_per_cell_decay_multi(sessions, max_cells=max_cells_fit)

    print("Computing statistics...")
    stats = print_bleaching_stats_from_metrics(
        per_day_mean=per_day,
        per_frame_mean=per_frame,
        half_means=halves,
        taus=taus,
        snr_per_day=snr_per_day,
    )

    print("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    animal_id = sessions[dates[0]]['session_data'].get('animal_id', '??')
    fig.suptitle(f'Bleaching Check — {animal_id}', fontsize=14, fontweight='bold')

    plot_daily_mean(axes[0, 0], per_day, stats=stats)
    plot_intra_session(axes[0, 1], per_frame, downsample=downsample, stats=stats)
    plot_decay_distribution(axes[1, 0], taus, stats=stats)
    plot_half_session(axes[1, 1], halves, stats=stats)

    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == '__main__':
    from df_processing import load_multiday_sessions
    mouse_id = '26'
    mouse_dir = Path('/Users/cs963/Desktop/sun_lab_projects/datasets', mouse_id)

    sessions = load_multiday_sessions(
        mouse_dir,
        date_range=('2025-09-02', '2025-09-15'),
        auto_process=True,
    )

    fig = plot_bleaching_summary(sessions, show=True)


