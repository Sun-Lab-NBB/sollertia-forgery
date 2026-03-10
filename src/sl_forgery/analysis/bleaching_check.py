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


# PLOTTING
def plot_daily_mean(
    ax: plt.Axes,
    per_day_mean: dict[str, float],
) -> None:
    """Overall mean F per day as scatter + line.

    Args:
        ax: matplotlib axes.
        per_day_mean: mean F per date string.
    """
    dates = sorted(per_day_mean.keys())
    values = [per_day_mean[d] for d in dates]
    x = range(len(dates))

    ax.plot(x, values, 'o-', color='#2E86AB', markersize=8, linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([d[5:] for d in dates], fontsize=9)  # MM-DD
    ax.set_xlabel('Session date')
    ax.set_ylabel('Mean raw F')
    ax.set_title('Overall mean F per day')
    ax.grid(True, alpha=0.3)


def plot_intra_session(
    ax: plt.Axes,
    per_frame_mean: dict[str, np.ndarray],
    downsample: int = 50,
) -> None:
    """Per-frame mean F within each session.

    Args:
        ax: matplotlib axes.
        per_frame_mean: 1D array of frame-level means per date.
        downsample: bin this many frames together to reduce plot density.
    """
    dates = sorted(per_frame_mean.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dates)))

    for i, date in enumerate(dates):
        trace = per_frame_mean[date]
        if downsample > 1:
            n_bins = len(trace) // downsample
            trace = trace[:n_bins * downsample].reshape(n_bins, downsample).mean(axis=1)
        x = np.arange(len(trace)) * downsample
        ax.plot(x, trace, color=colors[i], linewidth=1, alpha=0.8, label=date[5:])

    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean raw F (across cells)')
    ax.set_title('Intra-session signal')
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_decay_distribution(
    ax: plt.Axes,
    taus: np.ndarray,
) -> None:
    """Distribution of tau across cells.

    Args:
        ax: matplotlib axes.
        taus: array of tau values from compute_per_cell_decay().
    """
    valid = taus[~np.isnan(taus)]
    if len(valid) == 0:
        ax.text(0.5, 0.5, 'Not enough days\nfor decay fits',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Per-cell decay τ')
        return

    # Clip extreme outliers for readable histogram
    lo, hi = np.percentile(valid, [2, 98])
    clipped = valid[(valid >= lo) & (valid <= hi)]

    ax.hist(clipped, bins=50, color='#59A14F', edgecolor='white', linewidth=0.5)
    median_tau = np.median(valid)
    ax.axvline(median_tau, color='#E15759', linewidth=2, linestyle='--',
               label=f'median τ = {median_tau:.1f} days')   #if using compare_2, use "hrs" instead of "days"
    ax.set_xlabel('Decay τ (days)')  #if using compare_2, use ax.set_xlabel('Decay τ (hours)')
    ax.set_ylabel('Cell count')
    ax.set_title('Per-cell decay time constants')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)


def plot_half_session(
    ax: plt.Axes,
    half_means: dict[str, tuple[float, float]],
) -> None:
    """First half vs second half mean F per day (paired scatter).

    Args:
        ax: matplotlib axes.
        half_means: (first_half, second_half) per date.
    """
    dates = sorted(half_means.keys())
    x = np.arange(len(dates))
    first = [half_means[d][0] for d in dates]
    second = [half_means[d][1] for d in dates]

    width = 0.3
    ax.bar(x - width / 2, first, width, color='#4E79A7', label='First half', edgecolor='white')
    ax.bar(x + width / 2, second, width, color='#F28E2B', label='Second half', edgecolor='white')

    # Connect pairs
    for i in range(len(dates)):
        ax.plot([x[i] - width / 2, x[i] + width / 2], [first[i], second[i]],
                'k-', linewidth=0.8, alpha=0.5)

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
    print("Extracting signals...")
    signals = extract_session_signals(sessions, signal_col)

    dates = sorted(signals.keys())
    if max_sessions is not None and len(dates) > max_sessions:
        step = len(dates) // max_sessions + 1
        dates = dates[::step]
        sessions = {d: sessions[d] for d in dates}
        print(f"  Subsampled to {len(dates)} sessions (step={step})")
    n_cells = signals[dates[0]].shape[1]
    total_frames = sum(m.shape[0] for m in signals.values())
    print(f"  {len(dates)} sessions, {n_cells} cells, {total_frames} total frames")

    print("Computing metrics...")
    per_day = compute_per_day_mean(signals)
    per_frame = compute_per_frame_mean(signals)
    halves = compute_half_session_means(signals)
    taus = compute_per_cell_decay_multi(sessions, max_cells=max_cells_fit)

    print("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    animal_id = sessions[dates[0]]['session_data'].get('animal_id', '??')
    fig.suptitle(f'Bleaching Check — {animal_id}', fontsize=14, fontweight='bold')

    plot_daily_mean(axes[0, 0], per_day)
    plot_intra_session(axes[0, 1], per_frame, downsample=downsample)
    plot_decay_distribution(axes[1, 0], taus)
    plot_half_session(axes[1, 1], halves)

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
        date_range=('2025-08-20', '2025-09-16'),
        auto_process=True,
    )

    fig = plot_bleaching_summary(sessions, show=True)


