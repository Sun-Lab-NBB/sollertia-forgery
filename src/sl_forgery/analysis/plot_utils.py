"""
Shared plotting utilities for place cell analysis.

Central source for color palettes, cue/trial-type color and label lookups,
and reusable matplotlib helpers (title builder, cue shading, cue bars).

All analysis/plotting modules should import colors and helpers from here
rather than defining their own, unless needed.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import colorsys


# FONT CONFIGURATION

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Poppins', 'Liberation Sans', 'DejaVu Sans']


# CUE COLOR PALETTE

# Tableau 10 — colorblind friendly, fixed by cue ID across all experiments
CUE_COLOR_PALETTE = [
    '#4E79A7',  # Muted blue
    '#F28E2B',  # Warm orange
    '#59A14F',  # Forest green
    '#E15759',  # Soft red
    '#B07AA1',  # Dusty purple
    '#9C755F',  # Warm brown
    '#EDC948',  # Golden yellow
    '#76B7B2',  # Dusty teal
    '#FF9DA7',  # Soft pink
    '#BAB0AC',  # Warm gray
]

SPECIAL_CUE_COLORS = {
    0: '#A3A3A3',   # Light gray (gray zones)
    255: '#2D2D2D',  # Charcoal (dark periods)
}

SPECIAL_CUE_LABELS = {
    0: 'Gray',
    255: 'Dark',
}


# TRIAL TYPE PALETTES

# Light = individual traces, Dark = session average
# For example, trial type 1 (ABC) will always be blue, and trial type 2 red
TRIAL_TYPE_PALETTE = [
    '#2E86AB',  # Blue
    '#A23B72',  # Red/magenta
    '#59A14F',  # Green
    '#E15759',  # Coral
    '#B07AA1',  # Purple
]

TRIAL_TYPE_PALETTE_DARK = [
    '#0A4D68',  # Dark blue
    '#6B0848',
    '#2D6A2E',
    '#9E2B2D',
    '#7A4E7A',
]

#TODO make this work with all experiment configs
# Convenience lookup for the two standard trial types
TRIAL_TYPE_COLORS = {
    'ABC': TRIAL_TYPE_PALETTE[0],
    'ABDC': TRIAL_TYPE_PALETTE[1],
}


# COLOR / LABEL GETTERS

def get_trial_type_colors(config: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Auto-assign trial type colors from config trial_structures keys.

    Args:
        config: Experiment configuration dict with 'trial_structures' key.

    Returns:
        Tuple of (light_colors, dark_colors) dicts mapping trial_type -> hex color.
    """
    trial_types = sorted(config.get('trial_structures', {}).keys())
    colors = {}
    colors_dark = {}
    for i, tt in enumerate(trial_types):
        colors[tt] = TRIAL_TYPE_PALETTE[i % len(TRIAL_TYPE_PALETTE)]
        colors_dark[tt] = TRIAL_TYPE_PALETTE_DARK[i % len(TRIAL_TYPE_PALETTE_DARK)]
    return colors, colors_dark


def get_cue_colors(config: dict = None, max_cue_id: int = 20) -> dict[int, str]:
    """Get cue ID to color mapping.

    Assigns Tableau 10 colors by cue ID, with special colors for gray (0)
    and dark (255) zones.  Config can override via 'cue_colors' key.

    Args:
        config: Experiment configuration dict (optional).
        max_cue_id: Highest cue ID to generate colors for.

    Returns:
        Dict mapping cue_id -> hex color string.
    """
    colors = SPECIAL_CUE_COLORS.copy()

    for cue_id in range(1, max_cue_id + 1):
        idx = (cue_id - 1) % len(CUE_COLOR_PALETTE)
        colors[cue_id] = CUE_COLOR_PALETTE[idx]

    if config and 'cue_colors' in config:
        colors.update(config['cue_colors'])

    # Map string cue IDs to the same colors as their int IDs
    for int_id in range(1, max_cue_id + 1):
        str_id = chr(ord('A') + int_id - 1)  # 1→'A', 2→'B', ...
        if int_id in colors:
            colors[str_id] = colors[int_id]
            colors[f"0{str_id.lower()}"] = colors[0]  # gray zones: '0a', '0b'

    return colors


#TODO replace this with the cue_id column, though the gray zones needs better names
def get_cue_labels(config: dict = None, max_cue_id: int = 20) -> dict[int, str]:
    """Get cue ID to label mapping (A, B, C, ...).

    Args:
        config: Experiment configuration dict (optional).
        max_cue_id: Highest cue ID to generate labels for.

    Returns:
        Dict mapping cue_id -> label string.
    """
    labels = SPECIAL_CUE_LABELS.copy()

    for cue_id in range(1, max_cue_id + 1):
        if cue_id <= 26:
            labels[cue_id] = chr(ord('A') + cue_id - 1)
        else:
            labels[cue_id] = 'A' + chr(ord('A') + (cue_id - 27) % 26) #only really works for my current task
#TODO address the above ^^ for future tasks

    if config and 'cue_labels' in config:
        labels.update(config['cue_labels'])

    return labels


def scale_color(hex_color: str, factor: float) -> str:
    """Scale lightness of a hex color. factor > 1 = lighter, < 1 = darker.
    Option to add other color properties like saturation, etc
    Useful in UMAP plotting for comparing across trial types on the manifold"""
    r, g, b = int(hex_color[1:3], 16) / 255, int(hex_color[3:5], 16) / 255, int(hex_color[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def trial_type_colorscale(trial_type: str) -> list[list]:
    """Generate a Plotly colorscale from white to the trial type's standard color.

    Args:
        trial_type: Trial type name (e.g. 'ABC').

    Returns:
        Plotly-compatible colorscale (list of [fraction, color] pairs).
    """
    color = TRIAL_TYPE_COLORS.get(trial_type, '#999999')

    return [[.2, color], [1, color]]


# PLOT HELPERS

def build_title(
    description: str,
    trial_type: str | None = None,
    animal_id: str | None = None,
    date: str | None = None,
    day_x: str | None = None,
    day_y: str | None = None,
) -> str:
    """Build a consistent plot title with animal ID and date info.

    Joins non-empty parts with ' — '.  For multiday comparisons, shows
    'MM-DD vs MM-DD' or 'MM-DD split-half' when day_x == day_y.

    Args:
        description: Main plot description.
        trial_type: Trial type label.
        animal_id: Animal identifier.
        date: Single session date.
        day_x: First date for multiday comparisons.
        day_y: Second date for multiday comparisons.

    Returns:
        Formatted title string.
    """
    parts = []
    if animal_id:
        parts.append(animal_id)
    parts.append(description)
    if trial_type:
        parts.append(trial_type)
    if date:
        parts.append(date[5:])
    elif day_x and day_y:
        if day_x == day_y:
            parts.append(f'{day_x[5:]} split-half')
        else:
            parts.append(f'{day_x[5:]} vs {day_y[5:]}')
    return ' — '.join(parts)


def add_cue_shading(ax: Axes, config: dict, trial_type: str, alpha: float = 0.2, max_cm: float | None = None):
    """Add light cue region shading to an axis using standard gray.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        alpha: Shading transparency.
        max_cm: If provided, stops drawing cues at this position in cm.
    """
    cue_colors = get_cue_colors(config)
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})
    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        if max_cm is not None and pos >= max_cm:
            break
        if max_cm is not None:
            w = min(w, max_cm - pos)
        if cue_id != 0:
            color = cue_colors.get(cue_id, '#D3D3D3')  #defaults to light gray; reconsider this fallback
            ax.axvspan(pos, pos + w, alpha=alpha, color=color, zorder=0)
        pos += w


def add_cue_bar(
    ax: Axes,
    config: dict,
    trial_type: str,
    axis: str = 'x',
    bar_width: float = 0.03,
    max_cm: float | None = None,
):
    """Add a color-coded cue bar along an axis edge of a heatmap.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        axis: 'x' for bottom bar, 'y' for left bar.
        bar_width: Fraction of axis extent for bar thickness.
        max_cm: If provided, stops drawing cues at this position in cm.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    cue_colors = get_cue_colors(config)
    cue_labels = get_cue_labels(config)
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        if max_cm is not None and pos >= max_cm:
            break
        if max_cm is not None:
            w = min(w, max_cm - pos)
        color = cue_colors.get(cue_id, '#D3D3D3')

        mid = pos + w/2

        if axis == 'x':
            # x = data coords, y = axes fraction; bar sits below axes
            transform = blended_transform_factory(ax.transData, ax.transAxes)
            rect = Rectangle(
                (pos, -bar_width), w, bar_width,
                transform=transform, color=color, alpha=0.9,
                clip_on=False, zorder=10,
            )
            ax.add_patch(rect)
            if cue_id != 0:
                ax.text(mid, -bar_width / 2, cue_labels.get(cue_id, ''),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white', transform=transform,
                        clip_on=False, zorder=11)
        else:
            # x = axes fraction, y = data coords; bar sits left of axes
            transform = blended_transform_factory(ax.transAxes, ax.transData)
            rect = Rectangle(
                (-bar_width, pos), bar_width, w,
                transform=transform, color=color, alpha=0.9,
                clip_on=False, zorder=10,
            )
            ax.add_patch(rect)
            if cue_id != 0:
                ax.text(-bar_width / 2, mid, cue_labels.get(cue_id, ''),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white', transform=transform,
                        clip_on=False, zorder=11)
        pos += w

    ax.tick_params(axis='x',  pad=10)


def add_cue_shading_with_labels(
    ax: Axes,
    config: dict,
    trial_type: str,
    alpha: float = 0.15,
    label_y: float = 0.98,
    font_scale: float = 1.0,
    skip_gray: bool = True,
    max_cm: float | None = None,
):
    """Add cue region shading with labeled boxes at the top.

    Draws colored axvspan for each cue and places a rounded label box
    at the top of the shaded region.  Gray zones (cue_id=0) get shading
    but no label by default.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        alpha: Shading transparency.
        label_y: Y position for labels in axes-fraction coords.
        font_scale: Scale factor for label font size.
        skip_gray: If True, skip labels for gray zones.
        max_cm: max position to avoid plotting unused cue labels
    """
    cue_colors = get_cue_colors(config)
    cue_labels = get_cue_labels(config)
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        if max_cm is not None and pos >= max_cm:
            break
        color = cue_colors.get(cue_id, '#D3D3D3')
        ax.axvspan(pos, pos + w, alpha=alpha, color=color, zorder=0)

        if not (skip_gray and cue_id == 0):
            label = cue_labels.get(cue_id, '')
            mid = pos + w / 2
            ax.text(
                mid, label_y, label,
                ha='center', va='top',
                fontsize=9 * font_scale, fontweight='bold',
                transform=ax.get_xaxis_transform(),
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color, alpha=0.6, edgecolor='none',
                ),
            )
        pos += w


def add_cue_boundary_lines(
    ax: Axes,
    config: dict,
    trial_type: str,
    axis: str = 'both',
    color: str = 'white',
    linestyle: str = '--',
    linewidth: float = 0.8,
    alpha: float = 0.5,
):
    """Add dashed lines at cue boundaries on a heatmap.

    Draws lines at each transition between cues in the cue_sequence.
    Skips position 0 and the final track edge.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        axis: 'x', 'y', or 'both'.
        color: Line color.
        linestyle: Line style string.
        linewidth: Line width.
        alpha: Line transparency.
    """
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    boundaries = []
    pos = 0.0
    for cue_id in seq:
        pos += cue_widths[cue_id]
        boundaries.append(pos)
    boundaries = boundaries[:-1]

    line_kwargs = dict(color=color, linestyle=linestyle,
                       linewidth=linewidth, alpha=alpha, zorder=5)

    if axis in ('x', 'both'):
        for b in boundaries:
            ax.axvline(b, **line_kwargs)
    if axis in ('y', 'both'):
        for b in boundaries:
            ax.axhline(b, **line_kwargs)


def get_cue_boundaries(config: dict, trial_type: str, max_cm: float | None = None) -> list[float]:
    """Returns cumulative cue boundary positions in cm, starting at 0.

    Args:
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        max_cm: If provided, stops at the first boundary that exceeds this value.

    Returns:
        List of boundary positions starting with 0.0.
    """
    ts = config.get('trial_structures', {}).get(trial_type, {})
    cue_widths = config.get('cue_map', {})
    boundaries = [0.0]
    pos = 0.0
    for cue_id in ts.get('cue_sequence', []):
        pos += cue_widths[cue_id]
        if max_cm is not None and pos > max_cm + 1e-6:
            break
        boundaries.append(pos)
    return boundaries


def set_cue_boundary_ticks(ax: Axes, config: dict, trial_type: str, max_cm: float | None = None):
    """Set x-ticks at cue region boundaries.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        max_cm: If provided, only ticks at or below this position are included.
    """
    ticks = get_cue_boundaries(config=config, trial_type=trial_type, max_cm=max_cm)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:.0f}' for t in ticks])


def plot_pv_heatmap(
    ax: Axes,
    matrix: np.ndarray,
    config: dict,
    x_type: str,
    y_type: str,
    x_label: str,
    y_label: str,
    bin_size_cm: int = 5,
    diverge_cm: float | None = None,
    vmin: float = -0.3,
    vmax: float = 1.0,
    cmap: str = 'RdBu_r',
):
    """Render a bin×bin PV correlation heatmap with cue bars and boundary lines.

    Shared renderer for within-session (cross-track) and multiday (same-track)
    PV correlation matrices. Handles imshow, colorbar, cue bars, cue boundary
    lines, diagonal reference line, and tick padding.

    Args:
        ax: Matplotlib Axes to draw on.
        matrix: PV correlation matrix, shape (n_bins_x, n_bins_y). Will be
            transposed internally so x_type maps to the x-axis.
        config: Experiment configuration dict.
        x_type: Trial type for x-axis cue bar.
        y_type: Trial type for y-axis cue bar.
        x_label: X-axis label string.
        y_label: Y-axis label string.
        bin_size_cm: Spatial bin size in cm.
        diverge_cm: If provided, draw divergence lines at this position.
        vmin: Colorbar minimum.
        vmax: Colorbar maximum.
        cmap: Colormap name.

    Returns:
        AxesImage from imshow (for external colorbar customization if needed).
    """
    len_x = matrix.shape[0] * bin_size_cm
    len_y = matrix.shape[1] * bin_size_cm

    im = ax.imshow(
        matrix.T, origin='lower', aspect='auto',
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[0, len_x, 0, len_y],
    )
    ax.figure.colorbar(im, ax=ax, label='PV Correlation (r)', shrink=0.85)

    # Diagonal reference
    max_len = min(len_x, len_y)
    ax.plot([0, max_len], [0, max_len], color='white', linewidth=0.8,
            linestyle='--', alpha=0.5)

    # Divergence lines (within-session cross-track only)
    if diverge_cm is not None:
        ax.axvline(diverge_cm, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(diverge_cm, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Cue bars and boundary lines
    add_cue_bar(ax, config, x_type, axis='x')
    add_cue_boundary_lines(ax, config, x_type, axis='x')

    add_cue_bar(ax, config, y_type, axis='y')
    add_cue_boundary_lines(ax, config, y_type, axis='y')

    ax.tick_params(axis='x', pad=15)     #both
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    return im