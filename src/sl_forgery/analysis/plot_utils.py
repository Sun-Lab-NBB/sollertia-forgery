"""
Shared plotting utilities for place cell analysis.

Central source for color palettes, cue/trial-type color and label lookups,
and reusable matplotlib helpers (title builder, cue shading, cue bars).

All analysis/plotting modules should import colors and helpers from here
rather than defining their own, unless needed.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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
    0: '#D3D3D3',   # Light gray (gray zones)
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

    return colors


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


def add_cue_shading(ax: Axes, config: dict, trial_type: str, alpha: float = 0.2):
    """Add light cue region shading to an axis using standard gray.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        alpha: Shading transparency.
    """
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})
    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        if cue_id != 0:
            color = cue_colors.get(cue_id, '#D3D3D3')  #defaults to light gray; reconsider this fallback
            ax.axvspan(pos, pos + w, alpha=alpha, color=color, zorder=0)
        pos += w


def add_cue_bar(
    ax: Axes,
    config: dict,
    trial_type: str,
    axis: str = 'x',
    bar_width: float = 0.02,
):
    """Add a color-coded cue bar along an axis edge of a heatmap.

    Args:
        ax: Matplotlib Axes.
        config: Experiment configuration dict.
        trial_type: Trial type for cue layout.
        axis: 'x' for bottom bar, 'y' for left bar.
        bar_width: Fraction of axis extent for bar thickness.
    """
    cue_colors = get_cue_colors(config)
    cue_labels = get_cue_labels(config)
    ts = config.get('trial_structures', {}).get(trial_type, {})
    seq = ts.get('cue_sequence', [])
    cue_widths = config.get('cue_map', {})

    pos = 0.0
    for cue_id in seq:
        w = cue_widths[cue_id]
        color = cue_colors.get(cue_id, '#D3D3D3')
        mid = pos + w/2

        if axis == 'x':
            ax.axvspan(pos, pos + w, ymin=0, ymax=bar_width,
                       color=color, alpha=0.9, clip_on=False, zorder=10)
            if cue_id != 0:
                ax.text(mid, bar_width / 2, cue_labels.get(cue_id, ''),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white', transform=ax.get_xaxis_transform(),
                        clip_on=False, zorder=11)
        else:
            ax.axhspan(pos, pos + w, xmin=0, xmax=bar_width,
                       color=color, alpha=0.9, clip_on=False, zorder=10)
            if cue_id != 0:
                ax.text(bar_width / 2, mid, cue_labels.get(cue_id, ''),
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white', transform=ax.get_yaxis_transform(),
                        clip_on=False, zorder=11)