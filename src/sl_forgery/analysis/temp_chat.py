import numpy as np
import plotly
import plotly.graph_objects as go
from sl_forgery.utils.dataclass import TargetGroup, AnimalData
from sl_forgery.analysis.processing import Processing
from sl_forgery.analysis.plotting import Plotting

from typing import Dict, Any, List, Tuple

# ---------- Helpers (shared by single- and multi-session plots) ----------

def _axis_settings() -> Dict[str, Any]:
    return dict(visible=False, showbackground=False, showgrid=False, zeroline=False)

def _color_maps():
    cue_color_map = ['gray', 'black', 'blue', 'aqua', 'gold']
    region_color_map = ['#BEBEBE','#492323', '#BEBEBE', "#6D1B76", '#BEBEBE', '#9B3753', '#BEBEBE', '#D097BB']
    region_names = ['Cue 1', 'Gray 1', 'Cue 2', 'Gray 2', 'Cue 3', 'Gray 3', 'Cue 4', 'Gray 4']
    return cue_color_map, region_color_map, region_names

def _build_plot_columns(behavior_filtered) -> Dict[str, Any]:
    cue_color_map, region_color_map, region_names = _color_maps()
    cue_point_colors = np.array([cue_color_map[label] for label in behavior_filtered["cue"]])
    region_point_colors = np.array([region_color_map[label] for label in behavior_filtered["region"]])
    track_pos = np.array(behavior_filtered["track_position_cm"])
    trial_ids = np.array(behavior_filtered["trial"])

    twilight = str(plotly.colors.make_colorscale(plotly.colors.cyclical.Twilight)).replace("'", '"')
    thermal  = str(plotly.colors.make_colorscale(plotly.colors.sequential.thermal)).replace("'", '"')

    return dict(
        cue_point_colors=cue_point_colors,
        region_point_colors=region_point_colors,
        track_pos=track_pos,
        trial_ids=trial_ids,
        track_pos_cmin=0,
        track_pos_cmax=float(track_pos.max()) if track_pos.size else 1.0,
        trial_cmin=0,
        trial_cmax=float(trial_ids.max()) if trial_ids.size else 1.0,
        twilight=twilight,
        thermal=thermal,
        cue_color_map=cue_color_map,
        region_color_map=region_color_map,
        region_names=region_names,
    )

def _legend_traces(cue_color_map: List[str], region_color_map: List[str], region_names: List[str]) -> List[go.Scatter3d]:
    traces = []
    for cue_val, color in enumerate(cue_color_map):
        traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=(cue_val != 0),  # hide gray entry by default
                name=f"Cue {cue_val}",
            )
        )
    for idx, color in enumerate(region_color_map):
        traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=6, color=color),
                showlegend=False,
                name=region_names[idx],
            )
        )
    return traces

def _dropdown_buttons_for_session(colors: Dict[str, Any]) -> List[Dict[str, Any]]:
    cue_len = len(colors["cue_color_map"])
    reg_len = len(colors["region_color_map"])

    cue_legend_mask    = [False] + ([False] + [True] * (cue_len - 1)) + [False] * reg_len
    region_legend_mask = [False] + [False] * cue_len + [True] * reg_len
    no_legend_mask     = [False] * (1 + cue_len + reg_len)

    return [
        dict(
            label="Cues",
            method="update",
            args=[
                {
                    "marker.color":    [colors["cue_point_colors"]] + colors["cue_color_map"] + colors["region_color_map"],
                    "marker.showscale":[False] + [False] * (cue_len + reg_len),
                },
                {"showlegend": cue_legend_mask}
            ],
        ),
        dict(
            label="Region",
            method="update",
            args=[
                {
                    "marker.color":    [colors["region_point_colors"]] + colors["cue_color_map"] + colors["region_color_map"],
                    "marker.showscale":[False] + [False] * (cue_len + reg_len),
                },
                {"showlegend": region_legend_mask}
            ],
        ),
        dict(
            label="Track Position",
            method="update",
            args=[
                {
                    "marker.color":     [colors["track_pos"]] + colors["cue_color_map"] + colors["region_color_map"],
                    "marker.colorscale":[colors["twilight"]] + [None] * (cue_len + reg_len),
                    "marker.cmin":      [colors["track_pos_cmin"]] + [None] * (cue_len + reg_len),
                    "marker.cmax":      [colors["track_pos_cmax"]] + [None] * (cue_len + reg_len),
                    "marker.showscale": [True] + [False] * (cue_len + reg_len),
                },
                {"showlegend": no_legend_mask}
            ],
        ),
        dict(
            label="Trial",
            method="update",
            args=[
                {
                    "marker.color":     [colors["trial_ids"]] + colors["cue_color_map"] + colors["region_color_map"],
                    "marker.colorscale":[colors["thermal"]] + [None] * (cue_len + reg_len),
                    "marker.cmin":      [colors["trial_cmin"]] + [None] * (cue_len + reg_len),
                    "marker.cmax":      [colors["trial_cmax"]] + [None] * (cue_len + reg_len),
                    "marker.showscale": [True] + [False] * (cue_len + reg_len),
                },
                {"showlegend": no_legend_mask}
            ],
        ),
    ]

def _build_main_trace(embedding: np.ndarray, colors: Dict[str, Any]) -> go.Scatter3d:
    return go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=1, color=colors["cue_point_colors"]),
        showlegend=False,
        name="UMAP",
    )

# ---------- Frame builder ----------

def _build_frame(session_name: str, embedding: np.ndarray, behavior_filtered) -> Tuple[go.Frame, Dict[str, Any]]:
    colors = _build_plot_columns(behavior_filtered)
    frame_buttons = _dropdown_buttons_for_session(colors)
    frame = go.Frame(
        name=session_name,
        data=[_build_main_trace(embedding, colors)],
        layout=go.Layout(
            updatemenus=[dict(
                type="dropdown",
                xanchor="left", yanchor="bottom",
                x=1, y=1,
                direction="down",
                buttons=frame_buttons
            )]
        )
    )
    return frame, colors

# ---------- Main entrypoint ----------

def plot_all_single_session_umaps(target_group: str | TargetGroup, animal_data: AnimalData) -> go.Figure:
    """
    Single figure with a slider to transition between sessions in `animal_data.sessions`.
    Each frame mirrors the single-session dropdown behavior (Cues/Region/Track Position/Trial).
    """
    if isinstance(target_group, str):
        target_group = TargetGroup(target_group)

    sessions = getattr(animal_data, "sessions", None)
    if not sessions:
        raise ValueError("`animal_data.sessions` is empty or missing.")

    cue_color_map, region_color_map, region_names = _color_maps()

    frames: List[go.Frame] = []
    labels: List[str] = []

    first_colors: Dict[str, Any] = {}
    first_embedding: np.ndarray | None = None

    # Build frames for each session
    for i, session in enumerate(sessions):
        session_name = getattr(session, "name", f"session {i}")
        embedding, behavior_filtered = Processing.compute_single_session_umap(target_group, session)
        behavior_filtered = Plotting._add_plotting_columns(behavior_filtered)

        frame, colors = _build_frame(session_name, embedding, behavior_filtered)
        frames.append(frame)
        labels.append(session_name)

        if i == 0:
            first_colors = colors
            first_embedding = embedding

    if first_embedding is None:
        raise ValueError("No sessions produced an embedding.")

    # Seed figure with first session's cloud + legends
    fig = go.Figure()
    fig.add_trace(_build_main_trace(first_embedding, first_colors))
    for t in _legend_traces(cue_color_map, region_color_map, region_names):
        fig.add_trace(t)

    # Attach frames + slider
    fig.frames = frames
    slider_steps = [
        {
            "method": "animate",
            "args": [[lbl], {"mode": "immediate", "transition": {"duration": 0},
                             "frame": {"duration": 0, "redraw": True}}],
            "label": lbl,
        } for lbl in labels
    ]

    fig.update_layout(
        title="3D UMAP across sessions",
        scene=dict(xaxis=_axis_settings(), yaxis=_axis_settings(), zaxis=_axis_settings()),
        sliders=[{
            "active": 0,
            "pad": {"b": 0, "t": 30},
            "len": 0.8, "x": 0.1, "y": -0.08,
            "currentvalue": {"prefix": "Session: "},
            "steps": slider_steps,
        }],
        updatemenus=[
            # Initial dropdown for the first session; subsequent frames replace this.
            dict(
                type="dropdown",
                xanchor="left", yanchor="bottom",
                x=1, y=1,
                direction="down",
                buttons=_dropdown_buttons_for_session(first_colors),
            ),
            # Play/Pause
            dict(
                type="buttons",
                y=-0.18, x=0.1, xanchor="left",
                buttons=[
                    {"label": "Play",
                     "method": "animate",
                     "args": [None, {"fromcurrent": True,
                                     "frame": {"duration": 800, "redraw": True},
                                     "transition": {"duration": 300}}]},
                    {"label": "Pause",
                     "method": "animate",
                     "args": [[None], {"mode": "immediate",
                                       "frame": {"duration": 0, "redraw": False},
                                       "transition": {"duration": 0}}]},
                ],
            ),
        ],
        showlegend=True,
    )

    fig.show(renderer="browser")
    return fig
