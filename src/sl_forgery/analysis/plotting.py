from enum import Enum
from typing import ClassVar
from pathlib import Path

import numpy as np
import plotly
from plotly import graph_objects as go
import polars as pl
from ataraxis_base_utilities import console

from sl_forgery.utils.dataclass import AnimalData, ProjectData, TargetGroup, ProcessedSessionData
from sl_forgery.analysis.processing import Processing, bin_size, cue_length, track_length


class ColoringStrategy(str, Enum):
    CUE = "cue"
    REGION = "region"
    TRACK_POSITION = "track_position"
    TRIAL = "trial"


class TraceType(str, Enum):
    PER_TRIAL = "per_trial"
    AVERAGE = "average"


class Plotting:
    cue_color_map: ClassVar[list[str]] = ["gray", "black", "blue", "aqua", "gold"]
    region_color_map: ClassVar[list[str]] = [
        "#BEBEBE",
        "#492323",
        "#BEBEBE",
        "#6D1B76",
        "#BEBEBE",
        "#9B3753",
        "#BEBEBE",
        "#D097BB",
    ]

    @staticmethod
    def save_fig(fig, save_path: Path | None):
        """Saves the file at the specificed path.

        Args:
            fig (plotly.graph_objects.Figure):
                The Plotly figure object to be saved.
            save_path (Path | None, optional):
                If provided, the figure is saved to this path.

        Raises:
            ValueError: If `save_path` has an unsupported file extension.
        """
        if save_path is not None:
            match save_path.suffix:
                case ".html":
                    fig.write_html(str(save_path))
                case ".pdf" | ".png" | ".jpg" | ".jpeg":
                    fig.write_image(str(save_path))
                case _:
                    console.error(f"Cannot save as {save_path.suffix} file. Save as a .html file instead.")

    @staticmethod
    def plot_session(
        target_group: str | TargetGroup, cell: int, session_data: ProcessedSessionData, save_path: Path | None = None
    ):
        """Plots binned fluorescence activity for a single cell across a session.
        Uses pre-binned data from `Processing.bin_data` to generate either
        the session average (with SEM shading) or trial-by-trial averages.
        Cues are overlaid as shaded regions and annotated along the track.

        Args:
            target_group (str | TargetGroup): Which data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day"):
                    Uses the single-day data loader and filters for identified cells
                    based on the Suite2p `iscell` mask.
                - TargetGroup.MULTI_DAY (or "multi_day"):
                    Uses the multi-day data loader without additional cell filtering.
                Passing any other string will raise a ValueError.
            cell (int): Index of the cell to plot.
            session_data (ProcessedSessionData): Object containing behavior
                and fluorescence data for the session.
            save_path (Path | None, optional):
                If provided, the figure is saved to this path.

        Returns:
            plotly.graph_objects.Figure:
        """
        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        # %%%%%%%%%%%%%%%%%%
        # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
        #  plotting; plot cue regions under the graph; basically thick little
        #  vlines of different colors

        cue_positions = range(0, track_length, cue_length * 2)  # *2 bc of the gray region
        session_avg_df, sess_sem, result, trial_avg_df = Processing.bin_data(target_group, session_data)
        cell_val = session_avg_df[f"cell_{cell}_signal_binned"][-1]  # selects the last row of the col,
        # which has the avg session data

        mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
        # pulling ndarrays from polars df

        sem = sess_sem[
            cell
        ]  # Before Chelsea indexed sess_sem[0] every time, I think she meant to get the sem for the cell being plotted

        xaxis = np.arange(bin_size / 2, track_length, bin_size)  # plot the avg signal in center of bin

        fig = go.Figure()

        # Add the trace corresponding to the session average
        fig.add_trace(go.Scatter(x=xaxis, y=mean, mode="lines", line=dict(width=3), name="Mean"))

        upper = mean + sem
        lower = mean - sem

        # Add trace for the standard error of the mean
        fig.add_trace(
            go.Scatter(
                x=list(xaxis) + list(xaxis[::-1]),  # x followed by reversed x
                y=list(upper) + list(lower[::-1]),  # upper followed by reversed lower
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",  # RGBA for transparency
                line=dict(color="rgba(255,255,255,0)"),  # No border
                hoverinfo="skip",
                name="SEM",
            )
        )

        # Add the traces for each individual session
        trial_traces = [
            go.Scatter(
                x=xaxis,
                y=trial_avg_df[i, cell],
                mode="lines",
                line=dict(width=2),
                name=f"Trial {i + 1}",
                visible=False,
            )
            for i in range(result.shape[0])
        ]

        fig.add_traces(trial_traces)

        fig.update_layout(
            title=dict(
                text="Cell Fluorescence Trial Averages",
                x=0.5,
            ),
            plot_bgcolor="white",
            xaxis=dict(title="Track position (cm)", range=[0, track_length]),
            yaxis=dict(title="Flourescant Signal", range=[0, 6000]),
            updatemenus=[
                dict(
                    type="dropdown",
                    xanchor="left",
                    yanchor="bottom",
                    x=1,
                    y=1,
                    direction="down",
                    buttons=[
                        dict(
                            label="Session Average",
                            method="update",
                            args=[{"visible": [True] * 2 + [False] * len(trial_traces)}],
                        ),
                        dict(
                            label="Trial Binned Averages",
                            method="update",
                            args=[{"visible": [False] * 2 + [True] * len(trial_traces)}],
                        ),
                    ],
                )
            ],
            annotations=[
                *[
                    dict(
                        text=f"Cue {i + 1}",
                        xref="x",
                        yref="paper",
                        x=pos + cue_length / 2,
                        y=1,
                        xanchor="center",
                        yanchor="top",
                        align="center",
                        showarrow=False,
                    )
                    for i, pos in enumerate(cue_positions)
                ],
                dict(
                    text=f"Session: {ProjectData.parse_session(session_data.name)}<br>Cell: {cell}",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1,
                    xanchor="right",
                    yanchor="bottom",
                    align="left",
                    showarrow=False,
                ),
            ],
            shapes=[
                dict(
                    type="rect",
                    x0=pos,
                    x1=pos + cue_length,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="lightsteelblue",
                    opacity=0.4,
                    layer="below",
                    line_width=0,
                )
                for pos in cue_positions
            ],
        )
        Plotting.save_fig(fig, save_path)
        fig.show(renderer="browser")
        return fig

    @staticmethod
    def plot_multi_session(
        target_group: str | TargetGroup,
        trace_type: str | TraceType,
        cell: int,
        animal: AnimalData,
        save_path: Path | None = None,
    ):
        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        if isinstance(trace_type, str):
            trace_type = TraceType(trace_type)

        frames = []

        # TODO: Task specific
        xaxis = np.arange(bin_size / 2, track_length, bin_size)
        cue_positions = range(0, track_length, cue_length * 2)  # *2 bc of the gray region

        for session in animal.sessions:
            session_avg_df, sess_sem, result, trial_avg_df = Processing.bin_data(target_group, session)

            match trace_type:
                case TraceType.PER_TRIAL:
                    traces = [
                        go.Scatter(
                            x=xaxis,
                            y=trial_avg_df[i, cell],
                            mode="lines",
                            line=dict(width=2),
                            name=f"Trial {i + 1}",
                            visible=True,
                        )
                        for i in range(result.shape[0])
                    ]

                case TraceType.AVERAGE:
                    cell_val = session_avg_df[f"cell_{cell}_signal_binned"][-1]
                    sem = sess_sem[cell]
                    mean = cell_val.to_numpy()
                    traces = []
                    traces.append(go.Scatter(x=xaxis, y=mean, mode="lines", line=dict(width=3), name="Mean"))

                    upper = mean + sem
                    lower = mean - sem

                    # Add trace for the standard error of the mean
                    traces.append(
                        go.Scatter(
                            x=list(xaxis) + list(xaxis[::-1]),  # x followed by reversed x
                            y=list(upper) + list(lower[::-1]),  # upper followed by reversed lower
                            fill="toself",
                            fillcolor="rgba(0, 0, 255, 0.2)",  # RGBA for transparency
                            line=dict(color="rgba(255,255,255,0)"),  # No border
                            hoverinfo="skip",
                            name="SEM",
                        )
                    )

            frames.append(go.Frame(data=traces, name=session.name))

        fig = go.Figure(data=frames[0].data, frames=frames)

        slider_steps = [
            {
                "method": "animate",
                "args": [[session.name], dict(mode="immediate", transition=dict(duration=0))],
                "label": ProjectData.parse_session(session.name),
            }
            for session in animal.sessions
        ]

        # Slider
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "steps": slider_steps,
                }
            ],
            updatemenus=[
                {
                    "type": "buttons",
                    "y": -0.15,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000, "redraw": True},
                                    "transition": {"duration": 1000},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                    ],
                }
            ],
        )

        # Axes
        fig.update_layout(
            title=dict(
                text="Cell Fluorescence Trial Averages",
                x=0.5,
            ),
            plot_bgcolor="white",
            xaxis=dict(title="Track position (cm)", range=[0, track_length]),
            yaxis=dict(title="Flourescant Signal", range=[0, 6000]),
        )

        fig.update_layout(
            annotations=[
                *[
                    dict(
                        text=f"Cue {i + 1}",
                        xref="x",
                        yref="paper",
                        x=pos + cue_length / 2,
                        y=1,
                        xanchor="center",
                        yanchor="top",
                        align="center",
                        showarrow=False,
                    )
                    for i, pos in enumerate(cue_positions)
                ],
            ],
            shapes=[
                dict(
                    type="rect",
                    x0=pos,
                    x1=pos + cue_length,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor="lightsteelblue",
                    opacity=0.4,
                    layer="below",
                    line_width=0,
                )
                for pos in cue_positions
            ],
        )

        Plotting._clear_axes(fig)

        Plotting.save_fig(fig, save_path)
        fig.show(renderer="browser")
        return fig

    @staticmethod
    def _add_plotting_columns(behavior_df):
        """Adds columns for track_position, region, cue, to a behavior dataframe if not already present

        Args:
            behavior_df

        Returns:
            behavior_df with additional columns

        Notes:
            Helper function to plot_umap
        """

        def compute_track_position(distance_traveled_cm, initial_pos_cm=10):
            return (distance_traveled_cm + initial_pos_cm) % track_length

        def compute_region(track_pos):
            return int(track_pos // cue_length)

        cue_sequence = [1, 0, 2, 0, 3, 0, 4, 0]

        def compute_cue(region):
            return cue_sequence[region]

        if "track_position_cm" not in behavior_df.columns:
            behavior_df = behavior_df.with_columns(
                compute_track_position(pl.col("traveled_distance_cm")).alias("track_position_cm")
            )

        if "region" not in behavior_df.columns:
            behavior_df = behavior_df.with_columns(
                pl.col("track_position_cm").map_elements(compute_region, return_dtype=pl.Int64).alias("region")
            )

        if "cue" not in behavior_df.columns:
            behavior_df = behavior_df.with_columns(
                pl.col("region").map_elements(compute_cue, return_dtype=pl.Int64).alias("cue")
            )

        return behavior_df

    @staticmethod
    def _get_point_colors(behavior_filtered, coloring_strategy: ColoringStrategy):
        match coloring_strategy:
            case ColoringStrategy.CUE:
                return np.array([Plotting.cue_color_map[label] for label in behavior_filtered["cue"]])
            case ColoringStrategy.REGION:
                return np.array([Plotting.region_color_map[label] for label in behavior_filtered["region"]])
            case ColoringStrategy.TRACK_POSITION:
                return np.array(behavior_filtered["track_position_cm"])
            case ColoringStrategy.TRIAL:
                return np.array(behavior_filtered["trial"])

    @staticmethod
    def _make_umap_scatter(embedding, behavior_filtered, coloring_strategy: ColoringStrategy):
        behavior_filtered = Plotting._add_plotting_columns(behavior_filtered)

        return go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            marker={"size": 2, "opacity": 1, "color": Plotting._get_point_colors(behavior_filtered, coloring_strategy)},
            showlegend=False,
        )

    @staticmethod
    def plot_umap(
        target_group: str | TargetGroup,
        session_data: ProcessedSessionData,
        save_path: Path | None = None,
    ):
        """Creates an interactive 3D UMAP visualization of neural activity with behavioral annotations.
        The embedding is computed from filtered spike and behavioral data, and points can be colored
        dynamically by cue, region, track position, or trial using a dropdown menu.

        Args:
            target_group (str | TargetGroup): Which data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day"):
                    Uses the single-day data loader and filters for identified cells
                    based on the Suite2p `iscell` mask.
                - TargetGroup.MULTI_DAY (or "multi_day"):
                    Uses the multi-day data loader without additional cell filtering.
                Passing any other string will raise a ValueError.
            session_data (ProcessedSessionData): Object containing references to
                behavior and spike data loaders, including file paths.
            save_path (Path | None, optional):
                If provided, the figure is saved to this path.

        Returns:
            plotly.graph_objects.Figure:
                An interactive 3D scatter plot where:
                - Default coloring shows cue identity.
                - Dropdown menu allows switching between cue, region, track position, and trial views.
                - Legends are dynamically updated based on the selected coloring scheme.

        Notes:
            - Axes, grid, and background are hidden for clarity.
            - Uses Plotly's `Scatter3d` for visualization.
            - The figure is displayed in the browser and also returned for further manipulation.
        """
        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        embedding, behavior_filtered = Processing.compute_single_session_umap(target_group, session_data)

        behavior_filtered = Plotting._add_plotting_columns(behavior_filtered)

        # TODO this is specific to Ivan's first task
        region_names = ["Cue 1", "Gray 1", "Cue 2", "Gray 2", "Cue 3", "Gray 3", "Cue 4", "Gray 4"]

        cue_point_colors = np.array([Plotting.cue_color_map[label] for label in behavior_filtered["cue"]])
        region_point_colors = np.array([Plotting.region_color_map[label] for label in behavior_filtered["region"]])

        fig = go.Figure(Plotting._make_umap_scatter(embedding, behavior_filtered, ColoringStrategy.CUE))

        # Make the cue legend
        for cue_val, color in enumerate(Plotting.cue_color_map):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],  # no actual points
                    mode="markers",
                    marker=dict(size=6, color=color),  # same color map
                    showlegend=True if cue_val != 0 else False,  # don't view legend for the gray region
                    name=f" Cue {cue_val}",  # legend label
                )
            )

        # Make the region legend
        for cue_val, color in enumerate(Plotting.region_color_map):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],  # no actual points
                    mode="markers",
                    marker=dict(size=6, color=color),  # same color map
                    showlegend=False,
                    name=f"{region_names[cue_val]}",  # legend label
                )
            )

        # Same setting for each axis
        axis_settings = dict(
            visible=False,  # hides axis, labels, ticks
            showbackground=False,  # hides background plane
            showgrid=False,  # hides grid lines
            zeroline=False,  # hides zero line
        )

        fig.update_layout(
            scene=dict(xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings),
            updatemenus=[
                dict(
                    type="dropdown",
                    xanchor="left",
                    yanchor="bottom",
                    x=1,
                    y=1,
                    direction="down",
                    buttons=[
                        dict(
                            label="Cues",
                            method="update",
                            args=[
                                {
                                    "marker.color": [
                                        cue_point_colors,
                                        *Plotting.cue_color_map,
                                        *Plotting.region_color_map,
                                    ],
                                    "marker.showscale": False,
                                    "showlegend": [False]
                                    + ([False] + [True] * len(Plotting.cue_color_map[1:]))
                                    + [False] * len(Plotting.region_color_map),
                                },
                            ],
                        ),
                        dict(
                            label="Region",
                            method="update",
                            args=[
                                {
                                    "marker.color": [
                                        region_point_colors,
                                        *Plotting.cue_color_map,
                                        *Plotting.region_color_map,
                                    ],
                                    "marker.showscale": False,
                                    "showlegend": [False]
                                    + [False] * len(Plotting.cue_color_map)
                                    + [True] * len(Plotting.region_color_map),
                                },
                            ],
                        ),
                        dict(
                            label="Track Position",
                            method="update",
                            args=[
                                {
                                    "marker.color": np.array(behavior_filtered["track_position_cm"]),
                                    "marker.colorscale": str(
                                        plotly.colors.make_colorscale(plotly.colors.cyclical.Twilight)
                                    ).replace("'", '"'),
                                    "marker.cmin": 0,
                                    "marker.cmax": behavior_filtered["track_position_cm"].max(),
                                    "marker.showscale": True,
                                    "showlegend": False,
                                },
                            ],
                        ),
                        dict(
                            label="Trial",
                            method="update",
                            args=[
                                {
                                    "marker.color": np.array(behavior_filtered["trial"]),
                                    "marker.colorscale": str(
                                        plotly.colors.make_colorscale(plotly.colors.sequential.thermal)
                                    ).replace("'", '"'),
                                    "marker.cmin": 0,
                                    "marker.cmax": behavior_filtered["trial"].max(),
                                    "marker.showscale": True,
                                    "showlegend": False,
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

        Plotting.save_fig(fig, save_path)
        fig.show(renderer="browser")
        return fig

    @staticmethod
    def plot_all_single_session_umaps(
        target_group: str | TargetGroup,
        animal: AnimalData,
        save_path: Path | None = None,
    ):
        """Creates an animated 3D UMAP visualization across all sessions for a given animal.
        Each frame corresponds to one session, showing neural activity structure over time.

        Args:
            target_group (str | TargetGroup): Data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day")
                - TargetGroup.MULTI_DAY (or "multi_day")
            animal (AnimalData): Object containing metadata and session data.
            save_path (Path | None, optional): If provided, saves the figure to this path.

        Returns:
            plotly.graph_objects.Figure:
                Interactive 3D scatter animation with a session slider.
        """
        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        frames = []
        for session in animal.sessions:
            embedding, behavior_filtered = Processing.compute_single_session_umap(
                target_group=target_group, session_data=session
            )
            frames.append(
                go.Frame(
                    data=[Plotting._make_umap_scatter(embedding, behavior_filtered, ColoringStrategy.CUE)],
                    name=session.name,
                )
            )

        fig = go.Figure(data=frames[0].data, frames=frames)

        slider_steps = [
            {
                "method": "animate",
                "args": [[session.name], dict(mode="immediate", transition=dict(duration=0))],
                "label": ProjectData.parse_session(session.name),
            }
            for session in animal.sessions
        ]

        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "steps": slider_steps,
                }
            ],
        )

        Plotting._clear_axes(fig)

        Plotting.save_fig(fig, save_path)
        fig.show(renderer="browser")
        return fig

    @staticmethod
    def _clear_axes(fig):
        """Makes it so  axes are invisible for a plotly figure"""
        axis_settings = dict(
            visible=False,  # hides axis, labels, ticks
            showbackground=False,  # hides background plane
            showgrid=False,  # hides grid lines
            zeroline=False,  # hides zero line
        )

        fig.update_layout(scene=dict(xaxis=axis_settings, yaxis=axis_settings, zaxis=axis_settings))
