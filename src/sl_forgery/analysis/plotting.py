from sl_forgery.utils.dataclass import ProjectData, ProcessedSessionData #TODO file names, class names, location
from sl_forgery.analysis.processing import Processing #TODO file names, class names, location

from sl_forgery.analysis.processing import track_length, cue_length, bin_size

import numpy as np
from pathlib import Path
import plotly
from plotly import graph_objects as go
import polars as pl


class Plotting:
    @staticmethod
    def plot_session(target_group, cell, session_data : ProcessedSessionData):
        """
        Plots Binned Fluorescence. Relies on Chelsea's initial binning implementation, currently encapsulated in Data.bin_data
        
        Args:
            mouse (str): Mouse identifier.
            session (int | str): Session number (0-indexed) or session name.
            target_group (str): "single_day" or "multi_day"
            TODO fix this docstring

        Returns:
            The figure that is displayed

        """

        # %%%%%%%%%%%%%%%%%%
        # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
        #  plotting; plot cue regions under the graph; basically thick little
        #  vlines of different colors

        cue_positions = range(0, track_length, cue_length * 2) # *2 bc of the gray region
        session_avg_df, sess_sem, result, trial_avg_df = Processing.bin_data(target_group, session_data)
        cell_val = session_avg_df['cell_{}_signal_binned'.format(cell)][-1]  # selects the last row of the col,
        # which has the avg session data

        mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
        # pulling ndarrays from polars df
        
        sem = sess_sem[cell] # Before Chelsea indexed sess_sem[0] every time, I think she meant to get the sem for the cell being plotted
        
        xaxis = np.arange(bin_size / 2, track_length, bin_size)  # plot the avg signal in center of bin

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = xaxis,
            y = mean,
            mode='lines',
            line=dict(width=3),
            name="Mean"
        ))

        upper = mean + sem
        lower = mean - sem

        fig.add_trace(go.Scatter(
            x=list(xaxis) + list(xaxis[::-1]),  # x followed by reversed x
            y=list(upper) + list(lower[::-1]),  # upper followed by reversed lower
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',  # RGBA for transparency
            line=dict(color='rgba(255,255,255,0)'),  # No border
            hoverinfo='skip',
            name='SEM'
        ))

        trial_traces = [go.Scatter(
            x = xaxis,
            y = trial_avg_df[i, cell],
            mode='lines',
            line=dict(width=2),
            name=f"Trial {i+1}",
            visible=False,
        ) for i in range(result.shape[0])]

        fig.add_traces(trial_traces)

        fig.update_layout(
            title=dict(
                text=f"Cell Fluorescence Trial Averages",
                x=.5,
            ),
            plot_bgcolor='white',
            xaxis=dict(
                title="Track position (cm)",
                range=[0, track_length]
            ),
            yaxis=dict(
                title="Flourescant Signal",
                range=[0, 6000]
            ),
            updatemenus=[dict(
                type="dropdown",
                xanchor="left", yanchor="bottom",
                x=1, y=1,
                direction="down",
                buttons=[
                    dict(
                        label="Session Average",
                        method="update",
                        args=[{"visible": [True] * 2 + [False] * len(trial_traces)}]
                    ),
                    dict(
                        label="Trial Binned Averages",
                        method="update",
                        args=[{"visible": [False] * 2 + [True] * len(trial_traces)}]
                    ),
                ]
            )],
            annotations=
            [
                *[
                    dict(
                        text=f"Cue {i+1}",
                        xref="x", yref="paper",
                        x=pos + cue_length / 2, y=1, 
                        xanchor="center", yanchor="top",
                        align="center", 
                        showarrow=False,
                    ) for i, pos in enumerate(cue_positions)          
                ],
                dict(
                    text=f"Mouse: TODO<br>Session: {ProjectData.parse_session(session_data.name)}<br>Cell: {cell}",
                    xref="paper", yref="paper",
                    x=1, y=1, 
                    xanchor="right", yanchor="bottom",
                    align="left", 
                    showarrow=False,
                )
            ],
            shapes=[
                dict(type="rect", x0=pos, x1=pos+cue_length, y0=0, y1=1, xref="x", yref="paper",
                    fillcolor="lightsteelblue", opacity=0.4, layer="below", line_width=0) 
                for pos in cue_positions
            ],
        )
        fig.show(renderer="browser")
        return fig
    
    @staticmethod
    def _add_plotting_columns(behavior_df):
        """
        Adds columns for track_position, region, cue, to a behavior dataframe if not already present

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
    def plot_umap(target_group, session_data : ProcessedSessionData):
        """
        Makes an interactive umap plot of data

        Args:
            mouse (str): Mouse identifier.
            session (int | str): Session number (0-indexed) or session name.
            target_group (str): "single_day" or "multi_day"
            TODO fix docstring

        Returns:
            The figure that is displayed

        """
        embedding, behavior_filtered = Processing.compute_umap(target_group, session_data)

        behavior_filtered = Plotting._add_plotting_columns(behavior_filtered)
            
        cue_color_map = ['gray', 'black', 'blue', 'aqua', 'gold']
        region_color_map = ['#BEBEBE','#492323', '#BEBEBE', "#6D1B76", '#BEBEBE', '#9B3753', '#BEBEBE', '#D097BB']
        region_names = ['Cue 1', 'Gray 1', 'Cue 2', 'Gray 2', 'Cue 3', 'Gray 3', 'Cue 4', 'Gray 4']

        cue_point_colors = np.array([cue_color_map[label] for label in  behavior_filtered["cue"]])
        region_point_colors = np.array([region_color_map[label] for label in  behavior_filtered["region"]])
        
        fig = go.Figure(
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers',
                marker={
                    "size": 2,
                    "opacity": 1,
                    "color": cue_point_colors
                },
                showlegend=False,
            )
        )

        # Make the cue legend
        for cue_val, color in enumerate(cue_color_map):
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],        # no actual points
                    mode="markers",
                    marker=dict(size=6, color=color),    # same color map
                    showlegend=True if cue_val != 0 else False, # don't view legend for the gray region
                    name=f" Cue {cue_val}",               # legend label
                )
            )
        
        # Make the region legend
        for cue_val, color in enumerate(region_color_map):
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],        # no actual points
                    mode="markers",
                    marker=dict(size=6, color=color),    # same color map
                    showlegend=False,
                    name=f"{region_names[cue_val]}"               # legend label
                )
            )

        # Same setting for each axis
        axis_settings = dict(
            visible=False,        # hides axis, labels, ticks
            showbackground=False, # hides background plane
            showgrid=False,       # hides grid lines
            zeroline=False        # hides zero line
        )

        fig.update_layout(
            scene=dict(
                xaxis=axis_settings,
                yaxis=axis_settings,
                zaxis=axis_settings
            ),
            updatemenus=[dict(
                type="dropdown",
                xanchor="left", yanchor="bottom",
                x=1, y=1,
                direction="down",
                buttons=[
                    dict(label="Cues",
                        method="update",
                        args=[
                            {
                                "marker.color": [cue_point_colors] + cue_color_map + region_color_map,
                                "marker.showscale": False,
                                "showlegend":  [False] + ([False] + [True] * len(cue_color_map[1:])) + [False] * len(region_color_map),
                            },
                        ]),
                    dict(label="Region",
                        method="update",
                        args=[
                            {
                                "marker.color": [region_point_colors] + cue_color_map + region_color_map,
                                "marker.showscale": False,
                                "showlegend":  [False] + [False] * len(cue_color_map) + [True] * len(region_color_map),
                            },
                        ]),
                    dict(
                        label="Track Position",
                        method="update",
                        args=[
                            {
                                "marker.color": np.array(behavior_filtered["track_position_cm"]),
                                "marker.colorscale": str(plotly.colors.make_colorscale(plotly.colors.cyclical.Twilight)).replace("'", '"'),
                                "marker.cmin": 0,
                                "marker.cmax": behavior_filtered["track_position_cm"].max(),
                                "marker.showscale": True,
                                "showlegend": False,
                            },
                        ]
                    ),
                    dict(label="Trial",
                        method="update",
                        args=[
                            {
                                "marker.color": np.array(behavior_filtered["trial"]),
                                "marker.colorscale": str(plotly.colors.make_colorscale(plotly.colors.sequential.thermal)).replace("'", '"'),
                                "marker.cmin": 0,
                                "marker.cmax": behavior_filtered["trial"].max(),
                                "marker.showscale": True,
                                "showlegend": False,
                            },
                        ]
                    ),
                ],
            )]
        )

        fig.show(renderer="browser")
        return fig



