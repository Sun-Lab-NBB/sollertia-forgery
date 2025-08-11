import numpy as np
import polars as pl

from pathlib import Path
from typing import Union

import umap

import plotly
from plotly import graph_objects as go


def make_umap_plot(
        spikes_df: pl.DataFrame,
        behavior_df: pl.DataFrame,
        save_path: Union[str, Path],
        track_length: int = 240,
        cue_length: int = 30
    ) -> None:
    """
    Standalone function that contains all the necessary code to create a umap plot. This function should eventually be replaced by a more standard architechture but could be deployed now to make some simple representation of the data which will be useful to assess cognitive map formation during runtime.
    
    Args:
        spikes_df: all spike data
        behavior_df: all behavior data
        save_path: location to save the html file containing the umap plot
        track_length: how long the track is in cm
        cue_length: all long each cue is in centimeters
    """

    save_path = Path(save_path)

    active_state_mask = (behavior_df["experiment_stage"].is_in([2, 4])) & (behavior_df["system_state"] == 2)
    behavior_filtered = behavior_df.filter(active_state_mask)
    spikes_filtered = spikes_df.filter(active_state_mask)

    spikes = spikes_filtered.to_numpy() # umap needs cells x frames

    umap_data = umap.UMAP(
        n_neighbors=100,
        n_components=3,
        min_dist=0.1,
        n_jobs=-1,
        metric='correlation'
    ).fit(spikes)

    embedding = umap_data.embedding_

     # Add cue labels and traveled distance labels (more modern data might already have this)
    def compute_track_position(distance_traveled_cm, initial_pos_cm=10):
        return (distance_traveled_cm + initial_pos_cm) % track_length

    def compute_region(track_pos):
        return int(track_pos // cue_length)

    cue_sequence = [1, 0, 2, 0, 3, 0, 4, 0]
    def compute_cue(region):
        return cue_sequence[region]

    if "track_position_cm" not in behavior_filtered.columns:
        behavior_filtered = behavior_filtered.with_columns(
            compute_track_position(pl.col("traveled_distance_cm")).alias("track_position_cm")
        )

    if "region" not in behavior_filtered.columns:
        behavior_filtered = behavior_filtered.with_columns(
            pl.col("track_position_cm").map_elements(compute_region, return_dtype=pl.Int64).alias("region")
        )
    
    if "cue" not in behavior_filtered.columns:
        behavior_filtered = behavior_filtered.with_columns(
            pl.col("region").map_elements(compute_cue, return_dtype=pl.Int64).alias("cue")
        )
        
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

    fig.write_html(save_path)