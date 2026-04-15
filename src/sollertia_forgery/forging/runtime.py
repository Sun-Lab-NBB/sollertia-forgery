"""Provides assets for assembling runtime and experiment datasets from processed session data, including experiment
state, trial, cue, reward zone, and guidance state data sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numba import njit  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from sollertia_shared_assets import MesoscopeExperimentConfiguration
from ataraxis_data_structures import interpolate_data

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


@njit(cache=True)
def _check_reward_zones(
    traversed_distance: NDArray[np.float64],
    reward_zone_starts: NDArray[np.float64],
    reward_zone_ends: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Uses the provided reward zone boundary data to determine which portion of the processed runtime data corresponds
    to the animal traversing the reward zone.

    Args:
        traversed_distance: The NumPy array containing the cumulative distance traveled by the animal during the
            experiment at each sampling time-point.
        reward_zone_starts: The NumPy array containing the reward zone start boundaries for each sequential
            experiment trial.
        reward_zone_ends: The NumPy array containing the reward zone end boundaries for each sequential experiment
            trial.

    Returns:
        A NumPy array that stores whether each distance-point corresponds to a reward zone (1) or not (0).
    """
    # Pre-allocates the output boolean array.
    distance_value_count = len(traversed_distance)
    reward_zone_count = len(reward_zone_starts)
    in_zone = np.zeros(distance_value_count, dtype=np.uint8)

    # If no reward zones are defined, returns the binary array set to 0 everywhere.
    if reward_zone_count == 0:
        return in_zone

    # Tracks the current zone being checked
    zone_index = 0

    # Determines whether each distance-point falls into a reward zone. Note, this relies on the distance and reward zone
    # data being sorted and monotonically increasing.
    for i in range(distance_value_count):
        evaluated_distance = traversed_distance[i]

        # Moves the zone_index backward if needed (handles slight non-monotonicity in the distance data).
        while zone_index > 0 and reward_zone_ends[zone_index - 1] >= evaluated_distance:
            zone_index -= 1

        # Checks zone boundaries starting from the current position (evaluated_distance) onward.
        while zone_index < reward_zone_count:
            # If the checked distance less than the start of the next reward zone, the distance is not within a reward
            # zone.
            if evaluated_distance < reward_zone_starts[zone_index]:
                break

            # If the distance falls within the reward zone, marks the corresponding mask point as 1 (in reward zone).
            if evaluated_distance <= reward_zone_ends[zone_index]:
                in_zone[i] = 1
                break

            # If the distance is past the evaluated reward zone, moves to the next zone.
            zone_index += 1

    return in_zone


def _mask_non_run_experiment_data(experiment_data: pl.DataFrame) -> pl.DataFrame:
    """Masks cue, trial, and trial_type column values for non-run system states.

    When the system is in a non-run state (idle or rest), the cue, trial, and trial_type values are not meaningful.
    This function replaces them with the maximum value of each column's unsigned integer dtype (255 for cue's UInt8,
    65535 for trial's UInt16). These sentinels sit outside the range of legitimate cue codes and trial IDs, so sessions
    with many hundreds of trials can still be masked unambiguously. For trial_type, the value is set to the "undefined"
    Enum member.

    Args:
        experiment_data: The experiment dataset containing system_state, cue, trial, and trial_type columns.

    Returns:
        The experiment dataset with cue, trial, and trial_type values masked for non-run system states.
    """
    # Extracts the Enum dtypes to ensure type consistency.
    trial_type_dtype = experiment_data.schema["trial_type"]
    system_state_dtype = experiment_data.schema["system_state"]

    # Defines the non-run system states that should trigger masking, cast to the Enum type.
    non_run_states = pl.Series(["idle", "rest"]).cast(system_state_dtype)

    # Creates a boolean mask for rows where the system state is not "run".
    is_non_run = pl.col("system_state").is_in(non_run_states)

    return experiment_data.with_columns(
        pl.when(is_non_run).then(pl.lit(255, dtype=pl.UInt8)).otherwise(pl.col("cue")).alias("cue"),
        pl.when(is_non_run).then(pl.lit(65535, dtype=pl.UInt16)).otherwise(pl.col("trial")).alias("trial"),
        pl.when(is_non_run)
        .then(pl.lit("undefined").cast(trial_type_dtype))
        .otherwise(pl.col("trial_type"))
        .alias("trial_type"),
    )


def assemble_runtime_dataset(session_data_path: Path, reference_time: NDArray[np.uint64]) -> pl.DataFrame:
    """Assembles the target session's runtime and experiment dataset from the experiment metadata generated by the
    sollertia-forgery processing pipeline.

    Args:
        session_data_path: The path to the session's processed data directory.
        reference_time: The reference time vector to which to align the assembled dataset.

    Returns:
        The Polars DataFrame that contains the assembled experiment metadata.
    """
    # Resolves the paths to the root data directories.
    behavior_data_path = session_data_path.joinpath("processed_data", "behavior_data")
    source_data_path = session_data_path.joinpath("source_data")

    # Loads experiment configuration early to have mappings ready.
    experiment_config = MesoscopeExperimentConfiguration.from_yaml(
        source_data_path.joinpath("experiment_configuration.yaml")
    )

    # Uses the experiment configuration file to map the integer trial type codes and experiment state codes to
    # descriptive names. Adds "undefined" as a special value for masking non-run experiment states.
    trial_type_mapping = dict(enumerate(experiment_config.trial_structures.keys()))
    trial_type_categories = [*list(trial_type_mapping.values()), "undefined"]
    trial_enum_dtype = pl.Enum(trial_type_categories)
    experiment_state_mapping = {
        state_config.experiment_state_code: state_name
        for state_name, state_config in experiment_config.experiment_states.items()
    }
    experiment_state_mapping[0] = "idle"  # Adds the default system state
    experiment_state_enum_dtype = pl.Enum(list(experiment_state_mapping.values()))

    # Loads all experiment data sources.
    encoder_df = pl.read_ipc(behavior_data_path.joinpath("encoder_data.feather"), memory_map=True)
    reward_zones_df = pl.read_ipc(behavior_data_path.joinpath("vr_reward_zone_data.feather"), memory_map=True)
    cue_df = pl.read_ipc(behavior_data_path.joinpath("vr_cue_data.feather"), memory_map=True)
    trial_df = pl.read_ipc(behavior_data_path.joinpath("trial_data.feather"), memory_map=True)
    experiment_state_df = pl.read_ipc(behavior_data_path.joinpath("experiment_state_data.feather"), memory_map=True)
    guidance_state_df = pl.read_ipc(behavior_data_path.joinpath("guidance_state_data.feather"), memory_map=True)

    # Adds a trial number column to the trials dataframe.
    trial_df = trial_df.with_columns(pl.int_range(start=1, end=len(trial_df) + 1, dtype=pl.UInt32).alias("trial"))
    trial_distance = trial_df["traveled_distance_cm"].to_numpy()

    # Interpolates the traveled distance first as it's used as a reference for other interpolations.
    reference_distance: NDArray[np.float64] = interpolate_data(  # type: ignore[assignment]
        source_coordinates=encoder_df["time_us"].to_numpy(),
        source_values=encoder_df["traveled_distance_cm"].to_numpy(),
        target_coordinates=reference_time,
        is_discrete=False,
    )

    # Aligns all data sources to the reference time (or distance) and builds an aligned data dictionary.
    aligned_data: dict[str, NDArray[Any]] = {
        # Distance-based interpolations
        "trial": interpolate_data(
            source_coordinates=trial_distance,
            source_values=trial_df["trial"].to_numpy(),
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "trial_type": interpolate_data(
            source_coordinates=trial_distance,
            source_values=trial_df["trial_type_index"].to_numpy(),
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "cue": interpolate_data(
            source_coordinates=cue_df["traveled_distance_cm"].to_numpy(),
            source_values=cue_df["vr_cue"].to_numpy(),
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "in_reward_zone": _check_reward_zones(
            traversed_distance=reference_distance,
            reward_zone_starts=reward_zones_df["reward_zone_start_cm"].to_numpy(),
            reward_zone_ends=reward_zones_df["reward_zone_end_cm"].to_numpy(),
        ),
        # Time-based interpolations
        "experiment_state": interpolate_data(
            source_coordinates=experiment_state_df["time_us"].to_numpy(),
            source_values=experiment_state_df["experiment_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
        "guided": interpolate_data(
            source_coordinates=guidance_state_df["time_us"].to_numpy(),
            source_values=guidance_state_df["lick_guidance_state"].to_numpy().astype(np.uint8),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
    }

    # Creates the aligned dataframe, replaced categorical data with Polars Enum types and optimizes how the data is
    # stored in memory by casting some columns to preferred types.
    return pl.DataFrame(aligned_data).with_columns(
        [
            # Converts trial_type and experiment_state to Enum types
            pl.col("trial_type").replace_strict(trial_type_mapping).cast(trial_enum_dtype),
            pl.col("experiment_state").replace_strict(experiment_state_mapping).cast(experiment_state_enum_dtype),
            # Optimizes the trial column's datatype
            pl.col("trial").cast(pl.UInt16),
        ]
    )
