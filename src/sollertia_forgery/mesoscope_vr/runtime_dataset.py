"""Provides assets for assembling runtime and experiment datasets from processed session data, including runtime
state, trial, cue, trigger zone, and guidance state data sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
import numpy as np
import polars as pl
from ataraxis_data_structures import interpolate_data

from .metadata import BehaviorDataFiles

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from sollertia_shared_assets import MesoscopeExperimentConfiguration


_CUE_UNDEFINED: int = 255
"""The sentinel value used to mask the cue column when the system is not in the run state, equal to the maximum value
of UInt8 so it sits outside the valid cue code range."""

_TRIAL_UNDEFINED: int = 65535
"""The sentinel value used to mask the trial column when the system is not in the run state, equal to the maximum
value of UInt16 so it sits outside the expected trial ID range for any realistic session."""


def assemble_runtime_dataset(
    microcontroller_data_path: Path,
    runtime_data_path: Path,
    experiment_configuration: MesoscopeExperimentConfiguration,
    reference_time: NDArray[np.uint64],
) -> pl.DataFrame:
    """Assembles the target session's runtime and experiment dataset and aligns it to the reference time vector.

    Args:
        microcontroller_data_path: The path to the processed microcontroller-data directory holding the module-parsed
            encoder feather (the wheel-distance source for interpolation).
        runtime_data_path: The path to the processed runtime-data directory holding the runtime-parsed feathers (VR
            cue, trigger zone, trial, runtime state, and the optional guidance feathers).
        experiment_configuration: Provides the mappings from integer trial type and runtime state codes to descriptive
            names, loaded from the session's raw data.
        reference_time: The reference time vector to which to align the assembled dataset.

    Returns:
        A DataFrame aligned to the reference time vector with the columns ``trial``, ``trial_type``, ``cue``,
        ``in_trigger_zone``, and ``runtime_state``, plus the optional ``reinforcing_guided`` / ``aversive_guided``
        columns when the corresponding guidance feathers were produced.
    """
    # Uses the experiment configuration file to map the integer trial type codes and runtime state codes to
    # descriptive names. Adds "undefined" as a special value for masking non-run experiment states.
    trial_type_mapping = dict(enumerate(experiment_configuration.trial_structures.keys()))
    trial_type_categories = [*trial_type_mapping.values(), "undefined"]
    trial_enum_dtype = pl.Enum(trial_type_categories)
    runtime_state_mapping = {
        state_configuration.experiment_state_code: state_name
        for state_name, state_configuration in experiment_configuration.experiment_states.items()
    }
    # State code 0 is the implicit idle state and is never listed in the experiment_states configuration.
    runtime_state_mapping[0] = "idle"
    runtime_state_enum_dtype = pl.Enum(list(runtime_state_mapping.values()))

    # Loads all experiment data sources.
    encoder_df = pl.read_ipc(source=microcontroller_data_path.joinpath(BehaviorDataFiles.ENCODER), memory_map=True)
    trigger_zones_df = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.VR_TRIGGER_ZONE), memory_map=True
    )
    cue_df = pl.read_ipc(source=runtime_data_path.joinpath(BehaviorDataFiles.VR_CUE), memory_map=True)
    trial_df = pl.read_ipc(source=runtime_data_path.joinpath(BehaviorDataFiles.TRIAL), memory_map=True)
    runtime_state_df = pl.read_ipc(source=runtime_data_path.joinpath(BehaviorDataFiles.RUNTIME_STATE), memory_map=True)

    # Extracts the trial distance and generates sequential trial numbers directly as numpy arrays, avoiding an
    # intermediate Polars DataFrame since both are only consumed by interpolate_data.
    trial_distance = trial_df["traveled_distance_cm"].to_numpy()
    trial_numbers: NDArray[np.uint32] = np.arange(1, len(trial_df) + 1, dtype=np.uint32)

    # Interpolates the traveled distance first as it's used as a reference for other interpolations.
    reference_distance: NDArray[np.float64] = interpolate_data(  # type: ignore[assignment]
        source_coordinates=encoder_df["time_us"].to_numpy(),
        source_values=encoder_df["traveled_distance_cm"].to_numpy(),
        target_coordinates=reference_time,
        is_discrete=False,
    )

    # Loads guidance state data. The processing pipeline produces separate reinforcing and aversive guidance files,
    # each conditional on whether the corresponding events were recorded during the session.
    reinforcing_guidance_file = runtime_data_path.joinpath(BehaviorDataFiles.REINFORCING_GUIDANCE)
    aversive_guidance_file = runtime_data_path.joinpath(BehaviorDataFiles.AVERSIVE_GUIDANCE)

    # Aligns all data sources to the reference time (or distance) and builds an aligned data dictionary.
    aligned_data: dict[str, NDArray[np.number]] = {
        "trial": interpolate_data(
            source_coordinates=trial_distance,
            source_values=trial_numbers,
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
        "in_trigger_zone": _check_trigger_zones(
            traversed_distance=reference_distance,
            trigger_zone_starts=trigger_zones_df["trigger_zone_start_cm"].to_numpy(),
            trigger_zone_ends=trigger_zones_df["trigger_zone_end_cm"].to_numpy(),
        ),
        "runtime_state": interpolate_data(
            source_coordinates=runtime_state_df["time_us"].to_numpy(),
            source_values=runtime_state_df["runtime_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
    }

    # Adds reinforcing guidance state if the file was produced by the processing pipeline.
    if reinforcing_guidance_file.exists():
        reinforcing_df = pl.read_ipc(source=reinforcing_guidance_file, memory_map=True)
        aligned_data["reinforcing_guided"] = interpolate_data(
            source_coordinates=reinforcing_df["time_us"].to_numpy(),
            source_values=reinforcing_df["reinforcing_guidance_state"].to_numpy().astype(np.uint8),
            target_coordinates=reference_time,
            is_discrete=True,
        )

    # Adds aversive guidance state if the file was produced by the processing pipeline.
    if aversive_guidance_file.exists():
        aversive_df = pl.read_ipc(source=aversive_guidance_file, memory_map=True)
        aligned_data["aversive_guided"] = interpolate_data(
            source_coordinates=aversive_df["time_us"].to_numpy(),
            source_values=aversive_df["aversive_guidance_state"].to_numpy().astype(np.uint8),
            target_coordinates=reference_time,
            is_discrete=True,
        )

    # Creates the aligned dataframe, replaces categorical data with Polars Enum types and optimizes how the data is
    # stored in memory by casting some columns to preferred types.
    return pl.DataFrame(aligned_data).with_columns(
        pl.col("trial_type").replace_strict(trial_type_mapping).cast(trial_enum_dtype),
        pl.col("runtime_state").replace_strict(runtime_state_mapping).cast(runtime_state_enum_dtype),
        pl.col("trial").cast(pl.UInt16),
    )


@njit(cache=True)
def _check_trigger_zones(
    traversed_distance: NDArray[np.float64],
    trigger_zone_starts: NDArray[np.float64],
    trigger_zone_ends: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Uses the provided trigger zone boundary data to determine which portion of the processed runtime data corresponds
    to the animal traversing a trigger zone.

    Args:
        traversed_distance: The cumulative distance traveled by the animal during the experiment at each sampling
            time-point.
        trigger_zone_starts: The trigger zone start boundaries for each sequential experiment trial.
        trigger_zone_ends: The trigger zone end boundaries for each sequential experiment trial.

    Returns:
        Whether each distance-point falls within a trigger zone, encoded as 1 (inside) or 0 (outside).
    """
    distance_value_count = len(traversed_distance)
    trigger_zone_count = len(trigger_zone_starts)
    in_zone: NDArray[np.uint8] = np.zeros(distance_value_count, dtype=np.uint8)

    # If no trigger zones are defined, returns the binary array set to 0 everywhere.
    if not trigger_zone_count:
        return in_zone

    # Tracks the current zone being checked.
    zone_index = 0

    # Determines whether each distance-point falls into a trigger zone. This relies on the distance and trigger zone
    # data being sorted and monotonically increasing.
    for i in range(distance_value_count):
        evaluated_distance = traversed_distance[i]

        # Moves the zone_index backward if needed (handles slight non-monotonicity in the distance data).
        while zone_index > 0 and trigger_zone_ends[zone_index - 1] >= evaluated_distance:
            zone_index -= 1

        # Checks zone boundaries starting from the current position (evaluated_distance) onward.
        while zone_index < trigger_zone_count:
            # If the checked distance is less than the start of the next trigger zone, the distance is not within a
            # trigger zone.
            if evaluated_distance < trigger_zone_starts[zone_index]:
                break

            # If the distance falls within the trigger zone, marks the corresponding mask point as 1 (in trigger zone).
            if evaluated_distance <= trigger_zone_ends[zone_index]:
                in_zone[i] = 1
                break

            # If the distance is past the evaluated trigger zone, moves to the next zone.
            zone_index += 1

    return in_zone


def _mask_non_run_experiment_data(experiment_data: pl.DataFrame) -> pl.DataFrame:
    """Masks cue, trial, and trial_type column values for non-run (idle or rest) system states.

    Sets cue and trial to their dtype sentinels (``_CUE_UNDEFINED`` / ``_TRIAL_UNDEFINED``) and trial_type to the
    "undefined" Enum member.

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
        pl.when(is_non_run).then(pl.lit(_CUE_UNDEFINED, dtype=pl.UInt8)).otherwise(pl.col("cue")).alias("cue"),
        pl.when(is_non_run).then(pl.lit(_TRIAL_UNDEFINED, dtype=pl.UInt16)).otherwise(pl.col("trial")).alias("trial"),
        pl.when(is_non_run)
        .then(pl.lit("undefined").cast(trial_type_dtype))
        .otherwise(pl.col("trial_type"))
        .alias("trial_type"),
    )
