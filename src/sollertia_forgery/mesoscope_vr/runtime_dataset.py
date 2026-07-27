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

    encoder_data_frame = pl.read_ipc(
        source=microcontroller_data_path.joinpath(BehaviorDataFiles.ENCODER), memory_map=True
    )
    trigger_zones_data_frame = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.VR_TRIGGER_ZONE), memory_map=True
    )
    cue_data_frame = pl.read_ipc(source=runtime_data_path.joinpath(BehaviorDataFiles.VR_CUE), memory_map=True)
    trial_data_frame = pl.read_ipc(source=runtime_data_path.joinpath(BehaviorDataFiles.TRIAL), memory_map=True)
    runtime_state_data_frame = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.RUNTIME_STATE), memory_map=True
    )

    # Extracts the trial distance and generates sequential trial numbers directly as numpy arrays, avoiding an
    # intermediate Polars DataFrame since both are only consumed by interpolate_data.
    trial_distance = trial_data_frame["traveled_distance_cm"].to_numpy()
    trial_numbers: NDArray[np.uint32] = np.arange(1, len(trial_data_frame) + 1, dtype=np.uint32)

    reference_distance: NDArray[np.float64] = interpolate_data(  # type: ignore[assignment]
        source_coordinates=encoder_data_frame["time_us"].to_numpy(),
        source_values=encoder_data_frame["traveled_distance_cm"].to_numpy(),
        target_coordinates=reference_time,
        is_discrete=False,
    )

    # Loads guidance state data. The processing pipeline produces separate reinforcing and aversive guidance files,
    # each conditional on whether the corresponding events were recorded during the session.
    reinforcing_guidance_file = runtime_data_path.joinpath(BehaviorDataFiles.REINFORCING_GUIDANCE)
    aversive_guidance_file = runtime_data_path.joinpath(BehaviorDataFiles.AVERSIVE_GUIDANCE)

    aligned_data: dict[str, NDArray[np.number]] = {
        "trial": interpolate_data(
            source_coordinates=trial_distance,
            source_values=trial_numbers,
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "trial_type": interpolate_data(
            source_coordinates=trial_distance,
            source_values=trial_data_frame["trial_type_index"].to_numpy(),
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "cue": interpolate_data(
            source_coordinates=cue_data_frame["traveled_distance_cm"].to_numpy(),
            source_values=cue_data_frame["vr_cue"].to_numpy(),
            target_coordinates=reference_distance,
            is_discrete=True,
        ),
        "in_trigger_zone": _check_trigger_zones(
            traversed_distance=reference_distance,
            trigger_zone_starts=trigger_zones_data_frame["trigger_zone_start_cm"].to_numpy(),
            trigger_zone_ends=trigger_zones_data_frame["trigger_zone_end_cm"].to_numpy(),
        ),
        "runtime_state": interpolate_data(
            source_coordinates=runtime_state_data_frame["time_us"].to_numpy(),
            source_values=runtime_state_data_frame["runtime_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
    }

    if reinforcing_guidance_file.exists():
        reinforcing_data_frame = pl.read_ipc(source=reinforcing_guidance_file, memory_map=True)
        aligned_data["reinforcing_guided"] = interpolate_data(
            source_coordinates=reinforcing_data_frame["time_us"].to_numpy(),
            source_values=reinforcing_data_frame["reinforcing_guidance_state"].to_numpy().astype(np.uint8),
            target_coordinates=reference_time,
            is_discrete=True,
        )

    if aversive_guidance_file.exists():
        aversive_data_frame = pl.read_ipc(source=aversive_guidance_file, memory_map=True)
        aligned_data["aversive_guided"] = interpolate_data(
            source_coordinates=aversive_data_frame["time_us"].to_numpy(),
            source_values=aversive_data_frame["aversive_guidance_state"].to_numpy().astype(np.uint8),
            target_coordinates=reference_time,
            is_discrete=True,
        )

    return pl.DataFrame(aligned_data).with_columns(
        pl.col("trial_type").replace_strict(trial_type_mapping).cast(trial_enum_dtype),
        pl.col("runtime_state").replace_strict(runtime_state_mapping).cast(runtime_state_enum_dtype),
        pl.col("trial").cast(pl.UInt16),
    )


def mask_non_run_experiment_data(experiment_data: pl.DataFrame) -> pl.DataFrame:
    """Masks cue, trial, and trial_type column values for non-run (idle or rest) system states.

    Sets cue and trial to their dtype sentinels (``_CUE_UNDEFINED`` / ``_TRIAL_UNDEFINED``) and trial_type to the
    "undefined" Enum member.

    Args:
        experiment_data: The experiment dataset containing system_state, cue, trial, and trial_type columns.

    Returns:
        The experiment dataset with cue, trial, and trial_type values masked for non-run system states.
    """
    trial_type_dtype = experiment_data.schema["trial_type"]
    system_state_dtype = experiment_data.schema["system_state"]

    non_run_states = pl.Series(["idle", "rest"]).cast(system_state_dtype)

    is_non_run = pl.col("system_state").is_in(non_run_states)

    return experiment_data.with_columns(
        pl.when(is_non_run).then(pl.lit(_CUE_UNDEFINED, dtype=pl.UInt8)).otherwise(pl.col("cue")).alias("cue"),
        pl.when(is_non_run).then(pl.lit(_TRIAL_UNDEFINED, dtype=pl.UInt16)).otherwise(pl.col("trial")).alias("trial"),
        pl.when(is_non_run)
        .then(pl.lit("undefined").cast(trial_type_dtype))
        .otherwise(pl.col("trial_type"))
        .alias("trial_type"),
    )


def clip_to_runtime_end(assembled_data: pl.DataFrame, runtime_data_path: Path) -> pl.DataFrame:
    """Discards the assembled samples acquired after the session's runtime ended.

    Notes:
        Session teardown stops the acquisition assets in sequence, so each asset contributes data for a different
        span past the end of the runtime. The cameras stop about a second after the runtime, the mesoscope continues
        for several more seconds, and the microcontrollers log for several more minutes. Clipping the fully assembled
        dataset at the final runtime-state entry removes that span from every column at once, which keeps the
        sub-dataset assemblers free of teardown-specific handling.

        On the fluorescence clock the trailing samples carry the last camera value held constant, so clipping also
        removes fabricated data. On a camera clock every trailing sample is acquired, so clipping ends the session
        at the runtime rather than at the camera teardown.

    Args:
        assembled_data: The fully assembled DataFrame, ordered by its session's reference clock.
        runtime_data_path: The path to the session's processed runtime-data directory.

    Returns:
        The DataFrame containing only the samples acquired at or before the end of the runtime.
    """
    runtime_state_data = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.RUNTIME_STATE), memory_map=True
    )
    runtime_end_time = runtime_state_data["time_us"][-1]
    return assembled_data.filter(pl.col("time_us") <= runtime_end_time)


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

    zone_index = 0

    # Determines whether each distance-point falls into a trigger zone. This relies on the distance and trigger zone
    # data being sorted and monotonically increasing.
    for sample_index in range(distance_value_count):
        evaluated_distance = traversed_distance[sample_index]

        # Moves the zone_index backward if needed (handles slight non-monotonicity in the distance data).
        while zone_index > 0 and trigger_zone_ends[zone_index - 1] >= evaluated_distance:
            zone_index -= 1

        while zone_index < trigger_zone_count:
            if evaluated_distance < trigger_zone_starts[zone_index]:
                break

            if evaluated_distance <= trigger_zone_ends[zone_index]:
                in_zone[sample_index] = 1
                break

            zone_index += 1

    return in_zone
