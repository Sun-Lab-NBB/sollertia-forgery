"""Provides the Mesoscope-VR runtime log parser donated to the system-agnostic runtime pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
import numpy as np
import polars as pl
from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import SessionTypes, TaskTemplate, MesoscopeExperimentConfiguration

from .metadata import BehaviorDataFiles

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterable

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData, TrialStructure


RUNTIME_SOURCE_ID: str = "1"
"""The source ID used by the Mesoscope-VR runtime DataLogger for its log archive. Every processable session
contains exactly one runtime archive, and it always uses this ID."""

_CUE_SEQUENCE_MIN_LENGTH: int = 500
"""The exclusive byte-length threshold above which a runtime payload is treated as a VR wall cue sequence message."""

_SYSTEM_STATE_CODE: int = 1
"""The message code for VR system state data."""

_RUNTIME_STATE_CODE: int = 2
"""The message code for session runtime state data."""

_REINFORCING_GUIDANCE_STATE_CODE: int = 3
"""The message code for reinforcing trial guidance state data."""

_AVERSIVE_GUIDANCE_STATE_CODE: int = 4
"""The message code for aversive trial guidance state data."""

_DISTANCE_SNAPSHOT_CODE: int = 5
"""The message code for distance snapshot data logged when VR wall cue sequence changes."""

_ERROR_CONTEXT_CUE_COUNT: int = 20
"""The number of subsequent cues included in the error context when a cue sequence fails to decompose."""


def parse_runtime(decoded_messages: pl.DataFrame, output_directory: Path, session: SessionData) -> None:
    """Parses the decoded Mesoscope-VR runtime archive into the session's runtime behavior feathers.

    Routes each decoded payload by its leading code (or by length for VR wall cue sequences) and writes the resulting
    behavior feathers. The experiment-only feathers are written only for experiment sessions.

    Args:
        decoded_messages: The decoded runtime messages as a Polars DataFrame with a ``time_us`` UInt64 column and a
            ``payload`` Binary column, in archive order.
        output_directory: The path to the session's processed runtime-data directory where the runtime feathers are
            written.
        session: The loaded session, from which the experiment configuration is resolved.

    Raises:
        FileNotFoundError: If the session is an experiment session but its experiment configuration or VR task
            template YAML file is missing.
        ValueError: If the recorded VR wall cue sequences are absent, if their distance breakpoints are inconsistent,
            or if the experiment configuration references a trial name absent from the VR task template.
        RuntimeError: If a VR wall cue sequence cannot be fully decomposed into trial motifs.
    """
    experiment_configuration = _resolve_experiment_configuration(session=session)
    task_template = _resolve_task_template(session=session, experiment_configuration=experiment_configuration)
    messages = (
        (timestamp, np.frombuffer(payload, dtype=np.uint8))
        for timestamp, payload in zip(
            decoded_messages["time_us"].to_numpy(), decoded_messages["payload"].to_list(), strict=True
        )
    )
    _export_runtime_data(
        messages=messages,
        output_directory=output_directory,
        experiment_configuration=experiment_configuration,
        task_template=task_template,
    )


def _export_runtime_data(
    messages: Iterable[tuple[np.uint64, NDArray[np.uint8]]],
    output_directory: Path,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
    task_template: TaskTemplate | None,
) -> None:
    """Routes decoded runtime messages by payload code and exports the resulting behavior feathers.

    Writes the system-state and runtime-state feathers for every session. For experiment sessions it also writes the
    cue, trigger-zone, and trial feathers, plus the guidance feathers when the corresponding guidance events were
    recorded.

    Args:
        messages: An iterable of ``(timestamp, payload)`` records, where each payload is a uint8 byte array.
        output_directory: The path to the directory where the extracted data is written as uncompressed .feather files.
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session. Only
            required if the processed session is an experiment session.
        task_template: The VR task template supplying the trial geometry for the processed session. Present exactly
            when experiment_configuration is present, since only experiment sessions decode trial geometry.

    Raises:
        ValueError: If the recorded VR wall cue sequences are absent, if their distance breakpoints are inconsistent,
            or if the experiment configuration references a trial name absent from the VR task template.
        RuntimeError: If a VR wall cue sequence cannot be fully decomposed into trial motifs.
    """
    system_states: list[np.uint8] = []
    system_timestamps: list[np.uint64] = []
    runtime_states: list[np.uint8] = []
    runtime_timestamps: list[np.uint64] = []
    reinforcing_guidance_states: list[np.uint8] = []
    reinforcing_guidance_timestamps: list[np.uint64] = []
    aversive_guidance_states: list[np.uint8] = []
    aversive_guidance_timestamps: list[np.uint64] = []
    cue_sequences: list[NDArray[np.uint8]] = []
    distance_snapshots: list[np.float64] = []

    # The timestamps are already absolute UTC values resolved during decoding.
    for timestamp, payload in messages:
        # Long payloads (> _CUE_SEQUENCE_MIN_LENGTH bytes) are VR wall cue sequences, collected only for experiment
        # sessions.
        if len(payload) > _CUE_SEQUENCE_MIN_LENGTH and experiment_configuration is not None:
            cue_sequences.append(payload.astype(np.uint8))

        elif payload[0] == _SYSTEM_STATE_CODE:
            system_states.append(np.uint8(payload[1]))
            system_timestamps.append(timestamp)

        elif payload[0] == _RUNTIME_STATE_CODE:
            runtime_states.append(np.uint8(payload[1]))
            runtime_timestamps.append(timestamp)

        elif payload[0] == _REINFORCING_GUIDANCE_STATE_CODE:
            reinforcing_guidance_states.append(np.uint8(payload[1]))
            reinforcing_guidance_timestamps.append(timestamp)

        elif payload[0] == _AVERSIVE_GUIDANCE_STATE_CODE:
            aversive_guidance_states.append(np.uint8(payload[1]))
            aversive_guidance_timestamps.append(timestamp)

        elif payload[0] == _DISTANCE_SNAPSHOT_CODE:
            distance_bytes = payload[1:9]
            traveled_distance = np.float64(distance_bytes.view(dtype="<f8")[0])
            distance_snapshots.append(traveled_distance)

    ensure_directory_exists(path=output_directory, is_file=False)

    system_dataframe = pl.DataFrame({"time_us": system_timestamps, "system_state": system_states})
    system_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.SYSTEM_STATE, compression="uncompressed")

    runtime_dataframe = pl.DataFrame({"time_us": runtime_timestamps, "runtime_state": runtime_states})
    runtime_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.RUNTIME_STATE, compression="uncompressed")

    # Exports experiment-specific data only for experiment sessions. The task template is present exactly when the
    # experiment configuration is, so the combined guard also narrows the template to non-None for the geometry join.
    if experiment_configuration is not None and task_template is not None:
        if reinforcing_guidance_states:
            reinforcing_dataframe = pl.DataFrame(
                {"time_us": reinforcing_guidance_timestamps, "reinforcing_guidance_state": reinforcing_guidance_states}
            )
            reinforcing_dataframe.write_ipc(
                file=output_directory / BehaviorDataFiles.REINFORCING_GUIDANCE, compression="uncompressed"
            )

        if aversive_guidance_states:
            aversive_dataframe = pl.DataFrame(
                {"time_us": aversive_guidance_timestamps, "aversive_guidance_state": aversive_guidance_states}
            )
            aversive_dataframe.write_ipc(
                file=output_directory / BehaviorDataFiles.AVERSIVE_GUIDANCE, compression="uncompressed"
            )

        trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
            experiment_configuration=experiment_configuration,
            task_template=task_template,
            cue_sequences=cue_sequences,
            distance_breakpoints=distance_snapshots,
        )

        cue_sequence, distance_sequence, trigger_start, trigger_end, trial_start = _process_trial_sequence(
            experiment_configuration=experiment_configuration,
            task_template=task_template,
            trial_types=trial_types,
            trial_distances=trial_distances,
        )

        cue_dataframe = pl.DataFrame({"vr_cue": cue_sequence, "traveled_distance_cm": distance_sequence})
        cue_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.VR_CUE, compression="uncompressed")

        trigger_zone_dataframe = pl.DataFrame(
            {"trigger_zone_start_cm": trigger_start, "trigger_zone_end_cm": trigger_end}
        )
        trigger_zone_dataframe.write_ipc(
            file=output_directory / BehaviorDataFiles.VR_TRIGGER_ZONE, compression="uncompressed"
        )

        trial_dataframe = pl.DataFrame({"trial_type_index": trial_types, "traveled_distance_cm": trial_start})
        trial_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.TRIAL, compression="uncompressed")


def _resolve_experiment_configuration(session: SessionData) -> MesoscopeExperimentConfiguration | None:
    """Loads the MesoscopeExperimentConfiguration for experiment sessions or returns None otherwise.

    Args:
        session: The loaded session whose runtime data is being parsed.

    Returns:
        The loaded MesoscopeExperimentConfiguration instance for experiment sessions, or None for non-experiment
        sessions.

    Raises:
        FileNotFoundError: If the session is an experiment session but no experiment configuration YAML file is present
            at the session's canonical location.
    """
    if session.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
        return None

    experiment_configuration_path = session.raw_data.experiment_configuration_path
    if not experiment_configuration_path.is_file():
        message = (
            f"Unable to load experiment configuration for session '{session.session_name}'. No experiment "
            f"configuration YAML file was found at '{experiment_configuration_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeExperimentConfiguration.from_yaml(file_path=experiment_configuration_path)


def _resolve_task_template(
    session: SessionData, experiment_configuration: MesoscopeExperimentConfiguration | None
) -> TaskTemplate | None:
    """Loads the VR task template geometry for experiment sessions or returns None otherwise.

    Notes:
        The task template is the session's ``vr_configuration.yaml`` snapshot. It holds the spatial trial geometry the
        runtime parser needs: the cue catalog, the corridor cue offset, and each trial's cue sequence and trigger
        zone. The experiment configuration carries the per-trial stimulus parameters and the experiment state machine,
        so the two configurations are joined by trial name.

    Args:
        session: The loaded session whose runtime data is being parsed.
        experiment_configuration: The resolved experiment configuration, or None for non-experiment sessions. The
            template is loaded only when this is present, since only experiment sessions decode trial geometry.

    Returns:
        The loaded TaskTemplate instance for experiment sessions, or None for non-experiment sessions.

    Raises:
        FileNotFoundError: If the session is an experiment session but no VR task template YAML file is present at the
            session's canonical location.
    """
    if experiment_configuration is None:
        return None

    vr_configuration_path = session.raw_data.vr_configuration_path
    if not vr_configuration_path.is_file():
        message = (
            f"Unable to load the VR task template for session '{session.session_name}'. No VR configuration YAML "
            f"file was found at '{vr_configuration_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return TaskTemplate.from_yaml(file_path=vr_configuration_path)


def _resolve_trial_geometries(task_template: TaskTemplate, trial_names: list[str]) -> list[TrialStructure]:
    """Joins each experiment trial to its spatial geometry in the VR task template, keyed by trial name.

    Notes:
        The runtime parser indexes trials by their position in the experiment configuration, so the returned
        geometries follow that same order. Each trial's geometry is looked up from the task template by trial name,
        the shared key between the two configurations.

    Args:
        task_template: The VR task template holding the per-trial spatial geometry.
        trial_names: The experiment configuration's trial names, in their canonical order.

    Returns:
        The list of TrialStructure geometries, one per trial name in the given order.

    Raises:
        ValueError: If the experiment configuration references a trial name absent from the task template.
    """
    missing = [name for name in trial_names if name not in task_template.trial_structures]
    if missing:
        message = (
            f"Unable to resolve trial geometry for the runtime parser. The experiment configuration references "
            f"trial(s) {missing} absent from the VR task template. Available template trials: "
            f"{sorted(task_template.trial_structures)}."
        )
        console.error(message=message, error=ValueError)
    return [task_template.trial_structures[name] for name in trial_names]


def _decompose_multiple_cue_sequences_into_trials(
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
    cue_sequences: list[NDArray[np.uint8]],
    distance_breakpoints: list[np.float64],
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Decomposes multiple Virtual Reality environment cue sequences into a unified sequence of trials.

    Notes:
        Handles cases where the original cue sequence was interrupted and a new sequence was generated during runtime.
        Uses distance breakpoints to stitch sequences together correctly.

    Args:
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session, which
            defines the canonical trial ordering that trial_type_index refers to.
        task_template: The VR task template supplying each trial's cue motif and length, joined by trial name.
        cue_sequences: The Virtual Reality environment cue sequences in the order they were used during runtime.
        distance_breakpoints: The cumulative distances, in centimeters, at which each sequence ends. It holds one
            fewer element than the cue_sequences list.

    Returns:
        A tuple of two elements. The first element is an array of trial type indices stored in the order encountered
        during runtime. The second element is an array of cumulative distances at the end of each trial.

    Raises:
        ValueError: If there is more than one cue sequence and the number of breakpoints does not match the number of
            sequences minus one, or if no cue sequences are provided.
        RuntimeError: If the function is unable to fully decompose any of the cue sequences.
    """
    if not cue_sequences:
        message = (
            "Unable to decompose input cue sequence(s) into trials. Expected at least one cue sequence as input, but "
            "received none."
        )
        console.error(message=message, error=ValueError)

    if len(cue_sequences) > 1 and len(distance_breakpoints) != len(cue_sequences) - 1:
        message = (
            f"Unable to decompose input cue sequence(s) into trials. Expected the number of distance breakpoints "
            f"to be {len(cue_sequences) - 1} (number of sequences - 1), but encountered {len(distance_breakpoints)}."
        )
        console.error(message=message, error=ValueError)

    # The experiment configuration defines the canonical trial ordering that trial_type_index refers to, and the VR
    # task template supplies each trial's spatial geometry. The two are joined by trial name.
    trial_names = list(experiment_configuration.trial_structures.keys())
    trial_geometries = _resolve_trial_geometries(task_template=task_template, trial_names=trial_names)
    cue_code_by_name = {cue.name: cue.code for cue in task_template.cues}
    cue_length_by_name = {cue.name: cue.length_cm for cue in task_template.cues}

    trial_motifs: list[NDArray[np.uint8]] = [
        np.array([cue_code_by_name[name] for name in geometry.cue_sequence], dtype=np.uint8)
        for geometry in trial_geometries
    ]
    trial_distances: list[float] = [
        float(sum(cue_length_by_name[name] for name in geometry.cue_sequence)) for geometry in trial_geometries
    ]

    motifs_flat, motif_starts, motif_lengths, motif_indices, distances_array = _prepare_motif_data(
        trial_motifs=trial_motifs, trial_distances=trial_distances
    )

    min_motif_length = min(len(motif) for motif in trial_motifs)
    total_cue_length = sum(len(sequence) for sequence in cue_sequences)
    max_trials = total_cue_length // min_motif_length + 1

    all_trial_indices: list[int] = []
    all_trial_distances: list[float] = []
    cumulative_distance = 0.0

    for sequence_index, cue_sequence in enumerate(cue_sequences):
        trial_indices_array, trial_count = _decompose_cue_sequence_into_trials(
            cue_sequence=cue_sequence,
            motifs_flat=motifs_flat,
            motif_starts=motif_starts,
            motif_lengths=motif_lengths,
            motif_indices=motif_indices,
            max_trials=max_trials,
        )

        if trial_count == -1:
            sequence_position = 0
            trial_indices_list = trial_indices_array[:max_trials].tolist()

            for trial_index in trial_indices_list:
                if trial_index == 0 and sequence_position > 0:
                    break
                sequence_position += len(trial_motifs[trial_index])

            remaining_sequence = cue_sequence[sequence_position : sequence_position + _ERROR_CONTEXT_CUE_COUNT]
            message = (
                f"Unable to decompose VR wall cue sequence {sequence_index + 1} of {len(cue_sequences)} into a "
                f"sequence of trial distances. No trial motif matched at position {sequence_position}. The next "
                f"{_ERROR_CONTEXT_CUE_COUNT} cues: {remaining_sequence.tolist()}"
            )
            console.error(message=message, error=RuntimeError)

        sequence_trial_indices = trial_indices_array[:trial_count].tolist()

        for trial_index in sequence_trial_indices:
            trial_distance = distances_array[trial_index]
            new_cumulative_distance = cumulative_distance + trial_distance

            if sequence_index < len(cue_sequences) - 1:
                breakpoint_distance = distance_breakpoints[sequence_index]

                if new_cumulative_distance > breakpoint_distance:
                    truncated_distance = breakpoint_distance - cumulative_distance

                    if truncated_distance > 0:
                        all_trial_indices.append(trial_index)
                        all_trial_distances.append(float(breakpoint_distance))

                    cumulative_distance = breakpoint_distance
                    break

            all_trial_indices.append(trial_index)
            all_trial_distances.append(float(new_cumulative_distance))
            cumulative_distance = new_cumulative_distance

    trial_type_sequence: NDArray[np.int32] = np.array(all_trial_indices, dtype=np.int32)
    trial_distance_sequence: NDArray[np.float64] = np.array(all_trial_distances, dtype=np.float64)

    return trial_type_sequence, trial_distance_sequence


def _prepare_motif_data(
    trial_motifs: list[NDArray[np.uint8]], trial_distances: list[float]
) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Prepares the flattened motif data for faster cue sequence-to-trial decomposition.

    Args:
        trial_motifs: The trial motifs (wall cue sequences) to decompose.
        trial_distances: The trial motif distances, in centimeters.

    Returns:
        A tuple of five elements. The first element is the flattened array that stores all motifs. The second
        element is the array that stores the starting indices of each motif in the flattened array. The third
        element is the array that stores the length of each motif, in cues. The fourth element is the array
        that stores the original indices of motifs before sorting. The fifth element is the array of trial distances
        in centimeters.
    """
    motif_data: list[tuple[int, NDArray[np.uint8], int]] = [
        (index, motif, len(motif)) for index, motif in enumerate(trial_motifs)
    ]
    motif_data.sort(key=lambda entry: entry[2], reverse=True)

    total_size: int = sum(len(motif) for motif in trial_motifs)
    motif_count: int = len(trial_motifs)

    motifs_flat: NDArray[np.uint8] = np.zeros(total_size, dtype=np.uint8)
    motif_starts: NDArray[np.int32] = np.zeros(motif_count, dtype=np.int32)
    motif_lengths: NDArray[np.int32] = np.zeros(motif_count, dtype=np.int32)
    motif_indices: NDArray[np.int32] = np.zeros(motif_count, dtype=np.int32)

    current_position: int = 0
    for index, (original_index, motif, length) in enumerate(motif_data):
        motif_uint8 = motif.astype(np.uint8) if motif.dtype != np.uint8 else motif
        motifs_flat[current_position : current_position + length] = motif_uint8
        motif_starts[index] = current_position
        motif_lengths[index] = length
        motif_indices[index] = original_index
        current_position += length

    distances_array: NDArray[np.float32] = np.array(trial_distances, dtype=np.float32)

    return motifs_flat, motif_starts, motif_lengths, motif_indices, distances_array


@njit(cache=True)
def _decompose_cue_sequence_into_trials(
    cue_sequence: NDArray[np.uint8],
    motifs_flat: NDArray[np.uint8],
    motif_starts: NDArray[np.int32],
    motif_lengths: NDArray[np.int32],
    motif_indices: NDArray[np.int32],
    max_trials: int,
) -> tuple[NDArray[np.int32], int]:
    """Decomposes a long sequence of Virtual Reality wall cues into individual trial motifs.

    Notes:
        Longer motifs are matched preferentially over shorter ones to prevent partial matches.

    Args:
        cue_sequence: The full Virtual Reality environment cue sequence to decompose.
        motifs_flat: All trial type motifs concatenated into a single 1D array, sorted by length.
        motif_starts: The starting index of each unique motif in the motifs_flat array.
        motif_lengths: The length of each unique motif in the motifs_flat array.
        motif_indices: The original trial type motif indices before sorting.
        max_trials: The maximum number of trials that can make up the entire cue sequence.

    Returns:
        A tuple of two elements. The first element is the array of trial-type indices decoded from the cue sequence,
        trimmed to the extracted trial count on success and returned as the full max_trials buffer on failure. The
        second element is the total number of trials extracted, or -1 if decomposition failed.
    """
    trial_indices: NDArray[np.int32] = np.zeros(max_trials, dtype=np.int32)
    trial_count = 0
    sequence_position = 0
    sequence_length = len(cue_sequence)
    motif_count = len(motif_lengths)

    while sequence_position < sequence_length and trial_count < max_trials:
        motif_found = False

        for motif_index in range(motif_count):
            motif_length = motif_lengths[motif_index]

            if sequence_position + motif_length <= sequence_length:
                motif_start = motif_starts[motif_index]

                match = True
                for element_index in range(motif_length):
                    if cue_sequence[sequence_position + element_index] != motifs_flat[motif_start + element_index]:
                        match = False
                        break

                if match:
                    trial_indices[trial_count] = motif_indices[motif_index]
                    trial_count += 1
                    sequence_position += motif_length
                    motif_found = True
                    break

        if not motif_found:
            return trial_indices, -1

    return trial_indices[:trial_count], trial_count


def _process_trial_sequence(
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
    trial_types: NDArray[np.int32],
    trial_distances: NDArray[np.float64],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Processes the sequence of trials experienced by the animal during runtime to extract trial metadata.

    Args:
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session, which
            defines the canonical trial ordering the trial_types indices refer to.
        task_template: The VR task template supplying each trial's cue sequence and trigger zone, the cue catalog,
            and the corridor cue offset, joined by trial name.
        trial_types: The indices used to query the trial data for each trial experienced by the animal during runtime.
        trial_distances: The cumulative traveled distance, in centimeters, at which the animal fully completed each
            trial during runtime. The elements in this array use the same order as elements in the trial_types array.

    Returns:
        A tuple of five NumPy arrays. The first array stores the IDs of the Virtual Reality environment cues
        experienced by the animal during runtime. The second array stores the cumulative distance, in centimeters,
        traveled by the animal at the onset of each cue. The third array stores the cumulative distance traveled by
        the animal when it entered a trial's trigger zone, and the fourth the distance at which it left that zone,
        clamped to the trial's end distance. Trials that ended before their trigger zone began contribute no entry,
        so these two arrays can be shorter than the trial-type array. The fifth array stores the cumulative distance
        traveled by the animal at the start of each trial.
    """
    # The experiment configuration defines the canonical trial ordering, and the VR task template supplies each
    # trial's cue sequence, trigger zone, and the cue catalog, joined by trial name.
    trial_names = list(experiment_configuration.trial_structures.keys())
    trial_geometries = _resolve_trial_geometries(task_template=task_template, trial_names=trial_names)

    cue_offset = task_template.vr_environment.cue_offset_cm
    cue_code_by_name = {cue.name: cue.code for cue in task_template.cues}
    cue_length_by_name = {cue.name: cue.length_cm for cue in task_template.cues}

    distances_list: list[np.float64] = []
    cues_list: list[np.uint8] = []
    trigger_zone_starts_list: list[np.float64] = []
    trigger_zone_ends_list: list[np.float64] = []
    trial_start_distances_list: list[np.float64] = []

    cumulative_distance = np.float64(0)
    apply_offset_to_next_cue = True
    previous_trial_end_distance = np.float64(0)

    index: int
    trial: np.int32
    for index, trial in enumerate(trial_types):
        trial_geometry = trial_geometries[trial]
        trial_start_distances_list.append(previous_trial_end_distance)

        actual_trial_distance = trial_distances[index] - previous_trial_end_distance
        trial_cue_sequence = trial_geometry.cue_sequence
        distance_within_trial = np.float64(0)

        for cue_index, cue_name in enumerate(trial_cue_sequence):
            cue_code = cue_code_by_name[cue_name]
            cue_length = cue_length_by_name[cue_name]

            if apply_offset_to_next_cue and cue_index == 0:
                effective_distance_to_next_cue = cue_length - cue_offset
                apply_offset_to_next_cue = False
            else:
                effective_distance_to_next_cue = cue_length

            # Handles trial truncation when the trial was abruptly ended before completion.
            if distance_within_trial + effective_distance_to_next_cue > actual_trial_distance:
                cues_list.append(np.uint8(cue_code))
                distances_list.append(cumulative_distance)
                cumulative_distance = previous_trial_end_distance + actual_trial_distance
                apply_offset_to_next_cue = True
                break

            cues_list.append(np.uint8(cue_code))
            distances_list.append(cumulative_distance)
            cumulative_distance += effective_distance_to_next_cue
            distance_within_trial += effective_distance_to_next_cue

        trigger_start_relative = trial_geometry.stimulus_trigger_zone_start_cm
        trigger_end_relative = trial_geometry.stimulus_trigger_zone_end_cm
        trigger_start_absolute = previous_trial_end_distance + trigger_start_relative
        trigger_end_absolute = previous_trial_end_distance + trigger_end_relative

        if trigger_start_absolute <= trial_distances[index]:
            trigger_zone_starts_list.append(trigger_start_absolute)

            if trigger_end_absolute <= trial_distances[index]:
                trigger_zone_ends_list.append(trigger_end_absolute)
            else:
                trigger_zone_ends_list.append(np.float64(trial_distances[index]))

        previous_trial_end_distance = trial_distances[index]

    distances: NDArray[np.float64] = np.array(distances_list, dtype=np.float64)
    cues: NDArray[np.uint8] = np.array(cues_list, dtype=np.uint8)
    trigger_zone_starts: NDArray[np.float64] = np.array(trigger_zone_starts_list, dtype=np.float64)
    trigger_zone_ends: NDArray[np.float64] = np.array(trigger_zone_ends_list, dtype=np.float64)
    trial_start_distances: NDArray[np.float64] = np.array(trial_start_distances_list, dtype=np.float64)

    return cues, distances, trigger_zone_starts, trigger_zone_ends, trial_start_distances
