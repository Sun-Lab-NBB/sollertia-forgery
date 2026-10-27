"""Provides the Mesoscope-VR runtime log parser donated to the system-agnostic runtime pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from numba import njit
import numpy as np
import polars as pl
from ataraxis_base_utilities import console, convert_bytes_to_scalar, ensure_directory_exists
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
"""The message code for the distance traveled since the active VR corridor's onset, logged when Unity terminates."""

_ERROR_CONTEXT_CUE_COUNT: int = 20
"""The number of subsequent cues included in the error context when a cue sequence fails to decompose."""

_ACTOR_TRIGGER_LEAD_CM: np.float64 = np.float64(4.0)
"""The distance, in centimeters, by which the Virtual Reality runtime registers a stimulus trigger zone crossing ahead
of the animal's tracked position. Gimbl represents the animal as a CharacterController sphere of 0.5 Unity units (5 cm)
with a 0.05 unit skin width, and Unity fires a zone trigger while that sphere overlaps the zone collider, so the
sphere's leading surface enters the zone this far before the tracked center does and its trailing surface leaves this
far after. The value is bracketed to 3.8 to 4.1 cm by the last unrewarded and first rewarded lick positions in
native-rate encoder data and equals the radius less the skin width on each side."""


@dataclass(frozen=True, slots=True)
class _Trial:
    """Describes one trial the animal ran, decomposed from the cue sequence of its corridor."""

    trial_type: int
    """The index of the trial's type in the experiment configuration's declaration order."""
    start_cm: np.float64
    """The cumulative distance, in centimeters, the animal had traveled when the trial began."""
    end_cm: np.float64
    """The cumulative distance, in centimeters, the animal had traveled when the trial ended or its corridor closed."""
    length_cm: np.float64
    """The distance, in centimeters, the animal traveled inside the trial. It is measured within the trial's own
    corridor, so it carries none of the rounding that the cumulative distances accumulate across corridors."""
    opens_corridor: bool
    """Determines whether the trial is the first of its corridor, which the animal enters partway into its first cue."""
    truncated: bool
    """Determines whether a corridor swap cut the trial short."""


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
        ValueError: If the recorded VR wall cue sequences are absent, if their distance snapshots are inconsistent,
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
    corridor, cue, trigger-zone, and trial feathers, plus the guidance feathers when the corresponding guidance events
    were recorded.

    Args:
        messages: The decoded runtime records, in archive order.
        output_directory: The path to the directory where the extracted data is written as uncompressed .feather files.
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session. Only
            required if the processed session is an experiment session.
        task_template: The VR task template supplying the trial geometry for the processed session. Present exactly
            when experiment_configuration is present, since only experiment sessions decode trial geometry.

    Raises:
        ValueError: If the recorded VR wall cue sequences are absent, if their distance snapshots are inconsistent,
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
    cue_sequence_timestamps: list[np.uint64] = []
    distance_snapshots: list[np.float64] = []

    # The timestamps are already absolute UTC values resolved during decoding.
    for timestamp, payload in messages:
        # Long payloads (> _CUE_SEQUENCE_MIN_LENGTH bytes) are VR wall cue sequences, collected only for experiment
        # sessions.
        # The length alone classifies the message, so a long payload is consumed here whether or not the session
        # collects it. Testing the session type alongside the length would let a wall cue sequence fall into the
        # state-code chain below, where its first two cue codes read as a state transition.
        if len(payload) > _CUE_SEQUENCE_MIN_LENGTH:
            if experiment_configuration is not None:
                cue_sequences.append(payload.astype(np.uint8))
                cue_sequence_timestamps.append(timestamp)

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
            traveled_distance = np.float64(convert_bytes_to_scalar(data=distance_bytes, dtype=np.dtype("<f8")))
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

        trials = _decompose_multiple_cue_sequences_into_trials(
            experiment_configuration=experiment_configuration,
            task_template=task_template,
            cue_sequences=cue_sequences,
            corridor_distances=distance_snapshots,
        )

        # The cue sequence is logged immediately before the tracker reset that opens its corridor, so its timestamp
        # marks the corridor onset.
        corridor_starts = _resolve_corridor_starts(corridor_distances=distance_snapshots)[: len(cue_sequences)]
        corridor_dataframe = pl.DataFrame({"time_us": cue_sequence_timestamps, "traveled_distance_cm": corridor_starts})
        corridor_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.VR_CORRIDOR, compression="uncompressed")

        cue_sequence, distance_sequence, trigger_start, trigger_end = _process_trial_sequence(
            experiment_configuration=experiment_configuration, task_template=task_template, trials=trials
        )

        cue_dataframe = pl.DataFrame({"vr_cue": cue_sequence, "traveled_distance_cm": distance_sequence})
        cue_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.VR_CUE, compression="uncompressed")

        trigger_zone_dataframe = pl.DataFrame(
            {"trigger_zone_start_cm": trigger_start, "trigger_zone_end_cm": trigger_end}
        )
        trigger_zone_dataframe.write_ipc(
            file=output_directory / BehaviorDataFiles.VR_TRIGGER_ZONE, compression="uncompressed"
        )

        trial_dataframe = pl.DataFrame(
            {
                "trial_type_index": np.array([trial.trial_type for trial in trials], dtype=np.int32),
                "traveled_distance_cm": np.array([trial.start_cm for trial in trials], dtype=np.float64),
            }
        )
        trial_dataframe.write_ipc(file=output_directory / BehaviorDataFiles.TRIAL, compression="uncompressed")


def _resolve_experiment_configuration(session: SessionData) -> MesoscopeExperimentConfiguration | None:
    """Loads the MesoscopeExperimentConfiguration for experiment sessions or returns None otherwise.

    Args:
        session: The loaded session whose runtime data is being parsed.

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
        geometries follow that same order.

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
    corridor_distances: list[np.float64],
) -> list[_Trial]:
    """Decomposes multiple Virtual Reality environment cue sequences into a unified sequence of trials.

    Notes:
        Handles cases where the original cue sequence was interrupted and a new sequence was generated during runtime.
        Each corridor's trials are placed against the distance traveled since that corridor's onset, the frame in which
        the runtime records its snapshot, so whether a swap cuts a trial short never depends on the rounding of the
        cumulative distances. A swap landing on a trial boundary contributes no zero-length trial.

    Args:
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session, which
            defines the canonical trial ordering to which each trial's type index refers.
        task_template: The VR task template supplying each trial's cue motif and length, joined by trial name.
        cue_sequences: The Virtual Reality environment cue sequences in the order they were used during runtime.
        corridor_distances: The distances, in centimeters, the animal traveled in each corridor before Unity
            terminated and its cue sequence was replaced. It holds one fewer element than the cue_sequences list.

    Returns:
        Every trial the animal ran, in the order it ran them.

    Raises:
        ValueError: If there is more than one cue sequence and the number of corridor distances does not match the
            number of sequences minus one, or if no cue sequences are provided.
        RuntimeError: If the function is unable to fully decompose any of the cue sequences.
    """
    if not cue_sequences:
        message = (
            "Unable to decompose input cue sequence(s) into trials. Expected at least one cue sequence as input, but "
            "received none."
        )
        console.error(message=message, error=ValueError)

    if len(cue_sequences) > 1 and len(corridor_distances) != len(cue_sequences) - 1:
        message = (
            f"Unable to decompose input cue sequence(s) into trials. Expected the number of distance snapshots "
            f"to be {len(cue_sequences) - 1} (number of sequences - 1), but encountered {len(corridor_distances)}."
        )
        console.error(message=message, error=ValueError)

    # The experiment configuration defines the canonical trial ordering to which each trial's type index refers, and
    # the VR task template supplies each trial's spatial geometry. The two are joined by trial name.
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

    minimum_motif_length = min(len(motif) for motif in trial_motifs)
    total_cue_length = sum(len(sequence) for sequence in cue_sequences)
    maximum_trials = total_cue_length // minimum_motif_length + 1

    # The animal enters each corridor already 'cue_offset_cm' into its first cue, so the corridor's first trial ends
    # after it travels that much less than the trial's corridor length.
    cue_offset = np.float64(task_template.vr_environment.cue_offset_cm)
    corridor_starts = _resolve_corridor_starts(corridor_distances=corridor_distances)

    trials: list[_Trial] = []
    for sequence_index, cue_sequence in enumerate(cue_sequences):
        trial_indices_array, trial_count, failure_position = _decompose_cue_sequence_into_trials(
            cue_sequence=cue_sequence,
            motifs_flat=motifs_flat,
            motif_starts=motif_starts,
            motif_lengths=motif_lengths,
            motif_indices=motif_indices,
            maximum_trials=maximum_trials,
        )

        if trial_count == -1:
            remaining_sequence = cue_sequence[failure_position : failure_position + _ERROR_CONTEXT_CUE_COUNT]
            message = (
                f"Unable to decompose VR wall cue sequence {sequence_index + 1} of {len(cue_sequences)} into a "
                f"sequence of trial distances. No trial motif matched at position {failure_position}. The next "
                f"{_ERROR_CONTEXT_CUE_COUNT} cues: {remaining_sequence.tolist()}"
            )
            console.error(message=message, error=RuntimeError)

        # Every corridor but the last closes at the distance recorded by its snapshot. The last one runs to the end of
        # the session, so the rest of its sequence stays listed in full.
        swap = corridor_distances[sequence_index] if sequence_index < len(cue_sequences) - 1 else None
        corridor_start = corridor_starts[sequence_index]
        trial_start = np.float64(0)
        for position, trial_index in enumerate(trial_indices_array[:trial_count].tolist()):
            trial_length = distances_array[trial_index] - (cue_offset if position == 0 else np.float64(0))
            trial_end = trial_start + trial_length
            if swap is not None and trial_end > swap:
                if swap > trial_start:
                    trials.append(
                        _Trial(
                            trial_type=trial_index,
                            start_cm=corridor_start + trial_start,
                            end_cm=corridor_start + swap,
                            length_cm=swap - trial_start,
                            opens_corridor=position == 0,
                            truncated=True,
                        )
                    )
                break
            trials.append(
                _Trial(
                    trial_type=trial_index,
                    start_cm=corridor_start + trial_start,
                    end_cm=corridor_start + trial_end,
                    length_cm=trial_length,
                    opens_corridor=position == 0,
                    truncated=False,
                )
            )
            trial_start = trial_end

    return trials


def _resolve_corridor_starts(corridor_distances: list[np.float64]) -> NDArray[np.float64]:
    """Resolves the cumulative distance, in centimeters, at which each corridor opened.

    Notes:
        The acquisition system zeroes its distance tracker at every corridor onset, so each snapshot measures the
        corridor it closes. Their running total places every corridor flush against the one before it.

    Args:
        corridor_distances: The distances the animal traveled in each closed corridor, in the order the corridors ran.

    Returns:
        The opening distance of every corridor, one more than the number of closed corridors.
    """
    totals = np.cumsum(np.array(corridor_distances, dtype=np.float64))
    return np.concatenate((np.zeros(1, dtype=np.float64), totals))


def _prepare_motif_data(
    trial_motifs: list[NDArray[np.uint8]], trial_distances: list[float]
) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
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

    # Holds the width at which the cumulative trial distance is declared and persisted, since the accumulator that sums
    # these lengths adopts their dtype and a narrower one drifts off the recorded value over a session's trials.
    distances_array: NDArray[np.float64] = np.array(trial_distances, dtype=np.float64)

    return motifs_flat, motif_starts, motif_lengths, motif_indices, distances_array


@njit(cache=True)
def _decompose_cue_sequence_into_trials(
    cue_sequence: NDArray[np.uint8],
    motifs_flat: NDArray[np.uint8],
    motif_starts: NDArray[np.int32],
    motif_lengths: NDArray[np.int32],
    motif_indices: NDArray[np.int32],
    maximum_trials: int,
) -> tuple[NDArray[np.int32], int, int]:
    """Decomposes a long sequence of Virtual Reality wall cues into individual trial motifs.

    Notes:
        Longer motifs are matched preferentially over shorter ones to prevent partial matches.

    Args:
        cue_sequence: The full Virtual Reality environment cue sequence to decompose.
        motifs_flat: All trial type motifs concatenated into a single 1D array, sorted by length.
        motif_starts: The starting index of each unique motif in the motifs_flat array.
        motif_lengths: The length of each unique motif in the motifs_flat array.
        motif_indices: The original trial type motif indices before sorting.
        maximum_trials: The maximum number of trials that can make up the entire cue sequence.

    Returns:
        A tuple of three elements. The first element is the array of trial-type indices decoded from the cue sequence,
        trimmed to the extracted trial count. The second element is the total number of trials extracted, or -1 if
        decomposition failed. The third element is the cue sequence position at which decomposition stopped, which is
        the position of the unmatched cue when the second element is -1.
    """
    trial_indices: NDArray[np.int32] = np.zeros(maximum_trials, dtype=np.int32)
    trial_count = 0
    sequence_position = 0
    sequence_length = len(cue_sequence)
    motif_count = len(motif_lengths)

    while sequence_position < sequence_length and trial_count < maximum_trials:
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
            return trial_indices[:trial_count], -1, sequence_position

    return trial_indices[:trial_count], trial_count, sequence_position


def _process_trial_sequence(
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
    trials: list[_Trial],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Resolves the cues the animal entered and the trigger zones it crossed over the trials it ran.

    Notes:
        Every decision is taken against the distance traveled since the trial began, so no two cumulative distances are
        ever compared.

    Args:
        experiment_configuration: The MesoscopeExperimentConfiguration instance for the processed session, which
            defines the canonical trial ordering to which each trial's type index refers.
        task_template: The VR task template supplying each trial's cue sequence and trigger zone, the cue catalog,
            and the corridor cue offset, joined by trial name.
        trials: The trials the animal ran, as decomposed from the cue sequences of their corridors.

    Returns:
        A tuple of four NumPy arrays. The first array stores the IDs of the Virtual Reality environment cues
        experienced by the animal during runtime. The second array stores the cumulative distance, in centimeters,
        traveled by the animal at the onset of each cue. The third array stores the cumulative distance traveled by
        the animal when the runtime registered it entering a trial's trigger zone, and the fourth the distance at
        which it registered the exit, clamped to the trial's end distance. Both sit one actor lead outside the
        declared zone boundaries. Trials that ended before their trigger zone began contribute no entry, so these two
        arrays can be shorter than the trial list.
    """
    # The experiment configuration defines the canonical trial ordering, and the VR task template supplies each
    # trial's cue sequence, trigger zone, and the cue catalog, joined by trial name.
    trial_names = list(experiment_configuration.trial_structures.keys())
    trial_geometries = _resolve_trial_geometries(task_template=task_template, trial_names=trial_names)

    cue_offset = np.float64(task_template.vr_environment.cue_offset_cm)
    cue_code_by_name = {cue.name: cue.code for cue in task_template.cues}
    cue_length_by_name = {cue.name: cue.length_cm for cue in task_template.cues}

    distances_list: list[np.float64] = []
    cues_list: list[np.uint8] = []
    trigger_zone_starts_list: list[np.float64] = []
    trigger_zone_ends_list: list[np.float64] = []

    for trial in trials:
        trial_geometry = trial_geometries[trial.trial_type]

        # The animal enters the first trial of every corridor partway into its first cue, so it reaches that cue's end
        # and every position the template declares against the corridor that much sooner.
        entry_offset = cue_offset if trial.opens_corridor else np.float64(0)

        distance_within_trial = np.float64(0)
        for cue_index, cue_name in enumerate(trial_geometry.cue_sequence):
            # A trial cut short by a corridor swap holds only the cues the animal entered before the swap.
            if trial.truncated and distance_within_trial >= trial.length_cm:
                break
            cues_list.append(np.uint8(cue_code_by_name[cue_name]))
            distances_list.append(trial.start_cm + distance_within_trial)
            distance_within_trial += cue_length_by_name[cue_name] - (entry_offset if cue_index == 0 else 0)

        # Unity registers the crossing when the actor's collider surface meets the zone, so the tracked position is one
        # actor lead short of the declared start at entry and one lead past the declared end at exit.
        trigger_start = trial_geometry.stimulus_trigger_zone_start_cm - entry_offset - _ACTOR_TRIGGER_LEAD_CM
        trigger_end = trial_geometry.stimulus_trigger_zone_end_cm - entry_offset + _ACTOR_TRIGGER_LEAD_CM

        if trigger_start <= trial.length_cm:
            trigger_zone_starts_list.append(trial.start_cm + trigger_start)
            trigger_zone_ends_list.append(
                trial.start_cm + trigger_end if trigger_end <= trial.length_cm else trial.end_cm
            )

    distances: NDArray[np.float64] = np.array(distances_list, dtype=np.float64)
    cues: NDArray[np.uint8] = np.array(cues_list, dtype=np.uint8)
    trigger_zone_starts: NDArray[np.float64] = np.array(trigger_zone_starts_list, dtype=np.float64)
    trigger_zone_ends: NDArray[np.float64] = np.array(trigger_zone_ends_list, dtype=np.float64)

    return cues, distances, trigger_zone_starts, trigger_zone_ends
