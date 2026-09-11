"""Contains tests for the Mesoscope-VR runtime log parser and the runtime dataset assembler that reads the feathers it
writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import (
    Cue,
    TriggerType,
    TaskTemplate,
    VREnvironment,
    TrialStructure,
    ExperimentState,
    MesoscopeWaterRewardTrial,
    MesoscopeExperimentConfiguration,
)

from sollertia_forgery.runtime import run_runtime_processing_pipeline
from sollertia_forgery.mesoscope_vr.runtime import (
    parse_runtime,
    _process_trial_sequence,
    _decompose_cue_sequence_into_trials,
    _decompose_multiple_cue_sequences_into_trials,
)
from sollertia_forgery.mesoscope_vr.metadata import BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.runtime_dataset import (
    _check_trigger_zones,
    clip_to_session_bounds,
    assemble_runtime_dataset,
    mask_non_run_experiment_data,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_SYSTEM_STATE_CODE: int = 1
"""The leading payload byte the acquisition runtime stamps on a system state message."""

_RUNTIME_STATE_CODE: int = 2
"""The leading payload byte the acquisition runtime stamps on an experiment state message."""

_REINFORCING_GUIDANCE_CODE: int = 3
"""The leading payload byte the acquisition runtime stamps on a reinforcing guidance message."""

_AVERSIVE_GUIDANCE_CODE: int = 4
"""The leading payload byte the acquisition runtime stamps on an aversive guidance message."""

_DISTANCE_SNAPSHOT_CODE: int = 5
"""The leading payload byte the acquisition runtime stamps on a traveled distance snapshot message."""

_UNROUTED_CODE: int = 9
"""A leading payload byte the parser routes to none of its collectors, which every archive may carry."""

_GRATING_CODE: int = 1
"""The wall cue code of the first cue catalogued by every template this suite builds."""

_CHECKER_CODE: int = 2
"""The wall cue code of the second cue catalogued by every template this suite builds."""

_CUE_LENGTH_CM: float = 30.0
"""The corridor length every catalogued wall cue occupies."""

_TRIAL_COUNT: int = 251
"""The number of two-cue trials one recorded cue sequence carries, chosen so the payload passes the length gate."""

_ARCHIVE_ONSET_US: int = 1_700_000_000_000_000
"""The microsecond epoch on which the synthetic runtime archive anchors its elapsed timestamps."""


def _state_payload(code: int, value: int) -> bytes:
    """Builds the two-byte state payload the acquisition runtime logs for one state transition.

    Args:
        code: The leading byte identifying which state the message carries.
        value: The state code the message reports.

    Returns:
        The payload bytes.
    """
    return bytes((code, value))


def _distance_payload(distance_cm: float) -> bytes:
    """Builds the distance snapshot payload the acquisition runtime logs when it swaps the cue sequence.

    Args:
        distance_cm: The cumulative traveled distance the snapshot reports.

    Returns:
        The payload bytes, holding the leading code followed by the little-endian double.
    """
    return bytes((_DISTANCE_SNAPSHOT_CODE,)) + np.float64(distance_cm).tobytes()


def _cue_payload(cue_codes: Sequence[int]) -> bytes:
    """Builds the wall cue sequence payload the acquisition runtime logs once per generated corridor.

    Args:
        cue_codes: The wall cue codes making up the corridor, in the order the animal encounters them.

    Returns:
        The payload bytes, one byte per cue.
    """
    return bytes(cue_codes)


def _repeating_cue_sequence(motif: Sequence[int], repeats: int) -> list[int]:
    """Repeats one trial motif into a corridor cue sequence.

    Args:
        motif: The wall cue codes of a single trial.
        repeats: The number of trials the corridor holds.

    Returns:
        The corridor's wall cue codes.
    """
    return list(motif) * repeats


def _decoded_messages(messages: Sequence[tuple[int, bytes]]) -> pl.DataFrame:
    """Builds the decoded message table the runtime pipeline hands to the parser.

    Args:
        messages: The pairs of absolute message timestamp and raw payload bytes, in archive order.

    Returns:
        The table holding the timestamps and payloads in the schema the parser reads.
    """
    return pl.DataFrame(
        {
            "time_us": pl.Series(name="time_us", values=[timestamp for timestamp, _ in messages], dtype=pl.UInt64),
            "payload": pl.Series(name="payload", values=[payload for _, payload in messages], dtype=pl.Binary),
        }
    )


def _build_task_template(
    trial_cue_sequences: dict[str, list[str]], *, cue_offset_cm: float = 0.0, cue_length_cm: float = _CUE_LENGTH_CM
) -> TaskTemplate:
    """Builds a VR task template cataloging two wall cues and the requested trial geometries.

    Every trial receives the same trigger zone, which fits inside the shortest corridor segment this suite builds.

    Args:
        trial_cue_sequences: The mapping of trial name to the wall cue names making up that trial.
        cue_offset_cm: The corridor cue offset the parser subtracts from the first cue of a trial.
        cue_length_cm: The corridor length every catalogued wall cue occupies.

    Returns:
        The task template instance.
    """
    return TaskTemplate(
        cues=[
            Cue(name="grating", code=_GRATING_CODE, length_cm=cue_length_cm, texture="Cue 001 - 4x1.png"),
            Cue(name="checker", code=_CHECKER_CODE, length_cm=cue_length_cm, texture="Cue 002 - 4x1.png"),
        ],
        vr_environment=VREnvironment(
            corridor_spacing_cm=200.0,
            segments_per_corridor=4,
            padding_prefab_name="padding",
            cm_per_unity_unit=10.0,
            cue_offset_cm=cue_offset_cm,
        ),
        trial_structures={
            name: TrialStructure(
                cue_sequence=list(cue_sequence),
                stimulus_trigger_zone_start_cm=30.0,
                stimulus_trigger_zone_end_cm=45.0,
                stimulus_location_cm=40.0,
                show_stimulus_collision_boundary=False,
                trigger_type=TriggerType.COLLISION,
            )
            for name, cue_sequence in trial_cue_sequences.items()
        },
    )


def _build_experiment_configuration(trial_names: Sequence[str]) -> MesoscopeExperimentConfiguration:
    """Builds an experiment configuration whose canonical trial order matches the requested names.

    Args:
        trial_names: The trial names, in the order to which the trial type indices refer.

    Returns:
        The experiment configuration instance.
    """
    return MesoscopeExperimentConfiguration(
        trial_structures={
            name: MesoscopeWaterRewardTrial(reward_size_ul=5.0, reward_tone_duration_ms=300) for name in trial_names
        },
        experiment_states={
            "run_state": ExperimentState(
                experiment_state_code=1, system_state_code=2, state_duration_s=600.0, supports_trials=True
            )
        },
        unity_scene_name="TestScene",
    )


def _write_configurations(
    session: SessionData,
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
) -> None:
    """Replaces the session's experiment and VR snapshots with the given configurations.

    Args:
        session: The session whose raw snapshots are overwritten.
        experiment_configuration: The experiment configuration written into the session's raw data.
        task_template: The VR task template written into the session's raw data.
    """
    experiment_configuration.to_yaml(file_path=session.raw_data.experiment_configuration_path)
    task_template.to_yaml(file_path=session.raw_data.vr_configuration_path)


def _read_feather(directory: Path, name: BehaviorDataFiles) -> pl.DataFrame:
    """Reads one behavior feather the runtime parser wrote.

    Args:
        directory: The directory into which the parser wrote its feathers.
        name: The canonical filename of the feather to read.

    Returns:
        The feather's contents.
    """
    return pl.read_ipc(source=directory.joinpath(name), memory_map=False)


def _normalized(message: str) -> str:
    """Collapses the line wrapping that the console applies to a raised error message.

    Args:
        message: The message text as the raised exception carries it.

    Returns:
        The message with every run of whitespace replaced by a single space.
    """
    return " ".join(message.split())


def _write_experiment_runtime_feathers(session: SessionData, *, guidance: bool) -> Path:
    """Parses a synthetic experiment archive into the session's processed runtime directory.

    Args:
        session: The experiment session whose configurations the parser resolves.
        guidance: Determines whether the archive carries the reinforcing and aversive guidance transitions.

    Returns:
        The path to the directory that received the runtime feathers.
    """
    messages: list[tuple[int, bytes]] = [
        (100, _state_payload(_SYSTEM_STATE_CODE, 0)),
        (300, _state_payload(_SYSTEM_STATE_CODE, 2)),
        (400, _state_payload(_RUNTIME_STATE_CODE, 1)),
        (500, _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))),
        (900, _state_payload(_RUNTIME_STATE_CODE, 0)),
    ]
    if guidance:
        messages.extend(
            [(600, _state_payload(_REINFORCING_GUIDANCE_CODE, 1)), (700, _state_payload(_AVERSIVE_GUIDANCE_CODE, 1))]
        )

    directory = session.processed_data.runtime_data_path
    parse_runtime(
        decoded_messages=_decoded_messages(messages=sorted(messages)), output_directory=directory, session=session
    )
    return directory


def _write_encoder_feather(
    session: SessionData,
    *,
    times_us: Sequence[int] = (0, 500, 1000),
    distances_cm: Sequence[float] = (0.0, 90.0, 180.0),
) -> Path:
    """Writes the encoder feather from which the runtime assembler interpolates its reference distance.

    Args:
        session: The session whose processed microcontroller directory receives the feather.
        times_us: The timestamps of the recorded wheel-distance samples.
        distances_cm: The cumulative distance the animal had traveled at each of those timestamps.

    Returns:
        The path to the directory that received the encoder feather.
    """
    directory = session.processed_data.microcontroller_data_path
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "time_us": pl.Series(name="time_us", values=list(times_us), dtype=pl.UInt64),
            "traveled_distance_cm": list(distances_cm),
        }
    ).write_ipc(file=directory.joinpath(BehaviorDataFiles.ENCODER), compression="uncompressed")
    return directory


# Runtime log parsing


def test_runtime_pipeline_writes_experiment_behavior_feathers(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that the runtime pipeline decodes a real archive into every experiment behavior feather."""
    write_log_archive(
        experiment_session.raw_data.behavior_data_path.joinpath("1_log.npz"),
        1,
        [
            (1000, _state_payload(_SYSTEM_STATE_CODE, 0)),
            (2000, _state_payload(_SYSTEM_STATE_CODE, 2)),
            (3000, _state_payload(_RUNTIME_STATE_CODE, 1)),
            (4000, _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))),
            (5000, _state_payload(_REINFORCING_GUIDANCE_CODE, 1)),
            (6000, _state_payload(_AVERSIVE_GUIDANCE_CODE, 1)),
            (7000, _state_payload(code=_UNROUTED_CODE, value=7)),
            (8000, _state_payload(_RUNTIME_STATE_CODE, 0)),
        ],
        onset_us=_ARCHIVE_ONSET_US,
    )

    run_runtime_processing_pipeline(session_path=experiment_session.raw_data_path.parent, workers=1)

    directory = experiment_session.processed_data.runtime_data_path
    system_state = _read_feather(directory, BehaviorDataFiles.SYSTEM_STATE)
    assert system_state["time_us"].to_list() == [_ARCHIVE_ONSET_US + 1000, _ARCHIVE_ONSET_US + 2000]
    assert system_state["system_state"].to_list() == [0, 2]

    runtime_state = _read_feather(directory, BehaviorDataFiles.RUNTIME_STATE)
    assert runtime_state["time_us"].to_list() == [_ARCHIVE_ONSET_US + 3000, _ARCHIVE_ONSET_US + 8000]
    assert runtime_state["runtime_state"].to_list() == [1, 0]

    reinforcing = _read_feather(directory=directory, name=BehaviorDataFiles.REINFORCING_GUIDANCE)
    assert reinforcing["time_us"].to_list() == [_ARCHIVE_ONSET_US + 5000]
    assert reinforcing["reinforcing_guidance_state"].to_list() == [1]

    aversive = _read_feather(directory=directory, name=BehaviorDataFiles.AVERSIVE_GUIDANCE)
    assert aversive["time_us"].to_list() == [_ARCHIVE_ONSET_US + 6000]
    assert aversive["aversive_guidance_state"].to_list() == [1]

    cues = _read_feather(directory=directory, name=BehaviorDataFiles.VR_CUE)
    assert cues["vr_cue"].to_list() == _repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT)
    assert cues["traveled_distance_cm"].to_numpy() == pytest.approx(np.arange(2 * _TRIAL_COUNT) * _CUE_LENGTH_CM)

    trigger_zones = _read_feather(directory=directory, name=BehaviorDataFiles.VR_TRIGGER_ZONE)
    assert trigger_zones["trigger_zone_start_cm"].to_numpy() == pytest.approx(26.0 + 60.0 * np.arange(_TRIAL_COUNT))
    assert trigger_zones["trigger_zone_end_cm"].to_numpy() == pytest.approx(49.0 + 60.0 * np.arange(_TRIAL_COUNT))

    trials = _read_feather(directory, BehaviorDataFiles.TRIAL)
    assert trials["trial_type_index"].to_list() == [0] * _TRIAL_COUNT
    assert trials["traveled_distance_cm"].to_numpy() == pytest.approx(60.0 * np.arange(_TRIAL_COUNT))


def test_parse_runtime_writes_only_state_feathers_for_training_session(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that a session without an experiment configuration receives the two state feathers alone."""
    directory = tmp_path.joinpath("training_runtime_data")

    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 0)),
                (200, _state_payload(_RUNTIME_STATE_CODE, 1)),
                (300, _distance_payload(distance_cm=120.0)),
            ]
        ),
        output_directory=directory,
        session=training_session,
    )

    assert sorted(entry.name for entry in directory.iterdir()) == [
        BehaviorDataFiles.RUNTIME_STATE.value,
        BehaviorDataFiles.SYSTEM_STATE.value,
    ]
    assert _read_feather(directory, BehaviorDataFiles.SYSTEM_STATE)["system_state"].to_list() == [0]
    assert _read_feather(directory, BehaviorDataFiles.RUNTIME_STATE)["runtime_state"].to_list() == [1]


def test_parse_runtime_discards_a_cue_sequence_a_training_session_does_not_collect(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that a wall cue sequence is consumed by its length alone, whether or not the session collects it."""
    directory = tmp_path.joinpath("training_runtime_data")
    # The first two cue codes are the system-state code and the value that code would carry.
    corridor = _cue_payload(cue_codes=[_SYSTEM_STATE_CODE, 2] * 300)

    parse_runtime(
        decoded_messages=_decoded_messages(messages=[(100, corridor), (200, _state_payload(_SYSTEM_STATE_CODE, 0))]),
        output_directory=directory,
        session=training_session,
    )

    system_states = _read_feather(directory, BehaviorDataFiles.SYSTEM_STATE)
    # A training session collects no corridor, and the payload's leading byte is a wall cue code rather than a
    # message code, so a long payload reaching the state-code chain would be recorded as a fabricated state
    # transition.
    assert system_states["system_state"].to_list() == [0]
    assert system_states["time_us"].to_list() == [200]


def test_parse_runtime_omits_guidance_feathers_when_unrecorded(experiment_session: SessionData, tmp_path: Path) -> None:
    """Verifies that the guidance feathers stay unwritten when the session recorded no guidance transition."""
    directory = tmp_path.joinpath("runtime_data")

    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 2)),
                (200, _state_payload(_RUNTIME_STATE_CODE, 1)),
                (300, _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))),
            ]
        ),
        output_directory=directory,
        session=experiment_session,
    )

    assert not directory.joinpath(BehaviorDataFiles.REINFORCING_GUIDANCE).exists()
    assert not directory.joinpath(BehaviorDataFiles.AVERSIVE_GUIDANCE).exists()
    assert directory.joinpath(BehaviorDataFiles.TRIAL).is_file()


@pytest.mark.parametrize(
    ("guidance_code", "recorded_file", "recorded_column", "unrecorded_file"),
    [
        (
            _REINFORCING_GUIDANCE_CODE,
            BehaviorDataFiles.REINFORCING_GUIDANCE,
            "reinforcing_guidance_state",
            BehaviorDataFiles.AVERSIVE_GUIDANCE,
        ),
        (
            _AVERSIVE_GUIDANCE_CODE,
            BehaviorDataFiles.AVERSIVE_GUIDANCE,
            "aversive_guidance_state",
            BehaviorDataFiles.REINFORCING_GUIDANCE,
        ),
    ],
)
def test_parse_runtime_writes_each_guidance_feather_on_its_own_recorded_transitions(
    experiment_session: SessionData,
    tmp_path: Path,
    guidance_code: int,
    recorded_file: BehaviorDataFiles,
    recorded_column: str,
    unrecorded_file: BehaviorDataFiles,
) -> None:
    """Verifies that a session guiding the animal one way alone still receives that kind's guidance feather."""
    directory = tmp_path.joinpath("runtime_data")

    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 2)),
                (200, _state_payload(_RUNTIME_STATE_CODE, 1)),
                (300, _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))),
                (400, _state_payload(code=guidance_code, value=1)),
            ]
        ),
        output_directory=directory,
        session=experiment_session,
    )

    recorded = _read_feather(directory=directory, name=recorded_file)
    # Gating one kind's feather on the other kind's transitions would silently drop every transition the session
    # did record.
    assert recorded["time_us"].to_list() == [400]
    assert recorded[recorded_column].to_list() == [1]
    assert not directory.joinpath(unrecorded_file).exists()


def test_parse_runtime_truncates_the_trial_spanning_the_recorded_corridor_swap(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that the recorded distance of a mid-session corridor swap is decoded from its snapshot exactly."""
    directory = tmp_path.joinpath("runtime_data")
    corridor = _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))

    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 2)),
                (200, corridor),
                (300, _distance_payload(distance_cm=90.1)),
                (400, corridor),
            ]
        ),
        output_directory=directory,
        session=experiment_session,
    )

    trials = _read_feather(directory, BehaviorDataFiles.TRIAL)
    # The second trial is cut short by the swap, so the third trial starts at the recorded distance itself. 90.1 is
    # compared exactly, since it is deliberately not representable in single precision. The snapshot payload's
    # leading byte is its message code, so a double read one byte early would yield a breakpoint beyond any
    # distance the animal ran, stitching the two corridors as though no swap had happened.
    assert trials["traveled_distance_cm"].to_list()[:4] == [0.0, 60.0, 90.1, 150.1]
    # The first corridor contributes the completed trial and the truncated one alone. The rest of it is abandoned.
    assert len(trials) == _TRIAL_COUNT + 2


def test_parse_runtime_raises_for_missing_experiment_configuration(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that an experiment session missing its experiment snapshot is reported as an absent file."""
    experiment_session.raw_data.experiment_configuration_path.unlink()

    with pytest.raises(FileNotFoundError, match="Unable to load experiment configuration") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages([(100, _state_payload(_SYSTEM_STATE_CODE, 2))]),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert str(experiment_session.raw_data.experiment_configuration_path) in _normalized(str(raised.value))


def test_parse_runtime_raises_for_missing_vr_configuration(experiment_session: SessionData, tmp_path: Path) -> None:
    """Verifies that an experiment session missing its VR snapshot is reported as an absent file."""
    experiment_session.raw_data.vr_configuration_path.unlink()

    with pytest.raises(FileNotFoundError, match="Unable to load the VR task template") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages([(100, _state_payload(_SYSTEM_STATE_CODE, 2))]),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert str(experiment_session.raw_data.vr_configuration_path) in _normalized(str(raised.value))


def test_parse_runtime_raises_for_trial_absent_from_task_template(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that a trial absent from the VR task template is reported against the template's trial set."""
    _write_configurations(
        session=experiment_session,
        experiment_configuration=_build_experiment_configuration(trial_names=["absent_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}),
    )

    with pytest.raises(ValueError, match="Unable to resolve trial geometry") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages(
                messages=[(100, _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT)))]
            ),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert "trial(s) ['absent_trial'] absent" in _normalized(str(raised.value))
    assert "Available template trials: ['reward_trial']" in _normalized(str(raised.value))


def test_parse_runtime_raises_without_cue_sequences(experiment_session: SessionData, tmp_path: Path) -> None:
    """Verifies that an experiment session whose archive holds no cue sequence is rejected."""
    with pytest.raises(ValueError, match="Unable to decompose input cue sequence") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages([(100, _state_payload(_SYSTEM_STATE_CODE, 2))]),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert "Expected at least one cue sequence as input, but received none." in _normalized(str(raised.value))


def test_parse_runtime_raises_for_breakpoint_count_mismatch(experiment_session: SessionData, tmp_path: Path) -> None:
    """Verifies that a second cue sequence without its distance breakpoint is rejected."""
    corridor = _cue_payload(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], _TRIAL_COUNT))

    with pytest.raises(ValueError, match="Unable to decompose input cue sequence") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages(messages=[(100, corridor), (200, corridor)]),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert "Expected the number of distance breakpoints to be 1" in _normalized(str(raised.value))
    assert "but encountered 0." in _normalized(str(raised.value))


def test_parse_runtime_raises_for_undecomposable_cue_sequence(experiment_session: SessionData, tmp_path: Path) -> None:
    """Verifies that an unmatched wall cue is reported with the position and the cues that follow it."""
    _write_configurations(
        session=experiment_session,
        experiment_configuration=_build_experiment_configuration(["reward_trial", "control_trial"]),
        task_template=_build_task_template(
            {"reward_trial": ["grating", "checker"], "control_trial": ["checker", "grating", "grating"]}
        ),
    )
    corridor = [*_repeating_cue_sequence(motif=[_CHECKER_CODE, _GRATING_CODE, _GRATING_CODE], repeats=167), 3]

    with pytest.raises(RuntimeError, match="Unable to decompose VR wall cue sequence") as raised:
        parse_runtime(
            decoded_messages=_decoded_messages(messages=[(100, _cue_payload(corridor))]),
            output_directory=tmp_path.joinpath("runtime_data"),
            session=experiment_session,
        )

    assert "cue sequence 1 of 1" in _normalized(str(raised.value))
    assert "No trial motif matched at position 501. The next 20 cues: [3]" in _normalized(str(raised.value))


def test_decompose_reports_the_failure_position_when_no_trial_is_decoded() -> None:
    """Verifies the reported failure context of a corridor shorter than the shortest trial motif."""
    with pytest.raises(RuntimeError, match="Unable to decompose VR wall cue sequence") as raised:
        _decompose_multiple_cue_sequences_into_trials(
            experiment_configuration=_build_experiment_configuration(trial_names=["long_trial"]),
            task_template=_build_task_template(trial_cue_sequences={"long_trial": ["grating", "checker", "grating"]}),
            cue_sequences=[np.array([_GRATING_CODE, _CHECKER_CODE], dtype=np.uint8)],
            distance_breakpoints=[],
        )

    assert "No trial motif matched at position 0. The next 20 cues: [1, 2]" in _normalized(str(raised.value))


def test_decompose_reports_the_failure_position_of_a_corridor_built_from_the_first_trial_type() -> None:
    """Verifies that the reported failure position counts every decoded trial of the first configured trial type."""
    corridor = [*_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 3), 3]

    with pytest.raises(RuntimeError, match="Unable to decompose VR wall cue sequence") as raised:
        _decompose_multiple_cue_sequences_into_trials(
            experiment_configuration=_build_experiment_configuration(["reward_trial"]),
            task_template=_build_task_template({"reward_trial": ["grating", "checker"]}),
            cue_sequences=[np.array(corridor, dtype=np.uint8)],
            distance_breakpoints=[],
        )

    assert "No trial motif matched at position 6. The next 20 cues: [3]" in _normalized(str(raised.value))


def test_decompose_accumulates_trial_distances_of_a_single_sequence() -> None:
    """Verifies that one corridor decomposes into its trials with cumulative end distances."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}),
        cue_sequences=[np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 4), dtype=np.uint8)],
        distance_breakpoints=[],
    )

    assert trial_types.tolist() == [0, 0, 0, 0]
    assert trial_distances.tolist() == [60.0, 120.0, 180.0, 240.0]


def test_decompose_decodes_every_trial_of_a_configuration_mixing_motif_lengths() -> None:
    """Verifies that a corridor made of short trials decomposes in full when a longer trial type is also configured."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(trial_names=["reward_trial", "long_trial"]),
        task_template=_build_task_template(
            trial_cue_sequences={
                "reward_trial": ["grating", "checker"],
                "long_trial": ["checker", "grating", "checker", "grating"],
            }
        ),
        cue_sequences=[np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 4), dtype=np.uint8)],
        distance_breakpoints=[],
    )

    # The trial cap is the corridor length over the shortest configured motif, so every short trial decodes and
    # hitting the cap returns an ordinary trial count. A cap measured against the longest motif would truncate
    # the session with no error raised anywhere.
    assert trial_types.tolist() == [0, 0, 0, 0]
    assert trial_distances.tolist() == [60.0, 120.0, 180.0, 240.0]


def test_decompose_matches_the_longer_motif_when_a_shorter_trial_is_its_prefix() -> None:
    """Verifies that a trial whose motif is a prefix of another trial's motif never consumes that longer trial."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(trial_names=["control_trial", "reward_trial"]),
        task_template=_build_task_template(
            trial_cue_sequences={
                "control_trial": ["grating", "checker", "grating"],
                "reward_trial": ["grating", "checker"],
            }
        ),
        cue_sequences=[
            np.array(
                _repeating_cue_sequence(motif=[_GRATING_CODE, _CHECKER_CODE, _GRATING_CODE], repeats=2), dtype=np.uint8
            )
        ],
        distance_breakpoints=[],
    )

    # The kernel takes the first motif that matches, so the motifs reach it longest-first. Offered the prefix
    # first, it eats two cues of every three-cue trial and the rest of the corridor stops decomposing altogether.
    assert trial_types.tolist() == [0, 0]
    assert trial_distances.tolist() == [3 * _CUE_LENGTH_CM, 6 * _CUE_LENGTH_CM]


def test_decompose_reports_trial_types_in_the_configuration_declaration_order() -> None:
    """Verifies that the decoded trial type indices count against the experiment configuration's declaration order."""
    reward_motif = [_GRATING_CODE, _CHECKER_CODE]
    control_motif = [_CHECKER_CODE, _GRATING_CODE, _GRATING_CODE]

    trial_types, _trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial", "control_trial"]),
        task_template=_build_task_template(
            {"reward_trial": ["grating", "checker"], "control_trial": ["checker", "grating", "grating"]}
        ),
        cue_sequences=[np.array([*reward_motif, *control_motif, *reward_motif], dtype=np.uint8)],
        distance_breakpoints=[],
    )

    # The trial-sequence processor resolves each trial's corridor geometry by that same order, so indices counted
    # against an alphabetically sorted trial list would walk the wrong corridor for every trial declared out of
    # order.
    assert trial_types.tolist() == [0, 1, 0]


def test_decompose_accumulates_trial_distances_at_the_precision_of_the_declared_cue_lengths() -> None:
    """Verifies that a trial end is reported as the exact double sum of the cue lengths it spans."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template(
            trial_cue_sequences={"reward_trial": ["grating", "checker"]}, cue_length_cm=30.1
        ),
        cue_sequences=[np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 2), dtype=np.uint8)],
        distance_breakpoints=[],
    )

    assert trial_types.tolist() == [0, 0]
    # Single precision cannot hold 60.2, and rounds the first trial end to 60.20000076293945 instead. These
    # distances are the coordinate against which the assembled dataset interpolates its trial columns, and the
    # matching encoder distance is a double, so a narrower accumulator drifts off the declared corridor.
    assert trial_distances.tolist() == [60.2, 120.4]


def test_decompose_truncates_the_trial_spanning_a_distance_breakpoint() -> None:
    """Verifies that the trial interrupted by a corridor swap ends at the recorded breakpoint distance."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}),
        cue_sequences=[
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 4), dtype=np.uint8),
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 2), dtype=np.uint8),
        ],
        distance_breakpoints=[np.float64(90.0)],
    )

    assert trial_types.tolist() == [0, 0, 0, 0]
    assert trial_distances.tolist() == [60.0, 90.0, 150.0, 210.0]


def test_decompose_drops_the_trial_starting_at_a_distance_breakpoint() -> None:
    """Verifies that a corridor swap landing on a trial boundary contributes no zero-length trial."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}),
        cue_sequences=[
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 4), dtype=np.uint8),
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 2), dtype=np.uint8),
        ],
        distance_breakpoints=[np.float64(120.0)],
    )

    assert trial_types.tolist() == [0, 0, 0, 0]
    assert trial_distances.tolist() == [60.0, 120.0, 180.0, 240.0]


def test_decompose_reopens_the_accumulator_behind_the_cue_offset_after_a_corridor_swap() -> None:
    """Verifies that the trials following a corridor swap stay in the traveled-distance frame under a cue offset."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}, cue_offset_cm=10.0),
        cue_sequences=[
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 4), dtype=np.uint8),
            np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 2), dtype=np.uint8),
        ],
        distance_breakpoints=[np.float64(90.0)],
    )

    assert trial_types.tolist() == [0, 0, 0, 0]
    # Trial 1 completes after 50 cm because the first corridor is entered 10 cm in, and trial 2 is cut at the 90 cm
    # swap. The second corridor is entered 10 cm in as well, so its two trials complete from 80 cm traveled onwards.
    assert trial_distances.tolist() == [50.0, 90.0, 140.0, 200.0]


def test_every_distance_stream_shares_the_traveled_frame_under_a_cue_offset() -> None:
    """Verifies the four distance streams agree with each other when the animal starts partway into its first cue."""
    cues, cue_distances, trigger_starts, trigger_ends, trial_starts = _process_trial_sequence(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}, cue_offset_cm=10.0),
        trial_types=np.array([0, 0, 0], dtype=np.int32),
        # Trial 0 is entered 10 cm in, so it completes after 50 cm traveled. Each later trial takes its full 60.
        trial_distances=np.array([50.0, 110.0, 170.0], dtype=np.float64),
    )

    # A trial begins exactly where its own first cue does, which is the invariant tying the two distance frames
    # together. The assembled dataset matches all four streams against one axis interpolated from the encoder's
    # traveled distance, so a stream carrying corridor positions would sit a fixed offset away on the same rows.
    assert trial_starts.tolist() == [0.0, 50.0, 110.0]
    assert [cue_distances[index] for index in (0, 2, 4)] == trial_starts.tolist()
    assert cues.tolist() == [_GRATING_CODE, _CHECKER_CODE] * 3

    # The trigger zone sits 30 cm into the corridor, which the first trial reaches after traveling only 20.
    assert trigger_starts.tolist() == [16.0, 76.0, 136.0]
    assert trigger_ends.tolist() == [39.0, 99.0, 159.0]


def test_decomposition_reports_trial_ends_as_distances_traveled() -> None:
    """Verifies the decomposition reports how far the animal ran rather than how long the corridor was."""
    trial_types, trial_distances = _decompose_multiple_cue_sequences_into_trials(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}, cue_offset_cm=10.0),
        cue_sequences=[np.array(_repeating_cue_sequence([_GRATING_CODE, _CHECKER_CODE], 3), dtype=np.uint8)],
        distance_breakpoints=[],
    )

    assert trial_types.tolist() == [0, 0, 0]
    # Each trial spans 60 cm of corridor, and the animal enters the first one 10 cm along. The assembled dataset
    # interpolates its trial columns against the encoder's traveled distance, so a corridor length would place
    # every boundary too late.
    assert trial_distances.tolist() == [50.0, 110.0, 170.0]


def test_process_trial_sequence_resolves_cues_and_trigger_zones_of_truncated_trials() -> None:
    """Verifies the cue, trigger zone, and trial start series a partially truncated trial sequence produces."""
    cues, distances, trigger_starts, trigger_ends, trial_starts = _process_trial_sequence(
        experiment_configuration=_build_experiment_configuration(["reward_trial"]),
        task_template=_build_task_template({"reward_trial": ["grating", "checker"]}, cue_offset_cm=10.0),
        trial_types=np.array([0, 0, 0, 0], dtype=np.int32),
        trial_distances=np.array([60.0, 80.0, 150.0, 190.0], dtype=np.float64),
    )

    assert cues.tolist() == [1, 2, 1, 1, 2, 1, 2]
    assert distances.tolist() == [0.0, 20.0, 50.0, 80.0, 100.0, 130.0, 160.0]
    # Trial 0 and trial 2 are entered 10 cm into their first cue, so each reaches its declared trigger zone position
    # after traveling 10 cm less. Trial 2 follows trial 1's truncation, which re-enters the corridor, so the same offset
    # applies to it. Trial 1 is cut short before its zone begins and contributes no entry, and trial 3 follows a
    # complete trial, so its zone sits at the full 30 cm.
    assert trigger_starts.tolist() == [16.0, 96.0, 176.0]
    assert trigger_ends.tolist() == [39.0, 119.0, 190.0]
    assert trial_starts.tolist() == [0.0, 60.0, 80.0, 150.0]


def test_process_trial_sequence_walks_each_trial_through_its_own_corridor() -> None:
    """Verifies that a mixed trial sequence resolves each trial's cues and trigger zone from that trial's geometry."""
    cues, distances, trigger_starts, trigger_ends, trial_starts = _process_trial_sequence(
        experiment_configuration=_build_experiment_configuration(["reward_trial", "control_trial"]),
        task_template=_build_task_template(
            {"reward_trial": ["grating", "checker"], "control_trial": ["checker", "grating", "grating"]}
        ),
        trial_types=np.array([0, 1], dtype=np.int32),
        trial_distances=np.array([60.0, 150.0], dtype=np.float64),
    )

    # Walking every trial with the first configured trial type's corridor would fill the cue feather with cues the
    # animal never saw.
    assert cues.tolist() == [_GRATING_CODE, _CHECKER_CODE, _CHECKER_CODE, _GRATING_CODE, _GRATING_CODE]
    assert distances.tolist() == [0.0, 30.0, 60.0, 90.0, 120.0]
    assert trial_starts.tolist() == [0.0, 60.0]
    assert trigger_starts.tolist() == [26.0, 86.0]
    assert trigger_ends.tolist() == [49.0, 109.0]


def test_decompose_cue_sequence_kernel_prefers_the_longest_matching_motif() -> None:
    """Verifies that the decomposition kernel consumes the longer motif before the shorter one."""
    trial_indices, trial_count, stop_position = _decompose_cue_sequence_into_trials.py_func(
        cue_sequence=np.array([1, 2, 3, 1, 2], dtype=np.uint8),
        motifs_flat=np.array([1, 2, 3, 1, 2], dtype=np.uint8),
        motif_starts=np.array([0, 3], dtype=np.int32),
        motif_lengths=np.array([3, 2], dtype=np.int32),
        motif_indices=np.array([0, 1], dtype=np.int32),
        maximum_trials=5,
    )

    assert trial_count == 2
    assert trial_indices.tolist() == [0, 1]
    assert stop_position == 5


def test_decompose_cue_sequence_kernel_reports_an_unmatched_position() -> None:
    """Verifies that the decomposition kernel reports failure at the position of the unmatched cue."""
    trial_indices, trial_count, failure_position = _decompose_cue_sequence_into_trials.py_func(
        cue_sequence=np.array([1, 2, 3, 1, 2, 9], dtype=np.uint8),
        motifs_flat=np.array([1, 2, 3, 1, 2], dtype=np.uint8),
        motif_starts=np.array([0, 3], dtype=np.int32),
        motif_lengths=np.array([3, 2], dtype=np.int32),
        motif_indices=np.array([0, 1], dtype=np.int32),
        maximum_trials=5,
    )

    assert trial_count == -1
    assert trial_indices.tolist() == [0, 1]
    assert failure_position == 5


# Runtime dataset assembly


def test_assemble_runtime_dataset_aligns_every_column_to_the_reference_time(
    experiment_session: SessionData, experiment_configuration: MesoscopeExperimentConfiguration
) -> None:
    """Verifies the trial, cue, trigger zone, state, and guidance columns the assembler aligns."""
    runtime_directory = _write_experiment_runtime_feathers(session=experiment_session, guidance=True)
    microcontroller_directory = _write_encoder_feather(experiment_session)

    dataset = assemble_runtime_dataset(
        microcontroller_data_path=microcontroller_directory,
        runtime_data_path=runtime_directory,
        experiment_configuration=experiment_configuration,
        reference_time=np.array([0, 250, 500, 750, 1000], dtype=np.uint64),
    )

    assert dataset.schema["trial"] == pl.UInt16
    assert dataset["trial"].to_list() == [1, 1, 2, 3, 4]
    assert dataset["trial_type"].to_list() == ["reward_trial"] * 5
    assert dataset["cue"].to_list() == [1, 2, 2, 1, 1]
    assert dataset["in_trigger_zone"].to_list() == [0, 1, 1, 0, 0]
    assert dataset["runtime_state"].to_list() == ["run_state", "run_state", "run_state", "run_state", "idle"]
    assert dataset["reinforcing_guided"].to_list() == [1] * 5
    assert dataset["aversive_guided"].to_list() == [1] * 5


def test_assemble_runtime_dataset_omits_the_guidance_columns_when_their_feathers_are_absent(
    experiment_session: SessionData, experiment_configuration: MesoscopeExperimentConfiguration
) -> None:
    """Verifies that a session without recorded guidance yields a dataset carrying the required columns alone."""
    runtime_directory = _write_experiment_runtime_feathers(session=experiment_session, guidance=False)
    microcontroller_directory = _write_encoder_feather(experiment_session)

    dataset = assemble_runtime_dataset(
        microcontroller_data_path=microcontroller_directory,
        runtime_data_path=runtime_directory,
        experiment_configuration=experiment_configuration,
        reference_time=np.array([0, 250, 500, 750, 1000], dtype=np.uint64),
    )

    assert dataset.columns == ["trial", "trial_type", "cue", "in_trigger_zone", "runtime_state"]
    assert dataset["trial"].to_list() == [1, 1, 2, 3, 4]


def test_assemble_runtime_dataset_labels_a_mixed_session_against_the_reference_distance(
    experiment_session: SessionData,
) -> None:
    """Verifies that the trial type of every sample is looked up in the distance frame that keys the trial feather."""
    # A session running a single trial type cannot tell the two indexes apart, so this one runs two.
    configuration = _build_experiment_configuration(["reward_trial", "control_trial"])
    _write_configurations(
        session=experiment_session,
        experiment_configuration=configuration,
        task_template=_build_task_template(
            {"reward_trial": ["grating", "checker"], "control_trial": ["checker", "grating", "grating"]}
        ),
    )
    corridor = _repeating_cue_sequence(
        motif=[_GRATING_CODE, _CHECKER_CODE, _CHECKER_CODE, _GRATING_CODE, _GRATING_CODE], repeats=101
    )
    runtime_directory = experiment_session.processed_data.runtime_data_path
    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 2)),
                (200, _state_payload(_RUNTIME_STATE_CODE, 1)),
                (300, _cue_payload(corridor)),
            ]
        ),
        output_directory=runtime_directory,
        session=experiment_session,
    )
    # The animal covers 10 cm over the reference window, so every sample sits inside the session's first trial, which
    # is the reward trial. The timestamps themselves run to 1000, which is many trials along that same distance axis.
    microcontroller_directory = _write_encoder_feather(
        session=experiment_session, times_us=(0, 500, 1000), distances_cm=(0.0, 5.0, 10.0)
    )

    dataset = assemble_runtime_dataset(
        microcontroller_data_path=microcontroller_directory,
        runtime_data_path=runtime_directory,
        experiment_configuration=configuration,
        reference_time=np.array([0, 250, 500, 750, 1000], dtype=np.uint64),
    )

    # The trial feather indexes its trial types by traveled distance while the state feathers are indexed by time.
    # Matching them against the reference time instead labels every sample with whichever trial sits at that many
    # centimeters.
    assert dataset["trial"].to_list() == [1] * 5
    assert dataset["trial_type"].to_list() == ["reward_trial"] * 5
    assert dataset.schema["trial_type"] == pl.Enum(["reward_trial", "control_trial", "undefined"])


def test_mask_non_run_experiment_data_masks_idle_and_rest_samples() -> None:
    """Verifies that the cue, trial, and trial type columns are masked outside the run state."""
    experiment_data = pl.DataFrame(
        {
            "system_state": pl.Series(
                name="system_state",
                values=["idle", "rest", "run", "run"],
                dtype=pl.Enum(["idle", "rest", "run"]),
            ),
            "cue": pl.Series(name="cue", values=[1, 2, 1, 2], dtype=pl.UInt8),
            "trial": pl.Series(name="trial", values=[1, 1, 2, 2], dtype=pl.UInt16),
            "trial_type": pl.Series(
                name="trial_type",
                values=["reward_trial"] * 4,
                dtype=pl.Enum(["reward_trial", "undefined"]),
            ),
        }
    )

    masked = mask_non_run_experiment_data(experiment_data=experiment_data)

    assert masked["cue"].to_list() == [255, 255, 1, 2]
    assert masked["trial"].to_list() == [65535, 65535, 2, 2]
    assert masked["trial_type"].to_list() == ["undefined", "undefined", "reward_trial", "reward_trial"]
    assert masked.schema["cue"] == pl.UInt8
    assert masked.schema["trial"] == pl.UInt16


def test_clip_to_session_bounds_trims_the_setup_and_teardown_samples(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that the samples before the first non-idle system state and after the last runtime state are dropped."""
    directory = tmp_path.joinpath("runtime_data")
    parse_runtime(
        decoded_messages=_decoded_messages(
            messages=[
                (100, _state_payload(_SYSTEM_STATE_CODE, 0)),
                (300, _state_payload(_SYSTEM_STATE_CODE, 2)),
                (400, _state_payload(_RUNTIME_STATE_CODE, 1)),
                (900, _state_payload(_RUNTIME_STATE_CODE, 0)),
            ]
        ),
        output_directory=directory,
        session=training_session,
    )
    assembled = pl.DataFrame(
        {
            "time_us": pl.Series(name="time_us", values=[0, 100, 200, 300, 400, 900, 1000], dtype=pl.UInt64),
            "lick": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    clipped = clip_to_session_bounds(assembled_data=assembled, runtime_data_path=directory)

    assert clipped["time_us"].to_list() == [300, 400, 900]
    assert clipped["lick"].to_list() == [1, 0, 1]


def test_clip_to_session_bounds_retains_every_sample_without_state_transitions(
    training_session: SessionData, tmp_path: Path
) -> None:
    """Verifies that a session that never left idle and recorded no runtime state keeps its full sample range."""
    directory = tmp_path.joinpath("runtime_data")
    parse_runtime(
        decoded_messages=_decoded_messages(messages=[(100, _state_payload(_SYSTEM_STATE_CODE, 0))]),
        output_directory=directory,
        session=training_session,
    )
    assembled = pl.DataFrame(
        {
            "time_us": pl.Series(name="time_us", values=[0, 100, 200], dtype=pl.UInt64),
            "lick": [0, 1, 0],
        }
    )

    clipped = clip_to_session_bounds(assembled_data=assembled, runtime_data_path=directory)

    assert clipped["time_us"].to_list() == [0, 100, 200]
    assert clipped["lick"].to_list() == [0, 1, 0]


def test_check_trigger_zones_returns_zeros_without_recorded_zones() -> None:
    """Verifies that every sample falls outside a trigger zone when the session recorded no zone."""
    in_zone: NDArray[np.uint8] = _check_trigger_zones.py_func(
        traversed_distance=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        trigger_zone_starts=np.array([], dtype=np.float64),
        trigger_zone_ends=np.array([], dtype=np.float64),
    )

    assert in_zone.tolist() == [0, 0, 0]
    assert in_zone.dtype == np.uint8


def test_check_trigger_zones_marks_the_samples_inside_a_zone() -> None:
    """Verifies zone membership across, between, and beyond the zones, including a backward distance step."""
    in_zone: NDArray[np.uint8] = _check_trigger_zones.py_func(
        traversed_distance=np.array([5.0, 15.0, 25.0, 35.0, 45.0, 15.0], dtype=np.float64),
        trigger_zone_starts=np.array([10.0, 30.0], dtype=np.float64),
        trigger_zone_ends=np.array([20.0, 40.0], dtype=np.float64),
    )

    assert in_zone.tolist() == [0, 1, 0, 1, 0, 1]
