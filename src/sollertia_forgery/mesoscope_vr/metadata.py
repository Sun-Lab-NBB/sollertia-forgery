"""Provides the Mesoscope-VR-specific metadata schema: the BehaviorDataFiles and DatasetColumn enumerations, the
StimulusMode enumeration, and the per-session TrialGeometry data file consumed by both the forging and analysis
pipelines.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    MesoscopeGasPuffTrial as GasPuffTrial,
    MesoscopeWaterRewardTrial as WaterRewardTrial,
)
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from sollertia_shared_assets import MesoscopeExperimentConfiguration


class BehaviorDataFiles(StrEnum):
    """Enumerates the canonical filenames of the behavior feather files written into the session's
    ``processed_data/behavior_data`` directory by the Mesoscope-VR processing pipeline and consumed by the
    Mesoscope-VR dataset forging pipeline.

    Notes:
        This is the contract between the Mesoscope-VR processing and forging assets: the processing pipeline writes
        each file under these exact names, and the forging pipeline reads them back by the same names. All entries
        are forgery-internal and must not be referenced from outside the library.
    """

    ENCODER = "encoder_data.feather"
    """The encoder module feather holding the traveled-distance time series derived from the running-wheel encoder."""
    VALVE = "valve_data.feather"
    """The water valve module feather holding water-dispensing events and cumulative dispensed volume."""
    GAS_PUFF = "gas_puff_data.feather"
    """The gas puff valve module feather holding aversive-stimulus dispensing events."""
    LICK = "lick_data.feather"
    """The lick module feather holding thresholded lick events from the capacitive sensor."""
    BRAKE = "brake_data.feather"
    """The brake module feather holding the instantaneous brake torque applied to the running wheel."""
    TORQUE = "torque_data.feather"
    """The torque module feather holding the instantaneous torque exerted on the running wheel by the animal."""
    SCREEN = "screen_data.feather"
    """The screen module feather holding VR screen on/off state transitions."""
    MESOSCOPE_FRAME = "mesoscope_frame_data.feather"
    """The TTL module feather holding mesoscope scan-frame pulse edges used to align fluorescence to behavior."""
    SYSTEM_STATE = "system_state_data.feather"
    """The runtime feather holding system-state code transitions extracted from the acquisition runtime's log
    archive."""
    RUNTIME_STATE = "runtime_state_data.feather"
    """The runtime feather holding experiment-state code transitions extracted from the acquisition runtime's log
    archive."""
    REINFORCING_GUIDANCE = "reinforcing_guidance_state_data.feather"
    """The runtime feather holding reinforcing guidance-state transitions (written only when reinforcing guidance
    events were recorded during the session)."""
    AVERSIVE_GUIDANCE = "aversive_guidance_state_data.feather"
    """The runtime feather holding aversive guidance-state transitions (written only when aversive guidance events
    were recorded during the session)."""
    VR_CUE = "vr_cue_data.feather"
    """The runtime feather holding VR wall-cue transitions along the corridor."""
    VR_TRIGGER_ZONE = "vr_trigger_zone_data.feather"
    """The runtime feather holding VR trigger-zone entry and exit events."""
    TRIAL = "trial_data.feather"
    """The runtime feather holding per-trial metadata (trial index, trial type, traveled distance at trial start)."""


class DatasetColumn(StrEnum):
    """Defines every column that can appear in the assembled session data feather produced by the forging pipeline.

    Notes:
        Members covering optional columns (`REINFORCING_GUIDED`, `AVERSIVE_GUIDED`) are present in the feather only
        when the corresponding upstream events were recorded. All other members are guaranteed to exist in every
        forged session.
    """

    # Behavior alignment columns (from forging behavior assembly).
    TIME_US = "time_us"
    """Microsecond-precision sample timestamps from the acquisition reference clock."""
    ELAPSED_MINUTES = "elapsed_minutes"
    """Elapsed session time in minutes since the first sample."""
    BRAKE = "brake"
    """Wheel brake engagement at each sample."""
    SCREENS = "screens"
    """Display panel state at each sample."""
    TORQUE_N_CM = "torque_N_cm"
    """Wheel torque in N·cm at each sample. Forced to zero during 'run' periods upstream."""
    DISTANCE_CM = "distance_cm"
    """Cumulative distance traveled by the animal in centimeters at each sample."""
    SPEED_CM_S = "speed_cm_s"
    """Animal running speed in cm/s at each sample."""
    LICK = "lick"
    """Lick sensor state at each sample."""
    WATER_UL = "water_uL"
    """Per-sample water reward delivery in microliters."""
    REWARD = "reward"
    """Reward event flag at each sample."""
    SYSTEM_STATE = "system_state"
    """Acquisition system state at each sample (idle, rest, run)."""

    # Runtime/experiment columns (from forging runtime assembly).
    TRIAL = "trial"
    """One-based trial identifier at each sample. 255 marks samples outside any trial."""
    TRIAL_TYPE = "trial_type"
    """Trial type label at each sample (e.g. 'ABC', 'ABCD'). 'undefined' marks non-run samples."""
    CUE = "cue"
    """Active virtual reality cue identifier at each sample."""
    IN_TRIGGER_ZONE = "in_trigger_zone"
    """Boolean flag indicating whether the animal is inside a stimulus trigger zone at each sample."""
    RUNTIME_STATE = "runtime_state"
    """Experiment runtime state label at each sample."""
    REINFORCING_GUIDED = "reinforcing_guided"
    """Optional. Reinforcing guidance state at each sample. Present only when reinforcing guidance was recorded."""
    AVERSIVE_GUIDED = "aversive_guided"
    """Optional. Aversive guidance state at each sample. Present only when aversive guidance was recorded."""

    # Cindra fluorescence columns (from forging fluorescence assembly).
    SINGLE_DAY_CELL_FLUORESCENCE = "single_day_cell_fluorescence"
    """Single-recording raw cell fluorescence trace per ROI."""
    SINGLE_DAY_NEUROPIL_FLUORESCENCE = "single_day_neuropil_fluorescence"
    """Single-recording raw neuropil fluorescence trace per ROI."""
    SINGLE_DAY_SUBTRACTED_FLUORESCENCE = "single_day_subtracted_fluorescence"
    """Single-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence."""
    SINGLE_DAY_SPIKES = "single_day_spikes"
    """Single-recording OASIS-deconvolved spike rates per ROI."""
    MULTI_DAY_CELL_FLUORESCENCE = "multi_day_cell_fluorescence"
    """Multi-recording raw cell fluorescence trace per ROI."""
    MULTI_DAY_NEUROPIL_FLUORESCENCE = "multi_day_neuropil_fluorescence"
    """Multi-recording raw neuropil fluorescence trace per ROI."""
    MULTI_DAY_SUBTRACTED_FLUORESCENCE = "multi_day_subtracted_fluorescence"
    """Multi-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence aligned across recording days."""
    MULTI_DAY_SPIKES = "multi_day_spikes"
    """Multi-recording OASIS-deconvolved spike rates per ROI aligned across recording days."""


class StimulusMode(StrEnum):
    """Defines the semantic meaning of the stimulus delivered when a trial's stimulus trigger zone fires.

    Notes:
        Projects each upstream trial-class type onto the analysis-relevant axis of "what does the animal experience
        when the trigger fires." Decoupled from TriggerType, which describes the activation
        mechanism rather than the resulting outcome. Analysis modules use this enum to gate semantically appropriate
        pipelines (e.g., reward-cell analysis runs only against trial types whose stimulus_mode is REWARD).
    """

    REWARD = "reward"
    """Indicates an appetitive stimulus (e.g., water delivery in a WaterRewardTrial) delivered when the trigger
    condition is met."""
    AVERSIVE = "aversive"
    """Indicates an aversive stimulus (e.g., gas puff in a GasPuffTrial) delivered when the trigger condition fails."""


# noinspection PyUnhashable
_TRIAL_CLASS_TO_STIMULUS_MODE: dict[type[WaterRewardTrial | GasPuffTrial], StimulusMode] = {
    WaterRewardTrial: StimulusMode.REWARD,
    GasPuffTrial: StimulusMode.AVERSIVE,
}
"""Maps each upstream trial class to the stimulus mode it delivers. Update this mapping when a new trial subclass is
added to MesoscopeExperimentConfiguration; missing entries are surfaced as ValueError at forging time."""


@dataclass(frozen=True, slots=True)
class TrialGeometryEntry:
    """Defines the canonical geometry for a single trial type used in a forged session."""

    stimulus_mode: StimulusMode
    """The semantic meaning of the stimulus delivered when the trigger zone fires (reward or aversive)."""
    trial_length_cm: float
    """The canonical track length for this trial type, in centimeters."""
    stimulus_trigger_zone_start_cm: float
    """The trial-relative start of the stimulus trigger zone, in centimeters."""
    stimulus_trigger_zone_end_cm: float
    """The trial-relative end of the stimulus trigger zone, in centimeters."""
    stimulus_location_cm: float
    """The trial-relative location of the stimulus boundary, in centimeters."""
    cue_offset_cm: float = 0.0
    """The offset between the runtime's trial start and the canonical start of the first cue in the cue sequence,
    in centimeters. When non-zero, the runtime begins recording mid-cue, so analysis-side trial boundaries must be
    re-aligned to the first-cue transition before cue zones (and the trigger zone) read at canonical positions."""


@dataclass
class TrialGeometry(YamlConfig):
    """Maps each trial type name to its canonical geometry, written as a data file alongside data.feather.

    Notes:
        Projects the analysis-relevant slice of MesoscopeExperimentConfiguration.trial_structures so that downstream
        analysis can reconstruct canonical per-trial position without re-reading the upstream experiment configuration.
        This decouples the analysis dataset schema from the upstream configuration schema, limiting migration impact
        when MesoscopeExperimentConfiguration evolves.
    """

    entries: dict[str, TrialGeometryEntry]
    """The mapping from trial type name (the key used in MesoscopeExperimentConfiguration.trial_structures and the
    'trial_type' column in data.feather) to that trial type's canonical geometry."""

    @classmethod
    def from_experiment_configuration(cls, experiment_configuration: MesoscopeExperimentConfiguration) -> TrialGeometry:
        """Projects the canonical trial geometry out of the provided experiment configuration.

        Args:
            experiment_configuration: The MesoscopeExperimentConfiguration loaded from the session's raw data.

        Returns:
            A TrialGeometry instance with one entry per trial type defined in the experiment configuration.

        Raises:
            ValueError: If any trial structure has a class that is not registered in _TRIAL_CLASS_TO_STIMULUS_MODE.
        """
        entries: dict[str, TrialGeometryEntry] = {}
        for trial_type_name, trial in experiment_configuration.trial_structures.items():
            stimulus_mode = _TRIAL_CLASS_TO_STIMULUS_MODE.get(type(trial))
            if stimulus_mode is None:
                message = (
                    f"Unable to project trial '{trial_type_name}' into the trial geometry data file. The trial class "
                    f"'{type(trial).__name__}' has no entry in _TRIAL_CLASS_TO_STIMULUS_MODE. Add a mapping for any "
                    f"new trial subclass added to MesoscopeExperimentConfiguration."
                )
                console.error(message=message, error=ValueError)
            entries[trial_type_name] = TrialGeometryEntry(
                stimulus_mode=stimulus_mode,
                trial_length_cm=trial.trial_length_cm,
                stimulus_trigger_zone_start_cm=trial.stimulus_trigger_zone_start_cm,
                stimulus_trigger_zone_end_cm=trial.stimulus_trigger_zone_end_cm,
                stimulus_location_cm=trial.stimulus_location_cm,
                cue_offset_cm=experiment_configuration.cue_offset_cm,
            )
        return cls(entries=entries)
