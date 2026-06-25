"""Provides the Mesoscope-VR-specific metadata schema: the BehaviorDataFiles and DatasetColumn enumerations and the
per-session SessionDataFormat schema descriptor written by the forging pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from dataclasses import field, dataclass

from ataraxis_data_structures import YamlConfig

DATA_FORMAT_FILE: str = "data_format.yaml"
"""The canonical filename of the per-session data-format descriptor written alongside ``data.feather`` by the
Mesoscope-VR forging assembly worker. The descriptor is system-specific: it is named and produced by the donating
acquisition-system package, not by the agnostic forging pipeline."""


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
    """One-based trial identifier at each sample. 65535 marks samples outside any trial."""
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


@dataclass
class SessionDataFormat(YamlConfig):
    """Describes the schema of a forged session's ``data.feather``, written alongside it as ``data_format.yaml``.

    Notes:
        This is the per-session, system-specific data-format descriptor the Mesoscope-VR forging assembly worker
        donates to the dataset (the agnostic forging pipeline never names or interprets it). It records the ordered
        column-name-to-dtype mapping of the assembled feather so downstream consumers can introspect the columns that
        are present in a given session (which varies, for example, when optional guidance columns were not recorded)
        without reading the feather itself. The ``DatasetColumn`` enumeration documents the meaning of each column.
    """

    columns: dict[str, str] = field(default_factory=dict)
    """The ordered mapping from each column name in ``data.feather`` to its Polars dtype string."""
