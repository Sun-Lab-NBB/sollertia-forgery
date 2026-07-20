"""Provides the Mesoscope-VR-specific metadata schema: the BehaviorDataFiles and DatasetColumn enumerations and the
derived MESOSCOPE_COLUMN_DESCRIPTIONS mapping donated to the system-agnostic forging pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class BehaviorDataFiles(StrEnum):
    """Enumerates the canonical filenames of the behavior feather files written by the donated Mesoscope-VR parsers and
    read back by the donated assembly worker. The microcontroller parsers write the module feathers into the session's
    ``processed_data/microcontroller_data`` directory, while the runtime parser writes the runtime feathers into
    ``processed_data/runtime_data``.

    Notes:
        These names are the file-naming contract shared between the donated parsers (writers) and the assembly worker
        (reader). All entries are forgery-internal and must not be referenced from outside the library.
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
    """Defines every column that can appear in the assembled session data feather produced by the forging pipeline,
    pairing each column name with its human-readable description.

    Notes:
        Each member's value is the column name as it appears in ``data.feather`` (so members compare equal to the raw
        column strings), while the ``description`` attribute carries the column's meaning. This enum is the single
        source of truth for the Mesoscope-VR column descriptions baked into a forged dataset via
        ``MESOSCOPE_COLUMN_DESCRIPTIONS``.

        Several columns are conditional: ``REINFORCING_GUIDED`` and ``AVERSIVE_GUIDED`` are present only when the
        corresponding guidance events were recorded. ``BRAKE`` and ``SCREENS`` are present only for mesoscope
        experiments. ``TORQUE_N_CM`` is absent for run training. ``DISTANCE_CM`` and ``SPEED_CM_S`` are absent for lick
        training. The remaining members are present in every forged session.
    """

    description: str
    """The human-readable description of the column, recorded in the dataset's ``data_descriptions.feather``."""

    def __new__(cls, value: str, description: str) -> Self:
        """Builds a DatasetColumn member whose string value is the column name and that carries its description.

        Args:
            value: The column name as it appears in ``data.feather``.
            description: The human-readable description of the column.

        Returns:
            The constructed DatasetColumn member.
        """
        member = str.__new__(cls, value)
        member._value_ = value
        member.description = description
        return member

    # Behavior alignment columns (from forging behavior assembly).
    TIME_US = ("time_us", "Microsecond-precision sample timestamps from the acquisition reference clock.")
    ELAPSED_MINUTES = ("elapsed_minutes", "Elapsed session time in minutes since the session's onset.")
    BRAKE = ("brake", "The running wheel brake engagement at each sample.")
    SCREENS = ("screens", "The Virtual Reality displays state at each sample.")
    TORQUE_N_CM = (
        "torque_N_cm",
        "The torque applied by the animal to the running wheel in N·cm at each sample, forced to zero during "
        "'run' periods upstream.",
    )
    DISTANCE_CM = (
        "distance_cm",
        "Cumulative distance traveled by the animal in centimeters at each sample.",
    )
    SPEED_CM_S = ("speed_cm_s", "Animal's running speed in cm/s at each sample.")
    LICK = ("lick", "Lick sensor engagement state at each sample.")
    WATER_UL = ("water_uL", "The cumulative water reward volume delivered to the animal at each sample in microliters.")
    REWARD = ("reward", "The reward delivery state (on / off) at each sample.")
    SYSTEM_STATE = ("system_state", "Acquisition system state at each sample (idle, rest, run).")

    # Runtime/experiment columns (from forging runtime assembly).
    TRIAL = ("trial", "One-based trial identifier at each sample. 65535 marks samples outside any trial.")
    TRIAL_TYPE = (
        "trial_type",
        "Trial type label at each sample (e.g. 'ABC', 'ABCD'). 'undefined' marks non-run samples.",
    )
    CUE = ("cue", "Active virtual reality cue identifier at each sample. 255 marks samples outside the run state.")
    IN_TRIGGER_ZONE = (
        "in_trigger_zone",
        "Boolean flag indicating whether the animal is inside a stimulus trigger zone at each sample.",
    )
    RUNTIME_STATE = ("runtime_state", "Experiment runtime state label at each sample.")
    REINFORCING_GUIDED = (
        "reinforcing_guided",
        "Reinforcing guidance state at each sample. Present only when reinforcing guidance was recorded.",
    )
    AVERSIVE_GUIDED = (
        "aversive_guided",
        "Aversive guidance state at each sample. Present only when aversive guidance was recorded.",
    )

    # Cindra fluorescence columns (from forging fluorescence assembly).
    SINGLE_DAY_CELL_FLUORESCENCE = (
        "single_day_cell_fluorescence",
        "Single-recording raw cell fluorescence trace per ROI.",
    )
    SINGLE_DAY_NEUROPIL_FLUORESCENCE = (
        "single_day_neuropil_fluorescence",
        "Single-recording raw neuropil fluorescence trace per ROI.",
    )
    SINGLE_DAY_SUBTRACTED_FLUORESCENCE = (
        "single_day_subtracted_fluorescence",
        "Single-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence.",
    )
    SINGLE_DAY_SPIKES = ("single_day_spikes", "Single-recording OASIS-deconvolved spike rates per ROI.")
    MULTI_DAY_CELL_FLUORESCENCE = (
        "multi_day_cell_fluorescence",
        "Multi-recording raw cell fluorescence trace per ROI.",
    )
    MULTI_DAY_NEUROPIL_FLUORESCENCE = (
        "multi_day_neuropil_fluorescence",
        "Multi-recording raw neuropil fluorescence trace per ROI.",
    )
    MULTI_DAY_SUBTRACTED_FLUORESCENCE = (
        "multi_day_subtracted_fluorescence",
        "Multi-recording neuropil-subtracted, baseline-corrected dF/F0 fluorescence aligned across recording days.",
    )
    MULTI_DAY_SPIKES = (
        "multi_day_spikes",
        "Multi-recording OASIS-deconvolved spike rates per ROI aligned across recording days.",
    )


MESOSCOPE_COLUMN_DESCRIPTIONS: dict[str, str] = {column.value: column.description for column in DatasetColumn}
"""The Mesoscope-VR column-description binding donated to the forging pipeline. Maps every column name the
Mesoscope-VR assembly worker can emit into ``data.feather`` to its human-readable description, baked into each forged
dataset's ``data_descriptions.feather``. Derived from ``DatasetColumn`` so the descriptions stay single-sourced."""
