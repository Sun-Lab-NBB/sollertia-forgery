"""Enumerates the canonical filenames used across sollertia-forgery's internal processing and forging pipelines."""

from __future__ import annotations

from enum import StrEnum


class BehaviorDataFiles(StrEnum):
    """Enumerates the canonical filenames of the behavior feather files written into the session's
    ``processed_data/behavior_data`` directory by the sollertia-forgery processing pipeline and consumed by the
    sollertia-forgery dataset forging pipeline.

    Notes:
        This is the contract between the ``processing`` and ``forging`` subpackages: the processing pipeline writes
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
    """The runtime feather holding system-state code transitions extracted from the acquisition runtime's log archive."""
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
    FACE_CAMERA_TIMESTAMPS = "face_camera_timestamps.feather"
    """The face-camera timestamp feather hardlinked from the ataraxis-video-system processed camera_timestamps output
    under its legacy sollertia-forgery name."""
    BODY_CAMERA_TIMESTAMPS = "body_camera_timestamps.feather"
    """The body-camera timestamp feather hardlinked from the ataraxis-video-system processed camera_timestamps output
    under its legacy sollertia-forgery name."""
