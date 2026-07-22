"""Provides the Mesoscope-VR-specific metadata schema: the BehaviorDataFiles and VideoDataFiles filename
enumerations, the DatasetColumn enumeration, and the derived MESOSCOPE_COLUMN_DESCRIPTIONS mapping donated to the
system-agnostic forging pipeline.
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
    """The runtime feather holding per-trial metadata (trial type index and traveled distance at trial start)."""


class VideoDataFiles(StrEnum):
    """Enumerates the canonical filenames of the per-camera video feathers written by the video-processing pipeline and
    read back by the donated video-dataset assembler. The pipeline writes them into the session's
    ``processed_data/video_data`` directory under the acquisition-time camera names.

    Notes:
        These names are the file-naming contract for the fixed Mesoscope-VR camera set (the face and body cameras),
        mirroring how the microcontroller module feathers are a fixed set. All entries are forgery-internal and must
        not be referenced from outside the library.
    """

    FACE_CAMERA_TIMESTAMPS = "face_camera_timestamps.feather"
    """The face-camera timestamp feather holding one frame-acquisition timestamp per recorded frame."""
    FACE_CAMERA_ENERGY = "face_camera_energy.feather"
    """The face-camera motion-energy feather holding the per-frame motion energy and frame luminance."""
    FACE_CAMERA_PUPIL = "face_camera_pupil.feather"
    """The face-camera pupil-tracking feather holding the per-frame pupil and eye metrics."""
    BODY_CAMERA_TIMESTAMPS = "body_camera_timestamps.feather"
    """The body-camera timestamp feather holding one frame-acquisition timestamp per recorded frame."""
    BODY_CAMERA_ENERGY = "body_camera_energy.feather"
    """The body-camera motion-energy feather holding the per-frame motion energy and frame luminance."""


class DatasetColumn(StrEnum):
    """Defines every column that can appear in the assembled session data feather produced by the forging pipeline,
    pairing each column name with its human-readable description.

    Notes:
        Each member's value is the column name as it appears in ``data.feather`` (so members compare equal to the raw
        column strings), while the ``description`` attribute carries the column's meaning. This enum is the single
        source of truth for the Mesoscope-VR column descriptions baked into a forged dataset via
        ``MESOSCOPE_COLUMN_DESCRIPTIONS``.

        Several columns are conditional. The runtime columns (``TRIAL``, ``TRIAL_TYPE``, ``CUE``, ``IN_TRIGGER_ZONE``,
        ``RUNTIME_STATE``), the fluorescence columns (``FRAME`` and the ``SINGLE_DAY_*`` and ``MULTI_DAY_*`` members),
        ``BRAKE``, and ``SCREENS`` are present only for mesoscope experiment sessions. ``REINFORCING_GUIDED`` and
        ``AVERSIVE_GUIDED`` are present only when the corresponding guidance events were recorded. ``TORQUE_N_CM`` is
        absent for run training. ``DISTANCE_CM`` and ``SPEED_CM_S`` are absent for lick training. The per-camera video
        columns are present only when that camera's feathers were produced, and the pupil columns only when the face
        camera's pose predictions were processed. The remaining behavior columns (``TIME_US``, ``ELAPSED_MINUTES``,
        ``LICK``, ``WATER_UL``, ``REWARD``, ``SYSTEM_STATE``) are present in every forged session.
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
    REWARD = (
        "reward",
        "The reward-classification state at each sample, one of 'no' (no tone or reward), 'tone' (tone played, no "
        "water delivered), or 'yes' (water reward delivered).",
    )
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
    FRAME = ("frame", "One-based mesoscope acquisition frame index at each sample.")
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
        "Single-recording neuropil- and baseline-subtracted fluorescence trace per ROI.",
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
        "Multi-recording neuropil- and baseline-subtracted fluorescence trace per ROI aligned across recording days.",
    )
    MULTI_DAY_SPIKES = (
        "multi_day_spikes",
        "Multi-recording OASIS-deconvolved spike rates per ROI aligned across recording days.",
    )

    # Video motion-energy columns (from forging video assembly).
    FACE_CAMERA_MOTION_ENERGY = (
        "face_camera_motion_energy",
        "Face camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A "
        "within-session movement magnitude, high during movement and low during stillness.",
    )
    FACE_CAMERA_FRAME_LUMINANCE = (
        "face_camera_frame_luminance",
        "Face camera mean frame intensity at each sample in gray levels, tracking scene illumination.",
    )
    BODY_CAMERA_MOTION_ENERGY = (
        "body_camera_motion_energy",
        "Body camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A "
        "within-session movement magnitude, high during movement and low during stillness.",
    )
    BODY_CAMERA_FRAME_LUMINANCE = (
        "body_camera_frame_luminance",
        "Body camera mean frame intensity at each sample in gray levels, tracking scene illumination.",
    )

    # Pupil-tracking columns (from forging video assembly, face camera).
    PUPIL_CENTER_X_PX = (
        "pupil_center_x_px",
        "Horizontal position of the fitted pupil ellipse center in face-camera pixels at each sample.",
    )
    PUPIL_CENTER_Y_PX = (
        "pupil_center_y_px",
        "Vertical position of the fitted pupil ellipse center in face-camera pixels at each sample.",
    )
    PUPIL_DIAMETER_PX = (
        "pupil_diameter_px",
        "Mean of the fitted pupil ellipse's two axis diameters in face-camera pixels at each sample, the primary "
        "arousal proxy. NaN marks blink and dilation samples.",
    )
    PUPIL_AREA_PX2 = (
        "pupil_area_px2",
        "Area enclosed by the fitted pupil ellipse in square face-camera pixels at each sample.",
    )
    PUPIL_FIT_CONDITION = (
        "pupil_fit_condition",
        "Condition number of the fitted pupil ellipse at each sample, where higher values indicate a less reliable "
        "fit. NaN marks unmeasured samples.",
    )
    PUPIL_FIT_RESIDUAL_PX = (
        "pupil_fit_residual_px",
        "Root-mean-square distance in pixels between the confident pupil-perimeter points and the fitted ellipse at "
        "each sample, a fit-quality measure.",
    )
    EYE_CENTER_X_PX = (
        "eye_center_x_px",
        "Horizontal position of the fitted eye ellipse center in face-camera pixels at each sample.",
    )
    EYE_CENTER_Y_PX = (
        "eye_center_y_px",
        "Vertical position of the fitted eye ellipse center in face-camera pixels at each sample.",
    )
    EYE_WIDTH_PX = (
        "eye_width_px",
        "Length of the fitted eye ellipse's left-right chord in face-camera pixels at each sample.",
    )
    EYE_HEIGHT_PX = (
        "eye_height_px",
        "Length of the fitted eye ellipse's top-bottom chord in face-camera pixels at each sample.",
    )
    EYE_OPENNESS = (
        "eye_openness",
        "Ratio of the fitted eye ellipse's height to its width at each sample, a distance-invariant measure of eye "
        "openness.",
    )
    BLINKING_STATE = (
        "blinking_state",
        "Blink state at each sample, marking a closed or covered eye. Encoded as 1 during a blink and 0 otherwise.",
    )
    DILATION_STATE = (
        "dilation_state",
        "Dilation-clip state at each sample, marking a pupil dilated past the eye aperture. Encoded as 1 when clipped "
        "and 0 otherwise.",
    )
    REFLECTION_X_PX = (
        "reflection_x_px",
        "Horizontal position of the corneal reflection in face-camera pixels at each sample.",
    )
    REFLECTION_Y_PX = (
        "reflection_y_px",
        "Vertical position of the corneal reflection in face-camera pixels at each sample.",
    )
    PUPIL_REFLECTION_OFFSET_X_PX = (
        "pupil_reflection_offset_x_px",
        "Horizontal offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust "
        "horizontal eye-position signal.",
    )
    PUPIL_REFLECTION_OFFSET_Y_PX = (
        "pupil_reflection_offset_y_px",
        "Vertical offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust "
        "vertical eye-position signal.",
    )
    PUPIL_IN_EYE_X = (
        "pupil_in_eye_x",
        "Horizontal offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's "
        "horizontal semi-axis and dimensionless.",
    )
    PUPIL_IN_EYE_Y = (
        "pupil_in_eye_y",
        "Vertical offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's "
        "vertical semi-axis and dimensionless.",
    )


MESOSCOPE_COLUMN_DESCRIPTIONS: dict[str, str] = {column.value: column.description for column in DatasetColumn}
"""The Mesoscope-VR column-description binding donated to the forging pipeline. Maps every column name the
Mesoscope-VR assembly worker can emit into ``data.feather`` to its human-readable description, baked into each forged
dataset's ``data_descriptions.feather``. Derived from ``DatasetColumn`` so the descriptions stay single-sourced."""
