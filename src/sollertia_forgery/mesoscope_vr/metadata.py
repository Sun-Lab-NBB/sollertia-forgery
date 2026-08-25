"""Provides the Mesoscope-VR-specific metadata schema: the BehaviorDataFiles and VideoDataFiles filename
enumerations, the DatasetColumn enumeration, and the derived MESOSCOPE_COLUMN_DESCRIPTIONS mapping donated to the
system-agnostic forging pipeline.
"""

from __future__ import annotations

from enum import StrEnum


class BehaviorDataFiles(StrEnum):
    """Enumerates the canonical filenames of the behavior feather files written by the donated Mesoscope-VR parsers,
    most of which the donated assembly worker reads back. The microcontroller parsers write the module feathers into
    the session's ``processed_data/microcontroller_data`` directory, while the runtime parser writes them into
    ``processed_data/runtime_data``.
    """

    ENCODER = "encoder_data.feather"
    """The encoder module feather holding the traveled-distance time series derived from the running-wheel encoder."""
    VALVE = "valve_data.feather"
    """The water valve module feather holding the cumulative dispensed water volume and the reward tone state."""
    GAS_PUFF = "gas_puff_data.feather"
    """The gas puff valve module feather holding aversive-stimulus dispensing events."""
    LICK = "lick_data.feather"
    """The lick module feather holding the raw 12-bit ADC sensor voltage and the thresholded lick state derived from
    it."""
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
    """Defines every column name that can appear in the assembled session data feather produced by the forging
    pipeline.

    Notes:
        Each member's value is the column name as it appears in ``data.feather``, so members compare equal to the raw
        column strings.

        Several columns are conditional. The runtime columns (``TRIAL``, ``TRIAL_TYPE``, ``CUE``, ``IN_TRIGGER_ZONE``,
        ``RUNTIME_STATE``), the fluorescence columns (``FRAME`` and the ``SINGLE_DAY_*`` and ``MULTI_DAY_*`` members),
        ``BRAKE``, and ``SCREENS`` are present only for mesoscope experiment sessions. ``REINFORCING_GUIDED`` and
        ``AVERSIVE_GUIDED`` are present only when the corresponding guidance events were recorded. ``TORQUE_N_CM`` is
        absent for run training. ``DISTANCE_CM`` and ``SPEED_CM_S`` are absent for lick training. The per-camera video
        columns are present only when that camera's feathers were produced, and the pupil columns only when the face
        camera's pose predictions were processed. The remaining behavior columns (``TIME_US``, ``ELAPSED_MINUTES``,
        ``LICK``, ``WATER_UL``, ``REWARD``, ``SYSTEM_STATE``) are present in every forged session.
    """

    # Groups the behavior-alignment columns. The two time columns come from the fluorescence assembly for experiment
    # sessions and from the behavior assembly for training sessions.
    TIME_US = "time_us"
    """Microsecond-precision sample timestamps from the acquisition reference clock."""
    ELAPSED_MINUTES = "elapsed_minutes"
    """Elapsed time in minutes since the first sample of the session's reference clock. The clock starts during setup,
    so the clipped dataset's first row carries a value above zero.
    """
    BRAKE = "brake"
    """The running wheel brake engagement at each sample."""
    SCREENS = "screens"
    """The Virtual Reality display state at each sample."""
    TORQUE_N_CM = "torque_N_cm"
    """The torque applied by the animal to the running wheel in N·cm at each sample, forced to zero during 'run' periods
    upstream.
    """
    DISTANCE_CM = "distance_cm"
    """Cumulative distance traveled by the animal in centimeters at each sample."""
    SPEED_CM_S = "speed_cm_s"
    """Animal's running speed in cm/s at each sample."""
    LICK = "lick"
    """Lick sensor engagement state at each sample."""
    WATER_UL = "water_uL"
    """The cumulative water reward volume delivered to the animal at each sample in microliters."""
    REWARD = "reward"
    """The reward-classification state at each sample, one of 'no' (no reward tone playing), 'tone' (reward tone playing
    with no water delivered during the tone), or 'yes' (reward tone playing with water delivered during the tone).
    """
    SYSTEM_STATE = "system_state"
    """Acquisition system state at each sample (idle, rest, run for experiment sessions, lick training or run training
    for training sessions).
    """

    # Groups the runtime and experiment columns produced by the forging runtime assembly.
    TRIAL = "trial"
    """One-based trial identifier at each sample. 65535 marks samples outside the run state."""
    TRIAL_TYPE = "trial_type"
    """Trial type label at each sample (e.g. 'ABC', 'ABCD'). 'undefined' marks non-run samples."""
    CUE = "cue"
    """Active virtual reality cue identifier at each sample. 255 marks samples outside the run state."""
    IN_TRIGGER_ZONE = "in_trigger_zone"
    """Boolean flag indicating whether the animal is inside a stimulus trigger zone at each sample."""
    RUNTIME_STATE = "runtime_state"
    """Experiment runtime state label at each sample."""
    REINFORCING_GUIDED = "reinforcing_guided"
    """Reinforcing guidance state at each sample. Present only when reinforcing guidance was recorded."""
    AVERSIVE_GUIDED = "aversive_guided"
    """Aversive guidance state at each sample. Present only when aversive guidance was recorded."""

    # Groups the cindra fluorescence columns produced by the forging fluorescence assembly.
    FRAME = "frame"
    """One-based mesoscope acquisition frame index at each sample."""
    SINGLE_DAY_CELL_FLUORESCENCE = "single_day_cell_fluorescence"
    """Single-recording raw cell fluorescence trace per ROI, over the cells detected in this session alone."""
    SINGLE_DAY_NEUROPIL_FLUORESCENCE = "single_day_neuropil_fluorescence"
    """Single-recording raw neuropil fluorescence trace per ROI, over the cells detected in this session alone."""
    SINGLE_DAY_SUBTRACTED_FLUORESCENCE = "single_day_subtracted_fluorescence"
    """Single-recording neuropil- and baseline-subtracted fluorescence trace per ROI, over the cells detected in this
    session alone.
    """
    SINGLE_DAY_SPIKES = "single_day_spikes"
    """Single-recording OASIS-deconvolved spike rates per ROI, over the cells detected in this session alone."""
    MULTI_DAY_CELL_FLUORESCENCE = "multi_day_cell_fluorescence"
    """Multi-recording raw cell fluorescence trace per ROI, over the cells tracked across every session of this animal
    in the dataset.
    """
    MULTI_DAY_NEUROPIL_FLUORESCENCE = "multi_day_neuropil_fluorescence"
    """Multi-recording raw neuropil fluorescence trace per ROI, over the cells tracked across every session of this
    animal in the dataset.
    """
    MULTI_DAY_SUBTRACTED_FLUORESCENCE = "multi_day_subtracted_fluorescence"
    """Multi-recording neuropil- and baseline-subtracted fluorescence trace per ROI, over the cells tracked across every
    session of this animal in the dataset.
    """
    MULTI_DAY_SPIKES = "multi_day_spikes"
    """Multi-recording OASIS-deconvolved spike rates per ROI, over the cells tracked across every session of this animal
    in the dataset.
    """

    # Groups the video motion-energy columns produced by the forging video assembly.
    FACE_CAMERA_MOTION_ENERGY = "face_camera_motion_energy"
    """Face camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A
    within-session movement magnitude, high during movement and low during stillness.
    """
    FACE_CAMERA_FRAME_LUMINANCE = "face_camera_frame_luminance"
    """Face camera mean frame intensity at each sample in gray levels, tracking scene illumination."""
    BODY_CAMERA_MOTION_ENERGY = "body_camera_motion_energy"
    """Body camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A
    within-session movement magnitude, high during movement and low during stillness.
    """
    BODY_CAMERA_FRAME_LUMINANCE = "body_camera_frame_luminance"
    """Body camera mean frame intensity at each sample in gray levels, tracking scene illumination."""

    # Groups the pupil-tracking columns produced by the forging video assembly for the face camera.
    PUPIL_CENTER_X_PX = "pupil_center_x_px"
    """Horizontal position of the fitted pupil ellipse center in face-camera pixels at each sample."""
    PUPIL_CENTER_Y_PX = "pupil_center_y_px"
    """Vertical position of the fitted pupil ellipse center in face-camera pixels at each sample."""
    PUPIL_DIAMETER_PX = "pupil_diameter_px"
    """Mean of the fitted pupil ellipse's two axis diameters in face-camera pixels at each sample, the primary arousal
    proxy. NaN marks blink and dilation samples.
    """
    PUPIL_AREA_PX2 = "pupil_area_px2"
    """Area enclosed by the fitted pupil ellipse in square face-camera pixels at each sample."""
    PUPIL_FIT_CONDITION = "pupil_fit_condition"
    """Condition number of the fitted pupil ellipse at each sample, where higher values indicate a less reliable fit.
    NaN marks unmeasured samples.
    """
    PUPIL_FIT_RESIDUAL_PX = "pupil_fit_residual_px"
    """Root-mean-square distance in pixels between the confident pupil-perimeter points and the fitted ellipse at each
    sample, a fit-quality measure.
    """
    EYE_CENTER_X_PX = "eye_center_x_px"
    """Horizontal position of the fitted eye ellipse center in face-camera pixels at each sample."""
    EYE_CENTER_Y_PX = "eye_center_y_px"
    """Vertical position of the fitted eye ellipse center in face-camera pixels at each sample."""
    EYE_WIDTH_PX = "eye_width_px"
    """Length of the fitted eye ellipse's left-right chord in face-camera pixels at each sample."""
    EYE_HEIGHT_PX = "eye_height_px"
    """Length of the fitted eye ellipse's top-bottom chord in face-camera pixels at each sample."""
    EYE_OPENNESS = "eye_openness"
    """Ratio of the fitted eye ellipse's height to its width at each sample, a distance-invariant measure of eye
    openness.
    """
    BLINKING_STATE = "blinking_state"
    """Blink state at each sample, marking a closed or covered eye. Encoded as 1 during a blink and 0 otherwise."""
    DILATION_STATE = "dilation_state"
    """Dilation-clip state at each sample, marking a pupil dilated past the eye aperture. Encoded as 1 when clipped and
    0 otherwise.
    """
    REFLECTION_X_PX = "reflection_x_px"
    """Horizontal position of the corneal reflection in face-camera pixels at each sample."""
    REFLECTION_Y_PX = "reflection_y_px"
    """Vertical position of the corneal reflection in face-camera pixels at each sample."""
    PUPIL_REFLECTION_OFFSET_X_PX = "pupil_reflection_offset_x_px"
    """Horizontal offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust
    horizontal eye-position signal.
    """
    PUPIL_REFLECTION_OFFSET_Y_PX = "pupil_reflection_offset_y_px"
    """Vertical offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust
    vertical eye-position signal.
    """
    PUPIL_IN_EYE_X = "pupil_in_eye_x"
    """Horizontal offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's
    horizontal semi-axis and dimensionless.
    """
    PUPIL_IN_EYE_Y = "pupil_in_eye_y"
    """Vertical offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's vertical
    semi-axis and dimensionless.
    """


_COLUMN_DESCRIPTIONS: dict[DatasetColumn, str] = {
    # Behavior-alignment column descriptions.
    DatasetColumn.TIME_US: "Microsecond-precision sample timestamps from the acquisition reference clock.",
    DatasetColumn.ELAPSED_MINUTES: (
        "Elapsed time in minutes since the first sample of the session's reference clock. The clock starts during "
        "setup, so the clipped dataset's first row carries a value above zero."
    ),
    DatasetColumn.BRAKE: "The running wheel brake engagement at each sample.",
    DatasetColumn.SCREENS: "The Virtual Reality display state at each sample.",
    DatasetColumn.TORQUE_N_CM: (
        "The torque applied by the animal to the running wheel in N·cm at each sample, forced to zero during 'run' "
        "periods upstream."
    ),
    DatasetColumn.DISTANCE_CM: "Cumulative distance traveled by the animal in centimeters at each sample.",
    DatasetColumn.SPEED_CM_S: "Animal's running speed in cm/s at each sample.",
    DatasetColumn.LICK: "Lick sensor engagement state at each sample.",
    DatasetColumn.WATER_UL: "The cumulative water reward volume delivered to the animal at each sample in microliters.",
    DatasetColumn.REWARD: (
        "The reward-classification state at each sample, one of 'no' (no reward tone playing), 'tone' (reward tone "
        "playing with no water delivered during the tone), or 'yes' (reward tone playing with water delivered during "
        "the tone)."
    ),
    DatasetColumn.SYSTEM_STATE: (
        "Acquisition system state at each sample (idle, rest, run for experiment sessions, lick training or run "
        "training for training sessions)."
    ),
    # Runtime and experiment column descriptions.
    DatasetColumn.TRIAL: "One-based trial identifier at each sample. 65535 marks samples outside the run state.",
    DatasetColumn.TRIAL_TYPE: (
        "Trial type label at each sample (e.g. 'ABC', 'ABCD'). 'undefined' marks non-run samples."
    ),
    DatasetColumn.CUE: "Active virtual reality cue identifier at each sample. 255 marks samples outside the run state.",
    DatasetColumn.IN_TRIGGER_ZONE: (
        "Boolean flag indicating whether the animal is inside a stimulus trigger zone at each sample."
    ),
    DatasetColumn.RUNTIME_STATE: "Experiment runtime state label at each sample.",
    DatasetColumn.REINFORCING_GUIDED: (
        "Reinforcing guidance state at each sample. Present only when reinforcing guidance was recorded."
    ),
    DatasetColumn.AVERSIVE_GUIDED: (
        "Aversive guidance state at each sample. Present only when aversive guidance was recorded."
    ),
    # cindra fluorescence column descriptions.
    DatasetColumn.FRAME: "One-based mesoscope acquisition frame index at each sample.",
    DatasetColumn.SINGLE_DAY_CELL_FLUORESCENCE: (
        "Single-recording raw cell fluorescence trace per ROI, over the cells detected in this session alone."
    ),
    DatasetColumn.SINGLE_DAY_NEUROPIL_FLUORESCENCE: (
        "Single-recording raw neuropil fluorescence trace per ROI, over the cells detected in this session alone."
    ),
    DatasetColumn.SINGLE_DAY_SUBTRACTED_FLUORESCENCE: (
        "Single-recording neuropil- and baseline-subtracted fluorescence trace per ROI, over the cells detected in "
        "this session alone."
    ),
    DatasetColumn.SINGLE_DAY_SPIKES: (
        "Single-recording OASIS-deconvolved spike rates per ROI, over the cells detected in this session alone."
    ),
    DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE: (
        "Multi-recording raw cell fluorescence trace per ROI, over the cells tracked across every session of this "
        "animal in the dataset."
    ),
    DatasetColumn.MULTI_DAY_NEUROPIL_FLUORESCENCE: (
        "Multi-recording raw neuropil fluorescence trace per ROI, over the cells tracked across every session of "
        "this animal in the dataset."
    ),
    DatasetColumn.MULTI_DAY_SUBTRACTED_FLUORESCENCE: (
        "Multi-recording neuropil- and baseline-subtracted fluorescence trace per ROI, over the cells tracked "
        "across every session of this animal in the dataset."
    ),
    DatasetColumn.MULTI_DAY_SPIKES: (
        "Multi-recording OASIS-deconvolved spike rates per ROI, over the cells tracked across every session of this "
        "animal in the dataset."
    ),
    # Video motion-energy column descriptions.
    DatasetColumn.FACE_CAMERA_MOTION_ENERGY: (
        "Face camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A "
        "within-session movement magnitude, high during movement and low during stillness."
    ),
    DatasetColumn.FACE_CAMERA_FRAME_LUMINANCE: (
        "Face camera mean frame intensity at each sample in gray levels, tracking scene illumination."
    ),
    DatasetColumn.BODY_CAMERA_MOTION_ENERGY: (
        "Body camera motion energy at each sample, the mean absolute inter-frame intensity change in gray levels. A "
        "within-session movement magnitude, high during movement and low during stillness."
    ),
    DatasetColumn.BODY_CAMERA_FRAME_LUMINANCE: (
        "Body camera mean frame intensity at each sample in gray levels, tracking scene illumination."
    ),
    # Pupil-tracking column descriptions (face camera).
    DatasetColumn.PUPIL_CENTER_X_PX: (
        "Horizontal position of the fitted pupil ellipse center in face-camera pixels at each sample."
    ),
    DatasetColumn.PUPIL_CENTER_Y_PX: (
        "Vertical position of the fitted pupil ellipse center in face-camera pixels at each sample."
    ),
    DatasetColumn.PUPIL_DIAMETER_PX: (
        "Mean of the fitted pupil ellipse's two axis diameters in face-camera pixels at each sample, the primary "
        "arousal proxy. NaN marks blink and dilation samples."
    ),
    DatasetColumn.PUPIL_AREA_PX2: (
        "Area enclosed by the fitted pupil ellipse in square face-camera pixels at each sample."
    ),
    DatasetColumn.PUPIL_FIT_CONDITION: (
        "Condition number of the fitted pupil ellipse at each sample, where higher values indicate a less reliable "
        "fit. NaN marks unmeasured samples."
    ),
    DatasetColumn.PUPIL_FIT_RESIDUAL_PX: (
        "Root-mean-square distance in pixels between the confident pupil-perimeter points and the fitted ellipse at "
        "each sample, a fit-quality measure."
    ),
    DatasetColumn.EYE_CENTER_X_PX: (
        "Horizontal position of the fitted eye ellipse center in face-camera pixels at each sample."
    ),
    DatasetColumn.EYE_CENTER_Y_PX: (
        "Vertical position of the fitted eye ellipse center in face-camera pixels at each sample."
    ),
    DatasetColumn.EYE_WIDTH_PX: (
        "Length of the fitted eye ellipse's left-right chord in face-camera pixels at each sample."
    ),
    DatasetColumn.EYE_HEIGHT_PX: (
        "Length of the fitted eye ellipse's top-bottom chord in face-camera pixels at each sample."
    ),
    DatasetColumn.EYE_OPENNESS: (
        "Ratio of the fitted eye ellipse's height to its width at each sample, a distance-invariant measure of eye "
        "openness."
    ),
    DatasetColumn.BLINKING_STATE: (
        "Blink state at each sample, marking a closed or covered eye. Encoded as 1 during a blink and 0 otherwise."
    ),
    DatasetColumn.DILATION_STATE: (
        "Dilation-clip state at each sample, marking a pupil dilated past the eye aperture. Encoded as 1 when clipped "
        "and 0 otherwise."
    ),
    DatasetColumn.REFLECTION_X_PX: (
        "Horizontal position of the corneal reflection in face-camera pixels at each sample."
    ),
    DatasetColumn.REFLECTION_Y_PX: "Vertical position of the corneal reflection in face-camera pixels at each sample.",
    DatasetColumn.PUPIL_REFLECTION_OFFSET_X_PX: (
        "Horizontal offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust "
        "horizontal eye-position signal."
    ),
    DatasetColumn.PUPIL_REFLECTION_OFFSET_Y_PX: (
        "Vertical offset of the pupil center from the corneal reflection in pixels at each sample, a motion-robust "
        "vertical eye-position signal."
    ),
    DatasetColumn.PUPIL_IN_EYE_X: (
        "Horizontal offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's "
        "horizontal semi-axis and dimensionless."
    ),
    DatasetColumn.PUPIL_IN_EYE_Y: (
        "Vertical offset of the pupil center from the eye center at each sample, normalized to the eye ellipse's "
        "vertical semi-axis and dimensionless."
    ),
}
"""Maps each DatasetColumn to its human-readable description. Every member must appear here, which the
``MESOSCOPE_COLUMN_DESCRIPTIONS`` comprehension enforces at import time by raising ``KeyError`` on any omission."""


MESOSCOPE_COLUMN_DESCRIPTIONS: dict[str, str] = {column.value: _COLUMN_DESCRIPTIONS[column] for column in DatasetColumn}
"""The Mesoscope-VR column-description binding donated to the forging pipeline. Maps every column name the
Mesoscope-VR assembly worker can emit into ``data.feather`` to its human-readable description, baked into each forged
dataset's ``data_descriptions.feather``. Derived from ``DatasetColumn`` and ``_COLUMN_DESCRIPTIONS``."""
