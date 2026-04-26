"""Provides the TrialGeometry dataclass for recording trail geometry in forged datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import console
from sollertia_shared_assets import BaseTrial, GasPuffTrial, WaterRewardTrial
from ataraxis_data_structures import YamlConfig

from ..shared_assets import StimulusMode

if TYPE_CHECKING:
    from sollertia_shared_assets import MesoscopeExperimentConfiguration


TRIAL_GEOMETRY_FILENAME: str = "trial_geometry.yaml"
"""The filename of the trial geometry sidecar inside each session directory of a forged dataset."""

# noinspection PyUnhashable
_TRIAL_CLASS_TO_STIMULUS_MODE: dict[type[BaseTrial], StimulusMode] = {
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


@dataclass
class TrialGeometry(YamlConfig):
    """Maps each trial type name to its canonical geometry, written as a sidecar to data.feather.

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
                    f"Unable to project trial '{trial_type_name}' into the trial geometry sidecar. The trial class "
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
            )
        return cls(entries=entries)
