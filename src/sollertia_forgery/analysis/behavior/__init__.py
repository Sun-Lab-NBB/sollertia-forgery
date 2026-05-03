"""Provides behavior analysis helpers built on top of the per-session ``data.feather`` and trial geometry.

The package intentionally has no ``run_*_analysis`` orchestrator and persists no feather/yaml artifacts:
behavior data already lives in the forged ``data.feather`` and reward zones already live in the per-session
trial geometry data file, so per-session compute is cheap enough to repeat on demand. Consumers call
`aggregate_lick_events` for the per-animal `LickContext` consumed by `plot_lick_scatter`, or
`aggregate_trial_outcomes` for the per-animal `TrialOutcomeContext` consumed by `plot_trial_outcomes`.
"""

from .plotting import plot_lick_scatter, plot_trial_outcomes
from .lick_protocol import (
    CueSpan,
    TrialBlock,
    LickContext,
    SessionLickData,
    aggregate_lick_events,
    extract_session_lick_events,
)
from .outcome_protocol import (
    OUTCOME_GUIDED,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    TrialOutcome,
    TrialOutcomeContext,
    SessionTrialOutcomes,
    aggregate_trial_outcomes,
    extract_session_trial_outcomes,
)

__all__ = [
    "OUTCOME_FAILURE",
    "OUTCOME_GUIDED",
    "OUTCOME_SUCCESS",
    "CueSpan",
    "LickContext",
    "SessionLickData",
    "SessionTrialOutcomes",
    "TrialBlock",
    "TrialOutcome",
    "TrialOutcomeContext",
    "aggregate_lick_events",
    "aggregate_trial_outcomes",
    "extract_session_lick_events",
    "extract_session_trial_outcomes",
    "plot_lick_scatter",
    "plot_trial_outcomes",
]
