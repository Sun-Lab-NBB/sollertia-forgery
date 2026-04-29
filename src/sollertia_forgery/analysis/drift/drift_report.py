"""Per-animal cross-session tuning-drift report container.

Wraps the `DriftDetectionConfiguration` outputs as a triplet of persisted artifacts: a per-cell summary
feather, a per-session-pair pairwise feather, and a YAML summary. The report owns persistence (``save`` /
``load``), the human-readable summary, and the per-animal orchestrator `run_drift_analysis` that walks
every animal in a `DatasetData` instance. Methodological references for the drift pipeline live on
`compute_drift_report`.

The drift report consumes the persisted `..tuning.TuningReport` artifacts of every session for an
animal plus, when available, the animal's persisted `..bleaching.BleachingReport`. Both are produced
by the existing analysis sub-pipelines, so the drift evaluation does not re-run any per-session detection.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import YamlConfig

from .utilities import (
    AnimalSessionData,
    load_animal_session_data,
    fit_per_cell_baseline_slope,
)
from .drift_protocol import (
    CellDriftMetrics,
    PairwiseDriftMetrics,
    DriftDetectionConfiguration,
    fit_population_decay,
    compute_pairwise_drift_metrics,
    compute_per_cell_drift_metrics,
)
from ...shared_assets import delay_terminal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...shared_assets import DatasetData, DatasetAnimal, DatasetSession


_MINIMUM_SESSIONS_FOR_DRIFT: int = 2
"""Minimum number of sessions per animal required to evaluate any drift metric. The lowest meaningful unit is
a single session pair, so two sessions is the protocol floor."""


class DriftCellColumn(StrEnum):
    """Defines every column written to the per-animal ``drift_cells.feather`` per-cell summary table."""

    CELL_ID = "cell_id"
    """Multi-day-registered cell identifier; stable across the animal's sessions."""
    PLACE_TRAJECTORY = "place_trajectory"
    """Per-session ``IS_PLACE`` boolean trajectory as a list of length ``session_count``. Holds the unmasked
    flags so the persisted column is reversible against bleaching gating choices."""
    REWARD_TRAJECTORY = "reward_trajectory"
    """Per-session ``IS_REWARD_CELL`` boolean trajectory as a list of length ``session_count``."""
    STRICT_PLACE_TRAJECTORY = "strict_place_trajectory"
    """Per-session ``IS_STRICT_PLACE`` boolean trajectory as a list of length ``session_count``."""
    PLACE_SESSION_COUNT = "place_session_count"
    """Number of bleaching-clean sessions where the cell was flagged ``IS_PLACE``."""
    REWARD_SESSION_COUNT = "reward_session_count"
    """Number of bleaching-clean sessions where the cell was flagged ``IS_REWARD_CELL``."""
    STRICT_PLACE_SESSION_COUNT = "strict_place_session_count"
    """Number of bleaching-clean sessions where the cell was flagged ``IS_STRICT_PLACE``."""
    PLACE_SESSION_FRACTION = "place_session_fraction"
    """Fraction of bleaching-clean sessions where the cell was flagged ``IS_PLACE``."""
    REWARD_SESSION_FRACTION = "reward_session_fraction"
    """Fraction of bleaching-clean sessions where the cell was flagged ``IS_REWARD_CELL``."""
    STRICT_PLACE_SESSION_FRACTION = "strict_place_session_fraction"
    """Fraction of bleaching-clean sessions where the cell was flagged ``IS_STRICT_PLACE``."""
    PLACE_RECURRENCE_PROBABILITY = "place_recurrence_probability"
    """Consecutive-pair P(IS_PLACE_{t+1} | IS_PLACE_t) over bleaching-clean sessions; NaN when the
    conditioning event is empty."""
    REWARD_RECURRENCE_PROBABILITY = "reward_recurrence_probability"
    """Consecutive-pair P(IS_REWARD_CELL_{t+1} | IS_REWARD_CELL_t) over bleaching-clean sessions."""
    STRICT_PLACE_RECURRENCE_PROBABILITY = "strict_place_recurrence_probability"
    """Consecutive-pair P(IS_STRICT_PLACE_{t+1} | IS_STRICT_PLACE_t) over bleaching-clean sessions."""
    IS_PERSISTENT_PLACE = "is_persistent_place"
    """True for cells whose ``PLACE_SESSION_FRACTION`` exceeds the configured persistence threshold."""
    IS_PERSISTENT_REWARD = "is_persistent_reward"
    """True for cells whose ``REWARD_SESSION_FRACTION`` exceeds the configured persistence threshold."""
    IS_PERSISTENT_STRICT_PLACE = "is_persistent_strict_place"
    """True for cells whose ``STRICT_PLACE_SESSION_FRACTION`` exceeds the configured persistence threshold."""
    PAIR_COUNT = "pair_count"
    """Number of session pairs that contributed at least one finite per-cell rate-map correlation entry."""
    MEAN_RATE_MAP_CORRELATION = "mean_rate_map_correlation"
    """Per-cell mean Pearson r of rate maps across surviving session pairs. NaN when no pair contributes."""
    MEAN_CONSECUTIVE_RATE_MAP_CORRELATION = "mean_consecutive_rate_map_correlation"
    """Per-cell mean Pearson r restricted to consecutive (chronologically adjacent) session pairs."""
    MEAN_PEAK_SHIFT_CM = "mean_peak_shift_cm"
    """Per-cell mean absolute peak shift in centimeters across surviving session pairs."""
    MEAN_COM_SHIFT_CM = "mean_com_shift_cm"
    """Per-cell mean absolute COM shift in centimeters across surviving session pairs."""
    FISHER_PEAK_SHIFT_P_VALUE = "fisher_peak_shift_p_value"
    """Per-cell Fisher-combined random-remapping peak-shift p-value across surviving session pairs."""
    IS_FIELD_STABLE = "is_field_stable"
    """True for cells whose ``MEAN_RATE_MAP_CORRELATION`` exceeds the configured threshold."""
    IS_PEAK_STABLE = "is_peak_stable"
    """True for cells whose ``FISHER_PEAK_SHIFT_P_VALUE`` falls below the configured significance
    threshold."""
    CELL_BASELINE_SLOPE = "cell_baseline_slope"
    """Per-cell linear baseline-fluorescence slope in fluorescence units per day; NaN when bleaching
    cross-correlation is unavailable for the animal."""
    IS_HIGH_BLEACHING_DRIFT = "is_high_bleaching_drift"
    """True for cells whose absolute baseline slope exceeds the configured high-drift quartile of the per-cell
    distribution. Always False when bleaching cross-correlation is unavailable."""
    IS_STABLY_TUNED_PLACE = "is_stably_tuned_place"
    """Composite flag — True for cells that are persistently place AND field-stable AND peak-stable AND not
    high-bleaching-drift candidates."""
    IS_STABLY_TUNED_REWARD = "is_stably_tuned_reward"
    """Composite flag — True for cells that are persistently reward AND field-stable AND not
    high-bleaching-drift candidates."""
    IS_STABLY_TUNED_STRICT_PLACE = "is_stably_tuned_strict_place"
    """Composite flag — True for cells that are persistently strict-place AND field-stable AND peak-stable AND
    not high-bleaching-drift candidates."""


class DriftPairColumn(StrEnum):
    """Defines every column written to the per-animal ``drift_pairs.feather`` per-pair table."""

    SESSION_A = "session_a"
    """Session-directory timestamp of the earlier member of the pair."""
    SESSION_B = "session_b"
    """Session-directory timestamp of the later member of the pair."""
    SESSION_A_INDEX = "session_a_index"
    """Chronological index of ``SESSION_A`` in the animal's drift session set."""
    SESSION_B_INDEX = "session_b_index"
    """Chronological index of ``SESSION_B`` in the animal's drift session set."""
    LAG_DAYS = "lag_days"
    """Calendar-day lag computed as ``days[B] - days[A]``."""
    POPULATION_VECTOR_CORRELATION = "population_vector_correlation"
    """Pearson r between the two sessions' (cell, bin) rate-map matrices flattened to vectors."""
    PLACE_RECURRENCE_COUNT = "place_recurrence_count"
    """Count of cells classified ``IS_PLACE`` in both sessions; reflects the unmasked classification."""
    REWARD_RECURRENCE_COUNT = "reward_recurrence_count"
    """Count of cells classified ``IS_REWARD_CELL`` in both sessions."""
    STRICT_PLACE_RECURRENCE_COUNT = "strict_place_recurrence_count"
    """Count of cells classified ``IS_STRICT_PLACE`` in both sessions."""
    BOTH_BLEACHING_CLEAN = "both_bleaching_clean"
    """True when both sessions were bleaching-clean during the run that produced this report. False entries
    indicate that the per-cell shift columns for this pair are NaN."""
    RATE_MAP_CORRELATION_PER_CELL = "rate_map_correlation_per_cell"
    """Per-cell Pearson r between the two sessions' rate maps; NaN where either map has zero variance or the
    pair was masked out by bleaching gating. List of length ``cell_count``."""
    PEAK_SHIFT_CM_PER_CELL = "peak_shift_cm_per_cell"
    """Per-cell |peak_b - peak_a| in centimeters; NaN where either peak is undefined or the pair was
    bleaching-masked. List of length ``cell_count``."""
    COM_SHIFT_CM_PER_CELL = "com_shift_cm_per_cell"
    """Per-cell |COM_b - COM_a| in centimeters; NaN where either COM is undefined or the pair was
    bleaching-masked. List of length ``cell_count``."""
    PEAK_SHIFT_P_VALUE_PER_CELL = "peak_shift_p_value_per_cell"
    """Per-cell random-remapping peak-shift p-value; NaN where the peak shift is undefined."""


@dataclass
class DriftSummary(YamlConfig):
    """Per-animal YAML companion to ``drift_cells.feather`` and ``drift_pairs.feather``.

    Carries the drift configuration (load-bearing because the persisted boolean flags depend on its
    thresholds), the per-animal session metadata, the population-vector decay fit, and pre-aggregated counts
    of persistent / stably-tuned cells across the standard classifications.
    """

    configuration: DriftDetectionConfiguration
    """The drift configuration that produced the report."""
    cell_count: int
    """Number of multi-day-registered cells for the animal; constant across every session."""
    session_count: int
    """Number of sessions evaluated for this animal's drift report."""
    bleaching_available: bool
    """True when the animal's saved bleaching report was successfully aligned and used. False indicates the
    per-cell baseline slope and ``IS_HIGH_BLEACHING_DRIFT`` are NaN / False respectively for every cell."""

    session_names: list[str] = field(default_factory=list)
    """Session-directory timestamps in chronological order; aligns with the per-cell trajectory list columns."""
    days_since_first: list[float] = field(default_factory=list)
    """Per-session day offsets relative to the first session, parallel to ``session_names``."""
    trial_types: list[str] = field(default_factory=list)
    """Per-session evaluated trial type, parallel to ``session_names``."""
    bleaching_flagged_per_session: list[bool] = field(default_factory=list)
    """Per-session photobleaching flag from the bleaching report, parallel to ``session_names``. All-False
    when bleaching cross-correlation is unavailable."""

    persistent_place_count: int = 0
    """Number of cells flagged ``IS_PERSISTENT_PLACE`` after bleaching gating."""
    persistent_reward_count: int = 0
    """Number of cells flagged ``IS_PERSISTENT_REWARD`` after bleaching gating."""
    persistent_strict_place_count: int = 0
    """Number of cells flagged ``IS_PERSISTENT_STRICT_PLACE`` after bleaching gating."""
    stably_tuned_place_count: int = 0
    """Number of cells flagged ``IS_STABLY_TUNED_PLACE`` after bleaching gating."""
    stably_tuned_reward_count: int = 0
    """Number of cells flagged ``IS_STABLY_TUNED_REWARD`` after bleaching gating."""
    stably_tuned_strict_place_count: int = 0
    """Number of cells flagged ``IS_STABLY_TUNED_STRICT_PLACE`` after bleaching gating."""
    high_bleaching_drift_count: int = 0
    """Number of cells flagged ``IS_HIGH_BLEACHING_DRIFT``."""

    decay_amplitude: float = float("nan")
    """Decaying-component amplitude of the PV-correlation-vs-lag decay fit. NaN when the fit failed."""
    decay_tau_days: float = float("nan")
    """Decay time constant in days of the PV-correlation-vs-lag fit. NaN when the fit failed."""
    decay_offset: float = float("nan")
    """Asymptotic correlation offset of the PV-correlation-vs-lag fit. NaN when the fit failed."""
    decay_fit_succeeded: bool = False
    """True when the PV-correlation decay fit converged on finite parameters."""


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Top-level per-animal container for the cross-session tuning-drift evaluation.

    Holds the per-cell summary feather (``cells``), the per-session-pair feather (``pairs``), and the YAML
    summary (``summary``). Persistence is co-located here; plot regeneration lives in `.plotting`.
    """

    cells: pl.DataFrame
    """Per-cell wide table; one row per multi-day-registered cell. Schema enumerated by `DriftCellColumn`."""
    pairs: pl.DataFrame
    """Per-pair table; one row per ordered ``(session_a, session_b)`` pair with ``a < b``. Schema enumerated
    by `DriftPairColumn`."""
    summary: DriftSummary
    """YAML wrapper holding the configuration, per-animal session metadata, persistent / stably-tuned counts,
    and the population-vector decay fit."""

    @classmethod
    def load(cls, animal: DatasetAnimal) -> DriftReport:
        """Loads a previously saved drift report from the animal directory."""
        summary: DriftSummary = DriftSummary.from_yaml(file_path=animal.drift_summary_path)
        cells = pl.read_ipc(source=animal.drift_cells_path, memory_map=True)
        pairs = pl.read_ipc(source=animal.drift_pairs_path, memory_map=True)
        return cls(cells=cells, pairs=pairs, summary=summary)

    def save(self, animal: DatasetAnimal) -> None:
        """Persists the report to the three per-animal artifacts inside the animal directory."""
        self.summary.to_yaml(file_path=animal.drift_summary_path)
        self.cells.write_ipc(file=animal.drift_cells_path)
        self.pairs.write_ipc(file=animal.drift_pairs_path)

    def summarize(self) -> str:
        """Returns a multi-line human-readable summary of the drift evaluation."""
        summary = self.summary
        cell_count = summary.cell_count
        session_count = summary.session_count

        def percent(count: int) -> float:
            return 100.0 * count / cell_count if cell_count > 0 else 0.0

        lines: list[str] = [
            "Cross-session tuning drift",
            "==========================",
            f"Sessions:                   {session_count} "
            f"(spanning {summary.days_since_first[-1] - summary.days_since_first[0]:.1f} days)"
            if session_count > 0
            else "Sessions: 0",
            f"Cells (multi-day):          {cell_count}",
            f"Bleaching cross-correlation: {'on' if summary.bleaching_available else 'off'}",
            "",
            "Persistent classification (fraction-of-sessions threshold "
            f"= {summary.configuration.persistence_fraction_threshold:.2f})",
            "------------------------------------------------------------",
            f"  Persistent place cells:        {summary.persistent_place_count} "
            f"({percent(summary.persistent_place_count):.1f}%)",
            f"  Persistent reward cells:       {summary.persistent_reward_count} "
            f"({percent(summary.persistent_reward_count):.1f}%)",
            f"  Persistent strict-place cells: {summary.persistent_strict_place_count} "
            f"({percent(summary.persistent_strict_place_count):.1f}%)",
            "",
            "Stably tuned cells (persistent AND field-stable AND peak-stable AND bleaching-clean)",
            "-------------------------------------------------------------------------------------",
            f"  Stably tuned place:        {summary.stably_tuned_place_count} "
            f"({percent(summary.stably_tuned_place_count):.1f}%)",
            f"  Stably tuned reward:       {summary.stably_tuned_reward_count} "
            f"({percent(summary.stably_tuned_reward_count):.1f}%)",
            f"  Stably tuned strict-place: {summary.stably_tuned_strict_place_count} "
            f"({percent(summary.stably_tuned_strict_place_count):.1f}%)",
            "",
            "Bleaching-driven instability flag",
            "---------------------------------",
            f"  High-baseline-drift cells: {summary.high_bleaching_drift_count} "
            f"({percent(summary.high_bleaching_drift_count):.1f}%)",
            "",
            "Population-vector correlation decay (Climer 2025-style)",
            "-------------------------------------------------------",
        ]
        if summary.decay_fit_succeeded:
            lines.append(
                f"  amplitude={summary.decay_amplitude:.3f}, "
                f"tau={summary.decay_tau_days:.2f} days, offset={summary.decay_offset:.3f}  [converged]"
            )
        else:
            lines.append("  decay fit did not converge")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Prints the textual summary to the terminal via the ataraxis console."""
        console.echo(message=self.summarize(), raw=True)


def run_drift_analysis(
    dataset: DatasetData,
    *,
    animal: str | None = None,
    trial_type_overrides: dict[str, dict[str, str]] | None = None,
    use_bleaching: bool = True,
    configuration: DriftDetectionConfiguration | None = None,
) -> tuple[DriftReport, ...]:
    """Evaluates cross-session tuning drift for every animal in the dataset (or a single specified animal).

    Notes:
        Sole orchestrator for assembling `DriftReport` artifacts. Per-animal compute and methodological
        references live on `compute_drift_report`; this function resolves the in-scope animal set, walks
        them sequentially because per-animal compute is dominated by per-cell aggregations that are already
        numpy-vectorized, and persists each report to the animal directory. Animals with fewer than two
        sessions are surfaced as ValueError because the drift evaluation has no meaningful pair to evaluate.

    Args:
        dataset: The DatasetData whose animals are evaluated.
        animal: When provided, restricts the evaluation to a single animal. None evaluates every animal.
        trial_type_overrides: Optional nested map ``{animal: {session_name: trial_type}}`` that selects the
            trial type to use for sessions whose tuning summary lists more than one. Sessions not in the map
            default to the single trial type recorded in their summary.
        use_bleaching: When True (default), the animal's saved bleaching report is loaded and used to gate
            stability claims. When False, the drift report is bleaching-agnostic.
        configuration: Drift evaluation parameters shared across animals. Uses defaults if None.

    Returns:
        A tuple of DriftReports in the same order as the resolved animal set, with each report's three
        artifacts persisted under the corresponding animal directory.
    """
    resolved_configuration = configuration if configuration is not None else DriftDetectionConfiguration()
    overrides = trial_type_overrides if trial_type_overrides is not None else {}

    if animal is not None:
        resolved_animal = dataset.get_animal(animal=animal)
        animal_set: tuple[DatasetAnimal, ...] = (resolved_animal,)
    else:
        animal_set = dataset.animals

    if not animal_set:
        message = (
            f"Unable to run drift analysis on dataset {dataset.name!r}. The dataset contains no animals to "
            f"evaluate."
        )
        console.error(message=message, error=ValueError)

    console.echo(
        message=(
            f"Running drift analysis on dataset {dataset.name!r} for "
            f"{len(animal_set)} animal{'s' if len(animal_set) != 1 else ''}: "
            f"sequential per-animal loop."
        ),
        level=LogLevel.INFO,
    )
    delay_terminal()

    reports: list[DriftReport] = []
    with console.progress(
        total=len(animal_set), description="Running drift analysis", unit="animal"
    ) as progress_bar:
        for current_animal in animal_set:
            sessions = dataset.get_sessions_for_animal(animal=current_animal.animal)
            sorted_sessions = tuple(sorted(sessions, key=lambda dataset_session: dataset_session.session))
            if len(sorted_sessions) < _MINIMUM_SESSIONS_FOR_DRIFT:
                message = (
                    f"Unable to run drift analysis for animal {current_animal.animal!r}. The protocol "
                    f"requires at least {_MINIMUM_SESSIONS_FOR_DRIFT} sessions, but the animal has "
                    f"{len(sorted_sessions)}."
                )
                console.error(message=message, error=ValueError)

            report = compute_drift_report(
                animal=current_animal,
                sessions=sorted_sessions,
                trial_type_overrides=overrides.get(current_animal.animal),
                use_bleaching=use_bleaching,
                configuration=resolved_configuration,
            )
            report.save(animal=current_animal)
            reports.append(report)
            progress_bar.update()

    console.echo(
        message=(
            f"Drift analysis complete. Persisted {len(reports)} "
            f"report{'s' if len(reports) != 1 else ''} under {dataset.dataset_data_path.parent}."
        ),
        level=LogLevel.SUCCESS,
    )
    delay_terminal()
    return tuple(reports)


def compute_drift_report(
    animal: DatasetAnimal,
    sessions: tuple[DatasetSession, ...],
    *,
    trial_type_overrides: dict[str, str] | None = None,
    use_bleaching: bool = True,
    configuration: DriftDetectionConfiguration | None = None,
) -> DriftReport:
    """Computes the per-animal cross-session tuning-drift report.

    Notes:
        Per-animal algorithmic entry-point for the drift pipeline; consolidates every drift methodology that
        gates a column in the persisted feathers. Loads each session's `..tuning.TuningReport` and (when
        available) the animal's `..bleaching.BleachingReport`, stacks the per-session per-cell flags and
        rate maps into (session_count, cell_count) matrices, computes the pairwise drift metrics, aggregates
        them into per-cell scalars, fits a single-exponential decay to the per-pair population-vector
        correlation versus calendar lag, and produces composite ``IS_STABLY_TUNED_*`` flags that combine
        classification persistence, rate-map stability, peak stability, and bleaching cleanliness.

        Classification persistence and consecutive-pair recurrence probability follow the cross-session
        stability convention used by Ziv et al. (2013), Hainmueller & Bartos (2018), and Mau et al. (2018).
        Per-cell tuning-curve correlation (rate-map Pearson r averaged across pairs) and the random-remapping
        peak-shift null follow Krishnan & Sheffield (2024) and Hainmueller & Bartos (2018); the
        random-remapping null itself is the cell-ID-shuffle helper from
        `..tuning.utilities.random_remapping_peak_shift_p_values`. Population-vector correlation versus
        calendar lag with an exponential-decay summary follows Sheintuch et al. (2023) and Climer et al.
        (2025). The bleaching cross-correlation gates stability claims so peak instability driven by chronic
        photobleaching is not misattributed to representational drift, mirroring the convention used by
        Climer & Dombeck (2021) when discussing methodological controls for chronic two-photon imaging.

    References:
        Cross-session classification stability and recurrence rate framework:
            Ziv, Burns, Cocker, Hamel, Ghosh, Kitch, El Gamal & Schnitzer (2013). Long-term dynamics of
            CA1 hippocampal place codes. Nat Neurosci. https://doi.org/10.1038/nn.3329
            Hainmueller & Bartos (2018). Parallel emergence of stable and dynamic memory engrams in the
            hippocampus. Nature. https://doi.org/10.1038/s41586-018-0191-2
            Mau, Sullivan, Maurer, Hasselmo, Howard & Eichenbaum (2018). The same hippocampal CA1 population
            simultaneously codes temporal information over multiple timescales. Curr Biol.
            https://doi.org/10.1016/j.cub.2018.04.051
        Per-cell tuning-curve correlation, peak-shift null, and stable / drifting partition:
            Krishnan & Sheffield (2024). Mechanisms underlying the development and maintenance of stable
            spatial representations in the mouse hippocampus. Nat Commun.
            https://doi.org/10.1038/s41467-024-50596-3
            Gauthier & Tank (2018). A dedicated population for reward coding in the hippocampus. Neuron.
            https://doi.org/10.1016/j.neuron.2018.06.008
        Population-vector correlation versus calendar lag and exponential-decay summary:
            Sheintuch, Geva, Baumer, Rechavi, Rubin & Ziv (2023). Multiple maps of the same spatial context
            can stably coexist in the mouse hippocampus. Curr Biol.
            https://doi.org/10.1016/j.cub.2020.04.018
            Climer, Davoudi, Oh & Dombeck (2025). Hippocampal representations drift in stable multisensory
            environments. Nature. https://doi.org/10.1038/s41586-025-09245-y
        Methodological control for photobleaching-driven instability:
            Climer & Dombeck (2021). Choice of method of place cell classification determines the population
            of cells identified. PLoS Comput Biol. https://doi.org/10.1371/journal.pcbi.1008835
        Fisher's method for combining per-pair p-values across the pairwise drift matrix:
            Fisher (1925). Statistical Methods for Research Workers. Edinburgh: Oliver and Boyd.

    Args:
        animal: The DatasetAnimal whose drift report is being computed.
        sessions: Chronologically ordered DatasetSession entries for ``animal``.
        trial_type_overrides: Optional map from session name to the trial type to use when that session's
            tuning summary lists more than one. Sessions absent from the map default to the single trial
            type recorded in their summary.
        use_bleaching: When True, the animal's saved bleaching report is loaded and used to gate stability
            claims. When False, the drift report is bleaching-agnostic.
        configuration: Drift configuration. Uses defaults if None.

    Returns:
        An in-memory `DriftReport` ready to be saved or plotted.
    """
    resolved_configuration = configuration if configuration is not None else DriftDetectionConfiguration()

    animal_data = load_animal_session_data(
        animal=animal,
        sessions=sessions,
        trial_type_overrides=trial_type_overrides,
        use_bleaching=use_bleaching,
    )

    cell_count = animal_data.cell_count
    session_count = len(animal_data.sessions)

    # Resolve per-session masks. ``classification_session_mask`` controls trajectory aggregates; the
    # pair-level mask controls per-cell shift columns. Both default to "include every session" when
    # bleaching is unavailable or the user disabled the gate, and reduce to "include only bleaching-clean
    # sessions" when both conditions are met.
    if resolved_configuration.use_bleaching_session_mask and animal_data.bleaching_available:
        # noinspection PyTypeChecker
        clean_session_mask: NDArray[np.bool_] = ~animal_data.bleaching_flagged
    else:
        # noinspection PyTypeChecker
        clean_session_mask = np.ones(session_count, dtype=np.bool_)

    pairwise = compute_pairwise_drift_metrics(
        animal_data=animal_data,
        configuration=resolved_configuration,
        rate_map_session_mask=clean_session_mask,
    )

    cell_baseline_slope = fit_per_cell_baseline_slope(
        days_since_first=animal_data.days_since_first,
        cell_baseline_fluorescence=animal_data.cell_baseline_fluorescence,
    )

    cell_metrics = compute_per_cell_drift_metrics(
        animal_data=animal_data,
        pairwise=pairwise,
        configuration=resolved_configuration,
        classification_session_mask=clean_session_mask,
        cell_baseline_slope=cell_baseline_slope,
    )

    population = fit_population_decay(
        lag_days=pairwise.lag_days,
        population_vector_correlation=pairwise.population_vector_correlation,
    )

    cells_frame = _build_cells_frame(
        animal_data=animal_data,
        metrics=cell_metrics,
    )
    pairs_frame = _build_pairs_frame(
        animal_data=animal_data,
        pairwise=pairwise,
        clean_session_mask=clean_session_mask,
    )

    summary = DriftSummary(
        configuration=resolved_configuration,
        cell_count=cell_count,
        session_count=session_count,
        bleaching_available=animal_data.bleaching_available,
        session_names=list(animal_data.session_names),
        days_since_first=[float(value) for value in animal_data.days_since_first.tolist()],
        trial_types=list(animal_data.trial_types),
        bleaching_flagged_per_session=[bool(flag) for flag in animal_data.bleaching_flagged.tolist()],
        persistent_place_count=int(np.sum(cell_metrics.is_persistent_place)),
        persistent_reward_count=int(np.sum(cell_metrics.is_persistent_reward)),
        persistent_strict_place_count=int(np.sum(cell_metrics.is_persistent_strict_place)),
        stably_tuned_place_count=int(np.sum(cell_metrics.is_stably_tuned_place)),
        stably_tuned_reward_count=int(np.sum(cell_metrics.is_stably_tuned_reward)),
        stably_tuned_strict_place_count=int(np.sum(cell_metrics.is_stably_tuned_strict_place)),
        high_bleaching_drift_count=int(np.sum(cell_metrics.is_high_bleaching_drift)),
        decay_amplitude=population.decay_amplitude,
        decay_tau_days=population.decay_tau_days,
        decay_offset=population.decay_offset,
        decay_fit_succeeded=population.decay_fit_succeeded,
    )
    return DriftReport(cells=cells_frame, pairs=pairs_frame, summary=summary)


def _build_cells_frame(
    animal_data: AnimalSessionData,
    metrics: CellDriftMetrics,
) -> pl.DataFrame:
    """Assembles the per-cell drift feather from the live aggregation outputs."""
    cell_count = animal_data.cell_count
    if cell_count == 0:
        return pl.DataFrame(schema=_DRIFT_CELLS_EMPTY_SCHEMA)

    place_trajectory = [
        [bool(value) for value in animal_data.is_place[:, cell_index].tolist()]
        for cell_index in range(cell_count)
    ]
    reward_trajectory = [
        [bool(value) for value in animal_data.is_reward_cell[:, cell_index].tolist()]
        for cell_index in range(cell_count)
    ]
    strict_place_trajectory = [
        [bool(value) for value in animal_data.is_strict_place[:, cell_index].tolist()]
        for cell_index in range(cell_count)
    ]

    # noinspection PyTypeChecker
    cell_ids: NDArray[np.int32] = np.arange(cell_count, dtype=np.int32)

    return pl.DataFrame(
        {
            DriftCellColumn.CELL_ID.value: pl.Series(values=cell_ids, dtype=pl.Int32),
            DriftCellColumn.PLACE_TRAJECTORY.value: pl.Series(
                values=place_trajectory, dtype=pl.List(pl.Boolean)
            ),
            DriftCellColumn.REWARD_TRAJECTORY.value: pl.Series(
                values=reward_trajectory, dtype=pl.List(pl.Boolean)
            ),
            DriftCellColumn.STRICT_PLACE_TRAJECTORY.value: pl.Series(
                values=strict_place_trajectory, dtype=pl.List(pl.Boolean)
            ),
            DriftCellColumn.PLACE_SESSION_COUNT.value: pl.Series(
                values=metrics.place_session_count, dtype=pl.Int32
            ),
            DriftCellColumn.REWARD_SESSION_COUNT.value: pl.Series(
                values=metrics.reward_session_count, dtype=pl.Int32
            ),
            DriftCellColumn.STRICT_PLACE_SESSION_COUNT.value: pl.Series(
                values=metrics.strict_place_session_count, dtype=pl.Int32
            ),
            DriftCellColumn.PLACE_SESSION_FRACTION.value: pl.Series(
                values=metrics.place_session_fraction, dtype=pl.Float32
            ),
            DriftCellColumn.REWARD_SESSION_FRACTION.value: pl.Series(
                values=metrics.reward_session_fraction, dtype=pl.Float32
            ),
            DriftCellColumn.STRICT_PLACE_SESSION_FRACTION.value: pl.Series(
                values=metrics.strict_place_session_fraction, dtype=pl.Float32
            ),
            DriftCellColumn.PLACE_RECURRENCE_PROBABILITY.value: pl.Series(
                values=metrics.place_recurrence_probability, dtype=pl.Float32
            ),
            DriftCellColumn.REWARD_RECURRENCE_PROBABILITY.value: pl.Series(
                values=metrics.reward_recurrence_probability, dtype=pl.Float32
            ),
            DriftCellColumn.STRICT_PLACE_RECURRENCE_PROBABILITY.value: pl.Series(
                values=metrics.strict_place_recurrence_probability, dtype=pl.Float32
            ),
            DriftCellColumn.IS_PERSISTENT_PLACE.value: pl.Series(
                values=metrics.is_persistent_place, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_PERSISTENT_REWARD.value: pl.Series(
                values=metrics.is_persistent_reward, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_PERSISTENT_STRICT_PLACE.value: pl.Series(
                values=metrics.is_persistent_strict_place, dtype=pl.Boolean
            ),
            DriftCellColumn.PAIR_COUNT.value: pl.Series(
                values=metrics.pair_count_per_cell, dtype=pl.Int32
            ),
            DriftCellColumn.MEAN_RATE_MAP_CORRELATION.value: pl.Series(
                values=metrics.mean_rate_map_correlation, dtype=pl.Float32
            ),
            DriftCellColumn.MEAN_CONSECUTIVE_RATE_MAP_CORRELATION.value: pl.Series(
                values=metrics.mean_consecutive_rate_map_correlation, dtype=pl.Float32
            ),
            DriftCellColumn.MEAN_PEAK_SHIFT_CM.value: pl.Series(
                values=metrics.mean_peak_shift_cm, dtype=pl.Float32
            ),
            DriftCellColumn.MEAN_COM_SHIFT_CM.value: pl.Series(
                values=metrics.mean_com_shift_cm, dtype=pl.Float32
            ),
            DriftCellColumn.FISHER_PEAK_SHIFT_P_VALUE.value: pl.Series(
                values=metrics.fisher_peak_shift_p_value, dtype=pl.Float32
            ),
            DriftCellColumn.IS_FIELD_STABLE.value: pl.Series(
                values=metrics.is_field_stable, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_PEAK_STABLE.value: pl.Series(
                values=metrics.is_peak_stable, dtype=pl.Boolean
            ),
            DriftCellColumn.CELL_BASELINE_SLOPE.value: pl.Series(
                values=metrics.cell_baseline_slope, dtype=pl.Float32
            ),
            DriftCellColumn.IS_HIGH_BLEACHING_DRIFT.value: pl.Series(
                values=metrics.is_high_bleaching_drift, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_STABLY_TUNED_PLACE.value: pl.Series(
                values=metrics.is_stably_tuned_place, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_STABLY_TUNED_REWARD.value: pl.Series(
                values=metrics.is_stably_tuned_reward, dtype=pl.Boolean
            ),
            DriftCellColumn.IS_STABLY_TUNED_STRICT_PLACE.value: pl.Series(
                values=metrics.is_stably_tuned_strict_place, dtype=pl.Boolean
            ),
        }
    ).sort(DriftCellColumn.CELL_ID.value)


def _build_pairs_frame(
    animal_data: AnimalSessionData,
    pairwise: PairwiseDriftMetrics,
    *,
    clean_session_mask: NDArray[np.bool_],
) -> pl.DataFrame:
    """Assembles the per-pair drift feather from the live pairwise outputs."""
    pair_count = int(pairwise.pair_a_indices.size)
    if pair_count == 0:
        return pl.DataFrame(schema=_DRIFT_PAIRS_EMPTY_SCHEMA)

    a_indices = pairwise.pair_a_indices
    b_indices = pairwise.pair_b_indices
    session_a = [animal_data.session_names[int(index)] for index in a_indices]
    session_b = [animal_data.session_names[int(index)] for index in b_indices]
    both_clean = [
        bool(clean_session_mask[int(a)]) and bool(clean_session_mask[int(b)])
        for a, b in zip(a_indices.tolist(), b_indices.tolist(), strict=True)
    ]

    rate_map_corr_lists = [
        [float(value) for value in pairwise.rate_map_correlation_per_cell[pair_index].tolist()]
        for pair_index in range(pair_count)
    ]
    peak_shift_lists = [
        [float(value) for value in pairwise.peak_shift_cm_per_cell[pair_index].tolist()]
        for pair_index in range(pair_count)
    ]
    com_shift_lists = [
        [float(value) for value in pairwise.com_shift_cm_per_cell[pair_index].tolist()]
        for pair_index in range(pair_count)
    ]
    peak_p_value_lists = [
        [float(value) for value in pairwise.peak_shift_p_value_per_cell[pair_index].tolist()]
        for pair_index in range(pair_count)
    ]

    return pl.DataFrame(
        {
            DriftPairColumn.SESSION_A.value: pl.Series(values=session_a, dtype=pl.Utf8),
            DriftPairColumn.SESSION_B.value: pl.Series(values=session_b, dtype=pl.Utf8),
            DriftPairColumn.SESSION_A_INDEX.value: pl.Series(values=a_indices, dtype=pl.Int32),
            DriftPairColumn.SESSION_B_INDEX.value: pl.Series(values=b_indices, dtype=pl.Int32),
            DriftPairColumn.LAG_DAYS.value: pl.Series(values=pairwise.lag_days, dtype=pl.Float32),
            DriftPairColumn.POPULATION_VECTOR_CORRELATION.value: pl.Series(
                values=pairwise.population_vector_correlation, dtype=pl.Float32
            ),
            DriftPairColumn.PLACE_RECURRENCE_COUNT.value: pl.Series(
                values=pairwise.place_recurrence_count, dtype=pl.Int32
            ),
            DriftPairColumn.REWARD_RECURRENCE_COUNT.value: pl.Series(
                values=pairwise.reward_recurrence_count, dtype=pl.Int32
            ),
            DriftPairColumn.STRICT_PLACE_RECURRENCE_COUNT.value: pl.Series(
                values=pairwise.strict_place_recurrence_count, dtype=pl.Int32
            ),
            DriftPairColumn.BOTH_BLEACHING_CLEAN.value: pl.Series(
                values=both_clean, dtype=pl.Boolean
            ),
            DriftPairColumn.RATE_MAP_CORRELATION_PER_CELL.value: pl.Series(
                values=rate_map_corr_lists, dtype=pl.List(pl.Float32)
            ),
            DriftPairColumn.PEAK_SHIFT_CM_PER_CELL.value: pl.Series(
                values=peak_shift_lists, dtype=pl.List(pl.Float32)
            ),
            DriftPairColumn.COM_SHIFT_CM_PER_CELL.value: pl.Series(
                values=com_shift_lists, dtype=pl.List(pl.Float32)
            ),
            DriftPairColumn.PEAK_SHIFT_P_VALUE_PER_CELL.value: pl.Series(
                values=peak_p_value_lists, dtype=pl.List(pl.Float32)
            ),
        }
    ).sort([DriftPairColumn.SESSION_A_INDEX.value, DriftPairColumn.SESSION_B_INDEX.value])


_DRIFT_CELLS_EMPTY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    DriftCellColumn.CELL_ID.value: pl.Int32,
    DriftCellColumn.PLACE_TRAJECTORY.value: pl.List(pl.Boolean),
    DriftCellColumn.REWARD_TRAJECTORY.value: pl.List(pl.Boolean),
    DriftCellColumn.STRICT_PLACE_TRAJECTORY.value: pl.List(pl.Boolean),
    DriftCellColumn.PLACE_SESSION_COUNT.value: pl.Int32,
    DriftCellColumn.REWARD_SESSION_COUNT.value: pl.Int32,
    DriftCellColumn.STRICT_PLACE_SESSION_COUNT.value: pl.Int32,
    DriftCellColumn.PLACE_SESSION_FRACTION.value: pl.Float32,
    DriftCellColumn.REWARD_SESSION_FRACTION.value: pl.Float32,
    DriftCellColumn.STRICT_PLACE_SESSION_FRACTION.value: pl.Float32,
    DriftCellColumn.PLACE_RECURRENCE_PROBABILITY.value: pl.Float32,
    DriftCellColumn.REWARD_RECURRENCE_PROBABILITY.value: pl.Float32,
    DriftCellColumn.STRICT_PLACE_RECURRENCE_PROBABILITY.value: pl.Float32,
    DriftCellColumn.IS_PERSISTENT_PLACE.value: pl.Boolean,
    DriftCellColumn.IS_PERSISTENT_REWARD.value: pl.Boolean,
    DriftCellColumn.IS_PERSISTENT_STRICT_PLACE.value: pl.Boolean,
    DriftCellColumn.PAIR_COUNT.value: pl.Int32,
    DriftCellColumn.MEAN_RATE_MAP_CORRELATION.value: pl.Float32,
    DriftCellColumn.MEAN_CONSECUTIVE_RATE_MAP_CORRELATION.value: pl.Float32,
    DriftCellColumn.MEAN_PEAK_SHIFT_CM.value: pl.Float32,
    DriftCellColumn.MEAN_COM_SHIFT_CM.value: pl.Float32,
    DriftCellColumn.FISHER_PEAK_SHIFT_P_VALUE.value: pl.Float32,
    DriftCellColumn.IS_FIELD_STABLE.value: pl.Boolean,
    DriftCellColumn.IS_PEAK_STABLE.value: pl.Boolean,
    DriftCellColumn.CELL_BASELINE_SLOPE.value: pl.Float32,
    DriftCellColumn.IS_HIGH_BLEACHING_DRIFT.value: pl.Boolean,
    DriftCellColumn.IS_STABLY_TUNED_PLACE.value: pl.Boolean,
    DriftCellColumn.IS_STABLY_TUNED_REWARD.value: pl.Boolean,
    DriftCellColumn.IS_STABLY_TUNED_STRICT_PLACE.value: pl.Boolean,
}


_DRIFT_PAIRS_EMPTY_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    DriftPairColumn.SESSION_A.value: pl.Utf8,
    DriftPairColumn.SESSION_B.value: pl.Utf8,
    DriftPairColumn.SESSION_A_INDEX.value: pl.Int32,
    DriftPairColumn.SESSION_B_INDEX.value: pl.Int32,
    DriftPairColumn.LAG_DAYS.value: pl.Float32,
    DriftPairColumn.POPULATION_VECTOR_CORRELATION.value: pl.Float32,
    DriftPairColumn.PLACE_RECURRENCE_COUNT.value: pl.Int32,
    DriftPairColumn.REWARD_RECURRENCE_COUNT.value: pl.Int32,
    DriftPairColumn.STRICT_PLACE_RECURRENCE_COUNT.value: pl.Int32,
    DriftPairColumn.BOTH_BLEACHING_CLEAN.value: pl.Boolean,
    DriftPairColumn.RATE_MAP_CORRELATION_PER_CELL.value: pl.List(pl.Float32),
    DriftPairColumn.PEAK_SHIFT_CM_PER_CELL.value: pl.List(pl.Float32),
    DriftPairColumn.COM_SHIFT_CM_PER_CELL.value: pl.List(pl.Float32),
    DriftPairColumn.PEAK_SHIFT_P_VALUE_PER_CELL.value: pl.List(pl.Float32),
}
