"""Drift-pipeline utilities: per-animal session loading, classification trajectory assembly, bleaching pull.

These helpers are package-private to the drift analysis. They centralize the per-animal joins on the persisted
`TuningReport` and `BleachingReport` artifacts so the higher-level protocol and report modules can
operate on tidy numpy / polars structures without repeating the per-session loading boilerplate. Helpers that
any analysis package may need (e.g. acquisition-warmup trimming, session-day display unit) live in
`..shared_utilities`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from ataraxis_time import TimeUnits, TimestampFormats, convert_time, parse_timestamp
from ataraxis_base_utilities import console

from ..tuning import TuningColumn, TuningReport, TuningTrialSummary
from ..bleaching import BleachingColumn, BleachingReport

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...shared_assets import DatasetAnimal, DatasetSession


_SESSION_TIMESTAMP_FORMAT: str = "%Y-%m-%d-%H-%M-%S-%f"
"""``strptime`` format string for the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` session-directory name.
Mirrors the bleaching analyzer's constant; kept private here so the drift package does not reach into the
bleaching package's internals."""

_MINIMUM_SESSIONS_FOR_BASELINE_FIT: int = 2
"""Minimum number of sessions required to fit a per-cell linear baseline-fluorescence trend across days. The
slope is undefined for fewer than two finite samples on the day axis."""

_MINIMUM_SESSIONS_FOR_PAIR: int = 2
"""Minimum number of sessions required to enumerate at least one upper-triangular session pair."""

_MINIMUM_SESSIONS_FOR_RECURRENCE: int = 2
"""Minimum number of sessions required to compute a consecutive-pair recurrence probability. The conditioning
event needs at least one leading sample and one trailing sample."""


@dataclass(frozen=True, slots=True)
class AnimalSessionData:
    """Stores the per-animal cross-session tuning matrices and chronological session metadata.

    All arrays use ``session_count`` along the first axis and the multi-day-registered ``cell_count`` along the
    second so cross-session reductions collapse to a single numpy axis. The animal's bleaching report (when
    available and its session set matches this animal's tuning sessions) is carried alongside so the drift
    pipeline can gate stability claims by per-session bleaching status and per-cell baseline drift.
    """

    sessions: tuple[DatasetSession, ...]
    """Chronologically ordered DatasetSession entries for this animal."""
    session_names: tuple[str, ...]
    """Session-directory timestamp names, parallel to ``sessions``."""
    days_since_first: NDArray[np.float32]
    """Per-session day offsets relative to the first chronological session for this animal."""
    trial_types: tuple[str, ...]
    """Per-session evaluated trial type, parallel to ``sessions``. Each session contributes exactly one trial
    type to the drift analysis (selected upstream when the session's geometry has multiple entries)."""
    cell_count: int
    """Number of multi-day-registered cells for this animal; constant across every session."""
    bin_count: int
    """Spatial bin count; constant across every session because the same ``bin_size_cm`` is used throughout."""
    is_place: NDArray[np.bool_]
    """Per-session per-cell ``IS_PLACE`` flag with shape (session_count, cell_count)."""
    is_strict_place: NDArray[np.bool_]
    """Per-session per-cell ``IS_STRICT_PLACE`` flag with shape (session_count, cell_count)."""
    is_reward_cell: NDArray[np.bool_]
    """Per-session per-cell ``IS_REWARD_CELL`` flag with shape (session_count, cell_count)."""
    is_spatially_significant: NDArray[np.bool_]
    """Per-session per-cell Skaggs-significance flag with shape (session_count, cell_count)."""
    rate_maps: NDArray[np.float32]
    """Per-session per-cell smoothed rate map with shape (session_count, cell_count, bin_count)."""
    centers_of_mass_cm: NDArray[np.float32]
    """Per-session per-cell circular center-of-mass position with shape (session_count, cell_count). NaN where
    the COM is undefined (spatial information vanished or the cell did not pass the upstream COM gate)."""
    peak_positions_cm: NDArray[np.float32]
    """Per-session per-cell argmax position of the smoothed rate map in centimeters with shape (session_count,
    cell_count). NaN for cells whose rate map is all-NaN or all-zero in this session."""
    pf_centers_cm: tuple[tuple[NDArray[np.float32], ...], ...]
    """Per-session per-cell place-field centers in centimeters. Outer index is session, inner index is cell;
    each entry is a length-``field_count`` numpy array (possibly empty)."""
    bleaching_flagged: NDArray[np.bool_]
    """Per-session boolean from the saved `BleachingReport` if available, all-False otherwise. Shape
    (session_count,). True for sessions where any photobleaching threshold was exceeded."""
    cell_baseline_fluorescence: NDArray[np.float32]
    """Per-session per-cell baseline fluorescence with shape (session_count, cell_count) if a matching
    bleaching report was loaded, all-NaN otherwise."""
    bleaching_available: bool
    """True when a saved `BleachingReport` was loaded and aligned by session name; False when bleaching
    cross-correlation is disabled for this animal."""


def parse_session_microseconds(session_name: str) -> int:
    """Parses the canonical ``YYYY-MM-DD-HH-MM-SS-microseconds`` session-directory name into UTC microseconds.

    Args:
        session_name: Session directory name in the canonical dash-separated format.

    Returns:
        Microseconds elapsed since the UTC epoch corresponding to the session timestamp.

    Raises:
        ValueError: When the session name does not parse against the canonical format.
    """
    try:
        microseconds = parse_timestamp(
            date_string=session_name,
            format_string=_SESSION_TIMESTAMP_FORMAT,
            output_format=TimestampFormats.INTEGER,
        )
    except ValueError:
        message = (
            f"Unable to parse the session timestamp from {session_name!r}. The session directory name must "
            f"follow the 'YYYY-MM-DD-HH-MM-SS-microseconds' format."
        )
        console.error(message=message, error=ValueError)
    return int(microseconds)


def compute_days_since_first(session_names: tuple[str, ...]) -> NDArray[np.float32]:
    """Computes per-session day offsets relative to the first chronological session.

    Args:
        session_names: Chronologically ordered session-directory names.

    Returns:
        Per-session day offsets relative to ``session_names[0]`` as float32. The first entry is always 0.0.
    """
    if not session_names:
        # noinspection PyTypeChecker
        empty: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        return empty
    microseconds = tuple(parse_session_microseconds(session_name=name) for name in session_names)
    first_us = microseconds[0]
    days = [
        float(
            convert_time(
                time=session_us - first_us,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.DAY,
                as_float=True,
            )
        )
        for session_us in microseconds
    ]
    # noinspection PyTypeChecker
    days_array: NDArray[np.float32] = np.asarray(days, dtype=np.float32)
    return days_array


def load_animal_session_data(
    animal: DatasetAnimal,
    sessions: tuple[DatasetSession, ...],
    *,
    trial_type_overrides: dict[str, str] | None = None,
    use_bleaching: bool = True,
) -> AnimalSessionData:
    """Loads per-session `TuningReport` artifacts for an animal and stacks them into per-cell matrices.

    Notes:
        Each session contributes exactly one trial type. When the session's tuning report has a single trial
        type, it is used; when it has multiple entries, ``trial_type_overrides`` selects which one. The chosen
        trial type's per-cell rate map, place-field centers, classification flags, and COM are stacked into
        (session_count, cell_count[, bin_count]) numpy arrays. The cell count must match across every session
        because multi-day registration is per-animal — a mismatch indicates an upstream registration drift and
        is surfaced as ValueError.

        When ``use_bleaching=True`` and the animal has a saved `BleachingReport` whose session set matches
        the supplied tuning sessions exactly (by session name), the per-session ``flagged`` mask and per-cell
        baseline fluorescence are aligned to the same chronological order and attached to the result. Otherwise
        ``bleaching_available=False`` is set, ``bleaching_flagged`` is all-False, and
        ``cell_baseline_fluorescence`` is all-NaN; downstream consumers fall through to bleaching-agnostic
        behavior in that case.

    Args:
        animal: The DatasetAnimal whose drift report is being assembled.
        sessions: Chronologically ordered DatasetSession entries for ``animal``.
        trial_type_overrides: Optional map from session name to the trial type to use when that session's
            tuning summary lists more than one. Sessions absent from the map default to the single trial type
            recorded in their summary; sessions with multiple trial types and no override entry raise.
        use_bleaching: When True, attempt to load the animal's saved `BleachingReport` and align it. When
            False, skip bleaching loading entirely (``bleaching_available=False``).

    Returns:
        An AnimalSessionData with per-session per-cell matrices ready for drift quantification.

    Raises:
        ValueError: When session-level cell counts disagree, when a session's summary lists more than one
            trial type and no override is provided, or when a requested override trial type is missing.
    """
    if not sessions:
        message = (
            f"Unable to load tuning sessions for animal {animal.animal!r}. The supplied session tuple is empty."
        )
        console.error(message=message, error=ValueError)

    overrides = trial_type_overrides if trial_type_overrides is not None else {}
    session_names = tuple(session.session for session in sessions)
    days_since_first = compute_days_since_first(session_names=session_names)

    rate_maps_per_session: list[NDArray[np.float32]] = []
    centers_of_mass_per_session: list[NDArray[np.float32]] = []
    peak_positions_per_session: list[NDArray[np.float32]] = []
    pf_centers_per_session: list[tuple[NDArray[np.float32], ...]] = []
    is_place_per_session: list[NDArray[np.bool_]] = []
    is_strict_place_per_session: list[NDArray[np.bool_]] = []
    is_reward_per_session: list[NDArray[np.bool_]] = []
    is_spatial_per_session: list[NDArray[np.bool_]] = []
    trial_type_per_session: list[str] = []

    cell_count_reference: int | None = None
    bin_count_reference: int | None = None

    for session in sessions:
        report = TuningReport.load(session=session)
        available_trial_types = tuple(report.summary.trial_types)
        if not available_trial_types:
            message = (
                f"Unable to load drift inputs from session {session.session!r} of animal {animal.animal!r}. "
                f"The tuning summary lists no trial types."
            )
            console.error(message=message, error=ValueError)
        if len(available_trial_types) == 1:
            trial_type = available_trial_types[0]
        else:
            override = overrides.get(session.session)
            if override is None:
                message = (
                    f"Unable to load drift inputs from session {session.session!r} of animal "
                    f"{animal.animal!r}. The tuning summary lists multiple trial types "
                    f"{available_trial_types!r}; provide a ``trial_type_overrides`` entry for this session."
                )
                console.error(message=message, error=ValueError)
            elif override not in available_trial_types:
                message = (
                    f"Unable to load drift inputs from session {session.session!r} of animal "
                    f"{animal.animal!r}. The override trial type {override!r} is missing; the tuning "
                    f"summary lists {available_trial_types!r}."
                )
                console.error(message=message, error=ValueError)
            trial_type = override
        trial_type_per_session.append(trial_type)

        cells_frame = report.trial_cells(trial_type=trial_type)
        cell_count = int(cells_frame.height)
        if cell_count_reference is None:
            cell_count_reference = cell_count
        elif cell_count != cell_count_reference:
            message = (
                f"Cell count mismatch across sessions for animal {animal.animal!r}: session "
                f"{session.session!r} has {cell_count} cells but the first session had "
                f"{cell_count_reference}. Multi-day registration is per-animal, so cell counts must match "
                f"across the animal's tuning reports."
            )
            console.error(message=message, error=ValueError)

        rate_maps_list = cells_frame[TuningColumn.RATE_MAP.value].to_list()
        # noinspection PyTypeChecker
        rate_maps: NDArray[np.float32] = np.asarray(rate_maps_list, dtype=np.float32)
        if rate_maps.ndim == 1:
            # Polars empty-list-of-list edge case: rebuild as an explicit (cell_count, 0) matrix.
            # noinspection PyTypeChecker
            rate_maps = np.zeros((cell_count, 0), dtype=np.float32)
        bin_count = int(rate_maps.shape[1])
        if bin_count_reference is None:
            bin_count_reference = bin_count
        elif bin_count != bin_count_reference:
            message = (
                f"Bin count mismatch across sessions for animal {animal.animal!r}: session "
                f"{session.session!r} has {bin_count} rate-map bins but the first session had "
                f"{bin_count_reference}. The drift pipeline assumes a constant bin count across sessions."
            )
            console.error(message=message, error=ValueError)
        rate_maps_per_session.append(rate_maps)

        peak_positions = _resolve_peak_positions_cm(
            rate_maps=rate_maps,
            trial_summary=report.trial_summary(trial_type=trial_type),
        )
        peak_positions_per_session.append(peak_positions)

        # noinspection PyTypeChecker
        coms: NDArray[np.float32] = (
            cells_frame[TuningColumn.CENTER_OF_MASS_CM.value].to_numpy().astype(np.float32, copy=False)
        )
        # The persisted feather marks invalid COMs with -1; project that back onto NaN so the cross-session
        # comparisons can rely on numpy's standard NaN-propagation semantics.
        coms = np.where(coms < 0, np.nan, coms).astype(np.float32, copy=False)
        centers_of_mass_per_session.append(coms)

        pf_center_lists = cells_frame[TuningColumn.PF_CENTER_CM.value].to_list()
        pf_centers_per_session.append(
            tuple(np.asarray(values, dtype=np.float32) for values in pf_center_lists)
        )

        # noinspection PyTypeChecker
        is_place: NDArray[np.bool_] = cells_frame[TuningColumn.IS_PLACE.value].to_numpy()
        # noinspection PyTypeChecker
        is_strict_place: NDArray[np.bool_] = cells_frame[TuningColumn.IS_STRICT_PLACE.value].to_numpy()
        # noinspection PyTypeChecker
        is_reward: NDArray[np.bool_] = cells_frame[TuningColumn.IS_REWARD_CELL.value].to_numpy()
        # noinspection PyTypeChecker
        is_spatial: NDArray[np.bool_] = cells_frame[TuningColumn.IS_SPATIALLY_SIGNIFICANT.value].to_numpy()
        is_place_per_session.append(is_place)
        is_strict_place_per_session.append(is_strict_place)
        is_reward_per_session.append(is_reward)
        is_spatial_per_session.append(is_spatial)

    cell_count = int(cell_count_reference if cell_count_reference is not None else 0)
    bin_count = int(bin_count_reference if bin_count_reference is not None else 0)

    # noinspection PyTypeChecker
    rate_maps_stack: NDArray[np.float32] = np.stack(rate_maps_per_session, axis=0).astype(
        np.float32, copy=False
    )
    # noinspection PyTypeChecker
    coms_stack: NDArray[np.float32] = np.stack(centers_of_mass_per_session, axis=0).astype(
        np.float32, copy=False
    )
    # noinspection PyTypeChecker
    peaks_stack: NDArray[np.float32] = np.stack(peak_positions_per_session, axis=0).astype(
        np.float32, copy=False
    )
    # noinspection PyTypeChecker
    place_stack: NDArray[np.bool_] = np.stack(is_place_per_session, axis=0)
    # noinspection PyTypeChecker
    strict_place_stack: NDArray[np.bool_] = np.stack(is_strict_place_per_session, axis=0)
    # noinspection PyTypeChecker
    reward_stack: NDArray[np.bool_] = np.stack(is_reward_per_session, axis=0)
    # noinspection PyTypeChecker
    spatial_stack: NDArray[np.bool_] = np.stack(is_spatial_per_session, axis=0)

    bleaching_flagged, cell_baseline, bleaching_available = _try_load_bleaching(
        animal=animal,
        session_names=session_names,
        cell_count=cell_count,
        use_bleaching=use_bleaching,
    )

    return AnimalSessionData(
        sessions=sessions,
        session_names=session_names,
        days_since_first=days_since_first,
        trial_types=tuple(trial_type_per_session),
        cell_count=cell_count,
        bin_count=bin_count,
        is_place=place_stack,
        is_strict_place=strict_place_stack,
        is_reward_cell=reward_stack,
        is_spatially_significant=spatial_stack,
        rate_maps=rate_maps_stack,
        centers_of_mass_cm=coms_stack,
        peak_positions_cm=peaks_stack,
        pf_centers_cm=tuple(pf_centers_per_session),
        bleaching_flagged=bleaching_flagged,
        cell_baseline_fluorescence=cell_baseline,
        bleaching_available=bleaching_available,
    )


def _resolve_peak_positions_cm(
    rate_maps: NDArray[np.float32],
    trial_summary: TuningTrialSummary,
) -> NDArray[np.float32]:
    """Computes per-cell peak position in centimeters from a rate-map matrix and the trial summary.

    Args:
        rate_maps: Per-cell rate map matrix with shape (cell_count, bin_count).
        trial_summary: ``TuningTrialSummary`` providing the bin width and bin count used to convert peak bin
            indices into centimeters.

    Returns:
        Per-cell peak position in centimeters with length ``cell_count``. NaN for cells whose rate map is
        all-NaN or has zero variance.
    """
    cell_count = rate_maps.shape[0]
    bin_count = rate_maps.shape[1]
    # noinspection PyTypeChecker
    peaks: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if cell_count == 0 or bin_count == 0:
        return peaks
    bin_size_cm = float(trial_summary.bin_size_cm)
    for cell_index in range(cell_count):
        cell_map = rate_maps[cell_index]
        finite_mask = np.isfinite(cell_map)
        if not bool(np.any(finite_mask)):
            continue
        finite_values = cell_map[finite_mask]
        if float(finite_values.max() - finite_values.min()) <= 0.0:
            continue
        # ``argmax`` ignores the NaN entries because we replace them with -inf before the search; this avoids a
        # spurious peak at the first NaN bin while keeping the NaN-aware short-circuit at the entry.
        masked = np.where(finite_mask, cell_map, -np.inf)
        peak_bin = int(np.argmax(masked))
        peaks[cell_index] = (peak_bin + 0.5) * bin_size_cm
    return peaks


def _try_load_bleaching(
    animal: DatasetAnimal,
    session_names: tuple[str, ...],
    cell_count: int,
    *,
    use_bleaching: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.float32], bool]:
    """Attempts to load and align the animal's saved `BleachingReport` against the supplied session set.

    Notes:
        The bleaching report is per-animal but its session set may not match the drift pipeline's session
        selection one-for-one. The strict alignment rule used here keeps the cross-correlation simple: a
        bleaching report is consumed only when its session names are a superset of the drift session names
        and its cell count matches the multi-day-registered cell count from the tuning reports. Otherwise the
        bleaching cross-correlation is disabled for this animal and the caller falls through to the
        bleaching-agnostic branch.

    Args:
        animal: The DatasetAnimal whose bleaching report is sought.
        session_names: Chronologically ordered tuning session names.
        cell_count: Multi-day-registered cell count from the tuning reports.
        use_bleaching: When False, skip the load entirely.

    Returns:
        A tuple of ``(bleaching_flagged, cell_baseline_fluorescence, bleaching_available)`` where the first
        element has shape (session_count,), the second has shape (session_count, cell_count), and the third
        is True only when the report was successfully loaded and aligned.
    """
    session_count = len(session_names)
    # noinspection PyTypeChecker
    flagged_default: NDArray[np.bool_] = np.zeros(session_count, dtype=np.bool_)
    # noinspection PyTypeChecker
    baseline_default: NDArray[np.float32] = np.full(
        (session_count, cell_count), np.nan, dtype=np.float32
    )

    if not use_bleaching:
        return flagged_default, baseline_default, False
    if not animal.bleaching_path.exists() or not animal.bleaching_table_path.exists():
        return flagged_default, baseline_default, False

    try:
        bleaching_report = BleachingReport.load(animal=animal)
    except (ValueError, FileNotFoundError):
        return flagged_default, baseline_default, False

    table = bleaching_report.table
    available_names = table[BleachingColumn.SESSION.value].to_list()
    available_indices: dict[str, int] = {name: index for index, name in enumerate(available_names)}
    if not all(name in available_indices for name in session_names):
        return flagged_default, baseline_default, False

    # noinspection PyTypeChecker
    flagged_array: NDArray[np.bool_] = table[BleachingColumn.FLAGGED.value].to_numpy()
    cell_baseline_lists = table[BleachingColumn.CELL_BASELINE_FLUORESCENCE.value].to_list()

    if not cell_baseline_lists or len(cell_baseline_lists[0]) != cell_count:
        return flagged_default, baseline_default, False

    aligned_flagged = flagged_default.copy()
    aligned_baseline = baseline_default.copy()
    for index, name in enumerate(session_names):
        source_index = available_indices[name]
        aligned_flagged[index] = bool(flagged_array[source_index])
        aligned_baseline[index, :] = np.asarray(cell_baseline_lists[source_index], dtype=np.float32)

    return aligned_flagged, aligned_baseline, True


def fit_per_cell_baseline_slope(
    days_since_first: NDArray[np.float32],
    cell_baseline_fluorescence: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Fits a per-cell linear baseline-fluorescence trend in fluorescence units per day.

    Notes:
        Closed-form least-squares slope of ``baseline ~ alpha + beta * days`` per cell. Cells whose baseline
        column has fewer than two finite samples or zero variance in days receive NaN. The fit is intentionally
        unweighted: photobleaching is approximately log-linear over the observation window so a simple linear
        slope is informative as a quartile-based gate without committing to a specific decay model.

    Args:
        days_since_first: Per-session day offsets relative to the first session, length ``session_count``.
        cell_baseline_fluorescence: Per-session per-cell baseline fluorescence with shape (session_count,
            cell_count). May be all-NaN when bleaching is unavailable.

    Returns:
        Per-cell baseline slope in fluorescence units per day with length ``cell_count``. NaN where the fit
        cannot be produced (insufficient data or zero day-axis variance).
    """
    session_count, cell_count = cell_baseline_fluorescence.shape
    # noinspection PyTypeChecker
    slopes: NDArray[np.float32] = np.full(cell_count, np.nan, dtype=np.float32)
    if session_count < _MINIMUM_SESSIONS_FOR_BASELINE_FIT or cell_count == 0:
        return slopes

    # noinspection PyTypeChecker
    finite_session_mask: NDArray[np.bool_] = np.isfinite(days_since_first)
    if int(np.sum(finite_session_mask)) < _MINIMUM_SESSIONS_FOR_BASELINE_FIT:
        return slopes
    days = days_since_first[finite_session_mask].astype(np.float64, copy=False)
    day_variance = float(np.var(days))
    if day_variance <= 0.0:
        return slopes

    day_mean = float(np.mean(days))
    centered_days = days - day_mean
    denom = float(np.sum(centered_days * centered_days))

    for cell_index in range(cell_count):
        cell_baseline = cell_baseline_fluorescence[finite_session_mask, cell_index].astype(
            np.float64, copy=False
        )
        # noinspection PyTypeChecker
        valid_mask: NDArray[np.bool_] = np.isfinite(cell_baseline)
        valid_count = int(np.sum(valid_mask))
        if valid_count < _MINIMUM_SESSIONS_FOR_BASELINE_FIT:
            continue
        valid_days = centered_days[valid_mask]
        valid_baseline = cell_baseline[valid_mask] - float(np.mean(cell_baseline[valid_mask]))
        cell_denom = float(np.sum(valid_days * valid_days))
        if cell_denom > 0.0:
            slopes[cell_index] = float(np.sum(valid_days * valid_baseline) / cell_denom)
        elif denom > 0.0:
            # All cells share the same day axis when no per-cell NaN dropout occurred. Recover the closed-form
            # slope using the precomputed denominator from the full day vector.
            slopes[cell_index] = float(np.sum(centered_days * (cell_baseline - float(np.mean(cell_baseline))))
                                       / denom)
    return slopes


def session_pair_indices(session_count: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generates the upper-triangular ``(i, j)`` index pairs over a square session matrix with ``i < j``.

    Args:
        session_count: Number of sessions for the animal.

    Returns:
        A tuple of (a_indices, b_indices) numpy int64 arrays, each with length ``session_count * (session_count
        - 1) // 2``. Use these to drive every pairwise reduction in the drift pipeline so that ``(a, b)`` and
        ``(b, a)`` are not double-counted.
    """
    if session_count < _MINIMUM_SESSIONS_FOR_PAIR:
        # noinspection PyTypeChecker
        empty: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        return empty, empty
    # noinspection PyTypeChecker
    a_indices, b_indices = np.triu_indices(n=session_count, k=1)
    return a_indices.astype(np.int64, copy=False), b_indices.astype(np.int64, copy=False)


def boolean_recurrence_probability(per_session_flag: NDArray[np.bool_]) -> float:
    """Computes the consecutive-pair recurrence probability for a per-session boolean trajectory.

    Notes:
        Defined as ``P(flag_{t+1} = True | flag_t = True)`` averaged over consecutive session pairs in which
        ``flag_t = True``. Returns NaN when no consecutive ``flag_t = True`` is observed (the cell was never
        classified or only classified in the final session). Mirrors the recurrence rate used in Ziv et al.
        (2013) and Hainmueller & Bartos (2018) for cross-session classification stability.

    Args:
        per_session_flag: Per-session boolean trajectory with length ``session_count``.

    Returns:
        Recurrence probability as a float in ``[0.0, 1.0]``, or NaN when the conditioning event is empty.
    """
    if per_session_flag.size < _MINIMUM_SESSIONS_FOR_RECURRENCE:
        return float("nan")
    leading = per_session_flag[:-1]
    trailing = per_session_flag[1:]
    conditioning_count = int(np.sum(leading))
    if conditioning_count == 0:
        return float("nan")
    return float(np.sum(np.logical_and(leading, trailing)) / conditioning_count)


