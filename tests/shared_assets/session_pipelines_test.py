"""Contains tests for the per-session pipeline identity assets: the tracker-path resolver, the pipeline tuple derived
from it, the manifest schema invariants that both drive, and the Mesoscope-VR forging admission and dispatch assets.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sollertia_shared_assets import SessionTypes

from sollertia_forgery.shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    resolve_session_tracker_path,
)
from sollertia_forgery.managing.manifest import _PIPELINE_STATUS_COLUMNS
from sollertia_forgery.mesoscope_vr.forging import MESOSCOPE_ADMISSION_PIPELINES, assemble_mesoscope_session

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_EXPECTED_SESSION_PIPELINES: tuple[ProcessingPipelines, ...] = (
    ProcessingPipelines.CHECKSUM,
    ProcessingPipelines.RUNTIME,
    ProcessingPipelines.MICROCONTROLLER,
    ProcessingPipelines.VIDEO,
    ProcessingPipelines.TWO_PHOTON,
)
"""The pipelines for which a session carries a tracker, in the order the manifest presents them. Pinned explicitly,
because the order of the manifest's status columns follows it."""

_NON_SESSION_PIPELINES: tuple[ProcessingPipelines, ...] = (
    ProcessingPipelines.MANIFEST,
    ProcessingPipelines.FORGING,
)
"""The pipelines that operate on a project or a dataset, which therefore record no per-session tracker."""


def _make_session() -> SimpleNamespace:
    """Builds a stand-in session whose tracker properties report which accessor the resolver reached."""
    return SimpleNamespace(
        session_name="2026-01-02-03-04-05-000006",
        raw_data=SimpleNamespace(checksum_tracker_path="raw/checksum"),
        processed_data=SimpleNamespace(
            runtime_tracker_path="processed/runtime",
            microcontroller_tracker_path="processed/microcontroller",
            video_tracker_path="processed/video",
            two_photon_tracker_path="processed/two_photon",
        ),
    )


def test_session_pipelines_holds_every_per_session_pipeline_in_order() -> None:
    """Verifies that SESSION_PIPELINES matches the pinned per-session pipeline order."""
    assert SESSION_PIPELINES == _EXPECTED_SESSION_PIPELINES


def test_every_session_pipeline_resolves_its_own_tracker() -> None:
    """Verifies that every per-session pipeline resolves to its own distinct tracker accessor."""
    session = _make_session()
    resolved = {
        pipeline: resolve_session_tracker_path(session=session, pipeline=pipeline)  # type: ignore[arg-type]
        for pipeline in SESSION_PIPELINES
    }
    assert resolved == {
        ProcessingPipelines.CHECKSUM: "raw/checksum",
        ProcessingPipelines.RUNTIME: "processed/runtime",
        ProcessingPipelines.MICROCONTROLLER: "processed/microcontroller",
        ProcessingPipelines.VIDEO: "processed/video",
        ProcessingPipelines.TWO_PHOTON: "processed/two_photon",
    }
    assert len(set(resolved.values())) == len(SESSION_PIPELINES)


@pytest.mark.parametrize("pipeline", _NON_SESSION_PIPELINES)
def test_non_session_pipeline_resolution_raises(pipeline: ProcessingPipelines) -> None:
    """Verifies that a project-scoped or dataset-scoped pipeline is rejected rather than resolved to a session path."""
    with pytest.raises(ValueError, match="records a per-session tracker"):
        resolve_session_tracker_path(session=_make_session(), pipeline=pipeline)  # type: ignore[arg-type]


def test_every_session_pipeline_declares_a_manifest_status_column() -> None:
    """Verifies that the manifest declares one status column per per-session pipeline and no others."""
    # The manifest module asserts this at import, so this test pins the invariant that assertion protects.
    assert set(_PIPELINE_STATUS_COLUMNS) == set(SESSION_PIPELINES)


def test_status_column_names_stay_distinct() -> None:
    """Verifies that every manifest status column name is distinct."""
    assert len(set(_PIPELINE_STATUS_COLUMNS.values())) == len(_PIPELINE_STATUS_COLUMNS)


def test_checksum_reports_under_the_integrity_column() -> None:
    """Verifies that the checksum pipeline reports under the manifest's integrity column."""
    assert _PIPELINE_STATUS_COLUMNS[ProcessingPipelines.CHECKSUM] == "integrity"


def test_admission_requires_only_pipelines_a_session_records() -> None:
    """Verifies that every admission requirement names a pipeline that records a per-session tracker."""
    for pipelines in MESOSCOPE_ADMISSION_PIPELINES.values():
        assert pipelines <= set(SESSION_PIPELINES)


def test_admission_pins_every_pipeline_each_session_type_requires() -> None:
    """Verifies each session type's admission set exactly, since this gate is the only thing holding an unprocessed
    session out of a forged dataset and a subset assertion cannot tell a dropped requirement from a narrower one.
    """
    training_requirement = frozenset(
        {
            ProcessingPipelines.CHECKSUM,
            ProcessingPipelines.RUNTIME,
            ProcessingPipelines.MICROCONTROLLER,
            ProcessingPipelines.VIDEO,
        }
    )

    assert MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.MESOSCOPE_EXPERIMENT] == (
        training_requirement | {ProcessingPipelines.TWO_PHOTON}
    )
    assert MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.RUN_TRAINING] == training_requirement
    assert MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.LICK_TRAINING] == training_requirement


def test_admission_asks_a_training_session_for_no_imaging() -> None:
    """Verifies that a training session's admission omits the two-photon pipeline an experiment session requires."""
    experiment_requirement = MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.MESOSCOPE_EXPERIMENT]
    training_requirement = MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.RUN_TRAINING]

    assert ProcessingPipelines.TWO_PHOTON in experiment_requirement
    assert ProcessingPipelines.TWO_PHOTON not in training_requirement
    assert training_requirement == MESOSCOPE_ADMISSION_PIPELINES[SessionTypes.LICK_TRAINING]
    assert experiment_requirement - training_requirement == {ProcessingPipelines.TWO_PHOTON}


def test_a_window_checking_session_joins_no_dataset() -> None:
    """Verifies that a window checking session declares no admission requirement."""
    assert SessionTypes.WINDOW_CHECKING not in MESOSCOPE_ADMISSION_PIPELINES


def test_dispatch_rejects_a_window_checking_session(
    session_factory: Callable[..., SessionData], tmp_path: Path
) -> None:
    """Verifies that the dispatcher refuses a session type it cannot assemble and names the supported types."""
    session = session_factory(animal_id="404", session_type=SessionTypes.WINDOW_CHECKING)
    output_path = tmp_path.joinpath("forged", "data.feather")

    with pytest.raises(ValueError, match=r"(?s)'window checking' is not a\s+supported forging session type") as error:
        assemble_mesoscope_session(
            source_session_path=session.raw_data_path.parent, output_path=output_path, dataset_name="dataset"
        )

    reported = " ".join(str(error.value).split())
    assert reported.endswith("The supported session types are: lick training, mesoscope experiment, run training.")
    assert not output_path.exists()
