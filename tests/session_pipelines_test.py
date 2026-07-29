"""Tests the per-session pipeline identity assets: the tracker-path resolver, the pipeline tuple derived from it, and
the manifest schema invariants that both drive.

The manifest's per-pipeline status columns and the project job artifact's rows are both derived from these, so a
change to either surfaces here rather than in a generated artifact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sollertia_forgery.shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    resolve_session_tracker_path,
)
from sollertia_forgery.managing.manifest import PIPELINE_STATUS_COLUMNS

EXPECTED_SESSION_PIPELINES: tuple[ProcessingPipelines, ...] = (
    ProcessingPipelines.CHECKSUM,
    ProcessingPipelines.RUNTIME,
    ProcessingPipelines.MICROCONTROLLER,
    ProcessingPipelines.VIDEO,
    ProcessingPipelines.TWO_PHOTON,
)
"""The pipelines a session carries a tracker for, in the order the manifest presents them. Pinned explicitly, because
the order of the manifest's status columns follows it."""

NON_SESSION_PIPELINES: tuple[ProcessingPipelines, ...] = (
    ProcessingPipelines.MANIFEST,
    ProcessingPipelines.FORGING,
)
"""The pipelines that operate on a project or a dataset, which therefore record no per-session tracker."""


def make_session() -> SimpleNamespace:
    """Builds a stand-in session whose tracker properties report which accessor the resolver reached for."""
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
    """The derived tuple matches the pinned order, since the manifest's struct field order follows it."""
    assert SESSION_PIPELINES == EXPECTED_SESSION_PIPELINES


def test_every_session_pipeline_resolves_its_own_tracker() -> None:
    """Each pipeline reaches a distinct accessor, so no two pipelines share a tracker location."""
    session = make_session()
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


@pytest.mark.parametrize("pipeline", NON_SESSION_PIPELINES)
def test_a_pipeline_of_another_scope_has_no_session_tracker(pipeline: ProcessingPipelines) -> None:
    """A project-scoped or dataset-scoped pipeline is rejected rather than resolved to a session path."""
    with pytest.raises(ValueError, match="records a per-session tracker"):
        resolve_session_tracker_path(session=make_session(), pipeline=pipeline)  # type: ignore[arg-type]


def test_every_session_pipeline_declares_a_manifest_status_column() -> None:
    """The manifest reports one status column per pipeline a session carries a tracker for, and no others.

    The manifest module asserts this at import, so this test pins the invariant that assertion protects.
    """
    assert set(PIPELINE_STATUS_COLUMNS) == set(SESSION_PIPELINES)


def test_status_column_names_stay_distinct() -> None:
    """Every pipeline reports into its own manifest column, so one pipeline's status never overwrites another's."""
    assert len(set(PIPELINE_STATUS_COLUMNS.values())) == len(PIPELINE_STATUS_COLUMNS)


def test_checksum_reports_under_the_integrity_column() -> None:
    """The checksum pipeline's status lives under the historically named column the manifest already published."""
    assert PIPELINE_STATUS_COLUMNS[ProcessingPipelines.CHECKSUM] == "integrity"
