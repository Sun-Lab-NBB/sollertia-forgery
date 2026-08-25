"""Contains tests for the system-agnostic substrate: the per-session tracker locations, the shared utilities, and the
terminal configuration the distribution applies when it is imported.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import importlib

import polars as pl
import pytest
from ataraxis_base_utilities import console

import sollertia_forgery
from sollertia_forgery.shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    natural_sort,
    delay_terminal,
    multi_recording_dataset_name,
    resolve_session_tracker_path,
)

if TYPE_CHECKING:
    from sollertia_shared_assets import SessionData


# Per-session tracker locations


def test_every_per_session_pipeline_resolves_the_location_its_own_session_declares(
    experiment_session: SessionData,
) -> None:
    """Verifies that the reporting layer and the dispatch table both read this mapping, so it must name the session's
    own paths.
    """
    resolved = {
        pipeline: resolve_session_tracker_path(session=experiment_session, pipeline=pipeline)
        for pipeline in SESSION_PIPELINES
    }

    assert resolved == {
        ProcessingPipelines.CHECKSUM: experiment_session.raw_data.checksum_tracker_path,
        ProcessingPipelines.RUNTIME: experiment_session.processed_data.runtime_tracker_path,
        ProcessingPipelines.MICROCONTROLLER: experiment_session.processed_data.microcontroller_tracker_path,
        ProcessingPipelines.VIDEO: experiment_session.processed_data.video_tracker_path,
        ProcessingPipelines.TWO_PHOTON: experiment_session.processed_data.two_photon_tracker_path,
    }


def test_the_checksum_tracker_sits_beside_the_data_it_verifies(experiment_session: SessionData) -> None:
    """Verifies that the pipeline verifies the acquired data in place, so its record belongs under the acquired data."""
    resolved = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.CHECKSUM)

    assert resolved.parent == experiment_session.raw_data_path


@pytest.mark.parametrize("pipeline", [ProcessingPipelines.MANIFEST, ProcessingPipelines.FORGING])
def test_a_pipeline_that_processes_no_single_session_is_rejected(
    experiment_session: SessionData, pipeline: ProcessingPipelines
) -> None:
    """Verifies that one pipeline operates on a project and the other on a dataset, so neither records a per-session
    tracker.
    """
    with pytest.raises(ValueError, match=f"tracker path of pipeline '{pipeline.value}'"):
        resolve_session_tracker_path(session=experiment_session, pipeline=pipeline)


# Shared utilities


def test_the_terminal_delay_holds_the_runtime_for_its_declared_period() -> None:
    """Verifies that consecutive printouts stay visually separated only if the delay actually elapses."""
    start = time.perf_counter()

    delay_terminal()

    assert time.perf_counter() - start >= 0.09


def test_a_natural_sort_orders_an_identifier_by_the_number_it_embeds() -> None:
    """Verifies that every identifier on which this library orders embeds a number in text, so a listing has to place 2
    ahead of 10.

    Ordering the same identifiers as plain text puts 10 ahead of 2, which is the inversion this function exists to keep
    out of the project manifest, the project plan, the project jobs artifact, and the dataset state artifact.
    """
    frame = pl.DataFrame({"animal": ["10", "2", "1"], "session": ["a", "b", "c"]})

    ordered = natural_sort(frame=frame, by=["animal"])

    assert ordered["animal"].to_list() == ["1", "2", "10"]
    # The remaining columns travel with the row that supplied their identifier.
    assert ordered["session"].to_list() == ["c", "b", "a"]


def test_the_multi_recording_dataset_name_is_qualified_by_animal() -> None:
    """Verifies that an animal's multi-recording output stays separate from its peers only when the name carries the
    animal, and the fold that reaches the directory is cindra's, so the name itself keeps the casing it was given.
    """
    assert multi_recording_dataset_name(animal_id="305", dataset_name="PlaceCells") == "305_PlaceCells"


# Import-time terminal configuration


@pytest.mark.xdist_group(name="console")
def test_importing_the_library_turns_on_whichever_terminal_channel_is_off() -> None:
    """Verifies that every pipeline reports through the console and its progress bars, so importing enables both."""
    console.disable()

    importlib.reload(sollertia_forgery)

    assert console.enabled, "an import found the console off and left it off"
    assert console.progress_enabled

    console.disable_progress()

    importlib.reload(sollertia_forgery)

    assert console.enabled
    assert console.progress_enabled, "an import found the progress bars off and left them off"
