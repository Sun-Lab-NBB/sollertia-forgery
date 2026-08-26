"""Contains tests for the tools that enumerate the batches this host records and drop the records of a finished run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from sollertia_forgery.orchestration import record_prepared_batch
from sollertia_forgery.orchestration.graph import BatchDocument
from sollertia_forgery.orchestration.batches import record_batch_outcome, retire_prepared_batch
from sollertia_forgery.interfaces.orchestration_tools import (
    list_prepared_batches_tool,
    forget_prepared_batches_tool,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("isolated_working_directory")


def test_a_batch_that_has_yet_to_run_reports_no_outcome(isolated_working_directory: Path) -> None:
    """Verifies that a prepared batch is listed with the counts its document carries and no recorded outcome."""
    batch_id = _record_batch()

    entry = list_prepared_batches_tool()["batches"][0]

    assert entry["batch_id"] == batch_id
    assert entry["pipeline"] == "video"
    assert entry["unit_count"] == 1
    assert entry["job_count"] == 1
    assert not entry["outcome_recorded"]


def test_a_settled_batch_stays_listed_under_the_outcome_that_answers_for_it(
    isolated_working_directory: Path,
) -> None:
    """Verifies that closure retiring a document leaves the batch listed, which is what recovers its identifier."""
    batch_id = _record_batch()
    _settle_batch(batch_id=batch_id, outcome={"pipeline": "video", "host": "local", "total": 3, "blocked": 1})

    response = list_prepared_batches_tool()

    assert response["total_batches"] == 1
    assert response["total_jobs"] == 3
    entry = response["batches"][0]
    assert entry["batch_id"] == batch_id
    assert entry["pipeline"] == "video"
    assert entry["host"] == "local"
    assert entry["job_count"] == 3
    assert entry["blocked_count"] == 1
    assert entry["outcome_recorded"]
    # The prepared document is the only source of the unit count, and it is gone, so the entry drops that field.
    assert "unit_count" not in entry


def test_a_settled_batch_is_resolvable_by_the_identifier_it_ran_under(isolated_working_directory: Path) -> None:
    """Verifies that naming a settled batch resolves it rather than reporting the identifier as unheld."""
    batch_id = _record_batch()
    _settle_batch(batch_id=batch_id, outcome={"pipeline": "video", "host": "local", "total": 1})

    response = list_prepared_batches_tool(batch_ids=[batch_id])

    assert response["success"]
    assert [entry["batch_id"] for entry in response["batches"]] == [batch_id]


def test_forgetting_a_settled_batch_drops_it_from_the_listing(isolated_working_directory: Path) -> None:
    """Verifies that the forget reaches a batch held by its outcome alone, so nothing outlives every record of it."""
    batch_id = _record_batch()
    _settle_batch(batch_id=batch_id, outcome={"pipeline": "video", "host": "local", "total": 1})

    response = forget_prepared_batches_tool(batch_ids=[batch_id, "never_recorded"])

    assert response["forgotten"] == [batch_id]
    assert response["total_forgotten"] == 1
    assert response["unknown"] == ["never_recorded"]
    assert list_prepared_batches_tool()["total_batches"] == 0


def test_forgetting_a_prepared_batch_leaves_the_others_alone(isolated_working_directory: Path) -> None:
    """Verifies that a forget covers the batches it names and nothing else the registry holds."""
    forgotten = _record_batch()
    kept = _record_batch(pipeline="checksum")

    forget_prepared_batches_tool(batch_ids=[forgotten])

    assert [entry["batch_id"] for entry in list_prepared_batches_tool()["batches"]] == [kept]


def test_forgetting_without_an_identifier_is_refused(isolated_working_directory: Path) -> None:
    """Verifies that a forget naming no batch is refused rather than treated as covering every batch."""
    batch_id = _record_batch()

    response = forget_prepared_batches_tool(batch_ids=[])

    assert not response["success"]
    assert list_prepared_batches_tool(batch_ids=[batch_id])["success"]


def _record_batch(pipeline: str = "video") -> str:
    """Records one prepared batch holding a single dispatchable job.

    Args:
        pipeline: The pipeline the recorded batch dispatches.

    Returns:
        The identifier under which the batch was recorded.
    """
    return record_prepared_batch(
        document=BatchDocument(
            pipeline=pipeline,
            host="local",
            units=[{"unit_path": "/nonexistent/session", "unit_name": "session", "job_count": 1}],
            jobs=[{"job_id": "a_job", "job_name": "motion_energy", "unit_path": "/nonexistent/session"}],
        )
    )


def _settle_batch(batch_id: str, outcome: dict[str, Any]) -> None:
    """Records one batch's outcome and retires the document behind it, as closure does.

    Args:
        batch_id: The identifier of the batch to settle.
        outcome: The outcome closure recorded for it.
    """
    record_batch_outcome(batch_id=batch_id, outcome=outcome)
    retire_prepared_batch(batch_id=batch_id)
