"""Contains tests for the resource narrowing that the generic processing tools apply to a local batch before they
dispatch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import pytest

from sollertia_forgery.video import CAMERA_EXTRACTION_JOB_NAME
from sollertia_forgery.interfaces import processing_tools
from sollertia_forgery.orchestration import GenericPendingJob
from sollertia_forgery.interfaces.processing_tools import _execute_local_batch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_MEMORY_BUDGET_MB: int = 16_384
"""The memory budget against which every staged batch runs, kept far above what its jobs request so the core terms
alone decide the outcome."""


# Helpers


class _IdleThread:
    """Stands in for the manager thread, so a batch is reconciled, sized, and staged without any job running.

    Attributes:
        args: The positional arguments the batch would have handed the manager.
    """

    def __init__(self, target: object, args: tuple[Any, ...], daemon: bool) -> None:  # noqa: ARG002, FBT001
        self.args = args

    def start(self) -> None:
        """Stands in for starting the manager, which a staged batch never does."""

    def is_alive(self) -> bool:
        """Reports the staged manager as finished, so a later batch is never rejected as already running."""
        return False


class _RecordingHost:
    """Stands in for the host holding the trackers, recording the reset each staged batch applies through it.

    Attributes:
        reset_calls: One entry per reset, carrying the pipeline and the identifiers it cleared on each unit.
    """

    def __init__(self) -> None:
        self.reset_calls: list[tuple[str, dict[str, list[str]]]] = []

    def reset_jobs(self, pipeline: str, job_ids_by_unit: Mapping[Path, Sequence[str]]) -> None:
        """Records one reset rather than rewriting any tracker.

        Args:
            pipeline: The pipeline whose jobs are cleared.
            job_ids_by_unit: The identifiers of the cleared jobs, keyed by the unit whose tracker holds them.
        """
        self.reset_calls.append(
            (pipeline, {str(unit_path): list(job_ids) for unit_path, job_ids in job_ids_by_unit.items()})
        )


def _make_job(job_id: str, cores: int, job_name: str = CAMERA_EXTRACTION_JOB_NAME) -> GenericPendingJob:
    """Builds one dispatchable job carrying the width its own sizing pass gave it.

    Args:
        job_id: The identifier under which the tracker records the job.
        cores: The cores the sizing pass chose for this particular job.
        job_name: The job type name, which is what groups this job with the others of its type.

    Returns:
        The pending job.
    """
    return GenericPendingJob(
        tracker_path=Path("/nonexistent/tracker.yaml"),
        job_id=job_id,
        unit_path=Path("/nonexistent/session"),
        job_name=job_name,
        core_weight=cores,
        memory_mb=512,
        pipeline="video",
    )


def _stage_batch(jobs: list[GenericPendingJob], core_budget: int) -> dict[str, Any]:
    """Runs one batch through the local dispatch path up to the point the manager would start.

    Args:
        jobs: The batch's jobs, whose widths are narrowed in place.
        core_budget: The cores the batch may commit across all concurrently running jobs.

    Returns:
        The response dict the execute tool would return.
    """
    return _execute_local_batch(
        host=_RecordingHost(),  # type: ignore[arg-type]
        pending=jobs,
        batch_ids=["batch"],
        core_budget_override=core_budget,
        memory_budget_mb=_MEMORY_BUDGET_MB,
    )


@pytest.fixture
def staged_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Holds the manager thread and the host's own core count out of the dispatch path.

    The requested budget is honored exactly, so a test states the host it means rather than inheriting whatever the
    machine running it happens to hold. Recording the module's execution state has it restored afterwards, which
    keeps a staged batch from being mistaken for a live one.

    Args:
        monkeypatch: The fixture used to replace each dependency that the dispatch path uses.
    """

    def _honor_request(requested_workers: int, reserved_cores: int) -> int:
        """Answers with the budget the caller requested, whatever the host running the test holds."""
        return requested_workers

    monkeypatch.setattr(processing_tools, "Thread", _IdleThread)
    monkeypatch.setattr(processing_tools, "resolve_worker_count", _honor_request)
    monkeypatch.setattr(processing_tools, "_EXECUTION_STATE", None)


# Per-job core widths


def test_a_batch_keeps_the_width_each_job_was_sized_at(staged_batch: None) -> None:
    """Verifies that the archive-backed extraction stages resolve a width per job, so one type holds jobs of several
    widths and dispatch has to carry each of them through rather than flattening the type onto a single figure.
    """
    narrow = _make_job(job_id="narrow", cores=1)
    wide = _make_job(job_id="wide", cores=8)

    # The per-archive sizing is the whole reason these two jobs differ, so a run where they match would let the
    # assertions below pass without the narrowing ever being exercised.
    assert narrow.core_weight != wide.core_weight

    response = _stage_batch(jobs=[narrow, wide], core_budget=32)

    assert response["success"]
    assert narrow.core_weight == 1
    assert wide.core_weight == 8
    # The type's reported allocation is the widest of its jobs, since that figure is what divides the budget into the
    # concurrency the response reports for the type.
    assert response["job_allocations"][CAMERA_EXTRACTION_JOB_NAME]["cores_per_job"] == 8
    assert response["job_allocations"][CAMERA_EXTRACTION_JOB_NAME]["maximum_parallel"] == 4


def test_the_type_width_does_not_depend_on_the_order_its_jobs_arrive_in(staged_batch: None) -> None:
    """Verifies that every job of a type is a candidate representative, so the widest is picked rather than whichever
    the batch happens to hold last.
    """
    wide = _make_job(job_id="wide", cores=8)
    narrow = _make_job(job_id="narrow", cores=1)

    assert wide.core_weight != narrow.core_weight

    response = _stage_batch(jobs=[wide, narrow], core_budget=32)

    assert response["job_allocations"][CAMERA_EXTRACTION_JOB_NAME]["cores_per_job"] == 8
    assert wide.core_weight == 8
    assert narrow.core_weight == 1


def test_a_job_wider_than_the_host_is_capped_while_a_narrow_one_is_left_alone(staged_batch: None) -> None:
    """Verifies that a descriptor planned against a wider host would otherwise tell its pipeline to fan out past what
    this host supplies, and capping it never widens the job that already fits.
    """
    narrow = _make_job(job_id="narrow", cores=1)
    wide = _make_job(job_id="wide", cores=8)

    assert narrow.core_weight != wide.core_weight

    response = _stage_batch(jobs=[narrow, wide], core_budget=4)

    assert wide.core_weight == 4
    assert narrow.core_weight == 1
    assert response["job_allocations"][CAMERA_EXTRACTION_JOB_NAME]["cores_per_job"] == 4
    # The pool is sized off the narrowest job, so the one-core job's capacity is not spent on workers it cannot use.
    assert response["pool_size"] == 2


def test_a_zero_width_job_is_floored_at_one_core_rather_than_widened_to_its_type(staged_batch: None) -> None:
    """Verifies that a non-positive width in the plan artifact would leave admission with no core term at all, so the
    dispatch floor of one core stands whatever the descriptor carries. The floor is the job's own and not its type's,
    which is what a zero-width job sharing a type with a wide one shows.
    """
    empty = _make_job(job_id="empty", cores=0)
    wide = _make_job(job_id="wide", cores=8)

    response = _stage_batch(jobs=[empty, wide], core_budget=8)

    assert empty.core_weight == 1
    assert wide.core_weight == 8
    # The type is still represented by its widest job, so the floor applied to the zero-width one leaves the reported
    # allocation alone.
    assert response["job_allocations"][CAMERA_EXTRACTION_JOB_NAME]["cores_per_job"] == 8
    # The pool follows the narrowest job that the batch holds, which is the floored one rather than the type's
    # reported width. A run that widened the zero-width job onto its type would spawn a single worker here.
    assert response["pool_size"] == 2
