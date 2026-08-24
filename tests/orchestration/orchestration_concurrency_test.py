"""Tests the shared batch engine: core allocation, two-dimensional admission, and the graph a batch is dispatched as."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from pathlib import Path
from collections import deque
from dataclasses import field, dataclass
from concurrent.futures import Future

import cv2
import numba
from cindra import MEMORY_ESTIMATE_TOLERANCE
import psutil
import pytest
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
)
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.orchestration import (
    SESSION_UNIT,
    BATCH_PIPELINES,
    JobExecutionState,
    prepare_batch,
    resolve_dispatch,
    build_pending_job,
    index_rows_by_unit,
    plan_artifact_path,
    build_batch_document,
    partition_blocked_jobs,
    resolve_host_memory_mb,
    resolve_core_allocations,
    resolve_submission_order,
    resolve_concurrency_limits,
    resolve_dispatch_priorities,
)
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.orchestration.local import (
    PendingJob,
    _admit_pending_jobs,
    _initialize_worker_threads,
)
from sollertia_forgery.orchestration.dispatch import _JOB_CORE_ALLOCATIONS
from sollertia_forgery.orchestration.footprints import (
    _MEGABYTES_PER_GIGABYTE,
    _CHECKSUM_READER_MEMORY_MB,
    _apply_tolerance,
    _size_checksum_job,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor that converts the byte figure the host reports into the megabytes the probe answers with."""

_CORE_BUDGET: int = 64
"""The core budget the admission tests weigh their jobs against, pinned so a retune is adopted deliberately."""

_MEMORY_BUDGET_MB: int = 65536
"""The memory budget the admission tests weigh their jobs against, pinned so a retune is adopted deliberately."""

_TRACKER: Path = Path("/nonexistent/tracker.yaml")
"""The tracker path the admission tests attach their jobs to."""

_OTHER_TRACKER: Path = Path("/nonexistent/other_tracker.yaml")
"""A second tracker path, used where a test must show that two units keep separate tracker state."""

_UNIT: Path = Path("/nonexistent/session")
"""The processing unit the admission tests place their jobs under, which is what scopes a job identifier."""

_OTHER_UNIT: Path = Path("/nonexistent/other_session")
"""A second processing unit, used where a test must show that two units' identical identifiers stay separate."""

_PREPARED_PROJECT_ROOT: Path = Path("/nonexistent/project")
"""The project root the preparation tests resolve their units against, which is where their artifacts are read from."""


@dataclass
class RecordingPool:
    """Stands in for a process pool, recording each submitted job instead of running it."""

    submitted: list[PendingJob] = field(default_factory=list)
    """Every job handed to the pool, in submission order."""

    def submit(self, worker: Callable[[PendingJob], None], job: PendingJob) -> Future[None]:  # noqa: ARG002
        """Records a submitted job and returns a future that never completes."""
        self.submitted.append(job)
        return Future()


def make_pending_job(
    job_id: str,
    job_name: str = "processing",
    cores: int = 1,
    memory_mb: int = 512,
    prerequisites: tuple[str, ...] = (),
    tracker_path: Path = _TRACKER,
    unit_path: Path = _UNIT,
) -> PendingJob:
    """Builds a pending job with explicit resource weights and prerequisite identifiers."""
    return PendingJob(
        tracker_path=tracker_path,
        job_id=job_id,
        unit_path=unit_path,
        job_name=job_name,
        core_weight=cores,
        memory_mb=memory_mb,
        prerequisite_ids=prerequisites,
    )


def build_state(
    jobs: list[PendingJob], core_budget: int = _CORE_BUDGET, memory_budget_mb: int = _MEMORY_BUDGET_MB
) -> JobExecutionState[PendingJob]:
    """Builds an execution state holding the supplied jobs against explicit budgets."""
    return JobExecutionState(
        worker=lambda _job: None,
        all_jobs={job.dispatch_key: job for job in jobs},
        pending_jobs=deque(jobs),
        core_budget=core_budget,
        memory_budget_mb=memory_budget_mb,
        pool_size=len(jobs) or 1,
    )


def run_admission_pass(state: JobExecutionState[PendingJob]) -> RecordingPool:
    """Runs one admission pass against a recording pool and returns it."""
    pool = RecordingPool()
    _admit_pending_jobs(state=state, pool=pool)  # type: ignore[arg-type]
    return pool


def complete_job(state: JobExecutionState[PendingJob], job: PendingJob) -> None:
    """Marks a running job finished and successful, as the manager would after reaping its future."""
    state.active_jobs = [active for active in state.active_jobs if active.job.dispatch_key != job.dispatch_key]
    state.succeeded_job_keys.add(job.dispatch_key)


def test_core_allocations_come_from_the_table_and_clamp_to_the_budget() -> None:
    """Verifies that each job type takes its declared cores, narrowed to a budget that cannot supply them."""
    names = set(_JOB_CORE_ALLOCATIONS)
    generous = resolve_core_allocations(job_cores=_JOB_CORE_ALLOCATIONS, job_names=names, core_budget=128)
    for job_name, allocation in generous.items():
        assert allocation.cores_per_job == _JOB_CORE_ALLOCATIONS[job_name]
        assert allocation.maximum_parallel >= 1

    cramped = resolve_core_allocations(job_cores=_JOB_CORE_ALLOCATIONS, job_names=names, core_budget=2)
    for allocation in cramped.values():
        assert 1 <= allocation.cores_per_job <= 2


def test_unregistered_job_type_stops_the_batch() -> None:
    """Verifies that a job type with no registered core allocation raises instead of running at a default width."""
    with pytest.raises(ValueError, match="an_unregistered_job"):
        resolve_core_allocations(job_cores={}, job_names={"an_unregistered_job"}, core_budget=_CORE_BUDGET)


def test_a_partial_allocation_table_stops_the_batch() -> None:
    """Verifies that one unregistered type among registered ones is still caught."""
    with pytest.raises(ValueError, match="an_unregistered_job"):
        resolve_core_allocations(
            job_cores=_JOB_CORE_ALLOCATIONS,
            job_names={*_JOB_CORE_ALLOCATIONS, "an_unregistered_job"},
            core_budget=_CORE_BUDGET,
        )


def test_admission_stops_on_whichever_budget_binds_first() -> None:
    """Verifies that cores and memory are both enforced and that the scarcer one limits concurrency."""
    core_bound = build_state(
        jobs=[make_pending_job(job_id=f"c{index}", cores=16, memory_mb=1024) for index in range(8)],
        core_budget=_CORE_BUDGET,
        memory_budget_mb=_MEMORY_BUDGET_MB,
    )
    assert len(run_admission_pass(core_bound).submitted) == 4

    memory_bound = build_state(
        jobs=[make_pending_job(job_id=f"m{index}", cores=1, memory_mb=20000) for index in range(8)],
        core_budget=_CORE_BUDGET,
        memory_budget_mb=_MEMORY_BUDGET_MB,
    )
    assert len(run_admission_pass(memory_bound).submitted) == 3


def test_admission_never_commits_past_either_budget() -> None:
    """Verifies that the running set stays inside both budgets for a heterogeneous queue."""
    sizes = [50000, 20000, 20000, 6000, 6000, 6000, 6000, 6000]
    state = build_state(
        jobs=[make_pending_job(job_id=f"j{index}", cores=4, memory_mb=size) for index, size in enumerate(sizes)],
        core_budget=_CORE_BUDGET,
        memory_budget_mb=_MEMORY_BUDGET_MB,
    )
    pool = run_admission_pass(state)
    assert sum(job.core_weight for job in pool.submitted) <= _CORE_BUDGET
    assert sum(job.memory_mb for job in pool.submitted) <= _MEMORY_BUDGET_MB


def test_admission_considers_the_heaviest_job_first() -> None:
    """Verifies that a large job is dispatched before lighter ones rather than waiting for them to drain."""
    jobs = [make_pending_job(job_id=f"light{index}", memory_mb=1024) for index in range(6)]
    jobs.append(make_pending_job(job_id="heavy", memory_mb=40000))
    pool = run_admission_pass(build_state(jobs=jobs))
    assert pool.submitted[0].job_id == "heavy"


def test_light_jobs_backfill_the_capacity_a_heavy_job_leaves_spare() -> None:
    """Verifies that a heavy job runs alongside as many light jobs as the remaining budget allows."""
    jobs = [make_pending_job(job_id="heavy", cores=8, memory_mb=40000)]
    jobs += [make_pending_job(job_id=f"light{index}", cores=2, memory_mb=4000) for index in range(10)]
    pool = run_admission_pass(build_state(jobs=jobs, core_budget=_CORE_BUDGET, memory_budget_mb=_MEMORY_BUDGET_MB))
    submitted = {job.job_id for job in pool.submitted}
    assert "heavy" in submitted
    assert len(submitted) >= 6, "light jobs did not backfill around the heavy job"
    assert sum(job.memory_mb for job in pool.submitted) <= _MEMORY_BUDGET_MB


def test_freed_capacity_is_repacked_on_the_next_pass() -> None:
    """Verifies that finishing a job immediately releases its share to the jobs still queued."""
    jobs = [make_pending_job(job_id=f"j{index}", cores=1, memory_mb=20000) for index in range(6)]
    state = build_state(jobs=jobs, core_budget=_CORE_BUDGET, memory_budget_mb=_MEMORY_BUDGET_MB)

    first = run_admission_pass(state)
    assert len(first.submitted) == 3
    assert len(state.pending_jobs) == 3

    complete_job(state=state, job=first.submitted[0])
    second = run_admission_pass(state)
    assert len(second.submitted) == 1, "capacity freed by a finished job was not refilled"


def test_admission_withholds_a_job_until_its_prerequisite_succeeds() -> None:
    """Verifies that a dependent job is not dispatched while its upstream job has not yet succeeded."""
    upstream = make_pending_job(job_id="up", job_name="binarization")
    downstream = make_pending_job(job_id="down", prerequisites=("up",))
    state = build_state(jobs=[upstream, downstream])

    pool = run_admission_pass(state)
    assert [job.job_id for job in pool.submitted] == ["up"]

    complete_job(state=state, job=upstream)
    assert [job.job_id for job in run_admission_pass(state).submitted] == ["down"]


def test_a_completed_stage_in_one_session_does_not_satisfy_another_session() -> None:
    """Verifies that identical job identifiers in different sessions are tracked separately.

    A job identifier is derived from the job name and specifier alone, so every session's binarization shares one
    identifier. A batch spanning sessions must not treat one session's completed stage as every session's.
    """
    first_upstream = make_pending_job(job_id="shared", job_name="binarization", tracker_path=_TRACKER, unit_path=_UNIT)
    second_downstream = make_pending_job(
        job_id="down", prerequisites=("shared",), tracker_path=_OTHER_TRACKER, unit_path=_OTHER_UNIT
    )
    state = build_state(jobs=[first_upstream, second_downstream])

    pool = run_admission_pass(state)
    assert [job.job_id for job in pool.submitted] == ["shared"]

    complete_job(state=state, job=first_upstream)
    assert not run_admission_pass(state).submitted, (
        "another session's dependent was released by an unrelated session's job"
    )
    assert [job.job_id for job in state.pending_jobs] == ["down"]


def test_admission_blocks_a_dependent_whose_prerequisite_failed() -> None:
    """Verifies that a job whose upstream failed is moved aside rather than waiting for an outcome that cannot come."""
    downstream = make_pending_job(job_id="down", prerequisites=("up",))
    state = build_state(jobs=[downstream])
    state.failed_job_keys.add((str(_UNIT), "up"))

    assert not run_admission_pass(state).submitted
    assert [job.job_id for job in state.blocked_jobs] == ["down"]
    assert not state.pending_jobs


def test_forward_progress_floor_never_bypasses_the_prerequisite_check() -> None:
    """Verifies that the single-job floor does not dispatch a job whose input does not exist yet."""
    downstream = make_pending_job(job_id="down", memory_mb=10_000_000, prerequisites=("up",))
    assert not run_admission_pass(build_state(jobs=[downstream])).submitted


def test_forward_progress_floor_admits_a_job_larger_than_the_budget() -> None:
    """Verifies that a job no budget can hold still runs alone rather than stalling the batch."""
    oversized = make_pending_job(job_id="huge", cores=999, memory_mb=10_000_000)
    assert [job.job_id for job in run_admission_pass(build_state(jobs=[oversized])).submitted] == ["huge"]


def test_estimates_carry_the_shared_tolerance() -> None:
    """Verifies that a reported estimate carries the shared margin and lands on the first whole gigabyte above it."""
    margin = int(1000 * MEMORY_ESTIMATE_TOLERANCE) + 1
    reportable = _apply_tolerance(memory_mb=1000)

    # The reported figure is the smallest whole gigabyte the margin fits inside, so the margin clears the gigabyte
    # below it and does not clear the figure itself. Bounding it from both sides is what separates this rounding from
    # any larger figure that would also cover the margin.
    assert reportable % _MEGABYTES_PER_GIGABYTE == 0
    assert reportable - _MEGABYTES_PER_GIGABYTE < margin <= reportable

    # The modeled figure sits below one gigabyte, so only a margin of real magnitude carries it onto the second one.
    # Pinning the figure it reaches is what fails if the tolerance ever collapses toward one, which the bounds above
    # would still accept because the rounding adds an increment of its own.
    assert reportable == 2 * _MEGABYTES_PER_GIGABYTE

    # A modeled figure sitting exactly on a gigabyte shows the margin is applied before the rounding rather than
    # absorbed by it, since the tolerance carries it off its own quantum and onto the next one.
    assert _apply_tolerance(memory_mb=_MEGABYTES_PER_GIGABYTE) == 2 * _MEGABYTES_PER_GIGABYTE

    # A modeled figure of zero still reports a whole gigabyte, because the margin's own increment lands above nothing.
    assert _apply_tolerance(memory_mb=0) == _MEGABYTES_PER_GIGABYTE


def test_host_memory_is_readable_and_positive() -> None:
    """Verifies that the host memory probe reports a figure inside the host's physical memory."""
    reported = resolve_host_memory_mb()
    assert 0 < reported <= psutil.virtual_memory().total // _BYTES_PER_MEGABYTE


def test_every_dispatched_job_type_declares_a_core_allocation() -> None:
    """Verifies that the allocation table covers every job type with a positive core count."""
    assert _JOB_CORE_ALLOCATIONS
    for job_name, cores in _JOB_CORE_ALLOCATIONS.items():
        assert cores >= 1, f"{job_name} declares a non-positive core count"


def test_checksum_is_a_registered_batch_pipeline() -> None:
    """Verifies that the checksum pipeline is dispatchable and declares the quartet the batch tools drive it with."""
    assert ProcessingPipelines.CHECKSUM in BATCH_PIPELINES

    dispatch = resolve_dispatch(pipeline="checksum")
    assert dispatch is not None
    assert dispatch.pipeline is ProcessingPipelines.CHECKSUM
    assert CHECKSUM_JOB_NAME in _JOB_CORE_ALLOCATIONS

    # The pipeline resolves one job per session with no upstream stage, so every job maps to an empty ordering.
    universe = [(CHECKSUM_JOB_NAME, "a_session")]
    assert dispatch.prerequisites(None, universe) == {(CHECKSUM_JOB_NAME, "a_session"): ()}


def test_job_options_round_trip_from_descriptor_to_worker() -> None:
    """Verifies that pipeline-specific parameters survive the descriptor hop into the dispatched job."""
    descriptor = {
        "tracker_path": str(_TRACKER),
        "job_id": "a_job",
        "unit_path": "/nonexistent/session",
        "job_name": CHECKSUM_JOB_NAME,
        "pipeline": ProcessingPipelines.CHECKSUM.value,
        "cores": 8,
        "memory_mb": 1024,
        "options": {"regenerate_checksum": True},
    }
    assert build_pending_job(job=descriptor).options == {"regenerate_checksum": True}

    # A descriptor that names no options yields an empty mapping rather than None, so a worker reads it unguarded.
    del descriptor["options"]
    assert build_pending_job(job=descriptor).options == {}


def test_checksum_memory_is_flat_in_input_size_and_linear_in_cores() -> None:
    """Verifies that the checksum estimate tracks the cores a job holds rather than the bytes it reads.

    Every other estimator scales a per-byte ratio off an input file. A checksum worker streams its file in fixed
    chunks, so the session's size does not enter the estimate and only the reader count does.
    """
    single = _size_checksum_job(cores=1).memory_mb
    # Reportable figures land on whole gigabytes, so the per-reader growth shows across a wide core spread rather
    # than between two adjacent core counts, where the rounding absorbs it.
    many = _size_checksum_job(cores=16).memory_mb
    assert many - single >= 15 * _CHECKSUM_READER_MEMORY_MB
    assert _size_checksum_job(cores=8).memory_mb > single
    assert _size_checksum_job(cores=2).memory_mb >= single
    # The sizing pass answers both halves, so the width the job is dispatched at comes back beside its memory.
    assert _size_checksum_job(cores=16).cores == 16


def test_worker_initializer_leaves_the_numba_thread_variable_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker initializer controls numba through its runtime setter rather than its environment.

    numba reads NUMBA_NUM_THREADS once at import and compares the variable against that latched count on every
    compilation, rejecting a disagreement once its thread pool has started. A worker imports numba before the
    initializer runs, so pinning the variable there would fail every job that compiles a numba function.
    """
    for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "POLARS_MAX_THREADS"):
        monkeypatch.setenv(variable, os.environ.get(variable, ""))
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    numba_threads = numba.get_num_threads()
    opencv_threads = cv2.getNumThreads()

    _initialize_worker_threads(thread_ceiling=1)
    numba.set_num_threads(n=numba_threads)
    cv2.setNumThreads(opencv_threads)

    assert "NUMBA_NUM_THREADS" not in os.environ

    # The other threading layers stay pinned, since they read their variables when the job itself starts.
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["POLARS_MAX_THREADS"] == "1"


@pytest.mark.parametrize("pipeline", sorted(member.value for member in BATCH_PIPELINES))
def test_every_dispatch_entry_declares_the_whole_generic_contract(pipeline: str) -> None:
    """Verifies that each registered pipeline supplies every callable the unit-generic dispatch contract requires.

    The batch layer reads a unit only through these callables, so an entry omitting one fails at preparation rather
    than at registration.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    assert dispatch is not None

    for field_name in (
        "discover",
        "worker",
        "prerequisites",
        "tracker_path",
        "output_path",
        "unit_name",
        "size_jobs",
    ):
        assert callable(getattr(dispatch, field_name)), f"{pipeline} declares no {field_name}"


def test_forging_is_a_registered_batch_pipeline() -> None:
    """Verifies that the dataset-scoped pipeline is dispatchable and declares a core allocation for every stage."""
    assert ProcessingPipelines.FORGING in BATCH_PIPELINES

    dispatch = resolve_dispatch(pipeline="forging")
    assert dispatch is not None
    assert dispatch.pipeline is ProcessingPipelines.FORGING

    for job_name in (MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME, FORGING_JOB_NAME):
        assert job_name in _JOB_CORE_ALLOCATIONS, f"{job_name} declares no core allocation"


def test_only_assembly_carries_a_forging_concurrency_limit() -> None:
    """Verifies that the storage-bound forging stage is capped while the compute-bound stages are budget-bound.

    cindra treats its own cross-recording discovery and extraction as compute-bound and runs them at a wide core
    allocation, so a ceiling on top of the core budget would hold them below the concurrency they gain from.
    """
    limits = resolve_concurrency_limits(
        job_names={MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME, FORGING_JOB_NAME}
    )
    assert set(limits) == {FORGING_JOB_NAME}
    assert _JOB_CORE_ALLOCATIONS[MULTIDAY_DISCOVERY_JOB_NAME] > _JOB_CORE_ALLOCATIONS[FORGING_JOB_NAME]
    assert _JOB_CORE_ALLOCATIONS[MULTIDAY_EXTRACTION_JOB_NAME] > _JOB_CORE_ALLOCATIONS[FORGING_JOB_NAME]


# Batch document assembly

_PIPELINE: str = ProcessingPipelines.CHECKSUM.value
"""The pipeline every batch-document test prepares."""


def identifier(job_name: str, specifier: str = "") -> str:
    """Returns the identifier the processing tracker records the named job under."""
    return ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)


def plan_row(
    unit: str,
    job_name: str,
    specifier: str = "",
    *,
    cores: int = 4,
    memory_mb: int = 2048,
    prerequisites: tuple[tuple[str, str], ...] = (),
    pipeline: str = _PIPELINE,
    unit_column: str = "session",
) -> dict[str, Any]:
    """Renders one row of the project plan projection, carrying a job's figures and its recorded ordering."""
    return {
        unit_column: unit,
        "pipeline": pipeline,
        "job_id": identifier(job_name=job_name, specifier=specifier),
        "job_name": job_name,
        "specifier": specifier,
        "cores": cores,
        "memory_mb": memory_mb,
        "prerequisite_ids": [identifier(job_name=name, specifier=upstream) for name, upstream in prerequisites],
    }


def state_row(
    unit: str,
    job_name: str,
    specifier: str = "",
    *,
    status: str = "SCHEDULED",
    pipeline: str | None = _PIPELINE,
    unit_column: str = "session",
) -> dict[str, Any]:
    """Renders one row of the project state projection, carrying a tracked job's recorded status."""
    row: dict[str, Any] = {
        unit_column: unit,
        "job_id": identifier(job_name=job_name, specifier=specifier),
        "job_name": job_name,
        "specifier": specifier,
        "status": status,
    }
    if pipeline is not None:
        row["pipeline"] = pipeline
    return row


def descriptor(job_id: str, prerequisites: tuple[str, ...] = ()) -> dict[str, Any]:
    """Builds the minimal job descriptor the blocking partition reads."""
    return {
        "job_id": job_id,
        "job_name": "processing",
        "specifier": job_id,
        "pipeline": _PIPELINE,
        "unit_path": str(_UNIT),
        "unit_name": _UNIT.name,
        "prerequisite_ids": list(prerequisites),
    }


def test_a_batch_document_dispatches_only_the_outstanding_planned_jobs() -> None:
    """Verifies that preparing a unit twice queues the outstanding work, leaving a succeeded job out of the batch."""
    unit = Path("/nonexistent/project/305/a_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[
            plan_row(unit=unit.name, job_name="hash", specifier="a"),
            plan_row(unit=unit.name, job_name="hash", specifier="b", cores=8, memory_mb=4096),
            # A row of another pipeline never reaches this batch, whichever unit it names.
            plan_row(unit=unit.name, job_name="hash", specifier="c", pipeline="runtime"),
        ],
        state_rows=[
            state_row(unit=unit.name, job_name="hash", specifier="a", status=ProcessingStatus.SUCCEEDED.name),
            state_row(unit=unit.name, job_name="hash", specifier="b"),
            state_row(unit=unit.name, job_name="hash", specifier="c", pipeline="runtime"),
        ],
        unit_paths=[unit],
        options={"regenerate_checksum": True},
        tracker_paths={str(unit): "/nonexistent/project/305/a_session/tracker.yaml"},
    )

    assert document.pipeline == _PIPELINE
    assert document.host == "workstation"
    assert document.options == {"regenerate_checksum": True}
    assert [job["job_id"] for job in document.jobs] == [identifier(job_name="hash", specifier="b")]
    assert document.jobs[0] == {
        "job_id": identifier(job_name="hash", specifier="b"),
        "job_name": "hash",
        "specifier": "b",
        "unit_path": str(unit),
        "unit_name": unit.name,
        "pipeline": _PIPELINE,
        "tracker_path": "/nonexistent/project/305/a_session/tracker.yaml",
        "cores": 8,
        "memory_mb": 4096,
        "prerequisite_ids": [],
        "options": {"regenerate_checksum": True},
        # Reconciliation reads the recorded outcome off the descriptor, so preparation carries it across rather than
        # leaving a later stage to reopen the tracker the state artifact was regenerated from.
        "status": "SCHEDULED",
        "executor_id": "",
    }
    assert document.units == [{"unit_path": str(unit), "unit_name": unit.name, "job_count": 1, "blocked_count": 0}]
    assert document.blocked_jobs == []


def test_a_job_whose_upstream_stage_already_succeeded_is_dispatched_rather_than_blocked() -> None:
    """Verifies that a stage succeeding in an earlier run satisfies its dependents when the unit is prepared again.

    That is the ordinary resume path: the succeeded stage is left out of the batch while the job waiting on it is
    dispatched. Reading the recorded successes as empty would report every dependent blocked and stall the unit.
    """
    unit = Path("/nonexistent/project/305/a_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[
            plan_row(unit=unit.name, job_name="hash", specifier="a"),
            plan_row(unit=unit.name, job_name="hash", specifier="b", prerequisites=(("hash", "a"),)),
        ],
        state_rows=[
            state_row(unit=unit.name, job_name="hash", specifier="a", status=ProcessingStatus.SUCCEEDED.name),
            state_row(unit=unit.name, job_name="hash", specifier="b"),
        ],
        unit_paths=[unit],
        options={},
    )

    assert [job["job_id"] for job in document.jobs] == [identifier(job_name="hash", specifier="b")]
    assert document.blocked_jobs == []


def test_a_state_row_naming_no_pipeline_is_read_as_the_prepared_ones() -> None:
    """Verifies that a state table recording no pipeline column belongs wholly to the pipeline being prepared."""
    unit = Path("/nonexistent/project/305/a_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[plan_row(unit=unit.name, job_name="hash", specifier="a")],
        state_rows=[state_row(unit=unit.name, job_name="hash", specifier="a", pipeline=None)],
        unit_paths=[unit],
        options={},
    )

    # No tracker location was supplied, which is what a backend recording its outcomes elsewhere passes.
    assert document.jobs[0]["tracker_path"] == ""
    assert document.jobs[0]["options"] == {}


def test_a_recorded_edge_naming_an_untracked_stage_is_dropped() -> None:
    """Verifies that a prerequisite the unit can never produce is dropped rather than left waiting on it.

    The planned figures cover the whole universe of a pipeline's jobs while a tracker registers only the subset the
    unit can actually produce, so a stage that is planned but untracked is exactly the shape this drop exists for.
    """
    unit = Path("/nonexistent/project/305/a_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[
            plan_row(unit=unit.name, job_name="hash", specifier="a"),
            plan_row(
                unit=unit.name,
                job_name="hash",
                specifier="b",
                prerequisites=(("hash", "a"), ("hash", "never_possible")),
            ),
            plan_row(unit=unit.name, job_name="hash", specifier="never_possible"),
        ],
        state_rows=[
            state_row(unit=unit.name, job_name="hash", specifier="a"),
            state_row(unit=unit.name, job_name="hash", specifier="b"),
        ],
        unit_paths=[unit],
        options={},
    )

    # Keeping the edge would leave the downstream stage waiting on an outcome that can never be recorded, which the
    # batch reports as blocked rather than dispatching.
    assert document.blocked_jobs == []
    downstream = next(job for job in document.jobs if job["specifier"] == "b")
    assert downstream["prerequisite_ids"] == [identifier(job_name="hash", specifier="a")]


def test_a_unit_the_state_table_records_nothing_for_contributes_no_job() -> None:
    """Verifies that a unit carrying none of the pipeline's data is reported with the reason it contributes none."""
    known = Path("/nonexistent/project/305/a_session")
    unknown = Path("/nonexistent/project/321/another_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[plan_row(unit=known.name, job_name="hash", specifier="a")],
        state_rows=[state_row(unit=known.name, job_name="hash", specifier="a")],
        unit_paths=[known, unknown],
        options={},
    )

    unresolved = next(entry for entry in document.units if entry["unit_name"] == unknown.name)
    assert unresolved["job_count"] == 0
    assert unresolved["error"] == (
        f"The project's state table records no '{_PIPELINE}' job for this unit, so the unit carries none of the data "
        f"that pipeline consumes."
    )


def test_a_unit_whose_outstanding_job_carries_no_figures_is_rejected() -> None:
    """Verifies that an unplanned outstanding job stops its whole unit, since every job is sized before it runs."""
    unit = Path("/nonexistent/project/305/a_session")
    document = build_batch_document(
        pipeline=_PIPELINE,
        host="workstation",
        unit_column="session",
        plan_rows=[plan_row(unit=unit.name, job_name="hash", specifier="a")],
        state_rows=[
            state_row(unit=unit.name, job_name="hash", specifier="a"),
            state_row(unit=unit.name, job_name="hash", specifier="unplanned"),
            # An unplanned job that already succeeded is tolerated, since this run would never dispatch it.
            state_row(unit=unit.name, job_name="hash", specifier="retired", status=ProcessingStatus.SUCCEEDED.name),
        ],
        unit_paths=[unit],
        options={},
    )

    assert document.jobs == []
    assert document.units[0]["error"] == (
        f"The project's plan table carries no figures for job(s) "
        f"{[identifier(job_name='hash', specifier='unplanned')]}. Every outstanding job must be planned before it "
        f"can be sized for a host or a scheduler."
    )


def test_rows_naming_no_unit_are_left_out_of_the_index() -> None:
    """Verifies that a row whose unit column is empty belongs to no unit, so it indexes under none of them."""
    indexed = index_rows_by_unit(
        rows=[
            {"session": "a_session", "job_id": "first"},
            {"session": None, "job_id": "second"},
            {"dataset": "a_dataset", "job_id": "third"},
        ],
        key="session",
    )

    assert set(indexed) == {"a_session"}
    assert indexed["a_session"]["first"] == {"session": "a_session", "job_id": "first"}


def test_blocking_propagates_along_the_chain_that_waits_on_it() -> None:
    """Verifies that a stage waiting on a blocked stage is blocked in turn, so the whole chain is set aside."""
    jobs = [
        descriptor(job_id="root", prerequisites=("absent",)),
        descriptor(job_id="middle", prerequisites=("root",)),
        descriptor(job_id="leaf", prerequisites=("middle",)),
        descriptor(job_id="independent"),
    ]

    dispatchable, blocked = partition_blocked_jobs(jobs=jobs, succeeded=set())

    assert [job["job_id"] for job in dispatchable] == ["independent"]
    assert [entry["job_id"] for entry in blocked] == ["root", "middle", "leaf"]
    assert blocked[0]["unsatisfied_prerequisite_ids"] == ["absent"]
    assert blocked[2]["unsatisfied_prerequisite_ids"] == ["middle"]
    assert blocked[1]["unit_name"] == _UNIT.name


def test_an_already_succeeded_prerequisite_satisfies_its_dependent() -> None:
    """Verifies that a prerequisite recorded as succeeded blocks nothing, even where this run never dispatches it."""
    jobs = [descriptor(job_id="downstream", prerequisites=("upstream",))]

    dispatchable, blocked = partition_blocked_jobs(jobs=jobs, succeeded={"upstream"})

    assert [job["job_id"] for job in dispatchable] == ["downstream"]
    assert blocked == []


# Batch preparation


class _StubPreparationHost:
    """Stands in for an execution host holding one project's tables and its units' trackers.

    Args:
        plan_rows: The rows the project's plan table holds.
        state_rows: The rows the project's state table holds.

    Attributes:
        _plan_rows: The rows the project's plan table holds.
        _state_rows: The rows the project's state table holds.
        materialized: The project root, unit kind, and replan choice of every materialization this host was asked for.
    """

    def __init__(self, plan_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]]) -> None:
        self._plan_rows: list[dict[str, Any]] = plan_rows
        self._state_rows: list[dict[str, Any]] = state_rows
        self.materialized: list[tuple[Path, str, bool]] = []

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "workstation"

    def materialize(
        self,
        project_root: Path,
        unit_paths: Sequence[Path],  # noqa: ARG002
        unit_kind: str,
        *,
        replan: bool,
    ) -> None:
        """Records what the batch asked to be rewritten, since this stub's tables already hold their rows.

        Args:
            project_root: The project the artifacts are rewritten for.
            unit_paths: The processing units the artifacts are rewritten for.
            unit_kind: The kind of processing unit the paths name.
            replan: Determines whether the recorded figures are re-estimated.
        """
        self.materialized.append((project_root, unit_kind, replan))

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Answers the plan table for the project's plan artifact and the state table for every other path.

        Args:
            path: The artifact the rows are read from.

        Returns:
            The rows that artifact holds.
        """
        return self._plan_rows if path == plan_artifact_path(project_root=_PREPARED_PROJECT_ROOT) else self._state_rows

    @staticmethod
    def resolve_tracker_paths(pipeline: str, unit_paths: Sequence[Path]) -> dict[str, str]:
        """Resolves one tracker per unit, the way a host whose engine opens those files directly does.

        Args:
            pipeline: The pipeline whose tracker is located.
            unit_paths: The unit root directories to locate trackers for.

        Returns:
            The tracker path of each unit, keyed by the unit path as a string.
        """
        return {str(unit_path): str(unit_path.joinpath(f"{pipeline}_tracker.yaml")) for unit_path in unit_paths}


def test_a_prepared_batch_carries_the_tracker_location_the_host_resolved_for_each_unit() -> None:
    """Verifies that preparation stamps each unit's own tracker location onto the descriptors it hands the engine.

    The local engine opens those files directly to seed and extend its recorded outcomes, so descriptors carrying no
    location would leave every prerequisite unsatisfied and a multi-stage batch would report its later stages blocked.
    """
    unit = _PREPARED_PROJECT_ROOT.joinpath("305", "a_session")
    host = _StubPreparationHost(
        plan_rows=[plan_row(unit=unit.name, job_name="hash", specifier="a")],
        state_rows=[state_row(unit=unit.name, job_name="hash", specifier="a")],
    )

    document = prepare_batch(
        host=host,  # type: ignore[arg-type]
        pipeline=_PIPELINE,
        unit_paths=[str(unit)],
        options={"regenerate_checksum": True},
        replan=True,
    )

    assert document.jobs[0]["tracker_path"] == str(unit.joinpath(f"{_PIPELINE}_tracker.yaml"))
    assert document.jobs[0]["options"] == {"regenerate_checksum": True}
    assert document.host == "workstation"
    # A session pipeline materializes its whole project, and the caller's own replan choice reaches the host.
    assert host.materialized == [(_PREPARED_PROJECT_ROOT, SESSION_UNIT, True)]


# Dispatch ordering


def test_priority_weighs_a_job_by_the_cores_its_dependents_commit() -> None:
    """Verifies that ordering follows the queued work a job holds back, so a root outweighs the leaves it releases."""
    root = make_pending_job(job_id="root", cores=2)
    first = make_pending_job(job_id="first", cores=8, prerequisites=("root",))
    second = make_pending_job(job_id="second", cores=4, prerequisites=("root",))
    # Reachable along both middle stages at once, so it is counted a single time.
    leaf = make_pending_job(job_id="leaf", cores=16, prerequisites=("first", "second"))
    jobs = {job.dispatch_key: job for job in (root, first, second, leaf)}

    priorities = resolve_dispatch_priorities(jobs=jobs)

    assert priorities[root.dispatch_key] == 8 + 4 + 16
    assert priorities[first.dispatch_key] == 16
    assert priorities[leaf.dispatch_key] == 0


def test_a_prerequisite_outside_the_batch_contributes_no_priority() -> None:
    """Verifies that a job the batch does not hold cannot be ordered against the ones it does, so it weighs nothing."""
    inside = make_pending_job(job_id="inside", cores=4, prerequisites=("outside_the_batch",))
    other_unit = make_pending_job(job_id="root", cores=4, unit_path=_OTHER_UNIT)
    jobs = {job.dispatch_key: job for job in (inside, other_unit)}

    priorities = resolve_dispatch_priorities(jobs=jobs)

    assert priorities == {inside.dispatch_key: 0, other_unit.dispatch_key: 0}


def test_a_cyclic_ordering_resolves_to_a_finite_priority() -> None:
    """Verifies that a malformed pipeline ordering is weighed poorly rather than recursing without end."""
    first = make_pending_job(job_id="first", cores=3, prerequisites=("second",))
    second = make_pending_job(job_id="second", cores=5, prerequisites=("first",))
    jobs = {job.dispatch_key: job for job in (first, second)}

    priorities = resolve_dispatch_priorities(jobs=jobs)

    # The first key resolved reaches the whole cycle, and the one it memoized on the way stops at the truncation.
    assert priorities[first.dispatch_key] == 3 + 5
    assert priorities[second.dispatch_key] == 3


def test_submission_orders_every_job_behind_the_jobs_it_waits_on() -> None:
    """Verifies that a dependency names the upstream allocation's identifier, so upstream is submitted first."""
    leaf = make_pending_job(job_id="leaf", prerequisites=("first", "second"))
    first = make_pending_job(job_id="first", prerequisites=("root",))
    second = make_pending_job(job_id="second", prerequisites=("root",))
    root = make_pending_job(job_id="root")

    ordered = resolve_submission_order(jobs=[leaf, first, second, root])

    assert [job.job_id for job in ordered] == ["root", "first", "second", "leaf"]


def test_submission_ignores_a_prerequisite_outside_the_batch() -> None:
    """Verifies that a prerequisite the batch does not hold contributes no depth, so its dependent keeps its place."""
    outside = make_pending_job(job_id="outside", prerequisites=("never_submitted",))
    other_unit = make_pending_job(job_id="root", unit_path=_OTHER_UNIT)
    dependent = make_pending_job(job_id="dependent", prerequisites=("root",))

    ordered = resolve_submission_order(jobs=[outside, other_unit, dependent])

    # The dependent names its own unit's root, which this batch holds under another unit, so it sits at depth zero.
    assert [job.job_id for job in ordered] == ["outside", "root", "dependent"]


def test_a_cyclic_ordering_resolves_to_a_finite_submission_depth() -> None:
    """Verifies that a cycle is ordered poorly rather than stalling the submission it belongs to."""
    first = make_pending_job(job_id="first", prerequisites=("second",))
    second = make_pending_job(job_id="second", prerequisites=("first",))

    ordered = resolve_submission_order(jobs=[first, second])

    assert [job.job_id for job in ordered] == ["second", "first"]
