"""Tests the shared batch engine: core allocation, two-dimensional admission, and dependency ordering.

The engine's behavior is a scheduling policy expressed in arithmetic, so these tests pin it against explicit budgets
rather than against any host. A retune is expected to update the pinned values deliberately.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
from concurrent.futures import Future

import pytest

from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
)
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.orchestration import (
    BATCH_PIPELINES,
    JobExecutionState,
    resolve_dispatch,
    build_pending_job,
    resolve_host_memory_mb,
    resolve_core_allocations,
    resolve_concurrency_limits,
)
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.orchestration.local import (
    _PINNED_THREAD_VARIABLES,
    PendingJob,
    _admit_pending_jobs,
)
from sollertia_forgery.orchestration.dispatch import _JOB_CORE_ALLOCATIONS
from sollertia_forgery.orchestration.footprints import (
    _CHECKSUM_READER_MEMORY_MB,
    _MEMORY_ESTIMATE_TOLERANCE,
    _apply_tolerance,
    _estimate_checksum_memory,
)

CORE_BUDGET = 64
"""The core budget the admission tests weigh their jobs against."""

MEMORY_BUDGET_MB = 65536
"""The memory budget the admission tests weigh their jobs against."""

TRACKER = Path("/nonexistent/tracker.yaml")
OTHER_TRACKER = Path("/nonexistent/other_tracker.yaml")

UNIT = Path("/nonexistent/session")
"""The processing unit the admission tests place their jobs under, which is what scopes a job identifier."""

OTHER_UNIT = Path("/nonexistent/other_session")
"""A second processing unit, used where a test must show that two units' identical identifiers stay separate."""


class RecordingPool:
    """Stands in for a process pool, recording each submitted job instead of running it."""

    def __init__(self) -> None:
        """Initializes the recorded submission list."""
        self.submitted: list[PendingJob] = []

    def submit(self, worker: object, job: PendingJob) -> Future[None]:  # noqa: ARG002
        """Records a submitted job and returns a future that never completes."""
        self.submitted.append(job)
        return Future()


def make_job(
    job_id: str,
    job_name: str = "processing",
    cores: int = 1,
    memory_mb: int = 512,
    prerequisites: tuple[str, ...] = (),
    tracker_path: Path = TRACKER,
    unit_path: Path = UNIT,
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
    jobs: list[PendingJob], core_budget: int = CORE_BUDGET, memory_budget_mb: int = MEMORY_BUDGET_MB
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


def admit(state: JobExecutionState[PendingJob]) -> RecordingPool:
    """Runs one admission pass against a recording pool and returns it."""
    pool = RecordingPool()
    _admit_pending_jobs(state=state, pool=pool)  # type: ignore[arg-type]
    return pool


def complete(state: JobExecutionState[PendingJob], job: PendingJob) -> None:
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
        resolve_core_allocations(job_cores={}, job_names={"an_unregistered_job"}, core_budget=CORE_BUDGET)


def test_a_partial_allocation_table_stops_the_batch() -> None:
    """Verifies that one unregistered type among registered ones is still caught."""
    with pytest.raises(ValueError, match="an_unregistered_job"):
        resolve_core_allocations(
            job_cores=_JOB_CORE_ALLOCATIONS,
            job_names={*_JOB_CORE_ALLOCATIONS, "an_unregistered_job"},
            core_budget=CORE_BUDGET,
        )


def test_admission_stops_on_whichever_budget_binds_first() -> None:
    """Verifies that cores and memory are both enforced and that the scarcer one limits concurrency."""
    core_bound = build_state(
        jobs=[make_job(job_id=f"c{index}", cores=16, memory_mb=1024) for index in range(8)],
        core_budget=CORE_BUDGET,
        memory_budget_mb=MEMORY_BUDGET_MB,
    )
    assert len(admit(core_bound).submitted) == 4

    memory_bound = build_state(
        jobs=[make_job(job_id=f"m{index}", cores=1, memory_mb=20000) for index in range(8)],
        core_budget=CORE_BUDGET,
        memory_budget_mb=MEMORY_BUDGET_MB,
    )
    assert len(admit(memory_bound).submitted) == 3


def test_admission_never_commits_past_either_budget() -> None:
    """Verifies that the running set stays inside both budgets for a heterogeneous queue."""
    sizes = [50000, 20000, 20000, 6000, 6000, 6000, 6000, 6000]
    state = build_state(
        jobs=[make_job(job_id=f"j{index}", cores=4, memory_mb=size) for index, size in enumerate(sizes)],
        core_budget=CORE_BUDGET,
        memory_budget_mb=MEMORY_BUDGET_MB,
    )
    pool = admit(state)
    assert sum(job.core_weight for job in pool.submitted) <= CORE_BUDGET
    assert sum(job.memory_mb for job in pool.submitted) <= MEMORY_BUDGET_MB


def test_admission_considers_the_heaviest_job_first() -> None:
    """Verifies that a large job is dispatched before lighter ones rather than waiting for them to drain."""
    jobs = [make_job(job_id=f"light{index}", memory_mb=1024) for index in range(6)]
    jobs.append(make_job(job_id="heavy", memory_mb=40000))
    pool = admit(build_state(jobs=jobs))
    assert pool.submitted[0].job_id == "heavy"


def test_light_jobs_backfill_the_capacity_a_heavy_job_leaves_spare() -> None:
    """Verifies that a heavy job runs alongside as many light jobs as the remaining budget allows."""
    jobs = [make_job(job_id="heavy", cores=8, memory_mb=40000)]
    jobs += [make_job(job_id=f"light{index}", cores=2, memory_mb=4000) for index in range(10)]
    pool = admit(build_state(jobs=jobs, core_budget=CORE_BUDGET, memory_budget_mb=MEMORY_BUDGET_MB))
    submitted = {job.job_id for job in pool.submitted}
    assert "heavy" in submitted
    assert len(submitted) >= 6, "light jobs did not backfill around the heavy job"
    assert sum(job.memory_mb for job in pool.submitted) <= MEMORY_BUDGET_MB


def test_freed_capacity_is_repacked_on_the_next_pass() -> None:
    """Verifies that finishing a job immediately releases its share to the jobs still queued."""
    jobs = [make_job(job_id=f"j{index}", cores=1, memory_mb=20000) for index in range(6)]
    state = build_state(jobs=jobs, core_budget=CORE_BUDGET, memory_budget_mb=MEMORY_BUDGET_MB)

    first = admit(state)
    assert len(first.submitted) == 3
    assert len(state.pending_jobs) == 3

    complete(state=state, job=first.submitted[0])
    second = admit(state)
    assert len(second.submitted) == 1, "capacity freed by a finished job was not refilled"


def test_admission_withholds_a_job_until_its_prerequisite_succeeds() -> None:
    """Verifies that a dependent job is not dispatched while its upstream job has not yet succeeded."""
    upstream = make_job(job_id="up", job_name="binarization")
    downstream = make_job(job_id="down", prerequisites=("up",))
    state = build_state(jobs=[upstream, downstream])

    pool = admit(state)
    assert [job.job_id for job in pool.submitted] == ["up"]

    complete(state=state, job=upstream)
    assert [job.job_id for job in admit(state).submitted] == ["down"]


def test_one_sessions_completed_stage_does_not_satisfy_another_sessions() -> None:
    """Verifies that identical job identifiers in different sessions are tracked separately.

    A job identifier is derived from the job name and specifier alone, so every session's binarization shares one
    identifier. A batch spanning sessions must not treat one session's completed stage as every session's.
    """
    first_upstream = make_job(job_id="shared", job_name="binarization", tracker_path=TRACKER, unit_path=UNIT)
    second_downstream = make_job(
        job_id="down", prerequisites=("shared",), tracker_path=OTHER_TRACKER, unit_path=OTHER_UNIT
    )
    state = build_state(jobs=[first_upstream, second_downstream])

    pool = admit(state)
    assert [job.job_id for job in pool.submitted] == ["shared"]

    complete(state=state, job=first_upstream)
    assert not admit(state).submitted, "another session's dependent was released by an unrelated session's job"
    assert [job.job_id for job in state.pending_jobs] == ["down"]


def test_admission_blocks_a_dependent_whose_prerequisite_failed() -> None:
    """Verifies that a job whose upstream failed is moved aside rather than waiting for an outcome that cannot come."""
    downstream = make_job(job_id="down", prerequisites=("up",))
    state = build_state(jobs=[downstream])
    state.failed_job_keys.add((str(UNIT), "up"))

    assert not admit(state).submitted
    assert [job.job_id for job in state.blocked_jobs] == ["down"]
    assert not state.pending_jobs


def test_forward_progress_floor_never_bypasses_the_prerequisite_check() -> None:
    """Verifies that the single-job floor does not dispatch a job whose input does not exist yet."""
    downstream = make_job(job_id="down", memory_mb=10_000_000, prerequisites=("up",))
    assert not admit(build_state(jobs=[downstream])).submitted


def test_forward_progress_floor_admits_a_job_larger_than_the_budget() -> None:
    """Verifies that a job no budget can hold still runs alone rather than stalling the batch."""
    oversized = make_job(job_id="huge", cores=999, memory_mb=10_000_000)
    assert [job.job_id for job in admit(build_state(jobs=[oversized])).submitted] == ["huge"]


def test_estimates_carry_the_shared_tolerance() -> None:
    """Verifies that the reported memory of an estimate exceeds its modeled value by the shared margin."""
    assert _apply_tolerance(memory_mb=1000) == int(1000 * _MEMORY_ESTIMATE_TOLERANCE) + 1
    assert _apply_tolerance(memory_mb=0) == 1
    assert _MEMORY_ESTIMATE_TOLERANCE > 1.0


def test_host_memory_is_readable_and_positive() -> None:
    """Verifies that the host memory probe returns a usable figure on any supported host."""
    assert resolve_host_memory_mb() > 0


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
        "tracker_path": str(TRACKER),
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
    single = _estimate_checksum_memory(cores=1)
    doubled = _estimate_checksum_memory(cores=2)
    assert doubled - single == pytest.approx(_CHECKSUM_READER_MEMORY_MB * _MEMORY_ESTIMATE_TOLERANCE, rel=0.01)
    assert _estimate_checksum_memory(cores=8) > single


def test_worker_initializer_leaves_the_numba_thread_variable_alone() -> None:
    """Verifies that the worker initializer controls numba through its runtime setter rather than its environment.

    numba reads NUMBA_NUM_THREADS once at import and compares the variable against that latched count on every
    compilation, rejecting a disagreement once its thread pool has started. A worker imports numba before the
    initializer runs, so pinning the variable there would fail every job that compiles a numba function.
    """
    assert "NUMBA_NUM_THREADS" not in _PINNED_THREAD_VARIABLES

    # The other threading layers stay pinned, since they read their variables when the job itself starts.
    assert "OMP_NUM_THREADS" in _PINNED_THREAD_VARIABLES
    assert "POLARS_MAX_THREADS" in _PINNED_THREAD_VARIABLES


@pytest.mark.parametrize("pipeline", sorted(member.value for member in BATCH_PIPELINES))
def test_every_dispatch_entry_declares_the_whole_generic_contract(pipeline: str) -> None:
    """Verifies that each registered pipeline supplies every callable the unit-generic dispatch contract requires.

    The batch layer reads a unit only through these callables, so an entry omitting one fails at preparation rather
    than at registration.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    assert dispatch is not None

    for field in ("discover", "worker", "prerequisites", "tracker_path", "output_path", "unit_name", "estimate_memory"):
        assert callable(getattr(dispatch, field)), f"{pipeline} declares no {field}"


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
