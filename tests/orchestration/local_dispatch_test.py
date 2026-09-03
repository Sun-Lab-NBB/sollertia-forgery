"""Contains tests for the local batch execution engine and the pipeline dispatch table through which both execution
backends route jobs.
"""

from __future__ import annotations

import os
from time import sleep
from typing import TYPE_CHECKING, Any, Self
from pathlib import Path
from collections import deque
from dataclasses import replace
from concurrent.futures import Future

import cv2
import numba
import pytest
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SessionTypes,
    DatasetSession,
    AcquisitionSystems,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import FORGING_JOB_NAME
from sollertia_forgery.runtime import RUNTIME_JOB_NAME
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.two_photon import SingleRecordingJobNames
from sollertia_forgery.orchestration import (
    JobExecutionState,
    run_batch_job,
    build_pending_job,
    group_jobs_by_tracker,
    job_execution_manager,
    resolve_core_allocations,
    resolve_concurrency_limits,
    resolve_concurrency_reservations,
)
from sollertia_forgery.orchestration.graph import PendingJob
from sollertia_forgery.orchestration.local import (
    _reset_queued_jobs,
    _admit_pending_jobs,
    _pinned_pool_imports,
    _refresh_job_outcomes,
    _initialize_worker_threads,
    apply_decode_thread_ceiling,
)
import sollertia_forgery.orchestration.dispatch as dispatch_module
from sollertia_forgery.orchestration.dispatch import (
    DATASET_UNIT,
    SESSION_UNIT,
    resolve_dispatch,
    resolve_job_cores,
    resolve_unit_kind,
    resolve_job_command,
    resolve_unit_dispatches,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sollertia_shared_assets import ProjectData

_PINNED_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
    "TIFFFILE_NUM_THREADS",
)
"""The threading-layer environment variables the worker initializer is expected to pin.

Notes:
    The pinning itself belongs to ataraxis-data-structures, which keeps its own list of these variables private, so
    this module names them again rather than reaching for that private tuple. NUMBA_NUM_THREADS is deliberately
    absent, since numba is pinned through its runtime setter instead of through the environment.
"""

_CORE_BUDGET: int = 64
"""The cores against which every admission test in this module weighs its jobs."""

_MEMORY_BUDGET_MB: int = 65536
"""The memory against which every admission test in this module weighs its jobs."""

_SESSION_UNIT_PATH: str = "/data/Project/305/2026-01-02-03-04-05-000006"
"""The session directory named by the rendered command lines and the dispatched session jobs."""

_DATASET_UNIT_PATH: str = "/data/Project/Dataset"
"""The dataset directory named by the rendered forging command and the dispatched forging job."""

_SLOW_JOB_SECONDS: float = 0.25
"""How long the slow batch worker occupies its pool slot, which is long enough for the manager to observe it running
while it reaps the job that finished beside it."""

_PIPELINE_ENTRY_POINTS: dict[str, str] = {
    "checksum": "run_checksum_processing_pipeline",
    "runtime": "run_runtime_processing_pipeline",
    "microcontroller": "run_microcontroller_processing_pipeline",
    "video": "run_video_processing_pipeline",
    "two_photon": "run_two_photon_processing_pipeline",
    "forging": "run_forging_pipeline",
}
"""The pipeline entry point each dispatch worker calls, keyed by the pipeline the worker serves."""


# Module-level batch workers, which the shared process pool imports in each of its children


def complete_tracked_job(job: PendingJob) -> None:
    """Records one job as succeeded on its own tracker, the way a pipeline worker does when its stage finishes.

    Args:
        job: The dispatched job whose tracker records the outcome.
    """
    tracker = ProcessingTracker(file_path=job.tracker_path)
    tracker.start_job(job_id=job.job_id)
    tracker.complete_job(job_id=job.job_id)


def fail_tracked_job(job: PendingJob) -> None:
    """Records one job as failed on its own tracker, the way a pipeline worker does when its stage raises.

    Args:
        job: The dispatched job whose tracker records the outcome.
    """
    tracker = ProcessingTracker(file_path=job.tracker_path)
    tracker.start_job(job_id=job.job_id)
    tracker.fail_job(job_id=job.job_id, error_message="The stage raised.")


def complete_tracked_job_slowly(job: PendingJob) -> None:
    """Holds a pool slot for a fixed period and then records the job as succeeded.

    Args:
        job: The dispatched job whose tracker records the outcome.
    """
    if job.job_name == "slow_stage":
        sleep(_SLOW_JOB_SECONDS)
    complete_tracked_job(job=job)


# Helpers


class RecordingPool:
    """Stands in for the shared process pool, recording each admitted job rather than running it.

    Attributes:
        submitted: The jobs that admission handed to the pool, in the order it handed them over.
    """

    def __init__(self) -> None:
        self.submitted: list[PendingJob] = []

    def submit(self, worker: object, job: PendingJob) -> Future[None]:  # noqa: ARG002
        """Records one admitted job and answers a future that stays unresolved.

        Args:
            worker: The callable that admission would have invoked.
            job: The admitted job.

        Returns:
            An unresolved future, which keeps the job in the running set.
        """
        self.submitted.append(job)
        return Future()


def make_job(
    job_id: str,
    job_name: str = "processing",
    cores: int = 1,
    memory_mb: int = 512,
    prerequisites: tuple[str, ...] = (),
    tracker_path: Path = Path("/nonexistent/tracker.yaml"),
    unit_path: Path = Path("/nonexistent/session"),
) -> PendingJob:
    """Builds one pending job carrying explicit resource weights and prerequisite identifiers.

    Args:
        job_id: The identifier under which the tracker records the job.
        job_name: The job type name, which keys every concurrency term.
        cores: The cores the job occupies while it runs.
        memory_mb: The memory the job occupies while it runs.
        prerequisites: The identifiers of the jobs on which this one waits.
        tracker_path: The tracker on which the job records its outcome.
        unit_path: The processing unit on which the job operates.

    Returns:
        The pending job.
    """
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
    jobs: list[PendingJob],
    core_budget: int = _CORE_BUDGET,
    memory_budget_mb: int = _MEMORY_BUDGET_MB,
    limits: dict[str, int] | None = None,
    reservations: dict[str, int] | None = None,
) -> JobExecutionState[PendingJob]:
    """Builds one execution state holding the supplied jobs against explicit budgets.

    Args:
        jobs: The jobs the state queues.
        core_budget: The cores the batch may commit at once.
        memory_budget_mb: The memory the batch may commit at once.
        limits: The concurrent-job ceilings the queued types declare.
        reservations: The concurrency to which the queued types are held while other work can use the room.

    Returns:
        The execution state.
    """
    return JobExecutionState(
        worker=complete_tracked_job,
        all_jobs={job.dispatch_key: job for job in jobs},
        pending_jobs=deque(jobs),
        core_budget=core_budget,
        memory_budget_mb=memory_budget_mb,
        concurrency_limits=dict(limits or {}),
        concurrency_reservations=dict(reservations or {}),
        pool_size=max(1, len(jobs)),
    )


def admit(state: JobExecutionState[PendingJob]) -> RecordingPool:
    """Runs one admission pass against a recording pool.

    Args:
        state: The execution state whose pending queue is scanned.

    Returns:
        The pool holding whatever the pass admitted.
    """
    pool = RecordingPool()
    _admit_pending_jobs(state=state, pool=pool)  # type: ignore[arg-type]
    return pool


def write_tracker_universe(tracker_path: Path, jobs: list[tuple[str, str]]) -> ProcessingTracker:
    """Creates one processing tracker holding the given job universe in the scheduled state.

    Args:
        tracker_path: The path to which the tracker is written.
        jobs: The job name and specifier pairs the tracker registers.

    Returns:
        The written tracker.
    """
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=jobs, universe=jobs)
    return tracker


def build_descriptor(
    job_id: str = "a_job",
    job_name: str = "motion_energy",
    pipeline: str = "video",
    unit_path: str = _SESSION_UNIT_PATH,
    cores: int = 4,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds one job descriptor of the shape preparation emits.

    Args:
        job_id: The identifier under which the tracker records the job.
        job_name: The job type name.
        pipeline: The pipeline to which the job belongs.
        unit_path: The processing unit on which the job operates.
        cores: The cores at which the job was planned.
        options: The pipeline-specific parameters with which the job runs.

    Returns:
        The descriptor.
    """
    return {
        "job_id": job_id,
        "job_name": job_name,
        "specifier": "1",
        "unit_path": unit_path,
        "unit_name": Path(unit_path).name,
        "pipeline": pipeline,
        "cores": cores,
        "memory_mb": 4096,
        "prerequisite_ids": [],
        "options": dict(options or {}),
    }


# Fixtures


@pytest.fixture
def pinned_thread_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restores every threading-layer setting the worker initializer writes once the test finishes.

    The initializer writes process-global environment variables and calls the numba and OpenCV runtime setters, so a
    test that runs it has to hand the process back its original counts.

    Args:
        monkeypatch: The fixture used to record and restore each pinned environment variable.

    Yields:
        Nothing, since the fixture exists for the restoration it performs.
    """
    numba_threads = numba.get_num_threads()
    opencv_threads = cv2.getNumThreads()
    for variable in _PINNED_THREAD_VARIABLES:
        monkeypatch.setenv(variable, os.environ.get(variable, ""))
    yield
    numba.set_num_threads(n=numba_threads)
    cv2.setNumThreads(opencv_threads)


@pytest.fixture
def restored_console() -> Iterator[None]:
    """Hands the process back its original console state once the test finishes.

    The console is a process-global singleton, so a test that silences it to stand in for the MCP server on the stdio
    transport would otherwise leave every later test running against a silent console.

    Yields:
        Nothing, since the fixture exists for the restoration it performs.
    """
    enabled = console.enabled
    yield
    if enabled:
        console.enable()
    else:
        console.disable()


@pytest.fixture
def recorded_opencv_thread_counts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Records the width to which the worker initializer pins the OpenCV core thread pool.

    OpenCV selects its parallel backend at build time, and the backend shipped by the macOS wheels accepts a pinned
    width without reporting it back through the reader. The request itself is therefore what states that the pin
    happened on every platform this package supports.

    Args:
        monkeypatch: The fixture used to replace the OpenCV thread setter for the duration of one test.

    Returns:
        The list to which every requested thread count is appended, in request order.
    """
    counts: list[int] = []
    monkeypatch.setattr(cv2, "setNumThreads", counts.append)
    return counts


@pytest.fixture
def recorded_pipeline_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Stands a recorder in for every pipeline entry point the dispatch workers call.

    Each entry point runs a whole processing stage against acquired data, so the dispatch layer is exercised by
    recording the arguments it forwards rather than by running the stages themselves.

    Args:
        monkeypatch: The fixture used to replace each entry point the dispatch module holds.

    Returns:
        The keyword arguments each replaced entry point received, keyed by the pipeline that called it.
    """
    recorded: dict[str, dict[str, Any]] = {}

    def _recorder(pipeline: str) -> Callable[..., None]:
        """Builds the recorder standing in for one pipeline's entry point."""

        def _record(**keywords: Any) -> None:
            """Records the keyword arguments the dispatch worker forwarded."""
            recorded[pipeline] = keywords

        return _record

    for pipeline, entry_point in _PIPELINE_ENTRY_POINTS.items():
        monkeypatch.setattr(dispatch_module, entry_point, _recorder(pipeline=pipeline))
    return recorded


# Core allocation


def test_a_job_type_takes_its_declared_cores_narrowed_to_the_budget() -> None:
    """Verifies that a small host never promises a job more cores than it holds, and the concurrency follows from what
    it does.
    """
    generous = resolve_core_allocations(job_cores={"wide": 16}, job_names={"wide"}, core_budget=64)
    cramped = resolve_core_allocations(job_cores={"wide": 16}, job_names={"wide"}, core_budget=4)

    assert generous["wide"].cores_per_job == 16
    assert generous["wide"].maximum_parallel == 4
    assert generous["wide"].concurrency_limit is None
    assert generous["wide"].concurrency_reservation is None
    assert cramped["wide"].cores_per_job == 4
    assert cramped["wide"].maximum_parallel == 1


def test_a_declared_ceiling_narrows_the_concurrency_the_budget_alone_would_allow() -> None:
    """Verifies that the ceiling and the reservation are both reported, so a caller can tell which term produced the
    maximum.
    """
    allocations = resolve_core_allocations(
        job_cores={"streamed": 8},
        job_names={"streamed"},
        core_budget=64,
        job_limits={"streamed": 3},
        job_reservations={"streamed": 2},
    )

    assert allocations["streamed"].maximum_parallel == 3
    assert allocations["streamed"].concurrency_limit == 3
    assert allocations["streamed"].concurrency_reservation == 2


def test_an_unregistered_job_type_stops_the_whole_allocation() -> None:
    """Verifies that dispatching a type with no declared allocation would run it at a width nobody chose."""
    with pytest.raises(ValueError, match=r"No core allocation is registered for job type\(s\) \['unregistered'\]"):
        resolve_core_allocations(job_cores={"known": 1}, job_names={"known", "unregistered"}, core_budget=8)


# Admission


def test_admission_fills_both_budgets_and_defers_what_neither_can_hold() -> None:
    """Verifies that cores and memory are weighed together, so the batch's mix decides which of the two bounds the
    pass.
    """
    jobs = [make_job(job_id=f"job{index}", cores=16, memory_mb=1024) for index in range(6)]
    state = build_state(jobs=jobs)

    pool = admit(state=state)

    assert len(pool.submitted) == 4
    assert len(state.pending_jobs) == 2
    assert sum(job.core_weight for job in pool.submitted) == _CORE_BUDGET


def test_a_later_pass_counts_the_cores_the_already_running_jobs_committed() -> None:
    """Verifies that every pass recomputes the committed totals from the running set, so capacity already taken is never
    reoffered.
    """
    jobs = [make_job(job_id=f"job{index}", cores=16, memory_mb=1024) for index in range(6)]
    state = build_state(jobs=jobs)

    first = admit(state=state)
    second = admit(state=state)

    # The first pass commits all sixty-four cores, and the memory budget stays wide open throughout, so cores are the
    # only term that can hold the two remaining jobs back. A pass that began its core tally at zero would read a full
    # pool as an idle one and admit another budget's worth of work on top of it.
    assert len(first.submitted) == 4
    assert second.submitted == []
    assert len(state.pending_jobs) == 2


def test_a_job_larger_than_the_whole_budget_still_runs_alone() -> None:
    """Verifies a batch whose only job outsizes the host makes progress rather than stalling on an unreachable fit."""
    oversized = make_job(job_id="oversized", cores=999, memory_mb=10_000_000)

    assert [job.job_id for job in admit(state=build_state(jobs=[oversized])).submitted] == ["oversized"]


def test_a_job_whose_prerequisite_has_not_succeeded_waits_in_the_queue() -> None:
    """Verifies that the forward-progress floor never dispatches a job before the stage that writes its input has
    finished.
    """
    dependent = make_job(job_id="dependent", prerequisites=("upstream",))
    state = build_state(jobs=[dependent])

    assert admit(state=state).submitted == []
    assert [job.job_id for job in state.pending_jobs] == ["dependent"]
    assert state.blocked_jobs == []


def test_a_job_whose_prerequisite_failed_is_dropped_rather_than_queued() -> None:
    """Verifies an outcome that can never arrive would hold the job forever, so the job leaves the queue as blocked."""
    dependent = make_job(job_id="dependent", prerequisites=("upstream",))
    state = build_state(jobs=[dependent])
    state.failed_job_keys.add((str(dependent.unit_path), "upstream"))

    assert admit(state=state).submitted == []
    assert [job.job_id for job in state.blocked_jobs] == ["dependent"]
    assert not state.pending_jobs


def test_a_satisfied_prerequisite_releases_the_job_waiting_on_it() -> None:
    """Verifies that recorded success is what admission reads, so a job becomes runnable the moment its upstream stage
    is recorded.
    """
    dependent = make_job(job_id="dependent", prerequisites=("upstream",))
    state = build_state(jobs=[dependent])
    state.succeeded_job_keys.add((str(dependent.unit_path), "upstream"))

    assert [job.job_id for job in admit(state=state).submitted] == ["dependent"]


def test_a_declared_ceiling_holds_a_type_while_the_budgets_stay_open() -> None:
    """Verifies that a type held by its ceiling waits on a resource that idle cores do not supply, so spare capacity
    never lifts it.
    """
    jobs = [make_job(job_id=f"stream{index}", job_name="streamed", cores=1, memory_mb=64) for index in range(5)]
    state = build_state(jobs=jobs, limits={"streamed": 2})

    pool = admit(state=state)

    assert len(pool.submitted) == 2
    assert len(state.pending_jobs) == 3


def test_a_reservation_lifts_once_nothing_else_can_use_the_room_it_holds() -> None:
    """Verifies that a reservation exists to leave room for other work, so its own queue widens when no other work is
    runnable.
    """
    reserved = [make_job(job_id=f"wide{index}", job_name="wide_stage", cores=1, memory_mb=64) for index in range(4)]
    state = build_state(jobs=reserved, reservations={"wide_stage": 2})

    assert len(admit(state=state).submitted) == 4


def test_a_reservation_offers_its_room_to_other_runnable_work_first() -> None:
    """Verifies holding the reserved type back in the first pass is what keeps the other queues moving beside it."""
    reserved = [make_job(job_id=f"wide{index}", job_name="wide_stage", cores=20, memory_mb=64) for index in range(3)]
    others = [make_job(job_id=f"light{index}", job_name="light_stage", cores=14, memory_mb=64) for index in range(3)]
    state = build_state(jobs=[*reserved, *others], reservations={"wide_stage": 1})

    admitted = {job.job_id for job in admit(state=state).submitted}

    # The first pass admits one reserved job and hands the rest of the budget to the queue that has no reservation,
    # which leaves the second pass no room for widening the reserved type.
    assert admitted == {"wide0", "light0", "light1", "light2"}
    assert [job.job_id for job in state.pending_jobs] == ["wide1", "wide2"]


# Tracker reading and resetting


def test_jobs_are_grouped_so_each_tracker_is_opened_once(tmp_path: Path) -> None:
    """Verifies that a batch holding many jobs of one unit reads that unit's tracker a single time per pass."""
    first = tmp_path.joinpath("one", "tracker.yaml")
    second = tmp_path.joinpath("two", "tracker.yaml")
    jobs = [
        make_job(job_id="a", tracker_path=first),
        make_job(job_id="b", tracker_path=first),
        make_job(job_id="c", tracker_path=second, unit_path=Path("/nonexistent/other")),
    ]

    grouped = group_jobs_by_tracker(state=build_state(jobs=jobs))

    assert {path: [job.job_id for job in held] for path, held in grouped.items()} == {
        first: ["a", "b"],
        second: ["c"],
    }


def test_starting_a_batch_returns_only_its_own_jobs_to_the_scheduled_state(tmp_path: Path) -> None:
    """Verifies that a queued job whose tracker still records an earlier success would report as done before the batch
    dispatched.
    """
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    universe = [("queued_stage", ""), ("earlier_stage", "")]
    tracker = write_tracker_universe(tracker_path=tracker_path, jobs=universe)
    queued_id = ProcessingTracker.generate_job_id(job_name="queued_stage", specifier="")
    earlier_id = ProcessingTracker.generate_job_id(job_name="earlier_stage", specifier="")
    for job_id in (queued_id, earlier_id):
        tracker.start_job(job_id=job_id)
        tracker.complete_job(job_id=job_id)

    state = build_state(
        jobs=[
            make_job(job_id=queued_id, tracker_path=tracker_path),
            # A unit whose tracker has never been written contributes nothing to reset.
            make_job(job_id="absent", tracker_path=tmp_path.joinpath("missing", "tracker.yaml")),
        ]
    )
    _reset_queued_jobs(state=state)

    snapshot = ProcessingTracker(file_path=tracker_path).snapshot()
    assert snapshot[queued_id].status is ProcessingStatus.SCHEDULED
    assert snapshot[earlier_id].status is ProcessingStatus.SUCCEEDED


def test_recorded_outcomes_are_read_back_under_the_unit_that_holds_them(tmp_path: Path) -> None:
    """Verifies that one unit's completed stage never satisfies another unit's dependent, so outcomes are keyed by unit
    and job.
    """
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    universe = [("done_stage", ""), ("broken_stage", "")]
    tracker = write_tracker_universe(tracker_path=tracker_path, jobs=universe)
    done_id = ProcessingTracker.generate_job_id(job_name="done_stage", specifier="")
    broken_id = ProcessingTracker.generate_job_id(job_name="broken_stage", specifier="")
    tracker.start_job(job_id=done_id)
    tracker.complete_job(job_id=done_id)
    tracker.start_job(job_id=broken_id)
    tracker.fail_job(job_id=broken_id, error_message="The stage raised.")

    unit = tmp_path.joinpath("unit")
    state = build_state(
        jobs=[
            make_job(job_id=done_id, tracker_path=tracker_path, unit_path=unit),
            make_job(job_id=broken_id, tracker_path=tracker_path, unit_path=unit),
            make_job(job_id="absent", tracker_path=tmp_path.joinpath("missing", "tracker.yaml")),
        ]
    )
    _refresh_job_outcomes(state=state)

    assert state.succeeded_job_keys == {(str(unit), done_id)}
    assert state.failed_job_keys == {(str(unit), broken_id)}


# The execution manager


@pytest.mark.xdist_group(name="worker_pool")
def test_the_manager_runs_a_dependency_chain_through_to_its_last_stage(tmp_path: Path) -> None:
    """Verifies each finished job refreshes the recorded outcomes, which is what releases the stage waiting on it."""
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    universe = [("first_stage", ""), ("second_stage", "")]
    write_tracker_universe(tracker_path=tracker_path, jobs=universe)
    first_id = ProcessingTracker.generate_job_id(job_name="first_stage", specifier="")
    second_id = ProcessingTracker.generate_job_id(job_name="second_stage", specifier="")

    unit = tmp_path.joinpath("unit")
    jobs = [
        make_job(
            job_id=second_id,
            job_name="second_stage",
            tracker_path=tracker_path,
            unit_path=unit,
            prerequisites=(first_id,),
        ),
        make_job(job_id=first_id, job_name="first_stage", tracker_path=tracker_path, unit_path=unit),
    ]
    state = build_state(jobs=jobs)
    job_execution_manager(state=state)

    snapshot = ProcessingTracker(file_path=tracker_path).snapshot()
    assert snapshot[first_id].status is ProcessingStatus.SUCCEEDED
    assert snapshot[second_id].status is ProcessingStatus.SUCCEEDED
    # Reading the first stage back is what released the second, so the session records it as a satisfied prerequisite.
    assert (str(unit), first_id) in state.succeeded_job_keys
    assert state.blocked_jobs == []


@pytest.mark.xdist_group(name="worker_pool")
def test_the_manager_records_the_dependents_of_a_failed_stage_as_blocked(tmp_path: Path) -> None:
    """Verifies that a stage whose input can never be produced leaves the queue instead of waiting for an outcome that
    cannot come.
    """
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    universe = [("first_stage", ""), ("second_stage", "")]
    write_tracker_universe(tracker_path=tracker_path, jobs=universe)
    first_id = ProcessingTracker.generate_job_id(job_name="first_stage", specifier="")
    second_id = ProcessingTracker.generate_job_id(job_name="second_stage", specifier="")

    unit = tmp_path.joinpath("unit")
    state = build_state(
        jobs=[
            make_job(job_id=first_id, job_name="first_stage", tracker_path=tracker_path, unit_path=unit),
            make_job(
                job_id=second_id,
                job_name="second_stage",
                tracker_path=tracker_path,
                unit_path=unit,
                prerequisites=(first_id,),
            ),
        ]
    )
    state.worker = fail_tracked_job
    job_execution_manager(state=state)

    assert ProcessingTracker(file_path=tracker_path).snapshot()[first_id].status is ProcessingStatus.FAILED
    assert [job.job_id for job in state.blocked_jobs] == [second_id]
    assert not state.pending_jobs


@pytest.mark.xdist_group(name="worker_pool")
def test_the_manager_reaps_a_finished_job_while_another_is_still_running(tmp_path: Path) -> None:
    """Verifies that capacity is released the moment a job finishes, so the pass that reaps it leaves its neighbor
    running.
    """
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    universe = [("fast_stage", ""), ("slow_stage", "")]
    write_tracker_universe(tracker_path=tracker_path, jobs=universe)
    fast_id = ProcessingTracker.generate_job_id(job_name="fast_stage", specifier="")
    slow_id = ProcessingTracker.generate_job_id(job_name="slow_stage", specifier="")

    unit = tmp_path.joinpath("unit")
    state = build_state(
        jobs=[
            make_job(job_id=fast_id, job_name="fast_stage", tracker_path=tracker_path, unit_path=unit),
            make_job(job_id=slow_id, job_name="slow_stage", tracker_path=tracker_path, unit_path=unit),
        ]
    )
    state.worker = complete_tracked_job_slowly
    # One worker runs the pair in turn, so the fast job is reaped while the slow one still holds the pool.
    state.pool_size = 1
    job_execution_manager(state=state)

    snapshot = ProcessingTracker(file_path=tracker_path).snapshot()
    assert snapshot[fast_id].status is ProcessingStatus.SUCCEEDED
    assert snapshot[slow_id].status is ProcessingStatus.SUCCEEDED


@pytest.mark.xdist_group(name="worker_pool")
def test_a_canceled_session_stops_admitting_and_records_the_remainder(tmp_path: Path) -> None:
    """Verifies that cancellation stops new admissions, so the queue is reported rather than dispatched."""
    tracker_path = tmp_path.joinpath("unit", "tracker.yaml")
    write_tracker_universe(tracker_path=tracker_path, jobs=[("first_stage", "")])
    first_id = ProcessingTracker.generate_job_id(job_name="first_stage", specifier="")

    state = build_state(
        jobs=[
            make_job(
                job_id=first_id, job_name="first_stage", tracker_path=tracker_path, unit_path=tmp_path.joinpath("unit")
            )
        ]
    )
    state.canceled = True
    job_execution_manager(state=state)

    assert [job.job_id for job in state.blocked_jobs] == [first_id]
    assert ProcessingTracker(file_path=tracker_path).snapshot()[first_id].status is ProcessingStatus.SCHEDULED


# Worker thread pinning


@pytest.mark.xdist_group(name="worker_pool")
def test_the_decode_ceiling_follows_the_cores_a_job_holds(pinned_thread_environment: None) -> None:
    """Verifies that a decode stops shortening past the ceiling, so a wide job spends its remaining cores on the stage
    itself.
    """
    apply_decode_thread_ceiling(cores=2)
    assert os.environ["TIFFFILE_NUM_THREADS"] == "2"

    apply_decode_thread_ceiling(cores=32)
    assert os.environ["TIFFFILE_NUM_THREADS"] == "4"

    apply_decode_thread_ceiling(cores=0)
    assert os.environ["TIFFFILE_NUM_THREADS"] == "1"


@pytest.mark.xdist_group(name="worker_pool")
def test_the_worker_initializer_pins_every_declared_threading_layer(
    pinned_thread_environment: None, recorded_opencv_thread_counts: list[int]
) -> None:
    """Verifies that a worker holds one core, so every library pool it opens has to be pinned before the job starts."""
    _initialize_worker_threads(thread_ceiling=1)

    for variable in _PINNED_THREAD_VARIABLES:
        assert os.environ[variable] == "1"
    assert recorded_opencv_thread_counts == [1]
    assert numba.get_num_threads() == 1


@pytest.mark.xdist_group(name="worker_pool")
def test_the_worker_initializer_raises_a_non_positive_ceiling_to_one(
    pinned_thread_environment: None, recorded_opencv_thread_counts: list[int]
) -> None:
    """Verifies that a pool pinned to zero threads would open nothing at all, so the floor of one is applied first."""
    _initialize_worker_threads(thread_ceiling=0)

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert recorded_opencv_thread_counts == [1]


@pytest.mark.xdist_group(name="worker_pool")
def test_the_worker_initializer_mirrors_a_silenced_parent_console(
    pinned_thread_environment: None, recorded_opencv_thread_counts: list[int], restored_console: None
) -> None:
    """Verifies that a spawned worker comes up with a freshly enabled console while writing to the stream the parent
    handed it, so a parent that silenced its own has to have that silence carried into the child.
    """
    console.enable()
    _initialize_worker_threads(thread_ceiling=1, console_disabled=True)
    assert not console.enabled

    console.enable()
    _initialize_worker_threads(thread_ceiling=1, console_disabled=False)
    assert console.enabled


@pytest.mark.xdist_group(name="console")
def test_the_pool_hands_its_children_whatever_console_state_the_parent_holds(
    monkeypatch: pytest.MonkeyPatch, restored_console: None
) -> None:
    """Verifies that mirroring the parent is what keeps a batch run's worker output intact while silencing the children
    of the MCP server on the stdio transport, whose stdout carries the JSON-RPC stream an echoed line would corrupt.
    """
    recorded: list[tuple[Any, ...]] = []

    class CapturingPool(RecordingPool):
        """Stands in for the shared pool, recording the arguments with which its children would be initialized."""

        def __init__(self, max_workers: int, initializer: Callable[..., None], initargs: tuple[Any, ...]) -> None:  # noqa: ARG002
            super().__init__()
            recorded.append(initargs)

        def __enter__(self) -> Self:
            """Enters the pool's scope, which the manager holds for the session."""
            return self

        def __exit__(self, *_exception: object) -> None:
            """Leaves the pool's scope without suppressing anything."""

    monkeypatch.setattr("sollertia_forgery.orchestration.local.ProcessPoolExecutor", CapturingPool)

    console.enable()
    job_execution_manager(state=build_state(jobs=[]))

    console.disable()
    job_execution_manager(state=build_state(jobs=[]))

    assert [initargs[1] for initargs in recorded] == [False, True]


@pytest.mark.xdist_group(name="console")
def test_the_pool_spawns_the_batchs_own_width_while_pinning_its_children_to_the_thread_ceiling(
    monkeypatch: pytest.MonkeyPatch, restored_console: None
) -> None:
    """Verifies that the width at which the pool spawns and the threads to which each child pins its libraries are
    two unrelated figures.
    """
    recorded: list[tuple[int, tuple[Any, ...]]] = []

    class WidthCapturingPool(RecordingPool):
        """Stands in for the shared pool, recording its spawn width and the arguments handed to its children."""

        def __init__(self, max_workers: int, initializer: Callable[..., None], initargs: tuple[Any, ...]) -> None:  # noqa: ARG002
            super().__init__()
            recorded.append((max_workers, initargs))

        def __enter__(self) -> Self:
            """Enters the pool's scope, which the manager holds for the session."""
            return self

        def __exit__(self, *_exception: object) -> None:
            """Leaves the pool's scope without suppressing anything."""

    monkeypatch.setattr("sollertia_forgery.orchestration.local.ProcessPoolExecutor", WidthCapturingPool)
    state = build_state(jobs=[])
    state.pool_size = 8
    state.thread_ceiling = 2

    console.enable()
    job_execution_manager(state=state)

    # The batch tools size the pool from the host's cores and leave the thread ceiling at its default of one. A pool
    # spawned at the ceiling instead would run every local batch strictly serially.
    assert recorded == [(8, (2, False))]


@pytest.mark.xdist_group(name="worker_pool")
def test_the_pool_import_pin_hands_the_parent_back_what_it_was_using(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the parent has already imported the same libraries, so the pin covering the pool's lifetime has to
    leave the parent's own threading exactly as it found it.
    """
    monkeypatch.setenv("POLARS_MAX_THREADS", "12")

    with _pinned_pool_imports():
        assert os.environ["POLARS_MAX_THREADS"] == "1"
    assert os.environ["POLARS_MAX_THREADS"] == "12"

    # A variable the parent never set is removed again rather than left behind at the pinned width.
    monkeypatch.delenv("POLARS_MAX_THREADS")
    with _pinned_pool_imports():
        assert os.environ["POLARS_MAX_THREADS"] == "1"
    assert "POLARS_MAX_THREADS" not in os.environ


# The dispatch table


def test_an_identifier_outside_the_pipeline_enumeration_resolves_to_no_dispatch() -> None:
    """Verifies a caller reads the absent entry as an unsupported pipeline rather than receiving a partial table."""
    assert resolve_dispatch(pipeline="not_a_pipeline") is None


def test_a_registered_job_type_reports_the_cores_it_declares() -> None:
    """Verifies that a stage sized by this package itself reports the allocation the dispatch table declares for it,
    so it is reported unnarrowed.
    """
    assert resolve_job_cores(job_name=CHECKSUM_JOB_NAME) == 8
    assert resolve_job_cores(job_name=FORGING_JOB_NAME) == 1


def test_an_unregistered_job_type_has_no_declared_cores() -> None:
    """Verifies every job type resolved by a pipeline must declare its width before the batch tools may dispatch it."""
    with pytest.raises(ValueError, match="Unable to resolve the cores for job type 'unregistered'"):
        resolve_job_cores(job_name="unregistered")


def test_only_the_types_that_declare_one_carry_a_concurrency_ceiling() -> None:
    """Verifies that an absent name reads as bounded by the two budgets alone, which is what keeps the mapping a narrow
    statement.
    """
    limits = resolve_concurrency_limits(job_names={"motion_energy", CHECKSUM_JOB_NAME, RUNTIME_JOB_NAME})

    # Named against the declaration table rather than against frozen values, so retuning a ceiling leaves this alone.
    assert set(limits) == {"motion_energy", CHECKSUM_JOB_NAME}
    assert RUNTIME_JOB_NAME not in limits
    assert all(limit >= 1 for limit in limits.values())


def test_only_the_types_that_declare_one_carry_a_concurrency_reservation() -> None:
    """Verifies that a reserved type widens once nothing else claims its room, so only the types that give room up are
    recorded.
    """
    reservations = resolve_concurrency_reservations(
        job_names={str(SingleRecordingJobNames.REGISTER), CHECKSUM_JOB_NAME}
    )

    assert reservations == {str(SingleRecordingJobNames.REGISTER): 4}


def test_the_microcontroller_command_names_the_job_the_allocation_runs() -> None:
    """Verifies one dispatch table states both how a job runs in-process and how it runs as a scheduled allocation."""
    job = build_pending_job(job=build_descriptor(job_name="extraction", pipeline="microcontroller", cores=8))

    assert resolve_job_command(job=job) == (
        "slf",
        "process",
        "-sp",
        _SESSION_UNIT_PATH,
        "-w",
        "8",
        "-np",
        "-id",
        "a_job",
        "microcontroller",
    )


def test_rendering_a_command_for_an_unsupported_pipeline_is_rejected() -> None:
    """Verifies that a descriptor naming a pipeline absent from the table names no command a host could run."""
    job = build_pending_job(job=build_descriptor(pipeline="not_a_pipeline"))

    with pytest.raises(ValueError, match="Unable to render the command for job 'a_job'"):
        resolve_job_command(job=job)


@pytest.mark.xdist_group(name="worker_pool")
def test_running_a_job_of_an_unsupported_pipeline_is_rejected(pinned_thread_environment: None) -> None:
    """Verifies that the shared worker routes on the pipeline the job carries, so an unroutable job stops rather than
    running.
    """
    job = build_pending_job(job=build_descriptor(pipeline="not_a_pipeline"))

    with pytest.raises(ValueError, match="Unable to run batch job 'a_job'"):
        run_batch_job(job=job)


@pytest.mark.parametrize(
    ("pipeline", "job_name", "unit_path", "expected"),
    [
        (
            "checksum",
            "checksum_resolution",
            _SESSION_UNIT_PATH,
            {"session_path": Path(_SESSION_UNIT_PATH), "regenerate_checksum": False, "workers": 4},
        ),
        ("runtime", "runtime_processing", _SESSION_UNIT_PATH, {"session_path": Path(_SESSION_UNIT_PATH), "workers": 4}),
        (
            "microcontroller",
            "extraction",
            _SESSION_UNIT_PATH,
            {"session_path": Path(_SESSION_UNIT_PATH), "job_id": "a_job", "workers": 4},
        ),
        (
            "video",
            "motion_energy",
            _SESSION_UNIT_PATH,
            {"session_path": Path(_SESSION_UNIT_PATH), "job_id": "a_job", "workers": 4},
        ),
        (
            "two_photon",
            "binarize",
            _SESSION_UNIT_PATH,
            {"session_path": Path(_SESSION_UNIT_PATH), "job_id": "a_job", "workers": 4},
        ),
        (
            "forging",
            FORGING_JOB_NAME,
            _DATASET_UNIT_PATH,
            {
                "name": "Dataset",
                "project_root": Path("/data/Project"),
                "job_id": "a_job",
                "workers": 4,
            },
        ),
    ],
)
@pytest.mark.xdist_group(name="worker_pool")
def test_each_pipeline_worker_forwards_the_job_to_its_own_entry_point(
    pipeline: str,
    job_name: str,
    unit_path: str,
    expected: dict[str, Any],
    recorded_pipeline_calls: dict[str, dict[str, Any]],
    pinned_thread_environment: None,
) -> None:
    """Verifies that the pipeline a job names is what selects the stage it runs and that stage's width."""
    job = build_pending_job(job=build_descriptor(job_name=job_name, pipeline=pipeline, unit_path=unit_path))

    run_batch_job(job=job)

    assert recorded_pipeline_calls == {pipeline: expected}
    assert os.environ["TIFFFILE_NUM_THREADS"] == "4"


@pytest.mark.xdist_group(name="worker_pool")
def test_the_checksum_worker_takes_its_mode_from_the_options_the_job_carries(
    recorded_pipeline_calls: dict[str, dict[str, Any]],
    pinned_thread_environment: None,
) -> None:
    """Verifies that re-baselining is a deliberate correction, so a batch runs it only when it was prepared with that
    mode.
    """
    job = build_pending_job(
        job=build_descriptor(job_name="checksum_resolution", pipeline="checksum", options={"regenerate_checksum": True})
    )

    run_batch_job(job=job)

    assert recorded_pipeline_calls["checksum"]["regenerate_checksum"]


def test_a_session_pipeline_loads_the_session_its_jobs_operate_on(experiment_session: SessionData) -> None:
    """Verifies that a caller needing only the unit's own locations loads it without running any job resolution."""
    dispatch = resolve_dispatch(pipeline="runtime")
    assert dispatch is not None

    loaded = dispatch.load(experiment_session.raw_data_path.parent)

    assert isinstance(loaded, SessionData)
    assert loaded.session_name == experiment_session.session_name
    assert dispatch.unit_name(loaded) == experiment_session.session_name


def test_the_forging_pipeline_loads_the_dataset_its_jobs_operate_on(project: ProjectData) -> None:
    """Verifies that a dataset job resolves its own hierarchy, so the dispatch entry loads a dataset rather than a
    session.
    """
    dataset = DatasetData.create(
        name="test_dataset",
        project=project.project_name,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=(DatasetSession(session="2026-01-02-03-04-05-000006", animal="305", session_path=project.path),),
        datasets_root=project.path,
        column_descriptions={},
    )
    dispatch = resolve_dispatch(pipeline="forging")
    assert dispatch is not None

    loaded = dispatch.load(dataset.dataset_data_path.parent)

    assert isinstance(loaded, DatasetData)
    assert dispatch.unit_name(loaded) == "test_dataset"
    assert dispatch.output_path(loaded) == dataset.dataset_data_path.parent


def test_every_dispatch_entry_declares_the_unit_its_jobs_operate_on() -> None:
    """Verifies that the unit a pipeline processes is a property of its entry, so a second dataset pipeline is scoped
    correctly without being named at any of the sites that resolve a unit.
    """
    assert [dispatch.pipeline.value for dispatch in resolve_unit_dispatches(unit_kind=DATASET_UNIT)] == ["forging"]
    assert [dispatch.pipeline.value for dispatch in resolve_unit_dispatches(unit_kind=SESSION_UNIT)] == [
        "checksum",
        "runtime",
        "microcontroller",
        "video",
        "two_photon",
    ]
    assert resolve_unit_kind(pipeline="forging") == DATASET_UNIT
    assert resolve_unit_kind(pipeline="video") == SESSION_UNIT


def test_a_pipeline_outside_the_dispatch_table_is_scoped_to_a_session() -> None:
    """Verifies that a batch naming a pipeline the table does not carry still resolves a project root, since this
    library lays out the artifacts of an unrecognized pipeline per session.
    """
    assert resolve_unit_kind(pipeline="unregistered_pipeline") == SESSION_UNIT


def test_the_dispatch_table_is_held_to_the_pipelines_the_batch_tools_advertise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that a pipeline advertised by the tools without a dispatch entry would fail at preparation rather
    than at registration.
    """
    monkeypatch.setattr(dispatch_module, "_pipeline_dispatch", dict)

    with pytest.raises(RuntimeError, match="Unable to validate the pipeline dispatch table"):
        dispatch_module._assert_dispatch_coverage()


def test_the_dispatch_table_is_held_to_the_unit_kinds_its_callers_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an entry declaring a unit nothing resolves fails at registration rather than planning against a
    project root that sits at the wrong depth.
    """
    forging = resolve_dispatch(pipeline="forging")
    assert forging is not None
    table = dict(dispatch_module._pipeline_dispatch())
    table[forging.pipeline] = replace(forging, unit_kind="project")
    monkeypatch.setattr(dispatch_module, "_pipeline_dispatch", lambda: table)

    with pytest.raises(RuntimeError, match="must declare one of"):
        dispatch_module._assert_dispatch_coverage()
