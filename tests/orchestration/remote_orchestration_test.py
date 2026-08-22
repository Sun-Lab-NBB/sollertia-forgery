"""Tests the remote execution backend: the batch document, the dependency-graph submission, the rendered job
commands, the batch script the scheduler runs, and the durable ledger of what was submitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import field, dataclass

import polars as pl
import pytest
from sollertia_shared_assets import DATASET_MARKER_FILENAME

from sollertia_forgery.server import Job, JobStatus
from sollertia_forgery.forging import DATASET_STATE_SCHEMA, DATASET_STATE_FILENAME
from sollertia_forgery.managing import PROJECT_JOBS_SCHEMA, project_jobs_path
from sollertia_forgery.orchestration import (
    PROJECT_PLAN_SCHEMA,
    REMOTE_JOB_WALLTIME_MINUTES,
    SubmissionBatch,
    RemoteSubmission,
    ledger_path,
    read_ledger,
    record_batch,
    submit_batch,
    forget_batches,
    resolve_batches,
    connect_to_server,
    project_plan_path,
    query_submissions,
    render_submission,
    cancel_submissions,
    sync_project_state,
    environment_command,
    resolve_job_command,
    prepare_remote_batch,
    retire_settled_batches,
)
from sollertia_forgery.server.server import _parse_job_status
from sollertia_forgery.orchestration.graph import (
    BatchDocument,
    build_pending_job,
    build_batch_document,
    partition_blocked_jobs,
    resolve_submission_order,
)
from sollertia_forgery.orchestration.preparation import resolve_project_root

if TYPE_CHECKING:
    from sollertia_forgery.server import Server
    from sollertia_forgery.server.server_configuration import ServerConfiguration

SERVER_PROJECT_ROOT: Path = Path("/data/sollertia/TestProject")
"""The project directory every server-side preparation test addresses on the stubbed compute server."""

pytestmark: pytest.MarkDecorator = pytest.mark.usefixtures("isolated_working_directory")
"""Points every test in this module at an isolated platform working directory, which the ledger is written under."""


@dataclass
class StubServer:
    """Stands in for a connected compute server, recording the jobs it is asked to submit."""

    root: Path = Path("/server/root")
    """The server's data root."""
    environment: str = "slf_server"
    """The environment every submitted script activates."""
    submitted: list[Job] = field(default_factory=list)
    """The jobs the server accepted, in submission order."""
    created: list[Path] = field(default_factory=list)
    """The directories the server was asked to create."""

    def create(self, remote_path: Path, *, is_dir: bool = True, parents: bool = True) -> None:  # noqa: ARG002
        """Records a directory creation request."""
        self.created.append(remote_path)

    def submit_job(self, job: Job, *, verbose: bool = False) -> Job:  # noqa: ARG002
        """Assigns a sequential allocation identifier the way a scheduler would."""
        job.job_id = str(1000 + len(self.submitted))
        self.submitted.append(job)
        return job


def build_document(
    pipeline: str, plan: pl.DataFrame, state: pl.DataFrame, unit_paths: list[Path], options: dict[str, Any]
) -> BatchDocument:
    """Builds a batch document from the stand-in tables, matching how preparation reads its own artifacts."""
    return build_batch_document(
        pipeline=pipeline,
        host="remote",
        unit_column="dataset" if pipeline == "forging" else "session",
        plan_rows=plan.to_dicts(),
        state_rows=state.to_dicts(),
        unit_paths=unit_paths,
        options=options,
    )


def build_descriptor(
    job_id: str,
    job_name: str,
    specifier: str = "",
    prerequisite_ids: tuple[str, ...] = (),
    pipeline: str = "video",
    unit_path: str = "/data/Project/Animal/Session",
    cores: int = 2,
    memory_mb: int = 4096,
) -> dict[str, Any]:
    """Builds one job descriptor of the shape preparation emits."""
    return {
        "job_id": job_id,
        "job_name": job_name,
        "specifier": specifier,
        "unit_path": unit_path,
        "unit_name": Path(unit_path).name,
        "pipeline": pipeline,
        "cores": cores,
        "memory_mb": memory_mb,
        "prerequisite_ids": list(prerequisite_ids),
        "options": {},
    }


def build_submission(slurm_job_id: str, job_id: str = "job") -> RemoteSubmission:
    """Builds one recorded submission."""
    return RemoteSubmission(
        job_id=job_id,
        slurm_job_id=slurm_job_id,
        slurm_job_name=f"0000-Session-{job_id}",
        pipeline="video",
        job_name="motion_energy",
        specifier="1",
        unit_path="/data/Project/Animal/Session",
        unit_name="Session",
        cores=16,
        memory_mb=4096,
        output_log="/server/root/processing_batches/batch01/0000.out",
        error_log="/server/root/processing_batches/batch01/0000.err",
    )


def build_batch(batch_id: str, submissions: list[RemoteSubmission], submitted_at: int = 1) -> SubmissionBatch:
    """Builds one recorded batch holding the given submissions."""
    return SubmissionBatch(
        batch_id=batch_id,
        batch_directory=f"/server/root/processing_batches/{batch_id}",
        submitted_at=submitted_at,
        walltime_minutes=REMOTE_JOB_WALLTIME_MINUTES,
        submissions=submissions,
    )


def build_plan_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Builds a project plan table from partial rows, filling the columns a row omits."""
    return pl.DataFrame(
        data=[
            {
                "unit_kind": "session",
                "animal": "305",
                "session": "2024_11_04",
                "dataset": None,
                "pipeline": "video",
                "job_name": "motion_energy",
                "specifier": "1",
                "cores": 16,
                "memory_mb": 4096,
                "prerequisite_ids": [],
                **row,
            }
            for row in rows
        ],
        schema=PROJECT_PLAN_SCHEMA,
        strict=False,
    )


def build_state_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Builds a project job table from partial rows, filling the columns a row omits."""
    return pl.DataFrame(
        data=[
            {
                "animal": "305",
                "session": "2024_11_04",
                "pipeline": "video",
                "job_name": "motion_energy",
                "specifier": "1",
                "status": "SCHEDULED",
                "executor_id": None,
                "error_message": None,
                "started_at": None,
                "completed_at": None,
                **row,
            }
            for row in rows
        ],
        schema=PROJECT_JOBS_SCHEMA,
        strict=False,
    )


def place_server_table(transport: Any, remote_path: Path, frame: pl.DataFrame) -> Path:
    """Writes one table into the stubbed server's own filesystem.

    Args:
        transport: The stubbed transport whose temporary tree stands in for the server.
        remote_path: The absolute server path the table sits at.
        frame: The table to write.

    Returns:
        The local path the table was written to.
    """
    local_path = transport.local_path(remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_ipc(file=local_path, compression="uncompressed")
    return local_path


def build_dataset_state_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Builds a dataset state table from partial rows, filling the columns a row omits.

    Args:
        rows: The partial rows, each overriding the shared defaults.

    Returns:
        The dataset state table.
    """
    return pl.DataFrame(
        data=[
            {
                "dataset": "Dataset",
                "animal": "305",
                "session": "2024_11_04",
                "scope": "session",
                "job_name": "session_data_assembly",
                "specifier": "2024_11_04",
                "status": "SCHEDULED",
                "executor_id": None,
                "error_message": None,
                "started_at": None,
                "completed_at": None,
                **row,
            }
            for row in rows
        ],
        schema=DATASET_STATE_SCHEMA,
        strict=False,
    )


def test_a_job_whose_upstream_stage_cannot_run_is_blocked_rather_than_submitted() -> None:
    """A prerequisite that neither runs in this batch nor already succeeded blocks the job waiting on it."""
    jobs = [
        build_descriptor(job_id="rename", job_name="rename", prerequisite_ids=("timestamp_absent",)),
        build_descriptor(job_id="energy", job_name="motion_energy", specifier="1"),
    ]

    submittable, blocked = partition_blocked_jobs(jobs=jobs, succeeded=set())

    assert [job["job_id"] for job in submittable] == ["energy"]
    assert [entry["job_id"] for entry in blocked] == ["rename"]
    assert blocked[0]["unsatisfied_prerequisite_ids"] == ["timestamp_absent"]


def test_blocking_propagates_down_the_chain() -> None:
    """A stage waiting on a blocked stage is blocked in turn, so the whole chain stays out of the batch."""
    jobs = [
        build_descriptor(job_id="root", job_name="binarize", prerequisite_ids=("missing",)),
        build_descriptor(job_id="middle", job_name="process", prerequisite_ids=("root",)),
        build_descriptor(job_id="leaf", job_name="combine", prerequisite_ids=("middle",)),
    ]

    submittable, blocked = partition_blocked_jobs(jobs=jobs, succeeded=set())

    assert submittable == []
    assert {entry["job_id"] for entry in blocked} == {"root", "middle", "leaf"}


def test_a_prerequisite_the_batch_holds_does_not_block_its_dependent() -> None:
    """A job whose upstream stage is queued alongside it is submittable, since the batch produces that stage."""
    jobs = [
        build_descriptor(job_id="timestamp", job_name="timestamp", specifier="1"),
        build_descriptor(job_id="rename", job_name="rename", prerequisite_ids=("timestamp",)),
    ]

    submittable, blocked = partition_blocked_jobs(jobs=jobs, succeeded=set())

    assert {job["job_id"] for job in submittable} == {"timestamp", "rename"}
    assert blocked == []


def test_jobs_are_ordered_so_every_job_follows_the_jobs_it_waits_on() -> None:
    """Ordering by dependency depth is what lets each submission name an allocation the scheduler already assigned."""
    jobs = [
        build_pending_job(job=build_descriptor(job_id="combine", job_name="combine", prerequisite_ids=("process",))),
        build_pending_job(job=build_descriptor(job_id="process", job_name="process", prerequisite_ids=("binarize",))),
        build_pending_job(job=build_descriptor(job_id="binarize", job_name="binarize")),
    ]

    ordered = [job.job_id for job in resolve_submission_order(jobs=jobs)]

    assert ordered == ["binarize", "process", "combine"]


def test_a_cyclic_ordering_resolves_rather_than_recursing_without_end() -> None:
    """A malformed pipeline is submitted in a poor order, because ordering falls back to a fixed traversal when the
    graph carries a cycle.
    """
    jobs = [
        build_pending_job(job=build_descriptor(job_id="first", job_name="first", prerequisite_ids=("second",))),
        build_pending_job(job=build_descriptor(job_id="second", job_name="second", prerequisite_ids=("first",))),
    ]

    assert {job.job_id for job in resolve_submission_order(jobs=jobs)} == {"first", "second"}


def test_a_submission_names_the_allocations_of_the_upstream_jobs_the_batch_holds() -> None:
    """Each job's dependency directive names the identifiers the scheduler assigned to its upstream jobs."""
    server = StubServer()
    jobs = [
        build_descriptor(job_id="rename", job_name="rename", prerequisite_ids=("timestamp",)),
        build_descriptor(job_id="timestamp", job_name="timestamp", specifier="1"),
    ]

    submissions = submit_batch(server=server, jobs=jobs, batch_id="batch01")

    assert [submission.job_id for submission in submissions] == ["timestamp", "rename"]
    assert submissions[0].slurm_job_id == "1000"
    assert "#SBATCH --dependency=afterok:1000" in server.submitted[1].command_script
    assert "--dependency" not in server.submitted[0].command_script


def test_a_submission_requests_the_cores_and_memory_the_job_was_prepared_at() -> None:
    """Sizing each allocation from its own estimate is what keeps one large job from reserving its footprint for all."""
    server = StubServer()
    jobs = [build_descriptor(job_id="energy", job_name="motion_energy", specifier="1", cores=16, memory_mb=5000)]

    submit_batch(server=server, jobs=jobs, batch_id="batch01")
    script = server.submitted[0].command_script

    assert "#SBATCH --cpus-per-task=16" in script
    # 5000 megabytes rounds up to five gigabytes, since understating a request kills the allocation outright.
    assert "#SBATCH --mem=5G" in script
    assert "#SBATCH --time=08:00:00" in script


def test_every_allocation_writes_into_the_batch_directory() -> None:
    """One directory per batch holds the scripts and logs, so a submission creates one directory rather than many."""
    server = StubServer()
    jobs = [build_descriptor(job_id="energy", job_name="motion_energy", specifier="1")]

    submissions = submit_batch(server=server, jobs=jobs, batch_id="batch01")

    expected = server.root.joinpath("processing_batches", "batch01")
    assert server.created == [expected]
    assert submissions[0].output_log.startswith(str(expected))
    assert submissions[0].error_log.startswith(str(expected))


@pytest.mark.parametrize(
    ("pipeline", "job_name", "expected"),
    [
        ("checksum", "checksum", ("slf", "checksum", "-sp", "/data/Project/Animal/Session", "-w", "2", "-np")),
        ("runtime", "runtime", ("slf", "process", "-sp", "/data/Project/Animal/Session", "-w", "2", "-np", "runtime")),
        (
            "video",
            "motion_energy",
            ("slf", "process", "-sp", "/data/Project/Animal/Session", "-w", "2", "-np", "-id", "job", "video"),
        ),
        (
            "two_photon",
            "binarize",
            ("slf", "process", "-sp", "/data/Project/Animal/Session", "-w", "2", "-np", "-id", "job", "two-photon"),
        ),
    ],
)
def test_each_pipeline_renders_the_command_that_runs_one_of_its_jobs(
    pipeline: str, job_name: str, expected: tuple[str, ...]
) -> None:
    """One dispatch table states both how a job runs in-process and how it runs as a scheduled allocation."""
    job = build_pending_job(job=build_descriptor(job_id="job", job_name=job_name, pipeline=pipeline))

    assert resolve_job_command(job=job) == expected


def test_the_checksum_command_carries_the_mode_the_job_was_prepared_with() -> None:
    """A pipeline's options reach its command line, so a batch runs the mode it was prepared for."""
    prepared = build_descriptor(job_id="job", job_name="checksum", pipeline="checksum")
    prepared["options"] = {"regenerate_checksum": True}

    assert "-rc" in resolve_job_command(job=build_pending_job(job=prepared))


def test_the_forging_command_names_the_dataset_and_its_project_root() -> None:
    """A forging job resolves its dataset from the unit directory and its project from that directory's parent."""
    job = build_pending_job(
        job=build_descriptor(
            job_id="job",
            job_name="session_data_assembly",
            pipeline="forging",
            unit_path="/data/Project/Dataset",
        )
    )

    assert resolve_job_command(job=job) == (
        "slf",
        "forge",
        "-dn",
        "Dataset",
        "-pp",
        "/data/Project",
        "-id",
        "job",
        "-w",
        "2",
        "-np",
    )


def test_the_script_exits_with_the_status_of_the_work_it_ran(tmp_path: Path) -> None:
    """Error checking precedes the payload and cleanup runs through a trap, so a failure is never masked."""
    job = Job(
        job_name="job",
        output_log=tmp_path.joinpath("job.out"),
        error_log=tmp_path.joinpath("job.err"),
        working_directory=tmp_path,
        conda_environment="slf_server",
    )
    job.add_command("slf process -sp /data/Project/Animal/Session runtime")
    script = job.command_script.splitlines()

    assert script[-1] == "slf process -sp /data/Project/Animal/Session runtime"
    assert "set -eo pipefail" in script
    assert script.index("set -eo pipefail") < script.index("slf process -sp /data/Project/Animal/Session runtime")
    assert any(line.startswith("trap ") and "rm -f" in line for line in script)
    # Activation precedes error checking, since a conda hook's exit status does not describe whether the environment
    # is usable.
    assert script.index("source activate slf_server") < script.index("set -eo pipefail")


def test_rendering_a_script_twice_produces_the_same_script(tmp_path: Path) -> None:
    """Rendering does not mutate the job, so a retried submission uploads the script the first one would have."""
    job = Job(
        job_name="job",
        output_log=tmp_path.joinpath("job.out"),
        error_log=tmp_path.joinpath("job.err"),
        working_directory=tmp_path,
        conda_environment="slf_server",
    )
    job.add_command("slf checksum -sp /data/Project/Animal/Session")

    first_render = job.command_script
    second_render = job.command_script

    assert second_render == first_render


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("COMPLETED", JobStatus.COMPLETED),
        ("CANCELLED by 1234", JobStatus.CANCELLED),
        ("CANCELLED+", JobStatus.CANCELLED),
        ("OUT_OF_MEMORY", JobStatus.OUT_OF_MEMORY),
        ("SOMETHING_ELSE", JobStatus.UNKNOWN),
    ],
)
def test_a_decorated_accounting_state_resolves_to_the_state_it_names(state: str, expected: JobStatus) -> None:
    """Accounting decorates some states with a marker or an attribution clause, which names the same state."""
    assert _parse_job_status(state=state) == expected


def test_a_command_runs_inside_the_shared_server_environment() -> None:
    """A command issued over the connection activates the environment a submitted job script activates."""
    rendered = environment_command(environment="slf_server", command=["slf", "manifest", "-pp", "/a b/Project"])

    assert "conda shell.bash hook" in rendered
    assert "source activate slf_server" in rendered
    # The project path carries a space, so quoting is what keeps it one argument.
    assert "'/a b/Project'" in rendered


# Tests for the submission ledger


def test_the_ledger_round_trips_through_yaml() -> None:
    """A recorded batch reads back with every field it was written with, two dataclass levels deep."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1000")]))

    recorded = read_ledger().resolve_batch(batch_id="batch01")

    assert ledger_path().is_file()
    assert recorded is not None
    assert recorded.walltime_minutes == REMOTE_JOB_WALLTIME_MINUTES
    assert recorded.submissions[0] == build_submission(slurm_job_id="1000")


def test_a_submitted_batch_is_recorded_so_it_outlives_the_process_that_submitted_it() -> None:
    """A submission returns as soon as the scheduler accepts it, so the record has to survive on disk."""
    server = StubServer()
    jobs = [build_descriptor(job_id="energy", job_name="motion_energy", specifier="1")]

    submit_batch(server=server, jobs=jobs, batch_id="batch01")

    recorded = read_ledger().resolve_batch(batch_id="batch01")
    assert recorded is not None
    assert [entry.slurm_job_id for entry in recorded.submissions] == ["1000"]
    assert recorded.batch_directory == "/server/root/processing_batches/batch01"


def test_allocations_accepted_before_a_rejection_are_still_recorded() -> None:
    """A rejection leaves the earlier allocations queued, so recording only on a clean pass would orphan them."""

    @dataclass
    class RejectingServer(StubServer):
        """Accepts two submissions and then rejects, the way a scheduler refusing one job of a batch would."""

        def submit_job(self, job: Job, *, verbose: bool = False) -> Job:
            """Rejects the third submission after accepting the first two."""
            accepted_before_rejection = 2
            if len(self.submitted) >= accepted_before_rejection:
                message = "sbatch rejected the job."
                raise RuntimeError(message)
            return super().submit_job(job=job, verbose=verbose)

    jobs = [
        build_descriptor(job_id=f"job{index}", job_name="motion_energy", specifier=str(index)) for index in range(4)
    ]

    with pytest.raises(RuntimeError, match="sbatch rejected the job"):
        submit_batch(server=RejectingServer(), jobs=jobs, batch_id="batch01")

    recorded = read_ledger().resolve_batch(batch_id="batch01")
    assert recorded is not None
    assert [entry.slurm_job_id for entry in recorded.submissions] == ["1000", "1001"]


def test_recording_a_batch_preserves_the_batches_already_in_the_ledger() -> None:
    """Concurrent runs are both queryable, which is the whole reason the record is durable."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1000")]))
    record_batch(batch=build_batch(batch_id="batch02", submissions=[build_submission(slurm_job_id="2000")]))

    assert [recorded.batch_id for recorded in read_ledger().batches] == ["batch01", "batch02"]


def test_re_recording_a_batch_replaces_its_earlier_record() -> None:
    """A batch identifier names one submission, so recording it again supersedes rather than duplicates."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1000")]))
    record_batch(
        batch=build_batch(
            batch_id="batch01",
            submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
        )
    )

    ledger = read_ledger()
    assert [recorded.batch_id for recorded in ledger.batches] == ["batch01"]
    assert [entry.slurm_job_id for entry in ledger.batches[0].submissions] == ["1000", "1001"]


def test_a_batch_is_retired_once_every_allocation_reaches_a_terminal_state() -> None:
    """The ledger names outstanding allocations alone, so a finished batch has nothing left to answer."""
    record_batch(
        batch=build_batch(
            batch_id="batch01",
            submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
        )
    )

    retired = retire_settled_batches(statuses={"1000": JobStatus.COMPLETED, "1001": JobStatus.FAILED})

    assert retired == ["batch01"]
    assert read_ledger().batches == []


def test_a_partly_finished_batch_is_retained() -> None:
    """A batch is retired as a whole, so one finished allocation never drops the ones still running beside it."""
    record_batch(
        batch=build_batch(
            batch_id="batch01",
            submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
        )
    )

    retired = retire_settled_batches(statuses={"1000": JobStatus.COMPLETED, "1001": JobStatus.RUNNING})

    assert retired == []
    assert read_ledger().resolve_batch(batch_id="batch01") is not None


def test_an_allocation_the_query_did_not_cover_keeps_its_batch() -> None:
    """A partial query never retires a batch it did not fully observe, so an unqueried job is not assumed finished."""
    record_batch(
        batch=build_batch(
            batch_id="batch01",
            submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
        )
    )

    retired = retire_settled_batches(statuses={"1000": JobStatus.COMPLETED})

    assert retired == []
    assert read_ledger().resolve_batch(batch_id="batch01") is not None


def test_a_blocked_allocation_retires_its_batch() -> None:
    """A dependency that can never be satisfied is terminal, so it must not hold a batch open forever."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1000")]))

    assert retire_settled_batches(statuses={"1000": JobStatus.BLOCKED}) == ["batch01"]


def test_retirement_leaves_the_batches_that_are_still_running() -> None:
    """Concurrent runs are independent, so retiring one batch never disturbs another."""
    record_batch(batch=build_batch(batch_id="done", submissions=[build_submission(slurm_job_id="1")]))
    record_batch(batch=build_batch(batch_id="running", submissions=[build_submission(slurm_job_id="2")]))

    retire_settled_batches(statuses={"1": JobStatus.COMPLETED, "2": JobStatus.RUNNING})

    assert [recorded.batch_id for recorded in read_ledger().batches] == ["running"]


def test_naming_no_batch_resolves_every_outstanding_batch() -> None:
    """A status read covers what is still running rather than only the newest submission."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1")]))
    record_batch(batch=build_batch(batch_id="batch02", submissions=[build_submission(slurm_job_id="2")]))

    assert [recorded.batch_id for recorded in resolve_batches(ledger=read_ledger())] == ["batch01", "batch02"]


def test_naming_a_batch_narrows_the_report_to_it() -> None:
    """Naming one of several outstanding batches is how a single run is followed."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1")]))
    record_batch(batch=build_batch(batch_id="batch02", submissions=[build_submission(slurm_job_id="2")]))

    resolved = resolve_batches(ledger=read_ledger(), batch_ids=["batch02"])

    assert [recorded.batch_id for recorded in resolved] == ["batch02"]


def test_forgetting_a_batch_drops_it_from_the_ledger() -> None:
    """Dropping a batch is explicit, so a record is never removed as a side effect of reading it."""
    record_batch(batch=build_batch(batch_id="batch01", submissions=[build_submission(slurm_job_id="1000")]))
    record_batch(batch=build_batch(batch_id="batch02", submissions=[build_submission(slurm_job_id="2000")]))

    dropped = forget_batches(batch_ids=["batch01", "absent"])

    assert dropped == ["batch01"]
    assert [recorded.batch_id for recorded in read_ledger().batches] == ["batch02"]


def test_an_absent_ledger_reads_as_holding_no_batches() -> None:
    """A host that has never submitted anything reads cleanly rather than failing."""
    assert read_ledger().batches == []


# Tests for resolving a batch out of the project's own tables


def test_a_session_unit_resolves_the_project_two_levels_up() -> None:
    """A session sits under its animal, so its project is two directories above it."""
    resolved = resolve_project_root(unit_paths=[Path("/root/Project/305/2024_11_04")], unit_kind="session")

    assert resolved == Path("/root/Project")


def test_a_dataset_unit_resolves_the_project_one_level_up() -> None:
    """A dataset sits directly under its project root, unlike a session."""
    resolved = resolve_project_root(unit_paths=[Path("/root/Project/Dataset")], unit_kind="dataset")

    assert resolved == Path("/root/Project")


def test_units_spanning_two_projects_are_rejected() -> None:
    """The tables a batch is resolved from are written per project, so one batch reads one project."""
    with pytest.raises(ValueError, match="belong to the same project"):
        resolve_project_root(
            unit_paths=[Path("/root/ProjectA/305/2024_11_04"), Path("/root/ProjectB/306/2024_11_05")],
            unit_kind="session",
        )


def test_a_batch_joins_the_state_table_to_the_planned_figures() -> None:
    """State names which jobs exist and the plan sizes them, which is the whole descriptor."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame([{"job_id": "energy", "cores": 16, "memory_mb": 5000}]),
        state=build_state_frame([{"job_id": "energy"}]),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
    )

    assert len(document.jobs) == 1
    job = document.jobs[0]
    assert job["cores"] == 16
    assert job["memory_mb"] == 5000
    assert job["unit_path"] == "/root/Project/305/2024_11_04"
    assert job["unit_name"] == "2024_11_04"


def test_a_succeeded_job_is_not_submitted_again() -> None:
    """The state table is what makes a run resumable, so a job it records as succeeded stays done."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame([{"job_id": "done"}, {"job_id": "todo", "specifier": "2"}]),
        state=build_state_frame([{"job_id": "done", "status": "SUCCEEDED"}, {"job_id": "todo", "specifier": "2"}]),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
    )

    assert [job["job_id"] for job in document.jobs] == ["todo"]


def test_a_job_absent_from_the_state_table_is_never_submitted() -> None:
    """A job the unit cannot run never reaches a tracker, so its absence from state is what rules it out."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame([{"job_id": "possible"}, {"job_id": "impossible", "specifier": "9"}]),
        state=build_state_frame([{"job_id": "possible"}]),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
    )

    assert [job["job_id"] for job in document.jobs] == ["possible"]


def test_a_unit_the_state_table_does_not_cover_is_reported_without_aborting_the_others() -> None:
    """One unit carrying none of a pipeline's data never stops the units that do."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame([{"job_id": "energy"}]),
        state=build_state_frame([{"job_id": "energy"}]),
        unit_paths=[Path("/root/Project/305/2024_11_04"), Path("/root/Project/306/2024_11_05")],
        options={},
    )

    covered, uncovered = document.units
    assert covered["job_count"] == 1
    assert "error" in uncovered
    assert len(document.jobs) == 1


def test_an_outstanding_job_the_plan_does_not_size_is_reported_as_an_error() -> None:
    """A job must be planned before it can be sized for a scheduler, so an unplanned one stops its unit."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame([]),
        state=build_state_frame([{"job_id": "energy"}]),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
    )

    assert document.jobs == []
    assert "carries no figures" in document.units[0]["error"]


def test_the_planned_ordering_reaches_the_descriptor() -> None:
    """The plan carries the edges, which is what lets a scheduler build the graph without resolving the unit."""
    document = build_document(
        pipeline="video",
        plan=build_plan_frame(
            [
                {"job_id": "timestamp", "job_name": "timestamp"},
                {"job_id": "rename", "job_name": "rename", "specifier": "", "prerequisite_ids": ["timestamp"]},
            ]
        ),
        state=build_state_frame(
            [
                {"job_id": "timestamp", "job_name": "timestamp"},
                {"job_id": "rename", "job_name": "rename", "specifier": ""},
            ]
        ),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
    )

    rename = next(job for job in document.jobs if job["job_id"] == "rename")
    assert rename["prerequisite_ids"] == ["timestamp"]


def test_options_are_stamped_onto_every_descriptor() -> None:
    """A pipeline's mode rides on the descriptor, so every job of the batch runs the mode it was prepared for."""
    document = build_document(
        pipeline="checksum",
        plan=build_plan_frame([{"job_id": "sum", "pipeline": "checksum", "job_name": "checksum", "specifier": ""}]),
        state=build_state_frame([{"job_id": "sum", "pipeline": "checksum", "job_name": "checksum", "specifier": ""}]),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={"regenerate_checksum": True},
    )

    assert document.jobs[0]["options"] == {"regenerate_checksum": True}


def test_one_unit_completed_stage_never_satisfies_another_units_dependent() -> None:
    """A job identifier is derived from the job name and specifier alone, so the unit is what separates two copies."""
    first = build_pending_job(job=build_descriptor(job_id="a", job_name="rename", unit_path="/root/Project/305/one"))
    second = build_pending_job(job=build_descriptor(job_id="a", job_name="rename", unit_path="/root/Project/305/two"))

    assert first.dispatch_key != second.dispatch_key


def test_a_descriptor_carries_the_tracker_location_the_host_resolved() -> None:
    """The local engine opens these files directly, so a batch dispatched here must carry where they sit."""
    document = build_batch_document(
        pipeline="video",
        host="local",
        unit_column="session",
        plan_rows=build_plan_frame([{"job_id": "energy"}]).to_dicts(),
        state_rows=build_state_frame([{"job_id": "energy"}]).to_dicts(),
        unit_paths=[Path("/root/Project/305/2024_11_04")],
        options={},
        tracker_paths={"/root/Project/305/2024_11_04": "/root/Project/305/2024_11_04/processed_data/video.yaml"},
    )

    assert document.jobs[0]["tracker_path"] == "/root/Project/305/2024_11_04/processed_data/video.yaml"


def test_a_descriptor_carries_no_tracker_location_when_the_host_resolves_none() -> None:
    """A remotely dispatched job records its own outcome on the server, so naming a path here would mislead."""
    document = build_batch_document(
        pipeline="video",
        host="remote",
        unit_column="session",
        plan_rows=build_plan_frame([{"job_id": "energy"}]).to_dicts(),
        state_rows=build_state_frame([{"job_id": "energy"}]).to_dicts(),
        unit_paths=[Path("/data/Project/305/2024_11_04")],
        options={},
    )

    assert document.jobs[0]["tracker_path"] == ""


# Tests for preparing a batch against the remote compute server


def test_a_remote_batch_is_resolved_from_the_projects_own_artifacts(
    connected_server: Server, stub_ssh_transport: Any
) -> None:
    """A remote batch is resolved by the same code a local one is, so only the host that materializes it differs."""
    session_path = SERVER_PROJECT_ROOT.joinpath("305", "2024_11_04")
    place_server_table(
        transport=stub_ssh_transport,
        remote_path=project_plan_path(project_directory=SERVER_PROJECT_ROOT),
        frame=build_plan_frame([{"job_id": "energy", "cores": 16, "memory_mb": 5000}]),
    )
    place_server_table(
        transport=stub_ssh_transport,
        remote_path=project_jobs_path(project_directory=SERVER_PROJECT_ROOT),
        frame=build_state_frame([{"job_id": "energy"}]),
    )

    document = prepare_remote_batch(server=connected_server, pipeline="video", unit_paths=[str(session_path)])

    assert document.host == "remote"
    assert document.pipeline == "video"
    assert [job["job_id"] for job in document.jobs] == ["energy"]
    assert document.jobs[0]["cores"] == 16
    assert document.jobs[0]["memory_mb"] == 5000
    # A remotely dispatched job records its own outcome on the server, so no descriptor carries a local tracker.
    assert document.jobs[0]["tracker_path"] == ""
    assert document.units[0]["job_count"] == 1


def test_a_remote_forging_batch_reads_each_named_datasets_own_state(
    connected_server: Server, stub_ssh_transport: Any
) -> None:
    """A dataset batch resolves its project one level up and reads one state table per dataset it covers."""
    dataset_path = SERVER_PROJECT_ROOT.joinpath("Dataset")
    place_server_table(
        transport=stub_ssh_transport,
        remote_path=project_plan_path(project_directory=SERVER_PROJECT_ROOT),
        frame=build_plan_frame(
            [
                {
                    "unit_kind": "dataset",
                    "animal": None,
                    "session": None,
                    "dataset": "Dataset",
                    "pipeline": "forging",
                    "job_id": "forge",
                    "job_name": "session_data_assembly",
                    "specifier": "2024_11_04",
                    "cores": 1,
                }
            ]
        ),
    )
    place_server_table(
        transport=stub_ssh_transport,
        remote_path=dataset_path.joinpath(DATASET_STATE_FILENAME),
        frame=build_dataset_state_frame([{"job_id": "forge"}]),
    )

    document = prepare_remote_batch(server=connected_server, pipeline="forging", unit_paths=[str(dataset_path)])

    assert [job["job_id"] for job in document.jobs] == ["forge"]
    assert document.jobs[0]["unit_path"] == str(dataset_path)
    assert document.jobs[0]["unit_name"] == "Dataset"


def test_preparing_an_unsupported_pipeline_is_rejected(connected_server: Server) -> None:
    """The dispatch table states which pipelines the batch tools drive, so an absent one names no batch."""
    with pytest.raises(ValueError, match="which is not a supported batch pipeline"):
        prepare_remote_batch(
            server=connected_server,
            pipeline="not_a_pipeline",
            unit_paths=[str(SERVER_PROJECT_ROOT.joinpath("305", "2024_11_04"))],
        )


def test_preparing_a_batch_the_host_holds_no_plan_for_is_rejected(connected_server: Server) -> None:
    """A job must be planned before it can be sized, so a project with no plan table resolves no batch at all."""
    with pytest.raises(FileNotFoundError, match="holds no plan table for project 'TestProject'"):
        prepare_remote_batch(
            server=connected_server,
            pipeline="video",
            unit_paths=[str(SERVER_PROJECT_ROOT.joinpath("305", "2024_11_04"))],
        )


def test_preparing_a_batch_covering_no_unit_is_rejected() -> None:
    """Every artifact a batch is resolved from is written per project, so a batch has to name the project it reads."""
    with pytest.raises(ValueError, match="No processing unit was named"):
        resolve_project_root(unit_paths=[], unit_kind="session")


# Tests for the scheduler operations a submitted batch is followed and stopped through


def test_a_batch_that_queued_nothing_is_never_recorded(connected_server: Server) -> None:
    """The ledger names outstanding allocations, so a submission the scheduler never accepted leaves nothing behind."""
    assert submit_batch(server=connected_server, jobs=[], batch_id="batch01") == []
    assert read_ledger().batches == []


def test_the_scheduler_reports_the_state_of_every_submitted_allocation(
    connected_server: Server, stub_ssh_transport: Any
) -> None:
    """Observing a state and acting on it are separate, so a query reports what the scheduler holds and retires none."""
    stub_ssh_transport.job_statuses = {"1000": "RUNNING", "1001": "COMPLETED"}

    statuses = query_submissions(
        server=connected_server,
        submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
    )

    assert statuses == {"1000": JobStatus.RUNNING, "1001": JobStatus.COMPLETED}


def test_cancelling_a_batch_names_every_allocation_it_holds(connected_server: Server, stub_ssh_transport: Any) -> None:
    """One call carries the whole batch, which the scheduler applies to the queued and running allocations alone."""
    allocations = cancel_submissions(
        server=connected_server,
        submissions=[build_submission(slurm_job_id="1000"), build_submission(slurm_job_id="1001")],
    )

    assert allocations == ["1000", "1001"]
    assert any(issued.startswith("scancel ") for issued in stub_ssh_transport.commands)
    assert "1000" in stub_ssh_transport.commands[-1]
    assert "1001" in stub_ssh_transport.commands[-1]


def test_a_submission_renders_as_the_fields_it_was_recorded_with() -> None:
    """A caller reading a run's allocations needs the same fields the ledger holds, as a plain payload."""
    rendered = render_submission(submission=build_submission(slurm_job_id="1000"))

    assert rendered["slurm_job_id"] == "1000"
    assert rendered["unit_name"] == "Session"
    assert rendered["output_log"] == "/server/root/processing_batches/batch01/0000.out"


def test_the_configured_server_is_what_a_connection_opens(
    server_configuration: ServerConfiguration,
    stub_ssh_transport: Any,
) -> None:
    """A caller never names the host, so the recorded configuration is what every remote operation runs against."""
    with connect_to_server() as server:
        assert server.host == server_configuration.host
        assert server.root == Path(server_configuration.root)
        assert server.environment == server_configuration.environment


# Tests for mirroring a remote project's state onto this host


def test_mirroring_a_project_the_server_does_not_hold_is_rejected(connected_server: Server, tmp_path: Path) -> None:
    """A mirror answers from the server's own artifacts, so a project it holds no directory for mirrors nothing."""
    with pytest.raises(FileNotFoundError, match=r"Unable to mirror the state of project 'Absent'"):
        sync_project_state(server=connected_server, project="Absent", local_directory=tmp_path)


def test_mirroring_regenerates_the_state_before_it_pulls_it(
    connected_server: Server, stub_ssh_transport: Any, tmp_path: Path
) -> None:
    """Regeneration precedes the pull, so the mirrored tables describe the state after the runs rather than before."""
    server_project = stub_ssh_transport.local_path(SERVER_PROJECT_ROOT)
    dataset = server_project.joinpath("Dataset")
    dataset.mkdir(parents=True)
    dataset.joinpath(DATASET_MARKER_FILENAME).write_text("marker")
    dataset.joinpath(DATASET_STATE_FILENAME).write_text("dataset state")
    server_project.joinpath("305").mkdir()
    server_project.joinpath("TestProject_manifest.feather").write_text("manifest")
    server_project.joinpath("TestProject_jobs.feather").write_text("jobs")

    mirrored = sync_project_state(server=connected_server, project="TestProject", local_directory=tmp_path)

    issued = " ".join(stub_ssh_transport.commands)
    assert f"slf manifest -pp {SERVER_PROJECT_ROOT} create" in issued
    assert f"slf dataset-state -dp {SERVER_PROJECT_ROOT.joinpath('Dataset')}" in issued
    assert [path.relative_to(tmp_path).as_posix() for path in mirrored] == [
        "TestProject_manifest.feather",
        "TestProject_jobs.feather",
        "Dataset/dataset.yaml",
        "Dataset/dataset_state.feather",
    ]
    assert tmp_path.joinpath("Dataset", "dataset_state.feather").read_text() == "dataset state"


def test_a_project_holding_no_dataset_regenerates_its_manifest_alone(
    connected_server: Server, stub_ssh_transport: Any, tmp_path: Path
) -> None:
    """The dataset command names the datasets it refreshes, so a project holding none never issues it."""
    server_project = stub_ssh_transport.local_path(SERVER_PROJECT_ROOT)
    server_project.joinpath("305").mkdir(parents=True)
    server_project.joinpath("TestProject_plan.feather").write_text("plan")

    mirrored = sync_project_state(server=connected_server, project="TestProject", local_directory=tmp_path)

    issued = " ".join(stub_ssh_transport.commands)
    assert "slf manifest" in issued
    assert "slf dataset-state" not in issued
    assert [path.name for path in mirrored] == ["TestProject_plan.feather"]


def test_a_regeneration_that_fails_still_mirrors_what_the_server_holds(
    connected_server: Server, stub_ssh_transport: Any, tmp_path: Path
) -> None:
    """The artifacts a failed regeneration would have refreshed may still be worth pulling, so the failure warns."""
    server_project = stub_ssh_transport.local_path(SERVER_PROJECT_ROOT)
    server_project.mkdir(parents=True)
    server_project.joinpath("TestProject_jobs.feather").write_text("jobs")
    stub_ssh_transport.respond("bash -lc", stderr="slf: the manifest walk failed", return_code=1)

    mirrored = sync_project_state(server=connected_server, project="TestProject", local_directory=tmp_path)

    assert [path.name for path in mirrored] == ["TestProject_jobs.feather"]


def test_mirroring_without_regeneration_pulls_the_artifacts_as_the_server_last_wrote_them(
    connected_server: Server, stub_ssh_transport: Any, tmp_path: Path
) -> None:
    """Replanning is a preparation step rather than a mirroring one, so a caller may take the server as it stands."""
    server_project = stub_ssh_transport.local_path(SERVER_PROJECT_ROOT)
    server_project.mkdir(parents=True)
    server_project.joinpath("TestProject_jobs.feather").write_text("jobs")

    mirrored = sync_project_state(
        server=connected_server, project="TestProject", local_directory=tmp_path, regenerate=False
    )

    assert [path.name for path in mirrored] == ["TestProject_jobs.feather"]

    # Discovery reads the project's datasets with one server-side search, so the invocations the mirror issues carry
    # that search and nothing else.
    assert [command for command in stub_ssh_transport.commands if not command.startswith("find -L ")] == []
