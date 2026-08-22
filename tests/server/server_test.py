"""Tests the remote compute server transport, the SLURM batch script builder, and the accounting status parser."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path
import datetime

import pytest

from sollertia_forgery.server import TERMINAL_JOB_STATUSES, Job, Server, JobStatus, CommandResult
from sollertia_forgery.server.job import _SlurmScript
import sollertia_forgery.server.server as server_module
from sollertia_forgery.server.server import _parse_job_status

if TYPE_CHECKING:
    from conftest import StubSSHTransport

    from sollertia_forgery.server import ServerConfiguration


def _build_job(working_directory: Path, **overrides: Any) -> Job:
    """Builds one job whose script, log, and working paths all sit under the given server-side directory.

    Args:
        working_directory: The server-side directory the job's script is uploaded into.
        overrides: The job arguments replacing the defaults this helper supplies.

    Returns:
        The constructed job.
    """
    arguments: dict[str, Any] = {
        "job_name": "forge_job",
        "output_log": working_directory.joinpath("forge_job.out"),
        "error_log": working_directory.joinpath("forge_job.err"),
        "working_directory": working_directory,
        "conda_environment": "slf_server",
    }
    arguments.update(overrides)
    return Job(**arguments)


def _fail_inside_the_block(server: Server) -> None:
    """Raises inside the server's context block, so the exit path runs on a failure rather than on a clean end.

    Args:
        server: The connected server the block is entered on.

    Raises:
        RuntimeError: Always, from inside the context block.
    """
    with server:
        message = "work failed"
        raise RuntimeError(message)


@pytest.fixture
def instant_retry_timer(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replaces the connection loop's wall clock, so a whole retry sequence runs without real waiting.

    Args:
        monkeypatch: The fixture used to replace the timer the server module holds.

    Returns:
        The list recording the delay, in seconds, of every wait the connection loop requested.
    """
    delays: list[int] = []

    class _InstantTimer:
        """Records a requested wait instead of sleeping through it."""

        def __init__(self, precision: object) -> None:
            self.precision = precision

        def delay(self, delay: int, *, allow_sleep: bool, block: bool) -> None:  # noqa: ARG002
            """Records the requested wait and returns immediately."""
            delays.append(delay)

    monkeypatch.setattr(server_module, "PrecisionTimer", _InstantTimer)
    return delays


@pytest.fixture
def unreachable_transport(
    monkeypatch: pytest.MonkeyPatch,
    instant_retry_timer: list[int],  # Requested so the retry sequence runs without waiting.
) -> SimpleNamespace:
    """Replaces the paramiko binding with a stack whose handshake always raises the configured failure.

    Args:
        monkeypatch: The fixture used to replace the paramiko binding the server module holds.
        instant_retry_timer: The stubbed timer that removes the delay between retries.

    Returns:
        A recorder carrying the reassignable ``error`` raised by every handshake, the ``attempts`` counter, and the
        ``authentication_exception`` class the server module recognizes.
    """

    class _AuthenticationError(Exception):
        """Stands in for the failure paramiko raises when the server rejects the supplied credentials."""

    recorder = SimpleNamespace(
        error=OSError("No route to host"),
        attempts=0,
        authentication_exception=_AuthenticationError,
    )

    class _FailingClient:
        """Refuses every handshake the connection loop attempts."""

        def set_missing_host_key_policy(self, policy: object) -> None:
            """Accepts the host key policy the server applies before it connects."""

        def connect(self, hostname: str, username: str, password: str) -> None:  # noqa: ARG002
            """Records the attempt and raises the configured failure."""
            recorder.attempts += 1
            raise recorder.error

    monkeypatch.setattr(
        server_module,
        "paramiko",
        SimpleNamespace(
            SSHClient=_FailingClient,
            AutoAddPolicy=object,
            AuthenticationException=_AuthenticationError,
        ),
    )
    return recorder


# Job and batch script rendering


def test_job_renders_the_scheduler_script_it_submits() -> None:
    """Verifies the directive header, the cleanup trap, the activation preamble, and the command body of a job."""
    job = _build_job(working_directory=Path("/data/sollertia/scratch"), cpu_threads=8, ram=24, time=90)
    job.add_command(command="slf forge --project TestProject")

    assert job.job_id is None
    assert job.job_name == "forge_job"
    assert job.remote_script_path == "/data/sollertia/scratch/forge_job.sh"
    assert job.command_script == (
        "#!/bin/bash\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --job-name=forge_job\n"
        "#SBATCH --output=/data/sollertia/scratch/forge_job.out\n"
        "#SBATCH --error=/data/sollertia/scratch/forge_job.err\n"
        "#SBATCH --mem=24G\n"
        "#SBATCH --time=01:30:00\n"
        "\n"
        "trap 'rm -f /data/sollertia/scratch/forge_job.sh' EXIT\n"
        "eval $(conda shell.bash hook)\n"
        "conda init bash\n"
        "source activate slf_server\n"
        "\n"
        "set -eo pipefail\n"
        "slf forge --project TestProject\n"
    )


def test_job_renders_dependency_directives_for_a_sequenced_stage() -> None:
    """Verifies that a job naming dependencies waits on every one of them and is cancelled when one cannot run."""
    job = _build_job(working_directory=Path("/data/sollertia/scratch"), dependencies=("1000", "1001"))

    lines = job.command_script.splitlines()

    assert "#SBATCH --dependency=afterok:1000:1001" in lines
    assert "#SBATCH --kill-on-invalid-dep=yes" in lines


def test_job_renders_a_day_field_for_a_multi_day_walltime() -> None:
    """Verifies that a wall-time of at least one day carries the leading day field SLURM expects."""
    job = _build_job(working_directory=Path("/data/sollertia/scratch"), time=1500)

    assert "#SBATCH --time=1-01:00:00" in job.command_script.splitlines()


def test_slurm_script_without_a_cleanup_path_renders_no_trap() -> None:
    """Verifies that a script asked to leave itself in place renders no exit trap."""
    script = _SlurmScript(
        cpus_per_task=2,
        job_name="bare",
        output="/data/out.txt",
        error="/data/err.txt",
        memory="4G",
        time=datetime.timedelta(minutes=5),
    )
    script.add_preamble(command="module load cuda")
    script.add_command(command="echo done")

    assert script.render() == (
        "#!/bin/bash\n"
        "#SBATCH --cpus-per-task=2\n"
        "#SBATCH --job-name=bare\n"
        "#SBATCH --output=/data/out.txt\n"
        "#SBATCH --error=/data/err.txt\n"
        "#SBATCH --mem=4G\n"
        "#SBATCH --time=00:05:00\n"
        "\n"
        "module load cuda\n"
        "\n"
        "set -eo pipefail\n"
        "echo done\n"
    )


# Connection lifecycle


def test_server_connects_and_exposes_the_configured_locations(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a connected server authenticates once and resolves every path from the configured data root."""
    assert stub_ssh_transport.connections == [("test.server.com", "tester")]
    assert connected_server.host == "test.server.com"
    assert connected_server.user == "tester"
    assert connected_server.environment == "slf_server"
    assert connected_server.root == Path("/data/sollertia")
    assert connected_server.cindra_configurations_directory == Path("/data/sollertia/cindra_configurations")
    assert connected_server.dlc_projects_directory == Path("/data/sollertia/deeplabcut_projects")


def test_server_close_is_idempotent(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that closing an already closed connection leaves the instance closed and issues no further work."""
    connected_server.close()

    assert stub_ssh_transport.closed is True
    assert connected_server._open is False

    stub_ssh_transport.closed = False
    connected_server.close()

    assert stub_ssh_transport.closed is False
    assert connected_server._open is False


def test_server_context_manager_closes_the_connection(
    stub_ssh_transport: StubSSHTransport, server_configuration: ServerConfiguration
) -> None:
    """Verifies that the context manager yields the connected instance and closes it however the block ends."""
    with Server(configuration=server_configuration) as server:
        assert server._open is True
        entered = server

    assert entered._open is False
    assert stub_ssh_transport.closed is True


def test_server_context_manager_closes_the_connection_on_failure(
    stub_ssh_transport: StubSSHTransport, server_configuration: ServerConfiguration
) -> None:
    """Verifies that a failure inside the block still closes the connection before it propagates."""
    server = Server(configuration=server_configuration)

    with pytest.raises(RuntimeError, match=r"work failed"):
        _fail_inside_the_block(server=server)

    assert server._open is False
    assert stub_ssh_transport.closed is True


def test_server_raises_permission_error_when_credentials_are_rejected(
    unreachable_transport: SimpleNamespace, server_configuration: ServerConfiguration
) -> None:
    """Verifies that a rejected handshake fails immediately rather than retrying."""
    unreachable_transport.error = unreachable_transport.authentication_exception("rejected")

    with pytest.raises(PermissionError, match=r"Authentication failed when connecting to test\.server\.com"):
        Server(configuration=server_configuration)

    assert unreachable_transport.attempts == 1


def test_server_raises_connection_error_after_exhausting_retries(
    unreachable_transport: SimpleNamespace,
    instant_retry_timer: list[int],
    server_configuration: ServerConfiguration,
) -> None:
    """Verifies that an unreachable host is retried a fixed number of times before the runtime is aborted."""
    with pytest.raises(ConnectionError, match=r"Could not connect to test\.server\.com after 30 retries"):
        Server(configuration=server_configuration)

    assert unreachable_transport.attempts == 31
    assert instant_retry_timer == [2] * 30


# Job submission


def test_submit_job_uploads_the_script_and_records_the_allocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that submission uploads the script, makes it executable, and records the assigned identifier."""
    job = _build_job(working_directory=connected_server.root.joinpath("scratch"))
    job.add_command(command="slf forge")

    submitted = connected_server.submit_job(job=job)

    assert submitted is job
    assert job.job_id == "1000"
    assert stub_ssh_transport.submitted_scripts == ["/data/sollertia/scratch/forge_job.sh"]
    assert stub_ssh_transport.local_path("/data/sollertia/scratch/forge_job.sh").read_text() == job.command_script
    assert stub_ssh_transport.commands == [
        "chmod +x /data/sollertia/scratch/forge_job.sh",
        "sbatch /data/sollertia/scratch/forge_job.sh",
    ]


def test_submit_job_assigns_sequential_identifiers_when_quiet(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a quiet submission still uploads and records, so the identifier tracks the scheduler."""
    first = connected_server.submit_job(
        job=_build_job(working_directory=connected_server.root.joinpath("scratch"), job_name="first"), verbose=False
    )
    second = connected_server.submit_job(
        job=_build_job(working_directory=connected_server.root.joinpath("scratch"), job_name="second"), verbose=False
    )

    assert (first.job_id, second.job_id) == ("1000", "1001")
    assert stub_ssh_transport.submitted_scripts == [
        "/data/sollertia/scratch/first.sh",
        "/data/sollertia/scratch/second.sh",
    ]


def test_submit_job_returns_an_already_submitted_job_untouched(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a job carrying an identifier is returned as it is, without a second upload or submission."""
    job = _build_job(working_directory=connected_server.root.joinpath("scratch"))
    job.job_id = "4242"

    returned = connected_server.submit_job(job=job, verbose=False)

    assert returned is job
    assert job.job_id == "4242"
    assert stub_ssh_transport.commands == []
    assert stub_ssh_transport.uploads == []


def test_submit_job_raises_when_the_script_cannot_be_made_executable(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a failed permission change aborts the submission and reports the server's own error text."""
    stub_ssh_transport.respond("chmod ", stderr="chmod: Operation not permitted\n", return_code=1)
    job = _build_job(working_directory=connected_server.root.joinpath("scratch"))

    with pytest.raises(RuntimeError, match=r"chmod: Operation not permitted"):
        connected_server.submit_job(job=job)

    assert job.job_id is None
    assert stub_ssh_transport.submitted_scripts == []


def test_submit_job_raises_when_the_scheduler_rejects_the_script(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that an acknowledgement naming no allocation aborts the submission and leaves the job unidentified."""
    stub_ssh_transport.respond("sbatch ", stderr="sbatch: error: invalid partition\n", return_code=1)
    job = _build_job(working_directory=connected_server.root.joinpath("scratch"))

    with pytest.raises(RuntimeError, match=r"sbatch: error: invalid partition"):
        connected_server.submit_job(job=job)

    assert job.job_id is None


# Job cancellation and status


def test_abort_job_cancels_a_running_allocation(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that an allocation the scheduler still holds is cancelled."""
    stub_ssh_transport.job_statuses["1000"] = "RUNNING"

    connected_server.abort_job(slurm_job_id="1000")

    assert "scancel 1000" in stub_ssh_transport.commands


def test_abort_job_leaves_a_settled_allocation_alone(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that an allocation the scheduler has already finished is not cancelled again."""
    stub_ssh_transport.job_statuses["1000"] = "COMPLETED"

    connected_server.abort_job(slurm_job_id="1000")

    assert not any(command.startswith("scancel") for command in stub_ssh_transport.commands)


def test_abort_jobs_cancels_every_named_allocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a batched cancellation costs one invocation and quotes every named identifier."""
    connected_server.abort_jobs(slurm_job_ids=("1000", "10 01"))

    assert stub_ssh_transport.commands == ["scancel 1000 '10 01'"]


def test_abort_jobs_without_identifiers_issues_no_invocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that an empty cancellation reaches the server not at all."""
    connected_server.abort_jobs(slurm_job_ids=())

    assert stub_ssh_transport.commands == []


def test_get_job_statuses_without_identifiers_returns_an_empty_mapping(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that querying nothing answers nothing and reaches the server not at all."""
    assert connected_server.get_job_statuses(slurm_job_ids=()) == {}
    assert stub_ssh_transport.commands == []


def test_get_job_statuses_reads_allocation_rows_and_the_blocked_queue(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that step rows, unparsable rows, and unrequested rows are skipped and a stuck job reports as blocked."""
    stub_ssh_transport.respond(
        "sacct ",
        stdout=(
            "malformed-row-without-a-separator\n1000.batch|COMPLETED\n9999|COMPLETED\n1000|PENDING\n1001|COMPLETED\n"
        ),
    )
    stub_ssh_transport.respond("squeue ", stdout="1000|DependencyNeverSatisfied\n7777|DependencyNeverSatisfied\n")

    statuses = connected_server.get_job_statuses(slurm_job_ids=("1000", "1001", "1002"))

    assert statuses == {
        "1000": JobStatus.BLOCKED,
        "1001": JobStatus.COMPLETED,
        "1002": JobStatus.UNKNOWN,
    }


def test_get_job_statuses_skips_the_queue_lookup_without_a_pending_allocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a query holding no pending allocation never reads the queue."""
    stub_ssh_transport.job_statuses.update({"1000": "COMPLETED", "1001": "FAILED"})

    statuses = connected_server.get_job_statuses(slurm_job_ids=("1000", "1001"))

    assert statuses == {"1000": JobStatus.COMPLETED, "1001": JobStatus.FAILED}
    assert not any(command.startswith("squeue") for command in stub_ssh_transport.commands)


def test_get_job_status_reports_one_allocation(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that the single-allocation query names only the requested identifier."""
    stub_ssh_transport.job_statuses["1000"] = "TIMEOUT"

    assert connected_server.get_job_status(slurm_job_id="1000") is JobStatus.TIMEOUT
    assert stub_ssh_transport.commands[0] == "sacct -j 1000 --format=JobID,State --noheader --parsable2"


def test_get_blocked_job_ids_keeps_only_permanently_blocked_rows(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that the queue scan reads this user's whole queue and keeps only the unsatisfiable dependencies."""
    stub_ssh_transport.respond(
        "squeue ",
        stdout="1000|DependencyNeverSatisfied\n1001|Resources\nmalformed-row\n1002|DependencyNeverSatisfied\n",
    )

    assert connected_server.get_blocked_job_ids() == {"1000", "1002"}
    assert stub_ssh_transport.commands == ['squeue -h -u tester -o "%i|%r"']


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("COMPLETED", JobStatus.COMPLETED),
        ("CANCELLED by 1234", JobStatus.CANCELLED),
        ("CANCELLED+", JobStatus.CANCELLED),
        ("OUT_OF_MEMORY", JobStatus.OUT_OF_MEMORY),
        ("PREEMPTED", JobStatus.UNKNOWN),
    ],
)
def test_parse_job_status_normalizes_decorated_accounting_states(state: str, expected: JobStatus) -> None:
    """Verifies that a decorated or unrecognized accounting state resolves to the state it names."""
    assert _parse_job_status(state=state) is expected


def test_terminal_job_statuses_exclude_the_states_a_job_still_leaves() -> None:
    """Verifies that the terminal set holds every settled state and neither of the two a job still leaves."""
    settled = frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
            JobStatus.NODE_FAIL,
            JobStatus.OUT_OF_MEMORY,
            JobStatus.BLOCKED,
        }
    )

    assert settled == TERMINAL_JOB_STATUSES
    assert JobStatus.PENDING not in TERMINAL_JOB_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_JOB_STATUSES
    assert JobStatus.UNKNOWN not in TERMINAL_JOB_STATUSES


# File transfer


def test_pull_raises_for_an_absent_remote_path(connected_server: Server, tmp_path: Path) -> None:
    """Verifies that downloading a path the server does not hold names the missing path."""
    with pytest.raises(FileNotFoundError, match=re.escape("/data/sollertia/absent does not exist on the server")):
        connected_server.pull(local_path=tmp_path.joinpath("unused"), remote_path=Path("/data/sollertia/absent"))


def test_pull_downloads_a_directory_tree(
    connected_server: Server, stub_ssh_transport: StubSSHTransport, tmp_path: Path
) -> None:
    """Verifies that a directory download reproduces every nested file it holds."""
    source = stub_ssh_transport.local_path("/data/sollertia/outputs")
    source.joinpath("nested").mkdir(parents=True)
    source.joinpath("top.txt").write_text("top")
    source.joinpath("nested", "inner.txt").write_text("inner")

    destination = tmp_path.joinpath("pulled")
    connected_server.pull(local_path=destination, remote_path=Path("/data/sollertia/outputs"))

    assert destination.joinpath("top.txt").read_text() == "top"
    assert destination.joinpath("nested", "inner.txt").read_text() == "inner"


def test_pull_downloads_a_single_file_and_creates_its_parent(
    connected_server: Server, stub_ssh_transport: StubSSHTransport, tmp_path: Path
) -> None:
    """Verifies that a file download creates the local directory the file is placed into."""
    source = stub_ssh_transport.local_path("/data/sollertia/outputs/manifest.txt")
    source.parent.mkdir(parents=True)
    source.write_text("manifest")

    destination = tmp_path.joinpath("mirror", "TestProject", "manifest.txt")
    connected_server.pull(local_path=destination, remote_path=Path("/data/sollertia/outputs/manifest.txt"))

    assert destination.read_text() == "manifest"
    assert stub_ssh_transport.downloads == [(destination, Path("/data/sollertia/outputs/manifest.txt"))]


def test_push_raises_for_an_absent_local_path(connected_server: Server, tmp_path: Path) -> None:
    """Verifies that uploading a path this host does not hold names the missing path."""
    missing = tmp_path.joinpath("absent.txt")

    # The console wraps the rendered message at a width that depends on the temporary path length, so the full
    # message is compared after collapsing the wrapping whitespace.
    with pytest.raises(FileNotFoundError, match=r"does\s+not\s+exist") as error:
        connected_server.push(local_path=missing, remote_path=Path("/data/sollertia/absent.txt"))

    assert f"The local path {missing} does not exist." in " ".join(str(error.value).split())


def test_push_uploads_a_directory_tree(
    connected_server: Server, stub_ssh_transport: StubSSHTransport, tmp_path: Path
) -> None:
    """Verifies that a directory upload reproduces every nested file it holds on the server."""
    source = tmp_path.joinpath("payload")
    source.joinpath("nested").mkdir(parents=True)
    source.joinpath("top.txt").write_text("top")
    source.joinpath("nested", "inner.txt").write_text("inner")

    connected_server.push(local_path=source, remote_path=Path("/data/sollertia/uploaded"))

    destination = stub_ssh_transport.local_path("/data/sollertia/uploaded")
    assert destination.joinpath("top.txt").read_text() == "top"
    assert destination.joinpath("nested", "inner.txt").read_text() == "inner"


def test_push_uploads_a_single_file_into_a_created_directory(
    connected_server: Server, stub_ssh_transport: StubSSHTransport, tmp_path: Path
) -> None:
    """Verifies that a file upload creates the server-side directory the file is placed into."""
    source = tmp_path.joinpath("plan.txt")
    source.write_text("plan")

    connected_server.push(local_path=source, remote_path=Path("/data/sollertia/plans/plan.txt"))

    assert stub_ssh_transport.local_path("/data/sollertia/plans/plan.txt").read_text() == "plan"
    assert stub_ssh_transport.commands == ["mkdir -p /data/sollertia/plans"]
    assert stub_ssh_transport.uploads == [(source, Path("/data/sollertia/plans/plan.txt"))]


# Remote path creation and removal


def test_create_makes_a_nested_directory_in_one_invocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a nested directory is resolved by the shell in a single round trip."""
    connected_server.create(remote_path=Path("/data/sollertia/TestProject/305"))

    assert stub_ssh_transport.local_path("/data/sollertia/TestProject/305").is_dir()
    assert stub_ssh_transport.commands == ["mkdir -p /data/sollertia/TestProject/305"]


def test_create_without_parents_uses_the_transfer_protocol(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a directory created without parents is made once and left alone on a repeated call."""
    connected_server.create(remote_path=Path("/data/sollertia/single"), parents=False)
    stub_ssh_transport.local_path("/data/sollertia/single/marker.txt").write_text("marker")
    connected_server.create(remote_path=Path("/data/sollertia/single"), parents=False)

    assert stub_ssh_transport.local_path("/data/sollertia/single/marker.txt").read_text() == "marker"
    assert stub_ssh_transport.commands == []


def test_create_leaves_an_existing_file_in_place(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that creating a file makes it empty when it is absent and preserves it when it already exists."""
    connected_server.create(remote_path=Path("/data/sollertia/state/marker.txt"), is_dir=False)
    created = stub_ssh_transport.local_path("/data/sollertia/state/marker.txt")

    assert created.read_text() == ""

    created.write_text("recorded")
    connected_server.create(remote_path=Path("/data/sollertia/state/marker.txt"), is_dir=False)

    assert created.read_text() == "recorded"


def test_create_raises_when_the_server_refuses_the_directory(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a refused directory creation reports the server's own error text."""
    stub_ssh_transport.respond("mkdir -p ", stderr="mkdir: Permission denied\n", return_code=1)

    with pytest.raises(RuntimeError, match=r"mkdir: Permission denied"):
        connected_server.create(remote_path=Path("/data/sollertia/blocked"))


def test_remove_deletes_a_file(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that a file removal deletes the named file."""
    target = stub_ssh_transport.local_path("/data/sollertia/state/marker.txt")
    target.parent.mkdir(parents=True)
    target.write_text("marker")

    connected_server.remove(remote_path=Path("/data/sollertia/state/marker.txt"), is_dir=False)

    assert not target.exists()
    assert target.parent.is_dir()


def test_remove_deletes_an_empty_directory(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that a non-recursive removal deletes an empty directory."""
    target = stub_ssh_transport.local_path("/data/sollertia/empty")
    target.mkdir(parents=True)

    connected_server.remove(remote_path=Path("/data/sollertia/empty"), is_dir=True)

    assert not target.exists()


def test_remove_deletes_a_populated_directory_tree(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a recursive removal deletes every nested entry alongside the directory itself."""
    target = stub_ssh_transport.local_path("/data/sollertia/tree")
    target.joinpath("nested").mkdir(parents=True)
    target.joinpath("top.txt").write_text("top")
    target.joinpath("nested", "inner.txt").write_text("inner")

    connected_server.remove(remote_path=Path("/data/sollertia/tree"), is_dir=True, recursive=True)

    assert not target.exists()


def test_remove_leaves_the_target_intact_when_the_tree_cannot_be_walked(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a recursive removal of an unwalkable target warns rather than propagating the failure."""
    target = stub_ssh_transport.local_path("/data/sollertia/not_a_directory.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("payload")

    connected_server.remove(remote_path=Path("/data/sollertia/not_a_directory.txt"), is_dir=True, recursive=True)

    assert target.read_text() == "payload"


# Remote inspection


def test_exists_reports_presence(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that existence is reported for a present path and denied for an absent one."""
    stub_ssh_transport.local_path("/data/sollertia/present").mkdir(parents=True)

    assert connected_server.exists(remote_path=Path("/data/sollertia/present")) is True
    assert connected_server.exists(remote_path=Path("/data/sollertia/absent")) is False


def test_is_directory_separates_directories_from_files_and_absences(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that only a present directory reports as one."""
    stub_ssh_transport.local_path("/data/sollertia/present").mkdir(parents=True)
    stub_ssh_transport.local_path("/data/sollertia/present/file.txt").write_text("payload")

    assert connected_server.is_directory(remote_path=Path("/data/sollertia/present")) is True
    assert connected_server.is_directory(remote_path=Path("/data/sollertia/present/file.txt")) is False
    assert connected_server.is_directory(remote_path=Path("/data/sollertia/absent")) is False


def test_list_directory_returns_entry_names(connected_server: Server, stub_ssh_transport: StubSSHTransport) -> None:
    """Verifies that a listing names the entries of a directory rather than their full paths."""
    listed = stub_ssh_transport.local_path("/data/sollertia/TestProject")
    listed.joinpath("305").mkdir(parents=True)
    listed.joinpath("manifest.feather").write_text("manifest")

    assert connected_server.list_directory(remote_path=Path("/data/sollertia/TestProject")) == [
        "305",
        "manifest.feather",
    ]


def test_execute_command_returns_both_streams_and_the_exit_code(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that an invocation's output, error, and exit code are all reported back to the caller."""
    stub_ssh_transport.respond("slf prepare", stdout="prepared\n", stderr="deprecated\n", return_code=3)

    result = connected_server.execute_command(command="slf prepare --project TestProject")

    assert result == CommandResult(stdout="prepared\n", stderr="deprecated\n", return_code=3)
    assert stub_ssh_transport.commands == ["slf prepare --project TestProject"]


def test_find_paths_reports_matches_at_the_requested_depths_only(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a search reports the depths it was given and never the same name above or below them."""
    project_path = stub_ssh_transport.local_path("/data/sollertia/TestProject")
    for relative in (("marker.yaml",), ("305", "marker.yaml"), ("305", "session", "marker.yaml")):
        planted = project_path.joinpath(*relative)
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("marker")

    matches = connected_server.find_paths(
        remote_path=Path("/data/sollertia/TestProject"), names=("marker.yaml",), minimum_depth=2, maximum_depth=2
    )

    assert matches == [Path("/data/sollertia/TestProject/305/marker.yaml")]


def test_find_paths_rejects_a_path_that_is_not_a_directory(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a file answers as an absent directory, since the search itself reports one as an empty tree."""
    artifact = stub_ssh_transport.local_path("/data/sollertia/TestProject_manifest.feather")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("manifest")

    with pytest.raises(FileNotFoundError, match=r"holds no\s+directory at that path"):
        connected_server.find_paths(
            remote_path=Path("/data/sollertia/TestProject_manifest.feather"),
            names=("marker.yaml",),
            minimum_depth=1,
            maximum_depth=1,
        )


def test_find_paths_passes_the_searched_path_as_a_start_point_rather_than_a_pattern(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Verifies that a path carrying a glob metacharacter is searched as a location rather than matched as a pattern."""
    marker = stub_ssh_transport.local_path("/data/sollertia/Proj[1]").joinpath("305", "marker.yaml")
    marker.parent.mkdir(parents=True)
    marker.write_text("marker")

    matches = connected_server.find_paths(
        remote_path=Path("/data/sollertia/Proj[1]"), names=("marker.yaml",), minimum_depth=2, maximum_depth=2
    )

    assert matches == [Path("/data/sollertia/Proj[1]/305/marker.yaml")]
    assert stub_ssh_transport.commands == [
        "find -L '/data/sollertia/Proj[1]' -mindepth 2 -maxdepth 2 '(' -name marker.yaml ')' '!' -type l -print0"
    ]


def test_find_paths_rejects_a_record_the_search_did_not_produce(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A login shell that greets the connection writes into the stream the answer arrives on, which is not an answer."""
    stub_ssh_transport.local_path("/data/sollertia/TestProject").mkdir(parents=True)
    stub_ssh_transport.respond(
        prefix="find -L ",
        stdout="Lmod is replacing 'gcc/9.3' with 'gcc/11.2'\0/data/sollertia/TestProject/305/marker.yaml\0",
    )

    with pytest.raises(RuntimeError, match=r"does not sit under the searched directory"):
        connected_server.find_paths(
            remote_path=Path("/data/sollertia/TestProject"), names=("marker.yaml",), minimum_depth=2, maximum_depth=2
        )
