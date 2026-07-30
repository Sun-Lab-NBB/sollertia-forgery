"""Tests the execution-host operations that mutate a project, and the remote artifact fetch.

Every operation runs the same underlying function on either host, so these tests pin the commands the remote host
issues and pin that a batched reset and a cleanup cost one invocation rather than one per unit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from dataclasses import dataclass

import pytest
from ataraxis_data_structures import ProcessingTracker

import sollertia_forgery.orchestration.maintenance as maintenance
from sollertia_forgery.orchestration import (
    LocalHost,
    RemoteHost,
    reset_tracked_jobs,
    resolve_path_size,
    clean_pipeline_output,
    environment_commands,
)


@dataclass
class StubResult:
    """Stands in for a completed server-side invocation."""

    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class StubServer:
    """Stands in for a connected server, recording every command it is handed."""

    def __init__(self, stdout: str = "", return_code: int = 0) -> None:
        self._stdout: str = stdout
        self._return_code: int = return_code
        self.commands: list[str] = []

    @property
    def environment(self) -> str:
        """Returns the environment every command activates."""
        return "slf_server"

    @property
    def root(self) -> Path:
        """Returns the server's data root."""
        return Path("/data")

    @property
    def host(self) -> str:
        """Returns the server's hostname."""
        return "server"

    def execute_command(self, command: str) -> StubResult:
        """Records the command and answers the configured result."""
        self.commands.append(command)
        return StubResult(return_code=self._return_code, stdout=self._stdout)


def remote(stdout: str = "", return_code: int = 0) -> tuple[RemoteHost, StubServer]:
    """Builds a remote host over a recording stub server."""
    server = StubServer(stdout=stdout, return_code=return_code)
    return RemoteHost(server=server), server  # type: ignore[arg-type]


def test_a_batched_remote_reset_costs_one_invocation() -> None:
    """Each unit resets only the identifiers it tracks, so one command carries a whole batch."""
    host, server = remote()

    host.reset_jobs(pipeline="video", unit_paths=[Path("/data/P/305/a"), Path("/data/P/305/b")], job_ids=["j1", "j2"])

    assert len(server.commands) == 1
    issued = server.commands[0]
    assert "slf reset -p video" in issued
    assert issued.count("-up") == 2
    assert issued.count("-id") == 2


def test_a_remote_reset_naming_no_unit_issues_nothing() -> None:
    """A reset covering no unit has nothing to do, so it never reaches the server."""
    host, server = remote()

    host.reset_jobs(pipeline="video", unit_paths=[], job_ids=["j1"])

    assert not server.commands


def test_a_remote_cleanup_reports_the_bytes_each_removal_freed() -> None:
    """The command prints the bytes beside each path, so a remote cleanup returns the figures a local one does."""
    host, server = remote(stdout="4096 /data/P/305/a/processed_data/video\n17 /data/P/305/a/video_tracker.yaml\n")

    removed = host.clean(pipeline="video", unit_paths=[Path("/data/P/305/a")])

    assert removed == [
        {"path": "/data/P/305/a/processed_data/video", "removed_bytes": 4096},
        {"path": "/data/P/305/a/video_tracker.yaml", "removed_bytes": 17},
    ]
    assert "slf clean -p video" in server.commands[0]


def test_a_remote_cleanup_ignores_output_that_is_not_a_removal() -> None:
    """Unrelated lines never become removals, so a warning on the same stream cannot inflate the report."""
    host, _server = remote(stdout="warning: something happened\n2048 /data/P/305/a/processed_data/video\n")

    removed = host.clean(pipeline="video", unit_paths=[Path("/data/P/305/a")])

    assert removed == [{"path": "/data/P/305/a/processed_data/video", "removed_bytes": 2048}]


def test_defining_a_remote_dataset_names_its_sessions_and_rebuild_flags() -> None:
    """The forging command builds the hierarchy, so a remote definition drives it rather than this process."""
    host, server = remote()

    host.define_dataset(
        project_root=Path("/data/P"),
        dataset_name="ds",
        session_names=["s1", "s2"],
        recreate_animals=["305"],
        force_recreate=True,
    )

    issued = server.commands[0]
    assert "slf forge -dn ds -pp /data/P" in issued
    assert issued.count("-s ") == 2
    assert "-ra 305" in issued
    assert issued.rstrip("'").endswith("-f")


def test_a_failing_remote_operation_reports_the_invocation_it_ran() -> None:
    """A caller needs to know which command failed, since one invocation can carry several."""
    host, _server = remote(return_code=1)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        host.reset_jobs(pipeline="video", unit_paths=[Path("/data/P/305/a")], job_ids=["j1"])


def test_chained_commands_stop_at_the_first_failure() -> None:
    """Later steps depend on earlier ones, so one round trip must not run past a failure."""
    rendered = environment_commands(
        environment="slf_server", commands=[["slf", "plan", "project"], ["slf", "manifest"]]
    )

    assert rendered.count("&&") == 3, "the activation and both commands are chained"


def test_a_batched_local_reset_clears_only_the_identifiers_each_unit_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing a whole batch's identifiers is safe, because a unit resets its own share and ignores the rest."""
    session = tmp_path.joinpath("2024_11_04")
    session.mkdir(parents=True)
    tracker_path = session.joinpath("tracker.yaml")
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=[("motion_energy", "1")], universe=[("motion_energy", "1")])
    held = next(iter(tracker.snapshot()))
    tracker.start_job(job_id=held)

    class StubDispatch:
        """Stands in for a dispatch entry resolving a fixed tracker."""

        @staticmethod
        def load(path: Path) -> Path:
            """Returns the unit unchanged."""
            return path

        @staticmethod
        def tracker_path(_unit: Path) -> Path:
            """Returns the fixed tracker path."""
            return tracker_path

    monkeypatch.setattr(maintenance, "resolve_dispatch", lambda pipeline: StubDispatch())  # noqa: ARG005

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session], job_ids=[held, "absent"])

    assert reset == [held]


def test_a_local_reset_naming_no_identifier_clears_every_tracked_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming nothing returns the unit to a clean slate, which is how a caller discards a whole run's records."""
    session = tmp_path.joinpath("2024_11_04")
    session.mkdir(parents=True)
    tracker_path = session.joinpath("tracker.yaml")
    tracker = ProcessingTracker(file_path=tracker_path)
    jobs = [("motion_energy", "1"), ("motion_energy", "2")]
    tracker.align_jobs(jobs=jobs, universe=jobs)
    for job_id in tracker.snapshot():
        tracker.start_job(job_id=job_id)

    class StubDispatch:
        """Stands in for a dispatch entry resolving a fixed tracker."""

        @staticmethod
        def load(path: Path) -> Path:
            """Returns the unit unchanged."""
            return path

        @staticmethod
        def tracker_path(_unit: Path) -> Path:
            """Returns the fixed tracker path."""
            return tracker_path

    monkeypatch.setattr(maintenance, "resolve_dispatch", lambda pipeline: StubDispatch())  # noqa: ARG005

    assert len(reset_tracked_jobs(pipeline="video", unit_paths=[session])) == 2


def test_cleaning_reports_the_size_of_what_it_removed(tmp_path: Path) -> None:
    """A caller deciding whether a cleanup was worth running needs the bytes it freed."""
    directory = tmp_path.joinpath("output")
    directory.mkdir()
    directory.joinpath("a.bin").write_bytes(b"x" * 100)
    directory.joinpath("b.bin").write_bytes(b"y" * 23)

    assert resolve_path_size(path=directory) == 123
    assert resolve_path_size(path=directory.joinpath("a.bin")) == 100


def test_cleaning_an_unsupported_pipeline_removes_nothing(tmp_path: Path) -> None:
    """A pipeline the dispatch table does not hold names no output, so nothing is removed."""
    assert clean_pipeline_output(pipeline="nonexistent", unit_paths=[tmp_path]) == []


def test_the_local_host_reports_where_an_artifact_already_sits(tmp_path: Path) -> None:
    """This machine already holds its own artifacts, so nothing is copied to deliver one."""
    artifact = tmp_path.joinpath("project_jobs.feather")
    artifact.write_text("rows")

    assert LocalHost.fetch(path=artifact, destination=tmp_path.joinpath("elsewhere")) == artifact
    assert LocalHost.fetch(path=tmp_path.joinpath("absent.feather"), destination=tmp_path) is None


def test_the_mirrored_artifact_set_covers_what_the_read_tools_resolve_from(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirrored table is only readable alongside the file its tool resolves it from, so both travel together."""
    import sollertia_forgery.orchestration.remote as remote_module

    pulled: list[str] = []

    class StubServer:
        """Stands in for a server holding one forged dataset and every project artifact."""

        @property
        def root(self) -> Path:
            """Returns the server's data root."""
            return Path("/data")

        @staticmethod
        def is_directory(remote_path: Path) -> bool:  # noqa: ARG004
            """Reports the project directory as present."""
            return True

        @staticmethod
        def exists(remote_path: Path) -> bool:
            """Reports every artifact as present, and only the one dataset directory as a dataset."""
            if remote_path.name == "dataset.yaml":
                return remote_path.parent.name == "ds"
            return True

        @staticmethod
        def list_directory(remote_path: Path) -> list[str]:  # noqa: ARG004
            """Reports the project's entries."""
            return ["ds", "305"]

        @staticmethod
        def pull(local_path: Path, remote_path: Path) -> None:
            """Records what was asked for rather than copying it."""
            pulled.append(remote_path.name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("mirrored")

    monkeypatch.setattr(remote_module, "_discover_remote_datasets", lambda server, project_path: [project_path / "ds"])  # noqa: ARG005

    remote_module.sync_project_state(
        server=StubServer(),  # type: ignore[arg-type]
        project="Proj",
        local_directory=Path(tempfile.mkdtemp()),
        regenerate=False,
    )

    # The manifest read tool opens the snapshot while the status tool reads the tracker, and the dataset read tools
    # resolve a dataset from its marker before reading its rows, so all four travel with the tables.
    assert "manifest_processing_tracker.yaml" in pulled
    assert "dataset.yaml" in pulled
    assert "dataset_state.feather" in pulled
    assert "Proj_jobs.feather" in pulled
