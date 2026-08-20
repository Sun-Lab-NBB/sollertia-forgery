"""Tests the execution-host contract, the operations each host runs, and the artifact locations a batch reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import dataclass

import polars as pl
import pytest
from sollertia_shared_assets import (
    DatasetData,
    SessionTypes,
    DatasetSession,
    AcquisitionSystems,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.forging import DATASET_STATE_FILENAME
from sollertia_forgery.managing import project_jobs_path, project_manifest_path
from sollertia_forgery.orchestration import (
    DATASET_UNIT,
    SESSION_UNIT,
    PROJECT_PLAN_SCHEMA,
    LocalHost,
    RemoteHost,
    ExecutionHost,
    remote as remote_module,
    maintenance,
    project_plan_path,
    resolve_path_size,
    plan_artifact_path,
    reset_tracked_jobs,
    environment_commands,
    state_artifact_paths,
    clean_pipeline_output,
)
from sollertia_forgery.shared_assets import ProcessingPipelines, resolve_session_tracker_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import StubSSHTransport
    from sollertia_shared_assets import ProjectData, SessionData

    from sollertia_forgery.server import Server

SERVER_PROJECT_ROOT: Path = Path("/data/sollertia/TestProject")
"""The project directory every remote-host test addresses on the stubbed compute server."""


@dataclass(slots=True)
class StubResult:
    """Stands in for a completed server-side invocation."""

    return_code: int = 0
    """The status the invocation exited with."""
    stdout: str = ""
    """The invocation's standard output."""
    stderr: str = ""
    """The invocation's standard error."""


class StubServer:
    """Stands in for a connected server, recording every command it is handed.

    Args:
        stdout: The standard output every command answers with.
        return_code: The status every command exits with.

    Attributes:
        _stdout: The standard output every command answers with.
        _return_code: The status every command exits with.
        commands: The commands the server was handed, in the order it received them.
    """

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


def build_remote_host(stdout: str = "", return_code: int = 0) -> tuple[RemoteHost, StubServer]:
    """Builds a remote host over a recording stub server."""
    server = StubServer(stdout=stdout, return_code=return_code)
    return RemoteHost(server=server), server


def build_stub_dispatch(tracker_file: Path) -> Any:
    """Builds a dispatch entry that resolves every unit to the given tracker file."""

    class StubDispatch:
        """Stands in for a dispatch entry resolving a fixed tracker."""

        @staticmethod
        def load(path: Path) -> Path:
            """Returns the unit unchanged."""
            return path

        @staticmethod
        def tracker_path(_unit: Path) -> Path:
            """Returns the fixed tracker path."""
            return tracker_file

    return StubDispatch()


def test_a_batched_remote_reset_costs_one_invocation() -> None:
    """Each unit resets only the identifiers it tracks, so one command carries a whole batch."""
    host, server = build_remote_host()

    host.reset_jobs(pipeline="video", unit_paths=[Path("/data/P/305/a"), Path("/data/P/305/b")], job_ids=["j1", "j2"])

    assert len(server.commands) == 1
    issued = server.commands[0]
    assert "slf reset -p video" in issued
    assert issued.count("-up") == 2
    assert issued.count("-id") == 2


def test_a_remote_reset_naming_no_unit_issues_nothing() -> None:
    """A reset covering no unit has nothing to do, so it never reaches the server."""
    host, server = build_remote_host()

    host.reset_jobs(pipeline="video", unit_paths=[], job_ids=["j1"])

    assert not server.commands


def test_a_remote_cleanup_reports_the_bytes_each_removal_freed() -> None:
    """The command prints the bytes beside each path, so a remote cleanup returns the figures a local one does."""
    host, server = build_remote_host(
        stdout="4096 /data/P/305/a/processed_data/video\n17 /data/P/305/a/video_tracker.yaml\n"
    )

    removed = host.clean(pipeline="video", unit_paths=[Path("/data/P/305/a")])

    assert removed == [
        {"path": "/data/P/305/a/processed_data/video", "removed_bytes": 4096},
        {"path": "/data/P/305/a/video_tracker.yaml", "removed_bytes": 17},
    ]
    assert "slf clean -p video" in server.commands[0]


def test_a_remote_cleanup_ignores_output_that_is_not_a_removal() -> None:
    """Unrelated lines never become removals, so a warning on the same stream cannot inflate the report."""
    host, _server = build_remote_host(stdout="warning: something happened\n2048 /data/P/305/a/processed_data/video\n")

    removed = host.clean(pipeline="video", unit_paths=[Path("/data/P/305/a")])

    assert removed == [{"path": "/data/P/305/a/processed_data/video", "removed_bytes": 2048}]


def test_defining_a_remote_dataset_names_its_sessions_and_rebuild_flags() -> None:
    """A definition builds the hierarchy alone, so it carries the session set and the rebuild flags and no job."""
    host, server = build_remote_host()

    host.define_dataset(
        project_root=Path("/data/P"),
        dataset_name="ds",
        session_names=["s1", "s2"],
        recreate_animals=["305"],
        force_recreate=True,
    )

    issued = server.commands[0]
    assert "define_forging_dataset(" in issued
    assert 'name="ds"' in issued
    assert 'session_names=tuple(["s1", "s2"])' in issued
    assert 'project_root=Path("/data/P")' in issued
    assert "force_recreate=True" in issued
    assert 'recreate_animals=tuple(["305"])' in issued
    # The forging command runs every outstanding tracked job, so a definition-only step never names it.
    assert "slf forge" not in issued


def test_a_failing_remote_operation_reports_the_invocation_it_ran() -> None:
    """A caller needs to know which command failed, since one invocation can carry several."""
    host, _server = build_remote_host(return_code=1)

    with pytest.raises(RuntimeError, match=r"'slf reset -p video -up .+' exited with code 1"):
        host.reset_jobs(pipeline="video", unit_paths=[Path("/data/P/305/a")], job_ids=["j1"])


def test_chained_commands_stop_at_the_first_failure() -> None:
    """Later steps depend on earlier ones, so one round trip must not run past a failure."""
    rendered = environment_commands(
        environment="slf_server", commands=[["slf", "plan", "project"], ["slf", "manifest"]]
    )

    chained_operator_count = 3
    assert rendered.count("&&") == chained_operator_count, "The activation and both commands are chained."


def test_a_batched_local_reset_clears_only_the_identifiers_each_unit_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Passing a whole batch's identifiers is safe, because a unit resets its own share and ignores the rest."""
    session = tmp_path.joinpath("2024_11_04")
    session.mkdir(parents=True)
    tracker_path = session.joinpath("tracker.yaml")
    jobs = [("motion_energy", "1")]
    tracker = write_tracker(tracker_path, jobs, running=jobs)
    held = next(iter(tracker.snapshot()))

    monkeypatch.setattr(
        maintenance,
        "resolve_dispatch",
        lambda pipeline: build_stub_dispatch(tracker_file=tracker_path),  # noqa: ARG005
    )

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session], job_ids=[held, "absent"])

    assert reset == [held]


def test_a_local_reset_naming_no_identifier_clears_every_tracked_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Naming nothing returns the unit to a clean slate, which is how a caller discards a whole run's records."""
    session = tmp_path.joinpath("2024_11_04")
    session.mkdir(parents=True)
    tracker_path = session.joinpath("tracker.yaml")
    jobs = [("motion_energy", "1"), ("motion_energy", "2")]
    tracker = write_tracker(tracker_path, jobs, running=jobs)

    monkeypatch.setattr(
        maintenance,
        "resolve_dispatch",
        lambda pipeline: build_stub_dispatch(tracker_file=tracker_path),  # noqa: ARG005
    )

    reset = reset_tracked_jobs(pipeline="video", unit_paths=[session])

    assert sorted(reset) == sorted(tracker.snapshot())


def test_a_path_size_sums_every_file_it_holds(tmp_path: Path) -> None:
    """A caller deciding whether a cleanup was worth running needs the bytes each removal would free."""
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


def test_the_mirrored_artifact_set_covers_what_the_read_tools_resolve_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mirrored table is only readable alongside the file its tool resolves it from, so both travel together."""
    pulled: list[str] = []

    class StubMirrorServer:
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

    monkeypatch.setattr(
        remote_module,
        "_discover_remote_datasets",
        lambda server, project_path: [project_path.joinpath("ds")],  # noqa: ARG005
    )

    mirrored = remote_module.sync_project_state(
        server=StubMirrorServer(),
        project="Proj",
        local_directory=tmp_path.joinpath("mirror"),
        regenerate=False,
    )

    # The manifest read tool opens the snapshot while the status tool reads the tracker, and the dataset read tools
    # resolve a dataset from its marker before reading its rows, so all four travel with the tables.
    assert "manifest_processing_tracker.yaml" in pulled
    assert "dataset.yaml" in pulled
    assert "dataset_state.feather" in pulled
    assert "Proj_jobs.feather" in pulled
    assert {path.name for path in mirrored} == set(pulled)


# Helpers


def create_dataset(project: ProjectData, name: str) -> DatasetData:
    """Creates one forged dataset hierarchy under a project through the shared hierarchy's own creator.

    Args:
        project: The project the dataset is created under.
        name: The name of the dataset to create.

    Returns:
        The created dataset view, whose marker exists on disk.
    """
    return DatasetData.create(
        name=name,
        project=project.project_name,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
        sessions=(DatasetSession(session="2026-01-02-03-04-05-000006", animal="305", session_path=project.path),),
        datasets_root=project.path,
        column_descriptions={},
    )


def write_plan_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Writes one project plan projection holding the supplied partial rows.

    Args:
        path: The path the projection is written to.
        rows: The partial rows, each overriding the shared defaults.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        data=[
            {
                "unit_kind": SESSION_UNIT,
                "animal": "305",
                "session": None,
                "dataset": None,
                "pipeline": "video",
                "job_id": "a_job",
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
    ).write_ipc(file=path, compression="uncompressed")
    return path


def issued_command(server: StubSSHTransport, index: int = 0) -> str:
    """Returns one invocation the stubbed transport recorded.

    Args:
        server: The transport every invocation was recorded against.
        index: The position of the invocation to return.

    Returns:
        The recorded invocation.
    """
    return server.commands[index]


# The execution-host contract


def test_the_protocol_declares_the_operations_preparation_runs_against_a_host() -> None:
    """Both hosts run the same underlying functions, so the protocol states the operations rather than implementing
    any of them.
    """
    declared = {name for name in vars(ExecutionHost) if not name.startswith("_")}
    assert declared == {
        "label",
        "materialize",
        "plan",
        "generate_state",
        "read_rows",
        "fetch",
        "reset_jobs",
        "clean",
        "define_dataset",
        "resolve_tracker_paths",
    }
    for name in declared:
        assert hasattr(LocalHost, name)
        assert hasattr(RemoteHost, name)

    # Each declaration carries the contract alone, so invoking one through the protocol answers nothing.
    host = LocalHost()
    assert ExecutionHost.label.fget(host) is None
    assert ExecutionHost.materialize(host, Path("/p"), [], SESSION_UNIT, replan=False) is None
    assert ExecutionHost.plan(host, Path("/p"), [], SESSION_UNIT, replan=False) is None
    assert ExecutionHost.generate_state(host, Path("/p"), [], SESSION_UNIT) is None
    assert ExecutionHost.read_rows(host, Path("/p")) is None
    assert ExecutionHost.fetch(host, Path("/p"), Path("/d")) is None
    assert ExecutionHost.reset_jobs(host, "video", [], []) is None
    assert ExecutionHost.clean(host, "video", []) is None
    assert ExecutionHost.define_dataset(host, Path("/p"), "ds", [], [], force_recreate=False) is None
    assert ExecutionHost.resolve_tracker_paths(host, "video", []) is None


def test_each_host_reports_itself_under_the_name_a_batch_records(connected_server: Server) -> None:
    """A prepared batch records where it was prepared, so it runs on the host that holds the data it reads."""
    remote_host = RemoteHost(server=connected_server)

    assert LocalHost().label == "local"
    assert remote_host.label == "remote"
    assert remote_host.server is connected_server
    assert repr(remote_host) == f"RemoteHost(host={connected_server.host}, root={connected_server.root})"


# The local host


def test_materializing_a_session_batch_writes_the_plan_and_the_state_it_is_resolved_from(
    project_root: Path, experiment_session: SessionData
) -> None:
    """Planning registers a unit's jobs and the state step reads those registries, so the order is what carries it."""
    session_path = experiment_session.raw_data_path.parent

    LocalHost.materialize(project_root=project_root, unit_paths=[session_path], unit_kind=SESSION_UNIT, replan=False)

    plan_rows = LocalHost.read_rows(path=project_plan_path(project_directory=project_root))
    state_rows = LocalHost.read_rows(path=project_jobs_path(project_directory=project_root))
    # Every planned figure is modeled from the data its job reads, so the runtime pipeline, whose only job for this
    # session reads an archive the session never wrote, is refused and drops out of the plan rather than being
    # recorded at a figure nothing measured.
    assert {row["pipeline"] for row in plan_rows} == {"checksum"}
    assert {row["session"] for row in plan_rows} == {experiment_session.session_name}
    assert {row["pipeline"] for row in state_rows} == {"checksum"}
    assert {row["session"] for row in state_rows} == {experiment_session.session_name}
    assert project_manifest_path(project_directory=project_root).is_file()


def test_planning_reports_the_figures_each_unit_recorded(project_root: Path, experiment_session: SessionData) -> None:
    """A submission is sized against these figures, so planning reports the count and the memory each unit resolved."""
    session_path = experiment_session.raw_data_path.parent

    planned = LocalHost.plan(project_root=project_root, unit_paths=[session_path], unit_kind=SESSION_UNIT, replan=True)

    assert len(planned) == 1
    assert planned[0]["unit_path"] == str(session_path)
    assert planned[0]["unit_name"] == experiment_session.session_name
    # The checksum stage is the one pipeline whose every job reads data this session carries, so it is the one that
    # survives the sizing pass.
    assert planned[0]["job_count"] == 1
    assert planned[0]["summed_memory_mb"] > 0


def test_a_unit_no_pipeline_resolves_a_job_for_is_reported_beside_the_ones_that_planned(
    project_root: Path, experiment_session: SessionData
) -> None:
    """One unit carrying none of the data the pipelines consume never stops the units that carry it."""
    session_path = experiment_session.raw_data_path.parent
    absent = project_root.joinpath("305", "2026-01-02-03-04-05-000006")

    planned = LocalHost.plan(
        project_root=project_root, unit_paths=[absent, session_path], unit_kind=SESSION_UNIT, replan=False
    )

    assert planned[0]["job_count"] == 0
    assert "No pipeline planned any job" in planned[0]["error"]
    assert planned[1]["job_count"] == 1


def test_planning_a_dataset_reads_the_forging_pipeline_alone(project_root: Path) -> None:
    """A dataset batch is planned against the forging pipeline, which resolves nothing for a directory holding no
    dataset.
    """
    planned = LocalHost.plan(
        project_root=project_root,
        unit_paths=[project_root.joinpath("absent_dataset")],
        unit_kind=DATASET_UNIT,
        replan=False,
    )

    assert planned[0]["job_count"] == 0
    assert "Unable to plan the jobs of" in planned[0]["error"]
    assert "forging" in planned[0]["error"]


def test_refreshing_a_dataset_batch_writes_the_state_of_the_named_datasets_alone(project: ProjectData) -> None:
    """A dataset batch reads one table per named dataset, so a dataset the batch does not cover is left alone."""
    named = create_dataset(project=project, name="named_dataset")
    other = create_dataset(project=project, name="other_dataset")

    LocalHost.generate_state(
        project_root=project.path, unit_paths=[named.dataset_data_path.parent], unit_kind=DATASET_UNIT
    )

    assert named.dataset_data_path.parent.joinpath(DATASET_STATE_FILENAME).is_file()
    assert not other.dataset_data_path.parent.joinpath(DATASET_STATE_FILENAME).is_file()


def test_a_table_the_local_host_does_not_hold_reads_as_no_rows(tmp_path: Path) -> None:
    """A project whose artifacts were never written reads cleanly rather than failing."""
    assert LocalHost.read_rows(path=tmp_path.joinpath("absent.feather")) == []

    written = write_plan_table(path=tmp_path.joinpath("plan.feather"), rows=[{"job_id": "energy"}])
    assert [row["job_id"] for row in LocalHost.read_rows(path=written)] == ["energy"]


def test_a_local_reset_returns_the_units_tracked_jobs_to_the_scheduled_state(
    experiment_session: SessionData,
) -> None:
    """The trackers sit on this machine, so the reset runs in this process rather than over a command line."""
    tracker_path = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.RUNTIME)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [("runtime_processing", "1")]
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=jobs, universe=jobs)
    job_id = next(iter(tracker.snapshot()))
    tracker.start_job(job_id=job_id)
    tracker.complete_job(job_id=job_id)

    LocalHost.reset_jobs(pipeline="runtime", unit_paths=[experiment_session.raw_data_path.parent], job_ids=[])

    assert ProcessingTracker(file_path=tracker_path).snapshot()[job_id].status is ProcessingStatus.SCHEDULED


def test_a_local_cleanup_reports_the_bytes_each_removal_freed(experiment_session: SessionData) -> None:
    """A caller deciding whether a cleanup was worth running needs the bytes it freed."""
    output = experiment_session.processed_data.runtime_data_path
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("payload.bin").write_bytes(b"x" * 512)

    removed = LocalHost.clean(pipeline="runtime", unit_paths=[experiment_session.raw_data_path.parent])

    assert {entry["path"] for entry in removed} == {str(output)}
    assert removed[0]["removed_bytes"] == 512
    assert not output.exists()


def test_defining_a_local_dataset_rejects_a_session_the_project_does_not_hold(project_root: Path) -> None:
    """The hierarchy is built from the named sessions, so a name resolving to no directory stops the definition."""
    with pytest.raises(FileNotFoundError, match="2026-01-02-03-04-05-000006"):
        LocalHost.define_dataset(
            project_root=project_root,
            dataset_name="test_dataset",
            session_names=["2026-01-02-03-04-05-000006"],
            recreate_animals=[],
            force_recreate=False,
        )


def test_the_local_host_resolves_where_each_units_tracker_sits(experiment_session: SessionData) -> None:
    """The local engine opens these files directly, so a batch dispatched here carries the locations on its
    descriptors.
    """
    session_path = experiment_session.raw_data_path.parent
    unloadable = session_path.parent.joinpath("2026-01-02-03-04-05-000006")

    resolved = LocalHost.resolve_tracker_paths(pipeline="runtime", unit_paths=[session_path, unloadable])

    assert resolved == {
        str(session_path): str(
            resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.RUNTIME)
        )
    }


def test_an_unsupported_pipeline_resolves_no_tracker_at_all(experiment_session: SessionData) -> None:
    """A pipeline the dispatch table does not hold names no tracker, so no descriptor carries a location for it."""
    assert (
        LocalHost.resolve_tracker_paths(pipeline="not_a_pipeline", unit_paths=[experiment_session.raw_data_path.parent])
        == {}
    )


# The remote host


def test_materializing_a_session_batch_remotely_ships_its_steps_as_one_chained_invocation(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Chaining costs a single round trip and holds the order the steps require, since planning registers the jobs."""
    session_path = SERVER_PROJECT_ROOT.joinpath("305", "2026-01-02-03-04-05-000006")

    RemoteHost(server=connected_server).materialize(
        project_root=SERVER_PROJECT_ROOT, unit_paths=[session_path], unit_kind=SESSION_UNIT, replan=True
    )

    issued = issued_command(server=stub_ssh_transport)
    assert len(stub_ssh_transport.commands) == 1
    assert f"slf plan session -sp {session_path} -rp" in issued
    assert f"slf plan project -pp {SERVER_PROJECT_ROOT}" in issued
    assert f"slf manifest -pp {SERVER_PROJECT_ROOT} create" in issued
    assert issued.index("slf plan session") < issued.index("slf plan project") < issued.index("slf manifest")


def test_refreshing_a_dataset_batch_remotely_names_every_dataset_it_covers(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A dataset batch reads one table per named dataset, so one command carries every dataset the batch covers."""
    datasets = [SERVER_PROJECT_ROOT.joinpath("first"), SERVER_PROJECT_ROOT.joinpath("second")]

    RemoteHost(server=connected_server).generate_state(
        project_root=SERVER_PROJECT_ROOT, unit_paths=datasets, unit_kind=DATASET_UNIT
    )

    issued = issued_command(server=stub_ssh_transport)
    assert f"slf dataset-state -dp {datasets[0]} -dp {datasets[1]}" in issued


def test_a_remote_plan_reports_the_figures_the_projection_now_holds(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """The per-unit figures are read out of the projection, so a remote plan reports what a local one returns."""
    plan_path = project_plan_path(project_directory=SERVER_PROJECT_ROOT)
    write_plan_table(
        path=stub_ssh_transport.local_path(plan_path),
        rows=[
            {"job_id": "energy", "session": "2026-01-02-03-04-05-000006", "memory_mb": 4096},
            {"job_id": "rename", "session": "2026-01-02-03-04-05-000006", "memory_mb": 512},
            # A dataset row names no session, so a session batch's summary passes over it.
            {"job_id": "forge", "unit_kind": DATASET_UNIT, "animal": None, "dataset": "a_dataset"},
        ],
    )
    planned = SERVER_PROJECT_ROOT.joinpath("305", "2026-01-02-03-04-05-000006")
    unplanned = SERVER_PROJECT_ROOT.joinpath("305", "2026-01-03-03-04-05-000006")

    summarized = RemoteHost(server=connected_server).plan(
        project_root=SERVER_PROJECT_ROOT, unit_paths=[planned, unplanned], unit_kind=SESSION_UNIT, replan=False
    )

    assert summarized[0] == {
        "unit_path": str(planned),
        "unit_name": planned.name,
        "job_count": 2,
        "summed_memory_mb": 4608,
    }
    assert summarized[1]["job_count"] == 0
    assert "holds no job for this unit" in summarized[1]["error"]


def test_a_remote_plan_naming_no_unit_reprojects_the_project_alone(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """Naming no unit drops the per-unit planning step, which is how a caller reprojects an already planned project."""
    summarized = RemoteHost(server=connected_server).plan(
        project_root=SERVER_PROJECT_ROOT, unit_paths=[], unit_kind=SESSION_UNIT, replan=False
    )

    assert summarized == []
    issued = issued_command(server=stub_ssh_transport)
    assert "slf plan project" in issued
    assert "slf plan session" not in issued


def test_a_table_the_server_does_not_hold_reads_as_no_rows(connected_server: Server) -> None:
    """A project the server never planned reads cleanly rather than failing."""
    assert RemoteHost(server=connected_server).read_rows(path=SERVER_PROJECT_ROOT.joinpath("absent.feather")) == []


def test_fetching_an_artifact_copies_it_off_the_server_so_this_machine_keeps_it(
    connected_server: Server, stub_ssh_transport: StubSSHTransport, tmp_path: Path
) -> None:
    """The copy stays in place, so a snapshot survives the server regenerating its own artifacts afterward."""
    remote_artifact = SERVER_PROJECT_ROOT.joinpath("TestProject_jobs.feather")
    stub_ssh_transport.local_path(remote_artifact).parent.mkdir(parents=True, exist_ok=True)
    stub_ssh_transport.local_path(remote_artifact).write_bytes(b"recorded state")
    destination = tmp_path.joinpath("snapshot", "batch01")

    host = RemoteHost(server=connected_server)
    fetched = host.fetch(path=remote_artifact, destination=destination)

    assert fetched == destination.joinpath("TestProject_jobs.feather")
    assert fetched.read_bytes() == b"recorded state"
    assert host.fetch(path=SERVER_PROJECT_ROOT.joinpath("absent.feather"), destination=destination) is None


def test_a_remote_cleanup_naming_no_unit_issues_nothing(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A cleanup covering no unit has nothing to remove, so it never reaches the server."""
    assert RemoteHost(server=connected_server).clean(pipeline="video", unit_paths=[]) == []
    assert stub_ssh_transport.commands == []


def test_a_failing_remote_cleanup_reports_the_invocation_it_ran(
    connected_server: Server, stub_ssh_transport: StubSSHTransport
) -> None:
    """A caller needs the command and the server's own message, since one invocation can carry several commands."""
    stub_ssh_transport.respond("bash -lc", stderr="slf: no such pipeline", return_code=2)

    with pytest.raises(RuntimeError, match=r"'slf clean -p video -up .+' exited with code 2"):
        RemoteHost(server=connected_server).clean(pipeline="video", unit_paths=[SERVER_PROJECT_ROOT.joinpath("305")])


def test_the_remote_host_resolves_no_tracker_location_at_all(connected_server: Server) -> None:
    """A remotely dispatched job records its own outcome on the server, so naming a path here would only mislead."""
    assert (
        RemoteHost(server=connected_server).resolve_tracker_paths(
            pipeline="video", unit_paths=[SERVER_PROJECT_ROOT.joinpath("305", "2026-01-02-03-04-05-000006")]
        )
        == {}
    )


# Artifact locations


def test_the_state_artifacts_of_a_batch_follow_the_kind_of_unit_it_covers() -> None:
    """A session batch reads one table per project while a dataset batch reads one table per named dataset."""
    datasets = [SERVER_PROJECT_ROOT.joinpath("first"), SERVER_PROJECT_ROOT.joinpath("second")]

    assert state_artifact_paths(project_root=SERVER_PROJECT_ROOT, unit_paths=datasets, unit_kind=DATASET_UNIT) == [
        dataset.joinpath(DATASET_STATE_FILENAME) for dataset in datasets
    ]
    assert state_artifact_paths(
        project_root=SERVER_PROJECT_ROOT,
        unit_paths=[SERVER_PROJECT_ROOT.joinpath("305", "2026-01-02-03-04-05-000006")],
        unit_kind=SESSION_UNIT,
    ) == [project_jobs_path(project_directory=SERVER_PROJECT_ROOT)]


def test_the_plan_artifact_of_a_batch_is_its_projects_projection() -> None:
    """Every job's planned figures ship in one table per project, whichever kind of unit the batch covers."""
    assert plan_artifact_path(project_root=SERVER_PROJECT_ROOT) == project_plan_path(
        project_directory=SERVER_PROJECT_ROOT
    )
