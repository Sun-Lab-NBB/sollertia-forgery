"""Provides the execution hosts that materialize a project's artifacts and deliver them to this machine."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, Protocol
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from ataraxis_base_utilities import console

from ..forging import DATASET_STATE_FILENAME, generate_dataset_state, discover_project_datasets
from .dispatch import resolve_dispatch
from .planning import (
    DATASET_UNIT,
    project_plan_path,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
)
from ..managing import project_jobs_path, generate_project_manifest
from .reconcile import reset_tracked_jobs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..server import Server


class ExecutionHost(Protocol):
    """Declares the operations preparation runs against the host that holds a project's data.

    Notes:
        Every implementation runs the same underlying functions, so the artifacts a caller reads describe the same
        project state either way. A local host calls them in this process and a remote host runs the command line that
        calls them on the server, which is what keeps one preparation path serving both.
    """

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        ...

    def materialize(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact a batch is resolved from, for the named units and their project."""
        ...

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Reads a stored table into plain rows, returning nothing when the host holds no such table."""
        ...

    def reset_jobs(self, pipeline: str, unit_path: Path, job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs to the scheduled state on the host that records them."""
        ...

    def resolve_tracker_paths(self, pipeline: str, unit_paths: Sequence[Path]) -> dict[str, str]:
        """Resolves where each unit's pipeline tracker sits, for a caller that will open it directly."""
        ...


class LocalHost:
    """Runs a project's preparation steps in this process, against data this machine holds.

    Notes:
        Calls the same functions the command line exposes, so a locally prepared batch and a remotely prepared one are
        resolved from artifacts written by identical code.
    """

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "local"

    @staticmethod
    def materialize(project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact a batch is resolved from, for the named units and their project.

        Notes:
            Three steps run in a fixed order. Planning each unit records what its jobs will cost and registers them on
            their processing trackers. Projecting the project's plan caches gathers those figures into the single table
            that ships. Refreshing the state serializes what the trackers now record.

            The order matters, because planning is what registers a unit's jobs and the state step reads those
            registries. Planning also re-estimates nothing a unit's cache already holds unless a caller asks for it,
            which is what keeps the figures a submission was sized against from changing underneath it.

            A unit no pipeline resolves any job for is reported and skipped, leaving the units that did resolve jobs
            planned.

        Args:
            project_root: The path to the project's root directory.
            unit_paths: The unit root directories the batch covers.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.
        """
        resolve = resolve_dataset_plan if unit_kind == DATASET_UNIT else resolve_session_plan
        for unit_path in unit_paths:
            try:
                resolve(unit_path, regenerate_plan=replan)
            except Exception as exception:
                console.echo(message=f"Unable to plan '{unit_path}'. {exception}")

        generate_project_plan(project_directory=project_root)

        # A session batch reads the project manifest's walk, which writes the project job artifact alongside it, while a
        # dataset batch reads each named dataset's own state artifact.
        if unit_kind == DATASET_UNIT:
            named = set(unit_paths)
            for dataset in discover_project_datasets(project_root=project_root):
                if dataset.dataset_data_path.parent in named:
                    generate_dataset_state(dataset=dataset)
            return
        generate_project_manifest(project_directory=project_root)

    @staticmethod
    def read_rows(path: Path) -> list[dict[str, Any]]:
        """Reads a stored table into plain rows.

        Args:
            path: The path to the table.

        Returns:
            The table's rows, or an empty list when the table is absent.
        """
        if not path.is_file():
            return []
        return pl.read_ipc(source=path, memory_map=True).to_dicts()

    @staticmethod
    def reset_jobs(pipeline: str, unit_path: Path, job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs to the scheduled state.

        Notes:
            The trackers sit on this machine, so the reset runs in this process rather than over a command line.

        Args:
            pipeline: The pipeline whose jobs to reset.
            unit_path: The path to the processing unit that records them.
            job_ids: The identifiers of the jobs to reset.
        """
        reset_tracked_jobs(pipeline=pipeline, unit_path=unit_path, job_ids=job_ids)

    @staticmethod
    def resolve_tracker_paths(pipeline: str, unit_paths: Sequence[Path]) -> dict[str, str]:
        """Resolves where each unit's pipeline tracker sits on this machine.

        Notes:
            The local execution engine opens these files directly. It reads the recorded outcomes to decide which
            prerequisites are satisfied, and re-reads them to report a running batch's status. A batch dispatched here
            therefore carries the locations on its descriptors, and a unit that cannot be loaded contributes no entry.

        Args:
            pipeline: The pipeline whose tracker to locate.
            unit_paths: The unit root directories to locate trackers for.

        Returns:
            The tracker path of each unit, keyed by the unit path as a string.
        """
        dispatch = resolve_dispatch(pipeline=pipeline)
        if dispatch is None:
            return {}

        resolved: dict[str, str] = {}
        for unit_path in unit_paths:
            try:
                resolved[str(unit_path)] = str(dispatch.tracker_path(dispatch.load(unit_path)))
            except Exception as exception:
                console.echo(message=f"Unable to locate the '{pipeline}' tracker for '{unit_path}'. {exception}")
        return resolved


class RemoteHost:
    """Runs a project's preparation steps on the remote compute server and copies its artifacts home.

    Notes:
        Each operation issues the command line that calls the same function a local host calls in-process, so the two
        hosts write the same artifacts from the same code. Reading a table copies it into a temporary directory and
        parses it here, because the graph a batch dispatches is always built on this machine.

    Args:
        server: The connected compute server holding the project.

    Attributes:
        _server: The connected server every operation is issued through.
    """

    def __init__(self, server: Server) -> None:
        self._server: Server = server

    def __repr__(self) -> str:
        """Returns a string representation of the RemoteHost instance."""
        return f"RemoteHost(host={self._server.host}, root={self._server.root})"

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "remote"

    @property
    def server(self) -> Server:
        """Returns the connected server this host operates through."""
        return self._server

    def materialize(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact a batch is resolved from, for the named units and their project.

        Notes:
            The three steps ship as one invocation, chained so each runs only after the one before it succeeded. That
            costs a single round trip, and it holds the order the steps require, since planning registers the jobs the
            state step reads.

        Args:
            project_root: The path to the project's root directory on the server.
            unit_paths: The unit root directories the batch covers.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.

        Raises:
            RuntimeError: If the server-side commands fail.
        """
        is_dataset = unit_kind == DATASET_UNIT
        plan = ["slf", "plan", "dataset" if is_dataset else "session"]
        plan.extend(_repeated(flag="-dp" if is_dataset else "-sp", values=unit_paths))
        if replan:
            plan.append("-rp")

        state = (
            ["slf", "dataset-state", *_repeated(flag="-dp", values=unit_paths)]
            if is_dataset
            else ["slf", "manifest", "-pp", str(project_root), "create"]
        )

        self._run(commands=[plan, ["slf", "plan", "project", "-pp", str(project_root)], state])

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Copies a stored table home and reads it into plain rows.

        Args:
            path: The path to the table on the server.

        Returns:
            The table's rows, or an empty list when the server holds no such table.
        """
        if not self._server.exists(remote_path=path):
            return []
        with TemporaryDirectory() as staging_directory:
            local_path = Path(staging_directory).joinpath(path.name)
            self._server.pull(local_path=local_path, remote_path=path)
            return pl.read_ipc(source=local_path, memory_map=False).to_dicts()

    def reset_jobs(self, pipeline: str, unit_path: Path, job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs to the scheduled state on the server.

        Notes:
            The trackers sit beside the data on the server, so the reset runs there through a command line that calls
            the same tracker primitive a local reset calls directly.

        Args:
            pipeline: The pipeline whose jobs to reset.
            unit_path: The path, on the server, to the processing unit that records them.
            job_ids: The identifiers of the jobs to reset.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        if not job_ids:
            return
        command = ["slf", "reset", "-p", pipeline, "-up", str(unit_path)]
        for job_id in job_ids:
            command.extend(("-id", job_id))
        self._run(commands=[command])

    def resolve_tracker_paths(
        self,
        pipeline: str,  # noqa: ARG002
        unit_paths: Sequence[Path],  # noqa: ARG002
    ) -> dict[str, str]:
        """Resolves nothing, because this host's trackers are only ever read and written on the server.

        Notes:
            A remotely dispatched job records its own outcome on the server, its reset runs there over a command line,
            and its progress is read from the scheduler. Nothing on this machine opens one of these trackers, so
            recording a path this machine cannot read would only mislead.

        Args:
            pipeline: The pipeline whose tracker would be located.
            unit_paths: The unit root directories that would be located.

        Returns:
            An empty mapping.
        """
        return {}

    def _run(self, commands: Sequence[Sequence[str]]) -> None:
        """Issues one or more commands inside the server's shared processing environment.

        Args:
            commands: The commands to run, in order, each as an argument vector.

        Raises:
            RuntimeError: If the invocation exits with a non-zero status.
        """
        result = self._server.execute_command(
            command=environment_commands(environment=self._server.environment, commands=commands)
        )
        if result.return_code != 0:
            rendered = " && ".join(shlex.join(command) for command in commands)
            message = (
                f"Unable to prepare the batch. The server-side invocation '{rendered}' exited with code "
                f"{result.return_code}. {result.stderr.strip()}"
            )
            console.error(message=message, error=RuntimeError)


def environment_commands(environment: str, commands: Sequence[Sequence[str]]) -> str:
    """Wraps several commands so they run in order inside the server's shared processing environment.

    Notes:
        The commands are chained so each runs only after the one before it succeeded, which lets one round trip carry a
        sequence whose later steps depend on its earlier ones.

    Args:
        environment: The name of the conda environment to activate.
        commands: The commands to run, in order, each as an argument vector.

    Returns:
        The shell command to issue on the server.
    """
    activation = f'eval "$(conda shell.bash hook)" && source activate {shlex.quote(environment)}'
    chained = " && ".join(shlex.join(command) for command in commands)
    return f"bash -lc {shlex.quote(f'{activation} && {chained}')}"


def environment_command(environment: str, command: Sequence[str]) -> str:
    """Wraps one command so it runs inside the server's shared processing environment.

    Args:
        environment: The name of the conda environment to activate.
        command: The command to run, as an argument vector.

    Returns:
        The shell command to issue on the server.
    """
    return environment_commands(environment=environment, commands=[command])


def state_artifact_paths(project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> list[Path]:
    """Resolves the artifacts holding every tracked job's recorded status.

    Notes:
        A session batch reads one table per project, while a dataset batch reads one table per named dataset, so the
        two unit kinds resolve different counts of artifact.

    Args:
        project_root: The path to the project's root directory.
        unit_paths: The unit root directories the batch covers.
        unit_kind: Whether the units are sessions or datasets.

    Returns:
        The paths to read, in the order they are read.
    """
    if unit_kind == DATASET_UNIT:
        return [unit_path.joinpath(DATASET_STATE_FILENAME) for unit_path in unit_paths]
    return [project_jobs_path(project_directory=project_root)]


def plan_artifact_path(project_root: Path) -> Path:
    """Resolves the artifact holding every job's planned figures.

    Args:
        project_root: The path to the project's root directory.

    Returns:
        The path to the project's plan table.
    """
    return project_plan_path(project_directory=project_root)


def _repeated(flag: str, values: Sequence[Path]) -> list[str]:
    """Expands one repeated command-line option over every value it is given.

    Args:
        flag: The option flag to repeat.
        values: The paths to pass.

    Returns:
        The flattened argument list.
    """
    return [argument for value in values for argument in (flag, str(value))]
