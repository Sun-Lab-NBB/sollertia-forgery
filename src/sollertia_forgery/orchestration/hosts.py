"""Provides the execution hosts that materialize a project's artifacts and deliver them to this machine."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, Protocol
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from ataraxis_base_utilities import console

from ..forging import (
    DATASET_STATE_FILENAME,
    define_forging_dataset,
    generate_dataset_state,
    discover_project_datasets,
)
from .dispatch import resolve_dispatch
from .planning import (
    DATASET_UNIT,
    project_plan_path,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
)
from ..managing import project_jobs_path, generate_project_manifest
from .maintenance import reset_tracked_jobs, clean_pipeline_output

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

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Delivers a stored artifact to this machine durably, returning where it landed."""
        ...

    def plan(
        self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool
    ) -> list[dict[str, Any]]:
        """Records what the named units' jobs will cost and projects every plan cache into the table that ships."""
        ...

    def generate_state(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> None:
        """Rewrites the artifacts recording every tracked job's current status."""
        ...

    def reset_jobs(self, pipeline: str, unit_paths: Sequence[Path], job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs of several units to the scheduled state, in one operation."""
        ...

    def clean(self, pipeline: str, unit_paths: Sequence[Path]) -> list[dict[str, Any]]:
        """Removes a pipeline's output and tracker for the named units, reporting what each removal freed."""
        ...

    def define_dataset(
        self,
        project_root: Path,
        dataset_name: str,
        session_names: Sequence[str],
        recreate_animals: Sequence[str],
        *,
        force_recreate: bool,
    ) -> None:
        """Builds the dataset hierarchy a forging batch is resolved against."""
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
        LocalHost.plan(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind, replan=replan)
        LocalHost.generate_state(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind)

    @staticmethod
    def plan(project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> list[dict[str, Any]]:
        """Records what the named units' jobs will cost and projects every plan cache into the table that ships.

        Notes:
            Estimation reads each unit's raw acquisition data, so it costs far more than reading the cache it writes.
            Only the jobs a unit's cache does not already hold are estimated unless a caller asks for the recorded
            figures to be replaced, which is what keeps a figure a submission was sized against from moving underneath
            it.

            A unit no pipeline resolves any job for is reported in its own entry rather than aborting the others.

        Args:
            project_root: The path to the project's root directory.
            unit_paths: The unit root directories to plan.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.

        Returns:
            One entry per named unit, carrying its ``unit_path``, ``unit_name``, ``job_count``, and
            ``summed_memory_mb``, or its ``unit_path`` and the ``error`` that stopped it.
        """
        resolve = resolve_dataset_plan if unit_kind == DATASET_UNIT else resolve_session_plan

        planned: list[dict[str, Any]] = []
        for unit_path in unit_paths:
            try:
                unit_plan = resolve(unit_path, regenerate_plan=replan)
            except Exception as exception:
                planned.append({"unit_path": str(unit_path), "error": str(exception), "job_count": 0})
                continue
            planned.append(
                {
                    "unit_path": str(unit_path),
                    "unit_name": unit_plan.unit_name,
                    "job_count": len(unit_plan.entries),
                    "summed_memory_mb": sum(entry.memory_mb for entry in unit_plan.entries),
                }
            )

        generate_project_plan(project_directory=project_root)
        return planned

    @staticmethod
    def generate_state(project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> None:
        """Rewrites the artifacts recording every tracked job's current status.

        Notes:
            A session batch reads the project manifest's walk, which writes the project job artifact alongside it, while
            a dataset batch reads each named dataset's own state artifact.

        Args:
            project_root: The path to the project's root directory.
            unit_paths: The unit root directories the batch covers.
            unit_kind: Whether the units are sessions or datasets.
        """
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
    def fetch(path: Path, destination: Path) -> Path | None:  # noqa: ARG004
        """Reports where a stored artifact already sits, since this machine holds it.

        Notes:
            The artifact is already durable on this machine, so nothing is copied and the destination is ignored. A
            caller recording where a snapshot landed therefore records the artifact's own location.

        Args:
            path: The path to the artifact.
            destination: The directory a copy would land in, unused here.

        Returns:
            The artifact's own path, or None when it is absent.
        """
        return path if path.is_file() else None

    @staticmethod
    def reset_jobs(pipeline: str, unit_paths: Sequence[Path], job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs of several units to the scheduled state.

        Notes:
            The trackers sit on this machine, so the reset runs in this process rather than over a command line.

        Args:
            pipeline: The pipeline whose jobs to reset.
            unit_paths: The processing units that record them.
            job_ids: The identifiers to reset, or empty to reset every job each unit tracks.
        """
        reset_tracked_jobs(pipeline=pipeline, unit_paths=unit_paths, job_ids=job_ids)

    @staticmethod
    def clean(pipeline: str, unit_paths: Sequence[Path]) -> list[dict[str, Any]]:
        """Removes a pipeline's output and tracker for the named units.

        Args:
            pipeline: The pipeline whose output to remove.
            unit_paths: The processing units to clean.

        Returns:
            One entry per removed path, carrying the ``path`` and the ``removed_bytes`` it held.
        """
        return clean_pipeline_output(pipeline=pipeline, unit_paths=unit_paths)

    @staticmethod
    def define_dataset(
        project_root: Path,
        dataset_name: str,
        session_names: Sequence[str],
        recreate_animals: Sequence[str],
        *,
        force_recreate: bool,
    ) -> None:
        """Builds the dataset hierarchy a forging batch is resolved against.

        Args:
            project_root: The path to the project holding the sessions and the dataset.
            dataset_name: The name of the dataset to build.
            session_names: The sessions to admit into the dataset.
            recreate_animals: The animals to rebuild from the named sessions.
            force_recreate: Determines whether to rebuild the whole hierarchy from the named sessions.
        """
        define_forging_dataset(
            name=dataset_name,
            session_names=tuple(session_names),
            project_root=project_root,
            force_recreate=force_recreate,
            recreate_animals=tuple(recreate_animals),
        )

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
        self._run(
            commands=[
                *_plan_commands(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind, replan=replan),
                _state_command(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind),
            ]
        )

    def plan(
        self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool
    ) -> list[dict[str, Any]]:
        """Records what the named units' jobs will cost and projects every plan cache into the table that ships.

        Notes:
            The per-unit figures are read back out of the projection rather than parsed from the command's output, so a
            remote plan reports the same numbers a local one returns. A unit the projection holds no row for planned
            nothing.

        Args:
            project_root: The path to the project's root directory on the server.
            unit_paths: The unit root directories to plan.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.

        Returns:
            One entry per named unit, carrying its ``unit_path``, ``unit_name``, ``job_count``, and
            ``summed_memory_mb``, or its ``unit_path`` and the ``error`` that stopped it.

        Raises:
            RuntimeError: If the server-side commands fail.
        """
        self._run(
            commands=_plan_commands(
                project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind, replan=replan
            )
        )
        rows = self.read_rows(path=project_plan_path(project_directory=project_root))
        return _summarize_planned_units(rows=rows, unit_paths=unit_paths, unit_kind=unit_kind)

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

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Copies a stored artifact off the server so this machine keeps it.

        Notes:
            Unlike a read, this leaves the copy in place, so a snapshot taken at a run's closure survives the server
            regenerating its own artifacts afterward.

        Args:
            path: The path to the artifact on the server.
            destination: The local directory the copy lands in.

        Returns:
            The local path the copy landed at, or None when the server holds no such artifact.
        """
        if not self._server.exists(remote_path=path):
            return None
        destination.mkdir(parents=True, exist_ok=True)
        local_path = destination.joinpath(path.name)
        self._server.pull(local_path=local_path, remote_path=path)
        return local_path

    def generate_state(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> None:
        """Rewrites the artifacts recording every tracked job's current status.

        Args:
            project_root: The path to the project's root directory on the server.
            unit_paths: The unit root directories the batch covers.
            unit_kind: Whether the units are sessions or datasets.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        self._run(commands=[_state_command(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind)])

    def reset_jobs(self, pipeline: str, unit_paths: Sequence[Path], job_ids: Sequence[str]) -> None:
        """Returns the named tracked jobs of several units to the scheduled state on the server.

        Notes:
            One invocation carries every unit and every identifier, because each unit resets only the identifiers it
            actually tracks. A batch spanning many units therefore costs a single round trip.

        Args:
            pipeline: The pipeline whose jobs to reset.
            unit_paths: The paths, on the server, to the processing units that record them.
            job_ids: The identifiers to reset, or empty to reset every job each unit tracks.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        if not unit_paths:
            return
        command = ["slf", "reset", "-p", pipeline, *_repeated(flag="-up", values=unit_paths)]
        for job_id in job_ids:
            command.extend(("-id", job_id))
        self._run(commands=[command])

    def clean(self, pipeline: str, unit_paths: Sequence[Path]) -> list[dict[str, Any]]:
        """Removes a pipeline's output and tracker for the named units, on the server.

        Notes:
            The command reports each removed path with the bytes it held, so a remote cleanup returns the same figures
            a local one does.

        Args:
            pipeline: The pipeline whose output to remove.
            unit_paths: The paths, on the server, to the processing units to clean.

        Returns:
            One entry per removed path, carrying the ``path`` and the ``removed_bytes`` it held.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        if not unit_paths:
            return []
        command = ["slf", "clean", "-p", pipeline, *_repeated(flag="-up", values=unit_paths)]
        return _parse_removals(output=self._capture(commands=[command]))

    def define_dataset(
        self,
        project_root: Path,
        dataset_name: str,
        session_names: Sequence[str],
        recreate_animals: Sequence[str],
        *,
        force_recreate: bool,
    ) -> None:
        """Builds the dataset hierarchy a forging batch is resolved against, on the server.

        Notes:
            The forging command builds the hierarchy before running any tracked job, and naming no job leaves it with
            the definition alone to do.

        Args:
            project_root: The path, on the server, to the project holding the sessions and the dataset.
            dataset_name: The name of the dataset to build.
            session_names: The sessions to admit into the dataset.
            recreate_animals: The animals to rebuild from the named sessions.
            force_recreate: Determines whether to rebuild the whole hierarchy from the named sessions.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        command = ["slf", "forge", "-dn", dataset_name, "-pp", str(project_root), "-np"]
        for session_name in session_names:
            command.extend(("-s", session_name))
        for animal in recreate_animals:
            command.extend(("-ra", animal))
        if force_recreate:
            command.append("-f")
        self._run(commands=[command])

    @staticmethod
    def resolve_tracker_paths(
        pipeline: str,  # noqa: ARG004
        unit_paths: Sequence[Path],  # noqa: ARG004
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

    def _capture(self, commands: Sequence[Sequence[str]]) -> str:
        """Issues commands inside the server's shared processing environment and returns what they printed.

        Args:
            commands: The commands to run, in order, each as an argument vector.

        Returns:
            The invocation's standard output.

        Raises:
            RuntimeError: If the invocation exits with a non-zero status.
        """
        result = self._server.execute_command(
            command=environment_commands(environment=self._server.environment, commands=commands)
        )
        if result.return_code != 0:
            _raise_command_failure(commands=commands, return_code=result.return_code, stderr=result.stderr)
        return result.stdout

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
            _raise_command_failure(commands=commands, return_code=result.return_code, stderr=result.stderr)


def _plan_commands(project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> list[list[str]]:
    """Renders the commands that plan the named units and project every plan cache under their project.

    Args:
        project_root: The path to the project's root directory on the server.
        unit_paths: The unit root directories to plan.
        unit_kind: Whether the units are sessions or datasets.
        replan: Determines whether to re-estimate the figures a cache already holds.

    Returns:
        The commands as argument vectors, in the order they run.
    """
    is_dataset = unit_kind == DATASET_UNIT
    plan = ["slf", "plan", "dataset" if is_dataset else "session"]
    plan.extend(_repeated(flag="-dp" if is_dataset else "-sp", values=unit_paths))
    if replan:
        plan.append("-rp")

    projection = ["slf", "plan", "project", "-pp", str(project_root)]
    # Naming no unit leaves the projection alone to run, which is how a caller reprojects an already planned project.
    return [projection] if not unit_paths else [plan, projection]


def _summarize_planned_units(
    rows: list[dict[str, Any]], unit_paths: Sequence[Path], unit_kind: str
) -> list[dict[str, Any]]:
    """Summarizes what the projection now holds for each named unit.

    Args:
        rows: The project's plan rows.
        unit_paths: The unit root directories that were planned.
        unit_kind: Whether the units are sessions or datasets, which names the column holding each row's unit.

    Returns:
        One entry per named unit, carrying its counts or the reason it contributed none.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        unit = row.get(unit_kind)
        if unit is not None:
            grouped.setdefault(str(unit), []).append(row)

    summarized: list[dict[str, Any]] = []
    for unit_path in unit_paths:
        planned = grouped.get(unit_path.name, [])
        if not planned:
            summarized.append(
                {
                    "unit_path": str(unit_path),
                    "error": "The project's plan projection holds no job for this unit, so it planned nothing.",
                    "job_count": 0,
                }
            )
            continue
        summarized.append(
            {
                "unit_path": str(unit_path),
                "unit_name": unit_path.name,
                "job_count": len(planned),
                "summed_memory_mb": sum(int(row["memory_mb"]) for row in planned),
            }
        )
    return summarized


def _state_command(project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> list[str]:
    """Renders the command that rewrites the artifacts recording every tracked job's status.

    Args:
        project_root: The path to the project's root directory on the server.
        unit_paths: The unit root directories the batch covers.
        unit_kind: Whether the units are sessions or datasets.

    Returns:
        The command as an argument vector.
    """
    if unit_kind == DATASET_UNIT:
        return ["slf", "dataset-state", *_repeated(flag="-dp", values=unit_paths)]
    return ["slf", "manifest", "-pp", str(project_root), "create"]


def _parse_removals(output: str) -> list[dict[str, Any]]:
    """Reads the removals a server-side cleanup reported.

    Notes:
        Each line pairs the bytes a path held with the path itself, which is what lets a remote cleanup report the same
        figures a local one returns. A line that does not parse is skipped, so unrelated output never becomes a removal.

    Args:
        output: The cleanup command's standard output.

    Returns:
        One entry per removed path, carrying the ``path`` and the ``removed_bytes`` it held.
    """
    removals: list[dict[str, Any]] = []
    for line in output.splitlines():
        size, _, path = line.strip().partition(" ")
        if path and size.isdigit():
            removals.append({"path": path, "removed_bytes": int(size)})
    return removals


def _raise_command_failure(commands: Sequence[Sequence[str]], return_code: int, stderr: str) -> None:
    """Reports a server-side invocation that exited with a non-zero status.

    Args:
        commands: The commands the invocation carried.
        return_code: The status the invocation exited with.
        stderr: The invocation's standard error.

    Raises:
        RuntimeError: Always, since this reports a failure the caller cannot proceed past.
    """
    rendered = " && ".join(shlex.join(command) for command in commands)
    message = f"The server-side invocation '{rendered}' exited with code {return_code}. {stderr.strip()}"
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
