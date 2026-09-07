"""Provides the execution hosts that materialize a project's artifacts and deliver them to this machine."""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING, Any, NoReturn, Protocol
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
from ..shared_assets import posix_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..server import Server


class ExecutionHost(Protocol):
    """Declares the operations that preparation runs against the host that holds a project's data.

    Notes:
        Every implementation runs the same underlying functions, so the artifacts that a caller reads describe the
        same project state either way. A local host calls them in this process and a remote host runs the command
        line that calls them on the server, so one preparation path serves both.

        One operation differs by design. ``resolve_tracker_paths`` yields an empty mapping on a remote host, because
        a remotely dispatched job's trackers are read and written on the server alone.
    """

    @property
    def label(self) -> str:
        """Returns the name under which this host is reported."""
        ...

    def materialize(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact from which a batch is resolved, for the named units and their project."""
        ...

    def plan(
        self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool
    ) -> list[dict[str, Any]]:
        """Records what the named units' jobs will cost and projects every plan cache into the table that ships."""
        ...

    def generate_state(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> None:
        """Rewrites the artifacts recording every tracked job's current status."""
        ...

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Reads a stored table into plain rows, returning an empty list when the host holds no such table."""
        ...

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Delivers a stored artifact to this machine durably, returning where it landed, or None when it is absent."""
        ...

    def reset_jobs(self, pipeline: str, job_ids_by_unit: Mapping[Path, Sequence[str]]) -> None:
        """Resets each unit's own tracked jobs to the scheduled state, in one operation."""
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
        """Builds the dataset hierarchy against which a forging batch is resolved."""
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

    def __repr__(self) -> str:
        """Returns a string representation of the LocalHost instance."""
        return "LocalHost()"

    @property
    def label(self) -> str:
        """Returns the name under which this host is reported."""
        return "local"

    @classmethod
    def materialize(cls, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact from which a batch is resolved, for the named units and their project.

        Notes:
            Three steps run in a fixed order. Planning each unit records what its jobs will cost and registers them on
            their processing trackers. Projecting the project's plan caches gathers those figures into the single table
            that ships. Refreshing the state serializes what the trackers now record.

            The order matters, because planning is what registers a unit's jobs and the state step reads those
            registries. Planning also re-estimates nothing a unit's cache already holds unless a caller asks for it,
            so a submission's sizing figures never change underneath it.

            A unit for which no pipeline resolves a job is reported and skipped, leaving the units that did resolve
            jobs planned.

        Args:
            project_root: The path to the project's root directory.
            unit_paths: The unit root directories the batch covers.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.
        """
        cls.plan(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind, replan=replan)
        cls.generate_state(project_root=project_root, unit_paths=unit_paths, unit_kind=unit_kind)

    @staticmethod
    def plan(project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> list[dict[str, Any]]:
        """Records what the named units' jobs will cost and projects every plan cache into the table that ships.

        Notes:
            Estimation reads each unit's raw acquisition data, so it costs far more than reading the cache it writes.
            Only the jobs that a unit's cache does not already hold are estimated unless a caller asks for the
            recorded figures to be replaced, so a submission's sizing figures never move underneath it.

            A unit for which no pipeline resolves a job is reported in its own entry rather than aborting the others.
            A unit is planned without any job that the sizing pass refuses. Each refusal's reason is reported beside
            the figures the unit did plan.

        Args:
            project_root: The path to the project's root directory.
            unit_paths: The unit root directories to plan.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.

        Returns:
            One entry per named unit, carrying its ``unit_path``, ``unit_name``, ``job_count``, and
            ``summed_memory_mb``, and ``summed_resident_mb``, or its ``unit_path``, a ``job_count`` of zero, and the
            ``error`` that stopped it. An
            entry whose plan recorded a sizing refusal also carries ``unsized_jobs``, mapping each refusal to the
            reason it gave. A refusal raised for one job is keyed by its pipeline and job, and a refusal that ended a
            pipeline's one-pass sizing is keyed by that pipeline and ``all jobs``.
        """
        resolve = resolve_dataset_plan if unit_kind == DATASET_UNIT else resolve_session_plan

        planned: list[dict[str, Any]] = []
        for unit_path in unit_paths:
            try:
                unit_plan = resolve(unit_path, regenerate_plan=replan)
            except Exception as exception:
                planned.append({"unit_path": posix_text(path=unit_path), "error": str(exception), "job_count": 0})
                continue
            summary: dict[str, Any] = {
                "unit_path": posix_text(path=unit_path),
                "unit_name": unit_plan.unit_name,
                "job_count": len(unit_plan.entries),
                "summed_memory_mb": sum(entry.memory_mb for entry in unit_plan.entries),
                "summed_resident_mb": sum(entry.resident_mb for entry in unit_plan.entries),
            }
            if unit_plan.unsized_jobs:
                summary["unsized_jobs"] = dict(unit_plan.unsized_jobs)
            planned.append(summary)

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
            destination: The directory in which a copy would land, unused here.

        Returns:
            The artifact's own path, or None when it is absent.
        """
        return path if path.is_file() else None

    @staticmethod
    def reset_jobs(pipeline: str, job_ids_by_unit: Mapping[Path, Sequence[str]]) -> None:
        """Resets each unit's own tracked jobs to the scheduled state.

        Notes:
            The trackers sit on this machine, so the reset runs in this process rather than over a command line.

            Each unit is reset against its own identifiers alone. A job identifier carries no unit, so two units of one
            project share the identifier of the same stage, and a flat set applied to both would clear a record the
            caller never named.

        Args:
            pipeline: The pipeline whose jobs to reset.
            job_ids_by_unit: The identifiers to reset, keyed by the processing unit that records them. A unit mapped to
                an empty sequence has every job it tracks reset.
        """
        for unit_path, job_ids in job_ids_by_unit.items():
            reset_tracked_jobs(pipeline=pipeline, unit_paths=[unit_path], job_ids=job_ids)

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
        """Builds the dataset hierarchy against which a forging batch is resolved.

        Args:
            project_root: The path to the project holding the sessions and the dataset.
            dataset_name: The name of the dataset to build.
            session_names: The sessions to admit into the dataset.
            recreate_animals: The animals to rebuild from the named sessions.
            force_recreate: Determines whether to rebuild the whole hierarchy from the named sessions.

        Raises:
            ValueError: If the arguments contradict each other, or if the dataset's acquisition system is unknown.
            FileNotFoundError: If a named session resolves to no directory under the project root, or if the
                acquisition system's resolver reports a missing input it needs for an animal.
            RuntimeError: If a named session resolves to more than one directory under the project root.
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
            unit_paths: The unit root directories whose trackers to locate.

        Returns:
            The tracker path of each unit, keyed by the unit path as a string.
        """
        dispatch = resolve_dispatch(pipeline=pipeline)
        if dispatch is None:
            return {}

        resolved: dict[str, str] = {}
        for unit_path in unit_paths:
            try:
                resolved[posix_text(path=unit_path)] = posix_text(path=dispatch.tracker_path(dispatch.load(unit_path)))
            except Exception as exception:
                console.echo(message=f"Unable to locate the '{pipeline}' tracker for '{unit_path}'. {exception}")
        return resolved


class RemoteHost:
    """Runs a project's preparation steps on the remote compute server and copies its artifacts home.

    Notes:
        Each operation issues the command line that calls the same function a local host calls in-process, so the two
        hosts write the same artifacts from the same code. Reading a table copies it into a temporary directory and
        parses it here, because the graph that a batch dispatches is always built on this machine.

    Args:
        server: The connected compute server holding the project.

    Attributes:
        _server: The connected server through which every operation is issued.
    """

    def __init__(self, server: Server) -> None:
        self._server: Server = server

    def __repr__(self) -> str:
        """Returns a string representation of the RemoteHost instance."""
        return f"RemoteHost(host={self._server.host}, root={posix_text(path=self._server.root)})"

    @property
    def label(self) -> str:
        """Returns the name under which this host is reported."""
        return "remote"

    @property
    def server(self) -> Server:
        """Returns the connected server through which this host operates."""
        return self._server

    def materialize(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str, *, replan: bool) -> None:
        """Rewrites every artifact from which a batch is resolved, for the named units and their project.

        Notes:
            The steps ship as one invocation, chained so each runs only after the one before it succeeded. Naming no
            unit drops the per-unit planning step and ships the projection and the state refresh alone. Chaining costs
            a single round trip, and it holds the order the steps require, since planning registers the jobs the state
            step reads.

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
            remote plan reports the same numbers a local one returns. A unit for which the projection holds no row
            planned nothing.

        Args:
            project_root: The path to the project's root directory on the server.
            unit_paths: The unit root directories to plan.
            unit_kind: Whether the units are sessions or datasets.
            replan: Determines whether to re-estimate the figures a cache already holds.

        Returns:
            One entry per named unit, carrying its ``unit_path``, ``unit_name``, ``job_count``, and
            ``summed_memory_mb``, and ``summed_resident_mb``, or its ``unit_path``, a ``job_count`` of zero, and
            the ``error`` that stopped it.

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
            The copy stays in place, so a snapshot taken at a run's closure survives the server regenerating its own
            artifacts afterward.

        Args:
            path: The path to the artifact on the server.
            destination: The local directory in which the copy lands.

        Returns:
            The local path at which the copy landed, or None when the server holds no such artifact.
        """
        if not self._server.exists(remote_path=path):
            return None
        destination.mkdir(parents=True, exist_ok=True)
        local_path = destination.joinpath(path.name)
        self._server.pull(local_path=local_path, remote_path=path)
        return local_path

    def reset_jobs(self, pipeline: str, job_ids_by_unit: Mapping[Path, Sequence[str]]) -> None:
        """Resets each unit's own tracked jobs to the scheduled state on the server.

        Notes:
            One invocation is issued per unit, carrying that unit's identifiers alone. A job identifier carries no
            unit, so two units of one project share the identifier of the same stage, and a flat set applied to both
            would clear a record the caller never named.

            The invocations are chained into one round trip, so a batch spanning many units still costs a single
            connection.

        Args:
            pipeline: The pipeline whose jobs to reset.
            job_ids_by_unit: The identifiers to reset, keyed by the path, on the server, to the processing unit that
                records them. A unit mapped to an empty sequence has every job it tracks reset.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        commands: list[list[str]] = []
        for unit_path, job_ids in job_ids_by_unit.items():
            command = ["slf", "reset", "-p", pipeline, *_repeated(flag="-up", values=[unit_path])]
            for job_id in job_ids:
                command.extend(("-id", job_id))
            commands.append(command)
        if not commands:
            return
        self._run(commands=commands)

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
        """Builds the dataset hierarchy against which a forging batch is resolved, on the server.

        Notes:
            The invocation calls ``define_forging_dataset`` inside the server's processing environment, because the
            forging command couples the definition to the tracked forging jobs that follow it. A remote definition
            therefore performs the same work its local counterpart does, and the resulting jobs are dispatched by a
            prepared batch.

        Args:
            project_root: The path, on the server, to the project holding the sessions and the dataset.
            dataset_name: The name of the dataset to build.
            session_names: The sessions to admit into the dataset.
            recreate_animals: The animals to rebuild from the named sessions.
            force_recreate: Determines whether to rebuild the whole hierarchy from the named sessions.

        Raises:
            RuntimeError: If the server-side command fails.
        """
        self._run(
            commands=[
                _definition_command(
                    project_root=project_root,
                    dataset_name=dataset_name,
                    session_names=session_names,
                    recreate_animals=recreate_animals,
                    force_recreate=force_recreate,
                )
            ]
        )

    @staticmethod
    def resolve_tracker_paths(
        pipeline: str,  # noqa: ARG004
        unit_paths: Sequence[Path],  # noqa: ARG004
    ) -> dict[str, str]:
        """Returns an empty mapping, because this host's trackers are read and written on the server alone.

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
            command=_environment_commands(environment=self._server.environment, commands=commands)
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
            command=_environment_commands(environment=self._server.environment, commands=commands)
        )
        if result.return_code != 0:
            _raise_command_failure(commands=commands, return_code=result.return_code, stderr=result.stderr)


def _environment_commands(environment: str, commands: Sequence[Sequence[str]]) -> str:
    """Wraps several commands so they run in order inside the server's shared processing environment.

    Notes:
        The commands are chained so each runs only after the one before it succeeded, so one round trip carries a
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
    return _environment_commands(environment=environment, commands=[command])


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

    projection = ["slf", "plan", "project", "-pp", posix_text(path=project_root)]
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
                    "unit_path": posix_text(path=unit_path),
                    "error": "The project's plan projection holds no job for this unit, so it planned nothing.",
                    "job_count": 0,
                }
            )
            continue
        summarized.append(
            {
                "unit_path": posix_text(path=unit_path),
                "unit_name": unit_path.name,
                "job_count": len(planned),
                "summed_memory_mb": sum(int(row["memory_mb"]) for row in planned),
                "summed_resident_mb": sum(int(row["resident_mb"]) for row in planned),
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
    return ["slf", "manifest", "-pp", posix_text(path=project_root), "create"]


def _parse_removals(output: str) -> list[dict[str, Any]]:
    """Reads the removals a server-side cleanup reported.

    Notes:
        Each line pairs the bytes a path held with the path itself, so a remote cleanup reports the same figures a
        local one returns. A line that does not parse is skipped, so unrelated output never becomes a removal.

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


def _definition_command(
    project_root: Path,
    dataset_name: str,
    session_names: Sequence[str],
    recreate_animals: Sequence[str],
    *,
    force_recreate: bool,
) -> list[str]:
    """Renders the command that builds a dataset hierarchy on the server.

    Notes:
        Every string argument is embedded as a JSON literal, which the Python parser reads as the same literal, while
        the boolean is interpolated as its Python repr. A name carrying a space or a quote therefore survives both the
        parser and the shell quoting applied around it.

    Args:
        project_root: The path, on the server, to the project holding the sessions and the dataset.
        dataset_name: The name of the dataset to build.
        session_names: The sessions to admit into the dataset.
        recreate_animals: The animals to rebuild from the named sessions.
        force_recreate: Determines whether to rebuild the whole hierarchy from the named sessions.

    Returns:
        The command as an argument vector.
    """
    script = (
        f"from pathlib import Path; "
        f"from sollertia_forgery.forging import define_forging_dataset; "
        f"define_forging_dataset(name={json.dumps(dataset_name)}, "
        f"session_names=tuple({json.dumps(list(session_names))}), "
        f"project_root=Path({json.dumps(posix_text(path=project_root))}), "
        f"force_recreate={force_recreate}, "
        f"recreate_animals=tuple({json.dumps(list(recreate_animals))}))"
    )
    return ["python", "-c", script]


def _raise_command_failure(commands: Sequence[Sequence[str]], return_code: int, stderr: str) -> NoReturn:
    """Reports a server-side invocation that exited with a non-zero status.

    Args:
        commands: The commands the invocation carried.
        return_code: The exit status of the invocation.
        stderr: The invocation's standard error.

    Raises:
        RuntimeError: Always, since this reports a failure that stops the caller.
    """
    rendered = " && ".join(shlex.join(command) for command in commands)
    message = (
        f"Unable to complete the server-side invocation '{rendered}'. The invocation must exit with code 0, but it "
        f"exited with code {return_code}. {stderr.strip()}"
    )
    console.error(message=message, error=RuntimeError)


def _repeated(flag: str, values: Sequence[Path]) -> list[str]:
    """Expands one repeated command-line option over every value it is given.

    Args:
        flag: The option flag to repeat.
        values: The paths to pass.

    Returns:
        The flattened argument list.
    """
    return [argument for value in values for argument in (flag, posix_text(path=value))]
