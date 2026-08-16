"""Provides the shared fixtures the sollertia-forgery test suite builds its on-disk artifacts and its stubs from."""

from __future__ import annotations

from stat import S_IFDIR, S_IFREG
import shlex
from types import SimpleNamespace
import shutil
from typing import TYPE_CHECKING, Any
from pathlib import Path
import subprocess
from dataclasses import field, fields, dataclass

import cv2
import numpy as np
import pandas as pd
import polars as pl
import pytest
import platformdirs
from sollertia_shared_assets import (
    Cue,
    ProjectData,
    SessionData,
    TriggerType,
    SessionTypes,
    TaskTemplate,
    VREnvironment,
    TrialStructure,
    ExperimentState,
    AcquisitionSystems,
    MesoscopeHardwareState,
    MesoscopeWaterRewardTrial,
    MesoscopeExperimentConfiguration,
    set_working_directory,
    set_task_templates_directory,
)
from ataraxis_data_structures import ProcessingTracker
from sollertia_shared_assets.registries import DESCRIPTOR_REGISTRY, SESSION_TYPES_USING_VR_TASK

from sollertia_forgery.server import Server
from sollertia_forgery.managing import project_manifest_path, generate_project_manifest
import sollertia_forgery.server.server as server_module
from sollertia_forgery.shared_assets import SESSION_PIPELINES, resolve_session_tracker_path
import sollertia_forgery.orchestration.ledger as ledger_module
import sollertia_forgery.orchestration.remote as remote_module
import sollertia_forgery.orchestration.batches as batches_module
import sollertia_forgery.orchestration.closure as closure_module
import sollertia_forgery.orchestration.footprints as footprints_module
from sollertia_forgery.server.server_configuration import ServerConfiguration, create_server_configuration_file

if TYPE_CHECKING:
    from collections.abc import Mapping, Callable, Iterator, Sequence

    from numpy.typing import NDArray

FIXTURES_DIRECTORY: Path = Path(__file__).parent / "fixtures"
"""The directory holding the static reference files the suite pins its golden comparisons against. Resolving it here
keeps a test's own depth under the suite root out of the path."""

PROJECT_NAME: str = "TestProject"
"""The name of the project every hierarchy fixture builds under the temporary data root."""

EXPERIMENT_ANIMAL_ID: str = "305"
"""The animal identifier the experiment-session fixture records under."""

TRAINING_ANIMAL_ID: str = "321"
"""The animal identifier the training-session fixture records under, kept distinct so a project holds two animals."""

PYTHON_VERSION: str = "3.14.0"
"""The acquisition-time Python version every created session records in its marker."""

EXPERIMENT_VERSION: str = "1.0.0"
"""The acquisition-time sollertia-experiment version every created session records in its marker."""

SERVER_ROOT: str = "/data/sollertia"
"""The absolute server-side data root the server-configuration fixture records."""

SERVER_ENVIRONMENT: str = "slf_server"
"""The name of the server-side conda environment every remote command activates."""

FROZEN_TIMESTAMP_US: int = 1_700_000_000_000_000
"""The microsecond epoch the frozen clock reports until a test moves it."""

MOTION_ENERGY_FRAME_HEIGHT: int = 100
"""The height of a synthetic recording frame, chosen so it is not a multiple of the analysis bin size."""

MOTION_ENERGY_FRAME_WIDTH: int = 64
"""The width of a synthetic recording frame, chosen so it is not a multiple of the analysis bin size."""

_SLURM_FIRST_JOB_ID: int = 1000
"""The allocation identifier the stub scheduler assigns to the first submission it accepts."""


# Platform isolation


@pytest.fixture
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the Sollertia platform working directory at a pristine temporary location.

    The submission ledger, the prepared-batch registry, the batch outcomes, and the server configuration are all
    written under that directory. Requesting this fixture keeps every one of them inside the test's own temporary
    directory rather than in whatever this host already holds.

    Args:
        tmp_path: The temporary directory the platform state is placed under.
        monkeypatch: The fixture used to redirect the user data directory the platform resolves.

    Returns:
        The path to the isolated working directory.
    """
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda *_args, **_kwargs: str(tmp_path.joinpath("platform")))
    working_directory = tmp_path.joinpath("working")
    set_working_directory(path=working_directory)
    return working_directory


# Project hierarchy


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Creates the temporary data root every project fixture anchors its hierarchy on.

    Args:
        tmp_path: The temporary directory the data root is created inside.

    Returns:
        The path to the created data root directory.
    """
    root = tmp_path.joinpath("data_root")
    root.mkdir()
    return root


@pytest.fixture
def project(data_root: Path) -> ProjectData:
    """Creates the project directory structure through the shared hierarchy's own creator.

    Args:
        data_root: The temporary data root the project is created under.

    Returns:
        The created project view, whose configuration directory exists on disk.
    """
    return ProjectData(root=data_root, project_name=PROJECT_NAME).create()


@pytest.fixture
def project_root(project: ProjectData) -> Path:
    """Resolves the created project's root directory, which every project-scoped tool takes as its path argument.

    Args:
        project: The created project view.

    Returns:
        The absolute path to the project directory.
    """
    return project.path


@pytest.fixture
def hardware_state() -> MesoscopeHardwareState:
    """Builds a Mesoscope-VR hardware state marking every optional module as configured and used.

    Every module the microcontroller pipeline can parse reports as eligible, and the system state codes cover the
    idle, rest, and run states the behavior assembler reads.

    Returns:
        The populated hardware state instance.
    """
    return MesoscopeHardwareState(
        cm_per_pulse=0.0057652,
        maximum_brake_strength=11.30234233,
        minimum_brake_strength=0.42383811,
        lick_threshold=600,
        valve_scale_coefficient=2.0e-07,
        valve_nonlinearity_exponent=1.61941797,
        torque_per_adc_unit=0.00506377,
        screens_initially_on=False,
        recorded_mesoscope_ttl=True,
        delivered_gas_puffs=True,
        system_state_codes={"idle": 0, "rest": 1, "run": 2},
    )


@pytest.fixture
def experiment_configuration() -> MesoscopeExperimentConfiguration:
    """Builds a minimal Mesoscope-VR experiment configuration carrying one trial structure and one state.

    The configuration supplies the schema an experiment session's snapshot has to satisfy. The cue sequence a
    runtime archive records is supplied by the test that needs one.

    Returns:
        The experiment configuration instance.
    """
    return MesoscopeExperimentConfiguration(
        trial_structures={"reward_trial": MesoscopeWaterRewardTrial(reward_size_ul=5.0, reward_tone_duration_ms=300)},
        experiment_states={
            "run_state": ExperimentState(
                experiment_state_code=1, system_state_code=2, state_duration_s=600.0, supports_trials=True
            )
        },
        unity_scene_name="TestScene",
    )


@pytest.fixture
def task_template() -> TaskTemplate:
    """Builds a minimal VR task template whose trial geometry matches the experiment configuration fixture.

    Returns:
        The task template instance, holding two cues and the single named trial structure.
    """
    return TaskTemplate(
        cues=[
            Cue(name="grating", code=1, length_cm=30.0),
            Cue(name="checker", code=2, length_cm=30.0),
        ],
        vr_environment=VREnvironment(
            corridor_spacing_cm=200.0,
            segments_per_corridor=4,
            padding_prefab_name="padding",
            cm_per_unity_unit=10.0,
            cue_offset_cm=0.0,
        ),
        trial_structures={
            "reward_trial": TrialStructure(
                cue_sequence=["grating", "checker"],
                stimulus_trigger_zone_start_cm=30.0,
                stimulus_trigger_zone_end_cm=45.0,
                stimulus_location_cm=40.0,
                show_stimulus_collision_boundary=False,
                trigger_type=TriggerType.COLLISION,
            )
        },
    )


@pytest.fixture
def session_factory(
    project: ProjectData,
    isolated_working_directory: Path,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration,
    task_template: TaskTemplate,
) -> Callable[..., SessionData]:
    """Returns a builder that creates one acquired session through the shared hierarchy's own creator.

    The builder stages the experiment configuration and the VR task template where the creator sources them from,
    creates the session, writes the descriptor its type registers alongside the hardware state snapshot, and returns
    the session reloaded from disk so both its raw and processed paths resolve absolutely.

    Args:
        project: The created project the session is placed under.
        isolated_working_directory: The isolated platform state, requested so the task templates directory the
            builder registers is written under this test's own temporary directory rather than onto the host.
        hardware_state: The hardware state snapshot written into the session's raw data.
        experiment_configuration: The experiment configuration staged for the creator to copy into the session.
        task_template: The VR task template staged for the creator to copy into the session.

    Returns:
        A callable taking the animal identifier, the session type, the experimenter notes, an incomplete flag, and an
        optional experiment name, and returning the loaded session.
    """

    def _create(
        animal_id: str = EXPERIMENT_ANIMAL_ID,
        session_type: SessionTypes = SessionTypes.MESOSCOPE_EXPERIMENT,
        *,
        experimenter_notes: str = "A synthetic session.",
        incomplete: bool = False,
        experiment_name: str | None = None,
    ) -> SessionData:
        # A corridor-task session type is rejected at creation unless it names an experiment, since the shared
        # hierarchy resolves the VR task template from it. The builder therefore names one for those types whatever
        # the caller asked for, while the caller's own argument still decides whether the experiment configuration and
        # task template snapshots are written below. That keeps a test that exercises a session missing those
        # snapshots able to build one.
        declared_experiment = experiment_name
        if declared_experiment is None and SessionTypes(session_type) in SESSION_TYPES_USING_VR_TASK:
            declared_experiment = "unconfigured_experiment"

        # The shared hierarchy now sources both snapshots itself, copying the experiment configuration out of the
        # project's configuration directory and the task template out of the host's templates directory, so both have
        # to be staged before the session is created rather than written into raw data afterwards.
        if declared_experiment is not None:
            project.create()
            experiment_configuration.to_yaml(
                file_path=project.configuration_directory.joinpath(f"{declared_experiment}.yaml")
            )
            templates_directory = project.root.joinpath("task_templates")
            templates_directory.mkdir(parents=True, exist_ok=True)
            set_task_templates_directory(path=templates_directory)
            task_template.to_yaml(
                file_path=templates_directory.joinpath(f"{experiment_configuration.unity_scene_name}.yaml")
            )

        created = SessionData.create(
            animal=project.animal(animal_id),
            session_type=session_type,
            python_version=PYTHON_VERSION,
            sollertia_experiment_version=EXPERIMENT_VERSION,
            acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
            experiment_name=declared_experiment,
        )

        descriptor_class = DESCRIPTOR_REGISTRY[SessionTypes(session_type)]
        # Each session type registers its own descriptor, and a window-checking descriptor carries no animal weight,
        # so the arguments are narrowed to the fields the resolved class declares.
        declared = {declared_field.name for declared_field in fields(descriptor_class)}
        arguments = {
            "experimenter": "tester",
            "animal_weight_g": 25.0,
            "incomplete": incomplete,
            "experimenter_notes": experimenter_notes,
        }
        descriptor_class(**{name: value for name, value in arguments.items() if name in declared}).to_yaml(
            file_path=created.raw_data.session_descriptor_path
        )

        hardware_state.to_yaml(file_path=created.raw_data.hardware_state_path)

        # A caller that named no experiment wants a session carrying no experiment snapshots, which the creator now
        # always writes for a corridor-task type. Removing them here restores that shape, so a test can still build a
        # session whose experiment configuration or VR task template is absent.
        if experiment_name is None:
            created.raw_data.experiment_configuration_path.unlink(missing_ok=True)
            created.raw_data.vr_configuration_path.unlink(missing_ok=True)

        return SessionData.load(session_path=project.animal(animal_id).session_path(created.session_name))

    return _create


@pytest.fixture
def experiment_session(session_factory: Callable[..., SessionData]) -> SessionData:
    """Creates one acquired Mesoscope-VR experiment session carrying its experiment and VR snapshots.

    Args:
        session_factory: The builder that creates and reloads the session.

    Returns:
        The loaded experiment session.
    """
    return session_factory(
        animal_id=EXPERIMENT_ANIMAL_ID,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        experiment_name="test_experiment",
    )


@pytest.fixture
def training_session(session_factory: Callable[..., SessionData]) -> SessionData:
    """Creates one acquired Mesoscope-VR run-training session, which records no imaging and no experiment snapshot.

    Args:
        session_factory: The builder that creates and reloads the session.

    Returns:
        The loaded run-training session.
    """
    return session_factory(animal_id=TRAINING_ANIMAL_ID, session_type=SessionTypes.RUN_TRAINING)


# Processing trackers


@pytest.fixture
def write_tracker() -> Callable[..., ProcessingTracker]:
    """Returns a builder that writes one processing tracker holding jobs in the requested states.

    The builder aligns the tracker against the given universe, then drives each named job into the state the caller
    asked for. A job named in none of the state arguments stays scheduled.

    Returns:
        A callable taking the tracker path, the job universe, the succeeded, failed, and running job keys, and the
        executor identifier recorded against every started job, and returning the written tracker.
    """

    def _write(
        path: Path,
        jobs: Sequence[tuple[str, str]],
        *,
        succeeded: Sequence[tuple[str, str]] = (),
        failed: Mapping[tuple[str, str], str] | None = None,
        running: Sequence[tuple[str, str]] = (),
        executor_id: str | None = None,
    ) -> ProcessingTracker:
        path.parent.mkdir(parents=True, exist_ok=True)
        universe = list(jobs)
        tracker = ProcessingTracker(file_path=path)
        # An empty universe writes a tracker holding no jobs, which is how a caller builds the empty-registry case.
        # The tracker rejects an empty alignment request, so the file is created without one.
        if universe:
            tracker.align_jobs(jobs=universe, universe=universe)
        else:
            tracker.reset()

        for job in succeeded:
            job_id = ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])
            tracker.start_job(job_id=job_id, executor_id=executor_id)
            tracker.complete_job(job_id=job_id)

        for job, error_message in (failed or {}).items():
            job_id = ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])
            tracker.start_job(job_id=job_id, executor_id=executor_id)
            tracker.fail_job(job_id=job_id, error_message=error_message)

        for job in running:
            job_id = ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])
            tracker.start_job(job_id=job_id, executor_id=executor_id)

        return tracker

    return _write


@pytest.fixture
def mark_session_processed(write_tracker: Callable[..., ProcessingTracker]) -> Callable[[SessionData], None]:
    """Returns a helper that records every per-session pipeline of one session as fully succeeded.

    Each pipeline receives a tracker at the location the session resolves for it, holding one succeeded job. This is
    the state forging admission requires before a session may enter a dataset.

    Args:
        write_tracker: The builder that writes each pipeline's tracker.

    Returns:
        A callable taking the loaded session whose pipelines are marked as finished.
    """

    def _mark(session: SessionData) -> None:
        for pipeline in SESSION_PIPELINES:
            jobs = [(f"{pipeline.value}_stage", "")]
            write_tracker(resolve_session_tracker_path(session=session, pipeline=pipeline), jobs, succeeded=jobs)

    return _mark


# Project artifacts


@pytest.fixture
def project_manifest(
    project_root: Path,
    experiment_session: SessionData,
    training_session: SessionData,  # Requested so the generated manifest holds a second session.
    mark_session_processed: Callable[[SessionData], None],
) -> Path:
    """Generates the project's manifest and job artifacts through the project's own writer.

    The experiment session carries a succeeded tracker for every pipeline and the training session carries none, so
    the manifest holds one finished session beside one untouched session.

    Args:
        project_root: The project the manifest is generated for.
        experiment_session: The experiment session marked as fully processed.
        training_session: The training session left unprocessed.
        mark_session_processed: The helper that writes the experiment session's succeeded trackers.

    Returns:
        The path to the written manifest artifact, which sits beside the job artifact.
    """
    mark_session_processed(experiment_session)
    generate_project_manifest(project_directory=project_root)
    return project_manifest_path(project_directory=project_root)


# Synthetic acquisition inputs


@pytest.fixture
def write_log_archive() -> Callable[..., Path]:
    """Returns a writer that builds a real DataLogger log archive the archive reader decodes.

    The archive opens with the onset message the reader anchors every absolute timestamp on, followed by one entry
    per supplied message. Each entry carries the source identifier byte, the elapsed microseconds, and the payload.

    Returns:
        A callable taking the archive path, the source identifier, the message pairs of elapsed microseconds and
        payload bytes, and the onset epoch, and returning the written archive path.
    """

    def _write(
        path: Path,
        source_id: int,
        messages: Sequence[tuple[int, bytes]],
        *,
        onset_us: int = FROZEN_TIMESTAMP_US,
    ) -> Path:
        def _entry(elapsed_us: int, payload: bytes) -> NDArray[np.uint8]:
            header = np.frombuffer(np.uint64(elapsed_us).tobytes(), dtype=np.uint8)
            return np.concatenate(
                [
                    np.array([source_id], dtype=np.uint8),
                    header,
                    np.frombuffer(payload, dtype=np.uint8),
                ]
            )

        entries: dict[str, NDArray[np.uint8]] = {f"{source_id}_{0:020d}": _entry(0, np.uint64(onset_us).tobytes())}
        for index, (elapsed_us, payload) in enumerate(messages, start=1):
            entries[f"{source_id}_{index:020d}"] = _entry(elapsed_us, payload)

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **entries)
        # numpy appends the archive suffix when the given path carries none, so the written file is returned.
        return path if path.is_file() else path.with_suffix(".npz")

    return _write


@pytest.fixture
def write_camera_timestamps() -> Callable[[Path, NDArray[np.uint64]], Path]:
    """Returns a writer that builds one camera timestamp feather in the layout the video assembler reads.

    Returns:
        A callable taking the output path and the per-frame acquisition timestamps, and returning the written path.
    """

    def _write(path: Path, timestamps: NDArray[np.uint64]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"frame_time_us": np.asarray(timestamps, dtype=np.uint64)}).write_ipc(
            file=path, compression="uncompressed"
        )
        return path

    return _write


@pytest.fixture
def write_grayscale_video() -> Callable[..., Path]:
    """Returns a writer that encodes a grayscale frame stack into a recording the motion-energy stage decodes.

    The writer is lossy in the same way the acquisition encoder is, so a test comparing values recomputes its
    reference from the decoded frames rather than from the frames it wrote.

    Returns:
        A callable taking the output path, the frame stack, and the container frame rate, and returning the path.
    """

    def _write(path: Path, frames: NDArray[np.uint8], *, fps: int = 30) -> Path:
        height, width = frames.shape[1:]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), isColor=True)
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()
        return path

    return _write


@pytest.fixture
def moving_block_frames() -> NDArray[np.uint8]:
    """Builds a grayscale frame stack holding one bright block that moves between consecutive frames.

    Returns:
        The frame stack shaped as frames by height by width, whose dimensions exercise the analysis crop path.
    """
    frames = np.zeros((60, MOTION_ENERGY_FRAME_HEIGHT, MOTION_ENERGY_FRAME_WIDTH), dtype=np.uint8)
    for index in range(frames.shape[0]):
        offset = (index * 3) % 40
        frames[index, 20 + offset : 40 + offset, 10:40] = 255
    return frames


@pytest.fixture
def write_dlc_predictions() -> Callable[..., Path]:
    """Returns a writer that builds a DeepLabCut prediction file in the table-format layout the pupil stage reads.

    The columns carry the scorer, bodypart, and coordinate levels DeepLabCut emits, so the file is read back through
    the same reader a real prediction file goes through.

    Returns:
        A callable taking the output path, the mapping of bodypart to its per-frame array of horizontal position,
        vertical position, and likelihood, and the scorer name, and returning the written path.
    """

    def _write(
        path: Path,
        points: Mapping[str, NDArray[np.float64]],
        *,
        scorer: str = "DLC_resnet50_eye_tracking",
    ) -> Path:
        columns = pd.MultiIndex.from_tuples(
            [(scorer, bodypart, coordinate) for bodypart in points for coordinate in ("x", "y", "likelihood")],
            names=["scorer", "bodyparts", "coords"],
        )
        matrix = np.concatenate([np.asarray(points[bodypart], dtype=np.float64) for bodypart in points], axis=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data=matrix, columns=columns).to_hdf(path_or_buf=path, key="df_with_missing", format="table")
        return path

    return _write


# Server configuration and transports


@pytest.fixture
def server_configuration(isolated_working_directory: Path) -> ServerConfiguration:
    """Writes a fully configured server configuration under the isolated working directory.

    Args:
        isolated_working_directory: The working directory the configuration file is created under.

    Returns:
        The configuration the created file holds.
    """
    create_server_configuration_file(
        username="tester",
        password="test_pass",  # noqa: S106 - literal test credential.
        host="test.server.com",
        root=SERVER_ROOT,
        environment=SERVER_ENVIRONMENT,
    )
    return ServerConfiguration.from_yaml(
        file_path=isolated_working_directory.joinpath("configuration", "server_configuration.yaml")
    )


class StubSSHTransport:
    """Stands in for the paramiko stack a Server connects over, backed by a temporary server-side filesystem.

    Every file-transfer operation runs against a real directory tree, so pushing, pulling, listing, and existence
    checks answer truthfully. Shell invocations are recorded, and the scheduler primitives the server issues are
    answered from the state this class holds.

    Args:
        remote_root: The local directory that stands in for the whole server-side filesystem.

    Attributes:
        remote_root: The directory every absolute server path is resolved under.
        commands: Every shell invocation the server issued, in order.
        uploads: The local and server paths of every file the server uploaded.
        downloads: The local and server paths of every file the server downloaded.
        submitted_scripts: The server path of every script the scheduler accepted.
        job_statuses: The accounting state reported for each allocation identifier.
        blocked_job_ids: The allocation identifiers the queue reports as permanently blocked.
        connections: The host and user pairs the transport was asked to authenticate.
        closed: Whether the connection was closed.
    """

    def __init__(self, remote_root: Path) -> None:
        self.remote_root: Path = remote_root
        self.commands: list[str] = []
        self.uploads: list[tuple[Path, Path]] = []
        self.downloads: list[tuple[Path, Path]] = []
        self.submitted_scripts: list[str] = []
        self.job_statuses: dict[str, str] = {}
        self.blocked_job_ids: set[str] = set()
        self.connections: list[tuple[str, str]] = []
        self.closed: bool = False
        self._responses: dict[str, tuple[str, str, int]] = {}
        self._next_job_id: int = _SLURM_FIRST_JOB_ID

    def respond(self, prefix: str, *, stdout: str = "", stderr: str = "", return_code: int = 0) -> None:
        """Registers the result answered for every invocation starting with the given text.

        Args:
            prefix: The leading text of the invocations this response answers.
            stdout: The standard output the invocation reports.
            stderr: The standard error the invocation reports.
            return_code: The exit code the invocation reports.
        """
        self._responses[prefix] = (stdout, stderr, return_code)

    def local_path(self, remote_path: str | Path) -> Path:
        """Resolves one absolute server path to its location inside the temporary server-side filesystem.

        Args:
            remote_path: The absolute path as the server sees it.

        Returns:
            The corresponding local path under the transport's remote root.
        """
        path = Path(remote_path)
        relative = path.relative_to(path.anchor) if path.is_absolute() else path
        return self.remote_root.joinpath(relative)

    def execute(self, command: str) -> tuple[str, str, int]:
        """Runs one shell invocation against the transport's recorded state.

        Args:
            command: The invocation the server issued.

        Returns:
            A tuple of the standard output, the standard error, and the exit code.
        """
        self.commands.append(command)

        for prefix, response in self._responses.items():
            if command.startswith(prefix):
                return response

        if command.startswith("mkdir -p "):
            self.local_path(shlex.split(command)[2]).mkdir(parents=True, exist_ok=True)
            return "", "", 0

        if command.startswith("chmod "):
            return "", "", 0

        if command.startswith("sbatch "):
            self.submitted_scripts.append(shlex.split(command)[1])
            job_id = str(self._next_job_id)
            self._next_job_id += 1
            return f"Submitted batch job {job_id}\n", "", 0

        if command.startswith("sacct "):
            requested = command.split("-j ", maxsplit=1)[1].split(" ", maxsplit=1)[0].split(",")
            rows = [f"{job_id}|{self.job_statuses[job_id]}" for job_id in requested if job_id in self.job_statuses]
            return "\n".join(rows) + ("\n" if rows else ""), "", 0

        if command.startswith("squeue "):
            rows = [f"{job_id}|DependencyNeverSatisfied" for job_id in sorted(self.blocked_job_ids)]
            return "\n".join(rows) + ("\n" if rows else ""), "", 0

        return "", "", 0


@dataclass
class _StubChannel:
    """Reports the exit status of one completed invocation the way a paramiko channel does."""

    return_code: int

    def recv_exit_status(self) -> int:
        """Returns the exit code the invocation reported."""
        return self.return_code


@dataclass
class _StubChannelFile:
    """Delivers one stream of a completed invocation the way a paramiko channel file does."""

    payload: str
    channel: _StubChannel

    def read(self) -> bytes:
        """Returns the stream's contents as the bytes a caller decodes."""
        return self.payload.encode()


@dataclass
class _StubAttributes:
    """Describes one server-side directory entry the way a paramiko attribute record does."""

    filename: str
    st_mode: int


class _StubSFTPClient:
    """Serves the file-transfer half of the stubbed connection out of the transport's temporary filesystem.

    Args:
        transport: The transport whose server-side filesystem every operation runs against.
    """

    def __init__(self, transport: StubSSHTransport) -> None:
        self._transport: StubSSHTransport = transport

    def put(self, localpath: str, remotepath: str) -> None:
        """Uploads one file into the server-side filesystem."""
        destination = self._transport.local_path(remotepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=localpath, dst=destination)
        self._transport.uploads.append((Path(localpath), Path(remotepath)))

    def get(self, localpath: str, remotepath: str) -> None:
        """Downloads one file out of the server-side filesystem."""
        source = self._transport.local_path(remotepath)
        Path(localpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=source, dst=localpath)
        self._transport.downloads.append((Path(localpath), Path(remotepath)))

    def stat(self, path: str) -> SimpleNamespace:
        """Returns the mode of one server-side path, reporting an absent path as missing."""
        resolved = self._transport.local_path(path)
        if not resolved.exists():
            message = f"The remote path {path} does not exist."
            raise FileNotFoundError(message)
        return SimpleNamespace(st_mode=S_IFDIR | 0o755 if resolved.is_dir() else S_IFREG | 0o644)

    def mkdir(self, path: str) -> None:
        """Creates one server-side directory."""
        self._transport.local_path(path).mkdir(parents=True, exist_ok=True)

    def rmdir(self, path: str) -> None:
        """Removes one empty server-side directory."""
        self._transport.local_path(path).rmdir()

    def unlink(self, path: str) -> None:
        """Removes one server-side file."""
        self._transport.local_path(path).unlink()

    def listdir(self, path: str) -> list[str]:
        """Returns the entry names of one server-side directory."""
        return sorted(entry.name for entry in self._transport.local_path(path).iterdir())

    def listdir_attr(self, path: str) -> list[_StubAttributes]:
        """Returns the entry names and modes of one server-side directory."""
        return [
            _StubAttributes(filename=entry.name, st_mode=S_IFDIR | 0o755 if entry.is_dir() else S_IFREG | 0o644)
            for entry in sorted(self._transport.local_path(path).iterdir())
        ]

    def open(self, path: str, mode: str = "r") -> Any:
        """Opens one server-side file, creating its parent directories for a write."""
        resolved = self._transport.local_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved.open(mode)

    def close(self) -> None:
        """Closes the file-transfer half of the connection."""
        self._transport.closed = True


class _StubSSHClient:
    """Serves the shell half of the stubbed connection out of the transport's recorded state.

    Args:
        transport: The transport every invocation is recorded against.
    """

    def __init__(self, transport: StubSSHTransport) -> None:
        self._transport: StubSSHTransport = transport

    def set_missing_host_key_policy(self, policy: object) -> None:
        """Accepts the host key policy the server applies before it connects."""

    def connect(self, hostname: str, username: str, password: str) -> None:  # noqa: ARG002
        """Records one authentication attempt against the stubbed host."""
        self._transport.connections.append((hostname, username))

    def open_sftp(self) -> _StubSFTPClient:
        """Returns the file-transfer half of the connection."""
        return _StubSFTPClient(transport=self._transport)

    def exec_command(self, command: str) -> tuple[None, _StubChannelFile, _StubChannelFile]:
        """Runs one invocation and returns its input, output, and error streams."""
        stdout, stderr, return_code = self._transport.execute(command=command)
        channel = _StubChannel(return_code=return_code)
        output = _StubChannelFile(payload=stdout, channel=channel)
        error = _StubChannelFile(payload=stderr, channel=channel)
        return None, output, error

    def close(self) -> None:
        """Closes the shell half of the connection."""
        self._transport.closed = True


@pytest.fixture
def stub_ssh_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StubSSHTransport:
    """Replaces the paramiko stack the server module reaches for with a temporary-filesystem transport.

    Requesting this fixture lets a Server be constructed and driven without a network, while every file-transfer
    operation runs against a real directory tree under the test's temporary directory.

    Args:
        tmp_path: The temporary directory the server-side filesystem is created under.
        monkeypatch: The fixture used to replace the paramiko binding the server module holds.

    Returns:
        The transport, which records the invocations and answers the scheduler primitives.
    """
    remote_root = tmp_path.joinpath("server")
    remote_root.mkdir()
    transport = StubSSHTransport(remote_root=remote_root)

    monkeypatch.setattr(
        server_module,
        "paramiko",
        SimpleNamespace(
            SSHClient=lambda: _StubSSHClient(transport=transport),
            AutoAddPolicy=object,
            AuthenticationException=type("AuthenticationException", (Exception,), {}),
        ),
    )
    return transport


@pytest.fixture
def connected_server(
    stub_ssh_transport: StubSSHTransport,
    server_configuration: ServerConfiguration,
) -> Iterator[Server]:
    """Opens a real Server over the stubbed transport, so server-side behavior runs without a network.

    Args:
        stub_ssh_transport: The transport the connection is established over.
        server_configuration: The configuration naming the stubbed host and the server-side data root.

    Yields:
        The connected server, which is closed once the test finishes with it.
    """
    server = Server(configuration=server_configuration)
    yield server
    server.close()


@pytest.fixture
def stub_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces the process runner with a recorder that answers a configured result.

    Args:
        monkeypatch: The fixture used to replace the runner.

    Returns:
        A recorder carrying the ``calls`` list of invoked argument sequences and the ``result`` completed process
        every call answers, which a test may reassign before the code under test runs.
    """
    recorder = SimpleNamespace(
        calls=[],
        result=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )

    def _run(arguments: Sequence[str], **keywords: Any) -> subprocess.CompletedProcess[str]:
        recorder.calls.append((list(arguments), keywords))
        return recorder.result

    monkeypatch.setattr(subprocess, "run", _run)
    return recorder


# Host introspection and wall clock


@pytest.fixture
def stub_host_memory(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Returns a setter that pins the physical memory the host reports, so a budget is independent of this machine.

    Args:
        monkeypatch: The fixture used to replace the memory probe.

    Returns:
        A callable taking the total memory in megabytes the host reports.
    """

    def _pin(total_mb: int) -> None:
        monkeypatch.setattr(
            footprints_module.psutil,
            "virtual_memory",
            lambda: SimpleNamespace(total=total_mb * 1024 * 1024),
        )

    return _pin


@dataclass
class FrozenClock:
    """Reports one fixed microsecond epoch to every caller that stamps a record.

    Attributes:
        timestamp: The epoch reported until a test moves it.
    """

    timestamp: int = FROZEN_TIMESTAMP_US

    def advance(self, microseconds: int) -> int:
        """Moves the clock forward and returns the epoch it now reports.

        Args:
            microseconds: The number of microseconds to move the clock forward by.

        Returns:
            The epoch the clock reports after the move.
        """
        self.timestamp += microseconds
        return self.timestamp


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> FrozenClock:
    """Pins the wall clock every ledger, submission, and closure record is stamped with.

    Args:
        monkeypatch: The fixture used to replace the timestamp source each module holds.

    Returns:
        The clock, whose timestamp a test moves to order the records it writes.
    """
    clock = FrozenClock()
    for module in (ledger_module, remote_module, closure_module):
        monkeypatch.setattr(module, "current_timestamp", lambda: clock.timestamp)
    return clock


@dataclass
class BatchIdentifiers:
    """Issues the sequential identifiers every prepared batch is recorded under.

    Attributes:
        issued: The identifiers handed out so far, in the order they were issued.
    """

    issued: list[str] = field(default_factory=list)

    def next_identifier(self) -> str:
        """Issues the next identifier and records it.

        Returns:
            The issued identifier.
        """
        identifier = f"batch{len(self.issued):02d}"
        self.issued.append(identifier)
        return identifier


@pytest.fixture
def deterministic_batch_ids(monkeypatch: pytest.MonkeyPatch) -> BatchIdentifiers:
    """Replaces the random batch identifier with a sequential one, so a recorded batch is nameable in advance.

    Args:
        monkeypatch: The fixture used to replace the identifier source the batch registry holds.

    Returns:
        The issuer, whose ``issued`` list names every batch recorded during the test in order.
    """
    identifiers = BatchIdentifiers()
    monkeypatch.setattr(
        batches_module, "uuid4", lambda: SimpleNamespace(hex=identifiers.next_identifier().ljust(16, "0"))
    )
    return identifiers
