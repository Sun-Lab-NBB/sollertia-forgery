"""Provides the driver that runs DeepLabCut inference over a session's raw camera video through the isolated ``slvt``
command-line interface. The video pipeline uses it to produce the pose predictions its tracking donor consumes, rather
than relying on predictions produced out of band.

Notes:
    DeepLabCut pins ``numpy<2`` and cannot be imported into the ``numpy>=2`` sollertia-forgery process, so inference is
    driven out of process. The driver invokes ``slvt infer`` in a separate conda environment (named by the host's
    video-tracking configuration) and communicates with it entirely across the command-line boundary. The driver only
    launches the run and locates its outputs; it never imports DeepLabCut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import subprocess

from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

    from ..registries import VideoInferenceDescriptor
    from .configuration import VideoTrackingConfiguration

_VIDEO_SUFFIX: str = ".mp4"
"""The file extension of the raw camera videos ataraxis-video-system writes, each named ``{source_id:03d}.mp4``."""

_DLC_ARTIFACT_SUFFIXES: tuple[str, str] = (".h5", ".pickle")
"""The file extensions of the DeepLabCut prediction artifacts ``slvt infer`` writes for one video (the native ``.h5``
table and its companion pickles), all sharing the analyzed video's stem as their filename prefix."""

_STDERR_TAIL_CHARACTERS: int = 2000
"""The number of trailing characters of a failed ``slvt`` run's captured standard error to surface in the error
message, bounding the reported output while keeping the most recent, and usually most relevant, lines."""


def run_slvt_inference(
    session: SessionData,
    descriptor: VideoInferenceDescriptor,
    configuration: VideoTrackingConfiguration,
    output_directory: Path,
) -> str:
    """Runs DeepLabCut inference over a session's raw camera video through the isolated ``slvt`` command-line interface.

    Resolves the raw video for the descriptor's camera from the session's camera manifest. It then invokes ``slvt
    infer`` in the configured conda environment to write DeepLabCut's native ``.h5`` prediction file into the output
    directory, where the acquisition system's tracking donor reads it. The crop and model-selection parameters come
    from the host's video-tracking configuration, so a de-novo video that the project does not register can still be
    analyzed.

    Args:
        session: The loaded session whose raw camera video is analyzed.
        descriptor: The acquisition system's video-inference selectors, naming the camera to analyze and the DeepLabCut
            project that identifies the model.
        configuration: The processing host's video-tracking configuration, providing the ``slvt`` environment, the
            project configuration path, and the crop and model-selection parameters.
        output_directory: The processed video-data directory the prediction file is written into.

    Returns:
        The stem of the analyzed video (``{source_id:03d}``), used to locate and remove the prediction artifacts after
        the tracking donor has consumed them.

    Raises:
        FileNotFoundError: If the camera manifest is missing, or if the descriptor's camera has no raw video on disk.
        ValueError: If the descriptor's camera is not registered in the session's camera manifest.
        RuntimeError: If the ``slvt`` run exits with an error, or exits successfully without writing a prediction file.
    """
    camera_data_directory = session.raw_data.camera_data_path
    source_id = _resolve_camera_source_id(
        camera_data_directory=camera_data_directory, camera_name=descriptor.camera_name
    )
    raw_video = camera_data_directory.joinpath(f"{source_id:03d}{_VIDEO_SUFFIX}")
    if not raw_video.is_file():
        message = (
            f"Unable to run DeepLabCut inference for session '{session.session_name}'. No raw video was found for the "
            f"'{descriptor.camera_name}' camera (source ID {source_id}) at '{raw_video}'."
        )
        console.error(message=message, error=FileNotFoundError)

    console.echo(
        message=(
            f"Running DeepLabCut inference on '{raw_video.name}' for session '{session.session_name}' through the "
            f"'{configuration.environment}' environment..."
        ),
        level=LogLevel.INFO,
    )

    command = _build_inference_command(
        configuration=configuration, raw_video=raw_video, output_directory=output_directory
    )
    result = subprocess.run(args=command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (
            f"Unable to run DeepLabCut inference for session '{session.session_name}'. The 'slvt infer' command exited "
            f"with code {result.returncode}. Captured standard error: {result.stderr[-_STDERR_TAIL_CHARACTERS:]}"
        )
        console.error(message=message, error=RuntimeError)

    # Confirms the expected model's prediction file was produced. The scorer DeepLabCut bakes into the filename carries
    # the project (task) name, which is also the name the tracking donor globs for, so this both validates the run and
    # matches the donor's discovery contract.
    video_stem = raw_video.stem
    if not list(output_directory.glob(f"{video_stem}*{descriptor.dlc_project_name}*.h5")):
        message = (
            f"Unable to run DeepLabCut inference for session '{session.session_name}'. The 'slvt infer' command "
            f"reported success but wrote no '{descriptor.dlc_project_name}' prediction file for '{raw_video.name}' "
            f"into '{output_directory}'."
        )
        console.error(message=message, error=RuntimeError)

    console.echo(message=f"DeepLabCut inference for '{raw_video.name}' completed successfully.", level=LogLevel.SUCCESS)
    return video_stem


def cleanup_slvt_artifacts(output_directory: Path, video_stem: str) -> None:
    """Removes the DeepLabCut prediction artifacts ``slvt infer`` wrote for one analyzed video.

    DeepLabCut names every prediction file for a video with the video's stem as the filename prefix. This removes those
    transient artifacts (the native ``.h5`` table and its companion pickles) once the tracking donor has consumed them,
    leaving only the donor's own tracking outputs in the processed video-data directory.

    Args:
        output_directory: The processed video-data directory the prediction artifacts were written into.
        video_stem: The stem of the analyzed video (``{source_id:03d}``) that prefixes every artifact filename.
    """
    for suffix in _DLC_ARTIFACT_SUFFIXES:
        for artifact in output_directory.glob(f"{video_stem}*{suffix}"):
            artifact.unlink(missing_ok=True)


def _build_inference_command(
    configuration: VideoTrackingConfiguration, raw_video: Path, output_directory: Path
) -> list[str]:
    """Builds the ``conda run`` argument vector that runs ``slvt infer`` over one video in the isolated environment.

    Args:
        configuration: The processing host's video-tracking configuration, providing the environment, the project
            configuration path, and the crop and model-selection parameters.
        raw_video: The raw video to analyze.
        output_directory: The processed video-data directory the prediction file is written into.

    Returns:
        The command argument vector, running ``slvt infer`` in the configured conda environment with the destination,
        shuffle, and progress flags always set, and the snapshot-index and crop flags appended only when configured.
    """
    command = [
        "conda",
        "run",
        "-n",
        configuration.environment,
        "slvt",
        "infer",
        configuration.project_configuration_path,
        str(raw_video),
        "--destination",
        str(output_directory),
        "--shuffle",
        str(configuration.shuffle),
        "--no-progress",
    ]
    if configuration.snapshot_index >= 0:
        command.extend(("--snapshot-index", str(configuration.snapshot_index)))
    if configuration.crop:
        command.extend(("--crop", configuration.crop))
    return command


def _resolve_camera_source_id(camera_data_directory: Path, camera_name: str) -> int:
    """Resolves the numeric source ID of a named camera from the session's acquisition-time camera manifest.

    Args:
        camera_data_directory: The session's raw camera data directory (``session.raw_data.camera_data_path``), holding
            the camera manifest that maps each source ID to its colloquial name.
        camera_name: The colloquial camera name to resolve (for example, ``"face_camera"``).

    Returns:
        The numeric source ID registered for the named camera, which names its raw video ``{source_id:03d}.mp4``.

    Raises:
        FileNotFoundError: If the camera manifest file does not exist in the data directory.
        ValueError: If no camera with the requested name is registered in the manifest.
    """
    manifest_path = camera_data_directory.joinpath(CAMERA_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        message = (
            f"Unable to resolve the '{camera_name}' camera source ID. No camera manifest "
            f"('{CAMERA_MANIFEST_FILENAME}') was found in the raw camera data directory '{camera_data_directory}'."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = CameraManifest.from_yaml(file_path=manifest_path)
    for source in manifest.sources:
        if source.name == camera_name:
            return source.id

    registered = ", ".join(sorted(source.name for source in manifest.sources))
    message = (
        f"Unable to resolve the '{camera_name}' camera source ID. The camera manifest in '{camera_data_directory}' "
        f"registers no camera with that name. Registered cameras: {registered}."
    )
    console.error(message=message, error=ValueError)
    raise ValueError(message)  # pragma: no cover - console.error is NoReturn; satisfies RET503/return typing.
