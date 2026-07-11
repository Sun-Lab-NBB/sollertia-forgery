"""Provides the video-tracking configuration that lets the video pipeline drive DeepLabCut inference on a processing
host, locating the isolated ``slvt`` runtime and the trained DeepLabCut project that produces a session's pose
predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import get_working_directory
from ataraxis_data_structures import YamlConfig

if TYPE_CHECKING:
    from pathlib import Path

_VIDEO_TRACKING_CONFIG_FILENAME: str = "video_tracking_configuration.yaml"
"""Canonical filename for the VideoTrackingConfiguration YAML stored under the working directory's configuration
subdirectory."""

_CONFIGURATION_DIR: str = "configuration"
"""Subdirectory under the working directory that stores the video-tracking configuration YAML alongside the other
Sollertia configuration assets."""


@dataclass
class VideoTrackingConfiguration(YamlConfig):
    """Defines how a processing host drives DeepLabCut inference for the video-tracking pipeline.

    Notes:
        DeepLabCut runs in its own isolated conda environment, since it pins ``numpy<2`` and cannot be imported into
        the ``numpy>=2`` sollertia-forgery process. The pipeline therefore invokes it through the ``slvt`` command-line
        interface in that environment. This configuration records the environment name and the trained project the
        pipeline runs, along with the crop and model-selection parameters that decouple a de-novo inference run from the
        project's own configuration. When this file is absent, the pipeline drives no inference and the tracking donor
        reads predictions produced out of band, so configuring it is what turns pipeline-driven inference on.
    """

    environment: str = ""
    """The name of the conda environment, on the processing host, in which the isolated ``slvt`` (sollertia-video-
    tracking) command-line interface is installed. The pipeline invokes ``conda run -n <environment> slvt`` to run
    inference in the DeepLabCut environment."""
    project_configuration_path: str = ""
    """The absolute path, on the processing host, to the trained DeepLabCut project's ``config.yaml`` that the pipeline
    runs over each session's raw video."""
    crop: str = ""
    """The ``x1,x2,y1,y2`` crop rectangle applied to the raw video during inference, decoupling the analyzed region
    from the project's own configuration so a de-novo video that is not registered in the project can be analyzed. An
    empty value uses the project's configured crop."""
    shuffle: int = 1
    """The shuffle index whose trained model the pipeline runs."""
    snapshot_index: int = -1
    """The pose snapshot index the pipeline runs. A negative value uses the project's configured snapshot."""


def create_video_tracking_configuration_file(
    environment: str,
    project_configuration_path: str,
    crop: str = "",
    shuffle: int = 1,
    snapshot_index: int = -1,
) -> None:
    """Creates the video-tracking configuration file and configures the local machine to use it for pipeline-driven
    DeepLabCut inference.

    Args:
        environment: The name of the conda environment, on the processing host, in which the isolated ``slvt`` command-
            line interface is installed.
        project_configuration_path: The absolute path to the trained DeepLabCut project's ``config.yaml``.
        crop: The ``x1,x2,y1,y2`` crop rectangle applied during inference, or empty to use the project's configured
            crop.
        shuffle: The shuffle index whose trained model the pipeline runs.
        snapshot_index: The pose snapshot index to run, or a negative value to use the project's configured snapshot.
    """
    output_directory = get_working_directory().joinpath(_CONFIGURATION_DIR)
    VideoTrackingConfiguration(
        environment=environment,
        project_configuration_path=project_configuration_path,
        crop=crop,
        shuffle=shuffle,
        snapshot_index=snapshot_index,
    ).to_yaml(file_path=output_directory.joinpath(_VIDEO_TRACKING_CONFIG_FILENAME))
    console.echo(message="Video-tracking configuration file: Created.", level=LogLevel.SUCCESS)


def get_video_tracking_configuration() -> VideoTrackingConfiguration | None:
    """Resolves the processing host's video-tracking configuration, or None when the host does not drive inference.

    Returns:
        The loaded configuration when the file exists and names both an environment and a project configuration path,
        or None when the configuration file is absent, in which case the host leaves prediction production to an
        out-of-band run.

    Raises:
        ValueError: If the configuration file exists but is unconfigured, missing the environment or the project
            configuration path.
    """
    config_path = get_working_directory().joinpath(_CONFIGURATION_DIR, _VIDEO_TRACKING_CONFIG_FILENAME)
    if not config_path.exists():
        return None

    configuration = VideoTrackingConfiguration.from_yaml(file_path=config_path)
    if not configuration.environment or not configuration.project_configuration_path:
        message = (
            "Unable to load the video-tracking configuration. The 'video_tracking_configuration.yaml' file exists but "
            "is missing the required 'environment' or 'project_configuration_path' field. Reconfigure it to drive "
            "DeepLabCut inference, or remove it to read predictions produced out of band."
        )
        console.error(message=message, error=ValueError)

    return configuration


def get_video_tracking_configuration_path() -> Path:
    """Returns the path under which the ``video_tracking_configuration.yaml`` file is stored.

    Used by tools that write the configuration file directly and need its destination path without loading the
    configuration contents.
    """
    return get_working_directory().joinpath(_CONFIGURATION_DIR, _VIDEO_TRACKING_CONFIG_FILENAME)
