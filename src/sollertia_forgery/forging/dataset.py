"""Provides operations for resolving and creating the forged dataset hierarchy from acquisition sessions."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    SessionData,
    RawDataFiles,
    SessionTypes,
    discover_sessions,
)
from ataraxis_data_structures import delete_directory

from ..shared_assets import DatasetData, DatasetSession

if TYPE_CHECKING:
    from pathlib import Path


def resolve_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
    force_recreate: bool = False,
) -> DatasetData:
    """Creates, loads, or recreates a dataset hierarchy based on the provided session set.

    Acts as the dataset-definition entry point for both the forging pipeline and the MCP ``prepare`` tool so that
    tracker setup and batch dispatch can layer above this helper rather than duplicating the resolution logic. When
    the dataset already exists, loads it and, when a session list is provided, verifies that the provided set matches
    the existing definition. A mismatch surfaces as an error unless ``force_recreate`` is True, which unlocks
    deletion and fresh recreation from the provided session names. When the dataset does not exist, creates it from
    the provided session names; a non-empty session list is required in that case.

    Args:
        name: The unique name of the dataset.
        session_names: The session names to include in the dataset. Pass an empty tuple to work with an
            already-defined dataset without triggering the session-set verification step.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        required_session_type: The session type the dataset's sessions must have, supplied by the calling
            acquisition system to restrict dataset creation to its forgeable session type. When None, any session
            type is accepted (the cross-session consistency check still applies).
        force_recreate: Determines whether to allow deletion of the existing dataset hierarchy when the provided
            session list does not match the existing definition.

    Returns:
        The resolved DatasetData instance (either loaded from disk or freshly created).

    Raises:
        ValueError: If required_session_type is provided and the first session's type differs from it, if the
            dataset does not exist and no sessions were provided to create it, or if the provided session list does
            not match the existing dataset and force_recreate is False.
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
    """
    dataset_directory = project_root.joinpath(name)
    if dataset_directory.exists():
        dataset = DatasetData.load(dataset_path=dataset_directory)

        if session_names:
            provided_sessions = set(session_names)
            existing_sessions = {entry.session for entry in dataset.sessions}
            if provided_sessions != existing_sessions:
                if not force_recreate:
                    message = (
                        f"Unable to use the existing '{name}' dataset. The provided session list does not match "
                        f"the dataset's existing session set. Call with force_recreate=True to delete and "
                        f"recreate the dataset, or provide a matching session list."
                    )
                    console.error(message=message, error=ValueError)
                delete_directory(directory_path=dataset_directory)
                console.echo(
                    message=f"Dataset '{name}': Removed existing hierarchy for recreation.",
                    level=LogLevel.INFO,
                )
                dataset = _create_dataset(
                    name=name,
                    sessions=session_names,
                    project_root=project_root,
                    required_session_type=required_session_type,
                )
        return dataset

    if not session_names:
        message = (
            f"Unable to define dataset '{name}'. The dataset does not exist under '{project_root}' and no "
            f"sessions were provided to create it."
        )
        console.error(message=message, error=ValueError)
    return _create_dataset(
        name=name,
        sessions=session_names,
        project_root=project_root,
        required_session_type=required_session_type,
    )


def _create_dataset(
    name: str,
    sessions: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
) -> DatasetData:
    """Creates a fresh dataset hierarchy by resolving the provided session names under the project root.

    Each session name is resolved by probing the discovered ``<project_root>/<animal>/<session_name>`` layout and
    requiring the session's ``session_data.yaml`` marker to be present. Every included session must share the first
    session's session type and acquisition system, and when ``required_session_type`` is provided, the first
    session's type must also match it.

    Args:
        name: The unique name for the dataset.
        sessions: The non-empty tuple of session names to include in the dataset.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        required_session_type: The session type the first session must have. When None, any session type is
            accepted as the dataset's reference type.

    Returns:
        The newly created DatasetData instance.

    Raises:
        FileNotFoundError: If a session name does not resolve to any animal directory under the project root.
        RuntimeError: If a session name resolves to more than one animal directory under the project root.
        ValueError: If required_session_type is provided and the first session's type differs from it, or if any
            subsequent session's session type or acquisition system differs from the first session's.
    """
    # Builds a session-name -> session-root index via shared-assets discovery so the project layout is not
    # assumed here. A session name colliding across animals surfaces as a RuntimeError during lookup rather
    # than silently selecting the first match.
    discovered: dict[str, list[Path]] = {}
    for session_root in discover_sessions(root_path=project_root):
        discovered.setdefault(session_root.name, []).append(session_root)

    session_paths: list[Path] = []
    for session_name in sessions:
        matches = discovered.get(session_name, [])
        if len(matches) != 1:
            message = (
                f"Unable to resolve the directory for session '{session_name}' under '{project_root}'. "
                f"Expected exactly one session named '{session_name}' to be discoverable via "
                f"'{RawDataFiles.SESSION_DATA}' markers, but found {len(matches)}."
            )
            console.error(message=message, error=FileNotFoundError if not matches else RuntimeError)
        session_paths.append(matches[0])

    first_session_data = SessionData.load(session_path=session_paths[0])
    if required_session_type is not None and first_session_data.session_type != required_session_type:
        message = (
            f"Unable to define dataset '{name}'. Dataset creation for this acquisition system is supported only "
            f"for '{required_session_type}' sessions, but the first session's type resolved to "
            f"'{first_session_data.session_type}'."
        )
        console.error(message=message, error=ValueError)

    # Verifies that every remaining session shares the first session's type and acquisition system. A dataset
    # must contain only sessions of the same type acquired by the same acquisition system; the assembly logic
    # downstream assumes this invariant when deriving the dataset-level metadata from the first session.
    for session_path in session_paths[1:]:
        session_data = SessionData.load(session_path=session_path)
        if session_data.session_type != first_session_data.session_type:
            message = (
                f"Unable to define dataset '{name}'. All sessions in a dataset must share the same session "
                f"type, but session '{session_path.name}' has type '{session_data.session_type}' while the "
                f"first session has type '{first_session_data.session_type}'."
            )
            console.error(message=message, error=ValueError)
        if session_data.acquisition_system != first_session_data.acquisition_system:
            message = (
                f"Unable to define dataset '{name}'. All sessions in a dataset must be acquired by the same "
                f"acquisition system, but session '{session_path.name}' was acquired by "
                f"'{session_data.acquisition_system}' while the first session was acquired by "
                f"'{first_session_data.acquisition_system}'."
            )
            console.error(message=message, error=ValueError)

    dataset_sessions = tuple(DatasetSession(animal=path.parent.name, session=path.name) for path in session_paths)
    dataset = DatasetData.create(
        name=name,
        project=project_root.name,
        session_type=first_session_data.session_type,
        acquisition_system=first_session_data.acquisition_system,
        sessions=dataset_sessions,
        datasets_root=project_root,
    )

    _copy_animal_surgery_files(dataset_name=name, dataset=dataset, source_session_paths=session_paths)

    console.echo(
        message=(
            f"Dataset '{name}' data hierarchy: Defined with {len(sessions)} sessions from "
            f"{len(dataset.animals)} animals."
        ),
        level=LogLevel.SUCCESS,
    )
    return dataset


def _copy_animal_surgery_files(
    dataset_name: str,
    dataset: DatasetData,
    source_session_paths: list[Path],
) -> None:
    """Copies the surgery metadata YAML for each animal into the dataset's animal directory.

    For each animal in the dataset, selects that animal's most recent source session and copies its
    ``surgery_metadata.yaml`` from the session's raw data directory to the dataset's animal directory root.
    Surgery metadata is per-animal rather than per-session, so a single copy is materialized for each animal.

    Args:
        dataset_name: The name of the dataset, used for error messages.
        dataset: The freshly created DatasetData instance, used to locate per-animal directories.
        source_session_paths: The resolved source session directory paths used to define the dataset, in the
            order provided to ``_create_dataset``. Grouped by animal to pick each animal's latest session.

    Raises:
        FileNotFoundError: If the latest session for any animal does not contain a ``surgery_metadata.yaml`` file.
    """
    # Groups source session paths by owning animal. The animal name is the parent directory name in the source
    # project layout.
    sessions_by_animal: dict[str, list[Path]] = {}
    for source_path in source_session_paths:
        sessions_by_animal.setdefault(source_path.parent.name, []).append(source_path)

    # The dataset hierarchy stores each animal at ``<dataset_root>/<animal>/``. DatasetAnimal.surgery_path
    # resolves the per-animal destination relative to that directory.
    for dataset_animal in dataset.animals:
        animal_sessions = sessions_by_animal[dataset_animal.animal]

        # Picks the most recent session for the animal via natural sort over the timestamped session names.
        latest_session_name = natsorted([path.name for path in animal_sessions])[-1]
        latest_session_path = next(path for path in animal_sessions if path.name == latest_session_name)
        session_data = SessionData.load(session_path=latest_session_path)

        source_surgery_path = session_data.raw_data.surgery_metadata_path
        if not source_surgery_path.is_file():
            message = (
                f"Unable to define dataset '{dataset_name}'. The latest session '{latest_session_name}' for "
                f"animal '{dataset_animal.animal}' does not contain a '{RawDataFiles.SURGERY_METADATA}' file at "
                f"'{source_surgery_path}'. Surgery metadata is required for every animal in a forged dataset."
            )
            console.error(message=message, error=FileNotFoundError)

        shutil.copy2(src=source_surgery_path, dst=dataset_animal.surgery_path)
