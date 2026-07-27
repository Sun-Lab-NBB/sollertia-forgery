"""Provides the forging-specific dataset resolution policy layered above the shared dataset hierarchy."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    RawDataFiles,
    DatasetSession,
    discover_sessions,
)
from ataraxis_data_structures import delete_directory

from ..registries import resolve_forging_column_descriptions

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionTypes


def resolve_dataset(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
    force_recreate: bool = False,
    recreate_animals: tuple[str, ...] = (),
) -> DatasetData:
    """Creates, loads, extends, or rebuilds a dataset hierarchy based on the provided session set.

    The provided session list names the sessions the dataset must contain. A session the dataset already holds is
    left alone and a session it does not hold is appended.

    Notes:
        An animal already in the dataset is frozen. Providing a session it does not hold is rejected, because
        widening an animal's session set often requires rebuilding the entire animal's dataset. Adding an animal
        the dataset does not hold stays safe.

        Naming an animal in recreate_animals opts that animal out of the freeze. The animal is dropped from the
        dataset with its directory tree and rebuilt from the sessions provided for it, while every other animal
        keeps its data. The rebuilt animal's tracked jobs need a reset, which the calling pipeline owns.

    Args:
        name: The unique name of the dataset.
        session_names: The session names the dataset must contain. Pass an empty tuple to work with an
            already-defined dataset without changing its session set.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        required_session_type: The session type the dataset's sessions must have, supplied by the calling
            acquisition system to restrict dataset creation to its forgeable session type. When None, any session
            type is accepted (the cross-session consistency check still applies).
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions the
            provided list holds for them.

    Returns:
        The resolved DatasetData instance.

    Raises:
        ValueError: If the arguments contradict each other or leave the dataset without a definition to build from.
            Also raised when an animal named for rebuilding is absent from the dataset or has no provided sessions.
            A provided session that would widen a frozen animal's session set raises too, as does one whose session
            type or acquisition system differs from the dataset's.
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
    """
    if force_recreate and recreate_animals:
        message = (
            f"Unable to resolve the '{name}' dataset. The force_recreate and recreate_animals arguments are "
            f"mutually exclusive, since force_recreate rebuilds the whole dataset from the provided session list "
            f"while recreate_animals rebuilds only the animals it names."
        )
        console.error(message=message, error=ValueError)

    dataset_directory = project_root.joinpath(name)
    if not dataset_directory.exists():
        if recreate_animals:
            message = (
                f"Unable to rebuild the animal(s) {natsorted(recreate_animals)} in the '{name}' dataset. Rebuilding "
                f"an animal requires an existing dataset, but none exists under '{project_root}'."
            )
            console.error(message=message, error=ValueError)
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

    if force_recreate:
        if not session_names:
            message = (
                f"Unable to rebuild the '{name}' dataset. Recreating the dataset from scratch requires a session "
                f"list to define it, but no sessions were provided."
            )
            console.error(message=message, error=ValueError)
        delete_directory(directory_path=dataset_directory)
        console.echo(
            message=f"Dataset '{name}': Removed existing hierarchy for recreation.",
            level=LogLevel.INFO,
        )
        return _create_dataset(
            name=name,
            sessions=session_names,
            project_root=project_root,
            required_session_type=required_session_type,
        )

    dataset = DatasetData.load(dataset_path=dataset_directory)
    if not session_names:
        if recreate_animals:
            message = (
                f"Unable to rebuild the animal(s) {natsorted(recreate_animals)} in the '{name}' dataset. Rebuilding "
                f"an animal replaces its session set, so the session list must define that set, but no sessions "
                f"were provided."
            )
            console.error(message=message, error=ValueError)
        return dataset

    _update_dataset(
        dataset=dataset,
        session_names=session_names,
        project_root=project_root,
        recreate_animals=recreate_animals,
    )
    return dataset


def _create_dataset(
    name: str,
    sessions: tuple[str, ...],
    project_root: Path,
    *,
    required_session_type: SessionTypes | None = None,
) -> DatasetData:
    """Creates a fresh dataset hierarchy by resolving the provided session names under the project root.

    Every included session must share the first session's session type and acquisition system, and when
    ``required_session_type`` is provided, the first session's type must also match it. The dataset's acquisition
    system determines the column descriptions baked into the created dataset.

    Args:
        name: The unique name for the dataset.
        sessions: The non-empty tuple of session names to include in the dataset.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        required_session_type: The session type the first session must have. When None, any session type is
            accepted as the dataset's reference type.

    Returns:
        The newly created DatasetData instance.

    Raises:
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
        ValueError: If required_session_type is provided and the first session's type differs from it, or if any
            subsequent session's session type or acquisition system differs from the first session's.
    """
    session_paths = _resolve_session_paths(sessions=sessions, project_root=project_root)

    first_session_data = SessionData.load(session_path=session_paths[0])
    if required_session_type is not None and first_session_data.session_type != required_session_type:
        message = (
            f"Unable to define dataset '{name}'. Dataset creation for this acquisition system is supported only "
            f"for '{required_session_type}' sessions, but the first session's type resolved to "
            f"'{first_session_data.session_type}'."
        )
        console.error(message=message, error=ValueError)

    # The dataset-level metadata is derived from the first session alone, so the assembly logic downstream depends
    # on every other session matching it.
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

    column_descriptions = resolve_forging_column_descriptions(first_session_data.acquisition_system)

    dataset_sessions = tuple(DatasetSession(animal=path.parent.name, session=path.name) for path in session_paths)
    dataset = DatasetData.create(
        name=name,
        project=project_root.name,
        session_type=first_session_data.session_type,
        acquisition_system=first_session_data.acquisition_system,
        sessions=dataset_sessions,
        datasets_root=project_root,
        column_descriptions=column_descriptions,
    )

    _copy_animal_surgery_files(dataset_name=name, dataset=dataset, source_session_paths=session_paths)

    console.echo(
        message=(
            f"Dataset '{name}' data hierarchy: Defined with {len(sessions)} session(s) across "
            f"{len(dataset.animals)} animal(s)."
        ),
        level=LogLevel.SUCCESS,
    )
    return dataset


def _update_dataset(
    dataset: DatasetData,
    session_names: tuple[str, ...],
    project_root: Path,
    recreate_animals: tuple[str, ...],
) -> None:
    """Brings an existing dataset in line with the provided session list, rebuilding the animals named for it.

    Notes:
        The whole request is screened before the hierarchy is touched, so a rejected request leaves the dataset as
        it stands. A rebuilt animal is removed before anything is added, so a session it keeps across the rebuild
        clears its existing entry before it is appended again.

    Args:
        dataset: The loaded dataset to update in place.
        session_names: The session names the dataset must contain.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        recreate_animals: The identifiers of animals to rebuild from the sessions provided for them.

    Raises:
        ValueError: If an animal named for rebuilding is absent from the dataset or has no provided sessions. Also
            raised when a provided session would widen a frozen animal's session set. An added session whose session
            type or acquisition system differs from the dataset's raises too.
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
    """
    session_paths = _resolve_session_paths(sessions=session_names, project_root=project_root)

    # The animal name is the parent directory name in the source project layout.
    provided_sessions: dict[str, list[Path]] = {}
    for session_path in session_paths:
        provided_sessions.setdefault(session_path.parent.name, []).append(session_path)

    dataset_animals = {dataset_animal.animal for dataset_animal in dataset.animals}
    dataset_sessions = {entry.session for entry in dataset.sessions}

    unknown_animals = natsorted(set(recreate_animals) - dataset_animals)
    if unknown_animals:
        message = (
            f"Unable to rebuild the animal(s) {unknown_animals} in the '{dataset.name}' dataset. Every animal named "
            f"for rebuilding must already be part of the dataset, which holds {natsorted(dataset_animals)}."
        )
        console.error(message=message, error=ValueError)

    undefined_animals = natsorted(animal for animal in recreate_animals if animal not in provided_sessions)
    if undefined_animals:
        message = (
            f"Unable to rebuild the animal(s) {undefined_animals} in the '{dataset.name}' dataset. Rebuilding an "
            f"animal replaces its session set, so the provided session list must hold at least one session for it."
        )
        console.error(message=message, error=ValueError)

    frozen_animals = natsorted(
        animal
        for animal, paths in provided_sessions.items()
        if animal in dataset_animals
        and animal not in recreate_animals
        and any(path.name not in dataset_sessions for path in paths)
    )
    if frozen_animals:
        message = (
            f"Unable to extend the '{dataset.name}' dataset. Sessions absent from the dataset were provided for the "
            f"animal(s) {frozen_animals}, which the dataset already holds. Widening the session set of an animal "
            f"already in a dataset invalidates the outputs already forged for the sessions it keeps, so the animal "
            f"has to be rebuilt as a whole. Name the animal(s) in recreate_animals to rebuild them from the provided "
            f"sessions, or provide sessions only for animals the dataset does not hold."
        )
        console.error(message=message, error=ValueError)

    # The freeze check above leaves every provided session of an untouched animal already part of the dataset.
    added_paths = [
        path
        for animal, paths in provided_sessions.items()
        if animal in recreate_animals or animal not in dataset_animals
        for path in paths
    ]
    if not added_paths:
        console.echo(
            message=f"Dataset '{dataset.name}': Already contains every provided session.",
            level=LogLevel.INFO,
        )
        return

    _verify_session_compatibility(dataset=dataset, session_paths=added_paths)

    for animal in recreate_animals:
        dataset.remove_animal(animal=animal)
    dataset.add_sessions(
        sessions=tuple(DatasetSession(animal=path.parent.name, session=path.name) for path in added_paths)
    )
    _copy_animal_surgery_files(dataset_name=dataset.name, dataset=dataset, source_session_paths=added_paths)

    added_animals = natsorted({path.parent.name for path in added_paths})
    console.echo(
        message=(
            f"Dataset '{dataset.name}' data hierarchy: Updated with {len(added_paths)} session(s) across "
            f"{len(added_animals)} animal(s) {added_animals}."
        ),
        level=LogLevel.SUCCESS,
    )


def _resolve_session_paths(sessions: tuple[str, ...], project_root: Path) -> list[Path]:
    """Resolves each provided session name to its source session directory under the project root.

    Args:
        sessions: The session names to resolve.
        project_root: The path to the project's root directory that stores the animal and session data directories.

    Returns:
        The resolved source session directories, in the order the names were provided.

    Raises:
        FileNotFoundError: If a session name does not resolve to any directory under the project root.
        RuntimeError: If a session name resolves to more than one directory under the project root.
    """
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

    return session_paths


def _verify_session_compatibility(dataset: DatasetData, session_paths: list[Path]) -> None:
    """Verifies that every provided session matches the dataset's recorded session type and acquisition system.

    Args:
        dataset: The dataset the sessions are added to.
        session_paths: The source session directories to verify.

    Raises:
        ValueError: If a session's session type or acquisition system differs from the dataset's.
    """
    for session_path in session_paths:
        session_data = SessionData.load(session_path=session_path)
        if session_data.session_type != dataset.session_type:
            message = (
                f"Unable to add session '{session_path.name}' to the '{dataset.name}' dataset. All sessions in a "
                f"dataset must share the same session type, but the session has type "
                f"'{session_data.session_type}' while the dataset holds '{dataset.session_type}' sessions."
            )
            console.error(message=message, error=ValueError)
        if session_data.acquisition_system != dataset.acquisition_system:
            message = (
                f"Unable to add session '{session_path.name}' to the '{dataset.name}' dataset. All sessions in a "
                f"dataset must be acquired by the same acquisition system, but the session was acquired by "
                f"'{session_data.acquisition_system}' while the dataset was acquired by "
                f"'{dataset.acquisition_system}'."
            )
            console.error(message=message, error=ValueError)


def _copy_animal_surgery_files(
    dataset_name: str,
    dataset: DatasetData,
    source_session_paths: list[Path],
) -> None:
    """Copies the surgery metadata YAML of each covered animal into the dataset's animal directory.

    Surgery metadata is per-animal, so each animal receives a single copy taken from its most recent source session.
    The metadata is optional, and an animal whose latest session carries no snapshot is skipped with a warning. The
    provided paths bound which animals are covered, so extending a dataset covers the animals it gained.

    Args:
        dataset_name: The name of the dataset, used for reporting.
        dataset: The DatasetData instance the animals belong to.
        source_session_paths: The resolved source session directory paths whose animals to cover.
    """
    # The animal name is the parent directory name in the source project layout.
    sessions_by_animal: dict[str, list[Path]] = {}
    for source_path in source_session_paths:
        sessions_by_animal.setdefault(source_path.parent.name, []).append(source_path)

    for animal, animal_sessions in sessions_by_animal.items():
        dataset_animal = dataset.get_animal(animal)

        # Session names are timestamped, so the natural sort orders them chronologically.
        latest_session_name = natsorted([path.name for path in animal_sessions])[-1]
        latest_session_path = next(path for path in animal_sessions if path.name == latest_session_name)
        session_data = SessionData.load(session_path=latest_session_path)

        source_surgery_path = session_data.raw_data.surgery_metadata_path
        if not source_surgery_path.is_file():
            message = (
                f"The latest session '{latest_session_name}' for animal '{animal}' does not contain a "
                f"'{RawDataFiles.SURGERY_METADATA}' file. Skipping the surgery metadata snapshot for this animal in "
                f"the '{dataset_name}' dataset."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            continue

        shutil.copy2(src=source_surgery_path, dst=dataset_animal.surgery_path)
