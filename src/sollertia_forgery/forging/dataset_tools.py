"""Provides Model Context Protocol (MCP) tools for authoring, discovering, and inspecting analysis datasets."""

from __future__ import annotations

from enum import Enum
from typing import Any, get_type_hints
from pathlib import Path
from dataclasses import MISSING, fields, is_dataclass

from ataraxis_base_utilities import ensure_directory_exists
from sollertia_shared_assets import SessionTypes, AcquisitionSystems

from ..interfaces import mcp
from .dataset_data import DatasetData, DatasetSession
from ..shared_assets import validate_directory

_DATASET_MARKER_FILENAME: str = "dataset.yaml"
"""Marker filename used to identify dataset directories during recursive discovery walks."""


def _serialize(value: Any) -> Any:  # noqa: ANN401
    """Recursively converts a dataclass, Path, Enum, mapping, or sequence into JSON-friendly Python."""
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_definition.name: _serialize(value=getattr(value, field_definition.name))
            for field_definition in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize(value=item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(value=item) for item in value]
    return value


def _describe_type(type_hint: Any) -> str:  # noqa: ANN401
    """Returns a human-readable string for the given type hint."""
    if type_hint is None:
        return "None"
    if isinstance(type_hint, type):
        return type_hint.__name__
    return str(type_hint).replace("typing.", "")


def _describe_dataclass(cls: type) -> dict[str, Any]:
    """Returns a structured schema description of a dataclass type."""
    if not is_dataclass(cls):
        return {"type": _describe_type(type_hint=cls)}

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    schema: dict[str, Any] = {"class": cls.__name__, "fields": {}}
    # noinspection PyDataclass
    for field_definition in fields(cls):
        type_hint = hints.get(field_definition.name, field_definition.type)
        field_schema: dict[str, Any] = {"type": _describe_type(type_hint=type_hint)}
        if field_definition.default is not MISSING:
            field_schema["default"] = _serialize(value=field_definition.default)
        elif field_definition.default_factory is not MISSING:
            try:
                field_schema["default"] = _serialize(value=field_definition.default_factory())
            except Exception:
                field_schema["required"] = True
        else:
            field_schema["required"] = True
        schema["fields"][field_definition.name] = field_schema
    return schema


def _load_dataset_summary(marker: Path) -> dict[str, Any]:
    """Loads a DatasetData YAML and returns a flat summary dict for discovery responses."""
    dataset_root = marker.parent
    try:
        instance: DatasetData = DatasetData.load(dataset_path=dataset_root)
    except Exception as exception:
        return {
            "dataset_path": str(dataset_root),
            "marker": str(marker),
            "error": f"Failed to load dataset: {exception}",
        }
    return {
        "name": instance.name,
        "project": instance.project,
        "session_type": _serialize(value=instance.session_type),
        "acquisition_system": _serialize(value=instance.acquisition_system),
        "session_count": len(instance.sessions),
        "animal_count": len(instance.animals),
        "dataset_path": str(dataset_root),
        "dataset_data_path": str(instance.dataset_data_path),
    }


@mcp.tool()
def discover_datasets_tool(
    datasets_root: str,
    project: str | None = None,
) -> dict[str, Any]:
    """Recursively discovers all dataset directories under the datasets root.

    Walks the directory tree looking for ``dataset.yaml`` markers and returns a flat list of dataset summaries.

    Args:
        datasets_root: The absolute path to the datasets root directory to search. Searched recursively.
        project: When provided, only datasets belonging to this project are returned.

    Returns:
        A dictionary with ``datasets`` (list of dataset summary dicts), ``total_datasets``, and ``datasets_root``,
        or ``{"error": ...}`` on failure.
    """
    error = validate_directory(datasets_root)
    if error is not None:
        return {"error": error}

    root = Path(datasets_root)
    markers = sorted(root.rglob(_DATASET_MARKER_FILENAME))
    datasets: list[dict[str, Any]] = []
    for marker in markers:
        summary = _load_dataset_summary(marker=marker)
        if project is not None and summary.get("project") != project:
            continue
        datasets.append(summary)

    return {"datasets": datasets, "total_datasets": len(datasets), "datasets_root": str(root)}


@mcp.tool()
def read_dataset_tool(dataset_path: str) -> dict[str, Any]:
    """Loads the DatasetData YAML for a dataset.

    Args:
        dataset_path: Path to the dataset root directory (containing ``dataset.yaml``).

    Returns:
        A dictionary with ``data`` containing the full DatasetData payload (including the resolved session list)
        and ``dataset_path``, or ``{"error": ...}`` on failure.
    """
    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"Dataset path does not exist: {path}"}
    try:
        instance = DatasetData.load(dataset_path=path)
    except Exception as exception:
        return {"error": f"Failed to load DatasetData: {exception}"}
    return {"data": _serialize(value=instance), "dataset_path": str(path)}


@mcp.tool()
def write_dataset_tool(
    name: str,
    project: str,
    session_type: str,
    acquisition_system: str,
    sessions: list[dict[str, str]],
    datasets_root: str,
) -> dict[str, Any]:
    """Creates a new dataset by materializing the dataset hierarchy on disk.

    Wraps :meth:`DatasetData.create`: builds the dataset directory, creates animal and session subdirectories, and
    writes the ``dataset.yaml`` manifest. Each entry in ``sessions`` must specify the ``session`` name and
    ``animal`` ID; the ``session_path`` field is resolved automatically inside the dataset hierarchy.

    Args:
        name: The unique dataset name.
        project: The source project name.
        session_type: The SessionTypes value all sessions belong to.
        acquisition_system: The AcquisitionSystems value all sessions were acquired on.
        sessions: List of dicts, each with ``{"session": str, "animal": str}``.
        datasets_root: Absolute path to the datasets root directory under which the dataset hierarchy is created.

    Returns:
        A dictionary with ``dataset_path``, ``dataset_data_path``, and ``data`` (the materialized DatasetData
        payload), or ``{"error": ...}`` on failure.
    """
    try:
        session_type_enum = SessionTypes(session_type)
    except ValueError:
        valid = ", ".join(member.value for member in SessionTypes)
        return {"error": f"Invalid session_type '{session_type}'. Valid values: {valid}"}

    try:
        acquisition_system_enum = AcquisitionSystems(acquisition_system)
    except ValueError:
        valid = ", ".join(member.value for member in AcquisitionSystems)
        return {"error": f"Invalid acquisition_system '{acquisition_system}'. Valid values: {valid}"}

    if not sessions:
        return {"error": "The 'sessions' argument must contain at least one entry."}

    # Converts raw session dicts into typed DatasetSession objects with basic structure validation.
    dataset_session_objects: list[DatasetSession] = []
    for entry in sessions:
        if not isinstance(entry, dict) or "session" not in entry or "animal" not in entry:
            return {
                "error": f"Invalid session entry {entry!r}. Each entry must be a dict with 'session' and 'animal' keys.",
            }
        dataset_session_objects.append(DatasetSession(session=entry["session"], animal=entry["animal"]))

    root = Path(datasets_root)
    ensure_directory_exists(path=root)

    try:
        instance = DatasetData.create(
            name=name,
            project=project,
            session_type=session_type_enum,
            acquisition_system=acquisition_system_enum,
            sessions=tuple(dataset_session_objects),
            datasets_root=root,
        )
    except Exception as exception:
        return {"error": f"Failed to create dataset: {exception}"}

    return {
        "dataset_path": str(instance.dataset_data_path.parent),
        "dataset_data_path": str(instance.dataset_data_path),
        "data": _serialize(value=instance),
    }


@mcp.tool()
def validate_dataset_tool(dataset_path: str) -> dict[str, Any]:
    """Loads and validates a dataset, verifying that all referenced session paths still exist.

    Args:
        dataset_path: Path to the dataset root directory.

    Returns:
        A dictionary with ``valid``, ``issues``, ``missing_sessions`` (when invalid), and ``summary``.
    """
    path = Path(dataset_path)
    if not path.exists():
        return {"valid": False, "issues": [f"Dataset path does not exist: {path}"]}
    try:
        dataset = DatasetData.load(dataset_path=path)
    except Exception as exception:
        return {"valid": False, "issues": [str(exception)]}

    missing: list[dict[str, str]] = [
        {
            "session": session.session,
            "animal": session.animal,
            "expected_path": str(session.session_path),
        }
        for session in dataset.sessions
        if not session.session_path.exists()
    ]

    return {
        "valid": not missing,
        "issues": [f"Missing session: {entry['animal']}/{entry['session']}" for entry in missing],
        "missing_sessions": missing,
        "summary": {
            "name": dataset.name,
            "project": dataset.project,
            "session_type": _serialize(value=dataset.session_type),
            "session_count": len(dataset.sessions),
            "animal_count": len(dataset.animals),
        },
    }


@mcp.tool()
def describe_dataset_schema_tool() -> dict[str, Any]:
    """Returns the schema for DatasetData and its nested DatasetSession.

    Returns:
        A dictionary with ``schema`` containing the DatasetData schema and ``nested_classes`` mapping
        ``DatasetSession`` to its field schema.
    """
    schema = _describe_dataclass(cls=DatasetData)
    schema["nested_classes"] = {"DatasetSession": _describe_dataclass(cls=DatasetSession)}
    return {"schema": schema}
