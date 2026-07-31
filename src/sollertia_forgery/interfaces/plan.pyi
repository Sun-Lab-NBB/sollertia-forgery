from pathlib import Path

from ..orchestration import (
    project_plan_path as project_plan_path,
    resolve_dataset_plan as resolve_dataset_plan,
    resolve_session_plan as resolve_session_plan,
    generate_project_plan as generate_project_plan,
)

_CONTEXT_SETTINGS: dict[str, int]

def plan_cli() -> None: ...
def plan_session_command(session_path: tuple[Path, ...], *, regenerate_plan: bool) -> None: ...
def plan_dataset_command(dataset_path: tuple[Path, ...], *, regenerate_plan: bool) -> None: ...
def plan_project_command(project_path: Path) -> None: ...
