from pathlib import Path

from ..forging import (
    run_forging_pipeline as run_forging_pipeline,
    define_forging_dataset as define_forging_dataset,
)

_CONTEXT_SETTINGS: dict[str, int]

def forge_command(
    dataset_name: str,
    project_path: Path,
    session: tuple[str, ...],
    job_id: str | None,
    workers: int,
    recreate_animal: tuple[str, ...],
    *,
    force_recreate: bool,
    no_progress: bool,
) -> None: ...
