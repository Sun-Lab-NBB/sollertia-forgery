from pathlib import Path

import polars as pl

from ..shared_assets import natural_sort as natural_sort

_PROJECT_JOBS_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType]

def project_jobs_path(project_directory: Path) -> Path: ...
def write_project_jobs(project_directory: Path, job_rows: list[dict[str, str | int | None]]) -> Path: ...
