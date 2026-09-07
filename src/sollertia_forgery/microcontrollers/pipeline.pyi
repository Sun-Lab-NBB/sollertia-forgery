from pathlib import Path
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor

from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker
from ataraxis_communication_interface import (
    JobUniverse as JobUniverse,
    ControllerExtractionConfig,
)

from ..registries import (
    MicrocontrollerParser as MicrocontrollerParser,
    resolve_microcontroller_parsers as resolve_microcontroller_parsers,
    resolve_microcontroller_event_codes as resolve_microcontroller_event_codes,
    resolve_eligible_microcontroller_modules as resolve_eligible_microcontroller_modules,
)
from ..shared_assets import verify_openmp_runtime as verify_openmp_runtime

PARSE_JOB_NAME: str

def run_microcontroller_processing_pipeline(
    session_path: Path, job_id: str | None = None, *, workers: int = -1, display_progress: bool = False
) -> None: ...
def discover_microcontroller_jobs(
    session_path: Path,
) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]: ...
def microcontroller_job_prerequisites(
    session: SessionData, universe: list[tuple[str, str]]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def _resolve_eligible_event_codes(session: SessionData) -> dict[tuple[int, int], tuple[int, ...]]: ...
def _resolve_controllers(
    session: SessionData, event_codes: Mapping[tuple[int, int], tuple[int, ...]], job_universe: JobUniverse
) -> dict[str, ControllerExtractionConfig]: ...
def _materialize_extraction_configuration(
    controllers: Mapping[str, ControllerExtractionConfig], output_directory: Path
) -> Path: ...
def _extract_controller(
    archive_path: Path,
    output_directory: Path,
    controller_id: str,
    configuration_path: Path,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None = None,
) -> None: ...
def _discover_jobs(
    controllers: dict[str, ControllerExtractionConfig],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    job_universe: JobUniverse,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Path], dict[str, tuple[str, int, int]]]: ...
def _run_extraction_stage(
    extraction_archives: dict[str, Path],
    extraction_output: Path,
    configuration_path: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    executor: ProcessPoolExecutor | None,
    display_progress: bool,
) -> None: ...
def _run_parse_stage(
    parse_specifiers: dict[str, tuple[str, int, int]],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    session: SessionData,
    extraction_output: Path,
    parse_output: Path,
    tracker: ProcessingTracker,
    *,
    executor: ProcessPoolExecutor | None,
    display_progress: bool,
) -> None: ...
def _execute_parse_jobs_sequential(
    runnable: dict[str, tuple[Path, MicrocontrollerParser]],
    tracker: ProcessingTracker,
    session: SessionData,
    parse_output: Path,
    *,
    display_progress: bool,
) -> None: ...
def _execute_parse_jobs_parallel(
    runnable: dict[str, tuple[Path, MicrocontrollerParser]],
    tracker: ProcessingTracker,
    session: SessionData,
    parse_output: Path,
    *,
    executor: ProcessPoolExecutor,
    display_progress: bool,
) -> None: ...
def _execute_remote_job(
    job_id: str,
    universe: list[tuple[str, str]],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    session: SessionData,
    log_directory: Path,
    extraction_archives: Mapping[str, Path],
    extraction_output: Path,
    parse_output: Path,
    configuration_path: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None: ...
def _run_parse(
    feather_path: Path, module_parser: MicrocontrollerParser, output_directory: Path, session: SessionData
) -> None: ...
