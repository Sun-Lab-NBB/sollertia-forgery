from pathlib import Path
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor

from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker
from ataraxis_communication_interface.microcontroller import ControllerExtractionConfig

from ..registries import (
    MicrocontrollerParser as MicrocontrollerParser,
    resolve_microcontroller_parsers as resolve_microcontroller_parsers,
    resolve_microcontroller_event_codes as resolve_microcontroller_event_codes,
    resolve_eligible_microcontroller_modules as resolve_eligible_microcontroller_modules,
)
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX as LOG_ARCHIVE_SUFFIX,
    tracked_job as tracked_job,
    partition_events as partition_events,
    find_module_feathers as find_module_feathers,
    pinned_worker_threads as pinned_worker_threads,
    parse_module_feather_name as parse_module_feather_name,
)

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
    session: SessionData, event_codes: Mapping[tuple[int, int], tuple[int, ...]]
) -> dict[str, ControllerExtractionConfig]: ...
def _find_controller_archive(log_directory: Path, controller_id: str) -> Path | None: ...
def _extract_controller(
    archive_path: Path,
    output_directory: Path,
    controller_id: str,
    controller_config: ControllerExtractionConfig,
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
    log_directory: Path,
    extraction_job_name: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Path], dict[str, tuple[str, int, int]]]: ...
def _run_extraction_stage(
    extraction_archives: dict[str, Path],
    controllers: dict[str, ControllerExtractionConfig],
    extraction_output: Path,
    tracker: ProcessingTracker,
    extraction_job_name: str,
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
    extraction_job_name: str,
    controllers: dict[str, ControllerExtractionConfig],
    parsers: Mapping[tuple[int, int], MicrocontrollerParser],
    session: SessionData,
    log_directory: Path,
    extraction_output: Path,
    parse_output: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
) -> None: ...
def _run_parse(
    feather_path: Path, module_parser: MicrocontrollerParser, output_directory: Path, session: SessionData
) -> None: ...
def _index_module_feathers(extraction_output: Path) -> dict[tuple[str, int, int], Path]: ...
