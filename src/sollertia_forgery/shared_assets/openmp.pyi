from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass

_OPENMP_LIBRARY_NAME: str
_PACKAGE_MANAGER_DIRECTORIES: tuple[Path, ...]
_CONDA_PREFIX_VARIABLE: str
_OMPPOOL_MODULE: str
_VENDORED_RUNTIME_PATTERN: str
_LINK_DIRECTORY: Path
_VERIFICATION_SCRIPT: str
_VERIFICATION_TIMEOUT: float

class OpenMPStatus(StrEnum):
    AVAILABLE = "available"
    UNRESOLVED = "unresolved"
    PREVIEWED = "previewed"
    LINKED = "linked"

@dataclass(frozen=True, slots=True)
class _OpenMPSummary:
    status: OpenMPStatus
    unresolved_reason: str
    runtime_path: Path | None
    link_path: Path | None
    searched_paths: tuple[Path, ...]
    loadable: bool
    def describe(self) -> str: ...

def verify_openmp_runtime() -> None: ...
def resolve_openmp_runtime(
    *, runtime_path: Path | None = None, link_path: Path | None = None, execute: bool = False, force: bool = False
) -> _OpenMPSummary: ...
def _openmp_runtime_loadable() -> bool: ...
def _summarize_request(
    status: OpenMPStatus,
    *,
    reason: str = "",
    runtime: Path | None = None,
    link: Path | None = None,
    searched: tuple[Path, ...] = (),
    loadable: bool = False,
) -> _OpenMPSummary: ...
def _resolve_candidate_paths() -> tuple[Path, ...]: ...
def _discover_openmp_runtime(candidates: tuple[Path, ...]) -> Path | None: ...
def _link_would_replace_runtime(runtime_path: Path, link_path: Path) -> bool: ...
def _link_openmp_runtime(runtime_path: Path, link_path: Path) -> None: ...
def _verify_runtime_loadable() -> bool: ...
