"""Provides the OpenMP runtime discovery and linking that make Numba's OpenMP threading layer loadable on macOS."""

import os
import sys
from enum import StrEnum
import ctypes
from pathlib import Path
import sysconfig
import subprocess
from dataclasses import dataclass

from ataraxis_base_utilities import console

_OPENMP_LIBRARY_NAME: str = "libomp.dylib"
"""The file name of the OpenMP runtime that Numba's omppool extension loads on macOS."""

_PACKAGE_MANAGER_DIRECTORIES: tuple[Path, ...] = (
    Path("/opt/homebrew/opt/libomp/lib"),
    Path("/usr/local/opt/libomp/lib"),
    Path("/opt/local/lib/libomp"),
)
"""The directories the macOS package managers install the OpenMP runtime into.

The first two are the Homebrew keg paths for the Apple Silicon and the Intel prefix, and the third is the MacPorts
path. A package manager runtime is examined ahead of every other source, because its lifecycle is independent of the
Python distributions this library installs alongside itself.
"""

_CONDA_PREFIX_VARIABLE: str = "CONDA_PREFIX"
"""The environment variable naming the active conda environment, whose lib directory holds an llvm-openmp runtime."""

_VENDORED_RUNTIME_PATTERN: str = f"*/.dylibs/{_OPENMP_LIBRARY_NAME}"
"""Matches the OpenMP runtimes that delocate vendors into the macOS wheels of scikit-learn, torch, and their peers.

A vendored runtime is examined last, because linking it ties the threading layer to the lifecycle of the distribution
that carries it. Upgrading or removing that distribution leaves the link dangling.
"""

_LINK_DIRECTORY: Path = Path("/usr/local/lib")
"""The directory the OpenMP runtime is linked into so that the dynamic loader finds it.

Numba's omppool extension records its dependency as '@rpath/libomp.dylib' and carries no LC_RPATH entries, so the
loader resolves the file name against DYLD_FALLBACK_LIBRARY_PATH instead. This directory is on that path by default,
while the Apple Silicon Homebrew prefix is not.
"""

_VERIFICATION_SCRIPT: str = f"import ctypes; ctypes.CDLL({_OPENMP_LIBRARY_NAME!r})"
"""Loads the OpenMP runtime from a fresh interpreter, which reads the loader search path as it stands after linking."""

_VERIFICATION_TIMEOUT: float = 60.0
"""The seconds to wait for the post-link verification before treating the runtime as unloadable."""


class OpenMPStatus(StrEnum):
    """Defines the outcome of a request to make the OpenMP runtime loadable."""

    AVAILABLE = "available"
    """The runtime already loads, so nothing was changed."""
    UNRESOLVED = "unresolved"
    """No OpenMP runtime was found to link, so nothing was changed."""
    PREVIEWED = "previewed"
    """The link was resolved and reported as a dry run, so nothing was changed."""
    LINKED = "linked"
    """The discovered runtime was linked into a directory the dynamic loader searches."""


@dataclass(frozen=True, slots=True)
class OpenMPSummary:
    """Summarizes a request to make the OpenMP runtime loadable, whether previewed as a dry run or carried out."""

    status: OpenMPStatus
    """The outcome of the request."""
    unresolved_reason: str
    """The reason no OpenMP runtime was found, empty for every other outcome."""
    runtime_path: Path | None
    """The absolute path to the discovered OpenMP runtime, or None when none was found."""
    link_path: Path | None
    """The absolute path to the link that makes the runtime loadable, or None when no runtime was found."""
    searched_paths: tuple[Path, ...]
    """The paths examined while discovering the runtime, in the order they were examined."""
    loadable: bool
    """Determines whether the OpenMP runtime loads from a fresh interpreter once the call returns."""

    def describe(self) -> str:
        """Builds a one-line human-readable summary of what the call resolved and what it changed.

        Returns:
            A compact description of the outcome.
        """
        if self.status == OpenMPStatus.AVAILABLE:
            return "the OpenMP runtime already loads. Pass --force to link a runtime anyway."
        if self.status == OpenMPStatus.UNRESOLVED:
            return f"no OpenMP runtime to link: {self.unresolved_reason}"
        if self.status == OpenMPStatus.PREVIEWED:
            return f"dry run: would link {self.link_path} to {self.runtime_path}. Re-run with --yes to apply."
        if self.loadable:
            return f"linked {self.link_path} to {self.runtime_path}, and the runtime now loads."
        return (
            f"linked {self.link_path} to {self.runtime_path}, but the runtime still does not load. Set "
            f"DYLD_LIBRARY_PATH to include the directory holding the runtime instead."
        )


def verify_openmp_runtime() -> None:
    """Aborts the runtime when Numba's OpenMP threading layer has no runtime to load on this macOS host.

    Notes:
        Every processing pipeline calls this before it dispatches a job, so a host that cannot run a parallelized
        kernel fails while it has done no work rather than partway through a session. The error replaces the
        threading-layer error Numba raises at the first parallelized call, which names no remedy. The platforms whose
        threading layer needs no separately installed runtime return without checking anything.

        The check belongs to the pipelines rather than to the package import, so importing this library to read the
        dataset types it publishes costs nothing on a macOS host that never processes anything.

        This library selects the threading layer for the whole process, so the guarantee covers every kernel the
        process may compile, including the parallelized stages its dependencies own.

    Raises:
        RuntimeError: If the host is macOS and no OpenMP runtime is loadable.
    """
    if sys.platform != "darwin":
        return
    if _openmp_runtime_loadable():
        return

    message = (
        f"Unable to locate the OpenMP runtime ({_OPENMP_LIBRARY_NAME}) that the Numba threading layer loads on macOS. "
        f"Processing fails once it reaches a parallelized stage until the runtime is loadable. Run 'slf omp' to "
        f"report the runtimes found on this host, and 'slf omp --yes' to link one into {_LINK_DIRECTORY}. Install "
        f"one with 'brew install libomp' when the report finds none."
    )
    console.error(message=message, error=RuntimeError)


def resolve_openmp_runtime(
    *,
    runtime_path: Path | None = None,
    link_path: Path | None = None,
    execute: bool = False,
    force: bool = False,
) -> OpenMPSummary:
    """Links a discovered OpenMP runtime into a directory the dynamic loader searches by default.

    The link is what makes Numba's OpenMP threading layer resolve on macOS, because the omppool extension shipped in
    the Numba wheel names its dependency through an rpath it carries no entries for. A host whose runtime already
    loads is left alone unless the link is forced. Only macOS reaches the linking path, because every other platform
    runs the TBB threading layer and gains nothing from an OpenMP runtime.

    Args:
        runtime_path: The absolute path to the OpenMP runtime to link, or None to search the macOS package manager
            directories, the active conda environment, and the installed Python distributions for one.
        link_path: The absolute path to write the link to, or None to derive it from the directory the loader
            searches by default.
        execute: Determines whether to create the resolved link, where a dry run reports it and changes nothing.
        force: Determines whether to link a runtime on a host whose runtime already loads.

    Returns:
        The summary describing the resolved runtime, the resolved link, and what the call changed.

    Raises:
        RuntimeError: If the host runs a platform other than macOS, if the link directory cannot be created, or if
            the link cannot be written.
    """
    if sys.platform != "darwin":
        message = (
            f"Unable to resolve an OpenMP runtime on the '{sys.platform}' platform. Only macOS runs the OpenMP "
            f"threading layer. Every other platform runs TBB, which carries lower overhead on the flat prange loops "
            f"this library and its dependencies compile, so an OpenMP runtime linked here would go unused."
        )
        console.error(message=message, error=RuntimeError)

    if not force and _openmp_runtime_loadable():
        return _summarize_request(status=OpenMPStatus.AVAILABLE, loadable=True)

    candidates = (runtime_path,) if runtime_path is not None else _resolve_candidate_paths()
    resolved_runtime = _discover_openmp_runtime(candidates=candidates)
    if resolved_runtime is None:
        reason = (
            f"none of the examined paths holds a {_OPENMP_LIBRARY_NAME} file. Install the runtime with "
            f"'brew install libomp', or name an existing runtime explicitly."
        )
        return _summarize_request(status=OpenMPStatus.UNRESOLVED, reason=reason, searched=candidates)

    resolved_link = link_path if link_path is not None else _LINK_DIRECTORY / _OPENMP_LIBRARY_NAME
    if not execute:
        return _summarize_request(
            status=OpenMPStatus.PREVIEWED,
            runtime=resolved_runtime,
            link=resolved_link,
            searched=candidates,
        )

    _link_openmp_runtime(runtime_path=resolved_runtime, link_path=resolved_link)
    return _summarize_request(
        status=OpenMPStatus.LINKED,
        runtime=resolved_runtime,
        link=resolved_link,
        searched=candidates,
        loadable=_verify_runtime_loadable(),
    )


def _openmp_runtime_loadable() -> bool:
    """Determines whether the OpenMP runtime that Numba's threading layer needs loads in this interpreter.

    Returns:
        True when the dynamic loader resolves the runtime, and False when it does not.
    """
    try:
        ctypes.CDLL(_OPENMP_LIBRARY_NAME)
    except OSError:
        return False
    return True


def _summarize_request(
    status: OpenMPStatus,
    *,
    reason: str = "",
    runtime: Path | None = None,
    link: Path | None = None,
    searched: tuple[Path, ...] = (),
    loadable: bool = False,
) -> OpenMPSummary:
    """Builds the summary reporting one outcome of an OpenMP runtime request.

    Args:
        status: The outcome the summary reports.
        reason: The reason no OpenMP runtime was found.
        runtime: The absolute path to the discovered OpenMP runtime.
        link: The absolute path to the link that makes the runtime loadable.
        searched: The paths examined while discovering the runtime.
        loadable: Determines whether the runtime loads from a fresh interpreter.

    Returns:
        The assembled summary.
    """
    return OpenMPSummary(
        status=status,
        unresolved_reason=reason,
        runtime_path=runtime,
        link_path=link,
        searched_paths=searched,
        loadable=loadable,
    )


def _resolve_candidate_paths() -> tuple[Path, ...]:
    """Assembles the OpenMP runtime paths to examine, ordered from the most to the least durable installation.

    Returns:
        The candidate paths, covering the macOS package managers, the active conda environment, and the runtimes
        vendored inside the installed Python distributions.
    """
    candidates = [directory / _OPENMP_LIBRARY_NAME for directory in _PACKAGE_MANAGER_DIRECTORIES]

    conda_prefix = os.environ.get(_CONDA_PREFIX_VARIABLE)
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / _OPENMP_LIBRARY_NAME)

    site_packages = sysconfig.get_path("purelib")
    if site_packages:
        candidates.extend(sorted(Path(site_packages).glob(_VENDORED_RUNTIME_PATTERN)))

    return tuple(candidates)


def _discover_openmp_runtime(candidates: tuple[Path, ...]) -> Path | None:
    """Finds the first candidate path that resolves to an existing OpenMP runtime file.

    Args:
        candidates: The paths to examine, in the order they are examined.

    Returns:
        The path to the first existing runtime file, or None when no candidate resolves to one.
    """
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _link_openmp_runtime(runtime_path: Path, link_path: Path) -> None:
    """Creates the symbolic link that places the OpenMP runtime on the loader's default search path.

    Args:
        runtime_path: The absolute path to the OpenMP runtime the link points at.
        link_path: The absolute path to write the link to.

    Raises:
        RuntimeError: If the link directory cannot be created or the link cannot be written.
    """
    try:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(target=runtime_path)
    except OSError as error:
        message = (
            f"Unable to link the OpenMP runtime into {link_path.parent}. Writing the link requires permission to "
            f"modify that directory, which usually means running the command through sudo. The loader reported: "
            f"{error}."
        )
        console.error(message=message, error=RuntimeError)


def _verify_runtime_loadable() -> bool:
    """Determines whether the OpenMP runtime loads from a fresh interpreter, which rereads the loader search path.

    Returns:
        True when the fresh interpreter loads the runtime, and False when it does not.
    """
    # The command is this interpreter running a module-level literal, so no part of it comes from a caller.
    result = subprocess.run(  # noqa: S603 - the executable and the script are both fixed by this module.
        args=[sys.executable, "-c", _VERIFICATION_SCRIPT],
        capture_output=True,
        check=False,
        timeout=_VERIFICATION_TIMEOUT,
    )
    return result.returncode == 0
