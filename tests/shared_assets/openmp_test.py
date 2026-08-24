"""Tests the OpenMP runtime discovery and linking that make Numba's OpenMP threading layer loadable on macOS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sollertia_forgery.shared_assets import (
    OpenMPStatus,
    OpenMPSummary,
    openmp as openmp_module,
    verify_openmp_runtime,
    resolve_openmp_runtime,
)

if TYPE_CHECKING:
    from pathlib import Path


def _refuse(_name: str) -> None:
    """Stands in for a dynamic loader that resolves no runtime at all.

    Args:
        _name: The library name the loader was asked for, which this stand-in never resolves.

    Raises:
        OSError: Always, which is what ctypes raises for a library the loader cannot find.
    """
    raise OSError


@pytest.fixture
def darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presents the host as macOS, which is the only platform that runs the OpenMP threading layer."""
    monkeypatch.setattr(openmp_module.sys, "platform", "darwin")


@pytest.fixture
def unloadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presents the dynamic loader as resolving no OpenMP runtime."""
    monkeypatch.setattr(openmp_module.ctypes, "CDLL", _refuse)


@pytest.fixture
def loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presents the dynamic loader as resolving the OpenMP runtime."""
    monkeypatch.setattr(openmp_module.ctypes, "CDLL", lambda _name: None)


# Verification


def test_a_platform_running_the_tbb_layer_is_not_verified(monkeypatch: pytest.MonkeyPatch, unloadable: None) -> None:
    """Every platform other than macOS runs TBB, which this library pins as a dependency and never has to discover."""
    monkeypatch.setattr(openmp_module.sys, "platform", "linux")

    assert verify_openmp_runtime() is None


def test_a_loadable_runtime_passes_verification(darwin: None, loadable: None) -> None:
    """A host whose loader already resolves the runtime runs every parallelized stage, so nothing is reported."""
    assert verify_openmp_runtime() is None


def test_an_unloadable_runtime_names_the_command_that_links_one(darwin: None, unloadable: None) -> None:
    """The refusal replaces Numba's own threading-layer error, which names no remedy, so it has to name the command."""
    with pytest.raises(RuntimeError, match=r"slf omp"):
        verify_openmp_runtime()


# Discovery


def test_the_candidates_run_from_the_package_managers_to_the_vendored_runtimes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A package manager runtime outlives the distributions installed beside it, so it is examined first and a
    runtime vendored inside a wheel is examined last."""
    monkeypatch.setenv(openmp_module._CONDA_PREFIX_VARIABLE, str(tmp_path))
    vendored = tmp_path.joinpath("torch", ".dylibs")
    vendored.mkdir(parents=True)
    vendored.joinpath(openmp_module._OPENMP_LIBRARY_NAME).touch()
    monkeypatch.setattr(openmp_module.sysconfig, "get_path", lambda _name: str(tmp_path))

    candidates = openmp_module._resolve_candidate_paths()

    assert candidates[: len(openmp_module._PACKAGE_MANAGER_DIRECTORIES)] == tuple(
        directory / openmp_module._OPENMP_LIBRARY_NAME for directory in openmp_module._PACKAGE_MANAGER_DIRECTORIES
    )
    assert tmp_path.joinpath("lib", openmp_module._OPENMP_LIBRARY_NAME) in candidates
    assert candidates[-1] == vendored.joinpath(openmp_module._OPENMP_LIBRARY_NAME)


def test_an_unset_conda_prefix_contributes_no_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host running outside a conda environment names no environment lib directory to examine."""
    monkeypatch.delenv(openmp_module._CONDA_PREFIX_VARIABLE, raising=False)
    monkeypatch.setattr(openmp_module.sysconfig, "get_path", lambda _name: None)

    assert openmp_module._resolve_candidate_paths() == tuple(
        directory / openmp_module._OPENMP_LIBRARY_NAME for directory in openmp_module._PACKAGE_MANAGER_DIRECTORIES
    )


def test_discovery_answers_with_the_first_existing_runtime(tmp_path: Path) -> None:
    """The candidates are ordered by how durable each installation is, so the first hit is the one to link."""
    present = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    present.touch()

    assert openmp_module._discover_openmp_runtime(candidates=(tmp_path.joinpath("absent"), present)) == present


def test_discovery_answers_with_none_when_no_candidate_exists(tmp_path: Path) -> None:
    """A host carrying no runtime resolves nothing rather than naming a path that holds no file."""
    assert openmp_module._discover_openmp_runtime(candidates=(tmp_path.joinpath("absent"),)) is None


# Linking


def test_linking_off_the_openmp_platform_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TBB host gains nothing from an OpenMP runtime, so a link written there would go unused."""
    monkeypatch.setattr(openmp_module.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="Only macOS runs the OpenMP"):
        resolve_openmp_runtime()


def test_a_loadable_runtime_is_left_alone(darwin: None, loadable: None) -> None:
    """A host that already works is not relinked, so the command is safe to run repeatedly."""
    summary = resolve_openmp_runtime()

    assert summary.status == OpenMPStatus.AVAILABLE
    assert summary.loadable
    assert summary.runtime_path is None


def test_forcing_relinks_a_host_whose_runtime_already_loads(
    monkeypatch: pytest.MonkeyPatch, darwin: None, loadable: None, tmp_path: Path
) -> None:
    """Forcing is what repoints the link at a chosen runtime on a host the discovery would otherwise skip."""
    runtime = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    runtime.touch()
    link = tmp_path.joinpath("link", openmp_module._OPENMP_LIBRARY_NAME)
    monkeypatch.setattr(openmp_module, "_verify_runtime_loadable", lambda: True)

    summary = resolve_openmp_runtime(runtime_path=runtime, link_path=link, execute=True, force=True)

    assert summary.status == OpenMPStatus.LINKED
    assert link.is_symlink()
    assert link.resolve() == runtime.resolve()


def test_an_undiscoverable_runtime_is_reported_rather_than_linked(
    monkeypatch: pytest.MonkeyPatch, darwin: None, unloadable: None, tmp_path: Path
) -> None:
    """A host carrying no runtime is told to install one, and the paths it examined are reported alongside."""
    monkeypatch.delenv(openmp_module._CONDA_PREFIX_VARIABLE, raising=False)
    monkeypatch.setattr(openmp_module.sysconfig, "get_path", lambda _name: str(tmp_path))
    monkeypatch.setattr(openmp_module.Path, "is_file", lambda _self: False)

    summary = resolve_openmp_runtime(execute=True)

    assert summary.status == OpenMPStatus.UNRESOLVED
    assert "brew install libomp" in summary.unresolved_reason
    assert summary.searched_paths
    assert summary.link_path is None


def test_a_dry_run_resolves_the_link_and_changes_nothing(darwin: None, unloadable: None, tmp_path: Path) -> None:
    """The command reports before it writes, so an operator sees the link it would create under sudo."""
    runtime = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    runtime.touch()
    link = tmp_path.joinpath("link", openmp_module._OPENMP_LIBRARY_NAME)

    summary = resolve_openmp_runtime(runtime_path=runtime, link_path=link)

    assert summary.status == OpenMPStatus.PREVIEWED
    assert summary.runtime_path == runtime
    assert summary.link_path == link
    assert not link.parent.exists()


def test_the_default_link_lands_where_the_loader_searches(
    monkeypatch: pytest.MonkeyPatch, darwin: None, unloadable: None, tmp_path: Path
) -> None:
    """Numba's omppool extension carries no rpath entries, so the link has to sit on the loader's default path."""
    runtime = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    runtime.touch()

    summary = resolve_openmp_runtime(runtime_path=runtime)

    assert summary.link_path == openmp_module._LINK_DIRECTORY / openmp_module._OPENMP_LIBRARY_NAME


def test_naming_the_link_path_as_the_runtime_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a runtime already sitting at the link path is left alone rather than replaced by a self-link.

    Unlinking the destination before writing the link would remove the only copy of the runtime and leave a link
    resolving to nothing, which reports as a successful link while the host loses the runtime entirely.
    """
    runtime = tmp_path.joinpath("libomp.dylib")
    runtime.write_bytes(b"the runtime")
    monkeypatch.setattr(openmp_module.sys, "platform", "darwin")
    monkeypatch.setattr(openmp_module, "_openmp_runtime_loadable", lambda: False)

    with pytest.raises(RuntimeError, match="already sits where the link would be written"):
        resolve_openmp_runtime(runtime_path=runtime, link_path=runtime, execute=True)

    assert runtime.read_bytes() == b"the runtime"
    assert not runtime.is_symlink()


def test_naming_the_runtime_by_a_second_route_to_the_same_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a runtime reached by another spelling of its path is still recognized as sitting at the link.

    Neither the command's arguments nor the discovery resolve the paths they are given, so an operator naming the
    runtime relative to the link's own directory spells one file two ways. Comparing the spellings rather than the
    files they name would miss that and replace the host's only runtime with a link pointing at itself.
    """
    runtime = tmp_path.joinpath("lib", "libomp.dylib")
    runtime.parent.mkdir()
    runtime.write_bytes(b"the runtime")
    second_route = tmp_path.joinpath("lib", "..", "lib", "libomp.dylib")
    monkeypatch.setattr(openmp_module.sys, "platform", "darwin")
    monkeypatch.setattr(openmp_module, "_openmp_runtime_loadable", lambda: False)

    with pytest.raises(RuntimeError, match="already sits where the link would be written"):
        resolve_openmp_runtime(runtime_path=second_route, link_path=runtime, execute=True)

    assert runtime.read_bytes() == b"the runtime"
    assert not runtime.is_symlink()


def test_a_failed_link_leaves_the_previous_one_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a link this call cannot write leaves whatever the destination already held.

    The link is published by renaming a temporary onto the destination, so the destination is never empty between the
    removal of the old link and the arrival of the new one.
    """
    runtime = tmp_path.joinpath("libomp.dylib")
    runtime.write_bytes(b"the runtime")
    previous_target = tmp_path.joinpath("previous.dylib")
    previous_target.write_bytes(b"the previous runtime")
    link = tmp_path.joinpath("link", "libomp.dylib")
    link.parent.mkdir()
    link.symlink_to(target=previous_target)

    def _refuse(self: Path, target: Path) -> None:
        message = "the filesystem refused the link"
        raise OSError(message)

    monkeypatch.setattr(openmp_module.sys, "platform", "darwin")
    monkeypatch.setattr(openmp_module, "_openmp_runtime_loadable", lambda: False)
    monkeypatch.setattr(openmp_module.Path, "symlink_to", _refuse)

    with pytest.raises(RuntimeError, match="Unable to link the OpenMP runtime into"):
        resolve_openmp_runtime(runtime_path=runtime, link_path=link, execute=True)

    assert link.is_symlink()
    assert link.resolve() == previous_target
    # The temporary the publication would have renamed is cleaned up rather than left beside the destination.
    assert sorted(entry.name for entry in link.parent.iterdir()) == ["libomp.dylib"]


def test_linking_replaces_a_stale_link(darwin: None, unloadable: None, tmp_path: Path, monkeypatch) -> None:
    """A rerun repoints an existing link rather than failing on it, so the command stays idempotent."""
    runtime = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    runtime.touch()
    link = tmp_path.joinpath(f"stale_{openmp_module._OPENMP_LIBRARY_NAME}")
    link.symlink_to(tmp_path.joinpath("absent"))
    monkeypatch.setattr(openmp_module, "_verify_runtime_loadable", lambda: False)

    summary = resolve_openmp_runtime(runtime_path=runtime, link_path=link, execute=True)

    assert summary.status == OpenMPStatus.LINKED
    assert not summary.loadable
    assert link.resolve() == runtime.resolve()


def test_an_unwritable_link_directory_names_the_permission_remedy(
    monkeypatch: pytest.MonkeyPatch, darwin: None, unloadable: None, tmp_path: Path
) -> None:
    """The default link directory is root-owned, so the failure has to name sudo rather than report a bare errno."""
    runtime = tmp_path.joinpath(openmp_module._OPENMP_LIBRARY_NAME)
    runtime.touch()

    def refuse_mkdir(*_args: object, **_kwargs: object) -> None:
        raise OSError

    monkeypatch.setattr(openmp_module.Path, "mkdir", refuse_mkdir)

    with pytest.raises(RuntimeError, match="sudo"):
        resolve_openmp_runtime(runtime_path=runtime, link_path=tmp_path.joinpath("link", "x"), execute=True)


def test_the_post_link_verification_runs_a_fresh_interpreter() -> None:
    """The loader search path is read once per process, so only a new interpreter reports whether the link took."""
    assert openmp_module._verify_runtime_loadable() is True


# Reporting


@pytest.mark.parametrize(
    ("status", "loadable", "expected"),
    [
        (OpenMPStatus.AVAILABLE, True, "already loads"),
        (OpenMPStatus.UNRESOLVED, False, "no OpenMP runtime to link"),
        (OpenMPStatus.PREVIEWED, False, "dry run"),
        (OpenMPStatus.LINKED, True, "the runtime now loads"),
        (OpenMPStatus.LINKED, False, "still does not load"),
    ],
)
def test_every_outcome_describes_itself(status: OpenMPStatus, expected: str, *, loadable: bool) -> None:
    """The command prints this line as its whole result, so each outcome has to read on its own."""
    summary = OpenMPSummary(
        status=status,
        unresolved_reason="reason",
        runtime_path=None,
        link_path=None,
        searched_paths=(),
        loadable=loadable,
    )

    assert expected in summary.describe()
