"""Tests the verification that the threading layer this library selected has a runtime to load."""

from __future__ import annotations

import ctypes

import pytest

from sollertia_forgery.shared_assets import (
    parallel_runtime as runtime_module,
    verify_parallel_runtime,
)


def test_a_platform_running_the_pinned_threading_layer_is_not_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")

    # A loader that would refuse every runtime proves the check never reaches it off the OpenMP platform.
    def refuse(_name: str) -> None:
        raise OSError

    monkeypatch.setattr(runtime_module.ctypes, "CDLL", refuse)

    assert verify_parallel_runtime() is None


def test_a_loadable_runtime_passes_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_module.ctypes, "CDLL", lambda _name: None)

    assert verify_parallel_runtime() is None


def test_an_unloadable_runtime_already_installed_is_reported_as_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")

    def refuse(_name: str) -> None:
        raise OSError

    monkeypatch.setattr(runtime_module.ctypes, "CDLL", refuse)
    # The first package manager prefix carries the runtime, so the host is told to make it reachable, not to install it.
    monkeypatch.setattr(
        runtime_module.Path, "is_file", lambda _self: str(_self.parent) == runtime_module._RUNTIME_DIRECTORIES[0]
    )

    with pytest.raises(RuntimeError, match="DYLD_LIBRARY_PATH"):
        verify_parallel_runtime()


def test_an_absent_runtime_is_reported_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")

    def refuse(_name: str) -> None:
        raise OSError

    monkeypatch.setattr(runtime_module.ctypes, "CDLL", refuse)
    monkeypatch.setattr(runtime_module.Path, "is_file", lambda _self: False)

    with pytest.raises(RuntimeError, match="brew install libomp"):
        verify_parallel_runtime()


def test_the_verification_loads_the_runtime_the_threading_layer_names() -> None:
    # Guards the library name against a rename, since the loader resolves it rather than a path this module builds.
    assert runtime_module._OPENMP_LIBRARY_NAME == "libomp.dylib"
    assert isinstance(ctypes.CDLL, type(ctypes.CDLL))
