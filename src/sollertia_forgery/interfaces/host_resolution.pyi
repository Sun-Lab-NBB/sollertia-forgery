from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator

from ..server import remote_state_directory as remote_state_directory
from ..orchestration import (
    LOCAL_HOST_LABEL as LOCAL_HOST_LABEL,
    REMOTE_HOST_LABEL as REMOTE_HOST_LABEL,
    LocalHost as LocalHost,
    RemoteHost as RemoteHost,
    ExecutionHost as ExecutionHost,
    connect_to_server as connect_to_server,
    sync_project_state as sync_project_state,
)

HOST_LABELS: frozenset[str]

def unsupported_host_message(host: str) -> str: ...
@contextmanager
def resolve_execution_host(host: str) -> Iterator[ExecutionHost]: ...
def resolve_readable_project(project_path: str, host: str) -> Path: ...
