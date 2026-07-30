from pathlib import Path
import datetime
from collections.abc import Sequence

from _typeshed import Incomplete

_SECONDS_PER_DAY: int
_SECONDS_PER_HOUR: int
_SECONDS_PER_MINUTE: int

class Job:
    remote_script_path: str
    job_id: str | None
    job_name: str
    _command: _SlurmScript
    def __init__(
        self,
        job_name: str,
        output_log: Path,
        error_log: Path,
        working_directory: Path,
        conda_environment: str,
        cpu_threads: int = 10,
        ram: int = 10,
        time: int = 60,
        dependencies: Sequence[str] = (),
    ) -> None: ...
    def __repr__(self) -> str: ...
    def add_command(self, command: str) -> None: ...
    @property
    def command_script(self) -> str: ...

class _SlurmScript:
    __slots__: Incomplete
    _directives: list[str]
    _preamble: list[str]
    _commands: list[str]
    _cleanup_path: str
    def __init__(
        self,
        cpus_per_task: int,
        job_name: str,
        output: str,
        error: str,
        memory: str,
        time: datetime.timedelta,
        dependencies: tuple[str, ...] = (),
        cleanup_path: str = "",
    ) -> None: ...
    def add_preamble(self, command: str) -> None: ...
    def add_command(self, command: str) -> None: ...
    def render(self) -> str: ...
    @staticmethod
    def _format_time(time_delta: datetime.timedelta) -> str: ...
