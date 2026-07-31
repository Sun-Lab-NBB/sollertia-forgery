from contextlib import contextmanager
from collections.abc import Iterator

from ataraxis_time import PrecisionTimer

DELAY_TIMER: PrecisionTimer
LOG_ARCHIVE_SUFFIX: str
_WORKER_THREAD_VARIABLES: tuple[str, ...]

def delay_terminal() -> None: ...
def multi_recording_dataset_directory(animal_id: str, dataset_name: str) -> str: ...
@contextmanager
def pinned_worker_threads() -> Iterator[None]: ...
