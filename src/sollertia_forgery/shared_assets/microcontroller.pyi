import numpy as np
from numpy.typing import NDArray as NDArray

def merge_event_streams[ScalarT: np.generic](
    timestamps_a: NDArray[np.uint64],
    values_a: NDArray[ScalarT],
    timestamps_b: NDArray[np.uint64],
    values_b: NDArray[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]: ...
