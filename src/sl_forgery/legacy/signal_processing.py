from typing import Any

from numba import jit
import numpy as np
import scipy
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import filters
from suite2p.extraction import preprocess


class ChunkShuffler:
    """Class for shuffling data in chunks along the time axis."""

    def __init__(self, F: np.ndarray, num_bins: int) -> None:
        """Initializes a ChunkShuffler instance.

        Args:
            F (np.ndarray): Input fluorescence data (num_cells, num_frames).
            num_bins (int): Number of bins (chunks) to split the data into.
        """
        step_size = int(np.ceil(F.shape[1] / num_bins))
        self._num_bins = num_bins
        self._F_split = np.split(F, np.arange(step_size, F.shape[1], step_size, dtype=np.uint16), axis=1)

    def shuffle(self) -> np.ndarray:
        """Shuffle the data by randomly reordering chunks along the time axis.

        Returns:
            np.ndarray: Shuffled data, concatenated after randomized chunk reordering.
        """
        shuffle_ind = np.random.choice(np.arange(self._num_bins), self._num_bins, replace=False)
        return np.concatenate([self._F_split[i] for i in shuffle_ind], axis=1)


def bin_fluorescence_data(
    F: np.ndarray,
    data: pd.Series,
    edges: np.ndarray,
    method: str = "mean",
    threshold: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin fluorescence data according to a reference data array (or Series).

    Args:
        F (np.ndarray): Fluorescence data, shape (num_cells, num_frames).
        data (pd.Series): Values used to bin the data, length must match num_frames in F.
        edges (np.ndarray): Bin edges passed to pandas.cut.
        method (str, optional): Aggregation method for each bin. Valid options:
            'mean', 'sum', 'threshold sum'. Defaults to 'mean'.
        threshold (float, optional): Threshold value for the 'threshold sum' method. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]:
          - Binned fluorescence array of shape (num_cells, num_bins).
          - Array with the count of samples in each bin (num_bins,).

    Raises:
        ValueError: If an unsupported method is provided.
    """
    # return bin label (from 0 to edges.size-1) for each entry (nan if outside of edges).
    bins = pd.cut(data, edges, include_lowest=True, labels=False).to_numpy()
    # I do simple list comprehension here because I also want the values for missing bins.
    uni_bin_ids = np.arange(0, edges.size - 1)
    count = np.array([sum(bins == cbin) for cbin in uni_bin_ids])
    if method == "mean":
        f_binned = np.array(
            [
                (F[:, bins == cbin].mean(axis=1) if sum(bins == cbin) != 0 else np.full((F.shape[0]), np.nan))
                for cbin in uni_bin_ids
            ]
        ).T

    elif method == "sum" or method == "threshold sum":
        f_binned = np.array(
            [
                (F[:, bins == cbin].sum(axis=1) if sum(bins == cbin) != 0 else np.full((F.shape[0]), np.nan))
                for cbin in uni_bin_ids
            ]
        ).T

    else:
        raise ValueError(f"Unsupported method: {method}")

    return f_binned, count


def df_over_f0(
    F: np.ndarray,
    method_baseline: str = "maximin",
    subtract_min: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate ΔF/F₀ (df/f0) for given fluorescence data.

    Args:
        F (np.ndarray): Fluorescence data (num_cells, num_frames).
        method_baseline (str, optional): Baseline calculation method.
            See the baseline() function for valid methods. Defaults to "maximin".
        subtract_min (bool, optional): Whether to subtract the minimum fluorescence from
            each row in F before baseline calculation. Defaults to False.
        **kwargs: Additional keyword arguments passed to the baseline() function.

    Returns:
        tuple[np.ndarray, np.ndarray]:
          - dF_over_f0 (np.ndarray): Resulting (F - f0)/f0, same shape as F.
          - f0 (np.ndarray): Baseline values of shape (num_cells, num_frames).
    """
    if subtract_min:
        F = F - np.min(F, axis=1)[..., np.newaxis]
    f0 = baseline(F, method=method_baseline, **kwargs)
    dF = F - f0
    dF = np.divide(dF, f0)
    return dF, f0


def fold_change(F: np.ndarray, method_baseline: str = "maximin", **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute fold-change (F / F0) for given fluorescence data.

    Args:
        F (np.ndarray): Fluorescence data (num_cells, num_frames).
        method_baseline (str, optional): Baseline calculation method.
            See the baseline() function for valid methods. Defaults to "maximin".
        **kwargs: Additional keyword arguments passed to the baseline() function.

    Returns:
        tuple[np.ndarray, np.ndarray]:
          - fold_change (np.ndarray): F / f0, same shape as F.
          - f0 (np.ndarray): Baseline values of shape (num_cells, num_frames).
    """
    f0 = baseline(F, method=method_baseline, **kwargs)
    # Fix (compared to the original snippet) to use F instead of non-existent dF
    fold = np.divide(F, f0)
    return fold, f0


def baseline(
    F: np.ndarray,
    method: str = "maximin",
    sigma_baseline: float = 20.0,
    window_size: int = 600,
) -> np.ndarray:
    """Calculate the baseline of fluorescence data.

    Supported methods:
      - 'maximin': Applies Gaussian filter, followed by min and max filtering.
      - 'average': Computes mean for each row of F.

    Args:
        F (np.ndarray): Fluorescence data (num_cells, num_frames).
        method (str, optional): Baseline calculation method. Defaults to "maximin".
        sigma_baseline (float, optional): Sigma of the Gaussian filter for 'maximin',
            by default 20.0
        window_size (int, optional): Window size for min/max filtering in 'maximin',
            by default 600

    Returns:
        np.ndarray: Baseline array of the same shape as F.

    Raises:
        ValueError: If an unknown method is provided.
    """
    if method == "maximin":
        if F.ndim == 2:
            Flow = filters.gaussian_filter(F, [0.0, sigma_baseline])
        elif F.ndim == 1:
            Flow = filters.gaussian_filter(F, [sigma_baseline])
        else:
            raise ValueError("Expected F to be 1D or 2D.")

        Flow = filters.minimum_filter1d(Flow, window_size)
        Flow = filters.maximum_filter1d(Flow, window_size)

    elif method == "average":
        Flow = np.tile(np.mean(F, axis=1), (F.shape[1], 1)).T
    else:
        raise ValueError(f"Unknown baseline method: '{method}'")

    return Flow


def quantile_max_treshold(F: np.ndarray, base_quantile: float = 0.25, threshold_factor: float = 0.25) -> np.ndarray:
    """Threshold fluorescence data using a fractional difference
    between a baseline quantile and the maximum value.

    threshold = base_val + (max_val - base_val) * threshold_factor

    Args:
        F (np.ndarray): Fluorescence data (num_cells, num_frames).
        base_quantile (float, optional): Quantile used as baseline. Defaults to 0.25.
        threshold_factor (float, optional): Fraction of (max_val - base_val). Defaults to 0.25.

    Returns:
        np.ndarray: Boolean mask of the same shape as F,
            True where F is above the computed threshold.
    """
    F = F.copy()  # F gets adjusted later on (nan to -inf).
    # get max and baseline value (max peak and quantile value)
    max_val = np.nanmax(F, axis=1)
    quantile_val = np.nanquantile(F, base_quantile, axis=1)

    # get mean of lowest quantile bins.
    base_val = []
    for i in range(F.shape[0]):
        valid_samples = F[i, ~np.isnan(F[i, :])]
        base_val.append(np.nanmean(valid_samples[valid_samples <= quantile_val[i]]))

    # set threshold as percentage difference between baseline and max value.
    threshold = (base_val + ((max_val - base_val) * threshold_factor))[:, np.newaxis]
    threshold = np.tile(threshold, [1, F.shape[1]])
    F[np.isnan(F)] = -np.inf
    return threshold < F  # nan_to_num prevents warning.


def find_calcium_events(
    dF: np.ndarray, bin_size: int = 50, base_quantile: float = 0.5, onset_factor: float = 3.0, end_factor: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Identify significant transient calcium events from dF data.

    Event onset is assumed where dF >= (std_quant * onset_factor) from the corrected baseline,
    and ends where dF <= (std_quant * end_factor) of baseline.

    The baseline σ is calculated by binning the fluorescent trace in short periods of size [bin_size]
    and calculating the std. in each bin. Selected std. is based on the [base_quantile] percentile.

    Based on Tank 2010

    Args:
        dF (np.ndarray): Centered fluorescence data, shape (num_cells, num_frames).
        bin_size (int, optional): Number of frames to pool into each bin for std calculation. Defaults to 50.
        base_quantile (float, optional): Quantile used to pick stable std from each bin. Defaults to 0.5.
        onset_factor (float, optional): Multiplicative factor for the std threshold at which an event starts. Defaults to 3.0.
        end_factor (float, optional): Factor for the std threshold at which an event ends. Defaults to 0.5.

    Returns:
        tuple[np.ndarray, np.ndarray]:
          - event_mask (np.ndarray): Boolean array of shape (num_cells, num_frames), marking events.
          - std_quant (np.ndarray): Per-cell std quantile used for identifying events.
    """
    # reshape traces into equal size bins.
    num_bins = int(np.floor(dF.shape[1] / bin_size))
    bin_dF = dF[:, 0 : (num_bins * bin_size)]
    bin_dF = np.reshape(bin_dF, [dF.shape[0], -1, bin_size])
    # calculate std. per bin.
    std_bin = np.std(bin_dF, axis=2)
    # get quantile.
    std_quant = np.nanquantile(std_bin, base_quantile, axis=1)
    event_mask = np.zeros(dF.shape, np.bool)
    # get start and stop of events.
    for icell in range(dF.shape[0]):
        onset_mask = scipy.signal.convolve(dF[icell, :] >= (std_quant[icell] * onset_factor), [1, -1], "same") == 1
        end_mask = scipy.signal.convolve(dF[icell, :] <= (std_quant[icell] * end_factor), [1, -1], "same") == 1
        event_mask = match_start_end(onset_mask, end_mask, event_mask, icell, dF.shape[1])
    return event_mask, std_quant


@jit(nopython=True)
def np_apply_along_axis(func1d, axis: int, arr: np.ndarray) -> np.ndarray:
    """Apply a 1D function along a specified axis of a 2D NumPy array.

    Args:
        func1d (callable): Function to apply along the specified axis.
        axis (int): Axis along which to apply the function (0 or 1).
        arr (np.ndarray): Input array of shape (M, N).

    Returns:
        np.ndarray: Result of applying func1d along the specified axis.
    """
    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@jit(nopython=True)
def np_min(array: np.ndarray, axis: int) -> float:
    """Compute the minimum value along a specified axis using np_apply_along_axis.

    Args:
        array (np.ndarray): 2D array where the minimum is taken along the given axis.
        axis (int): Axis along which to compute the minimum (0 or 1).

    Returns:
        float: The computed minimum value(s). If axis=0, shape is (N,). If axis=1, shape is (M,).
    """
    return np_apply_along_axis(np.min, axis, array)


@jit(nopython=True)  # Equivalent to @njit for performance optimization
def match_start_end(
    onset_mask: np.ndarray, end_mask: np.ndarray, event_mask: np.ndarray, icell: int, num_frames: int
) -> np.ndarray:
    """Match onset events to end events, marking them in an event mask.

    For each onset index, locate the closest corresponding end index that occurs afterward.
    If no end is found, mark events until the final frame.

    Args:
        onset_mask (np.ndarray): Boolean array where True indicates event start.
        end_mask (np.ndarray): Boolean array where True indicates event end.
        event_mask (np.ndarray): Event mask to update with found events (shape: [num_cells, num_frames]).
        icell (int): Index of the current cell in event_mask.
        num_frames (int): Total number of frames in the data.

    Returns:
        np.ndarray: Updated event_mask for the specified cell.
    """
    # get indices
    onset_ind = np.argwhere(onset_mask)
    end_ind = np.argwhere(end_mask)
    # find minimum positive offset
    offset = np.transpose(end_ind) - onset_ind

    # Replace negative offsets with large positive values to exclude them
    for i in range(offset.shape[0]):
        for j in range(offset.shape[1]):
            if offset[i, j] < 0:
                offset[i, j] = 9999999

    matches = np_min(offset, axis=1)
    onset_flat = onset_ind.flatten()
    matched_end = onset_flat + matches

    # If we never find an end event, mark up to num_frames
    matched_end[matched_end == 9999999] = num_frames

    for i, start_ind in enumerate(onset_flat):
        event_mask[icell, start_ind : matched_end[i]] = True

    return event_mask
