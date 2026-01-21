"""This module provides functionality for detecting and analyzing place fields
in neural recordings. It implements methods for identifying spatial tuning
of neurons on linear tracks.
"""

from copy import deepcopy
from typing import Any

import dask
import vr2p
import numpy as np
import colorcet as cc
import dask.array as da
from scipy.signal import convolve2d
from scipy.ndimage import label
import dask.dataframe
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import vr2p.signal_processing


class PlaceFields1d:
    """Class representing detected place fields in one-dimensional space.

    This class holds data on detected place fields, including their labels, fluorescence data, and spatial centers.

    Args:
        label_im (np.ndarray): Labeled image of detected place fields (size: num_cells x num_bins).
        binF (np.ndarray): Binned fluorescence data (size: num_cells x num_bins).
        centers (np.ndarray, optional): Centers of detected place fields. Defaults to an empty array.
        bin_size (float, optional): Size of spatial bins. Defaults to 1.

    Attributes:
        label_im (np.ndarray): Labeled image of detected place fields.
        binF (np.ndarray): Binned fluorescence data.
        centers (np.ndarray): Centers of detected place fields.
        bin_size (float): Size of spatial bins.
    """

    def __init__(self, label_im: np.ndarray, binF: np.ndarray, centers: np.ndarray = np.array([]), bin_size: float = 1):
        self.bin_size = bin_size
        self.label_im = label_im.astype(int)
        self.binF = binF.astype(float)
        self.centers = centers
        if len(centers) == 0:
            props = regionprops(self.label_im, self.binF, cache=False)
            self.centers = np.array([prop["weighted_centroid"] * np.array([1, bin_size]) for prop in props])

    @property
    def mean_intensity(self) -> np.ndarray:
        """Calculate the mean intensity for each detected place field.

        Returns:
            np.ndarray: Mean intensity values for each place field.
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([prop["mean_intensity"] for prop in props])

    @property
    def max_intensity(self) -> np.ndarray:
        """Calculate the maximum intensity for each detected place field.

        Returns:
            np.ndarray: Maximum intensity values for each place field.
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([prop["max_intensity"] for prop in props])

    @property
    def cell_id(self) -> np.ndarray:
        """Get the cell ID for each detected place field.

        Returns:
            np.ndarray: Cell ID for each place field.
        """
        props = regionprops(self.label_im, self.binF, cache=False)
        return np.array([region["coords"][0, 0] for region in props]).astype(int)

    @property
    def has_place_field(self) -> np.ndarray:
        """Check whether each cell has a detected place field.

        Returns:
            np.ndarray: Boolean array indicating if each cell has a place field.
        """
        return np.any(self.label_im > 0, axis=1)

    @property
    def order(self) -> np.ndarray:
        """Get cell ordering based on place field centers.

        For cells with multiple place fields, the field with the highest mean intensity is used for
        ordering.

        Returns:
            np.ndarray: Indices to sort cells by place field position.
        """
        num_cells = self.binF.shape[0]
        order = np.full(num_cells, np.inf)
        intensity = self.mean_intensity
        centers = self.centers

        if not centers.any():
            return order

        cell_id = centers[:, 0].astype(int)

        # In case a cell has two place fields, order on one with highest mean intensity
        for icell in range(num_cells):
            cell_ind = np.argwhere(cell_id == icell)
            if cell_ind.size > 0:
                ind = np.argmax(intensity[cell_id == icell])
                order[icell] = centers[cell_ind[ind], 1]

        return np.argsort(order)

    def remove_fields(self, ind: np.ndarray) -> PlaceFields1d:
        """Remove specified place fields.

        Args:
            ind (np.ndarray): Indices of place fields to remove.

        Returns:
            PlaceFields1d: New PlaceFields1d object with the specified fields removed.
        """
        pf = deepcopy(self)

        # Remove from label image and renumber
        pf.label_im[np.isin(pf.label_im, ind + 1)] = 0
        for counter, value in enumerate(np.unique(pf.label_im)):
            if value != 0:
                pf.label_im[pf.label_im == value] = counter

        # Remove from centers
        pf.centers = np.delete(pf.centers, ind, axis=0)

        return pf

    def filter_cells(self, ind: np.ndarray) -> PlaceFields1d:
        """Filter to keep only specified cells.

        Args:
            ind (np.ndarray): Indices of cells to keep.

        Returns:
            PlaceFields1d: New PlaceFields1d object containing only the specified cells.
        """
        pf = deepcopy(self)

        # Remove from label image and renumber
        list_ind = np.arange(0, pf.label_im.shape[0])
        cell_id = pf.cell_id
        pf.label_im[~np.isin(list_ind, ind), :] = 0

        for counter, value in enumerate(np.unique(pf.label_im)):
            if value != 0:
                pf.label_im[pf.label_im == value] = counter

        pf.centers = pf.centers[np.isin(cell_id, ind), :]

        return pf

    def plot(
        self,
        color_bar: bool = True,
        title: str | None = None,
        sort: bool = True,
        cells: np.ndarray | None = None,
        dpi: int = 150,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot place field activity as a heatmap.

        Args:
            color_bar (bool, optional): Whether to display a color bar. Defaults to True.
            title (str, optional): Title for the plot. Defaults to None.
            sort (bool, optional): Whether to sort cells by place field position. Defaults to True.
            cells (np.ndarray, optional): Specific cells to plot. Defaults to None.
            dpi (int, optional): Figure DPI. Defaults to 150.
            vmin (float, optional): Minimum value for color scaling. Defaults to None.
            vmax (float, optional): Maximum value for color scaling. Defaults to None.
            **kwargs: Additional keyword arguments to pass to imshow.

        Returns:
            plt.Figure: Matplotlib figure object containing the heatmap.
        """
        # Format data
        data = self.binF

        # Sort
        order = self.order if sort else np.arange(0, data.shape[0])

        # Select specific cells
        if cells is not None:
            order = order[np.isin(order, np.argwhere(cells))]

        data = data[order, :]

        # Calculate image range
        if vmin is None:
            vmin = np.nanquantile(data, 0.5)
        if vmax is None:
            vmax = np.nanquantile(data, 0.9)

        # Setup figure
        fig, axes = plt.subplots(1, 1, figsize=(2, 3), facecolor="white", dpi=dpi)

        if title:
            plt.title(title, fontsize=8)

        # Plot heatmap
        extent = [0, self.bin_size * data.shape[1], 1, data.shape[0] + 1]
        plt.imshow(data, cmap=cc.cm.CET_CBL2, extent=extent, interpolation="none", vmin=vmin, vmax=vmax, **kwargs)

        axes.set_aspect("auto")
        plt.xlabel("Position (cm)")
        plt.ylabel("Cell #")

        # Colorbar
        if color_bar:
            cbar = plt.colorbar()

        return fig


class PlaceFields1dProtocol:
    """Abstract base class for 1D place field detection protocols.

    This class defines the interface for different place field detection methods.
    Specific implementations should inherit from this class and implement the required methods.

    Attributes:
        params (Params): Protocol-specific parameters.
    """

    class Params:
        """Base parameter class for place field detection."""

        value = 0

    params = Params()

    def detect(
        self, F: np.ndarray, pos: np.ndarray, speed: np.ndarray, track_length: float, bin_size: float
    ) -> PlaceFields1d:
        """Detect place fields from fluorescence and position data.

        Args:
            F (np.ndarray): Fluorescence data (size: num_cells x num_timepoints).
            pos (np.ndarray): Position data (size: num_timepoints).
            speed (np.ndarray): Speed data (size: num_timepoints).
            track_length (float): Length of the track.
            bin_size (float): Size of spatial bins.

        Returns:
            PlaceFields1d: Detected place fields.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def validate(
        self,
        client: Any,
        num_repeats: int,
        F: np.ndarray,
        pos: np.ndarray,
        speed: np.ndarray,
        track_length: float,
        bin_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate detected place fields using a shuffle test.

        Args:
            client (Any): Dask client for parallel computing.
            num_repeats (int): Number of shuffles to perform.
            F (np.ndarray): Fluorescence data (size: num_cells x num_timepoints).
            pos (np.ndarray): Position data (size: num_timepoints).
            speed (np.ndarray): Speed data (size: num_timepoints).
            track_length (float): Length of the track.
            bin_size (float): Size of spatial bins.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing (significant_cell_indices, p_values).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class Tank1dProtocol(PlaceFields1dProtocol):
    """Tank implementation of 1D place field detection protocol.

    This implementation detects place fields using thresholding and connected component analysis,
    with validation performed via shuffle tests.

    Attributes:
        params (Params): Protocol-specific parameters for field detection.
    """

    class Params:
        """Parameters for Tank 1D place field detection."""

        min_speed = 5
        smooth_size = 3
        base_quantile = 0.25
        signal_threshold = 0.25
        min_bins = 3
        outside_threshold = 3
        max_int_threshold = 0.1
        num_chunks = 100
        sig_threshold = 0.05

    params = Params()

    def detect(
        self,
        F: np.ndarray,
        pos: np.ndarray,
        speed: np.ndarray,
        track_length: float,
        bin_size: float,
        calc_df: bool = True,
    ) -> PlaceFields1d:
        """Detect place fields using the Tank protocol.

        Args:
            F (np.ndarray): Fluorescence data (size: num_cells x num_timepoints).
            pos (np.ndarray): Position data (size: num_timepoints).
            speed (np.ndarray): Speed data (size: num_timepoints).
            track_length (float): Length of the track.
            bin_size (float): Size of spatial bins.
            calc_df (bool, optional): Whether to calculate dF/F₀. Defaults to True.

        Returns:
            PlaceFields1d: Detected place fields.
        """
        # Calculate Delta F over F zero
        if calc_df:
            F = vr2p.signal_processing.df_over_f0(F)

        # Filter for speed
        ind = speed > self.params.min_speed
        pos = pos.loc[ind]
        F = F[:, ind]

        # Average bin fluorescent data
        edges = np.arange(0, track_length + bin_size, bin_size)
        binF, _ = vr2p.signal_processing.bin_fluorescence_data(F, pos, edges)

        # Smooth with a moving average filter
        binF = convolve2d(
            binF, np.ones((1, self.params.smooth_size)) / self.params.smooth_size, mode="same", boundary="wrap"
        )

        # Threshold based on quantile
        thres_binF = vr2p.signal_processing.quantile_max_treshold(
            binF, self.params.base_quantile, self.params.signal_threshold
        )

        # Detect circular place fields
        pf = circular_connected_placefields(thres_binF, binF, self.params.min_bins)
        pf.bin_size = bin_size

        # Filter based on outside field threshold
        pf = outside_field_threshold(pf, self.params.outside_threshold)

        # Filter based on maximum intensity
        pf = pf.remove_fields(np.argwhere(pf.max_intensity < self.params.max_int_threshold))

        return pf

    def validate(
        self,
        client: Any,
        num_repeats: int,
        F: np.ndarray,
        pos: np.ndarray,
        speed: np.ndarray,
        track_length: float,
        bin_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate place fields using a shuffle test.

        This method performs a shuffle test to identify significant place fields.
        It compares observed place fields with those detected in shuffled data to compute p-values.

        Args:
            client (Any): Dask client for parallel computing.
            num_repeats (int): Number of shuffles to perform.
            F (np.ndarray): Fluorescence data (size: num_cells x num_timepoints).
            pos (np.ndarray): Position data (size: num_timepoints).
            speed (np.ndarray): Speed data (size: num_timepoints).
            track_length (float): Length of the track.
            bin_size (float): Size of spatial bins.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (significant_cell_indices, p_values).
        """
        # Calculate Delta F over F zero
        F = vr2p.signal_processing.df_over_f0(F)

        # Convert to Dask array for parallel processing
        F = da.from_array(F)

        # Convert speed to numpy array and handle NaNs
        speed = speed.to_numpy()
        speed[np.isnan(speed)] = 0

        # Calculate original place fields
        results = [dask.delayed(self.detect)(F, pos, speed, track_length, bin_size, calc_df=False).has_place_field]

        # Calculate place fields for shuffled data
        for i in range(num_repeats):
            shuffled = self._shuffle(F, i, pos, speed, track_length, bin_size)
            res = dask.delayed(self.detect)(shuffled, pos, speed, track_length, bin_size, calc_df=False).has_place_field
            results.append(res)

        # Compute all results in parallel
        results = dask.compute(results)
        results = np.vstack(results).T

        observed = results[:, 0]  # Observed original data
        results = results[:, 1:]  # Shuffle data

        # Calculate p-values as proportion of shuffles with place fields
        p = np.sum(results, axis=1) / results.shape[1]

        # Identify significant cells
        sig_cells = np.argwhere((observed) & (p < self.params.sig_threshold)).flatten()

        return sig_cells, p

    @dask.delayed
    def _shuffle(
        self, data: np.ndarray, i: int, pos: np.ndarray, speed: np.ndarray, track_length: float, bin_size: float
    ) -> np.ndarray:
        """Shuffle fluorescence data for validation.

        Args:
            data (np.ndarray): Fluorescence data to be shuffled.
            i (int): Shuffle iteration (used as random seed).
            pos (np.ndarray): Position data.
            speed (np.ndarray): Speed data.
            track_length (float): Length of the track.
            bin_size (float): Size of spatial bins.

        Returns:
            np.ndarray: Shuffled fluorescence data.
        """
        data_shuffle = np.array_split(data, self.params.num_chunks, axis=1)
        np.random.seed(i)
        shuffle_ind = np.random.choice(np.arange(self.params.num_chunks), self.params.num_chunks, replace=False)
        data_shuffle = np.concatenate([data_shuffle[i] for i in shuffle_ind], axis=1)

        return data_shuffle


class DetectPlaceFields1d:
    """Main class for detecting and validating 1D place fields.

    This class provides a simplified interface for users to detect and validate
    place fields using the specified protocol.

    Args:
        F (np.ndarray): Fluorescence data (size: num_cells x num_timepoints).
        pos (np.ndarray): Position data (size: num_timepoints).
        speed (np.ndarray): Speed data (size: num_timepoints).
        track_length (float): Length of the track.
        bin_size (float): Size of spatial bins.
        protocol (type, optional): Place field detection protocol class. Defaults to Tank1dProtocol.

    Attributes:
        F (np.ndarray): Fluorescence data.
        pos (np.ndarray): Position data.
        speed (np.ndarray): Speed data.
        track_length (float): Length of the track.
        bin_size (float): Size of spatial bins.
        protocol (PlaceFields1dProtocol): Instance of the place field detection protocol.
    """

    def __init__(
        self,
        F: np.ndarray,
        pos: np.ndarray,
        speed: np.ndarray,
        track_length: float,
        bin_size: float,
        protocol: type = Tank1dProtocol,
    ):
        self.F = F
        self.pos = pos
        self.speed = speed
        self.track_length = track_length
        self.bin_size = bin_size
        self.protocol = protocol()

    def run(self) -> PlaceFields1d:
        """Run place field detection.

        Returns:
            PlaceFields1d: Detected place fields.
        """
        return self.protocol.detect(self.F, self.pos, self.speed, self.track_length, self.bin_size)

    def validate(self, client: Any, num_repeats: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Validate place fields using a shuffle test.

        Args:
            client (Any): Dask client for parallel computing.
            num_repeats (int): Number of shuffles to perform.
            **kwargs: Additional arguments to pass to the protocol's validate method.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (significant_cell_indices, p_values).
        """
        return self.protocol.validate(
            client, num_repeats, self.F, self.pos, self.speed, self.track_length, self.bin_size, **kwargs
        )


def circular_connected_placefields(thres_im: np.ndarray, binF: np.ndarray, min_bins: int = 3) -> PlaceFields1d:
    """Create a labeled image of circularly connected regions within each cell/row.

    This function takes a thresholded binary image and identifies connected regions that may wrap
    around the track (circular connectivity).

    Args:
        thres_im (np.ndarray): Thresholded binary image of binned place field activity
            (size: num_cells x num_bins).
        binF (np.ndarray): Binned fluorescence data (size: num_cells x num_bins).
        min_bins (int, optional): Minimal required size of a connected region. Defaults to 3.

    Returns:
        PlaceFields1d: Detected one-dimensional place fields.

    Notes:
        The function handles circular connectivity by padding the input arrays and detecting
        connected components.
    """
    num_bins = thres_im.shape[1]

    # Make circular by padding
    pad_thres = np.pad(thres_im, ((0, 0), (num_bins, num_bins)), mode="wrap")
    F_padded = np.pad(binF, ((0, 0), (num_bins, num_bins)), mode="wrap")

    # Detect connected components
    label_im, _ = label(pad_thres, [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    props = np.array(regionprops(label_im, F_padded, cache=False))

    # Get region centers and area
    centers = np.array([prop["weighted_centroid"] for prop in props])
    area = np.array([prop["area"] for prop in props], np.uint32)

    # Select components with center within original area (for circularity)
    # and minimum area size
    ind = (centers[:, 1] >= num_bins) & (centers[:, 1] < num_bins * 2) & (area >= min_bins)

    # Store result
    result_label_im = np.zeros(thres_im.shape, np.uint32)
    adj_centers = []

    for counter, prop in enumerate(props[ind]):
        coords = prop["coords"]
        new_ind = np.take(np.arange(0, num_bins), coords[:, 1], mode="wrap")
        result_label_im[coords[:, 0], new_ind] = counter + 1

        # Get center
        center = np.array(prop["weighted_centroid"])
        center[1] -= num_bins
        adj_centers.append(center)

    return PlaceFields1d(result_label_im, binF, np.vstack(adj_centers))


def outside_field_threshold(pf: PlaceFields1d, threshold_factor: float = 3) -> PlaceFields1d:
    """Filter place fields based on the signal-to-baseline ratio.

    This function excludes detected place field regions if the inside field values are below
    (threshold_factor) times the outside field values. In cases where a cell has multiple fields,
    both fields are excluded from the outside field calculation (one outside field per cell).

    Args:
        pf (PlaceFields1d): PlaceFields1d object with previously detected place fields.
        threshold_factor (float, optional): Scalar factor of the outside field signal to set the
            threshold. Defaults to 3.

    Returns:
        PlaceFields1d: Filtered PlaceFields1d object.

    Notes:
        This function is intended to remove false positives by requiring that detected place fields
        have significantly higher activity than the baseline outside the field.
    """
    # Get outside field values
    outside_im = pf.binF.copy()
    outside_im[pf.label_im != 0] = np.nan
    outside_values = np.nanmean(outside_im, axis=1)

    # Get threshold for each region
    threshold_values = outside_values[pf.cell_id] * threshold_factor
    invalid_regions = np.concatenate(np.argwhere(pf.mean_intensity < threshold_values))

    # Remove invalid regions
    return pf.remove_fields(invalid_regions)
