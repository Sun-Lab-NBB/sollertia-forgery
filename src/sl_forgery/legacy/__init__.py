"""VR2P: A library for analyzing virtual reality experimental data with
longitudinal two-photon imaging.
"""

import os
from typing import Any, Union
from pathlib import Path

import s3fs
import vr2p
import zarr
import gcsfs
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ExperimentData:
    """Container for all experimental data from an animal.

    This class organizes and provides access to virtual reality data,
    imaging signals, cell masks, and metadata across multiple sessions.

    Args:
        file (str | Path): Path to the data directory or cloud storage location.

    Attributes:
        behavior (BehaviorCollection):
            Collection of virtual reality session data, accessible by type
            (e.g., self.behavior.lick) and then by session index, returning a DataFrame.
        signals (SignalCollection): Neural activity signals (F, Fneu, Fns, Fdemix, spks).
        images (ImageCollection): Original and registered images.
        cells (CellCollection): Cell mask information for both single and multi-session data.
        meta (dict): Processing settings and animal metadata.
        sessions_data (SessionData): Session data for each session.

    Notes:
        Data organization structure:
        - self.behavior.lick[session]: Lick behavioral data (DataFrame). Similar for other types like 'frame', 'brake'.
        - self.signals.single_session.F[session][cell, frame]: Within-session cell signals.
        - self.signals.multi_session.F[session][cell, frame]: Registered cell signals.
        - self.images.original[session][key]: Original, non-registered images.
        - self.images.registered[session][key]: Registered, transformed images.
        - self.cells.single_session[session][cell][key]: Single-session cell mask info.
        - self.cells.multi_session.original[session][cell][key]: Original coordinates.
        - self.cells.multi_session.registered[session][cell][key]: Transformed coordinates.
    """

    def __init__(self, file: str | Path) -> None:
        # Check if cloud path
        is_gcp: bool = isinstance(file, str) and file.lower().startswith("gs://")
        is_aws: bool = isinstance(file, str) and file.lower().startswith("s3://")

        if not (is_gcp or is_aws):
            file = Path(file)
            if not file.is_dir():
                raise FileNotFoundError(f"Could not find directory: {file}")
            file = zarr.open(str(file), mode="r")
        elif is_gcp:
            gcsmap = gcsfs.mapping.GCSMap(file)
            file = zarr.open(gcsmap, mode="r")
        elif is_aws:
            s3map = s3fs.S3Map(file)
            file = zarr.open(s3map, mode="r")

        self._file = file
        self.behavior = BehaviorCollection(self._file, "behavior")
        self.signals = SignalCollection(file)
        self.images = ImageCollection(file)
        self.cells = CellCollection(file)
        self.meta = file["meta"][()]
        self.sessions_data = file["sessions_data"][()]


class SignalData:
    """Handles direct access to signal data arrays.

    Args:
        file (zarr.Group): Zarr group containing signal data.
        field (str): Complete path to specific signal data.
    """

    def __init__(self, file: zarr.Group, field: str) -> None:
        self._file = file
        self._field = field

    def __getitem__(self, indices: int | tuple[int, int]) -> Any:
        return self._file[f"{self._field}"].oindex[indices]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file[f"{self._field}"], name)

    @property
    def array(self) -> NDArray:
        """Get the full underlying array."""
        return self._file[f"{self._field}"]


class SessionSignalData:
    """Provides access to signal data for specific sessions.

    Args:
        file (zarr.Group): Zarr group containing signal data.
        field (str): Path to signal data within the zarr file.
    """

    def __init__(self, file: zarr.Group, field: str) -> None:
        self._file = file
        self._field = field

    def __getitem__(self, index: int) -> SignalData:
        if not isinstance(index, int):
            msg = "vr2p: Can only access one session at a time (Do you remember the session index?)"
            raise ValueError(msg)

        return SignalData(self._file, f"{self._field}/{index}")

    def __len__(self) -> int:
        return len(self._file[f"{self._field}"])


class SignalCollectionData:
    """Container for session-related neural signal data.

    Holds all signal types (raw, neuropil, etc.) for either
    single-session or multi-session aligned data.

    Args:
        file (zarr.Group): Zarr group containing signal data.
        field (str): Either "single_session" or "multi_session" to specify data type.

    Attributes:
        F (SessionSignalData): Raw cell fluorescence signal.
        Fneu (SessionSignalData): Raw neuropil fluorescence signal.
        Fns (SessionSignalData): Neuropil-subtracted fluorescence signal.
        Fdemix (SessionSignalData): Demixed, neuropil and baseline subtracted signal.
        spks (SessionSignalData): Inferred spike signal (based on Fdemix).
    """

    def __init__(self, file: zarr.Group, field: str) -> None:
        self._file = file
        self._field = field
        self.F = SessionSignalData(self._file, f"{self._field}/F")
        self.Fneu = SessionSignalData(self._file, f"{self._field}/Fneu")
        self.Fns = SessionSignalData(self._file, f"{self._field}/Fns")
        self.Fdemix = SessionSignalData(self._file, f"{self._field}/Fdemix")
        self.spks = SessionSignalData(self._file, f"{self._field}/spks")


class SignalCollection:
    """Collection of neural activity signals across sessions.

    Args:
        file (zarr.Group): Zarr group containing signal data.

    Attributes:
        single_session (SignalCollectionData): Single session data without alignment.
        multi_session (SignalCollectionData): Multi-session data with alignment between sessions.
    """

    def __init__(self, file: zarr.Group) -> None:
        self._file = file
        self.single_session = SignalCollectionData(file, "single_session")
        self.multi_session = SignalCollectionData(file, "multi_session")


class LogSessionData:
    """Handles access to pickled session information from a zarr file.

    Args:
        file (zarr.Group): Zarr group containing log data.
        field (str): Path to log data within the zarr file.
    """

    def __init__(self, file: zarr.Group, field: str) -> None:
        self._file = file
        self._field = field

    def __getitem__(self, index: int) -> object:
        if not isinstance(index, int):
            raise ValueError("can only access one session at a time")
        return self._file[f"{self._field}/{index}"][()].value


class ObjectSessionData:
    """Handles access to pickled session objects from a zarr file.

    Provides iteration capabilities over sessions.

    Args:
        file (zarr.Group): Zarr group containing session data.
        field (str): Path to session data within the zarr file.
    """

    def __init__(self, file: zarr.Group, field: str) -> None:
        self._file = file
        self._field = field
        self.index = -1

    def __getitem__(self, index: int | np.int32) -> Any:
        if not (isinstance(index, int) or isinstance(index, np.int32)):
            raise ValueError("can only access one session at a time")
        return self._file[f"{self._field}/{index}"][()]

    def __len__(self) -> int:
        return self._file[f"{self._field}"].__len__()

    def __iter__(self) -> "ObjectSessionData":
        self.index = -1
        return self

    def __next__(self) -> Any:
        self.index += 1
        # Stop iteration if limit is reached
        if self.index == len(self):
            raise StopIteration
        return self.__getitem__(self.index)


class ImageCollection:
    """Contains original and registered imaging data.

    Args:
        file (zarr.Group): Zarr group containing image data.

    Attributes:
        original (ObjectSessionData): Original, non-registered images.
        registered (ObjectSessionData): Registered, transformed images.
    """

    def __init__(self, file: zarr.Group) -> None:
        self._file = file
        self.original = ObjectSessionData(self._file, "images/original")
        self.registered = ObjectSessionData(self._file, "images/registered")


class RegisteredCells:
    """Provides access to registered cell data with iteration capability.

    Args:
        file (zarr.Group): Zarr group containing registered cell data.
    """

    def __init__(self, file: zarr.Group) -> None:
        self._file = file
        self._index = -1
        self._values = None

    def __getitem__(self, indices: Any) -> Any:
        return self._file["cells/multi_session/registered"][indices]

    def __len__(self) -> int:
        return self._file["cells/multi_session/registered"].__len__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file["cells/multi_session/registered"], name)

    def __iter__(
        self,
    ) -> "RegisteredCells":
        self._index = -1
        self._values = self._file["cells/multi_session/registered"][()]
        return self

    def __next__(self) -> Any:
        self._index += 1
        # Stop iteration if limit is reached
        if self._index == len(self._values):
            raise StopIteration
        return self._values[self._index]


class MultiSessionCells:
    """Container for multi-session cell data in original and registered formats.

    Args:
        file (zarr.Group): Zarr group containing multi-session cell data.

    Attributes:
        original (ObjectSessionData): Cell masks in original coordinates.
        registered (RegisteredCells): Cell masks in registered, transformed coordinates.
    """

    def __init__(self, file: zarr.Group) -> None:
        self._file = file
        self.original = ObjectSessionData(self._file, "cells/multi_session/original")
        self.registered = RegisteredCells(file)


class CellCollection:
    """Contains cell masks from single and multi-session experiments.

    Args:
        file (zarr.Group): Zarr group containing cell data.

    Attributes:
        single_session (ObjectSessionData): Cell masks from single-session data.
        multi_session (MultiSessionCells): Cell masks from multi-session data in both original and registered coordinates.
    """

    def __init__(self, file: zarr.Group) -> None:
        self._file = file
        self.single_session = ObjectSessionData(self._file, "cells/single_session")
        self.multi_session = MultiSessionCells(file)


class BehaviorTypeSessionData:
    """Handles access to a specific type of behavior data (e.g., lick, frame)
    across multiple sessions stored in a Zarr file.
    Data for each session is expected to be a Zarr dataset containing
    a NumPy structured array (from pandas DataFrame.to_records()),
    which is converted back to a pandas DataFrame upon access.

    Args:
        file (zarr.Group): The root Zarr group.
        behavior_type_path (str): The path within the Zarr group to this
                                  specific behavior type's data
                                  (e.g., "behavior/lick").
    """

    def __init__(self, file: zarr.Group, behavior_type_path: str):
        self._file = file
        self._behavior_type_path = behavior_type_path

    def __getitem__(self, session_index: int) -> pd.DataFrame:
        """Retrieves the behavior data for a given session index as a pandas DataFrame.

        Args:
            session_index (int): The index of the session.

        Returns:
            pd.DataFrame: The behavior data for the specified session.
                          Returns an empty DataFrame if data is not found.
        """
        if not isinstance(session_index, int):
            raise ValueError("Session index must be an integer.")

        dataset_path = f"{self._behavior_type_path}/{session_index}"
        if dataset_path in self._file:
            structured_array = self._file[dataset_path][()]
            return pd.DataFrame.from_records(structured_array)
        else:
            # Return an empty DataFrame if data for the session_index is not found
            return pd.DataFrame()

    def __len__(self) -> int:
        """Returns the number of sessions for which this behavior data type is available."""
        if self._behavior_type_path in self._file:
            return len(self._file[self._behavior_type_path])
        return 0

    def __repr__(self) -> str:
        return f"<BehaviorTypeSessionData for '{self._behavior_type_path}', {len(self)} sessions>"


class BehaviorCollection:
    """Provides attribute-based access to different types of behavior data
    (e.g., lick, frame, brake), each managed by BehaviorTypeSessionData.

    Args:
        file (zarr.Group): The root Zarr group.
        root_behavior_path (str): The base path in the Zarr file where all
                                  behavior type sub-groups are located
                                  (e.g., "behavior").
    """

    def __init__(self, file: zarr.Group, root_behavior_path: str):
        self._file = file
        self._root_behavior_path = root_behavior_path
        self._behavior_types = []

        if self._root_behavior_path in self._file:
            behavior_group = self._file[self._root_behavior_path]
            for behavior_key in behavior_group.keys():
                # behavior_key is e.g., "lick", "frame"
                type_path = f"{self._root_behavior_path}/{behavior_key}"
                setattr(self, behavior_key, BehaviorTypeSessionData(self._file, type_path))
                self._behavior_types.append(behavior_key)
        # If root_behavior_path doesn't exist, no attributes will be set,
        # and accessing them will naturally raise an AttributeError.

    def __dir__(self):
        # Enhance introspection (e.g., for tab completion)
        return sorted(super().__dir__() + self._behavior_types)

    def __repr__(self) -> str:
        return f"<BehaviorCollection at '{self._root_behavior_path}', types: {self._behavior_types}>"
