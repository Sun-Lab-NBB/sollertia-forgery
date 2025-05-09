"""Utilities for setting up and managing Dask clusters.

This module provides functions to set up:
- A Dask cluster using the LSF scheduler (at BioHPC Cluster)
- A local Dask cluster (fallback if bsub is unavailable)
- A helper generator to iterate over Dask array blocks
"""

import os
from collections.abc import Generator
from pathlib import Path
from shutil import which
from typing import Any

import dask
import dask.array as da
import IPython
import numpy as np

try:
    from distributed import Client, LocalCluster
    from dask_jobqueue import LSFCluster
except ImportError:
    print("Warning: distributed and/or dask_jobqueue packages are not installed.")
    print("To install them, run: pip install distributed dask-jobqueue")

# Set default timeout configurations
dask.config.set({"jobqueue.lsf.use-stdin": True})
dask.config.set({"distributed.comm.timeouts.connect": 60})


def setup_client(account_name: str) -> Client:
    """Create and display a Dask client connected to the chosen cluster.

    This is necessary to ensure that workers get the job script from stdin.

    Args:
        account_name: Account or project name to be used for the job queue cluster setup.

    Returns:
        Client: A dask.distributed Client instance connected to the cluster.
    """
    client = get_cluster(project=account_name)
    IPython.display.display(client)
    return client


def get_jobqueue_cluster(
    walltime: str = "1:00",
    ncpus: int = 1,
    cores: int = 1,
    memory: str = "16GB",
    threads_per_worker: int = 1,
    **kwargs: Any,
) -> LSFCluster:
    """Instantiate a Dask job-queue cluster using the LSF scheduler.

    This function configures and returns a LSFCluster object with sensible
    defaults, suitable for the BioHPC compute cluster.

    Args:
        walltime: Maximum allowed job run time in HH:MM format. Default is "1:00".
        ncpus: Number of host CPUs available to each job. Default is 1.
        cores: Number of cores (or workers) to request from LSF. Default is 1.
        memory: Amount of memory to request for each job. Default is "16GB".
        threads_per_worker: Number of threads per Dask worker. Must currently be 1. Default is 1.
        **kwargs: Additional keyword arguments to pass to LSFCluster().

    Returns:
        LSFCluster: A configured Dask cluster on the LSF scheduler.

    Raises:
        ValueError: If threads_per_worker is not 1.
    """
    if threads_per_worker != 1:
        raise ValueError("threads_per_worker can only be 1 at this time.")

    env_extra = [
        "export NUM_MKL_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export OPENMP_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
    ]

    user = os.environ["USER"]
    home = os.environ["HOME"]

    if "local_directory" not in kwargs:
        kwargs["local_directory"] = f"/workdir/{user}/"

    if "log_directory" not in kwargs:
        log_dir = f"{home}/.dask_distributed/"
        Path(log_dir).mkdir(parents=False, exist_ok=True)
        kwargs["log_directory"] = log_dir

    cluster = LSFCluster(
        queue="normal",
        walltime=walltime,
        cores=cores,
        ncpus=ncpus,
        memory=memory,
        env_extra=env_extra,
        **kwargs,
    )
    return cluster


def bsub_available() -> bool:
    """Check if the `bsub` command is available in PATH. `bsub` is used to check
    whether code is running on the LSF-based cluster.

    Returns:
        bool: True if 'bsub' is available, False otherwise.
    """
    return which("bsub") is not None


def get_cluster(**kwargs: Any) -> Client:
    """Create a dask.distributed Client on a job queue cluster (LSF) if available, or locally otherwise.

    Args:
        **kwargs: Keyword arguments for either get_jobqueue_cluster() or LocalCluster().

    Returns:
        Client: A dask.distributed client instance connected to the chosen cluster.
    """
    if bsub_available():
        cluster = get_jobqueue_cluster(**kwargs)
    else:
        # Default to local cluster setup
        if "host" not in kwargs:
            kwargs["host"] = ""
        cluster = LocalCluster(**kwargs)

    client = Client(cluster)
    return client


def blockwise(arr: da.Array) -> Generator[tuple[slice, da.Array], None, None]:
    """Yield slice and block pairs from a Dask array, chunk by chunk.

    Args:
        arr: The Dask array to iterate over.

    Yields:
        tuple: A tuple containing (slice, dask.array.Array) where:
            - slice: The slice describing the block location in the array
            - dask.array.Array: The corresponding block (a Dask sub-array)
    """
    for idx, slc in zip(
        np.ndindex(arr.numblocks),
        da.core.slices_from_chunks(arr.chunks),
        strict=False
    ):
        yield (slc, arr.blocks[idx])
