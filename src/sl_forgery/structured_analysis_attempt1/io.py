from pathlib import Path
import polars as pl



def parse_data(data):
    if isinstance(data, str):
        data = Path(data)
    if isinstance(data, Path):
        data = pl.read_ipc(data, use_pyarrow=True)

    if not isinstance(data, pl.DataFrame):
        raise Exception("data must be a string, pathlib.Path, or polars.DataFrame")

    return data
