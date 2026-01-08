from pathlib import Path
import polars as pl

dataset = Path("/home/cyberaxolotl/server/workdir/sun_data/Datasets/SSOData/26/2025-08-21-16-35-48-370149.feather")
target = pl.read_ipc(dataset, memory_map=True, use_pyarrow=True)
with pl.Config(
    set_fmt_table_cell_list_len=1,
    set_float_precision=2,
    set_tbl_cols=50,
    set_tbl_rows=42000,
    set_tbl_width_chars=300,
    set_tbl_formatting="ASCII_FULL_CONDENSED",
    set_tbl_hide_column_data_types=True,
):
    print(target.slice(offset=30500, length=100))
