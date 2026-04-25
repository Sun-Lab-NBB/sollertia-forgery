from pathlib import Path
import polars as pl

dataset = Path(
    "/home/cyberaxolotl/Desktop/test/2025-08-15-12-00-55-872035/processed_data/behavior_data/runtime_state_data.feather"
)
target = pl.read_ipc(dataset, memory_map=True)
with pl.Config(
    set_fmt_table_cell_list_len=1,
    set_float_precision=2,
    set_tbl_cols=50,
    set_tbl_rows=42000,
    set_tbl_width_chars=300,
    set_tbl_formatting="ASCII_FULL_CONDENSED",
    set_tbl_hide_column_data_types=True,
):
    print(target)
