from pathlib import Path

import polars as pl

from sl_forgery.dataset.data_assembly import DatasetTypes, assemble_session_data

# od = Path("/home/cyberaxolotl/data/")
# # ProjectData.create(
# #     output_directory=od,
# #     dataset_name="test_data",
# #     project="MaalstroomicFlow",
# #     dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT
# # )
#
# manifest_p = Path("/home/cyberaxolotl/data/MaalstroomicFlow/MaalstroomicFlow_manifest.feather")
# p_manifest = ProjectManifest(manifest_file=manifest_p)
# data = ProjectData(dataset_path=od)
# data.forge(project_manifest=p_manifest)

session = Path("/home/data/2025-09-16-18-44-32-476061")
dataset = Path("/home/data/md_data/2025-09-16-18-44-32-476061")

assemble_session_data(
    session_data_path=session,
    session_multiday_path=dataset,
    output_path=session.joinpath("/home/data/test.feather"),
    dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT,
)

target = pl.read_ipc(session.joinpath("/home/data/test.feather"), memory_map=True, use_pyarrow=True)
with pl.Config(
    set_fmt_table_cell_list_len=1,
    set_float_precision=2,
    set_tbl_cols=50,
    set_tbl_rows=1000,
    set_tbl_width_chars=300,
):
    print(target.slice(offset=2800, length=500))

# target = session.joinpath("processed_data", "camera_data", "face_camera_timestamps.feather")
# with pl.Config(set_fmt_table_cell_list_len=5, set_tbl_cols=10, set_tbl_rows=20):
#     print(pl.read_ipc(target, use_pyarrow=True, memory_map=True))
