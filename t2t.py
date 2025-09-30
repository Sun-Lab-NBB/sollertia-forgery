from pathlib import Path

import polars as pl

from sl_forgery.forging import DatasetTypes, assemble_session_dataset

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

session_root = Path("/mnt/data/data_checks/Data")

for session in session_root.glob("*"):
    session_name = session.name
    dataset = Path("/mnt/data/data_checks/Dataset/26").joinpath(session_name)
    assemble_session_dataset(
        session_data_path=session,
        session_multiday_path=dataset,
        output_path=session.joinpath(f"/mnt/data/data_checks/{session_name}.feather"),
        dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT,
    )

# session = Path("/mnt/data/data_checks/Data/2025-08-27-17-18-55-361099")
# dataset = Path("/mnt/data/data_checks/Dataset/26/2025-08-27-17-18-55-361099")
#
# assemble_session_dataset(
#     session_data_path=session,
#     session_multiday_path=dataset,
#     output_path=session.joinpath("/mnt/data/data_checks/test.feather"),
#     dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT,
# )
#
# target = pl.read_ipc(session.joinpath("/mnt/data/data_checks/test.feather"), memory_map=True, use_pyarrow=True)
# with pl.Config(
#     set_fmt_table_cell_list_len=1,
#     set_float_precision=2,
#     set_tbl_cols=50,
#     set_tbl_rows=42000,
#     set_tbl_width_chars=300,
# ):
#     print(target.slice(offset=30000, length=5000))

# target = session.joinpath("processed_data", "behavior_data", "experiment_state_data.feather")
# with pl.Config(set_fmt_table_cell_list_len=5, set_tbl_cols=10, set_tbl_rows=100):
#     print(pl.read_ipc(target, use_pyarrow=True, memory_map=True))
