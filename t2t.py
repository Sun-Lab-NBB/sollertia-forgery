from pathlib import Path

from sl_forgery.dataset.data_assembly import assemble_session_data
from sl_forgery.dataset.dataset import SessionTypes

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
    session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
)

# target = session.joinpath("processed_data", "behavior_data", "encoder_data.feather")
# with pl.Config(set_fmt_table_cell_list_len=5, set_tbl_cols=10, set_tbl_rows=20):
#     print(pl.read_ipc(target, use_pyarrow=True, memory_map=True))
