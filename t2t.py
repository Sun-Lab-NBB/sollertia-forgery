from pathlib import Path

from sl_forgery.dataset.dataset import ProjectData, DatasetTypes, ProjectManifest

od = Path("/home/cyberaxolotl/data/")
# ProjectData.create(
#     output_directory=od,
#     dataset_name="test_data",
#     project="MaalstroomicFlow",
#     dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT
# )

manifest_p = Path("/home/cyberaxolotl/data/MaalstroomicFlow/MaalstroomicFlow_manifest.feather")
p_manifest = ProjectManifest(manifest_file=manifest_p)
data = ProjectData(dataset_path=od)
data.forge(project_manifest=p_manifest)
