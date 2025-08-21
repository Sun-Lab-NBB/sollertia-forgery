from sl_shared_assets import ProjectManifest, generate_project_manifest, SessionData
from sl_forgery.utils.dataclass import ProjectData, TargetGroup
from sl_forgery.analysis.plotting import Plotting
from sl_forgery.analysis.processing import Processing


from pathlib import Path

filter_path = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\sl-forgery\src\sl_forgery\analysis\tm6_filter.yaml")
manifest_path = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\slf_data\TM_06_pilot_manifest.feather")
work_dir = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\slf_data")

data = ProjectData.create(project_name="TM_06_pilot", manifest_path=manifest_path, filter_path=filter_path, working_directory=work_dir)

# Plotting.plot_session("single_day", 4, data.get_session("2025-06-23-13-32-06-980761"))
# fig = Plotting.plot_umap("single_day", data.get_session("2025-06-23-13-32-06-980761"))

Plotting.plot_all_single_session_umaps(target_group=TargetGroup.SINGLE_DAY, animal=data.get_mouse(6))

# for session_name in data.manifest.get_sessions(animal=6):
#     Processing.compute_single_session_umap("single_day", data.get_session(session_name))


# fig.write_html(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\sl-forgery\src\sl_forgery\analysis\umap.html")
