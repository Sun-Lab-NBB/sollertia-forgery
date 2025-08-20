from sl_shared_assets import ProjectManifest, generate_project_manifest, SessionData
from sl_forgery.analysis.structured_dataclass import ProjectData
from sl_forgery.analysis.plotting import Plotting

from pathlib import Path

filter_path = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\sl-forgery\src\sl_forgery\analysis\tm6_filter.yaml")
manifest_path = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\slf_data\TM_06_pilot_manifest.feather")
work_dir = Path(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\slf_data")

data = ProjectData.create(project_name="TM_06_pilot", manifest_path=manifest_path, filter_path=filter_path, working_directory=work_dir)

# Plotting.plot_session("single_day", 4, data.get_session("2025-06-23-13-32-06-980761"))

Plotting.plot_umap("single_day", data.get_session("2025-06-23-13-32-06-980761")).write_html(r"C:\Users\jacob\OneDrive\Desktop\PlaceFields\sl-forgery\src\sl_forgery\analysis\umap.html")