from sl_forgery.processing import process_project_data, fetch_remote_project_manifest

target_sessions = ["2025-07-29-12-49-17-013336", "2025-07-30-12-14-37-233207"]

process_project_data(project="MaalstroomicFlow", sessions=target_sessions, update_manifest=False, reprocess_behavior=True)

# fetch_remote_project_manifest(project="MaalstroomicFlow")
#
# target = get_working_directory().joinpath("MaalstroomicFlow", "manifest.feather")
# manifest = ProjectManifest(manifest_file=target)
# manifest.print_summary(11)
