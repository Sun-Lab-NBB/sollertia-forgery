from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    update_manifest=True,
    force_lock=True,
    process_checksum=True,
    recalculate_checksum=True,
    reset_trackers=True,
)
