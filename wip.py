from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    sessions=("2025-08-15-12-00-55-872035",),
    update_manifest=True,
    force_lock=False,
    process_checksum=True,
    recalculate_checksum=True,
    reset_trackers=True,
)
