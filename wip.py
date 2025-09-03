from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    update_manifest=False,
    force_lock=True,
    reprocess=True,
    process_checksum=True,
    recalculate_checksum=True,
    prepare_sessions=True,
    reset_trackers=True,
)
