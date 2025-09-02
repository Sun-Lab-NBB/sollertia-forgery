from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    update_manifest=False,
    force_lock=False,
    reprocess=True,
    process_checksum=False,
    recalculate_checksum=False,
    prepare_sessions=True,
    reset_trackers=False,
)
