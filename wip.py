from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    sessions=("2025-08-15-12-00-55-872035",),
    update_manifest=False,
    force_lock=True,
    reprocess=True,
    process_checksum=False,
    recalculate_checksum=False,
    prepare_sessions=False,
    process_behavior=True,
    reset_trackers=True,
)
