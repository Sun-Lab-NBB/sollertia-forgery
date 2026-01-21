from sl_forgery.processing import process_project_data

process_project_data(
    project="StateSpaceOdyssey",
    # sessions=("2025-08-13-16-50-55-726851", "2025-09-08-19-00-55-267485"),
    update_manifest=False,
    force_lock=False,
    reprocess=False,
    process_checksum=False,
    recalculate_checksum=False,
    prepare_sessions=False,
    process_behavior=True,
    process_suite2p=False,
    reset_trackers=False,
    processing_batch_size=10,
)
