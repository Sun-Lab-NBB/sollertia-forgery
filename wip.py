from sl_forgery.processing import process_project_data

process_project_data(
    project="StateSpaceOdyssey",
    # sessions=("2025-07-14-13-49-04-018601",),
    update_manifest=True,
    force_lock=True,
    reprocess=True,
    process_checksum=False,
    recalculate_checksum=False,
    prepare_sessions=False,
    process_behavior=False,
    process_suite2p=True,
    reset_trackers=True,
)
