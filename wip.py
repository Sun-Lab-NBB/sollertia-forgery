from sl_forgery.processing import process_project_data

process_project_data(
    project="MaalstroomicFlow",
    animals=(11,),
    # sessions=("2025-08-15-12-00-55-872035",),
    update_manifest=False,
    force_lock=True,
)
