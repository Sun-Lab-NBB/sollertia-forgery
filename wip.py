from pathlib import Path

from ataraxis_time import PrecisionTimer
from sl_shared_assets import Server, ProjectManifest
from ataraxis_base_utilities import console

from sl_forgery.utils import get_working_directory, set_working_directory, get_credentials_file_path
from sl_forgery.processing.data_processing import submit_behavior_processing_job
from sl_forgery.processing.project_management import fetch_remote_project_manifest, generate_remote_project_manifest

credentials = get_credentials_file_path(require_service=True)
server = Server(credentials_path=credentials)

generate_remote_project_manifest(project="MaalstroomicFlow")

job = submit_behavior_processing_job(
    project="MaalstroomicFlow",
    session="2025-07-13-19-08-43-998260",
    server=server,
    reprocess=False,
    legacy=False,
    keep_job_logs=False,
)

console.echo("Waiting for the behavior processing job to complete...")
timer = PrecisionTimer("s")
while not server.job_complete(job):
    timer.delay_noblock(delay=10, allow_sleep=True)

fetch_remote_project_manifest(project="MaalstroomicFlow")

target = get_working_directory().joinpath("MaalstroomicFlow", "manifest.feather")
manifest = ProjectManifest(manifest_file=target)
manifest.print_summary(15)
