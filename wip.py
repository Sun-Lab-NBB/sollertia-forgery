from ataraxis_time import PrecisionTimer
from sl_shared_assets import Server, ProjectManifest
from ataraxis_base_utilities import console

from sl_forgery.utils import get_working_directory, get_credentials_file_path
from sl_forgery.processing.data_processing import verify_processing_job_outcome, submit_behavior_processing_job
from sl_forgery.processing.project_management import fetch_remote_project_manifest, generate_remote_project_manifest

credentials = get_credentials_file_path(require_service=True)
server = Server(credentials_path=credentials)

generate_remote_project_manifest(project="MaalstroomicFlow")

job_data = submit_behavior_processing_job(
    project="MaalstroomicFlow",
    session="2025-07-29-12-49-17-013336",
    server=server,
    reprocess=True,
    legacy=False,
    keep_job_logs=False,
)

if job_data is not None:
    console.echo("Waiting for the behavior processing job to complete...")
    timer = PrecisionTimer("s")
    while verify_processing_job_outcome(job_data=job_data) is None:
        timer.delay_noblock(delay=10, allow_sleep=True)

fetch_remote_project_manifest(project="MaalstroomicFlow")

target = get_working_directory().joinpath("MaalstroomicFlow", "manifest.feather")
manifest = ProjectManifest(manifest_file=target)
manifest.print_summary(15)
