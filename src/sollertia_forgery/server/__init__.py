"""Provides assets for interfacing with remote compute servers to manage data and run processing pipelines."""

from sollertia_shared_assets import DatasetTrackers, ManagingTrackers, ProcessingTrackers

from .job import Job, JupyterJob
from .server import Server, JobStatus, CommandResult, get_remote_job_work_directory
from .pipeline import ProcessingPipeline
from .managing_interface import adopt_project, manage_project_data, resolve_project_manifest
from .processing_interface import process_project_data
from .forging_interface import forge_dataset, generate_report_datasets

__all__ = [
    "CommandResult",
    "DatasetTrackers",
    "Job",
    "JobStatus",
    "JupyterJob",
    "ManagingTrackers",
    "ProcessingPipeline",
    "ProcessingTrackers",
    "Server",
    "adopt_project",
    "forge_dataset",
    "generate_report_datasets",
    "get_remote_job_work_directory",
    "manage_project_data",
    "process_project_data",
    "resolve_project_manifest",
]
