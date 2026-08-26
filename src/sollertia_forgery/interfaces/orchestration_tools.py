"""Provides the Model Context Protocol (MCP) tools that enumerate and forget the batches this host has prepared, and
report the declared resource model against which their jobs are admitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ataraxis_base_utilities import resolve_worker_count

from ..video import ENERGY_JOB_NAME, RENAME_JOB_NAME, TRACKING_JOB_NAME, CAMERA_EXTRACTION_JOB_NAME
from ..forging import FORGING_JOB_NAME, MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    error_response,
    resolve_detail_limit,
)
from ..two_photon import SingleRecordingJobNames
from .mcp_instance import mcp
from ..orchestration import (
    RESERVED_CORES,
    OUTCOME_FILE_SUFFIX,
    batch_directory,
    resolve_job_cores,
    read_batch_outcome,
    read_prepared_batch,
    forget_batch_records,
    resolve_host_memory_mb,
    resolve_concurrency_limits,
    resolve_concurrency_reservations,
)
from ..shared_assets import ProcessingPipelines
from .host_resolution import HOST_LABELS, unsupported_host_message
from ..microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

_BATCH_FILE_PATTERN: str = "*.yaml"
"""The pattern matching the prepared document and the recorded outcome of a batch, which are the two registry files
that name it. The lock file guarding each carries a further suffix, so the pattern passes over it."""

_BATCH_SEMI_FIELDS: tuple[str, ...] = (
    "batch_id",
    "pipeline",
    "host",
    "unit_count",
    "job_count",
    "blocked_count",
    "outcome_recorded",
)
"""The fields a batch listing carries, which are the identifier that dispatches the batch, what it dispatches, and
whether its run has already been closed. A settled batch carries no ``unit_count``, since closure retires the document
that counted its units."""

_BATCH_DETAIL_FIELDS: tuple[str, ...] = ("options", "job_names", "unit_names")
"""The fields detail adds, which are the parameters every job of the batch carries, the job types it holds, and the
units it covers."""

_MODEL_FIELDS: tuple[str, ...] = ("job_name", "pipeline", "cores", "concurrency_limit", "concurrency_reservation")
"""The fields one job type's declared model carries. A type that declares no ceiling and no reservation reports
neither, since an absent ceiling means the batch budgets bound it alone."""

_PIPELINE_JOB_NAMES: dict[ProcessingPipelines, tuple[str, ...]] = {
    ProcessingPipelines.CHECKSUM: (CHECKSUM_JOB_NAME,),
    ProcessingPipelines.RUNTIME: (RUNTIME_JOB_NAME,),
    ProcessingPipelines.MICROCONTROLLER: (CONTROLLER_EXTRACTION_JOB_NAME, PARSE_JOB_NAME),
    ProcessingPipelines.VIDEO: (CAMERA_EXTRACTION_JOB_NAME, RENAME_JOB_NAME, TRACKING_JOB_NAME, ENERGY_JOB_NAME),
    ProcessingPipelines.TWO_PHOTON: tuple(str(member) for member in SingleRecordingJobNames),
    ProcessingPipelines.FORGING: (MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME, FORGING_JOB_NAME),
}
"""The job types each batch pipeline dispatches, in the order its stages run. The dispatch table keys every allocation
by job name alone and names no pipeline for it, so this is what pairs a reported job type with the pipeline that owns
it. A stage added to a pipeline reaches this report once its name is listed here."""


@mcp.tool()
def list_prepared_batches_tool(
    batch_ids: list[str] | None = None,
    pipelines: list[str] | None = None,
    host: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Lists the batches this host has prepared, which is how a lost batch identifier is recovered.

    A host holds a handful of prepared batches rather than thousands, so the listing is the summary and appears in
    every response. Naming a filter narrows it. Opting into detail adds the parameters every job of a batch carries,
    the job types it holds, and the units it covers.

    A batch is recorded when it is prepared and outlives the server that prepared it, so an identifier issued in an
    earlier session is listed here. A batch stays listed once it has run, under the outcome closure recorded for it,
    which is why ``outcome_recorded`` separates a batch still to run from one already settled. Preparing a settled
    batch's work again queues only what is still outstanding, and ``forget_prepared_batches_tool`` drops the record.

    Args:
        batch_ids: Restricts the listing to these prepared batches.
        pipelines: Restricts the listing to the batches dispatching these pipelines.
        host: Restricts the listing to the batches prepared against one host, either ``local`` for this machine or
            ``remote`` for the configured compute server.
        limit: The batches to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        detailed: Determines whether each listed batch reports its options, its job types, and its units.

    Returns:
        A response dict with ``batch_directory``, ``total_batches``, ``total_jobs``, a ``breakdown`` per pipeline and
        host, and a ``batches`` list alongside ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row``. Each
        entry carries the batch's ``batch_id``, ``pipeline``, ``host``, ``unit_count``, ``job_count``,
        ``blocked_count``, and ``outcome_recorded``. A settled batch carries the counts its outcome recorded and no
        ``unit_count``, and it reports none of the detail fields. A detailed entry adds its ``options``, its
        ``job_names`` counts, and the ``unit_names`` it covers. Returns an error when the host is not supported, when
        the registry cannot be read, or when a named identifier is not held.
    """
    if host is not None and host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    directory = batch_directory()
    try:
        records = _read_recorded_batches(directory=directory)
    except Exception as exception:
        return error_response(message=f"Unable to read the prepared batches under '{directory}'. {exception}")

    if batch_ids is not None:
        held = {str(record["batch_id"]) for record in records}
        unknown = sorted(batch_id for batch_id in batch_ids if batch_id not in held)
        if unknown:
            return error_response(message=f"No prepared batch has identifier(s) {unknown}. Held: {sorted(held)}.")

    response = ok_response(
        batch_directory=str(directory),
        total_batches=len(records),
        total_jobs=sum(int(record["job_count"]) for record in records),
        breakdown={
            "pipeline": count_values(values=[record["pipeline"] for record in records]),
            "host": count_values(values=[record["host"] for record in records]),
        },
    )

    selectors: dict[str, list[str] | None] = {
        "batch_id": batch_ids,
        "pipeline": pipelines,
        "host": [host] if host is not None else None,
    }
    matched = [
        record
        for record in records
        if all(values is None or record[field] in values for field, values in selectors.items())
    ]
    fields = (*_BATCH_SEMI_FIELDS, *_BATCH_DETAIL_FIELDS) if detailed else _BATCH_SEMI_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["batches"] = [project_item(item=record, fields=fields) for record in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


@mcp.tool()
def forget_prepared_batches_tool(batch_ids: list[str]) -> dict[str, Any]:
    """Removes what this host records for the named batches, which is the prepared document and the recorded outcome.

    A settled batch keeps its outcome so a later caller still reads what its jobs recorded, and this is what drops that
    outcome once it has been read. A batch that has yet to run loses the descriptors preparation gave it, so its work
    is prepared again before anything dispatches it.

    Args:
        batch_ids: The batches to remove, as reported by :func:`list_prepared_batches_tool`.

    Returns:
        A response dict with ``batch_directory``, the ``forgotten`` identifiers this host held and removed,
        ``total_forgotten``, and the ``unknown`` identifiers for which it recorded nothing. Returns an error when no
        identifier is named or the registry cannot be written.
    """
    if not batch_ids:
        return error_response(
            message=(
                "Unable to forget a batch without an identifier. Name the batches to remove, since this removes what "
                "the registry records for each one rather than everything it holds."
            )
        )

    directory = batch_directory()
    try:
        forgotten = forget_batch_records(batch_ids=batch_ids)
    except Exception as exception:
        return error_response(message=f"Unable to forget the batches {batch_ids} under '{directory}'. {exception}")

    return ok_response(
        batch_directory=str(directory),
        forgotten=forgotten,
        total_forgotten=len(forgotten),
        unknown=sorted(set(batch_ids) - set(forgotten)),
    )


@mcp.tool()
def read_resource_model_tool(
    pipelines: list[str] | None = None,
    job_names: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
) -> dict[str, Any]:
    """Reports the width, the ceiling, and the reservation every job type declares, alongside this machine's capacity.

    The model holds a handful of job types rather than thousands, so the listing is the summary and appears in every
    response. Naming a filter narrows it.

    A type's ``cores`` is the width one of its jobs occupies. A ``concurrency_limit`` is a hard ceiling that spare
    capacity never lifts, and a ``concurrency_reservation`` is a soft hold that admission releases whenever no other
    job can use the capacity it gives up. Reading this answers what a stage costs without planning a unit, so nothing
    on disk is touched and no tracker is written.

    Args:
        pipelines: Restricts the listing to the job types these pipelines dispatch.
        job_names: Restricts the listing to these job type names.
        limit: The job types to list. Defaults to 200. A value at or below zero lists every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.

    Returns:
        A response dict with this machine's ``total_cores`` left to a batch after the ``reserved_cores`` the system
        keeps, and its ``total_memory_mb``. It also carries ``total_job_types``, the ``widest_job_cores`` any type
        declares, a ``breakdown`` per pipeline, and a ``job_types`` list with ``rows``, ``matched_rows``,
        ``start_row``, and ``next_start_row``. Each entry carries the type's ``job_name``, its ``pipeline``, its
        ``cores``, and the ``concurrency_limit`` and ``concurrency_reservation`` it declares. Returns an error when a
        named pipeline is not a batch pipeline, when a named job type is not one the model declares, or when a job type
        declares no width.
    """
    available = sorted(str(pipeline) for pipeline in _PIPELINE_JOB_NAMES)
    if pipelines is not None:
        unknown = sorted({pipeline for pipeline in pipelines if pipeline not in available})
        if unknown:
            return error_response(message=f"No batch pipeline is named {unknown}. Available: {available}.")

    try:
        entries = _render_job_types()
    except ValueError as exception:
        return error_response(message=f"Unable to report the declared resource model. {exception}")

    response = ok_response(
        total_cores=resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES),
        reserved_cores=RESERVED_CORES,
        total_memory_mb=resolve_host_memory_mb(),
        total_job_types=len(entries),
        widest_job_cores=max((int(entry["cores"]) for entry in entries), default=0),
        breakdown={"pipeline": count_values(values=[entry["pipeline"] for entry in entries])},
    )

    selectors: dict[str, list[str] | None] = {"pipeline": pipelines, "job_name": job_names}
    if job_names is not None:
        unknown = sorted({name for name in job_names if all(name != entry["job_name"] for entry in entries)})
        if unknown:
            return error_response(
                message=(
                    f"No job type is named {unknown}. Available: {sorted(str(entry['job_name']) for entry in entries)}."
                )
            )

    matched = [
        entry
        for entry in entries
        if all(values is None or entry[field] in values for field, values in selectors.items())
    ]
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=False), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["job_types"] = [project_item(item=entry, fields=_MODEL_FIELDS) for entry in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


def _read_recorded_batches(directory: Path) -> list[dict[str, Any]]:
    """Reads every batch the prepared-batch registry holds, prepared and settled alike.

    Notes:
        Closure retires a batch's prepared document behind the outcome it records, so a settled batch is held by its
        outcome file alone. Both files therefore name a batch the listing reports, and a settled entry answers from the
        outcome because the document that carried its units is gone.

    Args:
        directory: The registry directory.

    Returns:
        One entry per recorded batch, in identifier order.
    """
    if not directory.is_dir():
        return []

    identifiers = sorted(
        {
            path.name.removesuffix(OUTCOME_FILE_SUFFIX) if path.name.endswith(OUTCOME_FILE_SUFFIX) else path.stem
            for path in directory.glob(_BATCH_FILE_PATTERN)
        }
    )
    records: list[dict[str, Any]] = []
    for batch_id in identifiers:
        outcome = read_batch_outcome(batch_id=batch_id)
        document = read_prepared_batch(batch_id=batch_id)
        if document is not None:
            records.append(
                {
                    "batch_id": batch_id,
                    "pipeline": document.pipeline,
                    "host": document.host,
                    "unit_count": len(document.units),
                    "job_count": len(document.jobs),
                    "blocked_count": len(document.blocked_jobs),
                    "outcome_recorded": outcome is not None,
                    "options": dict(document.options),
                    "job_names": count_values(values=[job["job_name"] for job in document.jobs]),
                    "unit_names": [str(unit["unit_name"]) for unit in document.units if "unit_name" in unit],
                }
            )
            continue
        if outcome is None:
            continue
        records.append(
            {
                "batch_id": batch_id,
                "pipeline": str(outcome.get("pipeline", "")),
                "host": str(outcome.get("host", "")),
                "job_count": int(outcome.get("total", 0)),
                "blocked_count": int(outcome.get("blocked", 0)),
                "outcome_recorded": True,
            }
        )
    return records


def _render_job_types() -> list[dict[str, Any]]:
    """Renders the declared model of every job type a batch pipeline dispatches.

    Returns:
        One entry per job type, in the order its pipeline runs its stages.

    Raises:
        ValueError: If a job type declares no core allocation.
    """
    names = {name for pipeline_names in _PIPELINE_JOB_NAMES.values() for name in pipeline_names}
    limits = resolve_concurrency_limits(job_names=names)
    reservations = resolve_concurrency_reservations(job_names=names)

    entries: list[dict[str, Any]] = []
    for pipeline, pipeline_names in _PIPELINE_JOB_NAMES.items():
        for job_name in pipeline_names:
            entry: dict[str, Any] = {
                "job_name": job_name,
                "pipeline": str(pipeline),
                "cores": resolve_job_cores(job_name=job_name),
            }
            if job_name in limits:
                entry["concurrency_limit"] = limits[job_name]
            if job_name in reservations:
                entry["concurrency_reservation"] = reservations[job_name]
            entries.append(entry)
    return entries
