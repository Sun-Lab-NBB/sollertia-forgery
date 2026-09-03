"""Provides the sizing pass that resolves each job's cores and working set from the data it will process."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from hashlib import sha256
from numbers import Integral
from zipfile import BadZipFile
from functools import cache
from dataclasses import dataclass

import cv2
from cindra import (
    WORKER_MEMORY_MB,
    SPAWNED_CHILD_MEMORY_MB,
    MEMORY_ESTIMATE_TOLERANCE,
    RecordingArrays,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_array_path,
    resolve_output_path,
    size_multi_recording_job,
    size_single_recording_job,
    read_tracked_recording_geometry,
)
import pandas as pd
import psutil
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from ataraxis_video_system import (
    resolve_jobs,
    size_archive_job as size_camera_extraction_job,
)
from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, SessionTypes
from ataraxis_data_structures import find_log_archives, read_archive_message_count
from ataraxis_communication_interface import size_archive_job as size_controller_extraction_job

from ..video import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    # The chunk threshold is bound under a private name so that the sizing model's identifier digests it. The
    # threshold decides how many decoders a motion-energy job opens, so retuning it moves every motion-energy figure
    # and every plan stamped with the previous identifier has to be estimated again.
    MINIMUM_CHUNK_FRAMES as _MINIMUM_CHUNK_FRAMES,
    CAMERA_EXTRACTION_JOB_NAME,
    resolve_camera_video,
)
from ..forging import FORGING_JOB_NAME, MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from ..registries import (
    resolve_pose_prediction_locator,
    resolve_two_photon_data_locator,
    resolve_assembly_source_resolver,
    resolve_assembly_geometry_resolver,
    resolve_forging_admission_pipelines,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from ..shared_assets import ProcessingPipelines
from ..microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
    from sollertia_shared_assets import DatasetData

_MODEL_VERSION_DIGITS: int = 12
"""The width of the digest that identifies this module's sizing model."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds. Every reportable estimate is rounded up to a multiple of that figure."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, the type every modeled working array holds."""

_ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE: int = 768
"""The resident memory the runtime log reader holds per message its archive carries. Reading an archive builds one
directory entry per logged message, and that directory dominates the decoded payload itself, so the cost follows the
message count rather than the bytes those messages occupy."""

_CONTROLLER_ARCHIVE_TABLE_RATIO: float = 3.4
"""The resident memory a module parse job holds per byte of the whole log archive its controller recorded. A parse
job reads one module's share of that archive, and partitioning and merging the module's event streams materializes
that share several times over.

Notes:
    The term is charged against the whole archive rather than against the module's share of it, because no per-module
    share is derivable at plan time. It is therefore a bound every parse job of one controller carries in common, not
    a figure taken from the data any one of them reads.
"""

_DOUBLE_PRECISION_BYTES: int = 8
"""The width of one double-precision element, the type a pose prediction table holds on disk and the type every array
the pose-tracking stage reads out of it carries."""

_POSE_TABLE_COPIES: float = 6.0
"""The copies of a pose prediction table the pose-tracking job holds at its peak, each at the width the table itself
carries.

Notes:
    DeepLabCut writes the table in double precision already, so nothing is promoted on the way in and this figure
    counts copies alone. The reader sorts the table by its index, which holds the sorted copy beside the table it
    sorted, materializes the whole table again as a dense matrix beside that sorted copy, and stacks one array per
    bodypart out of that matrix, which comes to a third table between them. The per-metric arrays the pupil stage
    derives from those points are charged in the margin this figure carries above the three copies it counts.
"""

_DECODER_BUFFER_MEMORY_MB: int = 96
"""The resident memory one decode worker holds for its codec reference frames and its decoder state, beyond the frame
buffers the measurement itself retains."""

_RETAINED_FRAME_BUFFERS: int = 2
"""The number of full-resolution single-precision frame buffers a decode worker keeps live. Binning produces a
strided view that retains its full-frame base, and the current and previous binned frames are live at once."""

_UNRESOLVED_FRAME_PIXELS: int = 0
"""The pixels charged to a motion-energy job whose camera left no recording this pass could read. The stage skips a
camera whose recording is absent and completes, so such a job still holds its decoders and is charged them rather than
being refused."""

_UNRESOLVED_FRAME_COUNT: int = -1
"""The frame count charged to a motion-energy job whose camera left no recording this pass could read. A container no
pass opened reports no length, and the length is what decides how many decoders the stage opens, so such a job is
charged the full width its type declared rather than the single decoder a short recording earns. No container reports
a negative length, which is what makes this value unambiguous as the answer for a recording that was never read."""

_CHECKSUM_CHUNK_MEMORY_MB: int = 8
"""The read buffer one checksum worker allocates, which is the fixed chunk the data-structures library streams every
file through. The buffer is allocated once per worker and reused for every chunk of every file that worker hashes, so
it is the whole data-dependent term a worker carries."""

_CHECKSUM_READER_MEMORY_MB: int = SPAWNED_CHILD_MEMORY_MB + _CHECKSUM_CHUNK_MEMORY_MB
"""The resident memory one checksum worker holds. The pool that opens the workers is spawn-started, so each of them
pays the interpreter and import graph a spawned child carries, and each holds its own read buffer above that. Stating
the figure as the shared spawned-child cost plus the buffer is what makes a retune of that shared cross-library figure
reach this model as well, rather than leaving a checksum worker modeled as cheaper than the spawned child it is."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a cindra trace array carries, which are its regions and its samples."""

_UNREADABLE_OUTPUT_ERRORS: tuple[type[Exception], ...] = (OSError, EOFError, LookupError, ValueError, BadZipFile)
"""The failures a recording's combined metadata archive raises when it stands on disk but cannot be read.

Notes:
    The archive is a compressed entry store, so a truncated, emptied, or overwritten one fails in the shape of
    whichever layer first reaches the damage: the file layer, the archive layer, or the entry lookup that expects the
    field the geometry is read from. The five are unrelated types and none of them shares a base narrower than
    Exception, so the set is named here rather than approximated by one branch of it.

    Every one of them means the same thing to the sizing pass, which is that the recording carries no readable
    processed output. They are caught rather than propagated because the pass that reads them classifies OSError as
    an unreadable session marker, and a damaged archive would otherwise be reported under a remedy that would not
    repair it.
"""

_ASSEMBLY_SINGLE_DAY_COLUMNS: int = 4
"""The fluorescence columns an experiment assembly retains at the recording's own detected region count. Every column
is attached under its own name and none replaces another, so each stays live in the assembled frame for the rest of
the job."""

_ASSEMBLY_MULTI_DAY_COLUMNS: int = 4
"""The fluorescence columns an experiment assembly retains at the count of regions tracked across the animal's
recordings. Tracking keeps a region only where it appears in enough of those recordings, so these columns are narrower
than their single-day counterparts."""

_ASSEMBLY_WRITE_COPIES: int = 1
"""The copies of the assembled fluorescence volume charged at the write. The write streams the frame it was handed
rather than rebuilding it, so the stage peaks at the columns the assembly already holds."""

_ASSEMBLY_LOAD_TRANSIENT_COLUMNS: int = 1
"""The extra fluorescence columns, at the recording's own detected region count, that an assembly holds while it loads
one single-day column.

Notes:
    A single-day column is loaded by selecting the recording's cell rows out of the memory-mapped trace array and then
    making that selection contiguous in single precision. Both arrays are live at once, because the transposed copy is
    built from the masked selection and the selection is only released afterwards, so loading the k-th single-day
    column peaks at the k-1 columns already attached plus two more. The last of the four single-day columns therefore
    peaks at five, one above the four the retained-column term charges, and that one column is what this figure adds.

    The term is charged at the single-day region count alone. A multi-day column is loaded with no mask, so its
    contiguous copy is made straight off the memory map and only one copy of it is ever live, and by the time those
    columns load the single-day peak has already passed.

    It is added to the retained columns rather than compared against them, because the two peaks fall at different
    moments and the sum bounds both: the load peak carries no assembled sub-dataset yet, while the terminal peak
    carries every column and every sub-dataset but none of this transient. Adding therefore over-reserves the terminal
    peak by one single-day column, which is the safe direction, while a model taking neither peak's part under-reserves
    the load whenever a session's single-day column stands above its four multi-day columns and its sub-datasets
    together. Tracking prunes hard, so that is the ordinary case rather than the extreme one: a recording detecting two
    thousand regions of which five hundred are tracked holds one single-day column of eight hundred megabytes against
    four multi-day columns of a hundred and sixty and a sub-dataset term of fifty, which the fifteen percent tolerance
    and the gigabyte rounding cannot absorb.
"""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the assembled behavior, runtime, and video columns hold per sample of the reference clock on which they
are placed. Each sub-dataset emits one array per column at that height, the interpolation that lifts a column onto the
clock holds double-precision transients there, and the clip that closes the assembly copies the stacked frame once.
The figure charges the assembly's output alone: the source arrays those columns are interpolated from sit at their own
clocks and are charged separately."""

_SOURCE_INPUT_BYTES_PER_SAMPLE: int = 128
"""The memory an assembly holds per sample of one input source, at that source's own clock rather than at the clock
the assembled frame is placed on. The widest source is the camera carrying the eye, which is read through three
feathers of one height at once: its timestamp column at eight bytes a frame, its two single-precision motion-energy
columns at eight, and its nineteen pupil columns at seventy. The interpolation that lifts one of those columns onto
the reference clock holds a double-precision copy of that column and of the timestamps beside it, which adds sixteen.
That comes to a hundred and two bytes a frame, which this figure rounds up, and every other source an assembly reads
is narrower."""

_PERCENT_PER_FRACTION: float = 100.0
"""The divisor converting a percentage into a fraction."""

# This multiple belongs to the model this library keeps for its own assembly job, which is charged the multi-day
# columns the assembled frame holds. cindra keeps a separate model for its own multi-day tracking jobs, and answers
# the same question there for a different job. The two figures agreeing is the same domain reasoning reached twice
# rather than either copying the other, so retuning one carries no obligation to retune the other.
_TRACKED_REGION_HEADROOM: float = 1.5
"""The multiple of the most populated recording's region count that ceilings the templates multi-day tracking keeps.

Notes:
    The domain reading of the multiple, in the words of the model cindra states it in: "we have at most every cell in
    the most populated recording + maybe half of it coming from other recordings. That should be getting us very close
    to the actual maximum."
"""

_MODULE_SPECIFIER_SEPARATOR: str = "-"
"""The separator joining the controller, type, and identifier segments of a module parse job's specifier. The leading
segment names the controller that recorded the module. The job's estimate reads that controller's archive, which
every parse job of that controller reads in common."""

_ARCHIVE_JOB_NAMES: frozenset[str] = frozenset(
    {CAMERA_EXTRACTION_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME, RUNTIME_JOB_NAME}
)
"""The job types whose input is one source's log archive, which the data-structures locator resolves for them.

Notes:
    Each of these jobs is specified by the identifier of the source that recorded its archive, so one locating pass
    over the session's raw behavior data answers every one of them.
"""


@dataclass(frozen=True, slots=True)
class JobFootprint:
    """Describes the resources one job occupies while it runs, as this module's sizing pass resolved them.

    Notes:
        Carries the two fields every dependency's own sizing record carries, so a stage that a library owns and a
        stage that this package owns describe themselves the same way. Both halves follow from the job's input, and a
        job whose input cannot be read raises rather than reporting a footprint nothing measured.
    """

    cores: int
    """The cores the job occupies."""
    memory_mb: int
    """The reportable memory the job holds at its peak, in megabytes."""


def resolve_model_version() -> str:
    """Resolves the identifier of the sizing model this module implements.

    Notes:
        The identifier digests the name and the value of every private constant this module holds, so retuning any one
        of them answers with a new identifier and every plan stamped with the old one is estimated again. A stage
        constant this module binds from the pipeline it models is held under a private name for that reason, so
        retuning the stage reaches the plans its model stamped. A constant holding a collection is ordered before it
        is digested, which keeps one tuning answering with one identifier across processes.

    Returns:
        The hexadecimal identifier of the sizing model.
    """
    digest = sha256()
    for name, value in sorted(globals().items()):
        if not name.startswith("_") or not name.isupper():
            continue
        ordered = sorted(value) if isinstance(value, frozenset | set) else value
        digest.update(f"{name}={ordered!r}\n".encode())
    return digest.hexdigest()[:_MODEL_VERSION_DIGITS]


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def size_session_jobs(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible job of one session, reporting the cores it occupies and the memory it holds there.

    Notes:
        Reads on-disk metadata alone, so sizing a session never decodes a frame. The two extraction stages read their
        archive's central directory, which decodes no message and loads no payload. Every estimate scales with the
        input it describes, which keeps it correct on recordings longer, wider, or denser than any previously seen.

        One job is the exception, and it is charged a bound rather than an estimate. A module parse job reads a table
        the extraction job it is scheduled behind has yet to write, and the archive that table comes from attributes
        no message to a module, so every parse job of one controller is charged that whole archive. The figure is a
        bound those jobs share, and the model that charges it names what it would take to make it per-job.

        Every term is read from the session's raw acquisition data, so a footprint is available before any stage has
        run. Estimates cover anonymous memory, the term that forces a host to swap and a scheduler to kill a job, so
        the reclaimable pages a memory-mapped stage leaves resident are excluded.

        The stages that a dependency owns are sized by that dependency's own sizing pass, which reads the job's input
        once and answers both halves of its model from that read. It picks the width at which the stage actually
        runs, which for the extraction stages is one core for an archive below their parallel threshold and their
        declared allocation above it, and it estimates the memory at that width. Taking both figures whole is what
        keeps a retune of either half reaching slf without a change here, and it is what stops this package from
        reserving a width the library would never open.

        Every other stage is this package's own, so it is modeled here in the same shape, where one call answers both
        halves from the job's input. A stage whose cost holds one width, whatever data it reads, reports the allocation
        its type declared, which reaches this pass alongside the job.

        No stage answers with a floor. A job whose input cannot be read is a job that cannot run, so the refusal
        raised by the read propagates to the caller, which drops the target holding it rather than planning it at a
        figure nothing measured.

    Args:
        pipeline: The pipeline that owns the jobs.
        session: The loaded session on which the jobs operate.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job occupies and the memory it
        holds there, in megabytes.

    Raises:
        FileNotFoundError: If a job's input cannot be read, in which case the job that reads it cannot run either.
        OSError: If any directory beneath the session's raw behavior data cannot be read while its log archives are
            located.
        ValueError: If a two-photon job's specifier names an imaging plane the recording does not hold, or if the
            recording's acquisition metadata cannot be parsed. Also raised when the session's raw behavior data holds
            more than one archive for a source or more than one camera manifest, when a camera manifest registers no
            camera, when the session's pose prediction states no table width, or when a job name routes to no sizing
            model.
    """
    # Every job of the two-photon pipeline runs a cindra stage, so the whole pipeline is sized by cindra's own pass
    # and the recording's configuration and raw imaging location are resolved once for the session that carries them.
    if pipeline is ProcessingPipelines.TWO_PHOTON:
        resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
        locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
        configuration = resolve_configuration(session)
        data_path = locate_two_photon_data(session)
        return {
            (job_name, specifier): _size_two_photon_job(
                job_name=job_name,
                specifier=specifier,
                output_root=session.processed_data_path,
                configuration=configuration,
                data_path=data_path,
            )
            for job_name, specifier, _ in jobs
        }

    behavior_directory = session.raw_data.behavior_data_path

    # A motion-energy job reads the recording of the one camera its specifier names, so every such job's container
    # is read before any job is sized. A session planning no motion-energy job resolves nothing and opens nothing.
    energy_recordings = _resolve_energy_recordings(
        behavior_directory=behavior_directory,
        camera_directory=session.raw_data.camera_data_path,
        session_name=session.session_name,
        jobs=jobs,
    )

    # Locating a source's archive belongs to the library that writes it, and one pass answers every archive-reading
    # job the session holds. A source that the pass cannot resolve raises there, which is the same refusal each sizing
    # model raises for an input it cannot read.
    archives = _resolve_job_archives(behavior_directory=behavior_directory, jobs=jobs)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        if job_name == CHECKSUM_JOB_NAME:
            footprint = _size_checksum_job(cores=cores)
        elif job_name == CAMERA_EXTRACTION_JOB_NAME:
            footprint = _size_camera_extraction_job(archive_path=archives[specifier])
        elif job_name == CONTROLLER_EXTRACTION_JOB_NAME:
            footprint = _size_controller_extraction_job(archive_path=archives[specifier])
        elif job_name == RUNTIME_JOB_NAME:
            footprint = _size_runtime_job(archive_path=archives[specifier], cores=cores)
        elif job_name == ENERGY_JOB_NAME:
            footprint = _size_motion_energy_job(recording=energy_recordings[specifier], cores=cores)
        elif job_name == TRACKING_JOB_NAME:
            footprint = _size_pose_tracking_job(session=session, cores=cores)
        elif job_name == PARSE_JOB_NAME:
            # Every parse job of one controller resolves to that controller's archive, because nothing at plan time
            # states which share of it each module holds. The figure they receive is therefore one they share.
            footprint = _size_module_parse_job(
                archive_path=archives[specifier.split(_MODULE_SPECIFIER_SEPARATOR)[0]], cores=cores
            )
        elif job_name == RENAME_JOB_NAME:
            footprint = _size_rename_job(cores=cores)
        else:
            message = (
                f"Unable to size job '{job_name}' of session '{session.session_name}'. The job type routes to no "
                f"sizing model, and a job whose resources nothing resolves cannot be admitted to a batch."
            )
            console.error(message=message, error=ValueError)
        footprints[job_name, specifier] = footprint

    return footprints


def size_dataset_jobs(
    dataset: DatasetData, jobs: list[tuple[str, str, int]], *, planned_roi_count: int | None = None
) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible forging job from the processed data it will read, reporting its cores and its memory.

    Notes:
        Reads array headers, feather metadata and the presence of the recording metadata alone, so sizing a dataset
        decodes no fluorescence and reads no timestamp. Each cross-recording stage scales with the processed data the
        single-recording pipeline wrote for the sessions that carry two-photon data.

        Every job is routed to a model rather than to a blanket allowance, since a remote scheduler reserves memory
        per job. The two cross-recording stages belong to cindra, so both halves of their figures are cindra's own
        sizing pass, which refuses a dataset that any recording leaves short rather than sizing it from the recordings
        that happen to be complete. That refusal propagates, because a stage that cindra will not size is a stage the
        dataset cannot run until its recordings are complete. The recording set handed to that pass is the set the
        dataset names, and an animal whose sessions this host does not all hold is refused here on the same terms
        rather than measured over the subset the host happens to carry.

        The per-session assembly stage is this package's own, so no dependency models it and its projection stays
        here. Its width holds one value whatever data it reads, so it reports the allocation its type declared. Which
        of its two models applies follows from the acquisition system's admission policy: a session type that joins a
        dataset without completing the two-photon pipeline is assembled from its behavior sources onto a camera clock,
        so it is sized from that clock and from the heights those sources stand at rather than from fluorescence it
        never recorded. Both models charge that source family, since an assembly reads the same per-source feathers
        at the same per-source clocks whichever clock it goes on to place its frame on; the two differ in the frame
        alone.

        The one figure the assembly model cannot read from data this pass consumes is the count of regions the
        multi-day tracking goes on to keep, because that tracking is planned in the same universe as the jobs sized
        here and has therefore not run. It is bounded from the animal's recording set instead, and a caller that
        reasonably knows the count its animals carry states it and is charged it, which is the same override cindra's
        own sizing takes in place of the bound it would otherwise draw. The count is stated for the whole batch rather
        than per job, since one dataset's tracking runs at one configuration.

        A stated count reaches the two cross-recording stages as well as the assembly stage, because those are the
        stages that produce the templates the assembly reads. It is the same count for all three: one dataset tracks
        one set of templates, and a caller stating how many that set holds states it about the stage that builds it
        before the stage that consumes it.

    Args:
        dataset: The resolved dataset on which the jobs operate.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.
        planned_roi_count: The tracked templates to charge every job in the batch that spans them, which are the
            multi-day columns of an assembly job and the whole working set of a cross-recording job. Use None to
            accept the bound each animal's recording set carries, which is what the forging pipeline itself passes.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job occupies and the memory it
        holds there, in megabytes.

    Raises:
        FileNotFoundError: If a job's processed input cannot be read, in which case the job that reads it cannot run
            either. Also raised when a cross-recording job's animal holds fewer sessions under the project root than
            the dataset names for it, since the set this host can measure is then narrower than the set the job runs
            over and sizing it would under-reserve the job. An assembly job raises it on those same terms while no
            region count is stated, since the bound it would otherwise draw spans that whole set.
        ValueError: If the dataset's acquisition system donates no multi-recording configuration, if a job name
            routes to no sizing model, if the dataset's session type joins no dataset for its acquisition system, or
            if a stated region count is not a positive integer, which cindra's own sizing refuses for the
            cross-recording stages the count is handed to.
    """
    project_root = dataset.dataset_data_path.parent.parent
    animals = {entry.session: entry.animal for entry in dataset.sessions}
    configuration = _resolve_tracking_configuration(dataset=dataset, project_root=project_root)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        if job_name == MULTIDAY_DISCOVERY_JOB_NAME:
            # The discovery stage runs over one animal, so its specifier names that animal rather than a session.
            footprint = _size_multi_recording_job(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=specifier, project_root=project_root
                ),
                configuration=configuration,
                planned_roi_count=planned_roi_count,
            )
        elif job_name == MULTIDAY_EXTRACTION_JOB_NAME:
            footprint = _size_multi_recording_job(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=animals.get(specifier, ""), project_root=project_root
                ),
                configuration=configuration,
                planned_roi_count=planned_roi_count,
            )
        elif job_name == FORGING_JOB_NAME:
            footprint = _size_forging_job(
                dataset=dataset,
                animal=animals.get(specifier, ""),
                session=specifier,
                project_root=project_root,
                configuration=configuration,
                cores=cores,
                planned_roi_count=planned_roi_count,
            )
        else:
            message = (
                f"Unable to size job '{job_name}' of dataset '{dataset.name}'. The job type routes to no sizing "
                f"model, and a job whose resources nothing resolves cannot be admitted to a batch."
            )
            console.error(message=message, error=ValueError)
        footprints[job_name, specifier] = footprint

    return footprints


@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    """Describes the shape of a two-photon recording as its processing output reports it."""

    regions: int
    """The regions the single-recording pipeline detected."""
    samples: int
    """The samples each region's trace holds."""


@dataclass(frozen=True, slots=True)
class _EnergyRecording:
    """Describes the recording one motion-energy job decodes, as that recording's own container metadata reports it."""

    frame_pixels: int
    """The pixels one frame of the recording holds."""
    frame_count: int
    """The frames the recording's container reports, or ``_UNRESOLVED_FRAME_COUNT`` when no container was read."""


def _resolve_job_archives(behavior_directory: Path, jobs: list[tuple[str, str, int]]) -> dict[str, Path]:
    """Locates the log archive every archive-reading job of one session consumes.

    Notes:
        The archive filename that a source writes is the data-structures library's own contract, so the sources are
        handed to its locator rather than having their filenames rebuilt here. One traversal resolves every source,
        which is the same pass the acquisition libraries' own job resolvers make.

    Args:
        behavior_directory: The session's raw behavior data directory, whose tree holds every archive it recorded.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples, whose archive-reading members
            carry the identifier of the source that recorded their archive.

    Returns:
        The path to the archive of every source the jobs read, keyed by that source identifier. Empty when the jobs
        read no archive.

    Raises:
        FileNotFoundError: If the behavior data directory is absent, or if a source recorded no archive, in which case
            the job reading it cannot run either.
        OSError: If any directory beneath the behavior data directory cannot be read.
        ValueError: If the tree holds more than one archive for a source, which leaves the job's input ambiguous.
    """
    sources = {specifier for job_name, specifier, _ in jobs if job_name in _ARCHIVE_JOB_NAMES}
    # A parse job is specified by its module rather than by a source, and the controller that recorded the module is
    # the leading segment of that name, so the archive its estimate reads is recovered from the specifier.
    sources.update(
        specifier.split(_MODULE_SPECIFIER_SEPARATOR)[0] for job_name, specifier, _ in jobs if job_name == PARSE_JOB_NAME
    )
    if not sources:
        return {}
    return find_log_archives(log_directory=behavior_directory, source_ids=sorted(sources))


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return max(0, int(byte_count / _BYTES_PER_MEGABYTE) + 1) if byte_count > 0 else 0


def _round_to_gigabyte(memory_mb: int) -> int:
    """Rounds a memory figure up to a whole gigabyte, which is the quantum at which every estimate is reported.

    Notes:
        A scheduler reserves memory in whole gigabytes, so an estimate landing mid-gigabyte is rounded there by
        whatever consumes it. Rounding here instead keeps the figure a plan records identical to the figure a
        submission requests, so a planned batch and a submitted one compare directly.

        A figure that a dependency already sized is rounded here as well. The acquisition libraries report at a 256
        megabyte quantum while cindra already reports at a whole gigabyte, so this is the boundary where every figure
        reaches one scale whichever model produced it.

    Args:
        memory_mb: The memory to round, in megabytes.

    Returns:
        The memory in megabytes, rounded up to a whole gigabyte.
    """
    return math.ceil(memory_mb / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the shared estimate tolerance to a modeled memory figure and rounds it to a whole gigabyte.

    Notes:
        The tolerance is cindra's, which the video library carries at the same value while the communication library
        carries a wider one, so a mixed batch is weighed on close but not identical scales.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance and rounded up to a whole gigabyte.
    """
    return _round_to_gigabyte(memory_mb=int(memory_mb * MEMORY_ESTIMATE_TOLERANCE) + 1)


def _size_camera_extraction_job(archive_path: Path) -> JobFootprint:
    """Sizes one camera timestamp extraction job through the video library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read. It picks a single
        core for an archive below its parallel-extraction threshold and its declared allocation above it, then
        estimates the memory at the width it picked. Both figures are taken whole, so this package neither repeats the
        width rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is left to propagate, since a job with no
        readable input is a job to drop from the workflow rather than one to plan at a guessed figure.

    Args:
        archive_path: The path to the log archive the job reads.

    Returns:
        The job's footprint, holding the library's own width and its memory at that width.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    sizing = size_camera_extraction_job(archive_path=archive_path)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_controller_extraction_job(archive_path: Path) -> JobFootprint:
    """Sizes one microcontroller data extraction job through the communication library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read. It picks a single
        core for an archive below its parallel-extraction threshold and its declared allocation above it, then
        estimates the memory at the width it picked. Both figures are taken whole, so this package neither repeats the
        width rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is left to propagate, since a job with no
        readable input is a job to drop from the workflow rather than one to plan at a guessed figure.

    Args:
        archive_path: The path to the log archive the job reads.

    Returns:
        The job's footprint, holding the library's own width and its memory at that width.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    sizing = size_controller_extraction_job(archive_path=archive_path)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_runtime_job(archive_path: Path, cores: int) -> JobFootprint:
    """Sizes one runtime log job from the archive it reads.

    Notes:
        Every reader opens the archive itself, so each one holds the archive's directory. A single core decodes in the
        job's own process, which holds one directory and starts no child. A wider allocation opens one child per
        allocated core, and the job's own process keeps the reader that planned the batches, so the directory is held
        once more than the child count. The runtime pipeline is this package's own, so no dependency models it, and
        the pool it opens is the allocation its type declared.

    Args:
        archive_path: The path to the log archive the job reads, as the shared locating pass resolved it.
        cores: The cores the job is allocated, which is how many readers it opens.

    Returns:
        The job's footprint, holding the declared width and the memory the readers hold at it.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    per_reader = _bytes_to_megabytes(
        byte_count=read_archive_message_count(archive_path=archive_path) * _ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE
    )
    children = cores if cores > 1 else 0
    readers = children + 1
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB + readers * per_reader + children * SPAWNED_CHILD_MEMORY_MB
        ),
    )


def _size_checksum_job(cores: int) -> JobFootprint:
    """Sizes one raw-data checksum job from the readers it opens.

    Notes:
        Each worker streams its file in fixed chunks and holds one at a time, so the figure is flat across every
        session size. The stage is this package's own and gains nothing from a width the data picks, so it runs at the
        allocation its type declared.

        The flatness rests on one precondition. The parent retains a pending result per *file* under the session's raw
        data, and for an assembled session that is hundreds to a few thousand entries, which stays below the rounding
        this estimate already carries. That holds only because the acquisition runtime consolidates each logger's
        per-message arrays into one archive per source before the session reaches this pipeline, which no stage this
        package owns performs. A session whose behavior logs were left unconsolidated carries one file per logged
        message instead, at which point the parent's per-file retention dominates everything this model charges and
        the figure is no longer flat.

    Args:
        cores: The cores the job is allocated, which is how many files it hashes at once.

    Returns:
        The job's footprint, holding the declared width and the memory its readers hold at it.
    """
    return JobFootprint(
        cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * _CHECKSUM_READER_MEMORY_MB)
    )


def _size_rename_job(cores: int) -> JobFootprint:
    """Sizes one camera timestamp rename job.

    Notes:
        The stage performs a fixed handful of filesystem operations and reads no recording, so it holds a worker and
        nothing besides. No input scales the estimate, so the worker itself is the whole model rather than a floor
        standing in for one.

    Args:
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and one worker's memory.
    """
    return JobFootprint(cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB))


def _resolve_energy_recordings(
    behavior_directory: Path, camera_directory: Path, session_name: str, jobs: list[tuple[str, str, int]]
) -> dict[str, _EnergyRecording]:
    """Reads the container metadata of the one recording each motion-energy job of a session decodes.

    Notes:
        A motion-energy job is specified by the source identifier of the single camera it measures, and that camera's
        own recording is the only input it reads, so each job is charged the frame of that recording rather than the
        widest frame the session holds. The two figures diverge by the ratio between the session's cameras, which for
        a rig pairing a fast small-sensor camera with a slow large-sensor one is most of the reservation.

        The colloquial name that locates a camera's recording on disk lives in the acquisition-time camera manifest,
        so the source identifiers the jobs carry are mapped through the video library's own resolver rather than
        through a camera vocabulary rebuilt here. One read of that manifest answers every motion-energy job of the
        session, and the recording it names is then resolved through the same resolver the stage itself calls, so the
        container this pass measures is the container the job opens.

        Each recording is opened once, for the one job that decodes it, so the pass reads one container per job rather
        than one per recording the session holds, and it reads none at all for a session that plans no motion-energy
        job. Both figures the model needs come off that one open: the frame decides what a decoder holds, and the
        length decides how many decoders the stage opens, so the length is read where the frame already is rather than
        being charged at the job's full width for want of a second read.

        A camera the manifest does not register, and a registered camera that left no recording, contribute no pixels.
        Neither is refused: the stage skips a camera whose recording is absent and completes, so the job runs and
        holds its decoders whatever that camera recorded.

    Args:
        behavior_directory: The session's raw behavior data directory, whose tree holds the camera manifest.
        camera_directory: The session's raw camera data directory, which holds the recordings themselves.
        session_name: The name of the session, which prefixes every recording filename.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples, whose motion-energy members
            carry the source identifier of the camera they measure.

    Returns:
        The frame and the length of each motion-energy job's recording, keyed by that job's specifier. Empty when the
        jobs measure no camera.

    Raises:
        OSError: If any directory beneath the behavior data directory cannot be read while the camera manifest is
            located.
        ValueError: If the tree holds more than one camera manifest, or if the manifest registers no camera, in which
            case the video pipeline refuses the session at discovery and plans none of these jobs either.
    """
    specifiers = {specifier for job_name, specifier, _ in jobs if job_name == ENERGY_JOB_NAME}
    if not specifiers:
        return {}

    # A session that recorded no behavior data at all registers no camera, which is the same answer the resolver gives
    # for a tree holding no manifest. The directory is checked here because the resolver refuses an absent one, and a
    # refusal would drop a job the stage itself would run and complete.
    sources = resolve_jobs(log_directory=behavior_directory).sources if behavior_directory.is_dir() else ()
    camera_names = {source.source_id: source.name for source in sources}

    recordings: dict[str, _EnergyRecording] = {}
    for specifier in specifiers:
        camera_name = camera_names.get(specifier)
        recording = (
            resolve_camera_video(
                camera_data_directory=camera_directory, session_name=session_name, camera_name=camera_name
            )
            if camera_name is not None
            else None
        )
        recordings[specifier] = (
            _EnergyRecording(frame_pixels=_UNRESOLVED_FRAME_PIXELS, frame_count=_UNRESOLVED_FRAME_COUNT)
            if recording is None
            else _read_recording_metadata(recording=recording)
        )
    return recordings


def _read_recording_metadata(recording: Path) -> _EnergyRecording:
    """Reads the frame and the length one camera recording's container reports.

    Notes:
        Reads container metadata only, so no frame is decoded. The length is read from the same property the stage
        itself reads, off the same open, so the chunk plan this pass models is the chunk plan the job runs.

        A container reporting either frame extent as non-positive describes no frame at all, which is charged as no
        pixels rather than as a negative area. A length is passed through as the container reports it, including a
        negative one, which the sizing model reads as the same unplannable recording an unread container describes.

    Args:
        recording: The path to the camera recording whose container is read.

    Returns:
        The frame and the length the recording's container reports.
    """
    capture = cv2.VideoCapture(str(recording))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return _EnergyRecording(frame_pixels=max(0, width) * max(0, height), frame_count=frame_count)


def _size_motion_energy_job(recording: _EnergyRecording, cores: int) -> JobFootprint:
    """Sizes one video motion-energy job from the frame it decodes and the chunks its length earns.

    Notes:
        Each decode worker holds the binned frame and its predecessor. Binning slices a strided view out of a
        full-resolution buffer, and a view retains its base, so both cost the whole frame. The frame charged is the
        frame of the one recording this job decodes, so a session pairing cameras of different sensors reserves each
        of their jobs the memory that job holds rather than reserving every one of them the widest camera's frame.

        The decoders charged are the decoders the stage opens rather than the cores the job was allocated. The stage
        splits its recording into chunks no shorter than the shared minimum and opens one worker per chunk, so a
        recording holding fewer frames than that minimum times the allocation opens fewer workers than the allocation
        names, and one holding fewer than the minimum itself decodes in the job's own process and opens no pool at
        all. A short clip therefore holds one decoder and no spawned child, which for a rig recording calibration and
        training clips beside full sessions is most of what charging the allocation would have reserved. The
        allocation still bounds the chunk count, so it is the width the estimate assumes the job runs at.

        The recording's length reaches this model from the same container property the stage reads, so the chunk plan
        modeled here is the chunk plan the job runs rather than a rule restated beside it. A recording this pass could
        not read states no length, and is charged the full width instead, since the stage that skips such a camera
        would otherwise be reserved less than the job whose camera did record.

        The parent collects one energy and one luminance array per chunk, which together span the recording once at
        single precision and stay below the rounding this estimate already carries.

    Args:
        recording: The frame and the length of the recording this job decodes, which the pre-pass read from the
            container of the camera the job's specifier names.
        cores: The cores the job is allocated, which bounds how many chunks it decodes at once.

    Returns:
        The job's footprint, holding the declared width and the memory the decoders it opens hold at it.
    """
    frame_buffers = _bytes_to_megabytes(
        byte_count=recording.frame_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS
    )

    # Mirrors the stage's own chunk plan, which it builds from this same allocation and this same threshold. A
    # recording whose container this pass could not read reports a non-positive length, and is charged the full width
    # rather than the single chunk that a length of nothing would otherwise imply. The test is non-positive rather
    # than negative because a capture carrying no backend reports zero as readily as it reports a negative, and zero
    # would otherwise route the job to the cheapest branch of the two.
    chunks = cores if recording.frame_count <= 0 else max(1, min(cores, recording.frame_count // _MINIMUM_CHUNK_FRAMES))

    # A single chunk decodes in the job's own process, which opens no pool, so the job holds one decoder and starts
    # no child at all.
    if chunks == 1:
        return JobFootprint(
            cores=cores,
            memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + frame_buffers + _DECODER_BUFFER_MEMORY_MB),
        )

    per_worker = frame_buffers + _DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB
    return JobFootprint(cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + chunks * per_worker))


def _size_two_photon_job(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> JobFootprint:
    """Sizes one two-photon job through cindra's own per-stage sizing pass.

    Notes:
        cindra reads the recording once and answers both halves of its model from that read, so the width at which a
        stage runs is the width cindra declares for that stage rather than a figure this package repeats. Taking both
        figures whole is what keeps a retune of either half reaching slf without a change here.

        The binarization, registration and processing stages are sized from the recording's own geometry, and the two
        per-plane stages additionally from the plane their specifier names, so each of them receives a figure taken
        from what it reads. The combination stage is not: its whole data-dependent term is the region count, and no
        recording states that count until its detection stage has run. What cindra charges the combination stage is
        therefore the ceiling the recording's own configuration allows, which is the multiplier the detection loop
        applies to its configured iteration limit, taken across the recording's planes. Every recording of one
        acquisition system carrying the same plane count therefore receives the identical figure whatever it went on
        to detect, so this is a configuration bound the stage shares rather than an estimate taken from the recording,
        and it stands above what a recording detects by whatever margin its configured limit stands above its data.

        The bound stands because the region count is not a raw input. It is written by a stage scheduled ahead of this
        one, so at plan time no reading of the recording answers it, and a plan has to size every stage before any of
        them runs. Passing cindra a measured count off a recording whose combination output already exists is what
        would turn this into a per-recording estimate, and it would additionally have to gate on the detection
        settings being unchanged, since a re-detection under widened settings can exceed the count the previous pass
        recorded.

        cindra rejects a stage it cannot size, either because the recording carries no readable raw imaging data or
        because a per-plane specifier names a plane the recording does not hold. Both refusals propagate, since a
        recording that cindra will not size is a recording whose stages cannot run.

    Args:
        job_name: The tracker job name identifying the stage.
        specifier: The plane specifier for a registration or processing job, empty for the binarization and
            combination stages.
        output_root: The output root the recording's cindra configuration was given.
        configuration: The recording's resolved processing configuration.
        data_path: The raw imaging directory, consulted when the recording carries no output yet.

    Returns:
        The job's footprint, holding cindra's own width for the stage and its memory at that width.

    Raises:
        FileNotFoundError: If the recording carries neither pipeline output nor readable raw imaging data, in which
            case no stage of it can run.
        ValueError: If the specifier names an imaging plane the recording does not hold, or if both inputs were
            readable and still describe no whole imaging plane.
    """
    sizing = size_single_recording_job(
        job_name=SingleRecordingJobNames(job_name),
        specifier=specifier,
        output_root=output_root,
        configuration=configuration,
        data_path=data_path,
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: tuple[Path, ...],
    configuration: MultiRecordingConfiguration | None,
    *,
    planned_roi_count: int | None = None,
) -> JobFootprint:
    """Sizes one cross-recording job through cindra's own per-stage sizing pass.

    Notes:
        Both cross-recording stages read every recording of the animal over which they run, so the whole recording set
        is handed to cindra whichever stage is being sized, and both halves of the figure come back from that one read.

        cindra refuses a set that any recording leaves short rather than sizing it from the recordings that happen to
        be complete, and that refusal propagates. A dataset whose recordings carry no combined output cannot run
        either stage yet, so it is dropped from the workflow rather than planned at a floor.

        A stated template count is handed on to cindra rather than spent on this package's own assembly model alone.
        The count a caller states is a property of the tracking the dataset will run, and these two stages are the
        stages that run it, so the caller who knows it knows it first about them. Withholding it here would leave the
        jobs that produce the templates bounded from the recordings while the job that merely reads them is charged
        the stated figure, which is the one arrangement of the two that cannot be right.

    Args:
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a session for the extraction stage and an animal for the
            discovery stage.
        recording_directories: The cindra output directory of every recording the job spans.
        configuration: The dataset's resolved multi-recording configuration, or None when its acquisition system
            donates none.
        planned_roi_count: The tracked templates to plan the stage for, counting the dataset as a whole. Use None to
            accept the bound cindra draws from the per-recording region counts.

    Returns:
        The job's footprint, holding cindra's own width for the stage and its memory at that width.

    Raises:
        FileNotFoundError: If the job spans no recording, if any recording it spans carries no combined metadata
            archive, or if any recording reports no regions in its combined trace array, in which case neither
            cross-recording stage can run.
        ValueError: If the dataset's acquisition system donates no multi-recording configuration, which leaves the
            stage without the parameters its sizing needs, or if a stated template count is not a positive integer.
    """
    if configuration is None:
        message = (
            f"Unable to size the '{job_name.value}' job of '{specifier}'. The dataset resolved no multi-recording "
            f"configuration, either because its acquisition system tracks no regions across the sessions it holds "
            f"or because none of those sessions remains under the project root, so the stage carries neither sizing "
            f"parameters nor input data."
        )
        console.error(message=message, error=ValueError)
    sizing = size_multi_recording_job(
        job_name=job_name,
        specifier=specifier,
        recording_directories=recording_directories,
        configuration=configuration,
        planned_roi_count=planned_roi_count,
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_pose_tracking_job(session: SessionData, cores: int) -> JobFootprint:
    """Sizes one pose-tracking job from the prediction file its session carries.

    Notes:
        The stage reads predictions written upstream and never runs inference, so its working set follows the table it
        reads. The file is named by the system that produces it, so it is resolved through that system's donated
        locator. That locator answers with the file that the tracking worker opens. The stage is this package's own and
        its own fan-out is fixed, so it runs at the allocation its type declared.

        The table is charged at the width it holds rather than at the size its file occupies on disk. The two agree
        only for a prediction the writer left uncompressed, while a compressed one occupies a fraction of its table and
        is still expanded to full width the moment the stage reads it, so charging the file would understate exactly
        the prediction most able to exhaust the node. The row and column counts come from the table's own metadata, so
        no prediction is loaded to read them and the figure holds whatever the writer compressed.

    Args:
        session: The loaded session whose pose predictions the job reads.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory the prediction table implies.

    Raises:
        FileNotFoundError: If the session carries no pose prediction, in which case the job reading one cannot run
            either.
        ValueError: If the prediction holds other than one table, or holds one whose row and column counts its own
            metadata does not report, in which case nothing states how wide the job's working set stands.
    """
    prediction = resolve_pose_prediction_locator(system=session.acquisition_system)(session=session)
    if prediction is None:
        message = (
            f"Unable to size the pose-tracking job of session '{session.session_name}'. The session carries no pose "
            f"prediction, so nothing states how much the job holds and the job could not run either."
        )
        console.error(message=message, error=FileNotFoundError)
    rows, columns = _read_pose_table_shape(prediction=prediction)
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=rows * columns * _DOUBLE_PRECISION_BYTES * _POSE_TABLE_COPIES)
        ),
    )


def _read_pose_table_shape(prediction: Path) -> tuple[int, int]:
    """Reads the row and column counts a pose prediction table reports in its own metadata.

    Notes:
        Reads the table's description alone, so no prediction is loaded and the read costs the same on a prediction of
        any length. The counts are what the stage's working set scales with, and unlike the size the file occupies on
        disk they do not move with whatever compression the writer applied.

        The stage reads the prediction through a call that names no key, which requires the file to hold exactly one
        frame, so a file holding another number of them is refused here on the terms the stage would refuse it.

        A frame written in the fixed layout rather than the table layout DeepLabCut writes reports neither count, and
        is refused rather than charged a figure taken from the file's size, which is the read this model exists to
        replace.

    Args:
        prediction: The path to the pose prediction file whose table is measured.

    Returns:
        The rows and the columns the table holds.

    Raises:
        ValueError: If the file holds other than one frame, or holds one whose row and column counts its metadata does
            not report.
    """
    with pd.HDFStore(path=str(prediction), mode="r") as store:
        keys = store.keys()
        if len(keys) != 1:
            message = (
                f"Unable to size the pose-tracking job reading '{prediction.name}'. The prediction file must hold "
                f"exactly one table for the stage to read it without naming a key, but it holds {len(keys)}."
            )
            console.error(message=message, error=ValueError)
        # The published pandas stubs declare no 'get_storer', so the call resolves through the store's catch-all
        # attribute hook and is read as an attribute rather than as a method. The runtime call is the documented one.
        storer = store.get_storer(key=keys[0])  # type: ignore[operator]
        rows = getattr(storer, "nrows", None)
        columns = getattr(storer, "ncols", None)

    # The counts come back as whatever integral type the table's own reader carries, which for the row count is a
    # numpy width rather than a builtin one, so they are admitted by the numeric tower and narrowed here.
    if not isinstance(rows, Integral) or not isinstance(columns, Integral):
        message = (
            f"Unable to size the pose-tracking job reading '{prediction.name}'. The prediction table reports no row "
            f"and column count in its own metadata, which is what a frame written in the fixed layout rather than "
            f"the table layout DeepLabCut writes reports. The job's working set stands at the width of that table, "
            f"so a prediction that does not state it cannot be sized without charging the file's compressed size in "
            f"its place, which would under-reserve the job."
        )
        console.error(message=message, error=ValueError)
    return int(rows), int(columns)


def _size_module_parse_job(archive_path: Path, cores: int) -> JobFootprint:
    """Charges one module parse job the upper bound its controller's whole log archive states.

    Notes:
        This figure is a bound shared by every parse job of one controller, not an estimate of what this job holds.
        A controller carrying several parsed modules produces one parse job per module, and each of them is charged
        the same whole archive, so their reservations sum to that archive once per module while the tables they read
        sum to at most one archive between them. A controller pairing a continuously polled module with a sparse
        event-driven one therefore reserves the sparse module's job the memory the polled module's job holds. The
        bound is a per-job figure only where the controller carries a single parsed module.

        The bound stands because no per-module figure exists at plan time. A parse job reads the raw per-module
        feather its controller's extraction job writes, and it is scheduled behind that job, so its own input is
        absent while the plan is made. The archive that input is extracted from is present but states nothing about
        the module: the archive's index names each message by the recording source and the acquisition timestamp
        alone, and a message names its module only inside its own body, so attributing messages to modules costs one
        payload read per message, which is the pass the extraction job itself makes. The event codes the acquisition
        system registers do not separate the modules either, since the modules of one controller reuse them.

        A per-module message count carried by the archive index is what would turn this into a per-job estimate. The
        library that assembles the archive is the one that could carry each message's module into its entry name or
        into an index beside it, at which point this model would charge each job the share its own module holds.

        The stage is a single pass over one module's extracted table, so it runs at the allocation its type declared.

    Args:
        archive_path: The path to the log archive the job's controller recorded, which every parse job of that
            controller is charged in full.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory its controller's whole archive implies.
    """
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=archive_path.stat().st_size * _CONTROLLER_ARCHIVE_TABLE_RATIO)
        ),
    )


@cache
def _two_photon_output_root(project_root: Path, animal: str, session: str) -> Path:
    """Resolves the output root given to a session's two-photon processing.

    Notes:
        This is the root under which cindra creates its own output directory, so every location beneath it is resolved
        through cindra's own resolvers rather than by rebuilding its layout here.

        Read off the same cached marker a system's donated resolvers are handed, rather than loading a second copy of
        it. One assembly estimate reaches both, since it reads the session's own imaging geometry through this root
        and its assembly sources through that donation, and a marker read twice for one session is one YAML read per
        session more than the pass needs.

        Cached, because one dataset's estimates resolve the same session from several stages and each resolution
        otherwise re-resolves that session's root.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose output root is resolved.

    Returns:
        The path to the session's processed-data root, which is the output root its cindra jobs were given.
    """
    return _session_marker(project_root=project_root, animal=animal, session=session).processed_data_path


@cache
def _session_marker(project_root: Path, animal: str, session: str) -> SessionData:
    """Loads the marker of one session a dataset holds.

    Notes:
        The marker is what a system's donated sizer reads, and it is the same object that system's assembler is handed
        when the job actually runs, so a donation reads the locations its assembler reads rather than locations this
        pass resolved on its behalf.

        Cached on the same terms as the two-photon output root, because a session's marker is otherwise re-read for
        every stage that resolves a location beneath it.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose marker is loaded.

    Returns:
        The loaded session.
    """
    return SessionData.load(session_path=project_root.joinpath(animal, session))


def _animal_recording_directories(dataset: DatasetData, animal: str, project_root: Path) -> tuple[Path, ...]:
    """Resolves the cindra output directory of every recording one animal contributes to a dataset.

    Notes:
        This is the recording set named by the animal's materialized multi-recording configuration, resolved from the
        project root rather than read back from that file, so a dataset whose configurations have not been written
        yet is still sizable. The dataset's own session list states which recordings that configuration names, because
        the configuration is written from every session the dataset holds for the animal, and the job that reads it
        back runs over every directory it names.

        A planning host that holds fewer of those sessions than the dataset names is refused rather than sized from
        the ones it holds. The discovery stage is quadratic in the pooled region count of the set it runs over, so an
        animal planned from half its sessions is reserved a quarter of what the job goes on to hold, and a job that
        is reserved less than it holds is killed by the scheduler and cancels every dependent scheduled behind it.
        The refusal names the sessions the host is missing, which are the ones to stage before the animal is planned.

    Args:
        dataset: The resolved dataset that holds the animal.
        animal: The animal whose recordings are resolved.
        project_root: The path to the project's root directory.

    Returns:
        The cindra output directory of every recording the dataset names for the animal.

    Raises:
        FileNotFoundError: If the project root holds fewer of the animal's dataset sessions than the dataset names, in
            which case the set this host can measure is narrower than the set the job runs over.
    """
    entries = dataset.get_sessions_for_animal(animal=animal)
    missing = sorted(entry.session for entry in entries if not project_root.joinpath(animal, entry.session).is_dir())
    if missing:
        message = (
            f"Unable to size the cross-recording jobs of animal '{animal}' in dataset '{dataset.name}'. The dataset "
            f"names {len(entries)} session(s) for the animal and both cross-recording stages run over every one of "
            f"them, but {len(missing)} of them are absent under the project root '{project_root}': "
            f"{', '.join(missing)}. The discovery stage is quadratic in the pooled region count of the recordings it "
            f"spans, so sizing this animal from the sessions this host holds would reserve the job less memory than "
            f"it goes on to hold. Stage the named session(s) under the project root, or rebuild the dataset without "
            f"them, then plan the animal again."
        )
        console.error(message=message, error=FileNotFoundError)
    return tuple(
        resolve_output_path(
            output_root=_two_photon_output_root(project_root=project_root, animal=animal, session=entry.session)
        )
        for entry in entries
    )


def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None:
    """Reads a processed recording's shape from the output the single-recording pipeline wrote.

    Notes:
        The region count is read through cindra's own tracked-geometry reader rather than resolved here, so the
        regions this package bounds a tracked count from are the regions cindra's multi-recording stages go on to
        read from that same recording. The reader carries the completion gate along with them, reporting an
        unresolved geometry for a recording that has not reached the end of the combination stage and therefore
        carries no output the forging stages can read.

        The sample count is not taken from the same reader, because the frame count it reports is read from the
        combined metadata archive while a sample is a column of the trace array. cindra writes that frame count only
        from the release that introduced it and documents a missing one as reading back zero, so a recording
        processed by an earlier release reports no frames while its traces stand at their full width. The assembly
        model multiplies every retained column by the sample count, so accepting that zero would collapse the whole
        fluorescence term for exactly the recordings that are oldest and least likely to be reprocessed. The traces
        are therefore measured at their own header, which reports what the assembly will actually load.

        No file's contents are loaded. The resolution reads the archive's own entries and the trace array's header,
        so a recording of any length costs the same pair of small reads.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose processed output is read.

    Returns:
        The recording's geometry, or None when the session holds no readable processed imaging output.
    """
    cindra_root = resolve_output_path(
        output_root=_two_photon_output_root(project_root=project_root, animal=animal, session=session)
    )
    try:
        geometry = read_tracked_recording_geometry(cindra_root=cindra_root)
    except _UNREADABLE_OUTPUT_ERRORS:
        # A recording whose archive stands but cannot be read is as unmeasured as one carrying no archive at all, so
        # both answer None and are named in the same clause of the refusal.
        return None

    # cindra reports an unresolved geometry both for a recording carrying no combined metadata archive and for one
    # whose archive describes an empty combined field. Its own multi-recording sizing refuses that second recording
    # on the same terms, so treating the two alike keeps the set this bound is drawn from equal to the set the
    # tracking stages will accept.
    if not geometry.resolved:
        return None

    # The reader resolves a geometry from the metadata archive alone and reports a zero region count for a recording
    # whose trace array is absent or carries another rank, so the traces are gated here rather than left to answer a
    # region count of zero. That gate is the sample count's own read, which the reader does not supply.
    traces = _read_array_shape(
        array_path=resolve_array_path(root_path=cindra_root, array=RecordingArrays.CELL_FLUORESCENCE)
    )
    if traces is None:
        return None
    return _RecordingGeometry(regions=geometry.region_count, samples=traces[1])


def _read_array_shape(array_path: Path) -> tuple[int, int] | None:
    """Parses the shape a two-dimensional array's own header reports.

    Args:
        array_path: The path to the array whose header is parsed.

    Returns:
        The array's two extents, or None when it is absent or carries another rank.
    """
    if not array_path.is_file():
        return None
    with array_path.open("rb") as array_file:
        reader = read_array_header_1_0 if read_magic(array_file) == (1, 0) else read_array_header_2_0
        shape, _, _ = reader(array_file)
    if len(shape) != _TRACE_ARRAY_DIMENSIONS:
        return None
    return int(shape[0]), int(shape[1])


def _resolve_tracking_configuration(dataset: DatasetData, project_root: Path) -> MultiRecordingConfiguration | None:
    """Resolves the multi-recording configuration the dataset's acquisition system donates.

    Notes:
        Read from the system registry rather than from the file that ``define_forging_dataset`` materializes, so the
        parameters are available for a dataset whose configurations have not been written yet.

        Resolved from the first session still present under the project root, because a dataset outlives the source
        data of the animals it has already forged. Estimating a newly added animal therefore does not depend on
        sessions that have moved to long-term storage.

    Args:
        dataset: The resolved dataset whose acquisition system donates the configuration.
        project_root: The path to the project's root directory.

    Returns:
        The resolved configuration, or None when no session remains under the project root, or when the dataset's
        sessions need no multi-day processing.
    """
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)
    for entry in dataset.sessions:
        session_path = project_root.joinpath(entry.animal, entry.session)
        if not session_path.is_dir():
            continue
        return resolve_configuration(SessionData.load(session_path=session_path))
    return None


def _resolve_tracked_regions(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
    *,
    planned_roi_count: int | None = None,
) -> int:
    """Resolves how many regions the multi-day columns of a session's assembled frame span.

    Notes:
        The count is bounded from the single-recording geometries of the animal's recording set rather than read back
        from the multi-day arrays the tracking writes. The forging pipeline plans its cross-recording stages and its
        per-session assembly jobs in one universe, so the tracking has not run when either is sized and the arrays it
        writes do not exist to be read. Every input this bound takes is a single-recording output, which the pipeline
        consumes rather than produces. The bound is therefore the whole model here rather than a fallback standing in
        for a read that happens to be unavailable.

        This bound belongs to this library, which charges its assembly job the multi-day columns that job assembles.
        cindra bounds the same templates for its own multi-day tracking jobs, and keeps that model private to itself.
        Neither stands in for the other, and neither waits on the other to expose anything: two jobs are sized, and
        the same domain reasoning answers both.

        The bound is the smaller of two ceilings, because neither covers the other. The headroom ceiling allows every
        region of the most populated recording and half as many again, which is the domain reading of a template. The
        pooled ceiling divides the pooled region count by the recordings a cluster must span, which is combinatorial:
        every template consumes at least that many regions and consumes them exclusively. The pooled term is the
        tighter one over few recordings, where two recordings of a thousand regions at a minimum of two admit a
        thousand templates against the fifteen hundred the headroom term allows. The headroom term is the tighter one
        over many, where twenty recordings of fifteen thousand regions at half prevalence pool to thirty thousand
        templates against a real ceiling near fifteen thousand, which is twice the memory such an animal holds. A
        pooled-only bound was tried on those terms and rejected, so both are kept and the smaller is taken.

        The bound is only honest over the animal's whole recording set. Every one of its terms falls when the
        planning host holds a subset of that set: prevalence divides the pooled regions by a minimum drawn from the
        recording count, and the pooled sum and the most populated recording are both taken over the recordings this
        host can read, so a subset bound can land far below the count the job goes on to attach. The dataset's own
        session list states which recordings the tracking spans, so a host that cannot measure any of them is refused
        rather than bounded from the ones it can.

        Completeness is judged on the geometries the bound is actually drawn from rather than on the session
        directories those geometries live under, because the two disagree. A session whose directory stands while its
        processed output has been removed passes a directory check and then contributes nothing to the pooled sum or
        to the recording count the prevalence divides, so it is dropped from the bound while the gate meant to guard
        that bound reports the set as whole.

        An entry yielding no geometry is an anomaly rather than a normal planning state. The acquisition system's
        admission policy holds a session out of a dataset until every pipeline it names has completed, and for a
        session type reaching this bound at all that set includes the two-photon pipeline, whose completion is what
        writes the output a geometry is read from. An entry therefore carried that output when it was admitted, and
        its absence now means the output was removed afterwards, which the operator remedies rather than the sizing
        pass estimating around.

        The three ways an entry fails to resolve are reported apart, because they are remedied apart: an absent
        session directory is staged back under the project root, a directory standing without readable output is
        restored or reprocessed through the two-photon pipeline, and a directory whose own session marker cannot be
        read is one whose transfer is incomplete or whose hierarchy holds more than one marker, which is repaired
        before any pipeline reads it.

    Args:
        dataset: The resolved dataset that holds the session.
        animal: The animal that owns the session.
        session: The session name whose tracked regions are resolved.
        project_root: The path to the project's root directory.
        configuration: The resolved multi-recording configuration, which reports the prevalence a cluster must reach.
        planned_roi_count: The regions to plan the multi-day columns for. Use None to accept the bound the animal's
            recording set carries. A stated count is the count the columns are charged, and the recording set is then
            neither gathered nor gated, since the bound those steps exist to protect is not drawn.

    Returns:
        The tracked region count the multi-day columns are charged.

    Raises:
        FileNotFoundError: If no count is stated and any session the dataset names for the animal yields no recording
            geometry, whether because its directory is absent under the project root, because that directory carries
            no readable processed imaging output, or because its own session marker cannot be read. The recording set
            this host can measure is then narrower than the set the tracking runs over, and the bound drawn from it
            would under-reserve the job.
    """
    # A caller that knows the count states it and is charged it, which mirrors the override cindra's single-recording
    # sizing takes in place of its own detection ceiling. The recording set is gathered for the bound alone, so a
    # stated count skips the gathering and the completeness gate guarding it together with the bound they serve.
    if planned_roi_count is not None:
        return planned_roi_count

    # Only the bound needs the recording set, and it needs all of it. The refusal mirrors the one the cross-recording
    # sizer raises for the same reason: a job reserved less memory than it goes on to hold is killed by the scheduler
    # and cancels every dependent scheduled behind it. An unresolved recording's own region count cannot be recovered
    # without its geometry, so there is nothing to estimate it from and the session is refused instead.
    entries = dataset.get_sessions_for_animal(animal=animal)
    absent: list[str] = []
    unreadable: list[str] = []
    unmarked: list[str] = []
    geometries: list[_RecordingGeometry] = []
    for entry in entries:
        # The geometry resolver loads the session's own marker to reach the output root, and that marker is gone along
        # with the session directory, so directory presence is settled here rather than inside the resolver's read.
        if not project_root.joinpath(animal, entry.session).is_dir():
            absent.append(entry.session)
            continue
        try:
            geometry = _resolve_recording_geometry(project_root=project_root, animal=animal, session=entry.session)
        except OSError:
            # A standing directory can still fail the marker load behind the resolver, which refuses a hierarchy
            # holding no session marker or more than one. That leaves the entry as unmeasured as an unreadable one,
            # so it is classified here rather than propagating a message that names neither this bound nor the entry.
            unmarked.append(entry.session)
            continue
        # The gate is placed on the geometry rather than on the directory that holds it, because the bound is summed
        # and counted over the geometries alone. A session counted as present while resolving no geometry is therefore
        # dropped from every term of the bound, which is exactly the under-estimate the gate exists to stop.
        if geometry is None:
            unreadable.append(entry.session)
            continue
        geometries.append(geometry)

    # Reported apart because they are remedied apart, and each clause names its own sessions so the operator does not
    # have to work out which of the three remedies each one needs.
    clauses: list[str] = []
    if absent:
        clauses.append(
            f"{len(absent)} of them are absent under the project root '{project_root}': {', '.join(sorted(absent))}"
        )
    if unreadable:
        clauses.append(
            f"{len(unreadable)} of them stand under the project root carrying no readable processed imaging output: "
            f"{', '.join(sorted(unreadable))}"
        )
    if unmarked:
        clauses.append(
            f"{len(unmarked)} of them stand under the project root carrying no readable session marker: "
            f"{', '.join(sorted(unmarked))}"
        )

    if clauses:
        message = (
            f"Unable to size the assembly job of session '{session}'. Its multi-day columns span the regions tracked "
            f"across animal '{animal}', the dataset names {len(entries)} session(s) for that animal, and the tracking "
            f"runs over every one of them, but {', and '.join(clauses)}. The tracking has not run when this job is "
            f"planned, so the tracked count can only be bounded from the geometries of that recording set, and the "
            f"bound falls when the set is narrowed, so bounding it from the geometries this host can read would "
            f"reserve the job less memory than it goes on to hold. Stage any absent session(s) under the project "
            f"root, restore or reprocess the two-photon output of any session that carries none, repair the "
            f"hierarchy of any session whose marker cannot be read, or rebuild the dataset without them, then plan "
            f"the session again."
        )
        console.error(message=message, error=FileNotFoundError)

    # Defensive: the forging job universe never reaches this floor. '_build_forging_universe' emits an assembly job
    # only for a session the dataset lists, so a specifier always resolves a real animal and a non-empty entry list.
    # The floor is kept for a caller reaching this pass directly with a specifier naming no session of the dataset,
    # which resolves no animal and therefore no recording set. That is a narrower answer rather than an
    # under-estimate of one: every named entry above resolved, so an empty list here is an empty entry list rather
    # than a set this host could not measure.
    if not geometries:
        return 1

    # The floor is load-bearing rather than defensive. A system donating no multi-recording configuration states no
    # prevalence at all, which reads as zero here and carries the ceiling to zero with it, so an unfloored divisor
    # divides by nothing. One recording is also the honest answer in that state: nothing states how many recordings a
    # cluster has to span, so the pooled term relaxes to the whole pooled sum, which the headroom term then caps for
    # any set pooling past the headroom its widest recording allows.
    prevalence = configuration.roi_tracking.mask_prevalence if configuration is not None else 0.0
    minimum_recordings = max(1, math.ceil(prevalence / _PERCENT_PER_FRACTION * len(geometries)))
    # Combinatorial: every template consumes at least the recordings a cluster must span, and consumes their regions
    # exclusively, so the pooled region count divided by that minimum ceilings the templates the tracking can keep.
    pooled_ceiling = sum(geometry.regions for geometry in geometries) // minimum_recordings
    # The domain reading of the same count: every cell the most populated recording detected, plus the headroom the
    # other recordings contribute through cells that recording does not hold.
    headroom_ceiling = math.ceil(max(geometry.regions for geometry in geometries) * _TRACKED_REGION_HEADROOM)
    return max(1, min(pooled_ceiling, headroom_ceiling))


def _size_forging_job(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
    cores: int,
    *,
    planned_roi_count: int | None = None,
) -> JobFootprint:
    """Sizes one per-session assembly job from the processed output it reads.

    Notes:
        The assembled frame retains every fluorescence column it attaches. The write that closes the job streams the
        frame it was handed rather than rebuilding it, so the shape of the session's own fluorescence is what the job
        is charged. The columns fall into two groups, each spanning its own region count. A single-day column spans
        every region the recording detected, while a multi-day column spans only the regions tracked across the animal.
        Loading a single-day column holds one column above those it has already attached, which is charged on top of
        them. The stage is this package's own, and its fan-out is a fixed handful of threads, so it runs at the
        allocation its type declared.

        The fluorescence is not the whole of what the job holds. An imaging assembly builds its behavior, runtime and
        video sub-datasets from the same per-source feathers a behavior-only assembly reads, and reads each of them at
        the clock that source was sampled on before interpolating it onto the reference clock. A camera faster than
        the fluorescence clock therefore contributes arrays taller than the frame the job builds, and charging the
        fluorescence clock alone drops that whole family. Those heights belong to the system that assembles the
        session, so they are read through that system's donated source resolver, which routes the source set by the
        session's own type.

        The frame itself stays on the fluorescence clock. The system's experiment assembler builds the fluorescence
        sub-dataset first and takes its ``time_us`` column as the reference every other sub-dataset is interpolated
        onto, so the samples the recording's own traces hold are the height of every retained column and of every
        assembled sub-dataset column, and no camera clock stands in for them here.

        A session carrying no fluorescence has nothing to assemble, so its refusal propagates rather than resolving
        to a floor. That holds for a session whose type must complete the two-photon pipeline before it joins a
        dataset at all. A type admitted without that pipeline records no imaging by design, and its assembly attaches
        no fluorescence column, so this model does not describe its job and the behavior-only model sizes it instead.

    Args:
        dataset: The resolved dataset that holds the session.
        animal: The animal that owns the session.
        session: The session name whose assembly job is sized.
        project_root: The path to the project's root directory.
        configuration: The dataset's resolved multi-recording configuration, or None when its acquisition system
            donates none.
        cores: The cores the job is allocated, which its type declares.
        planned_roi_count: The regions to plan the multi-day columns for. Use None to accept the bound the animal's
            recording set carries, which is what the forging pipeline itself passes.

    Returns:
        The job's footprint, holding the declared width and the memory the assembled frame holds at it.

    Raises:
        FileNotFoundError: If a session whose type requires imaging carries no processed imaging output, or if a
            session whose type requires none carries no reference clock, in which case the job assembling it cannot
            run either. Also raised when any session the dataset names for the session's animal yields no recording
            geometry, since the tracked-region bound the multi-day columns are charged spans the animal's whole
            recording set and would under-reserve the job if drawn from part of it.
        ValueError: If the dataset's acquisition system is unknown, if its session type falls outside the platform
            vocabulary, if that type joins no dataset for the system and therefore matches neither model, or if the
            system's own source resolver does not cover that type and therefore states none of the heights the job's
            sources stand at.
    """
    # The admission policy states which pipelines a session type must complete before it joins a dataset, so a type
    # that joins without the two-photon pipeline is one the assembler builds from behavior sources alone. Sizing it
    # from fluorescence would charge columns its job never attaches, so the model is selected by the policy rather
    # than falling back to it when imaging happens to be absent.
    if not _requires_imaging(dataset=dataset):
        return _size_behavior_assembly_job(
            system=dataset.acquisition_system, project_root=project_root, animal=animal, session=session, cores=cores
        )

    geometry = _resolve_recording_geometry(project_root=project_root, animal=animal, session=session)
    if geometry is None:
        message = (
            f"Unable to size the assembly job of session '{session}'. The session carries no processed imaging "
            f"output, so nothing states the shape of the frame the job assembles and the job could not run either."
        )
        console.error(message=message, error=FileNotFoundError)
    regions = _resolve_tracked_regions(
        dataset=dataset,
        animal=animal,
        session=session,
        project_root=project_root,
        configuration=configuration,
        planned_roi_count=planned_roi_count,
    )

    # Resolved through the system's own donation, and handed the same cached marker the recording geometry above was
    # resolved from, so the session's marker is read once for the whole estimate. A session whose sources the system
    # cannot resolve is refused by that donation rather than reported as sourceless, since a job charged no source
    # family at all is reserved a fraction of what it holds.
    resolve_assembly_sources = resolve_assembly_source_resolver(system=dataset.acquisition_system)
    source_samples = resolve_assembly_sources(
        session=_session_marker(project_root=project_root, animal=animal, session=session)
    )

    retained_regions = _ASSEMBLY_SINGLE_DAY_COLUMNS * geometry.regions + _ASSEMBLY_MULTI_DAY_COLUMNS * regions
    columns = retained_regions * _ASSEMBLY_WRITE_COPIES * geometry.samples * _SINGLE_PRECISION_BYTES
    load_transient = _ASSEMBLY_LOAD_TRANSIENT_COLUMNS * geometry.regions * geometry.samples * _SINGLE_PRECISION_BYTES
    sub_datasets = geometry.samples * _SUB_DATASET_BYTES_PER_SAMPLE
    source_arrays = sum(source_samples) * _SOURCE_INPUT_BYTES_PER_SAMPLE
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=columns + load_transient + sub_datasets + source_arrays)
        ),
    )


def _requires_imaging(dataset: DatasetData) -> bool:
    """Reports whether the sessions a dataset holds must carry two-photon output to have joined it.

    Notes:
        Read from the acquisition system's own admission policy rather than from a session-type literal, so a system
        that admits a behavior-only session type inherits the behavior-only estimate with no change here. A dataset
        holds one session type, which the shared hierarchy enforces both when a dataset is defined and when a session
        is added to one, so the policy is read once for the dataset rather than once per session.

        A session type the policy omits joins no dataset at all, so neither model describes the assembly of a
        dataset carrying it. Such a type is refused here rather than read as requiring no imaging, because this pass
        is public and does not itself run the admission check: a hand-made dataset marker naming an unadmitted type
        reaches it directly, and answering it with a behavior-only figure would reserve memory for a job the forging
        pipeline never plans.

    Args:
        dataset: The resolved dataset whose session type is examined.

    Returns:
        True when the dataset's session type must complete the two-photon pipeline before joining a dataset, and
        False when it joins carrying no imaging at all.

    Raises:
        ValueError: If the dataset's acquisition system is unknown, if its session type falls outside the platform
            vocabulary, or if that type joins no dataset for the system.
    """
    requirements = resolve_forging_admission_pipelines(system=dataset.acquisition_system)

    # An absent entry states that the type joins no dataset, which is a different answer from a type admitted while
    # requiring no imaging. Collapsing the two would size a dataset the pipeline refuses to forge, so the absence is
    # reported on the same terms the admission check reports it.
    required = requirements.get(SessionTypes(dataset.session_type))
    if required is None:
        admissible = ", ".join(sorted(str(session_type) for session_type in requirements))
        message = (
            f"Unable to size the assembly job of dataset '{dataset.name}'. Its session type "
            f"'{dataset.session_type}' joins no dataset for the '{dataset.acquisition_system}' acquisition system, "
            f"which admits the session type(s): {admissible}, so no model describes the job assembling it."
        )
        console.error(message=message, error=ValueError)

    return ProcessingPipelines.TWO_PHOTON in required


def _size_behavior_assembly_job(system: str, project_root: Path, animal: str, session: str, cores: int) -> JobFootprint:
    """Sizes one per-session assembly job of a session type that records no imaging.

    Notes:
        The assembly of such a session attaches no fluorescence column at all, so its whole data-dependent charge is
        the two families of arrays it holds while it runs, and those two are counted on different clocks.

        The first family is the frame it builds. Its behavior, runtime and video columns are all placed on the
        reference clock its acquisition system's own assembler settles on, so the samples that clock holds are the
        height of every one of them. The clock is settled after the columns are stacked and before they are clipped to
        the session bounds, so the job peaks at the full recorded height rather than at the height it writes.

        The second family is the sources those columns are interpolated from. Each source arrives at the clock it was
        sampled on, which is its own: a camera's timestamp, motion-energy and pupil feathers are read at that camera's
        frame count and only then interpolated onto the reference clock. A camera faster than the reference one
        therefore holds arrays taller than the frame the job builds, and a rig pairing a five-hundred-frame-a-second
        camera with a ten-frame-a-second one holds them fifty times taller. Charging the reference clock alone drops
        that family entirely and reserves a fraction of what the job holds, while charging every source at the widest
        clock instead of at its own overcharges the frame by that same ratio. Both terms are needed, and each is
        charged at the height its own arrays stand at.

        Which clock the frame is placed on, and which sources are read at all, belong to the system that assembles the
        session rather than to this pass, so both heights are read through that system's donated resolver.

        A session whose system settles on no reference clock has nothing to place its columns on, so the refusal the
        resolver raises propagates rather than resolving to a floor. That is the same answer the assembler gives for
        such a session, because both read one selection.

    Args:
        system: The acquisition system that recorded the dataset being forged, which donates the assembly-geometry
            resolver.
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose assembly job is sized.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory the assembled frame and its sources hold at it.

    Raises:
        FileNotFoundError: If the session carries no reference clock its system's assembler would settle on, in which
            case nothing states the height of the frame the job builds.
        ValueError: If the acquisition system is unknown.
    """
    resolve_assembly_geometry = resolve_assembly_geometry_resolver(system=system)
    geometry = resolve_assembly_geometry(
        session=_session_marker(project_root=project_root, animal=animal, session=session)
    )

    assembled_columns = geometry.reference_samples * _SUB_DATASET_BYTES_PER_SAMPLE
    source_arrays = sum(geometry.source_samples) * _SOURCE_INPUT_BYTES_PER_SAMPLE
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=assembled_columns + source_arrays)
        ),
    )
