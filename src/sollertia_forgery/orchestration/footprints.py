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
    parse_plane_specifier,
    size_multi_recording_job,
    size_single_recording_job,
    resolve_recording_geometry,
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
    sorted. It then materializes the whole table again as a dense matrix beside that sorted copy, and stacks one array
    per bodypart out of that matrix, which comes to a third table between them. The per-metric arrays the pupil stage
    derives from those points are charged in the margin this figure carries above the three copies it counts.
"""

_DECODER_BUFFER_MEMORY_MB: int = 96
"""The resident memory one decode worker holds for its codec reference frames and its decoder state, beyond the frame
buffers the measurement itself retains."""

_SPAWNED_SLF_CHILD_MEMORY_MB: int = 512
"""The resident memory one child spawned inside this package holds before it touches data.

Notes:
    A child spawned here re-imports this package, which carries OpenCV, NumPy, cindra, and the compiled kernels, so it
    costs several times the bare interpreter that ``SPAWNED_CHILD_MEMORY_MB`` charges. That figure describes a child
    holding nothing but the interpreter, and the library exporting it opens no pool of its own, so a pool whose
    children carry this package's own imports is measured here instead. A sixteen-worker job charged the bare figure
    was allocated 6 GB against a measured anonymous peak of 6.4 GB, which the scheduler's memory cgroup answered by
    killing it.
"""

_DEPENDENCY_CHILD_SURPLUS_MB: int = _SPAWNED_SLF_CHILD_MEMORY_MB - SPAWNED_CHILD_MEMORY_MB
"""The memory a dependency's own sizing pass leaves unmodeled for each child its stage spawns inside this package.

Notes:
    A dependency sizes its stage against the interpreter that dependency alone fills, which is the whole model while
    that library runs the stage itself. This package runs it instead, so every child the stage spawns re-imports this
    package rather than that library, and holds the difference measured here on top of the figure the dependency
    reported. The dependency's own figure stays correct for its own runtime and is still taken whole, with this
    surplus added beside it rather than replacing it.
"""

_RESIDENT_ESTIMATE_TOLERANCE: float = 1.10
"""The margin every resident estimate carries above the terms it sums.

Notes:
    A two-photon stage's mapped term is derived from the geometry the acquisition configured rather than measured off
    the binaries the conversion writes, so it lands within a couple of percent of the extent either way, while the
    cross-recording term is measured off those binaries and carries no such error. A stage whose mapped term dominates
    its anonymous one therefore inherits that error directly, and the conversion stage measured four percent of
    headroom without this margin. The margin holds every stage above ten percent, which is the floor a resident figure
    is held to because the host answers a shortfall by stalling the job rather than by failing it.
"""

_PLANE_MAPPING_STAGES: frozenset[SingleRecordingJobNames] = frozenset(
    {SingleRecordingJobNames.BINARIZE, SingleRecordingJobNames.REGISTER, SingleRecordingJobNames.PROCESS}
)
"""The two-photon stages that hold a plane binary memory-mapped for the whole of their run."""

_TWO_CHANNEL_COUNT: int = 2
"""The channels a recording holds when its acquisition configured a second one, which doubles every plane binary."""

_PLANE_BINARY_ELEMENT_BYTES: int = 2
"""The bytes one pixel occupies in a plane binary. The conversion stage builds every binary at the imaging library's
default element width whatever the width its source carried, so the extent a later stage maps follows this figure
rather than the source geometry's own."""

_PLANE_BINARY_PATTERN: str = "*_data.bin"
"""The glob matching the binarized plane files the imaging library writes, which are the files a cross-recording
extraction job maps."""

_SHARED_LIBRARY_IMAGE_MB: int = 256
"""The file-backed image one job holds resident, covering the interpreter and every shared object its import graph maps.

Notes:
    The pages are shared by every process of the job, so the scheduler's memory cgroup charges them once however wide
    the job runs. Measurement holds the figure between 206 MB and 221 MB across jobs running one, six, ten, and
    eighteen processes, which is what identifies the term as per-job rather than per-process. Summing the resident set
    of each process instead counts the same pages once per process, so a figure derived that way overstates a wide
    job by the width it runs at.
"""

_RETAINED_FRAME_BUFFERS: int = 2
"""The number of full-resolution single-precision frame buffers a decode worker keeps live. Binning produces a
strided view that retains its full-frame base, and the current and previous binned frames are live at once."""

_UNRESOLVED_FRAME_PIXELS: int = 0
"""The pixels charged to a motion-energy job whose camera left no recording this pass could read. The stage skips a
camera whose recording is absent and completes, so such a job still holds its decoders and is charged them rather than
being refused."""

_UNRESOLVED_FRAME_COUNT: int = -1
"""The frame count charged to a motion-energy job whose camera left no recording this pass could read. A container no
pass opened reports no length, and the length is what decides how many decoders the stage opens. Such a job is therefore
charged the full width its type declared rather than the single decoder a short recording earns. Any non-positive length
routes to that same full-width charge, whether it is this sentinel or a length a container reported, so the value needs
no distinguishing from a container's own answer."""

_CHECKSUM_CHUNK_MEMORY_MB: int = 8
"""The read buffer one checksum worker allocates, which is the fixed chunk through which the data-structures library
streams every file. The buffer is allocated once per file and reused for every chunk of it, so one worker holds one
buffer at a time and that buffer is the whole data-dependent term the worker carries."""

_CHECKSUM_READER_MEMORY_MB: int = _SPAWNED_SLF_CHILD_MEMORY_MB + _CHECKSUM_CHUNK_MEMORY_MB
"""The resident memory one checksum worker holds.

Notes:
    The pool that opens the workers is spawn-started inside the data-structures library this package calls, and a
    spawned child re-runs this package's own entry point as its main module, so each worker re-imports this package
    rather than the library exporting the bare spawned-child figure, and each holds its own read buffer above that.
    Charging the bare figure left an eight-worker job modeled at 3072 MB against a measured 2966 MB of anonymous
    memory, which is under four percent of headroom on the one term a scheduler kills a job over.
"""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a cindra trace array carries, which are its regions and its samples."""

_UNREADABLE_OUTPUT_ERRORS: tuple[type[Exception], ...] = (OSError, EOFError, LookupError, ValueError, BadZipFile)
"""The failures a recording's combined metadata archive raises when it stands on disk but cannot be read.

Notes:
    The archive is a compressed entry store, so a truncated, emptied, or overwritten one fails in the shape of
    whichever layer first reaches the damage. That layer is the file layer, the archive layer, or the entry lookup that
    expects the field from which the geometry is read. The five are unrelated types and none of them shares a base
    narrower than Exception, so the set is named here rather than approximated by one branch of it.

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
    contiguous copy is made straight off the memory map and only one copy of it is ever live. By the time those columns
    load, the single-day peak has already passed.

    It is added to the retained columns rather than compared against them, because the two peaks fall at different
    moments and the sum bounds both. The load peak carries no assembled sub-dataset yet, while the terminal peak carries
    every column and every sub-dataset but none of this transient. Adding therefore over-reserves the terminal peak by
    one single-day column, which is the safe direction. A model taking neither peak's part under-reserves the load
    whenever a session's single-day column stands above its four multi-day columns and its sub-datasets together.
    Tracking prunes hard, so that is the ordinary case rather than the extreme one. A recording detecting two thousand
    regions of which one hundred are tracked holds one single-day column of eight hundred megabytes against four
    multi-day columns of a hundred and sixty and a sub-dataset term of fifty. The fifteen percent tolerance and the
    gigabyte rounding cannot absorb that difference.
"""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the assembled behavior, runtime, and video columns hold per sample of the reference clock on which they
are placed. Each sub-dataset emits one array per column at that height, the interpolation that lifts a column onto the
clock holds double-precision transients there, and the clip that closes the assembly copies the stacked frame once.
The figure charges the assembly's output alone. The source arrays from which those columns are interpolated sit at
their own clocks and are charged separately."""

_SOURCE_INPUT_BYTES_PER_SAMPLE: int = 128
"""The memory an assembly holds per sample of one input source, at that source's own clock rather than at the clock on
which the assembled frame is placed. The widest source is the camera carrying the eye, which is read through three
feathers of one height at once. Those are its timestamp column at eight bytes a frame, its two single-precision
motion-energy columns at eight, and its nineteen pupil columns at seventy. The interpolation that lifts one of those
columns onto the reference clock holds a double-precision copy of that column and of the timestamps beside it, which
adds sixteen. That comes to a hundred and two bytes a frame, which this figure rounds up, and every other source an
assembly reads is narrower."""

_PERCENT_PER_FRACTION: float = 100.0
"""The divisor converting a percentage into a fraction."""

# This multiple belongs to the model this library keeps for its own assembly job, which is charged the multi-day
# columns the assembled frame holds. A separate model in cindra answers the same question for that library's own
# multi-day tracking jobs. The two figures agreeing is the same domain reasoning reached twice rather than either
# copying the other, so retuning one carries no obligation to retune the other.
_TRACKED_REGION_HEADROOM: float = 1.5
"""The multiple of the most populated recording's region count that ceilings the templates multi-day tracking keeps.

Notes:
    The multiple is a domain assumption about how a dataset's recordings overlap rather than a figure derived from the
    pipeline. A tracked dataset holds at most every region of its most populated recording, plus about half that count
    again contributed by regions the other recordings hold and it does not.
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

        The mapped term is the one field a dependency's record does not carry, because a dependency sizes the memory
        its stage allocates rather than the pages the host holds resident behind it. It stays zero for every stage
        that reads its input through the file interface. A stage that maps its input states the bytes it holds mapped
        at its peak, which for a stage opening one array at a time is that array rather than the set it reads.
    """

    cores: int
    """The cores the job occupies."""
    memory_mb: int
    """The anonymous memory the job holds at its peak, in megabytes, which is the term a local pool schedules on."""
    mapped_mb: int = 0
    """The file-backed memory the job maps, in megabytes, which stays resident behind it beside the anonymous term."""

    @property
    def resident_mb(self) -> int:
        """Returns the peak resident memory the job holds, in megabytes, which is the term SLURM schedules on."""
        summed = self.memory_mb + self.mapped_mb + _SHARED_LIBRARY_IMAGE_MB
        return _round_to_gigabyte(memory_mb=int(summed * _RESIDENT_ESTIMATE_TOLERANCE))


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
        the extraction job it is scheduled behind has yet to write. The archive from which that table comes attributes
        no message to a module, so every parse job of one controller is charged that whole archive. The figure is a
        bound those jobs share, and the model that charges it names what it would take to make it per-job.

        Every term is read from the session's raw acquisition data, so a footprint is available before any stage has
        run. Each footprint carries two figures. The anonymous one covers the memory a job allocates, which is the
        term that forces a host to swap and a scheduler to kill a job, and a local process pool budgets against it.
        The resident one adds the pages a mapping stage holds, which a scheduler is given because it packs a node by
        what each allocation declares and reclaims the shortfall from a job declaring less than it holds.

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
            more than one archive for a source or more than one camera manifest. Raised as well when a camera manifest
            registers no camera, when the session's pose prediction states no table width, or when a job name routes to
            no sizing model.
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
        decodes no fluorescence and materializes no timestamp column, reading at most a camera clock's two endpoints.
        Each cross-recording stage scales with the processed data the single-recording pipeline wrote for the sessions
        that carry two-photon data.

        Every job is routed to a model rather than to a blanket allowance, since a remote scheduler reserves memory
        per job. The two cross-recording stages belong to cindra, so the cores and the anonymous memory of their
        figures come from cindra's own sizing pass, while the bytes the extraction stage maps are measured here off
        the recording's binaries. That pass refuses a dataset that any recording leaves short rather than sizing it
        from the recordings that happen to be complete, and the refusal propagates, because a stage cindra will not
        size is a stage the dataset cannot run until its recordings are complete. The recording set handed to that
        pass is the set the dataset names. An animal whose sessions this host does not all hold is refused here on the
        same terms rather than measured over the subset the host happens to carry.

        The per-session assembly stage is this package's own, so no dependency models it and its projection stays
        here. Its width holds one value whatever data it reads, so it reports the allocation its type declared. Which
        of its two models applies follows from the acquisition system's admission policy. A session type that joins a
        dataset without completing the two-photon pipeline is assembled from its behavior sources onto a camera clock.
        It is therefore sized from that clock and from the heights at which those sources stand rather than from
        fluorescence it never recorded. Both models charge that source family, since an assembly reads the same
        per-source feathers at the same per-source clocks, whatever the clock on which it goes on to place its frame.
        The two differ in the frame alone.

        The one figure the assembly model cannot read from data this pass consumes is the count of regions the
        multi-day tracking goes on to keep. That tracking is planned in the same universe as the jobs sized here, so it
        has not run. It is bounded from the animal's recording set instead. A caller that reasonably knows the count
        its animals carry states it and is charged it, which is the same override cindra's own sizing takes in place of
        the bound it would otherwise draw. The count is stated for the whole batch rather than per job, since one
        dataset's tracking runs at one configuration.

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
            the dataset names for it. The set this host can measure is then narrower than the set over which the job
            runs, and sizing it would under-reserve the job. An assembly job raises it on those same terms while no
            region count is stated, since the bound it would otherwise draw spans that whole set.
        ValueError: If the dataset's acquisition system donates no multi-recording configuration, if a job name
            routes to no sizing model, or if the dataset's session type joins no dataset for its acquisition system.
            Also raised if a stated region count is not a positive integer, which cindra's own sizing refuses for the
            cross-recording stages to which the count is handed.
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


def _dependency_child_surplus(cores: int) -> int:
    """Resolves the memory a dependency-sized stage holds beyond its own figure when this package runs it.

    Notes:
        A stage running at a single core opens no pool, so it spawns no child and holds nothing beyond what the
        dependency modeled. A wider stage opens one child per allocated core and keeps its own reader alongside them,
        which is the same shape every dependency here models, so the surplus covers one process more than the cores.

    Args:
        cores: The cores the dependency's sizing pass picked for the stage.

    Returns:
        The memory to add to the dependency's own figure, in megabytes.
    """
    if cores <= 1:
        return 0
    return (cores + 1) * _DEPENDENCY_CHILD_SURPLUS_MB


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
    return JobFootprint(
        cores=sizing.cores,
        memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb + _dependency_child_surplus(cores=sizing.cores)),
    )


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
    return JobFootprint(
        cores=sizing.cores,
        memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb + _dependency_child_surplus(cores=sizing.cores)),
    )


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
        own recording is the only input it reads. Each job is therefore charged the frame of that recording rather than
        the widest frame the session holds. The two figures diverge by the ratio between the session's cameras, which
        for a rig pairing a fast small-sensor camera with a slow large-sensor one is most of the reservation.

        The colloquial name that locates a camera's recording on disk lives in the acquisition-time camera manifest, so
        the source identifiers the jobs carry are mapped through the video library's own resolver rather than through a
        camera vocabulary rebuilt here. One read of that manifest answers every motion-energy job of the session. The
        recording it names is then resolved through the same resolver the stage itself calls, so the container this pass
        measures is the container the job opens.

        Each recording is opened once, for the one job that decodes it, so the pass reads one container per job rather
        than one per recording the session holds. It reads none at all for a session that plans no motion-energy job.
        Both figures the model needs come off that one open. The frame decides what a decoder holds, and the length
        decides how many decoders the stage opens. The length is therefore read where the frame already is rather than
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
        frame of the one recording this job decodes. A session pairing cameras of different sensors therefore reserves
        each of their jobs the memory that job holds rather than reserving every one of them the widest camera's frame.

        The decoders charged are the decoders the stage opens rather than the cores the job was allocated. The stage
        splits its recording into chunks no shorter than the shared minimum and opens one worker per chunk. A recording
        holding fewer frames than that minimum times the allocation therefore opens fewer workers than the allocation
        names. One holding fewer than the minimum itself decodes in the job's own process and opens no pool at all. A
        short clip therefore holds one decoder and no spawned child, which for a rig recording calibration and training
        clips beside full sessions is most of what charging the allocation would have reserved. The allocation still
        bounds the chunk count, so it is the width at which the estimate assumes the job runs.

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

    per_worker = frame_buffers + _DECODER_BUFFER_MEMORY_MB + _SPAWNED_SLF_CHILD_MEMORY_MB
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
        from what it reads. The combination stage takes its frame count and its channel count from that same geometry,
        while its region term alone is a configuration ceiling, because no recording states its region count until the
        detection stage has run. What cindra charges for that term is therefore the ceiling the recording's own
        configuration allows, which is the multiplier the detection loop applies to its configured iteration limit,
        taken across the recording's planes. That one term stands above what a recording detects by whatever margin its
        configured limit stands above its data.

        The bound stands because the region count is not a raw input. It is written by a stage scheduled ahead of this
        one, so at plan time no reading of the recording answers it, and a plan has to size every stage before any of
        them runs. Passing cindra a measured count off a recording whose combination output already exists is what
        would turn this into a per-recording estimate. It would additionally have to gate on the detection settings
        being unchanged, since a re-detection under widened settings can exceed the count the previous pass recorded.

        A stage cindra cannot size is refused, either because the recording carries no readable raw imaging data or
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
    return JobFootprint(
        cores=sizing.cores,
        memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb),
        mapped_mb=_mapped_plane_megabytes(
            job_name=job_name,
            specifier=specifier,
            output_root=output_root,
            configuration=configuration,
            data_path=data_path,
        ),
    )


def _mapped_plane_megabytes(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> int:
    """Resolves the plane binaries one two-photon stage memory-maps, in megabytes.

    Notes:
        The three stages that hold a plane binary open map every byte of it. The conversion stage creates one map per
        plane and writes the whole recording through it, and the two per-plane stages map the one plane their
        specifier names. A stage denied those pages stalls, because the host answers the shortfall by writing dirty
        pages back
        synchronously. Measurement puts the conversion stage at 67 seconds while it holds its binaries. The same stage
        charged a
        gigabyte takes 58 minutes to 2 hours for the same five and a half minutes of processor time, which makes
        these bytes a demand rather than a convenience.

        The extent is derived from the acquisition metadata and one source file header, so it answers before the
        conversion stage has written anything. That reading reports the frame count the acquisition configured, which
        stands about two percent above the frames a recording actually lands, so the figure leans high by that margin.

        The combination stage reads the plane binaries a plane at a time rather than holding them, so it maps none of
        them and is charged nothing here.

    Args:
        job_name: The tracker job name identifying the stage.
        specifier: The plane specifier for a per-plane stage, empty for the whole-recording stages.
        output_root: The output root the recording's cindra configuration was given.
        configuration: The recording's resolved processing configuration, whose excluded source files the geometry
            read leaves out, so this figure spans the frames the conversion stage actually writes.
        data_path: The raw imaging directory from which the geometry is read.

    Returns:
        The megabytes the stage maps, which is zero for every stage that maps nothing.
    """
    stage = SingleRecordingJobNames(job_name)
    if stage not in _PLANE_MAPPING_STAGES:
        return 0

    # An unreadable acquisition resolves to no plane at all, so the sum below answers zero without a guard of its own.
    geometry = resolve_recording_geometry(
        output_root=output_root,
        data_path=data_path,
        ignored_file_names=tuple(configuration.file_io.ignored_file_names),
    )
    index = parse_plane_specifier(specifier=specifier)
    planes = tuple(plane for plane in geometry.planes if index is None or plane.index == index)
    channels = _TWO_CHANNEL_COUNT if geometry.two_channels else 1
    mapped = sum(
        plane.height * plane.width * plane.frame_count * _PLANE_BINARY_ELEMENT_BYTES * channels for plane in planes
    )
    return _bytes_to_megabytes(byte_count=mapped)


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
        is handed to cindra whichever stage is being sized. Both halves of cindra's own figure come back from that one
        read, and the mapped term is resolved here from the
        plane binaries the recordings hold.

        A set that any recording leaves short draws a refusal from cindra rather than a figure sized from the
        recordings that happen to be complete, and that refusal propagates. A dataset whose recordings carry no
        combined output cannot run either stage yet, so it is dropped from the workflow rather than planned at a floor.

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
    return JobFootprint(
        cores=sizing.cores,
        memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb),
        mapped_mb=_mapped_recording_megabytes(
            job_name=job_name, specifier=specifier, directories=recording_directories
        ),
    )


def _size_pose_tracking_job(session: SessionData, cores: int) -> JobFootprint:
    """Sizes one pose-tracking job from the prediction file its session carries.

    Notes:
        The stage reads predictions written upstream and never runs inference, so its working set follows the table it
        reads. The file is named by the system that produces it, so it is resolved through that system's donated
        locator. That locator answers with the file that the tracking worker opens. The stage is this package's own and
        its own fan-out is fixed, so it runs at the allocation its type declared.

        The table is charged at the width it holds rather than at the size its file occupies on disk. The two agree
        only for a prediction the writer left uncompressed. A compressed one occupies a fraction of its table and is
        still expanded to full width the moment the stage reads it, so charging the file would understate exactly the
        prediction most able to exhaust the node. The row and column counts come from the table's own metadata, so
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
        any length. The stage's working set scales with those counts, and unlike the size the file occupies on disk
        they do not move with whatever compression the writer applied.

        The stage reads the prediction through a call that names no key, which requires the file to hold exactly one
        frame. A file holding another number of them is therefore refused here on the terms the stage would refuse it.

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
        the same whole archive. Their reservations therefore sum to that archive once per module, while the tables they
        read sum to at most one archive between them. A controller pairing a continuously polled module with a sparse
        event-driven one therefore reserves the sparse module's job the memory the polled module's job holds. The
        bound is a per-job figure only where the controller carries a single parsed module.

        The bound stands because no per-module figure exists at plan time. A parse job reads the raw per-module feather
        its controller's extraction job writes, and it is scheduled behind that job, so its own input is absent while
        the plan is made. The archive from which that input is extracted is present but states nothing about the module.
        The archive's index names each message by the recording source and the acquisition timestamp alone, and a
        message names its module only inside its own body. Attributing messages to modules therefore costs one payload
        read per message, which is the pass the extraction job itself makes. The event codes the acquisition system
        registers do not separate the modules either, since the modules of one controller reuse them.

        A per-module message count carried by the archive index is what would turn this into a per-job estimate. The
        library that assembles the archive is the one that could carry each message's module into its entry name or into
        an index beside it. This model would then charge each job the share its own module holds.

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
        and its assembly sources through that donation. A marker read twice for one session is one YAML read per
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
        when the job actually runs. A donation therefore reads the locations its assembler reads rather than locations
        this pass resolved on its behalf.

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


def _mapped_recording_megabytes(job_name: MultiRecordingJobNames, specifier: str, directories: tuple[Path, ...]) -> int:
    """Resolves the binarized imaging data one cross-recording job maps, in megabytes.

    Notes:
        The extraction stage reads its recording's plane binaries through a memory map, so every byte of them stays
        resident behind the job while it runs. Measurement puts the resident file-backed term of that stage at 99% of
        those binaries, against an anonymous term under a tenth of it. The mapped bytes are therefore the figure that
        decides how many of these jobs a host schedules at once.

        The discovery stage reads the combined traces of every recording it spans through the file interface, so it
        maps none of them and is charged nothing here. A recording whose binaries have been removed after
        processing is the one input this pass cannot recover, since the stage that wrote them belongs to a pipeline
        that has already completed. Charging it nothing would reserve a tenth of what the job holds, so it is refused
        here on the terms every other unreadable input is refused.

    Args:
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a session for the extraction stage.
        directories: The cindra output directory of every recording the job spans.

    Returns:
        The megabytes the job maps, which is zero for every stage that maps nothing.
    """
    if job_name is not MultiRecordingJobNames.EXTRACT:
        return 0
    mapped = [directory for directory in directories if specifier in directory.parts]
    total = sum(binary.stat().st_size for directory in mapped for binary in directory.rglob(_PLANE_BINARY_PATTERN))
    return _bytes_to_megabytes(byte_count=total)


def _animal_recording_directories(dataset: DatasetData, animal: str, project_root: Path) -> tuple[Path, ...]:
    """Resolves the cindra output directory of every recording one animal contributes to a dataset.

    Notes:
        This is the recording set named by the animal's materialized multi-recording configuration, resolved from the
        project root rather than read back from that file, so a dataset whose configurations have not been written
        yet is still sizable. The dataset's own session list states which recordings that configuration names, because
        the configuration is written from every session the dataset holds for the animal, and the job that reads it
        back runs over every directory it names.

        A planning host that holds fewer of those sessions than the dataset names is refused rather than sized from
        the ones it holds. The discovery stage is quadratic in the pooled region count of the set over which it runs,
        so an animal planned from half its sessions is reserved a quarter of what the job goes on to hold. A job that
        is reserved less than it holds is killed by the scheduler and cancels every dependent scheduled behind it. The
        refusal names the sessions the host is missing, which are the ones to stage before the animal is planned.

    Args:
        dataset: The resolved dataset that holds the animal.
        animal: The animal whose recordings are resolved.
        project_root: The path to the project's root directory.

    Returns:
        The cindra output directory of every recording the dataset names for the animal.

    Raises:
        FileNotFoundError: If the project root holds fewer of the animal's dataset sessions than the dataset names, in
            which case the set this host can measure is narrower than the set over which the job runs.
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

        The sample count is not taken from the same reader, because the frame count it reports is read from the combined
        metadata archive while a sample is a column of the trace array. The frame count reaches the archive only from
        the cindra release that introduced it, and that release documents a missing one as reading back zero. A
        recording processed by an earlier release therefore reports no frames while its traces stand at their full
        width. The assembly model multiplies every retained column by the sample count, so accepting that zero would
        collapse the whole fluorescence term for exactly the recordings that are oldest and least likely to be
        reprocessed. The traces are therefore measured at their own header, which reports what the assembly will
        actually load.

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

    # The reader resolves a geometry from the metadata archive alone, and it reports a zero region count for a
    # recording whose trace array is absent or carries another rank. The traces are therefore gated here rather than
    # left to answer a region count of zero. That gate is the sample count's own read, which the reader does not
    # supply.
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
        every template consumes at least that many regions and consumes them exclusively. The pooled term is the tighter
        one over few recordings, where two recordings of a thousand regions at a minimum of two admit a thousand
        templates against the fifteen hundred the headroom term allows. The headroom term is the tighter one over many,
        where twenty recordings of fifteen thousand regions at half prevalence pool to thirty thousand templates against
        a real ceiling near fifteen thousand. That pooled count is twice the memory such an animal holds. Both ceilings
        are therefore kept, and the smaller of the two is taken.

        The bound is only honest over the animal's whole recording set. Every one of its terms falls when the
        planning host holds a subset of that set. Prevalence divides the pooled regions by a minimum drawn from the
        recording count, and the pooled sum and the most populated recording are both taken over the recordings this
        host can read. A subset bound can therefore land far below the count the job goes on to attach. The dataset's
        own session list states which recordings the tracking spans, so a host that cannot measure any of them is
        refused rather than bounded from the ones it can.

        Completeness is judged on the geometries the bound is actually drawn from rather than on the session
        directories those geometries live under, because the two disagree. A session whose directory stands while its
        processed output has been removed passes a directory check and then contributes nothing to the pooled sum or
        to the recording count the prevalence divides. It is dropped from the bound, while the gate meant to guard
        that bound reports the set as whole.

        An entry yielding no geometry is an anomaly rather than a normal planning state. The acquisition system's
        admission policy holds a session out of a dataset until every pipeline it names has completed. For a session
        type reaching this bound at all, that set includes the two-photon pipeline, whose completion is what writes the
        output from which a geometry is read. An entry therefore carried that output when it was admitted, and
        its absence now means the output was removed afterwards, which the operator remedies rather than the sizing
        pass estimating around.

        The three ways an entry fails to resolve are reported apart, because they are remedied apart. An absent
        session directory is staged back under the project root, and a directory standing without readable output is
        restored or reprocessed through the two-photon pipeline. A directory whose own session marker cannot be read
        is one whose transfer is incomplete or whose hierarchy holds more than one marker, which is repaired before
        any pipeline reads it.

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
            geometry. That happens when its directory is absent under the project root, when that directory carries no
            readable processed imaging output, or when its own session marker cannot be read. The recording set this
            host can measure is then narrower than the set over which the tracking runs, and the bound drawn from it
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

    # Defensive: the forging job universe never reaches this floor. ``_build_forging_universe`` emits an assembly job
    # only for a session the dataset lists, so a specifier always resolves a real animal and a non-empty entry list. The
    # floor is kept for a caller reaching this pass directly with a specifier naming no session of the dataset, which
    # resolves no animal and therefore no recording set. That is a narrower answer rather than an under-estimate of one:
    # every named entry above resolved, so an empty list here is an empty entry list rather than a set this host could
    # not measure.
    if not geometries:
        return 1

    # The floor is load-bearing rather than defensive. A system donating no multi-recording configuration states no
    # prevalence at all, which reads as zero here and carries the ceiling to zero with it, so an unfloored divisor
    # divides by nothing. One recording is also the honest answer in that state. Nothing states how many recordings a
    # cluster has to span, so the pooled term relaxes to the whole pooled sum. The headroom term then caps that sum for
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
        video sub-datasets from the same per-source feathers a behavior-only assembly reads. It reads each of them at
        the clock on which that source was sampled, before interpolating it onto the reference clock. A camera faster
        than the fluorescence clock therefore contributes arrays taller than the frame the job builds, and charging the
        fluorescence clock alone drops that whole family. Those heights belong to the system that assembles the session,
        so they are read through that system's donated source resolver, which routes the source set by the session's own
        type.

        The frame itself stays on the fluorescence clock. The system's experiment assembler builds the fluorescence
        sub-dataset first and takes its ``time_us`` column as the reference onto which every other sub-dataset is
        interpolated. The samples the recording's own traces hold are therefore the height of every retained column and
        of every assembled sub-dataset column, and no camera clock stands in for them here.

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
            geometry. The tracked-region bound the multi-day columns are charged spans the animal's whole recording
            set, and it would under-reserve the job if drawn from part of it.
        ValueError: If the dataset's acquisition system is unknown, if its session type falls outside the platform
            vocabulary, or if that type joins no dataset for the system and therefore matches neither model. Also
            raised if the system's own source resolver does not cover that type and therefore states none of the
            heights at which the job's sources stand.
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
    # The stage opens one trace array at a time as a map and selects its rows out of it, so the pages it holds are that
    # one array rather than every array it reads. Measurement puts the stage's file-backed peak at that array plus the
    # library image every job holds, which is what identifies the extent as one array rather than the set.
    mapped = geometry.regions * geometry.samples * _SINGLE_PRECISION_BYTES
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=columns + load_transient + sub_datasets + source_arrays)
        ),
        mapped_mb=_bytes_to_megabytes(byte_count=mapped),
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
        is public and does not itself run the admission check. A hand-made dataset marker naming an unadmitted type
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

        The first family is the frame it builds. Its behavior, runtime and video columns are all placed on the reference
        clock on which its acquisition system's own assembler settles, so the samples that clock holds are the height of
        every one of them. The clock is settled after the columns are stacked and before they are clipped to the session
        bounds, so the job peaks at the full recorded height rather than at the height it writes.

        The second family is the sources from which those columns are interpolated. Each source arrives at the clock on
        which it was sampled, which is its own: a camera's timestamp, motion-energy and pupil feathers are read at that
        camera's frame count and only then interpolated onto the reference clock. A camera faster than the reference one
        therefore holds arrays taller than the frame the job builds, and a rig pairing a five-hundred-frame-a-second
        camera with a ten-frame-a-second one holds them fifty times taller. Charging the reference clock alone drops
        that family entirely and reserves a fraction of what the job holds, while charging every source at the widest
        clock instead of at its own overcharges the frame by that same ratio. Both terms are needed, and each is charged
        at the height at which its own arrays stand.

        The clock on which the frame is placed, and the sources that are read at all, belong to the system that
        assembles the session rather than to this pass, so both heights are read through that system's donated
        resolver.

        A session whose system settles on no reference clock has no clock on which to place its columns, so the refusal
        the resolver raises propagates rather than resolving to a floor. That is the same answer the assembler gives for
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
        FileNotFoundError: If the session carries no reference clock on which its system's assembler would settle, in
            which case nothing states the height of the frame the job builds.
        ValueError: If the acquisition system is unknown, or if the system's own resolver does not cover the session's
            type and therefore states none of the heights at which the job's sources stand.
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
