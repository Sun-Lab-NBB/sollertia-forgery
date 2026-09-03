.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Dataset Forging
===============

.. automodule:: sollertia_forgery.forging
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, since the automodule directive above discovers module-level data
   through the source of the module it documents, and therefore skips a constant that this package re-exports. Each
   directive names the defining module rather than the re-exporting package, because autodoc reads the attribute
   docstring from that module's source and otherwise falls back to the docstring of the value's own type.
.. autodata:: sollertia_forgery.forging.state.DATASET_STATE_FILENAME
.. autodata:: sollertia_forgery.forging.pipeline.FORGING_JOB_NAME
.. autodata:: sollertia_forgery.forging.pipeline.FORGING_JOB_CONCURRENCY_LIMITS
.. autodata:: sollertia_forgery.forging.pipeline.MULTIDAY_DISCOVERY_JOB_NAME
.. autodata:: sollertia_forgery.forging.pipeline.MULTIDAY_EXTRACTION_JOB_NAME

Project Management
==================

.. automodule:: sollertia_forgery.managing
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.managing.checksum.CHECKSUM_JOB_NAME
.. autodata:: sollertia_forgery.managing.manifest.MANIFEST_JOB_NAME
.. autodata:: sollertia_forgery.managing.manifest.MANIFEST_AXES
.. autodata:: sollertia_forgery.managing.manifest.MANIFEST_SEMI_FIELDS

Runtime Log Processing
======================

.. automodule:: sollertia_forgery.runtime
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.runtime.pipeline.RUNTIME_JOB_NAME

Microcontroller Log Processing
==============================

.. automodule:: sollertia_forgery.microcontrollers
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.microcontrollers.pipeline.PARSE_JOB_NAME
.. autodata:: ataraxis_communication_interface.orchestration.jobs.CONTROLLER_EXTRACTION_JOB_NAME

Video Processing
================

.. automodule:: sollertia_forgery.video
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.video.motion_energy.MINIMUM_CHUNK_FRAMES
.. autodata:: sollertia_forgery.video.pipeline.ENERGY_JOB_NAME
.. autodata:: sollertia_forgery.video.pipeline.RENAME_JOB_NAME
.. autodata:: sollertia_forgery.video.pipeline.TRACKING_JOB_NAME
.. autodata:: ataraxis_video_system.orchestration.jobs.CAMERA_EXTRACTION_JOB_NAME

Two-Photon Processing
=====================

.. automodule:: sollertia_forgery.two_photon
   :members:
   :undoc-members:
   :show-inheritance:

Job Orchestration
=================

.. automodule:: sollertia_forgery.orchestration
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.orchestration.batches.OUTCOME_FILE_SUFFIX
.. autodata:: sollertia_forgery.orchestration.dispatch.BATCH_PIPELINES
.. autodata:: sollertia_forgery.orchestration.dispatch.SESSION_UNIT
.. autodata:: sollertia_forgery.orchestration.dispatch.DATASET_UNIT
.. autodata:: sollertia_forgery.orchestration.local.RESERVED_CORES
.. autodata:: sollertia_forgery.orchestration.planning.PROJECT_PLAN_SCHEMA
.. autodata:: sollertia_forgery.orchestration.reconcile.LOCAL_HOST_LABEL
.. autodata:: sollertia_forgery.orchestration.reconcile.REMOTE_HOST_LABEL
.. autodata:: sollertia_forgery.orchestration.remote.REMOTE_JOB_WALLTIME_MINUTES
.. autodata:: sollertia_forgery.orchestration.remote.HELD_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.SETTLED_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.GONE_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.RUNNING_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.FINISHED_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.FAILED_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.ABANDONED_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.STRANDED_ALLOCATION
.. autodata:: sollertia_forgery.orchestration.remote.NO_REMEDIATION
.. autodata:: sollertia_forgery.orchestration.remote.DROP_REMEDIATION
.. autodata:: sollertia_forgery.orchestration.remote.RESET_REMEDIATION
.. autodata:: sollertia_forgery.orchestration.remote.CANCEL_REMEDIATION
.. autodata:: sollertia_forgery.orchestration.remote.PROGRESSING_BATCH
.. autodata:: sollertia_forgery.orchestration.remote.STALLED_BATCH
.. autodata:: sollertia_forgery.orchestration.remote.AWAITING_CLOSURE_BATCH

Compute Server Transport
========================

.. automodule:: sollertia_forgery.server
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.server.server.TERMINAL_JOB_STATUSES

Agnostic Substrate
==================

.. automodule:: sollertia_forgery.shared_assets
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.shared_assets.pipelines.SESSION_PIPELINES

Command-Line Interface
======================

.. click:: sollertia_forgery.interfaces.entry_points:slf_cli
   :prog: slf
   :nested: full

Dispatch Registries
===================

.. automodule:: sollertia_forgery.registries
   :members:
   :undoc-members:
   :show-inheritance:

Mesoscope-VR Assets
===================

.. automodule:: sollertia_forgery.mesoscope_vr
   :members:
   :undoc-members:
   :show-inheritance:

.. Documents the column enumerations explicitly, since each names the columns carried by a forged dataset, and the
   automodule directive above reaches only what the package re-exports.
.. autoclass:: sollertia_forgery.mesoscope_vr.metadata.DatasetColumn
   :members:
   :undoc-members:
.. autoclass:: sollertia_forgery.mesoscope_vr.video_tracking.PupilColumn
   :members:
   :undoc-members:

.. Documents the package constants explicitly, for the reason given above the Dataset Forging directives.
.. autodata:: sollertia_forgery.mesoscope_vr.forging.MESOSCOPE_ADMISSION_PIPELINES
.. autodata:: sollertia_forgery.mesoscope_vr.metadata.MESOSCOPE_COLUMN_DESCRIPTIONS
.. autodata:: sollertia_forgery.mesoscope_vr.runtime.RUNTIME_SOURCE_ID
.. autodata:: sollertia_forgery.mesoscope_vr.two_photon.MESOSCOPE_MULTI_RECORDING_SESSION_TYPES
