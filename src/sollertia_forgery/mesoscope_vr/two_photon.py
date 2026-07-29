"""Provides the Mesoscope-VR two-photon pipeline assets donated to the system-agnostic two-photon and forging worker
packages.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
from cindra.dataclasses import (
    Main,
    FileIO,
    ROIDetection,
    Registration,
    BaselineMethod,
    SignalExtraction,
    ReferenceImageType,
    SpikeDeconvolution,
    NonrigidRegistration,
    OnePhotonRegistration,
)
from ataraxis_base_utilities import console
from sollertia_shared_assets import SurgeryData, SessionTypes, MesoscopeDirectories
from cindra.dataclasses.multi_recording_configuration import (
    ROITracking,
    RecordingIO,
    ROISelection,
    DiffeomorphicRegistration,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData


class _CalciumIndicator(StrEnum):
    """Enumerates the calcium indicators the Mesoscope-VR cindra configurations are tuned for.

    Notes:
        The indicator sets the OASIS decay ``tau``, the neuropil coefficient, and the multi-recording ROI-selection
        probability threshold. It is resolved from the animal's genotype through ``_resolve_calcium_indicator``.
    """

    GCAMP6F = "GCaMP6f"
    """The Thy1-GCaMP6f transgenic line (GP5.17)."""
    JGCAMP8S = "jGCaMP8s"
    """The in-house cross of a jGCaMP8s reporter line and a CaMKII-Cre driver line (GCaMP8s x CamKIICre), the slow-decay
    member of the jGCaMP8 family."""


@dataclass(frozen=True, slots=True)
class _IndicatorParameters:
    """Bundles the cindra configuration parameters that depend on the calcium indicator.

    Notes:
        ``neuropil_coefficient`` is shared by both the single- and multi-recording configurations, ``tau`` applies only
        to the single-recording configuration, and ``probability_threshold`` applies only to the multi-recording
        configuration.
    """

    tau: float
    """The single-recording OASIS AR(1) sensor decay time constant, in seconds (``main.tau``)."""
    neuropil_coefficient: float
    """The neuropil-subtraction coefficient shared by both configurations
    (``spike_deconvolution.neuropil_coefficient``)."""
    probability_threshold: float
    """The multi-recording ROI-selection cell-probability threshold (``roi_selection.probability_threshold``)."""


_INDICATOR_PARAMETERS: dict[_CalciumIndicator, _IndicatorParameters] = {
    _CalciumIndicator.GCAMP6F: _IndicatorParameters(tau=0.4, neuropil_coefficient=0.7, probability_threshold=0.85),
    _CalciumIndicator.JGCAMP8S: _IndicatorParameters(tau=0.7, neuropil_coefficient=0.8, probability_threshold=0.80),
}
"""Maps each calcium indicator to its indicator-dependent cindra parameters. Every ``_CalciumIndicator`` member must
appear here, which ``_assert_indicator_coverage`` enforces at import time."""

_GENOTYPE_INDICATOR_REGISTRY: dict[str, _CalciumIndicator] = {
    "gp5.17": _CalciumIndicator.GCAMP6F,
    "gp5.17 (hemi)": _CalciumIndicator.GCAMP6F,
    "gcamp8s x camkiicre": _CalciumIndicator.JGCAMP8S,
}
"""Maps each recognized normalized genotype string to its calcium indicator. Keys are matched against the genotype
normalized by ``_resolve_calcium_indicator`` (casefolded, whitespace collapsed). Zygosity qualifiers are recorded as
their own keys, so a hemizygous line resolves to the same indicator as the homozygous line while the surgery metadata
keeps the distinction. Only the two indicators used with the reference mesoscope-vr system are recognized, so an
unrecognized genotype fails loudly rather than defaulting to a possibly-wrong sensor."""


def locate_two_photon_data(session: SessionData) -> Path:
    """Resolves the Mesoscope-VR session's raw two-photon imaging directory, which is the input to the cindra pipeline.

    Args:
        session: The loaded session whose raw two-photon imaging directory is resolved.

    Returns:
        The path to the session's ``mesoscope_data`` directory under its raw-data root. This directory stores the
        compressed 2-Photon Random Access Mesoscope (2P-RAM) acquisition output and accompanying metadata that the
        cindra single-recording pipeline consumes.
    """
    return session.raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA)


def resolve_single_recording_configuration(session: SessionData) -> SingleRecordingConfiguration:
    """Resolves the single-recording cindra configuration for a Mesoscope-VR session.

    Notes:
        The configuration is selected from the animal's genotype, read from the session's surgery metadata, so a
        GCaMP6f and a jGCaMP8s recording receive their indicator-tuned parameters automatically.

    Args:
        session: The loaded session whose single-recording configuration is resolved.

    Returns:
        The single-recording configuration for the session's calcium indicator.

    Raises:
        FileNotFoundError: If the session's surgery metadata file is missing.
        ValueError: If the animal's genotype does not map to a recognized calcium indicator.
    """
    return _build_single_recording_configuration(_read_session_genotype(session))


def resolve_multi_recording_configuration(session: SessionData) -> MultiRecordingConfiguration | None:
    """Resolves the multi-recording cindra configuration for a Mesoscope-VR session.

    Notes:
        Mesoscope-VR tracks cells across recordings only for experiment sessions, so a session of any other type
        returns no configuration. When it applies, the configuration is selected from the animal's genotype, read from
        the session's surgery metadata.

    Args:
        session: The loaded session whose multi-recording configuration is resolved.

    Returns:
        The multi-recording configuration for the session's calcium indicator, or None when the session is not a
        mesoscope experiment session and therefore carries no cross-recording cell tracking.

    Raises:
        FileNotFoundError: If the session's surgery metadata file is missing.
        ValueError: If the animal's genotype does not map to a recognized calcium indicator.
    """
    if session.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
        return None
    return _build_multi_recording_configuration(_read_session_genotype(session))


def _resolve_calcium_indicator(genotype: str) -> _CalciumIndicator:
    """Resolves an animal's genotype string to the calcium indicator its cindra configuration is tuned for.

    Notes:
        The genotype is normalized before matching. Normalization casefolds the string, strips surrounding whitespace,
        and collapses internal whitespace runs to one space. Zygosity qualifiers such as the ``(hemi)`` in
        ``GP5.17 (hemi)`` are preserved and carry their own registry key. The normalized string is then matched
        exactly against the recognized genotypes, so a jGCaMP8f or jGCaMP8m line does not silently resolve to the
        jGCaMP8s configuration.

    Args:
        genotype: The animal's genotype, read from the ``subject.genotype`` field of its surgery metadata.

    Returns:
        The calcium indicator the genotype maps to.

    Raises:
        ValueError: If the genotype does not match a recognized calcium indicator.
    """
    normalized = re.sub(pattern=r"\s+", repl=" ", string=genotype.strip().casefold())

    indicator = _GENOTYPE_INDICATOR_REGISTRY.get(normalized)
    if indicator is None:
        recognized = ", ".join(sorted(_GENOTYPE_INDICATOR_REGISTRY))
        message = (
            f"Unable to resolve the calcium indicator for the genotype '{genotype}'. The genotype normalized to "
            f"'{normalized}', which does not match a recognized indicator. The recognized genotypes are: {recognized}."
        )
        console.error(message=message, error=ValueError)

    return indicator


def _read_session_genotype(session: SessionData) -> str:
    """Reads the genotype of the session's animal from its surgery metadata.

    Args:
        session: The loaded session whose animal genotype is read.

    Returns:
        The animal's genotype string.

    Raises:
        FileNotFoundError: If the session's surgery metadata file is missing.
    """
    surgery_metadata_path = session.raw_data.surgery_metadata_path
    if not surgery_metadata_path.is_file():
        message = (
            f"Unable to resolve the cindra configuration for session '{session.session_name}'. No surgery metadata "
            f"file was found at '{surgery_metadata_path}'. The animal's genotype is read from this file to select the "
            f"cindra configuration."
        )
        console.error(message=message, error=FileNotFoundError)
    return SurgeryData.from_yaml(file_path=surgery_metadata_path).subject.genotype


def _build_single_recording_configuration(genotype: str) -> SingleRecordingConfiguration:
    """Builds the reference mesoscope-vr CA1 single-recording cindra configuration for an animal's genotype.

    Notes:
        Every parameter is written out explicitly, so the configuration is decoupled from cindra's evolving defaults.
        Only ``main.tau`` and ``spike_deconvolution.neuropil_coefficient`` depend on the indicator. The deploy-time
        fields (``file_io.data_path``, ``file_io.output_path``, and the ``runtime`` settings) are left at their cindra
        defaults, because the two-photon pipeline overrides them with the session-resolved locations and worker budget.

    Args:
        genotype: The animal's genotype, read from the ``subject.genotype`` field of its surgery metadata.

    Returns:
        The single-recording configuration for the genotype's calcium indicator.

    Raises:
        ValueError: If the genotype does not match a recognized calcium indicator.
    """
    parameters = _INDICATOR_PARAMETERS[_resolve_calcium_indicator(genotype)]
    return SingleRecordingConfiguration(
        main=Main(
            two_channels=False,
            first_channel_functional=True,
            second_channel_functional=False,
            tau=parameters.tau,
            ignored_flyback_planes=(),
            custom_classifier_path=None,
        ),
        file_io=FileIO(
            ignored_file_names=("zstack",),
            repeat_binarization=False,
        ),
        registration=Registration(
            repeat_registration=False,
            align_by_first_channel=True,
            reference_frame_count=500,
            batch_size=100,
            maximum_offset_fraction=0.1,
            spatial_smoothing_sigma=1.15,
            temporal_smoothing_sigma=0.0,
            two_step_registration=False,
            bad_frame_threshold=1.0,
            normalize_frames=True,
            registration_metric_principal_components=10,
            compute_bidirectional_phase_offset=False,
            bidirectional_phase_offset_override=0,
        ),
        one_photon_registration=OnePhotonRegistration(
            enabled=False,
            spatial_highpass_window=42,
            pre_smoothing_sigma=0.0,
            edge_taper_pixels=40.0,
        ),
        nonrigid_registration=NonrigidRegistration(
            enabled=True,
            block_size=(128, 128),
            signal_to_noise_threshold=1.2,
            maximum_block_offset=5.0,
        ),
        roi_detection=ROIDetection(
            enabled=True,
            preclassification_threshold=0.5,
            threshold_scaling=2.0,
            spatial_highpass_window=25,
            maximum_overlap=0.75,
            temporal_highpass_window=100,
            maximum_iterations=50,
            maximum_binned_frames=5000,
            denoise=False,
            crop_to_soma=True,
        ),
        signal_extraction=SignalExtraction(
            extract_neuropil=True,
            allow_overlap=False,
            minimum_neuropil_pixels=350,
            inner_neuropil_border_radius=2,
            cell_probability_percentile=50,
            classification_threshold=0.5,
            batch_size=500,
            colocalization_threshold=0.65,
        ),
        spike_deconvolution=SpikeDeconvolution(
            extract_spikes=True,
            neuropil_coefficient=parameters.neuropil_coefficient,
            baseline_method=BaselineMethod.MAXIMIN,
            baseline_window=60.0,
            baseline_sigma=10.0,
            baseline_percentile=8.0,
        ),
    )


def _build_multi_recording_configuration(genotype: str) -> MultiRecordingConfiguration:
    """Builds the reference mesoscope-vr CA1 multi-recording cindra configuration for an animal's genotype.

    Notes:
        Every parameter is written out explicitly, so the configuration is decoupled from cindra's evolving defaults.
        Only ``roi_selection.probability_threshold`` and ``spike_deconvolution.neuropil_coefficient`` depend on the
        indicator. The deploy-time fields (``recording_io.recording_directories``, ``recording_io.dataset_name``, and
        the ``runtime`` settings) are left at their cindra defaults, because the forging pipeline overrides them with
        the animal's recording directories, the per-animal dataset name, and the worker budget.

    Args:
        genotype: The animal's genotype, read from the ``subject.genotype`` field of its surgery metadata.

    Returns:
        The multi-recording configuration for the genotype's calcium indicator.

    Raises:
        ValueError: If the genotype does not match a recognized calcium indicator.
    """
    parameters = _INDICATOR_PARAMETERS[_resolve_calcium_indicator(genotype)]
    return MultiRecordingConfiguration(
        recording_io=RecordingIO(
            repeat_selection=False,
        ),
        roi_selection=ROISelection(
            probability_threshold=parameters.probability_threshold,
            maximum_size=1000,
            mroi_region_margin=30,
            probability_threshold_channel_2=None,
            maximum_size_channel_2=None,
            mroi_region_margin_channel_2=None,
        ),
        diffeomorphic_registration=DiffeomorphicRegistration(
            image_type=ReferenceImageType.ENHANCED_MEAN,
            grid_sampling_factor=1,
            scale_sampling=30,
            speed_factor=3,
            repeat_registration=False,
        ),
        roi_tracking=ROITracking(
            threshold=0.75,
            mask_prevalence=50,
            pixel_prevalence=50,
            step_sizes=(200, 200),
            bin_size=50,
            maximum_distance=20,
            minimum_size=25,
        ),
        signal_extraction=SignalExtraction(
            extract_neuropil=True,
            allow_overlap=True,
            minimum_neuropil_pixels=350,
            inner_neuropil_border_radius=2,
            cell_probability_percentile=50,
            classification_threshold=0.5,
            batch_size=500,
            colocalization_threshold=0.65,
        ),
        spike_deconvolution=SpikeDeconvolution(
            extract_spikes=True,
            neuropil_coefficient=parameters.neuropil_coefficient,
            baseline_method=BaselineMethod.MAXIMIN,
            baseline_window=60.0,
            baseline_sigma=10.0,
            baseline_percentile=8.0,
        ),
    )


def _assert_indicator_coverage() -> None:
    """Verifies at import time that every calcium indicator declares its indicator-dependent parameters.

    Raises:
        RuntimeError: If a ``_CalciumIndicator`` member is missing from ``_INDICATOR_PARAMETERS``. The error names the
            offending members so an added indicator without parameters fails loudly at import rather than at build time.
    """
    uncovered = sorted(indicator.name for indicator in _CalciumIndicator if indicator not in _INDICATOR_PARAMETERS)
    if uncovered:
        message = (
            f"Unable to validate calcium-indicator coverage. Every _CalciumIndicator member must declare its "
            f"indicator-dependent parameters in _INDICATOR_PARAMETERS, but the following members do not: "
            f"{', '.join(uncovered)}."
        )
        console.error(message=message, error=RuntimeError)


_assert_indicator_coverage()
