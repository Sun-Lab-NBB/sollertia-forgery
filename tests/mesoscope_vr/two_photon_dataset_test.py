"""Tests for the Mesoscope-VR two-photon fluorescence sub-dataset assembler and its pulse alignment helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cindra import CombinedData, DetectionData, ExtractionData
import polars as pl
import pytest
from sollertia_shared_assets import MesoscopeDirectories

from sollertia_forgery.mesoscope_vr.metadata import DatasetColumn, BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.two_photon_dataset import (
    assemble_cindra_dataset,
    _resolve_acquisition_sizes,
    _match_runs_to_acquisitions,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from numpy.typing import NDArray

SAMPLING_RATE_HZ: float = 10.0
"""The per-plane sampling rate written into every synthetic combined metadata archive. It places the accepted scan
pulse duration window at 80 to 120 milliseconds."""

IN_WINDOW_DURATION_US: int = 90_000
"""A scan pulse duration that falls inside the accepted window for ``SAMPLING_RATE_HZ``."""

OUT_OF_WINDOW_DURATION_US: int = 10
"""A scan pulse duration that falls outside the accepted window for ``SAMPLING_RATE_HZ``."""

COMBINED_FRAME_EXTENT: int = 512
"""The height and the width recorded in every synthetic combined metadata archive. The assembler reads the sampling
rate alone, so the extent only has to be a shape cindra's writer accepts."""


def _write_ttl_feather(path: Path, pulses: Sequence[tuple[int, int]]) -> None:
    """Writes the processed mesoscope-frame feather for the given TTL pulse train.

    The feather uses the schema and the leading rising-edge layout produced by the microcontroller TTL parser. A
    sacrificial pulse is prepended, because the assembler pairs edges by cumulative rising-edge count and therefore
    never pairs the very first logged pulse.

    Args:
        path: The path of the feather file to write.
        pulses: The ``(start_us, duration_us)`` pair of every pulse the assembler is expected to observe.
    """
    train = [(pulses[0][0] - 500_000, 1_000), *pulses]
    times: list[int] = []
    states: list[int] = []
    for start_us, duration_us in train:
        times.extend((start_us, start_us + duration_us))
        states.extend((1, 0))
    dataframe = pl.DataFrame(
        {
            "time_us": np.asarray(times, dtype=np.uint64),
            "ttl_state": np.asarray(states, dtype=np.uint8),
        }
    )
    dataframe.write_ipc(file=path, compression="uncompressed")


def _pulse_train(
    start_us: int, count: int, *, period_us: int = 100_000, duration_us: int = IN_WINDOW_DURATION_US
) -> list[tuple[int, int]]:
    """Returns an evenly spaced run of TTL pulses.

    Args:
        start_us: The rising-edge timestamp of the first pulse.
        count: The number of pulses in the run.
        period_us: The rising-edge to rising-edge interval.
        duration_us: The high-state duration of every pulse.

    Returns:
        The ``(start_us, duration_us)`` pair of every pulse in the run.
    """
    return [(start_us + index * period_us, duration_us) for index in range(count)]


def _build_extraction(
    *, roi_count: int, frame_count: int, offset: float, is_cell: Sequence[int] | None = None
) -> ExtractionData:
    """Builds the cindra extraction record holding the four fluorescence trace arrays and, optionally, the labels.

    Every trace array counts up from a shared ramp, so a value pins both the ROI row it came from and the frame it was
    sampled at, and the per-array offset keeps the four distinguishable.

    Args:
        roi_count: The number of ROI rows every array carries.
        frame_count: The number of frame columns every array carries.
        offset: The constant added to the base ramp, so each directory carries distinguishable values.
        is_cell: The per-ROI cell label, one entry per ROI row, or None for a record carrying no classification.

    Returns:
        The populated extraction record, which cindra's own writer saves under its canonical array names.
    """
    base = np.arange(roi_count * frame_count, dtype=np.float32).reshape(roi_count, frame_count) + offset
    classification: NDArray[np.float32] | None = None
    if is_cell is not None:
        classification = np.zeros((roi_count, 2), dtype=np.float32)
        classification[:, 0] = np.asarray(is_cell, dtype=np.float32)
        classification[:, 1] = 0.5
    return ExtractionData(
        cell_fluorescence=base,
        neuropil_fluorescence=base + 1,
        subtracted_fluorescence=base + 2,
        spikes=base + 3,
        cell_classification=classification,
    )


def _write_traces(directory: Path, *, roi_count: int, frame_count: int, offset: float) -> None:
    """Writes the four cindra fluorescence trace arrays into a processing output directory.

    Args:
        directory: The directory the trace arrays are written into.
        roi_count: The number of ROI rows every array carries.
        frame_count: The number of frame columns every array carries.
        offset: The constant added to the base ramp, so each directory carries distinguishable values.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _build_extraction(roi_count=roi_count, frame_count=frame_count, offset=offset).save_arrays(output_path=directory)


def _write_cindra_outputs(directory: Path, *, is_cell: Sequence[int], frame_count: int, offset: float = 0.0) -> None:
    """Writes a complete single-recording cindra output directory through cindra's own writer.

    Notes:
        The combined record is saved rather than assembled by hand, so the archive the assembler reads back is the one
        cindra's combination stage publishes, including every metadata field its reader expects.

    Args:
        directory: The directory the single-recording outputs are written into.
        is_cell: The per-ROI cell label, one entry per ROI row.
        frame_count: The number of frames every trace array carries.
        offset: The constant added to the base ramp of every trace array.
    """
    directory.mkdir(parents=True, exist_ok=True)
    CombinedData(
        detection=DetectionData(),
        extraction=_build_extraction(roi_count=len(is_cell), frame_count=frame_count, offset=offset, is_cell=is_cell),
        plane_count=1,
        frame_count=frame_count,
        combined_height=COMBINED_FRAME_EXTENT,
        combined_width=COMBINED_FRAME_EXTENT,
        sampling_rate=SAMPLING_RATE_HZ,
    ).save(root_path=directory)


def _write_frame_metadata(
    raw_data_path: Path,
    *,
    frame_numbers: Sequence[int],
    frame_seconds: Sequence[float] | None = None,
    acquisition_numbers: Sequence[int] | None = None,
) -> Path:
    """Writes the ScanImage per-frame metadata archive into a session's raw mesoscope_data directory.

    Args:
        raw_data_path: The session raw_data directory the archive is written under.
        frame_numbers: The per-frame ScanImage frame counter values.
        frame_seconds: The per-frame ScanImage clock timestamps in seconds. Defaults to None, which writes a zero
            timestamp for every frame.
        acquisition_numbers: The per-frame ScanImage acquisition index. Defaults to None, which omits the key.

    Returns:
        The path of the written archive.
    """
    mesoscope_data_path = raw_data_path.joinpath(MesoscopeDirectories.MESOSCOPE_DATA)
    mesoscope_data_path.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "frameNumberAcquisition": np.asarray(frame_numbers, dtype=np.int64),
        "frameTimestamps_sec": np.asarray(
            frame_seconds if frame_seconds is not None else [0.0] * len(frame_numbers), dtype=np.float64
        ),
    }
    if acquisition_numbers is not None:
        arrays["acquisitionNumbers"] = np.asarray(acquisition_numbers, dtype=np.int64)
    archive_path = mesoscope_data_path.joinpath("frame_variant_metadata.npz")
    np.savez(archive_path, **arrays)
    return archive_path


class _Layout:
    """Bundles the four directories ``assemble_cindra_dataset`` reads from.

    Args:
        root: The temporary directory the layout is created under.

    Attributes:
        cindra_data_path: The single-recording cindra output directory.
        microcontroller_data_path: The processed microcontroller-data directory.
        multiday_data_path: The multi-recording cindra output directory.
        raw_data_path: The session raw_data directory.
    """

    def __init__(self, root: Path) -> None:
        self.cindra_data_path = root.joinpath("processed_data", "cindra")
        self.microcontroller_data_path = root.joinpath("processed_data", "microcontroller_data")
        self.multiday_data_path = root.joinpath("processed_data", "multiday")
        self.raw_data_path = root.joinpath("session", "raw_data")
        for directory in (
            self.cindra_data_path,
            self.microcontroller_data_path,
            self.multiday_data_path,
            self.raw_data_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def assemble(self) -> pl.DataFrame:
        """Runs the assembler against this layout.

        Returns:
            The assembled fluorescence table.
        """
        return assemble_cindra_dataset(
            cindra_data_path=self.cindra_data_path,
            microcontroller_data_path=self.microcontroller_data_path,
            multiday_data_path=self.multiday_data_path,
            raw_data_path=self.raw_data_path,
        )


@pytest.fixture
def layout(tmp_path: Path) -> _Layout:
    """Returns an empty four-directory layout for the assembler."""
    return _Layout(tmp_path)


def _prepare(
    layout: _Layout,
    *,
    pulses: Sequence[tuple[int, int]],
    frame_count: int,
    is_cell: Sequence[int] = (1, 0, 1, 1),
    multiday_roi_count: int = 2,
) -> None:
    """Writes every input file the assembler reads for one scenario.

    Args:
        layout: The directory layout the files are written into.
        pulses: The TTL pulse train written into the processed mesoscope-frame feather.
        frame_count: The cindra frame count every trace array carries.
        is_cell: The per-ROI cell label of the single-recording classification array.
        multiday_roi_count: The number of ROI rows the multi-recording trace arrays carry.
    """
    _write_ttl_feather(layout.microcontroller_data_path.joinpath(BehaviorDataFiles.MESOSCOPE_FRAME), pulses)
    _write_cindra_outputs(layout.cindra_data_path, is_cell=is_cell, frame_count=frame_count)
    _write_traces(layout.multiday_data_path, roi_count=multiday_roi_count, frame_count=frame_count, offset=1000.0)


def test_assemble_cindra_dataset_matches_pulse_count_exactly(layout: _Layout) -> None:
    """Verifies the assembler pairs every in-window pulse with a cindra frame when the two counts already agree."""
    _prepare(layout, pulses=_pulse_train(1_000_000, 5), frame_count=5)

    dataset = layout.assemble()

    assert dataset["frame"].to_list() == [1, 2, 3, 4, 5]
    assert dataset["frame"].dtype == pl.UInt32
    assert dataset["time_us"].to_list() == [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000]
    assert dataset.columns == [
        "frame",
        "time_us",
        "elapsed_minutes",
        DatasetColumn.SINGLE_DAY_CELL_FLUORESCENCE,
        DatasetColumn.SINGLE_DAY_NEUROPIL_FLUORESCENCE,
        DatasetColumn.SINGLE_DAY_SUBTRACTED_FLUORESCENCE,
        DatasetColumn.SINGLE_DAY_SPIKES,
        DatasetColumn.MULTI_DAY_CELL_FLUORESCENCE,
        DatasetColumn.MULTI_DAY_NEUROPIL_FLUORESCENCE,
        DatasetColumn.MULTI_DAY_SUBTRACTED_FLUORESCENCE,
        DatasetColumn.MULTI_DAY_SPIKES,
    ]

    # The classification array marks ROIs 0, 2 and 3 as cells, so the single-recording traces keep three of four rows
    # while the multi-recording traces keep both of theirs.
    single_day = dataset[DatasetColumn.SINGLE_DAY_CELL_FLUORESCENCE].to_numpy()
    expected = np.arange(20, dtype=np.float32).reshape(4, 5)[[0, 2, 3], :].T
    assert single_day.shape == (5, 3)
    assert np.array_equal(single_day, expected)

    multi_day = dataset[DatasetColumn.MULTI_DAY_SPIKES].to_numpy()
    assert multi_day.shape == (5, 2)
    assert np.array_equal(multi_day, np.arange(10, dtype=np.float32).reshape(2, 5).T + 1003.0)


def test_assemble_cindra_dataset_records_elapsed_minutes(layout: _Layout) -> None:
    """Verifies the elapsed-minutes column measures each frame from the first retained pulse."""
    _prepare(layout, pulses=_pulse_train(4_000_000, 4, period_us=1_200_000), frame_count=4)

    dataset = layout.assemble()

    assert dataset["elapsed_minutes"].dtype == pl.Float32
    assert dataset["elapsed_minutes"].to_list() == pytest.approx([0.0, 0.02, 0.04, 0.06])


def test_assemble_cindra_dataset_clips_surplus_leading_pulses(layout: _Layout) -> None:
    """Verifies a log carrying more in-window pulses than cindra frames keeps only its trailing pulses."""
    _prepare(layout, pulses=_pulse_train(1_000_000, 5), frame_count=3)

    dataset = layout.assemble()

    assert dataset["frame"].to_list() == [1, 2, 3]
    assert dataset["time_us"].to_list() == [1_200_000, 1_300_000, 1_400_000]
    assert dataset[DatasetColumn.SINGLE_DAY_SPIKES].to_numpy().shape == (3, 3)


def test_assemble_cindra_dataset_falls_back_to_scanimage(layout: _Layout) -> None:
    """Verifies pulses rejected by the duration filter are recovered by matching them to ScanImage frame timestamps."""
    pulses = [
        (1_000_000, IN_WINDOW_DURATION_US),
        (1_100_000, OUT_OF_WINDOW_DURATION_US),
        (1_200_000, OUT_OF_WINDOW_DURATION_US),
        (1_300_000, OUT_OF_WINDOW_DURATION_US),
    ]
    _prepare(layout, pulses=pulses, frame_count=4)

    # The archive is deliberately written out of acquisition order, so the fallback's chronological sort is exercised.
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[4, 2, 1, 3], frame_seconds=[0.3, 0.1, 0.0, 0.2])

    dataset = layout.assemble()

    assert dataset["frame"].to_list() == [1, 2, 3, 4]
    assert dataset["time_us"].to_list() == [1_000_000, 1_100_000, 1_200_000, 1_300_000]


def test_scanimage_fallback_keeps_the_closest_pulse_per_frame(layout: _Layout) -> None:
    """Verifies two pulses claiming one ScanImage frame resolve to the closer pulse, whichever of them logged first."""
    pulses = [
        (5_000_000, OUT_OF_WINDOW_DURATION_US),
        (5_000_020, OUT_OF_WINDOW_DURATION_US),
        (5_199_980, OUT_OF_WINDOW_DURATION_US),
        (5_200_000, OUT_OF_WINDOW_DURATION_US),
        (5_400_000, OUT_OF_WINDOW_DURATION_US),
    ]
    _prepare(layout, pulses=pulses, frame_count=3)
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[1, 2, 3], frame_seconds=[0.0, 0.2, 0.4])

    dataset = layout.assemble()

    # The first frame keeps the earlier pulse and the second frame replaces its earlier claimant with the later,
    # closer pulse.
    assert dataset["frame"].to_list() == [1, 2, 3]
    assert dataset["time_us"].to_list() == [5_000_000, 5_200_000, 5_400_000]
    assert dataset["elapsed_minutes"].to_list() == pytest.approx([0.0, 0.0, 0.01])


def test_scanimage_fallback_missing_archive_errors(layout: _Layout) -> None:
    """Verifies the fallback raises when the session carries no ScanImage per-frame metadata archive."""
    pulses = [(5_000_000, OUT_OF_WINDOW_DURATION_US), (5_200_000, OUT_OF_WINDOW_DURATION_US)]
    _prepare(layout, pulses=pulses, frame_count=2)

    with pytest.raises(ValueError, match=r"(?s)expected\s+per-frame\s+metadata\s+archive.*does\s+not\s+exist"):
        layout.assemble()


def test_scanimage_fallback_archive_frame_count_mismatch_errors(layout: _Layout) -> None:
    """Verifies the fallback raises when the ScanImage archive holds a different frame count than cindra reports."""
    pulses = [(5_000_000, OUT_OF_WINDOW_DURATION_US), (5_200_000, OUT_OF_WINDOW_DURATION_US)]
    _prepare(layout, pulses=pulses, frame_count=3)
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[1, 2, 3, 4, 5])

    with pytest.raises(
        ValueError,
        match=r"(?s)reported\s+3\s+frames,\s+but\s+the\s+ScanImage\s+metadata\s+archive.*contains\s+5\s+entries",
    ):
        layout.assemble()


def test_scanimage_fallback_unmatched_pulses_error(layout: _Layout) -> None:
    """Verifies the fallback raises when the matched pulse count falls short of the cindra frame count."""
    pulses = [(5_000_000, OUT_OF_WINDOW_DURATION_US), (5_200_000, OUT_OF_WINDOW_DURATION_US)]
    _prepare(layout, pulses=pulses, frame_count=3)
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[1, 2, 3], frame_seconds=[0.0, 0.2, 0.4])

    with pytest.raises(
        ValueError, match=r"(?s)matching\s+produced\s+2\s+pulses,\s+but\s+cindra\s+reports\s+3\s+frames"
    ):
        layout.assemble()


def test_assemble_keeps_split_runs_without_scanimage_metadata(layout: _Layout) -> None:
    """Verifies a gap-split pulse log is kept whole when no ScanImage archive names the acquisition sizes."""
    pulses = [*_pulse_train(1_000_000, 3), *_pulse_train(10_000_000, 2)]
    _prepare(layout, pulses=pulses, frame_count=5)

    dataset = layout.assemble()

    assert dataset["time_us"].to_list() == [1_000_000, 1_100_000, 1_200_000, 10_000_000, 10_100_000]
    assert dataset["elapsed_minutes"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.15, 0.15])


def test_assemble_keeps_every_run_claimed_by_an_acquisition(layout: _Layout) -> None:
    """Verifies both runs survive when the acquisition sizes recovered from repeated frame numbers claim both."""
    pulses = [*_pulse_train(1_000_000, 3), *_pulse_train(10_000_000, 2)]
    _prepare(layout, pulses=pulses, frame_count=5)
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[1, 2, 3, 1, 2])

    dataset = layout.assemble()

    assert dataset["time_us"].to_list() == [1_000_000, 1_100_000, 1_200_000, 10_000_000, 10_100_000]


def test_assemble_discards_unacquired_pulse_runs(layout: _Layout) -> None:
    """Verifies a stray run of hand-triggered pulses is dropped when the acquisition sizes claim only the real run."""
    pulses = [*_pulse_train(1_000_000, 2), *_pulse_train(10_000_000, 4, period_us=1_000_000)]
    _prepare(layout, pulses=pulses, frame_count=4)
    _write_frame_metadata(layout.raw_data_path, frame_numbers=[1, 2, 3, 4], acquisition_numbers=[1, 1, 1, 1])

    dataset = layout.assemble()

    assert dataset["frame"].to_list() == [1, 2, 3, 4]
    assert dataset["time_us"].to_list() == [10_000_000, 11_000_000, 12_000_000, 13_000_000]
    assert dataset["elapsed_minutes"].to_list() == pytest.approx([0.0, 0.02, 0.03, 0.05])


def test_assemble_keeps_runs_when_no_assignment_covers_the_acquisitions(layout: _Layout) -> None:
    """Verifies the pulse log is kept whole when there are more acquisitions than gap-separated pulse runs."""
    pulses = [*_pulse_train(1_000_000, 3), *_pulse_train(10_000_000, 2)]
    _prepare(layout, pulses=pulses, frame_count=5)
    _write_frame_metadata(
        layout.raw_data_path, frame_numbers=[1, 2, 1, 2, 1, 2], acquisition_numbers=[1, 1, 2, 2, 3, 3]
    )

    dataset = layout.assemble()

    assert dataset["time_us"].to_list() == [1_000_000, 1_100_000, 1_200_000, 10_000_000, 10_100_000]


def test_resolve_acquisition_sizes_reads_the_acquisition_index(tmp_path: Path) -> None:
    """Verifies an explicit per-frame acquisition index yields its acquisition sizes in descending order."""
    _write_frame_metadata(tmp_path, frame_numbers=[1, 2, 3, 1, 2, 1], acquisition_numbers=[1, 1, 1, 2, 2, 3])

    assert _resolve_acquisition_sizes(raw_data_path=tmp_path) == [3, 2, 1]


def test_resolve_acquisition_sizes_peels_repeated_frame_numbers(tmp_path: Path) -> None:
    """Verifies acquisition sizes are recovered from the frame-number multiset when no acquisition index is stored."""
    _write_frame_metadata(tmp_path, frame_numbers=[1, 2, 3, 1, 2, 1])

    assert _resolve_acquisition_sizes(raw_data_path=tmp_path) == [3, 2, 1]


def test_resolve_acquisition_sizes_without_archive(tmp_path: Path) -> None:
    """Verifies a session without a ScanImage metadata archive reports no acquisition sizes."""
    assert _resolve_acquisition_sizes(raw_data_path=tmp_path) == []


def test_match_runs_to_acquisitions_spans_consecutive_runs() -> None:
    """Verifies an acquisition split across neighbouring runs claims the whole consecutive span."""
    assert _match_runs_to_acquisitions(run_lengths=[2, 3, 4], acquisition_sizes=[5, 4]) == [(0, 1), (2, 2)]


def test_match_runs_to_acquisitions_without_acquisitions() -> None:
    """Verifies an empty acquisition list produces no assignment."""
    assert _match_runs_to_acquisitions(run_lengths=[2, 3], acquisition_sizes=[]) is None


def test_match_runs_to_acquisitions_with_more_acquisitions_than_runs() -> None:
    """Verifies more acquisitions than runs produces no assignment, since every acquisition needs its own run."""
    assert _match_runs_to_acquisitions(run_lengths=[1, 2], acquisition_sizes=[1, 1, 1]) is None
