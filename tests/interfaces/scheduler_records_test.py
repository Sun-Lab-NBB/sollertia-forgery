"""Contains tests for the parse that turns the compute server's accounting response into one record per job."""

from __future__ import annotations

from sollertia_forgery.interfaces.server_tools import _parse_accounting_rows

_COLUMNS: tuple[str, ...] = (
    "JobID",
    "JobName",
    "State",
    "Elapsed",
    "NCPUS",
    "AveCPU",
    "ReqMem",
    "MaxRSS",
    "AveRSS",
    "MaxVMSize",
)
"""The accounting columns the scheduler names in the first line of its response."""


def test_a_job_reports_the_largest_figure_its_steps_measured() -> None:
    """Verifies that a job carries its own identity and state alongside the widest figure any of its steps recorded."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "JobName": "video", "State": "COMPLETED", "Elapsed": "00:10:00", "NCPUS": "8"},
            {"JobID": "5502.0", "AveCPU": "00:01:00", "MaxRSS": "2.00G", "AveRSS": "1.00G", "MaxVMSize": "3.00G"},
            {"JobID": "5502.1", "AveCPU": "00:00:30", "MaxRSS": "8.00G", "AveRSS": "4.00G", "MaxVMSize": "9.00G"},
        ]
    )

    rows = _parse_accounting_rows(output=output)

    assert len(rows) == 1
    assert rows[0]["job_id"] == "5502"
    assert rows[0]["state"] == "COMPLETED"
    assert rows[0]["elapsed"] == "00:10:00"
    assert rows[0]["maximum_resident_memory"] == "8.00G"
    assert rows[0]["maximum_virtual_memory"] == "9.00G"
    assert rows[0]["peak_step_resident_memory"] == "4.00G"
    assert rows[0]["peak_step_cpu_time"] == "00:01:00"


def test_the_largest_figure_wins_whatever_order_the_steps_arrive_in() -> None:
    """Verifies that the widest step is picked rather than whichever the scheduler happened to write last."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "State": "COMPLETED"},
            {"JobID": "5502.0", "MaxRSS": "8.00G"},
            {"JobID": "5502.1", "MaxRSS": "2.00G"},
        ]
    )

    assert _parse_accounting_rows(output=output)[0]["maximum_resident_memory"] == "8.00G"


def test_a_memory_figure_is_compared_by_magnitude_rather_than_by_its_text() -> None:
    """Verifies that the suffix the scheduler appends orders the figures, since their text does not."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "State": "COMPLETED"},
            {"JobID": "5502.0", "MaxRSS": "512.00M"},
            {"JobID": "5502.1", "MaxRSS": "2.00G"},
        ]
    )

    assert _parse_accounting_rows(output=output)[0]["maximum_resident_memory"] == "2.00G"


def test_a_processor_time_is_compared_by_seconds_rather_than_by_its_text() -> None:
    """Verifies that a duration leading with a day count outranks a longer-looking clock time."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "State": "COMPLETED"},
            {"JobID": "5502.0", "AveCPU": "23:00:00"},
            {"JobID": "5502.1", "AveCPU": "1-00:00:00"},
        ]
    )

    assert _parse_accounting_rows(output=output)[0]["peak_step_cpu_time"] == "1-00:00:00"


def test_the_external_shell_step_is_left_out_of_the_record() -> None:
    """Verifies that the step the scheduler records for a job's external shell contributes no figure."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "State": "COMPLETED"},
            {"JobID": "5502.extern", "MaxRSS": "99.00G"},
            {"JobID": "5502.0", "MaxRSS": "2.00G"},
        ]
    )

    assert _parse_accounting_rows(output=output)[0]["maximum_resident_memory"] == "2.00G"


def test_a_response_carrying_no_record_reports_nothing() -> None:
    """Verifies that a header line on its own is answered as an empty result rather than as a record."""
    assert _parse_accounting_rows(output=_render_response(steps=[])) == []
    assert _parse_accounting_rows(output="") == []


def test_every_job_the_response_names_reaches_the_result() -> None:
    """Verifies that the steps of one job never fold into the record of another."""
    output = _render_response(
        steps=[
            {"JobID": "5502", "State": "COMPLETED"},
            {"JobID": "5502.0", "MaxRSS": "2.00G"},
            {"JobID": "5503", "State": "FAILED"},
            {"JobID": "5503.0", "MaxRSS": "8.00G"},
        ]
    )

    rows = _parse_accounting_rows(output=output)

    assert [row["job_id"] for row in rows] == ["5502", "5503"]
    assert [row["maximum_resident_memory"] for row in rows] == ["2.00G", "8.00G"]


def _render_response(steps: list[dict[str, str]]) -> str:
    """Renders one accounting response, writing the empty text into every column a step leaves unnamed.

    Args:
        steps: The records the scheduler wrote, keyed by the column names it reports.

    Returns:
        The response text.
    """
    lines = ["|".join(_COLUMNS)]
    lines.extend("|".join(step.get(column, "") for column in _COLUMNS) for step in steps)
    return "\n".join(lines)
