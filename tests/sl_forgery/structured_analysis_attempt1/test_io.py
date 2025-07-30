from sl_forgery.structured_analysis_attempt1.io import parse_data


from pathlib import Path


def test_parse_data():

    path = "data/TM_06_pilot/6/2025-06-23-13-32-06-980761/"
    project_root = Path(__file__).resolve().parents[3]
    session_root = project_root / path

    target_group = "single_day"
    parse_data(project_root)

    assert True
