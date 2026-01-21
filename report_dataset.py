from pathlib import Path
from tqdm import tqdm
from sl_forgery.forging import assemble_report_dataset

sessions_root = Path("/Users/natalieyeung/Downloads/test_session/")
output_dir = Path("/Users/natalieyeung/Documents/GitHub/sl-forgery")

sessions = [session.name for session in sessions_root.glob("*") if session.is_dir()]
for session in tqdm(sessions, desc="Assembling datasets", unit="session"):
    # Skips rebuilding already existing datasets
    if output_dir.joinpath(f"{session}_report.feather").exists():
        continue

    assemble_report_dataset(
        session_data_path=sessions_root.joinpath(session),
        output_path=output_dir.joinpath(f"{session}_report.feather"),
        progress=False,
    )
