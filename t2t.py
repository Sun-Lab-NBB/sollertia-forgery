from pathlib import Path
from tqdm import tqdm
from sl_forgery.forging import DatasetTypes, assemble_session_dataset

data_root = Path(f"/home/cyberaxolotl/server/workdir/sun_data/Datasets/")

for dataset in data_root.glob("*"):
    for animal in dataset.glob("*"):
        project = "StateSpaceOdyssey" if dataset.name == "SSOData" else "MaalstroomicFlow"
        session_data_root = Path(f"/home/cyberaxolotl/server/workdir/sun_data/{project}/{animal.name}/")
        dataset_session_root = Path(
            f"/home/cyberaxolotl/server/workdir/sun_data/Datasets/{dataset.name}/{animal.name}/"
        )

        sessions = [session.name for session in dataset_session_root.glob("*") if session.is_dir()]
        for session in tqdm(sessions, desc=f"Assembling {animal.name} datasets", unit="session"):
            # Skips rebuilding already existing datasets
            if dataset_session_root.joinpath(f"{session}.feather").exists():
                continue

            assemble_session_dataset(
                session_data_path=session_data_root.joinpath(session),
                session_multiday_path=dataset_session_root.joinpath(session),
                output_path=dataset_session_root.joinpath(f"{session}.feather"),
                dataset_type=DatasetTypes.MESOSCOPE_VR_EXPERIMENT,
                progress=False,
            )
