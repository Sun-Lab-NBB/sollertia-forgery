from pathlib import Path

from sollertia_forgery.analysis import run_sce_analysis, run_tuning_analysis
from sollertia_forgery.shared_assets import DatasetData

dataset_path = Path("/home/data/Data/StateSpaceOdyssey/extension/")

dataset = DatasetData.load(dataset_path=dataset_path)
run_tuning_analysis(dataset=dataset)
run_sce_analysis(dataset=dataset)
