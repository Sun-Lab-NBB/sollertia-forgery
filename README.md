# Mesoscope Processing

## Setup

### Create conda environment
```bash
conda create --name suite2p python=3.12
conda activate suite2p
pip install git+https://github.com/kushaangupta/suite2p
pip install git+https://github.com/kushaangupta/multiday-suite2p-public
pip install git+https://github.com/Sun-Lab-NBB/vr2p-fork
pip install git+https://github.com/kushaangupta/2ACDC_parse
pip install ipykernel paramiko==3.5.1 colorcet dask[dataframe]
```

### Copy bash scripts to `$HOME`

```bash
mkdir ~/.multiday-suite2p
cp scripts/extract_session_job.sh ~/.multiday-suite2p/extract_session_job.sh
mkdir ~/.linear2ac
cp scripts/placefield_job.sh ~/.linear2ac/placefield_job.sh
```


## Usage

1. Run Suite2P on sessions either using `notebooks/Run Suite2P.ipynb` for single
sessions or `scripts/batchSuite2p.py` followed by
`scripts/postSuite2pCombine.py` for multiple sessions.
2. Run `notebooks/multiday_registration.ipynb` for registration across sessions.
Create `logs` directory wherever you are running the notebook from.
3. Run `notebooks/Create multi-day vr2p ExperimentData object.ipynb` for
generating preprocessed dataset. Create `logs` directory wherever you are
running the notebook from.
4. Run notebook `notebooks/placefield_cache.ipynb` to compute & cache
place fields for cells. If you want to skip the shuffle significance test, set
`bootstrap_do_test` to `False`, and `bootstrap_num_shuffles` to `1`.
