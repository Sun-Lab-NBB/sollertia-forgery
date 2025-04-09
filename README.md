# sl-forgery

## Setup

### Create conda environment
```bash
conda create --name suite2p python=3.12
conda activate suite2p
tox -e setup
```

<!-- ### Copy bash scripts to `$HOME`

```bash
mkdir ~/.multiday-suite2p
cp scripts/extract_session_job.sh ~/.multiday-suite2p/extract_session_job.sh
mkdir ~/.linear2ac
cp scripts/placefield_job.sh ~/.linear2ac/placefield_job.sh
``` -->


## Usage

Expected to run after
[mesoscope-processing](https://github.com/Sun-Lab-NBB/mesoscope-processing)
pipeline.

1. Run `notebooks/multiday_registration.ipynb` for registration across sessions.
Create `logs` directory wherever you are running the notebook from.
2. Run `notebooks/Create multi-day vr2p ExperimentData object.ipynb` for
generating preprocessed dataset. Create `logs` directory wherever you are
running the notebook from.
3. Run notebook `notebooks/placefield_cache.ipynb` to compute & cache
place fields for cells. If you want to skip the shuffle significance test, set
`bootstrap_do_test` to `False`, and `bootstrap_num_shuffles` to `1`.
