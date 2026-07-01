# FL Master Thesis

This repository contains the code, notebooks, and experiment structure used in a master's thesis focused on fall detection, preprocessing strategies, centralized baselines, and federated learning scenarios.

## Repository Scope

The repository tracks code, notebooks, and experiment definitions, but it does not version the large raw datasets or generated preprocessing outputs.

Dataset directories excluded from version control:

- `data/Kfall/`
- `data/sisfall/`
- `data/UpFall/`

Generated outputs excluded from version control:

- `data/preprocessing_results/`

## Main Areas

- `data/`
  Contains preprocessing scripts and exploratory notebooks.
- `experiments/`
  Contains centralized experiments and comparison notebooks for different split strategies.
- `fedavg-experiments/`
  Contains the Flower-based federated learning scaffold for the FedAvg experiments.

## Centralized Experiment Families

The repository currently includes four main centralized evaluation techniques:

- `baseline`
  Trial-disjoint split.
- `byclient`
  Client-disjoint split.
- `onedatasetout`
  Leave-one-dataset-out split.
- `kfoldbyclient`
  Cross-validated client-disjoint split.

Each family contains the same preprocessing strategies:

- `zscore`
- `client_zscore`
- `magnitude_features`
- `magnitude_only`
- `robust_clip`

Comparison notebooks are available inside each family, and a global comparison notebook is available at:

- `experiments/all_split_techniques_comparison.ipynb`

The text summaries for the current centralized results are stored in:

- `experiments/results_observations.txt`
- `experiments/data_split_strategies.txt`

## FedAvg Experiments

The folder `fedavg-experiments/` contains the initial Flower-based scaffold for the federated learning stage.

Current files:

- `fedavg-experiments/config.py`
- `fedavg-experiments/data_utils.py`
- `fedavg-experiments/metrics.py`
- `fedavg-experiments/modeling.py`
- `fedavg-experiments/flwr_utils.py`
- `fedavg-experiments/run_experiment.py`

The experiment catalog currently includes:

- `exp1_fedavg_base`
- `exp2_fraction_clients`
- `exp3_local_epochs`
- `exp4_unweighted_aggregation`
- `exp5_cross_dataset`
- `exp6_keep_best_local_model`
- `exp7_clustered_aggregation`
- `exp8_personalized_fedavg`
- `exp9_final_local_finetuning`

## Running Preprocessing

Typical usage starts from the preprocessing scripts:

```powershell
..\venv\Scripts\python.exe .\data\run_full_preprocessing.py
```

This generates local preprocessing outputs under:

```text
data/preprocessing_results/
```

## Running FedAvg

To run the first federated experiment from the repository root:

```powershell
..\venv\Scripts\python.exe .\fedavg-experiments\run_experiment.py --experiment exp1_fedavg_base
```

Flower simulation requires the simulation extra, which installs `ray`. If you get an error saying that `ray` cannot be imported, install:

```powershell
..\venv\Scripts\python.exe -m pip install -U "flwr[simulation]"
```

## Notes

- Generated experiment outputs are written locally and are not intended to be committed by default.
- The current best centralized preprocessing strategy is `magnitude_features`.
- The current best centralized neural model is the `mlp_wide` configuration.
