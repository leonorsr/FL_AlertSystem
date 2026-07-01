# Soft-label Sending Experiments

This folder contains no-weight federated experiments that test richer probability
payloads for soft-label communication. The goal is to check whether sending a few
more statistics per client improves local/client-side performance while keeping
communication much smaller than model-weight exchange.

## Payload Variants

| Payload | Scalars sent per client | Description |
| --- | ---: | --- |
| `global_mean_count` | 2 | Current baseline: mean predicted fall probability plus count. |
| `global_mean_std_count` | 3 | Adds global standard deviation of local predicted probabilities. |
| `class_mean_counts` | 4 | Sends fall/non-fall class-conditional means plus class counts. |
| `class_mean_std_counts` | 6 | Adds class-conditional standard deviations. |
| `class_quantile_stats` | 12 | Adds class-conditional q25/q50/q75 quantiles. |

For comparison, the MLP used in the existing experiments has 49,665 trainable
parameters, while the richest payload here sends only 12 scalar values per client.

## Run A Smoke Test

From the repository root:

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\run_all.py --num-rounds 3 --local-epochs 20 --runs 1
```

## Run Full Experiments

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\run_all.py --runs 10
```

To run one payload only:

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\run_experiment.py --payload class_quantile_stats --runs 10
```

## Summarize Results

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\generate_summary.py
```

The summary ranks payloads by local PR-AUC first, since these experiments are
intended to improve local/client-side training rather than produce a global
parameter model.

## Advanced Strategies

The folder also includes four higher-level strategies that may be more useful
than simply sending more global statistics:

| Strategy | Main idea |
| --- | --- |
| `cluster_mean_count` | Aggregate mean/count soft labels separately per client similarity cluster. |
| `cluster_class_mean_counts` | Aggregate class-conditional mean/count soft labels per cluster. |
| `warmup_global_mean_count` | Use hard-label local training only for the first 5 rounds, then enable global mean/count KD. |
| `uncertain_global_mean_count` | Apply global mean/count KD only to locally uncertain examples with probabilities in [0.3, 0.7]. |

Smoke test:

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\run_all_strategies.py --num-rounds 3 --local-epochs 20 --runs 1
```

Full run:

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\run_all_strategies.py --runs 10
```

Summarize advanced results:

```powershell
..\venv\Scripts\python.exe .\soft-labels-sending-experiments\generate_advanced_summary.py
```
