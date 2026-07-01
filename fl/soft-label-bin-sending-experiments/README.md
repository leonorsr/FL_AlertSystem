# Binned soft-label sending experiments

These experiments test probability-distribution payloads where each client sends
fixed-bin statistics of its predicted fall probabilities. The only experimental
factor is the number of probability bins.

For every bin the communicated values are:

- mean predicted probability
- standard deviation
- count

Configurations:

- `prob_bins_3`: 3 bins, 9 scalars per round
- `prob_bins_5`: 5 bins, 15 scalars per round
- `prob_bins_10`: 10 bins, 30 scalars per round

Run all configurations:

```powershell
python .\soft-label-bin-sending-experiments\run_all.py --runs 10
```

Generate the summary:

```powershell
python .\soft-label-bin-sending-experiments\generate_summary.py
```

Quick smoke test:

```powershell
python .\soft-label-bin-sending-experiments\run_all.py --runs 1 --num-rounds 2 --local-epochs 1 --results-dir .\soft-label-bin-sending-experiments\_smoke_results
```
