Usage
=====

Run a single experiment:

```bash
cd kd-experiments-without-weights
python .\run_experiment_no_weights.py --experiment exp1_kkd_base --mode soft_labels --results-dir ".\soft labels\exp1_kkd_base\results"
```

Run all experiments with 10 runs each for soft labels:

```bash
cd kd-experiments-without-weights
python .\run_soft_labels.py --runs 10 --resume
```

Generate the summary text for soft labels after rerunning:

```bash
cd kd-experiments-without-weights
python generate_summary_soft_labels.py
```

Notes
-----
- Soft-label runs do not exchange model weights. Each client sends a KD payload with soft-label statistics; the server aggregates that payload and sends the global soft label back in the next round.
- `run_soft_labels.py --resume` only counts completed full-length runs matching each experiment config, so short smoke-test runs are ignored.
- The summary generator writes `soft_labels_summary.txt` in `kd-experiments-without-weights`.
