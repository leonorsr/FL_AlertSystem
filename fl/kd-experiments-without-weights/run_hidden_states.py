from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kd-experiments"))

from config import EXPERIMENT_CATALOG
from run_experiment_no_weights import run_experiment_no_weights


EXPERIMENT_IDS = [
    "exp1_kkd_base",
    "exp2_fraction_clients",
    "exp3_local_epochs",
    "exp4_unweighted_aggregation",
    "exp5_cross_dataset",
    "exp6_keep_best_local_model",
    "exp7_clustered_aggregation",
    "exp8_personalized_fedavg",
    "exp9_final_local_finetuning",
    "exp10_clustered_keep_best_local",
    "exp11_baseline_final",
    "exp12_final_unweighted",
    "exp13_final_keep_best_local",
    "exp14_final_clustered",
]


def _completed_runs(experiment_id: str, results_dir: Path) -> int:
    if not results_dir.exists():
        return 0

    expected = EXPERIMENT_CATALOG[experiment_id]
    completed = 0
    for path in results_dir.glob("run_*"):
        if not (path / "run_summary.json").exists() or not (path / "config.json").exists():
            continue
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(config.get("num_rounds", -1)) != int(expected.num_rounds):
            continue
        if int(config.get("local_epochs", -1)) != int(expected.local_epochs):
            continue
        if config.get("communication_mode") != "hidden_states":
            continue
        completed += 1
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KD experiments using hidden states as no-weight communication.")
    parser.add_argument("--runs", type=int, default=10, help="Target number of runs per experiment.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENT_IDS,
        choices=EXPERIMENT_IDS,
        help="Subset of experiments to run. Defaults to exp1-exp14.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "hidden states",
        help="Output root for hidden-state experiment folders.",
    )
    parser.add_argument("--num-rounds", type=int, default=None, help="Optional global round override for smoke tests.")
    parser.add_argument("--local-epochs", type=int, default=None, help="Optional local epoch override for smoke tests.")
    parser.add_argument("--base-seed", type=int, default=None, help="Base seed. Defaults to each experiment config seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only run the missing repetitions needed to reach --runs in each experiment folder.",
    )
    args = parser.parse_args()

    for experiment_id in args.experiments:
        results_dir = args.root / experiment_id / "results"
        existing_runs = _completed_runs(experiment_id, results_dir) if args.resume else 0
        runs_to_execute = max(args.runs - existing_runs, 0)
        print(f"{experiment_id}: {existing_runs} existing completed runs, {runs_to_execute} to execute.")

        base_seed = args.base_seed if args.base_seed is not None else EXPERIMENT_CATALOG[experiment_id].random_seed
        for run_idx in range(runs_to_execute):
            run_seed = base_seed + existing_runs + run_idx
            print(f"  Run {existing_runs + run_idx + 1}/{args.runs} with seed {run_seed}")
            run_dir = run_experiment_no_weights(
                experiment_id=experiment_id,
                results_dir=results_dir,
                mode="hidden_states",
                num_rounds_override=args.num_rounds,
                local_epochs_override=args.local_epochs,
                random_seed_override=run_seed,
            )
            print(f"  Saved: {run_dir}")


if __name__ == "__main__":
    main()
