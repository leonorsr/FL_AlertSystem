from __future__ import annotations

import argparse
from pathlib import Path

from bin_payload_configs import DEFAULT_BIN_PAYLOAD_ORDER
from run_experiment import DEFAULT_EXPERIMENT_ID, DEFAULT_RESULTS_ROOT, run_binned_probability_payload_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all binned soft-label payload experiments.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--payloads", nargs="+", default=DEFAULT_BIN_PAYLOAD_ORDER, choices=DEFAULT_BIN_PAYLOAD_ORDER)
    args = parser.parse_args()

    for payload_name in args.payloads:
        for run_idx in range(args.runs):
            run_seed = None if args.random_seed is None else args.random_seed + run_idx
            print(f"Running payload={payload_name}, run={run_idx + 1}/{args.runs}, seed={run_seed}")
            run_dir = run_binned_probability_payload_experiment(
                payload_name=payload_name,
                results_dir=Path(args.results_dir),
                experiment_id=args.experiment,
                num_rounds_override=args.num_rounds,
                local_epochs_override=args.local_epochs,
                random_seed_override=run_seed,
            )
            print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
