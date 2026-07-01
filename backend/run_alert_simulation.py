from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    from alert_simulation import run_selected_alert_simulation
else:
    from .alert_simulation import run_selected_alert_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected soft-label alert simulation in the backend.")
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"), help="Where to save run artifacts")
    parser.add_argument("--num-rounds", type=int, default=None, help="Override the number of FL rounds")
    parser.add_argument("--local-epochs", type=int, default=None, help="Override local epochs per round")
    parser.add_argument("--random-seed", type=int, default=None, help="Override the random seed")
    parser.add_argument("--windows-path", default=None, help="Optional path to the preprocessing windows.csv file")
    parser.add_argument("--smoke-test", action="store_true", help="Run a compact smoke-test version of the simulation")
    args = parser.parse_args()

    run_dir, summary = run_selected_alert_simulation(
        results_dir=args.results_dir,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        random_seed=args.random_seed,
        smoke_test=args.smoke_test,
        windows_path=args.windows_path,
    )

    print(f"Selected strategy run completed. Results saved in: {run_dir}")
    print(f"Summary: {summary['selected_experiment_id']} | smoke_test={summary['smoke_test']}")


if __name__ == "__main__":
    main()
