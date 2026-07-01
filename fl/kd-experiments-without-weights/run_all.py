from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kd-experiments"))

from run_experiment_no_weights import run_experiment_no_weights


CHANNELS = ["soft labels", "hidden states", "class prototype"]


def main():
    parser = argparse.ArgumentParser(description="Run all no-weight KD experiments across channels.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per experiment")
    parser.add_argument("--root", default=Path(__file__).resolve().parent, help="Root folder for channels")
    parser.add_argument("--channels", nargs="+", default=CHANNELS, choices=CHANNELS, help="Channels to run")
    parser.add_argument("--num-rounds", type=int, default=None, help="Optional global round override for smoke tests")
    parser.add_argument("--local-epochs", type=int, default=None, help="Optional local epoch override for smoke tests")
    parser.add_argument("--base-seed", type=int, default=None, help="Optional base seed for repeated runs")
    args = parser.parse_args()

    root = Path(args.root)
    for channel in args.channels:
        channel_dir = root / channel
        print(f"Channel: {channel}")
        for exp in sorted([p.name for p in channel_dir.iterdir() if p.is_dir()]):
            results_dir = channel_dir / exp / "results"
            print(f" Running {exp} -> results in {results_dir}")
            for run_idx in range(args.runs):
                run_seed = args.base_seed + run_idx if args.base_seed is not None else None
                run_dir = run_experiment_no_weights(
                    exp,
                    results_dir,
                    mode=channel.replace(" ", "_").lower(),
                    num_rounds_override=args.num_rounds,
                    local_epochs_override=args.local_epochs,
                    random_seed_override=run_seed,
                )
                print(f"  Saved: {run_dir}")


if __name__ == "__main__":
    main()
