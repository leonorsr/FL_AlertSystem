from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gen_syntdata.build_mixed_datasets import RATIOS
from gen_syntdata.smoke_experiments.run_fl_smoke_grid import run_one, summarize_grid


STRATEGIES = ["fedavg", "soft_labels", "class_prototype"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4 x 3 SMOTE smoke tests.")
    parser.add_argument("--datasets-dir", type=Path, default=Path("gen_syntdata/baseline_smote/full_data_datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("gen_syntdata/baseline_smote/full_data_smoke_results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--mixtures", nargs="+", choices=list(RATIOS), default=list(RATIOS))
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "SMOTE",
        "datasets_dir": str(args.datasets_dir),
        "strategies": args.models,
        "mixtures": args.mixtures,
        "seeds": args.seeds,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "run_count": len(args.models) * len(args.mixtures) * len(args.seeds),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Grid: 1 SMOTE x {len(args.mixtures)} repartition(s) x "
        f"{len(args.models)} strategy/strategies x {len(args.seeds)} seed(s) "
        f"= {manifest['run_count']} runs"
    )

    for strategy in args.models:
        for mixture in args.mixtures:
            windows_path = args.datasets_dir / mixture / "mixed_windows.csv"
            if not windows_path.exists():
                raise FileNotFoundError(windows_path)
            for seed in args.seeds:
                if args.dry_run:
                    print(f"{strategy} SMOTE {mixture} seed={seed}: {windows_path}")
                    continue
                run_one(
                    model=strategy,
                    generator="smote",
                    ratio=mixture,
                    seed=seed,
                    windows_path=windows_path,
                    output_dir=args.output_dir,
                    num_rounds=args.num_rounds,
                    local_epochs=args.local_epochs,
                    rerun_complete=args.rerun_complete,
                )
    if not args.dry_run:
        summarize_grid(args.output_dir)
        summary = pd.read_csv(args.output_dir / "summary_mean_std.csv")
        if not summary.empty:
            summary["selection_score"] = summary[["test_roc_auc_mean", "test_pr_auc_mean"]].mean(axis=1)
            summary.sort_values("selection_score", ascending=False).to_csv(
                args.output_dir / "ranking.csv", index=False
            )


if __name__ == "__main__":
    main()
